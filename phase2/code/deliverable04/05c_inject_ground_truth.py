"""
Inject human-verified ground-truth dates (ranker.csv) into the FINAL timeline output (D4).

WHERE THIS SITS
---------------
This is a standalone TERMINAL step. It runs AFTER 05_select_dates.py (and after 06 if used) and
operates ONLY on timeline_project_dates.parquet. It does NOT re-run selection, does NOT rewrite the
candidates parquet, and finishes in seconds.

ranker.csv carries, per project, the human-verified initiation_candidate_id and decision_candidate_id
(the SAME file that trains the 05b learned ranker). Here those verified picks are written straight
into the output as ground truth — overwriting whatever selection produced.

WHY THIS DOESN'T TOUCH TRAIN/TEST/VALIDATION
--------------------------------------------
Model training reads ranker.csv / classifier.csv directly; the ranker's own --eval reads the
candidates parquet + ranker.csv. NEITHER reads timeline_project_dates.parquet. So writing verified
dates into the OUTPUT cannot leak into any training or validation set — it is a pure terminal write.
The only metric that reads the output is 05b --eval-output (end-to-end), which should therefore be
computed BEFORE this step (or with --scope train) to stay honest.

SCOPE
-----
  --scope all    (default) inject ALL ranker.csv projects — the shipped deliverable should carry
                 every human-verified date.
  --scope train  inject only split==train rows — leaves split==test as pipeline output so
                 05b --eval-output can measure honest end-to-end accuracy against held-out truth.

ONE-SIDED CONFLICT HANDLING
---------------------------
When only one side is verified (e.g. verified initiation, but decision is "none" in ranker.csv) and
the existing pipeline date on the OTHER side now contradicts it (decision before initiation), the
contradicted NON-VERIFIED date is dropped (set missing) rather than left to produce invalid_order.
A verified date never loses to a stale pipeline pick.

Usage:
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05c_inject_ground_truth.py
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05c_inject_ground_truth.py --scope train
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05c_inject_ground_truth.py --dates-path X --dry-run
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
TIMELINE_DIR = PHASE2 / "data" / "analysis" / "timeline"
TRAINING_DIR = PHASE2 / "training" / "deliverable04"

DATES_PATH = TIMELINE_DIR / "timeline_project_dates.parquet"
CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
GROUND_TRUTH_PATH = TRAINING_DIR / "ranker.csv"

SAME_DAY_FLAG = "same_day"
STALE_FLAGS = {"missing_initiation", "missing_decision", "missing_both", "invalid_order"}


def _recompute_row(init_date, init_gran, dec_date, dec_gran,
                   init_proxy, dec_proxy, existing_flags: str) -> dict:
    """Granularity-aware status/duration/flags for ONE row after dates were overwritten.

    Granularity-aware so a same-MONTH initiation/decision (which midpoint imputation later resolves
    to the 15th) is not spuriously flagged invalid_order."""
    has_init = pd.notna(init_date) and str(init_date) != ""
    has_dec = pd.notna(dec_date) and str(dec_date) != ""

    flags = [f for f in str(existing_flags or "").split("|") if f and f not in STALE_FLAGS]
    if "ground_truth_bypass" not in flags:
        flags.append("ground_truth_bypass")

    duration_days = None
    if has_init and has_dec:
        di = date.fromisoformat(str(init_date)[:10])
        dd = date.fromisoformat(str(dec_date)[:10])
        # Compare at the COARSER of the two granularities so month/year proxies don't false-trip.
        coarse = {"year", "month", "day"}
        if "year" in (init_gran, dec_gran):
            before = (di.year > dd.year)
            equal = (di.year == dd.year)
        elif "month" in (init_gran, dec_gran):
            before = (di.year, di.month) > (dd.year, dd.month)
            equal = (di.year, di.month) == (dd.year, dd.month)
        else:
            before = di > dd
            equal = di == dd
        if before:
            status = "invalid_order"
            flags.append("invalid_order")
        elif equal:
            status = "complete_clear" if not (init_proxy or dec_proxy) else "complete_with_proxy"
            flags.append(SAME_DAY_FLAG)
            if init_gran == "day" and dec_gran == "day":
                duration_days = 0
        else:
            if init_gran == "day" and dec_gran == "day":
                duration_days = (dd - di).days
            status = "complete_with_proxy" if (init_proxy or dec_proxy) else "complete_clear"
    elif has_dec:
        status = "missing_initiation"
        flags.append("missing_initiation")
    elif has_init:
        status = "missing_decision"
        flags.append("missing_decision")
    else:
        status = "missing_both"
        flags += ["missing_initiation", "missing_decision"]

    return {"timeline_status": status, "duration_days": duration_days,
            "timeline_flags": "|".join(f for f in flags if f)}


def inject(dates_path: Path, scope: str, dry_run: bool) -> None:
    if not dates_path.exists():
        raise SystemExit(f"No dates parquet at {dates_path}. Run 05_select_dates.py first.")
    if not GROUND_TRUTH_PATH.exists():
        raise SystemExit(f"No ground truth at {GROUND_TRUTH_PATH}.")

    dates = pd.read_parquet(dates_path)
    gt = pd.read_csv(GROUND_TRUTH_PATH, dtype=str, keep_default_na=False)
    if scope == "train":
        gt = gt[gt["split"].astype(str).str.strip().eq("train")].copy()
    cand = pd.read_parquet(CANDIDATES_PATH, columns=["candidate_id", "parsed_date", "date_granularity"])
    clook = cand.drop_duplicates("candidate_id").set_index("candidate_id")

    dmap = dates.set_index("project_id")
    n_proj = n_init = n_dec = n_conflict = n_absent = 0

    for r in gt.itertuples():
        pid = r.project_id
        if pid not in dmap.index:
            n_absent += 1
            continue
        init_id = str(r.initiation_candidate_id).strip()
        dec_id = str(r.decision_candidate_id).strip()
        touched = False

        if init_id not in ("", "none") and init_id in clook.index:
            c = clook.loc[init_id]
            dmap.loc[pid, "initiation_date"] = c["parsed_date"]
            dmap.loc[pid, "initiation_date_granularity"] = c.get("date_granularity", "day")
            dmap.loc[pid, "initiation_source_type"] = "ground_truth_verified"
            dmap.loc[pid, "initiation_confidence"] = "high"
            dmap.loc[pid, "initiation_is_proxy"] = False
            n_init += 1
            touched = True

        if dec_id not in ("", "none") and dec_id in clook.index:
            c = clook.loc[dec_id]
            dmap.loc[pid, "decision_date"] = c["parsed_date"]
            dmap.loc[pid, "decision_date_granularity"] = c.get("date_granularity", "day")
            dmap.loc[pid, "decision_source_type"] = "ground_truth_verified"
            dmap.loc[pid, "decision_confidence"] = "high"
            dmap.loc[pid, "decision_is_proxy"] = False
            n_dec += 1
            touched = True

        if not touched:
            continue
        n_proj += 1

        row = dmap.loc[pid]
        init_src = row["initiation_source_type"]
        dec_src = row["decision_source_type"]
        idt, idg = row["initiation_date"], row["initiation_date_granularity"]
        ddt, ddg = row["decision_date"], row["decision_date_granularity"]

        # One-sided conflict: a verified date contradicts a NON-verified date on the other side.
        # Drop the non-verified (stale pipeline) date rather than emit invalid_order.
        if pd.notna(idt) and pd.notna(ddt):
            di = date.fromisoformat(str(idt)[:10])
            dd = date.fromisoformat(str(ddt)[:10])
            if di > dd:
                init_v = init_src == "ground_truth_verified"
                dec_v = dec_src == "ground_truth_verified"
                if init_v and not dec_v:
                    dmap.loc[pid, "decision_date"] = None
                    dmap.loc[pid, "decision_date_granularity"] = "unknown"
                    dmap.loc[pid, "decision_source_type"] = None
                    ddt, ddg = None, "unknown"
                    n_conflict += 1
                elif dec_v and not init_v:
                    dmap.loc[pid, "initiation_date"] = None
                    dmap.loc[pid, "initiation_date_granularity"] = "unknown"
                    dmap.loc[pid, "initiation_source_type"] = None
                    idt, idg = None, "unknown"
                    n_conflict += 1

        upd = _recompute_row(idt, idg, ddt, ddg,
                             bool(row["initiation_is_proxy"]), bool(row["decision_is_proxy"]),
                             row["timeline_flags"])
        for k, v in upd.items():
            dmap.loc[pid, k] = v

    out = dmap.reset_index()
    print(f"Ground-truth injection (scope={scope}):")
    print(f"  projects written : {n_proj}  (initiation={n_init}, decision={n_dec})")
    print(f"  one-sided conflicts resolved (stale date dropped): {n_conflict}")
    print(f"  ranker projects not in this dates file: {n_absent}")
    print(f"  rows now flagged ground_truth_bypass: "
          f"{out['timeline_flags'].fillna('').str.contains('ground_truth_bypass').sum()}")

    if dry_run:
        print("  [dry-run] not written.")
        return
    # Safety: snapshot the existing output before overwriting in place.
    backup = dates_path.with_suffix(f".pre_gt_inject_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.parquet")
    pd.read_parquet(dates_path).to_parquet(backup, index=False)
    out.to_parquet(dates_path, index=False)
    print(f"  backup written -> {backup.name}")
    print(f"  wrote {dates_path} ({len(out):,} projects)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject verified ranker.csv dates into final D4 output.")
    ap.add_argument("--scope", choices=["all", "train"], default="all",
                    help="all = ship every verified project (default); train = hold out test split.")
    ap.add_argument("--dates-path", default=str(DATES_PATH))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    inject(Path(args.dates_path), args.scope, args.dry_run)


if __name__ == "__main__":
    main()
