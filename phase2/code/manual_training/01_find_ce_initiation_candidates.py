import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

"""
01_find_ce_initiation_candidates.py

Find CE dates that are likely initiation dates being mislabeled as 'decision'.

Strategy: anchor on the decision date. For CE projects where BERT found a
decision date but no initiation date, look up all candidate dates for that
project in the regex cache. Any date that falls BEFORE the decision date is a
candidate for initiation — on CE CX forms the initiator always signs first.

The matched date IS the initiator date when it is clearly earlier than the
decision date and the context window shows an initiator-related signature block.

Outputs:
    data/manual_training/review_ce_initiation_candidates.csv

Workflow:
    1. Run this script (requires --bert-run output to anchor on decision dates)
    2. Open the CSV. For each row:
         - 'days_before_decision' shows how much earlier this date is
         - 'context' shows what surrounds the date in the document
         - Set correct_type = 'initiation' for clear initiator signature dates
         - Leave blank if ambiguous
    3. Run: python code/manual_training/02_apply_corrections.py
    4. Retrain: --bert-generate → --bert-train --source CE
"""

import argparse
import sys
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_DIR = BASE_DIR / "data" / "manual_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BERT_OUTPUT = ANALYSIS_DIR / "projects_timeline_bert.parquet"
DEFAULT_CACHE = ANALYSIS_DIR / "regex_candidates_ce.parquet"
CORRECTIONS_FILE = ANALYSIS_DIR / "manual_training_corrections.csv"
OUTPUT_FILE = OUTPUT_DIR / "review_ce_initiation_candidates.csv"

sys.path.insert(0, str(BASE_DIR / "code" / "extract"))
from extract_timeline import auto_label_context  # noqa: E402


def _load_existing_keys() -> set:
    if CORRECTIONS_FILE.exists():
        df = pd.read_csv(CORRECTIONS_FILE)
        return set(zip(df["project_id"], df["date"]))
    return set()


def _parse_date(s) -> datetime | None:
    if not s or pd.isna(s):
        return None
    try:
        return pd.to_datetime(str(s)).to_pydatetime()
    except Exception:
        return None


def main(bert_file: Path, cache_file: Path, top_n: int, min_gap_days: int, max_gap_days: int):
    # ── Load BERT output ──────────────────────────────────────────────────────
    print(f"\nLoading BERT output from: {bert_file}")
    if not bert_file.exists():
        print(f"ERROR: {bert_file} not found. Run --bert-run first.")
        return
    bert = pd.read_parquet(bert_file)
    bert = bert[bert["dataset_source"] == "CE"].copy()
    print(f"  CE projects: {len(bert):,}")

    # Projects with a decision date but no initiation date
    missing_init = bert[
        bert["bert_timeline_status"] == "missing_initiation"
    ].copy()
    print(f"  Missing initiation: {len(missing_init):,} projects")

    # Keep only those where we have a confirmed decision date to anchor on
    missing_init = missing_init[missing_init["bert_decision_date_final"].notna()].copy()
    missing_init["decision_dt"] = missing_init["bert_decision_date_final"].apply(_parse_date)
    missing_init = missing_init[missing_init["decision_dt"].notna()]
    print(f"  With a decision date to anchor on: {len(missing_init):,} projects")

    if missing_init.empty:
        print("No anchored projects found. Run --bert-run on more projects first.")
        return

    project_ids = set(missing_init["project_id"])
    decision_by_pid = dict(zip(missing_init["project_id"], missing_init["decision_dt"]))

    # ── Load regex cache candidates for those projects ─────────────────────
    print(f"\nLoading CE regex cache from: {cache_file}")
    if not cache_file.exists():
        print(f"ERROR: {cache_file} not found.")
        return
    con = duckdb.connect()
    pid_list = "', '".join(project_ids)
    cache = con.execute(f"""
        SELECT project_id, date, context, section_label, dep_verb, sig_flag
        FROM read_parquet('{cache_file}')
        WHERE project_id IN ('{pid_list}')
    """).df()
    print(f"  Cache rows for these projects: {len(cache):,}")

    # ── Find dates earlier than decision date ──────────────────────────────
    cache["candidate_dt"] = cache["date"].apply(_parse_date)
    cache = cache[cache["candidate_dt"].notna()].copy()
    cache["decision_dt"] = cache["project_id"].map(decision_by_pid)
    cache["days_before_decision"] = cache.apply(
        lambda r: (r["decision_dt"] - r["candidate_dt"]).days, axis=1
    )

    # Keep dates that are between min_gap_days and max_gap_days before decision
    candidates = cache[
        (cache["days_before_decision"] >= min_gap_days) &
        (cache["days_before_decision"] <= max_gap_days)
    ].copy()
    print(f"  Dates {min_gap_days}–{max_gap_days} days before decision: {len(candidates):,}")

    # ── Exclude already-corrected pairs ────────────────────────────────────
    existing_keys = _load_existing_keys()
    if existing_keys:
        before = len(candidates)
        candidates = candidates[
            ~candidates.apply(lambda r: (r["project_id"], r["date"]) in existing_keys, axis=1)
        ]
        print(f"  Excluding {before - len(candidates):,} already in corrections file")

    if candidates.empty:
        print("No new candidates. Try expanding --max-gap or running --bert-run on more projects.")
        return

    # ── Apply weak supervision to show what the model currently thinks ─────
    candidates["auto_label"] = candidates["context"].apply(
        lambda c: auto_label_context(c, "CE") or ""
    )

    # Sort: one candidate per project (earliest date wins), then by project
    candidates = (
        candidates
        .sort_values(["project_id", "days_before_decision"], ascending=[True, False])
        .drop_duplicates(subset="project_id", keep="first")
        .head(top_n)
    )
    print(f"  Exporting {len(candidates):,} candidates (one per project)")

    rows = []
    for _, r in candidates.iterrows():
        decision_str = r["decision_dt"].strftime("%Y-%m-%d") if r["decision_dt"] else ""
        rows.append({
            "project_id": r["project_id"],
            "date": r["date"],
            "days_before_decision": int(r["days_before_decision"]),
            "decision_date": decision_str,
            "auto_label": r["auto_label"],
            "section_label": r.get("section_label", ""),
            "dep_verb": r.get("dep_verb", ""),
            "context": str(r["context"])[:300].replace("\n", " ").strip(),
            "correct_type": "",   # USER FILLS THIS IN
        })

    out = pd.DataFrame(rows).sort_values("days_before_decision", ascending=False)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(out):,} candidates → {OUTPUT_FILE}")
    print()
    print("=== How to review ===")
    print("'days_before_decision' = how many days before the confirmed decision date this date falls.")
    print("'auto_label' = what weak supervision currently assigns (usually 'decision' — that's the bug).")
    print()
    print("For each row, look at 'context':")
    print("  → DOE Initiator Signature / DOE INITIATOR ... DATE: {this date}  → set correct_type = 'initiation'")
    print("  → NEPA Compliance Officer ... DATE: {this date}                  → leave correct_type blank")
    print("  → ambiguous or form revision date                                → leave blank")
    print()
    print("After reviewing:")
    print("  python code/manual_training/02_apply_corrections.py")
    print("  python code/extract/extract_timeline.py --bert-generate")
    print("  python code/extract/extract_timeline.py --bert-train --source CE")
    print()
    print("auto_label distribution:")
    print(out["auto_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find CE initiation candidates anchored to known decision dates"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_BERT_OUTPUT),
        help=f"BERT output parquet (default: {DEFAULT_BERT_OUTPUT.name})"
    )
    parser.add_argument(
        "--cache", type=str, default=str(DEFAULT_CACHE),
        help=f"CE regex cache parquet (default: {DEFAULT_CACHE.name})"
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Max candidates to export, one per project (default: 50)"
    )
    parser.add_argument(
        "--min-gap", type=int, default=1,
        help="Min days before decision date to include (default: 1)"
    )
    parser.add_argument(
        "--max-gap", type=int, default=730,
        help="Max days before decision date to include (default: 730)"
    )
    args = parser.parse_args()
    main(Path(args.input), Path(args.cache), args.top, args.min_gap, args.max_gap)
