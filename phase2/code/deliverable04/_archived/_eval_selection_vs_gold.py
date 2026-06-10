"""
Evaluate 05_select_dates.py output against the 300-project gold (D4).

Compares the pipeline's SELECTED initiation/decision dates (timeline_project_dates.parquet) to the
hand-labeled gold (timeline_gold_projects.parquet), with date matching at the gold's granularity.
Reports the ranker's deterministic 60-project holdout separately from the 240 training projects;
the mixed 300-project score is useful for A/B debugging but is not an unbiased quality estimate.
Also emits a human-readable 50-project report showing, per project, the top learned-ranker candidates
the model chose between, what it selected, and the gold answer.

Outputs:
  output/deliverable04/selection_eval_summary.csv   split x process accuracy
  output/deliverable04/selection_eval_errors.csv    all 300 gold projects: selected vs gold + verdict
  output/deliverable04/selection_eval_report.txt    50 holdout projects with candidate contexts
"""

import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2/data/analysis/timeline"
OUT = ROOT / "phase2/output/deliverable04"
GOLD = TL / "gold/timeline_gold_projects.parquet"
DATES = TL / "timeline_project_dates.parquet"
CANDS = TL / "timeline_candidates.parquet"
GOLD_SAMPLE = OUT / "project_gold_sample.csv"
N_REPORT = 50
SEED = 42
TEST_FRACTION = 0.20


def _norm(v):
    v = ("" if v is None else str(v)).strip().lower()
    return "" if v in ("", "none", "nan", "nat") else v


def _match(sel, gold, gran):
    """Verdict for one head. Compares at the GOLD granularity (month/year truncation)."""
    s, g = _norm(sel), _norm(gold)
    if not g and not s:
        return "ok_both_none"          # correctly no date
    if not g and s:
        return "false_positive"        # picked a date when gold says none
    if g and not s:
        return "missed"                # gold has a date, pipeline found none
    gran = (gran or "day").strip().lower()
    n = {"year": 4, "month": 7}.get(gran, 10)   # YYYY / YYYY-MM / YYYY-MM-DD
    return "match" if s[:n] == g[:n] else "mismatch"


def _ranker_split_map():
    """Reproduce 05b_rank.py's project-level split without mutating the gold CSV."""
    sample = pd.read_csv(GOLD_SAMPLE, dtype=str, keep_default_na=False)
    filled = (
        sample["initiation_candidate_id"].str.strip().ne("")
        | sample["decision_candidate_id"].str.strip().ne("")
    )
    sample = sample[filled].copy()
    if "split" not in sample.columns or sample["split"].str.strip().eq("").all():
        sample["split"] = "train"
        for _proc, grp in sample.groupby("process_type"):
            n_test = max(1, round(len(grp) * TEST_FRACTION))
            n_test = min(n_test, len(grp) - 1) if len(grp) > 1 else 0
            test_ids = grp.sample(n=n_test, random_state=SEED).index
            sample.loc[test_ids, "split"] = "test"
    return sample.set_index("project_id")["split"].to_dict()


def main():
    gold = pd.read_parquet(GOLD)
    dates = pd.read_parquet(DATES)
    split_map = _ranker_split_map()
    d = dates.set_index("project_id")
    rows = []
    for r in gold.itertuples():
        sel = d.loc[r.project_id] if r.project_id in d.index else None
        sel_i = sel["initiation_date"] if sel is not None else None
        sel_d = sel["decision_date"] if sel is not None else None
        rows.append({
            "project_id": r.project_id, "process_type": r.process_type,
            "ranker_split": split_map.get(r.project_id, "unknown"),
            "sel_initiation": sel_i, "gold_initiation": r.gold_initiation_date,
            "init_verdict": _match(sel_i, r.gold_initiation_date, r.gold_initiation_granularity),
            "sel_decision": sel_d, "gold_decision": r.gold_decision_date,
            "dec_verdict": _match(sel_d, r.gold_decision_date, r.gold_decision_granularity),
        })
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "selection_eval_errors.csv", index=False)

    # ----- accuracy: two framings per head -----
    def acc(df, col):
        v = df[col]
        correct = v.isin(["match", "ok_both_none"]).sum()
        has_gold = ~v.isin(["ok_both_none", "false_positive"])     # gold actually had a date
        date_acc = (v == "match").sum() / max(1, has_gold.sum())
        return correct / len(df), date_acc

    print(f"End-to-end selection vs gold ({len(ev)} projects)")
    print("HOLDOUT is the unbiased ranker test split; ALL includes ranker training projects.\n")
    summ = []
    eval_splits = [
        ("HOLDOUT", ev[ev["ranker_split"] == "test"]),
        ("TRAIN", ev[ev["ranker_split"] == "train"]),
        ("ALL", ev),
    ]
    for split_name, split_df in eval_splits:
        print(f"{split_name}:")
        for proc in ["ALL"] + sorted(ev["process_type"].unique()):
            sub = split_df if proc == "ALL" else split_df[split_df["process_type"] == proc]
            ia, ida = acc(sub, "init_verdict")
            da, dda = acc(sub, "dec_verdict")
            summ.append({
                "split": split_name, "process": proc, "n": len(sub),
                "init_overall_acc": round(ia, 3),
                "init_dateacc_when_gold": round(ida, 3),
                "dec_overall_acc": round(da, 3),
                "dec_dateacc_when_gold": round(dda, 3),
            })
            print(
                f"  {proc:4s} (n={len(sub):3d})  init: overall={ia:.3f} "
                f"date-acc|gold={ida:.3f}   decision: overall={da:.3f} "
                f"date-acc|gold={dda:.3f}"
            )
        print()
    pd.DataFrame(summ).to_csv(OUT / "selection_eval_summary.csv", index=False)
    print("  overall_acc = correct incl. correctly-no-date; date-acc|gold = right date WHEN gold has one")
    print("\n  init verdicts: ", ev["init_verdict"].value_counts().to_dict())
    print("  dec  verdicts: ", ev["dec_verdict"].value_counts().to_dict())

    # ----- 50-project holdout visual report -----
    cand = pd.read_parquet(CANDS, columns=[
        "project_id", "candidate_id", "parsed_date", "date_granularity", "candidate_role",
        "learned_init_score", "learned_decision_score", "model_context"])
    report_pool = ev[ev["ranker_split"] == "test"]
    if report_pool.empty:
        report_pool = ev
    samp = report_pool.sample(
        min(N_REPORT, len(report_pool)), random_state=SEED
    ).sort_values(["process_type", "project_id"])

    def excerpt(c):
        c = str(c or ""); i = c.find("[[")
        return (c[max(0, i-60):i+80] if i >= 0 else c[:140]).replace("\n", " ")

    def top(cc, scorecol, k=3):
        cc = cc.dropna(subset=[scorecol]).sort_values(scorecol, ascending=False).head(k)
        return [f"{r.parsed_date}({r.date_granularity},{r.candidate_role},s={getattr(r, scorecol):.2f}) "
                f":: {excerpt(r.model_context)}" for r in cc.itertuples()]

    lines = ["50-project HOLDOUT selection vs gold — ✓=match, ✗=mismatch/miss/FP. "
             "Shows top-3 ranker candidates per head.\n"]
    for r in samp.itertuples():
        cc = cand[cand["project_id"] == r.project_id]
        mark = lambda v: "✓" if v in ("match", "ok_both_none") else "✗"
        lines.append(f"\n===== {r.project_id} [{r.process_type}] =====")
        lines.append(f"  INIT  {mark(r.init_verdict)} [{r.init_verdict}]  selected={r.sel_initiation}  gold={r.gold_initiation}")
        for t in top(cc, "learned_init_score"):
            lines.append(f"        cand: {t}")
        lines.append(f"  DEC   {mark(r.dec_verdict)} [{r.dec_verdict}]  selected={r.sel_decision}  gold={r.gold_decision}")
        for t in top(cc, "learned_decision_score"):
            lines.append(f"        cand: {t}")
    (OUT / "selection_eval_report.txt").write_text("\n".join(lines))
    print(f"\nWrote: selection_eval_summary.csv, selection_eval_errors.csv, "
          f"selection_eval_report.txt ({len(samp)} projects)")


if __name__ == "__main__":
    main()
