"""Validate the tiered EIS decision (ROD-first, FEIS-fallback) against gold (read-only).
Runs the real select_dates_for_project on the gold ROD/FEIS projects and reports has_rod,
decision_is_feis_fallback, and whether the picked decision matches the Codex-verified true date
(granularity-aware). Writes nothing."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sel05", HERE / "05_select_dates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
AUD = HERE.parents[2] / "phase2" / "output" / "deliverable04" / "eis_audit"

cands = pd.read_parquet(m.CANDIDATES_PATH)
cands = cands[cands.process_type == "EIS"]
by_proj = {pid: g.reset_index(drop=True) for pid, g in cands.groupby("project_id", sort=False)}

index_map = {}
if m.INDEX_PATH.exists():
    idx = pd.read_parquet(m.INDEX_PATH, columns=["project_id", "decision_doc_score", "initiation_doc_score"])
    for pid, grp in idx.groupby("project_id"):
        index_map[pid] = {"decision_doc_score": grp.decision_doc_score.max(),
                          "initiation_doc_score": grp.initiation_doc_score.max()}


def gran_match(picked, true_date, gran):
    if not picked or not true_date or str(true_date).lower() in ("nan", "none", "", "nat"):
        return None
    g = str(gran).strip().lower(); p, t = str(picked)[:10], str(true_date)[:10]
    n = 4 if g.startswith("year") else (7 if g.startswith("month") else 10)
    return p[:n] == t[:n]


def run(gold_csv, datecol, goldcol, golddate, grancol_yes, grancol_no, kind):
    g = pd.read_csv(AUD / gold_csv)
    rows, correct, has_rod_n, fb_n, n = [], 0, 0, 0, 0
    for _, r in g.iterrows():
        pid = r.project_id
        if pid not in by_proj:
            continue
        res, _ = m.select_dates_for_project(by_proj[pid], "EIS", index_map)
        res["project_id"] = pid
        corr = str(r[goldcol]).strip().lower()
        true_date = r[datecol] if corr == "yes" else r[golddate]
        gran = r.get(grancol_yes) if corr == "yes" else r.get(grancol_no)
        ok = gran_match(res["decision_date"], true_date, gran)
        n += 1
        has_rod_n += int(bool(res.get("has_rod")))
        fb_n += int(bool(res.get("decision_is_feis_fallback")))
        if ok:
            correct += 1
        rows.append({"has_rod": res.get("has_rod"), "fb": res.get("decision_is_feis_fallback"),
                     "flag": res.get("timeline_flags", "")[:24], "dec": str(res["decision_date"]),
                     "gran": res["decision_date_granularity"], "true": str(true_date)[:10], "ok": ok})
    print(f"\n=== {kind}: {n} gold projects in pool ===")
    print(f"has_rod=True: {has_rod_n}/{n} | decision_is_feis_fallback=True: {fb_n}/{n}")
    print(f"decision_date matches verified truth (granularity-aware): {correct}/{n}")
    df = pd.DataFrame(rows)
    print(df.head(12).to_string(index=False))
    return df


run("eis_rod_promotion_sample_labeled.csv", "decision_date", "gold_is_correct_rod", "gold_rod_date",
    "decision_date_granularity", "gold_rod_granularity", "ROD gold")
run("eis_feis_sample_labeled.csv", "final_eis_date", "gold_is_correct_feis", "gold_feis_date",
    "final_eis_date_granularity", "gold_feis_granularity", "FEIS gold")
