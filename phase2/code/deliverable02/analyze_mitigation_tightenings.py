"""D2 #53 — mitigation_dependent rule tightening analysis (read-only, $0, no API).

D6's #47 re-tag made the condition->resource_area tags validated-good (80-row gold: any-overlap
0.89). Yet D2's `mitigation_dependent_f1` did NOT rise (any-overlap 0.566 / primary 0.596) — because
the metric is PRECISION-bound (~0.41-0.45), not tag-bound: the current rule over-attributes. The rule is

    mitigation_resource_matched = mitigation_flag AND (scope=='project_overall' OR resource in areas)
    mitigation_dependent        = mitigation_resource_matched OR class=='less_than_significant_with_mitigation'

This script scores candidate TIGHTENINGS of the resource_matched branch against D2's own gold
(gold_mitigation_link), reusing 05_validate_significance.evaluate() so the numbers are directly
comparable to the committed metric. It ships NOTHING — it prints P/R/F1 for the reviewer to choose.

Levers available on significance_determinations.parquet (from the mitigation rejoin):
  - obligation_level_set  (require the matched condition to be required/committed, not descriptive)
  - matched_condition_row_count (require >= N matched conditions)
  - mitigation_resource_areas + determination_scope + shared_resource_area
    (drop the 'project_overall' free pass -> require a real resource overlap)
NOT available (dropped in rejoin): mitigation_same_section — flagged as a join-recompute candidate.

USAGE:  conda run -n nepa python phase2/code/deliverable02/analyze_mitigation_tightenings.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa'.")

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("v05", HERE / "05_validate_significance.py")
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)  # gives us v.evaluate, v._real_gold/_real_pred, v.C, v.prf

LTS = "less_than_significant_with_mitigation"


def _set(s: object) -> set[str]:
    return {t.strip().lower() for t in str(s or "").replace(";", ",").split(",") if t.strip()}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = v.C.q(f"SELECT * FROM read_parquet('{v.C.GOLD}')")
    det = pd.read_parquet(v.C.SIGNIFICANCE_DETERMINATIONS)
    return gold, det


def variants(det: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {name: recomputed mitigation_dependent boolean Series} for baseline + tightenings.
    Every variant keeps the authoritative class==LTS_with_mitigation term as a floor; they only
    restrict the resource_matched (join-based) branch."""
    is_lts = det["determination_class"].astype(str).eq(LTS)
    rmatched = det["mitigation_resource_matched"].fillna(False).astype(bool)
    flag = det["mitigation_flag"].fillna(False).astype(bool)
    scope = det["determination_scope"].astype(str)
    res = det["shared_resource_area"].astype(str).str.lower()
    areas = det["mitigation_resource_areas"].map(_set)
    oblig = det["obligation_level_set"].map(_set)
    count = pd.to_numeric(det.get("matched_condition_row_count"), errors="coerce").fillna(0)

    # resource_matched WITHOUT the project_overall free pass (require a real resource overlap)
    rmatched_specific = pd.Series(
        [bool(fl) and (r in a) for fl, r, a in zip(flag, res, areas)], index=det.index)
    has_committed = oblig.map(lambda s: bool(s & {"required", "committed"}))

    out = {
        "baseline (any-overlap, committed)": rmatched | is_lts,
        "T1 require obligation required/committed": (rmatched & has_committed) | is_lts,
        "T2 require >=2 matched conditions": (rmatched & (count >= 2)) | is_lts,
        "T3 drop project_overall free pass": rmatched_specific | is_lts,
        "T4 specific + obligation": (rmatched_specific & has_committed) | is_lts,
        "T5 specific + >=2 conditions": (rmatched_specific & (count >= 2)) | is_lts,
    }
    return out


def score_variant(gold: pd.DataFrame, det: pd.DataFrame, mit_dep: pd.Series, tag: str) -> dict:
    d = det.copy()
    d["mitigation_dependent"] = mit_dep.values
    rows, _ = v.evaluate(gold, d, tag)
    m = next((r for r in rows if r.get("metric") == "mitigation_dependent_f1"), {})
    return {"precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"),
            "tp": m.get("tp"), "fp": m.get("fp"), "fn": m.get("fn")}


def main() -> None:
    gold, det = load()
    vs = variants(det)
    print(f"[#53] gold rows: {len(gold)}  determinations: {len(det)}  "
          f"(scoring mitigation_dependent_f1 on matched window×resource pairs, overall scope)\n")
    print(f"{'variant':<44}  {'n_pred_dep':>10}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'tp/fp/fn':>12}")
    print("-" * 96)
    rows = []
    for name, dep in vs.items():
        s = score_variant(gold, det, dep, name)
        npred = int(dep.sum())
        print(f"{name:<44}  {npred:>10}  {s['precision']:>6}  {s['recall']:>6}  {s['f1']:>6}  "
              f"{str(s['tp'])+'/'+str(s['fp'])+'/'+str(s['fn']):>12}")
        rows.append({"variant": name, "n_pred_dependent": npred, **s})
    out = pd.DataFrame(rows)
    dest = HERE.parents[1] / "notes" / "deliverable02" / "mitigation_tightening_scores.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\n[#53] wrote {dest}")
    print("[#53] same-section tightening is NOT scored here (mitigation_same_section is dropped by the "
          "rejoin); it needs a join recompute — flagged as a follow-up if the reviewer wants it.")


if __name__ == "__main__":
    main()
