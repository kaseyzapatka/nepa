"""D6 — 11: expand analysis (generalized).

Item #39: generalize the transmission-only "expand" case (07_classify_and_rank.py's
EXPAND test) to ALL bounded CEs. For each candidate_category with a matched CE
(rank-1 in candidate_ce_comparison), compare the is_ce_shaped FONSIs' size
distribution against the CE's stated numeric cap(s), and suggest a raised cap
(the 90th percentile of our observed values).

Inputs (D6_ANALYSIS_DIR):
  candidate_facts.parquet
  candidate_ce_comparison.parquet

Output:
  output/deliverable06/expand_analysis.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import numpy as np
import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now

FACTS = D6_ANALYSIS_DIR / "candidate_facts.parquet"
CE = D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet"
OUT = D6_OUTPUT_DIR / "expand_analysis.csv"

# metric -> (candidate_facts size col, candidate_ce_comparison bound col)
METRIC_MAP = {
    "acres": ("max_acres", "bound_acres"),
    "miles": ("max_miles", "bound_miles"),
    "mw": ("max_megawatts", "bound_mw"),
    "kv": ("max_kilovolts", "bound_kv"),
    "wells": ("n_wells", "bound_wells"),
}


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()
    facts = pd.read_parquet(FACTS)
    ce = pd.read_parquet(CE)

    best_ce = (
        ce.sort_values("retrieval_rank")
        .groupby("candidate_category", as_index=False)
        .first()
    )

    rows = []
    for cat in sorted(set(facts["candidate_category"])):
        cat_facts = facts[facts["candidate_category"].eq(cat)]
        shaped = cat_facts[cat_facts["is_ce_shaped"].astype(bool)]
        if shaped.empty:
            continue
        best = best_ce[best_ce["candidate_category"].eq(cat)]
        if best.empty:
            continue
        best = best.iloc[0]
        label = str(cat_facts["candidate_label"].iloc[0]) if "candidate_label" in cat_facts.columns else cat

        for metric, (fact_col, bound_col) in METRIC_MAP.items():
            bound = best.get(bound_col)
            if bound is None or pd.isna(bound):
                continue
            vals = pd.to_numeric(shaped.get(fact_col), errors="coerce").dropna()
            n_fonsi = len(vals)
            if n_fonsi < 1:
                continue
            bound = float(bound)
            n_exceeding = int((vals > bound).sum())
            v_p90 = float(np.percentile(vals, 90))
            rows.append({
                "candidate_category": cat,
                "candidate_label": label,
                "metric": metric,
                "ce_structured_id": best.get("structured_id", ""),
                "ce_bound": bound,
                "n_fonsi": n_fonsi,
                "v_min": float(vals.min()),
                "v_median": float(vals.median()),
                "v_p90": v_p90,
                "v_max": float(vals.max()),
                "n_exceeding": n_exceeding,
                "pct_exceeding": round(n_exceeding / n_fonsi, 4),
                "suggested_cap": round(v_p90),
                "canonical_source_url": best.get("canonical_source_url", ""),
            })

    result = pd.DataFrame(rows)
    result["expand_analysis_extraction_run_at"] = run_at
    result["expand_analysis_llm_run_at"] = ""

    ensure_d6_dirs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print(result.to_string(index=False))
    n_pairs = len(result)
    n_with_exceeding = int((result["n_exceeding"] > 0).sum()) if n_pairs else 0
    print(f"\n{n_pairs} (category, metric) pairs fire; {n_with_exceeding} have at least one FONSI exceeding the CE cap.")


if __name__ == "__main__":
    main()
