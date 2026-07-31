"""D6 v2 — 13: post-FRA refresh (corpus-answerable part of A2).

Tabulates, per candidate cell, how many CE-shaped FONSIs were decided after the
FRA cut date (2023-06-03) vs before vs undated. This is the corpus-answerable
slice of the "post-FRA refresh" question — whether current CE-adoption usage and
agency implementation guidance need refreshing given the FRA — which otherwise
requires EXTERNAL sources not in NEPATEC 2.0 (flagged, not attempted here).

Inputs:
  data/analysis/deliverable06/candidate_facts.parquet

Outputs:
  output/deliverable06/postfra_recurrence.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now

FACTS = D6_ANALYSIS_DIR / "candidate_facts.parquet"
OUT = D6_OUTPUT_DIR / "postfra_recurrence.csv"

FRA_CUT_DATE = "2023-06-03"  # FRA enactment; matches D4/D5

CAVEAT = (
    "CAVEAT: NEPATEC 2.0 coverage of 2024-2025 documents is incomplete (ingestion lag); "
    "a low post-cut count is NOT evidence of low current activity. Current CE-adoption "
    "usage and agency implementation guidance require EXTERNAL sources not in this "
    "corpus — flagged as human follow-up, not attempted here."
)


def main() -> None:
    ensure_d6_dirs()

    facts = pd.read_parquet(FACTS)
    ce = facts[facts["is_ce_shaped"]].copy()
    ce["decision_date_parsed"] = pd.to_datetime(ce["decision_date"], errors="coerce")
    cut = pd.Timestamp(FRA_CUT_DATE)

    is_dated = ce["decision_date_parsed"].notna()
    is_post = ce["decision_date_parsed"] > cut
    is_pre = is_dated & (ce["decision_date_parsed"] <= cut)

    rows = []
    for category, group in ce.groupby("candidate_category"):
        label = group["candidate_label"].iloc[0]
        g_dated = group["decision_date_parsed"].notna()
        g_post = group["decision_date_parsed"] > cut
        g_pre = g_dated & (group["decision_date_parsed"] <= cut)
        rows.append({
            "candidate_category": category,
            "candidate_label": label,
            "n_ce_shaped": len(group),
            "n_dated": int(g_dated.sum()),
            "n_post_fra": int(g_post.sum()),
            "n_pre_fra": int(g_pre.sum()),
            "n_undated": int((~g_dated).sum()),
        })

    rows.append({
        "candidate_category": "TOTAL",
        "candidate_label": "TOTAL",
        "n_ce_shaped": len(ce),
        "n_dated": int(is_dated.sum()),
        "n_post_fra": int(is_post.sum()),
        "n_pre_fra": int(is_pre.sum()),
        "n_undated": int((~is_dated).sum()),
    })

    out = pd.DataFrame(rows)
    out["postfra_extraction_run_at"] = utc_now()
    out["postfra_llm_run_at"] = ""
    out = out.sort_values("n_ce_shaped", ascending=False).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(out.to_string(index=False))
    print(f"\nOverall post-FRA CE-shaped FONSIs: {int(is_post.sum())} of {len(ce)} CE-shaped "
          f"({int(is_dated.sum())} dated).")
    print(CAVEAT)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
