"""D6 QA gate — assert the pipeline outputs are internally consistent.

Run after the chain (03 --stage classify -> 09 -> 07 -> 08). Fails loudly on the invariants that
matter for the client-facing report: the candidate set is keyed on the corrected LLM
`action_category` (no rule-vs-LLM mismatch), the CE-shaped (Rule B) totals match the headline,
the verdicts match the CE-shaped project counts, the classification stage ran cleanly (451
parsed, all v2), enrichment coverage is what the report claims, and the renamed stats columns
exist. Update EXPECT / the CE-shaped total when a new classify pass legitimately changes them.

Usage:  CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable06/qa_deliverable06.py
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa'.")

from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent.parent / "data" / "analysis" / "deliverable06"
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main() -> None:
    facts = pd.read_parquet(D / "candidate_facts.parquet")
    enr = pd.read_parquet(D / "fonsi_enrichment.parquet")
    verd = pd.read_parquet(D / "candidate_verdicts.parquet")
    cstats = pd.read_parquet(D / "corpus_mitigation_stats.parquet")

    print("[qa] D6 invariants (corrected action_category model)")
    CANDS = {"transmission_upgrade", "solar", "geothermal_exploration",
             "temporary_resource_assessment", "wind_onshore"}
    EXPECT = {"transmission_upgrade": 34, "solar": 9, "geothermal_exploration": 9,
              "wind_onshore": 7, "temporary_resource_assessment": 4}   # CE-shaped per category

    # 1. schema: the corrected-category + CE-shaped columns are present
    for col in ("citation_verified", "is_ce_shaped", "action_category", "candidate_category"):
        check(col in facts.columns, f"candidate_facts has `{col}`")

    # 2. the LLM action_category IS the categorizer now — no rule-vs-LLM mismatch by construction
    check(bool((facts["candidate_category"] == facts["action_category"]).all()),
          "candidate_category == action_category for every candidate row (LLM is the categorizer)")
    check(set(facts["candidate_category"].unique()) <= CANDS, "every candidate category is an in-scope type")

    # 3. CE-shaped (Rule B) total + per-category match the report headline
    n_shaped = int(facts["is_ce_shaped"].sum())
    check(n_shaped == 63, f"CE-shaped (Rule B) == 63 (got {n_shaped})")
    bycat = facts.loc[facts["is_ce_shaped"]].groupby("candidate_category")["project_id"].nunique().to_dict()
    check(bycat == EXPECT, f"CE-shaped by category == {EXPECT} (got {bycat})")

    # 4. Rule B's transmission gate: every CE-shaped transmission row is modify-existing within ROW
    txs = facts.loc[facts["candidate_category"].eq("transmission_upgrade") & facts["is_ce_shaped"]]
    check(bool((txs["within_existing_row"] == True).all()),
          "every CE-shaped transmission row is within_existing_row (Rule B gate)")

    # 5. verdict n_profile_fonsi must equal the CE-shaped project count per category
    for _, r in verd.iterrows():
        cat = r["candidate_category"]
        nb = facts.loc[facts["is_ce_shaped"] & facts["candidate_category"].eq(cat), "project_id"].nunique()
        check(int(r["n_profile_fonsi"]) == nb,
              f"{cat}: verdict n_profile_fonsi ({int(r['n_profile_fonsi'])}) == CE-shaped projects ({nb})")

    # 6. classification stage ran cleanly: every read parsed, all stamped v2
    n_clf_ok = int(enr["classification_parse_ok"].sum())
    check(n_clf_ok == 451, f"classification parse_ok == 451 (got {n_clf_ok})")
    ok = enr.loc[enr["classification_parse_ok"]]
    check(bool((ok["classification_prompt_version"] == "d6_classify_prompt_v2").all()),
          "all classified rows stamped d6_classify_prompt_v2")

    # 7. enrichment coverage matches the report's denominators
    n_enriched = int((enr["action_summary"].fillna("").str.len() > 0).sum())
    check(n_enriched == 451, f"enriched == 451 (got {n_enriched}); {len(enr) - n_enriched} no-evidence")

    # 8. quote-verification rate stays high
    vr = facts["citation_verified"].mean()
    check(vr >= 0.90, f"quote-verification rate >= 90% (got {vr:.1%})")

    # 9. renamed stats columns (not the old regex-era names)
    check("n_case_specific_dependent" in cstats.columns and "n_design_or_none" in cstats.columns,
          "corpus_mitigation_stats uses renamed LLM columns")

    print(f"\n[qa] {'PASS' if not fails else 'FAIL (' + str(len(fails)) + ')'}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
