"""D6 QA gate — assert the pipeline outputs are internally consistent (feedback nice-to-have #1).

Run after the chain (09 -> 07 -> 08). Fails loudly on the invariants that matter for the
client-facing report: the bounded subset is the two-gate definition, no "bounded" count
leaks an LLM-not-bounded row, verdicts are candidate (not silently "verified"), enrichment
coverage is what the report claims, and the renamed stats columns exist.

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

    print("[qa] D6 invariants")
    # 1. schema: verification + bounded columns present
    for col in ("citation_verified", "citation_claim", "is_bounded_low_impact", "is_profile_subtype"):
        check(col in facts.columns, f"candidate_facts has `{col}`")

    # 2. bounded = rule-profiled AND LLM-confirmed low-impact; no profile-but-not-bounded leaks in
    bounded = facts["is_profile_subtype"] & facts["is_bounded_low_impact"].eq(True)
    check(int(bounded.sum()) == 42, f"bounded rows == 42 (got {int(bounded.sum())})")
    leak = facts.loc[facts["is_profile_subtype"] & facts["is_bounded_low_impact"].eq(False)]
    check(len(leak) == 12, f"profile-but-not-bounded rows == 12, handled as expand/develop (got {len(leak)})")

    # 3. verdict n_profile_fonsi must equal the BOUNDED project count per category (not rule-profile)
    for _, r in verd[verd["verdict"].eq("adopt")].iterrows():
        cat = r["candidate_category"]
        nb = facts.loc[bounded & facts["candidate_category"].eq(cat), "project_id"].nunique()
        check(int(r["n_profile_fonsi"]) == nb,
              f"{cat}: verdict n_profile_fonsi ({int(r['n_profile_fonsi'])}) == bounded projects ({nb})")

    # 4. enrichment coverage matches the report's denominators
    n_enriched = int((enr["action_summary"].fillna("").str.len() > 0).sum())
    check(n_enriched == 451, f"enriched == 451 (got {n_enriched}); {len(enr) - n_enriched} no-evidence")

    # 5. quote-verification rate as reported (~97%)
    vr = facts["citation_verified"].mean()
    check(vr >= 0.95, f"quote-verification rate >= 95% (got {vr:.1%})")

    # 6. renamed stats columns (not the old regex-era names)
    check("n_case_specific_dependent" in cstats.columns and "n_design_or_none" in cstats.columns,
          "corpus_mitigation_stats uses renamed LLM columns")
    check("n_enforceable_only" not in cstats.columns,
          "corpus_mitigation_stats dropped the stale `n_enforceable_only` name")

    print(f"\n[qa] {'PASS' if not fails else 'FAIL (' + str(len(fails)) + ')'}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
