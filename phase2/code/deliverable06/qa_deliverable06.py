"""D6 QA gate (v3 — tech_group x action grid) — assert the pipeline outputs are consistent.

Run after the chain (10_action_label -> 04 -> 05 -> 06 -> 09 -> 07 -> 08). Fails loudly on the
invariants that matter for the client-facing report under the refactor:
  - every clean FONSI is categorized into a `tech_group__action` grid cell (no "other" black hole);
  - the cell id is exactly `f"{tech_group}__{action}"`;
  - `is_codifiable` is derived correctly from the action verb;
  - every cell has a verdict, and verdicts are drawn from the allowed set;
  - the client develop shortlist (d6_new.csv) contains NO non-codifiable cell;
  - the classification stage ran cleanly and enrichment coverage matches the report.

Usage:  CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable06/qa_deliverable06.py
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa'.")

from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent.parent / "data" / "analysis" / "deliverable06"
REVIEW = Path(__file__).resolve().parent.parent.parent / "output" / "deliverable06" / "review"

# actions that a CE cannot codify (a CE encodes a physical action, not funding/manufacturing/admin)
NON_CODIFIABLE = {"manufacturing", "land_or_row_authorization"}
VERDICTS_OK = {"new", "expand", "adopt", "already_covered"}

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def _load(path: Path):
    try:
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        check(False, f"could not read {path.name}: {e}")
        return None


def main() -> None:
    print("[qa] D6 invariants (v3 — tech_group x action grid)")
    facts = _load(D / "candidate_facts.parquet")
    enr = _load(D / "fonsi_enrichment.parquet")
    verd = _load(D / "candidate_verdicts.parquet")
    labels = _load(D / "fonsi_action_labels.parquet")
    cstats = _load(D / "corpus_mitigation_stats.parquet")
    new_csv = _load(REVIEW / "d6_new.csv")

    # 1. schema: the grid columns are present
    if facts is not None:
        for col in ("tech_group", "action", "candidate_category", "is_ce_shaped",
                    "is_codifiable", "citation_verified"):
            check(col in facts.columns, f"candidate_facts has `{col}`")

    # 2. cell id == tech_group__action for every row
    if facts is not None and {"tech_group", "action", "candidate_category"} <= set(facts.columns):
        expect_cell = facts["tech_group"].astype(str) + "__" + facts["action"].astype(str)
        check(bool((facts["candidate_category"] == expect_cell).all()),
              "candidate_category == f'{tech_group}__{action}' for every row")

    # 3. coverage: every enriched FONSI is categorized (no drop-to-other)
    if facts is not None and enr is not None:
        n_enriched = int((enr["action_summary"].fillna("").str.len() > 0).sum())
        n_cells = facts["project_id"].nunique()
        check(n_cells == n_enriched,
              f"every enriched FONSI has a grid cell ({n_cells} facts vs {n_enriched} enriched)")

    # 4. is_codifiable derived correctly from the verb
    if facts is not None and {"action", "is_codifiable"} <= set(facts.columns):
        bad = facts.loc[facts["action"].isin(NON_CODIFIABLE) & (facts["is_codifiable"] != False)]
        check(len(bad) == 0, f"non-codifiable verbs have is_codifiable==False (violations: {len(bad)})")
        bad2 = facts.loc[~facts["action"].isin(NON_CODIFIABLE) & (facts["is_codifiable"] != True)]
        check(len(bad2) == 0, f"codifiable verbs have is_codifiable==True (violations: {len(bad2)})")

    # 5. labels join cleanly onto the enrichment
    if labels is not None and enr is not None:
        check(set(enr.loc[enr["action_summary"].fillna("").str.len() > 0, "project_id"].astype(str))
              <= set(labels["project_id"].astype(str)),
              "every enriched FONSI has an action label")

    # 6. every cell has a verdict; verdicts are from the allowed set
    if verd is not None and facts is not None:
        cells_f = set(facts["candidate_category"].unique())
        cells_v = set(verd["candidate_category"].unique())
        check(cells_f <= cells_v, f"every facts cell has a verdict ({len(cells_f - cells_v)} missing)")
        check(set(verd["verdict"].unique()) <= VERDICTS_OK,
              f"verdicts subset of {VERDICTS_OK} (got {set(verd['verdict'].unique())})")

    # 7. the CLIENT develop shortlist excludes non-codifiable cells
    if new_csv is not None and "is_codifiable" in new_csv.columns:
        nbad = int((new_csv["is_codifiable"] == False).sum())
        check(nbad == 0, f"no non-codifiable cell in the client develop shortlist d6_new.csv (got {nbad})")

    # 8. classification stage ran cleanly (unchanged from the enrichment; 03 not re-run)
    if enr is not None and "classification_parse_ok" in enr.columns:
        n_clf_ok = int(enr["classification_parse_ok"].sum())
        check(n_clf_ok == 451, f"classification parse_ok == 451 (got {n_clf_ok})")

    # 9. enrichment coverage matches the report denominators
    if enr is not None:
        n_enriched = int((enr["action_summary"].fillna("").str.len() > 0).sum())
        check(n_enriched == 451, f"enriched == 451 (got {n_enriched})")

    # 10. quote-verification stays high on the CE-shaped subset
    if facts is not None and "citation_verified" in facts.columns:
        sub = facts.loc[facts["is_ce_shaped"]] if "is_ce_shaped" in facts.columns else facts
        vr = sub["citation_verified"].mean() if len(sub) else 1.0
        check(vr >= 0.90, f"quote-verification rate >= 90% on CE-shaped (got {vr:.1%})")

    # 11. renamed stats columns
    if cstats is not None:
        check("n_case_specific_dependent" in cstats.columns and "n_design_or_none" in cstats.columns,
              "corpus_mitigation_stats uses renamed LLM columns")

    print(f"\n[qa] {'PASS' if not fails else 'FAIL (' + str(len(fails)) + ')'}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
