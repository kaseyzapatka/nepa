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

    # 12. G1 — client develop shortlist obeys the recurrence gate (main >= 5, exploratory 3-4, no < 3)
    if new_csv is not None and "shortlist_tier" in new_csv.columns and "n_profile_fonsi" in new_csv.columns:
        below = new_csv[new_csv["n_profile_fonsi"] < 3]
        check(len(below) == 0, f"G1: no sub-floor (< 3 CE-shaped) cell in d6_new.csv (got {len(below)})")
        tiers = set(new_csv["shortlist_tier"].unique())
        check(tiers <= {"main", "exploratory"}, f"G1: d6_new.csv tiers subset of main/exploratory (got {tiers})")
        main_bad = new_csv[(new_csv["shortlist_tier"] == "main") & (new_csv["n_profile_fonsi"] < 5)]
        check(len(main_bad) == 0, f"G1: every main-tier cell has >= 5 CE-shaped (violations: {len(main_bad)})")

    # 13. #38 — annotate-only crosswalk: net/gross columns exist, net ⊆ gross, and it moves NO verdict.
    # (Intent is "the crosswalk changes no verdicts" — expressed as net⊆gross consistency, NOT a hard
    # adopt count, so it doesn't fight the A1 coverage gate which legitimately flips one adopt cell.)
    if verd is not None and {"adopt_targets_net", "adopt_targets_gross"} <= set(verd.columns):
        def _toks(s):
            return {t.strip() for t in str(s).split(",") if t.strip()}
        subset_ok = all(_toks(n) <= _toks(g) for n, g in
                        zip(verd["adopt_targets_net"], verd["adopt_targets_gross"]))
        check(subset_ok, "#38: adopt_targets_net ⊆ adopt_targets_gross for every cell (crosswalk annotate-only)")
        # gross is the deterministic adopt-gap snapshot: it equals adopt_targets on every row (the crosswalk
        # only narrows to net, never widens gross) — this is the "crosswalk moves no verdict" invariant.
        gross_matches_baseline = all(
            _toks(g) == _toks(a) for g, a in zip(verd["adopt_targets_gross"], verd["adopt_targets"]))
        check(gross_matches_baseline, "#38: adopt_targets_gross == deterministic adopt_targets (no verdict move)")

    # 16. A1/#37 — eCFR coverage gate invariants (replaces the old hard adopt==22 assertion)
    cov = _load(D / "candidate_ce_coverage.parquet")
    if verd is not None and cov is not None and "cell_best_coverage" in verd.columns \
            and cov["coverage_verdict"].fillna("").ne("").any():
        orig = cov.groupby("candidate_category")["verdict"].first()   # pre-gate verdict snapshot
        flips = verd[verd["cell_best_coverage"] == "does_not_cover"]
        adopt_flips = sum(1 for cat in flips["candidate_category"] if orig.get(cat) == "adopt")
        n_adopt_now = int((verd["verdict"] == "adopt").sum())
        # baseline reconciliation: deterministic adopt (22) == post-gate adopt + adopt-flips
        check(n_adopt_now + adopt_flips == 22,
              f"A1: post-gate adopt ({n_adopt_now}) + adopt-flips ({adopt_flips}) == 22 baseline")
        # every flip traces to a does_not_cover cell-best (and only those flipped)
        check((flips["verdict"] == "new").all(),
              "A1: every does_not_cover cell was flipped to new")
        kept = verd[verd["cell_best_coverage"].isin(["covers", "partially_covers"])]
        check((kept["verdict"].isin(["adopt", "expand"])).all(),
              "A1: covers/partial cells kept their adopt/expand verdict (no spurious flip)")
        # every "verified" confidence cell has ≥1 covers row from verified eCFR-current text
        cov_covers = cov[(cov["coverage_verdict"] == "covers") & (cov["source_type"] == "ecfr_current")]
        verified_cats = set(verd.loc[verd["verdict_confidence"] == "verified", "candidate_category"])
        check(verified_cats <= set(cov_covers["candidate_category"]),
              "A1: every 'verified' cell has ≥1 verified eCFR-current 'covers' row")
        # no needs_review adopt cell in the client adopt list without its flag carried
        adopt_csv = _load(REVIEW / "d6_adopt.csv")
        if adopt_csv is not None and "needs_review" in adopt_csv.columns:
            nr = verd[(verd["needs_review"]) & (verd["verdict"] == "adopt")]
            in_csv = set(adopt_csv.loc[adopt_csv["needs_review"] == True, "candidate_category"])
            check(set(nr["candidate_category"]) <= in_csv,
                  "A1: every needs_review adopt cell is flagged in the client adopt list")

    # 14. #40 — the other-action theme table is terminal (does not alter cell membership)
    other_themes = _load(D / "other_action_themes.parquet")
    if other_themes is not None and facts is not None:
        n_other = facts.loc[facts["action"] == "other", "project_id"].nunique() if "action" in facts.columns else -1
        check(other_themes["project_id"].nunique() == n_other,
              f"#40: theme table covers exactly the action=='other' projects ({other_themes['project_id'].nunique()} vs {n_other})")

    # 15. #47 — condition resource_area enum stays within the shared 12 + unknown (no 'vegetation' leak).
    # (fonsi_conditions is rebuilt out-of-band by retag_condition_resources.py; this guards D2 alignment.)
    cond = _load(D / "fonsi_conditions.parquet")
    if cond is not None and "resource_area" in cond.columns:
        shared12 = {"air_quality", "water", "biological", "cultural", "visual", "noise", "soils_geology",
                    "socioeconomic", "transportation", "land_use", "climate_ghg", "public_health", "unknown"}
        vals = set(cond["resource_area"].dropna().unique())
        check(vals <= shared12, f"#47: condition resource_area within shared 12 + unknown (stray: {vals - shared12})")

    print(f"\n[qa] {'PASS' if not fails else 'FAIL (' + str(len(fails)) + ')'}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
