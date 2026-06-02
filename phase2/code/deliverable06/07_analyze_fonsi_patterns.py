import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import json
from html import escape

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now, write_parquet


TAXONOMY_PATH = D6_ANALYSIS_DIR / "fonsi_archetype_taxonomy.parquet"
ASSIGNMENTS_PATH = D6_ANALYSIS_DIR / "project_action_archetypes.parquet"
PROJECTS_PATH = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
ACTIONS_PATH = D6_ANALYSIS_DIR / "fonsi_actions.parquet"
CONDITIONS_PATH = D6_ANALYSIS_DIR / "fonsi_conditions.parquet"
CROSSWALK_PATH = D6_ANALYSIS_DIR / "ce_crosswalk.parquet"
CANDIDATES_PATH = D6_ANALYSIS_DIR / "fonsi_candidate_categories.parquet"
MATRIX_PATH = D6_OUTPUT_DIR / "fonsi_opportunity_matrix.csv"
SCAN_HTML = D6_OUTPUT_DIR / "fonsi_opportunity_scan.html"
ANALYSIS_VERSION = "d6_stage_a_v1"


def share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def json_list(values, limit: int = 12) -> str:
    return json.dumps([str(value) for value in list(values)[:limit]])


def load_optional(path, columns=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_parquet(path)


def scale_summary(group: pd.DataFrame) -> str:
    summary = {}
    for col in ("max_acres", "max_miles", "max_megawatts", "max_kilovolts", "max_wells"):
        values = pd.to_numeric(group.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if not values.empty:
            summary[col] = {
                "n": int(len(values)),
                "median": round(float(values.median()), 2),
                "p90": round(float(values.quantile(0.9)), 2),
                "max": round(float(values.max()), 2),
            }
    road_count = int(group.get("has_no_new_access_road_constraint", pd.Series(dtype=bool)).fillna(False).sum())
    if road_count:
        summary["no_new_access_road_constraint_projects"] = road_count
    return json.dumps(summary, sort_keys=True)


def crosswalk_best(crosswalk: pd.DataFrame, archetype_id: str, match_type: str) -> dict:
    if crosswalk.empty:
        return {}
    rows = crosswalk.loc[
        crosswalk["archetype_id"].eq(archetype_id)
        & crosswalk["match_type"].eq(match_type)
    ].sort_values(["retrieval_score", "retrieval_rank"], ascending=[False, True])
    return rows.iloc[0].to_dict() if not rows.empty else {}


def render_scan(matrix: pd.DataFrame, run_at: str) -> None:
    display = matrix[
        [
            "archetype_label", "recommendation_tier", "n_fonsi_projects",
            "fonsi_share", "ce_share", "eis_share", "mitigation_dependence_share",
            "assessment_total", "strongest_same_agency_ce", "gating_flags",
        ]
    ].copy()
    table = display.to_html(index=False, escape=True, classes="matrix")
    SCAN_HTML.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>D6 FONSI opportunity scan</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.45rem; vertical-align: top; }}
th {{ background: #f2f4f6; text-align: left; }}
.note {{ max-width: 70rem; color: #444; }}
</style></head><body>
<h1>D6 FONSI Opportunity Scan</h1>
<p>Generated {escape(run_at)}. This is a review aid, not a legal sufficiency determination.</p>
<p class="note">CE Explorer similarity scores rank material for manual review. CE-track
classification is frequently metadata-only, so CE shares must be read together with the
classification-coverage columns in the CSV matrix. EIS share is a composition metric, not
an escalation rate.</p>
{table}
</body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()
    taxonomy = pd.read_parquet(TAXONOMY_PATH)
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)
    projects = pd.read_parquet(PROJECTS_PATH)
    actions = load_optional(ACTIONS_PATH, ["project_id"])
    conditions = load_optional(CONDITIONS_PATH, ["project_id", "condition_role"])
    crosswalk = load_optional(CROSSWALK_PATH, ["archetype_id", "match_type"])

    process_totals = assignments.groupby("process_type")["project_id"].nunique().to_dict()
    specific = assignments.loc[assignments["archetype_id"].ne("other_clean_energy")]
    coverage = specific.groupby("process_type")["project_id"].nunique().to_dict()
    fonsi_ids = set(projects["project_id"].astype(str))
    condition_roles = (
        conditions.groupby("project_id")["condition_role"].agg(lambda values: set(values))
        if not conditions.empty else pd.Series(dtype=object)
    )
    records = []
    for archetype in taxonomy.itertuples(index=False):
        group = assignments.loc[assignments["archetype_id"].eq(archetype.archetype_id)].copy()
        group["project_id_str"] = group["project_id"].astype(str)
        n_by_process = group.groupby("process_type")["project_id"].nunique().to_dict()
        project_ids = group["project_id_str"].drop_duplicates()
        fonsi_project_ids = [pid for pid in project_ids if pid in fonsi_ids]
        fonsi_actions = actions.loc[actions["project_id"].astype(str).isin(fonsi_project_ids)].copy()
        role_sets = condition_roles.loc[condition_roles.index.astype(str).isin(fonsi_project_ids)]
        role_project_count = max(len(fonsi_project_ids), 1)
        has_role = lambda role: sum(role in roles for roles in role_sets)

        n_classified = len(project_ids)
        n_ce = int(n_by_process.get("CE", 0))
        n_ea = int(n_by_process.get("EA", 0))
        n_eis = int(n_by_process.get("EIS", 0))
        n_fonsi = len(fonsi_project_ids)
        method_counts = group["assignment_method"].value_counts().to_dict()
        ce_group = group.loc[group["process_type"].eq("CE")]
        ce_metadata_share = share(int(ce_group["assignment_method"].eq("metadata_only").sum()), len(ce_group))
        mitigation_share = share(has_role("mitigation_commitment"), role_project_count)
        bmp_share = share(has_role("best_management_practice"), role_project_count)
        monitoring_share = share(has_role("monitoring_requirement"), role_project_count)

        same = crosswalk_best(crosswalk, archetype.archetype_id, "same_agency_existing")
        other = crosswalk_best(crosswalk, archetype.archetype_id, "other_agency_adoption_candidate")
        already_covered = bool(
            same and same.get("manual_verification_status") == "verified_existing_coverage"
        )
        comparison_only = archetype.candidate_scope == "comparison_diagnostic"
        high_mitigation = mitigation_share >= 0.5 and n_fonsi >= 5
        insufficient_qa = n_fonsi < 5
        insufficient_coverage = min(
            share(int(coverage.get(process, 0)), int(process_totals.get(process, 0)))
            for process in ("CE", "EA", "EIS")
        ) < 0.5
        unresolved_boundary = n_eis > 0 and n_fonsi < 10

        volume = 2 if n_fonsi >= 20 else (1 if n_fonsi >= 5 else 0)
        state_diversity = group["project_state"].dropna().nunique()
        diversity = 2 if state_diversity >= 8 else (1 if state_diversity >= 3 else 0)
        homogeneity = 0 if archetype.archetype_id == "other_clean_energy" else (
            2 if share(int(group["assignment_confidence"].eq("high").sum()), len(group)) >= 0.5 else 1
        )
        boundaries = 0
        if not fonsi_actions.empty:
            has_scales = fonsi_actions[
                ["max_acres", "max_miles", "max_megawatts", "max_kilovolts", "max_wells"]
            ].notna().any(axis=1).mean()
            has_roads = fonsi_actions["has_no_new_access_road_constraint"].fillna(False).mean()
            boundaries = 2 if max(has_scales, has_roads) >= 0.5 else (1 if max(has_scales, has_roads) > 0 else 0)
        qa = 2 if n_fonsi >= 20 and not fonsi_actions.empty else (1 if n_fonsi >= 5 else 0)
        total = volume + diversity + homogeneity + boundaries + qa

        flags = []
        for flag, active in [
            ("already_covered_by_existing_ce", already_covered),
            ("comparison_only_archetype", comparison_only),
            ("high_case_specific_mitigation_dependence", high_mitigation),
            ("unresolved_boundary_cases", unresolved_boundary),
            ("insufficient_classification_coverage", insufficient_coverage),
            ("insufficient_qa", insufficient_qa),
        ]:
            if active:
                flags.append(flag)
        if already_covered or comparison_only:
            tier = "deprioritize"
        elif flags or total < 8:
            tier = "review" if total >= 4 else "deprioritize"
        else:
            tier = "advance"
        records.append(
            {
                "archetype_id": archetype.archetype_id,
                "archetype_label": archetype.archetype_label,
                "archetype_description": archetype.archetype_description,
                "candidate_scope": archetype.candidate_scope,
                "recommendation_tier": tier,
                "gating_flags": json.dumps(flags),
                "n_projects_classified": n_classified,
                "n_fonsi_projects": n_fonsi,
                "fonsi_share": share(n_fonsi, n_classified),
                "n_ce_projects": n_ce,
                "ce_share": share(n_ce, n_classified),
                "n_ea_projects": n_ea,
                "ea_share": share(n_ea, n_classified),
                "n_eis_projects": n_eis,
                "eis_share": share(n_eis, n_classified),
                "ce_classification_coverage": share(int(coverage.get("CE", 0)), int(process_totals.get("CE", 0))),
                "ea_classification_coverage": share(int(coverage.get("EA", 0)), int(process_totals.get("EA", 0))),
                "eis_classification_coverage": share(int(coverage.get("EIS", 0)), int(process_totals.get("EIS", 0))),
                "assignment_method_distribution": json.dumps(method_counts, sort_keys=True),
                "ce_metadata_only_share": ce_metadata_share,
                "ce_metadata_only_caveat": "CE-track assignments are often metadata-only; manually verify shortlisted CE comparisons.",
                "repeated_limitations_and_scales": scale_summary(fonsi_actions),
                "mitigation_dependence_share": mitigation_share,
                "bmp_dependence_share": bmp_share,
                "monitoring_dependence_share": monitoring_share,
                "strongest_same_agency_ce": normalize_ce(same),
                "strongest_other_agency_ce": normalize_ce(other),
                "representative_project_ids": json_list(fonsi_project_ids),
                "assessment_evidence_volume": volume,
                "assessment_evidence_diversity": diversity,
                "assessment_action_homogeneity": homogeneity,
                "assessment_enforceable_boundaries": boundaries,
                "assessment_qa_confidence": qa,
                "assessment_total": total,
                "qa_notes": "Manual inventory, archetype, extraction, and CE crosswalk review gates remain open.",
                "taxonomy_version": archetype.taxonomy_version,
                "analysis_version": ANALYSIS_VERSION,
                "analysis_run_at": run_at,
            }
        )
    matrix = pd.DataFrame(records).sort_values(
        ["recommendation_tier", "assessment_total", "n_fonsi_projects"],
        ascending=[True, False, False],
    )
    write_parquet(matrix, CANDIDATES_PATH)
    matrix.to_csv(MATRIX_PATH, index=False)
    render_scan(matrix, run_at)
    print(f"wrote {len(matrix):,} opportunity rows -> {MATRIX_PATH}")


def normalize_ce(row: dict) -> str:
    if not row:
        return ""
    return " | ".join(
        str(row.get(field, ""))
        for field in ("agency_unit", "structured_id", "ce_description", "canonical_source_url")
        if str(row.get(field, ""))
    )


if __name__ == "__main__":
    main()
