"""D2 Phase 1 — build the three-tier significance corpus + cohort table (plan v2.11 §3, R1).

Tiers:
  mitigated_fonsi  — clean EA-source FONSI, recall-screen: a textual mitigated-finding cue
                     (finding spans) co-occurring with >=1 enforceable mitigation condition.
  straight_fonsi   — the complement within the 452.
  eis_significant  — clean EIS projects (significance findings extracted later).

Two-stage mitigated flag (plan §3): THIS script is the recall-oriented Gate-1 screen. The
precise dual-signal page-window join (frozen `mitigation_flag`) is computed in 02. Gate 1 sees
strict-same-section AND windowed-±2 counts side by side (columns + printed) so it is bounded.

Also emits (Round 5): `agency_scope_status` ∈ {primary_blm_doe_family, context_other_agency,
manual_scope_review} on ALL tiers (the headline-denominator gate §4/§6 depend on), keeping the
coarse `agency` as a display label; broadened `off_mission_flag`; `time_scope_status` incl.
`boundary_review`; and `project_cohorts.parquet` (A4).

Run:  conda run -n nepa python phase2/code/deliverable02/01_build_d2_inventory.py
Out:  phase2/data/analysis/deliverable02/significance_corpus.parquet
      phase2/data/analysis/deliverable02/project_cohorts.parquet
      phase2/output/deliverable02/corpus_membership_review.csv
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

import common as C
from significance_taxonomy import (
    MITIGATED_SCREEN_CUES, MITIGATION_OBLIGATIONS, MITIGATION_ROLES,
)

ARRA = date(2009, 2, 17)
BOUNDARY_DAYS = 90
MITIGATED_CUE = "|".join(MITIGATED_SCREEN_CUES)
ROLES_SQL = ",".join(f"'{r}'" for r in MITIGATION_ROLES)
OBLIG_SQL = ",".join(f"'{o}'" for o in MITIGATION_OBLIGATIONS)


def _parse(d) -> date | None:
    s = str(d)[:10]
    if s in ("", "NaT", "None", "nan"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _time_scope(d) -> str:
    pd_ = _parse(d)
    if pd_ is None:
        return "missing_decision_date"
    if abs((pd_ - ARRA).days) <= BOUNDARY_DAYS:
        return "boundary_review"
    return "pre_ARRA_dated" if pd_ < ARRA else "in_scope_dated"


def _agency(lead: str, dept: str) -> str:
    """Coarse display label (kept for the BLM-vs-DOE-subagency cut)."""
    lead, dept = str(lead or ""), str(dept or "")
    if "Bureau of Land Management" in lead:
        return "BLM"
    if "Department of Energy" in dept:
        return "DOE-family"
    return "other"


def _agency_scope_status(lead: str, dept: str) -> str:
    """The headline-denominator inclusion gate (plan §3/§4/§6, A1 rule)."""
    lead, dept = str(lead or ""), str(dept or "")
    if lead.strip() in ("", "[]", "None", "nan"):
        return "manual_scope_review"
    if "Bureau of Land Management" in lead or dept == "Department of Energy":
        return "primary_blm_doe_family"
    return "context_other_agency"


def _off_mission(lead, f_nuc, f_nucwaste, f_mil, f_broadband, f_util) -> bool:
    """Full A1 off-mission screen: five project exclusion flags + lead-agency string cues."""
    lead = str(lead or "")
    flag = any(bool(x) for x in (f_nuc, f_nucwaste, f_mil, f_broadband, f_util))
    string_cue = ("National Nuclear" in lead) or ("Defense Activities" in lead) or ("Laboratory" in lead)
    return flag or string_cue


def fonsi_tier() -> pd.DataFrame:
    sql = f"""
    WITH inv AS (
        SELECT project_id, project_title, project_description,
               lead_agency_harmonized, tech_group, project_state
        FROM read_parquet('{C.FONSI_INVENTORY}')
        WHERE project_energy_type = 'Clean' AND stage_a_ea_source = TRUE
    ),
    dep AS (
        SELECT project_id, project_department,
               project_is_nuclear_tech_only, project_nuclear_waste_to_exclude,
               project_military_to_exclude, project_is_utilities_broadband_only,
               project_utilities_to_exclude
        FROM read_parquet('{C.PROJECTS_COMBINED}')
    ),
    cue_find AS (  -- finding-span mitigated cues (frozen contract: finding only)
        SELECT project_id, document_id, section_id, page_start, page_end
        FROM read_parquet('{C.FONSI_SPANS}')
        WHERE span_type = 'finding' AND regexp_matches(lower(span_text), '{MITIGATED_CUE}')
    ),
    cue AS (SELECT DISTINCT project_id, TRUE AS mitigated_cue_hit FROM cue_find),
    qual_cond AS (
        SELECT project_id, document_id, section_id, page_number
        FROM read_parquet('{C.FONSI_CONDITIONS}')
        WHERE condition_role IN ({ROLES_SQL}) AND obligation_level IN ({OBLIG_SQL})
    ),
    cond AS (SELECT project_id, count(*) AS n_enforceable_conditions FROM qual_cond GROUP BY 1),
    strict AS (  -- cue finding + qualifying condition in the SAME section
        SELECT DISTINCT cf.project_id, TRUE AS mitigated_strict_same_section
        FROM cue_find cf JOIN qual_cond qc
          ON cf.project_id = qc.project_id AND cf.document_id = qc.document_id
         AND cf.section_id = qc.section_id
    ),
    windowed AS (  -- cue finding + qualifying condition within page_start/end +/- 2
        SELECT DISTINCT cf.project_id, TRUE AS mitigated_windowed_pm2
        FROM cue_find cf JOIN qual_cond qc
          ON cf.project_id = qc.project_id AND cf.document_id = qc.document_id
         AND qc.page_number BETWEEN cf.page_start - 2 AND cf.page_end + 2
    ),
    dt AS (
        SELECT project_id, decision_date, decision_confidence, decision_is_proxy
        FROM read_parquet('{C.TIMELINE_DATES}')
    )
    SELECT inv.project_id, inv.project_title, inv.project_description,
           inv.lead_agency_harmonized, inv.tech_group, inv.project_state,
           dep.project_department,
           dep.project_is_nuclear_tech_only, dep.project_nuclear_waste_to_exclude,
           dep.project_military_to_exclude, dep.project_is_utilities_broadband_only,
           dep.project_utilities_to_exclude,
           coalesce(cue.mitigated_cue_hit, FALSE) AS mitigated_cue_hit,
           coalesce(cond.n_enforceable_conditions, 0) AS n_enforceable_conditions,
           coalesce(strict.mitigated_strict_same_section, FALSE) AS mitigated_strict_same_section,
           coalesce(windowed.mitigated_windowed_pm2, FALSE) AS mitigated_windowed_pm2,
           dt.decision_date, dt.decision_confidence, dt.decision_is_proxy
    FROM inv
    LEFT JOIN dep USING (project_id)
    LEFT JOIN cue USING (project_id)
    LEFT JOIN cond USING (project_id)
    LEFT JOIN strict USING (project_id)
    LEFT JOIN windowed USING (project_id)
    LEFT JOIN dt USING (project_id)
    """
    df = C.q(sql)
    df["process_type"] = "EA"
    df["doc_type"] = "FONSI"
    has_cond = df["n_enforceable_conditions"] > 0
    df["corpus_tier"] = (df["mitigated_cue_hit"] & has_cond).map(
        {True: "mitigated_fonsi", False: "straight_fonsi"})
    df["fonsi_subtype"] = "mitigated_dual_signal"
    df.loc[df["corpus_tier"] == "straight_fonsi", "fonsi_subtype"] = (
        df.loc[df["corpus_tier"] == "straight_fonsi", "n_enforceable_conditions"]
        .gt(0).map({True: "design_feature_or_partial", False: "no_mitigation_signal"}))
    return df


def eis_tier() -> pd.DataFrame:
    sql = f"""
    WITH p AS (
        SELECT project_id, project_title, project_description,
               lead_agency_harmonized, project_department, project_state,
               project_is_nuclear_tech_only, project_nuclear_waste_to_exclude,
               project_military_to_exclude, project_is_utilities_broadband_only,
               project_utilities_to_exclude
        FROM read_parquet('{C.PROJECTS_COMBINED}')
        WHERE project_energy_type = 'Clean' AND process_type = 'EIS'
    ),
    tech AS (  -- enrich EIS tech_group from document_sections (not in projects_combined)
        SELECT project_id, any_value(tech_group) AS tech_group
        FROM read_parquet('{C.DOCUMENT_SECTIONS}') GROUP BY 1
    ),
    dt AS (
        SELECT project_id, decision_date, decision_confidence, decision_is_proxy
        FROM read_parquet('{C.TIMELINE_DATES}')
    )
    SELECT p.project_id, p.project_title, coalesce(p.project_description, '') AS project_description,
           p.lead_agency_harmonized, coalesce(tech.tech_group, '') AS tech_group,
           coalesce(p.project_state, '') AS project_state, p.project_department,
           p.project_is_nuclear_tech_only, p.project_nuclear_waste_to_exclude,
           p.project_military_to_exclude, p.project_is_utilities_broadband_only,
           p.project_utilities_to_exclude,
           FALSE AS mitigated_cue_hit, 0 AS n_enforceable_conditions,
           FALSE AS mitigated_strict_same_section, FALSE AS mitigated_windowed_pm2,
           dt.decision_date, dt.decision_confidence, dt.decision_is_proxy
    FROM p LEFT JOIN tech USING (project_id) LEFT JOIN dt USING (project_id)
    """
    df = C.q(sql)
    df["process_type"] = "EIS"
    df["doc_type"] = "FEIS"
    df["corpus_tier"] = "eis_significant"
    df["fonsi_subtype"] = ""
    return df


def build_cohorts(corpus: pd.DataFrame) -> pd.DataFrame:
    """A4 cohort table: cohort_by_date (frozen bins) + separate D5 law_cited_* flags."""
    laws = C.q(f"""
        SELECT project_id,
               max(CASE WHEN law_name='ARRA' THEN 1 ELSE 0 END)::BOOLEAN AS law_cited_arra,
               max(CASE WHEN law_name='BIL' THEN 1 ELSE 0 END)::BOOLEAN AS law_cited_bil,
               max(CASE WHEN law_name='IRA' THEN 1 ELSE 0 END)::BOOLEAN AS law_cited_ira,
               max(CASE WHEN law_name='DOE_funding' THEN 1 ELSE 0 END)::BOOLEAN AS law_cited_doe_funding
        FROM read_parquet('{C.LAW_CITATIONS}') GROUP BY 1
    """)
    coh = corpus[["project_id", "process_type", "decision_date",
                  "agency_scope_status", "agency_scope_rule"]].copy()
    coh["cohort_by_date"] = coh["decision_date"].map(_cohort_by_date)
    coh = coh.merge(laws, on="project_id", how="left")
    for c in ("law_cited_arra", "law_cited_bil", "law_cited_ira", "law_cited_doe_funding"):
        coh[c] = coh[c].fillna(False).astype(bool)
    coh["cohort_run_at"] = C.utc_now()
    coh["schema_version"] = C.SCHEMA_VERSION
    return coh.drop(columns=["decision_date"])


def _cohort_by_date(d) -> str:
    pd_ = _parse(d)
    if pd_ is None:
        return "missing_decision_date"
    if pd_ < ARRA:
        return "pre_ARRA"
    bil, ira, fra = date(2021, 11, 15), date(2022, 8, 16), date(2023, 6, 3)
    if pd_ < bil:
        return "arra_to_bil"        # [ARRA, BIL)
    if pd_ < ira:
        return "bil_to_ira"         # [BIL, IRA)
    if pd_ < fra:
        return "ira_to_fra"         # [IRA, FRA)
    return "post_fra"               # [FRA, present]


def main() -> None:
    print("D2 Phase 1: building significance corpus + cohorts ...")
    df = pd.concat([fonsi_tier(), eis_tier()], ignore_index=True)

    df["agency"] = [_agency(l, d) for l, d in zip(df["lead_agency_harmonized"], df["project_department"])]
    df["agency_scope_status"] = [
        _agency_scope_status(l, d) for l, d in zip(df["lead_agency_harmonized"], df["project_department"])]
    df["agency_scope_rule"] = "blm_plus_doe_family"
    df["off_mission_flag"] = [
        _off_mission(l, a, b, c, e, f) for l, a, b, c, e, f in zip(
            df["lead_agency_harmonized"], df["project_is_nuclear_tech_only"],
            df["project_nuclear_waste_to_exclude"], df["project_military_to_exclude"],
            df["project_is_utilities_broadband_only"], df["project_utilities_to_exclude"])]
    df["time_scope_status"] = df["decision_date"].map(_time_scope)
    df["analysis_scope"] = df["time_scope_status"].map(
        lambda s: "primary" if s == "in_scope_dated" else "context_or_validation")
    df["corpus_run_at"] = C.utc_now()
    df["schema_version"] = C.SCHEMA_VERSION

    cols = ["project_id", "process_type", "doc_type", "corpus_tier", "fonsi_subtype",
            "mitigated_cue_hit", "n_enforceable_conditions",
            "mitigated_strict_same_section", "mitigated_windowed_pm2",
            "agency", "agency_scope_status", "agency_scope_rule", "off_mission_flag",
            "time_scope_status", "analysis_scope",
            "decision_date", "decision_confidence", "decision_is_proxy",
            "lead_agency_harmonized", "tech_group", "project_state",
            "project_title", "project_description", "corpus_run_at", "schema_version"]
    df = df[cols]
    C.write_parquet(df, C.SIGNIFICANCE_CORPUS, "corpus")

    cohorts = build_cohorts(df)
    C.write_parquet(cohorts, C.PROJECT_COHORTS, "cohorts")

    review = df[df["corpus_tier"] != "straight_fonsi"].copy()  # mitigated + EIS lists for Gate 1
    review["project_description"] = review["project_description"].str.slice(0, 240)
    C.write_csv(review[[
        "project_id", "corpus_tier", "agency", "agency_scope_status", "off_mission_flag",
        "mitigated_cue_hit", "n_enforceable_conditions", "mitigated_strict_same_section",
        "mitigated_windowed_pm2", "time_scope_status", "decision_date",
        "tech_group", "project_title", "project_description"]],
        C.D2_OUTPUT_DIR / "corpus_membership_review.csv", "Gate 1/2 review")

    # --- Gate-1 reporting ---
    print("\ncorpus_tier x time_scope_status:")
    print(pd.crosstab(df["corpus_tier"], df["time_scope_status"]).to_string())
    print("\nagency_scope_status x corpus_tier:")
    print(pd.crosstab(df["agency_scope_status"], df["corpus_tier"]).to_string())
    fonsi = df[df.process_type == "EA"]
    prim = df[(df.corpus_tier == "mitigated_fonsi") & (df.analysis_scope == "primary")]
    print("\nmitigated-FONSI signal counts (Gate 1 — strict vs windowed vs screen):")
    print(f"  recall screen (cue & cond anywhere): {int((df.corpus_tier=='mitigated_fonsi').sum())}"
          f"  (primary/in-scope={len(prim)})")
    print(f"  strict same-section              : {int(fonsi['mitigated_strict_same_section'].sum())}")
    print(f"  windowed +/-2 pages              : {int(fonsi['mitigated_windowed_pm2'].sum())}")
    print("\ncohort_by_date:")
    print(cohorts["cohort_by_date"].value_counts().to_string())


if __name__ == "__main__":
    main()
