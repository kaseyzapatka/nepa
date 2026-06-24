"""D2 Phase 1 — build the three-tier significance corpus (Gate 1/2 eyeball list).

Tiers:
  mitigated_fonsi  — clean EA-source FONSI with a textual mitigated-finding cue AND
                     >=1 enforceable mitigation condition (role+obligation qualify).
  straight_fonsi   — the complement within the 452.
  eis_significant  — clean EIS projects (significance findings extracted later).

Also tags: agency (BLM / DOE-family / other), time_scope_status, analysis_scope.
NOTE: the mitigated flag here is a project-level co-occurrence screen for Gate 1.
The frozen extractor uses a page-window join (plan Phase 1); Gate 1 reviews this list.

Run:  CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable02/01_build_d2_inventory.py
Out:  phase2/data/analysis/deliverable02/significance_corpus.parquet
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
    return "pre_ARRA_dated" if pd_ < ARRA else "in_scope_dated"


def _agency(lead: str, dept: str) -> str:
    lead, dept = str(lead or ""), str(dept or "")
    if "Bureau of Land Management" in lead:
        return "BLM"
    if "Department of Energy" in dept:
        return "DOE-family"
    return "other"


def _off_mission(lead: str) -> bool:
    lead = str(lead or "")
    return ("National Nuclear" in lead) or ("Defense Activities" in lead)


def fonsi_tier() -> pd.DataFrame:
    sql = f"""
    WITH inv AS (
        SELECT project_id, project_title, project_description,
               lead_agency_harmonized, tech_group, project_state
        FROM read_parquet('{C.FONSI_INVENTORY}')
        WHERE project_energy_type = 'Clean' AND stage_a_ea_source = TRUE
    ),
    dep AS (SELECT project_id, project_department FROM read_parquet('{C.PROJECTS_COMBINED}')),
    cue AS (
        SELECT DISTINCT project_id, TRUE AS mitigated_cue_hit
        FROM read_parquet('{C.FONSI_SPANS}')
        WHERE span_type IN ('finding', 'condition')
          AND regexp_matches(lower(span_text), '{MITIGATED_CUE}')
    ),
    cond AS (
        SELECT project_id, count(*) AS n_enforceable_conditions
        FROM read_parquet('{C.FONSI_CONDITIONS}')
        WHERE condition_role IN ({ROLES_SQL})
          AND obligation_level IN ({OBLIG_SQL})
        GROUP BY 1
    ),
    dt AS (
        SELECT project_id, decision_date, decision_confidence, decision_is_proxy
        FROM read_parquet('{C.TIMELINE_DATES}')
    )
    SELECT inv.project_id, inv.project_title, inv.project_description,
           inv.lead_agency_harmonized, inv.tech_group, inv.project_state,
           dep.project_department,
           coalesce(cue.mitigated_cue_hit, FALSE) AS mitigated_cue_hit,
           coalesce(cond.n_enforceable_conditions, 0) AS n_enforceable_conditions,
           dt.decision_date, dt.decision_confidence, dt.decision_is_proxy
    FROM inv
    LEFT JOIN dep USING (project_id)
    LEFT JOIN cue USING (project_id)
    LEFT JOIN cond USING (project_id)
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
        SELECT project_id, lead_agency_harmonized, project_department, project_type
        FROM read_parquet('{C.PROJECTS_COMBINED}')
        WHERE project_energy_type = 'Clean' AND process_type = 'EIS'
    ),
    dt AS (
        SELECT project_id, decision_date, decision_confidence, decision_is_proxy
        FROM read_parquet('{C.TIMELINE_DATES}')
    )
    SELECT p.project_id,
           CAST(p.project_type AS VARCHAR) AS project_title,
           '' AS project_description,
           p.lead_agency_harmonized, '' AS tech_group, '' AS project_state,
           p.project_department,
           FALSE AS mitigated_cue_hit, 0 AS n_enforceable_conditions,
           dt.decision_date, dt.decision_confidence, dt.decision_is_proxy
    FROM p LEFT JOIN dt USING (project_id)
    """
    df = C.q(sql)
    df["process_type"] = "EIS"
    df["doc_type"] = "FEIS"
    df["corpus_tier"] = "eis_significant"
    df["fonsi_subtype"] = ""
    return df


def main() -> None:
    print("D2 Phase 1: building significance corpus ...")
    df = pd.concat([fonsi_tier(), eis_tier()], ignore_index=True)

    df["agency"] = [_agency(l, d) for l, d in zip(df["lead_agency_harmonized"], df["project_department"])]
    df["agency_scope_rule"] = "blm_plus_doe_family"
    df["off_mission_flag"] = df["lead_agency_harmonized"].map(_off_mission)
    df["time_scope_status"] = df["decision_date"].map(_time_scope)
    df["analysis_scope"] = df["time_scope_status"].map(
        lambda s: "primary" if s == "in_scope_dated" else "context_or_validation")
    df["corpus_run_at"] = C.utc_now()
    df["schema_version"] = C.SCHEMA_VERSION

    cols = ["project_id", "process_type", "doc_type", "corpus_tier", "fonsi_subtype",
            "mitigated_cue_hit", "n_enforceable_conditions", "agency", "agency_scope_rule",
            "off_mission_flag", "time_scope_status", "analysis_scope",
            "decision_date", "decision_confidence", "decision_is_proxy",
            "lead_agency_harmonized", "tech_group", "project_state",
            "project_title", "project_description", "corpus_run_at", "schema_version"]
    df = df[cols]
    C.write_parquet(df, C.D2_ANALYSIS_DIR / "significance_corpus.parquet", "corpus")

    review = df[df["corpus_tier"] != "straight_fonsi"].copy()  # mitigated + EIS lists for Gate 1
    review["project_description"] = review["project_description"].str.slice(0, 240)
    C.write_csv(review[[
        "project_id", "corpus_tier", "agency", "off_mission_flag", "mitigated_cue_hit",
        "n_enforceable_conditions", "time_scope_status", "decision_date",
        "tech_group", "project_title", "project_description"]],
        C.D2_OUTPUT_DIR / "corpus_membership_review.csv", "Gate 1/2 review")

    print("\ncorpus_tier x time_scope_status:")
    print(pd.crosstab(df["corpus_tier"], df["time_scope_status"]).to_string())
    print("\nmitigated_fonsi by agency (primary window only):")
    prim = df[(df.corpus_tier == "mitigated_fonsi") & (df.analysis_scope == "primary")]
    print(prim["agency"].value_counts().to_string())
    print(f"\nmitigated_fonsi total={int((df.corpus_tier=='mitigated_fonsi').sum())}  "
          f"(primary/in-scope={len(prim)})")


if __name__ == "__main__":
    main()
