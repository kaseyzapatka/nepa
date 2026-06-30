"""D6 v2 — 09: wire the LLM enrichment into the report pipeline.

Rebuilds the report's fact + mitigation tables FROM the verified 451-FONSI
enrichment (03_enrich_llm.py -> fonsi_enrichment.parquet), replacing the
deterministic regex/conditions versions from 03 and 05 with the same schemas so
04/06/07/08 and deliverable06.qmd consume them unchanged.

What it swaps to LLM-backed:
  - candidate_facts.parquet       : sizes (acres/miles/MW/kV/WELLS), siting booleans,
                                    mitigation_dependence, action definition, and the
                                    citation now points at the VERIFIED action quote.
  - candidate_mitigation_summary  : mitigated share + recurring resource areas + boundary
                                    statements from significance_thresholds.
  - corpus_mitigation_stats       : corpus-wide mitigated share from is_mitigated_fonsi.

Grain bridge: enrichment is per project; candidate_facts/mitigation are per
(project x candidate_category). We join the per-project enrichment onto the
candidate corpus rows. is_profile_subtype (the taxonomy "recurring subtype" gate)
is kept from the corpus; is_bounded_low_impact (the LLM read) is carried alongside.

Run after 06, before 07:
  CONDA_DEFAULT_ENV=nepa python 09_wire_enrichment.py
"""

import json
import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_REVIEW_DIR, ensure_d6_dirs, normalize_space, utc_now, write_parquet
from candidates import TAXONOMY_VERSION
from prompts import ENRICHMENT_SCHEMA_VERSION

ENRICH = D6_ANALYSIS_DIR / "fonsi_enrichment.parquet"
CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
INVENTORY = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
TIMELINE = D6_ANALYSIS_DIR.parent / "timeline" / "timeline_project_dates.parquet"  # D4 authoritative dates
FACTS_OUT = D6_ANALYSIS_DIR / "candidate_facts.parquet"
MIT_SUMMARY_OUT = D6_ANALYSIS_DIR / "candidate_mitigation_summary.parquet"
CORPUS_STATS_OUT = D6_ANALYSIS_DIR / "corpus_mitigation_stats.parquet"
FACTS_REVIEW = D6_REVIEW_DIR / "candidate_facts_review.csv"

LLM_MODEL = "claude-sonnet-4-6"

# CE-candidate action types = the corrected action_category values in scope, with report labels.
# Everything else the model classifies ("other", etc.) is out of scope for Analysis 1.
CANDS = {
    "transmission_upgrade": "Transmission upgrades (modify existing line)",
    "solar": "Solar electricity generation",
    "geothermal_exploration": "Geothermal exploration",
    "temporary_resource_assessment": "Temporary resource assessment / site investigation",
    "wind_onshore": "Onshore wind (turbines)",
}


def _jlist(v) -> list:
    try:
        x = json.loads(v) if isinstance(v, str) and v.strip().startswith("[") else v
        return x if isinstance(x, list) else []
    except Exception:
        return []


def _b(v) -> bool:
    """enrichment booleans are True/False/None -> coerce None to False (matches 03)."""
    return v is True


def _num(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)


def _action_citation(evidence_cited: str) -> dict:
    """Pull the VERIFIED action quote's provenance (the report's worked-example basis)."""
    def pack(c):
        return {"citation_document_id": c.get("document_id", "") or "",
                "citation_document_role": c.get("document_role", "") or "",
                "citation_evidence_span_id": c.get("span_id", "") or "",
                "citation_page": c.get("page"), "quote": normalize_space(c.get("quote", "")),
                "citation_verified": bool(c.get("verified") is True),
                "citation_claim": c.get("claim", "") or ""}
    for c in _jlist(evidence_cited):
        if c.get("claim") == "action" and c.get("verified") is True:
            return pack(c)
    for want in ("action", "finding"):                  # fall back to any action, then finding quote
        for c in _jlist(evidence_cited):
            if c.get("claim") == want and c.get("quote"):
                return pack(c)
    return {"citation_document_id": "", "citation_document_role": "", "citation_evidence_span_id": "",
            "citation_page": None, "quote": "", "citation_verified": False, "citation_claim": ""}


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()

    en = pd.read_parquet(ENRICH)
    en["project_id"] = en["project_id"].astype(str)
    en = en[en["action_summary"].notna()].copy()        # drop the skipped (no-evidence) rows
    by_pid = {r.project_id: r for r in en.itertuples(index=False)}

    # ---- candidate_facts: ONE row per candidate FONSI, keyed on the CORRECTED action_category ----
    # The keyword candidate_corpus is no longer the categorizer; the LLM's action_category is.
    # is_ce_shaped is Rule B: LLM-bounded AND (transmission must be modify-existing within ROW,
    # no new access road); generation/exploration/temporary gated on the LLM bounded judgment only.
    rows = []
    for e in en.itertuples(index=False):
        cat = getattr(e, "action_category", None)
        if cat not in CANDS:
            continue                                     # not a CE-candidate action type
        cite = _action_citation(getattr(e, "evidence_cited", "[]"))
        areas = ", ".join(_jlist(getattr(e, "mitigation_resource_areas", "[]")))
        wc = getattr(e, "well_count", None)
        bounded = _b(getattr(e, "is_bounded_low_impact", None))
        within_row = _b(getattr(e, "within_existing_row", None))
        new_road = getattr(e, "new_access_road", None)
        ce_shaped = bounded and ((within_row and new_road is not True)
                                 if cat == "transmission_upgrade" else True)
        rows.append({
            "project_id": e.project_id,
            "candidate_category": cat,
            "candidate_label": CANDS[cat],
            "subtype": cat,
            "is_profile_subtype": True,                  # every action_category candidate is in scope now
            "is_bounded_low_impact": bounded,            # LLM read (carried)
            "is_ce_shaped": bool(ce_shaped),             # Rule B: the bounded, CE-shaped candidate
            "candidate_role": "profile",                 # all corrected candidates are actionable (incl. wind)
            "action_category": cat,
            "action_category_pass1": getattr(e, "action_category_pass1", "") or "",
            "classification_confidence": getattr(e, "classification_confidence", "") or "",
            "project_title": getattr(e, "project_title", "") or "",
            "project_type": getattr(e, "project_type", "") or "",
            "is_fonsi": True,                            # every enriched candidate is an EA-source FONSI
            "lead_agency_harmonized": getattr(e, "lead_agency_harmonized", "") or "",
            "project_state": getattr(e, "project_state", "") or "",
            "action_definition": normalize_space(getattr(e, "action_summary", "") or "")[:400],
            "max_acres": _num(getattr(e, "disturbance_acres", None)),
            "max_acres_any": _num(getattr(e, "disturbance_acres", None)),
            "acres_basis": "llm_disturbance" if _num(getattr(e, "disturbance_acres", None)) is not None else "none",
            "max_miles": _num(getattr(e, "line_miles", None)),
            "max_megawatts": _num(getattr(e, "capacity_mw", None)),
            "max_kilovolts": _num(getattr(e, "voltage_kv", None)),
            "n_wells": int(wc) if wc is not None and not (isinstance(wc, float) and pd.isna(wc)) else None,
            "duration": "",
            "within_existing_row": within_row,
            "no_new_access_road": (new_road is False),
            "previously_disturbed_land": _b(getattr(e, "previously_disturbed_land", None)),
            "is_temporary": _b(getattr(e, "is_temporary", None)),
            "has_sensitive_resource": bool(normalize_space(str(getattr(e, "extraordinary_circumstances", "") or ""))),
            "extraordinary_circumstances": normalize_space(str(getattr(e, "extraordinary_circumstances", "") or ""))[:200],
            "mitigation_dependence": getattr(e, "mitigation_dependence", "") or "",
            "mitigation_summary": normalize_space(str(getattr(e, "mitigation_summary", "") or ""))[:500],
            "mitigation_resource_areas": areas,
            "finding_rationale": cite["quote"][:300],
            "citation_document_id": cite["citation_document_id"],
            "citation_document_role": cite["citation_document_role"],
            "citation_section_id": "",
            "citation_evidence_span_id": cite["citation_evidence_span_id"],
            "citation_page": cite["citation_page"],
            "citation_verified": cite["citation_verified"],   # was the quoted_span quote-verified?
            "citation_claim": cite["citation_claim"],
            "quoted_span": cite["quote"][:900],
            "extraction_method": "llm_enrichment",
            "confidence": getattr(e, "extraction_confidence", "") or "medium",
            "llm_provider": "anthropic",
            "llm_model": LLM_MODEL,
            "prompt_version": "",
            "schema_version": ENRICHMENT_SCHEMA_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "candidate_extraction_run_at": getattr(e, "enrichment_extraction_run_at", run_at) or run_at,
            "candidate_llm_run_at": getattr(e, "enrichment_llm_run_at", "") or "",
        })
    facts = pd.DataFrame(rows)
    # merge the authoritative D4 timeline decision dates (for the FRA timing analysis)
    if TIMELINE.exists():
        td = pd.read_parquet(TIMELINE, columns=["project_id", "decision_date"])
        td["project_id"] = td["project_id"].astype(str)
        facts = facts.merge(td.drop_duplicates("project_id"), on="project_id", how="left")
        n_dt = int(facts.loc[facts["is_ce_shaped"], "decision_date"].notna().sum())
        print(f"[09] merged D4 decision_date: {n_dt} of the {int(facts['is_ce_shaped'].sum())} CE-shaped rows dated")
    else:
        facts["decision_date"] = pd.NaT
        print("[09] D4 timeline not found — decision_date left null")
    write_parquet(facts, FACTS_OUT)

    # ---- candidate_mitigation_summary (per candidate, profile subset) ----
    summ_rows = []
    for cat, grp in facts.groupby("candidate_category"):
        prof = grp[grp["is_ce_shaped"]]                  # mitigation pattern over the bounded candidates
        focus = prof if not prof.empty else grp
        # is_mitigated per project from the enrichment
        mit_flags = [_b(getattr(by_pid[p], "is_mitigated_fonsi", None)) for p in focus["project_id"]]
        n = len(focus); n_mit = int(sum(mit_flags))
        area_counts: dict[str, int] = {}
        for a in focus["mitigation_resource_areas"]:
            for x in [x.strip() for x in str(a).split(",") if x.strip()]:
                area_counts[x] = area_counts.get(x, 0) + 1
        top_areas = sorted(area_counts.items(), key=lambda kv: -kv[1])[:5]
        # boundary statements from significance_thresholds
        bstmts = []
        for p in focus["project_id"]:
            for th in _jlist(getattr(by_pid[p], "significance_thresholds", "[]")):
                s = normalize_space(th.get("statement", ""))
                if s:
                    bstmts.append(s)
        summ_rows.append({
            "candidate_category": cat, "n_focus": n, "n_mitigated_fonsi": n_mit,
            "mitigated_share": round(n_mit / n, 3) if n else 0.0,
            "n_with_boundary_language": int(sum(
                1 for p in focus["project_id"] if _jlist(getattr(by_pid[p], "significance_thresholds", "[]")))),
            "top_mitigation_resource_areas": "; ".join(f"{k}({v})" for k, v in top_areas),
            "example_boundary_statements": json.dumps(bstmts[:5]),
            "run_at": run_at,
        })
    summ = pd.DataFrame(summ_rows).sort_values("n_mitigated_fonsi", ascending=False)
    write_parquet(summ, MIT_SUMMARY_OUT)

    # ---- corpus_mitigation_stats (all enriched clean FONSIs) ----
    is_mit = en["is_mitigated_fonsi"].map(_b)
    n_clean = en["project_id"].nunique()
    mdep = en.loc[is_mit, "mitigation_dependence"].fillna("")   # dependence WITHIN the mitigated set
    n_case = int((mdep == "case_specific_dependent").sum())
    n_design = int(mdep.isin(["design_feature_only", "none"]).sum())
    stats = pd.DataFrame([{
        "n_clean_fonsi": n_clean, "n_with_packet": n_clean,
        "n_mitigated_fonsi": int(is_mit.sum()),
        "mitigated_share": round(float(is_mit.mean()), 3) if n_clean else 0.0,
        "n_case_specific_dependent": n_case,        # mitigated FONSIs that are case-specific dependent
        "n_design_or_none": n_design,               # mitigated but impacts avoided by design feature / none
        "run_at": run_at,
    }])
    write_parquet(stats, CORPUS_STATS_OUT)

    review_cols = ["project_id", "candidate_category", "is_profile_subtype", "is_bounded_low_impact",
                   "action_definition", "max_acres", "max_miles", "max_kilovolts", "n_wells",
                   "mitigation_dependence", "mitigation_resource_areas", "confidence",
                   "citation_document_id", "citation_page"]
    facts[review_cols].sort_values(["candidate_category", "project_id"]).to_csv(FACTS_REVIEW, index=False)

    cs = stats.iloc[0]
    print(f"[09] LLM-backed candidate_facts rows={len(facts):,} (project x category) -> {FACTS_OUT.name}")
    print(f"[09] size fill: acres={int(facts['max_acres'].notna().sum())} miles={int(facts['max_miles'].notna().sum())} "
          f"kv={int(facts['max_kilovolts'].notna().sum())} mw={int(facts['max_megawatts'].notna().sum())} "
          f"wells={int(facts['n_wells'].notna().sum())}")
    print(f"[09] corpus mitigated FONSIs: {int(cs.n_mitigated_fonsi)} of {int(cs.n_clean_fonsi)} "
          f"({cs.mitigated_share:.1%})  [case-specific={int(cs.n_case_specific_dependent)} design-only={int(cs.n_design_or_none)}]")
    print(f"[09] per-candidate mitigation summary -> {MIT_SUMMARY_OUT.name}")
    print(summ[["candidate_category", "n_focus", "n_mitigated_fonsi", "mitigated_share",
                "n_with_boundary_language"]].to_string(index=False))


if __name__ == "__main__":
    main()
