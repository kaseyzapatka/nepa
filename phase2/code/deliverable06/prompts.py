"""D6 — shared LLM prompt definitions (un-numbered helper).

Lives here (not in a numbered step) so both the extraction step
(03_extract_candidate_facts.py) and the standalone benchmark (benchmark_models.py)
import the *identical* production prompt. Numbered scripts can't be imported
(module names can't start with a digit), so shared functions live in helpers.
"""

from __future__ import annotations


def build_facts_prompt(packet_text: str, category: str) -> str:
    """The production fact-extraction prompt. Shared with the model benchmark so it
    measures the exact prompt used in production."""
    return (
        "Extract CE-relevant facts from this NEPA FONSI/EA evidence as strict JSON with keys: "
        "action_definition, max_acres, max_miles, max_megawatts, within_existing_row, "
        "no_new_access_road, previously_disturbed_land, mitigation_dependence "
        "(one of none/design_feature_only/case_specific_dependent), mitigation_summary, "
        "extraordinary_circumstances. "
        f"Category hint: {category}. Use null when unknown.\n\nEVIDENCE:\n" + packet_text[:6000]
    )


# ===========================================================================
# Single comprehensive enrichment pass (Sonnet) — one read of every clean-energy
# FONSI that populates EVERYTHING Analyses 1 and 2 need, so the paid pass is run
# ONCE and never repeated. Consumed by 03_enrich_llm.py.
#
# ENRICHMENT_FIELDS is the SINGLE SOURCE OF TRUTH: it builds the prompt AND defines
# the expected output columns, so the two can never drift. To evolve the schema,
# edit this list and bump ENRICHMENT_SCHEMA_VERSION (which forces a re-run only for
# changed rows via the input-hash cache).
# ===========================================================================

ENRICHMENT_PROMPT_VERSION = "d6_enrich_prompt_v4"   # v4: size-retrieval packet section
ENRICHMENT_SCHEMA_VERSION = "d6_enrich_schema_v4"   # v4: ce_development_span_ref (verifiable)

# (field, json_type, instruction). Grouped by which analysis consumes it.
ENRICHMENT_FIELDS: list[tuple[str, str, str]] = [
    # --- Analysis 1: action identity, scale, siting ---
    ("action_summary", "string",
     "1-2 plain-English sentences describing the federal action."),
    ("purpose_and_need", "string",
     "1 sentence on why the action is needed (the EA's purpose-and-need)."),
    ("action_category", "string",
     "exactly one of: transmission_upgrade, solar, geothermal_exploration, "
     "temporary_resource_assessment, wind_onshore, other."),
    ("action_category_other", "string|null",
     "if action_category is 'other', a short label for the action; else null."),
    ("action_label_freeform", "string",
     "a short, normalized free-form label for the action regardless of category (e.g. 'transmission "
     "reconductoring', 'rooftop solar', 'fish passage culvert') — used to cluster non-candidate FONSIs."),
    ("potential_ce_theme", "string|null",
     "if this recurring action looks like a plausible NEW categorical-exclusion theme, a short name for it; else null."),
    ("why_not_current_candidate", "string|null",
     "if action_category is 'other', one phrase on what kind of action it is and why it's outside the current "
     "candidate set; else null."),
    ("is_bounded_low_impact", "boolean|null",
     "true if this is a small, routine, low-impact version of the action that could plausibly be a "
     "categorical exclusion (NOT a large, greenfield, or major project); null if the excerpts don't say."),
    ("bounded_rationale", "string",
     "one sentence on why is_bounded_low_impact is true or false."),
    ("key_activities", "array of strings",
     "the discrete physical activities the action involves (e.g. 'reconductor line', 'replace structures', "
     "'vegetation clearing', 'build access road', 'install met tower'); [] if unclear."),
    ("disturbance_acres", "number|null",
     "acres of actual ground disturbance / project footprint. NEVER the study area, planning area, "
     "watershed, allotment, or analysis-area acreage."),
    ("line_miles", "number|null",
     "length in miles of the transmission/distribution LINE itself (the selected/proposed action, not a "
     "rejected alternative); null if not linear. Do NOT report access-road length here."),
    ("access_road_miles", "number|null",
     "miles of access road built/improved, kept SEPARATE from line_miles; null if not stated."),
    ("capacity_mw", "number|null",
     "generation or storage capacity in megawatts; null if not applicable."),
    ("voltage_kv", "number|null",
     "transmission voltage in kilovolts; null if not stated."),
    ("within_existing_row", "boolean|null",
     "true if the work is within an existing right-of-way or previously developed corridor."),
    ("new_access_road", "boolean|null",
     "true if NEW (not merely improved) access roads are built."),
    ("previously_disturbed_land", "boolean|null",
     "true if sited on previously disturbed / developed / degraded land."),
    ("is_temporary", "boolean|null",
     "true if the action/disturbance is temporary (survey, testing, short-term study) rather than a "
     "permanent facility."),
    ("land_ownership", "string|null",
     "the land jurisdiction of the project site: one of BLM, federal_other, private, mixed, other; "
     "null if not stated."),
    # --- Analysis 2: significance & mitigation ---
    ("is_mitigated_fonsi", "boolean|null",
     "true if the no-significant-impact finding DEPENDS on committed mitigation, rather than the "
     "action being inherently low-impact; null if the excerpts don't say."),
    ("mitigation_dependence", "string",
     "the role mitigation plays in the no-significant-impact finding, exactly one of: "
     "none (inherently low-impact), design_feature_only (impacts avoided by the project's own bounded design), "
     "case_specific_dependent (the FONSI depends on committed, project-specific mitigation), "
     "permit_or_consultation_condition (conditions required by another regime, not necessarily the FONSI basis), "
     "monitoring_only, unclear."),
    ("mitigation_summary", "string",
     "short summary of the committed mitigation measures; '' if none."),
    ("mitigation_resource_areas", "array of strings",
     "resource areas the mitigation addresses, each one of: water, biological, cultural, air_quality, "
     "noise, soils_geology, transportation, visual, public_health, recreation, other."),
    ("key_impacts", "array of strings",
     "the main environmental impacts the EA identifies, each tagged to a resource area "
     "(e.g. 'biological: habitat disturbance', 'water: temporary turbidity'); [] if none notable."),
    ("residual_impacts", "string|null",
     "impacts the EA says remain AFTER mitigation; '' or null if none/negligible."),
    ("significance_thresholds", "array of objects",
     "ONLY explicit threshold / counterfactual statements (e.g. 'impacts would be significant if X exceeded Y', "
     "'an EIS would be required unless Z', 'provided that', 'not to exceed') — NOT scoping comments, RMP "
     "conformance, table references, or generic decision language. Each object has keys: statement (EXACT "
     "verbatim text), span_ref (the [S#] tag of the excerpt it came from), metric (string|null), "
     "value (number|null), unit (string|null), is_project_fact (true if it states this project's value, "
     "false if a general threshold). Empty array [] if none."),
    ("extraordinary_circumstances", "string|null",
     "any extraordinary circumstances noted that could preclude a CE; else null."),
    ("decision_basis", "string",
     "exactly one of: inherently_low_impact, mitigated_to_below_significant, small_scale, other."),
    ("significance_factors", "array of strings",
     "which significance / intensity factors the EA leaned on, each one of: context, controversy, "
     "cumulative_effects, unique_characteristics, public_health_safety, threatened_endangered, "
     "cultural_historic, precedent, highly_uncertain; [] if none discussed."),
    # --- DIRECT TEXT: verbatim quotes (verified + cited downstream, not trusted blindly) ---
    ("evidence", "array of objects",
     "the document text backing the key claims, so a human can check every summary against the source. "
     "Provide one object per claim type you can support, with keys: claim (one of: action, finding, size, "
     "mitigation), span_ref (the [S#] tag of the excerpt the quote is from), and quote (EXACT verbatim text "
     "copied character-for-character from that excerpt — do not paraphrase). Include at least the 'action' "
     "and 'finding' claims when present."),
    ("referenced_ce_citations", "array of objects",
     "existing categorical exclusions or NEPA authorities the document itself cites as relevant (e.g. "
     "'516 DM 11.9 B1.3', 'B4.13', '40 CFR 1508.4'). Each object: citation (verbatim), span_ref (the [S#] tag), "
     "and context (the verbatim sentence around it). Empty array [] if none."),
    ("ce_development_language", "string|null",
     "any VERBATIM language signaling the action is routine/minor or resembles actions normally "
     "categorically excluded (e.g. 'routine maintenance', 'minor', 'would not individually or cumulatively "
     "have a significant effect'); else null. Copy EXACTLY from one excerpt so it can be located."),
    ("ce_development_span_ref", "string|null",
     "the [S#] tag of the excerpt ce_development_language was copied from; null if none."),
    # --- context extras ---
    ("cooperating_agencies", "array of strings",
     "federal/state agencies named as cooperating or consulting agencies; [] if none."),
    ("is_tiered", "boolean|null",
     "true if this EA tiers from / incorporates a programmatic EIS or EA; null if the excerpts don't say."),
    ("tiers_from", "string|null",
     "the programmatic document it tiers from, verbatim; else null."),
    # --- quality ---
    ("extraction_confidence", "string",
     "exactly one of: high, medium, low — your confidence in this extraction."),
]

# Columns piped into the analysis dataset (data/analysis). 03 ALSO adds a computed
# `evidence_cited` column (each verbatim quote verified against the source spans and
# stamped with page + document + main-doc) — see 03_enrich_llm.py.
ENRICHMENT_ANALYSIS_COLUMNS: list[str] = [f for f, _t, _d in ENRICHMENT_FIELDS]


def build_enrichment_prompt(packet_text: str) -> str:
    """The single production enrichment prompt. `packet_text` is the balanced,
    span-tagged evidence packet built by enrich_lib.build_evidence_packet() — each
    excerpt is labeled [S#] with its page/document so the model can cite span_refs.
    Built from ENRICHMENT_FIELDS so prompt and output columns can never drift."""
    fields = "\n".join(f"- {name} ({jtype}): {desc}" for name, jtype, desc in ENRICHMENT_FIELDS)
    return (
        "You are analyzing a U.S. federal Environmental Assessment (EA) that ended in a Finding of No "
        "Significant Impact (FONSI), to support categorical-exclusion (CE) development. The evidence below is "
        "a set of excerpts, each labeled with a tag like [S1] and its document/page. Use ONLY these excerpts. "
        "Return the fields:\n\n"
        f"{fields}\n\n"
        "Rules: use null (or [] / \"\") when the excerpts do not state something — never guess. Do NOT infer "
        "numbers that are not in the text, and report values for the SELECTED/proposed action only, not "
        "rejected alternatives. For acreage, report the project's disturbance/footprint, never the study- or "
        "planning-area size. Every `quote`, `statement`, and `context` must be copied EXACTLY (character-for-"
        "character) from one excerpt, and its `span_ref` must be that excerpt's [S#] tag, so it can be "
        "verified against the source. Do NOT include leading or trailing ellipsis (…) characters in any "
        "quote — copy only the real words.\n\n"
        "EVIDENCE EXCERPTS:\n" + packet_text
    )


# --- tool schema for structured output (Anthropic tool-use) ----------------
# Explicit item schemas for the nested array-of-object fields; scalars are mapped
# from the json_type strings. Keeps ENRICHMENT_FIELDS the source of field names.
_NESTED_ITEM_SCHEMAS = {
    "evidence": {"type": "object",
                 "properties": {"claim": {"type": "string"}, "span_ref": {"type": "string"},
                                "quote": {"type": "string"}},
                 "required": ["claim", "span_ref", "quote"]},
    "significance_thresholds": {"type": "object",
                                "properties": {"statement": {"type": "string"}, "span_ref": {"type": "string"},
                                               "metric": {"type": ["string", "null"]},
                                               "value": {"type": ["number", "null"]},
                                               "unit": {"type": ["string", "null"]},
                                               "is_project_fact": {"type": ["boolean", "null"]}},
                                "required": ["statement", "span_ref", "metric", "value", "unit", "is_project_fact"]},
    "referenced_ce_citations": {"type": "object",
                                "properties": {"citation": {"type": "string"}, "span_ref": {"type": "string"},
                                               "context": {"type": "string"}},
                                "required": ["citation", "span_ref", "context"]},
}


def _json_type(jtype: str):
    if jtype == "array of strings":
        return {"type": "array", "items": {"type": "string"}}
    if jtype == "array of objects":
        return {"type": "array"}                       # items filled per-field below
    base = {"string": "string", "boolean": "boolean", "number": "number", "integer": "integer"}
    parts = [p.strip() for p in jtype.split("|")]
    types = [base.get(p) for p in parts if base.get(p)]
    if "null" in parts:
        types.append("null")
    return {"type": types[0] if len(types) == 1 else types}


def enrichment_tool_schema() -> dict:
    """Anthropic tool input_schema forcing schema-valid JSON for all fields."""
    props = {}
    for name, jtype, _d in ENRICHMENT_FIELDS:
        if jtype == "array of objects":
            props[name] = {"type": "array", "items": _NESTED_ITEM_SCHEMAS[name]}
        else:
            props[name] = _json_type(jtype)
    return {"type": "object", "properties": props,
            "required": [n for n, _t, _d in ENRICHMENT_FIELDS]}
