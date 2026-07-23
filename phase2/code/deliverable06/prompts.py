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

ENRICHMENT_PROMPT_VERSION = "d6_enrich_prompt_v5"   # v5: require action quote; wells in threshold units
ENRICHMENT_SCHEMA_VERSION = "d6_enrich_schema_v5"   # v5: well_count (operative bound for geo/resource-assessment)

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
    ("well_count", "integer|null",
     "number of wells, borings, or boreholes the proposed action drills/installs (the selected action, "
     "not a rejected alternative); the operative scale unit for geothermal exploration and resource "
     "assessment. Counts written as words ('up to twelve exploratory wells') count. null if not applicable."),
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
     "value (number|null), unit (string|null — when numeric, prefer one of: acres, miles, kv, mw, wells, "
     "feet, percent), is_project_fact (true if it states this project's value, "
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
     "copied character-for-character from that excerpt — do not paraphrase). ALWAYS include an 'action' "
     "claim with its span_ref (this is required — it is the citable basis for the action), and a 'finding' "
     "claim when present."),
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


_ENUM_FIELDS = {
    "action_category": ["transmission_upgrade", "solar", "geothermal_exploration",
                        "temporary_resource_assessment", "wind_onshore", "other"],
    "land_ownership": ["BLM", "federal_other", "private", "mixed", "other", None],
    "mitigation_dependence": ["none", "design_feature_only", "case_specific_dependent",
                              "permit_or_consultation_condition", "monitoring_only", "unclear"],
    "decision_basis": ["inherently_low_impact", "mitigated_to_below_significant", "small_scale", "other"],
    "extraction_confidence": ["high", "medium", "low"],
}


def enrichment_tool_schema() -> dict:
    """Anthropic tool input_schema forcing schema-valid JSON for all fields."""
    props = {}
    for name, jtype, _d in ENRICHMENT_FIELDS:
        if jtype == "array of objects":
            props[name] = {"type": "array", "items": _NESTED_ITEM_SCHEMAS[name]}
        else:
            props[name] = _json_type(jtype)
        if name in _ENUM_FIELDS:
            props[name]["enum"] = _ENUM_FIELDS[name]
    return {"type": "object", "properties": props,
            "required": [n for n, _t, _d in ENRICHMENT_FIELDS]}


# ===========================================================================
# Stage 2 — action CLASSIFICATION (cheap, separately cached).
#
# The extraction `action_category` above gives the model only six BARE LABELS with no
# definitions and no enum constraint, so keyword-similar actions are mislabeled
# (a botanical "Experimental Garden Array" -> solar; a BLM land withdrawal -> solar;
# a VHF two-way-radio upgrade -> transmission). This stage re-asks ONLY the category,
# from the already-extracted summary, with real definitions + an enum-constrained
# schema. 03_enrich_llm.py runs it as `--stage classify` (reuses the cached extraction;
# ~$1.4 for 451) and OVERWRITES action_category. Bump CLASSIFICATION_PROMPT_VERSION to
# force a classify-only re-run — the expensive extraction cache is untouched.
# ===========================================================================

CLASSIFICATION_PROMPT_VERSION = "d6_classify_prompt_v2"   # v2: precedence rules + edge-case guidance

ACTION_CATEGORIES = [
    "transmission_upgrade", "solar", "geothermal_exploration",
    "temporary_resource_assessment", "wind_onshore", "other",
]


def build_classification_prompt(title: str, action_summary: str, key_activities: str,
                                action_label: str, purpose_and_need: str) -> str:
    """The Stage-2 classifier prompt. Operates on the cached extraction summary (no
    document re-read). Classify by the physical action, not by keywords — the rules
    below name the exact failure modes observed in the first pass."""
    return (
        "TASK: Classify ONE U.S. federal NEPA action into a clean-energy action type, to support "
        "categorical-exclusion (CE) development. You are given the already-extracted summary of an "
        "Environmental Assessment (EA) that ended in a Finding of No Significant Impact (FONSI). "
        "Decide what the federal action PHYSICALLY IS.\n\n"
        "Classify by the physical action, NOT by keywords in the title or summary:\n"
        "- Funding, grants, financial assistance, loan guarantees, or programmatic budget decisions are "
        "'other' — even if they fund a solar/wind/transmission project (the federal action is the funding).\n"
        "- Studies, research installations, demonstrations, and experimental arrays (e.g. a botanical "
        "'garden array') are 'other'.\n"
        "- Land withdrawals, right-of-way grants, leases, and land-management/planning decisions are 'other'.\n"
        "- Energy-efficiency retrofits, building upgrades, communications/IT, and control/SCADA systems are 'other'.\n"
        "- Standalone battery / energy-storage projects are 'other'; a solar+storage project IS 'solar' when the "
        "solar generation is the action.\n"
        "- A NEW transmission line on NEW (greenfield) right-of-way is 'other'.\n\n"
        "CATEGORIES:\n"
        "- transmission_upgrade: physically MODIFYING an EXISTING electric transmission OR distribution line — "
        "rebuild, reconductor, voltage upgrade, structure replacement, or a substation/interconnection upgrade to "
        "existing grid infrastructure. A new line or circuit placed WITHIN an existing right-of-way or developed "
        "corridor also counts here.\n"
        "- solar: constructing or operating a solar photovoltaic or solar-thermal ELECTRICITY-GENERATION facility "
        "(including its dedicated gen-tie / interconnection).\n"
        "- geothermal_exploration: geothermal temperature-gradient / exploratory drilling, or geophysical survey "
        "for geothermal resources (a geothermal POWER PLANT is 'other').\n"
        "- temporary_resource_assessment: TEMPORARY site characterization that leaves NO permanent facility — "
        "meteorological (met) towers, geotechnical borings, surveys, monitoring — for ANY technology.\n"
        "- wind_onshore: constructing or operating PERMANENT onshore wind TURBINES (a generating facility).\n"
        "- other: anything that does not clearly and physically match one of the above.\n\n"
        "PRECEDENCE (resolve overlaps in this order):\n"
        "1. Geothermal exploratory drilling/survey -> geothermal_exploration (not temporary_resource_assessment).\n"
        "2. TEMPORARY testing for wind or solar (met towers, surveys, borings, monitoring) -> "
        "temporary_resource_assessment, NOT wind_onshore or solar; wind_onshore and solar are only for PERMANENT "
        "generating facilities.\n"
        "3. A standalone substation/gen-tie interconnecting a NEW generator -> that generator's type (solar/wind) "
        "if the generation is the federal action; otherwise 'other'.\n\n"
        f"INPUT:\n  title: {title}\n  action_summary: {action_summary}\n  key_activities: {key_activities}\n"
        f"  action_label: {action_label}\n  purpose_and_need: {purpose_and_need}\n\n"
        "Return action_category (exactly one of the six), classification_confidence (high/medium/low), and "
        "classification_rationale (one sentence grounded in the input)."
    )


def classification_tool_schema() -> dict:
    """Enum-constrained tool schema — the model cannot return an off-list category."""
    return {"type": "object", "properties": {
        "action_category": {"type": "string", "enum": ACTION_CATEGORIES},
        "classification_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "classification_rationale": {"type": "string"}},
        "required": ["action_category", "classification_confidence", "classification_rationale"]}


# ===========================================================================
# Stage 3 — ACTION-VERB labeling (refactor: tech_group x action grid).
#
# Assigns each FONSI's action a controlled verb WITHIN its tech_group, so the categorizer
# (09) can form `tech_group__action` grid cells. Operates on the cached extraction summary
# (no document re-read). `is_codifiable` is derived DETERMINISTICALLY from the verb (not the
# LLM). 10_action_label.py runs this; bump ACTIONLABEL_PROMPT_VERSION to force a re-run.
# ===========================================================================

ACTIONLABEL_PROMPT_VERSION = "d6_actionlabel_v1"

ACTION_VERBS = [
    "new_build", "upgrade", "maintenance", "decommissioning", "exploration",
    "assessment", "research_or_demonstration", "manufacturing", "interconnection",
    "land_or_row_authorization", "other",
]

# a CE codifies a PHYSICAL action, not funding / manufacturing / administrative acts
NON_CODIFIABLE_VERBS = {"manufacturing", "land_or_row_authorization"}


def is_codifiable_for(action_verb: str) -> bool:
    """Deterministic is_codifiable from the verb — no LLM. Manufacturing (a factory) and
    land/ROW authorizations (administrative) are not physical actions a CE can codify."""
    return action_verb not in NON_CODIFIABLE_VERBS


def build_action_label_prompt(tech_group: str, action_summary: str, key_activities: str,
                              action_label: str, purpose_and_need: str) -> str:
    """Label the physical action with one controlled verb. Operates on the cached summary."""
    return (
        "TASK: Label ONE U.S. federal NEPA action with the single action VERB that best describes "
        "what the federal action PHYSICALLY DOES, to support categorical-exclusion (CE) development. "
        "You are given the already-extracted summary of an EA that ended in a FONSI, plus its technology "
        "group. Choose the verb by the physical action, NOT by keywords in the title or summary.\n\n"
        "VERBS:\n"
        "- new_build: constructing a NEW generating facility, plant, or line (greenfield or a new unit).\n"
        "- upgrade: physically MODIFYING EXISTING infrastructure — rebuild, reconductor, voltage upgrade, "
        "repower, structure replacement, or a substation upgrade to existing grid infrastructure.\n"
        "- maintenance: repair, vegetation / right-of-way upkeep, routine servicing of existing facilities.\n"
        "- decommissioning: removal, retirement, or demolition of existing facilities / lines / turbines.\n"
        "- exploration: resource-investigation drilling or geophysical survey (e.g. geothermal gradient wells).\n"
        "- assessment: TEMPORARY site characterization leaving NO permanent facility — met towers, borings, "
        "surveys, monitoring.\n"
        "- research_or_demonstration: a pilot / R&D / demonstration facility (first-of-kind, experimental).\n"
        "- manufacturing: building or expanding a FACTORY (e.g. battery components, modules, materials).\n"
        "- interconnection: a gen-tie / grid-tap / interconnection line for a generator.\n"
        "- land_or_row_authorization: an ADMINISTRATIVE land action — right-of-way grant / renewal / amendment, "
        "lease, withdrawal, or land-use plan (NON-physical).\n"
        "- other: anything that does not clearly match a verb, INCLUDING pure funding / financial assistance "
        "(the federal action is the funding itself).\n\n"
        "GUIDANCE (resolve overlaps):\n"
        "- Funding, grants, loan guarantees, financial/cost-share assistance -> 'other' (even if they fund a build).\n"
        "- A new line or circuit placed WITHIN an EXISTING right-of-way / developed corridor is 'upgrade' "
        "(modifying the existing corridor), NOT 'new_build'.\n"
        "- Temporary testing (met towers, surveys, borings, monitoring) is 'assessment', even for wind/solar.\n"
        "- A geothermal exploratory / gradient well is 'exploration'.\n\n"
        f"INPUT:\n  tech_group: {tech_group}\n  action_summary: {action_summary}\n"
        f"  key_activities: {key_activities}\n  action_label: {action_label}\n"
        f"  purpose_and_need: {purpose_and_need}\n\n"
        "Return action (exactly one verb), action_confidence (high/medium/low), and "
        "action_rationale (one sentence grounded in the input)."
    )


def action_label_tool_schema() -> dict:
    """Enum-constrained tool schema — the model cannot return an off-list verb."""
    return {"type": "object", "properties": {
        "action": {"type": "string", "enum": ACTION_VERBS},
        "action_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "action_rationale": {"type": "string"}},
        "required": ["action", "action_confidence", "action_rationale"]}
