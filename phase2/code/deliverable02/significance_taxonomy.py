"""D2 controlled vocabularies — versioned. See phase2/plans/deliverable02.md (v2.5).

These are the *output* taxonomy + a crosswalk over the shared D6 RESOURCE_AREAS.
Cue dictionaries (DETERMINATION_CUES, THRESHOLD_CUES) generate *candidates* only;
the determination class is assigned by ordered rules + LLM adjudication, not by a
raw cue hit.
"""
from __future__ import annotations

TAXONOMY_VERSION = "d2_tax_v1"

# ---- shared D6 resource areas (preserved verbatim; never renamed in place) ----
# This is the LLM's answer space for a resource-specific determination. `unknown` here means
# "genuinely could not place the resource" (a real review signal). Project-level conclusions
# (a FONSI / overall decision, scope=project_overall) are NOT tied to one resource — the code
# assigns them RESOURCE_PROJECT_WIDE instead, so `unknown` is never overloaded with "N/A by design".
SHARED_RESOURCE_AREAS = [
    "air_quality", "water", "biological", "cultural", "visual", "noise",
    "soils_geology", "socioeconomic", "transportation", "land_use",
    "climate_ghg", "public_health", "unknown",
]
RESOURCE_PROJECT_WIDE = "project_wide"   # code-assigned to project_overall rows; not an LLM value
ALL_RESOURCE_VALUES = SHARED_RESOURCE_AREAS + [RESOURCE_PROJECT_WIDE]

# ---- D2 subarea crosswalk: shared_area -> {d2_subarea: keyword cues} ----
# Report sections lead with the shared 12; subareas are nested where supported.
RESOURCE_CROSSWALK = {
    "water": {
        "water_quality": ["water quality", "groundwater", "stormwater", "turbidity", "section 401"],
        "wetlands": ["wetland", "section 404", "waters of the u", "jurisdictional water"],
        "floodplains": ["floodplain", "eo 11988", "executive order 11988", "100-year flood"],
    },
    "socioeconomic": {
        "socioeconomics": ["employment", "jobs", "economy", "fiscal", "population", "housing"],
        "environmental_justice": ["environmental justice", "ej", "eo 12898", "executive order 12898",
                                   "disproportionate", "minority", "low-income"],
    },
    "public_health": {
        "public_health": ["public health", "human health", "exposure", "safety"],
        "hazardous_materials": ["hazardous material", "hazardous waste", "rcra", "cercla",
                                 "contamination", "toxic", "spill"],
    },
    "biological": {"biological_special_status": []},  # default rename; subareas optional
    "cultural": {"cultural_historic": []},
    "climate_ghg": {"ghg_climate": []},
    "air_quality": {"air_quality": []},
    "visual": {"visual": []},
    "noise": {"noise": []},
    "soils_geology": {"soils_geology": []},
    "transportation": {"transportation": []},
    "land_use": {"land_use": []},
    "unknown": {"unknown": []},
}

# ---- determination classes / scopes / polarities (the output enums) ----
DETERMINATION_CLASSES = [
    "no_significant_impact", "less_than_significant",
    "less_than_significant_with_mitigation", "significant_adverse",
    "significant_unavoidable", "eis_required", "not_a_determination", "ambiguous",
]
DETERMINATION_SCOPES = [
    "project_overall", "resource_specific", "alternative_specific",
    "threshold_specific", "programmatic_or_tiered", "procedural",
]
DETERMINATION_POLARITIES = [
    "no_adverse", "adverse_not_significant", "adverse_significant", "mixed", "unknown",
]
THRESHOLD_TYPES = [
    "NAAQS", "PSD", "ESA_take", "ESA_jeopardy", "NHPA_adverse_effect",
    "wetland_floodplain", "noise_threshold", "visual_vrm", "other_quantitative",
    "none", "unknown",
]
THRESHOLD_STATUSES = [
    "exceeds", "does_not_exceed", "may_exceed", "mitigated_below",
    "not_evaluated", "unknown",
]

# ---- significance-factor keys (framework-agnostic; regime stored alongside) ----
FACTOR_KEYS = [
    # 1508.27 context/intensity
    "context_local", "context_regional", "intensity_public_health_safety",
    "intensity_unique_characteristics", "intensity_controversy",
    "intensity_uncertainty_risk", "intensity_precedent", "intensity_cumulative",
    "intensity_historic_cultural", "intensity_endangered_species",
    "intensity_legal_violation",
    # 2020/2024 rule
    "degree_of_effects", "affected_environment_proximity",
    # FRA statutory
    "reasonably_foreseeable", "significant_effects_statutory",
    "other",
]

# ---- cue dictionaries (generate candidates only) ----
DETERMINATION_CUES = {
    "document_outcome": [
        r"finding of no significant impact", r"\bfonsi\b", r"\brecord of decision\b",
    ],
    "explicit_less_than_significant": [
        r"less[- ]than[- ]significant", r"\bnot significant\b",
        r"no significant adverse (impact|effect)", r"would not be significant",
    ],
    "explicit_mitigated_lts": [
        r"less[- ]than[- ]significant with mitigation",
        r"(mitigated|reduced) to (a level that is |)(below |less than )significan",
        r"would be significant (absent|without)", r"with (the )?incorporation of .{0,40}mitigation",
    ],
    "explicit_significant_adverse": [
        r"significant adverse (impact|effect)", r"significant and unavoidable",
        r"unavoidable adverse", r"would remain significant",
    ],
}

# threshold cue -> threshold_type
THRESHOLD_CUES = {
    "NAAQS": [r"\bnaaqs\b", r"national ambient air quality standard"],
    "PSD": [r"prevention of significant deterioration", r"\bpsd\b"],
    "ESA_take": [r"incidental take", r"\bsection 7\b", r"biological opinion", r"\besa\b"],
    "ESA_jeopardy": [r"jeopardy", r"jeopardize the continued existence"],
    "NHPA_adverse_effect": [r"section 106", r"adverse effect", r"historic propert", r"\bnhpa\b"],
    "wetland_floodplain": [r"section 404", r"section 401", r"floodplain", r"wetland"],
    "noise_threshold": [r"\bdba\b", r"decibel", r"noise (criteria|standard|threshold)"],
    "visual_vrm": [r"visual resource management", r"\bvrm\b", r"visual contrast"],
}

# ---- mitigation roles/obligations that count as enforceable for the mitigated-FONSI flag ----
MITIGATION_ROLES = ("mitigation_commitment", "enforcement_or_permit_condition")
MITIGATION_OBLIGATIONS = ("required", "committed")

# Recall-oriented screen for the Gate-1 mitigated-FONSI list (the human prunes).
# Calibrated against clean finding+condition spans: BLM/DOE FONSIs do NOT use the
# CEQA-style "less than significant with mitigation" phrase (0 hits) — they use
# "would be significant [absent mitigation]" and "with incorporation of ...
# mitigation". Kept separate from DETERMINATION_CUES['explicit_mitigated_lts']
# (precision, used by the extractor's classifier).
MITIGATED_SCREEN_CUES = [
    r"would be significant",
    r"with (the )?(incorporation|implementation|inclusion) of .{0,60}(mitigation|measure|condition)",
    r"(reduce|minimize|lessen)[a-z]* .{0,50}(below|to less than|to a level|to less[- ]than) .{0,25}significan",
]
