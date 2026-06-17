"""D6 v2 candidate-category configuration (narrow-first).

This is the single source of truth for the Stage A candidate categories, their
membership rules (over D3 `tech_group` + a project_type/title/description text
blob), and their CE-development story. Versioned so downstream artifacts can
record which taxonomy produced them.

See `phase2/plans/deliverable06.md` (v2.2) for rationale and the review record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TAXONOMY_VERSION = "d6_v2_2"


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


@dataclass(frozen=True)
class TechCandidate:
    category: str
    label: str
    tech_groups: tuple[str, ...]
    ce_story: str
    role: str  # "profile" or "contrast"
    # ordered (subtype_name, regex); first match wins, else default_subtype
    subtypes: tuple[tuple[str, re.Pattern], ...] = field(default_factory=tuple)
    default_subtype: str = "other"
    profile_subtypes: tuple[str, ...] = ()  # CE-shaped subset to profile

    def subtype_for(self, blob: str) -> str:
        for name, rx in self.subtypes:
            if rx.search(blob):
                return name
        return self.default_subtype


# --- Tech-group candidates -------------------------------------------------

TECH_CANDIDATES: tuple[TechCandidate, ...] = (
    TechCandidate(
        category="transmission_upgrade",
        label="Transmission upgrades within existing ROW",
        tech_groups=("Transmission",),
        ce_story="new_or_expand",
        role="profile",
        # ordered most-specific first. Off-scope misclassifications and non-upgrade
        # action types are separated out so the CE-shaped "standalone_upgrade"
        # profile subset stays clean and precise.
        subtypes=(
            ("off_scope_misclassified", _rx(r"natrium|nuclear demonstration|mining plan|mine (plan|stockpile)|"
                                            r"air conditioner|heat pump|conservation standard|waste transfer|"
                                            r"combustor|energy recovery|loan guarantee for|manufactur")),
            ("row_vegetation_maintenance", _rx(r"vegetation (management|control)|herbicide|pesticide|\bweed\b|"
                                               r"invasive plant|integrated weed|noxious weed|clearance|"
                                               r"right[- ]of[- ]way maintenance|row maintenance|line maintenance")),
            ("telecom_communication", _rx(r"fiber optic|communication (facilit|site)|microwave|\bat&t\b|broadband")),
            ("row_renewal_amendment", _rx(r"right[- ]of[- ]way (renewal|amendment|grant|application)|"
                                          r"row (renewal|amendment)|\brenewal and amendment\b")),
            ("substation_switchyard", _rx(r"substation|switchyard|switchgear|switching station")),
            ("distribution_line", _rx(r"distribution line|pole line")),
            ("access_service_road", _rx(r"access (and )?(service )?road|service road")),
            ("gen_bundled", _rx(r"\bsolar\b|\bwind\b|geothermal|hydropower|biomass|photovoltaic")),
            ("new_line", _rx(r"new transmission line|new \d+[- ]?kv|new corridor|greenfield|"
                             r"new power ?line|new right[- ]of[- ]way")),
            ("standalone_upgrade", _rx(r"rebuild|reconductor|recondition|upgrade|replace|reconstruct|"
                                       r"in[- ]place|voltage conversion|series capacitor|"
                                       r"remedial action scheme|relocat|improvement|life extension|reinforce|"
                                       r"within (the )?existing|existing right[- ]of[- ]way|existing corridor")),
            ("interconnection", _rx(r"interconnect|gen[- ]?tie|tie[- ]line|generation tie")),
        ),
        default_subtype="other_transmission",
        profile_subtypes=("standalone_upgrade",),
    ),
    TechCandidate(
        category="geothermal_exploration",
        label="Geothermal exploration",
        tech_groups=("Geothermal",),
        ce_story="adopt_or_expand",
        role="profile",
        subtypes=(
            ("exploration", _rx(r"explorat|temperature gradient|geophysical|slim[- ]?hole|observation well|"
                                r"seismic survey|resource confirmation|test well|core ?hole|reconnaissance")),
            ("development", _rx(r"power plant|production|utilization|generation facilit|develop|binary plant")),
        ),
        default_subtype="other_geothermal",
        profile_subtypes=("exploration",),
    ),
    TechCandidate(
        category="solar",
        label="Solar (CE-shaped subset)",
        tech_groups=("Solar",),
        ce_story="new_or_adopt",
        role="profile",
        # disturbed-site siting (the CE feature) is checked before gen_tie, and
        # gen_tie now fires only on line-specific terms so it stops swallowing
        # ordinary solar generation projects that merely mention interconnection.
        subtypes=(
            ("manufacturing", _rx(r"manufactur|factory|assembly|production facilit")),
            ("row_vegetation_maintenance", _rx(r"weed management|invasive plant|vegetation management|"
                                               r"integrated weed|herbicide|noxious weed")),
            ("disturbed_developed", _rx(r"disturb|previously developed|brownfield|reclaim|degraded|former |"
                                        r"rooftop|parking|landfill|capped|mine|mining|industrial site")),
            ("gen_tie", _rx(r"gen[- ]?tie|tie[- ]line|generation tie")),
        ),
        default_subtype="greenfield_utility",
        profile_subtypes=("disturbed_developed",),
    ),
    TechCandidate(
        category="wind_onshore",
        label="Wind, onshore (contrast)",
        tech_groups=("Wind",),
        ce_story="likely_weak",
        role="contrast",
        subtypes=(),
        default_subtype="wind",
        profile_subtypes=(),
    ),
)


# --- Cross-tech candidate #4 ----------------------------------------------

@dataclass(frozen=True)
class KeywordCandidate:
    category: str
    label: str
    ce_story: str
    role: str
    include: re.Pattern
    exclude: re.Pattern | None = None


RESOURCE_ASSESSMENT = KeywordCandidate(
    category="temporary_resource_assessment",
    label="Temporary resource assessment / site investigation",
    ce_story="adopt_expand_crosswalk",
    role="profile",
    include=_rx(
        r"met(eorological)? tower|meteorological monitoring|geotechnical|geotech|soil boring|"
        r"exploratory drilling|exploratory boring|monitoring well|test well|site characterization|"
        r"resource assessment|wind resource|solar resource|data collection|temporary .{0,30}(survey|monitor|test)"
    ),
    exclude=None,
)


# --- Storage-deployment scan (Gate 2 evidence only) ------------------------

STORAGE_SCAN_INCLUDE = _rx(r"battery|energy storage|\bbess\b|storage system|grid storage|standalone storage")
STORAGE_SCAN_EXCLUDE = _rx(
    r"manufactur|recycl|anode|cathode|precursor|gigafactory|active material|battery component|"
    r"cell production|component manufacturing|materials? (plant|production)"
)


def text_blob(*values: object) -> str:
    return " ".join("" if v is None else str(v) for v in values)
