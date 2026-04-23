import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

# --------------------------
# DELIVERABLE 1: NEPA TRIGGERED — Trigger Classification
# --------------------------
# Five-tier classification of what triggered NEPA review per 20,725 clean energy projects:
#   Tier 1a — Agency metadata heuristics (deterministic agencies; DOE stays provisional)
#   Tier 1b — Title and description keyword matching (all 7 classes; specificity-ranked)
#   Tier 2  — Document title scan (documents.parquet; high-signal title cues)
#   Tier 3  — Candidate-section regex extraction (EA/EIS first pages; CE full text with conservative patterns)
#   Tier 4  — Retrieval-first local adjudication (chunk retrieval + local NLI/embedding fallback)
#   Tier 5  — Claude Haiku LLM fallback on the small uncertain queue (--use-llm only)
#
# [SELF-CONTAINED] — requires only projects_combined.parquet and CE/EA/EIS docs/pages.
#
# Usage:
#   python 01_extract_nepa_trigger.py --eda              # EDA check only; no extraction
#   python 01_extract_nepa_trigger.py --calibrate        # validate NLI hypotheses before full run
#   python 01_extract_nepa_trigger.py --sample 50        # test on 50 projects
#   python 01_extract_nepa_trigger.py                    # full run (~20,725 projects)
#   python 01_extract_nepa_trigger.py --use-llm          # full run + Haiku on low-confidence
#
# Output:
#   data/analysis/nepa_trigger/projects_nepa_trigger.parquet  (one row per project)
#   data/analysis/nepa_trigger/validation_batches.csv          (flagged cases grouped by rule)

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --------------------------
# CONSTANTS
# --------------------------

BASE_DIR     = Path(__file__).resolve().parent.parent.parent  # repo root
DATA_DIR     = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_DIR   = ANALYSIS_DIR / "nepa_trigger"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS_PATH  = ANALYSIS_DIR / "projects_combined.parquet"
CE_PAGES_PATH  = DATA_DIR / "processed" / "ce"  / "pages.parquet"
EA_PAGES_PATH  = DATA_DIR / "processed" / "ea"  / "pages.parquet"
EIS_PAGES_PATH = DATA_DIR / "processed" / "eis" / "pages.parquet"
CE_DOCS_PATH   = DATA_DIR / "processed" / "ce"  / "documents.parquet"
EA_DOCS_PATH   = DATA_DIR / "processed" / "ea"  / "documents.parquet"
EIS_DOCS_PATH  = DATA_DIR / "processed" / "eis" / "documents.parquet"

DOCS_PATH_MAP  = {"CE": CE_DOCS_PATH,  "EA": EA_DOCS_PATH,  "EIS": EIS_DOCS_PATH}
PAGES_PATH_MAP = {"CE": CE_PAGES_PATH, "EA": EA_PAGES_PATH, "EIS": EIS_PAGES_PATH}
DOC_COLUMNS    = ["project_id", "document_id", "document_title", "file_name", "main_document"]

CLEAN_ENERGY_FILTER = "project_energy_type = 'Clean'"  # n = 20,725; hard constraint

EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
EMBEDDING_THRESHOLD = 0.45
HAIKU_MODEL         = "claude-haiku-4-5-20251001"
MAX_PAGES_PAN       = 10   # pages to scan for Purpose and Need in EA/EIS
PAN_WINDOW          = 600  # chars to extract after section header

OUTPUT_COLS = [
    "project_id",
    "nepa_trigger_primary", "nepa_trigger_secondary", "nepa_trigger_multi",
    "nepa_trigger_evidence_text", "nepa_trigger_evidence_source",
    "nepa_trigger_confidence", "nepa_trigger_rule_id",
    "nepa_trigger_manual_review", "is_dual_nexus",
    "nepa_trigger_extraction_run_at", "nepa_trigger_llm_run_at",
]

CONTEXT_CANDIDATES_PATH = OUTPUT_DIR / "context_candidates.parquet"
TIER4_CHUNK_SCORES_PATH = OUTPUT_DIR / "tier4_chunk_scores.parquet"
TIER4_DOC_SCORES_PATH = OUTPUT_DIR / "tier4_doc_scores.parquet"
TIER5_QUEUE_PATH = OUTPUT_DIR / "tier5_queue.parquet"

TIER4_TOP_K = 4
TIER4_BASE_THRESHOLD = 0.90
TIER4_NO_PRIOR_THRESHOLD = 0.92
TIER4_MARGIN_THRESHOLD = 0.15
TIER4_CONTRADICTION_WINDOW = 0.10
TIER4_SUPPORT_THRESHOLD = 0.25

TIER5_TARGET_QUEUE = 250
TIER5_SOFT_WARNING = 150
TIER5_HARD_STOP_BUDGET = 10.0
ESTIMATED_TIER5_COST_PER_PROJECT = 0.04  # conservative placeholder for queue guardrails

LOCAL_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

AUTO_ACCEPT_RULE_IDS = frozenset({
    "T1a_FERC_permit",
    "T1a_FAA_permit",
    "T1a_FCC_permit",
    "T2_doc_title_peis",
    "T2_doc_title_row",
    "T2_doc_title_permit_app",
    "T2_doc_title_license_amendment",
    "T2_doc_title_loan_guarantee",
    "T1b_ferc_license",
    "T1b_special_use",
    "T1b_row_grant",
    "T1b_land_exchange",
    "T3_npdes",
    "T3_agency_grant",
    "T3_blm_land",
    "T3_nfs_land",
})

SEND_TO_TIER4_RULE_IDS = frozenset({
    "T1a_DOE_action",
    "T1a_DOE_funding",
    "T1b_arra",
    "T3_sec404",
    "T3_arra",
})

AUDIT_FIRST_RULE_IDS = frozenset({
    "T3_rmp",
})

AMBIGUOUS_METADATA_AGENCIES = frozenset({
    "DOE", "Department of Energy",
    "USACE", "Army Corps of Engineers",
})

# --------------------------
# PATTERN DICTIONARIES
# --------------------------

# --- Tier 1a: Agency maps ---

AGENCY_PERMIT_MAP = frozenset({
    "FERC", "Federal Energy Regulatory Commission",
    "FAA", "Federal Aviation Administration",
    "FCC", "Federal Communications Commission",
})
AGENCY_FUNDING_MAP = frozenset({
    "DOT", "Department of Transportation",
    "HUD", "Department of Housing and Urban Development",
    "FTA", "Federal Transit Administration",
    "FHWA", "Federal Highway Administration",
})
AGENCY_LAND_MAP = frozenset({
    "BLM", "Bureau of Land Management",
    "USFS", "FS", "US Forest Service", "Forest Service",
    "NPS", "National Park Service",
    "FWS", "USFWS", "Fish and Wildlife Service",
    "BOR", "USBR", "Bureau of Reclamation",
})
# DOE and USACE require verb disambiguation — not assigned in Tier 1a without verb check
AGENCY_AMBIGUOUS = frozenset({
    "DOE", "Department of Energy",
    "USACE", "Army Corps of Engineers",
})

_AGENCY_CODE_LOOKUP = {
    "BLM": "BLM", "BUREAU OF LAND MANAGEMENT": "BLM",
    "USFS": "USFS", "FS": "USFS", "FOREST SERVICE": "USFS", "US FOREST SERVICE": "USFS",
    "NPS": "NPS", "NATIONAL PARK SERVICE": "NPS",
    "FWS": "FWS", "USFWS": "FWS", "FISH AND WILDLIFE SERVICE": "FWS",
    "BOR": "BOR", "USBR": "BOR", "BUREAU OF RECLAMATION": "BOR",
    "FERC": "FERC", "FAA": "FAA", "FCC": "FCC",
    "DOT": "DOT", "HUD": "HUD", "FTA": "FTA", "FHWA": "FHWA",
    "DOE": "DOE", "USACE": "USACE", "ARMY CORPS OF ENGINEERS": "USACE",
}

# --- federal_action vs federal_land disambiguation ---

FEDERAL_ACTION_VERB_PATTERNS = [
    r'\b(?:proposes?\s+to|will|would)\s+(?:construct|install|build|operate|implement|manage|restore|undertake|develop|upgrade|expand)\b',
    r'\bagency.{0,20}(?:proposes|will\s+construct|will\s+install|will\s+implement)\b',
    r'\b(?:Forest\s+Service|Bureau\s+of\s+Land\s+Management|BLM|USFS|NPS|Bureau\s+of\s+Reclamation)\s+(?:proposes|will|plans\s+to)\b',
    r'\bfederal\s+(?:construction|facility|installation)\b',
    r'\bmilitary\s+(?:installation|base|facility|construction)\b',
]

FEDERAL_LAND_AUTHORIZER_PATTERNS = [
    r'\bright.of.way\s+(?:grant|application|request)\b',
    r'\b(?:applicant|proponent|developer|company)\b',
    r'\bspecial\s+use\s+(?:permit|authorization)\b',
    r'\bwould\s+(?:authorize|approve|grant|allow)\s+(?:a|the)\b',
    r'\b(?:application|request)\s+(?:by|from)\b',
    r'\bhas\s+(?:applied|submitted)\s+(?:an\s+application|a\s+request)\b',
]

# --- federal_program detection ---

PROGRAMMATIC_TITLE_PATTERNS = [
    r'\bprogrammatic\b',
    r'\bprogram[\-\s]?wide\b',
    r'\bpeis\b',
    r'\bpea\b',
]
PROGRAMMATIC_STRONG_PATTERNS = [
    r'(?:draft|final|supplemental)\s+programmatic\s+environmental\s+(?:impact\s+statement|assessment)',
    r'programmatic\s+environmental\s+(?:impact\s+statement|assessment)',
    r'\b(?:dpeis|fpeis|speis|peis|pea)\b',
    r'this\s+programmatic\s+(?:eis|ea|environmental)',
    r'resource\s+management\s+plan\s+(?:amendment|revision)',
    r'\bleasing\s+(?:program|framework)\b',
    r'\bcorridor\s+designation\b',
]
PROGRAMMATIC_EXCLUSION_PATTERNS = [
    r'programmatic\s+agreement',
    r'programmatic\s+biological\s+opinion',
    r'programmatic\s+consultation',
    r'programmatic\s+collaboration',
]

# --- Tier 1b: Title / description keyword patterns ---
# Format: (pattern, trigger_class, rule_slug, confidence)
# More specific / distinctive patterns appear first within each class.

TIER1B_PATTERNS = [
    # federal_program — most distinctive; check before land/permit patterns
    (r'programmatic\s+environmental\s+impact\s+statement', 'federal_program', 'peis', 'high'),
    (r'resource\s+management\s+plan\b', 'federal_program', 'rmp', 'high'),
    (r'leasing\s+(?:program|framework)\b', 'federal_program', 'leasing_prog', 'high'),
    (r'corridor\s+designation\b', 'federal_program', 'corridor', 'high'),
    # federal_property_transaction
    (r'land\s+exchange\b', 'federal_property_transaction', 'land_exchange', 'high'),
    (r'(?:disposal|conveyance)\s+of\s+federal\s+(?:land|property)', 'federal_property_transaction', 'disposal', 'high'),
    (r'parcel\s+transfer\b', 'federal_property_transaction', 'parcel_xfer', 'medium'),
    # federal_permit
    (r'\bSection\s+404\b', 'federal_permit', 'sec404', 'high'),
    (r'Section\s+10\b.{0,50}Rivers\s+and\s+Harbors', 'federal_permit', 'sec10_rha', 'high'),
    (r'\bFERC\b.{0,30}\b(?:licens|authoriz|approv)', 'federal_permit', 'ferc_license', 'high'),
    (r'incidental\s+take\s+permit\b', 'federal_permit', 'itp', 'high'),
    (r'\bNPDES\b', 'federal_permit', 'npdes', 'high'),
    (r'(?:license\s+amendment|permit\s+application)\b', 'federal_permit', 'permit_app', 'medium'),
    # federal_land — ROW and access language
    (r'right.of.way\s+(?:grant|application|request)\b', 'federal_land', 'row_grant', 'high'),
    (r'special\s+use\s+(?:permit|authorization)\b', 'federal_land', 'special_use', 'high'),
    (r'National\s+Forest\s+System\s+lands\b', 'federal_land', 'nfs_land', 'high'),
    (r'(?:BLM|Bureau\s+of\s+Land\s+Management)\s+(?:administered\s+)?lands?\b', 'federal_land', 'blm_land', 'high'),
    (r'crosses?\s+(?:federal|public)\s+lands?\b', 'federal_land', 'crosses_fed', 'high'),
    # federal_funding
    (r'\bTitle\s+XVII\b', 'federal_funding', 'title17', 'high'),
    (r'Inflation\s+Reduction\s+Act\b', 'federal_funding', 'ira', 'high'),
    (r'Bipartisan\s+Infrastructure\s+(?:Law|Act)\b', 'federal_funding', 'bil', 'high'),
    (r'American\s+Recovery\s+and\s+Reinvestment\b', 'federal_funding', 'arra', 'high'),
    (r'loan\s+guarantee\b', 'federal_funding', 'loan_guarantee', 'high'),
    (r'(?:DOE|DOT|HUD|USDA)\s+(?:grant|funding)\b', 'federal_funding', 'agency_grant', 'high'),
    (r'federal\s+(?:financial\s+assistance|grant\b)', 'federal_funding', 'fed_grant', 'medium'),
    # federal_action — agency as actor (more generic; checked last among high-priority classes)
    (r'(?:Forest\s+Service|BLM|USFS|Bureau\s+of\s+Reclamation)\s+(?:proposes\s+to|will)\s+(?:construct|install|implement|manage|restore)', 'federal_action', 'agency_actor', 'high'),
    (r'military\s+(?:installation|base|facility)\b', 'federal_action', 'military', 'high'),
    (r'federal\s+facility\s+(?:upgrade|expansion|construction)\b', 'federal_action', 'fed_facility', 'high'),
    (r'vegetation\s+management\b.{0,50}National\s+Forest', 'federal_action', 'usfs_veg_mgmt', 'high'),
]

# --- Tier 2: Document title patterns ---
# Format: (pattern, trigger_class, rule_slug)
# Programmatic detection uses PROGRAMMATIC_TITLE_PATTERNS + exclusion check (handled separately).

DOC_TITLE_PATTERNS = [
    (r'land\s+exchange', 'federal_property_transaction', 'land_exchange'),
    (r'right.of.way\b', 'federal_land', 'row'),
    (r'permit\s+application\b', 'federal_permit', 'permit_app'),
    (r'license\s+amendment\b', 'federal_permit', 'license_amendment'),
    (r'loan\s+guarantee\b', 'federal_funding', 'loan_guarantee'),
    (r'\bSection\s+404\b', 'federal_permit', 'sec404'),
]

# Tier 3 splits CE from EA/EIS so CE text is handled more conservatively.
TIER3_PATTERNS_EA_EIS = TIER1B_PATTERNS
TIER3_PATTERNS_CE = [
    pattern for pattern in TIER1B_PATTERNS
    if pattern[2] not in {"sec404", "arra", "rmp"}
]

TIER4_CUE_PATTERNS = {
    "federal_funding": [
        r"\bgrant\b",
        r"\bloan\s+guarantee\b",
        r"\bcooperative\s+agreement\b",
        r"\bfederal\s+(?:funding|financial\s+assistance)\b",
        r"\bcost\s+share\b",
        r"\bDOE\s+Funding\b",
        r"\bwould\s+provide\s+(?:approximately\s+)?\d+",
        r"\baward\b",
    ],
    "federal_action": [
        r"\b(?:the\s+)?(?:agency|department|bureau|forest\s+service|western|bonneville)\s+(?:proposes?\s+to|will)\s+(?:construct|install|build|operate|implement|restore|upgrade|develop|expand|demolish)\b",
        r"\b(?:proposes?\s+to|will)\s+(?:construct|install|build|operate|implement|restore|upgrade|develop|expand|demolish)\b",
        r"\bfederal\s+facility\b",
        r"\bproject\s+sponsor\b",
    ],
    "federal_land": [
        r"\bright.of.way\s+(?:grant|renewal|application|request|amendment)\b",
        r"\bspecial\s+use\s+(?:permit|authorization)\b",
        r"\beasement\b",
        r"\bland\s+use\s+permit\b",
        r"\bcross(?:es|ing)?\s+(?:federal|public)\s+land",
        r"\b(?:BLM|Bureau\s+of\s+Land\s+Management|USFS|Forest\s+Service)\b.{0,60}\b(?:land|lands|right-of-way|ROW|permit|authorization)\b",
        r"\badministered\s+by\s+(?:BLM|the\s+Bureau\s+of\s+Land\s+Management|USFS|the\s+Forest\s+Service)\b",
    ],
    "federal_permit": [
        r"\bpermit\s+(?:application|required|is\s+required)\b",
        r"\bauthorization\s+(?:required|requested)\b",
        r"\bCorps\s+permit\b",
        r"\bSection\s+404\b",
        r"\bDepartment\s+of\s+the\s+Army\s+permit\b",
        r"\bNPDES\b",
        r"\bFERC\b.{0,40}\b(?:license|approval|authorize|authorized)\b",
        r"\blicense\s+amendment\b",
    ],
    "federal_program": [
        r"\bprogrammatic\b",
        r"\bPEIS\b",
        r"\bprogrammatic\s+environmental\s+(?:impact\s+statement|assessment)\b",
        r"\bresource\s+management\s+plan\s+(?:amendment|revision)\b",
        r"\bleasing\s+(?:program|framework)\b",
        r"\bpolicy\s+framework\b",
        r"\brulemaking\b",
    ],
    "federal_property_transaction": [
        r"\bland\s+exchange\b",
        r"\bconveyance\b",
        r"\bdisposal\b",
        r"\bproperty\s+transfer\b",
        r"\bacquisition\s+of\s+(?:interests\s+in\s+)?(?:real\s+property|land)\b",
    ],
}

HYPOTHESIS_TEMPLATES = {
    "federal_funding": "This text shows that a federal agency is funding, financing, or providing financial assistance, a grant, or a loan guarantee for this project.",
    "federal_action": "This text shows that a federal agency is directly implementing, constructing, installing, operating, or restoring this project.",
    "federal_land": "This text shows that the project is located on or crosses federal land, or requires a right-of-way grant or special use permit on federal land.",
    "federal_permit": "This text shows that a federal permit, license, or authorization is required for this project.",
    "federal_program": "This text shows that this is a programmatic environmental review, a resource management plan revision, or a land use plan covering a class of actions.",
    "federal_property_transaction": "This text shows that this involves a federal land exchange, conveyance, or disposal.",
}

# Calibration thresholds (--calibrate mode)
CALIBRATION_POSITIVE_THRESHOLD = 0.75   # correct class must score at or above this
CALIBRATION_NEGATIVE_THRESHOLD = 0.50   # all classes must score at or below this

# Positive and hard-negative examples drawn from _example_bank.md.
# Format: (label, correct_class_or_None, chunk_text)
# correct_class=None means hard negative — all classes should score <= threshold.
CALIBRATION_EXAMPLES: list[tuple[str, str | None, str]] = [
    # ── Positive examples ──────────────────────────────────────────────────────
    ("federal_funding / loan guarantee doc title", "federal_funding",
     "FINAL ENVIRONMENTAL ASSESSMENT Volume I for Department of Energy Loan Guarantee "
     "to High Plains II, LLC for the California Valley Solar Ranch Project"),
    ("federal_funding / DOE award with cost share", "federal_funding",
     "NEPA PROVISION\nDOE has made a final NEPA determination for this award\n\n"
     "DOE Funding = $6,999,959\nCost Share = $20,999,876\nTotal Project Cost = $27,999,835"),
    ("federal_funding / DOE proposes to provide funding", "federal_funding",
     "DOE is proposing to provide federal funding to the Contra Costa Economic Partnership "
     "to support local and regional efforts to address and achieve measurable improvements "
     "in market conditions for both commercial and residential rooftop photovoltaic (PV) solar arrays."),
    ("federal_action / DOE constructs NREL facility", "federal_action",
     "The Department of Energy (DOE) prepared this Final Supplemental EA to assess the potential "
     "environmental effects resulting from the proposed improvements to the RFHP. Specifically, the DOE "
     "proposes to develop, construct and operate a woodchip fuel storage silo at the National Renewable "
     "Energy Laboratory's (NREL) South Table Mountain (STM) site in Golden, Colorado."),
    ("federal_action / Western constructs substation", "federal_action",
     "Western Area Power Administration (Western) will construct a new control building at the Lusk Rural "
     "Substation (LRS) located in Niobrara County, Wyoming. The proposed work at the LRS control building "
     "consists of the following; construct a new control building and associated foundation, demolish "
     "existing 69-kV switch, construct new Fault Interrupter foundations and install steel support structure "
     "and fault interrupter, and demolish existing control building."),
    ("federal_action / Western constructs communications building", "federal_action",
     "Western Area Power Administration (Western) will construct a new communications building on the Archer "
     "Microwave Site (ARW). This project will have the following components:\n"
     "* Construct a new communications building\n"
     "* Rebuild the fence along the existing fence line\n"
     "* Conduct the site work necessary to improve the driveway"),
    ("federal_land / USFS special use permit for ROW", "federal_land",
     "Forest Service Purpose and Need The USFS purpose and need is to determine whether to issue a special "
     "use permit for the proposed transmission lines upgrade and rebuild. In conjunction with the issuance, "
     "the USFS would bring Western's facilities under a current authorization with a defined ROW and an "
     "Operation and Maintenance Plan."),
    ("federal_land / BPA perpetual ROW grant from BLM parcel", "federal_land",
     "BPA proposes to acquire a perpetual right-of-way grant for BPA's existing Wautoma-Rock Creek "
     "transmission line across a parcel of land in Klickitat County, Washington. Originally BPA had "
     "acquired a 50-year easement for the right-of-way from Yakama Tribal Allottees. However, the easement "
     "has expired and the Bureau of Land Management now owns the parcel."),
    ("federal_land / BIA ROW renewal", "federal_land",
     "A new right-of-way grant from the Bureau of Indian Affairs for a 25 year term from March 29, 2016 "
     "through March 28, 2041, with the right to extend the right-of-way for an additional 25 years "
     "thorough March 28, 2066."),
    ("federal_permit / NPDES permit required", "federal_permit",
     "This would entail obtaining the National Pollutant Discharge Elimination System (NPDES) permit for "
     "Construction activities. Develop and implement a Stormwater Pollution Prevention (SWPP) Plan/Temporary "
     "Erosion Sediment (TESC) Plan to limit project impacts."),
    ("federal_permit / NPDES permit conditional", "federal_permit",
     "If Task 7.2 'Perform Data Acquisition' requires discharge into that stilling basin, a NPDES permit "
     "must be obtained to comply with Section 402 of the Clean Water Act."),
    ("federal_permit / Army Corps permit application", "federal_permit",
     "Department of the Army Environmental Assessment and Statement of Findings for the Above-Referenced "
     "Standard Individual Permit Application"),
    ("federal_program / programmatic EA doc title", "federal_program",
     "Parker-Davis Transmission System Routine Operation and Maintenance Project and Proposed Integrated "
     "Vegetation Management Program Programmatic Environmental Assessment"),
    ("federal_program / system-wide programmatic EA", "federal_program",
     "Programmatic Environmental Assessment for System-wide Operations and Maintenance Activities "
     "and Integrated Vegetation Management Program"),
    ("federal_program / Draft PEIS title", "federal_program",
     "Upper Great Plains Wind Energy Draft Programmatic Environmental Impact Statement"),
    ("federal_property_transaction / land exchange in title", "federal_property_transaction",
     "Falls Creek Hydroelectric Project and Land Exchange"),
    ("federal_property_transaction / DOE multi-party land exchange", "federal_property_transaction",
     "The U.S. Department of Energy (DOE) is proposing to conduct a multi-party land exchange with "
     "Jefferson County Open Space (JCOS) and the State of Colorado (the State)."),
    ("federal_property_transaction / congressional land exchange", "federal_property_transaction",
     "This final Environmental Impact Statement (EIS) describes a number of alternatives in a historical "
     "context for the purpose of illustrating how the long-term evolution of the project led to the "
     "selection of a new village site to be constructed at Mertarvik on Nelson Island, a site granted to "
     "the village in a land exchange approved by the U.S."),
    # ── Hard negatives ─────────────────────────────────────────────────────────
    ("hard negative / CE checklist unchecked B5.4", None,
     "Conservation, Fossil, and Renewable Energy Activities\n"
     "[ ] B5.3 - Modification (not expansion)/abandonment of oil storage access/\n"
     "brine injection/gas/geothermal wells; no site closure\n"
     "[ ] B5.4 - Repair/replacement of pipeline sections within maintenance\n"
     "provisions of a Section 404 permit\n"
     "[ ] B5.5 - Short crude oil/gas/steam/geothermal pipeline const/oper within a\n"
     "single industrial complex/existing right-of-way"),
    ("hard negative / CE checklist one checked item, B5.4 unchecked", None,
     "Conservation, Fossil, and Renewable Energy Activities\n"
     "[x] B5.1 - Actions to conserve energy, no indoor air quality degradation\n"
     "[ ] B5.4 - Repair/replacement of pipeline sections within maintenance\n"
     "provisions of a Section 404 permit"),
    ("hard negative / ARRA form header unchecked", None,
     "Department of Energy\nCategorical Exclusion Determination Form\n\n"
     "Program or Field Office: Energy Efficiency and Conservation Block Grant Program\n"
     "Project Title MD-City-Bowie\n\n"
     "Proposed Action or Project Description\nAmerican Recovery and Reinvestment Act: [ ]"),
    ("hard negative / ARRA checked box, template text only", None,
     "Department of Energy\nCategorical Exclusion Determination Form\n\n"
     "Program or Field Office: Energy Efficiency and Conservation Block Program\n"
     "Project Title: Energy efficiency lighting and plumbing retrofits for several City facilities.\n\n"
     "American Recovery and Reinvestment Act: [x]"),
    ("hard negative / Part II plan conformance review", None,
     "PART II – PLAN CONFORMANCE REVIEW\n"
     "This proposed action is subject to the following land use plan(s):\n"
     "Safford District Resource Management Plan (RMP and Record of Decision (September 1992).\n\n"
     "Land use authorizations (rights-of-way, leases, permits, easements) will continue to be issued on a "
     "case-by-case basis."),
    ("hard negative / consistent with Forest Plan", None,
     "COMPLIANCE WITH FOREST PLAN\n\n"
     "The proposal is consistent with the approved Forests' Land and Resource Management Plan\n"
     "(As Amended January 2012)."),
    ("hard negative / cultural resource management plan mention", None,
     "Idaho National Laboratory Cultural\nResource Management Plan."),
]

BOILERPLATE_HARD_FILTER_PATTERNS = [
    r"^\s*\[[ xX✓]\]\s*B\d+\.\d+\s*[-–—]",
    r"^\s*B\d+\.\d+\s*[-–—].{0,120}\[\s*[ xX✓]?\s*\]",
    r"(?i)(?:part\s+ii\s*[–—]\s*)?plan\s+conformance\s+review",
    r"(?i)compliance\s+with\s+(?:the\s+)?(?:forest\s+plan|land\s+use\s+plan|resource\s+management\s+plan)",
    r"(?i)the\s+proposed\s+action\s+is\s+(?:consistent\s+with|in\s+conformance\s+with)",
    r"^\s*(?:N/?A|None|TBD|n/a)\s*$",
]

CE_SECTION_PATTERNS = [
    ("project_description", r"(?i)\b(?:project\s+description|proposed\s+action\s+or\s+project\s+description|brief\s+description\s+of\s+(?:proposal|proposed\s+action)|description\s+of\s+(?:the\s+)?proposed\s+action|proposed\s+action\s+description|description\s+of\s+activities)\b"),
    ("proposed_action", r"(?i)\bproposed\s+action\b"),
    ("agency_action", r"(?i)\b(?:agency\s+action|purpose\s+and\s+need|need\s+for\s+agency\s+action|blm'?s\s+purpose\s+and\s+need|forest\s+service\s+purpose\s+and\s+need)\b"),
    ("funding", r"(?i)\b(?:DOE\s+Funding|cost\s+share|financial\s+assistance|loan\s+guarantee)\b"),
]

EA_EIS_SECTION_PATTERNS = [
    ("purpose_and_need", r"(?i)\bpurpose\s+and\s+need\b"),
    ("need_for_action", r"(?i)\bneed\s+for\s+(?:federal\s+)?action\b"),
    ("proposed_action", r"(?i)\bproposed\s+(?:federal\s+)?action\b"),
    ("decision", r"(?i)\bdecision\s+to\s+be\s+made\b"),
    ("agency_action", r"(?i)\b(?:agency\s+action|federal\s+action)\b"),
]

SECTION_PRIOR_WEIGHTS = {
    "doc_title": 0.25,
    "first_pages": 0.10,
    "purpose_and_need": 0.18,
    "need_for_action": 0.18,
    "proposed_action": 0.18,
    "agency_action": 0.18,
    "project_description": 0.15,
    "decision": 0.10,
    "funding": 0.15,
    "cue_window": 0.12,
    "ce_fallback": 0.08,
}

# --- Tier 4: Class prototype sentences for embedding similarity ---

CLASS_PROTOTYPES = {
    "federal_action": [
        "The Forest Service proposes to implement vegetation management on National Forest land.",
        "The Bureau of Land Management will construct a new facility at the site.",
        "This federal action consists of upgrading an existing federal facility.",
        "The agency proposes to build and operate a new transmission substation on federal property.",
    ],
    "federal_program": [
        "This programmatic environmental impact statement evaluates a regional leasing framework.",
        "The Bureau of Land Management is revising its resource management plan.",
        "This PEIS addresses program-wide impacts of a wind energy leasing program.",
    ],
    "federal_land": [
        "The project requires a right-of-way grant across Bureau of Land Management land.",
        "The proposed transmission line would cross National Forest System lands.",
        "The applicant has applied for a special use permit on federal land.",
        "The project is located on lands administered by the Bureau of Land Management.",
    ],
    "federal_permit": [
        "The project requires an individual permit from the U.S. Army Corps of Engineers under Section 404.",
        "FERC approval is required before the project can proceed.",
        "The project requires authorization under Section 10 of the Rivers and Harbors Act.",
        "A federal license amendment from the NRC is required for this project.",
    ],
    "federal_funding": [
        "The project is funded through a Department of Energy loan guarantee.",
        "Federal financial assistance is provided through a DOE grant under Title XVII.",
        "The project is a recipient of federal funding through the Bipartisan Infrastructure Law.",
        "The project receives federal financial assistance from the Department of Transportation.",
    ],
    "federal_property_transaction": [
        "The proposed action consists of a land exchange between the federal government and a private party.",
        "The Bureau of Land Management proposes to convey this federal parcel to the state.",
        "This action involves the disposal of surplus federal land.",
    ],
}

# --- Tier 5: Claude Haiku prompt ---

LLM_PROMPT = """\
You are classifying what triggered a NEPA environmental review for an energy project.

Your job is to identify the primary federal nexus from the evidence bundle below.
Ignore boilerplate, blank checkbox language, plan-conformance headers, and generic legal citations.
Prefer affirmative, project-specific evidence. Distinguish a mere mention from an actual trigger.
Return unknown if the evidence is insufficient.

Classes:
- federal_action: federal agency is the primary actor constructing or implementing the project
- federal_program: programmatic EIS, land-use plan, rulemaking, or leasing framework
- federal_property_transaction: land exchange, disposal, or conveyance
- federal_land: project on or crossing federal land; ROW grant or special use permit tied to land access
- federal_permit: federal permit, license, or authorization is the primary nexus
- federal_funding: federal grant, loan guarantee, or financial assistance
- unknown: cannot determine from the text provided

Project title: {project_title}
Lead agency: {lead_agency}
Dataset source: {dataset_source}
Provisional rule: {provisional_rule}
Provisional class: {provisional_class}
Local class scores: {local_scores}

Retrieved evidence chunks:
{chunks}

Respond with JSON only:
{{"primary": "federal_land", "secondary": ["federal_permit"], "confidence": "high", "reasoning": "..."}}"""

VALID_CLASSES = frozenset({
    "federal_action", "federal_program", "federal_property_transaction",
    "federal_land", "federal_permit", "federal_funding", "unknown",
})

TOP_LEVEL_CLASSES = [
    "federal_funding",
    "federal_action",
    "federal_land",
    "federal_permit",
    "federal_program",
    "federal_property_transaction",
]

# Negation patterns applied to the extracted evidence sentence before accepting a match.
# Catches CE checklist checkboxes ("No [x]") and ordinary sentence-level negations.
_NEGATION_PATTERNS = [
    r'\bno\s*\[(?:x|X|✓|√)\]',
    r'\[(?:x|X|✓|√)\]\s*no\b',
    r'\b(?:would\s+not|will\s+not|does\s+not|do\s+not|did\s+not)\s+require\b',
    r'\bnot\s+(?:require|applicable|trigger|needed|warranted)\b',
    r'\bno\s+(?:permit|authorization|license|funding|grant)\s+(?:is\s+)?required\b',
    r'\bwithdraw(?:n|ing)?\s+(?:the\s+)?(?:permit|application)\b',
    r'\b(?:permit|authorization)\s+(?:is\s+)?not\s+required\b',
    r'\bnot\s+funded\s+(?:by|through|under)\b',
    r'\bdoes\s+not\s+(?:involve|include|require|apply)\b',
]

# --------------------------
# HELPERS
# --------------------------

def _agency_matches(agency: str, agency_map: frozenset) -> bool:
    """True if any token from agency_map appears as a word in the agency string."""
    if not agency:
        return False
    for token in agency_map:
        if re.search(r'\b' + re.escape(token) + r'\b', agency, re.IGNORECASE):
            return True
    return False


def _get_agency_code(agency: str) -> str:
    """Return short agency code for rule_id construction."""
    upper = agency.upper()
    for key, code in _AGENCY_CODE_LOOKUP.items():
        if key in upper:
            return code
    return "UNK"


def _verb_class(text: str) -> Optional[str]:
    """
    Check action vs. authorizer verb signals to distinguish federal_action from federal_land.
    Returns 'federal_action', 'federal_land', or None if no signal found.
    Priority: action verbs win when both are present (federal_action > federal_land).
    """
    for pat in FEDERAL_ACTION_VERB_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "federal_action"
    for pat in FEDERAL_LAND_AUTHORIZER_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "federal_land"
    return None


def _is_programmatic_exclusion(text: str) -> bool:
    for pat in PROGRAMMATIC_EXCLUSION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _is_programmatic_strong(text: str) -> bool:
    for pat in PROGRAMMATIC_STRONG_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _is_programmatic_title(text: str) -> bool:
    for pat in PROGRAMMATIC_TITLE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


_SENTENCE_MODEL = None
_HYPOTHESIS_EMBEDDINGS = None
_LOCAL_SCORER_KIND = None
_CROSS_ENCODER = None


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def make_result(
    project_id: str,
    primary: str,
    confidence: str,
    evidence_text: str,
    evidence_source: str,
    rule_id: str,
    secondary: list[str] | None = None,
    manual_review: bool | None = None,
    notes: str = "",
    route_policy: str = "",
    route_reason: str = "",
    provisional_rule_id: str = "",
    provisional_confidence: str = "",
) -> dict[str, Any]:
    secondary = _unique_preserve_order(secondary or [])
    multi = [] if primary == "unknown" else _unique_preserve_order([primary] + secondary)
    if manual_review is None:
        manual_review = (confidence == "low")
    return {
        "project_id": project_id,
        "nepa_trigger_primary": primary,
        "nepa_trigger_secondary": secondary,
        "nepa_trigger_multi": multi,
        "nepa_trigger_evidence_text": evidence_text or "",
        "nepa_trigger_evidence_source": evidence_source or "",
        "nepa_trigger_confidence": confidence,
        "nepa_trigger_rule_id": rule_id,
        "nepa_trigger_manual_review": bool(manual_review),
        "nepa_trigger_llm_run_at": "",
        "_route_policy": route_policy,
        "_route_reason": route_reason,
        "_provisional_rule_id": provisional_rule_id or rule_id,
        "_provisional_confidence": provisional_confidence or confidence,
        "_notes": notes,
    }


def _make_unknown(
    project_id: str,
    rule_id: str = "no_match",
    evidence_text: str = "",
    evidence_source: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return make_result(
        project_id=project_id,
        primary="unknown",
        confidence="low",
        evidence_text=evidence_text,
        evidence_source=evidence_source,
        rule_id=rule_id,
        manual_review=True,
        notes=notes,
        route_policy="manual_review",
        route_reason="unknown",
    )


def _safe_sql_list(values) -> str:
    items = [str(v).replace(chr(39), "") for v in values if pd.notna(v)]
    return ", ".join(f"'{item}'" for item in items) if items else "''"


def _page_sort_key(page_number: Any) -> int:
    text = str(page_number or "")
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 999999


def _result_confidence_rank(result: dict[str, Any]) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(result.get("nepa_trigger_confidence", "low"), 0)


def extract_sentence(text: str, match: re.Match) -> str:
    """Extract the full sentence containing the regex match position."""
    start, end = match.start(), match.end()
    region_before = text[max(0, start - 500): start]
    sent_start = max(0, start - 500)
    for sep in (". ", ".\n", "?\n", "!\n", "\n\n"):
        idx = region_before.rfind(sep)
        if idx >= 0:
            sent_start = max(0, start - 500) + idx + len(sep)
            break
    region_after = text[end: min(len(text), end + 500)]
    sent_end = min(len(text), end + 500)
    for sep in (". ", ".\n", "?\n", "!\n", "\n\n"):
        idx = region_after.find(sep)
        if idx >= 0:
            sent_end = end + idx + 1
            break
    return text[sent_start:sent_end].strip()


def extract_purpose_and_need(text: str, window: int = PAN_WINDOW) -> str:
    """Return up to `window` chars following the first Purpose and Need header found."""
    for pat in [
        r"\bpurpose\s+and\s+need\b",
        r"\bneed\s+for\s+(?:federal\s+)?action\b",
        r"\bproposed\s+(?:federal\s+)?action\b",
        r"\bproject\s+purpose\b",
        r"\bstatement\s+of\s+need\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return text[m.start(): m.start() + window]
    return ""


def _apply_pattern_list(
    project_id: str,
    text: str,
    patterns: list,
    evidence_source: str,
    tier_prefix: str,
) -> Optional[dict[str, Any]]:
    for pat, trigger_class, rule_slug, confidence in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        if trigger_class == "federal_program" and _is_programmatic_exclusion(text):
            continue
        evidence = extract_sentence(text, m)
        if any(re.search(np, evidence, re.IGNORECASE) for np in _NEGATION_PATTERNS):
            continue
        return make_result(
            project_id=project_id,
            primary=trigger_class,
            confidence=confidence,
            evidence_text=evidence,
            evidence_source=evidence_source,
            rule_id=f"{tier_prefix}_{rule_slug}",
            manual_review=(confidence == "low"),
            route_policy="auto_accept" if f"{tier_prefix}_{rule_slug}" in AUTO_ACCEPT_RULE_IDS else "provisional",
            route_reason=f"pattern_match:{rule_slug}",
        )
    return None


def _project_metadata_priors(project_row: pd.Series) -> list[str]:
    agency = str(project_row.get("lead_agency_harmonized") or "")
    priors: list[str] = []
    if _agency_matches(agency, frozenset({"DOE", "Department of Energy"})):
        priors.extend(["federal_funding", "federal_action"])
    elif _agency_matches(agency, frozenset({"USACE", "Army Corps of Engineers"})):
        priors.extend(["federal_permit", "federal_land"])
    elif _agency_matches(agency, AGENCY_LAND_MAP):
        priors.extend(["federal_land", "federal_action", "federal_program"])
    elif _agency_matches(agency, AGENCY_PERMIT_MAP):
        priors.append("federal_permit")
    elif _agency_matches(agency, AGENCY_FUNDING_MAP):
        priors.append("federal_funding")
    return _unique_preserve_order(priors)


def should_auto_accept(result: dict[str, Any]) -> bool:
    rule_id = result.get("nepa_trigger_rule_id", "")
    if rule_id in SEND_TO_TIER4_RULE_IDS or rule_id in AUDIT_FIRST_RULE_IDS:
        return False
    if rule_id in AUTO_ACCEPT_RULE_IDS:
        return True
    if rule_id.startswith("T4_local_") and rule_id != "T4_local_uncertain":
        return result.get("nepa_trigger_confidence") in ("high", "medium")
    if rule_id.startswith("T4_embed_"):
        return result.get("nepa_trigger_confidence") in ("high", "medium")
    if rule_id == "T5_llm":
        return result.get("nepa_trigger_confidence") in ("high", "medium")
    if result.get("nepa_trigger_evidence_source") == "agency_metadata":
        evidence_text = result.get("nepa_trigger_evidence_text", "")
        if _agency_matches(evidence_text, AMBIGUOUS_METADATA_AGENCIES):
            return False
    return False


def should_send_to_tier4(result: dict[str, Any] | None) -> bool:
    if result is None:
        return True
    if should_auto_accept(result):
        return False
    if result.get("nepa_trigger_rule_id") in SEND_TO_TIER4_RULE_IDS:
        return True
    if result.get("nepa_trigger_rule_id") in AUDIT_FIRST_RULE_IDS:
        return True
    if result.get("nepa_trigger_evidence_source") == "agency_metadata":
        return True
    return result.get("nepa_trigger_confidence") != "high"


def build_tier4_candidate_ids(
    all_project_ids: set[str],
    provisional: dict[str, dict[str, Any]],
    finalized: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(pid for pid in all_project_ids if pid not in finalized and should_send_to_tier4(provisional.get(pid)))


def _select_preferred_documents(docs_df: pd.DataFrame) -> pd.DataFrame:
    if docs_df.empty:
        return docs_df
    docs = docs_df.copy()
    docs["main_rank"] = docs["main_document"].fillna("").astype(str).str.upper().eq("YES").astype(int)
    docs["title_rank"] = docs["document_title"].fillna("").astype(str).str.len()
    docs = docs.sort_values(["project_id", "main_rank", "title_rank", "document_id"], ascending=[True, False, False, True])
    return docs.groupby("project_id", as_index=False).head(1).drop(columns=["main_rank", "title_rank"])


def _fetch_source_docs_pages(
    project_ids: list[str],
    dataset_source: str,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_upper = str(dataset_source).upper()
    docs_path = DOCS_PATH_MAP.get(source_upper)
    pages_path = PAGES_PATH_MAP.get(source_upper)
    if docs_path is None or pages_path is None or not docs_path.exists() or not pages_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    docs = conn.execute(f"""
        SELECT
            project_id.value AS project_id,
            document_id,
            document_title,
            file_name,
            main_document
        FROM read_parquet('{docs_path}')
        WHERE project_id.value IN ({_safe_sql_list(project_ids)})
    """).fetchdf()
    docs = _select_preferred_documents(docs)
    if docs.empty:
        return docs, pd.DataFrame()

    pages = conn.execute(f"""
        SELECT document_id, page_number, page_text
        FROM read_parquet('{pages_path}')
        WHERE document_id IN ({_safe_sql_list(docs['document_id'].tolist())})
          AND page_text IS NOT NULL
    """).fetchdf()
    if not pages.empty:
        pages["page_sort"] = pages["page_number"].map(_page_sort_key)
    return docs, pages


def _normalize_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _chunk_hash(text: str) -> str:
    return hashlib.sha1(_normalize_chunk_text(text).encode("utf-8")).hexdigest()


def _extract_section_windows(text: str, patterns: list[tuple[str, str]], window: int, max_sections: int = 6) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    seen = set()
    for section_type, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = match.start()
            end = min(len(text), start + window)
            key = (section_type, start)
            if key in seen:
                continue
            seen.add(key)
            sections.append({
                "section_type": section_type,
                "chunk_text": text[start:end].strip(),
            })
            if len(sections) >= max_sections:
                return sections
    return sections


def _extract_cue_windows(text: str, window_before: int = 280, window_after: int = 520, max_windows: int = 8) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    seen_hashes = set()
    for cue_patterns in TIER4_CUE_PATTERNS.values():
        for pattern in cue_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                sentence = extract_sentence(text, match)
                if any(re.search(np, sentence, re.IGNORECASE) for np in _NEGATION_PATTERNS):
                    continue
                start = max(0, match.start() - window_before)
                end = min(len(text), match.end() + window_after)
                chunk_text = text[start:end].strip()
                chunk_key = _chunk_hash(chunk_text)
                if chunk_key in seen_hashes:
                    continue
                seen_hashes.add(chunk_key)
                sections.append({
                    "section_type": "cue_window",
                    "chunk_text": chunk_text,
                })
                if len(sections) >= max_windows:
                    return sections
                break
    return sections


def extract_ce_candidate_sections(text: str) -> list[dict[str, Any]]:
    sections = _extract_section_windows(text, CE_SECTION_PATTERNS, window=1600, max_sections=8)
    sections.extend(_extract_cue_windows(text))
    if not sections:
        sections.append({
            "section_type": "ce_fallback",
            "chunk_text": text[:3000].strip(),
        })
    return sections


def extract_ea_eis_candidate_sections(
    section_text: str,
    cue_text: Optional[str] = None,
) -> list[dict[str, Any]]:
    sections = _extract_section_windows(section_text, EA_EIS_SECTION_PATTERNS, window=2000, max_sections=8)
    sections.extend(_extract_cue_windows(cue_text if cue_text is not None else section_text))
    return sections


def _has_named_entity_like(text: str) -> bool:
    if re.search(r"\b(?:DOE|BLM|USFS|Forest Service|Bureau of Land Management|Bonneville Power Administration|Western Area Power Administration|Federal Energy Regulatory Commission|U\.S\.)\b", text):
        return True
    return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))


def _is_hard_filtered_chunk(chunk_text: str) -> bool:
    lines = [line.strip() for line in str(chunk_text or "").splitlines() if line.strip()]
    if not lines:
        return True
    hard_matches = 0
    ce_rows = 0
    for line in lines:
        if re.search(r"^\s*\[[ xX✓]?\]\s*B\d+\.\d+\s*[-–—]", line) or re.search(r"^\s*B\d+\.\d+\s*[-–—].{0,120}\[\s*[ xX✓]?\s*\]", line):
            ce_rows += 1
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in BOILERPLATE_HARD_FILTER_PATTERNS):
            hard_matches += 1
    return (hard_matches / len(lines) > 0.5) or (ce_rows / len(lines) > 0.5)


def _boilerplate_penalty(chunk_text: str) -> float:
    penalty = 0.0
    ce_rows = len(re.findall(r"\[[ xX✓]?\]\s*B\d+\.\d+", chunk_text))
    penalty += min(0.6, 0.3 * ce_rows)
    if re.search(r"(?i)(?:plan\s+conformance|consistent\s+with\s+the\s+approved\s+(?:forest|land|resource)\s+management\s+plan)", chunk_text):
        penalty += 0.2
    if not _has_named_entity_like(chunk_text):
        penalty += 0.2
    return min(1.0, penalty)


def score_chunk_cues(
    chunk_text: str,
    dataset_source: str,
    section_type: str = "",
    metadata_prior_classes: list[str] | None = None,
) -> dict[str, Any]:
    metadata_prior_classes = metadata_prior_classes or []
    if _is_hard_filtered_chunk(chunk_text):
        out = {
            "cue_classes": [],
            "cue_rules": [],
            "cue_score": 0.0,
            "retrieval_score": 0.0,
            "boilerplate_penalty": 1.0,
            "hard_filtered": True,
        }
        for cls in TOP_LEVEL_CLASSES:
            out[f"cue_score_{cls}"] = 0.0
        return out

    raw_scores = {cls: 0.0 for cls in TOP_LEVEL_CLASSES}
    cue_classes: list[str] = []
    cue_rules: list[str] = []
    section_bonus = SECTION_PRIOR_WEIGHTS.get(section_type, 0.0)

    for cls in TOP_LEVEL_CLASSES:
        for idx, pattern in enumerate(TIER4_CUE_PATTERNS[cls]):
            match = re.search(pattern, chunk_text, re.IGNORECASE)
            if not match:
                continue
            evidence = extract_sentence(chunk_text, match)
            if any(re.search(np, evidence, re.IGNORECASE) for np in _NEGATION_PATTERNS):
                continue
            raw_scores[cls] += 0.18
            cue_classes.append(cls)
            cue_rules.append(f"{cls}:{idx}")
            break
        if raw_scores[cls] > 0:
            raw_scores[cls] += section_bonus
            if cls in metadata_prior_classes:
                raw_scores[cls] += 0.10

    if section_type == "doc_title":
        for pat, trigger_class, _rule_slug in DOC_TITLE_PATTERNS:
            if re.search(pat, chunk_text, re.IGNORECASE):
                raw_scores[trigger_class] = max(raw_scores[trigger_class], 0.55)
        if _is_programmatic_title(chunk_text) and not _is_programmatic_exclusion(chunk_text) and _is_programmatic_strong(chunk_text):
            raw_scores["federal_program"] = max(raw_scores["federal_program"], 0.60)

    boilerplate_penalty = _boilerplate_penalty(chunk_text)
    penalized_scores = {cls: max(0.0, min(1.0, score - boilerplate_penalty)) for cls, score in raw_scores.items()}
    retrieval_score = max(penalized_scores.values()) if penalized_scores else 0.0
    out = {
        "cue_classes": _unique_preserve_order(cue_classes),
        "cue_rules": _unique_preserve_order(cue_rules),
        "cue_score": retrieval_score,
        "retrieval_score": retrieval_score,
        "boilerplate_penalty": boilerplate_penalty,
        "hard_filtered": False,
    }
    for cls in TOP_LEVEL_CLASSES:
        out[f"cue_score_{cls}"] = penalized_scores[cls]
    return out


def dedupe_chunks(chunks: pd.DataFrame) -> pd.DataFrame:
    if chunks.empty:
        return chunks
    deduped = (
        chunks.sort_values(["project_id", "retrieval_score"], ascending=[True, False])
        .groupby(["project_id", "chunk_hash"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return deduped


def build_tier4_contexts(
    project_ids: list[str],
    projects_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    if not project_ids:
        return pd.DataFrame()

    target_df = projects_df[projects_df["project_id"].isin(set(project_ids))].copy()
    if target_df.empty:
        return pd.DataFrame()

    project_lookup = target_df.set_index("project_id")
    rows: list[dict[str, Any]] = []

    for source, group in target_df.groupby("dataset_source"):
        docs_df, pages_df = _fetch_source_docs_pages(list(group["project_id"]), source, conn)
        if not pages_df.empty:
            pages_df = pages_df.merge(docs_df[["project_id", "document_id"]], on="document_id", how="left")
            pages_df = pages_df.sort_values(["project_id", "page_sort", "page_number"])

        for pid in group["project_id"]:
            project_row = project_lookup.loc[pid]
            metadata_priors = _project_metadata_priors(project_row)
            project_docs = docs_df[docs_df["project_id"] == pid] if not docs_df.empty else pd.DataFrame()
            page_rows = pages_df[pages_df["project_id"] == pid] if not pages_df.empty else pd.DataFrame()
            chunk_count = 0

            if not project_docs.empty:
                doc_row = project_docs.iloc[0]
                document_id = doc_row["document_id"]
                document_title = str(doc_row.get("document_title") or "").strip()
                if document_title:
                    rows.append({
                        "project_id": pid,
                        "document_id": document_id,
                        "dataset_source": source,
                        "chunk_id": f"{pid}_doc_title",
                        "section_type": "doc_title",
                        "page_start": None,
                        "page_end": None,
                        "chunk_text": document_title,
                        "metadata_prior_classes": metadata_priors,
                    })
                    chunk_count += 1
            else:
                document_id = ""

            if not page_rows.empty:
                page_texts = page_rows["page_text"].fillna("").astype(str).tolist()
                full_text = "\n\n".join(page_texts).strip()
                if full_text:
                    if str(source).upper() == "CE":
                        sections = extract_ce_candidate_sections(full_text)
                    else:
                        first_pages_text = "\n\n".join(page_texts[:MAX_PAGES_PAN]).strip()
                        if first_pages_text:
                            rows.append({
                                "project_id": pid,
                                "document_id": document_id,
                                "dataset_source": source,
                                "chunk_id": f"{pid}_first_pages",
                                "section_type": "first_pages",
                                "page_start": page_rows["page_number"].iloc[0],
                                "page_end": page_rows["page_number"].iloc[min(len(page_rows) - 1, MAX_PAGES_PAN - 1)],
                                "chunk_text": first_pages_text,
                                "metadata_prior_classes": metadata_priors,
                            })
                            chunk_count += 1
                        sections = extract_ea_eis_candidate_sections(first_pages_text or full_text, cue_text=full_text)

                    for idx, section in enumerate(sections):
                        chunk_text = str(section.get("chunk_text") or "").strip()
                        if not chunk_text:
                            continue
                        rows.append({
                            "project_id": pid,
                            "document_id": document_id,
                            "dataset_source": source,
                            "chunk_id": f"{pid}_{section['section_type']}_{idx}",
                            "section_type": section["section_type"],
                            "page_start": None,
                            "page_end": None,
                            "chunk_text": chunk_text,
                            "metadata_prior_classes": metadata_priors,
                        })
                        chunk_count += 1

            if chunk_count == 0:
                fallback_text = " ".join([
                    str(project_row.get("project_title") or ""),
                    str(project_row.get("project_description") or ""),
                ]).strip()
                if fallback_text:
                    rows.append({
                        "project_id": pid,
                        "document_id": document_id,
                        "dataset_source": source,
                        "chunk_id": f"{pid}_fallback",
                        "section_type": "ce_fallback" if str(source).upper() == "CE" else "first_pages",
                        "page_start": None,
                        "page_end": None,
                        "chunk_text": fallback_text[:3000],
                        "metadata_prior_classes": metadata_priors,
                    })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    raw["chunk_hash"] = raw["chunk_text"].map(_chunk_hash)
    scored = raw.apply(
        lambda row: pd.Series(
            score_chunk_cues(
                chunk_text=row["chunk_text"],
                dataset_source=row["dataset_source"],
                section_type=row["section_type"],
                metadata_prior_classes=row["metadata_prior_classes"],
            )
        ),
        axis=1,
    )
    chunks = pd.concat([raw, scored], axis=1)
    chunks = chunks[~chunks["hard_filtered"]].copy()
    if chunks.empty:
        return chunks

    chunks["section_priority"] = chunks["section_type"].map(lambda value: SECTION_PRIOR_WEIGHTS.get(value, 0.0))
    positive = chunks[chunks["retrieval_score"] > 0].copy()
    fallback_ids = set(chunks["project_id"]) - set(positive["project_id"])

    if fallback_ids:
        fallback = (
            chunks[chunks["project_id"].isin(fallback_ids)]
            .sort_values(["project_id", "section_priority", "chunk_text"], ascending=[True, False, True])
            .groupby("project_id", as_index=False)
            .head(TIER4_TOP_K)
        )
        positive = pd.concat([positive, fallback], ignore_index=True)

    chunks = dedupe_chunks(positive)
    return (
        chunks.sort_values(["project_id", "retrieval_score", "section_priority"], ascending=[True, False, False])
        .groupby("project_id", as_index=False)
        .head(TIER4_TOP_K)
        .drop(columns=["section_priority"], errors="ignore")
        .reset_index(drop=True)
    )


def _load_local_adjudicator() -> str:
    global _SENTENCE_MODEL, _HYPOTHESIS_EMBEDDINGS, _LOCAL_SCORER_KIND, _CROSS_ENCODER

    if _LOCAL_SCORER_KIND is not None:
        return _LOCAL_SCORER_KIND

    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder(LOCAL_NLI_MODEL, local_files_only=False)
        _LOCAL_SCORER_KIND = "nli"
        return _LOCAL_SCORER_KIND
    except Exception as exc:
        log.info(
            "Tier 4 local NLI model not available from local cache; falling back to embeddings (%s: %s)",
            type(exc).__name__,
            exc,
        )

    _get_sentence_model()

    import numpy as np

    hypothesis_embeddings = _SENTENCE_MODEL.encode(
        [HYPOTHESIS_TEMPLATES[cls] for cls in TOP_LEVEL_CLASSES],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _HYPOTHESIS_EMBEDDINGS = {
        cls: hypothesis_embeddings[idx]
        for idx, cls in enumerate(TOP_LEVEL_CLASSES)
    }
    _LOCAL_SCORER_KIND = "embedding"
    return _LOCAL_SCORER_KIND


def _get_sentence_model():
    global _SENTENCE_MODEL

    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for Tier 4") from exc

    _SENTENCE_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _SENTENCE_MODEL


def get_candidate_classes(
    project_row: pd.Series,
    provisional_result: dict | None,
    cue_classes: list[str],
) -> list[str]:
    candidates: list[str] = []
    agency = str(project_row.get("lead_agency_harmonized") or "")

    if provisional_result:
        primary = provisional_result.get("nepa_trigger_primary")
        if primary in VALID_CLASSES and primary != "unknown":
            candidates.append(primary)
        candidates.extend([cls for cls in provisional_result.get("nepa_trigger_secondary", []) if cls in VALID_CLASSES and cls != "unknown"])
        if provisional_result.get("nepa_trigger_rule_id") in AUDIT_FIRST_RULE_IDS:
            candidates.extend(["federal_program", "federal_land"])

    candidates.extend([cls for cls in cue_classes if cls in VALID_CLASSES and cls != "unknown"])

    if _agency_matches(agency, frozenset({"DOE", "Department of Energy"})):
        candidates.extend(["federal_funding", "federal_action"])
    elif _agency_matches(agency, frozenset({"USACE", "Army Corps of Engineers"})):
        candidates.extend(["federal_permit", "federal_land"])
    elif _agency_matches(agency, AGENCY_LAND_MAP):
        candidates.extend(["federal_land", "federal_action", "federal_program"])
    elif _agency_matches(agency, AGENCY_PERMIT_MAP):
        candidates.append("federal_permit")
    elif _agency_matches(agency, AGENCY_FUNDING_MAP):
        candidates.append("federal_funding")

    candidates = _unique_preserve_order([cls for cls in candidates if cls in TOP_LEVEL_CLASSES])
    if not candidates:
        candidates = TOP_LEVEL_CLASSES.copy()
    return candidates


def run_local_nli_on_chunks(
    chunks_df: pd.DataFrame,
    candidate_classes_by_project: dict[str, list[str]],
) -> pd.DataFrame:
    if chunks_df.empty:
        return pd.DataFrame()

    strategy = _load_local_adjudicator()
    records: list[dict[str, Any]] = []

    if strategy == "nli" and _CROSS_ENCODER is not None:
        id2label = {int(k): str(v).lower() for k, v in getattr(_CROSS_ENCODER.model.config, "id2label", {}).items()}
        pairs = []
        row_meta = []
        for row in chunks_df.itertuples(index=False):
            for candidate_class in candidate_classes_by_project.get(row.project_id, TOP_LEVEL_CLASSES):
                pairs.append((row.chunk_text, HYPOTHESIS_TEMPLATES[candidate_class]))
                row_meta.append((row, candidate_class))
        predictions = _CROSS_ENCODER.predict(pairs, apply_softmax=True, show_progress_bar=False)
        for meta, pred in zip(row_meta, predictions):
            row, candidate_class = meta
            pred_list = pred.tolist() if hasattr(pred, "tolist") else list(pred)
            entailment_score = 0.0
            if id2label:
                for idx, score in enumerate(pred_list):
                    if id2label.get(idx, "").startswith("entail"):
                        entailment_score = float(score)
                        break
            elif pred_list:
                entailment_score = float(pred_list[-1])
            cue_score = float(getattr(row, f"cue_score_{candidate_class}", 0.0))
            metadata_bonus = 0.03 if candidate_class in (row.metadata_prior_classes or []) else 0.0
            final_score = min(1.0, max(0.0, (0.65 * entailment_score) + (0.35 * cue_score) + metadata_bonus))
            records.append({
                "project_id": row.project_id,
                "chunk_id": row.chunk_id,
                "section_type": row.section_type,
                "chunk_text": row.chunk_text,
                "candidate_class": candidate_class,
                "cue_score_class": cue_score,
                "retrieval_score": float(row.retrieval_score),
                "model_score": entailment_score,
                "final_score": final_score,
                "is_supporting_chunk": bool(cue_score >= TIER4_SUPPORT_THRESHOLD or entailment_score >= 0.82),
                "metadata_prior_classes": row.metadata_prior_classes,
            })
    else:
        import numpy as np

        assert _SENTENCE_MODEL is not None
        assert _HYPOTHESIS_EMBEDDINGS is not None
        text_embeddings = _SENTENCE_MODEL.encode(
            chunks_df["chunk_text"].tolist(),
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        for idx, row in enumerate(chunks_df.itertuples(index=False)):
            for candidate_class in candidate_classes_by_project.get(row.project_id, TOP_LEVEL_CLASSES):
                hypothesis_emb = _HYPOTHESIS_EMBEDDINGS[candidate_class]
                model_score = (float(np.dot(text_embeddings[idx], hypothesis_emb)) + 1.0) / 2.0
                cue_score = float(getattr(row, f"cue_score_{candidate_class}", 0.0))
                metadata_bonus = 0.03 if candidate_class in (row.metadata_prior_classes or []) else 0.0
                final_score = min(1.0, max(0.0, (0.65 * model_score) + (0.35 * cue_score) + metadata_bonus))
                records.append({
                    "project_id": row.project_id,
                    "chunk_id": row.chunk_id,
                    "section_type": row.section_type,
                    "chunk_text": row.chunk_text,
                    "candidate_class": candidate_class,
                    "cue_score_class": cue_score,
                    "retrieval_score": float(row.retrieval_score),
                    "model_score": model_score,
                    "final_score": final_score,
                    "is_supporting_chunk": bool(cue_score >= TIER4_SUPPORT_THRESHOLD or model_score >= 0.82),
                    "metadata_prior_classes": row.metadata_prior_classes,
                })

    return pd.DataFrame(records)


def aggregate_tier4_scores(chunk_scores: pd.DataFrame) -> pd.DataFrame:
    if chunk_scores.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for project_id, group in chunk_scores.groupby("project_id"):
        class_rows = []
        for candidate_class, class_group in group.groupby("candidate_class"):
            ordered = class_group.sort_values(["final_score", "cue_score_class", "retrieval_score"], ascending=False)
            best_row = ordered.iloc[0]
            support_count = int(class_group["is_supporting_chunk"].sum())
            affirmative_support = bool((class_group["cue_score_class"] >= TIER4_SUPPORT_THRESHOLD).any())
            doc_score = min(1.0, float(best_row["final_score"]) + (0.03 * max(0, support_count - 1)))
            class_rows.append({
                "candidate_class": candidate_class,
                "doc_score": doc_score,
                "affirmative_support": affirmative_support,
                "support_count": support_count,
                "best_chunk_text": best_row["chunk_text"],
                "best_section_type": best_row["section_type"],
            })

        class_rows.sort(key=lambda item: item["doc_score"], reverse=True)
        top = class_rows[0] if class_rows else None
        second_score = class_rows[1]["doc_score"] if len(class_rows) > 1 else 0.0
        metadata_priors = group["metadata_prior_classes"].iloc[0] if len(group) else []
        threshold = TIER4_BASE_THRESHOLD if metadata_priors else TIER4_NO_PRIOR_THRESHOLD
        margin = (top["doc_score"] - second_score) if top else 0.0
        contradictory = bool(top and second_score >= max(0.0, top["doc_score"] - TIER4_CONTRADICTION_WINDOW))

        reasons = []
        if not top:
            reasons.append("no_class_scores")
        else:
            if top["doc_score"] < threshold:
                reasons.append("weak_top_score")
            if margin < TIER4_MARGIN_THRESHOLD:
                reasons.append("small_margin")
            if not top["affirmative_support"]:
                reasons.append("no_affirmative_support")
            if contradictory:
                reasons.append("contradictory_scores")

        top_chunks = (
            group.sort_values(["retrieval_score", "final_score"], ascending=False)
            .drop_duplicates("chunk_id")
            .head(TIER4_TOP_K)
        )
        top_chunks_json = json.dumps([
            {
                "chunk_id": row["chunk_id"],
                "section_type": row["section_type"],
                "score": round(float(row["retrieval_score"]), 3),
                "text": row["chunk_text"][:1200],
            }
            for _, row in top_chunks.iterrows()
        ])
        local_scores_json = json.dumps({
            item["candidate_class"]: round(item["doc_score"], 3)
            for item in class_rows
        })

        rows.append({
            "project_id": project_id,
            "top_class": top["candidate_class"] if top else "unknown",
            "top_class_score": float(top["doc_score"]) if top else 0.0,
            "second_class_score": float(second_score),
            "margin": float(margin),
            "threshold": float(threshold),
            "auto_resolve": bool(top and not reasons),
            "uncertainty_reason": ",".join(reasons) if reasons else "",
            "top_evidence_text": top["best_chunk_text"] if top else "",
            "top_evidence_source": "doc_title" if top and top["best_section_type"] == "doc_title" else "document_text",
            "local_scores_json": local_scores_json,
            "top_chunks_json": top_chunks_json,
        })

    return pd.DataFrame(rows)


def _serialize_diag_lists(df: pd.DataFrame, list_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in list_cols:
        if col in out.columns:
            out[col] = out[col].apply(json.dumps)
    return out


def _write_tier4_diagnostics(
    contexts: pd.DataFrame,
    chunk_scores: pd.DataFrame,
    doc_scores: pd.DataFrame,
) -> None:
    if contexts is not None:
        contexts_to_write = _serialize_diag_lists(contexts, ["metadata_prior_classes", "cue_classes", "cue_rules"])
        contexts_to_write.to_parquet(CONTEXT_CANDIDATES_PATH, index=False)
    if chunk_scores is not None:
        chunk_to_write = _serialize_diag_lists(chunk_scores, ["metadata_prior_classes"])
        chunk_to_write.to_parquet(TIER4_CHUNK_SCORES_PATH, index=False)
    if doc_scores is not None:
        doc_scores.to_parquet(TIER4_DOC_SCORES_PATH, index=False)


def tier1a_metadata(projects: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for _, row in projects.iterrows():
        pid = row["project_id"]
        agency = str(row.get("lead_agency_harmonized") or "").strip()
        text = " ".join([
            str(row.get("project_title") or ""),
            str(row.get("project_description") or ""),
        ])
        agency_code = _get_agency_code(agency)

        if _agency_matches(agency, AGENCY_PERMIT_MAP):
            results.append(make_result(
                project_id=pid,
                primary="federal_permit",
                confidence="high",
                evidence_text=agency,
                evidence_source="agency_metadata",
                rule_id=f"T1a_{agency_code}_permit",
                manual_review=False,
                route_policy="auto_accept",
                route_reason="deterministic_permit_metadata",
            ))
        elif _agency_matches(agency, AGENCY_FUNDING_MAP):
            results.append(make_result(
                project_id=pid,
                primary="federal_funding",
                confidence="high",
                evidence_text=agency,
                evidence_source="agency_metadata",
                rule_id=f"T1a_{agency_code}_funding",
                manual_review=False,
                route_policy="auto_accept",
                route_reason="deterministic_funding_metadata",
            ))
        elif _agency_matches(agency, AGENCY_LAND_MAP):
            verb_class = _verb_class(text)
            trigger = verb_class if verb_class else "federal_land"
            confidence = "high" if verb_class else "medium"
            verb_suffix = "action" if trigger == "federal_action" else "land"
            results.append(make_result(
                project_id=pid,
                primary=trigger,
                confidence=confidence,
                evidence_text=agency,
                evidence_source="agency_metadata",
                rule_id=f"T1a_{agency_code}_{verb_suffix}",
                manual_review=(confidence != "high"),
                route_policy="auto_accept" if confidence == "high" and f"T1a_{agency_code}_{verb_suffix}" in AUTO_ACCEPT_RULE_IDS else "provisional",
                route_reason="land_agency_metadata",
            ))
        elif _agency_matches(agency, frozenset({"DOE", "Department of Energy"})):
            doe_funding_patterns = [
                r"\b(?:loan\s+guarantee|financial\s+assistance|cooperative\s+agreement)\b",
                r"\bTitle\s+XVII\b",
                r"\b(?:ARRA|Recovery\s+Act|Bipartisan\s+Infrastructure|Inflation\s+Reduction\s+Act)\b",
                r"\bfunded\s+(?:by|through|under)\b",
                r"\b(?:DOE|Department\s+of\s+Energy)\s+(?:grant|award|funding)\b",
            ]
            has_funding = any(re.search(p, text, re.IGNORECASE) for p in doe_funding_patterns)
            if has_funding:
                results.append(make_result(
                    project_id=pid,
                    primary="federal_funding",
                    confidence="medium",
                    evidence_text=agency,
                    evidence_source="agency_metadata",
                    rule_id="T1a_DOE_funding",
                    manual_review=True,
                    route_policy="tier4_candidate",
                    route_reason="doe_metadata_ambiguous",
                ))
            elif _verb_class(text) == "federal_action":
                results.append(make_result(
                    project_id=pid,
                    primary="federal_action",
                    confidence="medium",
                    evidence_text=agency,
                    evidence_source="agency_metadata",
                    rule_id="T1a_DOE_action",
                    manual_review=True,
                    route_policy="tier4_candidate",
                    route_reason="doe_metadata_ambiguous",
                ))
    return results


def tier1b_title_description(projects: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for _, row in projects.iterrows():
        pid = row["project_id"]
        text = " ".join([
            str(row.get("project_title") or ""),
            str(row.get("project_description") or ""),
        ])
        if not text.strip():
            continue

        if _is_programmatic_title(text) and not _is_programmatic_exclusion(text) and _is_programmatic_strong(text):
            results.append(make_result(
                project_id=pid,
                primary="federal_program",
                confidence="high",
                evidence_text=text[:300].strip(),
                evidence_source="description",
                rule_id="T1b_programmatic_title",
                manual_review=False,
                route_policy="auto_accept" if "T1b_programmatic_title" in AUTO_ACCEPT_RULE_IDS else "provisional",
                route_reason="strong_programmatic_title",
            ))
            continue

        result = _apply_pattern_list(pid, text, TIER1B_PATTERNS, "description", "T1b")
        if result:
            results.append(result)
    return results


def tier2_doc_title(
    unresolved_ids: list[str],
    projects_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    if not unresolved_ids:
        return []

    unresolved_df = projects_df[projects_df["project_id"].isin(set(unresolved_ids))]
    results: dict[str, dict[str, Any]] = {}

    for source, group in unresolved_df.groupby("dataset_source"):
        docs_df, _pages_df = _fetch_source_docs_pages(list(group["project_id"]), source, conn)
        if docs_df.empty:
            continue

        for _, row in docs_df.iterrows():
            pid = row["project_id"]
            if pid in results:
                continue
            title = str(row.get("document_title") or "")
            if not title.strip():
                continue

            if _is_programmatic_title(title) and not _is_programmatic_exclusion(title):
                results[pid] = make_result(
                    project_id=pid,
                    primary="federal_program",
                    confidence="high",
                    evidence_text=title,
                    evidence_source="doc_title",
                    rule_id="T2_doc_title_peis",
                    manual_review=False,
                    route_policy="auto_accept",
                    route_reason="doc_title_programmatic",
                )
                continue

            for pat, trigger_class, rule_slug in DOC_TITLE_PATTERNS:
                if re.search(pat, title, re.IGNORECASE):
                    results[pid] = make_result(
                        project_id=pid,
                        primary=trigger_class,
                        confidence="high",
                        evidence_text=title,
                        evidence_source="doc_title",
                        rule_id=f"T2_doc_title_{rule_slug}",
                        manual_review=False,
                        route_policy="auto_accept" if f"T2_doc_title_{rule_slug}" in AUTO_ACCEPT_RULE_IDS else "provisional",
                        route_reason=f"doc_title:{rule_slug}",
                    )
                    break

    return list(results.values())


def tier3_purpose_and_need(
    unresolved_ids: list[str],
    projects_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> list[dict[str, Any]]:
    if not unresolved_ids:
        return []

    unresolved_df = projects_df[projects_df["project_id"].isin(set(unresolved_ids))]
    results: dict[str, dict[str, Any]] = {}

    for source, group in unresolved_df.groupby("dataset_source"):
        docs_df, pages_df = _fetch_source_docs_pages(list(group["project_id"]), source, conn)
        if docs_df.empty or pages_df.empty:
            continue
        pages_df = pages_df.merge(docs_df[["project_id", "document_id"]], on="document_id", how="left")
        pages_df = pages_df.sort_values(["project_id", "page_sort", "page_number"])

        for pid, page_group in pages_df.groupby("project_id"):
            if pid in results:
                continue
            page_texts = page_group["page_text"].fillna("").astype(str).tolist()
            full_text = "\n\n".join(page_texts).strip()
            if not full_text:
                continue

            is_ce = str(source).upper() == "CE"
            candidate_sections = extract_ce_candidate_sections(full_text) if is_ce else extract_ea_eis_candidate_sections("\n\n".join(page_texts[:MAX_PAGES_PAN]))
            patterns = TIER3_PATTERNS_CE if is_ce else TIER3_PATTERNS_EA_EIS

            for section in candidate_sections:
                chunk_text = str(section.get("chunk_text") or "").strip()
                if not chunk_text or _is_hard_filtered_chunk(chunk_text):
                    continue
                evidence_source = "document_text" if is_ce else "purpose_and_need"
                result = _apply_pattern_list(pid, chunk_text, patterns, evidence_source, "T3")
                if result:
                    results[pid] = result
                    break

    return list(results.values())


def tier4_embedding(
    unresolved_ids: list[str],
    projects_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not unresolved_ids:
        return []

    try:
        import numpy as np
    except ImportError:
        log.warning("numpy not installed; skipping Tier 4 embedding fallback")
        return []

    model = _get_sentence_model()
    centroids = {}
    for cls, sentences in CLASS_PROTOTYPES.items():
        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        centroids[cls] = np.mean(embeddings, axis=0)

    unresolved_df = projects_df[projects_df["project_id"].isin(set(unresolved_ids))]
    results = []

    for _, row in unresolved_df.iterrows():
        pid = row["project_id"]
        candidate_text = " ".join([
            str(row.get("project_title") or ""),
            str(row.get("project_description") or ""),
        ]).strip()

        if not candidate_text:
            results.append(_make_unknown(pid, rule_id="T4_embed_no_text", evidence_source="embedding"))
            continue

        emb = model.encode([candidate_text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = {cls: float(np.dot(emb, centroid)) for cls, centroid in centroids.items()}
        best_cls = max(sims, key=sims.get)
        best_score = sims[best_cls]

        if best_score >= EMBEDDING_THRESHOLD:
            results.append(make_result(
                project_id=pid,
                primary=best_cls,
                confidence="medium",
                evidence_text=candidate_text[:300],
                evidence_source="embedding",
                rule_id=f"T4_embed_{best_cls}",
                manual_review=False,
                route_policy="fallback_auto_accept",
                route_reason="embedding_fallback",
                notes=f"embedding_score={best_score:.3f}",
            ))
        else:
            results.append(_make_unknown(
                pid,
                rule_id="T4_embed_below_threshold",
                evidence_text=candidate_text[:300],
                evidence_source="embedding",
                notes=f"embedding_score={best_score:.3f}",
            ))

    return results


def tier4_retrieval_local(
    candidate_ids: list[str],
    projects_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
    provisional: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    provisional = provisional or {}
    if not candidate_ids:
        empty = pd.DataFrame()
        return [], empty, empty, empty

    contexts = build_tier4_contexts(candidate_ids, projects_df, conn)
    if contexts.empty:
        return tier4_embedding(candidate_ids, projects_df), contexts, pd.DataFrame(), pd.DataFrame()

    project_lookup = projects_df.set_index("project_id")
    cue_classes_by_project = (
        contexts.groupby("project_id")["cue_classes"]
        .apply(lambda series: _unique_preserve_order([cls for values in series for cls in values]))
        .to_dict()
    )
    candidate_classes_by_project = {
        pid: get_candidate_classes(project_lookup.loc[pid], provisional.get(pid), cue_classes_by_project.get(pid, []))
        for pid in candidate_ids
        if pid in project_lookup.index
    }

    chunk_scores = run_local_nli_on_chunks(contexts, candidate_classes_by_project)
    doc_scores = aggregate_tier4_scores(chunk_scores)
    results: list[dict[str, Any]] = []
    covered_ids = set()

    if not doc_scores.empty:
        for _, row in doc_scores.iterrows():
            pid = row["project_id"]
            covered_ids.add(pid)
            provisional_result = provisional.get(pid, {})
            if row["auto_resolve"]:
                confidence = "high" if row["top_class_score"] >= 0.95 else "medium"
                results.append(make_result(
                    project_id=pid,
                    primary=row["top_class"],
                    confidence=confidence,
                    evidence_text=row["top_evidence_text"][:300],
                    evidence_source=row["top_evidence_source"],
                    rule_id=f"T4_local_{row['top_class']}",
                    manual_review=False,
                    route_policy="auto_accept",
                    route_reason="strong_retrieved_evidence",
                    provisional_rule_id=provisional_result.get("nepa_trigger_rule_id", ""),
                    provisional_confidence=provisional_result.get("nepa_trigger_confidence", ""),
                    notes=f"top_score={row['top_class_score']:.3f};margin={row['margin']:.3f}",
                ))
            else:
                results.append(make_result(
                    project_id=pid,
                    primary="unknown",
                    confidence="low",
                    evidence_text=str(row["top_evidence_text"])[:300],
                    evidence_source=row["top_evidence_source"],
                    rule_id="T4_local_uncertain",
                    manual_review=True,
                    route_policy="tier5_candidate",
                    route_reason=row["uncertainty_reason"],
                    provisional_rule_id=provisional_result.get("nepa_trigger_rule_id", ""),
                    provisional_confidence=provisional_result.get("nepa_trigger_confidence", ""),
                    notes=row["local_scores_json"],
                ))

    missing_ids = [pid for pid in candidate_ids if pid not in covered_ids]
    if missing_ids:
        log.info("  Tier 4 embedding fallback on %s projects with no usable retrieved contexts", len(missing_ids))
        results.extend(tier4_embedding(missing_ids, projects_df))

    return results, contexts, chunk_scores, doc_scores


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def estimate_tier5_spend(queue_df: pd.DataFrame) -> float:
    return len(queue_df) * ESTIMATED_TIER5_COST_PER_PROJECT


def build_tier5_queue(
    low_conf_ids: list[str],
    doc_scores: pd.DataFrame,
    projects_df: pd.DataFrame,
    provisional: dict[str, dict[str, Any]],
    tier4_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if not low_conf_ids:
        return pd.DataFrame()

    projects_lookup = projects_df.set_index("project_id")
    doc_lookup = doc_scores.set_index("project_id") if not doc_scores.empty else pd.DataFrame()
    rows = []

    for pid in low_conf_ids:
        if pid not in projects_lookup.index:
            continue
        project_row = projects_lookup.loc[pid]
        provisional_result = provisional.get(pid, {})
        tier4_result = tier4_results.get(pid, {})

        if not doc_lookup.empty and pid in doc_lookup.index:
            doc_row = doc_lookup.loc[pid]
        else:
            doc_row = None

        if doc_row is not None:
            top_chunks_json = doc_row["top_chunks_json"]
            local_scores_json = doc_row["local_scores_json"]
            tier4_reason = doc_row["uncertainty_reason"]
        else:
            fallback_text = " ".join([
                str(project_row.get("project_title") or ""),
                str(project_row.get("project_description") or ""),
            ]).strip()[:1200]
            top_chunks_json = json.dumps([{
                "chunk_id": f"{pid}_fallback",
                "section_type": "project_text",
                "score": 0.0,
                "text": fallback_text,
            }])
            local_scores_json = json.dumps({})
            tier4_reason = tier4_result.get("_route_reason", "no_retrieved_evidence")

        rows.append({
            "project_id": pid,
            "project_title": str(project_row.get("project_title") or ""),
            "lead_agency_harmonized": str(project_row.get("lead_agency_harmonized") or ""),
            "dataset_source": str(project_row.get("dataset_source") or ""),
            "provisional_rule_id": provisional_result.get("nepa_trigger_rule_id", ""),
            "provisional_class": provisional_result.get("nepa_trigger_primary", ""),
            "local_scores_json": local_scores_json,
            "top_chunks_json": top_chunks_json,
            "tier4_rule_id": tier4_result.get("nepa_trigger_rule_id", ""),
            "tier4_reason": tier4_reason,
        })

    return pd.DataFrame(rows)


def tier5_llm(
    candidate_df: pd.DataFrame,
    projects_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    if candidate_df.empty:
        return []

    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed; skipping Tier 5")
        return []

    client = anthropic.Anthropic()
    results = []

    for _, row in candidate_df.iterrows():
        pid = row["project_id"]
        chunks = json.loads(row.get("top_chunks_json") or "[]")
        chunk_text = "\n\n".join(
            f"[{idx + 1}] ({chunk.get('section_type', 'chunk')}, score={chunk.get('score', 0)}) {chunk.get('text', '')}"
            for idx, chunk in enumerate(chunks[:TIER4_TOP_K])
        )

        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": LLM_PROMPT.format(
                        project_title=row.get("project_title", ""),
                        lead_agency=row.get("lead_agency_harmonized", ""),
                        dataset_source=row.get("dataset_source", ""),
                        provisional_rule=row.get("provisional_rule_id", ""),
                        provisional_class=row.get("provisional_class", ""),
                        local_scores=row.get("local_scores_json", "{}"),
                        chunks=chunk_text,
                    ),
                }],
            )
            raw = response.content[0].text.strip()
            parsed = _parse_json_object(raw)
            primary = parsed.get("primary", "unknown")
            if primary not in VALID_CLASSES:
                primary = "unknown"
            secondary = [
                cls for cls in parsed.get("secondary", [])
                if cls in VALID_CLASSES and cls != primary
            ]
            confidence = parsed.get("confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            result = make_result(
                project_id=pid,
                primary=primary,
                confidence=confidence,
                evidence_text=(chunks[0]["text"] if chunks else "")[:300],
                evidence_source="llm",
                rule_id="T5_llm",
                secondary=secondary,
                manual_review=(confidence == "low"),
                route_policy="llm",
                route_reason=row.get("tier4_reason", ""),
            )
            result["nepa_trigger_llm_run_at"] = datetime.now(timezone.utc).isoformat()
            results.append(result)
        except Exception as exc:
            log.warning("Tier 5 failure for %s: %s", pid, exc)
            results.append(_make_unknown(
                pid,
                rule_id="T5_llm_error",
                evidence_text=(chunks[0]["text"] if chunks else "")[:300],
                evidence_source="llm",
                notes=str(exc),
            ))

    return results


def build_validation_batches(
    df: pd.DataFrame,
    projects_df: pd.DataFrame,
    sample_per_rule: int = 20,
    sample_per_process: int = 20,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    merged = df.merge(
        projects_df[["project_id", "process_type", "dataset_source", "lead_agency_harmonized"]],
        on="project_id",
        how="left",
    )
    batches = []

    for rule_id, group in merged.groupby("nepa_trigger_rule_id"):
        sample = group.sample(min(sample_per_rule, len(group)), random_state=42).copy()
        sample["validation_batch"] = f"rule::{rule_id}"
        sample["batch_kind"] = "rule"
        sample["batch_size"] = len(group)
        batches.append(sample)

    for process_type, group in merged.groupby("process_type"):
        sample = group.sample(min(sample_per_process, len(group)), random_state=42).copy()
        sample["validation_batch"] = f"process::{process_type}"
        sample["batch_kind"] = "process"
        sample["batch_size"] = len(group)
        batches.append(sample)

    doe_mask = merged["lead_agency_harmonized"].fillna("").astype(str).str.contains(r"DOE|Department of Energy", case=False, regex=True)
    if doe_mask.any():
        doe_group = merged[doe_mask]
        sample = doe_group.sample(min(sample_per_process, len(doe_group)), random_state=42).copy()
        sample["validation_batch"] = "agency::DOE"
        sample["batch_kind"] = "agency"
        sample["batch_size"] = len(doe_group)
        batches.append(sample)

    ce_mask = merged["dataset_source"].fillna("").astype(str).str.upper().eq("CE")
    if ce_mask.any():
        ce_group = merged[ce_mask]
        sample = ce_group.sample(min(sample_per_process, len(ce_group)), random_state=42).copy()
        sample["validation_batch"] = "dataset::CE"
        sample["batch_kind"] = "dataset"
        sample["batch_size"] = len(ce_group)
        batches.append(sample)

    return pd.concat(batches, ignore_index=True).sort_values(["batch_kind", "batch_size"], ascending=[True, False])


def run_eda(conn: duckdb.DuckDBPyConnection) -> None:
    result = conn.execute(f"""
        SELECT
            process_type,
            COUNT(*) AS n,
            SUM(CASE WHEN project_description IS NOT NULL
                      AND LENGTH(project_description) > 50 THEN 1 ELSE 0 END) AS n_with_desc,
            ROUND(AVG(LENGTH(project_description)), 0) AS avg_desc_len
        FROM read_parquet('{PROJECTS_PATH}')
        WHERE {CLEAN_ENERGY_FILTER}
        GROUP BY process_type
        ORDER BY process_type
    """).fetchdf()
    print("\n=== Description coverage by process type (clean energy projects only) ===")
    print(result.to_string(index=False))
    print(
        "\nNote: Low n_with_desc for CE projects means Tier 1b will have reduced coverage;\n"
        "      Tier 3 (full CE document scan) is the primary fallback for CEs.\n"
    )


def extract_nepa_triggers(
    conn: duckdb.DuckDBPyConnection,
    use_llm: bool = False,
    sample: Optional[int] = None,
    force_tier5: bool = False,
    tier5_budget: float = TIER5_HARD_STOP_BUDGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    projects = conn.execute(f"""
        SELECT project_id, lead_agency_harmonized, project_title,
               project_description, process_type, dataset_source
        FROM read_parquet('{PROJECTS_PATH}')
        WHERE {CLEAN_ENERGY_FILTER}
    """).fetchdf()

    if sample:
        projects = projects.sample(sample, random_state=42)
        log.info("Sample mode: %s projects", len(projects))

    all_project_ids = set(projects["project_id"])
    log.info("Processing %s clean energy projects", f"{len(all_project_ids):,}")

    finalized: dict[str, dict[str, Any]] = {}
    provisional: dict[str, dict[str, Any]] = {}

    def _ingest(results: list[dict[str, Any]]) -> None:
        for result in results:
            pid = result["project_id"]
            if should_auto_accept(result):
                finalized[pid] = result
            else:
                existing = provisional.get(pid)
                if existing is None or _result_confidence_rank(result) >= _result_confidence_rank(existing):
                    provisional[pid] = result

    def _remaining() -> list[str]:
        return sorted(pid for pid in all_project_ids if pid not in finalized)

    def _pct() -> str:
        return f"{len(finalized) / len(all_project_ids):.1%}"

    log.info("Tier 1a: agency metadata")
    _ingest(tier1a_metadata(projects))
    log.info("  → %s finalized (%s)", f"{len(finalized):,}", _pct())

    log.info("Tier 1b: title and description keywords")
    unresolved_df = projects[projects["project_id"].isin(_remaining())]
    _ingest(tier1b_title_description(unresolved_df))
    log.info("  → %s finalized (%s)", f"{len(finalized):,}", _pct())

    log.info("Tier 2: document title scan")
    _ingest(tier2_doc_title(_remaining(), projects, conn))
    log.info("  → %s finalized (%s)", f"{len(finalized):,}", _pct())

    log.info("Tier 3: purpose-and-need / candidate section extraction")
    _ingest(tier3_purpose_and_need(_remaining(), projects, conn))
    log.info("  → %s finalized (%s)", f"{len(finalized):,}", _pct())

    tier4_ids = build_tier4_candidate_ids(all_project_ids, provisional, finalized)
    log.info("Tier 4: retrieval-first local adjudication on %s projects", f"{len(tier4_ids):,}")
    tier4_results, contexts, chunk_scores, doc_scores = tier4_retrieval_local(tier4_ids, projects, conn, provisional)
    _write_tier4_diagnostics(contexts, chunk_scores, doc_scores)

    tier4_low_conf: dict[str, dict[str, Any]] = {}
    tier4_result_lookup = {result["project_id"]: result for result in tier4_results}
    for result in tier4_results:
        if result["nepa_trigger_confidence"] in ("high", "medium"):
            finalized[result["project_id"]] = result
        else:
            tier4_low_conf[result["project_id"]] = result
    log.info("  → %s finalized after Tier 4 (%s)", f"{len(finalized):,}", _pct())

    if use_llm:
        low_conf_ids = sorted(tier4_low_conf)
        queue_df = build_tier5_queue(low_conf_ids, doc_scores, projects, provisional, tier4_result_lookup)
        if not queue_df.empty:
            queue_df.to_parquet(TIER5_QUEUE_PATH, index=False)
            estimated_spend = estimate_tier5_spend(queue_df)
            log.info(
                "Tier 5 queue preflight: %s projects, estimated spend about $%.2f",
                f"{len(queue_df):,}",
                estimated_spend,
            )
            if len(queue_df) > TIER5_SOFT_WARNING:
                log.warning("Tier 5 queue exceeds soft warning threshold (%s)", TIER5_SOFT_WARNING)
            if len(queue_df) > TIER5_TARGET_QUEUE:
                log.warning("Tier 5 queue exceeds target threshold (%s)", TIER5_TARGET_QUEUE)
            if estimated_spend > tier5_budget and not force_tier5:
                raise SystemExit(
                    f"Tier 5 queue written to {TIER5_QUEUE_PATH}, but estimated spend "
                    f"(${estimated_spend:.2f}) exceeds budget (${tier5_budget:.2f}). "
                    "Re-run with --force-tier5 to override."
                )
            llm_results = tier5_llm(queue_df, projects)
            for result in llm_results:
                finalized[result["project_id"]] = result
        else:
            log.info("Tier 5: no uncertain queue to process")
    else:
        log.info("Tier 5: skipped (--use-llm not set)")

    for pid, result in tier4_low_conf.items():
        if pid not in finalized:
            finalized[pid] = result

    for pid in all_project_ids:
        if pid not in finalized:
            finalized[pid] = _make_unknown(
                pid,
                rule_id="unresolved_after_tier4",
                evidence_text=provisional.get(pid, {}).get("nepa_trigger_evidence_text", ""),
                evidence_source=provisional.get(pid, {}).get("nepa_trigger_evidence_source", ""),
                notes=provisional.get(pid, {}).get("nepa_trigger_rule_id", ""),
            )

    final = pd.DataFrame(list(finalized.values()))
    return final, projects


def run_calibration() -> bool:
    """
    Validate HYPOTHESIS_TEMPLATES against the example bank before a full corpus run.

    Passing criteria:
      - Positive examples: correct class entailment score >= CALIBRATION_POSITIVE_THRESHOLD (0.75)
      - Hard negatives:    all class scores <= CALIBRATION_NEGATIVE_THRESHOLD (0.50)

    Returns True if all checks pass, False otherwise.
    Run with:  python 01_extract_nepa_trigger.py --calibrate
    """
    strategy = _load_local_adjudicator()
    if strategy != "nli" or _CROSS_ENCODER is None:
        log.error("NLI model not available — cannot calibrate. Check that sentence-transformers is installed.")
        return False

    id2label = {
        int(k): str(v).lower()
        for k, v in getattr(_CROSS_ENCODER.model.config, "id2label", {}).items()
    }

    def _entailment(chunk: str, hypothesis: str) -> float:
        pred = _CROSS_ENCODER.predict([[chunk, hypothesis]], apply_softmax=True, show_progress_bar=False)[0]
        pred_list = pred.tolist() if hasattr(pred, "tolist") else list(pred)
        if id2label:
            for idx, score in enumerate(pred_list):
                if id2label.get(idx, "").startswith("entail"):
                    return float(score)
        return float(pred_list[-1]) if pred_list else 0.0

    all_pass = True
    positives = [(lbl, cls, chunk) for lbl, cls, chunk in CALIBRATION_EXAMPLES if cls is not None]
    negatives = [(lbl, chunk) for lbl, cls, chunk in CALIBRATION_EXAMPLES if cls is None]

    print(f"\n{'=' * 70}")
    print(f"POSITIVE EXAMPLES  (correct class must score >= {CALIBRATION_POSITIVE_THRESHOLD:.2f})")
    print("=" * 70)
    for label, correct_class, chunk in positives:
        score = _entailment(chunk, HYPOTHESIS_TEMPLATES[correct_class])
        status = "PASS" if score >= CALIBRATION_POSITIVE_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}]  {score:.3f}  {label}")
        if status == "FAIL":
            print(f"       class={correct_class}")
            print(f"       hypothesis: {HYPOTHESIS_TEMPLATES[correct_class]}")
            print(f"       chunk: {chunk[:120].strip()}...")

    print(f"\n{'=' * 70}")
    print(f"HARD NEGATIVES  (all classes must score <= {CALIBRATION_NEGATIVE_THRESHOLD:.2f})")
    print("=" * 70)
    for label, chunk in negatives:
        scores = {cls: _entailment(chunk, hyp) for cls, hyp in HYPOTHESIS_TEMPLATES.items()}
        worst_cls = max(scores, key=scores.__getitem__)
        worst_score = scores[worst_cls]
        status = "PASS" if worst_score <= CALIBRATION_NEGATIVE_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}]  max={worst_score:.3f} ({worst_cls})  {label}")
        if status == "FAIL":
            print(f"       all scores: { {k: round(v, 3) for k, v in scores.items()} }")
            print(f"       chunk: {chunk[:120].strip()}...")

    print(f"\n{'=' * 70}")
    if all_pass:
        print("ALL CHECKS PASSED — hypotheses are well-calibrated for a full corpus run.")
    else:
        print("SOME CHECKS FAILED — adjust HYPOTHESIS_TEMPLATES and re-run --calibrate before proceeding.")
    print("=" * 70 + "\n")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1: NEPA Trigger Classification — 20,725 clean energy projects"
    )
    parser.add_argument("--eda", action="store_true", help="Run EDA check only; do not extract")
    parser.add_argument("--calibrate", action="store_true", help="Validate NLI hypothesis templates against the example bank; do not extract")
    parser.add_argument("--use-llm", action="store_true", help="Enable Tier 5 Claude Haiku fallback for uncertain cases")
    parser.add_argument("--force-tier5", action="store_true", help="Override Tier 5 budget guardrail")
    parser.add_argument("--tier5-budget", type=float, default=TIER5_HARD_STOP_BUDGET, help="Hard stop budget for Tier 5")
    parser.add_argument("--sample", type=int, default=None, help="Process only N projects (random sample; for testing)")
    args = parser.parse_args()

    conn = duckdb.connect()

    if args.calibrate:
        log.info("Loading NLI model for calibration (downloads ~67MB on first run)...")
        passed = run_calibration()
        sys.exit(0 if passed else 1)

    if args.eda:
        run_eda(conn)
        return

    run_at = datetime.now(timezone.utc).isoformat()
    final, projects = extract_nepa_triggers(
        conn,
        use_llm=args.use_llm,
        sample=args.sample,
        force_tier5=args.force_tier5,
        tier5_budget=args.tier5_budget,
    )

    final["is_dual_nexus"] = (
        (final["nepa_trigger_primary"] == "federal_land") &
        (final["nepa_trigger_secondary"].apply(lambda x: "federal_permit" in x if isinstance(x, list) else False))
    )
    final["nepa_trigger_extraction_run_at"] = run_at
    final["nepa_trigger_llm_run_at"] = final.get("nepa_trigger_llm_run_at", "").fillna("")

    assert final["project_id"].is_unique, "Duplicate project_ids in output — check tier logic"
    assert final["project_id"].isin(set(projects["project_id"])).all(), "Non-clean project IDs in output"
    assert final["nepa_trigger_secondary"].apply(isinstance, args=(list,)).all(), "nepa_trigger_secondary must be list type"

    final = final[OUTPUT_COLS]

    out_path = OUTPUT_DIR / "projects_nepa_trigger.parquet"
    final.to_parquet(out_path, index=False)
    log.info("Written: %s (%s rows)", out_path, f"{len(final):,}")

    batches = build_validation_batches(final, projects)
    if not batches.empty:
        batch_path = OUTPUT_DIR / "validation_batches.csv"
        batches.to_csv(batch_path, index=False)
        log.info(
            "Validation batches: %s sampled rows across %s batches → %s",
            f"{len(batches):,}",
            batches["validation_batch"].nunique(),
            batch_path,
        )
    else:
        log.info("No validation batches were generated.")

    print("\n=== Primary trigger distribution ===")
    print(final["nepa_trigger_primary"].value_counts().to_string())
    print("\n=== Confidence distribution ===")
    print(final["nepa_trigger_confidence"].value_counts().to_string())
    print("\n=== Evidence source distribution ===")
    print(final["nepa_trigger_evidence_source"].value_counts().to_string())
    print(f"\nDual-nexus projects (federal_land + federal_permit): {final['is_dual_nexus'].sum():,}")


if __name__ == "__main__":
    main()
