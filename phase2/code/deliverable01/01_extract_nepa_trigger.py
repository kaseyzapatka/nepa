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
#   data/analysis/deliverable01/projects_nepa_trigger.parquet  (one row per project)
#   data/validation/deliverable01/validation_batches.csv        (flagged cases grouped by rule)

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
ANALYSIS_DIR  = DATA_DIR / "analysis"
OUTPUT_DIR    = ANALYSIS_DIR / "deliverable01"
TRAINING_DIR  = DATA_DIR / "training" / "deliverable01"
VALIDATION_DIR = DATA_DIR / "validation" / "deliverable01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

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
    "nepa_trigger_count", "nepa_trigger_combo", "nepa_trigger_primary_hierarchy",
    "nepa_trigger_evidence_text", "nepa_trigger_evidence_source",
    "nepa_trigger_confidence", "nepa_trigger_rule_id",
    "nepa_trigger_manual_review", "is_dual_nexus",
    "nepa_trigger_extraction_run_at", "nepa_trigger_llm_run_at",
]

CONTEXT_CANDIDATES_PATH = OUTPUT_DIR / "context_candidates.parquet"
TIER4_CHUNK_SCORES_PATH = OUTPUT_DIR / "tier4_chunk_scores.parquet"
TIER4_DOC_SCORES_PATH = OUTPUT_DIR / "tier4_doc_scores.parquet"
TIER5_QUEUE_PATH = OUTPUT_DIR / "tier5_queue.parquet"
PROJECTS_NEPA_TRIGGER_PATH = OUTPUT_DIR / "projects_nepa_trigger.parquet"
PROJECTS_FUNDING_DETAILS_PATH = OUTPUT_DIR / "projects_funding_details.parquet"

FUNDING_DETAIL_COLS = [
    "project_id",
    "federal_funding_type_primary",
    "federal_funding_type_multi",
    "federal_funding_program_multi",
    "federal_funding_amount_usd",
    "federal_funding_total_project_cost_usd",
    "federal_funding_recipient_cost_share_usd",
    "federal_funding_share_pct",
    "federal_funding_evidence_text",
    "federal_funding_evidence_source",
    "federal_funding_confidence",
    "federal_funding_manual_review",
    "federal_funding_amount_candidates_json",
    "federal_funding_extraction_run_at",
]

TIER4_TOP_K = 4
TIER4_BASE_THRESHOLD = 0.72
TIER4_NO_PRIOR_THRESHOLD = 0.78
TIER4_MARGIN_THRESHOLD = 0.08
TIER4_CONTRADICTION_WINDOW = 0.10
TIER4_SUPPORT_THRESHOLD = 0.25

TIER5_TARGET_QUEUE = 250
TIER5_SOFT_WARNING = 150
TIER5_HARD_STOP_BUDGET = 10.0
ESTIMATED_TIER5_COST_PER_PROJECT = 0.004  # measured 2026-07: ~$1.80 / 501 projects ≈ $0.0036/project at claude-haiku-4-5 pricing ($1/$5 per MTok), rounded up

LOCAL_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

SETFIT_MODEL_PATH        = Path("phase2/models/trigger_setfit")
# Ground-truth label files used as Tier 0 pass-through during inference so that
# training examples are never re-classified by the pipeline. Named by the model
# each file feeds: SetFit (DOE CE classifier) and NLI (EA/EIS classifier).
SETFIT_TRAINING_FILES    = (
    TRAINING_DIR / "doe_ce_samples.csv",
)
NLI_TRAINING_FILES       = (
    TRAINING_DIR / "ea_eis_samples.csv",
)
SETFIT_CONFIDENCE_THRESHOLD = 0.65
SETFIT_MARGIN_THRESHOLD     = 0.08

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
    "T1a_BPA_pma",
    "T1a_WAPA_pma",
    "T1a_SEPA_pma",
    "T1a_SWPA_pma",
    "T1a_TVA_pma",
    "T1a_PMA_pma",
    "T1a_CBP_direct_action",
    "T1a_BLM_USFS_land_control",
    "T1b_doe_export_auth",
    "T1b_presidential_permit_decision",
    "T1b_doe_gas_export_auth",
    "T1b_pma_name_in_description",
})

SEND_TO_TIER4_RULE_IDS = frozenset({
    "T1a_DOE_direct_action",
    "T1a_DOE_funding",
    "T3_sec404",
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
AGENCY_PMA_MAP = frozenset({
    "Power Marketing Administration", "PMA",
    "Bonneville Power Administration", "BPA",
    "Western Area Power Administration", "WAPA",
    "Southeastern Power Administration", "SEPA",
    "Southwestern Power Administration", "SWPA",
    "Tennessee Valley Authority", "TVA",
})
AGENCY_DIRECT_ACTION_MAP = frozenset({
    "CBP",
    "U.S. Customs and Border Protection",
    "Customs and Border Protection",
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
    "BPA": "BPA", "BONNEVILLE POWER ADMINISTRATION": "BPA",
    "WAPA": "WAPA", "WESTERN AREA POWER ADMINISTRATION": "WAPA",
    "SEPA": "SEPA", "SOUTHEASTERN POWER ADMINISTRATION": "SEPA",
    "SWPA": "SWPA", "SOUTHWESTERN POWER ADMINISTRATION": "SWPA",
    "TVA": "TVA", "TENNESSEE VALLEY AUTHORITY": "TVA",
    "CBP": "CBP", "U.S. CUSTOMS AND BORDER PROTECTION": "CBP", "CUSTOMS AND BORDER PROTECTION": "CBP",
    "POWER MARKETING ADMINISTRATION": "PMA",
}

FOREST_SERVICE_SPONSOR_PATTERN = (
    r"\b(?:USDA(?:,\s*)?)?(?:U\.?S\.?\s*)?Forest\s+Service\b|"
    r"\bUSFS\b"
)
BLM_USFS_LAND_CONTROL_PATTERN = (
    r"\bright[-\s]of[-\s]way\b|"
    r"\bROW\b|"
    r"\bwithdrawal\b|"
    r"\bPublic\s+Land\s+Order\b|"
    r"\bFLPMA\b"
)
CLEAN_WATER_ACT_PERMIT_CONTEXT_PATTERN = (
    r"\bClean\s+Water\s+Act\b[\s\S]{0,180}\b(?:"
    r"Section\s+404\b[\s\S]{0,80}\bpermit(?:\s+application)?\b|"
    r"permit\s+application\b|"
    r"issue\s+a\s+permit(?:\s+with\s+conditions)?\b|"
    r"deny\s+a\s+permit\b|"
    r"Section\s+10\b"
    r")"
    r"|"
    r"\b(?:"
    r"Section\s+404\b[\s\S]{0,80}\bpermit(?:\s+application)?\b|"
    r"permit\s+application\b|"
    r"issue\s+a\s+permit(?:\s+with\s+conditions)?\b|"
    r"deny\s+a\s+permit\b|"
    r"Section\s+10\b"
    r")[\s\S]{0,180}\bClean\s+Water\s+Act\b"
)

# --- federal_direct_action vs federal_land disambiguation ---

FEDERAL_ACTION_ACTOR_PATTERN = (
    r'(?:DOE|Department\s+of\s+Energy|NNSA|National\s+Nuclear\s+Security\s+Administration|'
    r'BPA|Bonneville(?:\s+Power\s+Administration)?|'
    r'WAPA|Western(?:\s+Area\s+Power\s+Administration)?|'
    r'Reclamation|Bureau\s+of\s+Reclamation|USBR|'
    r'CBP|U\.S\.\s+Customs\s+and\s+Border\s+Protection|'
    r'Forest\s+Service|U\.S\.\s+Forest\s+Service|USFS|'
    r'Bureau\s+of\s+Land\s+Management|BLM|'
    r'NPS|National\s+Park\s+Service|'
    r'PNNL|Pacific\s+Northwest\s+National\s+Laboratory)'
)
FEDERAL_ACTION_INTRO_PATTERN = r'(?:proposes?\s+to|is\s+proposing\s+to|will|would|would\s+be\s+to|is\s+to)'
FEDERAL_ACTION_DIRECT_VERB_PATTERN = (
    r'(?:construct|install|build|operate|implement|manage|restore|undertake|develop|upgrade|'
    r'expand|demolish|replace|retrofit|rebuild|reconductor|renovate|refurbish|relocate|repair|'
    r'reconfigure|dismantle|modernize|improve)'
)
FEDERAL_ACTION_STEP_OWNERSHIP_PATTERN = (
    r'\bnow\s+that\s+DOE\s+has\s+acquired\s+ownership\s+of\s+the\s+parcel,\s+DOE\s+proposes\s+to\s+'
    r'operate\s+and\s+maintain\s+the\s+site\b'
)

FEDERAL_ACTION_VERB_PATTERNS = [
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,80}}\b{FEDERAL_ACTION_INTRO_PATTERN}\s+{FEDERAL_ACTION_DIRECT_VERB_PATTERN}\b',
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,80}}\b{FEDERAL_ACTION_INTRO_PATTERN}\s+remove\s+and\s+replace\b',
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,120}}\bconstruct,\s*own,\s*operate,\s*and\s+maintain\b',
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\b(?:constructed\s+and\s+operated|would\s+be\s+constructed\s+and\s+operated)\b',
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\bcontinue\s+to\s+occupy\s+and\s+maintain\s+existing\s+facilities\b[\s\S]{{0,180}}\brefurbish\s+existing\s+facilities\b',
    FEDERAL_ACTION_STEP_OWNERSHIP_PATTERN,
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\bwould\s+functionally\s+replace\b',
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,520}}\brebuild\s+the\s+existing\b',
    rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,200}}\b(?:upgrade|rebuild)\b[\s\S]{{0,160}}\bby\s+removing\b[\s\S]{{0,160}}\band\s+installing\b',
    r'\bfederal\s+(?:construction|facility|installation)\b',
    r'\bmilitary\s+(?:installation|base|facility|construction)\b',
]

FEDERAL_LAND_AUTHORIZER_PATTERNS = [
    r'\bapplication\s+for\s+a\s+right[-\s]of[-\s]way\s+grant\b',
    r'\b30-year\s+right[-\s]of[-\s]way\s+grant\b[\s\S]{0,120}\bBLM-?\s*administered\s+lands\b',
    r'\bright[-\s]of[-\s]way\s+renewal\s+applications?\b',
    r'\bright[-\s]of[-\s]way\s+renewal\s+and\s+amendment\b',
    r'\bspecial\s+use\s+permit\b',
    r'\bcurrent\s+authorization\s+with\s+a\s+defined\s+ROW\b[\s\S]{0,80}\bOperation\s+(?:and|&)\s+Maintenance\s+Plan\b',
    r'\btemporary\s+and\s+permanent\s+easements?\b',
    r'\beasement\s+has\s+expired\b',
    r'\beasement\s+for\s+the\s+right[-\s]of[-\s]way\b',
    r'\bperpetual\s+right[-\s]of[-\s]way\s+grant\b',
    r'\bgrant\s+a\s+perpetual\s+ROW\s+on\s+BLM\s+managed\s+public\s+land\b',
    r'\bTitle\s+V\s+of\s+the\s+Federal\s+Land\s+Policy\s+and\s+Management\s+Act\b[\s\S]{0,180}\brespond\s+to\s+requests\s+for\s+rights?-of-way\s+across\s+public\s+lands\b',
    r'\brights?-of-way\s+over,?\s+upon,?\s+under,?\s+or\s+through\s+public\s+lands\b',
    r'\bright[-\s]of[-\s]way\s+\(ROW\)\b[\s\S]{0,120}\bpublic\s+land\s+administered\s+by\s+the\s+Bureau\s+of\s+Land\s+Management\b',
    r'\brequest\s+for\s+a\s+right[-\s]of[-\s]way\b[\s\S]{0,120}\bpublic\s+land\s+managed\s+by\s+BLM\b',
    r'\b(?:The\s+)?Bureau\s+of\s+Indian\s+Affairs\s+is\s+requesting\s+a\s+new\s+right[-\s]of[-\s]way\s*\(ROW\)\b',
    r'\bRequest\s+to\s+Amend\s+Existing\s+Authorization\b',
    r'\bamend\s+its\s+ROW\s+grant\b',
    r'\blands?\s+administered\s+by\s+the\s+Bureau\s+of\s+Reclamation\b[\s\S]{0,120}\bpermissions?\s+must\s+be\s+sought\b',
    r'\b2920\s+Land\s+Use\s+Authorization\b',
]

# --- federal_program detection ---

PROGRAMMATIC_TITLE_PATTERNS = [
    r'\bprogrammatic\s+environmental\s+(?:impact\s+statement|assessment)\b',
    r'\bprogrammatic\s+(?:eis|ea)\b',
    r'\b(?:dpeis|fpeis|speis|peis|pea)\b',
]
PROGRAMMATIC_STRONG_PATTERNS = [
    r'(?:draft|final|supplemental)\s+programmatic\s+environmental\s+(?:impact\s+statement|assessment)',
    r'programmatic\s+environmental\s+(?:impact\s+statement|assessment)',
    r'\bprogrammatic\s+(?:eis|ea)\b',
    r'\b(?:dpeis|fpeis|speis|peis|pea)\b',
    r'this\s+programmatic\s+(?:eis|ea|environmental)',
    r'this\s+(?:peis|pea)\s+(?:analyzes|addresses|evaluates)',
    r'\btier\s*(?:1|i|one)\s+(?:nepa\s+)?(?:review|environmental\s+(?:impact\s+statement|assessment)|eis|ea)\b',
    r'\b(?:environmental\s+(?:impact\s+statement|assessment)|eis|ea)\s+tier\s*(?:1|i|one)\b',
    r'\bsite[-\s]?wide\s+environmental\s+(?:impact\s+statement|assessment)\b',
    r'\b(?:sweis|swea)\b',
    r'\b(?:\d{4}\s+)?integrated\s+resource\s+plan\b',
    r'\blong-term\s+experimental\s+and\s+management\s+plan\b[\s\S]{0,160}\benvironmental\s+impact\s+statement\b',
]

# Patterns for federal_land programmatic reviews (land-management PEISs/PEAs).
# Used in tier1a to detect PMA/TVA sponsor projects that are land-management programs.
FEDERAL_LAND_PROGRAM_PATTERNS = [
    r'\brevision\s+of\s+the\b[\s\S]{0,160}\bland\s+and\s+resource\s+management\s+plan\b',
    r'\b(?:final|proposed)\b[\s\S]{0,120}\bland\s+and\s+resource\s+management\s+plan\b',
    r'\bintegrated\s+vegetation\s+management\s+program\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+assessment\b',
    r'\bsystem-wide\s+operations\s+and\s+maintenance\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+assessment\b',
    r'\buranium\s+leasing\s+program\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+(?:assessment|impact\s+statement)\b',
    r'\bouter\s+continental\s+shelf\s+(?:oil\s+and\s+gas\s+)?leasing\s+program\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b',
    r'\bsolar\s+energy\s+development\s+in\s+six\s+southwestern\s+states\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b',
    r'\bwind\s+energy\s+development\s+on\s+bureau\s+of\s+land\s+management-administered\s+lands\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b',
    r'\b(?:updates?\s+to\s+the\s+western\s+solar\s+plan|2023\s+draft\s+solar\s+peis)\b[\s\S]{0,160}\b(?:solar\s+peis|programmatic\s+environmental\s+impact\s+statement)\b|\b2023\s+draft\s+solar\s+peis\b',
    r'\bdesignation\s+of\s+energy\s+corridors\s+on\s+federal\s+land\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b',
    r'\bsection\s+368\s+energy\s+corridor\s+revisions\b[\s\S]{0,160}\b(?:resource\s+management\s+plan\s+amendment|environmental\s+impact\s+statement)\b',
]
PROGRAMMATIC_EXCLUSION_PATTERNS = [
    r'programmatic\s+agreement',
    r'programmatic\s+biological\s+opinion',
    r'programmatic\s+consultation',
    r'programmatic\s+collaboration',
    r'cultural\s+resource\s+management\s+plan',
]

PROPERTY_TRANSACTION_EXCLUSION_PATTERNS = [
    r'acquired\s+the\s+property\s+as\s+part\s+of[\s\S]{0,100}land\s+exchange',
    r'completed\s+a\s+NEPA\s+review\s+of\s+the\s+land\s+exchange',
    r'no\s+transfer\s+of\s+land\s+ownership',
    r'only\s+change\s+would\s+be\s+in\s+ownership\s+of\s+assets',
    r'land\s+exchanges?,\s+withdrawals?,\s+and\s+the\s+implementation\s+of\s+RMP',
    r'disposals?\s+of\s+land\s+parcels',
    r'land\s+exchanges?\s+could[\s\S]{0,120}(?:lower|allow|play\s+a\s+role|be\s+considered)',
    r'land\s+exchanges?\s+are\s+considered\s+on\s+a\s+case-by-case\s+basis',
]

# --- Tier 1b: Title / description keyword patterns ---
# Format: (pattern, trigger_class, rule_slug, confidence)
# More specific / distinctive patterns appear first within each class.

TIER1B_PATTERNS = [
    # federal_program — most distinctive; check before land/permit patterns
    (r'programmatic\s+environmental\s+impact\s+statement', 'federal_program', 'peis', 'high'),
    (r'programmatic\s+environmental\s+assessment', 'federal_program', 'pea', 'high'),
    # Generic nuclear EIS/EA must come BEFORE generic_review — nuclear context → federal_permit, not federal_program
    (r'generic\s+(?:environmental\s+impact\s+statement|eis|geis)\b[\s\S]{0,200}\b(?:nuclear|NRC|reactor|license\s+renewal)\b', 'federal_permit', 'generic_nuclear_eis', 'high'),
    (r'generic\s+(?:environmental\s+assessment|ea|gea)\b[\s\S]{0,200}\b(?:nuclear|NRC|reactor|license\s+renewal)\b', 'federal_permit', 'generic_nuclear_ea', 'high'),
    (r'\bNUREG-1437\b', 'federal_permit', 'nureg1437_geis', 'high'),
    (r'generic\s+(?:environmental\s+(?:impact\s+statement|assessment)|eis|ea)\b', 'federal_program', 'generic_review', 'high'),
    (r'tier\s*(?:1|i|one)\s+(?:nepa\s+)?(?:review|environmental\s+(?:impact\s+statement|assessment)|eis|ea)\b', 'federal_program', 'tier1_review', 'high'),
    (r'(?:environmental\s+(?:impact\s+statement|assessment)|eis|ea)\s+tier\s*(?:1|i|one)\b', 'federal_program', 'tier1_review_rev', 'high'),
    (r'site[-\s]?wide\s+environmental\s+(?:impact\s+statement|assessment)\b', 'federal_program', 'sitewide_review', 'high'),
    (r'\b(?:SWEIS|SWEA)\b', 'federal_program', 'sitewide_acronym', 'high'),
    (r'(?:\d{4}\s+)?integrated\s+resource\s+plan\b[\s\S]{0,160}\b(?:programmatic\s+environmental\s+impact\s+statement|supplemental\s+environmental\s+impact\s+statement|draft\s+eis)\b', 'federal_program', 'integrated_resource_plan', 'high'),
    (r'long-term\s+experimental\s+and\s+management\s+plan\b[\s\S]{0,160}\benvironmental\s+impact\s+statement\b', 'federal_program', 'ltemp_eis', 'high'),
    # federal_land — land-management and leasing program PEISs/PEAs (moved from federal_program)
    (r'revision\s+of\s+the\b[\s\S]{0,160}\bland\s+and\s+resource\s+management\s+plan\b', 'federal_land', 'lrm_plan_revision', 'high'),
    (r'(?:final|proposed)\b[\s\S]{0,120}\bland\s+and\s+resource\s+management\s+plan\b', 'federal_land', 'lrm_plan_title', 'medium'),
    (r'integrated\s+vegetation\s+management\s+program\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+assessment\b', 'federal_land', 'ivm_program_pea', 'high'),
    (r'system-wide\s+operations\s+and\s+maintenance\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+assessment\b', 'federal_land', 'systemwide_om_pea', 'high'),
    (r'uranium\s+leasing\s+program\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+(?:assessment|impact\s+statement)\b', 'federal_land', 'uranium_leasing_pea', 'high'),
    (r'outer\s+continental\s+shelf\s+(?:oil\s+and\s+gas\s+)?leasing\s+program\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b', 'federal_land', 'ocs_leasing_peis', 'high'),
    (r'solar\s+energy\s+development\s+in\s+six\s+southwestern\s+states\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b', 'federal_land', 'solar_program_peis', 'high'),
    (r'wind\s+energy\s+development\s+on\s+bureau\s+of\s+land\s+management-administered\s+lands\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b', 'federal_land', 'wind_program_peis', 'high'),
    (r'(?:updates?\s+to\s+the\s+western\s+solar\s+plan\b[\s\S]{0,160}\b(?:solar\s+peis|programmatic\s+environmental\s+impact\s+statement)\b)|(?:2023\s+draft\s+solar\s+peis\b)', 'federal_land', 'western_solar_peis', 'high'),
    (r'designation\s+of\s+energy\s+corridors\s+on\s+federal\s+land\b[\s\S]{0,160}\bprogrammatic\s+environmental\s+impact\s+statement\b', 'federal_land', 'energy_corridors_peis', 'high'),
    (r'section\s+368\s+energy\s+corridor\s+revisions\b[\s\S]{0,160}\b(?:resource\s+management\s+plan\s+amendment|environmental\s+impact\s+statement)\b', 'federal_land', 'section368_corridor', 'high'),
    # federal_property_transaction
    (r'land\s+exchange\b', 'federal_property_transaction', 'land_exchange', 'high'),
    (r'fee[-\s]for[-\s]fee\s+land\s+exchange\b', 'federal_property_transaction', 'fee_land_exchange', 'high'),
    (r'exchange\s+property\s+with\b', 'federal_property_transaction', 'exchange_property', 'high'),
    (r'dispose\s+of\s+(?:the\s+)?(?:underlying\s+)?land\s+rights\b', 'federal_property_transaction', 'dispose_land_rights', 'high'),
    (r'land\s+disposal\b', 'federal_property_transaction', 'land_disposal', 'high'),
    (r'sale\s+of\s+land\s+rights\b', 'federal_property_transaction', 'sale_land_rights', 'high'),
    (r'sell\s+in\s+fee\b', 'federal_property_transaction', 'sell_in_fee', 'high'),
    (r'acquire\s+and\s+release\s+access\s+road\s+rights\b', 'federal_property_transaction', 'access_rights_exchange', 'high'),
    (r'acquire\s+several\s+road\s+easements\b', 'federal_property_transaction', 'road_easement_acquisition', 'high'),
    (r'acquire\s+land\s+rights\b', 'federal_property_transaction', 'land_rights_acquisition', 'high'),
    (r'acquire\s+access\s+road\s+rights\b', 'federal_property_transaction', 'access_rights_acquisition', 'high'),
    (r'purchase\s+(?:two\s+lots?|lots?)\s+and\s+(?:line\s+)?easements\b', 'federal_property_transaction', 'land_purchase_easement', 'high'),
    (r'transfer\s+ownership\b[\s\S]{0,160}\b(?:associated\s+easements?|easements?|rights?-of-way|land\s+rights)\b', 'federal_property_transaction', 'transfer_easements', 'high'),
    (r'title\s+transfer\b[\s\S]{0,160}\b(?:easements?|rights?-of-way|land\s+rights)\b', 'federal_property_transaction', 'title_transfer', 'high'),
    (r'asset\s+exchange\b[\s\S]{0,160}\b(?:rights?-of-way|line\s+easements?|easements?)\b', 'federal_property_transaction', 'asset_exchange_easements', 'medium'),
    (r'(?:disposal|conveyance)\s+of\s+federal\s+(?:land|property)', 'federal_property_transaction', 'disposal', 'high'),
    (r'parcel\s+transfer\b', 'federal_property_transaction', 'parcel_xfer', 'medium'),
    # federal_permit
    (r'Department\s+of\s+the\s+Army\s+Environmental\s+Assessment\s+and\s+Statement\s+of\s+Find(?:ing|ings)\b[\s\S]{0,120}\b(?:Standard\s+)?Individual\s+Permit\s+Application\b', 'federal_permit', 'permit_app', 'high'),
    (r'Section\s+404\s+permit\s+application\b', 'federal_permit', 'sec404', 'high'),
    (r'applied\s+for\s+an\s+individual\s+permit\s+under\s+Section\s+404\b', 'federal_permit', 'sec404_individual', 'high'),
    (r'Department\s+of\s+(?:the\s+)?Army(?:\s+\(DA\))?\s+permit\s+pursuant\s+to\s+Section\s+404\b[\s\S]{0,180}\bSection\s+10\s+of\s+the\s+Rivers\s+and\s+Harbors', 'federal_permit', 'sec404_da', 'high'),
    (CLEAN_WATER_ACT_PERMIT_CONTEXT_PATTERN, 'federal_permit', 'clean_water_act_permit_context', 'medium'),
    (r'Section\s+10\b[\s\S]{0,80}Rivers\s+and\s+Harbors', 'federal_permit', 'sec10_rha', 'high'),
    (r'Nationwide\s+Permit\s+\(NWP\)\s+Verification\b', 'federal_permit', 'nwp_verification', 'medium'),
    (r'Nationwide,\s+Regional\s+General,\s+or\s+Standard\s+Individual\s+Permit\s+may\s+be\s+required\b', 'federal_permit', 'nwp_general_individual', 'medium'),
    (r'issuance\s+of\s+a\s+National\s+Pollutant\s+Discharge\s+Elimination\s+System\s+permit\b', 'federal_permit', 'npdes', 'high'),
    (r'NPDES\s+permit\s+must\s+be\s+obtained\b', 'federal_permit', 'npdes', 'high'),
    (r'National\s+Pollutant\s+Discharge\s+Elimination\s+System\s*\(NPDES\)\s+Construction\s+Storm\s+Water\s+General\s+Permit\s+is\s+required\b', 'federal_permit', 'npdes_general', 'high'),
    (r'incidental\s+take\s+permit\s+application\b', 'federal_permit', 'itp', 'high'),
    (r'Incidental\s+Take\s+Permit\s*\(ITP\)\s+under\s+Section\s+10\(a\)\(1\)\(B\)', 'federal_permit', 'itp_section10', 'high'),
    (r'Renewed/Amended\s+ITP\s+is\s+needed\b', 'federal_permit', 'itp_renewal', 'high'),
    (r'Habitat\s+Conservation\s+Plan\s+and\s+Incidental\s+Take\s+Permit\b', 'federal_permit', 'itp_hcp', 'high'),
    (r'\bhydropower\s+license\b', 'federal_permit', 'ferc_license', 'high'),
    (r'\brelicens(?:e|ing)\b', 'federal_permit', 'relicensing', 'medium'),
    (r'application\s+for\s+a\s+certificate\s+of\s+public\s+convenience\s+and\s+necessity\b', 'federal_permit', 'ferc_certificate', 'high'),
    (r'Amendment\s+to\s+Presidential\s+Permit\b', 'federal_permit', 'presidential_permit_amend', 'high'),
    (r'Issuance\s+of\s+Presidential\s+Permit\s+PP-\d+\b', 'federal_permit', 'presidential_permit_issue', 'high'),
    (r'Presidential\s+Permit\s+Application\s+Review\b', 'federal_permit', 'presidential_permit_review', 'high'),
    # "granting or denying a Presidential Permit" — DOE cross-border transmission EA framing
    (r'granting\s+or\s+denying\s+a\s+Presidential\s+Permit\b', 'federal_permit', 'presidential_permit_decision', 'high'),
    # DOE Section 202(e) electricity export authorizations under the Federal Power Act
    (r'electricity\s+export\s+authorization\b', 'federal_permit', 'doe_export_auth', 'high'),
    (r'export\s+electricity\b[\s\S]{0,100}\bSection\s+202\(e\)', 'federal_permit', 'doe_export_auth', 'high'),
    # DOE Natural Gas Act / LNG export authorizations (same DOE permit structure as electricity)
    (r'(?:LNG|liquefaction|natural\s+gas)\s+export\s+authorization\b', 'federal_permit', 'doe_gas_export_auth', 'high'),
    # PMA full name in project description — catches NEPATEC records listing DOE instead of BPA/WAPA/TVA
    (r'\b(?:Western\s+Area\s+Power\s+Administration|Bonneville\s+Power\s+Administration|Southwestern\s+Power\s+Administration|Southeastern\s+Power\s+Administration|Tennessee\s+Valley\s+Authority)\b', 'pma', 'pma_name_in_description', 'high'),
    (r'\bEarly\s+Site\s+Permit\b', 'federal_permit', 'early_site_permit', 'high'),
    (r'\bCombined\s+License\b', 'federal_permit', 'combined_license', 'high'),
    (r'\b(?:Subsequent\s+)?License\s+Renewal\b', 'federal_permit', 'license_renewal', 'high'),
    (r'issuance\s+of\s+renewed\s+facility\s+operating\s+licenses\b', 'federal_permit', 'renewed_operating_license', 'high'),
    # federal_land — ROW and access language
    (r'application\s+for\s+a\s+right[-\s]of[-\s]way\s+grant\b', 'federal_land', 'row_grant_application', 'high'),
    (r'30-year\s+right[-\s]of[-\s]way\s+grant\b[\s\S]{0,120}BLM-?\s*administered\s+lands', 'federal_land', 'row_grant_blm', 'high'),
    (r'right[-\s]of[-\s]way\s+renewal\s+applications?\b', 'federal_land', 'row_renewal_apps', 'high'),
    (r'right[-\s]of[-\s]way\s+renewal\s+and\s+amendment\b', 'federal_land', 'row_renewal_amend', 'high'),
    (r'special\s+use\s+permit\b', 'federal_land', 'special_use', 'high'),
    (r'current\s+authorization\s+with\s+a\s+defined\s+ROW\b[\s\S]{0,120}Operation\s+(?:and|&)\s+Maintenance\s+Plan', 'federal_land', 'defined_row_omp', 'high'),
    (r'temporary\s+and\s+permanent\s+easements?\b', 'federal_land', 'temp_perm_easement', 'high'),
    (r'easement\s+has\s+expired\b', 'federal_land', 'easement_expired', 'high'),
    (r'easement\s+for\s+the\s+right[-\s]of[-\s]way\b', 'federal_land', 'easement_row', 'high'),
    (r'perpetual\s+right[-\s]of[-\s]way\s+grant\b', 'federal_land', 'perpetual_row_grant', 'high'),
    (r'grant\s+a\s+perpetual\s+ROW\s+on\s+BLM\s+managed\s+public\s+land\b', 'federal_land', 'perpetual_row_blm', 'high'),
    (r'Title\s+V\s+of\s+the\s+Federal\s+Land\s+Policy\s+and\s+Management\s+Act\b[\s\S]{0,180}respond\s+to\s+requests\s+for\s+rights?-of-way\s+across\s+public\s+lands', 'federal_land', 'flpma_public_land_row', 'high'),
    (r'rights?-of-way\s+over,?\s+upon,?\s+under,?\s+or\s+through\s+public\s+lands', 'federal_land', 'public_land_row', 'high'),
    (r'right[-\s]of[-\s]way\s+\(ROW\)\b[\s\S]{0,120}public\s+land\s+administered\s+by\s+the\s+Bureau\s+of\s+Land\s+Management', 'federal_land', 'public_land_admin_blm', 'high'),
    (r'request\s+for\s+a\s+right[-\s]of[-\s]way\b[\s\S]{0,120}public\s+land\s+managed\s+by\s+BLM', 'federal_land', 'public_land_managed_blm', 'high'),
    (r'(?:The\s+)?Bureau\s+of\s+Indian\s+Affairs\s+is\s+requesting\s+a\s+new\s+right[-\s]of[-\s]way\s*\(ROW\)', 'federal_land', 'bia_row', 'high'),
    (r'lands?\s+administered\s+by\s+the\s+Bureau\s+of\s+Reclamation\b[\s\S]{0,120}permissions?\s+must\s+be\s+sought', 'federal_land', 'bor_permission', 'high'),
    (r'\b2920\s+Land\s+Use\s+Authorization\b', 'federal_land', 'land_use_2920', 'high'),
    (r'Request\s+to\s+Amend\s+Existing\s+Authorization', 'federal_land', 'auth_amend_title', 'medium'),
    (r'amend\s+its\s+ROW\s+grant\b', 'federal_land', 'auth_amend_row', 'medium'),
    # federal_funding
    (r'\bTitle\s+XVII\b', 'federal_funding', 'title17', 'high'),
    (r'Inflation\s+Reduction\s+Act\b', 'federal_funding', 'ira', 'high'),
    (r'Bipartisan\s+Infrastructure\s+(?:Law|Act)\b', 'federal_funding', 'bil', 'high'),
    (r'loan\s+guarantee\b', 'federal_funding', 'loan_guarantee', 'high'),
    (r'through\s+(?:a\s+)?cooperative\s+agreement[\s\S]{0,120}\bpartially\s+fund\b', 'federal_funding', 'coop_partial_fund', 'high'),
    (r'providing\s+financial\s+assistance\s+to[\s\S]{0,120}\b(?:under|through)\s+(?:a\s+)?cooperative\s+agreement\b', 'federal_funding', 'fin_assist_coop', 'high'),
    (r'\bawarding\s+a\s+grant\b[\s\S]{0,120}\bpartially\s+fund\b', 'federal_funding', 'grant_partial_fund', 'high'),
    (r'DOE\s+Funding\s*=\s*\$?\d[\s\S]{0,120}Cost\s+Share\s*=\s*\$?\d', 'federal_funding', 'doe_cost_share', 'high'),
    (r'provide\s+federal\s+funding\b', 'federal_funding', 'provide_fed_funding', 'high'),
    (r'DOE.?s\s+(?:proposed\s+)?action\s+is\s+to\s+provide[\s\S]{0,160}\bcost[-\s]shared\s+arrangement\b', 'federal_funding', 'cost_shared_arrangement', 'high'),
    (r'Federal\s+Cost\s+Share\b[\s\S]{0,120}Total\s+Project\s+Value\b', 'federal_funding', 'federal_cost_share', 'high'),
    (r'\b(?:DOE\s+)?EECBG\s+funding\b', 'federal_funding', 'eecbg_funding', 'high'),
    (r'\bformula(?:-based)?\s+(?:awards?|grants?)\b', 'federal_funding', 'formula_awards', 'high'),
    (r'\b(?:State\s+Energy\s+Program|SEP|WAP)\b[\s\S]{0,120}\bformula(?:-based)?\s+(?:awards?|grants?)\b', 'federal_funding', 'program_formula_awards', 'high'),
    (r'(?:Administrative\s+(?:and\s+)?Legal\s+Requirements\s+Document|\bALRD\b)[\s\S]{0,160}\bformula(?:-based)?\s+(?:awards?|grants?)\b', 'federal_funding', 'alrd_formula', 'medium'),
    (r'(?:DOE|DOT|HUD|USDA)\s+(?:grant|funding)\b', 'federal_funding', 'agency_grant', 'high'),
    (r'federal\s+(?:financial\s+assistance|grant\b)', 'federal_funding', 'fed_grant', 'medium'),
    # federal_land — vegetation management on National Forest land (moved from federal_direct_action)
    (r'vegetation\s+management\b.{0,50}National\s+Forest', 'federal_land', 'usfs_veg_mgmt', 'high'),
    # pma — Power Marketing Administration + Tennessee Valley Authority
    (r'\b(?:Bonneville\s+Power\s+Administration|BPA)\b[\s\S]{0,80}\b(?:proposes?\s+to|will|would)\b', 'pma', 'bpa_actor', 'high'),
    (r'\b(?:Western\s+Area\s+Power\s+Administration|WAPA)\b[\s\S]{0,80}\b(?:proposes?\s+to|will|would)\b', 'pma', 'wapa_actor', 'high'),
    (r'\b(?:Southeastern\s+Power\s+Administration|SEPA)\b[\s\S]{0,80}\b(?:proposes?\s+to|will|would)\b', 'pma', 'sepa_actor', 'high'),
    (r'\b(?:Southwestern\s+Power\s+Administration|SWPA)\b[\s\S]{0,80}\b(?:proposes?\s+to|will|would)\b', 'pma', 'swpa_actor', 'high'),
    (r'\b(?:Tennessee\s+Valley\s+Authority|TVA)\b[\s\S]{0,80}\b(?:proposes?\s+to|will|would)\b', 'pma', 'tva_actor', 'high'),
    (r'\bPower\s+Marketing\s+Administration\b', 'pma', 'pma_generic', 'medium'),
    # federal_direct_action — agency as actor (checked last among high-priority classes)
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,80}}\b{FEDERAL_ACTION_INTRO_PATTERN}\s+{FEDERAL_ACTION_DIRECT_VERB_PATTERN}\b', 'federal_direct_action', 'agency_actor_direct', 'high'),
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,80}}\b{FEDERAL_ACTION_INTRO_PATTERN}\s+remove\s+and\s+replace\b', 'federal_direct_action', 'agency_remove_replace', 'high'),
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,120}}\bconstruct,\s*own,\s*operate,\s*and\s+maintain\b', 'federal_direct_action', 'construct_own_operate_maintain', 'high'),
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\b(?:constructed\s+and\s+operated|would\s+be\s+constructed\s+and\s+operated)\b', 'federal_direct_action', 'constructed_operated', 'high'),
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\bcontinue\s+to\s+occupy\s+and\s+maintain\s+existing\s+facilities\b[\s\S]{{0,180}}\brefurbish\s+existing\s+facilities\b', 'federal_direct_action', 'occupy_maintain_refurbish', 'high'),
    (FEDERAL_ACTION_STEP_OWNERSHIP_PATTERN, 'federal_direct_action', 'ownership_transition_site_operation', 'high'),
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\bwould\s+functionally\s+replace\b', 'federal_direct_action', 'functional_replace', 'high'),
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,520}}\brebuild\s+the\s+existing\b', 'federal_direct_action', 'rebuild_existing_facility', 'high'),
    (rf'\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,200}}\b(?:upgrade|rebuild)\b[\s\S]{{0,160}}\bby\s+removing\b[\s\S]{{0,160}}\band\s+installing\b', 'federal_direct_action', 'upgrade_remove_install', 'high'),
    (r'military\s+(?:installation|base|facility)\b', 'federal_direct_action', 'military', 'high'),
    (r'federal\s+facility\s+(?:upgrade|expansion|construction)\b', 'federal_direct_action', 'fed_facility', 'high'),
]

# --- Tier 2: Document title patterns ---
# Format: (pattern, trigger_class, rule_slug)
# Programmatic detection uses PROGRAMMATIC_TITLE_PATTERNS + exclusion check (handled separately).

DOC_TITLE_PATTERNS = [
    # Nuclear generic EIS/EA → federal_permit; must precede generic_review
    (r'generic\s+(?:environmental\s+impact\s+statement|eis|geis)\b[\s\S]{0,200}\b(?:nuclear|NRC|reactor|license\s+renewal)\b', 'federal_permit', 'generic_nuclear_eis'),
    (r'generic\s+(?:environmental\s+assessment|ea|gea)\b[\s\S]{0,200}\b(?:nuclear|NRC|reactor|license\s+renewal)\b', 'federal_permit', 'generic_nuclear_ea'),
    (r'\bNUREG-1437\b', 'federal_permit', 'nureg1437_geis'),
    (r'generic\s+(?:environmental\s+(?:impact\s+statement|assessment)|eis|ea)\b', 'federal_program', 'generic_review'),
    (r'tier\s*(?:1|i|one)\s+(?:nepa\s+)?(?:review|environmental\s+(?:impact\s+statement|assessment)|eis|ea)\b', 'federal_program', 'tier1_review'),
    (r'(?:environmental\s+(?:impact\s+statement|assessment)|eis|ea)\s+tier\s*(?:1|i|one)\b', 'federal_program', 'tier1_review_rev'),
    (r'site[-\s]?wide\s+environmental\s+(?:impact\s+statement|assessment)\b', 'federal_program', 'sitewide_review'),
    (r'\b(?:SWEIS|SWEA)\b', 'federal_program', 'sitewide_acronym'),
    (r'(?:\d{4}\s+)?integrated\s+resource\s+plan\b', 'federal_program', 'integrated_resource_plan'),
    (r'revision\s+of\s+the\b[\s\S]{0,160}\bland\s+and\s+resource\s+management\s+plan\b', 'federal_land', 'lrm_plan_revision'),
    (r'(?:final|proposed)\b[\s\S]{0,120}\bland\s+and\s+resource\s+management\s+plan\b', 'federal_land', 'lrm_plan_title'),
    (r'land\s+exchange', 'federal_property_transaction', 'land_exchange'),
    (r'land\s+disposal\b', 'federal_property_transaction', 'land_disposal'),
    (r'sale\s+of\s+land\s+rights\b', 'federal_property_transaction', 'sale_land_rights'),
    (r'land\s+purchase\s+and\s+easement\s+acquisition\b', 'federal_property_transaction', 'land_purchase_easement'),
    (r'land\s+rights\s+acquisition\b', 'federal_property_transaction', 'land_rights_acquisition'),
    (r'easement\s+exchange\b', 'federal_property_transaction', 'easement_exchange'),
    (r'title\s+transfer\b', 'federal_property_transaction', 'title_transfer'),
    (r'(?:transmission\s+line|substation).{0,40}property\s+transfer\b', 'federal_property_transaction', 'property_transfer'),
    (r'right.of.way\b', 'federal_land', 'row'),
    (r'(?:Standard\s+)?Individual\s+Permit\s+Application\b', 'federal_permit', 'permit_app'),
    (r'Hydropower\s+License\b', 'federal_permit', 'ferc_license'),
    (r'Incidental\s+Take\s+Permit\b', 'federal_permit', 'itp'),
    (r'Presidential\s+Permit\b', 'federal_permit', 'presidential_permit'),
    (r'Early\s+Site\s+Permit\b', 'federal_permit', 'early_site_permit'),
    (r'Combined\s+License\b', 'federal_permit', 'combined_license'),
    (r'(?:Subsequent\s+)?License\s+Renewal\b', 'federal_permit', 'license_renewal'),
    (r'certificate\s+of\s+public\s+convenience\s+and\s+necessity\b', 'federal_permit', 'ferc_certificate'),
    (r'loan\s+guarantee\b', 'federal_funding', 'loan_guarantee'),
]

# Tier 3 splits CE from EA/EIS so CE text is handled more conservatively.
TIER3_PATTERNS_EA_EIS = TIER1B_PATTERNS
TIER3_PATTERNS_CE = [
    pattern for pattern in TIER1B_PATTERNS
    if pattern[2] not in {"sec404", "rmp"}
]

TIER4_CUE_PATTERNS = {
    "federal_funding": [
        r"\b(?:federal\s+grant|DOE\s+grant|DOT\s+grant|HUD\s+grant|USDA\s+grant|grant\s+funding)\b",
        r"\bloan\s+guarantee\b",
        r"\bthrough\s+(?:a\s+)?cooperative\s+agreement\b[\s\S]{0,120}\bpartially\s+fund\b",
        r"\bproviding\s+financial\s+assistance\s+to\b[\s\S]{0,120}\b(?:under|through)\s+(?:a\s+)?cooperative\s+agreement\b",
        r"\bawarding\s+a\s+grant\b[\s\S]{0,120}\bpartially\s+fund\b",
        r"\bfederal\s+(?:funding|financial\s+assistance)\b",
        r"\bcost\s+share\b",
        r"\bDOE\s+Funding\b",
        r"\b(?:DOE|Department\s+of\s+Energy)\b[\s\S]{0,80}\bwould\s+provide\b[\s\S]{0,120}\b(?:funds?|funding|grant|awards?|cost[-\s]share)\b",
        r"\bcost[-\s]shared\s+arrangement\b",
        r"\bFederal\s+Cost\s+Share\b",
        r"\b(?:DOE\s+)?EECBG\s+funding\b",
        r"\btotal\s+award\s+value\b",
        r"\bTotal\s+Project\s+Value\b",
        r"\bformula(?:-based)?\s+(?:awards?|grants?)\b",
        r"\b(?:State\s+Energy\s+Program|SEP|WAP)\b[\s\S]{0,120}\bformula(?:-based)?\s+(?:awards?|grants?)\b",
        r"(?:Administrative\s+(?:and\s+)?Legal\s+Requirements\s+Document|\bALRD\b)[\s\S]{0,160}\bformula(?:-based)?\s+(?:awards?|grants?)\b",
    ],
    "federal_direct_action": [
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,80}}\b{FEDERAL_ACTION_INTRO_PATTERN}\s+{FEDERAL_ACTION_DIRECT_VERB_PATTERN}\b",
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,80}}\b{FEDERAL_ACTION_INTRO_PATTERN}\s+remove\s+and\s+replace\b",
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,120}}\bconstruct,\s*own,\s*operate,\s*and\s+maintain\b",
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\b(?:constructed\s+and\s+operated|would\s+be\s+constructed\s+and\s+operated)\b",
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\bcontinue\s+to\s+occupy\s+and\s+maintain\s+existing\s+facilities\b[\s\S]{{0,180}}\brefurbish\s+existing\s+facilities\b",
        FEDERAL_ACTION_STEP_OWNERSHIP_PATTERN,
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,160}}\bwould\s+functionally\s+replace\b",
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,520}}\brebuild\s+the\s+existing\b",
        rf"\b{FEDERAL_ACTION_ACTOR_PATTERN}\b[\s\S]{{0,200}}\b(?:upgrade|rebuild)\b[\s\S]{{0,160}}\bby\s+removing\b[\s\S]{{0,160}}\band\s+installing\b",
        r"\bfederal\s+facility\b",
        r"\bmilitary\s+(?:installation|base|facility)\b",
    ],
    "federal_land": [
        r"\bapplication\s+for\s+a\s+right[-\s]of[-\s]way\s+grant\b",
        r"\bspecial\s+use\s+permit\b",
        r"\bcurrent\s+authorization\s+with\s+a\s+defined\s+ROW\b",
        r"\bOperation\s+(?:and|&)\s+Maintenance\s+Plan\b[\s\S]{0,120}\bROW\b|\bROW\b[\s\S]{0,120}\bOperation\s+(?:and|&)\s+Maintenance\s+Plan\b",
        r"\btemporary\s+and\s+permanent\s+easements?\b",
        r"\beasement\s+has\s+expired\b",
        r"\beasement\b[\s\S]{0,120}\bright[-\s]of[-\s]way\b",
        r"\bperpetual\s+right[-\s]of[-\s]way\s+grant\b",
        r"\bgrant\s+a\s+perpetual\s+ROW\s+on\s+BLM\s+managed\s+public\s+land\b",
        r"\b30-year\s+right[-\s]of[-\s]way\s+grant\b[\s\S]{0,120}\bBLM-?\s*administered\s+lands\b",
        r"\bTitle\s+V\s+of\s+the\s+Federal\s+Land\s+Policy\s+and\s+Management\s+Act\b[\s\S]{0,180}\brespond\s+to\s+requests\s+for\s+rights?-of-way\s+across\s+public\s+lands\b",
        r"\brights?-of-way\s+over,?\s+upon,?\s+under,?\s+or\s+through\s+public\s+lands\b",
        r"\bright[-\s]of[-\s]way\s+\(ROW\)\b[\s\S]{0,120}\bpublic\s+land\s+administered\s+by\s+the\s+Bureau\s+of\s+Land\s+Management\b",
        r"\brequest\s+for\s+a\s+right[-\s]of[-\s]way\b[\s\S]{0,120}\bpublic\s+land\s+managed\s+by\s+BLM\b",
        r"\bpublic\s+lands\s+managed\s+by\s+the\s+Bureau\s+of\s+Land\s+Management\b",
        r"\b(?:The\s+)?Bureau\s+of\s+Indian\s+Affairs\s+is\s+requesting\s+a\s+new\s+right[-\s]of[-\s]way\s*\(ROW\)\b",
        r"\blands?\s+administered\s+by\s+(?:the\s+)?Bureau\s+of\s+Reclamation\b[\s\S]{0,160}\bpermissions?\s+must\s+be\s+sought\b",
        r"\b2920\s+Land\s+Use\s+Authorization\b",
        r"\bRequest\s+to\s+Amend\s+Existing\s+Authorization\b",
        r"\bamend\s+its\s+ROW\s+grant\b",
        r"\bvegetation\s+management\b[\s\S]{0,60}\bNational\s+Forest\b",
        r"\bRevision\s+of\s+the\b[\s\S]{0,160}\bLand\s+and\s+Resource\s+Management\s+Plan\b",
        r"\b(?:Final|Proposed)\b[\s\S]{0,120}\bLand\s+and\s+Resource\s+Management\s+Plan\b",
        r"\bIntegrated\s+Vegetation\s+Management\s+Program\b[\s\S]{0,160}\bProgrammatic\s+Environmental\s+Assessment\b",
        r"\bSystem-wide\s+Operations\s+and\s+Maintenance\b[\s\S]{0,160}\bProgrammatic\s+Environmental\s+Assessment\b",
        r"\bUranium\s+Leasing\s+Program\b[\s\S]{0,160}\bProgrammatic\s+Environmental\s+(?:Assessment|Impact\s+Statement)\b",
        r"\bOuter\s+Continental\s+Shelf\s+(?:Oil\s+and\s+Gas\s+)?Leasing\s+Program\b[\s\S]{0,160}\bProgrammatic\s+Environmental\s+Impact\s+Statement\b",
        r"\bSolar\s+Energy\s+Development\s+in\s+Six\s+Southwestern\s+States\b[\s\S]{0,160}\bProgrammatic\s+Environmental\s+Impact\s+Statement\b",
        r"\bWind\s+Energy\s+Development\s+on\s+Bureau\s+of\s+Land\s+Management-Administered\s+Lands\b[\s\S]{0,160}\bProgrammatic\s+Environmental\s+Impact\s+Statement\b",
        r"\b(?:Updates?\s+to\s+the\s+Western\s+Solar\s+Plan|2023\s+Draft\s+Solar\s+PEIS)\b",
        r"\bDesignation\s+of\s+Energy\s+Corridors\s+on\s+Federal\s+Land\b[\s\S]{0,160}\bProgrammatic\s+Environmental\s+Impact\s+Statement\b",
        r"\bSection\s+368\s+Energy\s+Corridor\s+Revisions\b[\s\S]{0,160}\b(?:Resource\s+Management\s+Plan\s+Amendment|Environmental\s+Impact\s+Statement)\b",
    ],
    "federal_permit": [
        r"\b(?:Standard\s+)?Individual\s+Permit\s+Application\b",
        r"\bSection\s+404\s+permit\s+application\b",
        r"\bapplied\s+for\s+an\s+individual\s+permit\s+under\s+Section\s+404\b",
        r"\bDepartment\s+of\s+(?:the\s+)?Army(?:\s+\(DA\))?\s+permit\b[\s\S]{0,180}\bSection\s+404\b",
        CLEAN_WATER_ACT_PERMIT_CONTEXT_PATTERN,
        r"\bSection\s+10\b[\s\S]{0,80}\bRivers\s+and\s+Harbors\b",
        r"\bNationwide\s+Permit\s+\(NWP\)\s+Verification\b",
        r"\bNationwide,\s+Regional\s+General,\s+or\s+Standard\s+Individual\s+Permit\s+may\s+be\s+required\b",
        r"\bNational\s+Pollutant\s+Discharge\s+Elimination\s+System\s*\(NPDES\)\s+permit\b",
        r"\bNPDES\s+permit\s+must\s+be\s+obtained\b",
        r"\bNPDES\b[\s\S]{0,80}\bpermitting\s+decision\b",
        r"\bConstruction\s+Storm\s+Water\s+General\s+Permit\s+is\s+required\b",
        r"\bincidental\s+take\s+permit\s+application\b",
        r"\bIncidental\s+Take\s+Permit\s*\(ITP\)\s+under\s+Section\s+10\(a\)\(1\)\(B\)\b",
        r"\bRenewed/Amended\s+ITP\s+is\s+needed\b",
        r"\bHabitat\s+Conservation\s+Plan\s+and\s+Incidental\s+Take\s+Permit\b",
        r"\bhydropower\s+license\b",
        r"\brelicens(?:e|ing)\b",
        r"\bapplication\s+for\s+a\s+certificate\s+of\s+public\s+convenience\s+and\s+necessity\b",
        r"\b(?:Amendment\s+to\s+)?Presidential\s+Permit\b",
        r"\bIssuance\s+of\s+Presidential\s+Permit\s+PP-\d+\b",
        r"\bEarly\s+Site\s+Permit\b",
        r"\bCombined\s+License\b",
        r"\b(?:Subsequent\s+)?License\s+Renewal\b",
        r"\bissuance\s+of\s+renewed\s+facility\s+operating\s+licenses\b",
        r"\b(?:NRC|FERC)\b[\s\S]{0,80}\blicense\s+amendment\b|\blicense\s+amendment\b[\s\S]{0,80}(?:NRC|FERC|10\s+CFR\s+50\.90|FERC\s+order)\b",
        r"\bgeneric\s+(?:environmental\s+impact\s+statement|EIS|GEIS)\b[\s\S]{0,200}\b(?:nuclear|NRC|reactor|license\s+renewal)\b",
        r"\bgeneric\s+(?:environmental\s+assessment|EA|GEA)\b[\s\S]{0,200}\b(?:nuclear|NRC|reactor|license\s+renewal)\b",
        r"\bNUREG-1437\b",
        # DOE electricity/gas export authorizations (Section 202(e) FPA; Natural Gas Act)
        r"\belectricity\s+export\s+authorization\b",
        r"\bSection\s+202\(e\)\b[\s\S]{0,120}\b(?:export|DOE|Department\s+of\s+Energy)\b",
        r"\b(?:LNG|liquefaction|natural\s+gas)\s+export\s+authorization\b",
        # DOE Presidential Permit for cross-border transmission ("granting or denying" framing)
        r"\bgranting\s+or\s+denying\s+a\s+Presidential\s+Permit\b",
        # FERC as document author — only fires when FERC literally prepared the EIS/EA
        r"\bprepared\s+by\s+the\s+Federal\s+Energy\s+Regulatory\s+Commission\b",
        # FERC + hydroelectric within 100 chars — catches hydro license review language
        r"\bFederal\s+Energy\s+Regulatory\s+Commission[\s\S]{0,100}(?:hydroelectric|hydropower)\b",
    ],
    "pma": [
        r"\b(?:Bonneville\s+Power\s+Administration|BPA)\b",
        r"\b(?:Western\s+Area\s+Power\s+Administration|WAPA)\b",
        r"\b(?:Southeastern\s+Power\s+Administration|SEPA)\b",
        r"\b(?:Southwestern\s+Power\s+Administration|SWPA)\b",
        r"\b(?:Tennessee\s+Valley\s+Authority|TVA)\b",
        r"\bPower\s+Marketing\s+Administration\b|\bPMA\b",
    ],
    "federal_program": [
        r"\bprogrammatic\s+environmental\s+(?:impact\s+statement|assessment)\b",
        r"\bprogrammatic\s+(?:EIS|EA)\b",
        r"\b(?:DPEIS|FPEIS|SPEIS|PEIS|PEA)\b",
        r"\bthis\s+programmatic\s+(?:EIS|EA|environmental)\b",
        r"\bthis\s+GEIS\b",
        r"\btier\s*(?:1|I|one)\s+(?:NEPA\s+)?(?:review|environmental\s+(?:impact\s+statement|assessment)|EIS|EA)\b",
        r"\b(?:environmental\s+(?:impact\s+statement|assessment)|EIS|EA)\s+tier\s*(?:1|I|one)\b",
        r"\bsite[-\s]?wide\s+environmental\s+(?:impact\s+statement|assessment)\b",
        r"\b(?:SWEIS|SWEA)\b",
        r"\b(?:\d{4}\s+)?Integrated\s+Resource\s+Plan\b[\s\S]{0,160}\b(?:Programmatic\s+Environmental\s+Impact\s+Statement|Supplemental\s+Environmental\s+Impact\s+Statement|Draft\s+EIS)\b",
        r"\bLong-Term\s+Experimental\s+and\s+Management\s+Plan\b[\s\S]{0,160}\bEnvironmental\s+Impact\s+Statement\b",
    ],
    "federal_property_transaction": [
        r"\bland\s+exchange\b",
        r"\bfee[-\s]for[-\s]fee\s+land\s+exchange\b",
        r"\bexchange\s+property\s+with\b",
        r"\bdispose\s+of\s+(?:the\s+)?(?:underlying\s+)?land\s+rights\b|\bland\s+disposal\b",
        r"\bsale\s+of\s+land\s+rights\b|\bsell\s+in\s+fee\b",
        r"\bacquire\s+and\s+release\s+access\s+road\s+rights\b",
        r"\bacquire\s+(?:several\s+road\s+easements|access\s+road\s+rights|land\s+rights)\b",
        r"\bpurchase\s+(?:two\s+lots?|lots?)\s+and\s+(?:line\s+)?easements\b",
        r"\btransfer\s+ownership\b[\s\S]{0,140}\b(?:associated\s+easements?|easements?|rights?-of-way|land\s+rights)\b",
        r"\btitle\s+transfer\b[\s\S]{0,140}\b(?:easements?|rights?-of-way|land\s+rights)\b",
        r"\basset\s+exchange\b[\s\S]{0,140}\b(?:rights?-of-way|line\s+easements?|easements?)\b",
        r"\bconveyance\b",
    ],
}

HYPOTHESIS_TEMPLATES = {
    "federal_funding": "This text shows that a federal agency is funding, financing, or providing financial assistance for this project, including through a grant, loan, loan guarantee, cost-sharing arrangement, cooperative agreement, or formula-based award.",
    "federal_direct_action": "This text shows that a federal agency is the primary actor directly proposing, constructing, installing, operating, managing, upgrading, rebuilding, restoring, or otherwise implementing this project, rather than merely approving or permitting someone else's project.",
    "federal_land": "This text shows that this project is located on, crosses, or requires access to federally managed land, or that the project requires a right-of-way grant, easement, special use permit, land use authorization, or similar approval tied to use of federal land.",
    "federal_permit": "This text shows that a federal permit, license, certification, or regulatory approval is required for this project, even if the project is otherwise privately or state-led.",
    "federal_program": "This text shows that this is a programmatic, generic, site-wide, or Tier 1 environmental review covering a class of actions or a geographic area, or a broader federal planning document such as a resource management plan revision, leasing program, corridor designation, or rulemaking.",
    "federal_property_transaction": "This text shows that this involves a federal land exchange, sale, disposal, conveyance, acquisition, or transfer of land, land rights, easements, or other real-property interests.",
    "pma": "This text shows that Bonneville Power Administration (BPA), Western Area Power Administration (WAPA), Southeastern Power Administration (SEPA), Southwestern Power Administration (SWPA), or Tennessee Valley Authority (TVA) is the lead or sponsoring agency proposing or implementing this project.",
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
    ("federal_direct_action /DOE constructs NREL facility", "federal_direct_action",
     "The Department of Energy (DOE) prepared this Final Supplemental EA to assess the potential "
     "environmental effects resulting from the proposed improvements to the RFHP. Specifically, the DOE "
     "proposes to develop, construct and operate a woodchip fuel storage silo at the National Renewable "
     "Energy Laboratory's (NREL) South Table Mountain (STM) site in Golden, Colorado."),
    ("federal_direct_action /Western constructs substation", "federal_direct_action",
     "Western Area Power Administration (Western) will construct a new control building at the Lusk Rural "
     "Substation (LRS) located in Niobrara County, Wyoming. The proposed work at the LRS control building "
     "consists of the following; construct a new control building and associated foundation, demolish "
     "existing 69-kV switch, construct new Fault Interrupter foundations and install steel support structure "
     "and fault interrupter, and demolish existing control building."),
    ("federal_direct_action /Western constructs communications building", "federal_direct_action",
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
    ("federal_program / generic EIS title", "federal_program",
     "Final Generic Environmental Impact Statement for License Renewal of Nuclear Plants"),
    ("federal_program / tier 1 EIS title", "federal_program",
     "Draft Tier 1 Environmental Impact Statement"),
    ("federal_program / site-wide EIS title", "federal_program",
     "Final Site-Wide Environmental Impact Statement for the Y-12 National Security Complex"),
    ("federal_program / integrated resource plan PEIS", "federal_program",
     "2025 Integrated Resource Plan and Programmatic Environmental Impact Statement"),
    ("federal_program / energy corridors PEIS title", "federal_program",
     "Final Programmatic Environmental Impact Statement, Designation of Energy Corridors on Federal Land in the 11 Western States"),
    ("pma / WAPA constructs substation control building", "pma",
     "Western Area Power Administration (Western) will construct a new control building at the Lusk Rural "
     "Substation (LRS) located in Niobrara County, Wyoming. The proposed work at the LRS control building "
     "consists of the following; construct a new control building and associated foundation, demolish "
     "existing 69-kV switch, construct new Fault Interrupter foundations and install steel support structure "
     "and fault interrupter, and demolish existing control building."),
    ("pma / BPA radio antenna replacement", "pma",
     "Bonneville Power Administration (BPA) proposes to replace and upgrade the existing radio antennas "
     "at its substations. The proposed work will be conducted at multiple BPA substation sites throughout "
     "the Pacific Northwest."),
    ("pma / TVA transmission upgrade", "pma",
     "The Tennessee Valley Authority (TVA) proposes to construct a new 500-kV transmission line and "
     "associated substation facilities in Tennessee to improve grid reliability and support regional load growth."),
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
    # Broader proposed-action header variants (32–34% coverage in EA/EIS corpus)
    ("proposed_action", r"(?i)\b(?:proposed\s+(?:federal\s+)?action(?:\s+and\s+alternatives?)?|description\s+of\s+(?:the\s+)?proposed\s+(?:federal\s+)?action|proposed\s+federal\s+undertaking)\b"),
    ("decision", r"(?i)\bdecision\s+to\s+be\s+made\b"),
    ("agency_action", r"(?i)\b(?:agency\s+action|federal\s+action)\b"),
    # Regulatory framework: cites FLPMA, CFR parts, Section 404, etc. — high precision trigger signal
    ("regulatory_framework", r"(?i)\b(?:regulatory\s+(?:framework|context|requirements?|background|setting)|legal\s+(?:framework|background|authority|requirements?)|statutory\s+(?:background|authority|requirements?)|applicable\s+(?:laws?\s+(?:and\s+)?regulations?|statutes?\s+and\s+regulations?)|federal\s+regulatory\s+(?:framework|requirements?))\b"),
    # Executive summary: concentrates agency role + proposed action in one passage (6–9% coverage)
    ("executive_summary", r"(?i)\bexecutive\s+summary\b"),
]

SECTION_PRIOR_WEIGHTS = {
    "doc_title": 0.25,
    "first_pages": 0.10,
    "purpose_and_need": 0.18,
    "need_for_action": 0.18,
    "proposed_action": 0.18,
    "agency_action": 0.18,
    "regulatory_framework": 0.16,
    "project_description": 0.15,
    "decision": 0.10,
    "funding": 0.15,
    "cue_window": 0.12,
    "executive_summary": 0.12,
    "ce_fallback": 0.08,
}

# --- Tier 4: Class prototype sentences for embedding similarity ---

CLASS_PROTOTYPES = {
    "federal_direct_action": [
        "The Forest Service proposes to implement vegetation management on National Forest land.",
        "The Bureau of Land Management will construct a new facility at the site.",
        "This federal action consists of upgrading and replacing infrastructure at an existing federal facility.",
        "Bonneville Power Administration proposes to replace and upgrade the existing radio antennas at its substations.",
        "Bonneville Power Administration proposes to relocate laboratories and renovate an existing garage at the Ross Complex.",
        "WAPA would construct, own, operate, and maintain an interconnection switchyard in the project area.",
        "DOE proposes to develop, construct, and operate a new facility on federal property.",
    ],
    "federal_program": [
        "Programmatic Environmental Assessment for System-wide Operations and Maintenance Activities and Integrated Vegetation Management Program.",
        "Final Generic Environmental Impact Statement for License Renewal of Nuclear Plants.",
        "Draft Tier 1 Environmental Impact Statement.",
        "Final Site-Wide Environmental Impact Statement for the Y-12 National Security Complex.",
        "2025 Integrated Resource Plan and Programmatic Environmental Impact Statement.",
        "Final Programmatic Environmental Impact Statement, Designation of Energy Corridors on Federal Land in the 11 Western States.",
    ],
    "federal_land": [
        "The USFS purpose and need is to determine whether to issue a special use permit for the proposed transmission lines upgrade and rebuild.",
        "BPA proposes to acquire a perpetual right-of-way grant for BPA's existing Wautoma-Rock Creek transmission line.",
        "The Bureau of Indian Affairs is requesting a new right-of-way (ROW) for an existing 12.5kV overhead distribution line.",
        "These temporary work areas are under lands administered by the Bureau of Reclamation (BOR) and permissions must be sought from them.",
    ],
    "federal_permit": [
        "The applicant has applied for an individual permit under Section 404 of the Clean Water Act.",
        "EPA's preferred permit action is to issue the NPDES permit with conditions.",
        "Issuance of a federal Incidental Take Permit (ITP) under Section 10(a)(1)(B) of the Federal Endangered Species Act (FESA; the federal action under the National Environmental Policy Act \"NEPA\") will be required by the USFWS after the City's approval process is complete.",
        "Issuance of Presidential Permit PP-89 to Bangor Hydro-Electric Company.",
    ],
    "federal_funding": [
        "The project is funded through a Department of Energy loan guarantee.",
        "Federal financial assistance is provided through a DOE grant under Title XVII.",
        "The project is a recipient of federal funding through the Bipartisan Infrastructure Law.",
        "The project receives federal financial assistance from the Department of Transportation.",
    ],
    "federal_property_transaction": [
        "The proposed action consists of a land exchange between the federal government and a private party.",
        "BPA proposes to dispose of the underlying land rights beneath an existing substation.",
        "BPA proposes to acquire and release access road rights to ensure permanent legal access to transmission facilities.",
        "BPA proposes to sell its substation, including land rights, to the city.",
        "The agency proposes to transfer ownership of the transmission line and associated easements to the local utility.",
    ],
    "pma": [
        "Bonneville Power Administration (BPA) proposes to replace and upgrade the existing radio antennas at its substations.",
        "Western Area Power Administration (Western) will construct a new control building at the Lusk Rural Substation.",
        "WAPA would construct, own, operate, and maintain an interconnection switchyard in the project area.",
        "The Tennessee Valley Authority (TVA) proposes to install new transmission lines and associated equipment.",
        "Southeastern Power Administration (SEPA) is proposing to upgrade existing facilities at its hydropower project.",
        "The Southwestern Power Administration (SWPA) proposes to reconductor existing transmission lines.",
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
- pma: Power Marketing Administration (PMA) or Tennessee Valley Authority (TVA) — BPA, WAPA, SEPA, SWPA, or TVA — is the lead or sponsoring agency; use even when land or permit cues are also present
- federal_direct_action: federal agency (non-PMA/TVA) is the primary actor constructing or implementing the project
- federal_program: programmatic EIS, site-wide review, Tier 1 review, integrated resource plan, rulemaking, or policy framework (not primarily land-management on federal lands)
- federal_property_transaction: land exchange, sale, disposal, transfer, or acquisition of land, land rights, easements, or other real-property interests
- federal_land: project on or crossing federal land; ROW grant, special use permit, land-use plan, resource management plan, leasing program on federal lands, or land-management programmatic review
- federal_permit: federal permit, license, or authorization is the primary nexus; includes generic nuclear EIS/EA (GEIS/GEA/NUREG-1437) for NRC license renewal
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
    "federal_direct_action", "federal_program", "federal_property_transaction",
    "federal_land", "federal_permit", "federal_funding", "pma", "unknown",
})

TOP_LEVEL_CLASSES = [
    "federal_funding",
    "federal_direct_action",
    "pma",
    "federal_land",
    "federal_permit",
    "federal_program",
    "federal_property_transaction",
]

# Hierarchy for resolving primary trigger when multi-label evidence exists.
# pma sits above federal_land and federal_permit so PMA/TVA-led projects stay primary=pma
# even when ROW, easement, or permit cues are also present.
TRIGGER_HIERARCHY = [
    "federal_program",
    "federal_direct_action",
    "pma",
    "federal_property_transaction",
    "federal_land",
    "federal_permit",
    "federal_funding",
    "unknown",
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
    r'\bno\s+transfer\s+of\s+land\s+ownership\b',
]

# --- Federal funding detail sidecar patterns ---

FUNDING_CUE_RE = re.compile(
    r"\b(?:"
    r"federal\s+fund(?:ing|s)?|DOE\s+funding|Department\s+of\s+Energy\s+funding|"
    r"grant|grants|award|awards|financial\s+assistance|loan\s+guarantee|"
    r"guaranteed\s+loan|federal\s+loan|revolving\s+loan|cooperative\s+agreement|"
    r"cost[-\s]?share|Federal\s+Cost\s+Share|Title\s+XVII|EECBG|ARRA|"
    r"American\s+Recovery\s+and\s+Reinvestment\s+Act|Recovery\s+Act|"
    r"Bipartisan\s+Infrastructure\s+(?:Law|Act)|Inflation\s+Reduction\s+Act|"
    r"State\s+Energy\s+Program|SEP|Weatherization\s+Assistance\s+Program|WAP|"
    r"funding\s+opportunity\s+announcement|FOA"
    r")\b",
    re.IGNORECASE,
)

LAND_GRANT_FALSE_POSITIVE_RE = re.compile(
    r"\b(?:right[-\s]of[-\s]way|ROW|land\s+use|easement|perpetual)\s+grant\b|"
    r"\bgrant\s+(?:a\s+)?(?:right[-\s]of[-\s]way|ROW|easement|perpetual)\b|"
    r"\bROW\s+grant\b|\bright[-\s]of[-\s]way\s+\(ROW\)\s+grant\b",
    re.IGNORECASE,
)

FUNDING_PROJECT_SPECIFIC_RE = re.compile(
    r"\b(?:"
    r"proposes?\s+to\s+(?:provide|award|fund|partially\s+fund|use)|"
    r"would\s+(?:provide|award|fund|partially\s+fund|use)|"
    r"is\s+proposing\s+to\s+(?:provide|award|fund|partially\s+fund|use)|"
    r"selected\s+.+?\s+to\s+receive|recipient\s+of|receive[sd]?\s+.+?\s+(?:funding|grant|award)|"
    r"DOE\s+Funding|Federal\s+Cost\s+Share|Total\s+Project\s+(?:Cost|Value)|"
    r"amount\s+to\s+be\s+released|NEPA\s+PROVISION|Rationale?\s+for\s+determination|"
    r"Project\s+Description|Proposed\s+Action|award\s+before\s+proceeding|"
    r"loan\s+guarantee\s+to|cooperative\s+agreement\s+with|sub\s*grant|"
    r"pass\s+through\s+\$|provide\s+\$|providing\s+\$|selected\s+.+?\s+\$"
    r")\b",
    re.IGNORECASE,
)

FUNDING_GENERIC_BOILERPLATE_RE = re.compile(
    r"\bThese\s+actions\s+may\s+involve\s+financial\s+and\s+technical\s+assistance\b|"
    r"\bCovered\s+actions\s+include,\s+but\s+are\s+not\s+limited\s+to\b|"
    r"\bdo\s+not\s+include\s+rulemakings,\s+standard-settings,\s+or\s+proposed\s+DOE\s+legislation\b",
    re.IGNORECASE,
)

MONEY_RE = re.compile(
    r"(?:"
    r"\$\s*(?P<dollar_amount>\d[\d,]*(?:\.\d+)?)\s*(?P<dollar_scale>million|billion|thousand|m|b|k)?"
    r"|"
    r"\b(?P<word_amount>\d+(?:\.\d+)?)\s*(?P<word_scale>million|billion|thousand)\s+dollars?\b"
    r")",
    re.IGNORECASE,
)

PERCENT_RE = re.compile(r"\b(?P<pct>\d{1,3}(?:\.\d+)?)\s*(?:%|percent)\b", re.IGNORECASE)

FUNDING_MECHANISM_PATTERNS = {
    "loan_guarantee": re.compile(r"\bloan\s+guarantee\b|\bguaranteed\s+loan\b", re.IGNORECASE),
    "revolving_loan": re.compile(r"\brevolving\s+loan\b", re.IGNORECASE),
    "federal_loan": re.compile(
        r"\bfederal\s+loan\b|\bloans?\s+from\s+(?:DOE|Department\s+of\s+Energy|USDA|DOT|HUD)\b",
        re.IGNORECASE,
    ),
    "cooperative_agreement": re.compile(r"\bcooperative\s+agreement\b", re.IGNORECASE),
    "formula_grant": re.compile(
        r"\bformula[-\s]based\s+(?:grant|award)s?\b|\bformula\s+(?:grant|award)s?\b|\bEECBG\b",
        re.IGNORECASE,
    ),
    # DOE EERE PMC-ND determination form: presence of both RECIPIENT: and
    # "Procurement Instrument Number" reliably identifies a federal grant/award.
    "pmc_nd_form": re.compile(
        r"(?=[\s\S]*RECIPIENT\s*:)(?=[\s\S]*Procurement\s+Instrument\s+Number)",
        re.IGNORECASE,
    ),
    # All ARPA-E projects are competitively awarded federal grants.
    "arpa_e": re.compile(
        r"\bARPA[-\s]?E\b|\bAdvanced\s+Research\s+Projects\s+Agency[-\s–—]*Energy\b",
        re.IGNORECASE,
    ),
    "grant_or_award": re.compile(
        r"\b(?:federal\s+|DOE\s+|Department\s+of\s+Energy\s+)?(?:grant|grants|award|awards)\b",
        re.IGNORECASE,
    ),
    "cost_share": re.compile(r"\bcost[-\s]?share(?:d|s|r|ing|ment)?\b|\bfederal\s+cost\s+share\b", re.IGNORECASE),
    "financial_assistance": re.compile(r"\bfinancial\s+assistance\b", re.IGNORECASE),
    "generic_funding": re.compile(
        r"\b(?:provide|providing|provided|receive|receives|recipient\s+of)\s+(?:federal\s+)?fund(?:ing|s)?\b|"
        r"\bfederal\s+funding\b",
        re.IGNORECASE,
    ),
}

FUNDING_MECHANISM_PRIORITY = [
    "loan_guarantee",
    "revolving_loan",
    "federal_loan",
    "cooperative_agreement",
    "formula_grant",
    "pmc_nd_form",
    "arpa_e",
    "grant_or_award",
    "cost_share",
    "financial_assistance",
    "generic_funding",
]

FUNDING_PROGRAM_PATTERNS = {
    "ARRA": re.compile(r"\bAmerican\s+Recovery\s+and\s+Reinvestment\s+Act\b|\bARRA\b|\bRecovery\s+Act\b", re.IGNORECASE),
    "EECBG": re.compile(r"\bEECBG\b|Energy\s+Efficiency\s+and\s+Conservation\s+Block\s+Grant", re.IGNORECASE),
    "SEP": re.compile(r"\bState\s+Energy\s+Program\b|\bSEP\b", re.IGNORECASE),
    "WAP": re.compile(r"\bWeatherization\s+Assistance\s+Program\b|\bWAP\b", re.IGNORECASE),
    "Title XVII": re.compile(r"\bTitle\s+XVII\b", re.IGNORECASE),
    "BIL": re.compile(r"\bBipartisan\s+Infrastructure\s+(?:Law|Act)\b|\bBIL\b", re.IGNORECASE),
    "IRA": re.compile(r"\bInflation\s+Reduction\s+Act\b|\bIRA\b", re.IGNORECASE),
    "FOA": re.compile(r"\bfunding\s+opportunity\s+announcement\b|\bFOA\b", re.IGNORECASE),
}

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
    Check action vs. authorizer verb signals to distinguish federal_direct_action from federal_land.
    Returns 'federal_direct_action', 'federal_land', or None if no signal found.
    Priority: action verbs win when both are present (federal_direct_action > federal_land).
    """
    for pat in FEDERAL_ACTION_VERB_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "federal_direct_action"
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


def _is_property_transaction_exclusion(text: str) -> bool:
    for pat in PROPERTY_TRANSACTION_EXCLUSION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


_SENTENCE_MODEL = None
_HYPOTHESIS_EMBEDDINGS = None
_LOCAL_SCORER_KIND = None
_CROSS_ENCODER = None
_SETFIT_MODEL = None
_SETFIT_LABELS: list[str] = []


def _hierarchy_primary(classes: list[str]) -> str:
    """Return the highest-priority class from a multi-label list per TRIGGER_HIERARCHY."""
    for cls in TRIGGER_HIERARCHY:
        if cls in classes:
            return cls
    return "unknown"


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
        if trigger_class == "federal_property_transaction" and _is_property_transaction_exclusion(evidence):
            continue
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
    sponsor = str(project_row.get("project_sponsor") or "")
    priors: list[str] = []
    if _agency_matches(agency, AGENCY_PMA_MAP) or _agency_matches(sponsor, AGENCY_PMA_MAP):
        priors.append("pma")
    elif _agency_matches(agency, frozenset({"DOE", "Department of Energy"})):
        if _agency_matches(sponsor, AGENCY_PMA_MAP):
            priors.append("pma")
        else:
            priors.extend(["federal_funding", "federal_direct_action"])
    elif _agency_matches(agency, frozenset({"USACE", "Army Corps of Engineers"})):
        priors.extend(["federal_permit", "federal_land"])
    elif _agency_matches(agency, AGENCY_DIRECT_ACTION_MAP):
        priors.append("federal_direct_action")
    elif _agency_matches(agency, AGENCY_LAND_MAP):
        priors.extend(["federal_land", "federal_direct_action", "federal_program"])
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
    # Auto-accept any high-confidence result not explicitly requiring T4 verification.
    # Agency metadata for ambiguous agencies (DOE, USACE) still requires document evidence
    # and is handled by SEND_TO_TIER4_RULE_IDS; all other high-confidence results are trusted.
    if result.get("nepa_trigger_confidence") == "high":
        if result.get("nepa_trigger_evidence_source") == "agency_metadata":
            evidence_text = result.get("nepa_trigger_evidence_text", "")
            return not _agency_matches(evidence_text, AMBIGUOUS_METADATA_AGENCIES)
        return True
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
    sections = _extract_section_windows(section_text, EA_EIS_SECTION_PATTERNS, window=2000, max_sections=10)
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
            if cls == "federal_property_transaction" and _is_property_transaction_exclusion(evidence):
                continue
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

            # Always add project description when no page text was extracted — a doc_title
            # chunk alone (chunk_count=1) is too sparse for NLI scoring, and projects with
            # no documents at all need the description as their only text signal.
            if page_rows.empty:
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
                        "section_type": "project_description",
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

    sponsor = str(project_row.get("project_sponsor") or "")
    if _agency_matches(agency, AGENCY_PMA_MAP) or _agency_matches(sponsor, AGENCY_PMA_MAP):
        candidates.extend(["pma", "federal_land", "federal_permit"])
    elif _agency_matches(agency, frozenset({"DOE", "Department of Energy"})):
        if _agency_matches(sponsor, AGENCY_PMA_MAP):
            candidates.extend(["pma", "federal_land", "federal_permit"])
        else:
            candidates.extend(["federal_funding", "federal_direct_action"])
    elif _agency_matches(agency, frozenset({"USACE", "Army Corps of Engineers"})):
        candidates.extend(["federal_permit", "federal_land"])
    elif _agency_matches(agency, AGENCY_DIRECT_ACTION_MAP):
        candidates.append("federal_direct_action")
    elif _agency_matches(agency, AGENCY_LAND_MAP):
        candidates.extend(["federal_land", "federal_direct_action", "federal_program"])
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
        log.info("  Running NLI cross-encoder on %s (premise, hypothesis) pairs", f"{len(pairs):,}")
        predictions = _CROSS_ENCODER.predict(pairs, apply_softmax=True, show_progress_bar=True)
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


def _load_label_files(
    files: tuple,
    rule_id: str,
    route_reason: str,
    projects_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    existing = [p for p in files if p.exists()]
    if not existing:
        return []

    frames = []
    for p in existing:
        try:
            frames.append(pd.read_csv(p, usecols=["project_id", "manual_trigger"]))
        except Exception as exc:
            log.warning("Could not read label file %s: %s", p, exc)

    if not frames:
        return []

    labels_df = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["manual_trigger"])
        .query("manual_trigger != '' and manual_trigger != 'ambiguous'")
        .drop_duplicates("project_id", keep="last")
    )
    labels_df = labels_df[labels_df["manual_trigger"].isin(set(TOP_LEVEL_CLASSES))]

    in_scope = projects_df[["project_id"]].merge(labels_df, on="project_id", how="inner")
    if in_scope.empty:
        return []

    return [
        make_result(
            project_id=row["project_id"],
            primary=row["manual_trigger"],
            confidence="high",
            evidence_text="manual_label",
            evidence_source="description",
            rule_id=rule_id,
            manual_review=False,
            route_policy="auto_accept",
            route_reason=route_reason,
        )
        for _, row in in_scope.iterrows()
    ]


def tier0_manual_labels(projects_df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Tier 0: directly finalize any project that already has a manual_trigger label
    in one of the hand-labeled training CSVs (SETFIT_TRAINING_FILES or
    NLI_TRAINING_FILES). Runs before all other tiers so training examples are
    never re-processed by the pipeline.
    """
    combined = (
        _load_label_files(
            SETFIT_TRAINING_FILES, "T0_setfit_training", "setfit_training_example", projects_df
        )
        + _load_label_files(
            NLI_TRAINING_FILES, "T0_nli_training", "nli_training_example", projects_df
        )
    )
    # Deduplicate across file groups: NLI labels override SetFit labels for any
    # project_id that appears in both files.
    seen: dict[str, dict] = {}
    for r in combined:
        seen[r["project_id"]] = r
    results = list(seen.values())
    log.info("  → %d projects finalized from manual labels", len(results))
    return results


def _load_setfit_model() -> None:
    """Load SetFit model from disk if available. Silent no-op if not found."""
    global _SETFIT_MODEL, _SETFIT_LABELS
    if not SETFIT_MODEL_PATH.exists():
        return
    try:
        import json
        from setfit import SetFitModel
        _SETFIT_MODEL = SetFitModel.from_pretrained(str(SETFIT_MODEL_PATH))
        label_file = SETFIT_MODEL_PATH / "label_list.json"
        if label_file.exists():
            _SETFIT_LABELS = json.loads(label_file.read_text())
        else:
            _SETFIT_LABELS = list(getattr(_SETFIT_MODEL, "labels", []))
        log.info("SetFit model loaded from %s (%d classes)", SETFIT_MODEL_PATH, len(_SETFIT_LABELS))
    except Exception as exc:
        log.warning("SetFit model found at %s but failed to load: %s", SETFIT_MODEL_PATH, exc)
        _SETFIT_MODEL = None
        _SETFIT_LABELS = []


def _prep_setfit_text(row: Any) -> str:
    """Build inference text from project_title + project_description (matches training prep)."""
    title = str(row.get("project_title") or "").strip()
    desc  = str(row.get("project_description") or "").strip()
    if desc.startswith("[") and desc.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(desc)
            if isinstance(parsed, list):
                desc = " ".join(str(x) for x in parsed)
        except Exception:
            pass
    return f"{title} {desc[:2000]}".strip()


def tier3b_setfit_doe_ce(
    project_ids: list[str],
    projects_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    SetFit classifier for DOE CE projects.

    Runs between Tier 3 and Tier 4. Only fires when:
      - A trained model exists at SETFIT_MODEL_PATH
      - The project is DOE (lead_agency_harmonized) + CE (process_type)

    Projects with top-class probability >= SETFIT_CONFIDENCE_THRESHOLD and
    margin >= SETFIT_MARGIN_THRESHOLD are auto-accepted. The rest fall through
    to Tier 4 unchanged.
    """
    if _SETFIT_MODEL is None or not _SETFIT_LABELS:
        return []

    import numpy as np

    target = projects_df[
        projects_df["project_id"].isin(set(project_ids))
        & projects_df["lead_agency_harmonized"].fillna("").astype(str).str.contains(
            "Department of Energy", case=False, na=False
        )
        & (projects_df["process_type"].fillna("").astype(str).str.upper() == "CE")
    ].copy()

    if target.empty:
        return []

    texts = [_prep_setfit_text(row) for _, row in target.iterrows()]

    try:
        probs = _SETFIT_MODEL.predict_proba(texts)
        if hasattr(probs, "numpy"):
            probs = probs.numpy()
        probs = np.array(probs)
    except Exception as exc:
        log.warning("SetFit predict_proba failed (%s); skipping Tier 3b", exc)
        return []

    results = []
    for (_, row), prob_vec in zip(target.iterrows(), probs):
        top_idx    = int(np.argmax(prob_vec))
        top_prob   = float(prob_vec[top_idx])
        sorted_p   = sorted(prob_vec, reverse=True)
        second_prob = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
        margin     = top_prob - second_prob
        top_class  = _SETFIT_LABELS[top_idx]

        if top_class not in TOP_LEVEL_CLASSES:
            continue

        if top_prob >= SETFIT_CONFIDENCE_THRESHOLD and margin >= SETFIT_MARGIN_THRESHOLD:
            results.append(make_result(
                project_id=row["project_id"],
                primary=top_class,
                confidence="high",
                evidence_text=f"setfit prob={top_prob:.3f} margin={margin:.3f}",
                evidence_source="description",
                rule_id="T3b_setfit_doe_ce",
                manual_review=False,
                route_policy="auto_accept",
                route_reason="setfit_high_confidence",
            ))

    return results


def tier1a_metadata(projects: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for _, row in projects.iterrows():
        pid = row["project_id"]
        agency = str(row.get("lead_agency_harmonized") or "").strip()
        sponsor = str(row.get("project_sponsor") or "").strip()
        text = " ".join([
            str(row.get("project_title") or ""),
            str(row.get("project_description") or ""),
        ])
        agency_code = _get_agency_code(agency)
        land_control_match = re.search(BLM_USFS_LAND_CONTROL_PATTERN, text, re.IGNORECASE)

        if (
            _agency_matches(agency, frozenset({"BLM", "Bureau of Land Management"}))
            and re.search(FOREST_SERVICE_SPONSOR_PATTERN, sponsor, re.IGNORECASE)
            and land_control_match
        ):
            results.append(make_result(
                project_id=pid,
                primary="federal_land",
                confidence="high",
                evidence_text=f"{agency} | sponsor={sponsor} | text={land_control_match.group(0)}",
                evidence_source="agency_metadata",
                rule_id="T1a_BLM_USFS_land_control",
                manual_review=False,
                route_policy="auto_accept",
                route_reason="blm_usfs_land_control_metadata",
            ))
        elif _agency_matches(agency, AGENCY_PERMIT_MAP):
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
        elif _agency_matches(agency, AGENCY_PMA_MAP) or _agency_matches(agency, AGENCY_DIRECT_ACTION_MAP):
            # PMA/TVA agencies → primary=pma; CBP → primary=federal_direct_action.
            is_pma = _agency_matches(agency, AGENCY_PMA_MAP)
            is_cbp = _agency_matches(agency, frozenset({"CBP", "U.S. Customs and Border Protection", "Customs and Border Protection"}))
            if is_pma:
                land_cues = re.search(
                    r'\bright[-\s]of[-\s]way\b|\bROW\b|\bperpetual\b|\beasement\b'
                    r'|\bland\s+exchange\b|\bdispose\b|\bdisposal\b|\bacquire\b|\bacquisition\b'
                    r'|\btransfer\s+ownership\b|\btitle\s+transfer\b|\bsale\s+of\s+land\b',
                    text, re.IGNORECASE,
                )
                permit_cues = re.search(
                    r'\bright[-\s]of[-\s]way\s+grant\b|\bspecial\s+use\s+permit\b|\bSection\s+404\b'
                    r'|\bNPDES\b|\bferc\s+license\b|\blicense\s+renewal\b',
                    text, re.IGNORECASE,
                )
                secondary = []
                if land_cues:
                    secondary.append("federal_land")
                if permit_cues:
                    secondary.append("federal_permit")
                results.append(make_result(
                    project_id=pid,
                    primary="pma",
                    confidence="high",
                    evidence_text=agency,
                    evidence_source="agency_metadata",
                    rule_id=f"T1a_{agency_code}_pma",
                    secondary=secondary,
                    manual_review=False,
                    route_policy="auto_accept",
                    route_reason="deterministic_pma_metadata",
                ))
            elif is_cbp:
                results.append(make_result(
                    project_id=pid,
                    primary="federal_direct_action",
                    confidence="high",
                    evidence_text=agency,
                    evidence_source="agency_metadata",
                    rule_id=f"T1a_{agency_code}_direct_action",
                    manual_review=False,
                    route_policy="auto_accept",
                    route_reason="deterministic_direct_action_metadata",
                ))
        elif _agency_matches(agency, AGENCY_LAND_MAP):
            verb_class = _verb_class(text)
            trigger = verb_class if verb_class else "federal_land"
            confidence = "high"
            verb_suffix = "direct_action" if trigger == "federal_direct_action" else "land"
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
            # PMA/TVA projects are organizationally under DOE but are PMA-category agencies.
            # lead_agency_harmonized = "Department of Energy" for these projects, so the
            # AGENCY_PMA_MAP branch above never fires. Check project_sponsor instead.
            if _agency_matches(sponsor, AGENCY_PMA_MAP):
                _sponsor_code = (
                    "BPA" if _agency_matches(sponsor, frozenset({"BPA", "Bonneville Power Administration"}))
                    else "WAPA" if _agency_matches(sponsor, frozenset({"WAPA", "Western Area Power Administration"}))
                    else "SEPA" if _agency_matches(sponsor, frozenset({"SEPA", "Southeastern Power Administration"}))
                    else "SWPA" if _agency_matches(sponsor, frozenset({"SWPA", "Southwestern Power Administration"}))
                    else "TVA" if _agency_matches(sponsor, frozenset({"TVA", "Tennessee Valley Authority"}))
                    else "PMA"
                )
                _land_cues = re.search(
                    r'\bright[-\s]of[-\s]way\b|\bROW\b|\bperpetual\b|\beasement\b'
                    r'|\bland\s+exchange\b|\bdispose\b|\bdisposal\b|\bacquire\b|\bacquisition\b'
                    r'|\btransfer\s+ownership\b|\btitle\s+transfer\b|\bsale\s+of\s+land\b',
                    text, re.IGNORECASE,
                )
                _permit_cues = re.search(
                    r'\bright[-\s]of[-\s]way\s+grant\b|\bspecial\s+use\s+permit\b|\bSection\s+404\b'
                    r'|\bNPDES\b|\bferc\s+license\b|\blicense\s+renewal\b',
                    text, re.IGNORECASE,
                )
                _secondary = []
                if _land_cues:
                    _secondary.append("federal_land")
                if _permit_cues:
                    _secondary.append("federal_permit")
                results.append(make_result(
                    project_id=pid,
                    primary="pma",
                    confidence="high",
                    evidence_text=f"sponsor={sponsor}",
                    evidence_source="agency_metadata",
                    rule_id=f"T1a_{_sponsor_code}_pma",
                    secondary=_secondary,
                    manual_review=False,
                    route_policy="auto_accept",
                    route_reason="deterministic_pma_sponsor_metadata",
                ))
            else:
                doe_funding_patterns = [
                    r"\b(?:loan\s+guarantee|financial\s+assistance)\b",
                    r"\bTitle\s+XVII\b",
                    r"\bfunded\s+(?:by|through|under)\b",
                    r"\b(?:DOE|Department\s+of\s+Energy)\s+(?:grant|award|funding)\b",
                    r"\bthrough\s+(?:a\s+)?cooperative\s+agreement\b[\s\S]{0,120}\bpartially\s+fund\b",
                    r"\bproviding\s+financial\s+assistance\s+to\b[\s\S]{0,120}\b(?:under|through)\s+(?:a\s+)?cooperative\s+agreement\b",
                    r"\bawarding\s+a\s+grant\b[\s\S]{0,120}\bpartially\s+fund\b",
                    r"\bFederal\s+Cost\s+Share\b",
                    r"\b(?:DOE\s+)?EECBG\s+funding\b",
                    r"\bformula(?:-based)?\s+(?:awards?|grants?)\b",
                    r"(?:Administrative\s+(?:and\s+)?Legal\s+Requirements\s+Document|\bALRD\b)[\s\S]{0,160}\bformula(?:-based)?\s+(?:awards?|grants?)\b",
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
                elif _verb_class(text) == "federal_direct_action":
                    results.append(make_result(
                        project_id=pid,
                        primary="federal_direct_action",
                        confidence="medium",
                        evidence_text=agency,
                        evidence_source="agency_metadata",
                        rule_id="T1a_DOE_direct_action",
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
            title = str(row.get("document_title") or row.get("file_name") or "")
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

    # Zero-score bypass: projects where every retrieved chunk has retrieval_score == 0.0
    # have no cue signal (all boilerplate). Running NLI on them produces near-zero entailment
    # scores across all classes and yields unreliable results. Route directly to Tier 5 instead.
    zero_score_ids: set[str] = set()
    if not contexts.empty:
        max_retrieval = contexts.groupby("project_id")["retrieval_score"].max()
        zero_score_ids = set(max_retrieval[max_retrieval == 0.0].index)

    nli_contexts = contexts[~contexts["project_id"].isin(zero_score_ids)].copy() if zero_score_ids else contexts

    chunk_scores = run_local_nli_on_chunks(nli_contexts, candidate_classes_by_project)
    doc_scores = aggregate_tier4_scores(chunk_scores)
    results: list[dict[str, Any]] = []
    covered_ids = set()

    if zero_score_ids:
        log.info(
            "  Tier 4 zero-score bypass: %d projects skipped NLI (no cue signal) → Tier 5 queue",
            len(zero_score_ids),
        )
        for pid in sorted(zero_score_ids):
            provisional_result = provisional.get(pid, {})
            results.append(make_result(
                project_id=pid,
                primary="unknown",
                confidence="low",
                evidence_text="no retrieval signal in document chunks",
                evidence_source="document_text",
                rule_id="T4_local_uncertain",
                manual_review=True,
                route_policy="tier5_candidate",
                route_reason="zero_retrieval_score",
                provisional_rule_id=provisional_result.get("nepa_trigger_rule_id", ""),
                provisional_confidence=provisional_result.get("nepa_trigger_confidence", ""),
            ))
        covered_ids.update(zero_score_ids)

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
        SELECT project_id, lead_agency_harmonized, project_sponsor, project_title,
               project_description, process_type, dataset_source
        FROM read_parquet('{PROJECTS_PATH}')
        WHERE {CLEAN_ENERGY_FILTER}
    """).fetchdf()

    if sample:
        projects = projects.sample(sample, random_state=42)
        log.info("Sample mode: %s projects", len(projects))

    import time as _time

    all_project_ids = set(projects["project_id"])
    log.info("Processing %s clean energy projects", f"{len(all_project_ids):,}")
    _load_setfit_model()

    finalized: dict[str, dict[str, Any]] = {}
    provisional: dict[str, dict[str, Any]] = {}
    _run_start = _time.time()

    def _ingest(results: list[dict[str, Any]]) -> None:
        for result in results:
            pid = result["project_id"]
            if should_auto_accept(result):
                if pid not in finalized:  # earlier tier's auto-accept takes precedence
                    finalized[pid] = result
            else:
                existing = provisional.get(pid)
                if existing is None or _result_confidence_rank(result) >= _result_confidence_rank(existing):
                    provisional[pid] = result

    def _remaining() -> list[str]:
        return sorted(pid for pid in all_project_ids if pid not in finalized)

    def _pct() -> str:
        return f"{len(finalized) / len(all_project_ids):.1%}"

    def _elapsed() -> str:
        return f"{(_time.time() - _run_start) / 60:.1f}m"

    log.info("Tier 0: manual labels")
    _ingest(tier0_manual_labels(projects))
    log.info("  → %s finalized (%s) [%s elapsed]", f"{len(finalized):,}", _pct(), _elapsed())

    log.info("Tier 1a: agency metadata")
    _ingest(tier1a_metadata(projects))
    log.info("  → %s finalized (%s) [%s elapsed]", f"{len(finalized):,}", _pct(), _elapsed())

    log.info("Tier 1b: title and description keywords")
    unresolved_df = projects[projects["project_id"].isin(_remaining())]
    _ingest(tier1b_title_description(unresolved_df))
    log.info("  → %s finalized (%s) [%s elapsed]", f"{len(finalized):,}", _pct(), _elapsed())

    log.info("Tier 2: document title scan")
    _ingest(tier2_doc_title(_remaining(), projects, conn))
    log.info("  → %s finalized (%s) [%s elapsed]", f"{len(finalized):,}", _pct(), _elapsed())

    log.info("Tier 3: purpose-and-need / candidate section extraction")
    _ingest(tier3_purpose_and_need(_remaining(), projects, conn))
    log.info("  → %s finalized (%s) [%s elapsed]", f"{len(finalized):,}", _pct(), _elapsed())

    if _SETFIT_MODEL is not None:
        log.info("Tier 3b: SetFit DOE CE classifier")
        _ingest(tier3b_setfit_doe_ce(_remaining(), projects))
        log.info("  → %s finalized (%s) [%s elapsed]", f"{len(finalized):,}", _pct(), _elapsed())

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
    log.info("  → %s finalized after Tier 4 (%s) [%s elapsed]", f"{len(finalized):,}", _pct(), _elapsed())

    # Always persist the Tier 5 queue (prompt-ready context for every uncertain project),
    # even when --use-llm is not set — this is what makes a later standalone Tier 5 replay
    # possible without re-running tiers 0-4.
    low_conf_ids = sorted(tier4_low_conf)
    queue_df = build_tier5_queue(low_conf_ids, doc_scores, projects, provisional, tier4_result_lookup)
    if not queue_df.empty:
        queue_df.to_parquet(TIER5_QUEUE_PATH, index=False)
        log.info("Tier 5 queue persisted: %s projects → %s", f"{len(queue_df):,}", TIER5_QUEUE_PATH)

    if use_llm:
        if not queue_df.empty:
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


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_land_grant_false_positives(text: str) -> str:
    """Remove land-authorization grant phrases before funding mechanism scans."""
    return LAND_GRANT_FALSE_POSITIVE_RE.sub(" ", str(text or ""))


def _is_project_specific_funding_context(context_text: str, source: str) -> bool:
    text = _collapse_ws(context_text)
    if not text or not FUNDING_CUE_RE.search(text):
        return False
    if FUNDING_GENERIC_BOILERPLATE_RE.search(text) and not FUNDING_PROJECT_SPECIFIC_RE.search(text):
        return False
    if source in {"project_metadata", "trigger_evidence", "doc_title"}:
        return True
    return bool(FUNDING_PROJECT_SPECIFIC_RE.search(text))


def _funding_context_windows(
    source_texts: list[tuple[str, str]],
    window_before: int = 420,
    window_after: int = 760,
    max_windows_per_source: int = 14,
) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for source, raw_text in source_texts:
        text = _strip_land_grant_false_positives(raw_text)
        if not text.strip():
            continue
        source_count = 0
        for match in FUNDING_CUE_RE.finditer(text):
            start = max(0, match.start() - window_before)
            end = min(len(text), match.end() + window_after)
            context = _collapse_ws(text[start:end])
            if not _is_project_specific_funding_context(context, source):
                continue
            context_hash = _chunk_hash(context)
            if context_hash in seen_hashes:
                continue
            seen_hashes.add(context_hash)
            contexts.append({
                "source": source,
                "text": context,
            })
            source_count += 1
            if source_count >= max_windows_per_source:
                break
    return contexts


def _parse_money_match(match: re.Match) -> Optional[float]:
    amount_raw = match.group("dollar_amount") or match.group("word_amount")
    scale_raw = match.group("dollar_scale") or match.group("word_scale") or ""
    if not amount_raw:
        return None
    try:
        value = float(str(amount_raw).replace(",", ""))
    except ValueError:
        return None

    scale = scale_raw.lower()
    if scale in {"billion", "b"}:
        value *= 1_000_000_000
    elif scale in {"million", "m"}:
        value *= 1_000_000
    elif scale in {"thousand", "k"}:
        value *= 1_000
    if value < 0:
        return None
    return round(value, 2)


def _money_candidate_kind(context: str, match: re.Match) -> str:
    before = context[max(0, match.start() - 140): match.start()]
    after = context[match.end(): min(len(context), match.end() + 90)]
    label = _collapse_ws(f"{before} {after}")
    before_tail = _collapse_ws(before[-90:])

    federal_pat = (
        r"\b(?:DOE\s*:|DOE\s+Funding|Total\s+DOE\s+Funding|Federal\s+Cost\s+Share|"
        r"Federal\s*/|Federal\s+fund(?:ing|s)?|federal\s+share|grant|award|"
        r"sub\s*grant|loan\s+guarantee|SEP\s+funding|EECBG\s+funding|"
        r"Recovery\s+Act\s+funds?|amount\s+to\s+be\s+released\s+in\s+this\s+determination\s+DOE)\b"
    )
    total_pat = (
        r"\b(?:Total\s+Project\s+(?:Cost|Value)|Total\s+Project|Total\s*:|overall\s+project\s+cost|"
        r"project\s+cost)\b"
    )
    recipient_pat = (
        r"\b(?:Cost\s+Share|Recipient\s+Share|Non[-\s]?Federal|Applicant\s+Share|Private\s+Share|"
        r"recipient\s+cost)\b"
    )

    def _last_match_start(pattern: str, text: str) -> int:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        return matches[-1].start() if matches else -1

    federal_pos = _last_match_start(federal_pat, before_tail)
    total_pos = _last_match_start(total_pat, before_tail)
    recipient_pos = _last_match_start(recipient_pat, before_tail)

    # Use the closest preceding label first. This prevents "DOE Funding: $x Cost Share: $y"
    # from treating the cost-share amount as another federal funding amount.
    if recipient_pos > max(federal_pos, total_pos):
        return "recipient_cost_share"
    if total_pos > max(federal_pos, recipient_pos):
        return "total_project_cost"
    if federal_pos >= 0 or re.search(federal_pat, label, re.IGNORECASE):
        return "federal_amount"
    if re.search(recipient_pat, label, re.IGNORECASE):
        return "recipient_cost_share"
    if re.search(total_pat, label, re.IGNORECASE):
        return "total_project_cost"
    return "unlabeled_amount"


def _extract_amount_candidates(contexts: list[dict[str, str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()
    for context in contexts:
        text = context["text"]
        for match in MONEY_RE.finditer(text):
            amount = _parse_money_match(match)
            if amount is None:
                continue
            kind = _money_candidate_kind(text, match)
            if kind == "unlabeled_amount":
                nearby = text[max(0, match.start() - 220): min(len(text), match.end() + 220)]
                if not FUNDING_CUE_RE.search(nearby):
                    continue
            evidence = _collapse_ws(text[max(0, match.start() - 180): min(len(text), match.end() + 220)])
            key = (kind, amount, evidence[:120])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "amount_usd": amount,
                "kind": kind,
                "source": context["source"],
                "match_text": match.group(0),
                "evidence_text": evidence,
            })
    return candidates


def _extract_percent_candidates(contexts: list[dict[str, str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()
    for context in contexts:
        text = context["text"]
        for match in PERCENT_RE.finditer(text):
            try:
                pct = float(match.group("pct"))
            except ValueError:
                continue
            if pct < 0 or pct > 100:
                continue
            nearby = text[max(0, match.start() - 220): min(len(text), match.end() + 220)]
            if not FUNDING_CUE_RE.search(nearby):
                continue
            evidence = _collapse_ws(text[max(0, match.start() - 180): min(len(text), match.end() + 220)])
            key = (pct, evidence[:120])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "percent": pct,
                "source": context["source"],
                "match_text": match.group(0),
                "evidence_text": evidence,
            })
    return candidates


def _single_distinct_amount(candidates: list[dict[str, Any]], kind: str) -> tuple[Optional[float], bool]:
    values = sorted({round(float(c["amount_usd"]), 2) for c in candidates if c.get("kind") == kind and c.get("amount_usd") is not None})
    if len(values) == 1:
        return values[0], False
    if len(values) > 1:
        return None, True
    return None, False


def _single_distinct_percent(candidates: list[dict[str, Any]]) -> tuple[Optional[float], bool]:
    values = sorted({round(float(c["percent"]), 4) for c in candidates if c.get("percent") is not None})
    if len(values) == 1:
        return values[0], False
    if len(values) > 1:
        return None, True
    return None, False


def _extract_funding_mechanisms_and_programs(contexts: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    combined = "\n\n".join(context["text"] for context in contexts)
    cleaned = _strip_land_grant_false_positives(combined)

    mechanisms = [
        mechanism
        for mechanism, pattern in FUNDING_MECHANISM_PATTERNS.items()
        if pattern.search(cleaned)
    ]
    mechanisms = [m for m in FUNDING_MECHANISM_PRIORITY if m in mechanisms]
    programs = [
        program
        for program, pattern in FUNDING_PROGRAM_PATTERNS.items()
        if pattern.search(cleaned)
    ]
    return mechanisms, programs


def _best_funding_evidence_context(contexts: list[dict[str, str]], amount_candidates: list[dict[str, Any]]) -> tuple[str, str]:
    federal_amounts = [c for c in amount_candidates if c.get("kind") == "federal_amount"]
    if federal_amounts:
        best = federal_amounts[0]
        return str(best.get("evidence_text") or ""), str(best.get("source") or "")
    if amount_candidates:
        best = amount_candidates[0]
        return str(best.get("evidence_text") or ""), str(best.get("source") or "")
    if contexts:
        return contexts[0]["text"][:900], contexts[0]["source"]
    return "", ""


def _extract_funding_detail_from_sources(
    project_id: str,
    source_texts: list[tuple[str, str]],
    run_at: str,
) -> dict[str, Any]:
    contexts = _funding_context_windows(source_texts)
    mechanisms, programs = _extract_funding_mechanisms_and_programs(contexts) if contexts else ([], [])
    primary_mechanism = next((m for m in FUNDING_MECHANISM_PRIORITY if m in mechanisms), "unknown_funding")
    mechanism_multi = mechanisms if mechanisms else ["unknown_funding"]

    amount_candidates = _extract_amount_candidates(contexts)
    percent_candidates = _extract_percent_candidates(contexts)

    federal_amount, federal_conflict = _single_distinct_amount(amount_candidates, "federal_amount")
    total_cost, total_conflict = _single_distinct_amount(amount_candidates, "total_project_cost")
    recipient_share, recipient_conflict = _single_distinct_amount(amount_candidates, "recipient_cost_share")
    explicit_pct, pct_conflict = _single_distinct_percent(percent_candidates)

    computed_pct: Optional[float] = None
    computed_pct_conflict = False
    if federal_amount is not None and total_cost is not None and total_cost > 0:
        computed_pct = round(100 * federal_amount / total_cost, 2)
        if computed_pct < 0 or computed_pct > 100:
            computed_pct = None
            computed_pct_conflict = True

    funding_share_pct = explicit_pct if explicit_pct is not None else computed_pct
    amount_conflict = federal_conflict or total_conflict or recipient_conflict or pct_conflict or computed_pct_conflict
    evidence_text, evidence_source = _best_funding_evidence_context(contexts, amount_candidates)

    if primary_mechanism == "unknown_funding":
        confidence = "low"
    elif amount_conflict:
        confidence = "medium"
    elif federal_amount is not None or funding_share_pct is not None:
        confidence = "high"
    else:
        confidence = "medium"

    manual_review = amount_conflict or primary_mechanism == "unknown_funding"
    candidates_payload = {
        "amount_candidates": amount_candidates,
        "percent_candidates": percent_candidates,
        "amount_conflict": amount_conflict,
    }

    return {
        "project_id": project_id,
        "federal_funding_type_primary": primary_mechanism,
        "federal_funding_type_multi": mechanism_multi,
        "federal_funding_program_multi": programs,
        "federal_funding_amount_usd": federal_amount,
        "federal_funding_total_project_cost_usd": total_cost,
        "federal_funding_recipient_cost_share_usd": recipient_share,
        "federal_funding_share_pct": funding_share_pct,
        "federal_funding_evidence_text": evidence_text,
        "federal_funding_evidence_source": evidence_source,
        "federal_funding_confidence": confidence,
        "federal_funding_manual_review": manual_review,
        "federal_funding_amount_candidates_json": json.dumps(candidates_payload, sort_keys=True),
        "federal_funding_extraction_run_at": run_at,
    }


def _make_funding_detail_row(
    project_row: pd.Series,
    doc_text: str,
    doc_title: str,
    run_at: str,
) -> dict[str, Any]:
    project_metadata = " ".join([
        str(project_row.get("project_title") or ""),
        str(project_row.get("project_description") or ""),
        str(project_row.get("project_sponsor") or ""),
    ])
    source_texts = [
        ("trigger_evidence", str(project_row.get("nepa_trigger_evidence_text") or "")),
        ("project_metadata", project_metadata),
        ("doc_title", doc_title),
        ("document_text", doc_text),
    ]
    return _extract_funding_detail_from_sources(str(project_row["project_id"]), source_texts, run_at)


def _fetch_funding_preferred_document_texts(
    funding_projects: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    doc_text_by_project: dict[str, str] = {}
    doc_title_by_project: dict[str, str] = {}
    page_count_by_project: dict[str, int] = {}

    if funding_projects.empty:
        return doc_text_by_project, doc_title_by_project, page_count_by_project

    for source, group in funding_projects.groupby("dataset_source"):
        source_lower = str(source).lower()
        docs_path = DOCS_PATH_MAP.get(str(source).upper())
        pages_path = PAGES_PATH_MAP.get(str(source).upper())
        if docs_path is None or pages_path is None or not docs_path.exists() or not pages_path.exists():
            continue

        project_ids = group["project_id"].tolist()
        docs = conn.execute(f"""
            SELECT
                project_id.value AS project_id,
                document_id,
                document_title,
                file_name,
                main_document,
                length(coalesce(document_title, '')) AS title_len
            FROM read_parquet('{docs_path}')
            WHERE project_id.value IN ({_safe_sql_list(project_ids)})
            QUALIFY row_number() OVER (
                PARTITION BY project_id.value
                ORDER BY upper(coalesce(main_document, '')) = 'YES' DESC,
                         title_len DESC,
                         document_id
            ) = 1
        """).fetchdf()
        if docs.empty:
            continue

        for _, doc_row in docs.iterrows():
            doc_title_by_project[str(doc_row["project_id"])] = str(doc_row.get("document_title") or doc_row.get("file_name") or "")

        pages = conn.execute(f"""
            SELECT document_id, page_number, page_text
            FROM read_parquet('{pages_path}')
            WHERE document_id IN ({_safe_sql_list(docs['document_id'].tolist())})
              AND page_text IS NOT NULL
        """).fetchdf()
        if pages.empty:
            continue

        pages["page_sort"] = pages["page_number"].map(_page_sort_key)
        pages = pages.merge(docs[["project_id", "document_id"]], on="document_id", how="left")
        pages = pages.sort_values(["project_id", "page_sort", "page_number"])

        for pid, page_group in pages.groupby("project_id"):
            pid_str = str(pid)
            page_text = "\n\n".join(page_group["page_text"].fillna("").astype(str).tolist())
            doc_text_by_project[pid_str] = page_text
            page_count_by_project[pid_str] = len(page_group)

        log.info(
            "Funding details: scanned preferred %s documents for %s %s projects",
            source_lower.upper(),
            f"{len(docs):,}",
            source_lower.upper(),
        )

    return doc_text_by_project, doc_title_by_project, page_count_by_project


def _validate_funding_details(details: pd.DataFrame, funding_project_ids: set[str]) -> None:
    assert set(details["project_id"]) == funding_project_ids, (
        "Funding sidecar project set must exactly match federal_funding primary projects"
    )
    assert details["project_id"].is_unique, "Duplicate project_ids in funding sidecar"

    for col in [
        "federal_funding_amount_usd",
        "federal_funding_total_project_cost_usd",
        "federal_funding_recipient_cost_share_usd",
    ]:
        non_null = details[col].dropna()
        assert (non_null >= 0).all(), f"{col} contains negative values"

    pct = details["federal_funding_share_pct"].dropna()
    assert ((pct >= 0) & (pct <= 100)).all(), "Funding percentage must be between 0 and 100"

    assert details["federal_funding_type_multi"].apply(isinstance, args=(list,)).all(), (
        "federal_funding_type_multi must be list type"
    )
    assert details["federal_funding_program_multi"].apply(isinstance, args=(list,)).all(), (
        "federal_funding_program_multi must be list type"
    )


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _run_funding_detail_smoke_tests() -> None:
    run_at = "smoke-test"
    cases = [
        (
            "doe_funding",
            "DOE Funding: $726,199 Cost Share: $138,114",
            "cost_share",
            726199.0,
        ),
        (
            "loan_guarantee",
            "FINAL ENVIRONMENTAL ASSESSMENT FOR DEPARTMENT OF ENERGY LOAN GUARANTEE TO MOJAVE SOLAR, LLC",
            "loan_guarantee",
            None,
        ),
        (
            "eecbg_formula",
            "ARRA appropriates funding for DOE to issue formula-based grants under EECBG for this project.",
            "formula_grant",
            None,
        ),
        (
            "row_grant_negative",
            "BPA proposes to acquire a perpetual right-of-way grant across BLM-managed land.",
            "unknown_funding",
            None,
        ),
    ]

    for label, text, expected_type, expected_amount in cases:
        row = _extract_funding_detail_from_sources(label, [("project_metadata", text)], run_at)
        assert row["federal_funding_type_primary"] == expected_type, (
            f"Funding smoke test failed for {label}: expected {expected_type}, got {row['federal_funding_type_primary']}"
        )
        if expected_amount is not None:
            assert row["federal_funding_amount_usd"] == expected_amount, (
                f"Funding amount smoke test failed for {label}: expected {expected_amount}, got {row['federal_funding_amount_usd']}"
            )


def extract_funding_details(
    conn: duckdb.DuckDBPyConnection,
    triggers: pd.DataFrame,
    run_at: str,
) -> pd.DataFrame:
    _run_funding_detail_smoke_tests()

    funding_ids = (
        triggers.loc[triggers["nepa_trigger_primary"] == "federal_funding", ["project_id", "nepa_trigger_evidence_text"]]
        .drop_duplicates("project_id")
        .copy()
    )
    funding_project_ids = set(funding_ids["project_id"].astype(str))
    if funding_ids.empty:
        return pd.DataFrame(columns=FUNDING_DETAIL_COLS)

    conn.register("_funding_trigger_ids", funding_ids)
    funding_projects = conn.execute(f"""
        SELECT
            ids.project_id,
            p.dataset_source,
            p.process_type,
            p.project_title,
            p.project_description,
            p.project_sponsor,
            ids.nepa_trigger_evidence_text
        FROM _funding_trigger_ids ids
        JOIN read_parquet('{PROJECTS_PATH}') p USING (project_id)
        WHERE {CLEAN_ENERGY_FILTER}
    """).fetchdf()
    try:
        conn.unregister("_funding_trigger_ids")
    except Exception:
        pass

    assert set(funding_projects["project_id"].astype(str)) == funding_project_ids, (
        "Funding detail extraction can only run on clean projects already classified as federal_funding"
    )

    doc_text_by_project, doc_title_by_project, _page_count_by_project = _fetch_funding_preferred_document_texts(
        funding_projects,
        conn,
    )

    rows = []
    for _, project_row in funding_projects.iterrows():
        pid = str(project_row["project_id"])
        rows.append(_make_funding_detail_row(
            project_row=project_row,
            doc_text=doc_text_by_project.get(pid, ""),
            doc_title=doc_title_by_project.get(pid, ""),
            run_at=run_at,
        ))

    details = pd.DataFrame(rows)
    details = details[FUNDING_DETAIL_COLS]
    _validate_funding_details(details, funding_project_ids)
    return details


def write_funding_details_sidecar(
    conn: duckdb.DuckDBPyConnection,
    triggers: pd.DataFrame,
    run_at: str,
) -> pd.DataFrame:
    details = extract_funding_details(conn, triggers, run_at)
    _write_parquet_atomic(details, PROJECTS_FUNDING_DETAILS_PATH)
    log.info(
        "Written: %s (%s funding-primary rows)",
        PROJECTS_FUNDING_DETAILS_PATH,
        f"{len(details):,}",
    )
    if not details.empty:
        log.info(
            "Funding amount coverage: %s rows with federal amount (%.1f%%)",
            f"{details['federal_funding_amount_usd'].notna().sum():,}",
            100 * details["federal_funding_amount_usd"].notna().mean(),
        )
    return details


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
    parser.add_argument("--funding-details-only", action="store_true", help="Regenerate projects_funding_details.parquet from the existing trigger output; do not rerun trigger classification")
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
    if args.funding_details_only:
        if args.sample is not None:
            log.warning("--sample is ignored with --funding-details-only; sidecar validation requires the full funding-primary set")
        if not PROJECTS_NEPA_TRIGGER_PATH.exists():
            raise SystemExit(f"Missing trigger output: {PROJECTS_NEPA_TRIGGER_PATH}. Run trigger extraction first.")
        trigger_stat_before = PROJECTS_NEPA_TRIGGER_PATH.stat()
        triggers = pd.read_parquet(PROJECTS_NEPA_TRIGGER_PATH)
        write_funding_details_sidecar(conn, triggers, run_at)
        trigger_stat_after = PROJECTS_NEPA_TRIGGER_PATH.stat()
        assert (
            trigger_stat_before.st_mtime_ns == trigger_stat_after.st_mtime_ns
            and trigger_stat_before.st_size == trigger_stat_after.st_size
        ), "--funding-details-only must not rewrite projects_nepa_trigger.parquet"
        return

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

    def _sorted_multi(classes: list[str]) -> list[str]:
        ranked = {cls: TRIGGER_HIERARCHY.index(cls) if cls in TRIGGER_HIERARCHY else 99 for cls in classes}
        return sorted(classes, key=ranked.__getitem__)

    final["nepa_trigger_count"] = final["nepa_trigger_multi"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    final["nepa_trigger_combo"] = final["nepa_trigger_multi"].apply(
        lambda x: "|".join(_sorted_multi(x)) if isinstance(x, list) and x else ""
    )
    final["nepa_trigger_primary_hierarchy"] = final["nepa_trigger_multi"].apply(
        lambda x: _hierarchy_primary(x) if isinstance(x, list) else "unknown"
    )

    final["nepa_trigger_extraction_run_at"] = run_at
    final["nepa_trigger_llm_run_at"] = final.get("nepa_trigger_llm_run_at", "").fillna("")

    assert final["project_id"].is_unique, "Duplicate project_ids in output — check tier logic"
    assert final["project_id"].isin(set(projects["project_id"])).all(), "Non-clean project IDs in output"
    assert final["nepa_trigger_secondary"].apply(isinstance, args=(list,)).all(), "nepa_trigger_secondary must be list type"

    final = final[OUTPUT_COLS]

    out_path = PROJECTS_NEPA_TRIGGER_PATH
    final.to_parquet(out_path, index=False)
    log.info("Written: %s (%s rows)", out_path, f"{len(final):,}")
    write_funding_details_sidecar(conn, final, run_at)

    batches = build_validation_batches(final, projects)
    if not batches.empty:
        batch_path = VALIDATION_DIR / "validation_batches.csv"
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
    print("\n=== Hierarchy-resolved primary distribution ===")
    print(final["nepa_trigger_primary_hierarchy"].value_counts().to_string())
    multi_mask = final["nepa_trigger_count"] > 1
    print(f"\nMulti-class projects (2+ triggers): {multi_mask.sum():,}")
    if multi_mask.any():
        print(final.loc[multi_mask, "nepa_trigger_combo"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
