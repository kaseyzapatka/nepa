import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

# --------------------------
# DELIVERABLE 1: NEPA TRIGGERED — Trigger Classification
# --------------------------
# Five-tier classification of what triggered NEPA review per 20,725 clean energy projects:
#   Tier 1a — Agency metadata heuristics (pure-signal agencies; verb disambiguation for land agencies)
#   Tier 1b — Title and description keyword matching (all 7 classes; specificity-ranked)
#   Tier 2  — Document title scan (documents.parquet; no page text access)
#   Tier 3  — Purpose and Need section extraction (pages.parquet, pages 1-10; CE: full doc)
#   Tier 4  — Embedding similarity (zero-shot, all-MiniLM-L6-v2)
#   Tier 5  — Claude Haiku LLM fallback (--use-llm flag only)
#
# [SELF-CONTAINED] — requires only projects_combined.parquet and CE/EA/EIS docs/pages.
#
# Usage:
#   python 01_extract_nepa_trigger.py --eda              # EDA check only; no extraction
#   python 01_extract_nepa_trigger.py --sample 50        # test on 50 projects
#   python 01_extract_nepa_trigger.py                    # full run (~20,725 projects)
#   python 01_extract_nepa_trigger.py --use-llm          # full run + Haiku on low-confidence
#
# Output:
#   data/analysis/nepa_trigger/projects_nepa_trigger.parquet  (one row per project)
#   data/analysis/nepa_trigger/validation_batches.csv          (flagged cases grouped by rule)

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

# Tier 3 reuses the Tier 1b pattern list (patterns are class-agnostic to input source)
TIER3_PATTERNS = TIER1B_PATTERNS

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
Given the text below, identify the primary federal nexus.

Classes:
- federal_action: federal agency is the primary actor constructing or implementing the project
- federal_program: programmatic EIS, land-use plan, rulemaking, or leasing framework
- federal_property_transaction: land exchange, disposal, or conveyance
- federal_land: project on or crossing federal land; ROW grant to a private developer
- federal_permit: federal permit, license, or authorization is the primary nexus
- federal_funding: federal grant, loan guarantee, or financial assistance
- unknown: cannot determine from the text provided

Text: {text}

Respond with JSON only:
{{"primary": "federal_land", "secondary": ["federal_permit"], "confidence": "high", "reasoning": "..."}}"""

VALID_CLASSES = frozenset({
    "federal_action", "federal_program", "federal_property_transaction",
    "federal_land", "federal_permit", "federal_funding", "unknown",
})

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


def extract_sentence(text: str, match: re.Match) -> str:
    """Extract the full sentence containing the regex match position."""
    start, end = match.start(), match.end()
    region_before = text[max(0, start - 500): start]
    sent_start = max(0, start - 500)
    for sep in ('. ', '.\n', '?\n', '!\n', '\n\n'):
        idx = region_before.rfind(sep)
        if idx >= 0:
            sent_start = max(0, start - 500) + idx + len(sep)
            break
    region_after = text[end: min(len(text), end + 500)]
    sent_end = min(len(text), end + 500)
    for sep in ('. ', '.\n', '?\n', '!\n', '\n\n'):
        idx = region_after.find(sep)
        if idx >= 0:
            sent_end = end + idx + 1
            break
    return text[sent_start:sent_end].strip()


def extract_purpose_and_need(text: str, window: int = PAN_WINDOW) -> str:
    """Return up to `window` chars following the first Purpose and Need header found."""
    for pat in [
        r'\bpurpose\s+and\s+need\b',
        r'\bneed\s+for\s+(?:federal\s+)?action\b',
        r'\bproposed\s+(?:federal\s+)?action\b',
        r'\bproject\s+purpose\b',
        r'\bstatement\s+of\s+need\b',
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
) -> Optional[dict]:
    """
    Apply a list of (pattern, trigger_class, rule_slug, confidence) tuples.
    Returns the first match as a result dict, or None.
    """
    for pat, trigger_class, rule_slug, confidence in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            if trigger_class == "federal_program" and _is_programmatic_exclusion(text):
                continue
            return {
                "project_id": project_id,
                "nepa_trigger_primary": trigger_class,
                "nepa_trigger_secondary": [],
                "nepa_trigger_multi": [trigger_class],
                "nepa_trigger_evidence_text": extract_sentence(text, m),
                "nepa_trigger_evidence_source": evidence_source,
                "nepa_trigger_confidence": confidence,
                "nepa_trigger_rule_id": f"{tier_prefix}_{rule_slug}",
                "nepa_trigger_manual_review": confidence == "low",
                "nepa_trigger_llm_run_at": "",
            }
    return None


def _safe_pid_list(pids) -> str:
    """Build a SQL-safe IN-list string from a collection of project_ids."""
    return ", ".join(f"'{str(p).replace(chr(39), '')}'" for p in pids)


def _make_unknown(project_id: str) -> dict:
    return {
        "project_id": project_id,
        "nepa_trigger_primary": "unknown",
        "nepa_trigger_secondary": [],
        "nepa_trigger_multi": [],
        "nepa_trigger_evidence_text": "",
        "nepa_trigger_evidence_source": "",
        "nepa_trigger_confidence": "low",
        "nepa_trigger_rule_id": "no_match",
        "nepa_trigger_manual_review": True,
        "nepa_trigger_llm_run_at": "",
    }


# --------------------------
# TIER FUNCTIONS
# --------------------------

def tier1a_metadata(projects: pd.DataFrame) -> list[dict]:
    """
    Agency metadata heuristics.
    - Pure-signal agencies (FERC, FAA, DOT, HUD, etc.) → direct assignment, confidence=high
    - Land agencies (BLM, USFS, NPS, BOR, FWS) → verb disambiguation:
        action verbs → federal_action (high); authorizer verbs → federal_land (high);
        no verb signal → federal_land (medium)
    - DOE, USACE: not assigned here; fall through to Tier 1b
    """
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
            results.append({
                "project_id": pid,
                "nepa_trigger_primary": "federal_permit",
                "nepa_trigger_secondary": [],
                "nepa_trigger_multi": ["federal_permit"],
                "nepa_trigger_evidence_text": agency,
                "nepa_trigger_evidence_source": "agency_metadata",
                "nepa_trigger_confidence": "high",
                "nepa_trigger_rule_id": f"T1a_{agency_code}_permit",
                "nepa_trigger_manual_review": False,
                "nepa_trigger_llm_run_at": "",
            })
        elif _agency_matches(agency, AGENCY_FUNDING_MAP):
            results.append({
                "project_id": pid,
                "nepa_trigger_primary": "federal_funding",
                "nepa_trigger_secondary": [],
                "nepa_trigger_multi": ["federal_funding"],
                "nepa_trigger_evidence_text": agency,
                "nepa_trigger_evidence_source": "agency_metadata",
                "nepa_trigger_confidence": "high",
                "nepa_trigger_rule_id": f"T1a_{agency_code}_funding",
                "nepa_trigger_manual_review": False,
                "nepa_trigger_llm_run_at": "",
            })
        elif _agency_matches(agency, AGENCY_LAND_MAP):
            verb_class = _verb_class(text)
            trigger = verb_class if verb_class else "federal_land"
            confidence = "high" if verb_class else "medium"
            verb_suffix = "action" if trigger == "federal_action" else "land"
            results.append({
                "project_id": pid,
                "nepa_trigger_primary": trigger,
                "nepa_trigger_secondary": [],
                "nepa_trigger_multi": [trigger],
                "nepa_trigger_evidence_text": agency,
                "nepa_trigger_evidence_source": "agency_metadata",
                "nepa_trigger_confidence": confidence,
                "nepa_trigger_rule_id": f"T1a_{agency_code}_{verb_suffix}",
                "nepa_trigger_manual_review": False,
                "nepa_trigger_llm_run_at": "",
            })
        # DOE and USACE: intentionally not assigned here
    return results


def tier1b_title_description(projects: pd.DataFrame) -> list[dict]:
    """
    Keyword matching on project_title + project_description.
    Applies TIER1B_PATTERNS in specificity order (most specific first).
    Programmatic detection uses title pattern + exclusion check + strong confirmation.
    """
    results = []
    for _, row in projects.iterrows():
        pid = row["project_id"]
        text = " ".join([
            str(row.get("project_title") or ""),
            str(row.get("project_description") or ""),
        ])
        if not text.strip():
            continue

        # Programmatic check (high specificity — test before generic patterns)
        if _is_programmatic_title(text) and not _is_programmatic_exclusion(text):
            if _is_programmatic_strong(text):
                results.append({
                    "project_id": pid,
                    "nepa_trigger_primary": "federal_program",
                    "nepa_trigger_secondary": [],
                    "nepa_trigger_multi": ["federal_program"],
                    "nepa_trigger_evidence_text": text[:300].strip(),
                    "nepa_trigger_evidence_source": "title",
                    "nepa_trigger_confidence": "high",
                    "nepa_trigger_rule_id": "T1b_programmatic_title",
                    "nepa_trigger_manual_review": False,
                    "nepa_trigger_llm_run_at": "",
                })
                continue

        result = _apply_pattern_list(pid, text, TIER1B_PATTERNS, "description", "T1b")
        if result:
            results.append(result)
    return results


def tier2_doc_title(
    unresolved_ids: list,
    projects_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """
    Scan NEPA document titles for unambiguous trigger phrases.
    Routes each project to its source documents.parquet via dataset_source.
    Prefers main_document=YES rows; falls back to all docs if none flagged.
    """
    if not unresolved_ids:
        return []

    unresolved_df = projects_df[projects_df["project_id"].isin(set(unresolved_ids))]
    results: dict[str, dict] = {}

    for source, group in unresolved_df.groupby("dataset_source"):
        source_upper = source.upper()
        docs_path = DOCS_PATH_MAP.get(source_upper)
        if docs_path is None or not docs_path.exists():
            log.warning(f"documents.parquet not found for {source_upper}; skipping Tier 2 for this source")
            continue

        pids = list(group["project_id"])
        docs = conn.execute(f"""
            SELECT project_id, document_title, main_document
            FROM read_parquet('{docs_path}')
            WHERE project_id IN ({_safe_pid_list(pids)})
              AND document_title IS NOT NULL
        """).fetchdf()

        if docs.empty:
            continue

        # Prefer main documents
        preferred = []
        for pid, grp in docs.groupby("project_id"):
            main = grp[grp["main_document"].fillna("").str.upper() == "YES"]
            preferred.append(main if not main.empty else grp)
        docs = pd.concat(preferred, ignore_index=True)

        for _, row in docs.iterrows():
            pid = row["project_id"]
            if pid in results:
                continue
            title = str(row["document_title"] or "")

            # Programmatic check on title
            if _is_programmatic_title(title) and not _is_programmatic_exclusion(title):
                results[pid] = {
                    "project_id": pid,
                    "nepa_trigger_primary": "federal_program",
                    "nepa_trigger_secondary": [],
                    "nepa_trigger_multi": ["federal_program"],
                    "nepa_trigger_evidence_text": title,
                    "nepa_trigger_evidence_source": "doc_title",
                    "nepa_trigger_confidence": "high",
                    "nepa_trigger_rule_id": "T2_doc_title_peis",
                    "nepa_trigger_manual_review": False,
                    "nepa_trigger_llm_run_at": "",
                }
                continue

            for pat, trigger_class, rule_slug in DOC_TITLE_PATTERNS:
                m = re.search(pat, title, re.IGNORECASE)
                if m:
                    results[pid] = {
                        "project_id": pid,
                        "nepa_trigger_primary": trigger_class,
                        "nepa_trigger_secondary": [],
                        "nepa_trigger_multi": [trigger_class],
                        "nepa_trigger_evidence_text": title,
                        "nepa_trigger_evidence_source": "doc_title",
                        "nepa_trigger_confidence": "high",
                        "nepa_trigger_rule_id": f"T2_doc_title_{rule_slug}",
                        "nepa_trigger_manual_review": False,
                        "nepa_trigger_llm_run_at": "",
                    }
                    break

    return list(results.values())


def tier3_purpose_and_need(
    unresolved_ids: list,
    projects_df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
) -> list[dict]:
    """
    Extract Purpose and Need section from pages.parquet.
    EA/EIS: scan pages 1-10 for P&N section header; extract 600-char window.
    CE: scan full document (no page limit; CEs are typically 1-3 pages).
    Apply TIER3_PATTERNS to extracted text.
    """
    if not unresolved_ids:
        return []

    unresolved_df = projects_df[projects_df["project_id"].isin(set(unresolved_ids))]
    results: dict[str, dict] = {}

    for source, group in unresolved_df.groupby("dataset_source"):
        source_upper = source.upper()
        pages_path = PAGES_PATH_MAP.get(source_upper)
        if pages_path is None or not pages_path.exists():
            log.warning(f"pages.parquet not found for {source_upper}; skipping Tier 3 for this source")
            continue

        pids = list(group["project_id"])
        is_ce = (source_upper == "CE")
        page_filter = "" if is_ce else f"AND page_num <= {MAX_PAGES_PAN}"

        page_rows = conn.execute(f"""
            SELECT project_id, string_agg(page_text, ' ' ORDER BY page_num) AS combined_text
            FROM read_parquet('{pages_path}')
            WHERE project_id IN ({_safe_pid_list(pids)})
              {page_filter}
            GROUP BY project_id
        """).fetchdf()

        for _, row in page_rows.iterrows():
            pid = row["project_id"]
            if pid in results:
                continue
            full_text = str(row["combined_text"] or "")
            if not full_text.strip():
                continue

            if is_ce:
                scan_text = full_text
                evidence_source = "document_text"
            else:
                pan_text = extract_purpose_and_need(full_text)
                if not pan_text:
                    continue  # No P&N section found; fall through to Tier 4
                scan_text = pan_text
                evidence_source = "purpose_and_need"

            result = _apply_pattern_list(pid, scan_text, TIER3_PATTERNS, evidence_source, "T3")
            if result:
                results[pid] = result

    return list(results.values())


def tier4_embedding(
    unresolved_ids: list,
    projects_df: pd.DataFrame,
) -> list[dict]:
    """
    Zero-shot embedding similarity using all-MiniLM-L6-v2.
    Candidate text: project_title + project_description.
    Assigns to class with highest cosine similarity ≥ EMBEDDING_THRESHOLD (medium confidence).
    Falls below threshold → unknown (low confidence, manual_review=True).
    """
    if not unresolved_ids:
        return []

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        log.warning(
            "sentence-transformers not installed; skipping Tier 4. "
            "Install with: pip install sentence-transformers"
        )
        return []

    log.info(f"  Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute class centroids from prototype sentences
    centroids = {}
    for cls, sentences in CLASS_PROTOTYPES.items():
        embeddings = model.encode(sentences, normalize_embeddings=True)
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
            results.append(_make_unknown(pid))
            continue

        emb = model.encode([candidate_text], normalize_embeddings=True)[0]
        # Normalized embeddings: dot product == cosine similarity
        sims = {cls: float(np.dot(emb, centroid)) for cls, centroid in centroids.items()}
        best_cls = max(sims, key=sims.get)
        best_score = sims[best_cls]

        if best_score >= EMBEDDING_THRESHOLD:
            results.append({
                "project_id": pid,
                "nepa_trigger_primary": best_cls,
                "nepa_trigger_secondary": [],
                "nepa_trigger_multi": [best_cls],
                "nepa_trigger_evidence_text": candidate_text[:300],
                "nepa_trigger_evidence_source": "embedding",
                "nepa_trigger_confidence": "medium",
                "nepa_trigger_rule_id": f"T4_embed_{best_cls}",
                "nepa_trigger_manual_review": False,
                "nepa_trigger_llm_run_at": "",
            })
        else:
            r = _make_unknown(pid)
            r["nepa_trigger_rule_id"] = "T4_embed_below_threshold"
            results.append(r)

    return results


def tier5_llm(
    low_conf_ids: list,
    projects_df: pd.DataFrame,
) -> list[dict]:
    """
    Claude Haiku structured fallback for confidence=low residuals.
    Only called when --use-llm is set. Uses structured JSON prompt.
    Sets nepa_trigger_llm_run_at per-row on success.
    """
    if not low_conf_ids:
        return []

    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed; skipping Tier 5")
        return []

    client = anthropic.Anthropic()
    unresolved_df = projects_df[projects_df["project_id"].isin(set(low_conf_ids))]
    results = []

    for _, row in unresolved_df.iterrows():
        pid = row["project_id"]
        candidate_text = " ".join([
            str(row.get("project_title") or ""),
            str(row.get("project_description") or ""),
        ]).strip()[:1000]

        if not candidate_text:
            continue

        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": LLM_PROMPT.format(text=candidate_text)}],
            )
            raw = response.content[0].text.strip()
            parsed = json.loads(raw)
            primary = parsed.get("primary", "unknown")
            if primary not in VALID_CLASSES:
                primary = "unknown"
            secondary = [
                s for s in parsed.get("secondary", [])
                if s in VALID_CLASSES and s != primary
            ]
            confidence = parsed.get("confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            results.append({
                "project_id": pid,
                "nepa_trigger_primary": primary,
                "nepa_trigger_secondary": secondary,
                "nepa_trigger_multi": [primary] + secondary,
                "nepa_trigger_evidence_text": candidate_text[:300],
                "nepa_trigger_evidence_source": "llm",
                "nepa_trigger_confidence": confidence,
                "nepa_trigger_rule_id": "T5_llm",
                "nepa_trigger_manual_review": confidence == "low",
                "nepa_trigger_llm_run_at": datetime.now(timezone.utc).isoformat(),
            })
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Tier 5 parse error for {pid}: {e}")
            results.append(_make_unknown(pid))
        except Exception as e:
            log.warning(f"Tier 5 API error for {pid}: {e}")
            results.append(_make_unknown(pid))

    return results


# --------------------------
# VALIDATION
# --------------------------

def build_validation_batches(df: pd.DataFrame, sample_per_batch: int = 20) -> pd.DataFrame:
    """
    Group flagged cases by nepa_trigger_rule_id.
    Sample up to 20 per rule for efficient batch review.
    Sort batches by batch_size descending — fix largest broken rule first.
    """
    flagged = df[df["nepa_trigger_manual_review"]].copy()
    if flagged.empty:
        return pd.DataFrame()
    batches = []
    for rule_id, group in flagged.groupby("nepa_trigger_rule_id"):
        sample = group.sample(min(sample_per_batch, len(group)), random_state=42).copy()
        sample["validation_batch"] = rule_id
        sample["batch_size"] = len(group)
        batches.append(sample)
    return pd.concat(batches, ignore_index=True).sort_values("batch_size", ascending=False)


# --------------------------
# EDA
# --------------------------

def run_eda(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Print description coverage and length by process type.
    Run once before full pipeline to gauge Tier 1b effectiveness per source.
    """
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


# --------------------------
# ORCHESTRATOR
# --------------------------

def extract_nepa_triggers(
    conn: duckdb.DuckDBPyConnection,
    use_llm: bool = False,
    sample: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full five-tier classification pipeline.
    Returns (final_df, projects_df). projects_df is used downstream for scope assertion.

    A project is 'resolved' when confidence is high or medium. Each tier receives only
    the projects not yet resolved by all prior tiers. Tier 4 takes all remaining unresolved
    (including low-confidence from prior tiers). Tier 5 re-processes confidence=low residuals.
    """
    projects = conn.execute(f"""
        SELECT project_id, lead_agency_harmonized, project_title,
               project_description, process_type, dataset_source
        FROM read_parquet('{PROJECTS_PATH}')
        WHERE {CLEAN_ENERGY_FILTER}
    """).fetchdf()

    if sample:
        projects = projects.sample(sample, random_state=42)
        log.info(f"Sample mode: {len(projects)} projects")

    all_project_ids = set(projects["project_id"])
    log.info(f"Processing {len(all_project_ids):,} clean energy projects")

    resolved: dict[str, dict] = {}

    def _update(results: list[dict]) -> None:
        for r in results:
            if r["nepa_trigger_confidence"] in ("high", "medium"):
                resolved[r["project_id"]] = r

    def _remaining() -> list:
        return list(all_project_ids - set(resolved))

    def _pct() -> str:
        return f"{len(resolved)/len(all_project_ids):.1%}"

    # Tier 1a
    log.info("Tier 1a: agency metadata")
    _update(tier1a_metadata(projects))
    log.info(f"  → {len(resolved):,} resolved ({_pct()})")

    # Tier 1b
    log.info("Tier 1b: title and description keywords")
    unresolved_df = projects[~projects["project_id"].isin(resolved)]
    _update(tier1b_title_description(unresolved_df))
    log.info(f"  → {len(resolved):,} resolved ({_pct()})")

    # Tier 2
    log.info("Tier 2: document title scan")
    _update(tier2_doc_title(_remaining(), projects, conn))
    log.info(f"  → {len(resolved):,} resolved ({_pct()})")

    # Tier 3
    log.info("Tier 3: Purpose and Need section extraction")
    _update(tier3_purpose_and_need(_remaining(), projects, conn))
    log.info(f"  → {len(resolved):,} resolved ({_pct()})")

    # Tier 4 — takes all remaining (confidence=medium or low)
    log.info("Tier 4: embedding similarity (zero-shot)")
    for r in tier4_embedding(_remaining(), projects):
        resolved[r["project_id"]] = r
    log.info(f"  → {len(resolved):,} resolved ({_pct()})")

    # Tier 5 — re-process confidence=low residuals only
    if use_llm:
        low_conf_ids = [pid for pid, r in resolved.items() if r["nepa_trigger_confidence"] == "low"]
        log.info(f"Tier 5: Claude Haiku on {len(low_conf_ids):,} low-confidence projects")
        for r in tier5_llm(low_conf_ids, projects):
            resolved[r["project_id"]] = r
    else:
        log.info("Tier 5: skipped (--use-llm not set)")

    # Fill any remaining gaps
    for pid in all_project_ids:
        if pid not in resolved:
            resolved[pid] = _make_unknown(pid)

    final = pd.DataFrame(list(resolved.values()))
    return final, projects


# --------------------------
# MAIN
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1: NEPA Trigger Classification — 20,725 clean energy projects"
    )
    parser.add_argument("--eda", action="store_true",
                        help="Run EDA check only; do not extract")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable Tier 5 Claude Haiku fallback for low-confidence cases")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N projects (random sample; for testing)")
    args = parser.parse_args()

    conn = duckdb.connect()

    if args.eda:
        run_eda(conn)
        return

    run_at = datetime.now(timezone.utc).isoformat()
    final, projects = extract_nepa_triggers(conn, use_llm=args.use_llm, sample=args.sample)

    # Compute is_dual_nexus: federal_land primary + federal_permit secondary
    final["is_dual_nexus"] = (
        (final["nepa_trigger_primary"] == "federal_land") &
        (final["nepa_trigger_secondary"].apply(
            lambda x: "federal_permit" in x if isinstance(x, list) else False
        ))
    )

    # Timestamps
    final["nepa_trigger_extraction_run_at"] = run_at
    if "nepa_trigger_llm_run_at" not in final.columns:
        final["nepa_trigger_llm_run_at"] = ""
    else:
        final["nepa_trigger_llm_run_at"] = final["nepa_trigger_llm_run_at"].fillna("")

    # Assertions before writing
    assert final["project_id"].is_unique, \
        "Duplicate project_ids in output — check tier logic"
    assert final["project_id"].isin(set(projects["project_id"])).all(), \
        "Non-clean project IDs in output"
    assert final["nepa_trigger_secondary"].apply(isinstance, args=(list,)).all(), \
        "nepa_trigger_secondary must be list type"

    # Enforce column order
    final = final[OUTPUT_COLS]

    # Write parquet
    out_path = OUTPUT_DIR / "projects_nepa_trigger.parquet"
    final.to_parquet(out_path, index=False)
    log.info(f"Written: {out_path} ({len(final):,} rows)")

    # Write validation batches
    batches = build_validation_batches(final)
    if not batches.empty:
        batch_path = OUTPUT_DIR / "validation_batches.csv"
        batches.to_csv(batch_path, index=False)
        flag_rate = final["nepa_trigger_manual_review"].mean()
        log.info(
            f"Validation batches: {len(batches):,} flagged cases ({flag_rate:.1%} flag rate) "
            f"across {batches['validation_batch'].nunique()} rules → {batch_path}"
        )
        if flag_rate > 0.05:
            log.warning(
                f"Flag rate {flag_rate:.1%} exceeds 5% target. "
                "Tighten confidence thresholds or fix low-precision rules before finalizing."
            )
    else:
        log.info("No cases flagged for manual review.")

    # Summary to stdout
    print("\n=== Primary trigger distribution ===")
    print(final["nepa_trigger_primary"].value_counts().to_string())
    print("\n=== Confidence distribution ===")
    print(final["nepa_trigger_confidence"].value_counts().to_string())
    print("\n=== Evidence source distribution ===")
    print(final["nepa_trigger_evidence_source"].value_counts().to_string())
    print(f"\nDual-nexus projects (federal_land + federal_permit): {final['is_dual_nexus'].sum():,}")


if __name__ == "__main__":
    main()
