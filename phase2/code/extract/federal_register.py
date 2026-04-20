"""
Federal Register NOI enrichment for NEPATEC projects.

Phase 2 treats this as a refreshable, standalone data source. The default
extract_data.py path remains offline and only merges an existing
federal_register.parquet artifact unless a refresh is explicitly requested.
"""

from __future__ import annotations

import ast
import calendar
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
FEDERAL_REGISTER_DIR = ANALYSIS_DIR / "federal_register"

FR_ENDPOINT = "https://www.federalregister.gov/api/v1/documents.json"

DEFAULT_NOI_CORPUS_OUTPUT = FEDERAL_REGISTER_DIR / "noi_corups.parquet"
DEFAULT_NOI_CANDIDATES_OUTPUT = FEDERAL_REGISTER_DIR / "noi_candidates.parquet"
DEFAULT_NOA_CORPUS_OUTPUT = FEDERAL_REGISTER_DIR / "noa_corpus.parquet"
DEFAULT_NOA_CANDIDATES_OUTPUT = FEDERAL_REGISTER_DIR / "noa_candidates.parquet"
DEFAULT_PROJECT_OUTPUT = FEDERAL_REGISTER_DIR / "federal_register.parquet"
DEFAULT_CACHE_PATH = FEDERAL_REGISTER_DIR / "fr_noi_cache.json"
DEFAULT_FETCH_REPORT_OUTPUT = FEDERAL_REGISTER_DIR / "fr_noi_fetch_report.csv"
DEFAULT_EVIDENCE_OUTPUT = FEDERAL_REGISTER_DIR / "nepatec_fr_evidence.parquet"
DEFAULT_AMBIGUOUS_CANDIDATES_OUTPUT = FEDERAL_REGISTER_DIR / "noi_manual_review_candidates.csv"
DEFAULT_LOW_OVERLAP_ACCEPTED_OUTPUT = FEDERAL_REGISTER_DIR / "noi_manual_review_accepted_low_title_overlap.csv"
DEFAULT_NOA_LOW_OVERLAP_ACCEPTED_OUTPUT = FEDERAL_REGISTER_DIR / "noa_manual_review_accepted_low_title_overlap.csv"
DEFAULT_NOA_REVIEW_OUTPUT = FEDERAL_REGISTER_DIR / "noa_manual_review_candidates.csv"

NOI_CORPUS_QUERIES = (
    '"Notice of Intent"',
    '"Intent To Prepare"',
    '"Notice To Prepare"',
    '"Notice of Preparation"',
    '"Notice of Scoping"',
    '"Notice of Public Scoping"',
    '"Scoping for Environmental Impact"',
)

FR_FIELDS = (
    "title",
    "publication_date",
    "document_number",
    "html_url",
    "pdf_url",
    "raw_text_url",
    "agency_names",
    "agencies",
    "type",
    "subtype",
    "comments_close_on",
    "abstract",
    "excerpts",
)

PROJECT_OUTPUT_COLUMNS = [
    "project_id",
    "project_title",
    "noi_publication_date",
    "noi_document_number",
    "noi_url",
    "noi_project_title",
    "noi_type",
    "noi_subtype",
    "noi_comments_close_on",
    "noi_scoping_meeting_dates",
    "noi_match_score",
    "noi_query",
    "noi_match_tier",
    "noi_match_confidence",
    "noi_match_status",
    "noi_candidate_count",
    "noi_high_confidence_candidate_count",
    "noi_title_overlap_count",
    "noi_title_overlap_tokens",
    "noi_agency_match",
    "noi_state_match",
    "noi_sponsor_match",
    "noi_process_match",
    "noi_process_conflict",
    "noi_match_reason",
    "noi_date_evidence_type",
    "noi_nepatec_evidence_document_id",
    "noi_nepatec_evidence_file_name",
    "noi_nepatec_evidence_page_number",
    # NOA (Notice of Availability) — end-of-process signal (FEIS/FONSI)
    "noa_availability_date",
    "noa_document_number",
    "noa_url",
    "noa_fr_title",
    "noa_match_status",
    "noa_match_reason",
    "noa_match_score",
    "noa_title_overlap_count",
    "noa_title_overlap_tokens",
    "noa_date_evidence_type",
    "noa_nepatec_evidence_document_id",
    "noa_nepatec_evidence_file_name",
    "noa_nepatec_evidence_page_number",
]

CANDIDATE_OUTPUT_COLUMNS = [
    "project_id",
    "project_title",
    "process_type",
    "project_energy_type",
    "fr_document_number",
    "fr_title",
    "fr_publication_date",
    "fr_url",
    "fr_type",
    "fr_subtype",
    "fr_comments_close_on",
    "fr_scoping_meeting_dates",
    "fr_query_terms",
    "candidate_rank",
    "match_score",
    "match_confidence",
    "match_reason",
    "title_overlap_count",
    "title_containment_ratio",
    "title_overlap_tokens",
    "exact_phrase_match",
    "agency_match",
    "state_match",
    "sponsor_match",
    "process_match",
    "process_conflict",
    "document_number_evidence",
    "fr_citation_evidence",
    "raw_text_scoping_date_count",
    "nepatec_fr_document_number_evidence",
]

CORPUS_OUTPUT_COLUMNS = [
    "fr_document_number",
    "fr_title",
    "fr_publication_date",
    "fr_url",
    "fr_pdf_url",
    "fr_raw_text_url",
    "fr_agency_names",
    "fr_agencies",
    "fr_type",
    "fr_subtype",
    "fr_comments_close_on",
    "fr_abstract",
    "fr_excerpts",
    "fr_scoping_meeting_dates",
    "fr_query_terms",
    "fr_query_count",
    "fr_fetch_run_at",
]

FETCH_REPORT_COLUMNS = [
    "fr_query_terms",
    "query_index",
    "query_total",
    "window_level",
    "window_start",
    "window_end",
    "count",
    "total_pages",
    "pages_fetched",
    "cached_pages",
    "network_pages",
    "records_returned",
    "unique_documents_returned",
    "documents_added",
    "capped",
    "split",
    "split_to",
    "fr_fetch_run_at",
]

EVIDENCE_OUTPUT_COLUMNS = [
    "project_id",
    "process_type",
    "project_title",
    "document_id",
    "file_name",
    "document_title",
    "document_type",
    "main_document",
    "page_number",
    "evidence_type",
    "fr_document_number",
    "fr_document_number_raw",
    "fr_url",
    "fr_citation",
    "fr_date_text",
    "fr_date_text_parsed",
    "notice_title_snippet",
    "evidence_context",
    "nearby_noi_phrase",
    "nearby_project_title_token_count",
    "evidence_rank",
]

FOCUSED_PROJECT_REVIEW_COLUMNS = [
    "project_id",
    "process_type",
    "project_energy_type",
    "project_department",
    "lead_agency",
    "project_state",
    "project_sponsor",
    "project_title",
    "noi_document_number",
    "noi_publication_date",
    "noi_url",
    "noi_project_title",
    "noi_match_score",
    "noi_match_reason",
    "noi_candidate_count",
    "noi_high_confidence_candidate_count",
    "noi_title_overlap_count",
    "noi_title_overlap_tokens",
    "noi_agency_match",
    "noi_state_match",
    "noi_sponsor_match",
    "noi_process_match",
    "noi_process_conflict",
    "noi_nepatec_evidence_file_name",
    "noi_nepatec_evidence_page_number",
]

FOCUSED_NOA_PROJECT_REVIEW_COLUMNS = [
    "project_id",
    "process_type",
    "project_energy_type",
    "project_department",
    "lead_agency",
    "project_state",
    "project_sponsor",
    "project_title",
    "noa_document_number",
    "noa_availability_date",
    "noa_url",
    "noa_fr_title",
    "noa_match_score",
    "noa_match_reason",
    "noa_title_overlap_count",
    "noa_title_overlap_tokens",
    "noa_nepatec_evidence_file_name",
    "noa_nepatec_evidence_page_number",
]

FOCUSED_CANDIDATE_REVIEW_COLUMNS = [
    "project_id",
    "process_type",
    "project_title",
    "project_energy_type",
    "fr_document_number",
    "fr_publication_date",
    "fr_url",
    "fr_title",
    "match_confidence",
    "match_score",
    "match_reason",
    "candidate_rank",
    "title_overlap_count",
    "title_overlap_tokens",
    "agency_match",
    "state_match",
    "sponsor_match",
    "process_match",
    "process_conflict",
    "nepatec_fr_document_number_evidence",
]

_TITLE_PREFIX_PATTERNS = (
    r"^(?:proposed\s+)?construction and operation of(?:\s+(?:a|an|the))?\s+",
    r"^department of energy loan guarantee for\s+",
    r"^license renewal of\s+",
    r"^subsequent license renewal of\s+",
    r"^early site permit at(?:\s+the)?\s+",
    r"^expansion and modernization of\s+",
    r"^bonneville power administration proposed\s+",
)

_SEARCH_NOISE_TOKENS = {
    "a",
    "an",
    "and",
    "assessment",
    "at",
    "construction",
    "department",
    "draft",
    "energy",
    "environmental",
    "facility",
    "facilities",
    "final",
    "for",
    "impact",
    "in",
    "intent",
    "license",
    "management",
    "meeting",
    "meetings",
    "notice",
    "of",
    "operation",
    "permit",
    "plan",
    "prepare",
    "program",
    "programmatic",
    "project",
    "proposed",
    "public",
    "resource",
    "scoping",
    "site",
    "sites",
    "statement",
    "subsequent",
    "the",
    "to",
}

_MATCH_STOPWORDS = _SEARCH_NOISE_TOKENS | {
    "area",
    "county",
    "line",
    "state",
    "use",
}

_EMPTY_HINT_PATTERNS = (
    r"^none\b",
    r"^unk$",
    r"^unknown$",
    r"sponsored by the lead agency",
)

_NOI_LIKE_RE = re.compile(
    r"\b(?:notice\s+of\s+intent|intent\s+to\s+prepare|notice\s+to\s+prepare"
    r"|notice\s+of\s+preparation|notice\s+of\s+scoping|notice\s+of\s+public\s+scoping)\b",
    re.IGNORECASE,
)
_REJECT_NOTICE_RE = re.compile(r"\b(?:termination|withdrawals?|cancel(?:lation|ed)?)\b", re.IGNORECASE)
_FR_CITATION_RE = re.compile(r"\b\d+\s*FR\s*\d+\b", re.IGNORECASE)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE_PATTERN = rf"(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}"

_FR_DOC_RE = re.compile(
    r'[\[\(]?FR Doc\.?\s+(\d{4}[\-\u2013\u2014]\d+)',
    re.IGNORECASE,
)

_FR_URL_RE = re.compile(
    r'federalregister\.gov/documents/\d{4}/\d{2}/\d{2}/([\d\-]+)',
    re.IGNORECASE,
)

_FR_DATE_TEXT_RE = re.compile(
    r'(?:published\s+in\s+the\s+Federal\s+Register\s+on'
    r'|Federal\s+Register,\s+Vol\.?\s*\d+,\s*No\.?\s*\d+,?)'
    r'[^.]{0,60}?((?:' + _MONTHS + r')\s+\d{1,2},\s+\d{4})',
    re.IGNORECASE,
)

_NOI_PROXIMITY_PHRASES = (
    "notice of intent",
    "intent to prepare",
    "notice to prepare",
    "notice of preparation",
    "notice of scoping",
    "notice of public scoping",
)

_CE_PROXIMITY_PHRASES = (
    "notice of application",
    "notice of proposed action",
    "categorical exclusion",
)

_NOA_PROXIMITY_PHRASES = (
    "final environmental impact statement",
    "final eis",
    "finding of no significant impact",
    "fonsi",
    "final environmental assessment",
    "final ea",
    "availability of the final",
    "notice of availability",
    "record of decision",
    "final supplemental environmental impact statement",
    "final supplemental eis",
)

_NOA_LIKE_RE = re.compile(
    r"\b(?:final\s+environmental\s+impact\s+statement|final\s+eis"
    r"|finding\s+of\s+no\s+significant\s+impact|fonsi"
    r"|final\s+environmental\s+assessment|final\s+ea"
    r"|final\s+supplemental\s+environmental\s+impact\s+statement"
    r"|final\s+supplemental\s+eis)\b",
    re.IGNORECASE,
)


@dataclass
class FederalRegisterConfig:
    process_types: tuple[str, ...] = tuple()
    energy_types: tuple[str, ...] = tuple()
    sample_n: Optional[int] = None
    random_state: int = 7
    per_page: int = 100
    throttle_seconds: float = 0.25
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    fetch_raw_text: bool = False
    conservative: bool = True
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    max_candidates_per_project: int = 10
    api_windowing: bool = True
    api_cap_total_pages: int = 50
    api_cap_count: int = 5000
    show_progress: bool = True
    progress_page_interval: int = 10
    progress_project_interval: int = 5000


def _normalize_space(text: object) -> str:
    return " ".join(str(text).split())


def _normalize_phrase(text: object) -> str:
    return _normalize_space(re.sub(r"[^A-Za-z0-9]+", " ", str(text))).lower()


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_project_id(value: object) -> str:
    """Unwrap NEPATEC project_id values that arrive as Arrow struct payloads."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return _normalize_project_id(value.get("value"))
        return _normalize_text(value)

    text = _normalize_text(value)
    if not text:
        return ""

    if text.startswith("{") and text.endswith("}") and "value" in text:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict) and "value" in parsed:
                return _normalize_project_id(parsed.get("value"))

    return text


def _json_safe(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _parse_listish(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [_normalize_text(v) for v in value if _normalize_text(v)]
    text = _normalize_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [_normalize_text(v) for v in parsed if _normalize_text(v)]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    if ";" in text:
        return [_normalize_text(part) for part in text.split(";") if _normalize_text(part)]
    if "," in text and text.count(",") <= 6:
        return [_normalize_text(part) for part in text.split(",") if _normalize_text(part)]
    return [text]


def _clean_term(value: object) -> str:
    """
    Normalize agency/state terms that may appear as list-like strings.
    """
    parts = _parse_listish(value)
    text = parts[0] if parts else _normalize_text(value)
    text = text.strip("[]").strip("\"' ")
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in _EMPTY_HINT_PATTERNS):
        return ""
    return text


def _strip_title_prefixes(title: str) -> str:
    text = _normalize_space(title)
    if not text:
        return ""
    changed = True
    while changed:
        changed = False
        for pattern in _TITLE_PREFIX_PATTERNS:
            updated = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" ,;:-")
            if updated != text:
                text = _normalize_space(updated)
                changed = True
    return text


def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9]+", _normalize_text(text).lower()):
        if len(token) < 3:
            continue
        if token in stopwords:
            continue
        tokens.append(token)
    return tokens


def _distinctive_tokens(text: str) -> list[str]:
    seen = set()
    tokens = []
    for token in _tokenize(text, _MATCH_STOPWORDS):
        if len(token) < 4 and not any(char.isdigit() for char in token):
            continue
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def _search_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[A-Za-z0-9]+", text)
        if len(word) > 1 and word.lower() != "s"
    ]


def _token_weight(token: str) -> int:
    lowered = token.lower()
    weight = min(len(lowered), 12)
    if lowered in _SEARCH_NOISE_TOKENS:
        weight -= 4
    if any(char.isdigit() for char in lowered):
        weight += 2
    return max(weight, 0)


def _window_score(words: list[str]) -> tuple[int, int]:
    lowered = [word.lower() for word in words]
    informative = sum(1 for word in lowered if _token_weight(word) > 0)
    weighted = sum(_token_weight(word) for word in lowered)
    return informative, weighted


def _select_title_phrase(title: str, max_words: int = 8) -> str:
    """
    Select a short phrase from the project title to reduce API noise.
    Prefer a window with more distinctive words rather than the raw prefix.
    """
    title = _strip_title_prefixes(_normalize_text(title))
    words = _search_words(title)
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)

    min_words = min(4, len(words))
    best_phrase = " ".join(words[:max_words])
    best_key = (-1, -1, -1)
    for size in range(max_words, min_words - 1, -1):
        for start in range(0, len(words) - size + 1):
            window = words[start : start + size]
            score_key = (*_window_score(window), start)
            if score_key > best_key:
                best_key = score_key
                best_phrase = " ".join(window)
    return best_phrase


def _request_key(terms: str, start_date: str, end_date: Optional[str], page: int = 1) -> str:
    payload = f"{terms}|{start_date}|{end_date or ''}|{page}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _effective_end_date(end_date: Optional[str]) -> str:
    return end_date or date.today().isoformat()


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def _quarter_month(month: int) -> int:
    return ((month - 1) // 3) * 3 + 1


def _window_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _iter_date_windows(start_date: str, end_date: str, level: str = "year") -> list[tuple[str, str]]:
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    if start > end:
        raise ValueError(f"start_date must be <= end_date: {start_date} > {end_date}")

    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        if level == "year":
            window_stop = date(cursor.year, 12, 31)
            next_cursor = date(cursor.year + 1, 1, 1)
        elif level == "quarter":
            start_month = _quarter_month(cursor.month)
            end_month = start_month + 2
            window_stop = _window_end(cursor.year, end_month)
            if end_month == 12:
                next_cursor = date(cursor.year + 1, 1, 1)
            else:
                next_cursor = date(cursor.year, end_month + 1, 1)
        elif level == "month":
            window_stop = _window_end(cursor.year, cursor.month)
            if cursor.month == 12:
                next_cursor = date(cursor.year + 1, 1, 1)
            else:
                next_cursor = date(cursor.year, cursor.month + 1, 1)
        else:
            raise ValueError(f"Unsupported date window level: {level}")

        segment_end = min(window_stop, end)
        windows.append((cursor.isoformat(), segment_end.isoformat()))
        cursor = next_cursor

    return windows


def _next_window_level(level: str) -> Optional[str]:
    if level == "year":
        return "quarter"
    if level == "quarter":
        return "month"
    return None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _response_is_capped(response: dict, config: FederalRegisterConfig) -> bool:
    total_pages = _safe_int(response.get("total_pages"), 0)
    count = _safe_int(response.get("count"), 0)
    return total_pages >= config.api_cap_total_pages or count >= config.api_cap_count


def _progress(config: FederalRegisterConfig, message: str) -> None:
    if config.show_progress:
        print(message, flush=True)


def search_noi(
    terms: str,
    start_date: str,
    end_date: Optional[str],
    per_page: int,
    max_retries: int,
    retry_backoff_seconds: float,
    page: int = 1,
) -> dict:
    params = {
        "conditions[term]": terms,
        "conditions[type][]": "NOTICE",
        "conditions[publication_date][gte]": start_date,
        "conditions[publication_date][lte]": _effective_end_date(end_date),
        "per_page": per_page,
        "page": page,
        "order": "oldest",
        "fields[]": list(FR_FIELDS),
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(FR_ENDPOINT, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(retry_backoff_seconds * (attempt + 1))
    raise last_error


def _is_noi_title(title: str) -> bool:
    title_lower = _normalize_text(title).lower()
    if not _NOI_LIKE_RE.search(title_lower):
        return False
    if _REJECT_NOTICE_RE.search(title_lower):
        return False
    return True


def _is_noa_title(title: str) -> bool:
    """Return True if FR title describes a Final EIS NOA, FONSI, or Final EA availability notice.

    FEIS and FONSI titles are accepted unconditionally — in the Federal Register context,
    any document titled with "Final Environmental Impact Statement" or "Finding of No
    Significant Impact" is an end-of-process notice. "Final EA" / "Final Environmental
    Assessment" titles require "availab" to avoid false positives from interim EA notices.
    """
    t = _normalize_text(title).lower()
    if not _NOA_LIKE_RE.search(t):
        return False
    is_feis = bool(re.search(
        r"\b(?:final\s+(?:supplemental\s+)?environmental\s+impact\s+statement|final\s+(?:supplemental\s+)?eis)\b", t
    ))
    if is_feis:
        return True
    is_fonsi = bool(re.search(r"\b(?:finding\s+of\s+no\s+significant\s+impact|fonsi)\b", t))
    if is_fonsi:
        return True
    # Final EA / Final environmental assessment: require "availab" to be specific
    return "availab" in t


def _is_valid_candidate_title(title: str) -> bool:
    title_lower = _normalize_text(title).lower()
    if not _NOI_LIKE_RE.search(title_lower):
        return False
    return True


def _extract_scoping_dates(text: str) -> list[str]:
    if not text:
        return []
    matches = []
    for line in text.splitlines():
        if "scoping meeting" in line.lower():
            matches.extend(re.findall(_DATE_PATTERN, line))
    return sorted(set(matches))


def _fetch_raw_text(
    url: str,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.5,
) -> str:
    if not url:
        return ""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_backoff_seconds * (attempt + 1))
    raise last_error


def _field(item: object, *names: str) -> object:
    if isinstance(item, pd.Series):
        for name in names:
            if name in item.index:
                return item.get(name)
    elif isinstance(item, dict):
        for name in names:
            if name in item:
                return item.get(name)
    else:
        for name in names:
            if hasattr(item, name):
                return getattr(item, name)
    return None


def _fr_document_number(item: object) -> str:
    return _normalize_text(_field(item, "fr_document_number", "document_number"))


def _fr_title(item: object) -> str:
    return _normalize_text(_field(item, "fr_title", "title"))


def _fr_abstract(item: object) -> str:
    return _normalize_text(_field(item, "fr_abstract", "abstract"))


def _fr_url(item: object) -> str:
    return _normalize_text(_field(item, "fr_url", "html_url"))


def _fr_agency_text(item: object) -> str:
    return " ".join(_parse_listish(_field(item, "fr_agency_names", "agency_names")))


def _fr_query_terms(item: object) -> str:
    return _normalize_text(_field(item, "fr_query_terms", "noi_query"))


def _normalize_fr_doc_number(raw: str) -> str:
    """Normalize FR Doc number: replace en-dash/em-dash with ASCII hyphen."""
    return re.sub(r'[\u2013\u2014]', '-', raw).strip()


def _extract_fr_doc_numbers_from_text(text: str) -> list[tuple[str, str, int]]:
    """Return list of (normalized_doc_number, raw_doc_number, char_position) from page text."""
    results = []
    for m in _FR_DOC_RE.finditer(text):
        raw = m.group(1)
        normalized = _normalize_fr_doc_number(raw)
        results.append((normalized, raw, m.start()))
    return results


def _extract_fr_url_doc_number(text: str) -> list[tuple[str, str, int]]:
    """Return list of (normalized_doc_number, raw_url_match, char_position) from FR URLs."""
    results = []
    for m in _FR_URL_RE.finditer(text):
        raw = m.group(1)
        normalized = _normalize_fr_doc_number(raw)
        results.append((normalized, m.group(0), m.start()))
    return results


def _noi_proximity_check(text: str, pos: int, process_type: str, window: int = 500) -> Optional[str]:
    """Return the nearest NOI-like phrase within window chars of pos, or None."""
    text_lower = text.lower()
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    region = text_lower[start:end]
    rel_pos = pos - start

    all_phrases = _NOI_PROXIMITY_PHRASES
    if process_type.upper() == "CE":
        all_phrases = _NOI_PROXIMITY_PHRASES + _CE_PROXIMITY_PHRASES

    best: Optional[str] = None
    best_dist = window + 1
    for phrase in all_phrases:
        idx = region.find(phrase)
        if idx >= 0:
            dist = abs(idx - rel_pos)
            if dist < best_dist:
                best_dist = dist
                best = phrase
    return best


def _noa_proximity_check(text: str, pos: int, window: int = 500) -> Optional[str]:
    """Return the nearest NOA-like phrase within window chars of pos, or None."""
    text_lower = text.lower()
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    region = text_lower[start:end]
    rel_pos = pos - start

    best: Optional[str] = None
    best_dist = window + 1
    for phrase in _NOA_PROXIMITY_PHRASES:
        idx = region.find(phrase)
        if idx >= 0:
            dist = abs(idx - rel_pos)
            if dist < best_dist:
                best_dist = dist
                best = phrase
    return best


def _parse_fr_date_text(text: str) -> tuple[str, str]:
    """Extract (raw_text, parsed_iso_date_str) from FR date prose. Returns ('', '') if not found."""
    m = _FR_DATE_TEXT_RE.search(text)
    if not m:
        return "", ""
    raw = m.group(0)
    date_str = m.group(1)
    try:
        parsed = datetime.strptime(date_str, "%B %d, %Y").date()
        return raw, parsed.isoformat()
    except ValueError:
        return raw, ""


def _project_context_text(row: pd.Series) -> str:
    parts = [
        row.get("project_title"),
        row.get("project_description"),
        row.get("project_sponsor"),
        row.get("lead_agency"),
        row.get("project_department"),
        row.get("project_state"),
    ]
    return " ".join(_normalize_text(part) for part in parts if _normalize_text(part))


def _term_has_overlap(term: object, text: str) -> bool:
    term_text = _clean_term(term)
    if not term_text:
        return False
    text_norm = _normalize_phrase(text)
    term_norm = _normalize_phrase(term_text)
    if term_norm and term_norm in text_norm:
        return True
    term_tokens = set(_distinctive_tokens(term_text))
    text_tokens = set(_distinctive_tokens(text))
    return bool(term_tokens & text_tokens)


def _process_flags(process_type: object, candidate_text: str) -> tuple[bool, bool]:
    ptype = _normalize_text(process_type).upper()
    text = _normalize_text(candidate_text).lower()
    has_eis = "environmental impact statement" in text or re.search(r"\beis\b", text) is not None
    has_ea = "environmental assessment" in text or re.search(r"\bea\b", text) is not None
    has_ce = "categorical exclusion" in text
    if ptype == "EIS":
        return has_eis, (has_ea or has_ce) and not has_eis
    if ptype == "EA":
        return has_ea, (has_eis or has_ce) and not has_ea
    if ptype == "CE":
        return has_ce, (has_eis or has_ea) and not has_ce
    return False, False


def _candidate_match_metrics(item: object, row: pd.Series) -> dict:
    title_text = _fr_title(item)
    abstract_text = _fr_abstract(item)
    agency_text = _fr_agency_text(item)
    combined_text = " ".join([title_text, abstract_text, agency_text])
    combined_norm = _normalize_phrase(combined_text)

    project_title = _normalize_text(row.get("project_title"))
    project_tokens = set(_distinctive_tokens(_strip_title_prefixes(project_title)))
    candidate_tokens = set(_distinctive_tokens(combined_text))
    title_overlap = sorted(project_tokens & candidate_tokens)
    selected_phrase = _select_title_phrase(project_title)
    selected_phrase_norm = _normalize_phrase(selected_phrase)

    exact_phrase = bool(selected_phrase_norm and selected_phrase_norm in combined_norm)
    agency_match = _term_has_overlap(row.get("lead_agency"), agency_text) or _term_has_overlap(
        row.get("project_department"), agency_text
    )
    state_match = _term_has_overlap(row.get("project_state"), combined_text)
    sponsor_match = _term_has_overlap(row.get("project_sponsor"), combined_text)
    process_match, process_conflict = _process_flags(row.get("process_type"), combined_text)

    doc_number = _fr_document_number(item)
    context_text = _project_context_text(row)
    document_number_evidence = bool(doc_number and doc_number.lower() in context_text.lower())
    fr_citation_evidence = bool(_FR_CITATION_RE.search(context_text) and document_number_evidence)
    scoping_dates = _parse_listish(_field(item, "fr_scoping_meeting_dates", "noi_scoping_meeting_dates"))

    return {
        "exact_phrase_match": exact_phrase,
        "title_overlap_tokens": title_overlap,
        "title_overlap_count": len(title_overlap),
        "project_token_count": len(project_tokens),
        "title_containment_ratio": len(title_overlap) / max(len(project_tokens), 1),
        "agency_match": agency_match,
        "state_match": state_match,
        "sponsor_match": sponsor_match,
        "process_match": process_match,
        "process_conflict": process_conflict,
        "document_number_evidence": document_number_evidence,
        "fr_citation_evidence": fr_citation_evidence,
        "raw_text_scoping_date_count": len(scoping_dates),
    }


def _passes_candidate_threshold(metrics: dict) -> bool:
    if metrics["document_number_evidence"] or metrics["fr_citation_evidence"]:
        return True
    if metrics["exact_phrase_match"]:
        return True
    if metrics["title_overlap_count"] >= 2:
        return True
    if metrics["title_overlap_count"] == 1:
        token = metrics["title_overlap_tokens"][0]
        contextual = metrics["agency_match"] or metrics["state_match"] or metrics["sponsor_match"]
        if contextual and (len(token) >= 8 or any(char.isdigit() for char in token)):
            return True
    return False


def _score_candidate(item: object, row: pd.Series, metrics: Optional[dict] = None) -> int:
    metrics = metrics or _candidate_match_metrics(item, row)
    score = 0
    score += metrics["title_overlap_count"] * 5
    score += int(round(metrics["title_containment_ratio"] * 20))
    if metrics["exact_phrase_match"]:
        score += 15
    if metrics["agency_match"]:
        score += 6
    if metrics["state_match"]:
        score += 4
    if metrics["sponsor_match"]:
        score += 8
    if metrics["process_match"]:
        score += 5
    if metrics["document_number_evidence"]:
        score += 50
    if metrics["fr_citation_evidence"]:
        score += 25
    score += metrics["raw_text_scoping_date_count"] * 2
    if metrics["process_conflict"]:
        score -= 12
    return int(score)


def _is_ce_project(row: pd.Series) -> bool:
    return _normalize_text(row.get("process_type")).upper() == "CE"


def _required_title_overlap(n_project_tokens: int) -> int:
    """Minimum distinctive title token overlap required for auto-accept.

    Scales with title length so short titles require all tokens to match
    while longer titles require at least 3:

        1 token  → require 1 (all)
        2 tokens → require 2 (all)
        3 tokens → require 2
        4+tokens → require 3
    """
    if n_project_tokens <= 0:
        return 1  # no tokens → threshold can never be met → always review
    if n_project_tokens <= 2:
        return n_project_tokens  # require all
    if n_project_tokens == 3:
        return 2
    return 3  # 4+


def _classify_noi_candidate(
    item: object,
    row: pd.Series,
    conservative: bool = True,
    nepatec_doc_numbers: frozenset = frozenset(),
    nepatec_evidence_types: Optional[dict[str, set[str]]] = None,
) -> tuple[str, str]:
    metrics = _candidate_match_metrics(item, row)
    contextual = metrics["agency_match"] or metrics["state_match"] or metrics["sponsor_match"]
    nepatec_evidence_types = nepatec_evidence_types or {}

    if _REJECT_NOTICE_RE.search(_fr_title(item)):
        if metrics["document_number_evidence"] or metrics["fr_citation_evidence"]:
            return "medium", "termination_or_withdrawal_notice_requires_review"
        return "low", "termination_or_withdrawal_notice_rejected"

    # NEPATEC direct evidence path: FR doc number found in project's own NEPATEC documents
    candidate_doc_number = _fr_document_number(item)
    has_nepatec_doc_evidence = bool(
        candidate_doc_number and nepatec_doc_numbers and candidate_doc_number in nepatec_doc_numbers
    )

    if has_nepatec_doc_evidence:
        evidence_types = nepatec_evidence_types.get(candidate_doc_number, set())
        has_noi_doc_evidence = "fr_doc_noi" in evidence_types
        has_url_evidence = "fr_url" in evidence_types

        # CE: always review regardless of corroboration
        if _is_ce_project(row):
            return "medium", "ce_nepatec_doc_number_evidence_requires_review"

        # EA/EIS: require both direct doc number evidence AND title token overlap.
        # Agency/state/sponsor alone is not sufficient — title match guards against
        # the same FR doc number appearing in multiple projects' files (e.g. a
        # programmatic EIS cited by many site-specific reviews).
        # Threshold scales with title length (see _required_title_overlap).
        required = _required_title_overlap(metrics["project_token_count"])
        title_ok = metrics["title_overlap_count"] >= required
        # A hard process conflict (e.g. EIS project but FR record is explicitly EA)
        # overrides even a strong doc number + title match — send to review.
        conflict = metrics["process_conflict"]
        # The fetched FR record must itself be an NOI-type notice (not a Notice of
        # Availability of a Final EIS or other non-initiation notice).  If it isn't,
        # its publication date would contaminate the initiation-date field.
        has_noi_title = bool(_NOI_LIKE_RE.search(_fr_title(item)))

        if has_noi_doc_evidence and title_ok and not conflict:
            if has_noi_title:
                return "high", "nepatec_fr_doc_number_with_title_match"
            return "medium", "nepatec_fr_doc_number_non_noi_fr_title_requires_review"
        if has_noi_doc_evidence and title_ok and conflict:
            return "medium", "nepatec_fr_doc_number_process_conflict_requires_review"
        if has_noi_doc_evidence:
            return "medium", "nepatec_fr_doc_number_no_title_match_requires_review"

        # Federal Register URLs: same two-gate rule (doc number + title).
        if has_url_evidence and title_ok and not conflict:
            if has_noi_title:
                return "high", "nepatec_fr_url_with_title_match"
            return "medium", "nepatec_fr_url_non_noi_fr_title_requires_review"
        if has_url_evidence:
            return "medium", "nepatec_fr_url_no_title_match_requires_review"

        return "medium", "nepatec_direct_evidence_requires_review"

    # Title-only path (no NEPATEC doc number evidence)
    # CE: apply CE-specific rules
    if _is_ce_project(row) and metrics["title_overlap_count"] < 4:
        if metrics["title_overlap_count"] >= 2 and contextual and not metrics["process_conflict"]:
            return "medium", "ce_match_requires_distinctive_token_review"
        return "low", "ce_low_distinctive_title_overlap"

    # Title-only: cap at medium — strong matches go to review, not auto-accept
    if (
        metrics["title_overlap_count"] >= 3
        and metrics["title_containment_ratio"] >= 0.60
        and contextual
        and not metrics["process_conflict"]
    ):
        return "medium", "title_only_strong_overlap_requires_review"
    if metrics["exact_phrase_match"] and metrics["title_overlap_count"] >= 2 and not metrics["process_conflict"]:
        return "medium", "title_only_exact_phrase_requires_review"
    if metrics["title_overlap_count"] >= 2 and contextual and not metrics["process_conflict"]:
        return "medium", "moderate_title_overlap_with_context"
    if not conservative and metrics["title_overlap_count"] >= 2 and not metrics["process_conflict"]:
        return "medium", "moderate_title_overlap"
    return "low", "weak_or_contextless_match"


def _candidate_record(
    item: object,
    row: pd.Series,
    conservative: bool = True,
    nepatec_doc_numbers: frozenset = frozenset(),
    nepatec_evidence_types: Optional[dict[str, set[str]]] = None,
) -> Optional[dict]:
    title = _fr_title(item)
    if not title:
        return None

    # Check direct NEPATEC doc number evidence before applying title filters —
    # a doc number found in the project's own files must always be evaluated,
    # even if the resolved FR record is not itself an NOI (e.g. it turns out to
    # be an EA notice, which triggers a process-conflict review path).
    candidate_doc_number = _fr_document_number(item)
    has_nepatec_doc_evidence = bool(
        candidate_doc_number and nepatec_doc_numbers and candidate_doc_number in nepatec_doc_numbers
    )

    if not has_nepatec_doc_evidence and not _is_valid_candidate_title(title):
        return None

    metrics = _candidate_match_metrics(item, row)
    if not _passes_candidate_threshold(metrics) and not has_nepatec_doc_evidence:
        return None

    confidence, reason = _classify_noi_candidate(
        item,
        row,
        conservative=conservative,
        nepatec_doc_numbers=nepatec_doc_numbers,
        nepatec_evidence_types=nepatec_evidence_types,
    )
    score = _score_candidate(item, row, metrics=metrics)
    return {
        "project_id": _normalize_project_id(row.get("project_id")),
        "project_title": row.get("project_title"),
        "process_type": row.get("process_type"),
        "project_energy_type": row.get("project_energy_type"),
        "fr_document_number": candidate_doc_number,
        "fr_title": title,
        "fr_publication_date": _normalize_text(_field(item, "fr_publication_date", "publication_date")),
        "fr_url": _fr_url(item),
        "fr_type": _normalize_text(_field(item, "fr_type", "type")),
        "fr_subtype": _normalize_text(_field(item, "fr_subtype", "subtype")),
        "fr_comments_close_on": _normalize_text(_field(item, "fr_comments_close_on", "comments_close_on")),
        "fr_scoping_meeting_dates": _normalize_text(
            _field(item, "fr_scoping_meeting_dates", "noi_scoping_meeting_dates")
        ),
        "fr_query_terms": _fr_query_terms(item),
        "candidate_rank": None,
        "match_score": score,
        "match_confidence": confidence,
        "match_reason": reason,
        "title_overlap_count": metrics["title_overlap_count"],
        "title_containment_ratio": metrics["title_containment_ratio"],
        "title_overlap_tokens": ", ".join(metrics["title_overlap_tokens"]),
        "exact_phrase_match": metrics["exact_phrase_match"],
        "agency_match": metrics["agency_match"],
        "state_match": metrics["state_match"],
        "sponsor_match": metrics["sponsor_match"],
        "process_match": metrics["process_match"],
        "process_conflict": metrics["process_conflict"],
        "document_number_evidence": metrics["document_number_evidence"],
        "fr_citation_evidence": metrics["fr_citation_evidence"],
        "raw_text_scoping_date_count": metrics["raw_text_scoping_date_count"],
        "nepatec_fr_document_number_evidence": bool(candidate_doc_number and candidate_doc_number in nepatec_doc_numbers),
    }


def _load_cache(cache_path: Optional[Path]) -> dict:
    if not cache_path or not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text())


def _save_cache(cache_path: Optional[Path], cache: dict) -> None:
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))


def _normalize_fr_document(item: dict, query_terms: str, fetch_run_at: str) -> dict:
    doc_number = _normalize_text(item.get("document_number"))
    if not doc_number:
        fallback = f"{item.get('title', '')}|{item.get('publication_date', '')}|{item.get('html_url', '')}"
        doc_number = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:16]
    return {
        "fr_document_number": doc_number,
        "fr_title": _normalize_text(item.get("title")),
        "fr_publication_date": _normalize_text(item.get("publication_date")),
        "fr_url": _normalize_text(item.get("html_url")),
        "fr_pdf_url": _normalize_text(item.get("pdf_url")),
        "fr_raw_text_url": _normalize_text(item.get("raw_text_url")),
        "fr_agency_names": _json_safe(item.get("agency_names") or []),
        "fr_agencies": _json_safe(item.get("agencies") or []),
        "fr_type": _normalize_text(item.get("type")),
        "fr_subtype": _normalize_text(item.get("subtype")),
        "fr_comments_close_on": _normalize_text(item.get("comments_close_on")),
        "fr_abstract": _normalize_text(item.get("abstract")),
        "fr_excerpts": _json_safe(item.get("excerpts") or []),
        "fr_scoping_meeting_dates": "",
        "fr_query_terms": query_terms,
        "fr_query_count": 1,
        "fr_fetch_run_at": fetch_run_at,
    }


def _search_noi_cached(
    terms: str,
    start_date: str,
    end_date: str,
    page: int,
    config: FederalRegisterConfig,
    cache: dict,
) -> tuple[dict, bool]:
    cache_key = _request_key(f"corpus|{terms}", start_date, end_date, page)
    if cache_key in cache:
        return cache[cache_key], True

    response = search_noi(
        terms,
        start_date,
        end_date,
        config.per_page,
        config.max_retries,
        config.retry_backoff_seconds,
        page=page,
    )
    cache[cache_key] = response
    time.sleep(config.throttle_seconds)
    return response, False


def _add_fr_records(
    results: list[dict],
    terms: str,
    fetch_run_at: str,
    records_by_doc: dict[str, dict],
    query_terms_by_doc: dict[str, set[str]],
) -> tuple[set[str], int]:
    seen_in_page = set()
    added = 0
    for item in results:
        record = _normalize_fr_document(item, terms, fetch_run_at)
        doc_number = record["fr_document_number"]
        seen_in_page.add(doc_number)
        if doc_number not in records_by_doc:
            records_by_doc[doc_number] = record
            added += 1
        query_terms_by_doc[doc_number].add(terms)
    return seen_in_page, added


def _append_fetch_report_row(
    report_rows: list[dict],
    *,
    terms: str,
    query_index: int,
    query_total: int,
    level: str,
    start_date: str,
    end_date: str,
    response: dict,
    pages_fetched: int,
    cached_pages: int,
    records_returned: int,
    unique_documents_returned: int,
    documents_added: int,
    capped: bool,
    split: bool,
    split_to: Optional[str],
    fetch_run_at: str,
) -> None:
    report_rows.append(
        {
            "fr_query_terms": terms,
            "query_index": query_index,
            "query_total": query_total,
            "window_level": level,
            "window_start": start_date,
            "window_end": end_date,
            "count": _safe_int(response.get("count"), 0),
            "total_pages": _safe_int(response.get("total_pages"), 0),
            "pages_fetched": pages_fetched,
            "cached_pages": cached_pages,
            "network_pages": pages_fetched - cached_pages,
            "records_returned": records_returned,
            "unique_documents_returned": unique_documents_returned,
            "documents_added": documents_added,
            "capped": capped,
            "split": split,
            "split_to": split_to or "",
            "fr_fetch_run_at": fetch_run_at,
        }
    )


def _fetch_query_window(
    terms: str,
    start_date: str,
    end_date: str,
    level: str,
    *,
    config: FederalRegisterConfig,
    cache: dict,
    records_by_doc: dict[str, dict],
    query_terms_by_doc: dict[str, set[str]],
    report_rows: list[dict],
    fetch_run_at: str,
    query_index: int,
    query_total: int,
) -> None:
    response, from_cache = _search_noi_cached(terms, start_date, end_date, 1, config, cache)
    total_pages = max(1, _safe_int(response.get("total_pages"), 1))
    count = _safe_int(response.get("count"), 0)
    results = response.get("results", []) or []
    capped = _response_is_capped(response, config)
    split_to = _next_window_level(level) if config.api_windowing else None

    if capped and split_to:
        _progress(
            config,
            (
                f"[FR API] query {query_index}/{query_total} {terms} "
                f"{start_date}..{end_date} ({level}) appears capped "
                f"(count={count:,}, total_pages={total_pages:,}); splitting to {split_to}"
            ),
        )
        _append_fetch_report_row(
            report_rows,
            terms=terms,
            query_index=query_index,
            query_total=query_total,
            level=level,
            start_date=start_date,
            end_date=end_date,
            response=response,
            pages_fetched=1,
            cached_pages=int(from_cache),
            records_returned=len(results),
            unique_documents_returned=0,
            documents_added=0,
            capped=True,
            split=True,
            split_to=split_to,
            fetch_run_at=fetch_run_at,
        )
        for child_start, child_end in _iter_date_windows(start_date, end_date, split_to):
            _fetch_query_window(
                terms,
                child_start,
                child_end,
                split_to,
                config=config,
                cache=cache,
                records_by_doc=records_by_doc,
                query_terms_by_doc=query_terms_by_doc,
                report_rows=report_rows,
                fetch_run_at=fetch_run_at,
                query_index=query_index,
                query_total=query_total,
            )
        return

    _progress(
        config,
        (
            f"[FR API] query {query_index}/{query_total} {terms} "
            f"{start_date}..{end_date} ({level}) pages={total_pages:,} count={count:,}"
        ),
    )

    pages_fetched = 1
    cached_pages = int(from_cache)
    records_returned = 0
    unique_documents = set()
    documents_added = 0

    page_documents, page_added = _add_fr_records(
        results,
        terms,
        fetch_run_at,
        records_by_doc,
        query_terms_by_doc,
    )
    records_returned += len(results)
    documents_added += page_added
    unique_documents.update(page_documents)

    for page in range(2, total_pages + 1):
        response, from_cache = _search_noi_cached(terms, start_date, end_date, page, config, cache)
        pages_fetched += 1
        cached_pages += int(from_cache)
        results = response.get("results", []) or []
        page_documents, page_added = _add_fr_records(
            results,
            terms,
            fetch_run_at,
            records_by_doc,
            query_terms_by_doc,
        )
        records_returned += len(results)
        documents_added += page_added
        unique_documents.update(page_documents)
        if not results:
            break
        if page == total_pages or page % max(1, config.progress_page_interval) == 0:
            _progress(
                config,
                (
                    f"[FR API] query {query_index}/{query_total} {terms} "
                    f"{start_date}..{end_date}: page {page:,}/{total_pages:,}, "
                    f"records={records_returned:,}, new_docs={documents_added:,}"
                ),
            )

    _append_fetch_report_row(
        report_rows,
        terms=terms,
        query_index=query_index,
        query_total=query_total,
        level=level,
        start_date=start_date,
        end_date=end_date,
        response=response,
        pages_fetched=pages_fetched,
        cached_pages=cached_pages,
        records_returned=records_returned,
        unique_documents_returned=len(unique_documents),
        documents_added=documents_added,
        capped=capped,
        split=False,
        split_to=None,
        fetch_run_at=fetch_run_at,
    )


def extract_nepatec_federal_register_evidence(
    projects: pd.DataFrame,
    *,
    analysis_dir: Path,
    process_types: tuple[str, ...] = ("EA", "EIS", "CE"),
    evidence_output: Optional[Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Scan EA/EIS/CE NEPATEC page text for Federal Register evidence using DuckDB.

    Returns an evidence table (one row per FR Doc. number found per page).
    Results are cached to evidence_output if provided.
    """
    import duckdb

    analysis_dir = Path(analysis_dir)
    processed_dir = analysis_dir.parent / "processed"

    # Build project_id -> project_title lookup
    project_lookup: dict[str, str] = {}
    if not projects.empty and "project_id" in projects.columns:
        for _, prow in projects.iterrows():
            pid = _normalize_project_id(prow.get("project_id"))
            if pid and pid not in project_lookup:
                project_lookup[pid] = _normalize_text(prow.get("project_title"))

    all_evidence: list[dict] = []

    for process_type in process_types:
        pt_lower = process_type.lower()
        docs_path = processed_dir / pt_lower / "documents.parquet"
        pages_path = processed_dir / pt_lower / "pages.parquet"

        if not docs_path.exists() or not pages_path.exists():
            if show_progress:
                print(f"[fr-evidence] {process_type}: processed data not found at {docs_path.parent}, skipping", flush=True)
            continue

        con = duckdb.connect()
        try:
            doc_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{docs_path}')").fetchone()[0]
            page_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pages_path}')").fetchone()[0]
            if show_progress:
                print(
                    f"[fr-evidence] {process_type}: scanning {doc_count:,} documents / {page_count:,} pages (DuckDB)",
                    flush=True,
                )

            result_df = con.execute(
                """
                SELECT p.document_id, p.page_number, p.page_text,
                       d.project_id, d.file_name, d.document_title,
                       d.document_type, d.main_document
                FROM read_parquet(?) AS p
                JOIN read_parquet(?) AS d USING (document_id)
                WHERE p.page_text LIKE '%FR Doc%'
                   OR p.page_text LIKE '%federalregister.gov%'
                   OR p.page_text LIKE '%Federal Register%'
                """,
                [str(pages_path), str(docs_path)],
            ).df()
        finally:
            con.close()

        if show_progress:
            print(
                f"[fr-evidence] {process_type}: {len(result_df):,} pages matched LIKE filter; extracting FR Doc numbers",
                flush=True,
            )

        projects_with_evidence: set[str] = set()

        for _, page_row in result_df.iterrows():
            page_text = _normalize_text(page_row.get("page_text"))
            if not page_text:
                continue

            project_id = _normalize_project_id(page_row.get("project_id"))
            document_id = _normalize_text(page_row.get("document_id"))
            page_number = page_row.get("page_number")
            file_name = _normalize_text(page_row.get("file_name"))
            document_title = _normalize_text(page_row.get("document_title"))
            document_type = _normalize_text(page_row.get("document_type"))
            main_document = bool(page_row.get("main_document"))
            project_title = project_lookup.get(project_id, "")
            evidence_rank = 1 if main_document else 2

            project_tokens = set(_distinctive_tokens(project_title))

            # Extract FR Doc. numbers from bracket pattern
            for normalized, raw, pos in _extract_fr_doc_numbers_from_text(page_text):
                noi_phrase = _noi_proximity_check(page_text, pos, process_type)
                if noi_phrase:
                    evidence_type = "fr_doc_noi"
                    nearby_phrase = noi_phrase
                else:
                    noa_phrase = _noa_proximity_check(page_text, pos)
                    if noa_phrase:
                        evidence_type = "fr_doc_noa"
                        nearby_phrase = noa_phrase
                    else:
                        evidence_type = "fr_doc_non_noi"
                        nearby_phrase = None
                ctx_start = max(0, pos - 200)
                ctx_end = min(len(page_text), pos + 200)
                context = page_text[ctx_start:ctx_end]

                context_tokens = set(_distinctive_tokens(context))
                nearby_token_count = len(project_tokens & context_tokens)
                citation_m = _FR_CITATION_RE.search(context)
                fr_citation = citation_m.group(0) if citation_m else ""
                fr_date_text, fr_date_text_parsed = _parse_fr_date_text(context)

                all_evidence.append({
                    "project_id": project_id,
                    "process_type": process_type,
                    "project_title": project_title,
                    "document_id": document_id,
                    "file_name": file_name,
                    "document_title": document_title,
                    "document_type": document_type,
                    "main_document": main_document,
                    "page_number": page_number,
                    "evidence_type": evidence_type,
                    "fr_document_number": normalized,
                    "fr_document_number_raw": raw,
                    "fr_url": "",
                    "fr_citation": fr_citation,
                    "fr_date_text": fr_date_text,
                    "fr_date_text_parsed": fr_date_text_parsed,
                    "notice_title_snippet": nearby_phrase if nearby_phrase else "",
                    "evidence_context": context,
                    "nearby_noi_phrase": nearby_phrase if nearby_phrase else "",
                    "nearby_project_title_token_count": nearby_token_count,
                    "evidence_rank": evidence_rank,
                })
                projects_with_evidence.add(project_id)

            # Extract FR URLs
            for normalized, url_raw, pos in _extract_fr_url_doc_number(page_text):
                ctx_start = max(0, pos - 200)
                ctx_end = min(len(page_text), pos + 200)
                context = page_text[ctx_start:ctx_end]

                all_evidence.append({
                    "project_id": project_id,
                    "process_type": process_type,
                    "project_title": project_title,
                    "document_id": document_id,
                    "file_name": file_name,
                    "document_title": document_title,
                    "document_type": document_type,
                    "main_document": main_document,
                    "page_number": page_number,
                    "evidence_type": "fr_url",
                    "fr_document_number": normalized,
                    "fr_document_number_raw": normalized,
                    "fr_url": url_raw,
                    "fr_citation": "",
                    "fr_date_text": "",
                    "fr_date_text_parsed": "",
                    "notice_title_snippet": "",
                    "evidence_context": context,
                    "nearby_noi_phrase": "",
                    "nearby_project_title_token_count": 0,
                    "evidence_rank": evidence_rank,
                })
                projects_with_evidence.add(project_id)

        if show_progress:
            suffix = " (all → review)" if process_type.upper() == "CE" else ""
            print(
                f"[fr-evidence] {process_type}: found {len(projects_with_evidence):,} projects with FR Doc evidence{suffix}",
                flush=True,
            )

    if all_evidence:
        evidence_df = pd.DataFrame(all_evidence, columns=EVIDENCE_OUTPUT_COLUMNS)
    else:
        evidence_df = pd.DataFrame(columns=EVIDENCE_OUTPUT_COLUMNS)

    if evidence_output:
        evidence_path = Path(evidence_output)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_df.to_parquet(evidence_path, index=False)
        if show_progress:
            print(f"[fr-evidence] cached to {evidence_path} ({len(evidence_df):,} rows)", flush=True)

    return evidence_df


def fetch_federal_register_noi_corpus(
    config: FederalRegisterConfig,
    cache_path: Optional[Path] = None,
    fetch_report_output: Optional[Path] = None,
) -> pd.DataFrame:
    cache = _load_cache(cache_path)
    records_by_doc: dict[str, dict] = {}
    query_terms_by_doc: dict[str, set[str]] = defaultdict(set)
    report_rows: list[dict] = []
    fetch_run_at = datetime.now(timezone.utc).isoformat()
    end_date = _effective_end_date(config.end_date)
    initial_windows = _iter_date_windows(config.start_date, end_date, "year") if config.api_windowing else [
        (config.start_date, end_date)
    ]

    _progress(
        config,
        (
            f"[FR API] Starting corpus refresh: {len(NOI_CORPUS_QUERIES)} queries x "
            f"{len(initial_windows)} initial date windows from {config.start_date} to {end_date}"
        ),
    )

    query_total = len(NOI_CORPUS_QUERIES)
    for query_index, terms in enumerate(NOI_CORPUS_QUERIES, start=1):
        _progress(config, f"[FR API] Starting query {query_index}/{query_total}: {terms}")
        for window_start, window_end in initial_windows:
            _fetch_query_window(
                terms,
                window_start,
                window_end,
                "year" if config.api_windowing else "all",
                config=config,
                cache=cache,
                records_by_doc=records_by_doc,
                query_terms_by_doc=query_terms_by_doc,
                report_rows=report_rows,
                fetch_run_at=fetch_run_at,
                query_index=query_index,
                query_total=query_total,
            )
        _save_cache(cache_path, cache)
        _progress(
            config,
            f"[FR API] Finished query {query_index}/{query_total}: {terms}; unique docs so far={len(records_by_doc):,}",
        )

    _save_cache(cache_path, cache)

    records = []
    for doc_number, record in records_by_doc.items():
        query_terms = sorted(query_terms_by_doc[doc_number])
        record = dict(record)
        record["fr_query_terms"] = "; ".join(query_terms)
        record["fr_query_count"] = len(query_terms)
        records.append(record)

    corpus = pd.DataFrame(records, columns=CORPUS_OUTPUT_COLUMNS)
    fetch_report = pd.DataFrame(report_rows, columns=FETCH_REPORT_COLUMNS)
    if fetch_report_output:
        fetch_report_path = Path(fetch_report_output)
        fetch_report_path.parent.mkdir(parents=True, exist_ok=True)
        fetch_report.to_csv(fetch_report_path, index=False)
        capped_count = int(fetch_report["capped"].sum()) if not fetch_report.empty else 0
        split_count = int(fetch_report["split"].sum()) if not fetch_report.empty else 0
        _progress(
            config,
            (
                f"[FR API] Saved fetch report: {fetch_report_path} "
                f"({len(fetch_report):,} windows; capped={capped_count:,}; split={split_count:,})"
            ),
        )
    if not corpus.empty:
        years = pd.to_datetime(corpus["fr_publication_date"], errors="coerce").dt.year
        year_counts = years.value_counts().sort_index()
        if not year_counts.empty:
            recent_counts = year_counts[year_counts.index >= 2020]
            _progress(
                config,
                (
                    f"[FR API] Corpus complete: {len(corpus):,} unique docs, "
                    f"date range {corpus['fr_publication_date'].min()}..{corpus['fr_publication_date'].max()}, "
                    f"2020+ docs={int(recent_counts.sum()):,}"
                ),
            )
    if config.fetch_raw_text and not corpus.empty:
        scoping_dates = []
        for _, row in corpus.iterrows():
            raw_text = ""
            raw_url = _normalize_text(row.get("fr_raw_text_url"))
            if raw_url:
                try:
                    raw_text = _fetch_raw_text(
                        raw_url,
                        max_retries=config.max_retries,
                        retry_backoff_seconds=config.retry_backoff_seconds,
                    )
                    time.sleep(config.throttle_seconds)
                except requests.RequestException:
                    raw_text = ""
            dates = _extract_scoping_dates(raw_text or _normalize_text(row.get("fr_abstract")))
            scoping_dates.append("; ".join(dates) if dates else "")
        corpus["fr_scoping_meeting_dates"] = scoping_dates

    return corpus


def _build_corpus_index(corpus: pd.DataFrame) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for idx, row in corpus.iterrows():
        text = " ".join(
            [
                _normalize_text(row.get("fr_title")),
                _normalize_text(row.get("fr_abstract")),
                _fr_agency_text(row),
            ]
        )
        for token in _distinctive_tokens(text):
            index[token].add(idx)
    return index


def _build_corpus_doc_number_index(corpus: pd.DataFrame) -> dict[str, int]:
    """Map fr_document_number -> integer row index in corpus."""
    index: dict[str, int] = {}
    for idx, row in corpus.iterrows():
        doc_num = _normalize_text(row.get("fr_document_number"))
        if doc_num and doc_num not in index:
            index[doc_num] = idx
    return index


def fetch_documents_by_doc_numbers(
    doc_numbers: list[str],
    *,
    throttle_seconds: float = 0.25,
    cache_path: Optional[Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Direct-fetch FR documents by document number.

    For each doc number, calls:
        GET https://www.federalregister.gov/api/v1/documents/{doc_num}.json

    This replaces the keyword-corpus approach for projects with NEPATEC direct
    evidence. It is targeted (at most one call per unique doc number), faster
    than paginated keyword searches, and eliminates keyword-search false positives.

    Results are cached with key ``docnum|{doc_num}`` so subsequent runs skip
    already-fetched numbers. 404s are also cached as None to avoid re-fetching
    known-missing numbers.

    Returns a DataFrame with the same column schema as the keyword corpus
    (CORPUS_OUTPUT_COLUMNS).
    """
    if not doc_numbers:
        return pd.DataFrame(columns=CORPUS_OUTPUT_COLUMNS)

    cache = _load_cache(cache_path)
    fetch_run_at = datetime.now(timezone.utc).isoformat()
    records_by_doc: dict[str, dict] = {}
    not_found: list[str] = []
    network_calls = 0

    unique_doc_numbers = sorted(set(doc_numbers))

    if show_progress:
        print(
            f"[FR direct] Fetching {len(unique_doc_numbers):,} unique doc numbers from FR API",
            flush=True,
        )

    for i, doc_num in enumerate(unique_doc_numbers, start=1):
        cache_key = f"docnum|{doc_num}"
        if cache_key in cache:
            item = cache[cache_key]
        else:
            url = f"https://www.federalregister.gov/api/v1/documents/{doc_num}.json"
            try:
                resp = requests.get(
                    url,
                    params=[("fields[]", f) for f in FR_FIELDS],
                    timeout=30,
                )
                if resp.status_code == 404:
                    cache[cache_key] = None
                    not_found.append(doc_num)
                    network_calls += 1
                    time.sleep(throttle_seconds)
                    continue
                resp.raise_for_status()
                item = resp.json()
                cache[cache_key] = item
                network_calls += 1
                time.sleep(throttle_seconds)
            except Exception as exc:
                print(f"[FR direct] Warning: error fetching {doc_num}: {exc}", flush=True)
                continue

        if item is None:
            if doc_num not in not_found:
                not_found.append(doc_num)
            continue

        record = _normalize_fr_document(item, "direct_fetch", fetch_run_at)
        rec_doc_num = record["fr_document_number"]
        if rec_doc_num and rec_doc_num not in records_by_doc:
            records_by_doc[rec_doc_num] = record

        if show_progress and (i % 50 == 0 or i == len(unique_doc_numbers)):
            print(
                f"[FR direct] {i:,}/{len(unique_doc_numbers):,} processed; "
                f"found={len(records_by_doc):,}; not_found={len(not_found):,}; "
                f"network_calls={network_calls:,}",
                flush=True,
            )

    _save_cache(cache_path, cache)

    if show_progress:
        print(
            f"[FR direct] Complete: {len(records_by_doc):,} fetched, "
            f"{len(not_found):,} not in FR API, {network_calls:,} network calls",
            flush=True,
        )

    if not records_by_doc:
        return pd.DataFrame(columns=CORPUS_OUTPUT_COLUMNS)

    corpus = pd.DataFrame(list(records_by_doc.values()))
    for col in CORPUS_OUTPUT_COLUMNS:
        if col not in corpus.columns:
            corpus[col] = ""
    return corpus[CORPUS_OUTPUT_COLUMNS]


def _columns_present(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _sort_if_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    ascending: Optional[list[bool]] = None,
) -> pd.DataFrame:
    sort_columns = _columns_present(df, columns)
    if not sort_columns:
        return df
    sort_ascending = ascending[: len(sort_columns)] if ascending else True
    return df.sort_values(sort_columns, ascending=sort_ascending, kind="stable")


def write_focused_manual_review_exports(
    project_matches: pd.DataFrame,
    review: pd.DataFrame,
    projects: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, int]:
    """
    Write small, reproducible review packets for high-priority manual checks.

    These are derived from the canonical Federal Register artifacts:
    ambiguous project outputs, candidate rows for those ambiguous projects,
    and accepted rows with only 0-1 title-overlap tokens.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_meta_cols = _columns_present(
        projects,
        [
            "project_id",
            "process_type",
            "project_energy_type",
            "project_department",
            "lead_agency",
            "project_state",
            "project_sponsor",
        ],
    )
    project_rows = project_matches.copy()
    project_meta_add_cols = [
        col for col in project_meta_cols
        if col == "project_id" or col not in project_rows.columns
    ]
    project_meta = (
        projects[project_meta_add_cols].drop_duplicates("project_id")
        if project_meta_add_cols and "project_id" in project_meta_add_cols
        else pd.DataFrame()
    )
    if not project_meta.empty and len(project_meta.columns) > 1:
        project_rows = project_rows.merge(project_meta, on="project_id", how="left")

    ambiguous_ids = (
        set(project_rows.loc[project_rows["noi_match_status"].eq("ambiguous"), "project_id"].astype(str))
        if "noi_match_status" in project_rows.columns
        else set()
    )
    if not review.empty and ambiguous_ids and "project_id" in review.columns:
        ambiguous_candidates = review[review["project_id"].astype(str).isin(ambiguous_ids)].copy()
    else:
        ambiguous_candidates = pd.DataFrame(columns=review.columns)
    ambiguous_candidates = _sort_if_columns(
        ambiguous_candidates,
        ["project_id", "candidate_rank", "match_score"],
        ascending=[True, True, False],
    )
    ambiguous_candidates_path = output_dir / DEFAULT_AMBIGUOUS_CANDIDATES_OUTPUT.name
    ambiguous_candidates[
        _columns_present(ambiguous_candidates, FOCUSED_CANDIDATE_REVIEW_COLUMNS)
    ].to_csv(ambiguous_candidates_path, index=False)

    if "noi_publication_date" in project_rows.columns:
        accepted = project_rows[project_rows["noi_publication_date"].notna()].copy()
    else:
        accepted = pd.DataFrame(columns=project_rows.columns)
    if "noi_title_overlap_count" in accepted.columns:
        title_overlap = pd.to_numeric(accepted["noi_title_overlap_count"], errors="coerce").fillna(0)
    else:
        title_overlap = pd.Series(0, index=accepted.index)
    low_overlap_accepted = accepted[title_overlap <= 1].copy()
    low_overlap_accepted = _sort_if_columns(
        low_overlap_accepted,
        ["noi_title_overlap_count", "noi_match_score", "project_title"],
        ascending=[True, True, True],
    )
    low_overlap_accepted_path = output_dir / DEFAULT_LOW_OVERLAP_ACCEPTED_OUTPUT.name
    low_overlap_accepted[
        _columns_present(low_overlap_accepted, FOCUSED_PROJECT_REVIEW_COLUMNS)
    ].to_csv(low_overlap_accepted_path, index=False)

    if "noa_availability_date" in project_rows.columns:
        noa_accepted = project_rows[project_rows["noa_availability_date"].notna()].copy()
    else:
        noa_accepted = pd.DataFrame(columns=project_rows.columns)
    if "noa_title_overlap_count" in noa_accepted.columns:
        noa_title_overlap = pd.to_numeric(
            noa_accepted["noa_title_overlap_count"], errors="coerce"
        ).fillna(0)
    else:
        noa_title_overlap = pd.Series(0, index=noa_accepted.index)
    noa_low_overlap_accepted = noa_accepted[noa_title_overlap <= 1].copy()
    noa_low_overlap_accepted = _sort_if_columns(
        noa_low_overlap_accepted,
        ["noa_title_overlap_count", "noa_match_score", "project_title"],
        ascending=[True, True, True],
    )
    noa_low_overlap_accepted_path = output_dir / DEFAULT_NOA_LOW_OVERLAP_ACCEPTED_OUTPUT.name
    noa_low_overlap_accepted[
        _columns_present(noa_low_overlap_accepted, FOCUSED_NOA_PROJECT_REVIEW_COLUMNS)
    ].to_csv(noa_low_overlap_accepted_path, index=False)

    return {
        str(ambiguous_candidates_path): len(ambiguous_candidates),
        str(low_overlap_accepted_path): len(low_overlap_accepted),
        str(noa_low_overlap_accepted_path): len(noa_low_overlap_accepted),
    }


def _sort_candidate_records(candidates: list[dict]) -> list[dict]:
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    sorted_candidates = sorted(
        candidates,
        key=lambda rec: (
            -confidence_rank.get(rec["match_confidence"], 0),
            -rec["match_score"],
            rec.get("fr_publication_date") or "9999-99-99",
        ),
    )
    for rank, record in enumerate(sorted_candidates, start=1):
        record["candidate_rank"] = rank
    return sorted_candidates


def _empty_project_output(row: pd.Series, status: str = "unmatched", confidence: str = "none") -> dict:
    return {
        "project_id": _normalize_project_id(row.get("project_id")),
        "project_title": row.get("project_title"),
        "noi_publication_date": None,
        "noi_document_number": None,
        "noi_url": None,
        "noi_project_title": None,
        "noi_type": None,
        "noi_subtype": None,
        "noi_comments_close_on": None,
        "noi_scoping_meeting_dates": None,
        "noi_match_score": None,
        "noi_query": None,
        "noi_match_tier": None,
        "noi_match_confidence": confidence,
        "noi_match_status": status,
        "noi_candidate_count": 0,
        "noi_high_confidence_candidate_count": 0,
        "noi_title_overlap_count": None,
        "noi_title_overlap_tokens": None,
        "noi_agency_match": False,
        "noi_state_match": False,
        "noi_sponsor_match": False,
        "noi_process_match": False,
        "noi_process_conflict": False,
        "noi_match_reason": status,
        "noi_date_evidence_type": None,
        "noi_nepatec_evidence_document_id": None,
        "noi_nepatec_evidence_file_name": None,
        "noi_nepatec_evidence_page_number": None,
    }


def _project_output_from_candidate(
    row: pd.Series,
    candidate: dict,
    *,
    accepted: bool,
    status: str,
    confidence: str,
    candidate_count: int,
    high_count: int,
    reason: str,
    nepatec_evidence_row: Optional[dict] = None,
) -> dict:
    return {
        "project_id": _normalize_project_id(row.get("project_id")),
        "project_title": row.get("project_title"),
        "noi_publication_date": candidate.get("fr_publication_date") if accepted else None,
        "noi_document_number": candidate.get("fr_document_number"),
        "noi_url": candidate.get("fr_url"),
        "noi_project_title": candidate.get("fr_title"),
        "noi_type": candidate.get("fr_type"),
        "noi_subtype": candidate.get("fr_subtype"),
        "noi_comments_close_on": candidate.get("fr_comments_close_on"),
        "noi_scoping_meeting_dates": candidate.get("fr_scoping_meeting_dates") or None,
        "noi_match_score": candidate.get("match_score"),
        "noi_query": candidate.get("fr_query_terms"),
        "noi_match_tier": candidate.get("match_reason"),
        "noi_match_confidence": confidence,
        "noi_match_status": status,
        "noi_candidate_count": candidate_count,
        "noi_high_confidence_candidate_count": high_count,
        "noi_title_overlap_count": candidate.get("title_overlap_count"),
        "noi_title_overlap_tokens": candidate.get("title_overlap_tokens"),
        "noi_agency_match": candidate.get("agency_match"),
        "noi_state_match": candidate.get("state_match"),
        "noi_sponsor_match": candidate.get("sponsor_match"),
        "noi_process_match": candidate.get("process_match"),
        "noi_process_conflict": candidate.get("process_conflict"),
        "noi_match_reason": reason,
        "noi_date_evidence_type": "nepatec_fr_doc_number" if (accepted and nepatec_evidence_row) else None,
        "noi_nepatec_evidence_document_id": nepatec_evidence_row.get("document_id") if (accepted and nepatec_evidence_row) else None,
        "noi_nepatec_evidence_file_name": nepatec_evidence_row.get("file_name") if (accepted and nepatec_evidence_row) else None,
        "noi_nepatec_evidence_page_number": nepatec_evidence_row.get("page_number") if (accepted and nepatec_evidence_row) else None,
    }


def _empty_project_noa_output(row: pd.Series) -> dict:
    return {
        "noa_availability_date": None,
        "noa_document_number": None,
        "noa_url": None,
        "noa_fr_title": None,
        "noa_match_status": "unmatched",
        "noa_match_reason": "unmatched",
        "noa_match_score": None,
        "noa_title_overlap_count": None,
        "noa_title_overlap_tokens": None,
        "noa_date_evidence_type": None,
        "noa_nepatec_evidence_document_id": None,
        "noa_nepatec_evidence_file_name": None,
        "noa_nepatec_evidence_page_number": None,
    }


def _project_noa_output_from_candidate(
    row: pd.Series,
    candidate: dict,
    *,
    accepted: bool,
    status: str,
    reason: str,
    nepatec_evidence_row: Optional[dict] = None,
) -> dict:
    return {
        "noa_availability_date": candidate.get("fr_publication_date") if accepted else None,
        "noa_document_number": candidate.get("fr_document_number"),
        "noa_url": candidate.get("fr_url"),
        "noa_fr_title": candidate.get("fr_title"),
        "noa_match_status": status,
        "noa_match_reason": reason,
        "noa_match_score": candidate.get("match_score"),
        "noa_title_overlap_count": candidate.get("title_overlap_count"),
        "noa_title_overlap_tokens": candidate.get("title_overlap_tokens"),
        "noa_date_evidence_type": "nepatec_fr_doc_noa" if (accepted and nepatec_evidence_row) else None,
        "noa_nepatec_evidence_document_id": nepatec_evidence_row.get("document_id") if (accepted and nepatec_evidence_row) else None,
        "noa_nepatec_evidence_file_name": nepatec_evidence_row.get("file_name") if (accepted and nepatec_evidence_row) else None,
        "noa_nepatec_evidence_page_number": nepatec_evidence_row.get("page_number") if (accepted and nepatec_evidence_row) else None,
    }


def _classify_noa_candidate(
    item: object,
    row: pd.Series,
    nepatec_doc_numbers: frozenset = frozenset(),
    nepatec_evidence_types: Optional[dict[str, set[str]]] = None,
) -> tuple[str, str]:
    nepatec_evidence_types = nepatec_evidence_types or {}
    candidate_doc_number = _fr_document_number(item)
    has_noa_evidence = bool(
        candidate_doc_number and nepatec_doc_numbers and candidate_doc_number in nepatec_doc_numbers
    )
    if not has_noa_evidence:
        return "low", "no_nepatec_noa_evidence"

    if _REJECT_NOTICE_RE.search(_fr_title(item)):
        return "medium", "termination_or_withdrawal_notice_requires_review"

    noa_evidence_types = nepatec_evidence_types.get(candidate_doc_number, set())
    has_noa_doc_evidence = "fr_doc_noa" in noa_evidence_types
    has_noa_title = _is_noa_title(_fr_title(item))
    metrics = _candidate_match_metrics(item, row)
    required = _required_title_overlap(metrics["project_token_count"])
    title_ok = metrics["title_overlap_count"] >= required

    # Process alignment: EIS → FEIS title, EA → FONSI/Final EA title
    process_type = _normalize_text(row.get("process_type")).upper()
    fr_title_lower = _fr_title(item).lower()
    is_feis = bool(re.search(r"\bfinal\s+(?:environmental\s+impact\s+statement|eis)\b", fr_title_lower))
    is_fonsi = bool(re.search(
        r"\b(?:fonsi|finding\s+of\s+no\s+significant\s+impact|final\s+(?:ea|environmental\s+assessment))\b",
        fr_title_lower,
    ))
    process_aligned = (process_type == "EIS" and is_feis) or (process_type == "EA" and is_fonsi)

    if not has_noa_title:
        return "medium", "noa_doc_number_non_noa_fr_title_requires_review"
    if _is_ce_project(row):
        return "medium", "ce_noa_evidence_requires_review"
    if has_noa_doc_evidence and title_ok and process_aligned:
        return "high", "nepatec_fr_doc_noa_with_title_match"
    if has_noa_doc_evidence and title_ok and not process_aligned:
        return "medium", "noa_process_mismatch_requires_review"
    if has_noa_doc_evidence:
        return "medium", "noa_doc_number_insufficient_title_overlap"
    return "medium", "noa_evidence_requires_review"


def build_project_noi_matches(
    projects: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    conservative: bool = True,
    max_candidates_per_project: int = 10,
    show_progress: bool = False,
    progress_interval: int = 5000,
    nepatec_evidence: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    projects_unique = projects.drop_duplicates(subset=["project_id"], keep="first").copy()
    corpus = corpus.copy()
    project_total = len(projects_unique)

    if corpus.empty:
        project_rows = [_empty_project_output(row) for _, row in projects_unique.iterrows()]
        empty_candidates = pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)
        empty_review = empty_candidates.copy()
        return pd.DataFrame(project_rows, columns=PROJECT_OUTPUT_COLUMNS), empty_candidates, empty_review

    corpus_index = _build_corpus_index(corpus)
    corpus_doc_number_index = _build_corpus_doc_number_index(corpus)

    # Build per-project NEPATEC evidence lookup:
    # project_id -> {fr_document_number: best evidence row dict}
    nepatec_by_project: dict[str, dict[str, dict]] = defaultdict(dict)
    nepatec_types_by_project: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if nepatec_evidence is not None and not nepatec_evidence.empty:
        # Only use fr_doc_noi and fr_url evidence (passed proximity filter)
        valid_evidence = nepatec_evidence[
            nepatec_evidence["evidence_type"].isin(["fr_doc_noi", "fr_url"])
        ]
        for _, ev_row in valid_evidence.iterrows():
            pid = _normalize_project_id(ev_row.get("project_id"))
            doc_num = _normalize_text(ev_row.get("fr_document_number"))
            if not pid or not doc_num:
                continue
            evidence_type = _normalize_text(ev_row.get("evidence_type"))
            if evidence_type:
                nepatec_types_by_project[pid][doc_num].add(evidence_type)
            # Keep the best row per doc number (prefer main_document pages)
            existing = nepatec_by_project[pid].get(doc_num)
            if existing is None or (ev_row.get("main_document") and not existing.get("main_document")):
                nepatec_by_project[pid][doc_num] = ev_row.to_dict()

    project_rows = []
    all_candidate_rows = []
    review_rows = []

    if show_progress:
        print(
            f"[FR match] Starting offline matching: {project_total:,} projects x {len(corpus):,} corpus docs",
            flush=True,
        )

    for project_index, (_, row) in enumerate(projects_unique.iterrows(), start=1):
        project_id = _normalize_project_id(row.get("project_id"))
        title = _strip_title_prefixes(_normalize_text(row.get("project_title")))
        project_tokens = _distinctive_tokens(title)

        # Collect candidate corpus indices from token search
        candidate_counter: Counter[int] = Counter()
        for token in project_tokens:
            candidate_counter.update(corpus_index.get(token, set()))

        # Also include any corpus entries matched via NEPATEC doc number evidence
        project_nepatec = nepatec_by_project.get(project_id, {})
        project_nepatec_types = nepatec_types_by_project.get(project_id, {})
        nepatec_doc_numbers: frozenset = frozenset(project_nepatec.keys())
        for doc_num in nepatec_doc_numbers:
            if doc_num in corpus_doc_number_index:
                corpus_idx = corpus_doc_number_index[doc_num]
                candidate_counter[corpus_idx] = max(candidate_counter.get(corpus_idx, 0), 999)

        candidate_records = []
        for corpus_idx, _ in candidate_counter.most_common(75):
            candidate = _candidate_record(
                corpus.loc[corpus_idx], row,
                conservative=conservative,
                nepatec_doc_numbers=nepatec_doc_numbers,
                nepatec_evidence_types=project_nepatec_types,
            )
            if candidate is None:
                continue
            candidate_records.append(candidate)

        candidate_records = _sort_candidate_records(candidate_records)[:max_candidates_per_project]
        all_candidate_rows.extend(candidate_records)

        high_candidates = [rec for rec in candidate_records if rec["match_confidence"] == "high"]
        medium_candidates = [rec for rec in candidate_records if rec["match_confidence"] == "medium"]

        if high_candidates:
            top = high_candidates[0]
            if len(high_candidates) > 1 and (top["match_score"] - high_candidates[1]["match_score"]) < 10:
                # Multiple competing high-confidence direct-evidence candidates
                # Pick the one with the earlier publication date
                sorted_by_date = sorted(
                    high_candidates,
                    key=lambda c: c.get("fr_publication_date") or "9999-99-99",
                )
                top = sorted_by_date[0]
                ev_row = project_nepatec.get(top.get("fr_document_number") or "")
                output = _project_output_from_candidate(
                    row, top,
                    accepted=False,
                    status="ambiguous",
                    confidence="ambiguous",
                    candidate_count=len(candidate_records),
                    high_count=len(high_candidates),
                    reason="multiple_high_confidence_candidates",
                    nepatec_evidence_row=ev_row,
                )
                review_rows.extend(high_candidates[:max_candidates_per_project])
            else:
                ev_row = project_nepatec.get(top.get("fr_document_number") or "")
                output = _project_output_from_candidate(
                    row, top,
                    accepted=True,
                    status="accepted",
                    confidence="high",
                    candidate_count=len(candidate_records),
                    high_count=len(high_candidates),
                    reason=top["match_reason"],
                    nepatec_evidence_row=ev_row,
                )
        elif medium_candidates:
            top = medium_candidates[0]
            ev_row = project_nepatec.get(top.get("fr_document_number") or "")
            output = _project_output_from_candidate(
                row, top,
                accepted=False,
                status="review_required",
                confidence="medium",
                candidate_count=len(candidate_records),
                high_count=0,
                reason=top["match_reason"],
                nepatec_evidence_row=ev_row,
            )
            review_rows.extend(medium_candidates[:max_candidates_per_project])
        elif candidate_records:
            top = candidate_records[0]
            output = _project_output_from_candidate(
                row, top,
                accepted=False,
                status="unmatched",
                confidence="low",
                candidate_count=len(candidate_records),
                high_count=0,
                reason=top["match_reason"],
            )
        else:
            output = _empty_project_output(row)

        project_rows.append(output)
        if show_progress and (
            project_index == project_total or project_index % max(1, progress_interval) == 0
        ):
            print(
                (
                    f"[FR match] scored {project_index:,}/{project_total:,} projects; "
                    f"candidate rows={len(all_candidate_rows):,}; review rows={len(review_rows):,}; "
                    f"accepted={sum(r.get('noi_match_status') == 'accepted' for r in project_rows):,}"
                ),
                flush=True,
            )

    project_matches = pd.DataFrame(project_rows, columns=PROJECT_OUTPUT_COLUMNS)
    candidates = pd.DataFrame(all_candidate_rows, columns=CANDIDATE_OUTPUT_COLUMNS)
    review = pd.DataFrame(review_rows, columns=CANDIDATE_OUTPUT_COLUMNS)
    return project_matches, candidates, review


def build_project_noa_matches(
    projects: pd.DataFrame,
    noa_corpus: pd.DataFrame,
    *,
    max_candidates_per_project: int = 10,
    show_progress: bool = False,
    progress_interval: int = 5000,
    nepatec_evidence: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match projects to NOA (Notice of Availability) FR records.

    Mirrors build_project_noi_matches() but uses fr_doc_noa evidence and
    _classify_noa_candidate(). Returns (project_noa_matches, candidates, review)
    where project_noa_matches has project_id + noa_* columns only.
    """
    NOA_OUTPUT_COLUMNS = [
        "project_id",
        "noa_availability_date",
        "noa_document_number",
        "noa_url",
        "noa_fr_title",
        "noa_match_status",
        "noa_match_reason",
        "noa_match_score",
        "noa_title_overlap_count",
        "noa_title_overlap_tokens",
        "noa_date_evidence_type",
        "noa_nepatec_evidence_document_id",
        "noa_nepatec_evidence_file_name",
        "noa_nepatec_evidence_page_number",
    ]

    projects_unique = projects.drop_duplicates(subset=["project_id"], keep="first").copy()
    project_total = len(projects_unique)

    empty_noa = lambda row: {**{"project_id": _normalize_project_id(row.get("project_id"))}, **_empty_project_noa_output(row)}  # noqa: E731

    if noa_corpus.empty:
        project_rows = [empty_noa(row) for _, row in projects_unique.iterrows()]
        empty_candidates = pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)
        return pd.DataFrame(project_rows, columns=NOA_OUTPUT_COLUMNS), empty_candidates, empty_candidates.copy()

    corpus_index = _build_corpus_index(noa_corpus)
    corpus_doc_number_index = _build_corpus_doc_number_index(noa_corpus)

    # Build per-project NOA NEPATEC evidence lookup (fr_doc_noa only)
    nepatec_by_project: dict[str, dict[str, dict]] = defaultdict(dict)
    nepatec_types_by_project: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if nepatec_evidence is not None and not nepatec_evidence.empty:
        valid_evidence = nepatec_evidence[
            nepatec_evidence["evidence_type"].isin(["fr_doc_noa"])
        ]
        for _, ev_row in valid_evidence.iterrows():
            pid = _normalize_project_id(ev_row.get("project_id"))
            doc_num = _normalize_text(ev_row.get("fr_document_number"))
            if not pid or not doc_num:
                continue
            evidence_type = _normalize_text(ev_row.get("evidence_type"))
            if evidence_type:
                nepatec_types_by_project[pid][doc_num].add(evidence_type)
            existing = nepatec_by_project[pid].get(doc_num)
            if existing is None or (ev_row.get("main_document") and not existing.get("main_document")):
                nepatec_by_project[pid][doc_num] = ev_row.to_dict()

    project_rows = []
    all_candidate_rows = []
    review_rows = []

    if show_progress:
        print(
            f"[FR noa-match] Starting NOA matching: {project_total:,} projects x {len(noa_corpus):,} corpus docs",
            flush=True,
        )

    for project_index, (_, row) in enumerate(projects_unique.iterrows(), start=1):
        project_id = _normalize_project_id(row.get("project_id"))
        title = _strip_title_prefixes(_normalize_text(row.get("project_title")))
        project_tokens = _distinctive_tokens(title)

        candidate_counter: Counter[int] = Counter()
        for token in project_tokens:
            candidate_counter.update(corpus_index.get(token, set()))

        project_nepatec = nepatec_by_project.get(project_id, {})
        project_nepatec_types = nepatec_types_by_project.get(project_id, {})
        nepatec_doc_numbers: frozenset = frozenset(project_nepatec.keys())
        for doc_num in nepatec_doc_numbers:
            if doc_num in corpus_doc_number_index:
                corpus_idx = corpus_doc_number_index[doc_num]
                candidate_counter[corpus_idx] = max(candidate_counter.get(corpus_idx, 0), 999)

        candidate_records = []
        for corpus_idx, _ in candidate_counter.most_common(75):
            cand_row = noa_corpus.loc[corpus_idx]
            candidate_doc_number = _fr_document_number(cand_row)
            has_noa_evidence = bool(
                candidate_doc_number and nepatec_doc_numbers and candidate_doc_number in nepatec_doc_numbers
            )
            if not has_noa_evidence:
                continue  # NOA matching requires direct NEPATEC doc number evidence only
            metrics = _candidate_match_metrics(cand_row, row)
            confidence, reason = _classify_noa_candidate(
                cand_row,
                row,
                nepatec_doc_numbers=nepatec_doc_numbers,
                nepatec_evidence_types=project_nepatec_types,
            )
            if confidence == "low":
                continue
            score = _score_candidate(cand_row, row, metrics=metrics)
            candidate_records.append({
                "project_id": project_id,
                "fr_document_number": candidate_doc_number,
                "fr_title": _fr_title(cand_row),
                "fr_publication_date": _normalize_text(_field(cand_row, "fr_publication_date", "publication_date")),
                "fr_url": _fr_url(cand_row),
                "match_score": score,
                "match_confidence": confidence,
                "match_reason": reason,
                "title_overlap_count": metrics["title_overlap_count"],
                "title_overlap_tokens": ", ".join(metrics["title_overlap_tokens"]),
                "process_conflict": metrics["process_conflict"],
                "nepatec_fr_document_number_evidence": True,
            })

        candidate_records = sorted(
            candidate_records,
            key=lambda r: (-{"high": 3, "medium": 2, "low": 1}.get(r["match_confidence"], 0), -r["match_score"]),
        )[:max_candidates_per_project]
        all_candidate_rows.extend(candidate_records)

        high_candidates = [r for r in candidate_records if r["match_confidence"] == "high"]
        medium_candidates = [r for r in candidate_records if r["match_confidence"] == "medium"]

        if high_candidates:
            top = high_candidates[0]
            ev_row = project_nepatec.get(top.get("fr_document_number") or "")
            noa_out = _project_noa_output_from_candidate(
                row, top,
                accepted=True,
                status="accepted",
                reason=top["match_reason"],
                nepatec_evidence_row=ev_row,
            )
        elif medium_candidates:
            top = medium_candidates[0]
            ev_row = project_nepatec.get(top.get("fr_document_number") or "")
            noa_out = _project_noa_output_from_candidate(
                row, top,
                accepted=False,
                status="review_required",
                reason=top["match_reason"],
                nepatec_evidence_row=ev_row,
            )
            review_rows.extend(medium_candidates[:max_candidates_per_project])
        else:
            noa_out = _empty_project_noa_output(row)

        project_rows.append({"project_id": project_id, **noa_out})

        if show_progress and (
            project_index == project_total or project_index % max(1, progress_interval) == 0
        ):
            print(
                f"[FR noa-match] scored {project_index:,}/{project_total:,} projects; "
                f"accepted={sum(r.get('noa_availability_date') is not None for r in project_rows):,}",
                flush=True,
            )

    noa_output_df = pd.DataFrame(project_rows, columns=NOA_OUTPUT_COLUMNS)
    candidates_df = pd.DataFrame(all_candidate_rows)
    if candidates_df.empty:
        candidates_df = pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)
    review_df = pd.DataFrame(review_rows)
    if review_df.empty:
        review_df = pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)
    return noa_output_df, candidates_df, review_df


def _build_noa_title_search_term(project_title: str) -> Optional[str]:
    """Return an FR title search phrase for NOA lookup, or None if < 3 distinctive tokens.

    Requires at least 3 distinctive tokens to avoid excessively broad searches
    that would produce low-precision NOA matches without direct doc evidence.
    """
    title = _strip_title_prefixes(_normalize_text(project_title))
    if len(_distinctive_tokens(title)) < 3:
        return None
    return _select_title_phrase(title)


def _search_noa_by_title_cached(
    terms: str,
    min_date: str,
    cache: dict,
    *,
    throttle_seconds: float = 0.25,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.5,
) -> list[dict]:
    """Query FR for NOA notices by title keywords anchored by min_date. Cached.

    Cache key: ``noa_title_search|{terms}|{min_date}``
    Returns a list of raw result dicts from the FR API (may be empty).
    """
    cache_key = f"noa_title_search|{terms}|{min_date}"
    if cache_key in cache:
        return cache[cache_key] or []
    try:
        response = search_noi(
            terms,
            min_date,
            None,  # end_date=None → today
            per_page=20,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            page=1,
        )
        results = response.get("results") or []
    except Exception as exc:
        print(f"[FR noa-title] Warning: search error for '{terms}': {exc}", flush=True)
        results = []
    cache[cache_key] = results
    time.sleep(throttle_seconds)
    return results


_FEIS_TITLE_RE = re.compile(
    r"\b(?:final\s+(?:supplemental\s+)?environmental\s+impact\s+statement|final\s+(?:supplemental\s+)?eis)\b",
    re.IGNORECASE,
)


def _supplement_noa_by_title_search(
    noa_matches: pd.DataFrame,
    projects: pd.DataFrame,
    project_matches: pd.DataFrame,
    *,
    throttle_seconds: float = 0.25,
    cache: dict,
    cache_path: Optional[Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Supplement NOA matches for unmatched EIS projects by FR title search.

    For EIS projects where build_project_noa_matches() found no direct fr_doc_noa
    evidence, attempt a targeted FR API search using the project title and a
    date window anchored by noi_publication_date (NOI date + 365 days → today).

    Eligibility criteria:
    - EIS project (EA/CE excluded — FONSI matching is less reliable without
      direct doc evidence)
    - noi_publication_date is known (provides temporal anchor)
    - Project title has >= 3 distinctive tokens (prevents overly broad searches)

    Acceptance criteria (all must pass):
    - FR record has an FEIS/FSEIS-type title (_is_noa_title + FEIS regex)
    - Title token overlap >= max(_required_title_overlap(n), min(n, 3))
      (more conservative than the direct-evidence path)
    - Not a termination/withdrawal notice

    Provenance: noa_date_evidence_type = "fr_title_search_noi_anchored"
    """
    # Build lookup: project_id → noi_publication_date
    noi_dates: dict[str, str] = {}
    if "noi_publication_date" in project_matches.columns:
        for _, pm_row in project_matches.iterrows():
            pid = _normalize_project_id(pm_row.get("project_id"))
            noi_date_val = _normalize_text(pm_row.get("noi_publication_date"))
            if pid and noi_date_val:
                noi_dates[pid] = noi_date_val

    # Build lookup: project_id → {project_title, process_type}
    project_info: dict[str, dict] = {}
    for _, p_row in projects.drop_duplicates(subset=["project_id"]).iterrows():
        pid = _normalize_project_id(p_row.get("project_id"))
        if pid:
            project_info[pid] = {
                "project_title": _normalize_text(p_row.get("project_title")),
                "process_type": _normalize_text(p_row.get("process_type")).upper(),
            }

    fetch_run_at = datetime.now(timezone.utc).isoformat()
    noa_matches = noa_matches.copy()
    attempted = 0
    updated_count = 0

    for idx, noa_row in noa_matches.iterrows():
        if pd.notna(noa_row.get("noa_availability_date")):
            continue  # already matched

        pid = _normalize_project_id(noa_row.get("project_id"))
        info = project_info.get(pid, {})
        process_type = info.get("process_type", "")

        if process_type != "EIS":
            continue

        noi_date_str = noi_dates.get(pid)
        if not noi_date_str:
            continue

        project_title = info.get("project_title", "")
        search_term = _build_noa_title_search_term(project_title)
        if not search_term:
            continue

        try:
            noi_date = date.fromisoformat(noi_date_str)
        except (ValueError, TypeError):
            continue
        min_date = (noi_date + timedelta(days=365)).isoformat()

        attempted += 1
        results = _search_noa_by_title_cached(
            search_term,
            min_date,
            cache,
            throttle_seconds=throttle_seconds,
        )
        if not results:
            continue

        # Minimal project row for _candidate_match_metrics; fields absent here
        # (lead_agency, project_state, project_sponsor) fall back to "" in helpers.
        project_row = pd.Series({
            "project_id": pid,
            "project_title": project_title,
            "process_type": process_type,
        })

        best_item: Optional[dict] = None
        best_metrics: Optional[dict] = None
        best_overlap = -1

        for raw_item in results:
            fr_title = _normalize_text(raw_item.get("title", ""))
            if not fr_title:
                continue
            if not _is_noa_title(fr_title):
                continue
            if _REJECT_NOTICE_RE.search(fr_title):
                continue
            # EIS → FEIS/FSEIS only (no FONSI — that is for EA projects)
            if not _FEIS_TITLE_RE.search(fr_title):
                continue

            norm_item = _normalize_fr_document(raw_item, "noa_title_search", fetch_run_at)
            metrics = _candidate_match_metrics(norm_item, project_row)
            n = metrics["project_token_count"]
            # More conservative threshold than direct-evidence path:
            # require max(standard, min(n, 3)) overlapping tokens.
            required = max(_required_title_overlap(n), min(n, 3))
            if metrics["title_overlap_count"] < required:
                continue

            if metrics["title_overlap_count"] > best_overlap:
                best_overlap = metrics["title_overlap_count"]
                best_item = norm_item
                best_metrics = metrics

        if best_item is None or best_metrics is None:
            continue

        pub_date = _normalize_text(best_item.get("fr_publication_date")) or None
        if not pub_date:
            continue

        noa_matches.at[idx, "noa_availability_date"] = pub_date
        noa_matches.at[idx, "noa_document_number"] = _normalize_text(best_item.get("fr_document_number")) or None
        noa_matches.at[idx, "noa_url"] = _normalize_text(best_item.get("fr_url")) or None
        noa_matches.at[idx, "noa_fr_title"] = _normalize_text(best_item.get("fr_title")) or None
        noa_matches.at[idx, "noa_match_status"] = "accepted"
        noa_matches.at[idx, "noa_match_reason"] = "fr_title_search_noi_anchored"
        noa_matches.at[idx, "noa_match_score"] = _score_candidate(best_item, project_row, metrics=best_metrics)
        noa_matches.at[idx, "noa_title_overlap_count"] = best_metrics["title_overlap_count"]
        noa_matches.at[idx, "noa_title_overlap_tokens"] = ", ".join(best_metrics["title_overlap_tokens"])
        noa_matches.at[idx, "noa_date_evidence_type"] = "fr_title_search_noi_anchored"
        noa_matches.at[idx, "noa_nepatec_evidence_document_id"] = None
        noa_matches.at[idx, "noa_nepatec_evidence_file_name"] = None
        noa_matches.at[idx, "noa_nepatec_evidence_page_number"] = None
        updated_count += 1

        if cache_path and attempted % 25 == 0:
            _save_cache(cache_path, cache)

    if cache_path:
        _save_cache(cache_path, cache)

    if show_progress:
        print(
            f"[FR noa-title] Title search: attempted={attempted:,}, "
            f"supplemented={updated_count:,} unmatched EIS projects",
            flush=True,
        )

    return noa_matches


def enrich_projects_with_noi(
    projects: pd.DataFrame,
    config: FederalRegisterConfig,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    df = projects.copy()
    if config.process_types:
        df = df[df["process_type"].isin(config.process_types)]
    if config.energy_types:
        df = df[df["project_energy_type"].isin(config.energy_types)]
    if config.sample_n:
        sample_n = min(config.sample_n, len(df))
        df = df.sample(n=sample_n, random_state=config.random_state)

    corpus = fetch_federal_register_noi_corpus(config, cache_path=cache_path)
    project_matches, _, _ = build_project_noi_matches(
        df,
        corpus,
        conservative=config.conservative,
        max_candidates_per_project=config.max_candidates_per_project,
        show_progress=config.show_progress,
        progress_interval=config.progress_project_interval,
    )
    return project_matches


def attach_noi_fields(
    projects: pd.DataFrame,
    config: FederalRegisterConfig,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    noi_df = enrich_projects_with_noi(projects, config=config, cache_path=cache_path)
    return projects.merge(noi_df.drop(columns=["project_title"], errors="ignore"), on="project_id", how="left")


def refresh_federal_register_noi(
    projects: pd.DataFrame,
    *,
    analysis_dir: Path,
    throttle_seconds: float = 0.25,
    output_path: Optional[Path] = None,
    corpus_output: Optional[Path] = None,
    candidates_output: Optional[Path] = None,
    noa_corpus_output: Optional[Path] = None,
    noa_candidates_output: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    max_candidates_per_project: int = 10,
    evidence_output: Optional[Path] = None,
    rescan_nepatec_evidence: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Refresh Federal Register NOI enrichment using direct doc-number fetches.

    Flow:
        1. Scan NEPATEC pages for FR Doc numbers (or load cached evidence).
        2. Collect unique doc numbers from fr_doc_noi + fr_url evidence rows.
        3. Direct-fetch each doc number from the FR API (one call per number).
        4. Match projects to fetched FR records: accept only when doc number
           AND >= 2 normalized title tokens both agree.
        5. Write all output artifacts and return the project-level match table.
    """
    analysis_dir = Path(analysis_dir)
    fr_dir = analysis_dir / "federal_register"
    fr_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path) if output_path else fr_dir / DEFAULT_PROJECT_OUTPUT.name
    corpus_output = Path(corpus_output) if corpus_output else fr_dir / DEFAULT_NOI_CORPUS_OUTPUT.name
    candidates_output = Path(candidates_output) if candidates_output else fr_dir / DEFAULT_NOI_CANDIDATES_OUTPUT.name
    noa_corpus_output = Path(noa_corpus_output) if noa_corpus_output else fr_dir / DEFAULT_NOA_CORPUS_OUTPUT.name
    noa_candidates_output = Path(noa_candidates_output) if noa_candidates_output else fr_dir / DEFAULT_NOA_CANDIDATES_OUTPUT.name
    cache_path = Path(cache_path) if cache_path else fr_dir / "fr_noi_cache.json"

    # ── Step 1: NEPATEC page scan (or load cache) ───────────────────────────
    evidence_path = Path(evidence_output) if evidence_output else fr_dir / "nepatec_fr_evidence.parquet"
    if evidence_path.exists() and not rescan_nepatec_evidence:
        print(f"Loading cached NEPATEC FR evidence: {evidence_path}", flush=True)
        nepatec_evidence = pd.read_parquet(evidence_path)
        print(f"Loaded {len(nepatec_evidence):,} evidence rows from cache", flush=True)
    else:
        nepatec_evidence = extract_nepatec_federal_register_evidence(
            projects,
            analysis_dir=analysis_dir,
            evidence_output=evidence_path,
            show_progress=show_progress,
        )

    # ── Step 2: Collect unique doc numbers from valid evidence (NOI + NOA) ──
    if not nepatec_evidence.empty:
        noi_valid_evidence = nepatec_evidence[
            nepatec_evidence["evidence_type"].isin(["fr_doc_noi", "fr_url"])
        ]
        noa_valid_evidence = nepatec_evidence[
            nepatec_evidence["evidence_type"].isin(["fr_doc_noa"])
        ]
    else:
        noi_valid_evidence = nepatec_evidence
        noa_valid_evidence = nepatec_evidence

    noi_doc_numbers = set(
        _normalize_text(v) for v in noi_valid_evidence["fr_document_number"].dropna() if _normalize_text(v)
    )
    noa_doc_numbers = set(
        _normalize_text(v) for v in noa_valid_evidence["fr_document_number"].dropna() if _normalize_text(v)
    )
    all_doc_numbers_to_fetch = sorted(noi_doc_numbers | noa_doc_numbers)
    print(
        f"[FR direct] {len(all_doc_numbers_to_fetch):,} unique doc numbers to fetch "
        f"({len(noi_doc_numbers):,} NOI/URL + {len(noa_doc_numbers):,} NOA, "
        f"{len(noi_doc_numbers & noa_doc_numbers):,} shared)",
        flush=True,
    )

    # ── Step 3: Direct-fetch all doc numbers; split by title type ───────────
    all_fetched = fetch_documents_by_doc_numbers(
        all_doc_numbers_to_fetch,
        throttle_seconds=throttle_seconds,
        cache_path=cache_path,
        show_progress=show_progress,
    )
    corpus = all_fetched[all_fetched["fr_title"].apply(_is_noi_title)] if not all_fetched.empty else all_fetched
    noa_corpus = all_fetched[all_fetched["fr_title"].apply(_is_noa_title)] if not all_fetched.empty else all_fetched

    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(corpus_output, index=False)
    noa_corpus_output.parent.mkdir(parents=True, exist_ok=True)
    noa_corpus.to_parquet(noa_corpus_output, index=False)
    print(f"Saved NOI corpus: {corpus_output} ({len(corpus):,} documents)", flush=True)
    print(f"Saved NOA corpus: {noa_corpus_output} ({len(noa_corpus):,} documents)", flush=True)
    legacy_fetch_report_path = fr_dir / DEFAULT_FETCH_REPORT_OUTPUT.name
    if legacy_fetch_report_path.exists():
        legacy_fetch_report_path.unlink()
        print(
            f"Removed legacy keyword fetch report (not used by direct-fetch workflow): "
            f"{legacy_fetch_report_path}",
            flush=True,
        )

    # ── Step 4: Match NOI + NOA ──────────────────────────────────────────────
    project_matches, candidates, review = build_project_noi_matches(
        projects,
        corpus,
        max_candidates_per_project=max_candidates_per_project,
        show_progress=show_progress,
        nepatec_evidence=nepatec_evidence,
    )
    noa_matches, noa_candidates, noa_review = build_project_noa_matches(
        projects,
        noa_corpus,
        max_candidates_per_project=max_candidates_per_project,
        show_progress=show_progress,
        nepatec_evidence=nepatec_evidence,
    )

    # ── Step 4b: Supplement unmatched EIS NOA by title search ────────────────
    # For EIS projects with noi_publication_date but no fr_doc_noa NEPATEC
    # evidence, search the FR API by project title keywords anchored to the
    # NOI date window. This recovers coverage for projects whose FEIS bodies
    # cannot self-cite their own FR doc number (assigned only at publication).
    noa_supplement_cache = _load_cache(cache_path)
    noa_matches = _supplement_noa_by_title_search(
        noa_matches,
        projects,
        project_matches,
        throttle_seconds=throttle_seconds,
        cache=noa_supplement_cache,
        cache_path=cache_path,
        show_progress=show_progress,
    )

    # Merge authoritative NOA columns into the project output. The NOI match
    # frame carries empty placeholder noa_* columns, so drop them first to avoid
    # pandas _x/_y suffixes in the persisted artifact.
    project_matches = project_matches.drop(
        columns=[col for col in project_matches.columns if col.startswith("noa_")],
        errors="ignore",
    )
    project_matches = project_matches.merge(noa_matches, on="project_id", how="left")

    # ── Step 5: Write outputs ────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    noa_candidates_output.parent.mkdir(parents=True, exist_ok=True)

    project_matches.to_parquet(output_path, index=False)
    candidates.to_parquet(candidates_output, index=False)
    noa_candidates.to_parquet(noa_candidates_output, index=False)
    print(f"Saved NOA candidates: {noa_candidates_output} ({len(noa_candidates):,} rows)")
    focused_review_counts = write_focused_manual_review_exports(
        project_matches,
        review,
        projects,
        output_dir=candidates_output.parent,
    )
    noa_review_path = candidates_output.parent / DEFAULT_NOA_REVIEW_OUTPUT.name
    noa_review.to_csv(noa_review_path, index=False)
    print(f"Saved NOA review: {noa_review_path} ({len(noa_review):,} rows)")

    print(f"Saved project NOI+NOA matches: {output_path} ({len(project_matches):,} projects)")
    print(f"Saved NOI candidates: {candidates_output} ({len(candidates):,} rows)")
    for path, row_count in focused_review_counts.items():
        print(f"Saved focused review: {path} ({row_count:,} rows)")
    print(f"NEPATEC FR evidence rows: {len(nepatec_evidence):,}")
    _print_coverage_report(project_matches.merge(
        projects[["project_id", "process_type", "project_energy_type"]],
        on="project_id",
        how="left",
    ))
    return project_matches


def _print_coverage_report(project_matches: pd.DataFrame) -> None:
    if project_matches.empty:
        print("Federal Register NOI/NOA coverage: no project rows")
        return
    print("\n=== Federal Register NOI Coverage ===")
    print(f"Projects: {len(project_matches):,}")
    print(f"Accepted NOI dates: {project_matches['noi_publication_date'].notna().sum():,}")
    if {"process_type", "project_energy_type"}.issubset(project_matches.columns):
        summary = project_matches.groupby(["process_type", "project_energy_type"], dropna=False)[
            "noi_publication_date"
        ].agg(rows="size", accepted=lambda s: int(s.notna().sum()))
        print(summary.to_string())
    if "noi_match_status" in project_matches.columns:
        print("\nNOI match statuses:")
        print(project_matches["noi_match_status"].value_counts(dropna=False).to_string())
    if "noa_availability_date" in project_matches.columns:
        print("\n=== Federal Register NOA Coverage ===")
        print(f"Accepted NOA dates: {project_matches['noa_availability_date'].notna().sum():,}")
        if {"process_type", "project_energy_type"}.issubset(project_matches.columns):
            noa_summary = project_matches.groupby(["process_type", "project_energy_type"], dropna=False)[
                "noa_availability_date"
            ].agg(rows="size", accepted=lambda s: int(s.notna().sum()))
            print(noa_summary.to_string())
        if "noa_match_status" in project_matches.columns:
            print("\nNOA match statuses:")
            print(project_matches["noa_match_status"].value_counts(dropna=False).to_string())


def _parse_list_arg(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _sample_arg(value: Optional[int]) -> Optional[int]:
    if value is None or value <= 0:
        return None
    return value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Federal Register NOI enrichment (direct doc-number fetch)"
    )
    parser.add_argument(
        "--projects-path",
        default=str(ANALYSIS_DIR / "projects_combined.parquet"),
        help="Path to projects_combined.parquet",
    )
    parser.add_argument("--output", default=str(DEFAULT_PROJECT_OUTPUT), help="Project-level NOI+NOA output parquet")
    parser.add_argument("--corpus-output", default=str(DEFAULT_NOI_CORPUS_OUTPUT), help="Directly-fetched NOI FR documents parquet")
    parser.add_argument("--candidates-output", default=str(DEFAULT_NOI_CANDIDATES_OUTPUT), help="All scored NOI candidates parquet")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="FR API response cache JSON")
    parser.add_argument("--evidence-output", default=str(DEFAULT_EVIDENCE_OUTPUT), help="NEPATEC page scan evidence parquet")
    parser.add_argument("--all-projects", action="store_true", help="Include all projects; ignore process/energy filters")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size (0 = all)")
    parser.add_argument("--process-types", default="", help="Comma-separated process types; ignored with --all-projects")
    parser.add_argument("--energy-types", default="", help="Comma-separated energy types; ignored with --all-projects")
    parser.add_argument("--throttle-seconds", type=float, default=0.25, help="Delay between FR API calls (default 0.25)")
    parser.add_argument("--max-candidates-per-project", type=int, default=10)
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress progress logs")
    parser.add_argument("--report-n", type=int, default=10, help="Examples to print at end")
    parser.add_argument("--rescan-nepatec-evidence", action="store_true", help="Force fresh NEPATEC page scan even if cache exists")

    args = parser.parse_args()

    projects = pd.read_parquet(args.projects_path)
    if not args.all_projects:
        process_types = _parse_list_arg(args.process_types)
        energy_types = _parse_list_arg(args.energy_types)
        if process_types:
            projects = projects[projects["process_type"].isin(process_types)]
        if energy_types:
            projects = projects[projects["project_energy_type"].isin(energy_types)]
    sample_n = _sample_arg(args.sample)
    if sample_n:
        projects = projects.sample(n=min(sample_n, len(projects)), random_state=7)

    project_matches = refresh_federal_register_noi(
        projects,
        analysis_dir=ANALYSIS_DIR,
        throttle_seconds=args.throttle_seconds,
        output_path=Path(args.output),
        corpus_output=Path(args.corpus_output),
        candidates_output=Path(args.candidates_output),
        cache_path=Path(args.cache_path),
        max_candidates_per_project=args.max_candidates_per_project,
        evidence_output=Path(args.evidence_output),
        rescan_nepatec_evidence=args.rescan_nepatec_evidence,
        show_progress=not args.quiet_progress,
    )

    report_n = max(0, args.report_n)
    if report_n:
        accepted = project_matches[project_matches["noi_publication_date"].notna()]
        print(f"\nTop {report_n} accepted matches:")
        print(
            accepted.sort_values("noi_match_score", ascending=False)
            .head(report_n)[["project_id", "noi_publication_date", "noi_document_number", "noi_match_score", "noi_url"]]
        )


if __name__ == "__main__":
    main()
