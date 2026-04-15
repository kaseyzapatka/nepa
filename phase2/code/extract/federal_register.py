"""
Federal Register NOI enrichment for NEPATEC projects.

Phase 2 treats this as a refreshable, standalone data source. The default
extract_data.py path remains offline and only merges an existing
noi_federal_register.parquet artifact unless a refresh is explicitly requested.
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
from datetime import date, datetime, timezone
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

DEFAULT_CORPUS_OUTPUT = FEDERAL_REGISTER_DIR / "fr_noi_documents.parquet"
DEFAULT_CANDIDATES_OUTPUT = FEDERAL_REGISTER_DIR / "project_noi_candidates.parquet"
DEFAULT_REVIEW_OUTPUT = FEDERAL_REGISTER_DIR / "project_noi_manual_review.csv"
DEFAULT_PROJECT_OUTPUT = FEDERAL_REGISTER_DIR / "noi_federal_register.parquet"
DEFAULT_CACHE_PATH = FEDERAL_REGISTER_DIR / "fr_noi_cache.json"
DEFAULT_FETCH_REPORT_OUTPUT = FEDERAL_REGISTER_DIR / "fr_noi_fetch_report.csv"

NOI_CORPUS_QUERIES = (
    '"Notice of Intent"',
    '"Intent To Prepare"',
    '"Notice To Prepare"',
    '"Notice of Preparation"',
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
    r"\b(?:notice\s+of\s+intent|intent\s+to\s+prepare|notice\s+to\s+prepare|notice\s+of\s+preparation)\b",
    re.IGNORECASE,
)
_REJECT_NOTICE_RE = re.compile(r"\b(?:termination|withdrawals?|cancel(?:lation|ed)?)\b", re.IGNORECASE)
_FR_CITATION_RE = re.compile(r"\b\d+\s*FR\s*\d+\b", re.IGNORECASE)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE_PATTERN = rf"(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}"


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


def _is_valid_noi_title(title: str) -> bool:
    title_lower = _normalize_text(title).lower()
    if not _NOI_LIKE_RE.search(title_lower):
        return False
    if _REJECT_NOTICE_RE.search(title_lower):
        return False
    return True


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


def _classify_candidate(item: object, row: pd.Series, conservative: bool = True) -> tuple[str, str]:
    metrics = _candidate_match_metrics(item, row)
    contextual = metrics["agency_match"] or metrics["state_match"] or metrics["sponsor_match"]
    if _REJECT_NOTICE_RE.search(_fr_title(item)):
        if metrics["document_number_evidence"] or metrics["fr_citation_evidence"]:
            return "medium", "termination_or_withdrawal_notice_requires_review"
        return "low", "termination_or_withdrawal_notice_rejected"
    if metrics["document_number_evidence"] or metrics["fr_citation_evidence"]:
        return "high", "document_number_or_fr_citation_evidence"
    if _is_ce_project(row) and metrics["title_overlap_count"] < 4:
        if metrics["title_overlap_count"] >= 2 and contextual and not metrics["process_conflict"]:
            return "medium", "ce_match_requires_distinctive_token_review"
        return "low", "ce_low_distinctive_title_overlap"
    if (
        metrics["title_overlap_count"] >= 3
        and metrics["title_containment_ratio"] >= 0.60
        and contextual
        and not metrics["process_conflict"]
    ):
        return "high", "strong_title_overlap_with_context"
    if metrics["exact_phrase_match"] and metrics["title_overlap_count"] >= 2 and not metrics["process_conflict"]:
        return "high", "exact_phrase_match"
    if metrics["title_overlap_count"] >= 2 and contextual and not metrics["process_conflict"]:
        return "medium", "moderate_title_overlap_with_context"
    if not conservative and metrics["title_overlap_count"] >= 2 and not metrics["process_conflict"]:
        return "medium", "moderate_title_overlap"
    return "low", "weak_or_contextless_match"


def _candidate_record(item: object, row: pd.Series, conservative: bool = True) -> Optional[dict]:
    title = _fr_title(item)
    if not title:
        return None
    if not _is_valid_candidate_title(title):
        return None

    metrics = _candidate_match_metrics(item, row)
    if not _passes_candidate_threshold(metrics):
        return None

    confidence, reason = _classify_candidate(item, row, conservative=conservative)
    score = _score_candidate(item, row, metrics=metrics)
    return {
        "project_id": row.get("project_id"),
        "project_title": row.get("project_title"),
        "process_type": row.get("process_type"),
        "project_energy_type": row.get("project_energy_type"),
        "fr_document_number": _fr_document_number(item),
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
        "project_id": row.get("project_id"),
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
) -> dict:
    return {
        "project_id": row.get("project_id"),
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
    }


def build_project_noi_matches(
    projects: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    conservative: bool = True,
    max_candidates_per_project: int = 10,
    show_progress: bool = False,
    progress_interval: int = 5000,
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
    project_rows = []
    all_candidate_rows = []
    review_rows = []

    if show_progress:
        print(
            f"[FR match] Starting offline matching: {project_total:,} projects x {len(corpus):,} corpus docs",
            flush=True,
        )

    for project_index, (_, row) in enumerate(projects_unique.iterrows(), start=1):
        title = _strip_title_prefixes(_normalize_text(row.get("project_title")))
        project_tokens = _distinctive_tokens(title)
        candidate_counter: Counter[int] = Counter()
        for token in project_tokens:
            candidate_counter.update(corpus_index.get(token, set()))

        candidate_records = []
        for corpus_idx, _ in candidate_counter.most_common(75):
            candidate = _candidate_record(corpus.loc[corpus_idx], row, conservative=conservative)
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
                output = _project_output_from_candidate(
                    row,
                    top,
                    accepted=False,
                    status="ambiguous",
                    confidence="ambiguous",
                    candidate_count=len(candidate_records),
                    high_count=len(high_candidates),
                    reason="multiple_high_confidence_candidates",
                )
                review_rows.extend(high_candidates[:max_candidates_per_project])
            else:
                output = _project_output_from_candidate(
                    row,
                    top,
                    accepted=True,
                    status="accepted",
                    confidence="high",
                    candidate_count=len(candidate_records),
                    high_count=len(high_candidates),
                    reason=top["match_reason"],
                )
        elif medium_candidates:
            top = medium_candidates[0]
            output = _project_output_from_candidate(
                row,
                top,
                accepted=False,
                status="review_required",
                confidence="medium",
                candidate_count=len(candidate_records),
                high_count=0,
                reason=top["match_reason"],
            )
            review_rows.extend(medium_candidates[:max_candidates_per_project])
        elif candidate_records:
            top = candidate_records[0]
            output = _project_output_from_candidate(
                row,
                top,
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
                    f"accepted={sum(row.get('noi_match_status') == 'accepted' for row in project_rows):,}"
                ),
                flush=True,
            )

    project_matches = pd.DataFrame(project_rows, columns=PROJECT_OUTPUT_COLUMNS)
    candidates = pd.DataFrame(all_candidate_rows, columns=CANDIDATE_OUTPUT_COLUMNS)
    review = pd.DataFrame(review_rows, columns=CANDIDATE_OUTPUT_COLUMNS)
    return project_matches, candidates, review


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
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    fetch_raw_text: bool = False,
    conservative: bool = True,
    output_path: Optional[Path] = None,
    corpus_output: Optional[Path] = None,
    candidates_output: Optional[Path] = None,
    review_output: Optional[Path] = None,
    fetch_report_output: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    max_candidates_per_project: int = 10,
) -> pd.DataFrame:
    analysis_dir = Path(analysis_dir)
    fr_dir = analysis_dir / "federal_register"
    fr_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path) if output_path else fr_dir / "noi_federal_register.parquet"
    corpus_output = Path(corpus_output) if corpus_output else fr_dir / "fr_noi_documents.parquet"
    candidates_output = Path(candidates_output) if candidates_output else fr_dir / "project_noi_candidates.parquet"
    review_output = Path(review_output) if review_output else fr_dir / "project_noi_manual_review.csv"
    fetch_report_output = Path(fetch_report_output) if fetch_report_output else fr_dir / "fr_noi_fetch_report.csv"
    cache_path = Path(cache_path) if cache_path else fr_dir / "fr_noi_cache.json"

    config = FederalRegisterConfig(
        process_types=tuple(),
        energy_types=tuple(),
        sample_n=None,
        start_date=start_date,
        end_date=end_date,
        fetch_raw_text=fetch_raw_text,
        conservative=conservative,
        max_candidates_per_project=max_candidates_per_project,
    )
    corpus = fetch_federal_register_noi_corpus(
        config,
        cache_path=cache_path,
        fetch_report_output=fetch_report_output,
    )
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(corpus_output, index=False)
    print(f"Saved Federal Register corpus: {corpus_output} ({len(corpus):,} documents)")

    project_matches, candidates, review = build_project_noi_matches(
        projects,
        corpus,
        conservative=conservative,
        max_candidates_per_project=max_candidates_per_project,
        show_progress=config.show_progress,
        progress_interval=config.progress_project_interval,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    review_output.parent.mkdir(parents=True, exist_ok=True)

    project_matches.to_parquet(output_path, index=False)
    candidates.to_parquet(candidates_output, index=False)
    review.to_csv(review_output, index=False)

    print(f"Saved Federal Register project matches: {output_path} ({len(project_matches):,} projects)")
    print(f"Saved Federal Register candidates: {candidates_output} ({len(candidates):,} candidates)")
    print(f"Saved Federal Register manual review packet: {review_output} ({len(review):,} rows)")
    _print_coverage_report(project_matches.merge(
        projects[["project_id", "process_type", "project_energy_type"]],
        on="project_id",
        how="left",
    ))
    return project_matches


def _print_coverage_report(project_matches: pd.DataFrame) -> None:
    if project_matches.empty:
        print("Federal Register NOI coverage: no project rows")
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
        print("\nMatch statuses:")
        print(project_matches["noi_match_status"].value_counts(dropna=False).to_string())


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

    parser = argparse.ArgumentParser(description="Federal Register NOI enrichment")
    parser.add_argument(
        "--projects-path",
        default=str(ANALYSIS_DIR / "projects_combined.parquet"),
        help="Path to projects_combined.parquet",
    )
    parser.add_argument("--output", default=str(DEFAULT_PROJECT_OUTPUT))
    parser.add_argument("--corpus-output", default=str(DEFAULT_CORPUS_OUTPUT))
    parser.add_argument("--candidates-output", default=str(DEFAULT_CANDIDATES_OUTPUT))
    parser.add_argument("--review-output", default=str(DEFAULT_REVIEW_OUTPUT))
    parser.add_argument("--fetch-report-output", default=str(DEFAULT_FETCH_REPORT_OUTPUT))
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--all-projects", action="store_true", help="Include all projects, process types, and energy types")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size; 0 means all")
    parser.add_argument("--process-types", default="", help="Comma-separated process types; ignored with --all-projects")
    parser.add_argument("--energy-types", default="", help="Comma-separated energy types; ignored with --all-projects")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--fetch-raw-text", action="store_true", help="Fetch raw text for scoping date extraction")
    parser.add_argument("--balanced", action="store_true", help="Use less conservative medium-match classification")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--throttle-seconds", type=float, default=0.25)
    parser.add_argument("--max-candidates-per-project", type=int, default=10)
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress Federal Register progress logs")
    parser.add_argument("--report-n", type=int, default=10, help="Examples to print")

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

    config = FederalRegisterConfig(
        process_types=tuple(),
        energy_types=tuple(),
        sample_n=None,
        per_page=args.per_page,
        throttle_seconds=args.throttle_seconds,
        start_date=args.start_date,
        end_date=args.end_date,
        fetch_raw_text=args.fetch_raw_text,
        conservative=not args.balanced,
        max_candidates_per_project=args.max_candidates_per_project,
        show_progress=not args.quiet_progress,
    )

    corpus = fetch_federal_register_noi_corpus(
        config,
        cache_path=Path(args.cache_path) if args.cache_path else None,
        fetch_report_output=Path(args.fetch_report_output) if args.fetch_report_output else None,
    )
    corpus_path = Path(args.corpus_output)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(corpus_path, index=False)
    print(f"Saved Federal Register corpus: {corpus_path} ({len(corpus):,} documents)")

    project_matches, candidates, review = build_project_noi_matches(
        projects,
        corpus,
        conservative=config.conservative,
        max_candidates_per_project=config.max_candidates_per_project,
        show_progress=config.show_progress,
        progress_interval=config.progress_project_interval,
    )

    output_path = Path(args.output)
    candidates_path = Path(args.candidates_output)
    review_path = Path(args.review_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    project_matches.to_parquet(output_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    review.to_csv(review_path, index=False)

    print(f"Saved Federal Register project matches: {output_path} ({len(project_matches):,} projects)")
    print(f"Saved Federal Register candidates: {candidates_path} ({len(candidates):,} candidates)")
    print(f"Saved Federal Register manual review packet: {review_path} ({len(review):,} rows)")

    _print_coverage_report(project_matches.merge(
        projects[["project_id", "process_type", "project_energy_type"]],
        on="project_id",
        how="left",
    ))

    report_n = max(0, args.report_n)
    if report_n:
        accepted = project_matches[project_matches["noi_publication_date"].notna()]
        print(f"\nTop {report_n} accepted matches:")
        print(
            accepted.sort_values("noi_match_score", ascending=False)
            .head(report_n)[["project_id", "noi_publication_date", "noi_document_number", "noi_match_score", "noi_url"]]
        )
        print(f"\nFirst {report_n} manual review rows:")
        print(review.head(report_n)[["project_id", "fr_document_number", "match_confidence", "match_score", "fr_title"]])


if __name__ == "__main__":
    main()
