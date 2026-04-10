"""
Federal Register NOI enrichment for NEPATEC projects.

Default scope: clean energy EIS projects only.
Designed to be run as a standalone script for sampling and QA
before integration into the main extract pipeline.
"""

from __future__ import annotations

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

from dataclasses import dataclass
from pathlib import Path
import time
import hashlib
import json
import re
import ast
from typing import Iterable, Optional

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # phase2/code/extract/ -> repo root
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

FR_ENDPOINT = "https://www.federalregister.gov/api/v1/documents.json"

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
    "at",
    "construction",
    "department",
    "energy",
    "facility",
    "facilities",
    "final",
    "for",
    "guarantee",
    "impact",
    "in",
    "intent",
    "license",
    "management",
    "notice",
    "of",
    "operation",
    "permit",
    "plan",
    "prepare",
    "program",
    "project",
    "proposed",
    "public",
    "renewal",
    "statement",
    "subsequent",
    "the",
    "to",
}

_MATCH_STOPWORDS = _SEARCH_NOISE_TOKENS | {
    "assessment",
    "county",
    "draft",
    "environmental",
    "line",
    "meeting",
    "meetings",
    "plant",
    "plants",
    "programmatic",
    "resource",
    "scoping",
    "site",
    "sites",
    "solar",
    "state",
    "transmission",
    "use",
    "wind",
}

_EMPTY_HINT_PATTERNS = (
    r"^none\b",
    r"^unk$",
    r"^unknown$",
    r"sponsored by the lead agency",
)


@dataclass
class FederalRegisterConfig:
    process_types: tuple[str, ...] = ("EIS", "EA")
    energy_types: tuple[str, ...] = ("Clean",)
    sample_n: Optional[int] = 100
    random_state: int = 7
    per_page: int = 100
    throttle_seconds: float = 0.25
    start_date: str = "2000-01-01"
    end_date: Optional[str] = "2025-12-31"
    fetch_raw_text: bool = True
    max_results_before_refine: int = 100
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    debug_log_path: Optional[Path] = None


def _normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def _normalize_phrase(text: str) -> str:
    return _normalize_space(re.sub(r"[^A-Za-z0-9]+", " ", text)).lower()


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
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    ]


def _search_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[A-Za-z0-9]+", text)
        if len(word) > 1 and word.lower() != "s"
    ]


def _token_weight(token: str) -> int:
    weight = min(len(token), 12)
    if token in _SEARCH_NOISE_TOKENS:
        weight -= 4
    if any(char.isdigit() for char in token):
        weight += 2
    return max(weight, 0)


def _window_score(words: list[str]) -> tuple[int, int]:
    lowered = [word.lower() for word in words]
    informative = sum(1 for word in lowered if _token_weight(word) > 0)
    weighted = sum(_token_weight(word) for word in lowered)
    return informative, weighted


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _clean_term(value: object) -> str:
    """
    Normalize agency/state terms that may appear as list-like strings.
    """
    text = _normalize_text(value)
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        # Try JSON list
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                return _normalize_text(parsed[0])
        except json.JSONDecodeError:
            pass
        # Try Python list
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                return _normalize_text(parsed[0])
        except (ValueError, SyntaxError):
            pass
        # Fallback: strip brackets/quotes
        text = text.strip("[]").strip("\"' ")
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in _EMPTY_HINT_PATTERNS):
        return ""
    return text


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


def _select_title_keywords(title: str, max_tokens: int = 4) -> list[str]:
    cleaned = _strip_title_prefixes(_normalize_text(title))
    keywords = []
    seen = set()
    for token in _tokenize(cleaned, _SEARCH_NOISE_TOKENS):
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= max_tokens:
            break
    return keywords


def _build_search_plans(row: pd.Series) -> list[tuple[str, str]]:
    plans = [
        ("title_only", _build_search_terms(row, include_agency=False, include_state=False, include_sponsor=False)),
        (
            "title_agency_state",
            _build_search_terms(row, include_agency=True, include_state=True, include_sponsor=True),
        ),
    ]

    keyword_terms = _build_search_terms(
        row,
        include_agency=True,
        include_state=True,
        include_sponsor=True,
        keyword_mode=True,
    )
    if keyword_terms not in {terms for _, terms in plans}:
        plans.append(("keywords_agency_state", keyword_terms))

    return plans


def _build_search_terms(
    row: pd.Series,
    include_agency: bool = False,
    include_state: bool = False,
    include_sponsor: bool = False,
    keyword_mode: bool = False,
) -> str:
    title = _normalize_text(row.get("project_title"))
    department = _clean_term(row.get("project_department"))
    lead_agency = _clean_term(row.get("lead_agency"))
    sponsor = _clean_term(row.get("project_sponsor"))
    state = _clean_term(row.get("project_state"))

    terms = ['("Notice of Intent" OR "Intent to Prepare")']

    if keyword_mode:
        keywords = _select_title_keywords(title)
        terms.extend(keywords)
    else:
        title_phrase = _select_title_phrase(title)
        if title_phrase:
            terms.append(f"\"{title_phrase}\"")

    if include_agency:
        for hint in [lead_agency, department]:
            if hint and hint not in terms:
                terms.append(hint)

    if include_sponsor and sponsor and sponsor not in terms:
        terms.append(sponsor)

    if include_state and state:
        terms.append(state)

    return " AND ".join(terms)


def _request_key(terms: str, start_date: str, end_date: Optional[str]) -> str:
    payload = f"{terms}|{start_date}|{end_date or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def search_noi(
    terms: str,
    start_date: str,
    end_date: Optional[str],
    per_page: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict:
    params = {
        "conditions[term]": terms,
        "conditions[type][]": "NOTICE",
        "conditions[publication_date][gte]": start_date,
        "per_page": per_page,
        "order": "oldest",
        "fields[]": [
            "title",
            "publication_date",
            "document_number",
            "html_url",
            "raw_text_url",
            "agency_names",
            "type",
            "subtype",
            "comments_close_on",
            "abstract",
        ],
    }
    if end_date:
        params["conditions[publication_date][lte]"] = end_date

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
    title_lower = title.lower()
    if "notice of intent" not in title_lower:
        return False
    if "termination" in title_lower or "withdrawal" in title_lower:
        return False
    return True


def _candidate_match_metrics(item: dict, row: pd.Series) -> dict:
    title_text = _normalize_text(item.get("title"))
    title_norm = _normalize_phrase(title_text)
    selected_phrase = _select_title_phrase(_normalize_text(row.get("project_title")))
    selected_phrase_norm = _normalize_phrase(selected_phrase)

    project_tokens = set(_tokenize(_strip_title_prefixes(_normalize_text(row.get("project_title"))), _MATCH_STOPWORDS))
    candidate_tokens = set(_tokenize(title_text, _MATCH_STOPWORDS))
    agency_tokens = set(_tokenize(" ".join(item.get("agency_names") or []), _MATCH_STOPWORDS))
    sponsor_tokens = set(_tokenize(_normalize_text(row.get("project_sponsor")), _MATCH_STOPWORDS))

    title_overlap = sorted(project_tokens & candidate_tokens)
    sponsor_overlap = sorted(sponsor_tokens & (candidate_tokens | agency_tokens))
    exact_phrase = bool(selected_phrase_norm and selected_phrase_norm in title_norm)

    return {
        "exact_phrase_match": exact_phrase,
        "title_overlap_tokens": title_overlap,
        "title_overlap_count": len(title_overlap),
        "sponsor_overlap_tokens": sponsor_overlap,
        "sponsor_overlap_count": len(sponsor_overlap),
    }


def _passes_candidate_threshold(metrics: dict) -> bool:
    if metrics["exact_phrase_match"]:
        return True
    if metrics["title_overlap_count"] >= 2:
        return True
    if metrics["title_overlap_count"] == 1:
        token = metrics["title_overlap_tokens"][0]
        if len(token) >= 8 or any(char.isdigit() for char in token):
            return True
    return False


def _score_candidate(item: dict, row: pd.Series, metrics: Optional[dict] = None) -> int:
    metrics = metrics or _candidate_match_metrics(item, row)
    score = 0
    title = _normalize_text(item.get("title")).lower()
    agency_names = " ".join(item.get("agency_names") or []).lower()

    if "notice of intent" in title:
        score += 5
    selected_phrase = _select_title_phrase(_normalize_text(row.get("project_title"))).lower()
    if selected_phrase and _normalize_phrase(selected_phrase) in _normalize_phrase(title):
        score += 4
    score += metrics["title_overlap_count"] * 2
    score += metrics["sponsor_overlap_count"] * 2

    lead_agency = _normalize_text(row.get("lead_agency")).lower()
    department = _normalize_text(row.get("project_department")).lower()
    state = _normalize_text(row.get("project_state")).lower()

    for hint in [lead_agency, department]:
        if hint and hint in title:
            score += 3
            break
    if lead_agency and lead_agency in agency_names:
        score += 3
    if department and department in agency_names:
        score += 2
    if state and state in title:
        score += 1

    return score


_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE_PATTERN = rf"(?:{_MONTHS})\\s+\\d{{1,2}},\\s+\\d{{4}}"


def _extract_scoping_dates(text: str) -> list[str]:
    if not text:
        return []
    matches = []
    for line in text.splitlines():
        if "scoping meeting" in line.lower():
            matches.extend(re.findall(_DATE_PATTERN, line))
    return sorted(set(matches))


def _fetch_raw_text(url: str) -> str:
    if not url:
        return ""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def pick_best_noi(results: dict, row: pd.Series) -> Optional[dict]:
    candidates = []
    for item in results.get("results", []):
        title = _normalize_text(item.get("title"))
        if not title:
            continue
        if not _is_valid_noi_title(title):
            continue
        metrics = _candidate_match_metrics(item, row)
        if not _passes_candidate_threshold(metrics):
            continue
        score = _score_candidate(item, row, metrics=metrics)
        candidates.append((score, item, metrics))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1].get("publication_date") or "9999-99-99"))
    best_score, best_item, metrics = candidates[0]
    best_item = dict(best_item)
    best_item["match_score"] = best_score
    best_item["title_overlap_count"] = metrics["title_overlap_count"]
    best_item["title_overlap_tokens"] = metrics["title_overlap_tokens"]
    best_item["sponsor_overlap_count"] = metrics["sponsor_overlap_count"]
    return best_item


def enrich_projects_with_noi(
    projects: pd.DataFrame,
    config: FederalRegisterConfig,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    df = projects.copy()
    debug_log = config.debug_log_path

    if config.process_types:
        df = df[df["process_type"].isin(config.process_types)]
    if config.energy_types:
        df = df[df["project_energy_type"].isin(config.energy_types)]

    if config.sample_n:
        df = df.sample(n=min(config.sample_n, len(df)), random_state=config.random_state)

    cache = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    results_rows = []
    for _, row in df.iterrows():
        best = None
        terms = ""
        match_tier = "unmatched"

        for tier_name, candidate_terms in _build_search_plans(row):
            terms = candidate_terms
            cache_key = _request_key(terms, config.start_date, config.end_date)
            if debug_log:
                debug_log.parent.mkdir(parents=True, exist_ok=True)
                with debug_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"project_id={row.get('project_id')} tier={tier_name} terms={terms}\n")

            if cache_key in cache:
                results = cache[cache_key]
            else:
                try:
                    results = search_noi(
                        terms,
                        config.start_date,
                        config.end_date,
                        config.per_page,
                        config.max_retries,
                        config.retry_backoff_seconds,
                    )
                    cache[cache_key] = results
                except requests.RequestException:
                    results = {"count": 0, "results": []}
                time.sleep(config.throttle_seconds)

            best = pick_best_noi(results, row)
            if best is not None:
                match_tier = tier_name
                break

        scoping_dates = []
        if best and config.fetch_raw_text:
            raw_text = ""
            if best.get("raw_text_url"):
                try:
                    raw_text = _fetch_raw_text(best["raw_text_url"])
                    time.sleep(config.throttle_seconds)
                except requests.RequestException:
                    raw_text = ""
            scoping_dates = _extract_scoping_dates(raw_text or best.get("abstract") or "")

        results_rows.append(
            {
                "project_id": row.get("project_id"),
                "project_title": row.get("project_title"),
                "noi_publication_date": (best or {}).get("publication_date"),
                "noi_document_number": (best or {}).get("document_number"),
                "noi_url": (best or {}).get("html_url"),
                "noi_project_title": (best or {}).get("title"),
                "noi_type": (best or {}).get("type"),
                "noi_subtype": (best or {}).get("subtype"),
                "noi_comments_close_on": (best or {}).get("comments_close_on"),
                "noi_scoping_meeting_dates": "; ".join(scoping_dates) if scoping_dates else None,
                "noi_match_score": (best or {}).get("match_score"),
                "noi_title_overlap_count": (best or {}).get("title_overlap_count"),
                "noi_title_overlap_tokens": ", ".join((best or {}).get("title_overlap_tokens") or []),
                "noi_query": terms,
                "noi_match_tier": match_tier,
            }
        )

    if cache_path:
        cache_path.write_text(json.dumps(cache))

    return pd.DataFrame(results_rows)


def attach_noi_fields(
    projects: pd.DataFrame,
    config: FederalRegisterConfig,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    noi_df = enrich_projects_with_noi(projects, config=config, cache_path=cache_path)
    return projects.merge(noi_df, on="project_id", how="left")


def _parse_list_arg(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return tuple()
    return tuple([x.strip() for x in value.split(",") if x.strip()])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Federal Register NOI enrichment")
    parser.add_argument(
        "--projects-path",
        default=str(ANALYSIS_DIR / "projects_combined.parquet"),
        help="Path to projects_combined.parquet",
    )
    parser.add_argument("--sample", type=int, default=100, help="Sample size")
    parser.add_argument("--process-types", default="EIS,EA", help="Comma-separated list")
    parser.add_argument("--energy-types", default="Clean", help="Comma-separated list")
    parser.add_argument("--cache-path", default=str(ANALYSIS_DIR / "fr_noi_cache.json"))
    parser.add_argument("--output", default=str(ANALYSIS_DIR / "noi_federal_register.parquet"))
    parser.add_argument("--report-n", type=int, default=10, help="Examples to print")
    parser.add_argument("--fetch-raw-text", action="store_true", help="Fetch raw text for scoping dates")
    parser.add_argument("--debug-log", default=None, help="Optional path to write query debug log")

    args = parser.parse_args()

    projects = pd.read_parquet(args.projects_path)
    config = FederalRegisterConfig(
        process_types=_parse_list_arg(args.process_types) or ("EIS", "EA"),
        energy_types=_parse_list_arg(args.energy_types) or ("Clean",),
        sample_n=args.sample,
        fetch_raw_text=args.fetch_raw_text,
        debug_log_path=Path(args.debug_log) if args.debug_log else None,
    )
    cache_path = Path(args.cache_path) if args.cache_path else None
    output_path = Path(args.output)

    results = enrich_projects_with_noi(projects, config=config, cache_path=cache_path)
    results.to_parquet(output_path, index=False)
    print(f"Saved {len(results):,} rows to: {output_path}")

    # Lightweight report + examples
    total = len(results)
    matched = results["noi_publication_date"].notna().sum()
    print("\n=== Federal Register NOI Report ===")
    print(f"Total rows: {total:,}")
    print(f"Matched NOI: {matched:,} ({matched / total:.1%} if total else 0)")
    if "noi_match_score" in results.columns:
        print("\nMatch score summary:")
        print(results["noi_match_score"].describe())

    report_n = max(0, args.report_n)
    if report_n:
        print(f"\nTop {report_n} matches:")
        top = results.sort_values("noi_match_score", ascending=False).head(report_n)
        print(top[["project_id", "noi_publication_date", "noi_document_number", "noi_match_score", "noi_url"]])

        print(f"\nBottom {report_n} matches (lowest scores with NOI):")
        bottom = (
            results[results["noi_publication_date"].notna()]
            .sort_values("noi_match_score", ascending=True)
            .head(report_n)
        )
        print(bottom[["project_id", "noi_publication_date", "noi_document_number", "noi_match_score", "noi_url"]])

        print(f"\nUnmatched examples (first {report_n}):")
        missing = results[results["noi_publication_date"].isna()].head(report_n)
        print(missing[["project_id", "noi_query"]])


if __name__ == "__main__":
    main()
