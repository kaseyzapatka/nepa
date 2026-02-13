"""
Federal Register NOI enrichment for NEPATEC projects.

Default scope: clean energy EIS projects only.
Designed to be run as a standalone script for sampling and QA
before integration into the main extract pipeline.
"""

from __future__ import annotations

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


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

FR_ENDPOINT = "https://www.federalregister.gov/api/v1/documents.json"


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
    return text


def _select_title_phrase(title: str, max_words: int = 8) -> str:
    """
    Select a short phrase from the project title to reduce API noise.
    """
    title = " ".join(title.split())
    words = title.split(" ")
    if not words:
        return ""
    return " ".join(words[:max_words])


def _build_search_terms(row: pd.Series, include_agency: bool = False, include_state: bool = False) -> str:
    title = _normalize_text(row.get("project_title"))
    department = _clean_term(row.get("project_department"))
    lead_agency = _clean_term(row.get("lead_agency"))
    sponsor = _clean_term(row.get("project_sponsor"))
    state = _clean_term(row.get("project_state"))

    terms = ['("Notice of Intent" OR "Intent to Prepare")']

    title_phrase = _select_title_phrase(title)
    if title_phrase:
        terms.append(f"\"{title_phrase}\"")

    if include_agency:
        if lead_agency:
            terms.append(lead_agency)
        elif department:
            terms.append(department)
        elif sponsor:
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


def _score_candidate(item: dict, row: pd.Series) -> int:
    score = 0
    title = _normalize_text(item.get("title")).lower()
    agency_names = " ".join(item.get("agency_names") or []).lower()

    if "notice of intent" in title:
        score += 5
    if _normalize_text(row.get("project_title")).lower()[:30] in title:
        score += 4

    lead_agency = _normalize_text(row.get("lead_agency")).lower()
    department = _normalize_text(row.get("project_department")).lower()
    sponsor = _normalize_text(row.get("project_sponsor")).lower()
    state = _normalize_text(row.get("project_state")).lower()

    for hint in [lead_agency, department, sponsor]:
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
        score = _score_candidate(item, row)
        candidates.append((score, item))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1].get("publication_date") or "9999-99-99"))
    best_score, best_item = candidates[0]
    best_item = dict(best_item)
    best_item["match_score"] = best_score
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
        terms = _build_search_terms(row, include_agency=False, include_state=False)
        cache_key = _request_key(terms, config.start_date, config.end_date)
        if debug_log:
            debug_log.parent.mkdir(parents=True, exist_ok=True)
            with debug_log.open("a", encoding="utf-8") as handle:
                handle.write(f"project_id={row.get('project_id')} tier=title_only terms={terms}\n")

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

        match_tier = "title_only"
        if results.get("count", 0) == 0 or results.get("count", 0) > config.max_results_before_refine:
            refined_terms = _build_search_terms(row, include_agency=True, include_state=True)
            refined_key = _request_key(refined_terms, config.start_date, config.end_date)
            if debug_log:
                debug_log.parent.mkdir(parents=True, exist_ok=True)
                with debug_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"project_id={row.get('project_id')} tier=title_agency_state terms={refined_terms}\n"
                    )
            if refined_key in cache:
                results = cache[refined_key]
            else:
                try:
                    results = search_noi(
                        refined_terms,
                        config.start_date,
                        config.end_date,
                        config.per_page,
                        config.max_retries,
                        config.retry_backoff_seconds,
                    )
                    cache[refined_key] = results
                except requests.RequestException:
                    results = {"count": 0, "results": []}
                time.sleep(config.throttle_seconds)
            terms = refined_terms
            match_tier = "title_agency_state"

        best = pick_best_noi(results, row)
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
