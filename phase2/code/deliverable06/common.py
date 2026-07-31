"""Shared helpers for the D6 FONSI opportunity-scan pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PHASE2_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = PHASE2_DIR.parent
DATA_DIR = PHASE2_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
D6_ANALYSIS_DIR = ANALYSIS_DIR / "deliverable06"
D6_RAW_DIR = DATA_DIR / "raw" / "deliverable06"
D6_VALIDATION_DIR = DATA_DIR / "validation" / "deliverable06"
D6_OUTPUT_DIR = PHASE2_DIR / "output" / "deliverable06"
D6_REVIEW_DIR = D6_OUTPUT_DIR / "review"  # QA / drill-down tables (not client-facing)

PROJECTS_COMBINED = ANALYSIS_DIR / "projects_combined.parquet"
DOCUMENTS_COMBINED = ANALYSIS_DIR / "documents_combined.parquet"
D03_REVIEWS = ANALYSIS_DIR / "deliverable03" / "projects_nepa_reviews.parquet"
D03_CE_CITATIONS = ANALYSIS_DIR / "deliverable03" / "ce_citations.parquet"
EA_PAGES = DATA_DIR / "processed" / "ea" / "pages.parquet"
EA_DOCUMENTS = DATA_DIR / "processed" / "ea" / "documents.parquet"
TIMELINE_INDEX = ANALYSIS_DIR / "timeline" / "timeline_document_index.parquet"


def ensure_d6_dirs() -> None:
    for path in (D6_ANALYSIS_DIR, D6_RAW_DIR, D6_VALIDATION_DIR, D6_OUTPUT_DIR, D6_REVIEW_DIR):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def normalize_for_hash(value: object) -> str:
    return normalize_space(value).lower()


def sha256_text(value: object) -> str:
    return hashlib.sha256(normalize_for_hash(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(paths: Iterable[Path]) -> str:
    records = []
    for path in paths:
        if path.exists():
            records.append({"path": str(path), "sha256": file_sha256(path)})
    return json.dumps(records, sort_keys=True)


def slug(value: object) -> str:
    text = normalize_space(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unclassified"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def compact_join(values: Iterable[object], limit: int = 12_000) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        text = normalize_space(value)
        if not text or text in seen:
            continue
        seen.add(text)
        kept.append(text)
        if sum(len(item) for item in kept) >= limit:
            break
    return "\n\n".join(kept)[:limit]

