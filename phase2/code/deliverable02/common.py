"""D2 (significance determinations) — shared paths, IO, and helpers.

Self-contained within deliverable02. Reads shared / D6 / D5 artifacts read-only;
writes only to the D2 write set. See phase2/plans/deliverable02.md (v2.11).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

# ---- paths (resolved from repo root; this file lives at phase2/code/deliverable02/) ----
ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"

# D2 write set
D2_ANALYSIS_DIR = ANALYSIS_DIR / "deliverable02"
D2_OUTPUT_DIR = PHASE2 / "output" / "deliverable02"
D2_GOLD_DIR = D2_ANALYSIS_DIR / "gold"

# D2 output artifacts (the full surface — plan Phase 7)
PROJECT_REGIME = D2_ANALYSIS_DIR / "project_regime.parquet"
SIGNIFICANCE_CORPUS = D2_ANALYSIS_DIR / "significance_corpus.parquet"
PROJECT_COHORTS = D2_ANALYSIS_DIR / "project_cohorts.parquet"
MITIGATION_SIGNAL_MATCHES = D2_ANALYSIS_DIR / "mitigation_signal_matches.parquet"
SIGNIFICANCE_SECTION_CANDIDATES = D2_ANALYSIS_DIR / "significance_section_candidates.parquet"
SIGNIFICANCE_DETERMINATIONS = D2_ANALYSIS_DIR / "significance_determinations.parquet"
DETERMINATION_THRESHOLDS = D2_ANALYSIS_DIR / "determination_thresholds.parquet"
RUN_MANIFEST = D2_ANALYSIS_DIR / "significance_run_manifest.parquet"
GOLD = D2_GOLD_DIR / "significance_gold.parquet"
GOLD_THRESHOLDS = D2_GOLD_DIR / "significance_gold_thresholds.parquet"
GOLD_QUEUE_CSV = D2_OUTPUT_DIR / "significance_gold_queue.csv"

# read-only inputs
PROJECTS_COMBINED = ANALYSIS_DIR / "projects_combined.parquet"
DOCUMENT_SECTIONS = ANALYSIS_DIR / "document_sections.parquet"
TIMELINE_DATES = ANALYSIS_DIR / "timeline" / "timeline_project_dates.parquet"
LAW_CITATIONS = ANALYSIS_DIR / "deliverable05" / "law_citations.parquet"

D6_DIR = ANALYSIS_DIR / "deliverable06"
FONSI_INVENTORY = D6_DIR / "fonsi_project_inventory.parquet"
FONSI_SPANS = D6_DIR / "fonsi_evidence_spans.parquet"
FONSI_CONDITIONS = D6_DIR / "fonsi_conditions.parquet"
FONSI_SECTION_MANIFEST = D6_DIR / "fonsi_section_manifest.parquet"

SCHEMA_VERSION = "d2_v2_11"

# ---- cohort bin constants (plan A4, frozen) ----
ARRA = "2009-02-17"
BIL = "2021-11-15"
IRA = "2022-08-16"
FRA = "2023-06-03"


def ensure_dirs() -> None:
    for d in (D2_ANALYSIS_DIR, D2_OUTPUT_DIR, D2_GOLD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: object) -> str:
    return hashlib.sha256(str(text).strip().lower().encode("utf-8")).hexdigest()


def sha256_join(*parts: object) -> str:
    """Deterministic id from ordered parts (matches DuckDB sha256(concat_ws('|', ...)))."""
    return hashlib.sha256("|".join("" if p is None else str(p) for p in parts).encode("utf-8")).hexdigest()


def con() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection. Pass parquet paths as str() into read_parquet()."""
    return duckdb.connect()


def q(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a one-off DuckDB query and return a DataFrame."""
    c = duckdb.connect()
    try:
        return c.execute(sql, params or []).fetchdf()
    finally:
        c.close()


def write_parquet(df: pd.DataFrame, path: Path, label: str = "") -> None:
    ensure_dirs()
    df.to_parquet(path, index=False)
    tag = f" [{label}]" if label else ""
    print(f"  wrote {len(df):>7,} rows -> {path.relative_to(PHASE2)}{tag}")


def write_csv(df: pd.DataFrame, path: Path, label: str = "") -> None:
    ensure_dirs()
    df.to_csv(path, index=False)
    tag = f" [{label}]" if label else ""
    print(f"  wrote {len(df):>7,} rows -> {path.relative_to(PHASE2)}{tag}")
