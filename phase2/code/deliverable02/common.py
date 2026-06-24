"""D2 (significance determinations) — shared paths, IO, and helpers.

Self-contained within deliverable02. Reads shared / D6 artifacts read-only;
writes only to the D2 write set. See phase2/plans/deliverable02.md (v2.5).
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

# read-only inputs
PROJECTS_COMBINED = ANALYSIS_DIR / "projects_combined.parquet"
DOCUMENT_SECTIONS = ANALYSIS_DIR / "document_sections.parquet"
TIMELINE_DATES = ANALYSIS_DIR / "timeline" / "timeline_project_dates.parquet"

D6_DIR = ANALYSIS_DIR / "deliverable06"
FONSI_INVENTORY = D6_DIR / "fonsi_project_inventory.parquet"
FONSI_SPANS = D6_DIR / "fonsi_evidence_spans.parquet"
FONSI_CONDITIONS = D6_DIR / "fonsi_conditions.parquet"
FONSI_SECTION_MANIFEST = D6_DIR / "fonsi_section_manifest.parquet"

SCHEMA_VERSION = "d2_v2_5"


def ensure_dirs() -> None:
    for d in (D2_ANALYSIS_DIR, D2_OUTPUT_DIR, D2_GOLD_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: object) -> str:
    return hashlib.sha256(str(text).strip().lower().encode("utf-8")).hexdigest()


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
