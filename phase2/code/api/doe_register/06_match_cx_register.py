"""
Match DOE CX register records to NEPATEC CE project IDs via cx-NNNNNN.pdf filenames.

energy.gov stores each CX determination at a URL like:
    /nepa/articles/cx-019096-categorical-exclusion-determination
The same integer (019096) appears as the filename in NEPATEC: cx-019096.pdf
This provides a direct, lossless join key for ~90% of DOE CE projects.

Reads:
    phase2/data/processed/ce/documents.parquet
    phase2/data/analysis/doe_register/doe_cx_register.parquet   (from 05_fetch_cx_register.py)

Writes:
    phase2/data/analysis/doe_register/doe_cx_dates.parquet
      One row per matched CE project: project_id + cx_date as the determination date.

Usage:
    python 06_match_cx_register.py
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
CE_DOCS_PATH = PHASE2 / "data" / "processed" / "ce" / "documents.parquet"
CX_REG_PATH = PHASE2 / "data" / "analysis" / "doe_register" / "doe_cx_register.parquet"
OUTPUT_PATH = PHASE2 / "data" / "analysis" / "doe_register" / "doe_cx_dates.parquet"

# Filename pattern: (path/)cx-NNNNNN(.pdf or -slug.pdf)
# Case-insensitive: energy.gov portal uses lowercase "cx-019096.pdf" but WAPA and some
# DOE offices use uppercase "CX-026345.pdf". Both map to the same integer cx_number.
# Excludes BLM-style "DOI-BLM-...-CX-NNNNN.pdf" (those have a leading slash + DOI- prefix).
CX_FILENAME_RE = r"(?i)(^|/)cx-[0-9]{4,7}(-|[.]pdf)"


def main() -> None:
    if not CE_DOCS_PATH.exists():
        raise SystemExit(f"CE documents not found: {CE_DOCS_PATH}")
    if not CX_REG_PATH.exists():
        raise SystemExit(f"CX register not found — run 05_fetch_cx_register.py first: {CX_REG_PATH}")

    con = duckdb.connect()
    print(f"CE docs:     {CE_DOCS_PATH}")
    print(f"CX register: {CX_REG_PATH}")

    matched = con.execute(
        f"""
        WITH extracted AS (
            SELECT
                -- CE documents store project_id as STRUCT("value" VARCHAR); extract field.
                project_id."value" AS project_id,
                document_id AS document_id,
                file_name,
                TRY_CAST(
                    regexp_extract(lower(file_name), 'cx-([0-9]+)', 1)
                AS INTEGER) AS cx_number
            FROM read_parquet('{CE_DOCS_PATH}')
            WHERE regexp_matches(file_name, '{CX_FILENAME_RE}')
              AND regexp_extract(lower(file_name), 'cx-([0-9]+)', 1) != ''
        ),
        joined AS (
            SELECT
                e.project_id,
                e.document_id,
                e.file_name,
                e.cx_number,
                r.cx_date,
                r.cx_date_raw,
                r.office,
                r.location,
                r.cx_codes,
                r.cx_title
            FROM extracted e
            LEFT JOIN read_parquet('{CX_REG_PATH}') r
              ON e.cx_number = r.cx_number
        )
        SELECT
            project_id,
            document_id,
            file_name,
            cx_number,
            cx_date,
            cx_date_raw,
            office,
            location,
            cx_codes,
            cx_title,
            (cx_date IS NOT NULL) AS doe_cx_tier_a_eligible
        FROM joined
        ORDER BY project_id
        """
    ).df()

    n_total = len(matched)
    n_matched = matched["cx_date"].notna().sum()
    n_no_match = n_total - n_matched

    print(f"\nResults:")
    print(f"  CE projects with cx- filenames:  {n_total:,}")
    print(f"  Matched to register (date found): {n_matched:,}  ({n_matched/n_total*100:.1f}%)")
    print(f"  Unmatched (cx_number not in reg): {n_no_match:,}  ({n_no_match/n_total*100:.1f}%)")

    # Add standard columns for D4 integration
    matched["process_type"] = "CE"
    matched["doe_cx_decision_date"] = matched["cx_date"]
    matched["doe_cx_decision_date_type"] = matched["cx_date"].apply(
        lambda d: "cx_determination" if pd.notna(d) else None
    )
    matched["built_at"] = datetime.now(timezone.utc).isoformat()

    # Select final columns
    out = matched[[
        "project_id",
        "process_type",
        "document_id",
        "file_name",
        "cx_number",
        "doe_cx_decision_date",
        "doe_cx_decision_date_type",
        "doe_cx_tier_a_eligible",
        "cx_date_raw",
        "office",
        "location",
        "cx_codes",
        "cx_title",
        "built_at",
    ]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out):,} rows → {OUTPUT_PATH}")

    # Quick sanity sample
    sample = out[out["doe_cx_tier_a_eligible"]].head(5)
    print("\nSample matched projects:")
    for _, row in sample.iterrows():
        print(f"  {str(row['project_id'])[:12]}  cx_number={row['cx_number']}  date={row['doe_cx_decision_date']}  title={str(row['cx_title'])[:60]}")


if __name__ == "__main__":
    main()
