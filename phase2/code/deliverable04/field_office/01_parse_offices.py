#!/usr/bin/env python
"""Parse BLM field-office codes for D4 Extension #1 (field-office learning curve).

BLM NEPA case numbers follow ``DOI-BLM-<2-letter-state>-<office-code>-<year>-<seq>``
(e.g. ``DOI-BLM-CA-C060-2018-0018-EA``). The office code (``CA-C060``) identifies the
BLM field/district office that processed the review. This script maps every BLM-led
project to its field office so the learning-curve analysis (02_learning_curve.R) can
order an office's reviews by accumulated experience.

Approach (all in DuckDB; never ``pd.read_parquet`` on pages files):
  1. BLM universe = projects whose ``lead_agency_harmonized`` contains
     "Bureau of Land Management" (timeline_document_index.parquet).
  2. Per project, aggregate every document's ``blm_case_number`` and ``file_name``.
  3. Validated regex ``BLM-([A-Z]{2}-?[A-Z]?[0-9]{2,4})`` on the case-number string
     first, then the file-name string as a fallback. ~62% coverage.
  4. Normalize the captured code to ``STATE-OFFICE`` (upper-case, single dash);
     ``state`` = first two letters; ``parse_source`` records which string matched.

Output project->office map:
  phase2/data/analysis/deliverable04/blm_field_offices.parquet
  (project_id, office_code, state, parse_source in {case_number, file_name})

The script also prints parse coverage and characterizes the unparsed remainder
(by process type) so the ~38% that drop out are documented, not silent.

Usage:
  conda run -n nepa python phase2/code/deliverable04/field_office/01_parse_offices.py --run
  conda run -n nepa python phase2/code/deliverable04/field_office/01_parse_offices.py --run --threads 8
"""
import argparse
import os
import time
from pathlib import Path

import duckdb

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

PHASE2 = Path(__file__).resolve().parents[3]
TL = PHASE2 / "data" / "analysis" / "timeline"
OUT_DIR = PHASE2 / "data" / "analysis" / "deliverable04"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "blm_field_offices.parquet"
DIAG_DIR = PHASE2 / "output" / "deliverable04" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
COV_PATH = DIAG_DIR / "d4_fieldoffice_parse_coverage.csv"

IDX = (TL / "timeline_document_index.parquet").as_posix()

# Validated field-office regex (~62% of BLM projects). Captures e.g. CA-C060 / AK-R000.
OFFICE_RE = r"BLM-([A-Z]{2}-?[A-Z]?[0-9]{2,4})"


def build(con: duckdb.DuckDBPyConnection) -> None:
    """Collapse documents to one row per BLM project, parse the office code from the
    case-number string (then file-name fallback), normalize, and register `offices`."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE proj AS
        SELECT
            project_id,
            ANY_VALUE(lead_agency_harmonized)              AS lead_agency,
            ANY_VALUE(process_type)                        AS process_type,
            ANY_VALUE(project_energy_type)                 AS project_energy_type,
            string_agg(COALESCE(blm_case_number, ''), ' ') AS cases,
            string_agg(COALESCE(file_name, ''), ' ')       AS fnames
        FROM read_parquet('{IDX}')
        GROUP BY project_id
    """)
    # Restrict to BLM-led projects, then parse.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE parsed AS
        SELECT
            project_id, process_type, project_energy_type,
            regexp_extract(cases,  '{OFFICE_RE}', 1) AS code_case,
            regexp_extract(fnames, '{OFFICE_RE}', 1) AS code_file
        FROM proj
        WHERE lead_agency LIKE '%Bureau of Land Management%'
    """)
    # Normalize: case number wins; strip non-alnum; STATE-OFFICE; state = first 2 letters.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE offices AS
        WITH pick AS (
            SELECT project_id, process_type, project_energy_type,
                   CASE WHEN code_case <> '' THEN code_case ELSE code_file END AS raw_code,
                   CASE WHEN code_case <> '' THEN 'case_number'
                        WHEN code_file <> '' THEN 'file_name' END             AS parse_source
            FROM parsed
        ), clean AS (
            SELECT project_id, parse_source,
                   upper(regexp_replace(raw_code, '[^A-Za-z0-9]', '', 'g')) AS c
            FROM pick WHERE parse_source IS NOT NULL
        )
        SELECT project_id,
               substr(c, 1, 2) || '-' || substr(c, 3) AS office_code,
               substr(c, 1, 2)                        AS state,
               parse_source
        FROM clean
        WHERE length(c) >= 4
    """)


def report(con: duckdb.DuckDBPyConnection) -> None:
    """Print parse coverage and characterize the unparsed remainder."""
    total = con.execute("SELECT COUNT(*) FROM parsed").fetchone()[0]
    n_parsed = con.execute("SELECT COUNT(*) FROM offices").fetchone()[0]
    by_src = con.execute(
        "SELECT parse_source, COUNT(*) n FROM offices GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    print(f"\n=== BLM field-office parse coverage ===")
    print(f"BLM-led projects (universe):     {total:,}")
    print(f"Parsed to a field office:        {n_parsed:,}  ({100*n_parsed/total:.1f}%)")
    print(f"Unparsed:                        {total - n_parsed:,}  ({100*(total-n_parsed)/total:.1f}%)")
    print("\nParsed by source:")
    print(by_src.to_string(index=False))

    n_offices = con.execute("SELECT COUNT(DISTINCT office_code) FROM offices").fetchone()[0]
    print(f"\nDistinct field offices: {n_offices:,}")

    # Unparsed characterization: which process types fall out, and how heavily.
    unp = con.execute("""
        SELECT p.process_type,
               COUNT(*)                                                   AS blm_projects,
               COUNT(*) - COUNT(o.project_id)                             AS unparsed,
               ROUND(100.0 * (COUNT(*) - COUNT(o.project_id)) / COUNT(*), 1) AS pct_unparsed
        FROM parsed p LEFT JOIN offices o USING (project_id)
        GROUP BY 1 ORDER BY blm_projects DESC
    """).df()
    print("\nUnparsed by process type (the ~38% characterized):")
    print(unp.to_string(index=False))
    print("\nNote: EIS is disproportionately unparsed — programmatic/large EISs are filed under")
    print("titles rather than a DOI-BLM case number. Many unparsed EAs do carry an office-like")
    print("code in a non-standard form (bare 'C069-2023-...', 'AZ-A010-...') that lacks the")
    print("'BLM-' prefix the validated regex anchors on; they are left out rather than guessed.")

    top = con.execute(
        "SELECT office_code, state, COUNT(*) n FROM offices GROUP BY 1,2 ORDER BY n DESC LIMIT 12"
    ).df()
    print("\nTop field offices by parsed project count:")
    print(top.to_string(index=False))

    # Reproducible coverage CSV for the report (per process + ALL row).
    cov = con.execute("""
        SELECT COALESCE(p.process_type, 'NA') AS scope,
               COUNT(*)                                                       AS blm_projects,
               COUNT(o.project_id)                                            AS parsed,
               COUNT(*) - COUNT(o.project_id)                                 AS unparsed,
               ROUND(100.0 * COUNT(o.project_id) / COUNT(*), 1)               AS pct_parsed
        FROM parsed p LEFT JOIN offices o USING (project_id)
        GROUP BY 1
        UNION ALL
        SELECT 'ALL', COUNT(*), COUNT(o.project_id), COUNT(*) - COUNT(o.project_id),
               ROUND(100.0 * COUNT(o.project_id) / COUNT(*), 1)
        FROM parsed p LEFT JOIN offices o USING (project_id)
        ORDER BY blm_projects DESC
    """).df()
    cov.to_csv(COV_PATH, index=False)
    print(f"\nWrote parse-coverage CSV -> {COV_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="parse and write the office map")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads (default 4)")
    args = ap.parse_args()
    if not args.run:
        ap.error("pass --run to build blm_field_offices.parquet")

    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    build(con)
    report(con)

    out = OUT_PATH.as_posix().replace("'", "''")
    con.execute(f"COPY (SELECT * FROM offices ORDER BY office_code, project_id) TO '{out}' (FORMAT PARQUET)")
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"\nSaved {n:,} project->office rows -> {OUT_PATH}  ({time.time()-t0:.0f}s)")
    con.close()


if __name__ == "__main__":
    main()
