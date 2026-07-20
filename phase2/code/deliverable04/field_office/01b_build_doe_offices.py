#!/usr/bin/env python
"""Build a DOE administering-office map for D4 Extension #1 (field-office learning curve, DOE arm).

The BLM arm (01_parse_offices.py) parses a field-office code out of the DOI-BLM case number.
DOE has no analogous case-number office code, but the DOE CX (categorical exclusion) register
carries an ``office`` field per ``cx_number``. This script links every DOE-led project to that
office so 02_create_figures.R can run the same office fixed-effects learning-curve regression for
DOE that it runs for BLM — the structural mirror image (CE-only, document-anchored).

Approach (all in DuckDB; never ``pd.read_parquet`` on pages files):
  1. DOE universe = projects whose ``lead_agency_harmonized`` contains "Department of Energy"
     (NOT "Energy" — that wrongly catches Bureau of Ocean Energy Management / "Energy Programs").
     Bonneville / WAPA are DOE register offices, not separate lead agencies, so no exclusion.
  2. Per project, collect every document's ``cx_number`` and join to the register.
     JOIN BUG TO AVOID: ``cx_number`` is DOUBLE in timeline_document_index (``6176.0``) but
     BIGINT in the register (``6176``). Join on ``CAST(ROUND(cx_number) AS BIGINT)`` — a VARCHAR
     join silently yields ZERO matches.
  3. Harmonize the register office string (3-part rule):
       (1) take the segment after the LAST comma (strips the program-office prefix,
           e.g. "Energy Efficiency and Renewable Energy, Golden Field Office" -> "Golden Field Office");
       (2) normalize dash spacing (" - " -> "-", merging the ARPA-E variants) and strip the
           trailing-ellipsis scrape artifact ("Golden Field Offi..." / "Golden");
       (3) canonicalize truncated variants against a controlled vocabulary of dominant offices by
           the prefix/truncation relationship (one string is a prefix of the other).
  4. One office per project via the MODE over the project's matched non-null canonical offices.

Output project->office map:
  phase2/data/analysis/deliverable04/doe_offices.parquet
  (project_id, office, office_raw, n_cx_matched, project_energy_type)

Diagnostics (phase2/output/deliverable04/diagnostics/):
  d4_fieldoffice_doe_coverage.csv     — cx->register->office funnel + harmonization + register null share
  d4_fieldoffice_doe_office_counts.csv — per canonical office project count (+ ALL rollup)

Usage:
  conda run -n nepa python phase2/code/deliverable04/field_office/01b_build_doe_offices.py --run
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import duckdb

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

PHASE2 = Path(__file__).resolve().parents[3]
TL = PHASE2 / "data" / "analysis" / "timeline"
REG_DIR = PHASE2 / "data" / "analysis" / "doe_register"
OUT_DIR = PHASE2 / "data" / "analysis" / "deliverable04"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "doe_offices.parquet"
DIAG_DIR = PHASE2 / "output" / "deliverable04" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
COV_PATH = DIAG_DIR / "d4_fieldoffice_doe_coverage.csv"
COUNTS_PATH = DIAG_DIR / "d4_fieldoffice_doe_office_counts.csv"

IDX = (TL / "timeline_document_index.parquet").as_posix()
REG = (REG_DIR / "doe_cx_register.parquet").as_posix()

# Verified funnel references (live data 2026-07-20) — asserted by the hard checks below.
REF_DOE_LED = 32305
REF_WITH_CX = 27826
REF_REG_MATCHED = 25572
REF_WITH_OFFICE = 11707

# Controlled vocabulary of dominant DOE register offices (dash-normalized canonical names).
VOCAB = [
    "National Energy Technology Laboratory",
    "Golden Field Office",
    "Savannah River Operations Office",
    "Bonneville Power Administration",
    "River Protection-Richland Operations Office",
    "Western Area Power Administration-Rocky Mountain Region",
    "Western Area Power Administration-Desert Southwest Region",
    "Western Area Power Administration-Upper Great Plains Region",
    "Western Area Power Administration-Sierra Nevada Region",
    "Advanced Research Projects Agency-Energy",
    "Idaho Operations Office",
    "Argonne Site Office",
    "Fermi Site Office",
    "Sandia Site Office",
    "Nuclear Energy",
    "Energy Efficiency and Renewable Energy",
]
_VOCAB_L = [(c, c.lower()) for c in VOCAB]


def _norm_dash(x: str) -> str:
    return re.sub(r"\s*-\s*", "-", x.strip())


def _segment(s: str):
    """The office segment: scan comma-parts from the last backward until one survives the
    trailing-ellipsis strip (recovers the program-office prefix when the real office was
    truncated to junk, e.g. 'Energy Efficiency and Renewable Energy, ...' -> that prefix).
    Falls back to the normalized whole string so a non-null raw office is never dropped."""
    for part in reversed(s.split(",")):
        cand = re.sub(r"[.…\s]+$", "", _norm_dash(part))
        if cand:
            return cand
    return _norm_dash(s) or s


def seg_only(office_raw):
    """The harmonized office segment (steps 1-2) — used only for the seg-distinct diagnostic."""
    if office_raw is None:
        return None
    s = str(office_raw).strip()
    if not s:
        return None
    return _segment(s)


# Minimum length of a truncated segment before the prefix rule may canonicalize it — a floor
# that stops 1-5 char scrape stubs from matching a vocab office by their first letters.
CANON_MIN_PREFIX = 6


def canon(office_raw):
    """Harmonize a raw register office string to a canonical office (or None)."""
    if office_raw is None:
        return None
    s = str(office_raw).strip()
    if not s:
        return None
    # (1) office segment (last comma-part, ellipsis-stripped; prefix fallback); (2) dash-normalized
    seg = _segment(s)
    if not seg:
        return None
    # (3) canonicalize: exact match first, then a truncated variant (one string prefixes the
    #     other) once the segment clears the length floor.
    segl = seg.lower()
    for c, cl in _VOCAB_L:
        if segl == cl:
            return c
        if len(segl) >= CANON_MIN_PREFIX and (segl.startswith(cl) or cl.startswith(segl)):
            return c
    return seg


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.create_function("canon", canon, ["VARCHAR"], "VARCHAR", null_handling="special")
    con.create_function("seg_only", seg_only, ["VARCHAR"], "VARCHAR", null_handling="special")

    # DOE-led projects + one project_energy_type per project.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE proj AS
        SELECT project_id,
               first(project_energy_type) FILTER (WHERE project_energy_type IS NOT NULL) AS project_energy_type,
               first(process_type)        FILTER (WHERE process_type IS NOT NULL)        AS process_type
        FROM read_parquet('{IDX}')
        WHERE lead_agency_harmonized LIKE '%Department of Energy%'
        GROUP BY project_id
    """)
    # Every (project, cx) with a non-null cx_number — INTEGER join key.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE proj_cx AS
        SELECT DISTINCT p.project_id, CAST(ROUND(i.cx_number) AS BIGINT) AS cxn
        FROM proj p JOIN read_parquet('{IDX}') i USING (project_id)
        WHERE i.cx_number IS NOT NULL
    """)
    # Register matches: (project, cx, raw office, canonical office).
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE matched AS
        SELECT pc.project_id, pc.cxn, r.office AS office_raw, canon(r.office) AS office
        FROM proj_cx pc JOIN read_parquet('{REG}') r ON pc.cxn = r.cx_number
    """)
    # One canonical office per project = MODE over matched non-null canonical offices
    # (ties broken alphabetically); office_raw = the modal raw string for that office.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE offices AS
        WITH cc AS (
            SELECT project_id, office, COUNT(*) AS n_office
            FROM matched WHERE office IS NOT NULL
            GROUP BY 1, 2
        ), pick AS (
            SELECT project_id, office,
                   row_number() OVER (PARTITION BY project_id ORDER BY n_office DESC, office) AS rn
            FROM cc
        ), chosen AS (
            SELECT project_id, office FROM pick WHERE rn = 1
        ), raw_pick AS (
            SELECT m.project_id, m.office, m.office_raw, COUNT(*) AS n_raw,
                   row_number() OVER (PARTITION BY m.project_id, m.office
                                      ORDER BY COUNT(*) DESC, m.office_raw) AS rn
            FROM matched m WHERE m.office IS NOT NULL
            GROUP BY 1, 2, 3
        )
        SELECT c.project_id, c.office,
               rp.office_raw,
               (SELECT COUNT(DISTINCT cxn) FROM matched mm WHERE mm.project_id = c.project_id
                                                             AND mm.office IS NOT NULL) AS n_cx_matched,
               p.project_energy_type
        FROM chosen c
        JOIN raw_pick rp ON rp.project_id = c.project_id AND rp.office = c.office AND rp.rn = 1
        JOIN proj p ON p.project_id = c.project_id
    """)


def report_and_check(con: duckdb.DuckDBPyConnection) -> bool:
    import pandas as pd

    doe_led = con.execute("SELECT COUNT(*) FROM proj").fetchone()[0]
    with_cx = con.execute("SELECT COUNT(DISTINCT project_id) FROM proj_cx").fetchone()[0]
    reg_matched = con.execute("SELECT COUNT(DISTINCT project_id) FROM matched").fetchone()[0]
    with_office = con.execute("SELECT COUNT(*) FROM offices").fetchone()[0]

    # Harmonization distinct-office counts over the whole register.
    harm = con.execute(f"""
        SELECT
            COUNT(DISTINCT office)           AS raw_distinct,
            COUNT(DISTINCT seg_only(office)) AS seg_distinct,
            COUNT(DISTINCT canon(office))    AS canon_distinct
        FROM read_parquet('{REG}')
        WHERE office IS NOT NULL AND office <> ''
    """).fetchone()
    reg_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{REG}')").fetchone()[0]
    reg_null = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{REG}') WHERE office IS NULL OR office = ''"
    ).fetchone()[0]

    # All office-matched projects are CE (CX register is CE-only)?
    non_ce = con.execute("""
        SELECT COUNT(*) FROM offices o JOIN proj p ON p.project_id = o.project_id
        WHERE p.process_type IS DISTINCT FROM 'CE'
    """).fetchone()[0]
    all_ce = int(non_ce == 0)

    cov = pd.DataFrame([
        ("doe_led", doe_led),
        ("with_cx", with_cx),
        ("register_matched", reg_matched),
        ("with_office", with_office),
        ("register_rows", reg_rows),
        ("register_null_office", reg_null),
        ("register_raw_distinct_office", harm[0]),
        ("harmonized_seg_distinct", harm[1]),
        ("harmonized_canon_distinct", harm[2]),
        ("all_office_matched_are_ce", all_ce),
    ], columns=["metric", "value"])
    cov.to_csv(COV_PATH, index=False)

    counts = con.execute("""
        SELECT office, COUNT(*) AS n_parsed
        FROM offices GROUP BY 1
        UNION ALL
        SELECT 'ALL', COUNT(*) FROM offices
        ORDER BY n_parsed DESC
    """).df()
    counts.to_csv(COUNTS_PATH, index=False)

    print("\n=== DOE administering-office linkage funnel ===")
    print(f"DOE-led projects:                {doe_led:,}")
    print(f"  with >=1 cx_number:            {with_cx:,}")
    print(f"  matching a register row:       {reg_matched:,}")
    print(f"  with a non-null canonical office: {with_office:,}")
    print(f"Register: {reg_rows:,} rows, {reg_null:,} null office "
          f"({100*reg_null/reg_rows:.1f}%); distinct office {harm[0]} raw -> {harm[1]} seg -> {harm[2]} canon")
    print(f"All office-matched projects are CE: {'YES' if all_ce else 'NO'}")
    print(f"\nTop canonical offices by project count:")
    print(counts[counts.office != "ALL"].head(14).to_string(index=False))
    print(f"\nWrote coverage CSV  -> {COV_PATH}")
    print(f"Wrote office counts -> {COUNTS_PATH}")

    # ---- HARD CHECKS -------------------------------------------------------
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print(f"  MISMATCH: {msg}")
            ok = False

    check(with_cx == REF_WITH_CX, f"with_cx {with_cx} != {REF_WITH_CX}")
    check(reg_matched == REF_REG_MATCHED, f"register_matched {reg_matched} != {REF_REG_MATCHED}")
    check(with_office == REF_WITH_OFFICE, f"with_office {with_office} != {REF_WITH_OFFICE}")
    check(doe_led == REF_DOE_LED, f"doe_led {doe_led} != {REF_DOE_LED}")
    check(all_ce == 1, f"{non_ce} office-matched projects are not CE (expected all CE)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="build and write doe_offices.parquet")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads (default 4)")
    args = ap.parse_args()
    if not args.run:
        ap.error("pass --run to build doe_offices.parquet")

    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    build(con)
    ok = report_and_check(con)
    if not ok:
        print("\nHARD CHECK FAILED — not writing parquet.")
        sys.exit(1)

    out = OUT_PATH.as_posix().replace("'", "''")
    con.execute(
        f"COPY (SELECT * FROM offices ORDER BY office, project_id) TO '{out}' (FORMAT PARQUET)"
    )
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"\nHARD CHECK PASSED. Saved {n:,} project->office rows -> {OUT_PATH}  ({time.time()-t0:.0f}s)")
    con.close()


if __name__ == "__main__":
    main()
