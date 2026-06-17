#!/usr/bin/env python
"""Regulatory page-count extraction for the FULL Phase 2 EA/EIS corpus (ALL energy types).

Computes the FRA "regulatory page count" (40 C.F.R. § 1508.1(bb): a page = 500 words,
excluding maps/diagrams/tables/citations) for every EA and EIS project, reading the actual
page text — no back-of-envelope ratios. Self-contained in Phase 2: sources are
phase2/data/processed/{ea,eis}/{documents,pages}.parquet; output is
phase2/data/analysis/deliverable04/projects_page_counts.parquet.

For each project's main final document (final EA / FEIS):
  1. No-appendix-file shortcut: if a filename explicitly signals the appendix-free version
     (*_wo_appendices.pdf), use its raw page count directly (no OCR scan needed).
  2. Otherwise, scan the page text in DuckDB:
       - detect the embedded-appendix boundary (appendix/attachment/exhibit header near the
         top of a short page, at/after page MIN_APPENDIX_PAGE, excluding TOC dotted-leader rows)
       - count BODY pages (page < appendix_start, word_count >= WORD_COUNT_THRESHOLD)
       - regulatory_pages = ceil(body_word_count / 500)

Efficiency: the heavy work is a single multithreaded DuckDB query that STREAMS the pages
parquet (EIS = 6.1M pages, ~5.5 GB) and joins only the target main documents — nothing is
loaded into Python memory. Threads default to all cores. EA and EIS are processed in turn
(each query already saturates the cores).

Usage:
  conda run -n nepa python phase2/code/deliverable04/fra/01_extract_pages.py --run
  conda run -n nepa python phase2/code/deliverable04/fra/01_extract_pages.py --run --sample 100 --verbose
  conda run -n nepa python phase2/code/deliverable04/fra/01_extract_pages.py --run --threads 8
"""
import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

PHASE2 = Path(__file__).resolve().parents[3]
PROCESSED = PHASE2 / "data" / "processed"
OUT_DIR = PHASE2 / "data" / "analysis" / "deliverable04"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "projects_page_counts.parquet"

# --- Classification thresholds (identical to Phase 1 extract_pages.py) ---
WORD_COUNT_THRESHOLD = 50          # < this => map/figure/blank page, not body text
APPENDIX_HEADER_MAX_WORDS = 100    # appendix header pages are short
APPENDIX_HEADER_CHARS = 80         # scan only the first chars (real headers appear immediately)
MIN_APPENDIX_PAGE = 5              # don't detect appendix before this page (avoids TOC hits)
WORDS_PER_REGULATORY_PAGE = 500    # 40 C.F.R. § 1508.1(bb)

# Final document type per source; page limits apply to EA + EIS only.
DOC_TYPE = {"EA": "EA", "EIS": "FEIS"}

# Filenames that explicitly omit appendices => use raw pages directly.
NO_APPENDIX_RE = r'(without|wo|no)[_ -]?(appendix|appendices|app|appx)'


def select_main_docs(con, source: str, sample: int | None) -> int:
    """Pick ONE main final document per project (main_document=YES preferred, tie -> most
    pages), tag the no-appendix-file shortcut, register as the temp view `target_docs`.
    project_id is a struct {'value': ...} in the processed docs. Returns row count."""
    docs_path = (PROCESSED / source.lower() / "documents.parquet").as_posix()
    samp = f"USING SAMPLE {int(sample)} ROWS (reservoir, 42)" if sample else ""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE target_docs AS
        WITH d AS (
            SELECT project_id.value AS project_id, document_id,
                   TRY_CAST(total_pages AS INTEGER) AS raw_pages,
                   COALESCE(file_name, '') AS file_name,
                   (main_document = 'YES') AS is_main,
                   regexp_matches(lower(COALESCE(file_name, '')), '{NO_APPENDIX_RE}') AS no_appx
            FROM read_parquet('{docs_path}')
            WHERE document_type = '{DOC_TYPE[source]}'
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY project_id
                -- no-appendix-file preferred (smallest/clean), then main flag, then most pages
                ORDER BY no_appx DESC, is_main DESC, raw_pages DESC
            ) AS rn
            FROM d
        )
        SELECT project_id, document_id, raw_pages, file_name, no_appx
        FROM ranked WHERE rn = 1 {samp}
    """)
    return con.execute("SELECT COUNT(*) FROM target_docs").fetchone()[0]


def compute_source(con, source: str, run_at: str, verbose: bool) -> None:
    """Run the regulatory-page computation for one source into TEMP TABLE res_<source>."""
    pages_path = (PROCESSED / source.lower() / "pages.parquet").as_posix()
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE res_{source} AS
        WITH ocr_targets AS (SELECT * FROM target_docs WHERE NOT no_appx),
        pages_raw AS (
            SELECT p.document_id,
                COALESCE(TRY_CAST(regexp_extract(CAST(p.page_number AS VARCHAR), '(\\d+)', 1) AS INTEGER), 1000000000) AS page_num,
                left(COALESCE(p.page_text, ''), {APPENDIX_HEADER_CHARS}) AS page_head,
                CASE WHEN trim(COALESCE(p.page_text, '')) = '' THEN 0
                     ELSE array_length(regexp_split_to_array(trim(COALESCE(p.page_text, '')), '\\s+')) END AS word_count
            FROM read_parquet('{pages_path}') p
            INNER JOIN ocr_targets USING (document_id)
        ),
        classified AS (
            SELECT document_id, page_num, word_count,
                CASE WHEN page_num >= {MIN_APPENDIX_PAGE}
                      AND word_count < {APPENDIX_HEADER_MAX_WORDS}
                      AND regexp_matches(page_head, '(^|\\n)\\s*(APPENDIX|Appendix|ATTACHMENT|Attachment|EXHIBIT|Exhibit)\\s+[A-Z0-9][^A-Za-z0-9]')
                      AND NOT regexp_matches(page_head, '(APPENDIX|Appendix|ATTACHMENT|Attachment|EXHIBIT|Exhibit)\\s+[A-Z0-9][^\\n]{{0,25}}[\\.…]{{4,}}')
                     THEN 1 ELSE 0 END AS is_appendix_header
            FROM pages_raw
        ),
        boundary AS (
            SELECT document_id, MIN(CASE WHEN is_appendix_header = 1 THEN page_num END) AS appendix_start_page
            FROM classified GROUP BY document_id
        ),
        summ AS (
            SELECT c.document_id, b.appendix_start_page, COUNT(*) AS total_parquet_pages,
                SUM(CASE WHEN c.page_num < COALESCE(b.appendix_start_page, 999999) AND c.word_count >= {WORD_COUNT_THRESHOLD} THEN 1 ELSE 0 END) AS body_pages,
                SUM(CASE WHEN c.page_num < COALESCE(b.appendix_start_page, 999999) AND c.word_count <  {WORD_COUNT_THRESHOLD} THEN 1 ELSE 0 END) AS low_content_pages,
                SUM(CASE WHEN c.page_num >= COALESCE(b.appendix_start_page, 999999) THEN 1 ELSE 0 END) AS appendix_pages,
                SUM(CASE WHEN c.page_num < COALESCE(b.appendix_start_page, 999999) AND c.word_count >= {WORD_COUNT_THRESHOLD} THEN c.word_count ELSE 0 END) AS body_word_count
            FROM classified c LEFT JOIN boundary b USING (document_id)
            GROUP BY c.document_id, b.appendix_start_page
        ),
        ocr AS (
            SELECT t.project_id, t.document_id, '{source}' AS dataset_source, t.raw_pages, t.file_name,
                s.appendix_start_page, s.total_parquet_pages, s.body_pages, s.low_content_pages,
                s.appendix_pages, s.body_word_count,
                CASE WHEN s.body_word_count = 0 OR s.body_word_count IS NULL THEN NULL
                     ELSE CEIL(CAST(s.body_word_count AS DOUBLE) / {WORDS_PER_REGULATORY_PAGE}) END AS regulatory_pages,
                'ocr' AS regulatory_pages_method
            FROM ocr_targets t LEFT JOIN summ s USING (document_id)
        ),
        shortcut AS (
            SELECT project_id, document_id, '{source}' AS dataset_source, raw_pages, file_name,
                CAST(NULL AS INTEGER) AS appendix_start_page, CAST(NULL AS BIGINT) AS total_parquet_pages,
                raw_pages AS body_pages, 0 AS low_content_pages, CAST(NULL AS BIGINT) AS appendix_pages,
                CAST(NULL AS BIGINT) AS body_word_count, CAST(raw_pages AS DOUBLE) AS regulatory_pages,
                'no_appendix_file' AS regulatory_pages_method
            FROM target_docs WHERE no_appx
        )
        SELECT *, TIMESTAMP '{run_at}' AS pages_extraction_run_at FROM ocr
        UNION ALL BY NAME
        SELECT *, TIMESTAMP '{run_at}' AS pages_extraction_run_at FROM shortcut
    """)
    n = con.execute(f"SELECT COUNT(*) FROM res_{source}").fetchone()[0]
    if verbose:
        print(f"  {source}: computed {n} projects in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regulatory page counts for the full Phase 2 EA/EIS corpus")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sample", type=int, help="Sample N projects per source (testing)")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="DuckDB threads (default: all cores)")
    ap.add_argument("--output", type=str, help="Override output parquet path")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    if not args.run:
        ap.print_help()
        return

    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "")
    out_path = Path(args.output) if args.output else OUT_PATH
    print(f"=== Regulatory page extraction (Phase 2, ALL EA/EIS) | threads={args.threads} ===", flush=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(args.threads)}")
    total_t0 = time.time()
    parts = []
    for source in ("EA", "EIS"):
        nd = select_main_docs(con, source, args.sample)
        print(f"--- {source}: {nd} main final documents selected ---", flush=True)
        compute_source(con, source, run_at, args.verbose)
        parts.append(f"res_{source}")

    union_sql = " UNION ALL BY NAME ".join(f"SELECT * FROM {p}" for p in parts)
    out_sql = out_path.as_posix().replace("'", "''")
    con.execute(f"COPY ({union_sql}) TO '{out_sql}' (FORMAT PARQUET)")
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"\nSaved {n_total:,} rows -> {out_path}  ({time.time() - total_t0:.0f}s total)", flush=True)

    # Summary
    print("\n=== Summary (regulatory pages vs FRA limits) ===")
    summary = con.execute(f"""
        SELECT dataset_source,
            COUNT(*) AS projects,
            SUM(regulatory_pages_method = 'ocr')::INT AS n_ocr,
            SUM(regulatory_pages_method = 'no_appendix_file')::INT AS n_shortcut,
            SUM(appendix_start_page IS NOT NULL)::INT AS n_with_appendix,
            ROUND(MEDIAN(raw_pages), 0) AS raw_median,
            ROUND(MEDIAN(regulatory_pages), 0) AS reg_median,
            ROUND(100.0 * AVG(CASE WHEN dataset_source='EA'  THEN (regulatory_pages <= 75)::INT
                                   WHEN dataset_source='EIS' THEN (regulatory_pages <= 150)::INT END), 0) AS pct_within_limit
        FROM read_parquet('{out_sql}') GROUP BY dataset_source ORDER BY dataset_source
    """).df()
    print(summary.to_string(index=False), flush=True)
    con.close()


if __name__ == "__main__":
    main()
