import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

# --------------------------
# REGULATORY PAGE COUNT EXTRACTION
# --------------------------
# Estimates the "regulatory page count" for clean energy EA/EIS final documents.
#
# The FRA (40 C.F.R. § 1508.1(bb)) defines a "page" as 500 words and excludes
# maps, diagrams, graphs, tables, and citations. This script:
#   1. Restricts to the main final document per clean energy project
#   2. Detects embedded appendix sections and excludes pages from that point on
#   3. Excludes low-content pages (< 50 words) within the body as likely maps/figures
#   4. Computes regulatory_pages = ceil(body_word_count / 500)
#
# Uses DuckDB to query parquet files directly — avoids loading the 5.5 GB EIS
# pages file into memory.
#
# Usage:
#   python code/extract/extract_pages.py --run
#   python code/extract/extract_pages.py --run --sample 50
#   python code/extract/extract_pages.py --run --verbose
#   python code/extract/extract_pages.py --run --output custom_output.parquet

import argparse
import re
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# --------------------------
# CONFIGURATION
# --------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

# Page body classification thresholds
WORD_COUNT_THRESHOLD = 50        # Pages with fewer words treated as maps/figures/blanks
APPENDIX_HEADER_MAX_WORDS = 100  # Max words on a page to be flagged as appendix section header
                                  # (excludes most TOC pages; real headers are short)
APPENDIX_HEADER_CHARS = 80       # Characters at page start to scan for appendix pattern —
                                  # real headers appear immediately; TOC pages open with doc title
MIN_APPENDIX_PAGE = 5            # Don't detect appendix headers before this page (avoids TOC hits)

# FRA regulatory standard
WORDS_PER_REGULATORY_PAGE = 500  # 40 C.F.R. § 1508.1(bb): 1 page = 500 words


# --------------------------
# HELPERS
# --------------------------

def _extract_pid(pid) -> str:
    """Extract string value from project_id (stored as dict {'value': ...} in EA/EIS docs)."""
    if isinstance(pid, dict):
        return pid.get('value', str(pid))
    return str(pid)


def load_clean_energy_main_docs(source: str, project_ids: set) -> pd.DataFrame:
    """
    Load the single best main final document per clean energy project from one source.

    Selects main_document=YES, document_type=EA (or FEIS), then keeps the document
    with the highest total_pages per project as a tiebreaker.

    Returns columns: project_id (str), document_id, raw_pages, file_name
    """
    doc_type = 'EA' if source == 'EA' else 'FEIS'
    docs_path = PROCESSED_DIR / source.lower() / "documents.parquet"

    docs = pd.read_parquet(
        docs_path,
        columns=['project_id', 'document_id', 'document_type', 'main_document', 'total_pages', 'file_name']
    )
    docs['project_id'] = docs['project_id'].apply(_extract_pid)

    docs = docs[
        docs['project_id'].isin(project_ids) &
        (docs['document_type'] == doc_type)
    ].copy()

    if docs.empty:
        return docs

    # One document per project — mirrors R's deduplication logic in 00_setup.R:
    #   main_document=YES preferred first; ties broken by highest total_pages.
    # This ensures every project that appears in pages_analysis also appears here,
    # preventing raw_fallback inflation in the R pipeline.
    docs['_is_main'] = (docs['main_document'] == 'YES').astype(int)
    docs = (
        docs.sort_values(['_is_main', 'total_pages'], ascending=[False, False])
            .drop_duplicates(subset='project_id', keep='first')
            .drop(columns=['_is_main', 'main_document'])
            .rename(columns={'total_pages': 'raw_pages'})
        [['project_id', 'document_id', 'raw_pages', 'file_name']]
        .reset_index(drop=True)
    )
    return docs


# Regex pattern matching filenames that explicitly omit appendices.
# Matches: wo_appendices, without_appendix, no_app, noappx, NoApp, etc.
_NO_APPENDIX_PATTERN = re.compile(
    r'(without|wo|no)[_\s-]?(appendix|appendices|app|appx)',
    re.IGNORECASE
)


def find_no_appendix_docs(source: str, project_ids: set) -> pd.DataFrame:
    """
    Find documents whose filename explicitly signals the appendix-free version
    (e.g. '*_wo_appendices.pdf', '*_no_app.pdf').

    Searches ALL documents for the project (not just main_document=YES) but
    restricts to the correct final document type (EA or FEIS).  When a match
    is found the file's total_pages is used directly as regulatory_pages —
    no OCR word-count extraction needed.

    Returns one row per matched project with columns:
      project_id, document_id, raw_pages, file_name, regulatory_pages_method
    When multiple matches exist for a project the one with the fewest pages is
    preferred (the stripped version is typically the smallest).
    """
    doc_type = 'EA' if source == 'EA' else 'FEIS'
    docs_path = PROCESSED_DIR / source.lower() / "documents.parquet"

    docs = pd.read_parquet(
        docs_path,
        columns=['project_id', 'document_id', 'document_type', 'total_pages', 'file_name']
    )
    docs['project_id'] = docs['project_id'].apply(_extract_pid)

    docs = docs[
        docs['project_id'].isin(project_ids) &
        (docs['document_type'] == doc_type)
    ].copy()

    if docs.empty:
        return pd.DataFrame()

    mask = docs['file_name'].fillna('').apply(
        lambda fn: bool(_NO_APPENDIX_PATTERN.search(fn))
    )
    matched = docs[mask].copy()

    if matched.empty:
        return pd.DataFrame()

    # Keep the smallest-page match per project (the stripped version)
    matched = (
        matched.sort_values('total_pages', ascending=True)
               .drop_duplicates(subset='project_id', keep='first')
               .rename(columns={'total_pages': 'raw_pages'})
        [['project_id', 'document_id', 'raw_pages', 'file_name']]
        .reset_index(drop=True)
    )
    matched['regulatory_pages_method'] = 'no_appendix_file'
    return matched


def compute_regulatory_pages(
    docs_df: pd.DataFrame,
    pages_path: Path,
    source: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Use DuckDB to classify pages and compute regulatory page counts for a set of documents.

    For each document:
      - Detect appendix section headers: pattern in first APPENDIX_HEADER_CHARS chars,
        page word count < APPENDIX_HEADER_MAX_WORDS, page number >= MIN_APPENDIX_PAGE
      - Find the first such page (appendix_start_page)
      - Count body pages (page_num < appendix_start_page) with word_count >= WORD_COUNT_THRESHOLD
      - Sum body word counts
      - regulatory_pages = ceil(body_word_count / 500)

    Returns a DataFrame with one row per document.
    """
    if docs_df.empty:
        return pd.DataFrame()

    pages_path_sql = pages_path.as_posix().replace("'", "''")

    con = duckdb.connect()
    try:
        con.register('target_docs', docs_df[['document_id', 'project_id', 'raw_pages']])

        query = f"""
        WITH pages_raw AS (
            SELECT
                p.document_id,
                COALESCE(
                    TRY_CAST(regexp_extract(CAST(p.page_number AS VARCHAR), '(\\d+)', 1) AS INTEGER),
                    1000000000
                ) AS page_num,
                left(coalesce(p.page_text, ''), {APPENDIX_HEADER_CHARS}) AS page_head,
                CASE
                    WHEN trim(coalesce(p.page_text, '')) = '' THEN 0
                    ELSE array_length(regexp_split_to_array(trim(coalesce(p.page_text, '')), '\\s+'))
                END AS word_count
            FROM read_parquet('{pages_path_sql}') p
            INNER JOIN target_docs USING (document_id)
        ),
        pages_classified AS (
            SELECT
                document_id,
                page_num,
                word_count,
                CASE
                    WHEN page_num >= {MIN_APPENDIX_PAGE}
                     AND word_count < {APPENDIX_HEADER_MAX_WORDS}
                     -- Pattern must appear near the very start of the page;
                     -- identifier must be a single char (A, B, 1, 2...) followed by non-alphanumeric
                     -- so "Exhibit 68" does not match but "Appendix A" or "Appendix 1" do
                     AND regexp_matches(
                           page_head,
                           '(^|\\n)\\s*(APPENDIX|Appendix|ATTACHMENT|Attachment|EXHIBIT|Exhibit)\\s+[A-Z0-9][^A-Za-z0-9]'
                         )
                     -- Exclude TOC entries: appendix label followed by dotted leaders and page number
                     AND NOT regexp_matches(
                           page_head,
                           '(APPENDIX|Appendix|ATTACHMENT|Attachment|EXHIBIT|Exhibit)\\s+[A-Z0-9][^\\n]{0,25}[\\.…]{4,}'
                         )
                    THEN 1 ELSE 0
                END AS is_appendix_header
            FROM pages_raw
        ),
        appendix_boundary AS (
            SELECT
                document_id,
                MIN(CASE WHEN is_appendix_header = 1 THEN page_num END) AS appendix_start_page
            FROM pages_classified
            GROUP BY document_id
        ),
        page_summary AS (
            SELECT
                pc.document_id,
                ab.appendix_start_page,
                COUNT(*) AS total_parquet_pages,
                SUM(CASE
                    WHEN pc.page_num < COALESCE(ab.appendix_start_page, 999999)
                     AND pc.word_count >= {WORD_COUNT_THRESHOLD}
                    THEN 1 ELSE 0 END) AS body_pages,
                SUM(CASE
                    WHEN pc.page_num < COALESCE(ab.appendix_start_page, 999999)
                     AND pc.word_count < {WORD_COUNT_THRESHOLD}
                    THEN 1 ELSE 0 END) AS low_content_pages,
                SUM(CASE
                    WHEN pc.page_num >= COALESCE(ab.appendix_start_page, 999999)
                    THEN 1 ELSE 0 END) AS appendix_pages,
                SUM(CASE
                    WHEN pc.page_num < COALESCE(ab.appendix_start_page, 999999)
                     AND pc.word_count >= {WORD_COUNT_THRESHOLD}
                    THEN pc.word_count ELSE 0 END) AS body_word_count
            FROM pages_classified pc
            LEFT JOIN appendix_boundary ab USING (document_id)
            GROUP BY pc.document_id, ab.appendix_start_page
        )
        SELECT
            td.project_id,
            ps.document_id,
            '{source}' AS dataset_source,
            td.raw_pages,
            ps.appendix_start_page,
            ps.total_parquet_pages,
            ps.body_pages,
            ps.low_content_pages,
            ps.appendix_pages,
            ps.body_word_count,
            CASE WHEN ps.body_word_count = 0 THEN NULL
                 ELSE CEIL(CAST(ps.body_word_count AS DOUBLE) / {WORDS_PER_REGULATORY_PAGE})
            END AS regulatory_pages,
            'ocr' AS regulatory_pages_method
        FROM page_summary ps
        JOIN target_docs td USING (document_id)
        ORDER BY td.project_id
        """

        if verbose:
            print(f"  Running DuckDB on {pages_path.name} ({len(docs_df)} docs)...")

        t0 = time.time()
        result = con.execute(query).df()
        elapsed = time.time() - t0

        if verbose:
            print(f"  Query complete in {elapsed:.1f}s — {len(result)} rows returned")

    finally:
        con.close()

    return result


# --------------------------
# MAIN EXTRACTION FUNCTION
# --------------------------

def run_page_extraction(
    sample_size: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Extract regulatory page counts for all clean energy EA and EIS final documents.
    """
    print("\n=== Regulatory Page Count Extraction ===")
    print(f"Settings: word_threshold={WORD_COUNT_THRESHOLD} words, "
          f"appendix_header_max={APPENDIX_HEADER_MAX_WORDS} words, "
          f"min_appendix_page={MIN_APPENDIX_PAGE}, "
          f"regulatory_page={WORDS_PER_REGULATORY_PAGE} words")

    # Load clean energy EA/EIS projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return None

    projects = pd.read_parquet(
        projects_path,
        columns=['project_id', 'dataset_source', 'project_energy_type']
    )
    print(f"Loaded {len(projects):,} total projects")

    projects = projects[
        (projects['project_energy_type'] == 'Clean') &
        (projects['dataset_source'].isin(['EA', 'EIS']))
    ].copy()
    print(f"Filtered to {len(projects):,} clean energy EA/EIS projects")

    if sample_size:
        projects = projects.sample(min(sample_size, len(projects)), random_state=42)
        print(f"Sampled {len(projects):,} projects")

    if projects.empty:
        print("No projects after filtering.")
        return None

    # Process each source
    all_results = []
    total_t0 = time.time()

    for source in ['EA', 'EIS']:
        source_projects = projects[projects['dataset_source'] == source]
        if source_projects.empty:
            continue

        project_ids = set(source_projects['project_id'].tolist())
        print(f"\n--- {source}: {len(project_ids)} projects ---")

        # --- Step 1: shortcut for explicitly appendix-free documents ----------
        # When a project has a file explicitly named as the appendix-free version
        # (e.g. *_wo_appendices.pdf), use its page count directly — no OCR needed.
        no_appx_df = find_no_appendix_docs(source, project_ids)
        no_appx_ids = set()

        if not no_appx_df.empty:
            no_appx_ids = set(no_appx_df['project_id'].tolist())
            # Build result rows to match the OCR output schema
            shortcut_rows = no_appx_df.assign(
                dataset_source=source,
                appendix_start_page=None,
                total_parquet_pages=None,
                body_pages=no_appx_df['raw_pages'],
                low_content_pages=0,
                appendix_pages=None,
                body_word_count=None,
                regulatory_pages=no_appx_df['raw_pages'].astype(float),
            )
            all_results.append(shortcut_rows)
            print(f"  No-appendix file shortcut: {len(no_appx_df)} projects "
                  f"({', '.join(no_appx_df['file_name'].tolist())})")

        # --- Step 2: OCR word-count extraction for remaining projects ----------
        ocr_ids = project_ids - no_appx_ids
        if not ocr_ids:
            continue

        docs_df = load_clean_energy_main_docs(source, ocr_ids)
        doc_type = 'EA' if source == 'EA' else 'FEIS'
        print(f"  Main {doc_type} documents for OCR: {len(docs_df)}")

        if docs_df.empty:
            continue

        pages_path = PROCESSED_DIR / source.lower() / "pages.parquet"
        result = compute_regulatory_pages(
            docs_df=docs_df,
            pages_path=pages_path,
            source=source,
            verbose=verbose,
        )

        if not result.empty:
            all_results.append(result)
            if verbose:
                n_appx = result['appendix_start_page'].notna().sum()
                pct_appx = n_appx / len(result) * 100
                print(f"  Embedded appendix detected: {n_appx}/{len(result)} ({pct_appx:.0f}%)")

    if not all_results:
        print("No results produced.")
        return None

    final_df = pd.concat(all_results, ignore_index=True)
    elapsed_total = time.time() - total_t0

    # Save
    save_path = Path(output_path) if output_path else ANALYSIS_DIR / "projects_page_counts.parquet"
    final_df.to_parquet(save_path, index=False)
    print(f"\nSaved {len(final_df):,} rows to: {save_path}  ({elapsed_total:.0f}s total)")

    # Summary
    print("\n=== Summary ===")
    for src in ['EA', 'EIS']:
        sub = final_df[final_df['dataset_source'] == src]
        if sub.empty:
            continue
        has_appendix = sub['appendix_start_page'].notna()
        n_no_appx_file = (sub['regulatory_pages_method'] == 'no_appendix_file').sum()
        n_ocr = (sub['regulatory_pages_method'] == 'ocr').sum()
        limit = 75 if src == 'EA' else 150
        within_limit = (sub['regulatory_pages'] <= limit).sum()
        within_limit_pct = within_limit / len(sub) * 100

        print(f"\n{src} ({len(sub):,} projects):")
        print(f"  Method: {n_ocr} OCR, {n_no_appx_file} no-appendix-file shortcut")
        print(f"  Embedded appendix (OCR): {has_appendix.sum()} ({has_appendix.mean()*100:.0f}%)")
        print(f"  Raw pages:        median={sub['raw_pages'].median():.0f}, "
              f"mean={sub['raw_pages'].mean():.1f}, max={sub['raw_pages'].max():.0f}")
        print(f"  Regulatory pages: median={sub['regulatory_pages'].median():.0f}, "
              f"mean={sub['regulatory_pages'].mean():.1f}, max={sub['regulatory_pages'].max():.0f}")
        print(f"  Within {limit}-page FRA limit: {within_limit}/{len(sub)} ({within_limit_pct:.0f}%)")

    return final_df


# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract regulatory page counts for clean energy EA/EIS final documents"
    )
    parser.add_argument('--run', action='store_true',
                        help='Run extraction')
    parser.add_argument('--sample', type=int,
                        help='Sample N projects (for testing)')
    parser.add_argument('--output', type=str,
                        help='Output path (default: data/analysis/projects_page_counts.parquet)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output including per-source stats')

    args = parser.parse_args()

    if args.run:
        run_page_extraction(
            sample_size=args.sample,
            output_path=args.output,
            verbose=args.verbose,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python code/extract/extract_pages.py --run")
        print("  python code/extract/extract_pages.py --run --sample 50")
        print("  python code/extract/extract_pages.py --run --verbose")
        print("  python code/extract/extract_pages.py --run --output custom.parquet")


if __name__ == "__main__":
    main()
