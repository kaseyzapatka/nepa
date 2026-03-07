#!/usr/bin/env python3
"""
Build the NEPA document browser DuckDB database from existing parquet files.

Run once locally; upload the resulting data/rag/nepa_reader.duckdb to HF Spaces.

Usage:
    python code/rag/01_build_text_store.py
    python code/rag/01_build_text_store.py --threads 6 --memory-limit 24GB
"""

from __future__ import annotations

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import time
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "rag"
DB_PATH = OUTPUT_DIR / "nepa_reader.duckdb"

INPUT_FILES = {
    "projects": DATA_DIR / "analysis" / "projects_combined.parquet",
    "documents": DATA_DIR / "analysis" / "documents_combined.parquet",
    "ce_pages": DATA_DIR / "processed" / "ce" / "pages.parquet",
    "ea_pages": DATA_DIR / "processed" / "ea" / "pages.parquet",
    "eis_pages": DATA_DIR / "processed" / "eis" / "pages.parquet",
}

PROJECTS_COLS = [
    "project_id",
    "project_title",
    "lead_agency",
    "lead_agency_harmonized",
    "project_state",
    "process_type",
    "project_energy_type",
    "project_type",
    "project_description",
    "project_department",
]

DOCS_COLS = [
    "document_id",
    "project_id",
    "dataset_source",
    "document_type",
    "document_type_clean",
    "document_type_category",
    "main_document",
    "file_name",
    "total_pages",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/rag/nepa_reader.duckdb from existing parquet sources."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional DuckDB worker thread cap. Useful if you want to reduce system load.",
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Optional DuckDB memory limit, e.g. 16GB.",
    )
    parser.add_argument(
        "--skip-fts",
        action="store_true",
        help="Skip full-text index creation on pages.page_text.",
    )
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def ensure_inputs_exist() -> None:
    missing = [name for name, path in INPUT_FILES.items() if not path.exists()]
    if missing:
        details = "\n".join(f"  - {name}: {INPUT_FILES[name]}" for name in missing)
        raise FileNotFoundError(f"Missing required input files:\n{details}")


def configure_connection(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    extension_dir = OUTPUT_DIR / ".duckdb_extensions"
    extension_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET extension_directory='{sql_path(extension_dir)}'")

    if args.threads is not None:
        con.execute(f"PRAGMA threads={int(args.threads)}")
    if args.memory_limit:
        con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")


def build_projects_table(con: duckdb.DuckDBPyConnection) -> None:
    projects_path = sql_path(INPUT_FILES["projects"])

    con.execute(f"""
        CREATE TABLE projects AS
        SELECT {', '.join(PROJECTS_COLS)}
        FROM read_parquet('{projects_path}')
        WHERE project_energy_type = 'Clean'
    """)

    n = con.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    print(f"projects table: {n:,} rows")


def build_documents_table(con: duckdb.DuckDBPyConnection) -> None:
    docs_path = sql_path(INPUT_FILES["documents"])

    con.execute(f"""
        CREATE TABLE documents AS
        SELECT {', '.join(DOCS_COLS)}
        FROM read_parquet('{docs_path}')
        WHERE project_id IN (SELECT project_id FROM projects)
    """)

    n = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"documents table: {n:,} rows")


def build_pages_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE pages (
            document_id VARCHAR,
            page_number VARCHAR,
            page_number_int INTEGER,
            page_text VARCHAR
        )
        """
    )

    for source in ("ce", "ea", "eis"):
        pages_key = f"{source}_pages"
        pages_path = sql_path(INPUT_FILES[pages_key])
        source_upper = source.upper()

        print(f"Loading {source_upper} pages from {INPUT_FILES[pages_key]} ...")

        con.execute(f"""
            INSERT INTO pages
            SELECT
                p.document_id,
                CAST(p.page_number AS VARCHAR) AS page_number,
                COALESCE(
                    TRY_CAST(
                        regexp_extract(CAST(p.page_number AS VARCHAR), '([0-9]+)', 1)
                        AS INTEGER
                    ),
                    0
                ) AS page_number_int,
                CAST(p.page_text AS VARCHAR) AS page_text
            FROM read_parquet('{pages_path}') p
            INNER JOIN documents d
                ON p.document_id = d.document_id
               AND d.dataset_source = '{source_upper}'
        """)

        n = con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        print(f"  pages table now: {n:,} rows")


def create_indexes(con: duckdb.DuckDBPyConnection, skip_fts: bool) -> None:
    print("Creating B-tree indexes ...")
    con.execute("CREATE INDEX idx_pages_doc ON pages(document_id)")
    con.execute("CREATE INDEX idx_docs_project ON documents(project_id)")

    if skip_fts:
        print("Skipping FTS index (--skip-fts).")
        return

    print("Creating FTS index on pages.page_text ...")
    try:
        con.execute("LOAD fts")
    except duckdb.Error:
        # First run may need extension install.
        try:
            con.execute("INSTALL fts")
            con.execute("LOAD fts")
        except duckdb.Error as exc:
            raise RuntimeError(
                "Could not install/load DuckDB FTS extension. "
                "If you are offline or blocked by permissions, rerun with --skip-fts."
            ) from exc

    con.execute(
        """
        PRAGMA create_fts_index(
            'pages',
            'rowid',
            'page_text',
            overwrite=1
        )
        """
    )
    print("FTS index created.")


def print_summary(con: duckdb.DuckDBPyConnection) -> None:
    print("\nTable counts:")
    for table in ("projects", "documents", "pages"):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,} rows")


def main() -> None:
    args = parse_args()
    start = time.time()

    ensure_inputs_exist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    try:
        configure_connection(con, args)

        print("Building nepa_reader.duckdb ...")
        build_projects_table(con)
        build_documents_table(con)
        build_pages_table(con)
        create_indexes(con, skip_fts=args.skip_fts)

        # Force persistence before file size reporting.
        con.execute("CHECKPOINT")
        print_summary(con)
    finally:
        con.close()

    elapsed = time.time() - start
    size_gb = DB_PATH.stat().st_size / 1e9
    print(f"\nDatabase written to: {DB_PATH}")
    print(f"Size: {size_gb:.2f} GB")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
