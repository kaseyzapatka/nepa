#!/usr/bin/env python3
"""Build the Phase 2 RAG catalog and sharded page parquet artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

RAG_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = RAG_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nepa_rag.config import RagConfig, load_config
from nepa_rag.db import sql_path, write_manifest

PROJECT_COLS = [
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
    "noi_publication_date",
    "noi_document_number",
    "noi_url",
    "noi_match_confidence",
    "noi_match_status",
]

DOC_COLS = [
    "document_id",
    "project_id",
    "dataset_source",
    "document_type",
    "document_title",
    "document_type_clean",
    "document_type_category",
    "main_document",
    "file_name",
    "total_pages",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-types", nargs="+", default=["CE", "EA", "EIS"])
    parser.add_argument("--eis-shard-documents", type=int, default=500)
    parser.add_argument("--project-energy-type", default=None, help="Default comes from RAG_PROJECT_ENERGY_TYPE.")
    parser.add_argument("--all-projects", action="store_true", help="Do not filter to clean energy projects.")
    parser.add_argument(
        "--sample-documents-per-process",
        type=int,
        default=None,
        help="Limit documents per process type for smoke-test builds.",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--memory-limit", default=None)
    return parser.parse_args()


def configure(con: duckdb.DuckDBPyConnection, config: RagConfig, args: argparse.Namespace) -> None:
    extension_dir = config.data_dir / ".duckdb_extensions"
    extension_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET extension_directory='{sql_path(extension_dir)}'")
    if args.threads:
        con.execute(f"PRAGMA threads={int(args.threads)}")
    if args.memory_limit:
        con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")


def ensure_inputs(config: RagConfig, process_types: list[str]) -> None:
    missing = [config.projects_path, config.documents_path]
    missing.extend(config.pages_path(pt) for pt in process_types)
    missing = [path for path in missing if not path.exists()]
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required inputs:\n{detail}")


def remove_old_artifacts(config: RagConfig) -> None:
    if config.manifest_path.exists():
        try:
            old_manifest = json.loads(config.manifest_path.read_text())
        except json.JSONDecodeError:
            old_manifest = {}
        for shard in old_manifest.get("shards", []):
            for key in ("page_path", "chunk_path", "index_path"):
                value = shard.get(key)
                if not value:
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = config.data_dir / path
                if path.exists() and path.is_file():
                    path.unlink()
    if config.catalog_path.exists():
        config.catalog_path.unlink()
    wal = config.catalog_path.with_suffix(config.catalog_path.suffix + ".wal")
    if wal.exists():
        wal.unlink()


def build_catalog(con: duckdb.DuckDBPyConnection, config: RagConfig, args: argparse.Namespace) -> None:
    projects_path = sql_path(config.projects_path)
    documents_path = sql_path(config.documents_path)
    process_types = [pt.upper() for pt in args.process_types]
    process_placeholders = ",".join("?" for _ in process_types)

    energy_type = args.project_energy_type or config.project_energy_type
    where_clauses = [f"upper(process_type) IN ({process_placeholders})"]
    params: list[object] = list(process_types)
    if not args.all_projects and energy_type and energy_type.upper() != "ALL":
        where_clauses.append("project_energy_type = ?")
        params.append(energy_type)
    project_where = "WHERE " + " AND ".join(where_clauses)

    con.execute(
        f"""
        CREATE TABLE projects AS
        SELECT {', '.join(PROJECT_COLS)}
        FROM read_parquet('{projects_path}')
        {project_where}
        """,
        params,
    )

    con.execute(
        f"""
        CREATE TABLE documents_base AS
        SELECT {', '.join(DOC_COLS)}
        FROM read_parquet('{documents_path}')
        WHERE project_id IN (SELECT project_id FROM projects)
          AND upper(dataset_source) IN ({process_placeholders})
        """,
        process_types,
    )

    if args.sample_documents_per_process:
        con.execute(
            """
            CREATE TABLE documents AS
            SELECT *
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY upper(dataset_source)
                        ORDER BY main_document DESC, total_pages DESC NULLS LAST, document_id
                    ) AS doc_rank
                FROM documents_base
            )
            WHERE doc_rank <= ?
            """,
            [args.sample_documents_per_process],
        )
        con.execute("ALTER TABLE documents DROP COLUMN doc_rank")
    else:
        con.execute("CREATE TABLE documents AS SELECT * FROM documents_base")

    con.execute("DROP TABLE documents_base")
    con.execute(
        """
        CREATE TABLE projects_with_documents AS
        SELECT *
        FROM projects
        WHERE project_id IN (SELECT DISTINCT project_id FROM documents)
        """
    )
    con.execute("DROP TABLE projects")
    con.execute("ALTER TABLE projects_with_documents RENAME TO projects")

    con.execute(
        """
        CREATE TABLE shard_documents AS
        WITH ordered AS (
            SELECT
                document_id,
                project_id,
                upper(dataset_source) AS process_type,
                row_number() OVER (
                    PARTITION BY upper(dataset_source)
                    ORDER BY document_id
                ) AS rn
            FROM documents
        )
        SELECT
            document_id,
            project_id,
            process_type,
            CASE
                WHEN process_type = 'EIS'
                    THEN 'eis_' || lpad(CAST(floor((rn - 1) / ?) AS VARCHAR), 3, '0')
                ELSE lower(process_type)
            END AS shard_id
        FROM ordered
        """,
        [max(1, args.eis_shard_documents)],
    )

    con.execute(
        """
        CREATE TABLE shards AS
        SELECT
            shard_id,
            process_type,
            COUNT(DISTINCT document_id) AS document_count,
            COUNT(DISTINCT project_id) AS project_count,
            NULL::VARCHAR AS page_path,
            NULL::BIGINT AS page_rows,
            NULL::VARCHAR AS chunk_path,
            NULL::BIGINT AS chunk_rows,
            NULL::VARCHAR AS index_path,
            NULL::BIGINT AS index_rows
        FROM shard_documents
        GROUP BY shard_id, process_type
        ORDER BY process_type, shard_id
        """
    )

    con.execute("CREATE INDEX idx_projects_id ON projects(project_id)")
    con.execute("CREATE INDEX idx_documents_project ON documents(project_id)")
    con.execute("CREATE INDEX idx_documents_doc ON documents(document_id)")
    con.execute("CREATE INDEX idx_shard_documents_project ON shard_documents(project_id)")
    con.execute("CREATE INDEX idx_shard_documents_doc ON shard_documents(document_id)")


def export_page_shards(con: duckdb.DuckDBPyConnection, config: RagConfig) -> list[dict[str, object]]:
    shard_rows = con.execute(
        "SELECT shard_id, process_type FROM shards ORDER BY process_type, shard_id"
    ).fetchall()

    manifest_shards: list[dict[str, object]] = []
    for shard_id, process_type in shard_rows:
        pages_path = config.pages_path(process_type)
        output_name = f"pages_{shard_id}.parquet"
        output_path = config.data_dir / output_name
        if output_path.exists():
            output_path.unlink()

        print(f"[rag-store] writing {output_name}")
        con.execute(
            f"""
            COPY (
                SELECT
                    s.shard_id,
                    s.process_type,
                    p.document_id,
                    s.project_id,
                    CAST(p.page_number AS VARCHAR) AS page_number,
                    COALESCE(
                        TRY_CAST(
                            regexp_extract(CAST(p.page_number AS VARCHAR), '([0-9]+)', 1)
                            AS INTEGER
                        ),
                        0
                    ) AS page_number_int,
                    CAST(p.page_text AS VARCHAR) AS page_text
                FROM read_parquet('{sql_path(pages_path)}') p
                INNER JOIN shard_documents s
                    ON p.document_id = s.document_id
                WHERE s.shard_id = ?
            ) TO '{sql_path(output_path)}' (FORMAT PARQUET)
            """,
            [shard_id],
        )

        page_rows = duckdb.sql(
            "SELECT COUNT(*) FROM read_parquet(?)",
            params=[output_path.as_posix()],
        ).fetchone()[0]
        con.execute(
            "UPDATE shards SET page_path = ?, page_rows = ? WHERE shard_id = ?",
            [output_name, page_rows, shard_id],
        )

        counts = con.execute(
            """
            SELECT document_count, project_count
            FROM shards
            WHERE shard_id = ?
            """,
            [shard_id],
        ).fetchone()
        manifest_shards.append(
            {
                "shard_id": shard_id,
                "process_type": process_type,
                "page_path": output_name,
                "page_rows": int(page_rows),
                "document_count": int(counts[0]),
                "project_count": int(counts[1]),
            }
        )
    return manifest_shards


def main() -> None:
    args = parse_args()
    start = time.time()
    config = load_config()
    process_types = [pt.upper() for pt in args.process_types]

    ensure_inputs(config, process_types)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    remove_old_artifacts(config)

    con = duckdb.connect(str(config.catalog_path))
    try:
        configure(con, config, args)
        print("[rag-store] building catalog")
        build_catalog(con, config, args)
        manifest_shards = export_page_shards(con, config)
        con.execute("CHECKPOINT")
    finally:
        con.close()

    manifest = {
        "build_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog_path": config.catalog_path.name,
        "project_energy_type": "ALL" if args.all_projects else (args.project_energy_type or config.project_energy_type),
        "process_types": process_types,
        "eis_shard_documents": args.eis_shard_documents,
        "sample_documents_per_process": args.sample_documents_per_process,
        "shards": manifest_shards,
    }
    write_manifest(config, manifest)

    elapsed = time.time() - start
    print(f"[rag-store] wrote {len(manifest_shards)} page shard(s) in {elapsed / 60:.1f} minutes")
    print(f"[rag-store] catalog: {config.catalog_path}")
    print(f"[rag-store] manifest: {config.manifest_path}")


if __name__ == "__main__":
    main()
