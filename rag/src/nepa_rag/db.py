from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import RagConfig


def sql_path(path: Path | str) -> str:
    return Path(path).as_posix().replace("'", "''")


def read_manifest(config: RagConfig) -> dict[str, Any]:
    if not config.manifest_path.exists():
        raise FileNotFoundError(f"RAG manifest not found: {config.manifest_path}")
    return json.loads(config.manifest_path.read_text())


def write_manifest(config: RagConfig, manifest: dict[str, Any]) -> None:
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(config.manifest_path)


def resolve_artifact_path(config: RagConfig, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (config.data_dir / path).resolve()


def connect_catalog(config: RagConfig, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if read_only and not config.catalog_path.exists():
        raise FileNotFoundError(f"RAG catalog not found: {config.catalog_path}")
    return duckdb.connect(str(config.catalog_path), read_only=read_only)


def run_df(con: duckdb.DuckDBPyConnection, query: str, params: list[Any] | None = None) -> pd.DataFrame:
    if params is None:
        return con.execute(query).df()
    return con.execute(query, params).df()


def project_index(config: RagConfig) -> pd.DataFrame:
    with connect_catalog(config) as con:
        return con.execute(
            """
            SELECT
                p.project_id,
                p.project_title,
                p.lead_agency,
                p.lead_agency_harmonized,
                p.project_state,
                p.process_type,
                p.project_type,
                COUNT(DISTINCT d.document_id) AS n_documents
            FROM projects p
            LEFT JOIN documents d USING (project_id)
            GROUP BY 1,2,3,4,5,6,7
            ORDER BY p.project_title, p.project_id
            """
        ).df()


def search_projects(
    config: RagConfig,
    *,
    title_query: str = "",
    process_types: list[str] | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    title_query = (title_query or "").strip()
    process_types = process_types or []
    clauses = ["1=1"]
    params: list[Any] = []

    if title_query:
        clauses.append("project_title ILIKE ?")
        params.append(f"%{title_query}%")
    if process_types:
        placeholders = ",".join("?" for _ in process_types)
        clauses.append(f"process_type IN ({placeholders})")
        params.extend(process_types)

    params.append(limit)
    with connect_catalog(config) as con:
        return con.execute(
            f"""
            SELECT project_id, project_title, lead_agency_harmonized, project_state,
                   process_type, project_type
            FROM projects
            WHERE {' AND '.join(clauses)}
            ORDER BY project_title, project_id
            LIMIT ?
            """,
            params,
        ).df()


def get_project(config: RagConfig, project_id: str) -> pd.DataFrame:
    with connect_catalog(config) as con:
        return con.execute("SELECT * FROM projects WHERE project_id = ?", [project_id]).df()


def get_documents(config: RagConfig, project_id: str) -> pd.DataFrame:
    with connect_catalog(config) as con:
        return con.execute(
            """
            SELECT *
            FROM documents
            WHERE project_id = ?
            ORDER BY
                CASE WHEN main_document = 'YES' THEN 0 ELSE 1 END,
                total_pages DESC NULLS LAST,
                file_name
            """,
            [project_id],
        ).df()


def shard_ids_for_project(config: RagConfig, project_id: str) -> list[str]:
    with connect_catalog(config) as con:
        rows = con.execute(
            """
            SELECT DISTINCT shard_id
            FROM shard_documents
            WHERE project_id = ?
            ORDER BY shard_id
            """,
            [project_id],
        ).fetchall()
    return [row[0] for row in rows]


def process_for_project(config: RagConfig, project_id: str) -> str | None:
    with connect_catalog(config) as con:
        row = con.execute(
            "SELECT process_type FROM projects WHERE project_id = ? LIMIT 1",
            [project_id],
        ).fetchone()
    return row[0] if row else None


def local_shards(config: RagConfig, *, shard_ids: list[str] | None = None) -> list[dict[str, Any]]:
    manifest = read_manifest(config)
    shards = manifest.get("shards", [])
    if shard_ids:
        wanted = set(shard_ids)
        shards = [shard for shard in shards if shard.get("shard_id") in wanted]
    return shards


def pages_for_document(config: RagConfig, document_id: str) -> pd.DataFrame:
    with connect_catalog(config) as con:
        shard_rows = con.execute(
            """
            SELECT DISTINCT shard_id
            FROM shard_documents
            WHERE document_id = ?
            ORDER BY shard_id
            """,
            [document_id],
        ).fetchall()

    if not shard_rows:
        return pd.DataFrame()

    shard_id = shard_rows[0][0]
    shard = next((s for s in local_shards(config) if s.get("shard_id") == shard_id), None)
    if not shard:
        return pd.DataFrame()

    page_path = resolve_artifact_path(config, shard.get("page_path") or shard.get("page_glob"))
    if not page_path:
        return pd.DataFrame()

    with duckdb.connect() as con:
        return con.execute(
            """
            SELECT
                row_number() OVER (ORDER BY page_number_int, page_number) AS page_ordinal,
                page_number,
                page_number_int,
                page_text
            FROM read_parquet(?)
            WHERE document_id = ?
            ORDER BY page_number_int, page_number
            """,
            [page_path.as_posix(), document_id],
        ).df()
