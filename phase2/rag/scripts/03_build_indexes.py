#!/usr/bin/env python3
"""Build one DuckDB FTS index file per RAG chunk shard."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

RAG_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = RAG_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nepa_rag.config import load_config
from nepa_rag.db import read_manifest, resolve_artifact_path, sql_path, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-id", action="append", default=None)
    parser.add_argument("--skip-fts", action="store_true", help="Materialize DuckDB shard without FTS.")
    return parser.parse_args()


def load_fts(con: duckdb.DuckDBPyConnection, extension_dir: Path) -> None:
    extension_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET extension_directory='{sql_path(extension_dir)}'")
    try:
        con.execute("LOAD fts")
    except duckdb.Error:
        try:
            con.execute("INSTALL fts")
            con.execute("LOAD fts")
        except duckdb.Error as exc:
            raise RuntimeError(
                "Could not install/load DuckDB FTS. "
                "If this machine is offline or blocks extension downloads, rerun with --skip-fts. "
                "The app will still use the LIKE fallback, but full-text ranking will be weaker."
            ) from exc


def build_index(config, shard: dict[str, object], *, skip_fts: bool) -> dict[str, object]:
    shard_id = str(shard["shard_id"])
    chunk_path = resolve_artifact_path(config, str(shard.get("chunk_path", "")))
    if chunk_path is None or not chunk_path.exists():
        raise FileNotFoundError(f"Chunk shard not found for {shard_id}: {chunk_path}")

    index_name = f"rag_chunks_{shard_id}.duckdb"
    index_path = config.data_dir / index_name
    if index_path.exists():
        index_path.unlink()
    wal = index_path.with_suffix(index_path.suffix + ".wal")
    if wal.exists():
        wal.unlink()

    print(f"[rag-index] building {index_name}")
    con = duckdb.connect(str(index_path))
    try:
        con.execute("DROP TABLE IF EXISTS rag_chunks")
        con.execute(
            f"""
            CREATE TABLE rag_chunks AS
            SELECT *
            FROM read_parquet('{sql_path(chunk_path)}')
            """
        )
        con.execute("CREATE INDEX idx_rag_chunks_project ON rag_chunks(project_id)")
        con.execute("CREATE INDEX idx_rag_chunks_doc ON rag_chunks(document_id)")
        con.execute("CREATE INDEX idx_rag_chunks_shard ON rag_chunks(shard_id)")
        row_count = con.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]

        if not skip_fts:
            load_fts(con, config.data_dir / ".duckdb_extensions")
            con.execute(
                """
                PRAGMA create_fts_index(
                    'rag_chunks',
                    'chunk_id',
                    'chunk_text',
                    stemmer='porter',
                    stopwords='english',
                    overwrite=1
                )
                """
            )
        con.execute("CHECKPOINT")
    finally:
        con.close()

    with duckdb.connect(str(config.catalog_path)) as catalog:
        catalog.execute(
            "UPDATE shards SET index_path = ?, index_rows = ? WHERE shard_id = ?",
            [index_name, row_count, shard_id],
        )

    updated = dict(shard)
    updated["index_path"] = index_name
    updated["index_rows"] = int(row_count)
    return updated


def main() -> None:
    args = parse_args()
    start = time.time()
    config = load_config()
    manifest = read_manifest(config)

    requested = set(args.shard_id or [])
    updated_shards = []
    for shard in manifest.get("shards", []):
        if requested and shard.get("shard_id") not in requested:
            updated_shards.append(shard)
            continue
        updated_shards.append(build_index(config, shard, skip_fts=args.skip_fts))

    manifest["shards"] = updated_shards
    write_manifest(config, manifest)

    elapsed = time.time() - start
    print(f"[rag-index] completed in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
