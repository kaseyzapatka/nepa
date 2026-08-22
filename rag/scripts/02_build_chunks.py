#!/usr/bin/env python3
"""Build paragraph-aware RAG chunk parquet shards from page shards."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

RAG_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = RAG_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nepa_rag.chunking import chunk_page_text, source_label
from nepa_rag.config import load_config
from nepa_rag.db import read_manifest, resolve_artifact_path, sql_path, write_manifest

CHUNK_COLUMNS = [
    "chunk_id",
    "shard_id",
    "process_type",
    "project_id",
    "project_title",
    "lead_agency",
    "lead_agency_harmonized",
    "project_state",
    "project_type",
    "document_id",
    "file_name",
    "document_type",
    "document_type_clean",
    "document_type_category",
    "main_document",
    "page_number",
    "page_number_int",
    "chunk_index_on_page",
    "char_start",
    "char_end",
    "chunk_text",
    "token_estimate",
    "source_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-id", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--target-tokens", type=int, default=1200)
    parser.add_argument("--overlap-tokens", type=int, default=150)
    return parser.parse_args()


def create_chunk_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS rag_chunks")
    con.execute(
        """
        CREATE TABLE rag_chunks (
            chunk_id VARCHAR,
            shard_id VARCHAR,
            process_type VARCHAR,
            project_id VARCHAR,
            project_title VARCHAR,
            lead_agency VARCHAR,
            lead_agency_harmonized VARCHAR,
            project_state VARCHAR,
            project_type VARCHAR,
            document_id VARCHAR,
            file_name VARCHAR,
            document_type VARCHAR,
            document_type_clean VARCHAR,
            document_type_category VARCHAR,
            main_document VARCHAR,
            page_number VARCHAR,
            page_number_int INTEGER,
            chunk_index_on_page INTEGER,
            char_start INTEGER,
            char_end INTEGER,
            chunk_text VARCHAR,
            token_estimate INTEGER,
            source_label VARCHAR
        )
        """
    )


def insert_chunk_rows(con: duckdb.DuckDBPyConnection, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=CHUNK_COLUMNS)
    con.register("chunk_batch", df)
    try:
        con.execute(f"INSERT INTO rag_chunks SELECT {', '.join(CHUNK_COLUMNS)} FROM chunk_batch")
    finally:
        con.unregister("chunk_batch")


def build_shard(config, shard: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    shard_id = str(shard["shard_id"])
    page_path = resolve_artifact_path(config, str(shard.get("page_path", "")))
    if page_path is None or not page_path.exists():
        raise FileNotFoundError(f"Page shard not found for {shard_id}: {page_path}")

    output_name = f"chunks_{shard_id}.parquet"
    output_path = config.data_dir / output_name
    if output_path.exists():
        output_path.unlink()

    stage_path = config.data_dir / f".chunk_build_{shard_id}.duckdb"
    if stage_path.exists():
        stage_path.unlink()

    print(f"[rag-chunks] building {output_name}")
    stage = duckdb.connect(str(stage_path))
    create_chunk_table(stage)

    catalog = duckdb.connect(str(config.catalog_path), read_only=True)
    chunk_count = 0
    try:
        reader = catalog.execute(
            f"""
            SELECT
                p.shard_id,
                p.process_type,
                p.document_id,
                p.project_id,
                p.page_number,
                p.page_number_int,
                p.page_text,
                pr.project_title,
                pr.lead_agency,
                pr.lead_agency_harmonized,
                pr.project_state,
                pr.project_type,
                d.file_name,
                d.document_type,
                d.document_type_clean,
                d.document_type_category,
                d.main_document
            FROM read_parquet('{sql_path(page_path)}') p
            JOIN documents d
                ON p.document_id = d.document_id
               AND p.project_id = d.project_id
            JOIN projects pr
                ON p.project_id = pr.project_id
            ORDER BY p.document_id, p.page_number_int, p.page_number
            """
        ).fetch_record_batch(rows_per_batch=max(1, args.batch_size))

        for batch in reader:
            df = batch.to_pandas()
            rows: list[dict[str, object]] = []
            for record in df.to_dict("records"):
                label = source_label(
                    str(record.get("project_title") or ""),
                    str(record.get("project_id") or ""),
                    str(record.get("file_name") or ""),
                    str(record.get("document_id") or ""),
                    str(record.get("page_number") or ""),
                )
                chunks = chunk_page_text(
                    record.get("page_text"),
                    shard_id=shard_id,
                    document_id=str(record.get("document_id") or ""),
                    page_number=str(record.get("page_number") or ""),
                    target_tokens=args.target_tokens,
                    overlap_tokens=args.overlap_tokens,
                )
                for chunk in chunks:
                    rows.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "shard_id": shard_id,
                            "process_type": record.get("process_type"),
                            "project_id": record.get("project_id"),
                            "project_title": record.get("project_title"),
                            "lead_agency": record.get("lead_agency"),
                            "lead_agency_harmonized": record.get("lead_agency_harmonized"),
                            "project_state": record.get("project_state"),
                            "project_type": record.get("project_type"),
                            "document_id": record.get("document_id"),
                            "file_name": record.get("file_name"),
                            "document_type": record.get("document_type"),
                            "document_type_clean": record.get("document_type_clean"),
                            "document_type_category": record.get("document_type_category"),
                            "main_document": record.get("main_document"),
                            "page_number": record.get("page_number"),
                            "page_number_int": int(record.get("page_number_int") or 0),
                            "chunk_index_on_page": chunk.chunk_index_on_page,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "chunk_text": chunk.chunk_text,
                            "token_estimate": chunk.token_estimate,
                            "source_label": label,
                        }
                    )
            insert_chunk_rows(stage, rows)
            chunk_count += len(rows)
    finally:
        catalog.close()

    stage.execute(f"COPY rag_chunks TO '{sql_path(output_path)}' (FORMAT PARQUET)")
    stage.close()
    stage_path.unlink(missing_ok=True)

    with duckdb.connect(str(config.catalog_path)) as con:
        con.execute(
            "UPDATE shards SET chunk_path = ?, chunk_rows = ? WHERE shard_id = ?",
            [output_name, chunk_count, shard_id],
        )

    updated = dict(shard)
    updated["chunk_path"] = output_name
    updated["chunk_rows"] = chunk_count
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
        updated_shards.append(build_shard(config, shard, args))

    manifest["shards"] = updated_shards
    write_manifest(config, manifest)

    elapsed = time.time() - start
    print(f"[rag-chunks] completed in {elapsed / 60:.1f} minutes")
    print("[rag-chunks] run 03_build_indexes.py next; FTS indexes are stale after chunk rebuilds")


if __name__ == "__main__":
    main()
