from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb

from .config import RagConfig
from .db import local_shards, resolve_artifact_path, shard_ids_for_project
from .evidence import Evidence
from .query_router import QueryScope

STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before", "between",
    "could", "does", "from", "have", "into", "nepa", "over", "project", "show",
    "that", "their", "there", "these", "this", "what", "when", "where", "which",
    "with", "would", "your",
}


def keywords_for_query(question: str, *, max_terms: int = 10) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", question.lower()):
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:max_terms]


def retrieve_evidence(config: RagConfig, question: str, scope: QueryScope) -> list[Evidence]:
    shard_ids: list[str] | None = None
    if scope.project_id:
        shard_ids = shard_ids_for_project(config, scope.project_id)

    shards = local_shards(config, shard_ids=shard_ids)
    if scope.process_types and not scope.project_id:
        wanted = {item.upper() for item in scope.process_types}
        shards = [s for s in shards if str(s.get("process_type", "")).upper() in wanted]

    results: list[Evidence] = []
    for shard in shards:
        evidence = _search_shard(config, shard, question, project_id=scope.project_id)
        results.extend(evidence)

    results.sort(key=_evidence_sort_key, reverse=True)
    return dedupe_evidence(results)[: config.max_context_passages]


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str]] = set()
    output: list[Evidence] = []
    for item in items:
        key = (item.document_id, str(item.page_number), item.chunk_text[:120])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _evidence_sort_key(item: Evidence) -> tuple[int, float]:
    main_boost = 1 if str(item.main_document).upper() == "YES" else 0
    return main_boost, float(item.search_score or 0.0)


def _search_shard(
    config: RagConfig,
    shard: dict[str, Any],
    question: str,
    *,
    project_id: str | None,
) -> list[Evidence]:
    index_path = resolve_artifact_path(config, shard.get("index_path"))
    if not index_path or not index_path.exists():
        return []

    query_terms = keywords_for_query(question)
    fts_query = " ".join(query_terms) if query_terms else question
    try:
        return _search_shard_fts(index_path, fts_query, project_id, config.top_k_per_shard)
    except duckdb.Error:
        return _search_shard_like(index_path, query_terms, project_id, config.top_k_per_shard)


def _search_shard_fts(index_path: Path, query: str, project_id: str | None, limit: int) -> list[Evidence]:
    where = ["score IS NOT NULL"]
    params: list[Any] = [query]
    if project_id:
        where.append("project_id = ?")
        params.append(project_id)
    params.append(limit)

    with duckdb.connect(str(index_path), read_only=True) as con:
        try:
            con.execute("LOAD fts")
        except duckdb.Error:
            pass
        rows = con.execute(
            f"""
            WITH scored AS (
                SELECT
                    *,
                    fts_main_rag_chunks.match_bm25(chunk_id, ?) AS score
                FROM rag_chunks
            )
            SELECT *
            FROM scored
            WHERE {' AND '.join(where)}
            ORDER BY
                CASE WHEN main_document = 'YES' THEN 1 ELSE 0 END DESC,
                score DESC
            LIMIT ?
            """,
            params,
        ).df()
    return [_row_to_evidence(row) for _, row in rows.iterrows()]


def _search_shard_like(index_path: Path, terms: list[str], project_id: str | None, limit: int) -> list[Evidence]:
    if not terms:
        return []
    clauses = []
    params: list[Any] = []
    for term in terms[:6]:
        clauses.append("chunk_text ILIKE ?")
        params.append(f"%{term}%")
    where = [f"({' OR '.join(clauses)})"]
    if project_id:
        where.append("project_id = ?")
        params.append(project_id)
    params.append(limit)

    escaped_terms = [term.replace("'", "''") for term in terms[:6]]
    score_expr = " + ".join(
        [f"CASE WHEN chunk_text ILIKE '%{term}%' THEN 1 ELSE 0 END" for term in escaped_terms]
    )

    with duckdb.connect(str(index_path), read_only=True) as con:
        rows = con.execute(
            f"""
            SELECT *, ({score_expr})::DOUBLE AS score
            FROM rag_chunks
            WHERE {' AND '.join(where)}
            ORDER BY
                CASE WHEN main_document = 'YES' THEN 1 ELSE 0 END DESC,
                score DESC
            LIMIT ?
            """,
            params,
        ).df()
    return [_row_to_evidence(row) for _, row in rows.iterrows()]


def _row_to_evidence(row: Any) -> Evidence:
    def value(name: str, default: Any = "") -> Any:
        val = row.get(name, default)
        return default if val is None else val

    return Evidence(
        chunk_id=str(value("chunk_id")),
        shard_id=str(value("shard_id")),
        process_type=str(value("process_type")),
        project_id=str(value("project_id")),
        project_title=str(value("project_title")),
        document_id=str(value("document_id")),
        file_name=str(value("file_name")),
        document_type=str(value("document_type_clean") or value("document_type")),
        main_document=str(value("main_document")),
        page_number=str(value("page_number")),
        page_number_int=int(value("page_number_int", 0) or 0),
        chunk_text=str(value("chunk_text")),
        source_label=str(value("source_label")),
        search_score=float(value("score", 0.0) or 0.0),
    )
