from __future__ import annotations

import re
from typing import Any

from .config import RagConfig
from .db import connect_catalog
from .query_router import QueryScope

PROJECT_COUNT_RE = re.compile(
    r"\b(how many|number of|count of|total)\b.*\bprojects?\b|\bprojects?\b.*\b(count|total)\b",
    re.IGNORECASE,
)
PROJECT_TITLE_LIST_RE = re.compile(
    r"\b(sample|example|few|some|list|show|give)\b.*\bproject\s+titles?\b|"
    r"\bproject\s+titles?\b.*\b(sample|example|few|some|list|show|give)\b|"
    r"\blist\b.*\bprojects?\b|"
    r"\bshow\b.*\bprojects?\b",
    re.IGNORECASE,
)


def try_structured_answer(config: RagConfig, question: str, scope: QueryScope) -> str | None:
    if PROJECT_COUNT_RE.search(question or ""):
        return _project_count_answer(config, scope)
    if PROJECT_TITLE_LIST_RE.search(question or ""):
        return _project_title_list_answer(config, question, scope)
    return None


def _project_count_answer(config: RagConfig, scope: QueryScope) -> str:
    clauses: list[str] = []
    params: list[Any] = []
    scope_note = "loaded in the current local RAG catalog"

    if scope.project_id:
        clauses.append("project_id = ?")
        params.append(scope.project_id)
        scope_note = "in the selected project scope"
    elif scope.process_types:
        placeholders = ",".join("?" for _ in scope.process_types)
        clauses.append(f"process_type IN ({placeholders})")
        params.extend(scope.process_types)
        scope_note = (
            "loaded in the current local RAG catalog for process type(s): "
            + ", ".join(scope.process_types)
        )

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect_catalog(config) as con:
        count = con.execute(
            f"SELECT COUNT(DISTINCT project_id) FROM projects {where}",
            params,
        ).fetchone()[0]

    return (
        f"There are {count:,} project(s) {scope_note}. "
        "This is a direct DuckDB count, not a model-generated estimate. "
        "If you are using the smoke build, it only reflects the sampled local build."
    )


def _project_title_list_answer(config: RagConfig, question: str, scope: QueryScope) -> str:
    clauses: list[str] = []
    params: list[Any] = []
    scope_note = "loaded in the current local RAG catalog"
    limit = _requested_limit(question, default=5)

    if scope.project_id:
        clauses.append("project_id = ?")
        params.append(scope.project_id)
        scope_note = "in the selected project scope"
    elif scope.process_types:
        placeholders = ",".join("?" for _ in scope.process_types)
        clauses.append(f"process_type IN ({placeholders})")
        params.extend(scope.process_types)
        scope_note = (
            "loaded in the current local RAG catalog for process type(s): "
            + ", ".join(scope.process_types)
        )

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with connect_catalog(config) as con:
        rows = con.execute(
            f"""
            SELECT project_title, process_type, project_id
            FROM projects
            {where}
            ORDER BY project_title, project_id
            LIMIT ?
            """,
            params,
        ).fetchall()

    if not rows:
        return f"I did not find any project titles {scope_note}."

    lines = [f"Here are {len(rows)} project title(s) {scope_note}:"]
    for title, process_type, project_id in rows:
        lines.append(f"- {title} [{process_type}; {project_id}]")
    lines.append(
        "This is a direct DuckDB lookup, not a model-generated answer. "
        "If you are using the smoke build, it only reflects the sampled local build."
    )
    return "\n".join(lines)


def _requested_limit(question: str, *, default: int) -> int:
    match = re.search(r"\b(\d{1,2})\b", question or "")
    if not match:
        return default
    return min(max(int(match.group(1)), 1), 20)
