from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from .config import RagConfig
from .db import connect_catalog

PROJECT_ID_RE = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryScope:
    mode: str
    label: str
    project_id: str | None = None
    process_types: tuple[str, ...] = ()
    whole_corpus: bool = False
    warning: str | None = None


def route_question(
    config: RagConfig,
    question: str,
    *,
    selected_project_id: str | None = None,
    process_types: list[str] | None = None,
    force_whole_corpus: bool = False,
) -> QueryScope:
    if force_whole_corpus:
        return QueryScope(
            mode="whole_corpus",
            label="Searching across the selected corpus.",
            process_types=tuple(process_types or ()),
            whole_corpus=True,
        )

    if selected_project_id:
        return QueryScope(
            mode="selected_project",
            label="Answering within the selected project.",
            project_id=selected_project_id,
            process_types=tuple(process_types or ()),
        )

    exact_id = PROJECT_ID_RE.search(question or "")
    if exact_id:
        project_id = exact_id.group(0).lower()
        with connect_catalog(config) as con:
            exists = con.execute(
                "SELECT COUNT(*) FROM projects WHERE project_id = ?",
                [project_id],
            ).fetchone()[0]
        if exists:
            return QueryScope(
                mode="project_id",
                label="Answering within the project ID mentioned in the question.",
                project_id=project_id,
            )

    title_match = _resolve_project_title(config, question)
    if title_match:
        project_id, title, count = title_match
        if count == 1:
            return QueryScope(
                mode="project_title",
                label=f"Answering within the project matched by title: {title}",
                project_id=project_id,
            )
        return QueryScope(
            mode="ambiguous",
            label="Project title match is ambiguous.",
            process_types=tuple(process_types or ()),
            warning=(
                "I found multiple possible project title matches. Select a project in the "
                "sidebar, then ask again for a project-specific answer."
            ),
        )

    return QueryScope(
        mode="whole_corpus",
        label="No project is selected, so retrieval will search across the selected corpus.",
        process_types=tuple(process_types or ()),
        whole_corpus=True,
    )


def _resolve_project_title(config: RagConfig, question: str) -> tuple[str, str, int] | None:
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", question or "") if len(w) >= 4]
    if len(words) < 2:
        return None

    # Conservative fallback: require a multi-word phrase from the question to appear in title.
    phrases = [" ".join(words[i : i + 3]) for i in range(max(0, len(words) - 2))]
    phrases.extend(" ".join(words[i : i + 2]) for i in range(max(0, len(words) - 1)))

    with connect_catalog(config) as con:
        for phrase in phrases:
            try:
                rows = con.execute(
                    """
                    SELECT project_id, project_title
                    FROM projects
                    WHERE project_title ILIKE ?
                    ORDER BY length(project_title), project_title
                    LIMIT 5
                    """,
                    [f"%{phrase}%"],
                ).fetchall()
            except duckdb.Error:
                continue
            if rows:
                return rows[0][0], rows[0][1], len(rows)
    return None
