from __future__ import annotations

import re
from dataclasses import dataclass


SOURCE_LABEL_RE = re.compile(r"\[Project: .*? \([^)]+\); Document: .*? \([^)]+\); Page: .*?\]")


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    shard_id: str
    process_type: str
    project_id: str
    project_title: str
    document_id: str
    file_name: str
    document_type: str
    main_document: str
    page_number: str
    page_number_int: int
    chunk_text: str
    source_label: str
    search_score: float | None = None


def extract_citations(text: str) -> set[str]:
    return set(SOURCE_LABEL_RE.findall(text or ""))


def validate_citations(answer_text: str, evidence: list[Evidence]) -> list[str]:
    valid = {item.source_label for item in evidence}
    warnings: list[str] = []
    for label in extract_citations(answer_text):
        if label not in valid:
            warnings.append(f"Unsupported citation label: {label}")
    return warnings


def evidence_context(evidence: list[Evidence], *, max_tokens: int) -> str:
    parts: list[str] = []
    used_tokens = 0
    for idx, item in enumerate(evidence, start=1):
        approx_tokens = max(1, len(item.chunk_text.split()))
        if used_tokens + approx_tokens > max_tokens and parts:
            break
        used_tokens += approx_tokens
        parts.append(
            "\n".join(
                [
                    f"Source {idx}: {item.source_label}",
                    f"Process: {item.process_type}",
                    f"Document type: {item.document_type or '-'}",
                    f"Main document: {item.main_document or '-'}",
                    "Text:",
                    item.chunk_text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def source_card_title(item: Evidence) -> str:
    return f"{item.project_title} | {item.file_name} | Page {item.page_number}"
