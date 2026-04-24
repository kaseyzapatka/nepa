from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from .text_formatting import clean_text, estimate_tokens, paragraph_blocks


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index_on_page: int
    char_start: int
    char_end: int
    chunk_text: str
    token_estimate: int


def source_label(project_title: str, project_id: str, file_name: str, document_id: str, page_number: str) -> str:
    return (
        f"[Project: {project_title or 'Untitled project'} ({project_id}); "
        f"Document: {file_name or 'Untitled document'} ({document_id}); "
        f"Page: {page_number}]"
    )


def _chunk_id(parts: Iterable[object]) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def chunk_page_text(
    text: object,
    *,
    shard_id: str,
    document_id: str,
    page_number: str,
    target_tokens: int = 1200,
    overlap_tokens: int = 150,
) -> list[TextChunk]:
    raw = clean_text(text)
    if not raw:
        return []

    blocks = paragraph_blocks(raw)
    if not blocks:
        blocks = [raw]

    chunks: list[TextChunk] = []
    current_blocks: list[str] = []
    current_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_index, current_blocks, current_tokens
        if not current_blocks:
            return
        chunk_text = "\n\n".join(current_blocks).strip()
        if not chunk_text:
            current_blocks = []
            current_tokens = 0
            return

        start = raw.find(current_blocks[0])
        if start < 0:
            start = 0
        end = raw.rfind(current_blocks[-1])
        if end < 0:
            end = min(len(raw), start + len(chunk_text))
        else:
            end += len(current_blocks[-1])

        chunks.append(
            TextChunk(
                chunk_id=_chunk_id([shard_id, document_id, page_number, chunk_index, start, end]),
                chunk_index_on_page=chunk_index,
                char_start=start,
                char_end=end,
                chunk_text=chunk_text,
                token_estimate=estimate_tokens(chunk_text),
            )
        )
        chunk_index += 1

        if overlap_tokens <= 0:
            current_blocks = []
            current_tokens = 0
            return

        overlap: list[str] = []
        overlap_count = 0
        for block in reversed(current_blocks):
            tokens = estimate_tokens(block)
            if overlap and overlap_count + tokens > overlap_tokens:
                break
            overlap.insert(0, block)
            overlap_count += tokens
        current_blocks = overlap
        current_tokens = overlap_count

    for block in blocks:
        block_tokens = estimate_tokens(block)

        if block_tokens > target_tokens:
            flush()
            words = block.split()
            step = max(1, target_tokens - overlap_tokens)
            for start_idx in range(0, len(words), step):
                sub_words = words[start_idx : start_idx + target_tokens]
                if not sub_words:
                    continue
                sub_text = " ".join(sub_words)
                chunks.append(
                    TextChunk(
                        chunk_id=_chunk_id([shard_id, document_id, page_number, chunk_index, start_idx]),
                        chunk_index_on_page=chunk_index,
                        char_start=0,
                        char_end=0,
                        chunk_text=sub_text,
                        token_estimate=estimate_tokens(sub_text),
                    )
                )
                chunk_index += 1
            current_blocks = []
            current_tokens = 0
            continue

        if current_blocks and current_tokens + block_tokens > target_tokens:
            flush()

        current_blocks.append(block)
        current_tokens += block_tokens

    flush()
    return chunks
