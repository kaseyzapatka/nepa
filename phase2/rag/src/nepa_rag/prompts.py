from __future__ import annotations

from .evidence import Evidence, evidence_context
from .query_router import QueryScope


def build_answer_prompt(
    question: str,
    scope: QueryScope,
    evidence: list[Evidence],
    *,
    max_context_tokens: int,
) -> str:
    context = evidence_context(evidence, max_tokens=max_context_tokens)
    warning = scope.warning or "None"
    return f"""You are answering questions about NEPATEC2.0 source text and Phase 2 NEPA analysis.

Rules:
- Answer using only the evidence provided below.
- Keep the answer concise, usually under 250 words.
- If the evidence is incomplete, give the best supported answer and include a clear warning.
- Cite factual claims using exact source labels copied from the evidence.
- Separate direct source evidence from inference.
- Do not invent dates, counts, agencies, project titles, document numbers, or citations.
- If the question asks for exact counts or tables and those are not provided, say that a structured query is required.

Question:
{question}

Scope:
{scope.label}

Scope warning:
{warning}

Evidence:
{context or "No retrieved evidence."}

Return this structure:
Answer
Confidence / warning
Evidence used
Notes and limitations
"""
