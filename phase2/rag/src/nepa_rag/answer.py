from __future__ import annotations

from dataclasses import dataclass

from .config import RagConfig
from .evidence import Evidence, validate_citations
from .llm import LLMError, generate_text
from .prompts import build_answer_prompt
from .query_router import QueryScope, route_question
from .retrieval import retrieve_evidence
from .structured import try_structured_answer


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    evidence: list[Evidence]
    scope: QueryScope
    model: str | None
    warnings: list[str]


@dataclass(frozen=True)
class PreparedAnswer:
    prompt: str | None
    evidence: list[Evidence]
    scope: QueryScope
    warnings: list[str]
    early_answer: str | None = None


def answer_question(
    config: RagConfig,
    question: str,
    *,
    selected_project_id: str | None = None,
    process_types: list[str] | None = None,
    force_whole_corpus: bool = False,
) -> AnswerResult:
    prepared = prepare_answer(
        config,
        question,
        selected_project_id=selected_project_id,
        process_types=process_types,
        force_whole_corpus=force_whole_corpus,
    )
    if prepared.early_answer is not None or prepared.prompt is None:
        return AnswerResult(
            answer=prepared.early_answer or "",
            evidence=prepared.evidence,
            scope=prepared.scope,
            model=None,
            warnings=prepared.warnings,
        )

    warnings = list(prepared.warnings)
    try:
        llm_response = generate_text(config, prepared.prompt)
    except LLMError as exc:
        warnings.append(str(exc))
        return AnswerResult(
            answer=(
                "Retrieved source evidence, but the local model did not return an answer. "
                "Review the evidence cards below or start Ollama and try again."
            ),
            evidence=prepared.evidence,
            scope=prepared.scope,
            model=None,
            warnings=warnings,
        )

    citation_warnings = validate_citations(llm_response.text, prepared.evidence)
    warnings.extend(citation_warnings)
    if config.require_source_labels and citation_warnings:
        warnings.append("The answer contains citations that were not in the retrieved evidence set.")

    return AnswerResult(
        answer=llm_response.text,
        evidence=prepared.evidence,
        scope=prepared.scope,
        model=llm_response.model,
        warnings=warnings,
    )


def prepare_answer(
    config: RagConfig,
    question: str,
    *,
    selected_project_id: str | None = None,
    process_types: list[str] | None = None,
    force_whole_corpus: bool = False,
) -> PreparedAnswer:
    scope = route_question(
        config,
        question,
        selected_project_id=selected_project_id,
        process_types=process_types,
        force_whole_corpus=force_whole_corpus,
    )

    if scope.mode == "ambiguous":
        return PreparedAnswer(
            prompt=None,
            evidence=[],
            scope=scope,
            warnings=[scope.warning or "Ambiguous project scope."],
            early_answer=scope.warning or "The project scope is ambiguous.",
        )

    structured_answer = try_structured_answer(config, question, scope)
    if structured_answer is not None:
        return PreparedAnswer(
            prompt=None,
            evidence=[],
            scope=scope,
            warnings=[],
            early_answer=structured_answer,
        )

    evidence = retrieve_evidence(config, question, scope)
    warnings: list[str] = []
    if not evidence:
        warnings.append("No source evidence was retrieved. The model answer is suppressed.")
        return PreparedAnswer(
            prompt=None,
            evidence=[],
            scope=scope,
            warnings=warnings,
            early_answer=(
                "I do not have enough retrieved NEPATEC evidence to answer this reliably. "
                "Try selecting a project, narrowing the process type, or using more specific terms."
            ),
        )

    prompt = build_answer_prompt(
        question,
        scope,
        evidence,
        max_context_tokens=config.max_context_tokens,
    )
    return PreparedAnswer(
        prompt=prompt,
        evidence=evidence,
        scope=scope,
        warnings=warnings,
    )
