#!/usr/bin/env python3
"""Local Streamlit RAG app for NEPATEC artifacts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

RAG_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = RAG_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nepa_rag.answer import prepare_answer
from nepa_rag.config import RagConfig, load_config
from nepa_rag.db import get_documents, get_project, pages_for_document, project_index, search_projects
from nepa_rag.evidence import Evidence, source_card_title, validate_citations
from nepa_rag.llm import LLMError, stream_text
from nepa_rag.text_formatting import clean_text


@st.cache_resource
def get_config() -> RagConfig:
    return load_config()


@st.cache_data(show_spinner=False)
def cached_project_index() -> pd.DataFrame:
    return project_index(get_config())


@st.cache_data(show_spinner=False)
def cached_search_projects(title_query: str, process_types: tuple[str, ...]) -> pd.DataFrame:
    return search_projects(
        get_config(),
        title_query=title_query,
        process_types=list(process_types),
        limit=300,
    )


@st.cache_data(show_spinner=False)
def cached_documents(project_id: str) -> pd.DataFrame:
    return get_documents(get_config(), project_id)


@st.cache_data(show_spinner=False)
def cached_pages(document_id: str) -> pd.DataFrame:
    return pages_for_document(get_config(), document_id)


def init_state() -> None:
    defaults = {
        "selected_project_id": None,
        "selected_project_title": None,
        "selected_document_id": None,
        "selected_document_name": None,
        "current_page": 1,
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_artifacts(config: RagConfig) -> None:
    missing = [path for path in (config.catalog_path, config.manifest_path) if not path.exists()]
    if not missing:
        return

    st.error("RAG artifacts are not built yet.")
    st.write("Run the local build pipeline first:")
    st.code(
        "\n".join(
            [
                "conda run -n nepa python rag/scripts/01_build_text_store.py --sample-documents-per-process 25",
                "conda run -n nepa python rag/scripts/02_build_chunks.py",
                "conda run -n nepa python rag/scripts/03_build_indexes.py",
            ]
        ),
        language="bash",
    )
    st.caption("Remove the sample flag when you are ready to build the full sharded store.")
    st.stop()


def render_sidebar(config: RagConfig) -> tuple[tuple[str, ...], bool, str]:
    st.sidebar.header("Scope")
    index = cached_project_index()
    process_options = sorted([x for x in index["process_type"].dropna().unique().tolist() if x])
    process_types = tuple(
        st.sidebar.multiselect("Process type", options=process_options, default=process_options)
    )
    whole_corpus = st.sidebar.checkbox("Search whole corpus", value=False)

    model_options = {
        f"Fast ({config.llm_model})": config.llm_model,
        f"Quality ({config.llm_quality_model})": config.llm_quality_model,
    }
    selected_model_label = st.sidebar.selectbox("Answer model", options=list(model_options.keys()))
    selected_model = model_options[selected_model_label]

    st.sidebar.divider()
    st.sidebar.subheader("Project selector")
    title_query = st.sidebar.text_input("Project title contains")
    results = cached_search_projects(title_query, process_types)
    if results.empty:
        st.sidebar.info("No projects match the current filters.")
    else:
        labels = {
            row.project_id: f"{row.project_title} [{row.process_type}]"
            for row in results.itertuples(index=False)
        }
        selected = st.sidebar.selectbox(
            "Selected project",
            options=[""] + list(labels.keys()),
            format_func=lambda pid: "No selected project" if not pid else labels.get(pid, pid),
        )
        if selected:
            st.session_state["selected_project_id"] = selected
            st.session_state["selected_project_title"] = labels[selected]
        elif st.sidebar.button("Clear selected project"):
            st.session_state["selected_project_id"] = None
            st.session_state["selected_project_title"] = None
            st.session_state["selected_document_id"] = None
            st.session_state["selected_document_name"] = None
            st.session_state["current_page"] = 1
            st.rerun()

    selected_project_id = st.session_state.get("selected_project_id")
    if selected_project_id:
        st.sidebar.success(f"Selected project: {selected_project_id}")

    st.sidebar.divider()
    st.sidebar.caption(f"Catalog: {config.catalog_path}")
    st.sidebar.caption(f"Manifest: {config.manifest_path}")
    st.sidebar.caption(f"Fast model: {config.llm_model}")
    st.sidebar.caption(f"Quality model: {config.llm_quality_model}")
    st.sidebar.caption(f"Fallback: {config.llm_fallback_model}")
    return process_types, whole_corpus, selected_model


def render_ask_tab(process_types: tuple[str, ...], whole_corpus: bool, selected_model: str) -> None:
    st.subheader("Ask NEPATEC")
    selected_project_id = st.session_state.get("selected_project_id")
    selected_label = st.session_state.get("selected_project_title")
    if selected_project_id and not whole_corpus:
        st.caption(f"Scope: selected project, {selected_label}")
    elif whole_corpus:
        st.caption("Scope: whole corpus search across selected process types.")
    else:
        st.caption("Scope: no project selected; the router may search the corpus or ask for clarification.")

    question = st.text_area(
        "Question",
        placeholder="Example: What evidence supports the NOI date for the selected project?",
        height=120,
    )
    col1, col2 = st.columns([1, 1])
    ask = col1.button("Ask", type="primary", use_container_width=True)
    if col2.button("Clear session", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
            return
        config = get_config()
        started_at = time.monotonic()
        status = st.status("Retrieving evidence from the local RAG store...", expanded=True)
        status.write("Searching source shards and preparing the answer prompt.")
        with st.spinner("Retrieving evidence..."):
            prepared = prepare_answer(
                config,
                question,
                selected_project_id=None if whole_corpus else selected_project_id,
                process_types=list(process_types),
                force_whole_corpus=whole_corpus,
            )
        status.write(f"Retrieved {len(prepared.evidence)} source passage(s).")
        if prepared.early_answer is not None or prepared.prompt is None:
            if prepared.early_answer and not prepared.evidence and not prepared.warnings:
                status.update(label="Answered from DuckDB without model generation.", state="complete")
            else:
                status.update(label="Stopped before model generation.", state="complete")
            st.session_state["history"].append(
                {
                    "question": question,
                    "answer": prepared.early_answer or "",
                    "model": None,
                    "warnings": prepared.warnings,
                    "scope": prepared.scope.label,
                    "evidence": prepared.evidence,
                }
            )
            st.rerun()

        answer_placeholder = st.empty()
        answer_text = ""
        warnings = list(prepared.warnings)
        try:
            status.update(
                label=f"Running local model: {selected_model}",
                state="running",
                expanded=True,
            )
            status.write("Waiting for the first token from Ollama. This can take a while if the model is loading.")
            first_token_at: float | None = None
            for token in stream_text(config, prepared.prompt, model=selected_model):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                    status.write(f"First token received after {first_token_at - started_at:.1f} seconds.")
                answer_text += token
                answer_placeholder.markdown(answer_text + "▌")
            answer_placeholder.markdown(answer_text)
            status.update(
                label=f"Answer complete in {time.monotonic() - started_at:.1f} seconds.",
                state="complete",
                expanded=False,
            )
        except LLMError as exc:
            warnings.append(str(exc))
            status.update(label="Model generation failed.", state="error", expanded=True)
            status.write(str(exc))
            if not answer_text:
                answer_text = (
                    "Retrieved source evidence, but the local model did not return an answer. "
                    "Review the evidence cards below or start Ollama and try again."
                )
                answer_placeholder.warning(answer_text)

        citation_warnings = validate_citations(answer_text, prepared.evidence)
        warnings.extend(citation_warnings)
        if config.require_source_labels and citation_warnings:
            warnings.append("The answer contains citations that were not in the retrieved evidence set.")

        st.session_state["history"].append(
            {
                "question": question,
                "answer": answer_text,
                "model": selected_model,
                "warnings": warnings,
                "scope": prepared.scope.label,
                "evidence": prepared.evidence,
            }
        )
        st.rerun()

    for turn in reversed(st.session_state["history"]):
        st.markdown("### Answer")
        st.caption(turn["scope"])
        if turn.get("model"):
            st.caption(f"Model: {turn['model']}")
        if turn.get("warnings"):
            for warning in turn["warnings"]:
                st.warning(warning)
        st.write(turn["answer"])
        render_evidence_cards(turn.get("evidence", []))
        st.divider()


def render_evidence_cards(evidence: list[Evidence]) -> None:
    if not evidence:
        return
    with st.expander(f"Evidence used ({len(evidence)} source passages)", expanded=True):
        for idx, item in enumerate(evidence, start=1):
            st.markdown(f"**{idx}. {source_card_title(item)}**")
            st.caption(item.source_label)
            st.text(clean_text(item.chunk_text)[:1600])
            jump_key = f"jump_{idx}_{item.chunk_id}"
            if st.button("Open source page", key=jump_key):
                st.session_state["selected_project_id"] = item.project_id
                st.session_state["selected_project_title"] = item.project_title
                st.session_state["selected_document_id"] = item.document_id
                st.session_state["selected_document_name"] = item.file_name
                st.session_state["current_page"] = max(1, item.page_number_int)
                st.rerun()


def render_browse_tab() -> None:
    st.subheader("Browse selected project")
    project_id = st.session_state.get("selected_project_id")
    if not project_id:
        st.info("Select a project in the sidebar to browse its documents.")
        return

    project = get_project(get_config(), project_id)
    if project.empty:
        st.warning("Selected project was not found in the catalog.")
        return

    row = project.iloc[0]
    st.markdown(f"### {row.get('project_title', 'Untitled project')}")
    st.caption(f"Project ID: {project_id}")
    cols = st.columns(4)
    cols[0].metric("Process", row.get("process_type") or "-")
    cols[1].metric("Agency", row.get("lead_agency_harmonized") or "-")
    cols[2].metric("State", row.get("project_state") or "-")
    cols[3].metric("Energy", row.get("project_energy_type") or "-")

    docs = cached_documents(project_id)
    if docs.empty:
        st.warning("No documents found for this project.")
        return

    doc_labels = {
        doc.document_id: f"{doc.file_name} | {doc.document_type_clean or doc.document_type} | main={doc.main_document}"
        for doc in docs.itertuples(index=False)
    }
    selected_doc = st.selectbox(
        "Document",
        options=list(doc_labels.keys()),
        format_func=lambda doc_id: doc_labels.get(doc_id, doc_id),
    )
    if selected_doc:
        st.session_state["selected_document_id"] = selected_doc
        st.session_state["selected_document_name"] = doc_labels[selected_doc]

    render_document_viewer(selected_doc)


def render_document_viewer(document_id: str | None) -> None:
    if not document_id:
        return
    pages = cached_pages(document_id)
    if pages.empty:
        st.info("No page text available for this document in the local shards.")
        return

    total = len(pages)
    current = int(st.session_state.get("current_page", 1))
    current = min(max(current, 1), total)
    st.session_state["current_page"] = current

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    if nav1.button("Previous", disabled=current <= 1):
        st.session_state["current_page"] = current - 1
        st.rerun()
    nav2.markdown(f"<div style='text-align:center'>Page {current} of {total}</div>", unsafe_allow_html=True)
    if nav3.button("Next", disabled=current >= total):
        st.session_state["current_page"] = current + 1
        st.rerun()

    jump = st.number_input("Jump to ordinal page", min_value=1, max_value=total, value=current)
    if int(jump) != current:
        st.session_state["current_page"] = int(jump)
        st.rerun()

    page = pages.iloc[current - 1]
    st.markdown(f"**Source page label:** {page.get('page_number')}")
    st.text(clean_text(page.get("page_text", ""))[:12000])


def main() -> None:
    st.set_page_config(page_title="Phase 2 NEPA RAG", layout="wide")
    init_state()
    config = get_config()
    check_artifacts(config)
    process_types, whole_corpus, selected_model = render_sidebar(config)

    st.title("Phase 2 NEPA RAG")
    ask_tab, browse_tab = st.tabs(["Ask", "Browse Sources"])
    with ask_tab:
        render_ask_tab(process_types, whole_corpus, selected_model)
    with browse_tab:
        render_browse_tab()


if __name__ == "__main__":
    main()
