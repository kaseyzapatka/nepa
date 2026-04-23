# Phase 2 RAG

This directory will contain the CAFT-facing RAG application for NEPATEC2.0 source text, project metadata, and selected Phase 2 deliverable artifacts.

The implementation plan lives in `phase2/plans/rag.md`. This README is the project overview and use-case index.

## Local build

Copy `phase2/rag/.env.example` to `phase2/rag/.env` and adjust the model tags if needed. The first local build can use a tiny sample:

```bash
make -f phase2/rag/Makefile rag-smoke-build
conda run -n nepa streamlit run phase2/rag/app/app.py
```

The smoke build uses two CE documents and skips DuckDB FTS, so it is quick and does not require downloading the DuckDB `fts` extension. It is meant to test the catalog, shard build, chunking, retrieval fallback, app startup, and Ollama connection behavior.

For the full local RAG store, remove `--sample-documents-per-process` and include all process types:

```bash
make -f phase2/rag/Makefile rag-build
```

If DuckDB cannot install/load the `fts` extension, rebuild indexes with the LIKE fallback:

```bash
make -f phase2/rag/Makefile rag-index-skip-fts
```

The app will still run, but retrieval ranking will be weaker than with FTS.

The app calls Ollama locally. Pull the configured model before asking questions:

```bash
ollama pull gemma4:e2b
ollama pull gemma4:e4b
```

The fast default model is `gemma4:e2b`. The app sidebar also offers `gemma4:e4b` as a quality option, and the fallback is `gemma4:e2b`. Adjust these tags in `phase2/rag/.env` if Ollama uses different Gemma 4 tag names on your machine.

For local CPU-bound testing, the defaults intentionally keep prompts small:

```text
RAG_TOP_K_PER_SHARD=3
RAG_MAX_CONTEXT_PASSAGES=4
RAG_MAX_CONTEXT_TOKENS=2500
LLM_NUM_CTX=4096
LLM_NUM_PREDICT=350
```

If Ollama reports the model is running on `100% CPU`, use the default `gemma4:e2b` mode for smoke tests and reserve `gemma4:e4b` for questions where answer quality matters more than latency.

## Use cases

### Project evidence assistant

Use a RAG model to produce source-cited project summaries from NEPATEC2.0 records. Given a `project_id` or project title, the system should retrieve the relevant project metadata, documents, page text, extracted dates, and Federal Register evidence, then generate a concise project profile with citations back to `document_id` and `page_number`.

### Federal Register NOI/NOA evidence review

Use RAG to explain why a Federal Register date was selected for a project — either the Notice of Intent initiation date (`noi_publication_date`) or the Notice of Availability end-of-process date (`noa_availability_date`). The model should retrieve NEPATEC page evidence, FR Doc numbers, Federal Register API records, and candidate match scores, then summarize the evidence trail and flag weak or ambiguous matches for review.

### Manual review packets

Use RAG to accelerate review of ambiguous project-to-notice matches. For each project requiring review, the system can assemble the top candidate Federal Register records, relevant NEPATEC snippets, match rationale, disagreement risks, and a recommended accept/reject/review disposition.

### Timeline reconstruction

Use RAG to reconstruct project timelines from multiple source documents. The model can retrieve evidence for milestones such as NOI, Draft EIS, Final EIS, ROD, EA, FONSI, or other decision documents, then produce a timeline table where every date includes a source citation and confidence note.

### Deliverable QA

Use RAG as a quality-control layer before deliverables are finalized. It can compare project-level tables against source documents and flag unsupported dates, inconsistent agencies, mismatched process types, missing source evidence, or records that conflict with exclusion rules.

### Evidence-backed narrative drafting

Use RAG to draft prose for reports using already-computed statistics and retrieved source examples. Deterministic analysis code should produce the counts, medians, and tables; the RAG model should retrieve representative examples, cite evidence, and turn those results into client-facing narrative text.

### Comparable project finder

Use RAG to identify similar NEPA projects for benchmarking and precedent research. Analysts could search for comparable projects by agency, process type, energy category, geography, document type, date range, or project characteristics, then receive cited summaries of why each project is relevant.

### Appendix and citation generation

Use RAG to generate evidence appendices for deliverables. Possible appendices include project evidence tables, date-source tables, Federal Register match rationales, manual-review notes, and excluded-agency justifications. Each row should preserve traceability through `project_id`, `document_id`, `page_number`, source field, and review status.

### Natural language data access

Use RAG with a structured query layer so analysts can ask natural language questions about the NEPATEC2.0 database. Exact analytical questions should route to DuckDB or parquet tables, while source-text questions should use retrieval over document/page chunks.

### Refresh and change reports

Use RAG to summarize changes after NEPATEC2.0 or Federal Register refreshes. The system can report which projects gained or lost dates, which matches changed confidence level, which records moved from fuzzy matching to direct FR Doc evidence, and which projects newly require manual review.

## Recommended implementation pattern

Use a hybrid design rather than a generic chatbot. Store exact facts in structured tables and constrain the model to produce cited summaries, QA notes, review recommendations, and draft prose. Dates, counts, classifications, and joins should continue to come from deterministic pipeline code.
