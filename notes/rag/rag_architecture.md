# RAG System for NEPA Document Exploration

## Context

The client wants a way to query the NEPATEC 2.0 corpus with natural language and read specific
documents. Currently documents are locked in JSONL/parquet format accessible only through
Python scripts. The goal is a low-barrier chat interface + document viewer accessible as a new
tab on the existing Quarto website (kaseyzapatka.com/nepa/), usable by non-technical clients,
at near-zero cost.

**Scope decided**: Clean energy only (~20K projects)
**Budget**: Free / near-zero ($0–5/month)
**Document fidelity**: Extracted page text (no original PDFs needed)

---

## Specific RAG Model Questions:

### 1. How would the system be set up for clients?

**Architecture**: A Streamlit app deployed to **Hugging Face Spaces** (free tier) embedded
or linked from the Quarto site as a new navbar tab.

- Quarto side: add one entry to `_quarto.yml` navbar pointing to the HF Spaces URL, or
  create a `rag.qmd` page with an iframe embedding the app
- Client experience: opens a URL in the browser, types a question in plain English,
  reads the answer with cited documents, optionally clicks into a document to read its text

HF Spaces free tier limitations to know about:
- **Cold start**: up to 60 seconds on first load after inactivity
- **Storage**: 16 GB (enough for our pre-built index)
- **Compute**: 2 vCPU / 16 GB RAM (fine for inference-time retrieval)

Alternative for always-on reliability: Render.com free tier or a $7/month Fly.io instance.

---

### 2. What would it take to build?

#### Components

| Component | Tool | Cost | Notes |
|-----------|------|------|-------|
| Vector index (offline, built once) | ChromaDB or LanceDB | $0 | Built locally, uploaded to HF Space |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | $0 | Runs locally; no API key needed |
| LLM for answers | Claude Haiku API | ~$0.001/query | Only variable cost |
| Frontend / hosting | Streamlit on HF Spaces | $0 free tier | |
| Text store (document viewer) | DuckDB on pre-filtered parquet | $0 | Extracted locally, ~300–600 MB |

#### One-time build process (run locally, ~2–4 hours total)

1. **Filter text to clean energy projects**: query `data/processed/{ce,ea,eis}/pages.parquet`
   using DuckDB + `projects_combined.parquet` project IDs → write a `clean_energy_pages.db`
   SQLite or small parquet (estimated ~400–600 MB)
2. **Chunk and embed**: split pages into ~600-char chunks (50-char overlap), embed with
   `all-MiniLM-L6-v2`, store in ChromaDB on disk (estimated 1–2 hours, ~800K–1M chunks)
3. **Upload** the ChromaDB directory + filtered text store to HF Space repo

Compute cost: ~$0 (your local machine). API key cost during build: $0 (open-source embeddings).

#### Ongoing (query-time) cost

- Claude Haiku: ~$0.25/1M input tokens + $1.25/1M output tokens
- A typical query: ~2,000 tokens in context → $0.0005 per question
- Even 500 queries/month ≈ **$0.25/month**

An `ANTHROPIC_API_KEY` would be set as a HF Spaces secret (not visible to clients).

#### Hallucination mitigation (critical)

Four complementary techniques:

1. **Closed-context prompting**: system prompt says "Answer ONLY from the provided context
   passages. If the context is insufficient, say 'I don't have enough information.'"
2. **Mandatory citations**: require the model to name the project title, document type, and
   page number for every factual claim
3. **Temperature = 0**: deterministic outputs only
4. **Show retrieved context**: display the 3–5 source passages alongside the answer so
   clients can verify claims themselves (this is the most powerful transparency tool)

With these in place, the model cannot fabricate — it can only fail to find an answer.

---

### 3. How should clients read specific documents?

**Best approach: document browser tab within the same Streamlit app** — no TextBlob needed,
no RShiny needed. The RAG retrieval already locates the right passages, and a second tab lets
clients browse and read full documents.

The app would have two modes:

**Mode A — Chat/Q&A**: ask a question → get answer + source excerpts → click "View full
document" to enter Mode B

**Mode B — Document browser**: search by project title / project ID / agency / state → see
project metadata card (title, agency, location, process type, decision date) → browse
full extracted page text formatted as readable text with page numbers

**Why not TextBlob?** TextBlob is an NLP preprocessing library (tokenization, sentiment, POS
tagging) — it's for analyzing text, not displaying it. Not the right tool here.

**Why not RShiny?** Possible but adds a second language stack (R + Python) for a use case
where Streamlit in Python is cleaner given the existing Python pipeline.

**Original PDFs**: NEPATEC doesn't include PDF download links. Extracted text is the best
available source. If a client needs the original PDF, you could add an optional "agency
website" link using the `file_provider` and `lead_agency` metadata (e.g., EPLANNING.BLM.GOV
for BLM projects), but this is optional scope.

---

## Recommended Implementation Plan

### Phase 1: Data prep (offline, Python scripts)

**New script**: `code/rag/01_build_text_store.py`
- Input: `data/processed/{ce,ea,eis}/pages.parquet` + `data/analysis/projects_combined.parquet`
- Filter: `project_energy_type == "Clean"`, cap at 50 pages/project (mirrors existing pipeline)
- Output: `data/rag/clean_energy_pages.duckdb` (~400–600 MB)
- Also writes `data/rag/projects_metadata.parquet` (title, agency, state, process_type,
  energy_type, decision_date — from projects_combined.parquet)

**New script**: `code/rag/02_build_index.py`
- Input: `data/rag/clean_energy_pages.duckdb`
- Chunk pages, embed with `sentence-transformers/all-MiniLM-L6-v2`
- Output: `data/rag/chroma_index/` (ChromaDB on disk)
- Estimated runtime: 2–4 hours locally

### Phase 2: Streamlit app

**New files**: `app/app.py`, `app/requirements.txt`

Two-tab layout:
- **Tab 1 "Ask a Question"**: query box → retrieval → Claude Haiku response → source cards
- **Tab 2 "Browse Documents"**: filters (agency, state, process type, energy type) + search →
  project card → page-by-page text viewer

Anti-hallucination: closed-context prompt + cited sources shown below every answer.

### Phase 3: Deploy to Hugging Face Spaces

- Create HF Space: Streamlit SDK, set `ANTHROPIC_API_KEY` as a Space secret
- Upload `app/`, `data/rag/chroma_index/`, `data/rag/clean_energy_pages.duckdb`,
  `data/rag/projects_metadata.parquet`
- Add navbar link in `_quarto.yml`:
  ```yaml
  navbar:
    right:
      - text: "Document Explorer"
        href: https://huggingface.co/spaces/<username>/nepa-explorer
  ```

---

## Key Files to Touch

| File | Role |
|------|------|
| `data/processed/ce/pages.parquet` | CE source text (read only) |
| `data/processed/ea/pages.parquet` | EA source text (read only) |
| `data/processed/eis/pages.parquet` | EIS source text (~5.5 GB, read only) |
| `data/analysis/projects_combined.parquet` | Project metadata filter + display |
| `_quarto.yml` | Add navbar entry |
| `code/rag/01_build_text_store.py` | New: filter + export text store |
| `code/rag/02_build_index.py` | New: chunk + embed + write ChromaDB |
| `app/app.py` | New: Streamlit frontend |

## Reusable Patterns to Leverage

- DuckDB predicate pushdown pattern from `extract_reviews.py` / `extract_gencap.py`
  for efficient page loading without reading full parquets into memory
- Project-document lookup pattern (`build_project_document_lookup()` in reviews)
- `main_document == "YES"` filter for prioritizing primary documents over appendices

---

## Verification

End-to-end test sequence:
1. Run `01_build_text_store.py` → confirm `clean_energy_pages.duckdb` size ~400–600 MB
2. Run `02_build_index.py` on a 100-project sample → confirm ChromaDB writes successfully
3. Run `streamlit run app/app.py` locally → test 3–5 representative queries
4. Confirm retrieved passages are from the right projects and answers are grounded in context
5. Test document browser with a known project by title
6. Deploy to HF Spaces → test cold start, confirm API key works
7. Add link to `_quarto.yml` → `quarto render` → confirm navbar entry appears
