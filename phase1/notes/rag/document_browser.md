# NEPA Document Browser — Architecture & Plan

## Context

The existing `code/ce_page_viewer.ipynb` notebook lets a developer explore NEPA documents by
typing a `project_id` into a code cell and calling helper functions (`list_documents`,
`view_pages`, `search_pages`). Clients cannot use this — it requires running Python and
knowing a project ID in advance.

This plan builds a Streamlit web app that gives non-technical clients the same capabilities
through a browser: search for projects by name/agency/state, browse documents with a table
of contents, and read full page text. No RAG, no LLM, no API key, zero ongoing cost.

---

## What the Notebook Does (and What We're Replacing)

| Notebook function | Web equivalent |
|---|---|
| `list_documents(project_id)` | Document table on project detail page |
| `view_page(doc_id, page_num)` | Main reading panel in document viewer |
| `view_pages(doc_id, start, end)` | Prev/Next navigation in document viewer |
| `view_project_pages(project_id)` | Click through all docs in a project |
| `search_pages(project_id, term)` | Search-within-document box |
| Manual: set `project_id` in a cell | Project search + filter sidebar |

The key addition is **discoverability** — the notebook requires knowing the project ID.
The web app lets clients find projects by title, agency, state, or energy type.

---

## Data Architecture

### One-time build: `nepa_reader.duckdb`

A single DuckDB file containing three tables, built locally from existing parquets.

**`projects` table** (from `data/analysis/projects_combined.parquet`, filtered to clean energy):
```
project_id, project_title, lead_agency, state, process_type,
project_energy_type, project_description, project_type
```

**`documents` table** (from `data/analysis/documents_combined.parquet`, joined to clean energy project IDs):
```
document_id, project_id, dataset_source, document_type, document_type_category,
main_document, file_name, total_pages
```

**`pages` table** (from `data/processed/{ce,ea,eis}/pages.parquet`, filtered by clean energy project IDs):
```
document_id, page_number, page_number_int, page_text
```
→ FTS index on `page_text` for fast keyword search.

**Estimated size:** ~600 MB–1.5 GB compressed (CE pages are tiny; EIS pages are the bulk).
Fits on HF Spaces free tier (16 GB storage cap). DuckDB queries pages on demand — it does
not load the full file into RAM.

---

## App Structure — Three Views

```
SEARCH VIEW (default)
  ├── Sidebar: filters + title search
  │     ├── Process type (CE / EA / EIS) — multiselect
  │     ├── Energy type (Solar, Wind, Geothermal, …) — multiselect
  │     ├── Agency — dropdown
  │     ├── State — dropdown
  │     └── Project title search (freetext, ILIKE)
  └── Main: paginated results table
        ├── Project title | Agency | State | Process type | # Documents
        └── Click row → PROJECT VIEW

PROJECT VIEW
  ├── Breadcrumb: "← Back to Search"
  ├── Metadata card:
  │     Title | Lead agency | State | Process type | Energy type
  │     Project description (collapsible)
  └── Documents table (mirrors notebook's `list_documents()`):
        File name | Doc type | Main doc? | Pages
        Click row → DOCUMENT VIEWER

DOCUMENT VIEWER
  ├── Breadcrumb: "← Back to [Project Title]"
  ├── Two-column layout:
  │     Left (25%): Table of Contents
  │     │   ├── Page entries, scrollable
  │     │   ├── Detected section headers highlighted in bold
  │     │   ├── Search-within-document box
  │     │   │     (highlights matching pages in TOC, jumps to first match)
  │     │   └── Click page → jump to that page
  │     └── Right (75%): Page reader
  │           ├── Document header (file name, type, main_document flag)
  │           ├── "Page N of M" indicator
  │           ├── Page text (preformatted, preserves whitespace)
  │           └── ← Prev | Jump to page [  ] | Next →
  └── (no LLM, no API)
```

---

## Table of Contents Logic

NEPA documents don't have machine-readable section headers — they're OCR'd page images.
The TOC is generated heuristically by scanning the first few lines of each page:

**Header detection rules (in priority order):**
1. Numbered section: `"1."`, `"1.1"`, `"Chapter 3"`, `"Section 4"`, `"SECTION IV"` — first match in lines 1–5
2. ALL-CAPS short line: 5–60 chars, appears in lines 1–3, not a standalone word like "THE"
3. Fallback: first non-empty line, truncated to 60 chars

This produces TOC entries like:
```
Page 1:  FINDING OF NO SIGNIFICANT IMPACT
Page 2:  1. Project Description
Page 3:  1.1 Background
Page 8:  2. Environmental Setting
Page 14: 3. Environmental Consequences
Page 22: APPENDIX A - Species List
```

Pages with no detected header show `Page N: [first 60 chars of text]…` in lighter style.

Keyword search highlights matching page entries in orange and jumps to the first hit.

---

## Build Script: `code/rag/01_build_text_store.py`

```bash
python code/rag/01_build_text_store.py
```

Steps:
1. Load `projects_combined.parquet`, filter `project_energy_type == "Clean"`,
   write `projects` table → ~20K rows
2. Load `documents_combined.parquet`, inner join to clean energy project IDs,
   write `documents` table → ~60K rows (estimated)
3. For each source (CE, EA, EIS):
   - Load pages parquet using DuckDB (predicate pushdown on clean energy project IDs)
   - Write `pages` table in batches (EIS pages parquet is 5.5 GB — stream, don't load)
4. Create FTS index: `PRAGMA create_fts_index('pages', 'rowid', 'page_text')`
5. Output: `data/rag/nepa_reader.duckdb`

**Estimated runtime**: 30–60 minutes (EIS page loading is the slow step).
**Estimated output size**: 600 MB–1.5 GB.

Reuses the DuckDB predicate pushdown pattern already in `extract_reviews.py` and `extract_gencap.py`.

---

## Streamlit App: `app/app.py`

~350 lines. Key patterns:

- `st.session_state["view"]` — tracks which of the three views is active
  (`"search"`, `"project"`, `"document"`)
- `st.session_state["selected_project_id"]` — persists across reruns
- `st.session_state["selected_document_id"]` — persists across reruns
- `st.session_state["current_page"]` — current page number in the document viewer
- `@st.cache_resource` — caches the DuckDB connection (one connection for entire session)
- `@st.cache_data` — caches search results, project metadata, page lists

Queries at runtime:
- Search: `SELECT … FROM projects WHERE process_type IN (?) AND lead_agency LIKE ? …`
- Project docs: `SELECT … FROM documents WHERE project_id = ?`
- Page list (TOC): `SELECT page_number_int, page_text FROM pages WHERE document_id = ? ORDER BY page_number_int`
- Single page: same query with `LIMIT 1 OFFSET N`
- Full-text search within doc: DuckDB FTS `match_bm25` filtered by `document_id`

---

## Deployment: Hugging Face Spaces

```
nepa-explorer/
├── app.py
├── requirements.txt       # streamlit, duckdb, pandas (~3 packages)
└── data/
    └── nepa_reader.duckdb
```

**requirements.txt:**
```
streamlit>=1.35
duckdb>=0.10
pandas>=2.0
```

No API keys, no secrets. The Space is fully public.

**Cold start**: ~10–20 seconds (much faster than a RAG app with model loading).
After first load, navigation between pages is instant (DuckDB queries are <100ms).

**Website integration** (`_quarto.yml`):
```yaml
navbar:
  right:
    - text: "Document Explorer"
      href: https://huggingface.co/spaces/<username>/nepa-explorer
```
Or embed as iframe in a `docs-explorer.qmd` Quarto page.

---

## File Summary

| File | Status | Notes |
|------|--------|-------|
| `code/rag/01_build_text_store.py` | New | Build script; run once locally |
| `app/app.py` | New | Streamlit frontend (~350 lines) |
| `app/requirements.txt` | New | 3 dependencies |
| `data/rag/nepa_reader.duckdb` | Generated | Upload to HF Space |
| `_quarto.yml` | Edit | Add navbar entry |

**Source parquets (read-only inputs):**
- `data/analysis/projects_combined.parquet`
- `data/analysis/documents_combined.parquet`
- `data/processed/ce/pages.parquet`
- `data/processed/ea/pages.parquet`
- `data/processed/eis/pages.parquet` (~5.5 GB — streamed, not loaded whole)

---

## Build Time & Effort Estimate

| Task | Time |
|------|------|
| Write `01_build_text_store.py` | 2–3 hrs |
| Run build script (one-time) | 30–60 min |
| Write `app/app.py` | 4–6 hrs |
| HF Spaces deploy + test | 1–2 hrs |
| Add to `_quarto.yml` | 15 min |
| **Total dev effort** | **~1 day** |
| **Ongoing cost** | **$0** |

---

## Verification

1. Run `01_build_text_store.py` → check DuckDB size, spot-check a known project ID
2. `streamlit run app/app.py` locally → search "solar Nevada" → open a project → open a document
3. Confirm TOC generates sensible headers for an EIS document (long doc, good test)
4. Confirm keyword search within a document highlights correct pages
5. Test CE project (very short, 1–2 pages) — edge case for TOC with single page
6. Deploy to HF Spaces → test from a fresh browser (no local data)
7. Test cold start time; verify it loads within 30 seconds
