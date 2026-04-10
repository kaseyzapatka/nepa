# NEPA Document Browser — Implementation Instructions for Codex

## Overview

Build a two-file Streamlit application that lets non-technical clients search and read NEPA
documents from the NEPATEC 2.0 dataset. The app is scoped to the **~20,725 clean energy
projects** already analyzed in this report. It consists of:

1. **`code/rag/01_build_text_store.py`** — a one-time build script run locally that creates
   a self-contained DuckDB database from existing parquet files
2. **`app/app.py`** — a Streamlit app that reads that database and provides a three-view
   browser UI (search → project detail → document viewer)

No LLM, no API keys, no vector index. The only dependencies are `streamlit`, `duckdb`, and
`pandas`.

---

## Part 1: Build Script — `code/rag/01_build_text_store.py`

### Purpose

Read existing parquet files, filter to clean energy projects only, and write a single
`data/rag/nepa_reader.duckdb` file containing three tables (`projects`, `documents`, `pages`)
plus a full-text search index on the `pages` table.

### Input files (all read-only — do not modify)

| File | What it contains |
|------|-----------------|
| `data/analysis/projects_combined.parquet` | One row per project; ~91 columns |
| `data/analysis/documents_combined.parquet` | One row per document; 14 columns |
| `data/processed/ce/pages.parquet` | CE page text; columns: `document_id`, `page_number`, `page_text` |
| `data/processed/ea/pages.parquet` | EA page text; same columns |
| `data/processed/eis/pages.parquet` | EIS page text; same columns (~5.5 GB — must be streamed) |

### Output

`data/rag/nepa_reader.duckdb` — a persistent DuckDB database.

### Exact column names (verified from codebase)

**From `projects_combined.parquet` — use these exact names:**
- `project_id` — unique project identifier
- `project_title` — display name
- `lead_agency` — lead federal agency
- `project_state` — state (NOT `state`, NOT `location`)
- `process_type` — one of `"CE"`, `"EA"`, `"EIS"`
- `project_energy_type` — filter value is exactly `"Clean"` (capital C)
- `project_type` — technology tag (Solar, Wind, etc.)
- `project_description` — free-text description of the proposed action
- `project_department` — parent department (e.g., "Department of the Interior")

**From `documents_combined.parquet` — use these exact names:**
- `document_id` — unique document identifier
- `project_id` — foreign key to projects
- `dataset_source` — one of `"CE"`, `"EA"`, `"EIS"`
- `document_type` — raw type label
- `document_type_category` — cleaned category (`"decision"`, `"draft"`, `"final"`, `"other"`)
- `document_type_clean` — cleaned type (`"CE"`, `"EA"`, `"FEIS"`, `"DEIS"`, etc.)
- `main_document` — `"YES"` or `"NO"`
- `file_name` — original PDF filename
- `total_pages` — integer page count

**From pages parquets — all three sources have identical columns:**
- `document_id`
- `page_number` — string (e.g., `"1"`, `"Page-1"`, `"1-6"`) — needs numeric extraction
- `page_text` — full OCR'd text of that page

### Step-by-step logic

#### Step 1 — Create output directory and DuckDB connection

```python
import duckdb
import pandas as pd
from pathlib import Path

Path("data/rag").mkdir(parents=True, exist_ok=True)
db_path = "data/rag/nepa_reader.duckdb"

# Remove existing DB if present so we rebuild cleanly
if Path(db_path).exists():
    Path(db_path).unlink()

con = duckdb.connect(db_path)
```

#### Step 2 — Build `projects` table

Read `projects_combined.parquet`, filter to clean energy, write only the columns the app
needs (to keep the DB small):

```python
projects_cols = [
    "project_id", "project_title", "lead_agency", "project_state",
    "process_type", "project_energy_type", "project_type",
    "project_description", "project_department"
]

con.execute(f"""
    CREATE TABLE projects AS
    SELECT {', '.join(projects_cols)}
    FROM read_parquet('data/analysis/projects_combined.parquet')
    WHERE project_energy_type = 'Clean'
""")

n = con.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
print(f"projects table: {n:,} rows")
# Expected: ~20,725
```

#### Step 3 — Build `documents` table

Get the clean energy project IDs from the just-created projects table, then filter documents:

```python
docs_cols = [
    "document_id", "project_id", "dataset_source", "document_type",
    "document_type_clean", "document_type_category", "main_document",
    "file_name", "total_pages"
]

con.execute(f"""
    CREATE TABLE documents AS
    SELECT {', '.join(docs_cols)}
    FROM read_parquet('data/analysis/documents_combined.parquet')
    WHERE project_id IN (SELECT project_id FROM projects)
""")

n = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
print(f"documents table: {n:,} rows")
```

#### Step 4 — Build `pages` table (three sources, streamed)

The EIS pages parquet is ~5.5 GB. Use DuckDB predicate pushdown via `INNER JOIN` on
`document_id` — this is the same pattern used in `code/extract/extract_reviews.py`
(`load_project_pages_with_duckdb`). Do NOT load the full parquet into pandas.

Process all three sources with the same logic:

```python
for source in ["ce", "ea", "eis"]:
    pages_path = f"data/processed/{source}/pages.parquet"
    print(f"Loading {source.upper()} pages from {pages_path} ...")

    con.execute(f"""
        INSERT INTO pages
        SELECT
            p.document_id,
            p.page_number,
            -- Extract integer from page_number strings like "1", "Page-1", "1-6"
            COALESCE(
                TRY_CAST(
                    regexp_extract(CAST(p.page_number AS VARCHAR), '(\d+)', 1)
                AS INTEGER),
                0
            ) AS page_number_int,
            p.page_text
        FROM read_parquet('{pages_path}') p
        INNER JOIN documents d USING (document_id)
    """)

    n = con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    print(f"  pages table now: {n:,} rows")
```

Create the table before the loop:

```python
con.execute("""
    CREATE TABLE pages (
        document_id VARCHAR,
        page_number VARCHAR,
        page_number_int INTEGER,
        page_text VARCHAR
    )
""")
```

#### Step 5 — Create indexes for fast queries

```python
# Index for filtering pages by document_id (used in every page-load query)
con.execute("CREATE INDEX idx_pages_doc ON pages(document_id)")

# Index for filtering documents by project_id
con.execute("CREATE INDEX idx_docs_project ON documents(project_id)")

# Full-text search index on page_text
# DuckDB FTS extension
con.execute("INSTALL fts; LOAD fts;")
con.execute("""
    PRAGMA create_fts_index(
        'pages',        -- table name
        'rowid',        -- row identifier (DuckDB adds this automatically)
        'page_text',    -- column to index
        overwrite=1
    )
""")
print("FTS index created.")
```

#### Step 6 — Print summary and close

```python
for table in ["projects", "documents", "pages"]:
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n:,} rows")

con.close()
print(f"\nDatabase written to: {db_path}")
print(f"Size: {Path(db_path).stat().st_size / 1e9:.2f} GB")
```

#### Complete script structure

```python
#!/usr/bin/env python3
"""
Build the NEPA document browser DuckDB database from existing parquets.
Run once locally; upload the resulting data/rag/nepa_reader.duckdb to HF Spaces.

Usage:
    python code/rag/01_build_text_store.py
"""
import duckdb
from pathlib import Path

# ... (all steps above in sequence)
```

---

## Part 2: Streamlit App — `app/app.py`

### Overview

Three views controlled by `st.session_state`. Navigation flows forward (search → project →
document) and backward (breadcrumb buttons). All DuckDB queries use parameterized calls to
prevent SQL injection.

### Dependencies (`app/requirements.txt`)

```
streamlit>=1.35
duckdb>=0.10
pandas>=2.0
```

### Session state variables

Declare these at the top of `app.py` with defaults if not already set:

```python
import streamlit as st
import duckdb
import pandas as pd
import re

DB_PATH = "data/rag/nepa_reader.duckdb"

def init_state():
    defaults = {
        "view": "search",               # "search" | "project" | "document"
        "project_id": None,
        "project_title": None,
        "document_id": None,
        "document_name": None,
        "current_page": 1,
        "search_term": "",
        "doc_search_term": "",          # search-within-document term
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()
```

### DuckDB connection (cached)

```python
@st.cache_resource
def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

con = get_connection()
```

### Filter option helpers (cached)

```python
@st.cache_data
def get_filter_options():
    agencies = con.execute(
        "SELECT DISTINCT lead_agency FROM projects WHERE lead_agency IS NOT NULL ORDER BY 1"
    ).df()["lead_agency"].tolist()

    states = con.execute(
        "SELECT DISTINCT project_state FROM projects WHERE project_state IS NOT NULL ORDER BY 1"
    ).df()["project_state"].tolist()

    process_types = ["CE", "EA", "EIS"]

    energy_types = con.execute(
        "SELECT DISTINCT project_type FROM projects WHERE project_type IS NOT NULL ORDER BY 1"
    ).df()["project_type"].tolist()

    return agencies, states, process_types, energy_types
```

---

### View 1: Search

**Sidebar** — filters that narrow the results table. All filters are optional; omitting them
returns all ~20,725 clean energy projects. Show sidebar filters even when in project/document
view (they persist).

```python
def render_sidebar():
    st.sidebar.header("Filter Projects")

    agencies, states, process_types, energy_types = get_filter_options()

    title_search = st.sidebar.text_input(
        "Project title contains", value=st.session_state["search_term"],
        key="title_input"
    )
    selected_process = st.sidebar.multiselect("Process type", process_types)
    selected_energy = st.sidebar.multiselect("Energy type", energy_types)
    selected_agency = st.sidebar.selectbox("Agency", ["(all)"] + agencies)
    selected_state = st.sidebar.selectbox("State", ["(all)"] + states)

    return title_search, selected_process, selected_energy, selected_agency, selected_state
```

**Search results query** — parameterized, returns a dataframe:

```python
@st.cache_data
def search_projects(title, process_types, energy_types, agency, state):
    conditions = ["1=1"]
    params = []

    if title:
        conditions.append("LOWER(project_title) LIKE LOWER(?)")
        params.append(f"%{title}%")
    if process_types:
        placeholders = ", ".join(["?" for _ in process_types])
        conditions.append(f"process_type IN ({placeholders})")
        params.extend(process_types)
    if energy_types:
        # project_type is a tag that may contain multiple values separated by comma
        # Use OR LIKE for each selected type
        type_conditions = " OR ".join(["LOWER(project_type) LIKE LOWER(?)" for _ in energy_types])
        conditions.append(f"({type_conditions})")
        params.extend([f"%{t}%" for t in energy_types])
    if agency and agency != "(all)":
        conditions.append("lead_agency = ?")
        params.append(agency)
    if state and state != "(all)":
        conditions.append("project_state = ?")
        params.append(state)

    where = " AND ".join(conditions)

    query = f"""
        SELECT
            p.project_id,
            p.project_title,
            p.lead_agency,
            p.project_state,
            p.process_type,
            p.project_type,
            COUNT(d.document_id) AS n_documents
        FROM projects p
        LEFT JOIN documents d USING (project_id)
        WHERE {where}
        GROUP BY 1,2,3,4,5,6
        ORDER BY p.project_title
        LIMIT 500
    """
    return con.execute(query, params).df()
```

**Search view render:**

```python
def render_search():
    st.title("NEPA Document Explorer")
    st.caption("Browse ~20,000 clean energy environmental review documents.")

    title, proc, energy, agency, state = render_sidebar()
    st.session_state["search_term"] = title

    results = search_projects(
        title,
        tuple(proc),        # cache_data requires hashable args — use tuple
        tuple(energy),
        agency,
        state
    )

    st.write(f"**{len(results):,} projects** match your filters.")

    if results.empty:
        st.info("No projects found. Try broadening your filters.")
        return

    # Display as a table with clickable rows
    # Use st.dataframe with on_select (Streamlit >= 1.35)
    display_cols = {
        "project_title": "Project Title",
        "lead_agency": "Agency",
        "project_state": "State",
        "process_type": "Type",
        "project_type": "Energy Type",
        "n_documents": "# Docs",
    }
    display_df = results[list(display_cols.keys())].rename(columns=display_cols)

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection.rows:
        row_idx = event.selection.rows[0]
        selected = results.iloc[row_idx]
        st.session_state["project_id"] = selected["project_id"]
        st.session_state["project_title"] = selected["project_title"]
        st.session_state["view"] = "project"
        st.rerun()
```

---

### View 2: Project Detail

Shows project metadata and a table of its documents.

```python
@st.cache_data
def get_project(project_id):
    return con.execute(
        "SELECT * FROM projects WHERE project_id = ?", [project_id]
    ).df().iloc[0]

@st.cache_data
def get_documents(project_id):
    return con.execute("""
        SELECT
            document_id,
            file_name,
            document_type_clean,
            document_type_category,
            main_document,
            total_pages,
            dataset_source
        FROM documents
        WHERE project_id = ?
        ORDER BY
            CASE WHEN main_document = 'YES' THEN 0 ELSE 1 END,
            total_pages DESC NULLS LAST
    """, [project_id]).df()


def render_project():
    # Breadcrumb
    if st.button("← Back to Search"):
        st.session_state["view"] = "search"
        st.rerun()

    proj = get_project(st.session_state["project_id"])

    st.title(proj["project_title"])

    # Metadata card
    col1, col2, col3 = st.columns(3)
    col1.metric("Agency", proj["lead_agency"] or "—")
    col2.metric("State", proj["project_state"] or "—")
    col3.metric("Process Type", proj["process_type"] or "—")

    col4, col5 = st.columns(2)
    col4.metric("Energy Type", proj["project_type"] or "—")
    col5.metric("Department", proj["project_department"] or "—")

    if proj["project_description"]:
        with st.expander("Project Description", expanded=False):
            st.write(proj["project_description"])

    st.divider()
    st.subheader("Documents")

    docs = get_documents(st.session_state["project_id"])

    if docs.empty:
        st.warning("No documents found for this project.")
        return

    display_cols = {
        "file_name": "File Name",
        "document_type_clean": "Type",
        "document_type_category": "Category",
        "main_document": "Main Doc",
        "total_pages": "Pages",
    }
    display_df = docs[list(display_cols.keys())].rename(columns=display_cols)

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection.rows:
        row_idx = event.selection.rows[0]
        selected = docs.iloc[row_idx]
        st.session_state["document_id"] = selected["document_id"]
        st.session_state["document_name"] = selected["file_name"]
        st.session_state["current_page"] = 1
        st.session_state["doc_search_term"] = ""
        st.session_state["view"] = "document"
        st.rerun()
```

---

### View 3: Document Viewer

Two-column layout: left TOC panel, right page reader.

#### Page list loader

```python
@st.cache_data
def get_page_list(document_id):
    """Load all pages for a document, sorted by page_number_int."""
    return con.execute("""
        SELECT page_number, page_number_int, page_text
        FROM pages
        WHERE document_id = ?
        ORDER BY page_number_int, page_number
    """, [document_id]).df()
```

#### Section header detector (TOC logic)

```python
def detect_header(page_text: str, max_len: int = 60) -> str:
    """
    Return a short label for a page for use in the table of contents.
    Priority order:
      1. Numbered section heading in lines 1-5 (e.g., "1.", "1.1", "Chapter 3", "Section IV")
      2. ALL-CAPS short line in lines 1-3 (5–60 chars)
      3. First non-empty line, truncated to max_len chars
    """
    lines = [l.strip() for l in page_text.strip().splitlines() if l.strip()]
    if not lines:
        return "(empty page)"

    # Rule 1: numbered section heading in first 5 lines
    numbered = re.compile(
        r'^(\d+\.\d*|\d+\)|Chapter\s+\d+|Section\s+\d+|SECTION\s+[IVXLC\d]+)',
        re.IGNORECASE
    )
    for line in lines[:5]:
        if numbered.match(line):
            return line[:max_len]

    # Rule 2: ALL-CAPS short line in first 3 lines
    for line in lines[:3]:
        if line.isupper() and 5 <= len(line) <= 60 and len(line.split()) > 1:
            return line[:max_len]

    # Rule 3: first non-empty line
    return lines[0][:max_len]
```

#### Within-document full-text search

```python
@st.cache_data
def search_within_document(document_id: str, term: str):
    """Return page_number_int values where page_text matches term (case-insensitive)."""
    if not term or len(term.strip()) < 2:
        return set()
    results = con.execute("""
        SELECT page_number_int
        FROM pages
        WHERE document_id = ?
          AND LOWER(page_text) LIKE LOWER(?)
    """, [document_id, f"%{term}%"]).df()
    return set(results["page_number_int"].tolist())
```

#### Document viewer render

```python
def render_document():
    # Breadcrumb
    if st.button(f"← Back to {st.session_state['project_title']}"):
        st.session_state["view"] = "project"
        st.rerun()

    doc_id = st.session_state["document_id"]
    pages_df = get_page_list(doc_id)

    if pages_df.empty:
        st.warning("No page text found for this document.")
        return

    total_pages = len(pages_df)
    st.markdown(f"**{st.session_state['document_name']}** — {total_pages} pages")

    # Build TOC entries: list of (page_number_int, display_label)
    toc_entries = [
        (row["page_number_int"], f"Page {row['page_number_int']}: {detect_header(row['page_text'])}")
        for _, row in pages_df.iterrows()
    ]

    # Two-column layout
    toc_col, reader_col = st.columns([1, 3])

    with toc_col:
        st.subheader("Contents")

        # Search within document
        doc_search = st.text_input(
            "Search in document",
            value=st.session_state["doc_search_term"],
            key="doc_search_input",
            placeholder="e.g. wetlands"
        )
        st.session_state["doc_search_term"] = doc_search

        # Get matching page numbers
        matching_pages = search_within_document(doc_id, doc_search) if doc_search else set()

        # Scroll to first match if search term just entered
        if matching_pages and doc_search:
            first_match = min(matching_pages)
            if st.session_state["current_page"] not in matching_pages:
                st.session_state["current_page"] = first_match

        # Render TOC entries as buttons
        for page_int, label in toc_entries:
            is_current = (page_int == st.session_state["current_page"])
            is_match = (page_int in matching_pages)

            # Style: bold if current, orange highlight if search match
            if is_current:
                prefix = "▶ "
            elif is_match:
                prefix = "🔍 "
            else:
                prefix = "   "

            display = f"{prefix}{label}"
            btn_type = "primary" if is_current else "secondary"

            if st.button(display, key=f"toc_{page_int}", use_container_width=True, type=btn_type):
                st.session_state["current_page"] = page_int
                st.rerun()

    with reader_col:
        current = st.session_state["current_page"]

        # Clamp current_page to valid range
        valid_ints = pages_df["page_number_int"].tolist()
        if current not in valid_ints:
            current = valid_ints[0]
            st.session_state["current_page"] = current

        current_idx = valid_ints.index(current)

        # Navigation bar
        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("← Prev", disabled=(current_idx == 0)):
                st.session_state["current_page"] = valid_ints[current_idx - 1]
                st.rerun()
        with nav2:
            st.markdown(
                f"<div style='text-align:center'>Page {current_idx+1} of {total_pages}</div>",
                unsafe_allow_html=True
            )
        with nav3:
            if st.button("Next →", disabled=(current_idx == total_pages - 1)):
                st.session_state["current_page"] = valid_ints[current_idx + 1]
                st.rerun()

        # Jump to page
        jump = st.number_input(
            "Jump to page", min_value=1, max_value=total_pages,
            value=current_idx + 1, step=1
        )
        if jump - 1 != current_idx:
            st.session_state["current_page"] = valid_ints[jump - 1]
            st.rerun()

        # Page text
        page_row = pages_df[pages_df["page_number_int"] == current].iloc[0]
        page_text = page_row["page_text"]

        # Highlight search term if active
        if doc_search and doc_search.lower() in page_text.lower():
            # Simple highlight: wrap matches in bold (case-insensitive)
            highlighted = re.sub(
                f"({re.escape(doc_search)})",
                r"**\1**",
                page_text,
                flags=re.IGNORECASE
            )
        else:
            highlighted = page_text

        st.markdown(f"---\n**Page {current}**\n\n---")
        # Use code block to preserve whitespace/formatting of OCR text
        st.code(highlighted, language=None)
```

---

### Main app router

```python
def main():
    st.set_page_config(
        page_title="NEPA Document Explorer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    view = st.session_state["view"]

    if view == "search":
        render_search()
    elif view == "project":
        render_project()
    elif view == "document":
        render_document()

if __name__ == "__main__":
    main()
```

---

## Part 3: Deployment to Hugging Face Spaces

### File structure to upload

```
(HF Space repo root)
├── app.py
├── requirements.txt
└── data/
    └── rag/
        └── nepa_reader.duckdb
```

`app.py` references `data/rag/nepa_reader.duckdb` as a relative path — this works because
HF Spaces sets the working directory to the repo root.

### `requirements.txt`

```
streamlit>=1.35
duckdb>=0.10
pandas>=2.0
```

### HF Spaces settings

- SDK: `Streamlit`
- Python version: 3.11
- Hardware: CPU Basic (free)
- No environment secrets needed (no API keys)

### Upload steps

1. Create a new HF Space at huggingface.co/spaces with Streamlit SDK
2. Clone the Space repo locally
3. Copy `app.py`, `requirements.txt`, and `data/rag/nepa_reader.duckdb` into the repo
4. `git add . && git commit -m "initial deploy" && git push`
5. HF Spaces builds and deploys automatically (~2–3 min)

**Note on large file:** `nepa_reader.duckdb` may exceed GitHub's 100 MB file limit. Use
HF's native large file support (no LFS configuration needed — HF Spaces handles large files
automatically when pushed via the HF Hub API or git with HF's backend).

### `_quarto.yml` edit (existing project website)

Add one entry to the navbar in `_quarto.yml`:

```yaml
navbar:
  right:
    - text: "Document Explorer"
      href: https://huggingface.co/spaces/YOUR_HF_USERNAME/nepa-explorer
```

Replace `YOUR_HF_USERNAME` and `nepa-explorer` with the actual Space name after creating it.

---

## Part 4: Local Development & Testing

### Run locally

```bash
cd /path/to/nepa/project

# Build the database (run once)
python code/rag/01_build_text_store.py

# Launch the app
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Test cases to verify before deploying

1. **Search**: Type "solar Nevada" — should return results with process_type and agency populated
2. **Empty search**: Clear all filters — should show ~20,725 results (or up to the 500-row cap)
3. **Project view**: Click any result — metadata card should show title, agency, state, description
4. **Document viewer**: Click any document — TOC should populate, page 1 should show text
5. **TOC headers**: Open a long EIS document — headers like "1. Environmental Setting" should
   appear in the TOC panel, not just "Page N: ..."
6. **Search within doc**: Type "wetlands" — matching pages should get 🔍 prefix in TOC and
   app should jump to first match
7. **CE edge case**: Open a CE project (1–2 pages) — TOC should show 1–2 entries, no crash
8. **Navigation**: Prev/Next/Jump-to-page should all work without errors

---

## Key Implementation Notes for Codex

- **`project_state` not `state`** — the state column in `projects_combined.parquet` is named
  `project_state`. Do not use `state`, `location`, or `project_location`.

- **`project_energy_type == 'Clean'`** — the filter value is exactly `'Clean'` with capital C.
  This is what selects the ~20,725 clean energy projects.

- **Page number is a string** — `page_number` in the pages parquets is a VARCHAR that may
  contain values like `"1"`, `"Page-1"`, `"1-6"`. Always use `page_number_int` (the regex-
  extracted integer) for ordering and navigation. The extraction regex is:
  `regexp_extract(CAST(page_number AS VARCHAR), '(\d+)', 1)`

- **DuckDB and Streamlit caching** — `@st.cache_resource` for the DuckDB connection
  (one connection per session, not per rerun); `@st.cache_data` for query results
  (cached by arguments, invalidated when args change). Never call `con.close()` in the app.

- **`on_select="rerun"` requires Streamlit ≥ 1.35** — the `st.dataframe` click-to-select
  pattern uses this argument. Pin `streamlit>=1.35` in requirements.txt.

- **EIS parquet is ~5.5 GB** — the build script MUST use DuckDB's `read_parquet()` with an
  `INNER JOIN` on `document_id` (predicate pushdown). Do not use `pd.read_parquet()` on the
  EIS pages file — it will exhaust memory. See the pattern in
  `code/extract/extract_reviews.py` → `load_project_pages_with_duckdb()`.

- **FTS index** — DuckDB's FTS extension must be installed and loaded before calling
  `PRAGMA create_fts_index`. In the build script: `con.execute("INSTALL fts; LOAD fts;")`.
  In the app, the simpler `LOWER(page_text) LIKE LOWER(?)` is used for within-document
  search (fast enough for single-document queries). The FTS index is available for
  cross-document full-text search if that feature is added later.
