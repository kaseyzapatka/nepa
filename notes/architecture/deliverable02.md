# Deliverable 02: Data Architecture and Generation Methods

This document describes how each major dataset used in Deliverable 02 was constructed, including key design decisions, pipeline steps, and known limitations. Intended as a technical reference for the final project report.

---

## 1. Review Type Classification

**Report section:** "Programmatic & Tiered Reviews"
**Primary output:** `data/analysis/projects_reviews.parquet`
**Primary code:** `code/extract/extract_reviews.py`
**Analysis code:** `code/deliverable02/01_reviews.R`
**Setup code:** `code/deliverable02/00_setup.R`

### Source data

The raw data comes from PNNL's NEPATEC 2.0 dataset. Deliverable 02 operates exclusively on **EA and EIS projects** — the 1,416 clean energy projects from the EA and EIS process types after applying the same exclusion filters used across all deliverables (military nuclear, nuclear waste; see Deliverable 03 documentation). Categorical exclusions (CEs) are excluded from this analysis because programmatic/tiered review concepts apply specifically to the EA/EIS process.

### Scope of analysis

The deliverable addresses two questions:

1. How many clean energy EA/EIS projects are programmatic or tiered reviews, compared to the total?
2. Are tiered reviews completed faster than standard reviews?

---

## 2. Review Type Extraction Pipeline

### Overview: Three-tier strategy

Review type extraction uses a strict priority cascade to minimize document I/O while maximizing detection accuracy. Each project is classified exactly once, using the first tier that yields a confident result:

| Tier | Method | Trigger | Speed |
|------|--------|---------|-------|
| 1 | Title-based detection | Always run first | Instant |
| 2 | Regex on document text | If title does not match | ~5–10 sec/project |
| 3 | LLM (Ollama) | Only if `--use-llm` set AND medium-confidence candidates exist | ~5 sec/call |

Each project is assigned exactly one of four `project_review_type` values: `programmatic`, `tiered`, `standard`, or `unknown` (the last only on extraction error).

---

### 2.1 Tier 1: Title-Based Detection

**What it detects:** Whether this project IS a programmatic review (not whether it tiers from one).

The project title is checked against a set of programmatic keyword patterns:

- `programmatic` (with `include_generic=True`, the default)
- `program-wide`
- `PEIS` (Programmatic Environmental Impact Statement)
- `PEA` (Programmatic Environmental Assessment)

If a title match is found, the project is immediately classified as `programmatic` with `confidence="high"` and `review_source="title"`. No document pages are loaded. This fast path is appropriate because a PEIS or PEA title is unambiguous.

**Known limitation:** A small number of projects titled with "Programmatic Collaborations" (DOE grants to state energy offices) can trigger a false positive under `include_generic=True`. These projects are DOE grant programs, not NEPA programmatic reviews. They are filtered by the upstream clean energy / EA/EIS scope restriction (they live in the CE dataset), so they do not appear in the EA/EIS extraction.

---

### 2.2 Tier 2: Regex on Document Text

For projects not resolved by title, `extract_reviews.py` loads and scans the first 60 pages of each project's main documents using DuckDB-based bulk page loading (see §2.4). Two separate detection tasks run in parallel on each page:

#### 2.2.1 Programmatic detection (from text)

The function `check_text_for_programmatic()` looks for text patterns indicating the scanned document itself is a programmatic review. High-confidence patterns include explicit PEIS/PEA language in document headers or introductory text. Medium-confidence patterns include "this programmatic environmental" constructions. If a high- or medium-confidence match is found, the project is classified as `programmatic` and scanning stops immediately.

#### 2.2.2 Tiered detection

The function `extract_review_from_text()` applies a list of regex patterns looking for tiering language — phrases indicating the current review builds upon an existing programmatic review. Example patterns include:

| Pattern type | Example match |
|---|---|
| Direct tier statement | "This EA tiers from the 2012 NPR-A IAP/EIS" |
| Tier-to construction | "tiers to the ... PEIS" |
| Pursuant-to construction | "pursuant to the ... Programmatic EIS" |
| Site-specific tier | "site-specific EA that tiers from ..." |
| Tier reference | "Tier 1 EIS" / "Tier 2 EIS" |

Each candidate match is scored as `high` or `medium` confidence. High-confidence matches trigger immediate classification and stop scanning. Medium-confidence matches are accumulated across all pages and handled by LLM (if enabled) or selected by the best available candidate.

For each match, three pieces of information are captured:
- `review_match_text`: the exact matched text that triggered classification
- `review_tiers_from`: the extracted name of the programmatic review being referenced (cleaned)
- `review_tiers_from_context`: a window of surrounding text (~200 characters) providing evidence

#### 2.2.3 Source file tracking

As of the current pipeline version, each match is also linked back to its source document:
- `project_review_match_document_id`: the `document_id` of the file containing the matched text
- `project_review_match_file_name`: the `file_name` (PDF filename) of that document

This allows downstream users to locate and read the specific PDF that triggered a tiered classification.

#### 2.2.4 False positive filtering

The function `is_false_positive()` filters out a set of known non-NEPA uses of "tier" terminology before any candidate is accepted:

| False positive category | Example |
|---|---|
| EPA engine emission standards | "Tier 1", "Tier 2", "Tier 3", "Tier 4" engine standards |
| Road classifications | "Tier 1 highway", "road tier" |
| Pricing / service tiers | "tiered pricing", "tiered rate" |
| Generic ranking language | "first-tier", "second-tier" when not connected to NEPA |

This filter is especially important for solar and wind projects on federal land, which frequently reference EPA Tier 4 engine requirements for construction equipment.

---

### 2.3 Tier 3: LLM Adjudication (Optional)

If `--use-llm` is set and Tier 2 produced medium-confidence (but no high-confidence) tiering candidates, a local Ollama model (`llama3.2:3b-instruct-q4_K_M` by default) is queried to adjudicate. The LLM receives the candidate context texts and classifies whether they represent genuine NEPA tiering language.

LLM adjudication is not used in the primary production run (extraction was run with `--no-llm`). All 1,416 current classifications derive from Tiers 1 and 2 only.

---

### 2.4 DuckDB-Based Page Loading

NEPATEC 2.0 page text is stored in per-source parquet files (`data/processed/ea/pages.parquet`, `data/processed/eis/pages.parquet`). These files are large and cannot be loaded fully into memory for each project iteration.

The pipeline uses **DuckDB** for bulk page retrieval across all projects in a single pass. The core query:

1. Registers a `project_docs` lookup table (project_id → document_id → file_name) as an in-memory DuckDB relation
2. Joins the pages parquet directly from disk using `read_parquet()` — DuckDB applies predicate pushdown to scan only matching rows
3. Assigns each page a `ROW_NUMBER()` partitioned by `project_id`, ordered by `page_num` (numeric page order)
4. Filters to `rn <= max_pages` (default cap: 60 pages per project)
5. Returns a grouped dict: `{project_id: [{"page_text": ..., "document_id": ..., "file_name": ...}, ...]}`

The page entries are returned as dicts (not plain strings) so that the downstream extraction loop can record which document a match came from.

Documents are selected via a `main_document == "YES"` filter applied in `build_project_document_lookup()`: if a project has any documents flagged as main documents, only those are scanned. This ensures the primary EA or EIS body is searched before supporting appendices or supplemental files.

---

### 2.5 Extraction Results (Current Run)

| Review Type | Count | Share |
|---|---|---|
| Standard | 1,390 | 98.2% |
| Programmatic | 16 | 1.1% |
| Tiered | 10 | 0.7% |
| **Total** | **1,416** | **100%** |

- Confidence: **high** for 1,413 projects, **medium** for 3
- Detection source: **text_regex** (1,413), **title** (3)

The low prevalence of non-standard reviews is consistent with domain expectations: most NEPA projects are standalone reviews. The 10 tiered reviews identified represent a conservative lower bound; the regex approach may miss cases where tiering language is buried in later document pages or phrased in non-standard ways.

---

## 3. Output Schema

### Primary output: `data/analysis/projects_reviews.parquet`

| Field | Type | Description |
|---|---|---|
| `project_id` | String | Unique project identifier (from NEPATEC) |
| `project_review_is_programmatic` | Boolean | TRUE if this project IS a programmatic review |
| `project_review_type` | Categorical | `programmatic`, `tiered`, `standard`, `unknown` |
| `project_review_confidence` | Categorical | `high`, `medium`, `low` |
| `project_review_tiers_from` | String | Name of the programmatic review being tiered from (tiered only) |
| `project_review_tiers_from_context` | String | Surrounding text context for tiering language (tiered only) |
| `project_review_source` | String | Detection source: `title`, `doc_metadata`, `text_regex`, `llm`, `none`, `error` |
| `project_review_match_text` | String | Exact text that triggered the classification |
| `project_review_match_document_id` | String | `document_id` of the file containing the matched text (text_regex only) |
| `project_review_match_file_name` | String | PDF filename of the file containing the matched text (text_regex only) |
| `project_review_pages_scanned` | Integer | Number of pages examined before classification |
| `project_review_candidates_found` | Integer | Number of tiering candidates found across all scanned pages |
| `dataset_source` | String | Process type: `EA` or `EIS` |

---

## 4. Analysis and Figures

**Setup script:** `code/deliverable02/00_setup.R` loads `projects_reviews.parquet`, merges with timeline data, and creates analysis-ready objects:

| R object | Contents |
|---|---|
| `reviews` | Full reviews dataset (1,416 clean energy EA/EIS projects) |
| `reviews_tl` | Reviews merged with timeline data (includes `duration_days`) |
| `duration_data` | Subset with valid, positive duration |
| `non_standard` | Only programmatic + tiered projects (26 rows) |
| `reviews_long_agency` | Unnested by lead agency (for agency-level analysis) |
| `reviews_long_state` | Unnested by state (for geographic analysis) |

**Analysis script:** `code/deliverable02/01_reviews.R` produces all figures and tables:

| Output | File | Description |
|---|---|---|
| Figure 1 | `02_review_share.png` | Review type distribution (counts + %) |
| Figure 2 | `02_review_by_process.png` | Review type by NEPA process (100% stacked bar + zoom) |
| Figure 3 | `02_agency.png` | Top agencies for non-standard reviews |
| Figure 3b | `02_department.png` | Department-level breakdown |
| Figure 4 | `02_state.png` | Geographic distribution (top states) |
| Figure 5 | `02_duration.png` | Duration by review type and process type |
| Figure 6 | `02_tiered_parents.png` | Parent programmatic reviews cited by tiered projects |
| Table 1 | `02_snapshot.csv` | Count and % by review_type × process_type |
| Table 2 | `02_duration_summary.csv` | Duration descriptive statistics |

---

## 5. Timeline Integration

To answer "are tiered reviews completed faster?", timeline data is merged via a left join in `00_setup.R`:

- EA timelines: `data/analysis/projects_timeline_bert_ea_llm.parquet`
- EIS timelines: `data/analysis/projects_timeline_bert_eis_llm.parquet`

Key variables used:
- `llm_initiation_date` — start date of the NEPA process
- `llm_decision_date` — final approval/signature date
- `duration_days` = `decision_date − initiation_date`

**Caveat:** With only 10 tiered reviews in the dataset, duration comparisons are exploratory and lack statistical power. The current findings (tiered EAs take longer than standard EAs; programmatic EISs are modestly faster than standard EISs) should be framed as descriptive observations.

---

## 6. Known Limitations

### 6.1 Rarity of examples

Only 26 of 1,416 EA/EIS clean energy projects (1.8%) are non-standard. This is lower than some domain estimates (5–15% tiered). Possible reasons include:
- NEPATEC 2.0 includes project-level records but may not systematically include all supplemental and tiered documents
- Tiering language may appear after page 60 in some documents (the scan cap)
- Some tiered documents use non-standard phrasing not captured by current patterns

### 6.2 Missing file name search

The current extraction searches project titles and document text. It does not scan document file names for keywords. One project (`3621210fbd086bddbbf6fbedc1d6a488`) was found to have "programmatic" in its file name but was classified as standard because the document text did not trigger a match. Adding file name search is a potential improvement.

### 6.3 Evidence context for programmatic classification

The `project_review_tiers_from_context` field (surrounding text) is populated only for tiered reviews. Programmatic reviews identified via `text_regex` have `review_match_text` but no broader context field. The `tiers_from` reference string and associated context were architecturally scoped to tiered detection and are not available for programmatic detection.

### 6.4 "Programmatic Collaborations" false positives

DOE grant programs titled "Programmatic Collaboration" or similar are not NEPA programmatic reviews. These appear only in the CE dataset and are excluded from the EA/EIS scope; they are not a source of false positives in the current output. If CE projects are added to the scope in the future, additional filtering would be needed.

---

## 7. Validation

A Google Sheet was prepared for client validation of the 26 non-standard projects:

- **Programmatic tab** (16 projects): with `evidence_text` and `correct` / `notes` columns
- **Tiered tab** (10 projects): with `evidence_text`, `tiers_from`, and `correct` / `notes` columns

Validation status as of 2026-02-04: **awaiting client review**.

Two known potential issues flagged for review:
1. Two sets of apparent duplicate project IDs (same title, different IDs) — may represent separate review documents within the same project record
2. Project `3621210fbd086bddbbf6fbedc1d6a488` is potentially misclassified as standard (see §6.2)
