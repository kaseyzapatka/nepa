# Deliverable 05: Data Architecture and Generation Methods

This document describes how each major dataset used in Deliverable 05 was constructed, including key design decisions, pipeline steps, and known limitations. Intended as a technical reference for the final project report.

---

## 1. Document Length Over Time and FRA Compliance

**Report section:** "Pages Over Time / FRA Analysis"
**Setup code:** `code/deliverable05/00_setup.R`
**Analysis code:** `code/deliverable05/01_pages.R`
**Page extraction script:** `code/extract/extract_pages.py`
**Report:** `reports/deliverable05.qmd`

### Deliverable question

> Data on the number of pages over time, including pre- and post-Fiscal Responsibility Act of 2023 (FRA), which set page limit requirements.

The Fiscal Responsibility Act of 2023 (Public Law 118-5, signed June 3, 2023) introduced statutory page limits for NEPA environmental reviews: EAs are limited to 75 pages and EISs to 150 pages (up to 300 for extraordinarily complex projects). Under 40 C.F.R. § 1508.1(bb), a "page" is defined as containing 500 words, and maps, diagrams, graphs, tables, and citations are excluded from the count. This deliverable examines whether clean energy document length declined after FRA enactment and how well Post-FRA projects comply with those limits, using a page measure that reflects the statutory definition.

### Source data

| File | Description |
|------|-------------|
| `data/analysis/projects_combined.parquet` | Project-level metadata for all NEPA records |
| `data/analysis/documents_combined.parquet` | Document-level metadata including `document_type_clean`, `total_pages`, and `main_document` flag |
| `data/analysis/projects_timeline_bert_ea_llm.parquet` | LLM-adjudicated initiation and decision dates for EA projects |
| `data/analysis/projects_timeline_bert_eis_llm.parquet` | LLM-adjudicated initiation and decision dates for EIS projects |
| `data/analysis/projects_page_counts.parquet` | Regulatory page counts computed by `extract_pages.py` (see §2.6) |

---

## 2. Data Pipeline

### 2.1 Project filtering

`00_setup.R` loads `projects_combined.parquet` and immediately restricts scope to:

- `project_energy_type == "Clean"` — clean energy projects only
- `process_type %in% c("EA", "EIS")` — EAs and EISs only

**Rationale for CE exclusion:** FRA page limits apply only to EAs and EISs. Categorical exclusions (CEs) are not subject to page limits and are therefore out of scope.

---

### 2.2 Timeline data loading

Timeline records are loaded from two separate files (one per process type) and combined using `bind_rows()`. The key transformation step harmonizes the LLM-adjudicated date columns:

- `timeline_initiation_date` ← `llm_initiation_date` (cast to Date)
- `timeline_decision_date` ← `llm_decision_date` (cast to Date)

After combining, `distinct(project_id, .keep_all = TRUE)` removes any duplicate project IDs that arise from the row-binding step.

**Why LLM dates?** Both EA and EIS timelines use the hybrid BERT+LLM pipeline's final adjudicated dates, consistent with the harmonization approach used in Deliverable 03.

---

### 2.3 Document filtering and deduplication

Documents are filtered from `documents_combined.parquet` to retain only final documents per process type:

| Process type | Document type retained | Rationale |
|---|---|---|
| EIS | `document_type_clean == "FEIS"` | Final EIS (FEIS), not draft (DEIS) |
| EA | `document_type_clean == "EA"` | Final EA; draft EAs tagged "DEA" are excluded |

**Deduplication rule:** One document per project is selected using a two-key priority:

1. `main_document == "YES"` is preferred over non-main documents
2. Among ties, the document with the **highest `total_pages`** is selected

This rule favors the document most likely to represent the primary NEPA body (not an appendix or supplemental volume). Critically, if no `main_document == "YES"` document exists for a project, the rule falls back to the highest-page document of the correct type. This fallback is mirrored exactly in `extract_pages.py`'s `load_clean_energy_main_docs()` function to ensure all projects in the R analysis sample are also covered in the page count extraction.

---

### 2.4 Three-way merge

Projects, timelines, and documents are merged using **inner joins** on `project_id`. Only projects present in all three datasets are retained.

---

### 2.5 Analysis subset construction

After the merge, `pages_analysis` is created by filtering to projects with **complete timelines**: non-missing `timeline_initiation_date` AND non-missing `timeline_decision_date`.

**Derived variables added:**

| Variable | Derivation | Purpose |
|---|---|---|
| `fra_period` | `"Post-FRA"` if `timeline_decision_date >= 2023-06-03`, else `"Pre-FRA"` | FRA compliance classification |
| `decision_year` | `year(timeline_decision_date)` | Year-level grouping |
| `decision_month` | `floor_date(timeline_decision_date, "month")` | Month-level grouping for rolling average |
| `duration_days` | `timeline_decision_date − timeline_initiation_date` | Review duration |
| `duration_months` | `duration_days / 30.44` | Duration in calendar months |

**FRA classification rule:** Projects are classified as Post-FRA when their *decision* date falls on or after June 3, 2023. Using the decision date (not the initiation date) reflects the forward-looking nature of the statute: a project that received its final decision after FRA was enacted is expected to comply with the page limits regardless of when it was initiated.

---

### 2.6 Regulatory page count extraction (`extract_pages.py`)

Raw PDF page counts (`total_pages` from document metadata) overstate document length relative to what the FRA actually limits for two reasons: (1) many main PDF files contain embedded appendices bundled into the same file, and (2) the FRA defines a page as 500 words and excludes maps, figures, tables, and other low-text content. `extract_pages.py` addresses both issues.

**Script:** `code/extract/extract_pages.py`
**Output:** `data/analysis/projects_page_counts.parquet`
**Run command:** `python code/extract/extract_pages.py --run [--sample N] [--verbose]`

The script processes clean energy EA and EIS projects only (same filter as the R pipeline). It runs in two stages per project:

#### Stage 1 — No-appendix file shortcut

Before running OCR extraction, the script checks whether any document for the project has a filename explicitly indicating an appendix-free version (e.g. `*_wo_appendices.pdf`, `*_no_app.pdf`, `*_NoApp.pdf`). The pattern matched is:

```
(without|wo|no)[_\s-]?(appendix|appendices|app|appx)   [case-insensitive]
```

If a match is found among EA or FEIS documents for that project, the matched document's `total_pages` is used directly as `regulatory_pages` — no OCR extraction is needed. This is the most reliable possible estimate because the agency itself produced the appendix-free version. When multiple matches exist for a project, the file with the fewest pages is preferred (the stripped version is smallest). These rows are flagged `regulatory_pages_method = "no_appendix_file"`.

#### Stage 2 — OCR-based word count extraction (DuckDB)

For all remaining projects, the script queries the pages parquet files directly using DuckDB (required because the EIS pages parquet is ~5.5 GB). The query:

1. **Scans the first 80 characters of each page** for appendix section header patterns:
   ```
   (^|\n)\s*(APPENDIX|Appendix|ATTACHMENT|Attachment|EXHIBIT|Exhibit)\s+[A-Z0-9][^A-Za-z0-9]
   ```
   A page is flagged as an appendix header only if it also has fewer than 100 words (excluding TOC pages) and appears at page 5 or later (avoiding false matches at the start of the document).

   A second guard excludes TOC entries: pages where the pattern is immediately followed by dotted leaders (`....`) are not flagged as true appendix headers.

2. **Finds `appendix_start_page`**: the minimum page number where an appendix header is detected. All pages at or after this page are excluded.

3. **Classifies body pages**: pages before `appendix_start_page` with word count ≥ 50. Pages with fewer than 50 words are counted separately as `low_content_pages` (likely maps, figures, section dividers, or blank pages) and excluded from the word count sum.

4. **Computes regulatory pages**:
   ```
   regulatory_pages = CEIL(body_word_count / 500)
   ```
   Projects with zero body word count (fully scanned/image-only PDFs with no extractable OCR text) receive `regulatory_pages = NULL`. These rows are flagged `regulatory_pages_method = "ocr"`.

**Output schema:**

| Column | Description |
|---|---|
| `project_id` | FK to projects |
| `document_id` | The document processed |
| `dataset_source` | `"EA"` or `"EIS"` |
| `raw_pages` | `total_pages` of the document processed |
| `appendix_start_page` | First detected appendix page (NULL if none detected) |
| `total_parquet_pages` | Total pages in the pages parquet for this document |
| `body_pages` | Physical page count of body (before appendix, ≥50 words) |
| `low_content_pages` | Physical page count of low-content body pages (<50 words) |
| `appendix_pages` | Physical page count at or after appendix start |
| `body_word_count` | Total word count of body pages |
| `regulatory_pages` | `CEIL(body_word_count / 500)`; NULL if no OCR text |
| `regulatory_pages_method` | `"ocr"` or `"no_appendix_file"` |

---

### 2.7 Joining regulatory pages into the R pipeline

`00_setup.R` left-joins `projects_page_counts.parquet` onto `pages_data` by `project_id` after the main three-way merge.

**Fallback logic:** When a project has no row in the page counts parquet (e.g., because it has no extractable text and `regulatory_pages = NULL`, or rarely due to a join mismatch), `regulatory_pages` is filled with `total_pages` (the raw PDF page count) as a conservative fallback. These projects are flagged `reg_pages_source = "raw_fallback"`. The setup script prints a source breakdown at runtime:

```
regulatory_pages source: ocr = N | no_appendix_file = N | raw_fallback = N
```

A high `raw_fallback` count indicates that `extract_pages.py` should be re-run against an updated `projects_combined.parquet`.

**Backwards compatibility:** Parquets generated before `regulatory_pages_method` was added to the output schema are handled automatically: if the column is absent, it is added with value `"ocr"` before the join.

**`reg_pages_source` values:**

| Value | Meaning |
|---|---|
| `"ocr"` | Word-count extraction via DuckDB |
| `"no_appendix_file"` | No-appendix filename shortcut; page count is the file's `total_pages` |
| `"raw_fallback"` | Not in page counts parquet; raw PDF page count used |

---

### 2.8 Coverage tracking

`00_setup.R` tracks project counts at each filtering step. Steps:

| Step label | Description |
|---|---|
| Total clean energy with timeline data | All clean energy EA/EIS projects in both `projects_combined` and timeline data |
| With final document | Subset with a matched final document in `documents_combined` |
| With timeline + document | Inner join result (projects present in all three sources) |
| Complete timeline (analysis) | Final analysis sample after dropping projects with missing initiation or decision dates |

---

## 3. Analysis and Figures

All figures and tables are produced by `code/deliverable05/01_pages.R`, which sources `00_setup.R`.

**All figures use `regulatory_pages` as the page measure** — the FRA-compliant estimate computed by `extract_pages.py`. `total_pages` (raw PDF page count) is not shown in any figure but is retained as a fallback for projects without OCR-computed regulatory pages.

### Figure 1: Coverage Funnel (`05_coverage.png`)

Horizontal grouped bar chart showing project counts at each inclusion step, split by EA and EIS. Percentage labels are computed relative to the starting count for each process type.

### Figure 2: Document Length Over Time (`05_pages_over_time.png`)

Individual project points (semi-transparent, colored by FRA period) overlaid with a **3-month trailing rolling average** of monthly mean regulatory page counts, faceted by process type (EA, EIS). The x-axis covers 2010–2025. A red dashed vertical line marks FRA enactment (June 3, 2023). Projects without extractable OCR text are excluded. Rolling average computed using `zoo::rollmean(k = 3, align = "right")`.

### Figure 3: Pre/Post FRA Bar Chart (`05_pages_pre_post_fra.png`)

Grouped bar chart comparing mean regulatory pages Pre- vs. Post-FRA, faceted by process type. Each bar is annotated with the mean and sample size. A diamond marker indicates the median. The distribution is right-skewed so both statistics are shown.

### Figure 3b: Regulatory vs Body Pages Comparison (`05_pages_reg_vs_body.png`)

Grouped bar chart comparing `regulatory_pages` (word count ÷ 500) against `body_pages` (physical page count of body section with ≥50 words) within each FRA period. The gap between the two bars represents pages "lost" to the word-count normalization: sparse pages (covers, section dividers, tables) that count as full physical pages but contribute less than one regulatory page. This figure exists to validate and interpret the regulatory page measure and is not included in the primary deliverable report.

### Figure 4: Distribution Comparison (`05_pages_distribution_boxplot.png`)

Violin + box plot overlay of regulatory page counts Pre- vs. Post-FRA, faceted by process type. Median values are annotated. The y-axis is capped at the 99th percentile of all observations to prevent extreme outliers from collapsing the scale.

### Figure 5: FRA Page Limit Compliance (`05_fra_compliance.png`)

Stacked bar chart restricted to Post-FRA projects showing share compliant with FRA page limits, using `regulatory_pages` as the measure. EA compliance is binary (≤ 75 pages / > 75 pages). EIS compliance uses three tiers: compliant (≤ 150 pages), between standard and extraordinary complexity limits (151–300 pages), and exceeding all limits (> 300 pages). Color coding: teal = compliant, amber = middle tier, magenta = non-compliant.

### Summary tables

| File | Description |
|---|---|
| `output/deliverable5/tables/05_pages_summary.csv` | Descriptive statistics (mean, median, SD, IQR) by process type and FRA period — **all stats computed from `regulatory_pages`**; also includes `n_ocr`, `n_no_appx_file`, `n_raw_fallback` source breakdown |
| `output/deliverable5/tables/05_coverage.csv` | Project counts at each filter step, wide format |
| `output/deliverable5/tables/05_fra_compliance.csv` | FRA compliance breakdown for Post-FRA projects |

---

## 4. Key Design Decisions

### 4.1 Regulatory pages as the primary page measure

All figures and tables use `regulatory_pages` rather than `total_pages` (raw PDF page count) for three reasons:

1. **Statutory alignment:** The FRA defines a page as 500 words. Using word-count-normalized pages is the correct measure for the legal compliance question.
2. **Embedded appendices:** Approximately 52% of clean energy main EA documents and 34% of main FEIS documents contain appendices embedded within the same PDF. Raw page counts include those appendix pages; regulatory pages exclude them.
3. **Low-content pages:** Maps, figures, section dividers, and blank pages have few words and do not count toward the FRA limit. The regulatory page calculation excludes them by counting only words on pages with ≥50 words.

`total_pages` is retained in the dataset as a fallback and for reference, but is not shown in published figures.

### 4.2 Body pages vs regulatory pages

`body_pages` (physical page count before the appendix, ≥50 words) is a simpler alternative to `regulatory_pages`. It excludes appendix pages but still treats sparse pages as full pages. `regulatory_pages` is the correct FRA measure because it applies the 500-words-per-page normalization. The two measures diverge by roughly 15–30% in practice: a document with 60 body pages averaging ~300 words/page yields ~36 regulatory pages.

### 4.3 Decision date for FRA classification

Projects are classified as Post-FRA when their *decision* date falls on or after June 3, 2023. Using the decision date (not initiation date) reflects that a project concluding after FRA enactment is expected to comply, regardless of when it began.

### 4.4 Three-month rolling average

A 3-month trailing rolling average smooths the time series in Figure 2 without losing the shape of multi-year trends. Monthly raw averages can be noisy due to small sample sizes in any given month.

### 4.5 No-appendix file shortcut priority

When a project has an explicit appendix-free document on record (e.g., `*_wo_appendices.pdf`), that file's page count is used directly as `regulatory_pages`, bypassing OCR extraction. This is more reliable than algorithmic appendix detection because the agency itself produced the clean version. Only a handful of projects qualify for this shortcut in the current dataset.

### 4.6 P99 cap on distribution figure

A small number of very long documents would dominate the y-axis scale without truncation. The 99th percentile cap retains the bulk of the distribution while acknowledging outliers in the caption.

---

## 5. Output Schema

### Analysis-ready R objects (produced by `00_setup.R`)

| Object | Contents |
|---|---|
| `pages_data` | Full merged dataset: clean energy EA/EIS projects with timeline, document, and regulatory page count data. Includes `total_pages` (raw), `regulatory_pages` (primary measure), `reg_pages_source` (method flag), and page count components (`body_pages`, `low_content_pages`, `appendix_start_page`). |
| `pages_analysis` | Analysis subset: `pages_data` filtered to complete timelines, with `fra_period`, `decision_year`, `decision_month`, `duration_days`, and `duration_months` added. |
| `coverage` | Long-format tibble with project counts at each filter step, by process type. Used for Figure 1. |
| `fra_date` | Constant: `as.Date("2023-06-03")`. |

### Upstream parquet inputs (not modified)

`projects_combined.parquet`, `documents_combined.parquet`, and the two timeline parquets are read-only inputs.

### Intermediate output written by `extract_pages.py`

`data/analysis/projects_page_counts.parquet` — one row per clean energy main EA/FEIS document. Must be regenerated when `projects_combined.parquet` is updated.

---

## 6. Known Limitations

### 6.1 Small Post-FRA sample

Only approximately 2.5 years of Post-FRA data are available (June 2023 to present). This produces a small Post-FRA sample, limiting the statistical power of Pre/Post comparisons.

### 6.2 Descriptive analysis only

Changes in page counts after FRA could reflect shifts in project complexity, agency composition, or technology mix — not solely FRA-driven compliance. The analysis does not control for confounders. Figure 2 suggests document length was already declining before FRA enactment, which further complicates attribution.

### 6.3 Regulatory page estimates are approximate

OCR quality varies across documents. Fully scanned (image-only) PDFs return no extractable text and fall back to `total_pages`. In ~5% of cases, `regulatory_pages` exceeds `total_pages` because table-dense OCR text inflates word counts. The appendix detection heuristic may occasionally miss non-standard section headers or misidentify a body page as an appendix start.

### 6.4 Compliance without context

A project exceeding FRA page limits may have received a regulatory waiver, may have been initiated before FRA, or may involve extraordinary complexity. The compliance figure treats all Post-FRA projects uniformly.

### 6.5 Timeline dependency

Projects without complete timelines (both initiation and decision dates) are excluded from all figures. These may be systematically different from the included set.

### 6.6 Separate supplemental volumes not summed

The page count reflects the single primary document selected per project. Agencies that split their NEPA documents across many separate volumes may appear to have shorter documents than they actually do.

---

## 7. Validation

Inspection points recommended before finalizing the report:

- At script runtime, check the `regulatory_pages source: ocr = N | no_appendix_file = N | raw_fallback = N` line. A high `raw_fallback` count means `extract_pages.py` should be re-run.
- Review the coverage funnel numbers to assess whether the analysis sample is representative of the full clean energy EA/EIS population.
- For Figure 3b (regulatory vs body pages comparison), check that the gap between the two bars is plausible (~15–30% difference is typical). A very large gap may indicate over-detection of appendix headers.
- Compare the Pre-FRA and Post-FRA sample compositions (energy type, agency, project size) to assess whether confounders could explain observed differences.
