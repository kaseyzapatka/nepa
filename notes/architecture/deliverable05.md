# Deliverable 05: Data Architecture and Generation Methods

This document describes how each major dataset used in Deliverable 05 was constructed, including key design decisions, pipeline steps, and known limitations. Intended as a technical reference for the final project report.

---

## 1. Document Length Over Time and FRA Compliance

**Report section:** "Pages Over Time / FRA Analysis"
**Primary output:** `pages_analysis` (R session object; no intermediate parquet written)
**Setup code:** `code/deliverable05/00_setup.R`
**Analysis code:** `code/deliverable05/01_pages.R`
**Report:** `reports/deliverable05.qmd`

### Deliverable question

> Data on the number of pages over time, including pre- and post-Fiscal Responsibility Act of 2023 (FRA), which set page limit requirements.

The Fiscal Responsibility Act of 2023 (Public Law 118-5, signed June 3, 2023) introduced statutory page limits for NEPA environmental reviews: EAs are limited to 75 pages and EISs to 150 pages (up to 300 for extraordinarily complex projects). This deliverable examines whether clean energy document length declined after FRA enactment and how well Post-FRA projects comply with those limits.

### Source data

Three parquet files are merged for this analysis:

| File | Description |
|------|-------------|
| `data/analysis/projects_combined.parquet` | Project-level metadata for all NEPA records (see Deliverable 03 documentation for construction) |
| `data/analysis/documents_combined.parquet` | Document-level metadata including `document_type_clean`, `total_pages`, and `main_document` flag |
| `data/analysis/projects_timeline_bert_ea_llm.parquet` | LLM-adjudicated initiation and decision dates for EA projects |
| `data/analysis/projects_timeline_bert_eis_llm.parquet` | LLM-adjudicated initiation and decision dates for EIS projects |

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

**Why LLM dates?** Both EA and EIS timelines use the hybrid BERT+LLM pipeline's final adjudicated dates (`llm_initiation_date`, `llm_decision_date`), consistent with the harmonization approach used in Deliverable 03. These represent the pipeline's best estimate after BERT candidate generation and LLM selection.

---

### 2.3 Document filtering and deduplication

Documents are filtered from `documents_combined.parquet` to retain only final documents per process type:

| Process type | Document type retained | Rationale |
|---|---|---|
| EIS | `document_type_clean == "FEIS"` | Final EIS (FEIS), not draft (DEIS) |
| EA | `document_type_clean == "EA"` | Final EA; draft EAs tagged as "DEA" are excluded |

This ensures page counts reflect the final submitted document rather than earlier draft versions.

**Deduplication rule:** Some projects have multiple records that qualify as final documents (e.g., multi-volume FEIS, supplemental EIS files). One document per project is selected using a two-key priority:

1. `main_document == "YES"` is preferred over non-main documents
2. Among ties, the document with the **highest `total_pages`** is selected

This rule favors the document most likely to represent the primary NEPA body (not an appendix or supplemental volume) and uses page count as a secondary tiebreaker when the main_document flag is missing or ambiguous.

---

### 2.4 Three-way merge

Projects, timelines, and documents are merged using **inner joins** on `project_id`. Only projects present in all three datasets are retained. This means any clean energy EA/EIS project that lacks either a matched timeline record or a matched final document is excluded from the analysis sample.

Coverage is tracked at each step (see §2.5).

---

### 2.5 Analysis subset construction

After the merge, a final analysis subset (`pages_analysis`) is created by filtering to projects with **complete timelines**: non-missing `timeline_initiation_date` AND non-missing `timeline_decision_date`.

Projects with partial timelines (only one date available) are dropped at this stage.

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

### 2.6 Coverage tracking

`00_setup.R` tracks project counts at each filtering step using a `coverage_steps` list, building a `coverage` tibble used in Figure 1. The steps are:

| Step label | Description |
|---|---|
| Total clean energy with timeline data | All clean energy EA/EIS projects present in both `projects_combined` and timeline data |
| With final document | Subset that also has a matched final document in `documents_combined` |
| With timeline + document | Inner join result (projects present in all three sources) |
| Complete timeline (analysis) | Final analysis sample after dropping projects with missing initiation or decision dates |

Coverage percentages in Figure 1 are computed relative to the first step within each process type.

---

## 3. Analysis and Figures

All figures and tables are produced by `code/deliverable05/01_pages.R`, which sources `00_setup.R` to load all required objects.

### Figure 1: Coverage Funnel (`05_coverage.png`)

Horizontal grouped bar chart showing project counts at each inclusion step, split by EA and EIS. Percentage labels are computed relative to the starting count for each process type. This figure documents the analysis coverage so readers can assess how representative the final sample is.

### Figure 2: Average Pages Over Time (`05_pages_over_time.png`)

A 6-month rolling average of mean monthly page counts, faceted by process type (EA, EIS). The x-axis covers 2010–2025. Monthly raw averages are not shown directly because individual months often have small sample sizes that introduce noise. The rolling average uses `zoo::rollmean()` with `align = "right"` (trailing window). A red dashed vertical line marks FRA enactment (June 3, 2023).

### Figure 3: Pre/Post FRA Bar Chart (`05_pages_pre_post_fra.png`)

Grouped bar chart comparing mean total pages Pre- vs. Post-FRA, faceted by process type. Each bar is annotated with the mean page count and sample size. A diamond marker on each bar indicates the median, providing a visual indicator of skewness. The bar height (mean) and median are reported separately because the page count distribution is right-skewed, and the mean is more sensitive to extreme outliers than the median.

### Figure 4: Distribution Comparison (`05_pages_distribution_boxplot.png`)

Violin + box plot overlay showing the full distribution of page counts Pre- vs. Post-FRA, faceted by process type. The violin shape shows the overall density; the embedded box plot shows quartiles and outliers. Median values are annotated with labeled markers. The y-axis is capped at the 99th percentile of all observations (`p99_pages`) to prevent a small number of very long documents from collapsing the scale and making the bulk of the distribution unreadable.

### Figure 5: Scatter with LOESS Trend (`05_pages_scatter.png`)

Project-level scatter plot of total pages vs. decision date (2010–2025), colored by FRA period. A LOESS smooth with a 95% confidence interval is overlaid to show non-parametric trends without assuming a linear relationship. Points are semi-transparent (`alpha = 0.35`) to reduce overplotting. The red dashed FRA line is included for reference. This figure complements Figure 2 by showing the full dispersion of individual data points.

### Figure 6: FRA Page Limit Compliance (`05_fra_compliance.png`)

Stacked bar chart restricted to Post-FRA projects, showing the share of projects that comply with FRA page limits. EA compliance is binary (≤ 75 pages / > 75 pages). EIS compliance uses three tiers: compliant (≤ 150 pages), between the standard and extraordinary complexity limits (151–300 pages), and exceeding all limits (> 300 pages). Color coding uses CATF brand colors: teal for compliant, amber for the middle tier, and magenta for non-compliant.

### Summary tables

| File | Description |
|---|---|
| `output/deliverable5/tables/05_pages_summary.csv` | Descriptive statistics (mean, median, SD, min, max, IQR) by process type and FRA period |
| `output/deliverable5/tables/05_coverage.csv` | Project counts at each filter step, wide format |
| `output/deliverable5/tables/05_fra_compliance.csv` | FRA compliance breakdown for Post-FRA projects |

---

## 4. Key Design Decisions

### 4.1 Decision date for FRA classification

Projects are classified as Post-FRA when their decision date (not their initiation date) falls on or after June 3, 2023. A project that began before the FRA but received its final decision after enactment is expected to comply. Using the initiation date would misclassify some projects: a project that started in 2022 and concluded in 2024 should be subject to the limits.

### 4.2 Rolling average (not raw monthly)

Individual months often contain very few projects, producing noisy monthly averages. A 6-month trailing rolling average smooths the time series without obscuring the general trend. The rolling window was chosen empirically as wide enough to reduce noise but narrow enough to preserve the shape of trends over multi-year periods.

### 4.3 Page count = main document only

The `total_pages` field in `documents_combined.parquet` reflects the page count of a single PDF file. For multi-volume FEIS documents, only the primary volume (selected via the deduplication rule in §2.3) is counted. This is the appropriate measure for FRA compliance analysis: the FRA page limits apply to the main body of the document, not to appendices or supplemental volumes. Summing all volumes would overstate both the pre-FRA baseline and the post-FRA compliance gap.

### 4.4 Time window 2010–2025

Figures 2 and 5 are restricted to 2010–2025. Earlier years have few clean energy projects in the NEPATEC database and are not relevant to the FRA comparison. The 2025 cutoff reflects the current data availability.

### 4.5 P99 cap on distribution figure

A small number of very long documents (e.g., multi-year large-scale EISs running into thousands of pages) would dominate the y-axis scale if plotted without truncation. The 99th percentile cap retains the bulk of the distribution in view while acknowledging the existence of extreme outliers in the figure caption.

---

## 5. Output Schema

### Analysis-ready R objects (produced by `00_setup.R`)

| Object | Contents |
|---|---|
| `pages_data` | Full merged dataset: all clean energy EA/EIS projects with both a timeline record and a final document. Includes `total_pages`, `timeline_initiation_date`, `timeline_decision_date`. Missing timeline dates are retained. |
| `pages_analysis` | Analysis subset: `pages_data` filtered to complete timelines (non-missing initiation + decision), with `fra_period`, `decision_year`, `decision_month`, `duration_days`, and `duration_months` added. |
| `coverage` | Long-format tibble with project counts at each filter step, by process type. Used for Figure 1. |
| `fra_date` | Constant: `as.Date("2023-06-03")`. Used for FRA classification and vertical line placement in figures. |

### Upstream parquet inputs (not modified)

This deliverable reads from three upstream parquet files but does not write any new parquet outputs. All outputs are figures (PNG) and tables (CSV).

---

## 6. Known Limitations

### 6.1 Small Post-FRA sample

Only approximately 2.5 years of post-FRA projects are available (June 2023 to present as of analysis date). This produces a small Post-FRA sample (dozens of projects vs. hundreds Pre-FRA), limiting the statistical power of Pre/Post comparisons. Differences observed may not be statistically significant and should be interpreted cautiously.

### 6.2 Descriptive analysis only

Changes in page counts after FRA could reflect shifts in project complexity, agency composition, technology mix, or the types of projects reviewed — not solely FRA-driven compliance behavior. The analysis does not control for confounders and cannot establish a causal link between FRA enactment and changes in document length. Figures 2 and 5 suggest that document length was already declining before FRA enactment, which further complicates attribution.

### 6.3 Compliance without context

A project exceeding FRA page limits may have received a regulatory waiver, may have been initiated substantially before FRA enactment, or may involve extraordinary complexity that justifies a higher limit. The compliance figure (Figure 6) treats all Post-FRA projects uniformly and does not distinguish these cases. The framing should note that "non-compliance" in this analysis means exceeding the numerical threshold, not necessarily a violation of the statute.

### 6.4 Timeline dependency

The analysis requires complete timelines (both initiation and decision dates) because the FRA classification and all time-series figures depend on the decision date. Projects without extractable timeline dates are excluded from the analysis sample. The coverage funnel (Figure 1) documents the share excluded at this step, but the excluded set may be systematically different from the included set (e.g., older projects or projects where dates appear in non-standard formats may be underrepresented).

### 6.5 No appendices or supplemental volumes

The page count measure reflects the single primary document selected per project (see §4.3). For agencies that routinely split their NEPA documents across many volumes or appendices, this measure understates the total documentation burden and overstates apparent compliance with FRA limits.

---

## 7. Validation

No formal validation step is implemented for this deliverable. Inspection points recommended before finalizing the report:

- Review the coverage funnel numbers to assess whether the analysis sample is representative of the full clean energy EA/EIS population
- Check for zero-page or implausibly low page count records in the final sample, which may indicate metadata extraction errors in the upstream pipeline
- Compare the Pre-FRA and Post-FRA sample compositions (by energy type, agency, and project size) to assess whether confounders could explain observed page count differences
