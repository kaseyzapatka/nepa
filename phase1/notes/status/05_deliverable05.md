# Deliverable 5 Status: Pages Over Time / FRA Analysis

**Date**: 2026-02-18
**Status**: Scripts created, ready for initial run
**Due**: Feb 20, 2026

---

## Overview

Deliverable 5 analyzes whether NEPA document length (number of pages) decreased after the Fiscal Responsibility Act of 2023 (FRA), which set page limit requirements for environmental assessments and environmental impact statements. The analysis is limited to clean energy EA and EIS projects with final documents and complete timelines.

**Key question**: Is document length (total pages) lower after 6/3/2023 (FRA enactment)?

---

## Scope & Inclusion Criteria

- **Process types**: EA and EIS only (CE excluded; FRA page limits don't apply to CEs)
- **Document types**: Final Environmental Assessments (EA) and Final Environmental Impact Statements (FEIS) only (no drafts, no decision docs)
- **Timeline requirement**: Only projects with both non-missing initiation AND decision dates
- **FRA classification**: Projects with decision date >= June 3, 2023 are classified as "Post-FRA" (they should comply even if initiated before FRA)
- **Deduplication**: If multiple final documents per project, prefer `main_document == "YES"`, then highest `total_pages`

---

## Data Pipeline

### Inputs

| File | Description |
|------|-------------|
| `data/analysis/projects_combined.parquet` | All projects (filtered to clean energy EA/EIS) |
| `data/analysis/projects_timeline_bert_ea_llm.parquet` | LLM-adjudicated timeline dates for EA projects |
| `data/analysis/projects_timeline_bert_eis_llm.parquet` | LLM-adjudicated timeline dates for EIS projects |
| `data/analysis/documents_combined.parquet` | All documents (filtered to FEIS and final EA, `total_pages` column) |

### Merge Logic

1. Start with clean energy EA + EIS projects from `projects_combined`
2. Inner join with combined EA + EIS timeline data (LLM dates)
3. Inner join with deduplicated final documents (one per project)
4. Filter to complete timelines (non-missing initiation + decision)
5. Add FRA period classification and time variables

### Key Output Objects (in R session)

- `pages_data` — full merged dataset (projects + timeline + documents)
- `pages_analysis` — analysis subset with complete timelines + FRA classification
- `coverage` — project counts at each filter step

---

## Scripts

| Script | Purpose |
|--------|---------|
| `code/deliverable05/00_setup.R` | Load libraries, merge data, define theme/helpers |
| `code/deliverable05/01_pages.R` | Generate 5 figures + summary tables |

### Figures Produced

| # | File | Description |
|---|------|-------------|
| 1 | `05_coverage.png` | Horizontal bar chart: project counts at each inclusion filter step, by EA/EIS |
| 2 | `05_pages_over_time.png` | 6-month rolling average of page counts over time, with red FRA line, faceted by EA/EIS |
| 3 | `05_pages_pre_post_fra.png` | Bar chart: mean pages Pre vs Post FRA, with median diamonds, faceted by process type |
| 4 | `05_pages_distribution_boxplot.png` | Violin + box plot of page distributions Pre vs Post FRA, with median labels |
| 5 | `05_pages_scatter.png` | Project-level scatter (decision date vs pages) with LOESS trend + 95% CI, FRA line |

### Tables Produced

| File | Description |
|------|-------------|
| `05_pages_summary.csv` | Descriptive stats (mean, median, sd, min, max, IQR) by process type and FRA period |
| `05_coverage.csv` | Project counts at each filter step |

---

## Design Decisions

1. **LLM dates for both EA and EIS**: Follows D3 harmonization pattern — uses `llm_initiation_date` and `llm_decision_date` from the LLM-adjudicated timeline files
2. **Decision date for FRA classification**: If `timeline_decision_date >= 2023-06-03`, the project is Post-FRA. Projects in-progress when FRA was enacted should comply
3. **Rolling average (not monthly raw)**: Monthly averages are noisy due to small N in some months. The 6-month rolling average smooths this out
4. **Time range 2010-2025**: Focuses figures on the most relevant recent period
5. **P99 cap on distribution figure**: Extreme outliers distort violin/box plots; y-axis capped at 99th percentile

---

## Known Considerations

- **Post-FRA sample size**: Only ~2.5 years of post-FRA data (June 2023 to present). Small N limits statistical power
- **Page count = main document only**: `total_pages` reflects the single final document, not appendices or supplementary volumes. FRA limits target the main document, so this is appropriate
- **Confounders**: Changes in page counts may reflect project complexity, agency mix, or technology shifts — not just FRA compliance. Analysis is descriptive, not causal
- **Zero-page documents**: Some may exist as data quality artifacts; worth inspecting

---

## Next Steps

- [ ] Run `01_pages.R` and inspect figures
- [ ] Review coverage numbers to assess analysis representativeness
- [ ] Check for zero-page or extreme outlier documents that may need filtering
- [ ] Create `reports/deliverable05.qmd` incorporating the figures
- [ ] Consider adding statistical tests (e.g., Wilcoxon) for Pre vs Post FRA comparison
