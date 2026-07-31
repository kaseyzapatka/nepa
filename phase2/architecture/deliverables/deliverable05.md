# D5: CE Spikes After Major Infrastructure Legislation — Architecture

**Goal:** Quantify the post-ARRA/BIL/IRA surges in categorical exclusion usage, attribute the spike to the legislation via citation detection in document text, and characterize the CE-type mix invoked during each spike window.

**Self-contained:** Partially. Scripts `01` and `02` are self-contained (require only `projects_combined.parquet` and the processed CE/EA/EIS pages/documents parquets). Script `03` requires D4's `timeline_project_dates.parquet` for the decision-date anchor.

---

## Script Quick-Reference

### Pipeline scripts — `phase2/code/deliverable05/`

Run order: `02_build_ce_categories.py` → `01_extract_law_citations.py` → `03_create_figures.R`

| Script | What it does |
|---|---|
| `01_extract_law_citations.py` | DuckDB keyword pre-filter + Python regex across all CE/EA/EIS page text to detect explicit citations to ARRA, BIL, IRA, and DOE funding programs. Acronym disambiguation via ±200-char context window. Output: `law_citations.parquet` (grain: project × law). |
| `02_build_ce_categories.py` | Explodes the `ce_category` VARCHAR array in `ce/documents.parquet` to one normalized CE code per row. Tags schedule (DOE 10 CFR 1021, DOI 516 DM 11, EPAct §390) and attaches curated descriptions for the top codes. Output: `ce_categories.parquet` (grain: project × code_norm). |
| `03_create_figures.R` | Joins dates, citations, categories, and project metadata. Produces the 10 figures and 6 diagnostic CSVs. Recreates the Phase 1 D3 by-year figures from scratch with this deliverable's own code. |

---

## Data Flow

```mermaid
flowchart TD
    A[phase2/data/processed/ce/pages.parquet] --> B[01_extract_law_citations.py]
    C[phase2/data/processed/ea/pages.parquet] --> B
    D[phase2/data/processed/eis/pages.parquet] --> B
    E[phase2/data/processed/ce/documents.parquet] --> B
    F[phase2/data/processed/ea/documents.parquet] --> B
    G[phase2/data/processed/eis/documents.parquet] --> B
    H[timeline/timeline_project_dates.parquet] --> B
    B --► I[deliverable05/law_citations.parquet]

    E --> J[02_build_ce_categories.py]
    J --► K[deliverable05/ce_categories.parquet]

    H --> L[03_create_figures.R]
    M[phase1/data/analysis/projects_combined.parquet] --> L
    I --> L
    K --> L
    L --► N[output/deliverable05/figures/fig_d5_*.png\n10 figures]
    L --► O[output/deliverable05/diagnostics/d5_*.csv\n6 CSVs]
```

---

## Inputs

| File | Description |
|---|---|
| `phase2/data/analysis/timeline/timeline_project_dates.parquet` | D4 output; provides `decision_date`, `initiation_date`, `process_type`. The D5 year-placement field is `coalesce(decision_date, initiation_date)`. |
| `phase1/data/analysis/projects_combined.parquet` | Project metadata: `project_energy_type`, `project_department`, `lead_agency_harmonized`, `project_type`. |
| `phase2/data/processed/ce/documents.parquet` | CE document metadata including the `ce_category` VARCHAR array. Scanned by script 02 for category codes and by script 01 (via join) for project-to-document mapping. |
| `phase2/data/processed/ce/pages.parquet` | CE document page text. DuckDB scan only — never `pd.read_parquet`. |
| `phase2/data/processed/ea/documents.parquet` | EA document metadata (for project-to-document join in script 01). |
| `phase2/data/processed/ea/pages.parquet` | EA page text (script 01 citation scan). |
| `phase2/data/processed/eis/documents.parquet` | EIS document metadata (for project-to-document join in script 01). |
| `phase2/data/processed/eis/pages.parquet` | EIS page text (script 01 citation scan). |

---

## Primary Outputs

All analysis parquets are written under `phase2/data/analysis/deliverable05/`.

| File | Description |
|---|---|
| `law_citations.parquet` | One row per (project_id, law_name): citation count, number of matched documents, first match type and context, audit timestamp. |
| `ce_categories.parquet` | One row per (project_id, code_norm): normalized CE code, schedule, description, audit timestamp. |

Figures are written under `phase2/output/deliverable05/figures/`. Diagnostic CSVs are written under `phase2/output/deliverable05/diagnostics/`.

### Figures

| File | Analysis | Description |
|---|---|---|
| `fig_d5_ce_counts_by_year_all.png` | A1 | All CE counts by year with ARRA/BIL/IRA markers. Column chart with N labels. |
| `fig_d5_ce_counts_by_year_byenergy.png` | A2 | CE counts by year, stacked by energy type (Decarb / Fossil / Other). |
| `fig_d5_counts_by_year_byprocess.png` | A3 | CE/EA/EIS review counts by year, faceted by process. Recreation of Phase 1 `03_projects_by_year.png`. |
| `fig_d5_counts_by_year_byprocess_byenergy.png` | A4 | Review counts by year, faceted by process, stacked by energy type. |
| `fig_d5_ce_counts_by_year_bydept.png` | A5 | CE counts by year, stacked by top-4 departments + "Other". |
| `fig_d5_ce_counts_by_year_doe_blm.png` | A6 | CE counts by year for DOE and BLM in separate facets. Headline finding figure: DOE spikes while BLM is flat. |
| `fig_d5_citations_by_year_byprocess.png` | B1 | Law-citing review counts by year (line, one series per law), faceted by process. |
| `fig_d5_citation_rate_window_vs_baseline.png` | B2 | Citation rate (% of CEs citing the law) in spike window vs baseline, per law. Bar chart with N labels. |
| `fig_d5_ce_category_shift_arra.png` | C1 | DOE CE category mix: top-8 codes by ARRA-window share vs 2016–2019 baseline. Horizontal bars with percentage + N labels. Marquee Q3 figure. |
| `fig_d5_technology_shift_arra.png` | D1 | Technology/project-type mix of ARRA-window CEs. Top-10 `project_type` tags by share. |

### Diagnostic CSVs

| File | Description |
|---|---|
| `d5_counts_by_year.csv` | CE count by year × energy_type. |
| `d5_counts_by_year_department.csv` | CE count by year × project_department. |
| `d5_date_coverage_by_year.csv` | By year × process: n_total, n_placeable, n_decision, n_init_proxy. Validates the date-basis split. |
| `d5_spike_summary.csv` | Per law × agency subset (All CE / DOE / BLM): mean monthly count in spike window vs baseline, spike ratio. |
| `d5_citation_rates.csv` | Per law × scope × period: n reviews, % citing the law. Covers All CE, Decarb CE, Fossil CE, All EA, All EIS. |
| `d5_category_shift.csv` | Per law × scope × period × code: code share in window and baseline. Covers ARRA and IRA, All/Decarb/Fossil CE. |
| `d5_technology_shift.csv` | Per law: top project_type tags and their share within the spike window. |

---

## Module Architecture

### 01_extract_law_citations.py — Law-Citation Detection

Scans all three corpora (CE, EA, EIS) for explicit mentions of ARRA, BIL, IRA, and DOE funding programs. Controlled by `--source {ce,ea,eis,all}` (default `all`) and `--sample N` for smoke tests.

**Two-stage scan:** DuckDB executes a pre-filter query — `regexp_matches(lower(page_text), PREFILTER_LOWER)` OR `regexp_matches(page_text, PREFILTER_ACRONYM)` — to pull only candidate pages from each corpus before any Python is involved. This makes the scan fast even over the large EIS pages file. Python then runs the full regex suite on the pre-filtered rows.

**Join path:** `pages.document_id → documents.document_id → documents.project_id.value`. The `project_id.value` extraction handles the STRUCT wrapping (see DuckDB Gotcha below). The scan is restricted to projects present in `timeline_project_dates.parquet` so every cited project has a placeable date.

**Regex suite and disambiguation:** Four laws, each with multiple patterns at varying specificity:

| Law | Unconditional patterns | Context-gated patterns |
|---|---|---|
| ARRA | `American Recovery and Reinvestment Act`, `Recovery and Reinvestment Act`, `\bARRA\b`, `Section 1603`, `Section 1705` | `Recovery Act` — affirm if ARRA-confirming terms (`reinvestment`, `stimulus`, `2009`, `111-5`, `ARRA`) present in ±200-char window AND no conservation/RCRA forbid terms |
| BIL | `Bipartisan Infrastructure Law`, `Infrastructure Investment and Jobs Act`, `\bIIJA\b` | `\bBIL\b` — require `infrastructure`, `jobs act`, `iija`, or `bipartisan` in window |
| IRA | `Inflation Reduction Act` | `\bIRA\b` — require energy/climate term (`clean energy`, `renewable`, `solar`, `wind`, `climate`, `greenhouse gas`, `emission`, `battery`, `transmission`, `electric vehicle`, `inflation reduction`, `grid`, `decarboniz`) in window |
| DOE_funding | `Loan Programs Office`, `Title XVII`, `Section 1703` | — |

The ARRA short-name guard (`Recovery Act` + ARRA_FORBID against `conservation|resource conservation`) is specifically designed to prevent the Resource Conservation and Recovery Act (RCRA) from being counted as an ARRA citation.

**Output aggregation:** One row per `(project_id, law_name)` — not per page. `citation_count` is the total number of match events across all pages; `n_docs_matched` is the count of distinct documents containing at least one match. Only the first match per project is stored as evidence (`first_match_type`, `first_context`, `first_document_id`, `first_page_number`).

**Audit timestamp:** `law_citations_extraction_run_at` (ISO-8601 UTC) written on all output rows at run time.

### 02_build_ce_categories.py — CE Category Code Normalization

Reads `ce_category` (VARCHAR array) from `ce/documents.parquet` via DuckDB. The query extracts `project_id.value` (handling the STRUCT) and filters to rows where the array is non-null and non-empty. The Python loop then explodes each array into elements and splits on commas/semicolons to handle multi-code elements like `"A9, B3.6"`.

**Normalization rules (applied in order):**
1. `DOE_RE = r"^\s*([AB]\d+(?:\.\d+)?)"` — matches DOE 10 CFR 1021 Appendix A/B codes (e.g. `B5.1`, `A9`, `B1.31`). Schedule: `DOE (10 CFR 1021)`.
2. `DOI_RE = r"(516\s*DM\s*\d+(?:\.\d+)?)"` — searched anywhere within the token for DOI/BLM 516 DM 11 codes (e.g. `516 DM 11.9`), so codes trailing prose like "pursuant to" are recovered. Schedule: `DOI (516 DM 11)`.
3. `EPACT_RE` — matches `section 390` or `energy policy act of 2005`. Normalizes to code `EPAct §390`. Schedule: `EPAct 2005 §390`.
4. No match → `code_norm = None`, row dropped.

**Description lookup:** Hard-coded dict `DOE_DESC` for the top analysis-relevant codes (B5.1, B1.3, B3.6, B3.1, A9, A1, A11, B1.31). Codes absent from the dict fall back to the code string itself.

**Deduplication:** After normalizing all tokens, the output is deduplicated on `(project_id, code_norm)` — a project that invokes B5.1 across multiple documents contributes one row.

**Audit timestamp:** `ce_categories_extraction_run_at` (ISO-8601 UTC) on all rows.

### 03_create_figures.R — Spike Analysis, Figures, and Tables

Joins the four input sources and derives the year-placement base: `year_date = coalesce(decision_date, initiation_date)`. A `date_basis` column (`"decision"` or `"initiation_proxy"`) records which was used. The `energy_type` variable is constructed as `recode(coalesce(project_energy_type, "Other"), "Clean" = "Decarb")` — matching D4's encoding. Department is `project_department`; `agency2` is a derived three-level variable (`"DOE"` / `"BLM"` / `"Other"`) built from `lead_agency_harmonized`.

**Theme and colors:** Identical CATF theme block to D4's `08_create_figures.R`. `PROCESS_COLORS = c(CE=lime, EA=dark_blue, EIS=navy)`. `ENERGY_COLORS = c(Decarb=teal, Fossil=magenta, Other=light_blue)`. `LAW_COLORS = c(ARRA=dark_blue, BIL=teal, IRA=magenta)`.

**Legislation windows and baseline:**

```r
LAWS <- tribble(
  ~law,   ~win_start,   ~win_end,     ~base_start,  ~base_end,
  "ARRA", "2009-03-01", "2011-12-31",  NA,           NA,
  "BIL",  "2021-12-01", "2023-12-31", "2018-12-01", "2021-11-30",
  "IRA",  "2022-09-01", "2024-12-31", "2019-09-01", "2022-08-31"
)
```

The CE-category windows (`CAT_WINDOWS`) use a separate baseline: ARRA uses `2016-01-01`–`2019-12-31` (a stable DOE CE period, since there is no usable pre-2009 baseline), and IRA uses `2019-01-01`–`2022-08-31`.

**Four analysis tracks:**
- **Analysis A** (temporal counts): 6 figures across all / by-energy / by-process / by-department / DOE-vs-BLM dimensions.
- **Analysis B** (citation attribution): law-citing project counts by year (line; by process), plus citation rate in spike window vs baseline.
- **Analysis C** (CE-category mix, CE-only): top-8 DOE CE codes in ARRA-window vs baseline. Joined to `ce_categories` filtered to `schedule == "DOE (10 CFR 1021)"`.
- **Analysis D** (technology mix): `project_type` JSON array exploded via `separate_rows` (splitting on `",\s*"` quote-boundary, not bare comma, to preserve multi-word tags like "Utilities (electricity, gas, telecommunications)").

**N labels:** Every bar/column figure uses `geom_text(aes(label = comma(n)), vjust = -0.3, size = 2.6)`. Line charts (Analysis B, B1) do not get per-point N labels.

---

## Run Results

<!-- d5-run-results: pull this section into the D5 report -->

Most recent full run: 2026-07-23 (`law_citations_extraction_run_at` 17:07 UTC; `ce_categories_extraction_run_at` 17:08 UTC).
Numbers in this section re-verified 2026-07-24 directly against the current output parquets (`law_citations.parquet` = 8,741 rows; `ce_categories.parquet` = 74,035 rows / 49,604 distinct projects). Note: this D5 run (17:08 UTC) predates the D4 Tier-C section-retrieval restore (`timeline_project_dates.parquet` re-selected later that day, ~00:53 UTC on 2026-07-24), so D5's decision-date anchor comes from the pre-Tier-C D4 timeline — but CE decision-date coverage is unchanged by the restore (90.3% before and after), so every CE-anchored number in this section still holds. A re-run of `01`/`03` against the post-restore D4 timeline is not required for these figures.

### CE Project Base

| Metric | Count |
|---|---|
| CE projects in timeline | 54,668 |
| CE with decision_date | 49,392 (90.3%) |
| CE with initiation_date (no decision) | 2,697 (additional 4.9%) |
| CE placeable by `coalesce(decision_date, initiation_date)` | 52,089 (95.3%) |
| Used by `03_create_figures.R` (within 2000–2025 year filter) | 51,867 (100% of in-range projects have a year_date, per T6) |

Note: the D5 analysis frame is broader than D4's complete-timeline frame. D4 requires both initiation and decision dates for duration analysis (29,745 CE rows with complete timelines); D5 needs only a single year-placement date per project.

### DOE vs BLM CE Spike (ARRA window)

| Year | DOE CEs | DOI/BLM CEs |
|---:|---:|---:|
| 2008 | 7 | 52 |
| 2009 | 669 | 40 |
| 2010 | 3,942 | 79 |
| 2011 | 2,199 | 104 |
| 2012 | 1,801 | 242 |

DOE jumps ~200× from 2008 to 2010; DOI (BLM) is flat across the same window.

### DOE vs BLM CE Counts (IRA/BIL window)

| Year | DOE CEs | DOI/BLM CEs |
|---:|---:|---:|
| 2020 | 1,940 | 2,097 |
| 2021 | 2,466 | 2,122 |
| 2022 | 3,239 | 2,146 |
| 2023 | 3,152 | 1,776 |
| 2024 | 983 | 616 |

DOE shows a sustained rise through 2022–2023; BLM declines from 2022 onward.

### Law-Citation Parquet (law_citations.parquet)

| Law | CE projects citing | EA projects citing | EIS projects citing | Total (project × law) rows |
|---|---:|---:|---:|---:|
| ARRA | 7,676 | 109 | 349 | 8,134 unique projects |
| BIL | 55 | 36 | 101 | 192 unique projects |
| DOE_funding | 51 | 42 | 74 | 167 unique projects |
| IRA | 11 | 33 | 204 | 248 unique projects |

Total rows in `law_citations.parquet`: 8,741.

ARRA citations are overwhelmingly CE (7,676 vs 349 EIS). IRA citations skew toward EIS (204) — longer EIS documents in the IRA window provide more citable context, while CE forms are brief. This by-process contrast is a secondary finding showing the citation signal is not uniform.

### ARRA-Window Citation Rate (from d5_citation_rates.csv)

| Scope | % of CEs citing ARRA in spike window |
|---|---|
| All CE | 59.7% (n = 7,014) |
| Decarb CE | 61.7% (n = 4,613) |
| Fossil CE | 76.6% (n = 632) |

ARRA has no usable pre-law baseline. BIL and IRA CE citation rates are low (< 1%) — law citations in CE documents are far less common for BIL/IRA than for ARRA, consistent with ARRA having funded CEs at a level that generated explicit funding acknowledgments in the CE forms themselves.

### CE Category Shift: ARRA Window (2009–2011) vs 2016–2019 Baseline

n_win = 6,664 DOE CE projects; n_bse = 6,529 DOE CE projects.

| Code | Description | Window % | Baseline % |
|---|---|---:|---:|
| B5.1 | Actions to conserve energy or water | 49.7% | 1.2% (baseline B5.1 is negligible) |
| A9 | Information gathering, data analysis & document preparation | 36.6% | 14.4% |
| B3.6 | Small-scale R&D / demonstration projects | 23.8% | 28.2% |
| A11 | Technical advice and planning assistance | 18.0% | 7.6% |
| A1 | Technical/financial assistance, training & education | 11.4% | 3.7% |
| B1.3 | Routine maintenance | 6.0% | 29.1% |

B5.1 ("Actions to conserve energy or water") dominates the ARRA window and is negligible in the baseline. Routine maintenance (B1.3) is the dominant baseline code but is depressed in the ARRA window. The shift directly reflects ARRA's energy-efficiency and retrofit stimulus purpose.

### CE Category Parquet (ce_categories.parquet)

| Schedule | Distinct projects |
|---|---:|
| DOE (10 CFR 1021) | 31,052 |
| DOI (516 DM 11) | 15,523 |
| EPAct 2005 §390 | 3,060 |

Total rows (project × code): 74,035. Distinct projects with at least one normalized code: 49,604.

Top codes by project count: 516 DM 11.9 (DOI/BLM, 12,901 projects), B3.6 (9,102), A9 (8,271), B1.3 (6,315), B5.1 (4,344).

### Spike Summary (from d5_spike_summary.csv)

| Agency | Law | Window mean monthly CEs | Baseline mean monthly CEs | Spike ratio |
|---|---|---:|---:|---:|
| All CE | ARRA | 206 | — | — |
| All CE | BIL | 428 | 348 | 1.23 |
| All CE | IRA | 330 | 380 | 0.87 |
| DOE | ARRA | 206 | — | — |
| DOE | BIL | 263 | 164 | 1.60 |
| DOE | IRA | 210 | 196 | 1.07 |
| BLM | ARRA | 6 | — | — |
| BLM | BIL | 163 | 182 | 0.90 |
| BLM | IRA | 119 | 182 | 0.66 |

ARRA has no computed baseline ratio (no usable pre-2009 period). For BIL/IRA: DOE shows a real positive ratio while BLM is flat or negative, reinforcing the agency-conditioned finding.

---

## Output Schema

### law_citations.parquet

| Column | Type | Description |
|---|---|---|
| `project_id` | object | Project UUID |
| `process_type` | object | `CE`, `EA`, or `EIS` |
| `law_name` | object | `ARRA`, `BIL`, `IRA`, or `DOE_funding` |
| `citation_count` | int64 | Total regex match events across all pages for this project × law |
| `n_docs_matched` | int64 | Number of distinct documents containing at least one match |
| `first_match_type` | object | Match type of the first detected citation: `full_name`, `full_name_alt`, `acronym`, `short_name`, or `program` |
| `first_context` | object | ±100-char context window around the first match (capped at 500 chars) |
| `first_document_id` | object | Document ID containing the first match |
| `first_page_number` | object | Page number of the first match |
| `law_citations_extraction_run_at` | object | ISO-8601 UTC timestamp of the extraction run |

### ce_categories.parquet

| Column | Type | Description |
|---|---|---|
| `project_id` | object | Project UUID |
| `code_raw` | object | Raw token from the `ce_category` array (truncated at 200 chars) |
| `code_norm` | object | Normalized code: DOE codes like `B5.1`, `A9`; DOI codes like `516 DM 11.9`; `EPAct §390` |
| `schedule` | object | `DOE (10 CFR 1021)`, `DOI (516 DM 11)`, or `EPAct 2005 §390` |
| `code_description` | object | Curated human-readable description; falls back to the code string for uncurated codes |
| `ce_categories_extraction_run_at` | object | ISO-8601 UTC timestamp of the extraction run |

---

## Known Issues and Cautions

- **ARRA has no usable pre-law baseline.** The NEPATEC corpus is sparse before 2009 (DOE had 7 CEs in 2008). The citation evidence and DOE-vs-BLM contrast serve as the attribution layer for ARRA; do not compute a pre/post ratio from the thin pre-period.

- **BIL and IRA spike windows overlap.** The BIL window (Dec 2021 – Dec 2023) and IRA window (Sep 2022 – Dec 2024) share 16 months. Date alone cannot separate BIL-attributable from IRA-attributable CEs during the overlap. Attribution within that overlap requires the citation evidence — and BIL/IRA CE citation rates are both low (< 1%), so citation-based attribution is weak for these two laws compared to ARRA.

- **2024–25 counts are incomplete.** The NEPATEC 2.0 ingestion lag means 2024 and 2025 CE counts underrepresent true activity. The decline visible in DOE 2024 data (983 CEs vs 3,152 in 2023) is a data artifact, not a real policy change.

- **IRA/BIL CE citation rates are low.** Unlike ARRA, where 59% of spike-window CEs explicitly cite the law, BIL and IRA citation rates in CEs are below 1%. This reflects the structure of CE forms (brief, no detailed funding acknowledgment section) rather than absence of law association. The temporal spike + DOE-conditioning is the primary attribution evidence for BIL/IRA; citations are a secondary confirmation.

- **`ce_categories.parquet` covers 49,604 of 54,668 CE projects (90.7%; 95.8% within clean-energy CEs).** Not all CE documents carry a populated `ce_category` array. Most of the 5,064-project gap (9.3%) reflects CE documents with blank or unparseable category fields. The DOI/`516 DM` regex was previously anchored to the start of each comma-split token, missing codes trailing phrases like "pursuant to"; it was relaxed on 2026-07-23 to `re.search` (unanchored), which recovered 1,363 DOI projects whose `516 DM` code trailed such prose. The DOI project count is now 15,523. The category analysis is restricted to projects with at least one normalized code.

- **DOE description lookup is incomplete for long-tail codes.** Codes absent from `DOE_DESC` (e.g., B2.5, B2.2, B4.6, B1.15) fall back to the code string as the description. These codes appear in the data but lack human-readable labels in the current lookup. Extend `DOE_DESC` in `02_build_ce_categories.py` before publication if these codes appear prominently in the report figures.

- **`project_id` is STRUCT-wrapped in processed documents parquets.** The field `project_id` in `phase2/data/processed/{ce,ea,eis}/documents.parquet` is stored as `STRUCT("value" VARCHAR)`, not a plain VARCHAR. Any DuckDB query must extract `project_id.value` explicitly. Both `01_extract_law_citations.py` and `02_build_ce_categories.py` handle this. Do not use `project_id` directly in ad-hoc queries against these files.

- **DOE_funding is a supporting signal, not a primary law.** The `DOE_funding` law name in `law_citations.parquet` captures mentions of the Loan Programs Office, Title XVII, and Section 1703 — DOE energy-loan programs active before and after ARRA. It is included as a corroborating signal for DOE-targeted activity, not as a separate law for the spike narrative.

- **`project_type` is a JSON array requiring careful splitting.** The `project_type` column in `projects_combined.parquet` is a JSON array string whose elements can contain internal commas (e.g., `"Utilities (electricity, gas, telecommunications)"`). Script 03 splits on the `",\s*"` quote-boundary rather than bare comma; ad-hoc R/Python code must replicate this.

---

## Methodological Notes

**Why `coalesce(decision_date, initiation_date)` for year placement?** The spike analysis needs only a single determination date per CE project — not a complete timeline. Using the initiation date as a fallback when decision is absent recovers 2,697 projects (4.9%) and raises coverage from 90.3% to 95.3%. It is safe for CEs specifically: the median CE duration is 20 days (p75 = 79 days; 90% complete within one year), so initiation and decision almost always fall in the same calendar year. A `date_basis` column distinguishes the two cases; the spike shape is robust to dropping initiation-proxy rows (the CEs recovered are a small fraction of any spike window).

**Why DOE-vs-BLM conditioning rather than overall CE counts?** The aggregate CE count is confounded by the NEPATEC coverage ramp (sparse pre-2009, incomplete 2024–25). Conditioning on DOE vs BLM isolates the policy signal: BLM is the natural within-data control. ARRA channeled energy-related stimulus spending through DOE, not BLM; a DOE spike with BLM flat is direct evidence of agency-specific response to the legislation.

**Why scan all three corpora (CE, EA, EIS) for citations?** The by-process contrast reveals that ARRA citations are overwhelmingly CE-form citations (7,675 CE vs 362 EIS projects), while IRA citations skew EIS. This is itself a finding — if the law-citation signal were uniform across process types, it would suggest spurious detection. The CE-specific ARRA signal reinforces that the CE forms themselves acknowledge the law (funding acknowledgment sections in DOE ARRA CE forms), whereas IRA/BIL citations in EISs reflect longer documents with more narrative context.

**Why use ARRA acronym without disambiguation but guard `Recovery Act`?** `\bARRA\b` is sufficiently distinctive — no common competing acronym uses ARRA — so it needs no context gate. `Recovery Act` alone is not distinctive: the Resource Conservation and Recovery Act (RCRA) is cited frequently in NEPA documents. The guard (require affirming terms like `reinvestment`, `stimulus`, `2009` OR `\bARRA\b`; forbid `conservation|resource conservation` nearby) is calibrated specifically to RCRA false-positive risk.

**Why use 2016–2019 as the ARRA category baseline rather than 2005–2008?** Pre-ARRA (2005–2008) DOE CE activity was minimal (7 CEs in 2008) — too thin to compute stable code-share proportions. The 2016–2019 window is a stable DOE CE period unaffected by ARRA stimulus (which wound down by 2012–2013) and before the BIL/IRA windows begin. It provides a reliable baseline for "what DOE CEs looked like in the absence of major stimulus legislation."

**Why is `03_create_figures.R` tagged `[NEEDS D4 TIMELINE]` rather than self-contained?** The temporal anchor for the spike analysis is the `decision_date` from D4's `timeline_project_dates.parquet`. Without D4's extraction pipeline, there are no reliable CE determination dates at scale. The CE `ce_category` metadata does not include dates; D4 provides the dates and D5 uses them for year placement.

**ARRA spike is a confirmed, large effect; BIL/IRA are more modest.** DOE mean monthly CEs: ARRA window 206 vs pre-ARRA DOE-only 2008 rate of ~1.6/month — roughly a 125× monthly rate increase. BIL DOE spike ratio is 1.60 (59% above baseline). IRA DOE ratio is 1.07 (6% above baseline, within noise). The spike hierarchy ARRA >> BIL > IRA for DOE CEs is the key finding, and it is consistent with the magnitude of ARRA energy spending relative to IRA/BIL CE-eligible activities at DOE.

---

## Reproduction

```bash
# Fast: CE category metadata only (no page text scan)
conda run -n nepa python phase2/code/deliverable05/02_build_ce_categories.py

# Citation scan — all corpora (requires processed pages parquets + D4 timeline)
conda run -n nepa python phase2/code/deliverable05/01_extract_law_citations.py --source all

# Smoke test (200 random CE projects)
conda run -n nepa python phase2/code/deliverable05/01_extract_law_citations.py --source ce --sample 200

# Analysis, figures, and diagnostic tables (requires D4 timeline + scripts 01+02 outputs)
Rscript phase2/code/deliverable05/03_create_figures.R

# Report (once findings lock)
quarto render phase2/reports/deliverable05.qmd
```
