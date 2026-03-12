# Deliverable 06: Data Architecture and Generation Methods

This document describes how each major dataset used in Deliverable 06 was constructed, including key design decisions, pipeline steps, and known limitations. Intended as a technical reference for the final project report.

---

## 1. Technology-Specific Analyses Overview

**Report section:** "Technology-Specific Analyses: Transmission Lines, Geothermal, and Pipelines"
**Primary extraction code:** `code/extract/extract_technology.py`
**Setup code:** `code/deliverable06/00_setup.R`
**Analysis scripts:** `code/deliverable06/01_transmission.R`, `02_geothermal.R`, `03_pipelines.R`
**QC scripts:** `code/deliverable06/04_identification_qc.R`, `05_length_validation.R`
**Report:** `reports/deliverable06.qmd`

### Deliverable questions

1. **Transmission lines** — How do extracted line lengths correlate with NEPA review duration, geographic location, and project action type?
2. **Geothermal energy** — What NEPA timeline patterns emerge across phases of geothermal development (exploration, drilling, plant construction)?
3. **Carbon and hydrogen pipelines** — How do NEPA timelines for emerging pipeline technologies compare to natural gas pipelines, and does pipeline length predict review duration?

All three analyses are restricted to **clean energy projects** (`project_energy_type == "Clean"`) across all three NEPA process types (CE, EA, EIS).

---

## 2. Source Data

| File | Description |
|------|-------------|
| `data/analysis/projects_combined.parquet` | Project-level metadata including all technology/length extraction fields written by the Python pipeline |
| `data/analysis/projects_timeline_bert.parquet` | BERT-extracted timeline dates for CE projects |
| `data/analysis/projects_timeline_bert_ea_llm.parquet` | LLM-adjudicated timeline dates for EA projects |
| `data/analysis/projects_timeline_bert_eis_llm.parquet` | LLM-adjudicated timeline dates for EIS projects |

The technology-specific fields (`project_is_transmission`, `project_transmission_length_miles`, `project_geothermal_phase`, `project_pipeline_group`, etc.) are written entirely by the Python extraction pipeline into `projects_combined.parquet`. R does not re-derive these fields — if they are missing they remain `NA`.

---

## 3. Python Extraction Pipeline

**Primary code:** `code/extract/extract_technology.py`
**Invoked via:** `python code/extract/extract_data.py --mode analysis`

`extract_technology.py` is imported by `extract_data.py` and run as part of the standard analysis-mode pipeline. It generates technology flags, phase classifications, and length extraction fields for all three domains.

---

### 3.1 Transmission Line Classification

Transmission projects are identified through a two-tier flag system: **broad** and **strict**.

#### 3.1.1 Broad classification (`project_is_transmission_broad`)

A project is flagged as broad-transmission if its combined text (title + description + type) contains any of:
- `\belectricity transmission\b`
- `\btransmission line\b`
- `\btransmission\b`

This is intentionally permissive and produces more matches than the analysis uses. It serves as an efficient pre-filter before the more expensive length extraction.

#### 3.1.2 Maintenance exclusion (`project_is_transmission_maintenance`)

Before length extraction, projects matching a vegetation management or routine maintenance regex on their **title** are flagged as `project_is_transmission_maintenance = TRUE`. These are excluded from the strict classification and candidate extraction. The title-only check avoids excluding genuine transmission projects where incidental maintenance language appears in the description (e.g., "access road maintenance" in a pole-replacement project).

Maintenance patterns include: vegetation management, herbicide treatment, weed control, road maintenance, reclamation, routine inspection, and related terms.

#### 3.1.3 Strict classification (`project_is_transmission_strict` / `project_is_transmission`)

A project passes the strict filter only if **all three** conditions hold:

1. `project_type` contains `Electricity Transmission` (`project_has_transmission_type_tag == TRUE`)
2. Title or description contains explicit build-related transmission text (`project_has_transmission_build_text == TRUE`), matched against `TRANSMISSION_BUILD_RE` — a multi-pattern regex covering:
   - "new transmission line"
   - "transmission line project / route / corridor"
   - "transmission project / corridor / facility" (no "line" required — catches e.g. "Gateway West Transmission Project")
   - "construct / build / install / upgrade / rebuild ... [kV] transmission line"
   - "double-circuit / single-circuit ... transmission line"
   - `\d{2,4} kV (transmission) line`
   - `HVDC` / "high-voltage direct current"
   - "gen-tie line / transmission" / "generating tie line"
   - "right-of-way ... **new** transmission line" (narrowed from old branch that matched any ROW renewal mentioning a transmission line)
3. Extracted transmission length `>= 1` mile (after adjudication)

`project_is_transmission` is an alias of `project_is_transmission_strict`. The `>= 1 mile` threshold was chosen conservatively to exclude administrative or measurement artifacts.

---

### 3.2 Transmission Length Extraction

Length extraction is a multi-step process: candidate extraction → value grouping → rule-based selection → optional LLM adjudication.

#### 3.2.1 Candidate extraction (`_extract_length_candidates`)

The full project text is split into sentences. Each sentence is checked for hint terms (`"transmission"`, `"powerline"`, `"power line"`, `"kV transmission"`, `"electric line"`, `"line route"`). Sentences with at least one hint term are scanned for:

- **Miles patterns** (`MILES_RE`): numeric values followed by `mile`, `miles`, or `mi`
- **Feet patterns** (`FEET_RE`): numeric values followed by `foot`, `feet`, or `ft`, converted to miles at 5,280 ft/mile

Each candidate record includes:
- `value_miles`: the extracted numeric value (rounded to 3 decimal places)
- `hint_score`: count of hint terms in the sentence, plus bonuses
- `sentence_has_build_verb`: whether the sentence contains a construction/authorization verb
- `is_partial_crossing`: whether context indicates a partial land-crossing extent rather than total line length
- `source_text`: a bounded snippet (~500 characters) centered on the matched numeric value

Several false-positive filters are applied before a candidate is accepted:
- **Geographic direction filter**: candidates where the miles value is immediately followed by a cardinal direction (e.g., "26 miles north of Helena") are dropped — these are location references, not line lengths.
- **Width context filter** (feet only): candidates where the sentence contains width-context words (`wide`, `corridor width`, `right-of-way width`) are dropped — these measure easement width, not line length.
- **Partial crossing filter**: candidates where context indicates the value measures how far a line crosses a specific land type (`crosses public lands for X miles`, `X miles on federal land`) are flagged with `is_partial_crossing = TRUE` and deprioritized during adjudication.
- **Mile-post filter**: sentences containing `mile post` or `MP ` are skipped entirely.

A `+2` bonus is added to `hint_score` for sentences containing explicit total-length language (`miles long`, `miles in length`, `total length of`, `overall length`).

#### 3.2.2 Candidate grouping (`_collapse_candidates_by_value`)

Near-equal candidate values (within 0.01 miles) are collapsed into groups to avoid treating unit-conversion duplicates as distinct candidates. Within each group, the representative (highest hint_score) candidate is retained.

#### 3.2.3 Rule-based length selection (`_rule_based_length_selection`)

After grouping, the rule-based selector resolves the final length according to a priority cascade:

| Taxonomy | Condition | Result |
|----------|-----------|--------|
| `none` | Zero candidates | NaN |
| `unique_match` | Exactly one non-trivial candidate | That value |
| `single_nontrivial` | One non-partial candidate, others are partial | The non-partial value |
| `alternative_take_max` | Multiple candidates, project text signals alternatives (route options) | Largest value |
| `sum` | Multiple candidates, project text signals additive segments (`also included`, `lateral`, `segment`) | Sum of selected values |
| `build_verb_winner` | One candidate's sentence uniquely contains a build verb | That candidate's value |
| `take_max` | Ambiguous multi-candidate | Highest hint_score, then largest value |

The rule-based result is stored in `project_transmission_length_miles` (the "comparison baseline").

#### 3.2.4 LLM adjudication (`_run_llm_transmission_adjudication`)

LLM adjudication is **triggered** when two or more non-trivial, non-partial candidate groups remain after rule-based resolution (`llm_trigger = TRUE`). When LLM is enabled (`--use-llm` flag passed to the extraction script), a Claude API call is made for each triggered project.

The prompt presents up to 8 candidates with their source snippets and instructs the model to pick the candidate most likely to represent the total proposed line length. The model is explicitly told to:
- Prefer candidates with "X miles long" or "X miles in length" language
- Ignore partial-crossing candidates
- Ignore geographic direction mentions
- Use the line being built, not existing reference lines

**Provider:** Claude API only — `claude-haiku-4-5-20251001` (`CLAUDE_DEFAULT_MODEL`). Requires `ANTHROPIC_API_KEY` environment variable. Ollama support has been removed.

**Run command:**
```bash
python code/extract/extract_technology.py --run transmission --use-llm --workers 4 \
  --page-length-recovery --output data/analysis/projects_combined.parquet
```

The LLM-adjudicated result is stored in `project_transmission_length_final`. If LLM was not triggered or failed, `project_transmission_length_final` equals `project_transmission_length_miles` (both are rule-based).

The analysis uses `project_transmission_length_final` (LLM-adjudicated when available) as the primary length variable.

**Audit columns** — all written to `projects_combined.parquet` per project:

| Column | Type | Meaning |
|--------|------|---------|
| `project_transmission_length_llm_trigger` | bool | Whether 2+ distinct candidates triggered adjudication |
| `project_transmission_length_llm_used` | bool | Whether Claude API was actually called and returned a result |
| `project_transmission_length_llm_status` | str | `success` / `not_triggered` / `not_requested` / `failed_fallback_rule` |
| `project_transmission_length_llm_reasoning` | str | Claude's one-sentence explanation |
| `project_transmission_length_llm_model` | str | Model string (e.g. `claude-haiku-4-5-20251001`) or `""` if not used |

To confirm Claude was used for a specific row: `llm_used == True` and `llm_model != ""`.

#### 3.2.5 Page-level length recovery (`_extract_tx_length_from_pages`)

A secondary extraction pass recovers lengths for projects that passed the build-text gate but have no mileage in their title/description text. Diagnostic analysis showed 1,268 such projects (1,184 CE, 60 EIS, 24 EA) — the full potential population above the current 151 strict projects.

**Trigger condition:** `project_has_transmission_type_tag & project_has_transmission_build_text & ~project_is_transmission_maintenance & (project_transmission_length_final < 1 or NaN)`. Only these projects are searched; the pass for all other rows is a no-op.

**Document targeting (efficiency):**
- **CE projects**: Each CE document is a single page blob — all pages are read (no filtering needed). DuckDB joins `documents.parquet` → `pages.parquet` on `document_id`, filtering to target project IDs.
- **EA/EIS projects**: Only `main_document = 'YES'` documents are queried, and only the first `max_ea_eis_pages` pages (default 10) per document via `ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY page_number)`. The "Proposed Action" and "Project Description" sections appear in these opening pages in virtually all EA/EIS formats.

**DuckDB join:** `project_id` in `documents.parquet` is a struct `{value: UUID-with-hyphens}`; `project_id` in `projects_combined.parquet` is a plain hex string. Both sides are normalized by stripping hyphens (`replace(d.project_id.value, '-', '')`) before joining. Target project IDs are registered as an in-memory DuckDB table (`_target_ids`) to avoid per-project queries.

**Extraction:** Page texts are concatenated per project and passed through the same `_extract_length_candidates()` → `_adjudicate_transmission_length()` pipeline used for title/description text. No new logic is introduced.

**Write-back:** For projects where the recovered length ≥ 1 mile, all standard transmission length columns are overwritten with the page-derived values. A new boolean column `project_transmission_length_from_pages = TRUE` marks these rows for provenance tracking. `project_is_transmission_strict` is re-evaluated after write-back, so recovered projects automatically enter the strict set.

**CLI:** Enabled by `--page-length-recovery` flag. `--page-search-max-pages N` controls the EA/EIS page depth (default 10). Not run by default (no flag = old behavior).

**Known limitation:** Recovery rate is lower than expected for the CE population because many projects that previously passed the build-text gate were ROW renewals entering via the old permissive ROW branch (`right-of-way.*transmission line`). That branch has been narrowed to require "new" in the vicinity, so fewer renewals enter the gate going forward. The remaining no-length projects are expected to be genuine builds where the length is not stated anywhere in the document text.

#### 3.2.6 Action type split (`project_transmission_new_build_miles`, `project_transmission_upgrade_miles`)

After adjudication, each length candidate is independently classified as `new_build` or `upgrade` using sentence-level regexes. Distinct candidate values are summed per action type to produce per-project estimates of how many miles are new-build versus upgrade activity.

---

### 3.3 Transmission Action Type Classification (`project_transmission_action`)

A project-level action type is derived by applying six regex patterns to the full project text (title + description). Categories:

| Action type | Example signals |
|---|---|
| `new_build` | "new transmission line", "new double-circuit", "new substation", "switchyard", "tap line" |
| `upgrade` | "reconductoring", "rebuilding", "component replacement", "crossarm", "insulator" |
| `maintenance` | "hazard tree", "structure inspection", "line inspection" (structural only, in-scope projects) |
| `fiber_optic` | "fiber optic", "OPGW", "optical ground wire", "telecom" |
| `renewal` | "right-of-way renewal", "ROW grant", "re-authorization" |
| `acquisition` | "acquire", "disposition", "transfer of easement" |
| `mixed` | Multiple categories match |
| `unknown` | No category matches |

---

### 3.4 Geothermal Phase Classification (`project_geothermal_phase`)

Geothermal projects are identified by `project_is_geothermal` (a keyword flag detecting `geothermal` in project text, set in the extraction pipeline). Phase is then classified by `_classify_geothermal_phase()`, which applies regex patterns to the full project text:

| Phase | Pattern examples |
|---|---|
| `exploration` | `\bexploration\b`, `\bexploratory\b`, `\bresource assessment\b`, `\bgeophysical survey\b` |
| `drilling` | `\bdrilling\b`, `\bdrill pad\b`, `\bproduction well\b`, `\binjection well\b` |
| `plant` | `\bpower plant\b`, `\bgenerating station\b`, `\bturbine\b`, `\binterconnection\b` |
| `multi_phase` | Multiple phase signals match simultaneously |
| `unknown` | "geothermal" is present in text but no phase pattern matches |
| `none` | "geothermal" does not appear in text at all |

Phase classification is limited to what appears in project title and description text; it does not scan document pages.

---

### 3.5 Pipeline Classification

Pipeline projects are identified via `project_is_pipeline` (keyword-based flag). Each pipeline project is then classified into a technology subtype:

| Field | Detection basis |
|---|---|
| `project_is_carbon_pipeline` | Carbon capture / CO₂ pipeline keywords |
| `project_is_hydrogen_pipeline` | Hydrogen pipeline keywords |
| `project_is_natural_gas_pipeline` | Natural gas pipeline keywords |
| `project_pipeline_group` | Rolled-up label for the above (with "Other pipeline" as the residual) |

Pipeline length extraction follows the same candidate extraction and adjudication logic as transmission, using `PIPELINE_HINTS` (`"pipeline"`, `"pipelines"`, `"right-of-way"`, `"row"`, `"buried line"`, `"flowline"`). The result is stored in `project_pipeline_length_miles` and `project_pipeline_length_confidence`.

---

## 4. R Data Pipeline

### 4.1 Timeline loading and harmonization (`00_setup.R`)

`load_timeline_for_deliverable6()` loads three separate timeline parquet files and row-binds them:

| File | Process types |
|---|---|
| `projects_timeline_bert.parquet` | CE (categorical exclusions, BERT-only dates) |
| `projects_timeline_bert_ea_llm.parquet` | EA (LLM-adjudicated dates) |
| `projects_timeline_bert_eis_llm.parquet` | EIS (LLM-adjudicated dates) |

After binding, dates are harmonized into unified `timeline_initiation_date_final` and `timeline_decision_date_final` columns:
- For EA and EIS: `llm_initiation_date` / `llm_decision_date` (the hybrid BERT+LLM pipeline's final output)
- For CE: `bert_initiation_date_final` / `bert_decision_date_final` (BERT-only)

The `timeline_method` column records which method produced each row's dates (`"llm"` or `"bert"`). Legacy field names (`bert_initiation_date_final`, `bert_decision_date_final`) are preserved as aliases of the harmonized dates so downstream scripts do not need conditional column selection.

**Why CE uses BERT-only:** CE projects do not have the LLM adjudication layer applied to them in the current pipeline, so they fall back to raw BERT predictions.

### 4.2 Technology field merge

After loading timelines, `00_setup.R` reads `projects_combined.parquet` and left-joins the technology columns by `project_id`. A fixed list of ~35 technology columns is requested; only columns actually present in `projects_combined.parquet` are joined (guarding against version mismatches). R does not recompute any missing technology fields.

### 4.3 `prepare_deliverable6_data()`

This helper function is called at the top of each analysis script:

1. Adds text columns (`project_title_txt`, `project_description_txt`, `project_type_txt`, `project_text_full`) by unnesting the JSON-stored list fields using `textify()` / `safe_fromJSON()`.
2. Calls `add_timeline_metrics()`: computes `bert_duration_days_final`, `bert_duration_months_final`, `project_state_primary` (first listed state), `project_region` (US Census region via state lookup), and `process_group`.
3. Applies the clean-energy filter (`project_energy_type == "Clean"`) when `clean_only = TRUE` (the default).

Technology extraction columns that are absent from `projects_combined.parquet` are stubbed as `NA` (logical, numeric, or character depending on type) so downstream code never fails on a missing column.

---

## 5. Transmission Analysis (`01_transmission.R`)

### 5.1 Analysis subset

The script filters to `project_is_transmission == TRUE` (strict classification, clean energy), then additionally excludes `project_transmission_action %in% c("fiber_optic", "renewal")` — these action types involve adding fiber optic cable to existing lines or renewing ROW grants, neither of which constitutes new line construction. `"unknown"` and `"mixed"` are retained as they may represent genuine builds with undetected action signals.

The working dataset `analysis_len` adds:
- `length_miles = project_transmission_length_final` (LLM-adjudicated when available, else rule-based)
- `duration_days = bert_duration_days_final`
- `length_bin`: `<10 mi`, `10–50 mi`, `50–100 mi`, `100+ mi`

### 5.2 Exploratory outputs (Google Sheets)

Three candidate-level tables are exported to a shared Google Sheet for QA:

| Sheet tab | Contents |
|---|---|
| `tx` | One row per extracted candidate across all strict transmission projects; includes taxonomy, LLM trigger/status, hint scores, and whether each candidate was selected |
| `tx_multiple` | Subset of projects with 2+ distinct candidate values; each row is one candidate |
| `tx_adjudication` | One row per project with 2+ distinct candidates; candidate values and LLM reasoning collapsed for review |

### 5.3 Tables

| File | Description |
|---|---|
| `table_transmission_summary.csv` | High-level counts: projects, with-length, with-duration, multi-state, median length, median duration |
| `table_transmission_length_bins.csv` | Projects and duration statistics (median, P90) by length band |
| `table_transmission_state_region.csv` | Projects, median length, median duration by state and census region |
| `table_transmission_action.csv` | Projects, median length, median duration by project action type |

### 5.4 Figures

| File | Description |
|---|---|
| `fig_transmission_length_distribution.png` | Histogram of extracted line lengths; dashed line at median |
| `fig_transmission_length_by_action.png` | Boxplot of length by action type (excludes none/unknown/mixed) |
| `fig_transmission_length_bins.png` | Side-by-side bar charts: project count and median NEPA duration by length band |
| `fig_transmission_state_n.png` | Lollipop: number of projects per state, colored by census region |
| `fig_transmission_state_length.png` | Lollipop: median extracted length per state, with project count labels |
| `fig_transmission_length_vs_duration.png` | Scatter: length vs. duration, colored by action type; overall linear trend |
| `fig_transmission_duration_by_region.png` | Boxplot + jitter: NEPA duration by census region (y-axis capped at 1,000 days) |
| `fig_transmission_duration_by_action.png` | Boxplot + jitter: NEPA duration by project action type |

---

## 6. Geothermal Analysis (`02_geothermal.R`)

### 6.1 Analysis subset

Filters to `project_is_geothermal == TRUE`, clean energy projects. A **normalized project key** (`geothermal_project_key`) is derived from the project title by stripping common geothermal-domain words (`geothermal`, `exploration`, `drilling`, `well`, `plant`, `project`, etc.) and punctuation. Projects whose key is fewer than 8 characters after normalization fall back to `project_id`. This key groups related NEPA actions within a single physical development for within-project sequencing analysis.

### 6.2 Tables

| File | Description |
|---|---|
| `table_geothermal_phase_distribution.csv` | Count and share of projects by geothermal phase |
| `table_geothermal_within_project_phases.csv` | Per-inferred-project summary: action count, distinct phases, date span, example title |
| `table_geothermal_phase_timeline.csv` | Duration statistics (median, P25, P75) by phase, restricted to projects with complete timelines |

### 6.3 Figures

| File | Description |
|---|---|
| `fig_geothermal_phase_duration_boxplot.png` | Boxplot of NEPA duration by phase (excludes `none`) |
| `fig_geothermal_within_project_sequence.png` | Gantt-style segment plot: initiation-to-decision segments per inferred project identity, colored by phase; top 250 rows by action count |

---

## 7. Pipeline Analysis (`03_pipelines.R`)

### 7.1 Analysis subset

Filters to `project_is_pipeline == TRUE`, clean energy projects. The `pipeline_group` variable is derived in R as an ordered factor from the Python-extracted subtype flags:
- `project_is_carbon_pipeline` → `"Carbon pipeline"`
- `project_is_hydrogen_pipeline` → `"Hydrogen pipeline"`
- `project_is_natural_gas_pipeline` → `"Natural gas pipeline"`
- else → `"Other pipeline"`

(Priority cascade: carbon > hydrogen > natural gas > other, so a project matching multiple subtype flags takes the first matching label.)

### 7.2 Tables

| File | Description |
|---|---|
| `table_pipeline_group_summary.csv` | Per-group counts, length coverage, and duration statistics |
| `table_pipeline_state_region.csv` | Projects, median length, median duration by region/state/group |
| `table_pipeline_baseline_compare.csv` | Median duration and length for carbon, hydrogen, and natural gas groups |
| `table_pipeline_correlations.csv` | Pearson correlations between pipeline length and duration/doc_count/multi_state |

### 7.3 Figures

| File | Description |
|---|---|
| `fig_pipeline_duration_by_group.png` | Boxplot of NEPA review duration by pipeline technology group |
| `fig_pipeline_length_vs_duration.png` | Scatter of pipeline length vs. duration, per-group trend lines |

---

## 8. QC and Validation Scripts

### 8.1 Identification QC (`04_identification_qc.R`)

Produces a structured overview of technology identification coverage and two stratified audit samples for manual review:

| Output | Description |
|---|---|
| `table_identification_overview.csv` | Key counts: total clean-energy projects, broad/strict/broad-only transmission, geothermal flagged/unknown-phase/keyword-not-flagged, pipeline flagged |
| `table_identification_by_process.csv` | Identification counts and percentages broken out by process type (CE / EA / EIS) |
| `table_transmission_identification_audit_sample.csv` | 60 strict + 60 broad-only (possible strict misses) projects with title, type, description snippet, and manual review columns |
| `table_geothermal_identification_audit_sample.csv` | 60 geothermal-flagged + 60 geothermal-keyword-not-flagged projects for false positive / false negative review |

### 8.2 Length Extraction Validation (`05_length_validation.R`)

Produces stratified samples for manual length validation:

| Output | Description |
|---|---|
| `table_length_extraction_coverage.csv` | Per-technology coverage: projects, with-length, high/medium/low confidence, median length |
| `table_transmission_length_validation_sample.csv` | 90 projects with extracted lengths (balanced by confidence bucket) + 30 missing-length projects (balanced by process type) |
| `table_pipeline_length_validation_sample.csv` | 90 projects with extracted lengths + 50 missing-length projects |

Each validation row includes: `extracted_length_miles`, `confidence_bucket`, `extraction_source_text`, and blank `manual_length_found`, `manual_length_miles`, `manual_source_excerpt`, `manual_notes` columns for the reviewer to fill in.

---

## 9. Output Schema

### Technology fields in `projects_combined.parquet`

All fields below are written by `extract_technology.py` and available for join in `00_setup.R`.

#### Transmission flags and classification

| Field | Type | Description |
|---|---|---|
| `project_is_transmission_broad` | Boolean | Broad transmission flag (keyword-based) |
| `project_has_transmission_type_tag` | Boolean | `project_type` contains `Electricity Transmission` |
| `project_has_transmission_build_text` | Boolean | Title/description contains explicit build-language match |
| `project_is_transmission_maintenance` | Boolean | Title indicates vegetation management or maintenance-only scope |
| `project_is_transmission_strict` | Boolean | All three strict criteria satisfied |
| `project_is_transmission` | Boolean | Alias of `project_is_transmission_strict` |
| `project_transmission_action` | String | `new_build`, `upgrade`, `maintenance`, `fiber_optic`, `renewal`, `acquisition`, `mixed`, `unknown` |

#### Transmission length extraction

| Field | Type | Description |
|---|---|---|
| `project_transmission_length_miles` | Float | Rule-based selected length (baseline, always populated if candidates found) |
| `project_transmission_length_final` | Float | Final length: LLM answer if LLM ran, else rule-based |
| `project_transmission_length_confidence` | String | `high`, `medium`, `none` |
| `project_transmission_length_source_text` | String | Text snippet containing the matched numeric value |
| `project_transmission_length_taxonomy` | String | Rule taxonomy used: `unique_match`, `build_verb_winner`, `take_max`, `sum`, `alternative_take_max`, `llm`, `none` |
| `project_transmission_length_selection_method` | String | `rule` or `llm` |
| `project_transmission_length_candidate_count` | Integer | Raw candidate count before grouping |
| `project_transmission_length_distinct_candidate_count` | Integer | Distinct value groups after near-equality collapse |
| `project_transmission_length_candidates_json` | String | JSON array of all candidates (for QA) |
| `project_transmission_length_selected_candidate_ids` | String | JSON array of IDs of selected candidates |
| `project_transmission_length_llm_trigger` | Boolean | Whether LLM adjudication was triggered |
| `project_transmission_length_llm_used` | Boolean | Whether LLM call succeeded |
| `project_transmission_length_llm_status` | String | `success`, `failed_fallback_rule`, `not_triggered`, `not_requested` |
| `project_transmission_length_llm_reasoning` | String | LLM's one-sentence explanation of its choice |
| `project_transmission_new_build_miles` | Float | Sum of candidate lengths classified as new-build action type |
| `project_transmission_upgrade_miles` | Float | Sum of candidate lengths classified as upgrade action type |
| `project_transmission_length_from_pages` | Boolean | TRUE if length was recovered from document page text (not title/description) |

#### Geothermal and pipeline

| Field | Type | Description |
|---|---|---|
| `project_is_geothermal` | Boolean | Geothermal keyword present in project text |
| `project_geothermal_phase` | String | `exploration`, `drilling`, `plant`, `multi_phase`, `unknown`, `none` |
| `project_is_pipeline` | Boolean | Pipeline keyword flag |
| `project_is_carbon_pipeline` | Boolean | Carbon capture / CO₂ pipeline |
| `project_is_hydrogen_pipeline` | Boolean | Hydrogen pipeline |
| `project_is_natural_gas_pipeline` | Boolean | Natural gas pipeline |
| `project_pipeline_group` | String | Rolled-up subtype label |
| `project_pipeline_length_miles` | Float | Extracted pipeline length (rule-based) |
| `project_pipeline_length_confidence` | String | `high`, `medium`, `none` |
| `project_pipeline_length_source_text` | String | Text snippet containing the matched length value |
| `project_pipeline_length_candidate_count` | Integer | Raw candidate count |
| `project_pipeline_length_distinct_candidate_count` | Integer | Distinct value groups |
| `project_pipeline_length_candidates_json` | String | JSON array of all pipeline candidates (for QA) |

---

## 10. Key Design Decisions

### 10.1 Strict vs. broad transmission definitions

Using only the broad definition (any mention of "transmission") would include thousands of projects — including many where "transmission" is incidental (e.g., a solar project with a short interconnect tie line). The strict definition requires a type tag, explicit build-related language, AND a non-trivial extracted length. This is conservative: some genuine transmission projects may be missed if their descriptions use non-standard phrasing or their lengths are not mentioned in title/description text. The broad-only set (projects passing broad but not strict) is provided in the QC output for manual review.

### 10.2 LLM adjudication only for ambiguous cases

The LLM adjudication layer runs only when 2+ non-trivial, non-partial candidate lengths remain after rule-based resolution. For the majority of projects (single unambiguous candidate, or rules resolve the contest), the LLM is never called, keeping runtime manageable. The rule-based result is always stored for comparison even when the LLM runs.

### 10.3 Geothermal within-project sequencing via normalized title keys

To identify that two different NEPA records represent sequential stages of the same physical geothermal development, titles are normalized by stripping domain-generic words. This is a heuristic: it may merge unrelated projects that happen to share a location name, and may split a single project whose titles vary across reviews. No ground-truth match dataset exists to validate grouping accuracy.

### 10.4 Pipeline technology comparison uses natural gas as a baseline

Carbon and hydrogen pipeline NEPA timelines are compared to natural gas pipelines because natural gas provides the closest established-technology analog with sufficient sample size. "Other pipeline" is retained as a residual category but excluded from the key comparison.

### 10.5 Timeline method varies by process type

CE timelines come from BERT-only extraction; EA and EIS timelines use the LLM-adjudicated dates from the hybrid pipeline. This is consistent with other deliverables (Deliverable 03, Deliverable 05) and reflects the availability of the LLM layer only for EA/EIS.

---

## 11. Known Limitations

### 11.1 Length extraction coverage gaps

Length extraction depends on numeric mile mentions in project title and description text. Projects whose lengths appear only in document body text (not in title or description) will have `NA` for length fields. Coverage is substantially better for transmission (nearly all strict projects have a usable length) than for pipelines (~52% coverage for natural gas; lower for carbon and hydrogen).

### 11.2 Small pipeline samples

Carbon (n=8) and hydrogen (n=4) pipeline projects are too few for reliable statistical comparisons. All pipeline findings should be framed as preliminary and descriptive.

### 11.3 Geothermal phase completeness

Half of all geothermal projects have `project_geothermal_phase == "unknown"` — the project text mentions geothermal but no phase-specific signal was detected. Phase-specific conclusions are therefore limited to the minority of projects with classifiable phases.

### 11.4 Transmission false positives from geographic directions

The location-direction filter (dropping "26 miles north of Helena") catches most geographic references, but multi-word directional phrases or less standard geographic mentions may still produce spurious length candidates. These would be selected only if no better candidate exists, and they would typically be outscored by legitimate build-language candidates.

### 11.5 Action type completeness

Approximately 15% of strict transmission projects receive `action = "unknown"` — no action-type signal is detectable in their text. These projects are excluded from action-type figures and tables.

### 11.6 Timeline dependency for duration analysis

All duration-based results require calculable timelines (both initiation and decision dates). Projects without extractable dates are excluded from duration comparisons. This affects pipeline analysis most severely, where the already-small samples are further reduced.

---

## 12. Validation

No formal automated validation pass exists for Deliverable 06. The QC and validation scripts (`04_identification_qc.R`, `05_length_validation.R`) produce structured tables for **manual review**:

1. **Identification audit**: Samples of strict-transmission, broad-only-transmission, geothermal-flagged, and geothermal-keyword-not-flagged projects for client review of classification accuracy.

2. **Length validation sample**: Stratified samples of projects with extracted lengths (balanced by confidence level) and projects with missing lengths (for gap analysis). Reviewers read the source text snippet and record whether the extracted length is correct, along with any manual correction.

3. **Google Sheets upload** (transmission only): The candidate-level tables (`tx`, `tx_multiple`, `tx_adjudication`) are written to a shared Google Sheet for detailed QA of the length selection logic, including auditing LLM adjudication reasoning.

Validation status as of 2026-02-13: **pending client review**; length thresholds and strict classification criteria may be adjusted based on findings.
