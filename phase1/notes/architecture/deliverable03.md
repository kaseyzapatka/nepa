# Deliverable 03: Data Architecture and Generation Methods

This document describes how each major dataset used in Deliverable 03 was constructed, including key design decisions, pipeline steps, and known limitations. Intended as a technical reference for the final project report.

---

## 1. Process Types by Energy Classification

**Report section:** "CE vs. EA vs. EIS by Energy Type"
**Primary output:** `data/analysis/projects_combined.parquet`
**Primary code:** `code/extract/extract_data.py`
**Analysis code:** `code/deliverable03/01_process.R`

### Source data

The raw data comes from PNNL's NEPATEC 2.0 dataset, loaded from HuggingFace. NEPATEC 2.0 contains JSON records for over 60,000 NEPA projects (and ~120,000 documents) across 60+ federal agencies. The three process types — CE, EA, and EIS — are stored as separate HuggingFace datasets and are extracted, cleaned, and merged into a single file (`projects_combined.parquet`) by `extract_data.py`.

### Energy type classification

Each project is assigned a `project_energy_type` based on the `project_type` tags in the raw NEPATEC records (e.g., "Solar", "Wind", "Oil and Gas", "Transmission Lines"). Because NEPATEC tags are multi-valued (a project can be tagged as both "Transmission Lines" and "Solar"), a hierarchical classification was applied:

- **Clean**: any project tagged with a clean energy type (solar, wind, geothermal, hydro, biomass, batteries/storage, transmission, hydrogen, carbon capture), unless also flagged as fossil or excluded
- **Fossil**: any project with oil/gas/coal/fossil tags
- **Other**: projects not matching either

The "Utilities" tag receives special handling: projects tagged with *only* Utilities plus non-energy tags are excluded as non-energy infrastructure. Projects tagged with Utilities *and* a clean or fossil energy tag are retained and classified by that energy tag.

### Exclusion filters

Two categories of projects are excluded from all analysis before deliverable construction:

1. **Military nuclear**: defense-sector projects involving nuclear activities, identified via a manually curated list (`data/processed/military_project_ids_to_filter.csv`)
2. **Nuclear waste**: projects associated with nuclear waste management at DOE agencies identified in `data/processed/agencies_to_be_excluded.txt`

The exclusion logic is applied inside `extract_data.py` before the combined parquet is written.

### Analysis

`01_process.R` loads the combined parquet, filters to clean energy projects, and produces:
- **Figure 1**: Grouped bar chart of project counts by CE/EA/EIS and energy type
- **Figure 2**: Stacked bar chart showing composition of review types within energy categories

---

## 2. Timeline Analysis

**Report section:** "Timeline Analysis"
**Primary output:** per-source timeline parquet files (CE: BERT output; EA/EIS: LLM hybrid output)
**Primary code:** `code/extract/extract_timeline.py`
**Analysis code:** `code/deliverable03/03_timeline.R`

### Overall approach

Timeline extraction aims to recover two key dates for each project:
- **Initiation date**: when the NEPA process started (Notice of Intent, scoping, application received)
- **Decision date**: when the NEPA process concluded (ROD, FONSI, CE determination signature)

The extraction pipeline has three stages: (1) regex date detection, (2) context classification, and (3) date selection.

### Stage 1: Regex date detection

Ten date patterns are compiled and run across all page text for each project. Patterns cover:

| Pattern name     | Example                          |
|------------------|----------------------------------|
| MDY_full         | January 15, 2024                 |
| MDY_short        | Jan 15, 2024                     |
| DMY_full         | 15 January 2024                  |
| numeric_slash    | 01/15/2024                       |
| numeric_slash_2y | 01/15/24                         |
| ISO              | 2024-01-15                       |
| numeric_dash     | 01-15-2024                       |
| digital_sig      | 2024.01.15 (digital signature)   |
| MY_full          | January 2024 (month-year only)   |
| MY_short         | Jan 2024                         |

For each matched date, a context window (~150 characters before and after) is extracted and stored alongside the date. This context is what the classifier uses.

### Stage 2: Context classification

Each date-context pair is classified into one of four labels: **Decision**, **Initiation**, **Review**, or **Other**.

#### Auto-labeling (weak supervision)

Before training, a rule-based auto-labeler assigns provisional labels based on pattern matching against the context. Patterns are organized by strength (Strong → Medium → Weak) and label type:

- **Decision (Strong)**: digital signature syntax (`YYYY.MM.DD`), "digitally signed by", FONSI, ROD, CE determination language, "record of decision", "finding of no significant impact", authorizing official signatures
- **Decision (Med/Weak)**: "final approval", "determination", "approval"
- **Initiation (Strong)**: "notice of intent", "scoping meeting", "scoping period", "application received", "right-of-way application", "NOI published"
- **Initiation (Med)**: "proposed action", "NEPA process started", "request received"
- **Review (Strong)**: specialist role titles (wildlife biologist, archaeologist, realty specialist), reviewer checkbox forms ("Yes / No / Reviewer/Title / Initials & Date"), MOA/Section 106 references
- **Other / Exclusion**: RMP reference dates, boilerplate form language, dates far removed from any cue

#### BERT classifier (CE)

For CE projects, a DistilBERT text classifier is fine-tuned on the auto-labeled training data. Class imbalance (particularly for Initiation, which is rare in CE documents) is addressed via per-class loss weighting during training. The model classifies each date-context pair independently and quickly, making it well-suited for the ~19,000 CE projects. The trained model is stored at `models/timeline_classifier/`.

#### LLM hybrid (EA/EIS)

For EA and EIS projects, where document language is more complex and decision signals more varied, an LLM-based hybrid approach is used. BERT classifications serve as a first pass; an LLM provides validation and override for EA/EIS-specific decision cues (e.g., ROD language, joint ROD variants, final alternative selection). This approach trades throughput for accuracy on the smaller EA/EIS project sets (~573 EA, ~753 EIS).

### Stage 3: Date selection

After classification, one date per label type is selected per project using a scoring system that combines:
- **Pattern strength** (Strong > Med > Weak)
- **BERT/LLM confidence score**
- **Document type boost**: dates appearing in ROD, FONSI, or CE determination documents receive a boost
- **Historical gap rule**: dates occurring more than 2 years before the next date in the sequence are flagged as historical (e.g., a reference date from a prior environmental review)

For Decision dates, the system prefers the strongest signature cue. For Initiation dates, dates that fall *after* the selected Decision date are excluded.

### Coverage and known issues

Timeline completeness varies significantly by process type:
- EA: ~62% of projects have both dates
- EIS: ~48% of projects have both dates
- CE: ~30% of projects have both dates

CE coverage is limited primarily by **missing initiation dates**: CE documents typically do not include formal scoping notices or NOI language. EA/EIS coverage is limited primarily by **missing decision dates**, often because RODs and FONSIs are in separate documents that were not fully digitized or are not included in NEPATEC.

Initiation class imbalance in the BERT training data is a known bottleneck. Improvement strategies include expanded pattern matching for CE-specific initiation signals and manual annotation of examples.

---

## 3. Generation Capacity

**Report section:** "Generation Capacity Analysis"
**Primary outputs:**
- `data/analysis/projects_gencap.parquet` (regex results)
- `data/analysis/projects_gencap_merged.parquet` (after LLM merge)
**Primary code:** `code/extract/extract_gencap.py`
**Analysis code:** `code/deliverable03/02_capacity.R`

### Overview

Generation capacity extraction recovers the nameplate capacity (in power units: MW, GW, kW) and annual energy output (in energy units: MWh, GWh, kWh) for each clean energy project. The pipeline runs in two phases: a **regex phase** that scans project metadata and documents for numeric capacity mentions, followed by an optional **LLM adjudication phase** that resolves ambiguous cases where multiple conflicting candidate values were found.

---

### 3.1 Power vs. Energy Separation

A foundational design decision is that power and energy are tracked as **separate fields** throughout the pipeline. This distinction matters because NEPA documents often include both:
- Nameplate capacity (MW): the rated output of the facility — the primary metric for project scale
- Annual generation (MWh): a production estimate — useful but distinct

The regex unit patterns are sorted longest-first to ensure more specific units are matched before shorter ambiguous ones (e.g., `MWh` is matched before `MW`, `MWac` before `MW`). Power unit variants recognized include:

> MW, MWac, MWdc, MWe, MWt, MWth, MWp, GW, GWe, GWac, GWdc, kW, kWe, kWac, kWdc, and spelled-out variants (megawatt, gigawatt, kilowatt, with optional "electric" or "thermal" suffixes)

Energy unit variants include: MWh, GWh, kWh, and spelled-out forms.

Each extraction result records two parallel field sets: `project_gencap_value` / `project_gencap_unit` (power) and `project_gencap_energy_value` / `project_gencap_energy_unit` (energy). When a title or document yields a power match, it is stored as the power field regardless of whether an energy value was also found.

**Known limitation:** Energy captures conflate battery storage capacity (MWh as a storage metric) with annual generation projections (MWh as output). Distinguishing these requires LLM extension not yet implemented.

---

### 3.2 Regex Patterns

Four CAPACITY_PATTERNS templates are applied to each text source. They are designed to match different syntactic forms seen in NEPA documents:

1. **Bare quantity**: `50 MW`, `1.5 GW`, `500 kW` — number directly followed by unit
2. **Contextual noun**: `capacity of 50 MW`, `nameplate 200 MW`, `generating 100 megawatts`
3. **Hyphenated modifier**: `50-MW facility`, `100-megawatt solar project` — unit modifies a following noun
4. **Article + noun phrase**: `a 100-megawatt facility` — article precedes the capacity noun phrase

All patterns also accept optional range syntax (`50–100 MW`), prefix approximations (`about 50 MW`, `up to 100 MW`, `~50 MW`), and comma-formatted numbers (`1,500 MW`).

For each match, a context window (±80 characters) is extracted. This context is scored for confidence:
- **High**: contexts containing project-forward words (`proposed`, `nameplate`, `rated`, `will generate`, `would generate`, `facility`, `farm`, `array`)
- **Low**: contexts containing historical/reference words (`existing`, `previously`, `adjacent`, `another`, `prior`)
- **Ambiguous**: contexts with hedging words (`similar`, `comparable`, `reference`, `example`)

A separate `is_invalid_match()` filter removes matches that appear to be date-like patterns (e.g., `MW` adjacent to a date in a signature field, which is a known false-positive pattern in CE documents).

---

### 3.3 DuckDB-based Page Loading

NEPATEC 2.0 page text is stored in per-source parquet files (`data/processed/ce/pages.parquet`, `data/processed/ea/pages.parquet`, `data/processed/eis/pages.parquet`). These files are large and cannot be loaded fully into memory for each project iteration.

The pipeline uses **DuckDB** for bulk page retrieval across all projects in a given pass. The core query:

1. Registers the `project_docs` lookup table (project_id → document_id pairs) as an in-memory DuckDB relation
2. Joins the pages parquet directly from disk using `read_parquet()` — DuckDB applies predicate pushdown to only scan matching rows
3. Assigns each page a `ROW_NUMBER()` partitioned by `project_id`, ordered by `doc_rank` (document priority) then `page_num` (numeric page order)
4. Filters to `rn <= max_pages` (default cap: 50 pages per project)
5. Returns a grouped dict: `{project_id: [page_text, ...]}`

This design avoids loading the full pages parquet into pandas and instead pushes the filtering and ordering into DuckDB's query engine, which is substantially faster for this join pattern.

Documents are ranked by `doc_rank` based on a `main_document == "YES"` flag in the documents metadata, so main documents (the primary EA, EIS, or CE form) are scanned before supporting appendices or supplemental documents.

---

### 3.4 Tiering System (Title → Description → Documents)

The extraction pipeline uses a strict **three-tier fallback** to minimize document I/O while maximizing coverage:

#### Pass 1 — Title and description (no page I/O)

For every project, the pipeline first attempts extraction from `project_title` and `project_description` (structured metadata fields, always in memory). Within this pass:

1. **Title first**: run regex against the project title. Title matches are treated as `confidence="high"` unconditionally and assigned `project_gencap_source="title"`. The `is_initials_date_context()` filter is *not* applied to title matches (titles are assumed clean).
2. **Description fallback**: if the title yields no match, run regex against the project description. Description matches are also treated as `confidence="high"` and assigned `project_gencap_source="description"`.

If either the title or description yields a match, the project is considered resolved and is **skipped in all subsequent passes**. No document pages are ever loaded for these projects.

**Coverage by source:** Titles are rarely informative for capacity (~0.3% of projects). Descriptions are more useful, particularly for CE projects (~1,561 CE projects resolved via description in current runs). EA and EIS projects rarely have capacity in structured metadata fields.

#### Pass 2A — Main documents (DuckDB, capped)

Projects not resolved in Pass 1 are batched and queried via DuckDB. Only documents flagged as `main_document == "YES"` are included in this sub-pass. Pages are capped at 50 per project. Regex is run across the concatenated page texts for each project.

Projects resolved here are assigned `project_gencap_source="document"`.

#### Pass 2B — Other documents (DuckDB, capped)

Projects that still have no capacity after Pass 2A are re-queried using non-main documents (appendices, supporting files). The same 50-page cap applies.

#### Pass 2C — Full rescan (no cap)

A final rescan removes the page cap for projects still unresolved after Passes 2A and 2B, allowing all available pages to be searched. This pass is computationally expensive but ensures that projects with capacity only mentioned late in long documents are not missed.

---

### 3.5 LLM Adjudication (Claude Haiku)

#### When LLM is triggered

The LLM is triggered only for projects that, after document scanning, have **2 or more distinct power capacity candidates** (different value+unit combinations). Projects with 0 or 1 power candidates skip LLM adjudication entirely. Projects resolved via title or description in Pass 1 also never trigger the LLM, as single-source metadata hits are considered unambiguous.

The trigger threshold (default: 2 candidates) can be adjusted via CLI flag (`--min-candidates`). The LLM is run per-source (CE, EA, EIS) as separate sub-jobs.

#### What the LLM receives

Each triggered project's prompt includes:
- Project title and project type
- A numbered list of capacity candidates, each with the extracted value, unit, and surrounding context snippet (up to ~200 characters)

The prompt instructs the LLM to:
1. Select the candidate representing the **proposed project being reviewed**, not comparisons, existing infrastructure, or neighboring projects
2. Prefer candidates with context words like "proposed", "nameplate", "rated", or "will generate"
3. Ignore candidates describing existing systems, regional totals, or reference projects
4. Return `null` if no candidate clearly represents the proposed project capacity

#### LLM response format

The model (`claude-haiku-4-5-20251001`) returns a compact JSON response (max 200 tokens, temperature 0.1):

```json
{"selected_index": 2, "confidence": "high", "reasoning": "candidate 2 uses 'proposed nameplate capacity'"}
```

Retry logic handles rate limit errors with exponential backoff (up to 3 retries). Timeout and API errors are recorded in `llm_error` fields and do not crash the pipeline.

#### Fallback behavior

If the LLM returns `null` (no clear selection), a secondary fallback attempts to extract a value directly from the concatenated candidate context strings using another regex pass. If that also fails, the project retains its regex result (which may be the highest-priority candidate by unit size).

---

### 3.6 Merge Logic and Final Capacity Fields

After the LLM phase, results are merged back into the regex dataset in `merge_llm_results_into_regex()`. The merge applies the following decision logic to produce final capacity fields (`project_gencap_final_value`, `project_gencap_final_unit`, etc.):

| Condition | Final value |
|-----------|-------------|
| LLM not triggered | Regex result (highest-priority power candidate) |
| LLM triggered, valid power unit selected (GW/MW/kW), non-rejected method | LLM overrides regex |
| LLM triggered, selected unit is not a power unit | Regex result retained |
| LLM triggered, method is `no_candidates` / `llm_no_selection` / `llm_error` / `llm_timeout` | Regex result retained |
| No regex result and no LLM result | `null` |

A `llm_merge_decision` field records which branch was taken for each project:
- `regex_no_llm`: LLM was not triggered; regex result used
- `llm_override_regex`: valid LLM selection replaced a regex result
- `llm_only_fill`: LLM filled a gap where regex found nothing
- `regex_invalid_or_rejected_llm`: LLM ran but was rejected; regex result used
- `no_capacity`: neither regex nor LLM found a value

A `project_gencap_llm_selection_logic` field provides a more detailed audit trail of LLM behavior:
- `not_triggered`, `triggered_no_selection`, `selected_regex_candidate`, `selected_non_regex_candidate`, `selected_invalid_or_non_power`, `selected_valid_power_rejected_by_method`

---

### 3.7 Validation

Two validation scripts produce outputs for manual review:

**Stratified sample** (`03_gencap_validation_sample.py`): draws 30 projects per source (CE/EA/EIS), filtered to clean energy, for manual inspection. Output: `output/deliverable3/gencap_validation_stratified_sample.csv`.

**Validation flags** (`04_gencap_validation_flags.py`): four heuristic flags are applied to the merged dataset to identify likely false positives:

| Flag | Description |
|------|-------------|
| `gencap_flag_initials_date` | MW/kW value appears adjacent to a date (e.g., an initialing field like "MW 5/21/15") |
| `gencap_flag_non_generation` | Context suggests beam power, particle accelerators, thermal storage, or battery storage (not generation) |
| `gencap_flag_non_build` | Context describes removal, decommissioning, or maintenance rather than new construction |
| `gencap_flag_equipment_list` | Match appears inside a bulleted equipment list (multiple MW values in close proximity) |

A 30-row quick audit sample oversamples flagged rows. Output: `output/deliverable3/gencap_validation_quick_sample.csv`.

**Manual validation results (current run):** 11 of 20 sampled projects were independently verifiable; precision on verifiable rows was 100%. LLM spot-check showed 9/10 successful CE extractions.

---

### 3.8 Coverage and Capacity Bins

Coverage rates by process type (current run):
- **EIS**: ~81% of projects have a capacity value extracted
- **EA**: moderate coverage (varies by run)
- **CE**: ~8% of projects have a capacity value (most CE documents are brief forms without MW statements; title/description often do not mention capacity)

Capacity bins used in `02_capacity.R` analysis tables:
- **Small**: < 10 MW
- **Medium**: 10–100 MW
- **Large**: 100–500 MW
- **Utility-scale**: > 500 MW

Central tendency by process type (from current run):
- EIS projects: median ~538 MW (utility-scale dominated)
- EA projects: median ~60 MW (medium-to-large range)
- CE projects: median ~1.2 MW (small, typically distributed/rooftop solar)

---

### 3.9 Output Schema (Key Fields)

| Field | Description |
|-------|-------------|
| `project_gencap_value` | Regex-extracted power capacity (MW, GW, or kW) |
| `project_gencap_unit` | Unit for above |
| `project_gencap_energy_value` | Regex-extracted energy value (MWh, GWh, or kWh) |
| `project_gencap_energy_unit` | Unit for above |
| `project_gencap_source` | Where value was found: `title`, `description`, `document`, `none`, `no_documents` |
| `project_gencap_confidence` | `high` (title/description) or scored from context |
| `project_gencap_context` | Context snippet around the matched value |
| `project_gencap_candidates_json` | JSON array of up to 5 candidate power matches with context |
| `project_gencap_llm_triggered` | Whether LLM adjudication was run |
| `project_gencap_llm_reasoning` | LLM's one-sentence justification for selected candidate |
| `project_gencap_final_value` | Final resolved power capacity (after merge) |
| `project_gencap_final_unit` | Final resolved unit |
| `project_gencap_final_source` | Source of final value (`llm`, `regex`, etc.) |
| `llm_merge_decision` | Merge outcome category |
| `project_gencap_llm_selection_logic` | Detailed LLM decision audit trail |
