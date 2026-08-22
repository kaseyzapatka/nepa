# Phase 1 Architecture Overview

This folder contains explanatory documentation for Phase 1 — how data flows through the
pipeline, why key design decisions were made, and what each script or deliverable produces.
These files are designed to be used as baseline context and for reference when drafting
technical reports.

These are **not runbooks** (step-by-step execution guides — those live in `phase1/runbooks/`).
Architecture docs explain the *why* and *what* of each pipeline.

**Phase 1 is pandas-based.** Every extraction script loads parquet files into pandas
DataFrames and processes them in memory. This is the single biggest architectural difference
from Phase 2, which is DuckDB-based throughout (Phase 2 never loads full page parquets into
pandas). Phase 1's page-level scans use `pandas.read_parquet()` plus row-wise regex/apply,
which is workable at Phase 1's scale (~62K projects, ~6M pages) but is markedly slower than
Phase 2's DuckDB predicate-pushdown scans.

---

## Core Pipeline Scripts

| File | Script | Description |
|---|---|---|
| [code/extract_data.md](code/extract_data.md) | `extract_data.py` | Main pipeline entry point: builds `projects_combined.parquet` (97 columns, 61,881 rows) from EA/EIS/CE raw data, applies energy classification and military/nuclear-waste exclusion filters |
| [code/extract_timeline.md](code/extract_timeline.md) | `extract_timeline.py` | Shared timeline engine: regex date-candidate extraction + a DistilBERT 4-class classifier (decision/initiation/review/other), with an optional Claude/Ollama LLM adjudication pass for EA/EIS. Feeds D2, D3, D5, and D6. |
| [code/federal_register.md](code/federal_register.md) | `federal_register.py` | Federal Register NOI (Notice of Intent) enrichment via the FR keyword-search API. Run once, upstream of `extract_data.py`, per [runbook 01](../runbooks/01_base_dataset.md). |

Deliverable-specific extraction scripts (`extract_reviews.py`, `extract_gencap.py`,
`extract_pages.py`, `extract_technology.py`, `extract_coagency.py`,
`extract_coagency_names.py`) are documented inline within their owning deliverable's
architecture doc rather than as standalone files here, since each is used by exactly one
deliverable.

---

## Deliverables

| File | Deliverable | Self-contained? |
|---|---|---|
| [deliverables/deliverable01.md](deliverables/deliverable01.md) | D1: Technology, Agency, and Location | Yes — needs only `projects_combined.parquet` |
| [deliverables/deliverable02.md](deliverables/deliverable02.md) | D2: Programmatic & Tiered Reviews | Partially — review classification is self-contained; duration analysis needs the D3-owned timeline outputs (`extract_timeline.py`) |
| [deliverables/deliverable03.md](deliverables/deliverable03.md) | D3: Process Type, Generation Capacity, and Timelines | Partially — process-type table is self-contained; capacity needs `extract_gencap.py` output; timeline needs `extract_timeline.py` output |
| [deliverables/deliverable04.md](deliverables/deliverable04.md) | D4: Geography and Multi-Agency Review | Partially — multi-state analysis is self-contained; multi-agency analysis needs `extract_coagency.py`/`extract_coagency_names.py` sidecars |
| [deliverables/deliverable05.md](deliverables/deliverable05.md) | D5: Regulatory Page Counts (FRA) | Partially — needs `extract_pages.py` output and the D3-owned timeline outputs to compute pre/post-FRA duration |
| [deliverables/deliverable06.md](deliverables/deliverable06.md) | D6: Transmission, Geothermal, and Pipelines | Partially — needs `extract_technology.py` output; duration figures need the D3-owned timeline outputs |

---

## Shared Foundation

All deliverables start from `extract_data.py`, which produces
`phase1/data/analysis/projects_combined.parquet` — one row per project (97 columns, 61,881
rows) with agency, process type (CE/EA/EIS), energy classification, geography, document
availability flags, and Federal Register NOI fields. `phase1/code/extract/extract_technology.py`
adds transmission/geothermal/pipeline columns to this same table as a second pass (see
[runbook 06](../runbooks/06_technology.md)) — this is why the final `projects_combined.parquet`
carries 97 columns rather than the smaller set `extract_data.py` alone would produce.

**The clean energy universe is `project_energy_type == "Clean"`, n = 20,725** (broad
definition). `project_energy_type_strict == "Clean"` gives a conservative sensitivity cut of
19,628 that excludes Utilities+Broadband-only and Nuclear-Technology-only borderline cases.
Every deliverable filters to the broad definition by default. Military/nuclear-waste exclusion
lists are produced by `phase1/code/validation/military_review.R` and
`nuclear_waste_review.R` (writing `phase1/notes/military_project_ids_to_filter.csv` and
`phase1/notes/nuclear_waste_projects_to_keep.csv`), which `extract_data.py` then reads back in
on the next run — this is a human-in-the-loop review cycle, not a one-shot filter (see
[code/extract_data.md](code/extract_data.md#energy-classification-and-exclusion-filters)).

**No cross-phase dependency.** Phase 1 and Phase 2 each build `projects_combined.parquet`
independently from the same underlying NEPATEC 2.0 source tables — Phase 2's copy has 77
columns (a different, DuckDB-built pipeline) versus Phase 1's 97. As of this writing, no
Phase 2 script reads from `phase1/`, and this documentation set does not imply otherwise.

**All 62 R scripts across `phase1/code/deliverable01`–`06` share the same boilerplate
pattern**: a `00_setup.R` per deliverable loads `projects_combined.parquet` (plus any
deliverable-specific sidecar parquet), filters to `project_energy_type == "Clean"`, defines
the CATF brand `ggplot2` theme (`theme_catf()`, navy `#002169` / dark blue `#0047BB` palette),
and exposes shared helpers (`explode_column()` for JSON-array columns, `create_crosstab()`,
`add_totals_row()`). Numbered scripts (`01_*.R`, `02_*.R`, …) then `source()` that setup file
and produce one deliverable's figures/tables. `phase1/code/utils/utils.R` adds a small
`unpack_json()` helper used across several deliverables.

---

## Timeline Data Integration

Unlike Phase 2 (where timeline extraction is its own deliverable, D4), Phase 1 treats the
timeline pipeline (`extract_timeline.py`) as **shared infrastructure** consumed by four
deliverables rather than a deliverable of its own:

- **D2** joins timeline dates to compute review duration by review type (`reviews_tl` /
  `duration_data` in `deliverable02/00_setup.R`), with a small set of manually verified date
  overrides for edge cases hard-coded directly in that setup script (see
  [deliverable02.md](deliverables/deliverable02.md#known-issues-and-cautions)).
- **D3** owns the canonical timeline analysis (`03_timeline.R`) and defines the harmonization
  rule reused by D6: **CE projects use the BERT classifier's final dates
  (`bert_initiation_date_final` / `bert_decision_date_final`); EA/EIS projects use the
  LLM-adjudicated dates (`llm_initiation_date` / `llm_decision_date`)** — see
  `load_timeline_for_deliverable3()`.
- **D5** joins the LLM-adjudicated EA/EIS dates to classify projects Pre-FRA/Post-FRA
  (FRA enactment date: **June 3, 2023**) and to compute review duration.
- **D6** reuses D3's exact harmonization rule via its own copy,
  `load_timeline_for_deliverable6()`, to compute transmission/geothermal/pipeline duration.

**Why the CE/EA-EIS split?** CE is Phase 1's largest process type by an order of magnitude
(19,399 clean CE projects vs. 573 EA / 753 EIS), so per-project LLM calls are only affordable
for EA/EIS. CE relies entirely on the DistilBERT classifier plus regex/rule-based selection
logic; LLM adjudication is reserved for the smaller EA/EIS pools where it is both affordable
and most useful (EA/EIS documents are longer and more likely to contain multiple competing
date candidates). See [code/extract_timeline.md](code/extract_timeline.md) for the full
tier structure and measured coverage.

**Coverage at the frozen Phase 1 build** (verified against the committed parquet outputs):

| Process | n (clean) | Initiation coverage | Decision coverage | Both (duration-calculable) |
|---|---:|---:|---:|---:|
| CE | 19,399 | 42.6% | 78.8% | 30.4% |
| EA | 573 | 90.1% | 63.5% | 62.0% |
| EIS | 753 | 79.0% | 51.7% | 48.1% |

Initiation coverage is the binding constraint everywhere — CE in particular has weak
initiation signal because most CE documents never state an application/start date explicitly.
