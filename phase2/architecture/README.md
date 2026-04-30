# Phase 2 Architecture Overview

This folder contains explanatory documentation for Phase 2 — how data flows through the pipeline, why key design decisions were made, and what each script or deliverable produces. These files are designed to be used as baseline context and for reference when drafting the technical report.

These are **not runbooks** (step-by-step execution guides — those live in `runbooks/`). Architecture docs explain the *why* and *what* of each pipeline.

---

## Core Pipeline Scripts

| File | Script | Description |
|---|---|---|
| [code/extract_data.md](code/extract_data.md) | `extract_data.py` | Main pipeline entry point: builds `projects_combined.parquet`, merges all enrichment data |
| [code/federal_register.md](code/federal_register.md) | `federal_register.py` | Federal Register NOI/NOA matching: NEPATEC page scan + targeted direct fetch |

---

## Deliverables

| File | Deliverable | Self-contained? |
|---|---|---|
| [deliverables/deliverable01.md](deliverables/deliverable01.md) | D1: NEPA Triggered | Yes |
| deliverables/deliverable02.md *(pending)* | D2: Significant Impact Factors | Yes |
| deliverables/deliverable03.md *(pending)* | D3: NEPA Review Process Application | Partially (scripts 01–04 yes; script 05 needs timeline) |
| deliverables/deliverable04.md *(pending)* | D4: Timelines | No — requires timeline pipeline |
| deliverables/deliverable05.md *(pending)* | D5: CE Spikes After Major Legislation | Mostly (count analysis yes; duration sub-analysis needs timeline) |
| deliverables/deliverable06.md *(pending)* | D6: Patterns in FONSIs | Yes |

Deliverable architecture docs are added as each deliverable is implemented. See `plans/deliverable0N.md` for the corresponding implementation spec (deleted after build).

---

## Shared Foundation

All deliverables start from `extract_data.py`, which produces `data/analysis/projects_combined.parquet`. This contains one row per project with metadata: agency, project type, process type (CE/EA/EIS), energy type, geography, document references, and Federal Register NOI dates.

Federal Register NOI/NOA data is a refreshable Phase 2 artifact. Default `extract_data.py --mode analysis` runs offline and merges the existing `data/analysis/federal_register/federal_register.parquet` file if present. Use `--refresh-federal-register` to re-run the NEPATEC page scan and direct-fetch FR records for all doc numbers found. See [code/federal_register.md](code/federal_register.md) for how matching works and [runbooks/federal_register.md](../runbooks/federal_register.md) for the refresh commands.

**DuckDB** is used throughout for page-level scanning and parquet joins. Never load full page parquets into pandas — always use DuckDB's `read_parquet()` for memory efficiency.

---

## Timeline Data Integration

Timeline-dependent deliverables (D3 script 05, D4) join via `project_id` against `timeline_*.parquet`.

**Initiation date hierarchy:**
1. `noi_publication_date` from Federal Register — authoritative where present
2. `bert_initiation_date` from `extract_timeline.py`

**End-of-process date:**
- `noa_availability_date` from Federal Register — FEIS notice for EIS projects; FONSI/Final EA for EA projects

**Decision date hierarchy:**
1. `bert_decision_date`
2. `llm_decision_date` (EA/EIS adjudication cases only)
