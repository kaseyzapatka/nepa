# Phase 2 Architecture Overview

This folder contains explanatory documentation for each Phase 2 deliverable — how data flows from `extract_data.py` through extraction, analysis, and final output. These files are designed to be used as a baseline context and for reference when drafting the technical report.

These are **not runbooks** (step-by-step execution guides — those live in `runbooks/`). Architecture docs explain the *why* and *what* of each pipeline.

---

## Deliverable Index

| File | Deliverable | Self-contained? |
|---|---|---|
| [deliverable01.md](deliverable01.md) | D1: NEPA Triggered | Yes |
| [deliverable02.md](deliverable02.md) | D2: Significant Impact Factors | Yes |
| [deliverable03.md](deliverable03.md) | D3: NEPA Review Process Application | Partially (scripts 01–04 yes; script 05 needs timeline) |
| [deliverable04.md](deliverable04.md) | D4: Timelines | No — requires timeline pipeline |
| [deliverable05.md](deliverable05.md) | D5: CE Spikes After Major Legislation | Mostly (count analysis yes; duration sub-analysis needs timeline) |
| [deliverable06.md](deliverable06.md) | D6: Patterns in FONSIs | Yes |

---

## Shared Foundation

All deliverables start from `extract_data.py`, which produces `data/analysis/projects_combined.parquet`. This contains one row per project with metadata: agency, project type, process type (CE/EA/EIS), energy type, geography, document references, and Federal Register NOI dates.

Federal Register NOI data is a refreshable Phase 2 artifact. Default `extract_data.py --mode analysis` runs offline and merges the existing `data/analysis/federal_register/noi_federal_register.parquet` file if present, falling back to the old top-level artifact until the new output is generated. Use `--refresh-federal-register` only when intentionally querying the Federal Register API and regenerating the artifacts in `data/analysis/federal_register/`. API refreshes use date-windowed Federal Register pulls and write `data/analysis/federal_register/fr_noi_fetch_report.csv` so capped windows and corpus coverage can be audited before match-threshold tuning.

**DuckDB** is used throughout for page-level scanning and parquet joins. Never load full page parquets into pandas — always use DuckDB's `read_parquet()` for memory efficiency.

---

## Timeline Data Integration

Timeline-dependent deliverables (D3 script 05, D4) join via `project_id` against `timeline_*.parquet`.

**Initiation date hierarchy:**
1. `noi_publication_date` from Federal Register — authoritative where present
2. `bert_initiation_date` from `extract_timeline.py`

**Decision date hierarchy:**
1. `bert_decision_date`
2. `llm_decision_date` (EA/EIS adjudication cases only)
