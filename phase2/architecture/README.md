# Phase 2 Architecture Overview

This folder contains explanatory documentation for Phase 2 — how data flows through the pipeline, why key design decisions were made, and what each script or deliverable produces. These files are designed to be used as baseline context and for reference when drafting the technical report.

These are **not runbooks** (step-by-step execution guides — those live in `runbooks/`). Architecture docs explain the *why* and *what* of each pipeline.

---

## Core Pipeline Scripts

| File | Script | Description |
|---|---|---|
| [code/extract_data.md](code/extract_data.md) | `extract_data.py` | Main pipeline entry point: builds `projects_combined.parquet`, merges all enrichment data |

---

## API Register Sources (Tier-A metadata for D4 timelines)

These document the cached external-register lookups that feed D4's Tier-A date metadata. Each is fetched once (network required) and cached to a parquet the pipeline reads offline.

| File | Script | Description |
|---|---|---|
| [api/federal_register.md](api/federal_register.md) | `federal_register.py` | Federal Register NOI/NOA matching: NEPATEC page scan + targeted direct fetch (produces `noi_publication_date`/`noa_availability_date`) |
| [api/blm_register.md](api/blm_register.md) | `blm_register/01–03` | BLM ePlanning case-number scan + register fetch → `blm_eplanning_dates.parquet` (initiation/decision dates) |
| [api/doe_register.md](api/doe_register.md) | `doe_register/01–06` | DOE ePlanning ROD/FONSI + energy.gov CX-determination crawl → `doe_eplanning_dates.parquet`, `doe_cx_dates.parquet`, `doe_cx_register.parquet` |

---

## Deliverables

| File | Deliverable | Self-contained? |
|---|---|---|
| [deliverables/deliverable01.md](deliverables/deliverable01.md) | D1: NEPA Triggered | Yes — needs only `projects_combined.parquet` + CE/EA/EIS pages |
| [deliverables/deliverable02.md](deliverables/deliverable02.md) | D2: Determinations of Significance | Partially — significance extraction reads D6 FONSI finding-section spans; validation needs a hand-labeled gold set |
| [deliverables/deliverable03.md](deliverables/deliverable03.md) | D3: NEPA Review Process Application | Partially — core review/CE/geography/visual outputs self-contained; trigger-stratified CE summaries use D1 |
| [deliverables/deliverable04.md](deliverables/deliverable04.md) | D4: Project Timelines | Partially — core extraction self-contained; Tier-A registers need network (cached); geothermal/technology sub-analyses need D3 |
| [deliverables/deliverable05.md](deliverables/deliverable05.md) | D5: CE Spikes After Major Legislation | Partially — scripts 01/02 self-contained; `03_create_figures.R` needs D4 `decision_date` for year placement |
| [deliverables/deliverable06.md](deliverables/deliverable06.md) | D6: Patterns in FONSIs | Partially — needs D3 (reviews, CE citations) + D4 (decision dates); also writes `fonsi_conditions.parquet` consumed by D2 |

All six deliverable architecture docs now exist under `deliverables/`. See `plans/deliverable0N.md` for the corresponding implementation spec (deleted after build).

---

## Shared Foundation

All deliverables start from `extract_data.py`, which produces `data/analysis/projects_combined.parquet`. This contains one row per project with metadata: agency, project type, process type (CE/EA/EIS), energy type, geography, document references, and Federal Register NOI dates.

Federal Register NOI/NOA data is a refreshable Phase 2 artifact. Default `extract_data.py --mode analysis` runs offline and merges the existing `data/analysis/federal_register/federal_register.parquet` file if present. Use `--refresh-federal-register` to re-run the NEPATEC page scan and direct-fetch FR records for all doc numbers found. See [api/federal_register.md](api/federal_register.md) for how matching works and [runbooks/api/federal_register.md](../runbooks/api/federal_register.md) for the refresh commands.

**DuckDB** is used throughout for page-level scanning and parquet joins. Never load full page parquets into pandas — always use DuckDB's `read_parquet()` for memory efficiency.

---

## Timeline Data Integration

The timeline pipeline is **D4** (`code/deliverable04/`). It writes one row per project to `data/analysis/timeline/timeline_project_dates.parquet` with `initiation_date`, `decision_date`, their granularities and `*_source_type` provenance, `timeline_status`, and `duration_days`. Timeline-consuming deliverables join it via `project_id`: **D5** anchors CE-spike year placement on `decision_date`; **D6** merges `decision_date` for its post-FRA tabulation. D3 produces no timeline/duration figures — its dead timeline section was removed 2026-07-24 (duration analysis is entirely D4's domain).

D4 selects each date through a cascade (see `deliverables/deliverable04.md` for the full logic):

**Initiation date** — Tier-A metadata first (`fr_noi` from the Federal Register `noi_publication_date` is authoritative where present; then BLM/DOE register start dates), then document-text extraction (`03_extract_candidates.py` regex + `04_classify_candidates.py` learned scorer), then LLM adjudication (`06_adjudicate_llm.py`) for gaps.

**Decision date** — Tier-A register ROD/FONSI/CX dates first, then document-text extraction, then LLM adjudication. For EIS, ROD is preferred with FEIS as a documented fallback (`decision_is_feis_fallback`).
