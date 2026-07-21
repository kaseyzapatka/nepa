# Deliverable 04 — Timeline Extraction Pipeline

Extracts NEPA review **initiation** and **decision** dates per project (CE / EA / EIS) and
computes timeline durations and coverage.

## Pipeline spine (run in this order)

| # | Script | Does | Key output |
|---|--------|------|-----------|
| 00  | `00_sample.py` | build the 100-project validation sample | `timeline_sample100.csv` |
| 00b | `00b_sections.py` | extract document section headings | sections parquet |
| 01  | `01_index.py` | score & prioritize documents (scan_priority, register dates) | `timeline_document_index.parquet` |
| 02  | `02_retrieve.py` | pull page/section text into context packets | `timeline_context_packets.parquet` |
| 03  | `03_extract_candidates.py` | regex date extraction + role prelabel + rejection | `timeline_candidates.parquet` |
| 04  | `04_classify_candidates.py` | SetFit 3-head classifier (init / decision / final_eis) | candidates + `p_initiation`/`p_decision`/`p_final_eis` |
| 04b | `04b_calibrate.py --apply` | Platt calibration | candidates + `p_init_cal`/`p_dec_cal`/`p_feis_cal` |
| 05b | `05b_rank.py --apply` | LightGBM within-project ordering | candidates + `ranking_score` |
| 05  | `05_select_dates.py` | pick one initiation + one decision date per project | `timeline_project_dates.parquet` |
| 05c | `05c_inject_ground_truth.py --scope all` | overwrite with human-verified dates | project_dates (gt-injected) |
| 06  | `06_adjudicate_llm.py` | LLM adjudication of ambiguous picks (Claude Haiku); replays cached adjudications deterministically | project_dates |
| 07  | `07_validate.py` | compare selected dates to the gold sample | validation reports |
| 08  | `08_create_figures.R` | coverage tables, durations, figures | `diagnostics/` + `figures/` |

**Canonical run sequence:** `02 → 03 → 04 → 04b --apply → 05b --apply → 05 → 05c → 07 → 08`.

**Run it with one command:** `CONDA_DEFAULT_ENV=nepa ./run_pipeline.py` (full `02→08`) or
`./run_pipeline.py --select` (selection-only: `05b → 05 → 05c → 08`, minutes). This is the single
orchestrator — `04b`/`05b`/`05c` are baked in (skipping them corrupts CE via stale `ranking_score`).

`04b` / `05b` / `05c` load `04` / `05` via `importlib` because digit-prefixed files can't be
imported normally — follow that pattern for any new sibling stage.

## Paths

- **Data** (gitignored): `phase2/data/analysis/timeline/` — packets, candidates, project_dates, `models/`
- **Labels (inputs):** `phase2/training/deliverable04/` — `classifier.csv`, `ranker.csv`, `frozen_eval_ids.txt`
- **Outputs:** `phase2/output/deliverable04/` — `diagnostics/`, `figures/`

## Tools (not pipeline stages)

- `run_pipeline.py` — the one-command pipeline orchestrator (full or `--select`)
- `_diagnostics.py` — classifier label-inventory / confusion / calibration diagnostics

## Sub-tracks

- `labeling/` — gold-label building sub-pipeline (feeds the classifier retrain); its own `01–05`

## Post-pipeline analyses (require the spine to have been run first)

Everything below reads the spine's outputs — `timeline_document_index.parquet` (from `01`) and
`timeline_project_dates.parquet` (from `05`/`05c`/`06`) — so **the pipeline spine must be complete
through at least `05c` before any of these will run.** None of them modify spine outputs; each
follows the Python-builds-data / R-draws-figures split.

- `09_sample_check.R` — stratified eyeball sample of selected dates (manual QA; not used by the report)
- `10_outliers.R` — duration-outlier deliverable (feeds the report's case-study section)
- `fra/` — FRA page-length analysis: `01_extract_pages.py` (also needs processed EA/EIS
  `pages.parquet`) → `02_create_figures.R`. (The solar and duration-by-technology analyses
  formerly in `fra/` now live at top level: `08_create_figures_solar.R`,
  `08_create_figures_technology.R` — run after `08_create_figures.R`.)
- `field_office/` — office experience vs. process change, two arms: `01_parse_offices.py` (BLM field
  offices from the DOI-BLM case number) → `01b_build_doe_offices.py` (DOE administering offices from
  the CX register's `office` field, joined on `cx_number`; writes `doe_offices.parquet` + DOE coverage/
  count diagnostics) → `02_create_figures.R`. The reframe carries two figures — a combined BLM/DOE
  inventory and a busier- vs quieter-office **convergence** lead figure (`d4_fieldoffice_convergence.csv`
  + `d4_fieldoffice_convergence_split.csv` for the half definitions) — plus a per-year within-office
  Spearman table (`d4_fieldoffice_withinyear_cor.csv`) and the pooled office fixed-effects regression
  (`d4_fieldoffice_model.csv`, with `agency` and `frame` columns). Symmetric no-filter design: BOTH
  arms use each office's full review history (no calendar cut); known pre-2012 BLM artifact rows are
  retained and flagged in the report caveats (they inflate the raw estimate only). The convergence
  figure is BLM-only unless the DOE quieter half is both stable and actually converges (it does not:
  busier DOE offices are structurally faster). **Run order: `01` → `01b` → `02`.** The takeaway: process change, not office
  experience, likely drove the CE speed-up — the DOE arm reproduces the null independently.
- `ceq_regime/` — CEQ regulatory-regime durations: `01_build_tables.py` →
  `02_create_figures.R`; additionally requires `08_create_figures.R` to have run (reads its
  `d4_duration_summary.csv` consistency anchor and `d4_duration_by_year.csv`)
- `geothermal/` — geothermal review timelines by cohort: `01_build_tables.py` →
  `02_create_figures.R`; a three-tier reframe (office-matched BLM / unmatched BLM / DOE & other)
  with an office inventory (including a DOE CX-register-office panel — ~456 of the 764 DOE-tier
  projects link to a named grant/administering office, Golden Field Office dominant) and a state
  bubble map (the `maps` package supplies lower-48 polygons only). The decision-year split is
  retained only as a diagnostic CSV (`d4_geothermal_timeline_points.csv`), not a report figure.

## Conventions

- Scripts hard-require `CONDA_DEFAULT_ENV=nepa` (run via `conda run -n nepa python …`).
- Never `pd.read_parquet()` the pages files — query via DuckDB with `document_id` filters.
- Isolated runs: `--sample-ids <file>` (02/03/04/05/07) or `--run-dir <dir>` (04b/05b); sample runs
  auto-isolate to `timeline/sample_runs/<stem>/`.
