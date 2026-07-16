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
| 08  | `08_analyze.R` | coverage tables, durations, figures | `diagnostics/` + `figures/` |

**Canonical run sequence:** `02 → 03 → 04 → 04b --apply → 05b --apply → 05 → 05c → 07 → 08`.

**Run it with one command:** `CONDA_DEFAULT_ENV=nepa ./run_pipeline.py` (full `02→08`) or
`./run_pipeline.py --select` (selection-only: `05b → 05 → 05c → 08`, minutes). This is the single
orchestrator — `04b`/`05b`/`05c` are baked in (skipping them corrupts CE via stale `ranking_score`).

`04b` / `05b` / `05c` load `04` / `05` via `importlib` because digit-prefixed files can't be
imported normally — follow that pattern for any new sibling stage.

## Paths

- **Data** (gitignored): `phase2/data/analysis/timeline/` — packets, candidates, project_dates, `models/`
- **Labels (inputs):** `phase2/training/deliverable04/` — `classifier.csv`, `ranker.csv`, `frozen_eval_ids.txt`
- **Outputs:** `phase2/output/deliverable04/` — `diagnostics/`, `figures/`, `review_queues/`

## Tools (not pipeline stages)

- `run_pipeline.py` — the one-command pipeline orchestrator (full or `--select`)
- `_diagnostics.py` — classifier label-inventory / confusion / calibration diagnostics

## Sub-tracks

- `labeling/` — gold-label building sub-pipeline (feeds the classifier retrain); its own `01–05`

## Conventions

- Scripts hard-require `CONDA_DEFAULT_ENV=nepa` (run via `conda run -n nepa python …`).
- Never `pd.read_parquet()` the pages files — query via DuckDB with `document_id` filters.
- Isolated runs: `--sample-ids <file>` (02/03/04/05/07) or `--run-dir <dir>` (04b/05b); sample runs
  auto-isolate to `timeline/sample_runs/<stem>/`.
