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
| 06  | `06_adjudicate_llm.py` | **NEEDS REBUILD** — LLM adjudication of ambiguous picks | project_dates |
| 07  | `07_validate.py` | compare selected dates to the gold sample | validation reports |
| 08  | `08_analyze.R` | coverage tables, durations, figures | `diagnostics/` + `figures/` |

**Canonical run sequence:** `02 → 03 → 04 → 04b --apply → 05b --apply → 05 → 05c → 07 → 08`.

`04b` / `05b` / `05c` load `04` / `05` via `importlib` because digit-prefixed files can't be
imported normally — follow that pattern for any new sibling stage.

> **Orchestrator caveat:** `_run.py` currently runs `00b → 01 → 02 → 03 → 04 → 05 → 06` and
> **omits `04b` / `05b` / `05c`** — run those manually for now (see `clean_up_plan.md` #2).

## Paths

- **Data** (gitignored): `phase2/data/analysis/timeline/` — packets, candidates, project_dates, `models/`
- **Labels (inputs):** `phase2/training/deliverable04/` — `classifier.csv`, `ranker.csv`, `frozen_eval_ids.txt`
- **Outputs:** `phase2/output/deliverable04/` — `diagnostics/`, `figures/`, `review_queues/`

## Tools (not pipeline stages)

- `_diagnostics.py` — classifier label-inventory / confusion / calibration diagnostics
- `_phase0_baseline.py` — point-in-time baseline + source-ceiling audit for an improvement cycle
- `_run.py` — orchestrator (see caveat above)

## Sub-tracks

- `labeling/` — gold-label building sub-pipeline (feeds the classifier retrain); its own `01–05`
- `_archived/` — completed one-off scripts, kept in git history

## Conventions

- Scripts hard-require `CONDA_DEFAULT_ENV=nepa` (env python: `/opt/anaconda3/envs/nepa/bin/python`).
- Never `pd.read_parquet()` the pages files — query via DuckDB with `document_id` filters.
- Isolated runs: `--sample-ids <file>` (02/03/04/05/07) or `--run-dir <dir>` (04b/05b); sample runs
  auto-isolate to `timeline/sample_runs/<stem>/`.
