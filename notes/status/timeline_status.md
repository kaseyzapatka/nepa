# Timeline Extraction Status (Prompt-Ready)

**Last updated:** 2026-02-18  
**Source of truth:** `code/extract/extract_timeline.py`

Use this file as the canonical, up-to-date context for timeline extraction. If older notes conflict, trust the code and this document.

## Diff Checklist (Fast Read)

- Sources supported in current run: `CE`, `EA`, `EIS` via `--source`.
- Canonical workflow: `--regex-prep` -> `--bert-run` -> `--llm-adjudicate`.
- `--regex-prep` writes per-source caches: `regex_candidates_ce/ea/eis.parquet`.
- `--bert-run` is clean-energy only in CLI dispatch (`clean_energy_only=True`).
- EIS gap rule is OFF; CE/EA gap rule is ON (730 days).
- EA/EIS can use non-main-doc fallback when no main docs exist; flag is `main_document_imputed`.
- LLM adjudication reads BERT candidates (`bert_dates_json`) only; no new date discovery.
- Decision modes: `priority_only`, `ea_eis_fallback`, `no_decision_candidates`.
- Guardrail strictness applies only to `priority_only`.
- Adjudication caps: default `50/300`; EIS `30/200` (max candidates/context chars).
- Claude provider forces `workers=1`; currently retries `429` only (not `529/5xx`).
- Default adjudication output naming: `<input_stem>_llm.parquet`.

## Scope

This pipeline extracts NEPA timeline dates from document text for **clean-energy** projects across:
- `CE`
- `EA`
- `EIS`

Primary workflow is:
1. Regex candidate extraction (`--regex-prep`)
2. BERT date classification + timeline assembly (`--bert-run`)
3. Optional LLM post-adjudication (`--llm-adjudicate`)

## Current Pipeline (Authoritative)

### 1) Regex candidate extraction

Command entrypoint: `--regex-prep`  
Function: `run_regex_prep()`

Key behavior:
- Parses source(s) from `--source` (`CE`, `EA`, `EIS`, comma-separated).
- Filters to clean energy in CLI dispatch (`clean_energy_only=True`).
- Saves **per-source cache** by default:
  - `data/analysis/regex_candidates_ce.parquet`
  - `data/analysis/regex_candidates_ea.parquet`
  - `data/analysis/regex_candidates_eis.parquet`
- Uses `document_type_clean` when available, else `document_type`.
- Candidate rows include:
  - `project_id`, `date`, `match`, `context`, `position`, `position_pct`, `doc_type`, `main_document_imputed`, `run_timestamp`

Main-document logic:
- `CE`: main-doc-only behavior enforced.
- `EA/EIS`: main-doc-first; if a project has no main docs, falls back to non-main docs sorted by source-specific doc priority; those rows set `main_document_imputed=True`.

### 2) BERT timeline extraction

Command entrypoint: `--bert-run`  
Function: `run_bert_timeline_extraction()`

Key behavior:
- Filters project universe by `dataset_source in --source`.
- Filters to clean energy in CLI dispatch (`clean_energy_only=True`).
- Requires per-source regex caches (CE legacy fallback to `regex_candidates.parquet` is kept).
- Applies per-source gap rule config:
  - `CE`: gap rule on (730 days)
  - `EA`: gap rule on (730 days)
  - `EIS`: gap rule off
- Adds NOI fallback for initiation when no initiation date exists.
- Carries filename-derived metadata bounds:
  - `project_date_earliest_file_name`
  - `project_date_latest_file_name`
  - `project_file_name_dates`
- Carries `main_document_imputed` into project-level output.

Timeline status values:
- `complete`
- `missing_initiation`
- `missing_decision`
- `no_dates`

Default output:
- `data/analysis/projects_timeline_bert.parquet`
- or custom via `--output`.

### 3) LLM adjudication (post-BERT)

Command entrypoint: `--llm-adjudicate --input <bert_output.parquet>`  
Function: `run_llm_adjudication()`

What it does:
- Reads BERT output parquet (`bert_dates_json` per project).
- Filters candidate dates (`_filter_candidates_for_llm`).
- Prompts LLM to pick one initiation and one decision date.
- Writes LLM picks plus diagnostics back to parquet.

Providers:
- `--provider ollama`
- `--provider claude` (default model for Claude provider: `claude-haiku-4-5-20251001`)

Claude-specific behavior:
- Requires `ANTHROPIC_API_KEY` env var.
- Forces `workers=1`.
- Retries `429` only (does **not** currently retry `529` overload or other 5xx).

Decision modes in adjudication:
- `priority_only` (Tier A docs present: FONSI/ROD/DR/Decision Record)
- `ea_eis_fallback` (Tier A absent, vetted fallback decision candidates)
- `no_decision_candidates`

Guardrail:
- Hard post-parse guardrail is only enforced for `priority_only`.

Cap/context limits:
- Default (non-EIS): `max_candidates=50`, `context_chars=300`
- EIS: `max_candidates=30`, `context_chars=200`

Default output naming:
- `<input_stem>_llm.parquet` in same folder unless `--output` is provided.

## Training Pipeline (BERT)

### Generate training data

Command: `--bert-generate`  
Function: `generate_bert_training_data()`

Behavior:
- Auto-discovers available regex caches across CE/EA/EIS.
- Falls back to legacy CE cache only if needed.
- Auto-labels contexts via weak supervision.
- Applies manual corrections if `data/analysis/manual_training_corrections.csv` exists.
- Oversamples EA and EIS by 3x to reduce CE dominance.
- Output: `data/analysis/bert_training_data.parquet`.

### Train classifier

Command: `--bert-train`  
Function: `train_bert_classifier()`

Behavior:
- Trains classifier from `bert_training_data.parquet`.
- Saves model to `models/timeline_classifier/`.

## Canonical Commands

### Multi-source full workflow (recommended)

```bash
python code/extract/extract_timeline.py --regex-prep --source CE
python code/extract/extract_timeline.py --regex-prep --source EA
python code/extract/extract_timeline.py --regex-prep --source EIS

python code/extract/extract_timeline.py --bert-run --source CE,EA,EIS --output projects_timeline_bert_all.parquet

python code/extract/extract_timeline.py --llm-adjudicate --input projects_timeline_bert_all.parquet --provider claude --output projects_timeline_bert_all_llm.parquet
```

### EA-only workflow

```bash
python code/extract/extract_timeline.py --regex-prep --source EA
python code/extract/extract_timeline.py --bert-run --source EA --output projects_timeline_bert_ea.parquet
python code/extract/extract_timeline.py --llm-adjudicate --input projects_timeline_bert_ea.parquet --provider claude
```

### EIS-only workflow

```bash
python code/extract/extract_timeline.py --regex-prep --source EIS
python code/extract/extract_timeline.py --bert-run --source EIS --output projects_timeline_bert_eis.parquet
python code/extract/extract_timeline.py --llm-adjudicate --input projects_timeline_bert_eis.parquet --provider claude
```

## Important Caveats

1. `--llm-adjudicate` can only choose among BERT-provided candidates (`bert_dates_json`). It cannot recover dates never extracted upstream.
2. EIS adjudication can still fail with `claude_api_error (529): Overloaded`; current code retries 429 only.
3. `--llm-run` path is legacy CE-oriented full-document/hybrid LLM flow; current EA/EIS production path should use `--llm-adjudicate`.
4. If you pass one shared `--regex-cache` filename while processing multiple sources in one `--regex-prep` call, later sources can overwrite earlier output. Prefer default per-source cache names.

## NOTES

When editing timeline tasks:
1. Assume this file and `code/extract/extract_timeline.py` are authoritative.
2. Use the regex -> BERT -> LLM-adjudicate mental model.
3. Preserve per-source behavior differences (especially EIS gap-rule off and EIS adjudication limits 30/200).
4. Do not infer behavior from older notes unless verified in code.
5. When proposing edits, state which stage is impacted: regex cache, BERT classification, or LLM adjudication.

## Quick Glossary

- **Tier A decision docs:** `FONSI`, `ROD`, `DR`, `DECISION RECORD`
- **Fallback decision docs:** `EA`, `DEA`, `DEIS`, `FEIS`, `FEA`, `EIS`
- **Imputed main-doc flag:** `main_document_imputed=True` means EA/EIS fallback used non-main docs because no main docs existed.
