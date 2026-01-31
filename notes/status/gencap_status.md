# Generation Capacity Workflow Status

Date: 2026-01-30

## Purpose
Capture the current state of generation-capacity extraction work for Deliverable 3, summarize key findings on transmission/utilities prevalence in clean energy projects, and provide a runbook for rerunning the workflow.

## Summary of changes implemented
- Title-first extraction: scan `project_title` before documents; if a capacity is found, mark the project as done and skip document scanning.
- Power vs energy split: power (MW/GW/kW) is tracked separately from energy (MWh/GWh/kWh) with new fields.
- Expanded unit patterns: MWac/MWdc/MWe/MWt/MWth/MWp, kWe, and hyphenated forms; also handles ranges and "up to" expressions.
- Invalid match filtering: filters MW-year and $/MW style matches.
- Confidence fields added: `project_gencap_confidence` (high/medium/low) and `project_gencap_context` (local snippet).
- Transmission-only gating has been removed; transmission-only projects are included in the regex pass.
- LLM model default set to `llama3.2:3b-instruct-q4_K_M`.
- LLM pass restricted to low/medium confidence cases by default and can run in parallel.
- LLM hardening: requires numeric source quotes, rejects no-numeric candidates, and falls back to extracting the **max numeric capacity** from candidate sentences when the LLM omits a quote (marks `extraction_method = fallback_from_candidates`).
- LLM can now be constrained to projects that already have regex capacity (`--require-regex-capacity`).
- Stratified validation sample script added.

## Transmission/utilities prevalence (clean energy only)
Clean energy projects: 22,279 total.

Overall:
- Any “Electricity Transmission” in project_type: 7,815 (35.1%)
- Transmission-only (strict: only Electricity Transmission + Utilities): 1,531 (6.9%)
- Transmission-only (relaxed: also allow Broadband): 1,784 (8.0%)
- Utilities-only (only Utilities, no transmission): 488 (2.2%)

By dataset source (percent of clean energy in each source):
- CE: any transmission 34.3%, transmission-only strict 7.0%, relaxed 8.0%, utilities-only 2.3%
- EA: any transmission 43.2%, transmission-only strict 5.6%, relaxed 6.1%, utilities-only 1.6%
- EIS: any transmission 49.1%, transmission-only strict 1.5%, relaxed 2.0%, utilities-only 0.6%

## Baseline validation (manual sample)
- Sample file created: `output/deliverable3/gencap_manual_validation_sample.csv`
- 20 clean-energy projects with non-null extracted capacity were sampled.
- 11/20 have a text snippet captured (match found within first 10 pages of the main doc). The remaining 9 are marked N/A for manual verification due to missing snippets.
- Baseline precision on verifiable rows: 11/11 = 100% (limited to rows with snippets).
- Main uncertainty noted: whether some matches clearly refer to the proposed project vs. another project (several notes flagged this).

## LLM spot-check (regex-capacity sample)
- 10-project CE sample restricted to **regex-capacity cases** resulted in 9/10 capacity extractions (90%).
- Methods: 5 `llm`, 4 `fallback_from_candidates`, 1 `no_candidates`.
- The only miss was a false-positive regex match (initials/date “MW”).
- Example fix: Barr‑Tech case now returns **2.2 MW** via fallback.

## Gating status (extract_gencap.py)
Transmission-only gating has been removed. All projects, including transmission-only, are scanned by the regex pipeline.

## Outputs and new fields (regex pipeline)
`data/analysis/projects_gencap.parquet` now includes:
- `project_gencap_value`, `project_gencap_unit` (power only)
- `project_gencap_energy_value`, `project_gencap_energy_unit` (energy only)
- `project_gencap_source` (title/document/none/skipped_transmission_only)
- `project_gencap_confidence` (high/medium/low)
- `project_gencap_context` (local text snippet)

## Runbook (commands)

### 1) Run regex extraction (clean energy only)
```bash
python code/extract/extract_gencap.py --run
```

### 2) Run regex extraction on ALL projects
```bash
python code/extract/extract_gencap.py --run --all
```

### 3) Run regex extraction in parallel (single command)
```bash
python code/extract/extract_gencap.py --run --parallel 3
```
This runs CE/EA/EIS in parallel and automatically combines to:
`data/analysis/projects_gencap.parquet`.

### 4) Run a single dataset source (optional)
```bash
python code/extract/extract_gencap.py --run --source ce
python code/extract/extract_gencap.py --run --source ea
python code/extract/extract_gencap.py --run --source eis
```

### 5) Quick regex test sample
```bash
python code/extract/extract_gencap.py --run --sample 100
```

### 6) Count title hits, transmission skips, and low/medium cases for LLM
```bash
python - <<'PY'
import pandas as pd

df = pd.read_parquet("data/analysis/projects_gencap.parquet")
df = df[df["project_energy_type"] == "Clean"]

title_hits = (df["project_gencap_source"] == "title").sum()

conf = df["project_gencap_confidence"].fillna("low").str.lower()
low_med = df[conf.isin(["low","medium"])].copy()
low_med = low_med[~low_med["project_gencap_source"].isin(["title"])]

print("Clean energy total:", len(df))
print("Title hits:", title_hits)
print("Low/medium to LLM:", len(low_med))
print(low_med["dataset_source"].value_counts())
PY
```

### 7) Run LLM extraction (low/medium only, parallel workers)
```bash
python code/extract/extract_gencap_llm.py \
  --source ce \
  --run \
  --regex-results data/analysis/projects_gencap.parquet \
  --workers 4
```

### 8) Include high-confidence cases in LLM pass (optional)
```bash
python code/extract/extract_gencap_llm.py \
  --source ce \
  --run \
  --regex-results data/analysis/projects_gencap.parquet \
  --include-high \
  --workers 4
```

### 9) LLM test run using regex-capacity cases only
```bash
python code/extract/extract_gencap_llm.py \
  --source ce \
  --run \
  --sample 10 \
  --regex-results data/analysis/projects_gencap.parquet \
  --require-regex-capacity \
  --workers 1
```

### 10) Stratified validation sample (30 per source)
```bash
python code/deliverable3/03_gencap_validation_sample.py --n 30
```
Output: `output/deliverable3/gencap_validation_stratified_sample.csv`

## Files updated
- `code/extract/extract_gencap.py`
- `code/extract/extract_gencap_llm.py`
- `code/utils/config.py`
- `code/deliverable3/02_capacity.R`
- `code/deliverable3/03_gencap_validation_sample.py`

## Notes
- Power/energy are now separated; update analysis logic accordingly (power only for capacity bins).
- LLM pass is restricted to low/medium confidence by default to save time and cost.
- Parallelization: use 4-6 workers on this machine; Ollama throughput will be the limiting factor.
