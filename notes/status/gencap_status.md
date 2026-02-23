# Generation Capacity Workflow Status

Date: 2026-02-23 (updated)

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
- LLM pass restricted to ambiguous cases (candidate_count >= 2) by default and can run in parallel.
- LLM hardening: requires numeric source quotes, rejects no-numeric candidates, and falls back to extracting the **max numeric capacity** from candidate sentences when the LLM omits a quote (marks `extraction_method = fallback_from_candidates`).
- LLM can now be constrained to projects that already have regex capacity (`--require-regex-capacity`).
- Regex extraction now skips likely initials/date false positives (e.g., “MW, 5/21/15”).
- Added merge script to combine regex + LLM outputs into a single dataset.
- Added lightweight validation flags and audit sample generator for regex capacities (initials/date, non-generation context, non-build context, equipment lists).
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
python code/extract/extract_gencap.py --run regex
```

### 2) Run regex extraction on ALL projects
```bash
python code/extract/extract_gencap.py --run regex --all
```

### 3) Run regex extraction in parallel (single command)
```bash
python code/extract/extract_gencap.py --run regex --parallel 3
```
This runs CE/EA/EIS in parallel and automatically combines to:
`data/analysis/projects_gencap.parquet`.

### 4) Quick regex test sample
```bash
python code/extract/extract_gencap.py --run regex --sample 100
```

### 5) Count title/description hits and ambiguous cases for LLM
```bash
python - <<'PY'
import pandas as pd

df = pd.read_parquet("data/analysis/projects_gencap.parquet")
df = df[df["project_energy_type"] == "Clean"]

print("Clean energy total:", len(df))
print("Source counts:", df["project_gencap_source"].value_counts(dropna=False).to_dict())
ambiguous = df[df["project_gencap_candidate_count"].fillna(0) >= 2]
print("Ambiguous (>=2 candidates):", len(ambiguous))
print(ambiguous["dataset_source"].value_counts())
PY
```

### 6) Run LLM adjudication + merge (all sources, ambiguous only)
The `--run llm` command runs CE, EA, EIS sequentially and writes the final merged parquet.
```bash
python code/extract/extract_gencap.py --run llm --workers 4
```

### 7) LLM test run (sample, all sources)
```bash
python code/extract/extract_gencap.py --run llm --sample 10 --workers 1
```

### 8) LLM run including non-ambiguous projects
```bash
python code/extract/extract_gencap.py --run llm --include-non-ambiguous --workers 4
```

### 9) LLM run restricted to projects with existing regex capacity
```bash
python code/extract/extract_gencap.py --run llm --require-regex-capacity --workers 4
```

### 10) Stratified validation sample (30 per source)
```bash
python code/deliverable03/03_gencap_validation_sample.py --n 30
```

### 11) Generate validation flags + quick audit sample
```bash
python code/deliverable03/04_gencap_validation_flags.py
```
Outputs:
- `data/analysis/projects_gencap_flagged.parquet`
- `output/deliverable3/gencap_validation_quick_sample.csv`

Note: `05_gencap_merge_llm.py` has been removed. Merging is now integrated into `--run llm`
(via `run_llm_merge_pipeline()` in `extract_gencap_llm.py`).
Output: `data/analysis/projects_gencap_merged.parquet`

## Files updated
- `code/extract/extract_gencap.py`
- `code/extract/extract_gencap_llm.py`
- `code/utils/config.py`
- `code/deliverable03/02_capacity.R`
- `code/deliverable03/03_gencap_validation_sample.py`
- `code/deliverable03/05_gencap_merge_llm.py` — **deleted** (merge now integrated into `--run llm`)

## Notes
- Power/energy are now separated; update analysis logic accordingly (power only for capacity bins).
- LLM pass triggers on `project_gencap_candidate_count >= 2` (ambiguous cases) by default.
  Use `--include-non-ambiguous` to include all projects.
- `project_gencap_candidate_count` is only populated after a regex rerun — run `--run regex` first.
- Parallelization: use 4-6 workers on this machine; Ollama throughput will be the limiting factor.
- Description hits now always get confidence='high' (same as title hits).

## Where to pick up next
1) **Re-run regex extraction** (populates `project_gencap_candidate_count`; required for LLM trigger)
   ```bash
   python code/extract/extract_gencap.py --run regex --parallel 3
   ```
2) **Full LLM adjudication + merge** (all sources, ambiguous cases only)
   ```bash
   python code/extract/extract_gencap.py --run llm --workers 4
   ```
   Output: `data/analysis/projects_gencap_merged.parquet`
3) **Review validation flags / sample**
   Open: `output/deliverable3/gencap_validation_quick_sample.csv` and confirm which flags you want to filter.
4) **Update analysis in Deliverable 3**
   Point analysis to `data/analysis/projects_gencap_merged.parquet` (or `projects_gencap_flagged.parquet` if you want to filter first).
5) **DuckDB optimization** (next session)
   Replace per-document pyarrow reads in `extract_gencap_llm.py` with bulk DuckDB query (pattern from `extract_reviews.py:240-297`).

## Recent progress recap
- Regex extraction now skips initials/date false positives (e.g., “MW, 5/21/15”).\n
- LLM hardening: requires numeric source quotes, rejects no-numeric candidates, and uses `fallback_from_candidates` when LLM omits a quote.\n
- Candidate selection improved for hyphenated units and long equipment-list sentences with numeric units.\n
- LLM spot-checks:\n
  - 20-project regex-capacity CE sample: **18/20 (90%)** extracted.\n
  - Misses were false positives / list-style cases; equipment list fix now captures 50 kW example.\n
