# Deliverable 06 Status (Technology-Specific Inquiries)

Date: 2026-02-13

## What Was Done

### 1) Extraction pipeline updates (Python)
Updated `code/extract/extract_data.py` to add Deliverable 06 technology fields directly in the core extraction output (`projects_combined.parquet`).

Added in `add_technology_columns()`:
- Flags:
  - `project_is_transmission_broad`
  - `project_is_transmission_strict`
  - `project_is_transmission` (alias of strict)
  - `project_is_geothermal`
  - `project_is_pipeline`
  - `project_is_carbon_pipeline`
  - `project_is_hydrogen_pipeline`
  - `project_is_natural_gas_pipeline`
- Geothermal:
  - `project_geothermal_phase`
- Transmission lengths:
  - `project_transmission_length_miles`
  - `project_transmission_length_confidence`
  - `project_transmission_length_source_text`
- Pipeline lengths:
  - `project_pipeline_length_miles`
  - `project_pipeline_length_confidence`
  - `project_pipeline_length_source_text`
- Pipeline grouping:
  - `project_pipeline_group`

Integrated this into `create_combined_projects()` so fields are generated during `--mode analysis`.

### 2) Strict transmission logic was added
To reduce false positives, transmission inclusion now requires:
1. `project_type` contains `Electricity Transmission`
2. explicit transmission line build text in title/description
3. extracted transmission length `>= 1` mile

This logic exists in Python and is reflected in analysis usage.

### 3) Deliverable 06 scripts were created
Created `code/deliverable06/`:
- `00_setup.R`
- `01_transmission.R`
- `02_geothermal.R`
- `03_pipelines.R`

### 4) Clean-energy-only enforcement
Deliverable 06 analysis is filtered to `project_energy_type == "Clean"`.

### 5) Removed R-side re-extraction for tech/length fields
Initially, R had fallback derivation for missing fields (this caused confusion and false positives).
This was removed.

Current behavior in `code/deliverable06/00_setup.R`:
- Load timeline metrics from `data/analysis/projects_timeline_bert.parquet`
- Load technology/length extraction fields from `data/analysis/projects_combined.parquet`
- Merge by `project_id`
- If extraction fields are missing, they remain `NA` (not re-derived in R)

### 6) Rebuilt extraction outputs
Ran:
- `python code/extract/extract_data.py --mode analysis`

Then re-ran Deliverable 06 scripts so outputs are based on merged Python extraction columns.

### 7) Transmission figures
`code/deliverable06/01_transmission.R` now generates:
1. `fig_transmission_length_vs_duration.png`
2. `fig_transmission_duration_by_region.png` (box plot)
3. `fig_transmission_start_vs_decision_lollipop.png`

## Why There Was Confusion

The timeline file (`projects_timeline_bert.parquet`) contains timeline fields but originally did not include new Deliverable 06 extraction columns.
R fallback logic was temporarily filling those fields from text, which made it look like R was redoing Python extraction.
This fallback is now removed for tech/length variables.

## Current Data Flow (Corrected)

1. Python extraction pipeline writes technology/length fields to `projects_combined.parquet`
2. Timeline extraction writes timeline fields to `projects_timeline_bert.parquet`
3. Deliverable 06 setup merges both by `project_id`
4. R scripts analyze merged dataset (no tech-length re-extraction in R)

## Output Locations

- Tables: `output/deliverable6/tables/`
- Figures: `output/deliverable6/figures/`

## What Still Needs To Be Done

1. QA validation pass for extracted lengths (transmission and pipeline)
- Build a sampled validation table including:
  - `project_id`
  - extracted length
  - confidence
  - source text snippet
  - optional manual review columns

2. Review strict transmission rule thresholds
- Current threshold `>= 1 mile` is intentionally conservative.
- Confirm with project goals whether this should remain 1 mile or be adjusted.

3. Report integration
- Add Deliverable 06 report document (similar structure to `reports/deliverable04.qmd`) and include the produced tables/figures.

## Useful Rerun Commands

Rebuild extraction outputs:
```bash
python code/extract/extract_data.py --mode analysis
```

Run Deliverable 06 scripts:
```bash
Rscript code/deliverable06/01_transmission.R
Rscript code/deliverable06/02_geothermal.R
Rscript code/deliverable06/03_pipelines.R
```

## Key Files Changed

- `code/extract/extract_data.py`
- `code/deliverable06/00_setup.R`
- `code/deliverable06/01_transmission.R`
- `code/deliverable06/02_geothermal.R`
- `code/deliverable06/03_pipelines.R`
- `notes/status/06_deliverable06.md` (this file)
