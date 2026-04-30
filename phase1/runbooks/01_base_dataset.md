# Base Dataset Build

**Purpose:** Build the primary analysis-ready dataset that all downstream extractions depend on.
**Input:** NEPATEC 2.0 raw data.
**Output:** `data/analysis/projects_combined.parquet`, `data/analysis/noi_federal_register.parquet`
**Prerequisites:** Environment set up ([runbook 00](00_environment.md)), NEPATEC 2.0 data available.

## Step 1 — Federal Register NOI enrichment

Fetches Notice of Intent records for Clean EA and Clean EIS projects from the Federal Register API.

```bash
python code/extract/federal_register.py --sample 0 --report-n 10 --fetch-raw-text
```

Output: `data/analysis/noi_federal_register.parquet`

## Step 2 — Build main dataset

Builds `projects_combined.parquet` and merges the NOI fields into it by `project_id`.

```bash
python code/extract/extract_data.py --mode analysis
```

Output: `data/analysis/projects_combined.parquet`

## Notes

- The merge drops `project_title` from the NOI output to avoid duplicate columns.
- NOI enrichment is expected at `data/analysis/noi_federal_register.parquet` before step 2 runs.
- Re-run both steps whenever the underlying NEPATEC data is refreshed.
