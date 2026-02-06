# NEPA Project Analysis: Clean Energy Environmental Reviews

Analysis of clean energy projects using the National Environmental Policy Act Text Corpus (NEPATEC) 2.0 dataset from PNNL's PermitAI project.

## Project Website

**[https://www.kaseyzapatka.com/nepa/project_overview.html](https://www.kaseyzapatka.com/nepa/project_overview.html)**

## Data Source

This project is based on an analysis of [NEPATEC 2.0 on Hugging Face](https://huggingface.co/datasets/PNNL/NEPATEC2.0).

## Database Build

This pipeline produces analysis-ready parquet files in `data/analysis/`. The Federal Register NOI enrichment is generated separately and then merged into the projects output by `project_id`.

1. Generate Federal Register NOI enrichment (Clean EA + Clean EIS by default):

```bash
python code/extract/federal_register.py --sample 0 --report-n 10 --fetch-raw-text
```

2. Build the main dataset and merge the NOI fields into `projects_combined.parquet`:

```bash
python code/extract/extract_data.py --mode analysis
```

The merge uses `project_id` and drops `project_title` from the NOI output to avoid duplicate columns. The NOI output is expected at `data/analysis/noi_federal_register.parquet`.

Note for later updates: consider tracking a run log with query statistics for reproducibility.
