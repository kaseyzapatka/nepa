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

## Timeline Data Build (Regex + BERT)

Use this after updating timeline patterns or models in `code/extract/extract_timeline.py`.

1. Build regex cache (uses updated context rules):

```bash
python code/extract/extract_timeline.py --regex-prep
```

2. Generate BERT training data (uses updated strong/med/weak patterns):

```bash
python code/extract/extract_timeline.py --bert-generate
```

3. Train the BERT classifier:

```bash
python code/extract/extract_timeline.py --bert-train
```

4. Run a small test sample (sanity check):

```bash
python code/extract/extract_timeline.py --bert-run --sample 50 --output test50_bert_v9.parquet
```

5. Full run to update timeline output:

```bash
python code/extract/extract_timeline.py --bert-run --output projects_timeline_bert.parquet
```
