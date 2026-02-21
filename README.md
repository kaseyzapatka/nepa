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

## Review Type Extraction (Programmatic + Tiered)

Use `code/extract/extract_reviews.py` to classify projects as `programmatic`, `tiered`, or `standard`.

The extractor now uses DuckDB for source-level page loading (both EA and EIS), then classifies projects from in-memory caches. `--workers` parallelizes project classification on top of that.
`generic` / `tier 1` stand-in terminology is included by default.
Scope is fixed to clean energy EA/EIS projects.

### Full production run (recommended)

Default scope is clean energy + EA/EIS only. Output writes to `data/analysis/projects_reviews.parquet`.

```bash
python code/extract/extract_reviews.py --run --workers 8
```

### Optional full run with LLM fallback (slower)

```bash
python code/extract/extract_reviews.py --run --use-llm --workers 8
```

Use `--use-llm` when:
- You want higher recall on borderline/ambiguous phrasing that regex scores as medium confidence.
- You are doing a focused QA pass on likely edge cases, not routine production refreshes.
- You can tolerate slower runtime and model-driven variability in classifications.

### Test run (safe output)

Writes to `data/analysis/projects_reviews_test.parquet` so the main output is not overwritten.

```bash
python code/extract/extract_reviews.py --test --workers 4
```

## Timeline Data Build (Regex + BERT)

Use this after updating timeline patterns or models in `code/extract/extract_timeline.py`.

The `--source` flag controls which NEPA process types to run. Accepts `CE`, `EA`, `EIS`, or comma-separated combinations. Defaults to `CE` if omitted.

### CE (Categorical Exclusions) — default

```bash
# 1. Build regex cache
python code/extract/extract_timeline.py --regex-prep

# 2. Generate BERT training data
python code/extract/extract_timeline.py --bert-generate

# 3. Train the BERT classifier
python code/extract/extract_timeline.py --bert-train

# 4. Test sample (optional)
python code/extract/extract_timeline.py --bert-run --sample 50 --output test50_bert_vX.parquet

# 5. Full run
python code/extract/extract_timeline.py --bert-run --output projects_timeline_bert.parquet
```

### EA (Environmental Assessments)

```bash
# 1. Build EA regex cache
python code/extract/extract_timeline.py --regex-prep --source EA

# 2. Test sample (BERT)
python code/extract/extract_timeline.py --bert-run --source EA --sample 50 --output test50_ea.parquet

# 3. LLM adjudication (Claude) on BERT output
python code/extract/extract_timeline.py --llm-adjudicate --input test50_ea.parquet --provider claude

# 4. Full run (BERT + Claude adjudication)
python code/extract/extract_timeline.py --bert-run --source EA --output projects_timeline_bert_ea.parquet
python code/extract/extract_timeline.py --llm-adjudicate --input projects_timeline_bert_ea.parquet --provider claude
```

### EIS (Environmental Impact Statements)

```bash
# 1. Build EIS regex cache
python code/extract/extract_timeline.py --regex-prep --source EIS

# 2. Test sample (BERT)
python code/extract/extract_timeline.py --bert-run --source EIS --sample 50 --output test50_eis.parquet

# 3. LLM adjudication (Claude) on BERT output
python code/extract/extract_timeline.py --llm-adjudicate --input test50_eis.parquet --provider claude

# 4. Full run (BERT + Claude adjudication)
python code/extract/extract_timeline.py --bert-run --source EIS --output projects_timeline_bert_eis.parquet
python code/extract/extract_timeline.py --llm-adjudicate --input projects_timeline_bert_eis.parquet --provider claude
```

### Multi-source runs

```bash
# Build regex caches per source
python code/extract/extract_timeline.py --regex-prep --source CE
python code/extract/extract_timeline.py --regex-prep --source EA
python code/extract/extract_timeline.py --regex-prep --source EIS

# Run BERT across all three sources
python code/extract/extract_timeline.py --bert-run --source CE,EA,EIS --output projects_timeline_bert_all.parquet

# Run Claude adjudication on combined BERT output
python code/extract/extract_timeline.py --llm-adjudicate --input projects_timeline_bert_all.parquet --provider claude --output projects_timeline_bert_all_llm.parquet
```

### Retraining with EA/EIS data

After building regex caches for EA and/or EIS, `--bert-generate` auto-discovers all available per-source caches and includes them in training data (with 3x oversampling for EA/EIS to prevent CE domination):

```bash
python code/extract/extract_timeline.py --bert-generate
python code/extract/extract_timeline.py --bert-train
```

### Filtering by energy type

Add `--clean-energy` to restrict to clean energy projects only:

```bash
python code/extract/extract_timeline.py --bert-run --source EA --clean-energy --output projects_timeline_bert_ea_clean.parquet
```

### Single project debugging

The `--project-id` flag searches CE, EA, and EIS sources automatically:

```bash
python code/extract/extract_timeline.py --project-id <UUID> --hybrid --use-regex-cache
```

## Technology Deliverables Build

Use `code/extract/extract_technology.py` to build technology-specific fields in `data/analysis/projects_combined.parquet`.

### Transmission

Rule-based extraction only:

```bash
python3 code/extract/extract_technology.py --run transmission
```

Run transmission with LLM adjudication (Anthropic/Claude):

```bash
python code/extract/extract_technology.py --run transmission --use-llm --provider anthropic
```

If using Anthropic, set `ANTHROPIC_API_KEY` in your environment before running.

### Geothermal

```bash
# TODO: add final geothermal run sequence for Deliverable 6
python3 code/extract/extract_technology.py --run geothermal
```

### Pipelines

```bash
# TODO: add final pipeline run sequence for Deliverable 6
python3 code/extract/extract_technology.py --run pipeline
```
