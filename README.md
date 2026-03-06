# NEPA Project Analysis: Clean Energy Environmental Reviews

Analysis of clean energy projects using the National Environmental Policy Act Text Corpus (NEPATEC) 2.0 dataset from PNNL's PermitAI project.

## Project Website

**[https://www.kaseyzapatka.com/nepa/project_overview.html](https://www.kaseyzapatka.com/nepa/project_overview.html)**

## Data Source

This project is based on an analysis of [NEPATEC 2.0 on Hugging Face](https://huggingface.co/datasets/PNNL/NEPATEC2.0).

## Reproducible Environment

Use the project-standard conda environment:

```bash
conda env create -f environment.yml
conda activate nepa
```

If you already created it and need to sync to the latest spec:

```bash
conda env update -n nepa -f environment.yml --prune
```

Environment design notes and dependency rationale:
`notes/architecture/environment_setup.md`

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

## Deliverable 4 Multi-Agency Refresh (Simple)

Run these three commands when you want the latest multi-agency outputs for Deliverable 4:

```bash
python code/extract/extract_coagency.py --run
Rscript code/deliverable04/01_geography.R
quarto render reports/deliverable04.qmd
```

What this does:
- Builds `data/analysis/coagency_projects.parquet` from page-text cues (`extract_coagency.py`).
- Rebuilds Deliverable 4 tables/figures (including strict vs expanded multi-agency outputs).
- Renders the updated Deliverable 4 report.

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

## Targeted Re-Adjudication for Programmatic & Tiered Reviews (Deliverable 2)

Run this when the full EA/EIS LLM adjudication is complete but some programmatic or tiered
projects are still missing initiation or decision dates. This re-runs adjudication only on those
incomplete projects, with a higher candidate cap, ROD-language promotion, and a 15-year date
window to cut noise.

**Prerequisites:** `projects_timeline_bert_ea_llm.parquet` and
`projects_timeline_bert_eis_llm.parquet` must already exist in `data/analysis/`.

```bash
export ANTHROPIC_API_KEY='sk-ant-...'

python code/extract/extract_timeline.py \
  --llm-adjudicate \
  --input data/analysis/projects_timeline_bert_ea_llm.parquet,data/analysis/projects_timeline_bert_eis_llm.parquet \
  --nonstandard-incomplete \
  --max-candidates 125 \
  --context-chars 400 \
  --promote-rod-language \
  --year-window 15 \
  --provider claude \
  --output data/analysis/projects_timeline_targeted_llm.parquet
```

What this does:
- `--nonstandard-incomplete` — auto-selects only programmatic/tiered projects with missing dates (~73 projects). No manual ID file needed.
- `--max-candidates 125` — raises the candidate cap from 30 (EIS default) to 125, so large programmatic EISs get adequate coverage.
- `--promote-rod-language` — promotes dates with ROD/FONSI language to Tier A even if BERT mislabeled them.
- `--year-window 15` — drops candidate dates more than 15 years before the latest date found, removing NEPA citation noise.
- Output is a small targeted parquet (~73 rows). The full timeline files are not modified.

**Cost:** ~$0.44 (Haiku, ~400K input tokens).

**After the run**, the targeted dates are automatically patched into the Deliverable 2 analysis
when you run `00_setup.R` — no further changes needed.

---

## Technology Deliverables Build

Use `code/extract/extract_technology.py` to build technology-specific fields.

**Output:**
- `data/analysis/projects_transmission.parquet` — all transmission columns (flags, lengths, LLM audit, timestamps)

Reads `data/analysis/projects_combined.parquet` for project metadata but does **not** modify it.

### Transmission

**Step 1 — Rule-based extraction + page-level length recovery (~5–15 min, no API cost):**

```bash
python code/extract/extract_technology.py --run transmission
```

**Step 2 — LLM adjudication for ambiguous multi-candidate rows (~$0.06, ~2 min with 4 workers):**

Requires `ANTHROPIC_API_KEY` in environment. Run Step 1 first.

```bash
export ANTHROPIC_API_KEY='sk-ant-...'

python code/extract/extract_technology.py --run llm --workers 4
```

**Verify the output:**

```python
import pandas as pd
tx = pd.read_parquet('data/analysis/projects_transmission.parquet')
print(f"Rows: {len(tx)}, Columns: {len(tx.columns)}")
print(f"Extraction built: {tx['project_tx_extraction_run_at'].iloc[0]}")
llm_rows = tx[tx['project_tx_llm_run_at'] != '']
print(f"LLM rows: {len(llm_rows)}, model: {llm_rows['project_transmission_length_llm_model'].iloc[0] if len(llm_rows) else 'none'}")
print(f"Strict projects: {tx['project_is_transmission'].sum()}")
```

**After rebuilding**, re-run the Deliverable 6 R scripts:

```bash
Rscript code/deliverable06/01_transmission.R
```

### Geothermal

```bash
python code/extract/extract_technology.py --run geothermal
```

### Pipelines

```bash
python code/extract/extract_technology.py --run pipeline
```

## Deliverable 5: Regulatory Page Count Extraction

`code/extract/extract_pages.py` estimates FRA-compliant page counts for clean energy EA and EIS final documents. The FRA defines a "page" as 500 words and excludes maps, figures, and appendices (40 C.F.R. § 1508.1(bb)). This script computes `regulatory_pages = CEIL(body_word_count / 500)` by detecting embedded appendix sections and excluding low-content pages. Output is written to `data/analysis/projects_page_counts.parquet` and joined into the Deliverable 5 R pipeline by `code/deliverable05/00_setup.R`.

Re-run this script whenever `projects_combined.parquet` is updated (new projects added).

### Full production run

```bash
python code/extract/extract_pages.py --run
```
---

## Generation Capacity Build

`code/extract/extract_gencap.py` extracts generation capacity (MW/GW/kW) in two phases: regex over all projects, then Claude Haiku adjudication for projects with 2+ distinct candidates.

### Phase 1: Regex (parallel)

```bash
python code/extract/extract_gencap.py --run regex --parallel 3
```

Output: `data/analysis/projects_gencap.parquet`

### Phase 2: LLM adjudication

```bash
python code/extract/extract_gencap.py --run llm --workers 4 # run with 2 to avoid rate limits
```

Runs on ambiguous multi-candidate projects only. Updates `projects_gencap.parquet` in place and writes per-source raw outputs to `data/analysis/gencap_{ce,ea,eis}_llm.parquet`.

```bash
# Test on 10 projects first
python code/extract/extract_gencap.py --run llm --sample 10 --workers 1

# Debug a single project
python code/extract/extract_gencap.py --run llm --project-id <UUID>
```
