# Phase 1: NEPA Clean Energy Analysis

Phase 1 is the foundational analysis of clean energy NEPA reviews using the NEPATEC 2.0 dataset. Status: **complete**.

## Key facts

| | Phase 1 |
|---|---|
| Status | Complete |
| Data pipeline | Pandas-based |
| Timeline extraction | BERT classifier |
| Output location | `phase1/data/analysis/` |
| Clean energy projects | 20,725 |
| Deliverables | D1–D6 complete |

## Structure

```
phase1/
├── code/
│   ├── extract/             # Core Python extraction scripts (timeline, reviews, gencap, etc.)
│   ├── deliverable01–06/    # Per-deliverable R analysis scripts
│   ├── _project_overview/   # Project overview scripts
│   ├── exploratory/         # Exploratory analysis
│   ├── utils/               # Shared utilities (config.py, R helpers)
│   ├── validation/          # QC and validation scripts
│   └── rag/                 # Document retrieval infrastructure
├── data/
│   ├── analysis/            # Primary analysis outputs (parquet files)
│   ├── processed/           # Per-source preprocessed documents (ce/, ea/, eis/)
│   ├── models/              # Geothermal classifier checkpoints
│   └── rag/                 # DuckDB text store
├── models/                  # BERT timeline classifier checkpoints
├── notes/                   # Status files, architecture notes, running todo
├── output/
│   └── deliverable{1–6}/    # Figures, tables, maps per deliverable
├── reports/                 # Quarto deliverable reports (.qmd)
└── runbooks/                # Step-by-step pipeline documentation (00–08)
```

## Environment

Phase 1 uses the shared conda environment:

```bash
conda activate nepa
```

See [runbook 00](runbooks/00_environment.md) to create or update the environment.

## Rebuilding Phase 1 data

Run runbooks in order. Each runbook documents prerequisites, full commands, and output files.

| Step | Runbook | Purpose | Primary output |
|------|---------|---------|----------------|
| 0 | [00_environment.md](runbooks/00_environment.md) | Create/update conda env | conda env `nepa` |
| 1 | [01_base_dataset.md](runbooks/01_base_dataset.md) | Build primary dataset from NEPATEC 2.0 | `projects_combined.parquet` |
| 2 | [02_timeline.md](runbooks/02_timeline.md) | Extract initiation + decision dates (BERT + LLM) | `projects_timeline_bert.parquet`, `*_ea_llm.parquet`, `*_eis_llm.parquet` |
| 3 | [03_reviews.md](runbooks/03_reviews.md) | Classify EA/EIS as programmatic, tiered, or standard | `projects_reviews.parquet` |
| 4 | [04_gencap.md](runbooks/04_gencap.md) | Extract generation capacity (MW/GW/kW) | `projects_gencap_merged.parquet` |
| 5 | [05_page_counts.md](runbooks/05_page_counts.md) | Estimate FRA-compliant regulatory page counts | `projects_page_counts.parquet` |
| 6 | [06_technology.md](runbooks/06_technology.md) | Classify transmission, geothermal phase, pipelines | `projects_transmission.parquet`, etc. |
| 7 | [07_geography.md](runbooks/07_geography.md) | Identify multi-agency reviews + geography outputs | `coagency_projects.parquet` |
| 8 | [app/runbook.md](../app/runbook.md) (moved to `app/`) | Build DuckDB text store + deploy to HF Spaces | `data/rag/nepa_reader.duckdb` |

## Rendering reports

From the `phase1/` directory:

```bash
# Render all deliverable reports
quarto render

# Render a single deliverable
quarto render reports/deliverable03.qmd

# Render Key Insights to Word
quarto render reports/key_insights.qmd --to docx
```

Output: `reports/key_insights.docx`

## Key output files

| File (`data/analysis/`) | Description |
|---|---|
| `projects_combined.parquet` | Base dataset — 20,725 clean energy projects |
| `projects_timeline_bert.parquet` | Timeline dates for CE projects |
| `projects_timeline_bert_ea_llm.parquet` | Timeline dates for EA projects (LLM-adjudicated) |
| `projects_timeline_bert_eis_llm.parquet` | Timeline dates for EIS projects (LLM-adjudicated) |
| `projects_timeline_targeted_llm.parquet` | Targeted re-adjudication for programmatic/tiered projects |
| `projects_reviews.parquet` | Programmatic / tiered / standard classification |
| `projects_gencap_merged.parquet` | Generation capacity (regex + LLM adjudication) |
| `projects_page_counts.parquet` | FRA-compliant regulatory page counts |
| `coagency_projects.parquet` | Multi-agency / cooperating agency classification |
| `projects_transmission.parquet` | Transmission-specific fields |

## Relationship to Phase 2

The two phases are independent. Phase 2 builds its own base tables by running its own copy of `extract_data.py` over the same processed NEPATEC corpus, so the project universe matches (61,881 rows in each `projects_combined.parquet`) while the schemas differ — Phase 1 has 97 columns, Phase 2 has 77. Neither table is derived from the other, and no Phase 2 script reads from or writes to `phase1/`.

To reproduce Phase 1, work from the current repository under `phase1/` and follow the runbooks in order. The tags `freeze/phase1_v1.0` and `checkpoint/phase1-cleanup` remain in the history as markers but predate the repository reorganization, so they cannot be used for replication.

See [phase2/README.md](../phase2/README.md) for the Phase 2 pipeline.
