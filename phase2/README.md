# Phase 2: Extended NEPA Analysis

Phase 2 extends the Phase 1 NEPA analysis with an improved data pipeline and deeper extraction capabilities.

## Key differences from Phase 1

| | Phase 1 | Phase 2 |
|---|---|---|
| Data pipeline | Pandas-based | DuckDB-based |
| Timeline extraction | BERT classifier | Improved BERT + LLM hybrid adjudication |
| Output location | `data/analysis/` (frozen) | `phase2/data/` |
| Status | Frozen at `freeze/v1.0` | In progress |

## Structure

```
phase2/
├── code/
│   ├── extract/             # Extraction scripts (improved timeline, reviews, gencap)
│   ├── manual_supervision/  # Weak supervision sample builders
│   ├── manual_training/     # Manual training data builders + gold standard
│   ├── validation/          # Timeline and extraction validation scripts
│   ├── deliverable01–04/    # Phase 2 deliverable POC scripts
│   ├── utils/               # Shared utilities
│   ├── 00_setup.R           # R session setup
│   ├── e01_clean_energy.R   # Clean energy filter exploratory
│   ├── e02_timeline.R       # Timeline exploratory
│   └── page_viewer_{ce,ea,eis}.ipynb  # Interactive document page viewers
├── data/               # Phase 2 processed outputs (never write to ../data/analysis/)
├── models/             # Phase 2 BERT model checkpoints (CE, EA, EIS, combined)
├── notes/              # Architecture notes, current_plan.md, model evaluation
├── output/             # Phase 2 deliverable outputs + timeline validation
├── reports/            # Quarto reports (index.qmd landing page; deliverables added as completed)
├── runbooks/           # Phase 2-specific pipeline docs
└── tests/              # Unit tests
```

## Running Phase 2 pipelines

Phase 2 scripts inherit the same conda environment as Phase 1:

```bash
conda activate nepa
```

All Phase 2 outputs write to `phase2/data/` by default. Phase 1 data in `data/analysis/` is read-only input — never modified.

Phase 2 runbooks include `phase2/runbooks/02_timeline.md` for the improved BERT + LLM adjudication pipeline and `phase2/runbooks/federal_register.md` for Federal Register NOI refreshes. For other extractions without Phase 2-specific runbooks, use the Phase 1 runbooks at `phase1/runbooks/`.

## Relationship to Phase 1

Phase 2 reads Phase 1 source data (e.g., `data/analysis/projects_combined.parquet`) but writes all new outputs to `phase2/data/`. To reproduce Phase 1 exactly, check out the `freeze/v1.0` tag.
