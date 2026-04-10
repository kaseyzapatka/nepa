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
│   ├── extract/        # Extraction scripts
│   ├── manual_supervision/  # Supervision sample builders
│   ├── manual_training/     # Training data builders
│   ├── deliverable01–04/    # Phase 2 deliverable analysis scripts
│   └── utils/
├── data/               # Phase 2 processed outputs (never write to ../data/analysis/)
├── models/             # Phase 2 model checkpoints
├── notes/              # Architecture and workflow documentation
├── output/             # Phase 2 deliverable outputs
├── reports/            # Quarto reports
├── runbooks/           # Step-by-step pipeline docs
└── tests/              # Tests
```

## Running Phase 2 pipelines

Phase 2 scripts inherit the same conda environment as Phase 1:

```bash
conda activate nepa
```

All Phase 2 outputs write to `phase2/data/` by default. Phase 1 outputs in `data/analysis/` are never modified.

See `phase2/runbooks/` for step-by-step instructions as they are developed.

## Relationship to Phase 1

Phase 2 reads Phase 1 source data (e.g., `data/analysis/projects_combined.parquet`) but writes all new outputs to `phase2/data/`. To reproduce Phase 1 exactly, check out the `freeze/v1.0` tag.
