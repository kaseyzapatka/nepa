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
├── architecture/
│   ├── README.md            # Index of all architecture docs
│   ├── code/                # Core pipeline script architecture
│   │   ├── extract_data.md  # Main pipeline entry point
│   │   └── federal_register.md  # FR NOI/NOA matching design
│   └── deliverables/        # Per-deliverable data flow and methodology
│       └── deliverable01.md # D1: NEPA Triggered (others added as built)
├── code/
│   ├── extract/             # Extraction scripts (timeline, reviews, gencap, federal_register)
│   ├── manual_supervision/  # Weak supervision sample builders
│   ├── manual_training/     # Manual training data builders + gold standard
│   ├── validation/          # Timeline and extraction validation scripts
│   ├── deliverable01–06/    # Phase 2 deliverable analysis scripts
│   └── utils/               # Shared utilities
├── data/               # Phase 2 processed outputs (never write to ../data/analysis/)
├── models/             # Local model weights (untracked; SetFit trigger classifier ships via the GitHub Release)
├── output/             # Phase 2 deliverable outputs + timeline validation
├── reports/            # Quarto reports (index.qmd landing page; deliverables added as completed)
├── runbooks/           # Step-by-step execution guides
└── training/           # Model training labels and locked sample IDs (see its README)
```

## Running Phase 2 pipelines

Phase 2 scripts inherit the same conda environment as Phase 1:

```bash
conda activate nepa
```

All Phase 2 outputs write to `phase2/data/` by default. Phase 1 data in `data/analysis/` is read-only input — never modified.

Architecture docs (`phase2/architecture/`) explain the *why* and *what* of each pipeline — data flow diagrams, design rationale, output schemas, and methodological notes. Runbooks (`phase2/runbooks/`) are step-by-step execution guides. For other extractions without Phase 2-specific runbooks, use the Phase 1 runbooks at `phase1/runbooks/`.

## Relationship to Phase 1

Phase 2 reads Phase 1 source data (e.g., `data/analysis/projects_combined.parquet`) but writes all new outputs to `phase2/data/`. To reproduce Phase 1 exactly, check out the `freeze/phase1_v1.0` tag.
