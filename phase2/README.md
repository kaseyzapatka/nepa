# Phase 2: Extended NEPA Analysis

Phase 2 extends the Phase 1 NEPA analysis with an improved data pipeline and deeper extraction capabilities.

## Key differences from Phase 1

| | Phase 1 | Phase 2 |
|---|---|---|
| Data pipeline | Pandas-based | DuckDB-based |
| Timeline extraction | BERT classifier | Improved BERT + LLM hybrid adjudication |
| Output location | `phase1/data/analysis/` | `phase2/data/` |
| Status | Complete | Complete (D1–D6) |

## Structure

```
phase2/
├── architecture/
│   ├── README.md            # Index of all architecture docs
│   ├── code/                # Core pipeline script architecture
│   │   ├── extract_data.md  # Main pipeline entry point
│   │   └── federal_register.md  # FR NOI/NOA matching design
│   └── deliverables/        # Per-deliverable data flow and methodology (D1–D6)
├── code/
│   ├── extract/             # Base build + extraction scripts (extract_data, timeline, reviews, technology, federal_register)
│   ├── manual_supervision/  # Weak supervision sample builders
│   ├── manual_training/     # Manual training data builders + gold standard
│   ├── validation/          # Timeline and extraction validation scripts
│   ├── deliverable01–06/    # Phase 2 deliverable analysis scripts
│   └── utils/               # Shared utilities
├── data/               # Phase 2 base tables + deliverable outputs (largely untracked)
├── factsheets/         # Client-facing one-page summaries per deliverable
├── models/             # Local model weights (untracked; SetFit classifiers ship via the GitHub Release)
├── notes/              # Published methods notes and coverage/limitations pages
├── output/             # Phase 2 deliverable figures, tables, and validation output
├── rag/                # Retrieval-augmented Q&A app over the document corpus
├── reports/            # Quarto deliverable reports (D1–D6)
├── runbooks/           # Step-by-step execution guides
└── training/           # Model training labels and locked sample IDs (see its README)
```

## Running Phase 2 pipelines

Phase 2 scripts inherit the same conda environment as Phase 1:

```bash
conda activate nepa
```

**Start with [runbooks/01_base_dataset.md](runbooks/01_base_dataset.md).** It builds the base tables (`projects_combined`, `processes_combined`, `documents_combined`, `document_sections`) that every deliverable depends on. All Phase 2 outputs write under `phase2/data/`; nothing in Phase 2 writes to `phase1/`.

Architecture docs (`phase2/architecture/`) explain the *why* and *what* of each pipeline — data flow diagrams, design rationale, output schemas, and methodological notes. Runbooks (`phase2/runbooks/`) are the step-by-step execution guides.

## Relationship to Phase 1

**Phase 2 does not read Phase 1's outputs.** It builds its own base tables from the NEPATEC corpus by running its own copy of `extract_data.py`. Both phases read the same processed EA/EIS/CE tables — byte-identical between `phase1/data/processed/` and `phase2/data/processed/` — so the project universe matches (61,881 rows in each `projects_combined.parquet`), but the schemas differ: Phase 1 has 97 columns, Phase 2 has 77.

Phase 1 carries the transmission and pipeline fields merged in from `extract_technology.py`; Phase 2 has the extended Federal Register `noi_*`/`noa_*` enrichment instead. Neither file is derived from the other, and Phase 1 outputs should never be backfilled into Phase 2 — recover anything missing through the Phase 2 pipeline. See [runbooks/01_base_dataset.md](runbooks/01_base_dataset.md#relationship-to-phase-1) for the full breakdown.
