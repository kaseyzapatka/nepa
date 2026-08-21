# Base Dataset Build (Phase 2)

**Purpose:** Build the analysis-ready base tables that every Phase 2 deliverable depends on.
**Input:** NEPATEC 2.0 (downloaded from HuggingFace into `phase2/data/processed/`).
**Output:** `phase2/data/analysis/projects_combined.parquet`, `processes_combined.parquet`, `documents_combined.parquet`, `document_sections.parquet`
**Prerequisites:** `conda activate nepa` (see the root [README](../../README.md); note Quarto is installed separately).

> **Phase 2 builds its own base tables from the NEPATEC corpus.** It does *not* read
> Phase 1's `projects_combined.parquet`. Both phases run their own copy of `extract_data.py`
> over the same processed NEPATEC tables, so the row universe matches (61,881 projects) but
> the columns differ — see [Relationship to Phase 1](#relationship-to-phase-1) below.

## Step 1 — Download the NEPATEC corpus

Pulls the EA, EIS, and CE datasets from HuggingFace into `phase2/data/processed/`.

```bash
python phase2/code/extract/extract_data.py --mode extract
```

Output: `phase2/data/processed/{ea,eis,ce}/{projects,processes,documents,pages}.parquet`

**Skip this step if `phase2/data/processed/` is already populated.** It is in the current
working tree, and those files are byte-identical to Phase 1's copies — the corpus pull has
effectively already happened.

## Step 2 — Build the combined analysis tables

Reads the processed EA/EIS/CE tables and writes the three combined parquets. This also runs
project-description enrichment and merges the Federal Register NOI/NOA fields.

```bash
python phase2/code/extract/extract_data.py --mode analysis
```

Add `--refresh-federal-register` to re-pull Federal Register records rather than reuse the
cache at `phase2/data/analysis/federal_register/federal_register.parquet`.

Outputs, all in `phase2/data/analysis/`:

| File | Grain |
|---|---|
| `projects_combined.parquet` | one row per project (61,881) |
| `processes_combined.parquet` | one row per NEPA process |
| `documents_combined.parquet` | one row per document |

`--mode all` runs steps 1 and 2 in a single command.

## Step 3 — Build the document section layer

`document_sections.parquet` is a reusable section-span index over EA/EIS documents. D2 and D6
both consume it, and D3's visual-impact extraction was its original use case.

It has an ordering dependency that is easy to miss: with no `--target-documents` allowlist, it
joins project metadata from `phase2/data/analysis/deliverable03/projects_nepa_reviews.parquet`,
so **deliverable 3's review builder must run first**.

```bash
python phase2/code/deliverable03/01_identify_visual_impact_candidates.py
python phase2/code/deliverable03/02_build_nepa_reviews.py --section-layer
python phase2/code/extract/build_document_sections.py --process EA EIS --main-only
```

Output: `phase2/data/analysis/document_sections.parquet`

## Relationship to Phase 1

Phase 2 does **not** consume Phase 1's outputs. Both phases read the same processed NEPATEC
tables — verified byte-identical between `phase1/data/processed/` and
`phase2/data/processed/` — and each runs its own `extract_data.py` over them.

The two `projects_combined.parquet` files therefore share a row universe but not a schema:

| | Phase 1 | Phase 2 |
|---|---|---|
| Rows | 61,881 | 61,881 |
| Columns | 97 | 77 |

Phase 1 carries 49 columns Phase 2 lacks — the transmission and pipeline fields from
`extract_technology.py`, which Phase 1 merged into its combined table. Phase 2 has its own copy
of that script but writes those fields to `projects_transmission.parquet` by default rather
than merging them back.

Phase 2 carries 28 columns Phase 1 lacks — the extended Federal Register `noi_*` and `noa_*`
enrichment fields added in Phase 2's `federal_register.py`.

Neither file is derived from the other. Do not backfill Phase 1 outputs into Phase 2; recover
anything missing through the Phase 2 pipeline.

## Notes

- `phase2/data/raw/` is a defined constant in `extract_data.py` but is **not** used by the base
  build. It currently holds unrelated deliverable 6 caches (eCFR, FONSI enrichment).
- The base tables are large and untracked. Only a small set of gold-label and replication cache
  files under `phase2/data/` is committed.
- Re-run steps 2 and 3 whenever the underlying NEPATEC corpus is refreshed.
- Design rationale and data-flow diagrams live in
  [`phase2/architecture/code/extract_data.md`](../architecture/code/extract_data.md); this
  runbook is the execution sequence.
