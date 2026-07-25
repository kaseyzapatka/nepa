# NEPA Project: Clean Energy Environmental Reviews

Analysis of clean energy projects using the National Environmental Policy Act Text Corpus (NEPATEC) 2.0 dataset from PNNL's PermitAI project. This work was produced in collaboration with the [Clean Air Task Force (CATF)](https://www.catf.us/), whose questions about clean-energy permitting shaped the deliverables in both phases.

**Project website:** [kaseyzapatka.com/nepa](https://www.kaseyzapatka.com/nepa/)

**Data source:** [NEPATEC 2.0 on Hugging Face](https://huggingface.co/datasets/PNNL/NEPATEC2.0)

---

## Repository structure

What's tracked in this repository (large data drops, trained models, and admin files live outside git — see notes below):

```
nepa/
├── README.md                  # This file — project overview
├── _quarto.yml                # Quarto website configuration
├── index.qmd                  # Website landing page
├── includes/                  # Site-wide HTML includes (navbar submenu)
├── scripts/                   # Quarto post-render hook (PDF generation)
├── environment.yml            # Conda environment spec (shared by phase1 + phase2)
├── app/                       # Streamlit document explorer (deployed to HF Spaces)
├── docs/                      # Built Quarto website output (published site)
├── literature/                # Reference materials
├── phase1/                    # Phase 1 analysis — complete, frozen at freeze/phase1_v1.0
│   ├── README.md              # Phase 1 pipeline documentation
│   ├── code/                  # Extraction scripts + per-deliverable R analysis
│   ├── notes/                 # Status files, architecture notes
│   ├── output/                # Deliverable figures, tables, maps
│   ├── reports/               # Quarto deliverable reports
│   └── runbooks/              # Step-by-step pipeline instructions (00–08)
├── phase2/                    # Phase 2 analysis — complete (D1–D6)
│   ├── README.md              # Phase 2 pipeline documentation
│   ├── code/                  # Extraction pipelines + per-deliverable scripts
│   ├── architecture/          # As-built technical documentation per deliverable
│   ├── notes/                 # Published methods notes and coverage pages
│   ├── output/                # Deliverable outputs (report-input CSVs are tracked)
│   ├── reports/               # Quarto reports
│   ├── runbooks/              # Pipeline runbooks
│   └── training/              # Model training labels and locked sample IDs
├── presentations/             # RevealJS slides (CATF stakeholder presentation)
```

Not in the repository: raw NEPATEC data and multi-GB processed parquets (`phase1/data/`, `phase2/data/` — local data drops; a small set of gold-label and replication cache files *is* tracked), trained model weights (`phase1/models/`, `phase2/models/` — available via the project's GitHub Release), and administrative files.

---

## How it works

Raw NEPATEC 2.0 documents are loaded and preprocessed into per-source parquet files. Python extraction pipelines (`phase1/code/extract/`, `phase2/code/`) use a combination of regex, fine-tuned classifiers (BERT/SetFit), and LLM adjudication to pull structured fields (dates, capacity, review type, page counts, technology) from document text. These parquet outputs feed per-deliverable analysis scripts (Python/DuckDB and R) that produce figures and tables. Quarto renders everything into HTML reports, published as a static website via `docs/`.

---

## Phase 1 and Phase 2

| | Phase 1 | Phase 2 |
|---|---|---|
| Status | Complete — frozen at `freeze/phase1_v1.0` | Complete (D1–D6) |
| Data pipeline | Pandas-based | DuckDB-based |
| Timeline extraction | BERT classifier | Multi-tier retrieval + SetFit/LightGBM + LLM adjudication |
| Deliverables | D1–D6 complete | D1–D6 complete |
| Output location | `phase1/data/analysis/` | `phase2/data/` |

**Data flow:** Phase 2 reads `phase1/data/analysis/projects_combined.parquet` as read-only input and writes all new outputs to `phase2/data/`. Phase 1 data is never modified by Phase 2 scripts.

To reproduce Phase 1 exactly: `git checkout freeze/phase1_v1.0`

---

## Getting started

```bash
conda env create -f environment.yml
conda activate nepa
```

Both phases share this environment — Python pipelines and the rendering stack (Quarto 1.3.433 + R 4.2, pinned to the versions that produced the published site). See [environment.yml](environment.yml) for the full dependency spec with the exact versions annotated, or `phase1/notes/architecture/environment_setup.md` for design rationale.

**Rendering the website**: `quarto render` from the repo root inside the activated env. Report-input CSVs are tracked, so the reports render from a fresh clone; re-running the extraction pipelines themselves requires the NEPATEC data drop.

- **Phase 1 pipeline:** [phase1/README.md](phase1/README.md)
- **Phase 2 pipeline:** [phase2/README.md](phase2/README.md)

---

## Document Explorer (HF Spaces)

A Streamlit app for browsing NEPA document text, deployed to Hugging Face Spaces with its multi-GB DuckDB stored in an HF Dataset repo. Build and deployment instructions live in **[app/runbook.md](app/runbook.md)**.

---

## License

Code in this repository is released under the [MIT License](LICENSE). The underlying NEPATEC 2.0 data is published by PNNL ([Hugging Face](https://huggingface.co/datasets/PNNL/NEPATEC2.0)); the analysis reports were produced in collaboration with the Clean Air Task Force.
