# NEPA Project: Clean Energy Environmental Reviews

Analysis of clean energy projects using the National Environmental Policy Act Text Corpus (NEPATEC) 2.0 dataset from PNNL's PermitAI project.

**Project website:** [Working project website](https://www.kaseyzapatka.com/nepa/reports/project_overview.html)
**Data source:** [NEPATEC 2.0 on Hugging Face](https://huggingface.co/datasets/PNNL/NEPATEC2.0)

---

## Repository structure

```
nepa/
├── README.md                  # This file — project overview
├── _quarto.yml                # Quarto website configuration
├── environment.yml            # Conda environment spec (shared by phase1 + phase2)
├── environment.lock.yml       # Locked conda dependencies
├── admin/                     # Administrative documents and notes
├── app/                       # Streamlit document explorer (deployed to HF Spaces)
├── docs/                      # Built Quarto website output
├── literature/                # Reference materials
├── phase1/                    # Phase 1 analysis — frozen at freeze/v1.0
│   ├── README.md              # Phase 1 pipeline documentation
│   ├── code/                  # Extraction scripts + per-deliverable R analysis
│   ├── data/                  # Phase 1 processed data outputs (parquet files)
│   ├── models/                # Trained BERT timeline classifiers
│   ├── notes/                 # Status files, architecture notes, running todo
│   ├── output/                # Deliverable figures, tables, maps
│   ├── reports/               # Quarto deliverable reports
│   └── runbooks/              # Step-by-step pipeline instructions (00–08)
├── phase2/                    # Phase 2 analysis — active development
│   ├── README.md              # Phase 2 pipeline documentation
│   ├── code/                  # Improved extraction scripts + deliverable POCs
│   ├── data/                  # Phase 2 processed data outputs
│   ├── models/                # Improved BERT model checkpoints (CE, EA, EIS)
│   ├── notes/                 # Architecture notes, current plan, model evaluation
│   ├── output/                # Phase 2 deliverable outputs + timeline validation
│   ├── reports/               # Quarto reports
│   ├── runbooks/              # Phase 2-specific pipeline docs
│   └── tests/                 # Unit tests
├── presentations/             # RevealJS slides (CATF stakeholder presentation)
└── scripts/                   # Utility scripts
```

---

## How it works

Raw NEPATEC 2.0 documents are loaded and preprocessed into per-source parquet files. Python extraction scripts (`code/extract/`) use a combination of regex, BERT classifiers, and LLM adjudication to pull structured fields (dates, capacity, review type, page counts, technology) from document text. These parquet outputs feed into per-deliverable R scripts that produce figures and tables. Quarto renders the R outputs into HTML reports, which are published as a static website via `docs/`.

---

## Phase 1 and Phase 2

| | Phase 1 | Phase 2 |
|---|---|---|
| Status | Frozen at `freeze/v1.0` | Active development |
| Data pipeline | Pandas-based | DuckDB-based |
| Timeline extraction | BERT classifier | Improved BERT + LLM hybrid adjudication |
| Deliverables | D1–D6 complete | In progress |
| Output location | `phase1/data/analysis/` | `phase2/data/` |

**Data flow:** Phase 2 reads `phase1/data/analysis/projects_combined.parquet` as read-only input and writes all new outputs to `phase2/data/`. Phase 1 data is never modified by Phase 2 scripts.

To reproduce Phase 1 exactly: `git checkout freeze/v1.0`

---

## Getting started

```bash
conda env create -f environment.yml
conda activate nepa
```

Both phases share this environment. See [environment.yml](environment.yml) for the full dependency spec, or `phase1/notes/architecture/environment_setup.md` for design rationale.

- **Phase 1 pipeline:** [phase1/README.md](phase1/README.md)
- **Phase 2 pipeline:** [phase2/README.md](phase2/README.md)

---

## Build the Document Explorer (HF Spaces)

Use this workflow to deploy the Streamlit NEPA document explorer. The 7+ GB DuckDB file is stored in a Hugging Face Dataset repo (not committed to the Space).

### 1) Build the DuckDB locally (one-time per data refresh)

```bash
python phase1/code/rag/01_build_text_store.py
```

Output: `phase1/data/rag/nepa_reader.duckdb`

### 2) Upload the DB to a Hugging Face Dataset repo

```bash
HF_USERNAME="YOUR_HF_USERNAME"
DB_REPO="nepa-document-explorer-db"

hf repo create "${HF_USERNAME}/${DB_REPO}" --repo-type dataset || true
hf upload "${HF_USERNAME}/${DB_REPO}" phase1/data/rag/nepa_reader.duckdb nepa_reader.duckdb --repo-type dataset
```

### 3) Deploy app to a Hugging Face Docker Space

```bash
SPACE_NAME="nepa-document-explorer"
hf repo create "${HF_USERNAME}/${SPACE_NAME}" --repo-type space --space_sdk docker || true

DEPLOY_DIR="$(mktemp -d)"
cp app/app.py "${DEPLOY_DIR}/app.py"
cp app/requirements.txt "${DEPLOY_DIR}/requirements.txt"

cat > "${DEPLOY_DIR}/Dockerfile" <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV NEPA_DB_HF_REPO=YOUR_HF_USERNAME/nepa-document-explorer-db
ENV NEPA_DB_HF_FILENAME=nepa_reader.duckdb
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=7860"]
EOF

hf upload "${HF_USERNAME}/${SPACE_NAME}" "${DEPLOY_DIR}" --repo-type space --commit-message "Deploy NEPA document explorer"
```

If your CLI does not accept `--space_sdk`, create the Space in the HF web UI as **Docker**, then continue with the upload step.

### 4) Routine updates

- **App-only update:** re-upload `app.py`, `requirements.txt`, `Dockerfile` to the Space.
- **Data refresh:** rebuild `nepa_reader.duckdb`, re-upload to the dataset repo, then restart the Space.

Space URL: `https://huggingface.co/spaces/YOUR_HF_USERNAME/nepa-document-explorer`

> **Note:** HF Space repos have strict storage limits on the free tier (~1 GB). Do not commit `.duckdb` into the Space repo — keep it in the dataset repo.
