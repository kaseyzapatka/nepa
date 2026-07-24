# Reproducible Working Environment (`nepa`)

This document defines the project-standard conda environment for NEPA analysis and reporting.

## 1. Why this environment exists

The repository uses:

- Python extraction pipelines (`code/extract/*.py`)
- Python model workflows (Hugging Face `datasets`, `transformers`, `torch`)
- Claude API integration (`anthropic` SDK + direct API calls)
- DuckDB-backed app/data tooling (`app/app.py`, `app/build_text_store.py`)
- R analysis scripts (`code/**/*.R`)
- Quarto reports and presentation rendering (`reports/*.qmd`, `presentations/*.qmd`)

A single, pinned conda environment keeps this stack reproducible end-to-end.

## 2. Dependency audit method

Dependencies were collected from:

- `code/requirements.txt` (starting point)
- `app/requirements.txt`
- Python imports across `code/` and `app/`
- Notebook imports in `code/*.ipynb` and `notebooks/*.ipynb`
- R `library()` / `require()` usage across `code/`, `reports/`, and `presentations/`

This ensured coverage from Claude API workflows through DuckDB and report rendering.

## 3. Python version decision

`Python 3.12` is used in `environment.yml`.

Rationale:

- Compatible with current project stack (`datasets`, `transformers`, `torch`, `spacy`, `streamlit`, `duckdb`)
- Matches the existing project bootstrap direction in `code/setup_textanalysis.sh`
- Avoids older Python versions that can break newer NLP/ML dependencies

## 4. Canonical files

- `environment.yml` (repo root): full reproducible environment (`name: nepa`) for Python + R + Quarto
- `environment.lock.yml` (repo root): exact lock snapshot exported from `nepa` on March 6, 2026 (`osx-64`)
- `code/requirements.txt`: cleaned Python-only fallback for pip installs
- `app/requirements.txt`: minimal dependency file for standalone Streamlit deployment

`environment.yml` is the source of truth for this project.

## 5. Create and use the environment

```bash
conda env create -f environment.yml
conda activate nepa
```

If the environment already exists:

```bash
conda env update -n nepa -f environment.yml --prune
```

## 6. Required credentials

Some workflows require credentials:

```bash
# Hugging Face dataset access
hf auth login

# Claude API-backed extraction/adjudication
export ANTHROPIC_API_KEY='sk-ant-...'
```

## 7. Quick verification checks

```bash
python -c "import pandas, numpy, pyarrow, duckdb, requests, geopy, tqdm, datasets, transformers, torch, anthropic, streamlit, spacy, scattertext, wordcloud"

Rscript -e "pkgs <- c('arrow','classInt','DT','ggalluvial','ggwordcloud','googlesheets4','gt','here','jsonlite','kableExtra','knitr','patchwork','purrr','readr','rmarkdown','rstudioapi','scales','sf','stringr','tibble','tidycensus','tidyr','tidyverse','tigris','zoo'); miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if (length(miss)) { stop(paste('Missing R packages:', paste(miss, collapse=', '))) }"

quarto --version
```

## 8. Reproducibility best practice (lock snapshot)

After a successful install, create a machine-specific lock snapshot:

```bash
conda env export -n nepa > environment.lock.yml
```

Commit that lock file when you need exact rebuilds on the same platform/architecture.
