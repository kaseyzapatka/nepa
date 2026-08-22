# Reproducible Working Environment (`nepa`)

This document defines the project-standard conda environment for NEPA analysis and reporting.

## 1. Why this environment exists

The repository uses:

- Python extraction pipelines (`phase1/code/extract/*.py`)
- Python model workflows (Hugging Face `datasets`, `transformers`, `torch`)
- Claude API integration (`anthropic` SDK + direct API calls)
- DuckDB-backed app/data tooling (`app/app.py`, `app/build_text_store.py`)
- R analysis scripts (`phase1/code/**/*.R`)
- Quarto reports and presentation rendering (`phase1/reports/*.qmd`, `presentations/*.qmd`)

A single, pinned conda environment keeps the Python and R stack reproducible end-to-end.

**Quarto is deliberately excluded from the conda environment.** It is installed separately from
[quarto.org](https://quarto.org/docs/get-started/), and the project requires **Quarto ≥ 1.10**.
The conda-forge quarto builds pull in a Deno that crashes `quarto render` on this project, and
the pin used to work around that (1.3.433) silently produced a *downgraded* site — it drops the
screen-reader-only callout labels that Quarto 1.9+ emits, a regression invisible to a sighted
reviewer. A pre-render hook, `scripts/check_quarto_version.sh`, now aborts the build if Quarto
is too old.

## 2. Dependency audit method

Dependencies were collected from:

- `phase1/code/requirements.txt` (starting point)
- `app/requirements.txt`
- Python imports across `phase1/code/` and `app/`
- Notebook imports in `phase1/code/*.ipynb`
- R `library()` / `require()` usage across `phase1/code/`, `phase1/reports/`, and `presentations/`

This ensured coverage from Claude API workflows through DuckDB and report rendering.

## 3. Python version decision

`Python 3.12` is used in `environment.yml`.

Rationale:

- Compatible with current project stack (`datasets`, `transformers`, `torch`, `spacy`, `streamlit`, `duckdb`)
- Matches the existing project bootstrap direction in `phase1/code/setup_textanalysis.sh`
- Avoids older Python versions that can break newer NLP/ML dependencies

## 4. Canonical files

- `environment.yml` (repo root): full reproducible environment (`name: nepa`) for Python + R. Does **not** include Quarto — see the note in section 1.
- `phase1/code/requirements.txt`: cleaned Python-only fallback for pip installs
- `app/requirements.txt`: minimal dependency file for standalone Streamlit deployment

`environment.yml` is the source of truth for the Python and R stack; Quarto is installed separately.

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

# Quarto is NOT part of the conda env — this must report 1.10.0 or newer.
# If it reports an older version, or resolves inside the conda env, see section 1.
quarto --version
which -a quarto
```

## 8. Reproducibility approach (no lock file)

The project keeps a single portable spec — `environment.yml`, with the exact versions used
annotated in comments — rather than a platform-specific lock file. The from-scratch environment
build is verified against the spec directly.

There is deliberately no `environment.lock.yml` in this repository. If you need byte-exact
rebuilds on one platform, generate a lock snapshot locally (`conda env export > environment.lock.yml`),
but treat it as a machine-specific artifact rather than the project's source of truth.
