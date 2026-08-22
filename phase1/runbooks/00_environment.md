# Environment Setup

**Purpose:** Create and maintain the reproducible conda environment for all project scripts.
**Output:** conda env `nepa`
**Prerequisites:** [conda](https://docs.conda.io/en/latest/) installed.

> **Quarto is not part of this environment.** The conda env covers Python and R only. Rendering
> the reports additionally requires **Quarto ≥ 1.10**, installed separately from
> [quarto.org](https://quarto.org/docs/get-started/) — *not* from conda-forge. See
> [environment_setup.md](environment_setup.md) §1 for why.

## Create (first time)

```bash
conda env create -f environment.yml
conda activate nepa
```

## Update (sync to latest spec)

```bash
conda env update -n nepa -f environment.yml --prune
```

## Notes

- `--prune` removes packages no longer in `environment.yml`, keeping the env clean.
- Design notes and dependency rationale: `environment_setup.md`
- The geothermal BERT classifier requires additional packages not in the base env — see [runbook 06](06_technology.md#geothermal).
