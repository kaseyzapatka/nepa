# Environment Setup

**Purpose:** Create and maintain the reproducible conda environment for all project scripts.
**Output:** conda env `nepa`
**Prerequisites:** [conda](https://docs.conda.io/en/latest/) installed.

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
- Design notes and dependency rationale: `notes/architecture/environment_setup.md`
- The geothermal BERT classifier requires additional packages not in the base env — see [runbook 06](06_technology.md#geothermal).
