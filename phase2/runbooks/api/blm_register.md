# BLM National NEPA Register

**Purpose:** Extract authoritative decision and initiation dates for BLM NEPA projects from the BLM National NEPA Register (eplanning.blm.gov). Matched projects receive Tier A dates in the D4 timeline pipeline at confidence 5.0 — the highest in the pipeline.

**Architecture:** [phase2/architecture/blm_eplanning_register.md](../architecture/blm_eplanning_register.md)

**Scripts:** `phase2/code/deliverable04/09a_scan_blm_case_numbers.py`, `09b_fetch_blm_register.py`, `09c_build_blm_dates.py`

---

## Full Rebuild (from scratch)

Run this when the NEPATEC pages have been re-processed, or when the scan regex or acceptance gate logic has changed.

```bash
cd /Users/Dora/git/consulting/nepa
conda activate nepa

# Step 1 — Scan all NEPATEC pages for BLM case numbers
# Reads: phase2/data/processed/{ce,ea,eis}/pages/, documents_combined.parquet, projects_combined.parquet
# Writes: phase2/data/analysis/blm_register/nepatec_case_evidence.parquet
python phase2/code/deliverable04/09a_scan_blm_case_numbers.py

# Step 2 — Fetch from BLM National NEPA Register
# Default: fetch EA only (fastest for decision dates — FONSI returned directly)
python phase2/code/deliverable04/09b_fetch_blm_register.py

# Fetch EIS (ROD dates — fewer records but slower to parse)
python phase2/code/deliverable04/09b_fetch_blm_register.py --case-types EIS

# Fetch CX (CE decision/initiation dates — ~14,500 records, ~8 hours)
# Run overnight; cache is saved every 25 records so it is safe to interrupt
python phase2/code/deliverable04/09b_fetch_blm_register.py --case-types CX

# Step 3 — Join to projects and build project-level date output
# Reads: nepatec_case_evidence.parquet, blm_register_records.parquet
# Writes: phase2/data/analysis/blm_register/blm_eplanning_dates.parquet
python phase2/code/deliverable04/09c_build_blm_dates.py
```

After step 3, re-run the D4 pipeline (scripts 01–03) to incorporate the new Tier A dates — see [Downstream D4 Pipeline](#downstream-d4-pipeline) below.

---

## Partial Refresh

### Re-fetch a specific case type after a parser fix

Use `--refetch` to clear cached entries for the target case type and re-fetch from the register. The existing cache entries for other case types are preserved.

```bash
# Re-fetch EIS records only (e.g., after fixing the ROD date parser)
python phase2/code/deliverable04/09b_fetch_blm_register.py \
    --case-types EIS --refetch

# Re-fetch EA records only
python phase2/code/deliverable04/09b_fetch_blm_register.py \
    --case-types EA --refetch

# Rebuild the parquet from the full cache (after any partial re-fetch)
python -c "
import json, pandas as pd
cache = json.load(open('phase2/data/analysis/blm_register/blm_register_cache.json'))
pd.DataFrame(list(cache.values())).to_parquet(
    'phase2/data/analysis/blm_register/blm_register_records.parquet', index=False)
print(f'Wrote {len(cache)} records')
"

# Rebuild project-level output
python phase2/code/deliverable04/09c_build_blm_dates.py
```

> **Important:** `09b` only writes records for the case numbers it fetched in that run. After any partial re-fetch, always rebuild `blm_register_records.parquet` from the full cache JSON (as shown above), not directly from `09b`'s output, to avoid overwriting records from other case types.

### Re-run 09c only (no re-fetch needed)

Use when 09c join logic has changed but the cache is still valid.

```bash
python phase2/code/deliverable04/09c_build_blm_dates.py
```

### Re-run 09a only (case number scan changed)

Use when the regex patterns, acceptance gate, or document join in 09a have changed.

```bash
python phase2/code/deliverable04/09a_scan_blm_case_numbers.py
# Then run 09b (new case numbers only — already-cached ones are skipped automatically)
python phase2/code/deliverable04/09b_fetch_blm_register.py
python phase2/code/deliverable04/09b_fetch_blm_register.py --case-types EIS
python phase2/code/deliverable04/09b_fetch_blm_register.py --case-types CX
# Rebuild parquet from full cache, then rebuild project output
python phase2/code/deliverable04/09c_build_blm_dates.py
```

---

## Downstream D4 Pipeline

After `blm_eplanning_dates.parquet` is updated, re-run D4 scripts 01–03 to inject the Tier A dates.

```bash
cd /Users/Dora/git/consulting/nepa
conda activate nepa

# Rebuild document index with BLM Tier A flags
python phase2/code/deliverable04/01_build_timeline_index.py

# Retrieve context packets (BLM Tier A packets have retrieval_reason blm_register_decision
# or blm_register_initiation and source_tier=metadata)
python phase2/code/deliverable04/02_retrieve_timeline_contexts.py

# Extract candidates and label BLM Tier A at confidence 5.0
python phase2/code/deliverable04/03_extract_timeline_candidates.py
```

---

## CLI Arguments

### 09a — Scan NEPATEC pages

| Argument | Default | Description |
|---|---|---|
| *(none)* | — | Scans all process types (CE, EA, EIS) |

### 09b — Fetch from register

| Argument | Default | Description |
|---|---|---|
| `--acceptance` | `accept` | Which rows from 09a to fetch: `accept`, `review`, `skip` |
| `--case-types` | *(all)* | Restrict to suffix: `EA`, `EIS`, `CX`, `DNA`, `DR`, etc. |
| `--refetch` | off | Clear matching cache entries and re-fetch |
| `--dry-run` | off | Print what would be fetched without hitting the register |

### 09c — Build project-level dates

| Argument | Default | Description |
|---|---|---|
| *(none)* | — | Reads evidence + records, writes blm_eplanning_dates.parquet |

---

## Outputs

| File | Description |
|---|---|
| `blm_register/nepatec_case_evidence.parquet` | One row per (project_id, case_number) found in NEPATEC; includes `acceptance`, `process_match`, `case_type`, `case_number_raw` |
| `blm_register/blm_register_cache.json` | Raw API responses keyed by case number; cumulative across runs |
| `blm_register/blm_register_records.parquet` | One row per case number; date fields from the register |
| `blm_register/blm_eplanning_dates.parquet` | One row per project; `blm_decision_date`, `blm_initiation_date`, `blm_match_status` |
| `blm_register/blm_manual_review.csv` | Projects with `acceptance=review` for human inspection |

---

## Interpreting Results

**Coverage summary** (after initial full build):

| Process | BLM projects | Accepted | Decision dates | Initiation dates |
|---|---|---|---|---|
| EA | 2,217 | 1,253 | 1,157 (52%) | 1,068 (48%) |
| EIS | 760 | 55 | 52 (7%) | 52 (7%) |
| CE | 23,039 | ~14,770 | 0 (0%) | ~12,400 est. (54%) |

**Why EIS coverage is low:** Many EIS documents cross-reference EA case numbers (tiered EAs), causing `process_match=False` and routing to `review`. Only unambiguous EIS case numbers in main documents reach `accept`.

**Why CE has no decision dates:** The BLM register tracks CX project initiation (start date) but not the CX determination/signing date. CE decision dates remain 0% from this pipeline — they must come from BERT/regex text extraction.

**`not_found` in 09b output:** Pre-2010 BLM case numbers and non-standard CA Coastal office codes (`C060` variants) frequently return `not_found`. No fix available without manual lookup.

---

## Troubleshooting

**Session / CSRF token errors during 09b:** The BLM register uses D365 CSRF tokens that expire. If 09b logs repeated HTTP 400 or 403 responses, re-run — the token is refreshed at startup and every 50 requests.

**09b run interrupted mid-batch:** Cache is saved every 25 records. Re-run without `--refetch` and 09b will skip already-cached case numbers and resume from where it left off.

**`blm_register_records.parquet` shows fewer rows than expected:** This happens when 09b was run for a subset (e.g., only EIS) and overwrote the full parquet. Rebuild from the cache JSON:
```bash
python -c "
import json, pandas as pd
cache = json.load(open('phase2/data/analysis/blm_register/blm_register_cache.json'))
pd.DataFrame(list(cache.values())).to_parquet(
    'phase2/data/analysis/blm_register/blm_register_records.parquet', index=False)
print(f'Wrote {len(cache)} records')
"
```

**09a `Binder Error: Table "d" does not have a column "document_type_category"`:** 09a must join against `documents_combined.parquet`, not the process-specific raw `documents.parquet`. Verify the `DOCUMENTS_PATH` constant in 09a points to `data/processed/documents_combined.parquet`.
