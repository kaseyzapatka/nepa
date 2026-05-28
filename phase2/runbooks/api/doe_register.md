# DOE Register Run Book

Scripts live in `phase2/code/api/doe_register/`.
All scripts require `conda activate nepa`.

---

## Full Rebuild (all four steps in order)

```bash
# Step 1: Scan NEPATEC for DOE doc numbers embedded in PDF text
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/01_scan_doe_doc_numbers.py

# Step 2a: Scrape energy.gov listing pages (ROD + FONSI listings)
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/02_fetch_doe_register.py

# Step 2b: Fetch individual project pages for unmatched doc numbers (~20 min, rate-limited)
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/03_fetch_project_pages.py

# Step 3: Join and build project-level date output
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/04_build_doe_dates.py
```

Expected total runtime: ~25–35 minutes (03 is the bottleneck).

---

## Re-fetch Project Pages (after network errors or new doc numbers)

```bash
# Re-fetch pages that errored or are new since last run
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/03_fetch_project_pages.py

# Force re-fetch all cached entries (use if energy.gov content was updated)
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/03_fetch_project_pages.py --refetch

# Dry run: show which doc numbers would be fetched
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/03_fetch_project_pages.py --dry-run
```

---

## Outputs and Where to Look

| File | Description |
|------|-------------|
| `doe_case_evidence.parquet` | NEPATEC doc number extractions (2,564 rows) |
| `doe_register_records.parquet` | Combined listing + project page dates (661 rows) |
| `doe_project_page_cache.json` | Raw HTTP cache for individual pages (367 entries) |
| `doe_eplanning_dates.parquet` | Final output — one row per project (516 rows) |
| `doe_manual_review.csv` | Projects with `acceptance=review` for human inspection |

All files are in `phase2/data/analysis/doe_register/`.

---

## Interpreting Results

After running 04, check coverage:

```python
import pandas as pd
df = pd.read_parquet('phase2/data/analysis/doe_register/doe_eplanning_dates.parquet')
print(df.groupby('process_type')[['doe_decision_date', 'doe_initiation_date']].apply(
    lambda g: g.notna().sum()
))
print(df['doe_match_status'].value_counts())
```

Expected ranges:
- EA accepted projects: ~344, decision date rate ~89%
- EIS accepted projects: ~172, decision date rate ~58%
- Overall accepted: ~516, decision date coverage ~78.7%
- NOI initiation dates: ~65 (mostly EIS)

Against the clean energy universe (projects_combined.parquet):
- DOE EA: ~42% decision coverage, <1% initiation
- DOE EIS: ~11% decision, ~7% initiation

---

## Checking Project Page Cache

```python
import json
cache = json.load(open('phase2/data/analysis/doe_register/doe_project_page_cache.json'))
print(f"Cached: {len(cache)}")
from collections import Counter
print(Counter(v['fetch_status'] for v in cache.values()))
```

Cache entries with `fetch_status == "not_found"` mean energy.gov has no page for that doc
number. This is normal (some old doc numbers have no web presence).

---

## Rebuilding Parquet from Cache (without re-fetching)

If you need to regenerate `doe_register_records.parquet` after editing the cache:

```bash
# Re-run 04 to rebuild the final output from existing records
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/04_build_doe_dates.py
```

To rebuild records from the project page cache directly:
```bash
CONDA_DEFAULT_ENV=nepa python phase2/code/api/doe_register/03_fetch_project_pages.py
```
This is idempotent — it reads the cache JSON, skips entries already cached (unless `--refetch`),
and writes the parquet outputs.

---

## Troubleshooting

**Low acceptance rate in 01**
- Multi-doc EIS projects are the main cause. Projects where dozens of cross-referenced doc
  numbers appear without a clear dominant (≥2× second, ≥3 pages) fall into `review`.
- Check `doe_manual_review.csv` for patterns — if a particular EIS program appears frequently,
  add its dominant doc number to a hardcoded override list in 01.

**energy.gov 404 / not_found**
- Not all DOE doc numbers have a page. Older docs (pre-2005) often don't.
- Check the URL pattern: `https://www.energy.gov/nepa/listings/ea-NNNN-documents-available-download`
- If a page exists but dates aren't extracted, run with `--refetch` and inspect the HTML
  to see if the page layout changed.

**Low decision date coverage after 04**
- Verify that `doe_register_records.parquet` has the expected rows (should be ~661).
- The `accepted_not_found` status means a project was accepted in evidence but its doc number
  is not in the register. These projects have no energy.gov page.
- Run 03 without `--refetch` first — it may have new entries to fetch since the last run.

**EPA CDX (EIS database) is not publicly accessible**
- The `cdxapps.epa.gov/cdx-enepa-II/...` endpoint requires CDX account authentication.
- Do not attempt to scrape it. NOI dates are sourced from individual energy.gov project pages.

---

## D4 Integration

After rebuilding `doe_eplanning_dates.parquet`, re-run the D4 pipeline:

```bash
# Script 01: rebuild timeline index (adds DOE dates as new columns)
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/01_build_timeline_index.py

# Script 02: emit Tier A metadata packets for DOE decision + initiation
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/02_retrieve_timeline_contexts.py

# Script 03: prelabel DOE packets as clear_decision / clear_initiation at confidence 5.0
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/03_extract_timeline_candidates.py
```

DOE packets are identifiable in the output by `retrieval_reason` containing
`"doe_register_decision"` or `"doe_register_initiation"`.
