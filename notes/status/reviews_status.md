# Deliverable #2: Programmatic & Tiered Reviews

**Status**: Analysis complete; 118/161 non-standard projects have full timelines
**Last Updated**: 2026-03-04
**Script**: `code/extract/extract_reviews.py`
**Output**: `data/analysis/projects_reviews.parquet`
**Report**: `reports/deliverable02.qmd`

## Deliverable Goal

Data on programmatic and tiered reviews: how many tiered reviews are there compared to total, and are they completed faster?

---

## Current Status

### Completed
- [x] Created `extract_reviews.py` with 3-tier extraction approach
- [x] Title-based detection for programmatic reviews
- [x] Regex extraction with false-positive filtering
- [x] LLM integration for ambiguous cases (optional flag)
- [x] Fixed bug: "tiering from PEIS" was incorrectly flagged as programmatic
- [x] **Full extraction completed** on 1,416 EA/EIS clean energy projects
- [x] Created exploratory analysis script (`code/exploratory/reviews/01_reviews_eda.R`)
- [x] Created Google Sheet for client validation
- [x] Created report (`reports/deliverable02.qmd`)

### Extraction Results

| Review Type | Count | Share |
|-------------|-------|-------|
| Standard | 1,390 | 98.2% |
| Programmatic | 16 | 1.1% |
| Tiered | 10 | 0.7% |
| **Total** | **1,416** | 100% |

- Confidence: **high** for 1,413 projects, **medium** for 3
- Detection source: **text_regex** (1,413), **title** (3)

### Awaiting Validation
- [ ] Client review of 26 identified programmatic/tiered reviews via Google Sheet
- [ ] Investigate potential duplicates (2 sets in programmatic tab, 1 in tiered)
- [ ] Add project `3621210fbd086bddbbf6fbedc1d6a488` — found via file name search but classified as "standard"
- [ ] Confirm "Programmatic Collaborations" (DOE grants) should be excluded

### Next Steps
1. **Client validation** — review Google Sheet, mark correct/incorrect
2. **Add missing project** — investigate `3621210fbd086bddbbf6fbedc1d6a488`
3. **Add evidence_context** to programmatic extraction (currently only in tiered)
4. **Merge with timeline data** — answer "are tiered reviews completed faster?"
5. **Refine patterns** if validation reveals false positives/negatives

---

## Validation Resources

### Google Sheet
**[NEPA Reviews Validation Sheet](https://docs.google.com/spreadsheets/d/1d25Arj2IFR3SLcgv8B6tPdLbs2cDZtC_IjMhZUcueBA/edit?usp=sharing)**

Two tabs:
- **Programmatic** (16 projects): Reviews that ARE programmatic EIS/EAs
- **Tiered** (10 projects): Reviews that tier FROM a programmatic review

Key columns for reviewers:
- `evidence_text` — text that triggered classification
- `tiers_from` — (tiered only) programmatic review being referenced
- `correct` — Yes/No/Unsure
- `notes` — comments

### Potential Issues Found

1. **Duplicates**: Some projects have same title but different `project_id` — need manual review
2. **Missing project**: `3621210fbd086bddbbf6fbedc1d6a488` has "programmatic" in file name but was classified as "standard" — may need file name search added to extraction
3. **"Programmatic Collaborations"**: DOE grants with "programmatic" in name are NOT programmatic NEPA reviews — confirmed these are in CE dataset and excluded from EA/EIS scope

---

## Known Issues

### Issue 1: Rarity of Examples
**Problem**: Programmatic/tiered reviews are very rare.
- Only 26 found out of 1,416 projects (1.8%)
- Tiered reviews (10) fewer than expected (estimated 5-15%)

**Implication**: Timeline comparisons will have small sample sizes — results should be framed as exploratory.

### Issue 2: Missing File Name Search
**Problem**: Extraction searches titles and document text, but not file names.
- Project `3621210fbd086bddbbf6fbedc1d6a488` has "programmatic" in file name but not detected
- May be missing other projects

**TODO**: Consider adding file name search to extraction.

### Issue 3: Programmatic Tab Missing Context
**Problem**: `evidence_context` (surrounding text) only captured for tiered reviews, not programmatic.

**TODO**: Update extraction to capture context for programmatic reviews too.

---

## Background

### What are Programmatic Reviews?

**Programmatic EIS/EA**: A broad environmental review that analyzes a class of similar actions (e.g., all solar projects on BLM land). These serve as "umbrella" documents.

Examples found:
- "Programmatic Environmental Assessment for System-wide Operations and Maintenance"
- "Recycle of Scrap Metal Originating From Radiological Areas" (THIS PROGRAMMATIC ENVIRONMENTAL...)

### What are Tiered Reviews?

**Tiered Reviews**: Project-specific reviews that reference and build upon a programmatic review. They don't repeat analysis already done, potentially making them faster.

Examples found:
- "This EA tiers from the 2012 NPR-A IAP/EIS and its ROD"
- "The EA tiered from the analysis conducted in [programmatic review]"

---

## Extraction Approach

### 3-Tier Strategy

| Tier | Method | When Used | Speed |
|------|--------|-----------|-------|
| 1 | Title detection | Always first | Instant |
| 2 | Regex with context | If title doesn't match | ~10 sec/project |
| 3 | LLM (Ollama) | Only if `--no-llm` not set AND medium-confidence matches | ~5 sec/call |

### Search Patterns

**Programmatic (titles):** `programmatic`, `program-wide`, `PEIS`, `PEA`

**Tiered (document text):**
- "this EA tiers from..."
- "tiers to the...PEIS"
- "pursuant to the...Programmatic EIS"
- "site-specific EA that tiers from..."

**False positive exclusions:** EPA Tier 1-4 engines, road classifications, tiered pricing, ranking language

### Variables Created

| Variable | Type | Description |
|----------|------|-------------|
| `project_review_is_programmatic` | Boolean | TRUE if this project IS a programmatic review |
| `project_review_type` | Categorical | `programmatic`, `tiered`, `standard`, `unknown` |
| `project_review_tiers_from` | String | Name of the programmatic review being tiered from |
| `project_review_tiers_from_context` | String | Full context text (tiered only currently) |
| `project_review_confidence` | Categorical | `high`, `medium`, `low` |
| `project_review_source` | String | Detection source: `title`, `text_regex`, `llm` |
| `project_review_match_text` | String | Actual matched text from document |

---

## Timeline Integration (TODO)

To answer "are tiered reviews completed faster":

1. Merge `projects_reviews.parquet` with timeline data
2. Compare duration between tiered vs standard reviews
3. Timeline extraction still being refined — revisit when ready

Key timeline variables needed:
- `llm_decision_date` — Final approval date
- `llm_application_date` — Start date
- Duration = decision_date - application_date

**Note**: With only 10 tiered reviews, statistical power will be limited.

---

## Usage

```bash
# Full extraction (EA + EIS, no LLM)
python code/extract/extract_reviews.py --run --no-llm

# Full extraction with LLM for ambiguous cases
python code/extract/extract_reviews.py --run

# Include CE projects
python code/extract/extract_reviews.py --run --include-ce
```

### Exploratory Analysis

```bash
# Run EDA and write to Google Sheet
Rscript code/exploratory/reviews/01_reviews_eda.R
```

---

## Development Log

### 2026-03-04

#### Timeline Coverage Improvements for Non-Standard Projects (D2)

The full review extraction (161 programmatic + tiered projects in NEPATEC 2.0) was complete.
The challenge was timeline coverage: only 82/161 had full timelines after the base EA/EIS LLM
adjudication. A series of targeted changes brought this to **118/161 (73%)**.

**Root cause**: EIS documents averaged 179 BERT candidates but the LLM cap was 30 — 150+
candidates were filtered before Claude saw them. Programmatic EISs are the hardest: BERT
(trained on CE data) frequently mislabels decision-era dates as 'review' or 'other'.

**Changes to `code/extract/extract_timeline.py`**:

1. **`--nonstandard-incomplete` flag for `--llm-adjudicate`**: Auto-selects programmatic/tiered
   projects still missing `llm_initiation_date` or `llm_decision_date`. Reads
   `projects_reviews.parquet` + all available timeline outputs internally.

2. **Multi-file `--input`**: Accept comma-separated parquet paths (e.g., EA + EIS in one call).

3. **`--max-candidates N`**: Override per-project candidate cap (used 125 vs default 30 for EIS).

4. **`--context-chars N`**: Override context snippet length.

5. **`--promote-rod-language`**: Promotes candidates with ROD/FONSI language to Tier A decision
   regardless of BERT doc_type classification.

6. **`--year-window N`**: Drops candidates more than N years before the latest found date
   (removes NEPA citation noise; used 15).

7. **529 retry**: `_call_claude_adjudication()` now retries HTTP 529 (service overload) with
   60-second backoff, same as 429.

8. **Silent BERT decision discard fix**: Layer 5 was dropping `dtype=='decision'` candidates
   that failed strict doc-type checks. They now pass to `non_decision` so Claude sees them.

9. **`best_effort` prompt for `no_decision_candidates` mode**: When `--nonstandard-incomplete`
   is active, `_build_adjudication_prompt()` uses a best-effort prompt ("pick most likely
   decision date, null only if truly nothing fits") instead of the conservative default
   ("only if context clearly indicates... otherwise null"). Standard runs are unaffected.

**Changes to `code/deliverable02/00_setup.R`**:
- Added `tl_full`: full timeline parquets (with `bert_dates_json`) patched with targeted dates.
- Added `browse_ns`: coverage table for all 161 non-standard projects (`has_initiation`,
  `has_decision`, `complete`, `llm_decision_mode`).
- Added `inspect_candidates(pid)`: parses `bert_dates_json` JSON and prints all BERT date
  candidates for a given project.

**Changes to `code/deliverable02/01_reviews.R`**:
- Added inspection block calling `browse_ns` and `inspect_candidates()` at the top.

**New output**: `data/analysis/projects_timeline_targeted_llm.parquet` (73 rows).
Patched into the main timeline via `coalesce()` in `00_setup.R`.

**Coverage results** (161 non-standard projects):

| Status | Count |
|---|---|
| Complete (both dates) | **118** |
| Missing decision only | 32 |
| Missing initiation only | 3 |
| Missing both | 8 |
| **Total incomplete** | **43** |

The 29 remaining `no_decision_candidates` projects have zero decision signals identifiable
by BERT in any candidate — hard floor without document-level improvements.

**Run command** (documented in README):
```bash
python code/extract/extract_timeline.py \
  --llm-adjudicate \
  --input data/analysis/projects_timeline_bert_ea_llm.parquet,data/analysis/projects_timeline_bert_eis_llm.parquet \
  --nonstandard-incomplete \
  --max-candidates 125 --context-chars 400 \
  --promote-rod-language --year-window 15 \
  --provider claude \
  --output data/analysis/projects_timeline_targeted_llm.parquet
```
**Cost**: ~$0.44 (Haiku, ~400K input tokens for 73 projects).

#### Manual Date Overrides (TODO: integrate into pipeline)

Six projects were manually patched in `code/deliverable02/00_setup.R` for the 2026-03-06
presentation. These are hardcoded in a `manual_overrides` tibble and should eventually be
moved into the pipeline (e.g., a `data/analysis/manual_dates.csv` patched via coalesce in
`00_setup.R`, or re-adjudicated with corrected inputs).

| project_id | field | value | source |
|---|---|---|---|
| `cf2fbe90d43ac57a9460fa857f34af6c` | initiation | `noi_publication_date` | Federal Register NOI |
| `f95ec9530b352e3dd46e6473cb80dccf` | decision | 2019-04-01 | Date in EA filename |
| `49cdaa3ff2e6c505c6822e8e9803eb9b` | decision | 2023-05-01 | Date in draft filename |
| `4af8ad4f47941e4ccb53fe4349c258c3` | decision | 1995-09-01 | Page 3 of FEIS |
| `00d09887554d7ab68e49e9ab628583bf` | decision | 2025-06-01 | Page 1 of DEIS |
| `8d13822f3d8b469efcdb2706caa463c7` | decision | 2022-03-01 | TVA Final EIS target date |

**TODO**: After Thursday, move these to a persistent override file and remove the hardcoded
block from `00_setup.R`. Also investigate the remaining 37 incomplete projects using
`inspect_candidates()` / `inspect_llm_prompt()` in `01_reviews.R`.

### 2026-02-04
- **Extraction complete**: 16 programmatic, 10 tiered, 1,390 standard
- Created `code/exploratory/reviews/01_reviews_eda.R` with summary stats, figures, and Google Sheet export
- Created Google Sheet for client validation (separate Programmatic/Tiered tabs)
- Updated `reports/deliverable02.qmd` with figures, examples, and validation link
- Investigated file name search — found 1 potential miss (`3621210fbd086bddbbf6fbedc1d6a488`)
- Confirmed "Programmatic Collaborations" (DOE grants) are in CE dataset, not EA/EIS
- **Next**: Client validation, timeline merge, add missing project

### 2026-01-30
- Created `extract_reviews.py` with 3-tier extraction approach
- Implemented title-based, regex, and LLM extraction tiers
- Added false positive filtering for EPA Tier standards, road tiers, etc.
- Fixed bug: "tiering from PEIS" incorrectly flagged as programmatic
- Started first full extraction (background, `--no-llm`)

---

## Files

- Script: `code/extract/extract_reviews.py`
- Output: `data/analysis/projects_reviews.parquet`
- EDA: `code/exploratory/reviews/01_reviews_eda.R`
- Report: `reports/deliverable02.qmd`
- This doc: `notes/status/reviews_status.md`

## Related Files

- `code/extract/extract_timeline.py` — Timeline extraction (for merge)
- `notes/project_overview.md` — Overall project deliverables
- `notes/status/timeline_status.md` — Timeline extraction status
