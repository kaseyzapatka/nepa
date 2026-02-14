# Timeline Extraction Status

**Last updated**: 2026-02-13

This document summarizes the current state of timeline extraction for the NEPA project. Read this file to understand timeline-related work without needing to explore the full codebase.

---

## Project Context

The goal is to construct timeline variables for NEPA projects to analyze how long environmental reviews take (Phase 2, Deliverable 4). The NEPATEC 2.0 dataset contains 60,000+ projects with millions of pages of text, but **no explicit date fields in the metadata**. Dates must be extracted from document text.

Key timeline deliverables from `notes/project_overview.md`:
- Timelines for CEs, EAs, and EISs segmented by year, pre/post-FRA, agency, project type
- Identify timeline outliers for case studies
- May need to cross-reference Federal Register NOIs for start dates

---

## What Is Currently Implemented

### File: `code/extract/extract_timeline.py`

#### Regex-based extraction (baseline)
- Parses multiple date formats (full/short month names, numeric slash/dash, ISO, digital signature, month-year).
- Filters dates to 1990–2030 for the baseline regex pass to avoid old law years.
- Deduplicates dates within each project.

#### Regex context detection (baseline)
- Classifies date context using nearby keywords:
  - `decision`: ROD, FONSI, approved, signed, issued
  - `start/submission/notice/draft/final/comment/scoping`
- Excludes dates near law/statute/citation references (keyword + citation pattern filters).

#### Document prioritization
- Prioritizes decision documents (ROD, FONSI, CE), then final > draft > other.
- Counts `project_document_count` and `project_main_document_count`.

#### Regex output fields
- `project_date_earliest`, `project_date_latest`, `project_date_decision`
- `project_duration_days`, `project_year`
- `project_timeline_needs_review` + `project_timeline_review_reasons`

---

## New Work (2026-01-30)

### Hybrid Regex + LLM (updated)
The hybrid approach now extracts **candidate dates only** for initiation/decision workflows, with stronger context handling:

1) **Initiation/Decision candidate filtering** (over-inclusive by design)
- Keeps dates only if their sentence context contains **decision** or **initiation** cues.
- Decision cues include: `signed`, `signature`, `digitally signed`, `approved`, `determination`, `decision memorandum`, `authorizing official`, `NEPA Compliance Officer`, `field office manager determination`, etc.
- Initiation cues include: `initiated`, `consultation`, `scoping`, `notice of intent`, `application received`, `submitted`, `prepared`, `revised`, `reviewed`, `document creation`, etc.
- This is intentionally inclusive to avoid missing initiation signals.

2) **Sentence-based context windows**
- Each date’s **full sentence** is used as context.
- If a sentence is too short, it expands with adjacent sentences to reach a minimum length (80 chars).
- If a sentence has initiation cues **but no date**, the next sentence’s date is linked to it.

3) **Citation / boilerplate exclusions** (hybrid-specific)
- Filters out contexts containing Federal Register/CFR/USC citations and obvious boilerplate URLs/OMB mentions.
- Does **not** remove “revised/reviewed/creation” dates (these can be useful for start vs approval analysis).

4) **Hybrid prompt narrowed**
- The LLM is asked to classify only:
  - `decision`
  - `initiation`
  - `other`

5) **Hybrid parsing**
- `initiation` is treated as the “application/start” date in output fields.
- Decision date still defaults to **latest** decision date in the classified set (a known issue; see below).

---

## Regex Cache (new)

To avoid re-running regex extraction for every LLM sample, a **single reusable cache** was added.

### Commands
1) Build cache once:
```bash
python extract_timeline.py --regex-prep
```
- Saves to `data/analysis/regex_candidates.parquet` (default).

2) Use cache during hybrid LLM runs:
```bash
python extract_timeline.py --llm-run --hybrid --use-regex-cache --sample 20 --model llama3.2:3b-instruct-q4_K_M --timeout 180 --workers 4 --output test20_hybrid.parquet
```

### Cache contents
- `project_id`, `date`, `match`, `context`, `position`, `position_pct`

### Behavior
- If `--use-regex-cache` is set and cache exists, the hybrid LLM skips regex extraction and does **not** rebuild page text for each project.

---

## BERT Classifier Approach (NEW - 2026-01-30)

### Why BERT?

The LLM approach works but is **slow**:
- 20 projects = 3.5 minutes
- 20,000 projects = ~58 hours (2.5 days)

BERT offers **50-100x speedup**:
- 20 projects = ~2-5 seconds
- 20,000 projects = ~30-60 minutes

### How It Works

1. **Weak Supervision (Auto-Labeling)**
   - Uses existing regex patterns to auto-label thousands of date contexts
   - Decision patterns: `digitally signed by`, `NEPA Compliance Officer`, `authorizing official`, etc.
   - Initiation patterns: `scoping meeting`, `application received`, `notice of intent`, etc.
   - Other patterns: `map created`, CFR/USC citations, etc.
   - No manual labeling required

2. **BERT Training**
   - Downloads `distilbert-base-uncased` from Hugging Face (~250MB, cached locally)
   - Fine-tunes on auto-labeled NEPA data
   - 3-class classifier: decision / initiation / other
   - Training takes ~5-10 minutes

3. **Fast Inference**
   - Classifies date contexts in batches (~5ms per context vs ~500ms for LLM)
   - Uses regex cache (same as hybrid LLM approach)
   - Outputs same format as LLM approach for easy comparison

### Implementation Added

New functions in `extract_timeline.py`:
- `auto_label_context()` - Pattern-based weak supervision
- `generate_bert_training_data()` - Creates training dataset from regex cache
- `train_bert_classifier()` - Fine-tunes DistilBERT
- `BertDateClassifier` - Inference class
- `extract_with_bert()` - Drop-in replacement for LLM classification
- `run_bert_timeline_extraction()` - Full pipeline

New CLI arguments:
- `--bert-generate` - Generate training data
- `--bert-train` - Train classifier
- `--bert-run` - Run extraction with BERT
- `--bert-model` - Choose base model (default: distilbert-base-uncased)
- `--epochs` - Training epochs (default: 3)

### Current Status (End of Day 2026-01-30)

**Training completed.** Model saved to `models/timeline_classifier/`.

**20-sample evaluation completed.** Results in `data/analysis/test20_bert.parquet`.

---

### BERT Evaluation Results

| Metric | BERT | LLM | Notes |
|--------|------|-----|-------|
| Decision coverage | **85%** (17/20) | 80% (16/20) | BERT slightly better |
| Initiation coverage | **0%** (0/20) | 35% (7/20) | BERT fails completely |
| Decision agreement | 12/13 (92%) | - | Where both found dates |

**Decision quality analysis (17 classified):**
- ✅ ~9 clearly correct (53%) - signature blocks, NEPA Compliance Officer, digitally signed
- ⚠️ ~4 questionable (24%) - "Date Determined" without full context
- ❌ ~4 false positives (24%) - form boilerplate ("Revised:", "Form Approved")

**Example good classifications:**
```
"NEPA Compliance Officer: STEPHEN WITMER Digitally signed by STEPHEN WITMER Date: 2023.08.14"
"Signed By: Casey Strickland NEPA Compliance Officer Date: 11/23/2022"
"ORO NEPA Compliance Officer Gary S. Hartman Date Determined: 6/29/2011"
```

**Example false positives:**
```
"NETL F 451. 1-1/1 Revised: 11/24/2014 Reviewed: 11/24/2014 (Previous Editions Obsolete)"
"DOE F 1325. 8e Electronic Form Approved by Forms Mgmt. 04/19/2006"
```

---

### Training Data Imbalance (Root Cause of Issues)

| Label | Count | % |
|-------|-------|---|
| decision | 15,250 | 89% |
| other | 1,810 | 10.5% |
| initiation | **122** | **0.7%** |

Only 122 initiation examples vs 15,250 decision examples. The model never learned initiation patterns.

---

## Monday Pickup Instructions

### Priority 1: Create Training Data for Initiation

The model has only 122 initiation examples (0.7%) - it never learned what initiation looks like.

**Option A: Expand initiation patterns** (in `extract_timeline.py` → `INITIATION_PATTERNS_STRONG`)
```python
# Add more patterns like:
r'proposed action',
r'project initiat',
r'environmental review began',
r'review process started',
r'eis process',
r'ea preparation',
```

**Option B: Add negative patterns to exclude form boilerplate** (in `OTHER_PATTERNS_STRONG`)
```python
# Add patterns to catch form templates:
r'previous editions obsolete',
r'form approved',
r'forms mgmt',
r'netl f \d+',
r'doe f \d+',
```

**Option C: Manually label ~50-100 initiation examples**
- Look at LLM results where `llm_application_date` was found
- Extract those contexts and verify they're correct
- Add to training data with `label='initiation'`

**Option D: Use class weighting during training**
- Modify `train_bert_classifier()` to weight initiation class higher
- This helps the model pay more attention to rare classes

### Priority 2: Use Larger BERT Model

Since BERT is so fast (~5ms/context), we can afford a better model:

```bash
# Retrain with RoBERTa (often better for classification)
python extract_timeline.py --bert-train --bert-model roberta-base --epochs 5

# Or full BERT
python extract_timeline.py --bert-train --bert-model bert-base-uncased --epochs 5
```

### Priority 3: Regenerate and Retrain

After updating patterns:
```bash
cd /Users/Dora/git/consulting/nepa/code/extract

# 1. Regenerate training data with new patterns
python extract_timeline.py --bert-generate

# 2. Check new label distribution
python -c "import pandas as pd; df=pd.read_parquet('../../data/analysis/bert_training_data.parquet'); print(df['label'].value_counts())"

# 3. Retrain with better model
python extract_timeline.py --bert-train --bert-model roberta-base --epochs 5

# 4. Test on sample
python extract_timeline.py --bert-run --sample 20 --output test20_bert_v2.parquet
```

### Priority 4: Full Run (if results look good)

```bash
python extract_timeline.py --bert-run --output projects_timeline_bert.parquet
```
Expected runtime: ~30-60 minutes for 20K projects

---

## Known Issues / Gaps (Current)

1) **Decision date selection can be wrong**
- When multiple decision dates are classified, the current logic picks the **latest**.
- Example: project `3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0` had a true signature date labeled as decision, but the “latest decision” was a weaker context (e.g., location header), causing the wrong decision date to be selected.
- Potential fix: choose decision dates with strong signature cues over “latest by date.”

2) **Initiation candidates may include prep/revision dates**
- This is intentional for now (to keep possible “start” signals), but it can cause initiation mislabels (e.g., map creation).

3) **Hybrid approach still depends on LLM consistency**
- The model sometimes labels non-signature contexts as decision.
- It can also label “document creation / revised” contexts as initiation because those cues were intentionally included.

4) **Initiation cue linking can still miss long-distance references**
- The current rule links “initiation cue sentence → next sentence date.” If the date appears much later (multiple sentences away), it can still be missed.
- Potential fix: allow linking across a small sentence window (e.g., next 2–3 sentences).

5) **Decision vs initiation disambiguation lacks heuristic weighting**
- There is no post-LLM scoring to prefer strong decision cues (signature blocks) over weak cues (headers/locations).
- Potential fix: rank decision candidates by cue strength and choose best, not latest.

6) **Hybrid LLM runs can be slow (LLM generation is the bottleneck)**
- Even with regex caching, per-project LLM latency dominates runtime.
- Speed levers already applied: lower context length (80), lower `num_predict` (256), parallel workers (default 4).

---

## Decisions Made (So Far)

- Adopted a **hybrid regex + LLM** approach focused on **initiation + decision** only.
- Switched to **sentence-based context** with min-length expansion and initiation→next-sentence linking.
- Added **citation/boilerplate filters** (FR/CFR/USC, URLs, OMB) while keeping revised/creation dates.
- Added a **single regex cache** (`data/analysis/regex_candidates.parquet`) to avoid re-running regex.
- Reduced hybrid token load: context length **80**, `num_predict` **256**.
- Switched default model to **`llama3.2:3b-instruct-q4_K_M`** for speed.
- Added **parallelization** with `--workers` (default 4).
- Added **context de-duplication** to prevent duplicate signature sentences from creating extra dates.

---

## Decisions Still Open

- **Decision date selection**: still chooses the latest decision date; should move to cue-strength ranking.
- **Initiation strictness**: keep inclusive cues (prepared/revised/creation) or tighten to explicit initiation verbs.
- **Skip-LLM logic**: decide when to auto-assign decision from strong signature blocks and bypass LLM.
- **Prompt examples**: add structured positive/negative examples + required short quote to reduce hallucinations.
- **Parallel workers**: test stability at 6 workers (current default is 4).
- **Minimal validation set**: decide on 20–30 projects for manual ground truth.

---

## Run Comparisons (Latest)

Comparison: `test20_workers.parquet` vs `test20_hybrid3_instruct.parquet` (both `llama3.2:3b-instruct-q4_K_M`):
- Decision coverage: **85% new vs 90% old**
- Initiation coverage: **35% vs 35%**
- Agreement: decision dates matched **16/20**, initiation matched **6/20**
- New run produced fewer total labels (more conservative).
- Parallelization speedup not measured (no timing logs yet).

---

## Parallelization Notes

- Current default in `extract_timeline.py`: `--workers 4`.
- Safe range discussed: **4–6** parallel requests (test with a 10-project sample).
- To test stability at 6 workers, run a 10-project sample and check for timeouts/empty decisions.

---

## Suggested Next Steps (Actionable)

1) **Decision cue ranking**: prioritize dates with signature cues over “latest by date.”
2) **Initiation tightening**: restrict initiation cues to explicit verbs for LLM classification.
3) **Skip-LLM rule**: if exactly one strong signature candidate exists, set decision directly.
4) **Prompt examples + quote requirement**: add short YES/NO examples; require a 5–10 word quote.
5) **Parallelization test**: run a 10-project test with `--workers 6` and compare time/Errors.
6) **Add timing logs**: print total elapsed time + avg sec/project for real speed tracking.
7) **BERT speed-up**: `--bert-run` currently runs single-threaded even though it accepts `--workers`. Add batching and/or multiprocessing for BERT inference to improve throughput on full runs.

---

## Updates Added (2026-02-02)

1) **New timeline class: `review` (BERT + hybrid)**  
   - **Change:** Added `review` as a third class alongside `initiation` and `decision`; summary now includes earliest/latest review and review count.  
   - **Reason:** Interim approvals and phase approvals were being mislabeled as decisions; review dates are valuable for timeline structure and for initiation backfill when explicit initiation is missing.

2) **Decision selection ranking (cue strength + confidence)**  
   - **Change:** Decision date now chosen by cue strength + confidence, not just latest date; boilerplate penalty added; footer/header position only used when top candidates are close in score but far apart in position.  
   - **Reason:** Prevents form boilerplate or weak header dates from overriding true signature decisions.

3) **Dynamic context window sizing**  
   - **Change:** Smaller contexts for strong signature cues, larger contexts for review/initiation cues.  
   - **Reason:** Avoids over-capturing noise around signatures while preserving enough context for weak review cues.

4) **Keep-all regex cache (default)**  
   - **Change:** Regex cache now keeps all dates by default (cue filtering moved downstream); `--regex-filtered` restores legacy behavior.  
   - **Reason:** Dates in tables/lists were being dropped before BERT could see them (e.g., 1/7/2021, 1/11/2021, 1/26/2021).

5) **Boilerplate context merging for decisions**  
   - **Change:** If a date’s sentence is boilerplate, expand context to adjacent sentence with decision cues (signature blocks).  
   - **Reason:** Prevents Recovery Act/boilerplate sentences from masking nearby signature lines.

6) **Full context preserved in `source`**  
   - **Change:** BERT `source` now stores full context rather than truncating to 100 chars.  
   - **Reason:** Allows direct validation of dates (e.g., include “5515-001 on November 30, 2011.”).

7) **Memo DATE fallback to review (when no decision cues)**  
   - **Change:** If no decision cues exist, memo “DATE:” lines are treated as review; DOE form boilerplate is forced to `other`.  
   - **Reason:** Environmental clearance memos often have no signatures; memo date is the best available review proxy and form approval dates should be excluded.

8) **Context cleaning for BERT classification**  
   - **Change:** Added `clean_context()` to strip boilerplate (Recovery Act checkbox lines, DOE/NETL form headers, OMB/PRA lines) before BERT classification. Both original and cleaned context are stored in output, with a `context_cleaned_flag`.  
   - **Reason:** Boilerplate was masking signature lines, causing “Approved by … NEPA Compliance Officer” to be labeled `other`.

9) **Use cleaned context for rule overrides**  
   - **Change:** Decision/initiator override checks now use cleaned context when available (fallback to original if cleaned is empty).  
   - **Reason:** Ensures overrides are based on the meaningful text rather than boilerplate noise.

10) **Recovery Act cleaning made non-destructive**  
   - **Change:** Recovery Act checkbox line now gets stripped without removing trailing signature text (e.g., “Approved by …”).  
   - **Reason:** Prior regex removed entire line, leaving empty cleaned context and preventing decision classification.

11) **Model comparison: v6 vs v8 (test50)**  
   - **Decision rate:** v6 **84.0%** → v8 **84.0%** (no change)  
   - **Initiation rate:** v6 **2.0%** → v8 **16.0%** (improved)  
   - **Inferred initiation:** v6 **26.0%** → v8 **36.0%** (improved)  
   - **Net decision shifts:** 1 lost, 1 gained (overall stable).

12) **Known gap remains: Recovery‑Act signature lines**  
   - **Finding:** 7 projects with “Recovery Act … Approved by … NEPA Compliance Officer” still have **no decision** in both v6 and v8.  
   - **Implication:** Decision capture is stable, but these are consistent false negatives.

13) **Next targeted fix (recommended)**  
   - **Action:** Implement a narrow Recovery‑Act signature extraction that keeps the signature tail when “Recovery Act” and signature cues appear on the same line.  
   - **Goal:** Flip those 7 misses to decisions without hurting overall decision precision.

## Updates Added (2026-02-12)

Reviewed 11 misclassified projects from the BERT full run (`projects_timeline_bert.parquet`) and identified 7 systematic error patterns. Implemented 9 targeted fixes in `extract_timeline.py` to address them. These fixes affect both BERT training data (via auto-labeling patterns) and post-BERT guardrails/scoring.

### Projects Reviewed

| Project ID | Issue |
|---|---|
| `1df6f8b5` | "Initial and Date:" specialist sign-off misread as initiation |
| `5ec95c90` | "AUTHORITY AND APPROVAL" not winning as final decision |
| `58cab57e` | "Form Status: Approved" caught by too-broad boilerplate pattern |
| `8de424f4` | Historical gap rule only fires once (single-pass) |
| `e74f6ef2` | "reviewed" in INITIATION_CUES causing misclassification |
| `b523e342` | YYYY.MM.DD timestamp not recognized as digital signature; "Approval and Contact Information" not a strong decision cue |
| `cec29e92` | Initiation date after decision not hard-rejected |
| `3e3bb9f5` | "expire on [date]" not caught for past dates; "Initial and Date:" issue |
| `6149175c` | ROW application as initiation (appears correct) |
| `5c0911d5` | "Date Determined" winning over digital signature date |
| `e0f39636` | "District Manager" absent from decision patterns; no other→decision guardrail |

### Systematic Patterns Found

- **A.** "Initial and Date:" on BLM specialist checklists misread as initiation
- **B.** Missing decision-maker patterns (District Manager, Approval and Contact Information, YYYY.MM.DD timestamps)
- **C.** No other→decision guardrail — BERT "other" not corrected even when strong decision cue exists
- **D.** Expiration detection gated on >2025 year — misses "expire on [past date]"
- **E.** Historical gap rule finds first gap only — misses multi-cluster projects
- **F.** Initiation after decision only penalized (-3), not rejected
- **G.** "reviewed" in INITIATION_CUES causes false initiation labels

### Fixes Implemented (9 of 10 proposed)

1. **"Initial and Date:" → review** — Added `initial and date` and `initials?\s*&\s*date` to `REVIEW_PATTERNS_STRONG` and `INITIATION_EXCLUSION_PATTERNS`. (Patterns A)

2. **Missing decision patterns** — Added `district manager`, `approval and contact information`, `\d{4}\.\d{2}\.\d{2}` (YYYY.MM.DD timestamps) to `DECISION_PATTERNS_STRONG`. (Pattern B)

3. **other→decision guardrail** — New guardrail in `extract_with_bert()`: if BERT classifies as "other" but context has a strong decision cue, reclassify to "decision". (Pattern C)

4. **Expiration detection expanded** — `_is_expiration_candidate()` now fires on expiration language cues regardless of date, not just for dates after 2025-12-31. (Pattern D)

5. **Historical gap rule: last gap wins** — `_apply_historical_gap_rule()` now finds the LAST gap > 730 days (not the first), marking all dates before it as historical. Catches multi-cluster projects. (Pattern E)

6. **Hard-reject initiation after decision** — `_select_best_initiation()` now skips (continues past) candidates with dates after the decision date instead of penalizing by -3. (Pattern F)

7. *(Skipped)* **Dedupe by date before type** — Not implemented. Would change dedupe key from `(date, type)` to `(date)`. Risk: loses legitimate same-day multi-event entries. Other fixes likely address the edge cases that motivated this.

8. **"AUTHORITY AND APPROVAL" → tier 4** — `_decision_strength()` now returns 4 for `authority and approval` and `determination and approval`, above the tier 3 default for strong patterns. (Pattern B)

9. **Tighten "form approved" boilerplate** — Changed `r'form approved'` to `r'form approved\s*(omb|omg)'` in `DECISION_BOILERPLATE_PATTERNS` so "Form Status: Approved" is not penalized as boilerplate. (Pattern related to `58cab57e`)

10. **Remove "reviewed" from INITIATION_CUES** — Deleted `'reviewed'` from the hybrid `INITIATION_CUES` list. This cue belongs in review contexts, not initiation. (Pattern G)

### Impact Assessment

- **Fixes 1, 2, 10** affect auto-labeling (`auto_label_context()`) — require `--bert-generate` + `--bert-train` to take full effect
- **Fixes 3, 4, 5, 6, 8, 9** are post-BERT guardrails/scoring — take effect immediately on next `--bert-run`
- Highest-impact fixes: **1, 2, 3, 6** (address the most projects)

### Next Steps

1. Regenerate BERT training data: `python extract_timeline.py --bert-generate`
2. Retrain BERT model: `python extract_timeline.py --bert-train`
3. Test on the 11 misclassified projects: `python extract_timeline.py --bert-run --sample 50 --output test50_bert_v9.parquet`
4. If results improve, full run: `python extract_timeline.py --bert-run --output projects_timeline_bert.parquet`

---

## Updates Added (2026-02-13) — EA/EIS Timeline Construction

Extended the timeline pipeline to support EA and EIS projects (previously CE-only), added Claude API as an LLM adjudication provider, and implemented tiered decision selection to improve decision date quality.

### Multi-source support (CE, EA, EIS)

- **`--source` CLI arg**: `--regex-prep --source EA`, `--bert-run --source EIS`, etc. Per-source regex caches saved to `regex_candidates_ea.parquet`, `regex_candidates_eis.parquet`.
- **Per-source expected counts**: `EXPECTED_CLEAN_COUNTS = {'CE': 19399, 'EA': 573, 'EIS': 753}` with tolerance of 50. Warns if actual count drifts.
- **Clean energy hardcoded**: `--regex-prep` and `--bert-run` always filter to clean energy projects (no need for `--clean-energy` flag).
- **Per-document regex processing**: `run_regex_prep()` now iterates documents individually instead of concatenating all pages, tagging each candidate with `doc_type` (FONSI, ROD, EA, DEA, etc.).
- **main_document only**: Only `main_document == 'YES'` documents are parsed for date candidates (both at page and document level).

### EA/EIS-specific BERT post-processing

- **ROD/FONSI doc-type boost**: +3 score for candidates from ROD/FONSI documents in `_select_best_decision()`.
- **Post-BERT FONSI/ROD override**: Forces `decision` label when EA/EIS decision language (FONSI, ROD, decision record) is present in context, even if BERT classified differently.
- **Historical year exemption**: Dates with FONSI/ROD context no longer overridden to `historical` by the pre-2000 year rule.

### LLM adjudication pipeline (Regex → BERT → LLM)

Three-stage pipeline where the LLM sees all candidates together and picks best initiation/decision:

1. **Regex** extracts date candidates with context from document text
2. **BERT** classifies each candidate (decision/initiation/review/other/historical/expiration)
3. **LLM** reads all candidates + BERT classifications + context and adjudicates

New CLI:
```bash
python extract_timeline.py --llm-adjudicate --input test50_ea.parquet --provider claude
python extract_timeline.py --llm-adjudicate --input test50_ea.parquet --provider ollama --model llama3.2:3b-instruct-q4_K_M
```

### Claude API support

- **Provider flag**: `--provider claude` (default: ollama). Uses `claude-haiku-4-5-20251001` by default.
- **API key**: reads `ANTHROPIC_API_KEY` environment variable (not hardcoded).
- **Rate limit handling**: retries on 429 with `retry-after` header, up to 3 retries.
- **Sequential by default**: Claude provider forces `workers=1` to respect rate limits.
- **Cost**: ~$0.78 for all 573 EA projects (Haiku pricing).

### Candidate noise filtering (`_filter_candidates_for_llm`)

Layered filtering to reduce noise before sending to the LLM:

| Layer | Filter | Purpose |
|-------|--------|---------|
| 1 | Dedup by date | Same date appearing Nx → keep best, note count |
| 2 | Confidence floor (0.3) | Drop BERT's worst guesses; exempt FONSI/ROD/EA/DEA |
| 3 | Smart type filter | Keep decision/initiation always; review only if conf >= 0.5 or has cues; drop other |
| 4 | Doc-type priority | Stricter thresholds for blank/unknown doc types |
| 5 | Tiered decision gating | See below |
| 6 | Hard cap at 50 | Absolute bound on prompt size |

### Tiered decision selection (newest)

Explicit gating so the LLM only considers appropriate decision candidates:

- **Tier A** (priority): FONSI, ROD, DR, Decision Record → if any decision candidates exist from these, only these are allowed.
- **Tier B** (fallback): EA, DEA → used only when Tier A is empty, and only if context has strong decision language (signed, approved, issued, finding, FONSI, determination) AND no exclusion patterns (draft, comment period, scoping, NOI).

The LLM prompt explicitly states the mode and allowed date set:
- `priority_only`: "You MUST pick the decision date from this set ONLY: [dates]"
- `ea_dea_fallback`: "No FONSI/ROD exists, pick from this limited set"
- `no_decision_candidates`: "You MUST respond with null"

**Post-parse guardrail**: if the LLM returns a decision date not in the allowed set, it's nulled out with `llm_adj_error = 'decision_not_in_allowed_tier'`.

### Diagnostic output columns

| Column | Description |
|--------|-------------|
| `llm_decision_date` | LLM's best decision date |
| `llm_initiation_date` | LLM's best initiation date |
| `llm_decision_reasoning` | Brief explanation |
| `llm_initiation_reasoning` | Brief explanation |
| `llm_decision_mode` | `priority_only`, `ea_dea_fallback`, or `no_decision_candidates` |
| `llm_n_priority_decision_candidates` | Count of Tier A candidates |
| `llm_n_eadea_fallback_candidates` | Count of Tier B candidates |
| `llm_adj_n_candidates` | Total candidates sent to LLM |
| `llm_adj_error` | Error or guardrail flag |
| `llm_adj_prompt` | Full prompt sent to LLM |

### BERT vs Claude accuracy (50-project EA test)

Spot-check of 10 projects showed Claude was more accurate than BERT for EA decisions:
- Claude correct in all 10 cases where it had an opinion
- BERT picked non-NEPA events in 4/10 cases (tribal agreements, FEMA revisions, wetland verifications)

### Known gaps (EA/EIS)

- Only 11% of EA projects have FONSI/ROD documents — 89% rely on EA text for decision dates
- 8 FONSI documents were previously excluded due to `main_document` flag (now filtered at doc level)
- Filename dates could serve as fallback but are 90% year-only precision (useful for sanity checks, not day-level dates)

### Workflow (EA)

```bash
# 1. Build regex cache
python extract_timeline.py --regex-prep --source EA

# 2. BERT classification
python extract_timeline.py --bert-run --source EA --sample 50 --output test50_ea.parquet

# 3. Claude adjudication
python extract_timeline.py --llm-adjudicate --input test50_ea.parquet --provider claude

# 4. Full run
python extract_timeline.py --bert-run --source EA --output projects_timeline_bert_ea.parquet
python extract_timeline.py --llm-adjudicate --input projects_timeline_bert_ea.parquet --provider claude
```

---

## File References

| File | Purpose |
|------|---------|
| `code/extract/extract_timeline.py` | Timeline extraction implementation (regex + LLM + hybrid + BERT) |
| `code/extract/preprocess_documents.py` | LLM preprocessing for full-document extraction (legacy) |
| `data/analysis/projects_combined.parquet` | Combined project data (input) |
| `data/analysis/projects_timeline.parquet` | Regex-only timeline output |
| `data/analysis/regex_candidates.parquet` | Hybrid regex cache |
| `data/analysis/bert_training_data.parquet` | Auto-labeled training data for BERT |
| `data/analysis/test20_workers.parquet` | LLM hybrid results (20 sample) |
| `data/analysis/test20_bert.parquet` | BERT results (20 sample) - to be created |
| `models/timeline_classifier/` | Trained BERT model - to be created |
| `notes/project_overview.md` | Project goals and deliverables |

---

## Quick Start

Regex-only extraction:
```bash
python extract_timeline.py --run --sample 100
```

Hybrid LLM extraction (cached regex):
```bash
python extract_timeline.py --regex-prep
python extract_timeline.py --llm-run --hybrid --use-regex-cache --sample 20 --model llama3.2:3b-instruct-q4_K_M --timeout 180 --workers 4
```

**BERT extraction (recommended - 50-100x faster):**
```bash
# One-time setup:
pip install numpy==1.26.4 transformers datasets torch
python extract_timeline.py --regex-prep        # Build regex cache (if not done)
python extract_timeline.py --bert-generate     # Generate training data
python extract_timeline.py --bert-train        # Train classifier (~5-10 min)

# Run extraction:
python extract_timeline.py --bert-run --sample 20 --output test20_bert.parquet
python extract_timeline.py --bert-run --output projects_timeline_bert.parquet  # Full run
```

---

## Change Log

### 2026-02-14 - EIS Cue + Tiering Enhancements
- **Expanded EIS decision regex cues** in `extract_timeline.py`:
  - Added variants like `joint record of decision`, `record of decision ... signed/issued`, `selected alternative`, `decision to implement`
- **Expanded EIS initiation regex cues**:
  - Added variants like `intent to prepare ... environmental impact statement`, `notice of intent ... EIS`, `NOI ... federal register`, `scoping notice`
- **Expanded hybrid cue library for EIS**:
  - Added EIS-oriented decision cues in `DECISION_CUES` (ROD/joint ROD/selected alternative)
  - Added EIS-oriented initiation cues in `INITIATION_CUES` (NOI published, federal register, scoping notice)
- **Generalized LLM fallback logic from EA-only to EA+EIS docs**:
  - Added fallback decision doc set: `EA`, `DEA`, `DEIS`, `FEIS`, `FEA`, `EIS`
  - Decision mode now supports:
    - `priority_only` (FONSI/ROD/DR/Decision Record candidates only)
    - `ea_eis_fallback` (EA/DEA/DEIS/FEIS fallback only when Tier A is empty)
    - `no_decision_candidates`
- **Tightened fallback decision filtering**:
  - Requires strong decision language for fallback candidates
  - Added non-decision exclusions for draft/public review signals (including EIS-specific noise like `draft EIS`, `notice of availability`, `public hearing`, comment-period language)
- **Prompt updates for EIS-aware adjudication**:
  - Updated decision-mode instructions and allowed decision-date set text
  - Updated initiation guidance to include EIS initiation context (`DEIS`/NOI/scoping)
- **Guardrail enforcement remains strict**:
  - If LLM returns a decision date outside allowed tier set, date is nulled and flagged with `decision_not_in_allowed_tier`
- **Regex prep doc typing improved**:
  - `run_regex_prep()` now prefers `document_type_clean` when available (falls back to raw `document_type` otherwise) to improve tiering quality for EIS docs with weak raw type labels
- **No change (intentional)**:
  - CLI hardcode behavior for `main_docs_only=True` in `--regex-prep` and `--bert-run` was not changed in this update
- **Validation**:
  - `python -m py_compile code/extract/extract_timeline.py` passed

### 2026-02-13 - EA/EIS Multi-Source + Claude LLM Adjudication
- **Extended pipeline to EA and EIS** with `--source` CLI arg, per-source regex caches, per-source expected counts
- **Per-document regex processing** with `doc_type` tagging (FONSI, ROD, EA, DEA, etc.)
- **main_document filtering** enforced at both page and document level
- **ROD/FONSI scoring boost** (+3) and post-BERT override for EA/EIS decision language
- **LLM adjudication pipeline**: regex → BERT classify → LLM adjudicate (`--llm-adjudicate`)
- **Claude API support** (`--provider claude`) with rate-limit retry and env var API key
- **Candidate noise filtering** (`_filter_candidates_for_llm`): dedup, confidence floor, type filtering, doc-type priority, hard cap at 50
- **Tiered decision selection**: Tier A (FONSI/ROD/DR) preferred; Tier B (EA/DEA) fallback only with strong decision language
- **Post-parse guardrail**: nulls LLM decision if not in allowed tier set
- **Diagnostic columns**: `llm_decision_mode`, `llm_n_priority_decision_candidates`, `llm_n_eadea_fallback_candidates`
- **Prompt enhancements**: explicit decision mode, allowed date set, document priority guidance
- **Graceful handling** of missing `document_date_from_file_name` column in EA/EIS data

### 2026-02-12 - BERT v9 Guardrail Fixes
- **Reviewed 11 misclassified projects** from BERT full run, identified 7 systematic error patterns
- **Implemented 9 fixes** in `extract_timeline.py` targeting misclassification root causes
- Added "Initial and Date:" to review patterns + initiation exclusions
- Added missing decision patterns: district manager, approval and contact information, YYYY.MM.DD timestamps
- New other→decision guardrail in `extract_with_bert()`
- Expiration detection no longer gated on >2025 year
- Historical gap rule now finds LAST gap (multi-cluster support)
- Hard-reject initiation dates after decision (was soft penalty)
- "AUTHORITY AND APPROVAL" boosted to tier 4 decision strength
- Tightened "form approved" boilerplate to OMB-only (`form approved\s*(omb|omg)`)
- Removed "reviewed" from `INITIATION_CUES`
- **Not implemented**: dedupe by date-only (risk of losing legitimate same-day events)
- **Next**: regenerate training data, retrain, re-run on sample + full

### 2026-01-30 (PM) - BERT Classifier
- **Added BERT-based classification as alternative to LLM** (50-100x faster)
- Implemented weak supervision using existing regex patterns for auto-labeling
- Added `--bert-generate`, `--bert-train`, `--bert-run` CLI commands
- Training data generated: 17,182 examples (decision: 15,250, other: 1,810, initiation: 122)
- Uses DistilBERT from Hugging Face (downloads automatically, ~250MB)
- **Trained model** saved to `models/timeline_classifier/`
- **Evaluated on 20 samples**: 85% decision coverage, 0% initiation coverage
- **Key finding**: Training data severely imbalanced - only 122 initiation examples (0.7%)
- **Key finding**: ~24% false positives from form boilerplate ("Revised:", "Form Approved")
- **Next steps**:
  1. Create more initiation training data (expand patterns or manual labeling)
  2. Add patterns to exclude form boilerplate
  3. Try larger model (roberta-base) since BERT is fast enough

### 2026-01-30 (AM) - Hybrid LLM
- Added hybrid initiation/decision cue filtering.
- Switched hybrid context to sentence-based extraction with min-length expansion.
- Linked initiation cue sentence to next sentence date if needed.
- Added FR/CFR/USC + URL/OMB exclusions for hybrid contexts.
- Limited hybrid prompt to `decision | initiation | other`.
- Added single regex cache workflow via `--regex-prep` and `--use-regex-cache`.

### 2026-01-26
- Document type classification improvements in `code/extract/extract_data.py` (appendix detection, filename patterns, etc.).
