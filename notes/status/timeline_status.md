# Timeline Extraction Status

**Last updated**: 2026-01-30 (end of day)

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
