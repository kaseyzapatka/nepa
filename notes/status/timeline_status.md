# Timeline Extraction Status

**Last updated**: 2026-01-30

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
| `code/extract/extract_timeline.py` | Timeline extraction implementation (regex + LLM + hybrid + cache) |
| `code/extract/preprocess_documents.py` | LLM preprocessing for full-document extraction (legacy) |
| `data/analysis/projects_combined.parquet` | Combined project data (input) |
| `data/analysis/projects_timeline.parquet` | Regex-only timeline output |
| `data/analysis/regex_candidates.parquet` | Hybrid regex cache (new) |
| `notes/project_overview.md` | Project goals and deliverables |

---

## Quick Start

Regex-only extraction:
```bash
python extract_timeline.py --run --sample 100
```

Hybrid LLM extraction (fresh regex each run):
```bash
python extract_timeline.py --llm-run --hybrid --sample 20 --model llama3.2:3b-instruct-q4_K_M --timeout 180 --workers 4
```

Hybrid LLM extraction (cached regex):
```bash
python extract_timeline.py --regex-prep
python extract_timeline.py --llm-run --hybrid --use-regex-cache --sample 20 --model llama3.2:3b-instruct-q4_K_M --timeout 180 --workers 4
```

---

## Change Log

### 2026-01-30
- Added hybrid initiation/decision cue filtering.
- Switched hybrid context to sentence-based extraction with min-length expansion.
- Linked initiation cue sentence to next sentence date if needed.
- Added FR/CFR/USC + URL/OMB exclusions for hybrid contexts.
- Limited hybrid prompt to `decision | initiation | other`.
- Added single regex cache workflow via `--regex-prep` and `--use-regex-cache`.

### 2026-01-26
- Document type classification improvements in `code/extract/extract_data.py` (appendix detection, filename patterns, etc.).
