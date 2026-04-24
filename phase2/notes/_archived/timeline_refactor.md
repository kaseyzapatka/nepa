# Timeline Extraction: Refactor Architecture

**Created:** 2026-03-19
**Scope:** Improving completion rates without modifying or overwriting validated existing timelines

---

## Problem Statement

The current regex → BERT → LLM-adjudicate pipeline produces defensible timelines but at low completion rates:

| Source | Complete | Primary Bottleneck |
|--------|----------|--------------------|
| CE | ~30% | Missing initiation (~70% of CE lacks it) |
| EA | ~62% | Missing decision (~25% lack FONSI/ROD date) |
| EIS | ~48% | Missing decision or both (~31% missing decisions) |

These rates cause problems for D2 (tiered/programmatic analysis), D3 (timeline distribution), D5 (pages analysis), and D6 (technology-specific timelines).

---

## Root Cause Analysis

### The fundamental constraint

From `notes/status/timeline_status.md`:

> `--llm-adjudicate` can only choose among BERT-provided candidates (`bert_dates_json`). It **cannot recover dates never extracted upstream.**

LLM adjudication is a **selection** step, not a **discovery** step. This is the central architectural gap.

### Per-source root causes

**CE — missing initiation:**
- CE documents are short (1–2 pages), informal, and often lack formal scoping/NOI language
- Initiation events (application received, RoW application submitted) may appear in prose with weak signal context — BERT classifies them as "Other" and they never become candidates
- Some CE documents genuinely do not include an initiation date

**EA/EIS — missing decision:**
- FONSI and ROD documents are often filed separately from the main EA/EIS and are not linked in NEPATEC (~25–31% of EA/EIS missing decisions)
- Many main documents contain indirect references ("the FONSI was signed in August 2018") that regex/BERT misses or underclassifies
- This is partly a data availability problem (separate documents) and partly a classification problem (indirect references)

**EA/EIS — BERT underfit:**
- 573 EA + 753 EIS training examples vs. 19,399 CE examples
- 3x oversampling helps but is insufficient for EA/EIS-specific vocabulary (ROD language, DEA/FEIS document types, joint ROD variants)

### Why a better classifier alone won't solve it

Upgrading from DistilBERT to a larger model (DeBERTa, RoBERTa) improves classification of dates that are present but misclassified. It cannot recover:
- Dates absent from the document corpus
- Dates in separate unlinked documents (many EA/EIS decisions)

A better classifier reduces the problem but doesn't eliminate it.

### Why a QA model is not the right tool here

Extractive QA models (e.g., `deepset/deberta-v3-base-squad2`) would need the same DuckDB page-loading infrastructure as an LLM recovery pass, but with lower accuracy and more local infrastructure complexity. Claude Haiku achieves better date extraction accuracy at comparable or lower per-project cost.

---

## Recommended Architecture: Three-Tier Strategy

### Tier 1 — Recovery LLM pass (highest leverage, low cost)

**New CLI mode:** `--recovery-run` in `code/extract/extract_timeline.py`

**Core idea:** Instead of selecting from BERT candidates, send actual document page text to Claude Haiku and let it freely discover dates. This is a discovery step, not a selection step.

**Logic:**
1. Load BERT/adjudicated output parquet; filter to `timeline_status in ['missing_initiation', 'missing_decision', 'no_dates']`
2. Apply `--source` and `--sample` filters (same CLI pattern as existing modes)
3. Per project, load pages via DuckDB (reuse `_load_project_pages_bulk()`); keyword-score pages:
   - Missing initiation → score for: `received | submitted | application | NOI | scoping | notice of intent | initiated`
   - Missing decision → score for: `signed | approved | decision | FONSI | ROD | determination | finding of no significant | record of decision`
   - Select top 3 scoring pages + always include page 1 (cover pages hold application dates); cap total at 3000 chars
4. Call Claude Haiku with structured extraction prompt (see below); request only what is missing per project
5. Parse JSON response → `recovery_initiation_date`, `recovery_decision_date`, `recovery_confidence`, `recovery_context`
6. Write to separate output parquet with `recovery_run_at` audit timestamp

**Prompt structure:**
```
You are extracting NEPA review timeline dates from a [CE/EA/EIS] document.

[page text, ≤3000 chars]

Extract (set null if not found):
- initiation_date: When was the NEPA review initiated? Look for: application received,
  NOI published, scoping meeting, RoW application submitted. Return YYYY-MM-DD.
- decision_date: When was a formal decision made? Look for: CE determination signed,
  FONSI signed, ROD signed, approval date. Return YYYY-MM-DD.

Return JSON only:
{"initiation_date": "...", "decision_date": "...", "initiation_context": "...", "decision_context": "..."}
```

**Key design principles:**
- For a project missing only initiation, tell the model `decision_date: null` to reduce hallucination
- Reuse same worker/retry pattern as `--llm-adjudicate` (workers=1 for Claude; retry 429 and 529)
- Recovery output is additive only — never overwrites existing BERT/LLM dates
- Fallback hierarchy in `03_timeline.R`: `bert/llm date → recovery date → noi_publication_date`

**Estimated cost:** ~$3–8 for full CE + EA + EIS recovery pass (CE initiation is the bulk volume at ~13,000 projects × ~500 tokens at Haiku pricing)

### Tier 2 — Source-specific DeBERTa models (better upstream classification)

**Why:** DeBERTa-v3-base generalizes substantially better than DistilBERT on small training sets. With 573 EA and 753 EIS examples, the existing combined BERT model is systematically biased toward CE document vocabulary. A source-specific DeBERTa model better learns EA/EIS-specific signals (ROD language, FEIS document types, FONSI phrasing).

**Implementation:**
- Add `--model-type {distilbert, deberta}` to `--bert-train` and `--bert-run`
- Train source-specific models to separate directories:
  - `models/timeline_classifier_ce/`
  - `models/timeline_classifier_ea/`
  - `models/timeline_classifier_eis/`
- `--bert-run` dispatches to source-specific model when available, falls back to combined model
- Uses `microsoft/deberta-v3-base` from HuggingFace (note: requires SentencePiece tokenizer — `pip install sentencepiece`)
- Training data: same weak supervision labels + manual corrections; DeBERTa is slower to train but inference is acceptable

**Priority order:** EA first (largest gap for smallest dataset), then EIS.

**Effect:** Reduces how many dates are misclassified as "Other" before reaching LLM adjudication — shrinks Tier 1 workload over time without changing Tier 1's design.

### Tier 3 — Priority-project targeted batch (fallback for critical deliverables)

If Tier 1 + 2 still leave specific deliverables below threshold, run targeted high-quality extraction for:

| Deliverable | Filter | Config |
|-------------|--------|--------|
| D2 tiered/programmatic | `project_is_tiered == TRUE \| project_is_programmatic == TRUE` | Claude Sonnet, 8k chars |
| D6 technology-specific | `project_is_transmission \| project_is_geothermal \| project_is_pipeline` | Claude Sonnet, 8k chars |

Same `--recovery-run` mechanism but larger context window and higher-accuracy model.

---

## Files to Modify

| File | Change |
|------|--------|
| `code/extract/extract_timeline.py` | Add `--recovery-run` mode, `run_recovery_pass()` function, page keyword scoring, structured API call, DeBERTa support in `--bert-train`/`--bert-run` |
| `code/deliverable03/03_timeline.R` | Add recovery parquet join, fallback hierarchy (bert → recovery → noi_publication_date) |
| `notes/status/timeline_status.md` | Document `--recovery-run` mode and output schema |

## Code to Reuse

| Existing function | Reused in |
|-------------------|-----------|
| `_load_project_pages_bulk()` | Recovery pass page loading |
| `_call_claude_adjudication()` / retry logic | Recovery pass API calls |
| `--sample`, `--source`, `--output` arg pattern | Recovery pass CLI |
| `noi_publication_date` fallback | Confirm active in `03_timeline.R` merge |

---

## Expected Impact

| Source | Current | Expected after Tier 1 |
|--------|---------|----------------------|
| CE | ~30% | ~45–55% |
| EA | ~62% | ~70–78% |
| EIS | ~48% | ~58–65% |

---

## Constraints

- Do not modify or overwrite validated existing timeline dates — recovery is fill-only
- Do not attempt to recover EA/EIS decisions from documents that genuinely don't contain them (separate unlinked ROD/FONSI documents are unrecoverable)
- Do not rebuild regex extraction — the 10-pattern suite is comprehensive
- Do not run recovery on projects with `timeline_status == 'complete'`

---

## Implementation Order

1. `--recovery-run` mode in `extract_timeline.py` (Tier 1)
2. Fallback merge update in `03_timeline.R`
3. Sample recovery test: `--recovery-run --source CE --sample 50 --output test50_recovery.parquet`; manually inspect 10 recovered dates against source documents
4. Full recovery pass: CE → EA → EIS
5. DeBERTa EA-specific model training (Tier 2, can run in parallel)
6. Tier 3 priority batch if deliverable thresholds still not met

## Verification Checklist

- [ ] Recovery rate improvement: `timeline_status` distribution before vs. after
- [ ] No recovered initiation dates that postdate the corresponding decision date
- [ ] No implausible duration outliers introduced by recovery dates
- [ ] `03_timeline.R` figure outputs re-run and reviewed
- [ ] Manual spot-check: 5 EA projects where recovery filled a decision date vs. actual document
