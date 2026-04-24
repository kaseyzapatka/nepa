# Timeline Extraction Improvement Plan

## Read these files before implementing anything:

* `code/extract/extract_timeline.py` — full pipeline; key functions:
  * `extract_dates_with_context()` (line ~2732) — main candidate extraction path
  * `auto_label_context()` (line ~1116) — weak supervision labeling; training ceiling
  * `extract_with_bert()` (line ~1977) — BERT inference + post-hoc guardrails stack
  * `_select_best_decision()` / `_select_best_initiation()` — final date picks
  * `run_regex_prep()` (line ~3948) — builds `regex_candidates_*.parquet`
  * `generate_bert_training_data()` (line ~1189) — builds training set from cache
  * `run_llm_adjudication()` (line ~4643) — Claude/Ollama post-BERT adjudication
* `notes/architecture/timeline_ideas/model_comparison.md` — model selection rationale (DeBERTa base vs small vs xsmall by source)
* `notes/architecture/timeline_ideas/model_evaluation.md` — weak supervision ceiling explanation; bootstrap loop
* `notes/architecture/timeline_ideas/add_spacy.md` — existing spaCy spec (NOTE: targets old `get_date_context()` in `build_project_timeline`; this plan redirects it to the main extraction path)

---

## Background

Current pipeline produces:
* CE: 30% full timelines (missing initiation is the primary bottleneck)
* EA: 62% (mostly missing decision)
* EIS: 48% (missing both or just decision)

The core architecture — regex extraction → BERT classification → rule-based candidate selection → Claude adjudication — is sound. The weak supervision ceiling is the main barrier. The problems are:

1. **Context is thin:** `extract_dates_with_context()` provides sentence-level windows but no structural metadata (governing verb, section heading, document position). BERT learns from these flat strings.
2. **No triage before BERT:** All candidates (strong, ambiguous, and junk) go to BERT equally, adding noise that hurts initiation recall.
3. **CE initiation coverage is the worst gap:** CE documents use form-field and signature-block layouts where dates are not preceded by typical keywords. These patterns are systematically missed by `auto_label_context()` and thus never learned by BERT.
4. **Claude adjudication operates at the wrong granularity:** `run_llm_adjudication()` currently sends batches of individual date candidates; Claude can't reason about cross-date consistency.

Scale constraints that do NOT change:
* CE (~19k projects): no per-project LLM calls; BERT + rules only
* EA (~500) + EIS (~700): Claude adjudication is feasible; use it for edge cases only

---

## Implementation Plan (Phased)

### Phase 1 — Enrich candidate context with section labels and governing verb
**Target:** `extract_dates_with_context()` and `run_regex_prep()`
**Goal:** Add two new structured fields to every candidate row in `regex_candidates_*.parquet`:
- `section_label` — which document section the date lives in (signature block, NOI, ROD, references, review checklist, etc.)
- `dep_verb` — governing verb lemma from spaCy dependency parse (e.g., "sign", "receive", "approve")

**Critical design constraint:** Section headings are NOT in the same sentence as the date — they are separate heading lines above it. spaCy running on the already-extracted 100-char sentence context cannot see a heading like `"AUTHORITY AND APPROVAL"` two paragraphs up. Therefore ALL four enrichment signals must be computed DURING `run_regex_prep()` while the full page text is in memory. They cannot be added as a post-hoc pass on saved contexts.

**What to build — all inside `run_regex_prep()`, per page:**

1. **Section heading detector (regex, not spaCy):** Before extracting dates from a page, scan the full page text once for heading patterns and build a `position → section_label` map. Use uppercase line-pattern regex (e.g., `r'^\s*(AUTHORITY AND APPROVAL|NOTICE OF INTENT|RECORD OF DECISION|…)\s*$'`), not spaCy — headings are structurally regular enough that a fast regex is sufficient and avoids adding spaCy overhead to full document text. Map headings to canonical labels per source type:
   - CE: `signature_block`, `ce_determination`, `review_checklist`, `project_description`, `compliance_checklist`
   - EA: `noi`, `fonsi`, `scoping`, `comment_period`, `project_description`
   - EIS: `rod`, `noi`, `scoping`, `draft_eis`, `final_eis`, `project_description`
   - All sources: `references`, `bibliography`, `legal_citations` (dates here are nearly always junk)
   For each date match position, look up the nearest preceding heading within 1,000 chars → `section_label`.

2. **spaCy dep-parse and NER (on date-containing sentences only):** Regex finds all date matches in the page (existing). Extract the sentence containing each date match (existing `_sentence_spans()`). Collect ALL date-containing sentences for a page/document batch, then call `nlp.pipe(sentences, batch_size=256)` once — NOT on the full page text, only on the sentences that already contain a date. This keeps the spaCy volume small: roughly 5–15 sentences per CE project × 19k projects ≈ 200k sentences total for all CE, which runs in under 5 minutes on CPU.

   From each parsed sentence extract:
   - `dep_verb`: head verb lemma (walk dependency tree from date token up to VERB/AUX root)
   - `ner_law_ref` (bool): sentence contains an ORG or PERSON entity immediately adjacent to "Act", "Law", "Policy", "Plan", "Agreement", or "Regulation" — catches "Smith Act of 2000", "Burlington Northern Agreement" that `DATE_EXCLUSION_KEYWORDS` misses because it can't match proper names
   - `ner_decision_signal` (bool): sentence contains a PERSON entity AND dep_verb is in `{"sign", "approve", "authorize"}`, OR sentence contains "NEPA Compliance Officer" / "Authorizing Official" / "initiator" as a token pattern — catches these role phrases when unusual spacing or capitalization breaks the existing regex patterns

3. Store `section_label`, `dep_verb`, `ner_law_ref`, and `ner_decision_signal` as new columns in `regex_candidates_*.parquet`. These columns should be `""` / `False` when not available (backwards-compatible).

4. Gate the entire spaCy path (dep-parse + NER) behind `--use-spacy` CLI flag. Section heading detection via regex is lightweight enough to always run. Without `--use-spacy`, `dep_verb`, `ner_law_ref`, and `ner_decision_signal` remain empty.

**Constraint check:** `add_spacy.md` targets the old `get_date_context()` in `build_project_timeline()`. Do NOT implement it there — that function is a legacy path not used by `--bert-run`. All enrichment goes into `run_regex_prep()` and is stored in the regex cache.

**Timing:** Section heading regex: negligible. spaCy on date-sentences only: ~5 min for all CE, ~1 min for EA/EIS. The 30–60 min estimate in the previous version assumed processing full document text — that is not the design.

---

### Phase 2 — Pre-BERT triage (three-tier filtering)
**Target:** `extract_with_bert()` — add a triage pass before the BERT `.classify()` call
**Goal:** Reduce noise entering BERT and recover candidates that should be auto-classified

The existing code already has per-date guardrails *after* BERT. Move the strongest exclusions to *before* BERT:

**Tier 0 — Hard discard (no BERT call):**
- `section_label` in `{"references", "bibliography", "legal_citations"}` → skip
- `ner_law_ref == True` — sentence contains a named legislative/plan reference (e.g., "Smith Act of 2000", "National Environmental Policy Act") → skip; these are the cases `DATE_EXCLUSION_KEYWORDS` misses because it can't match proper names
- `_is_decision_boilerplate(context)` → skip (already exists, move it here)
- `_is_historical_by_year(date_str)` AND no strong initiation/decision cue AND `ner_decision_signal == False` → skip

**Tier 1 — Auto-classify without BERT:**
- `auto_label_context(context)` returns `"decision"` AND pattern was from `DECISION_PATTERNS_STRONG` → label = `decision`, confidence = `0.95` (synthetic)
- `section_label == "signature_block"` AND `dep_verb` in `{"sign", "approve", "authorize", "execute"}` → label = `decision`, confidence = `0.95`
- `ner_decision_signal == True` AND `dep_verb` in `{"sign", "approve", "authorize"}` → label = `decision`, confidence = `0.92`; this catches "NEPA Compliance Officer" and "Authorizing Official" phrasing that varies enough to break regex but is reliably caught by spaCy token patterns
- `section_label == "noi"` (EA/EIS only) → label = `initiation`, confidence = `0.90`
- `"initiator signature"` or `"doe initiator signature"` in context → label = `initiation`, confidence = `0.95`
- `ner_decision_signal == True` AND `dep_verb` in `{"receive", "submit", "file"}` → label = `initiation`, confidence = `0.88`; catches "BLM received application" and similar
These rules use existing patterns plus new NER signals; they bypass BERT for the most confident cases.

**Tier 2 — Send to BERT:** everything that passes Tiers 0 and 1

**Tier 3 — Flag for Claude (EA/EIS only):** BERT-classified candidates where `confidence < 0.65` AND type is `initiation` or `decision` (most ambiguous). Store a `needs_claude_review` flag on these rows; use them as input to Claude adjudication in Phase 4.

**Implementation note:** This does not require any new training. It's a pre-processing step layered on existing functions. Store `triage_tier` as a new column in the classified dates output for debugging.

---

### Phase 3 — Richer BERT input strings
**Target:** `generate_bert_training_data()` and `_add_inference_prefix()`
**Dependency:** Phase 1 must be done first (section_label and dep_verb must be in the regex cache)
**Goal:** Replace the current flat prefix `[CE] [ROD] <context>` with a richer structured prefix that packages the spaCy-extracted features

**New prefix format:**
```
[CE] [ROD] [sec:signature_block] [verb:sign] [sig:Y] <context>
```

Rules:
- Include `[sec:<label>]` only if `section_label` is non-empty and not `"unknown"`
- Include `[verb:<lemma>]` only if `dep_verb` is non-empty
- Include `[sig:Y]` if `ner_decision_signal == True` (collapses the PERSON/role signal to a single binary token)
- Omit `[sig:Y]` entirely (not `[sig:N]`) when False — keeps the prefix short for the majority of candidates where it doesn't fire
- Do NOT include raw entity names in the prefix — they're too variable and waste token budget
- Keep existing source and doc_type tokens
- Max input is still 256 tokens — the prefix adds ~4–10 tokens, fine

**What to change:**
- In `generate_bert_training_data()`: read `section_label` and `dep_verb` from the cache and build the richer prefix (mirrors `_make_prefix()` but adds new tokens)
- In `_add_inference_prefix()`: add optional `section_label` and `dep_verb` parameters with empty defaults (backwards compatible)
- In `run_bert_run()`: pass section_label and dep_verb from the candidate dict when calling `_add_inference_prefix()`

**Retraining required:** Yes — after this change, retrain all three source models on the enriched contexts. Use the same model/epoch config as today (deberta-v3-base for CE, deberta-v3-small for EA/EIS).

**Expected improvement:** The section_label token alone should materially improve initiation recall for CE because `[sec:signature_block]` is a strong initiation signal the model can learn from the weak-supervision labels.

---

### Phase 4 — Timeline-level Claude adjudication (EA/EIS only)
**Target:** `run_llm_adjudication()` — restructure the prompt and input format
**Goal:** Send Claude the full set of high-priority candidates per project as a single timeline validation task, not individual date candidates

**Current problem:** Claude currently receives batches of individual rows (one prompt per project with a flat list of date-context pairs). This works but doesn't leverage Claude's strength in cross-date reasoning.

**New prompt structure:** Per project, send:
```
Process type: EA
Project ID: <id>

INITIATION CANDIDATES (ordered by score, descending):
1. 2019-03-15 | section: noi | verb: publish | context: "Notice of Intent published March 15, 2019..."
2. 2018-11-01 | section: project_description | context: "Application received November 2018..."

DECISION CANDIDATES (ordered by score, descending):
1. 2021-08-22 | section: fonsi | verb: sign | context: "FONSI signed August 22, 2021..."
2. 2021-08-20 | section: signature_block | verb: sign | context: "Authorizing Official signature..."

Task: Select the best initiation date and best decision date. Flag if:
- Decision date precedes or equals initiation date
- Either date type has no confident candidate (report null)
- Duration seems implausible (>15 years or <30 days for EA)

Respond as JSON: {"initiation_date": "YYYY-MM-DD"|null, "decision_date": "YYYY-MM-DD"|null, "rationale": "...", "flag": null|"..."}
```

**What to change:**
- Add a `_build_timeline_adjudication_prompt(project_id, initiation_candidates, decision_candidates, source)` function that constructs this prompt
- Modify `run_llm_adjudication()` to group candidates by project, run triage (keep top 5 initiation + top 5 decision candidates by score), and call this new prompt builder
- Parse the JSON response; fall back gracefully if Claude returns malformed output
- Add cross-date validation guardrails: reject any adjudication where `decision_date <= initiation_date`

**Input filtering before Claude:** Only send projects where BERT produced `timeline_status != "complete"` AND at least one candidate was flagged `needs_claude_review` (from Phase 2 Tier 3). This limits Claude calls to genuinely ambiguous cases.

**CE:** Do NOT call Claude per-project. If CE timeline coverage remains low after Phases 1–3, consider a targeted Claude batch on the ~500 CE pipeline projects only (small enough to be cost-feasible).

---

### Phase 5 — Process-type branching for section patterns
**Target:** Section heading patterns in Phase 1, and `_select_best_initiation()` doc_type boost map (`_INIT_DOC_TYPE_BOOST`)
**Goal:** Ensure section patterns, triage rules, and candidate selection scoring are process-specific

**CE-specific additions:**
- Section patterns: `"ce determination"`, `"ce designation"`, `"doe initiator"`, `"compliance checklist"`, `"authority and approval"`, `"determination and approval"`, `"environmental review form"`
- Triage: `section_label == "ce_determination"` → treat any date as a candidate for decision (currently many are dropped as `other`)
- `_INIT_DOC_TYPE_BOOST`: already has CE doc type boost; verify `"CE"` doc type (main CE form) gets highest initiation doc-type boost

**EA-specific additions:**
- Section patterns: `"notice of intent"`, `"scoping notice"`, `"finding of no significant impact"`, `"fonsi"`, `"environmental assessment"`, `"comment period"`
- Triage: `section_label == "fonsi"` → date is decision candidate; `section_label == "noi"` → date is initiation candidate

**EIS-specific additions:**
- Section patterns: `"record of decision"`, `"rod"`, `"notice of intent"`, `"scoping"`, `"draft environmental impact statement"`, `"final environmental impact statement"`
- Gap rule is already disabled for EIS (`SOURCE_CONFIG['EIS']['apply_gap_rule'] = False`) — preserve this

**Implementation:** Pass `source` into the section-heading detector and use a `SECTION_PATTERNS` dict keyed by source. Already passing `source` throughout `run_regex_prep()` — straightforward extension.

---

## Manual Training Data Strategy

Manual labels have the highest leverage when targeted at cases the weak-supervision ceiling can't reach. Three distinct labeling efforts, in priority order:

### 1. CE Initiation False Negatives (~50 examples)
**What:** Projects where BERT returned no initiation candidate but the date is clearly present in the document — signature blocks, form fields, "application received" lines with irregular spacing, DOE initiator sections.
**Scripts:**
```bash
# Step 1: export review CSV of misclassified candidates
python code/manual_training/01_find_ce_initiation_candidates.py

# Step 2: open data/manual_training/review_ce_initiation_candidates.csv
#   fill in correct_type = "initiation" for dates that are clearly initiation dates

# Step 3: apply corrections to the training corrections file
python code/manual_training/02_apply_corrections.py

# Step 4: retrain
python code/extract/extract_timeline.py --bert-generate
python code/extract/extract_timeline.py --bert-train --source CE
```
**Files:** `data/manual_training/review_ce_initiation_candidates.csv` → `data/analysis/manual_training_corrections.csv`
**Where it feeds in:** `generate_bert_training_data()` merges `manual_training_corrections.csv` and overrides auto-labels where `(project_id, date)` matches. Even 50 well-chosen CE initiation corrections can shift the model materially because `auto_label_context()` almost never fires for form-field layouts.

### 2. Gold Standard Evaluation Set (25–30 per source)
**What:** A held-out set of fully verified (project_id, initiation_date, decision_date) triples that never enters training — used only to measure real-world accuracy after each phase.
**Why:** Current F1 metrics measure agreement with regex labels, not real accuracy. Without a gold standard you can't tell whether Phase 1, 2, or 3 is actually helping.
**Scripts:**
```bash
# Step 1: generate stratified sample template
python code/manual_training/03_build_gold_standard_sample.py

# Step 2: open data/manual_training/gold_standard_template.csv
#   verify initiation_date_verified and decision_date_verified for each project
#   use page_viewer_ce.ipynb / page_viewer_ea.ipynb to look up raw documents

# Step 3: copy completed file to its permanent location
cp data/manual_training/gold_standard_template.csv data/analysis/gold_standard_timelines.csv
```
**Files:** `data/manual_training/gold_standard_template.csv` → `data/analysis/gold_standard_timelines.csv`
**Where it feeds in:** A `--eval-gold` flag (to be added to `--bert-run` as part of Phase 3 implementation) loads `gold_standard_timelines.csv`, compares against pipeline output, and prints precision/recall/F1 by source and date type. Run after every retrain.

### 3. EA/EIS Post-Claude Adjudication Feedback Loop
**What:** After running Phase 4 Claude adjudication on EA/EIS edge cases, review the cases where Claude's pick differs from BERT's pick. Add the correct label to `manual_training_corrections.csv` via `02_apply_corrections.py`.
**Why:** Claude's outputs include a rationale field, so spot-checking is fast. This converts expensive Claude calls into cheap training signal — the loop closes over time and fewer projects need Claude.
**When:** After the first full EA/EIS adjudication run in Phase 4. Repeat each retrain cycle until BERT and Claude agree on >90% of cases.
**Implementation note:** Store Claude's adjudicated outputs in a separate column (`llm_initiation_date`, `llm_decision_date`) alongside BERT's picks so comparisons are easy without destructive overwrites.

---

## Sequencing & Dependencies

```
Phase 1 (spaCy enrichment + section labels)
    ↓
Phase 5 (process-type section patterns) ← can be done alongside Phase 1
    ↓
Phase 3 (richer BERT input strings + retrain)
    ↓
Phase 2 (pre-BERT triage) ← can be added before or after Phase 3
    ↓
Phase 4 (timeline-level Claude adjudication)
```

Minimum viable improvement: **Phase 1 + Phase 3 + retrain.** This alone should improve CE initiation coverage by giving BERT the section_label signal it currently lacks. Phases 2, 4, 5 compound those gains.

---

## Validation at Each Phase

After each phase:
1. Run `--regex-prep` on a 200-project sample per source
2. Run `--bert-run --sample 200` and compare coverage metrics against the current baseline in `notes/status/timeline_status.md`
3. Spot-check 10 previously-uncovered CE projects: look for `section_label == "signature_block"` or `dep_verb == "sign"` and verify the resulting classification is correct
4. For EA/EIS after Phase 4: compare `llm_adjudication_date` vs `bert_*_date` for the projects Claude touched; review 20 cases where they differ

---

## Constraints

* Do NOT modify the output schema of `regex_candidates_*.parquet` in a breaking way — add new columns, don't rename existing ones
* `--bert-run` without `--use-spacy` must produce identical output to today's behavior
* CE LLM adjudication at full scale is not feasible; Phase 4 applies to EA+EIS only (or a targeted CE subset of ≤500 projects)
* Retraining CE model takes ~45–60 min on MPS; plan accordingly
* spaCy `en_core_web_sm` must be installed in the `nepa` conda environment; add to `requirements.txt`

---

## Output from this session

Implement Phases 1–5 in order, validating after each phase. Start by building and testing the section-heading detector and `get_date_context_spacy()` on a 50-project CE sample before wiring them into `run_regex_prep()`.
