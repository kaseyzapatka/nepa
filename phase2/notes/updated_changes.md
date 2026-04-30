# extract_timeline.py — Changes from Improvement Plan

Date: 2026-03-30
Branch: timeline-duckdb

All changes are in `code/extract/extract_timeline.py`.

---

## Phase 0 — CE Initiation Pattern Expansion

**Problem:** CE documents use form-field layouts (`DOE Initiator Signature ... Date: 01/15/2020`)
that the existing `INITIATION_PATTERNS_STRONG/MED` never matched, causing systematic misses in
CE initiation weak-supervision labeling.

**Changes:**
- Added `INITIATION_PATTERNS_CE_ONLY` list (9 patterns) after `INITIATION_PATTERNS_WEAK`
- Added `_source_initiation_patterns(source)` helper — returns base patterns + CE-specific ones
  when `source == 'CE'`
- Updated `auto_label_context(context)` → `auto_label_context(context, source=None)` to accept
  source and call `_source_initiation_patterns(source)` instead of the two flat lists
- Updated `generate_bert_training_data()` to pass `dataset_source` to `auto_label_context()`

**Retrain required:** Yes — `--bert-generate` then `--bert-train --source CE`

---

## Phase 1a — Context Window Fix and Widening

**Problem:** `min_context_chars` parameter in `extract_dates_with_context()` was declared but
never used — the function always called `_min_context_chars_for_sentence()` directly, ignoring
the parameter entirely.

**Changes:**
- Fixed `extract_dates_with_context()` to use `min_context_chars` as a floor:
  `min_chars = max(min_context_chars, _min_context_chars_for_sentence(span_text))`
- Updated `run_regex_prep()` → `_flush()` to pass `min_context_chars=200` (was effectively 80),
  giving 2–4 sentences of context per candidate instead of 1–2

**Effect:** Wider context improves section heading detection (Phase 1b) and richer BERT input.
Does not affect any other call site — the global `DATE_CONTEXT_WINDOW = 80` is unchanged.

---

## Phase 1b — Section Heading Detection

**Problem:** Every candidate in the regex cache was a flat context string with no structural
signal. BERT had no way to know whether a date appeared in a signature block, an NOI section,
or a references list.

**Changes:**
- Added `SECTION_PATTERNS` dict: per-source (CE, EA, EIS) + all-source heading patterns mapped
  to canonical labels (`ce_determination`, `signature_block`, `review_checklist`, `noi`, `fonsi`,
  `rod`, `draft_eis`, `final_eis`, `references`, `legal_citations`, etc.)
- Added `_build_section_label_map(page_text, source)` — scans full page text for headings,
  returns sorted `(position, label)` list
- Added `_lookup_section_label(char_pos, section_map, max_lookback=1000)` — finds nearest
  preceding heading within 1,000 chars of a date
- Updated `_flush()` in `run_regex_prep()` to build the section map per-document (while full
  page text is in memory) and annotate each candidate with `section_label`

**Effect:** `section_label` is now a column in all rebuilt regex cache files. Always runs —
no extra flag required.

---

## Phase 1c — spaCy Enrichment (opt-in)

**Problem:** The regex cache had no verb or role-phrase features. These are strong signals
(e.g., `dep_verb == 'sign'` + `sig_flag == True`) for distinguishing decision vs. review dates.

**Changes:**
- Added `_enrich_with_spacy(cache_df, source)` — runs `nlp.pipe(batch_size=512)` on cached
  context strings; extracts `dep_verb` (ROOT verb lemma), `sig_flag` (bool, from existing
  `_is_signature_block()`), `ner_decision_signal` (bool, role phrase or PERSON+decision verb)
- Added `use_spacy=False` parameter to `run_regex_prep()`; enrichment pass runs after the cache
  is built when `use_spacy=True`
- Added `--use-spacy` CLI flag to `--regex-prep`
- Graceful degradation: if spaCy or `en_core_web_sm` not installed, columns are set to
  `''`/`False` with a warning

**New dependency (opt-in):** `pip install spacy && python -m spacy download en_core_web_sm`

---

## Phase 1d — Cache Schema Compatibility

**Problem:** After adding new columns, `_make_prefix()` and `_add_inference_prefix()` needed to
handle both old caches (no enrichment columns) and new caches gracefully.

**Changes:**
- Updated `_make_prefix()` in `generate_bert_training_data()`:
  - Reads `section_label`, `dep_verb`, `sig_flag`/`ner_decision_signal` via `.get()` with
    empty fallbacks
  - Appends `[sec:X]`, `[verb:X]`, `[sig:Y/N]` tokens only when enrichment columns are present
  - Falls back to current `[SRC] [DOCTYPE]` format automatically for old caches
- Added stale-cache warnings in `generate_bert_training_data()` when `section_label` or
  `dep_verb` columns are absent

---

## Phase 2 — Pre-BERT Inference Triage

**Problem:** At inference time (`--bert-run`), every candidate went through BERT regardless of
how obvious the context was. `auto_label_context()` only ran during training, not inference.

**Key design rule:** `REVIEW_PATTERNS_STRONG` contexts always go to BERT — reviewer sign-off
dates and final decision dates look identical to a pattern matcher but BERT can distinguish them.

**Changes:**
- Added `_pre_bert_triage(context, source, section_label, dep_verb, sig_flag)` with 11 priority
  rules returning `(action, label, confidence)`:
  - `hard_discard`: boilerplate, OTHER_PATTERNS_STRONG, references/legal section labels,
    historical context without decision signal
  - `auto_classify → decision`: strong decision patterns (no review pattern), signature_block +
    decision verb, sig_flag + decision verb, ce_determination section label (CE only)
  - `auto_classify → initiation`: strong initiation patterns (no decision/review cue),
    noi section (EA/EIS only)
  - `bert`: any `REVIEW_PATTERNS_STRONG` match, all other candidates
- Wired into `extract_with_bert()`: only sends `bert`-tagged candidates to the model; merges
  results back in original order
- Added `triage_classified` bool and `section_label` string to `classified_dates` output
- Logs triage counts per run (hard_discard / auto_classify / bert)

---

## Phase 3 — Richer BERT Input Strings

**Problem:** `_add_inference_prefix()` only accepted `source` and `doc_type`; inference prefixes
could not include the new enrichment features.

**Changes:**
- Updated `_add_inference_prefix(context, source, doc_type, section_label='', dep_verb='',
  sig_flag=False)` to accept optional enrichment kwargs
- Builds the same structured prefix as `_make_prefix()` at training time
- Falls back to `[SRC] [DOCTYPE]` format when enrichment fields are absent (backwards compatible)
- Updated `extract_with_bert()` to pass `section_label`, `dep_verb`, `sig_flag` from each
  candidate dict when building prefixed contexts

---

## Phase 4a — Adjudication Validation Guardrails

**Problem:** `_parse_adjudication_response()` accepted any LLM-returned dates without validating
date ordering or plausible review duration. No QA flag was surfaced in output.

**Changes:**
- Added `_PLAUSIBLE_GAP_DAYS` dict: `CE (30–2000)`, `EA (60–3000)`, `EIS (180–7000)` days
- Added `_validate_adjudication(initiation_date_str, decision_date_str, source)` returning
  pipe-delimited flag string: `date_order_error`, `gap_too_short_Nd`, `gap_too_long_Nd`, or `''`
- Updated `_parse_adjudication_response(response_text, source=None)`:
  - Accepts optional `source` parameter
  - Parses both new (`reason_initiation`/`reason_decision`) and legacy
    (`initiation_reasoning`/`decision_reasoning`) JSON keys
  - Calls `_validate_adjudication()` and returns `validation_flag` in result dict
  - Returns `reason_initiation` and `reason_decision` columns
- Updated `run_llm_adjudication()` to pass `task['dataset_source']` to
  `_parse_adjudication_response()`

---

## Phase 4b — Adjudication Prompt Improvements

**Problem:** The adjudication prompt sent flat candidate lines without structural features.
The JSON response used `initiation_reasoning`/`decision_reasoning` keys that didn't match
the plan's `reason_*` naming.

**Changes:**
- Updated candidate line format in `_build_adjudication_prompt()` to include `[sec:X]` and
  `[verb:X]` tokens when `section_label`/`dep_verb` fields are present on the candidate
- Updated the JSON response format in the prompt to request `reason_initiation` and
  `reason_decision` (shorter, more direct)
- Added a rule hint explaining `[sec:]` labels to Claude

---

## Phase 5 — Process-type Branching

**Problem:** `_INIT_DOC_TYPE_BOOST` had an empty CE entry and there was no triage rule for
CE determination sections.

**Changes:**
- Updated `_INIT_DOC_TYPE_BOOST['CE']` to include `'CE_DETERMINATION': -2` (penalise CE
  determination forms for initiation scoring — they are decision documents)
- Added `ce_determination` → `auto_classify decision, 0.85` rule in `_pre_bert_triage()`
  for CE: many CE determination dates previously fell through as `other` because the section
  heading was the only reliable signal in short form-field contexts

---

## Summary of New CLI Flags

| Flag | Mode | Effect |
|---|---|---|
| `--use-spacy` | `--regex-prep` | Enables spaCy dep-parse enrichment on cached context strings |

## Summary of New Output Columns

| Column | Added to | Description |
|---|---|---|
| `section_label` | regex cache parquets | Nearest preceding document section heading |
| `dep_verb` | regex cache parquets (with `--use-spacy`) | ROOT verb lemma from dep parse |
| `sig_flag` | regex cache parquets (with `--use-spacy`) | Bool: signature block marker present |
| `ner_decision_signal` | regex cache parquets (with `--use-spacy`) | Bool: role phrase or PERSON+decision verb |
| `section_label` | BERT output (`classified_dates`) | Propagated from cache |
| `triage_classified` | BERT output (`classified_dates`) | Bool/str: whether candidate was auto-classified or hard-discarded |
| `validation_flag` | adjudication output | Date ordering / plausible gap validation |
| `reason_initiation` | adjudication output | One-sentence LLM rationale for initiation pick |
| `reason_decision` | adjudication output | One-sentence LLM rationale for decision pick |
