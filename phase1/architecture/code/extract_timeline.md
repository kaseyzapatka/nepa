# extract_timeline.py — Pipeline Architecture

**Script:** `phase1/code/extract/extract_timeline.py`

**Purpose:** Extract project `initiation` and `decision` dates for clean energy projects.
Shared infrastructure consumed by four deliverables (D2, D3, D5, D6) rather than a deliverable
in its own right — see [../README.md](../README.md#timeline-data-integration).

**Method:** A DistilBERT classifier trained on weak-supervision (auto-labeled) regex
candidates, run at scale on Categorical Exclusions (~19K projects, no per-project LLM calls),
plus an optional Claude/Ollama LLM adjudication pass reserved for the much smaller EA/EIS
pools (~500 / ~750 projects). This is architecturally distinct from Phase 2's timeline
pipeline (D4), which uses a different two-head candidate classifier and a different tier
structure — the two should not be conflated.

---

## Data Flow

```mermaid
flowchart TD
    A[projects_combined.parquet +\nEA/EIS/CE page & document parquets] --> B[--regex-prep\nextract_dates_from_text]
    B --> C[regex_candidates_{ce,ea,eis}.parquet]
    C --> D[--bert-generate\nauto_label_context]
    D --> E[bert_training_data.parquet]
    E --> F[--bert-train\ntrain DistilBERT 4-class classifier]
    F --> G[models/timeline_classifier/]
    C --> H[--bert-run\nextract_with_bert]
    G --> H
    H --> I[bert_dates_json +\nbert_initiation_date_final +\nbert_decision_date_final]
    I --> J[projects_timeline_bert.parquet\nCE — production endpoint]
    I --> K{--llm-run?}
    K -->|EA/EIS only| L[Claude API adjudication\nbuild_project_timeline_llm]
    L --> M[projects_timeline_bert_ea_llm.parquet\nprojects_timeline_bert_eis_llm.parquet]
    K -->|--hybrid + Ollama| N[local LLM validation pass\ntest20_hybrid.parquet — edge cases only]
```

---

## Phase 1 — Regex Date Extraction (`--regex-prep`)

`extract_dates_from_text()` scans document page text with a battery of date-pattern regexes
(numeric, month-name, and abbreviated forms), classifies the surrounding context via
`should_exclude_date()` (rejects table-of-contents/citation/expiration-style contexts) and
`get_date_context()` (captures a ±100-char window). Candidate dates are additionally sourced
from **document file names** (`extract_dates_from_filename()`,
`build_file_name_date_map()`) — several downstream manual overrides (see Known Issues in
[deliverable02.md](../deliverables/deliverable02.md)) exist specifically because a
project's true decision date only appears in a filename, not in body text.

Output cached per source: `regex_candidates_{ce,ea,eis}.parquet`, reused across later BERT
runs via `--use-regex-cache` to avoid re-scanning ~6M pages.

## Phase 2 — Weak-Supervision Labeling and BERT Training

`auto_label_context()` assigns one of four weak-supervision labels to each regex candidate's
context window using cue-word heuristics organized by strength (Strong → Medium → Weak):

| Label | Strong cues | Weaker cues |
|---|---|---|
| `decision` | Digital-signature syntax (`YYYY.MM.DD`), "digitally signed by", ROD/FONSI/CE-determination language, authorizing-official signature blocks | "final approval", "determination", "approval" |
| `initiation` | "notice of intent", "scoping meeting/period", "application received", "right-of-way application", "NOI published" | "proposed action", "NEPA process started", "request received" |
| `review` | Specialist role titles (wildlife biologist, archaeologist, realty specialist), reviewer-checkbox forms, MOA/Section 106 references | — |
| `other` | RMP reference dates, boilerplate form language, dates with no nearby cue | — |

`generate_bert_training_data()` builds `bert_training_data.parquet` from these labels — this
weak-supervision step, not hand-labeled ground truth, is what the DistilBERT classifier is
trained against.

`train_bert_classifier()` fine-tunes a **4-class sequence classifier** — labels
`{decision: 0, initiation: 1, review: 2, other: 3}` — via HuggingFace `Trainer`. The
`--bert-model` CLI argument defaults to `distilbert-base-uncased`, and **this is what is
actually trained and committed**: `phase1/models/timeline_classifier/config.json` shows
`architectures: ["DistilBertForSequenceClassification"]`. There is only one model directory
on disk (`models/timeline_classifier/`), not the per-source `timeline_classifier_ce/ea/eis`
directories with `deberta-v3-base`/`deberta-v3-small` described in
[runbook 02](../../runbooks/02_timeline.md) — that runbook text describes an intended/future
per-source setup that was not the model actually used to produce the committed
`projects_timeline_bert*.parquet` outputs. See Known Issues below.

## Phase 3 — BERT Inference and Date Selection (`--bert-run`)

`extract_with_bert()` classifies every regex candidate's context with the trained classifier,
then applies rule-based selection logic on top of the raw classification:

- **Decision date selection** (`_select_best_decision`): prioritizes contexts with strong
  decision cues (`_has_strong_decision_cue`, `_has_reviewer_checkbox`,
  `_has_authorizing_with_signature`) over generic classifier hits; applies
  `_apply_historical_gap_rule` (730-day gap heuristic) to suppress stale historical dates that
  otherwise outrank the true decision date. The gap rule is applied **per source, not
  uniformly**: CE and EA both enable it (`apply_gap_rule=True`, `gap_days=730`), but EIS
  disables it entirely (`apply_gap_rule=False`) because EIS projects legitimately span
  5–10+ years and the 730-day heuristic would incorrectly suppress genuine early dates.
- **Initiation date selection** (`_select_best_initiation`): prioritizes strong initiation
  cues, with an **inferred fallback** — `bert_inferred_application_date` uses the earliest
  review date as a proxy for initiation when no explicit initiation date is found
  (`bert_earliest_review_date` / `bert_latest_review_date` / `bert_n_review_dates`).
- `bert_timeline_status` records whether both dates, only one, or neither was found.

Output: `projects_timeline_bert.parquet` — the **production CE endpoint** (no per-project LLM
calls; 19,399 clean CE projects, 62 columns).

## Phase 4 — LLM Adjudication (EA/EIS only, `--llm-adjudicate`)

For the much smaller EA/EIS pools, `run_llm_adjudication()` sends the BERT-filtered candidate
list per project to Claude (`_call_claude_adjudication`) or a local Ollama model
(`_call_ollama_adjudication`), with `_build_adjudication_prompt()` constructing a
candidate-ranking prompt and `_filter_candidates_for_llm()` capping the candidate count (50 by
default) sent per project. The LLM returns `llm_initiation_date` / `llm_decision_date` plus
free-text reasoning (`llm_initiation_reasoning` / `llm_decision_reasoning`) and diagnostic
fields (`llm_decision_mode`, `llm_n_priority_decision_candidates`, …).

**`--llm-adjudicate` is the current EA/EIS production entrypoint** — distinct from `--llm-run`,
which is a separate, legacy CE-oriented full-document/hybrid LLM flow
(`run_llm_timeline_extraction()`, `ce_only=True` by default) not used for production EA/EIS
adjudication.

**Adjudication caps and decision modes** — three decision modes govern how the prompt is
built (`_build_adjudication_prompt()`):

| Decision mode | Condition | Guardrail |
|---|---|---|
| `priority_only` | Tier A decision docs present (FONSI/ROD/DR/Decision Record) | Hard post-parse guardrail enforced only in this mode |
| `ea_eis_fallback` | No Tier A docs, but vetted fallback decision candidates exist (EA/DEA/DEIS/FEIS/FEA/EIS) | — |
| `no_decision_candidates` | No decision-date signal identifiable in any candidate | — |

Candidate/context caps default to `max_candidates=50, context_chars=300`; EIS uses a tighter
`max_candidates=30, context_chars=200` (`LLM_ADJ_EIS_MAX_CANDIDATES` /
`LLM_ADJ_EIS_CONTEXT_CHARS`) — EIS documents run much longer, so a lower cap keeps prompt size
bounded. Both are overridable via `--max-candidates` / `--context-chars`.

**Provider behavior:** Claude (`--provider claude`, default model
`claude-haiku-4-5-20251001`) forces `workers=1` (sequential, to respect rate limits) and
retries both HTTP `429` (rate limited) and `529` (service overload) with backoff — `529`
waits 60s by default, `429` respects the `retry-after` header (30s fallback). Ollama
(`--provider ollama`, the CLI default) has no such retry/backoff logic.

**`main_document_imputed`:** EA/EIS regex candidate extraction is main-document-first; when a
project has no `main_document == "YES"` documents at all, it falls back to non-main documents
sorted by source-specific priority, and those candidate rows are flagged
`main_document_imputed=True`. This flag carries through regex candidates into the BERT
project-level output.

Default output naming: `<input_stem>_llm.parquet` unless `--output` is given, e.g.
`projects_timeline_bert_ea_llm.parquet` (573 rows), `projects_timeline_bert_eis_llm.parquet`
(753 rows) — both extend the BERT columns with the `llm_*` adjudication columns.

A separate `--hybrid` mode (`extract_with_hybrid_approach()`) runs a **local Ollama model**
(e.g. `llama3.2:3b-instruct-q4_K_M`) for validation/edge-case spot checks
(`test20_hybrid.parquet`) — this path is explicitly **not** used for production CE runs (see
[runbook 02](../../runbooks/02_timeline.md)).

A downstream, narrower re-adjudication pass — `projects_timeline_targeted_llm.parquet` — is
consumed by D2 to patch a small set of specific incomplete non-standard (programmatic/tiered)
projects; see [deliverable02.md](../deliverables/deliverable02.md).

---

## Coverage (verified against committed outputs)

| Process | n (clean) | Initiation found | Decision found | Both (duration-calculable) |
|---|---:|---:|---:|---:|
| CE | 19,399 | 8,260 (42.6%) | 15,295 (78.8%) | 5,899 (30.4%) |
| EA | 573 | 516 (90.1%) | 364 (63.5%) | 355 (62.0%) |
| EIS | 753 | 595 (79.0%) | 389 (51.7%) | 362 (48.1%) |

CE's low "both dates" rate is driven almost entirely by weak initiation signal (many CE
documents never state an explicit application/start date), not by weak decision-date
extraction. EA/EIS's LLM adjudication step substantially improves both coverage and precision
relative to a BERT-only pass, at the cost of being feasible only at EA/EIS scale.

---

## Output Schema (CE — `projects_timeline_bert.parquet`)

| Column | Type | Description |
|---|---|---|
| `bert_dates_json` | str (JSON) | All classified date candidates with type/confidence/context |
| `bert_n_dates_found` | int | Count of candidates found |
| `bert_decision_date`, `bert_decision_date_final` | date | Raw vs. rule-refined decision date |
| `bert_decision_date_source`, `bert_decision_confidence` | str | Provenance and confidence of the selected decision date |
| `bert_application_date`, `bert_inferred_application_date` | date | Explicit vs. proxy (earliest-review) initiation date |
| `bert_initiation_date_final` | date | Final selected initiation date (explicit preferred, inferred fallback) |
| `bert_earliest_review_date`, `bert_latest_review_date`, `bert_n_review_dates` | — | Review-window bounds used for the inferred initiation fallback |
| `bert_error` | str | Extraction error, if any |
| `project_file_name_dates` | str (JSON) | Dates parsed from document file names |
| `run_timestamp` | str | Build timestamp |

EA/EIS outputs add: `llm_initiation_date`, `llm_decision_date`, `llm_initiation_reasoning`,
`llm_decision_reasoning`, `llm_adj_n_candidates`, `llm_adj_prompt`, `llm_adj_raw_response`,
`llm_decision_mode`, `llm_n_priority_decision_candidates`, `llm_n_other_decision_candidates`,
`llm_n_fallback_decision_candidates`, `llm_adj_error`, `bert_timeline_status`.

---

## Known Issues

**Runbook describes a per-source DeBERTa setup that was not actually trained.**
[runbook 02](../../runbooks/02_timeline.md) documents "CE uses `deberta-v3-base`; EA/EIS use
`deberta-v3-small`" with separate `models/timeline_classifier_{ce,ea,eis}/` output
directories. The committed model on disk is a single unified `distilbert-base-uncased`
classifier at `models/timeline_classifier/`, confirmed via `config.json`
(`DistilBertForSequenceClassification`). The `--bert-model` and `--source` CLI flags exist to
support the per-source design, but the artifacts that actually produced the published
`projects_timeline_bert*.parquet` files reflect the single-model DistilBERT setup. Treat the
runbook's per-source DeBERTa description as an intended/future design, not the as-built
pipeline.

This traces to a prior refactor-planning note (dated 2026-03-19, no longer kept), a
three-tier improvement plan written specifically to raise the coverage numbers in the table
above: **Tier 1** — an LLM "recovery" pass that reads raw document pages directly (a discovery
step) rather than only selecting among BERT-extracted candidates (a selection step) — the
document is explicit that "`--llm-adjudicate` can only choose among BERT-provided candidates;
it cannot recover dates never extracted upstream," which is the fundamental architectural
limit of the pipeline as built; **Tier 2** — the per-source DeBERTa models referenced in the
runbook; **Tier 3** — targeted high-context-window Sonnet extraction for specific deliverables
(D2 tiered/programmatic, D6 technology-specific projects) if Tiers 1–2 are insufficient. None
of the three tiers were implemented in the committed Phase 1 pipeline — the coverage numbers
in this document are the pre-refactor baseline the plan itself cites (CE ~30%, EA ~62%, EIS
~48%). Phase 2 subsequently built an architecturally different multi-tier timeline pipeline
rather than implementing this specific plan; do not treat this file as describing Phase 2.

**Manual date overrides live in downstream deliverable code, not in this script.** A small
set of hand-verified dates for specific projects (verified from document text, filenames, or
NOI records) are patched in directly inside `deliverable02/00_setup.R` as a temporary
measure — see [deliverable02.md](../deliverables/deliverable02.md#known-issues-and-cautions).

---

## CLI Reference

```bash
# Full rebuild
python code/extract/extract_timeline.py --regex-prep
python code/extract/extract_timeline.py --bert-generate
python code/extract/extract_timeline.py --bert-train
python code/extract/extract_timeline.py --bert-run --source CE --output projects_timeline_bert.parquet
python code/extract/extract_timeline.py --bert-run --source EA --output projects_timeline_bert_ea.parquet
python code/extract/extract_timeline.py --llm-adjudicate --input projects_timeline_bert_ea.parquet --provider claude

# Quick rebuild reusing cached regex candidates
python code/extract/extract_timeline.py --bert-run --use-regex-cache --output projects_timeline_bert.parquet

# Hybrid local-LLM validation sample (not for production)
python code/extract/extract_timeline.py --llm-run --hybrid --use-regex-cache --sample 20 \
  --model llama3.2:3b-instruct-q4_K_M --timeout 180 --workers 4 --output test20_hybrid.parquet
```

See [runbook 02](../../runbooks/02_timeline.md) for the full command reference.
