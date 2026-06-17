# D4 Timeline: EIS Recovery Refactor

**Status:** implementation started. The first low-risk slice is implemented but not yet
materialized in production data:

- EIS Tier B and Tier D page context raised to 12,000 characters;
- EIS same-page document-text deduplication prefers the longer packet;
- text fallback added for an EIS project that otherwise emits no packet;
- stale `selected_for_initiation` / `selected_for_decision` flags reset per project.

An in-memory validation over the 223 projects with no canonical packet produced:

- packets for 212 projects;
- 377 packets total, including 272 fallback packets across 149 projects;
- at least one extracted candidate for 137 projects;
- 177 candidates: 148 `unknown`, 20 `proxy_initiation`, 4 `clear_decision`,
  2 `proxy_decision`, 2 `review`, and 1 `clear_initiation`.

This validates retrieval recovery, not final-date recovery. The predominance of
`unknown` candidates is the reason the selection policy is not being loosened in the
same patch.

**Two different targets, two different levers — do not conflate them.** The plan has two
distinct goals with non-overlapping bottlenecks:

- **Objective 1 — candidate coverage (70% = 2,891 with both candidate types).** Current
  1,892; gap 999. This is **purely a retrieval + extraction problem (Phase 2/3).
  Selection (Phase 4) cannot create candidates and contributes nothing here.** The EIS
  candidate decomposition is: 1,892 both-types, **1,050 init-only (need a decision
  candidate)**, 106 decision-only (need an init candidate), 418 other-role-only, 664
  zero-candidate. The dominant lever is the **1,050 init-only projects**, of which
  **623 have a ROD/FEIS document** (`decision_doc_score >= 3.5`, addressable by the
  cap/full-read/exclusion-window fixes) and **427 have no endpoint document at all**
  (a source gap → Phase 5/OCR/register, not recoverable from local text).
  **Feasibility of Objective 1 is currently UNMEASURED for its dominant cohort:** the
  only retrieval validation so far (the 223 zero-candidate test in the status block, which
  recovered "both types" for ~9 projects) did not touch the 623 init-only-with-endpoint-doc
  projects. Phase 2/3 must measure decision-candidate recovery on those 623 before the 70%
  candidate target can be called reachable.

- **Objective 2 — final complete-timeline coverage (70% with both selected DATES).**
  Current 842. This is largely a **selection problem (Phase 4):** the 645 ranker-blocked
  initiation, 182 pool-excluded decision, and 130 month-suppressed decision projects
  (overlapping) **already have candidates** — they sit inside the 1,892 both-types group
  and fail at *selection*, not extraction.

The earlier framing that "Phase 2 contributes ~1–2% and Phase 4 carries the rest" was
**wrong** because it attributed the Objective-1 candidate gap to Objective-2 selection
levers. Keep the targets separate: **Phase 2/3 is the lever for candidate coverage;
Phase 4 is the lever for final-date coverage.** Do not treat the overnight Phase 2 run as
evidence either target is reachable; that is answered by the Phase 0 source-ceiling audit
(§6 Phase 0), the Phase 2/3 measurement of the 623, and Phase 4.

**Baseline date:** 2026-06-15.

## Phase 0 RESULT (2026-06-15) — go/no-go ceiling

Ran `_phase0_baseline.py` against current production parquets. Fixtures + metrics in
`notes/deliverable04/phase0/`. Headline:

| Metric | Value |
|---|---:|
| EIS total | 4,130 |
| Complete timelines today | 842 (20.4%) |
| Decision evidence available (endpoint doc OR register decision) | 2,664 (64.5%) |
| Initiation evidence available (narrative text / register / init doc) | 4,119 (99.7%) |
| **Corrected joint source ceiling** | **2,664 (64.5%)** |
| Ceiling-definition validation (currently-complete projects passing) | 99.5% ✅ |

**GO/NO-GO verdict: the 70% target (2,891) EXCEEDS the local-document ceiling of 2,664.**
70%+ complete timelines is **not achievable from local documents** — it requires Phase 5
(OCR of image-only decision PDFs + external register supplementation). The realistic
**local-document ceiling is 64.5% (2,664)**, and **decision evidence is the binding
constraint** (initiation evidence is nearly universal at 99.7% because the NOI/scoping date
lives in the EIS/FEIS narrative). The corrected ceiling definition is sound (99.5% of the
842 currently-complete projects satisfy it; the naive `initiation_doc_score>0` proxy gave a
false 434). **Phases 2–4 should target the 842→~2,664 headroom; treat 70%+ as a Phase-5
decision.** Phase-4 initiation-labeling cohort = 331 projects (high-confidence init,
no selected init); fixtures written.

**Scope:** EIS only. The refactor must not change CE or EA behavior, rows, dates, or
candidate scores except for explicitly shared infrastructure whose output is proven
byte-equivalent for those processes.

## 1. Objective

Refactor the D4 EIS path so that retrieval, extraction, classification, ranking, and
selection have explicit contracts and independently measurable recall. The immediate
target is:

1. At least 70% of all 4,130 EIS projects have at least one usable initiation candidate
   and at least one usable endpoint candidate before final selection.
2. Final complete-timeline coverage reaches 70% only if source availability and
   validation precision support it. Do not manufacture dates to hit the target.
3. Candidate and selected-date precision must not regress in exchange for coverage.

The first target requires at least 2,891 projects with both candidate types. The current
count is 1,892 (45.8%), so at least 999 additional projects must gain the missing
candidate type. Recovering all 664 zero-candidate projects would still yield at most
2,556 projects (61.9%) if every one gained both types. The refactor therefore must address
both zero-candidate projects and projects that currently have only one candidate type.

The 70–80% target cannot be treated as a local-regex guarantee. Only about 2,457 projects
have a recognized ROD or FEIS document in the current index. A source-ceiling audit must
separate:

- projects recoverable from local text;
- projects with text but mislabeled document metadata;
- image-only documents requiring OCR;
- projects requiring Federal Register or agency-register supplementation;
- projects for which no initiation or endpoint evidence exists in the available corpus.

**The binding ceiling is the JOINT one, and it must be the first number computed in
Phase 0.** A complete timeline requires *both* a recognized endpoint document (ROD/FEIS)
*and* a recognized initiation-evidence source (NOI, scoping notice, application, or
register entry) for the same project. The 2,457 endpoint figure is only the decision
side. Compute the joint count in Phase 0:

```python
# From timeline_document_index.parquet, per EIS project:
#   has_endpoint_doc   = any doc with decision_doc_score >= 3.5  (FEIS/ROD)
#   has_init_evidence  = see WARNING below — NOT initiation_doc_score > 0
#   joint              = has_endpoint_doc AND has_init_evidence
# Report COUNT(joint) over the 4,130 EIS universe.
```

> **WARNING — do not define initiation evidence as `initiation_doc_score > 0`.** That
> proxy is wrong by ~4×: of the 842 EIS projects that ALREADY have a complete timeline,
> only 213 have `initiation_doc_score > 0`, and 726/842 draw their initiation date from
> `document_text` — i.e. the NOI / scoping / project-history date lives **inside the
> EIS/FEIS narrative**, not in a dedicated initiation document. Using the narrow proxy
> produces a falsely hopeless ceiling (~434) that is below the count already achieved.
> Define `has_init_evidence` as **any EIS-family document with extractable text** (the
> narrative can yield an initiation date) OR an authoritative NOI/register initiation OR
> `initiation_doc_score > 0`. Validate the definition by confirming it is satisfied for
> ≥95% of the 842 currently-complete projects before trusting the ceiling number.

If `COUNT(joint)` is below 2,891 (the 70% target), the target is **not reachable from
local documents alone**, regardless of how well selection is tuned. In that case, state
the adjusted local-text-achievable target and treat Phase 5 (OCR + external register
supplementation) as **mandatory**, not optional, to reach 70%. Do not begin Phase 4
threshold work before this number is known.

## 2. Current Baseline

From the current production parquets:

| Metric | Projects | Share |
|---|---:|---:|
| EIS universe | 4,130 | 100.0% |
| Has initiation date | 1,264 | 30.6% |
| Has decision date | 2,204 | 53.4% |
| Has both dates | 842 | 20.4% |
| Has an initiation-role candidate | 2,942 | 71.2% |
| Has a decision-role candidate | 1,998 | 48.4% |
| Has both candidate types | 1,892 | 45.8% |
| Has zero candidates | 664 | 16.1% |

For candidate coverage, `body_text` is excluded from the decision-role count. It is an
ambiguous holding role, not positive decision evidence.

### 2.1 Zero-candidate funnel

The 664 zero-candidate projects split into:

| Failure point | Projects |
|---|---:|
| No context packets | 223 |
| Has truncated Tier D packets | 382 |
| Has packets, but no truncated Tier D packet | 59 |

All 223 no-packet projects have indexed documents. All have page rows, and 212 have at
least one nonempty text page. This is a retrieval failure, not primarily an OCR failure.
The fallback builder can access these pages: `retrieve_for_process()` currently loads
all EIS pages and constructs each project's `pages_df` from every indexed document ID,
including defer-priority documents. The loss occurs in packet construction, not page
loading.

Document metadata among the zero-candidate projects is dominated by:

- 947 `OTHER` documents across 494 projects;
- 208 FEIS documents across 140 projects;
- 137 DEIS documents across 129 projects;
- 5 ROD documents across 4 projects.

Recognized types are scored correctly: ROD scores `5.0`; 206 of 208 FEIS documents score
`3.5` and reach priority 2. The broader metadata gap is that 884 `OTHER` documents have
`decision_doc_score == 0`. Only nine affected projects have an obvious ROD/FEIS cue in
the filename, so adding filename scoring is correct but not a major recovery lever.

### 2.2 Truncation and exclusion findings

EIS Tier D packets are capped at 2,000 characters. Raw page lengths in zero-candidate
ROD/FEIS documents are:

| Type | Median | 99th percentile | Maximum |
|---|---:|---:|---:|
| FEIS | 2,850 | 8,919 | 14,943 |
| ROD | 2,736 | 5,220 | 13,729 |

A 30-page ROD/FEIS test found:

- all 30 pages exceeded 2,000 characters;
- full text recovered candidates on 3 pages, representing 2 of 8 projects;
- all recovered dates occurred after character 2,000;
- an 8,000-character cap recovered the same candidates as full text.

Across all 1,209 truncated Tier D packets for zero-candidate projects:

| Test | Projects with any candidate | Initiation | Decision | Both |
|---|---:|---:|---:|---:|
| 8,000 chars, current exclusions | 40 | 25 | 8 | 5 |
| 8,000 chars, local EIS exclusions | 54 | 30 | 19 | 9 |
| Full page | Same as corresponding 8,000-char test | | | |

These counts overlap other failure buckets and must not be added directly to downstream
loss counts. They show that truncation and whole-block exclusions are real but too small
to explain the complete-timeline gap by themselves.

### 2.3 Downstream loss

The previously reported 1,194 incomplete projects with both candidate types uses
`clear_decision` and `proxy_decision`, excluding `body_text`. It contains:

- 841 projects with a selected decision but no initiation;
- 207 projects with a selected initiation but no decision;
- 146 projects missing both selected dates.

For the exact 841-project missing-initiation cohort:

- 601 have all `p_init_cal < 0.3`;
- 218 have at least one `p_init_cal > 0.5`;
- 196 have no initiation candidate before the selected decision;
- 645 have a chronologically valid initiation candidate but all candidates fail the
  learned-ranker acceptance gates;
- 168 projects have both `p_init_cal > 0.5` and a chronologically valid candidate, but
  are still blocked by the ranker gate.

For the exact 207-project missing-decision cohort:

- 141 have at least one `p_dec_cal > 0.5`;
- 182 have no candidate admitted to the current EIS ROD/FEIS eligibility pool;
- 167 are flagged `deis_only`;
- 130 have at least one month-level decision candidate suppressed.

The ranker does separate selected and non-selected candidates, but its raw scores are
being used incorrectly as absolute confidence:

| Head | Selected median | Non-selected median |
|---|---:|---:|
| Initiation | 1.167 | -4.390 |
| Decision | -0.439 | -3.914 |

LightGBM LambdaRank scores are group-relative ordering values. They are not calibrated
probabilities and do not support universal `> 0` or `> -2` existence thresholds.

### 2.4 State and orchestration defects

The current pipeline also has reproducibility defects:

- `05_select_dates.py` sets `selected_for_initiation` and
  `selected_for_decision` to `True` but does not reset old values. The candidate parquet
  currently has 2,005 EIS projects with multiple initiation flags and 1,679 with
  multiple decision flags.
- `_run.py` runs `02`, `03`, `04`, and `05`, but omits calibration (`04b`) and ranking
  (`05b`). A fresh candidate pool can therefore be selected with stale or missing
  calibrated/ranker scores.
- Append-oriented stage writes make it possible to combine candidates and selection
  flags from different model or retrieval versions.
- `06_adjudicate_llm.py` cannot recover projects with no useful packets and currently
  sends only the top three candidates, which is too narrow for a recall-recovery pass.

The frozen validation inventory is also insufficient for final threshold selection:

- `frozen_eval_ids.txt` contains 28 projects, all EIS;
- 28 have a filled decision pick, but only 13 have a filled initiation pick;
- the separate labeled ROD sample has 46 projects;
- the separate labeled FEIS sample has 25 projects;
- there is no frozen FEIS threshold-evaluation column in `ranker.csv`.

The current data can support regression checks, but not stable independent thresholds
for initiation, ROD, and FEIS without expanding and freezing the event-specific sets.

## 3. Refactor Principles

1. **Separate recall from precision.** Retrieval and extraction should maximize auditable
   candidate recall. Classification and selection should control precision.
2. **Separate eligibility from ordering.** Source evidence and calibrated probabilities
   decide whether a candidate is eligible. The ranker only orders eligible candidates.
3. **Preserve abstention.** Missing is preferable to an unsupported date.
4. **Keep event semantics explicit.** ROD, Final EIS publication, and generic project
   milestones must not be silently collapsed.
5. **Make every loss observable.** A project should carry a reason code for the stage at
   which each endpoint became unavailable.
6. **Use process-partition replacement, not incremental mutation.** An EIS run should
   atomically replace the EIS partition while leaving CE/EA untouched.
7. **Validate each phase independently.** Do not combine retrieval, model, and selection
   changes into one production rerun.

## 4. Target Architecture

The refactored EIS flow should be:

```text
01 document evidence
    -> 02 role-aware retrieval
    -> 03 date mentions + rejection audit
    -> 04 calibrated event probabilities
    -> 05b within-project ordering
    -> 05 event selection + abstention
    -> 06 optional adjudication/recovery
    -> validated EIS partition publish
```

Each stage consumes an immutable stage output and writes a new stage output with a
`run_id`, configuration hash, input hash, and model version.

## 5. Proposed Changes by Stage

### 5.1 `01_index.py`: document evidence, not a single opaque priority

Keep the existing scores, but add explicit EIS evidence fields:

- `eis_doc_role`: `rod`, `feis`, `deis`, `noi_scoping`, `application`, `other`;
- `eis_doc_role_source`: pipe-delimited `type`, `clean_type`, `title`, `filename`;
- `eis_doc_role_confidence`: `high`, `medium`, `low`;
- `has_rod_filename_cue`, `has_feis_filename_cue`, `has_noi_filename_cue`;
- `text_page_count` and `has_extractable_text`.

Document-role detection should examine `file_name` in addition to the existing clean
type, raw type, and title. Filename evidence should raise retrieval priority but should
not itself establish a date.

Do not force every EIS document through one scalar `scan_priority` decision. Preserve
the scalar for compatibility, but let `02` consume the explicit role fields.

### 5.2 `02_retrieve.py`: role-aware EIS retrieval

#### Character limits

Set a named EIS page cap:

```python
EIS_PAGE_CONTEXT_CHARS = 12_000
```

Use it for both Tier B and Tier D EIS page packets. Raising Tier D alone is insufficient:
Tier B and Tier D packets use the same full-text hash, and deduplication prefers Tier B.
A 2,000-character Tier B packet can therefore replace a longer Tier D packet.

Twelve thousand characters covers more than 99% of measured FEIS pages and all but one
measured ROD page. The cap should be configurable and recorded in the run manifest.

#### Dedicated EIS retrieval paths

Add three EIS-specific builders:

1. `build_eis_rod_full_read_packets`
   - read every nonempty page of high-confidence ROD documents;
   - use the 12,000-character cap;
   - retain all pages until the EIS per-project cap is applied;
   - give these packets the highest document-text retrieval priority.

2. `build_eis_final_document_packets`
   - for FEIS documents, retain first three, last five, top initiation-cue pages, top
     ROD/decision-cue pages, and top publication/availability pages;
   - use separate quotas for initiation and endpoint evidence;
   - do not let ten high-scoring pages of one role consume the entire Tier D allowance.

3. `build_eis_text_fallback_packets`
   - for a text-bearing project that otherwise emits no packet, retain at least the first
     and last nonempty page of the best available document;
   - mark `retrieval_reason = "eis_text_fallback"`;
   - allow extraction even when the page keyword score is zero.

The fallback directly addresses the 212 no-packet projects that have extractable text.
It must be conservative in selection, but retrieval should not silently discard them.

#### Quotas and caps

Replace the global `top_n=10` behavior for EIS with role-aware quotas, for example:

| Packet purpose | Minimum reserved pages |
|---|---:|
| ROD document pages | all, subject to project cap |
| Initiation cues | 8 |
| Decision/ROD cues | 8 |
| FEIS publication cues | 5 |
| First/last structural pages | 8 |
| Generic fallback | 2 |

These are initial tuning values, not permanent constants. The EIS project cap can remain
150 for the first experiment, but the audit must report cap hits and packets lost by
purpose. Raise it only if validated recall is being lost at the cap.

#### Deduplication

Deduplicate by source identity:

```text
project_id + document_id + page_number + normalized source span
```

When two packets represent the same page, retain the packet with:

1. authoritative source tier;
2. longer context among document-text packets;
3. higher retrieval priority when context length is equal;
4. higher role-specific score as a final tie-break.

Do not hash full page text and then retain a shorter slice solely because its tier is
earlier.

For the first implementation slice, keep the existing content hash but apply the
essential rule above: metadata remains authoritative, while same-page EIS document-text
packets prefer the longer context. A source-identity deduplication rewrite remains a
later structural cleanup and should not block validation of the cap change.

**`deduplicate_packets()` is shared across CE/EA/EIS — gate the new rule and prove
byte-equivalence.** The function at `02_retrieve.py:882` keeps the lowest `tier_order`
value (`tier_b=1` beats `tier_d=3`) and has no length tiebreaker today. Implement the
longer-context preference as a conditional on `process_type == "EIS"` so CE/EA dedup is
untouched. Before publishing Phase 2, run the Phase 0 CE/EA equality invariant against
the new code: candidate counts and selected dates for CE and EA must be byte-identical.
If the rule is (intentionally or not) applied to all processes, this check must pass
before publication regardless.

#### Retrieval diagnostics

Write project-level diagnostics, either as packet columns or a small sidecar:

- `retrieval_status`;
- `text_pages_available`;
- `packets_emitted`;
- `packet_cap_hit`;
- `rod_pages_emitted`;
- `feis_pages_emitted`;
- `init_pages_emitted`;
- `decision_pages_emitted`;
- `fallback_used`;
- `retrieval_loss_reason`.

### 5.3 `03_extract_candidates.py`: anchored filtering and rejection visibility

#### Local exclusion windows

For EIS, apply citation and exclusion keywords to a local date window rather than the
whole packet. Start with the containing sentence plus 120 characters on each side.
Retain the full packet as evidence, but make accept/reject decisions from the anchored
window.

Apply the same locality principle to `REJECT_CUES`. A historical sentence elsewhere on
a dense FEIS page should not reject an unrelated signature or publication date.

Do not simply broaden regex patterns during this phase. First measure whether misses are:

- page not retrieved;
- date text after truncation;
- regex pattern miss;
- exclusion rejection;
- short-context guard;
- role ambiguity.

#### Rejection audit

Write rejected date mentions to `timeline_candidate_rejections.parquet` with:

- project/document/page identifiers;
- raw date and parsed date;
- local anchored context;
- rejection reason;
- matched exclusion keyword or regex;
- retrieval tier and reason;
- document role.

This sidecar is diagnostic only and must never feed selection directly. It makes future
recall work measurable instead of requiring repeated raw-page reconstruction.

#### Candidate roles

Treat regex roles as evidence features, not the exclusive selection pool. Keep
`candidate_role`, but add:

- `role_rule_version`;
- `event_evidence_flags`;
- `event_negative_flags`;
- `is_authoritative_rule`;
- `eligible_for_model`.

Candidates labeled `unknown`, `historical`, or `body_text` must remain available to the
classifier when not hard-rejected. A high calibrated model probability should be able to
promote them into an endpoint pool without rewriting the regex role.

**`unknown` candidates are NOT a hypothetical — they already win selections today.** A
query against the current production candidates parquet shows **1,268 EIS decision dates
are currently selected from an `unknown`-role candidate** (initiation: 0). So `unknown`
candidates are already eligible for and participating in decision selection under the
existing code. Two consequences:

1. The 148 `unknown` candidates from the fallback recovery do **not** need a role
   promotion to be considered — they enter the classifier and selection as-is. The §5.3
   role-feature work is about *auditability and future EIS event modeling*, not about
   unblocking these candidates.
2. **This is a Phase 4 hazard, not a benefit (see §5.6).** If the Phase 4 eligibility
   pools are defined purely from explicit roles (`clear_decision`, `proxy_decision`,
   `final_eis`, authoritative sources), they will *exclude* the `unknown` candidates that
   currently supply 1,268 decision dates — turning a coverage refactor into a coverage
   regression. The pool definition must preserve access for `unknown` candidates that
   clear the calibrated probability threshold.

### 5.4 `04_classify_candidates.py` and `04b_calibrate.py`: event probabilities

For EIS, model three distinct events:

- initiation;
- ROD decision;
- Final EIS publication/availability.

Continue to use target-anchored `[[date]]` context. Add the explicit document-role and
retrieval features to the model input or downstream calibration.

Calibration must be evaluated separately for EIS. A global threshold is acceptable only
if EIS reliability plots show it is calibrated. Report:

- candidate precision/recall at thresholds from 0.1 to 0.9;
- project-level top-1 and top-5 recall by event;
- calibration error by event;
- performance by document role and retrieval reason.

Authoritative register candidates remain pass-through and do not need model acceptance.

Before choosing `T_init`, `T_rod`, or `T_feis`:

1. preserve the existing 28 EIS frozen projects as a protected regression set;
2. freeze event-specific threshold sets from the existing EIS labels;
3. target at least 30 verified positive and 30 verified negative projects per event;
4. treat any threshold as provisional when that minimum cannot be reached.

The existing ROD labels are large enough to start this construction. Initiation and FEIS
need additional labeling before their thresholds can be treated as stable.

### 5.5 `05b_rank.py`: ordering only

Keep LambdaRank for within-project ordering, but remove all interpretations of its raw
score as a probability that an event exists.

The ranker training set currently drops projects whose gold answer is `none`, which is
valid for ranking but proves the score cannot be an existence gate. Use:

- calibrated event probability for eligibility;
- explicit source/cue rules for authoritative eligibility;
- LambdaRank only to order candidates within the eligible pool.

If a learned project-level existence decision is desired later, train a separate binary
model with positive and `none` projects. Do not overload the ranker.

### 5.6 `05_select_dates.py`: explicit pools and abstention

Reset all `selected_for_*` columns to `False` before selecting any project. Mark exactly
one candidate per selected event.

Build event pools independently of `candidate_role`:

> **Regression guard (mandatory).** 1,268 EIS decision dates are currently selected from
> `unknown`-role candidates (§5.3). Defining the pools "independently of `candidate_role`"
> must mean **role-agnostic inclusion via calibrated probability**, not an allow-list of
> explicit decision roles. Any pool definition that admits only `clear_decision` /
> `proxy_decision` / `final_eis` / authoritative candidates will drop those 1,268 dates.
> Before/after this change, count selected decision dates whose winning candidate is
> `unknown`-role; the Phase 4 ship gate "no existing validated selection is lost without
> an explicit reason" applies directly to this cohort.

#### Initiation eligibility

A candidate is initiation-eligible when any of the following holds:

- authoritative NOI/register initiation;
- authoritative initiation rule;
- `p_init_cal >= T_init`, where `T_init` is selected on frozen EIS validation data.

Then apply chronology and rank eligible candidates. Chronology failures should produce a
separate reason code; they must not be conflated with ranker rejection.

#### ROD decision eligibility — bias hard toward keeping RODs

Real ROD dates are scarce and must be preserved aggressively: **a ROD candidate is dropped
only in extreme cases, never by a soft probability cutoff.** A candidate is ROD-eligible
when it is an authoritative register ROD, OR sits on a ROD-type document, OR has explicit
local ROD issuance/signature language — i.e. the current `_eis_rod_pool` definition stays
as the inclusion rule. `p_dec_cal` / the ranker are used **only to ORDER multiple ROD
candidates within a project**; they must never make a project's *only* ROD ineligible.

A ROD candidate is excluded only by an explicit hard-reject rule:

- it is a citation or a reference to a *different* project's ROD (e.g. "consistent with the
  Programmatic ROD issued …"), detected by the same windowed citation/exclusion logic
  used elsewhere;
- it is a schedule/expectation, not an issuance ("ROD expected Q3 2024");
- it is future-dated, pre-1970, or unparseable.

If none of those fire, the ROD is eligible. When a project has a ROD candidate but it is
ambiguous (e.g. competing dates, no clear issuance language), **route to adjudication —
never silently drop it and never fall through to FEIS while a plausible ROD exists.**

#### ROD-first / FEIS-fallback is a PRESERVED INVARIANT — do not redesign it

**DECISION (locked): keep the current tiered behavior and the current storage.** A usable
ROD is always used when one exists; the FEIS publication date is used as the decision date
only when no usable ROD is present. The chosen date stays in the **single `decision_date`
column**, exactly as today. This is the lowest-risk option and is correct best practice:
the EIS "decision date" is one coherent concept — "when the review concluded, by ROD or by
FEIS proxy when no ROD exists." Do **not** move FEIS dates into a separate `final_eis_date`
column, do **not** migrate the headline metric, and do **not** flip
`EIS_FINAL_EIS_ENABLED`. The existing tiering in `_select_eis_decision()`
([05_select_dates.py:404-433](phase2/code/deliverable04/05_select_dates.py)) — ROD pool
first, FEIS pool only when `has_rod` is False — is the behavior to keep.

The only additions in Phase 4:

1. **Transparency tag.** Emit `decision_event_type` (`rod` | `final_eis_fallback`) alongside
   the existing `decision_is_feis_fallback` / `has_rod` flags, so the report can label a
   date as a FEIS proxy without moving the data. No consumer migration; `decision_date`,
   `08_analyze.R:207`, the figures, and the report all keep working unchanged.
   **`08_analyze.R` must be wired to surface this split.** It currently ignores
   `has_rod` / `decision_is_feis_fallback` (it only derives `endpoint_source_type` =
   `decision` vs `final_eis`, which is always `decision` since `final_eis_date` is null), so
   a ROD-vs-FEIS figure is impossible as-is. Add a diagnostic table
   (`d4_eis_decision_event_type.csv`: ROD vs FEIS-fallback counts/share by energy type) and
   a figure, reading `decision_event_type` (or `has_rod` / `decision_is_feis_fallback`).
   This is the filterable ROD/FEIS breakdown the report needs.
2. **ROD ordering, not a stricter existence gate.** The current `_eis_rod_pool` inclusion
   rule stays; `p_dec_cal` / the ranker only *order* competing ROD candidates within a
   project (see "ROD decision eligibility" below). A project's only ROD is never made
   ineligible by a probability cutoff — RODs are dropped only by explicit hard-reject rules.

**Why this does not lower the timeline count.** The current 842 complete timelines and 2,204
EIS decisions **already include 1,564 FEIS-fallback dates** (638 ROD + 1,564 FEIS;
`final_eis_date` populated for 0). Keeping the fallback feature keeps those dates — there is
no loss. The count is preserved by construction because nothing moves out of `decision_date`.

> **Count protection — Phase 4 must report any drop.** With conservative ROD eligibility
> (RODs dropped only by explicit hard-reject rules — citation / wrong-project reference /
> future / pre-1970), the only projects that can lose a decision date are those whose sole
> decision candidate was a genuine false positive AND that have no FEIS — i.e. corrections,
> not real losses. Even so, the Phase 4 regression diff **must list every project that went
> from "has decision date" to "none," with its rejection reason,** plus the before/after
> `has_rod` / `decision_is_feis_fallback` split. Hard floor: decision coverage ≥ 2,204 and
> complete timelines ≥ 842 unless each drop is individually justified as a false-positive
> correction.

#### Month granularity

Do not globally discard month-level EIS events before event-specific selection.

- A month-only ROD remains too coarse for exact duration and should normally route to
  adjudication.
- A month-level FEIS publication can be a valid endpoint at month granularity.
- Never impute a day before selection; midpoint imputation is a reporting transformation,
  not evidence.

`EIS_GAP_EXEMPT` does not affect this behavior. It only disables the historical-gap rule
in `_apply_historical_gap_rule`; the separate block controlled by
`MONTH_DECISION_PROCESSES = {"CE"}` (`05_select_dates.py:75`, applied at line ≈697)
currently suppresses month candidates for EIS.
The correction must therefore be in event-specific EIS selection, not in the gap rule
and not by globally enabling month decisions.

**Implementation detail — do NOT add `"EIS"` to `MONTH_DECISION_PROCESSES`.** That set
gates a single suppression block that fires for *all* decision candidates; adding EIS
would un-suppress months for ROD and FEIS together, which is exactly the conflation the
month rule is meant to avoid. Instead, implement the escape *inside* the EIS event
selection so the two events are handled separately:

- **FEIS-publication pool:** month-granularity candidates are valid endpoints — keep them
  at `granularity="month"` (no day imputation, no duration).
- **ROD pool:** month-granularity candidates are too coarse for exact duration — do not
  hard-suppress them (that loses them entirely); set `route_to_llm = True` and skip them
  in deterministic selection so adjudication can resolve them later.

#### Selection diagnostics

Emit:

- `initiation_selection_status`;
- `decision_selection_status`;
- `final_eis_selection_status`;
- `initiation_pool_size`;
- `decision_pool_size`;
- `final_eis_pool_size`;
- `selection_abstention_reason`;
- selected candidate probability and rank;
- `chronology_candidates_removed`.

### 5.7 `06_adjudicate_llm.py`: bounded adjudication, not hidden extraction

Use LLM adjudication only after deterministic retrieval and extraction metrics are
materialized.

For candidate adjudication:

- send separate top candidates for initiation, ROD, and FEIS;
- retain authoritative candidates regardless of model label;
- start with top five per event, deduplicated by date and context;
- include document role and calibrated probability;
- require the response to select a provided candidate ID.

For document recovery:

- use the new EIS recovery packets, including fallback packets;
- do not run on projects with no packet evidence;
- write recovered raw mentions back through the same candidate schema before selection,
  rather than directly mutating final dates;
- validate document-recovery precision on a frozen sample before production use.

Paid adjudication is not part of the extraction-recall acceptance gate.

### 5.8 `_run.py`: reproducible EIS partition builds

The orchestrator must execute:

```text
02 -> 03 -> 04 -> 04b --apply -> 05b --apply -> 05
```

Confirmed defect: the current `_run.py` runs only `02 -> 03 -> 04 -> 05` (see its stage
list) and **omits `04b` and `05b`**, so a fresh candidate pool gets selected with stale or
missing calibrated/ranker scores. Add the two missing stages in order.

**Isolation-flag reality (affects the validation recipe).** `02`, `03`, `04`, `05`, `07`
support `--sample-ids`; `04b` and `05b` support **`--run-dir` but NOT `--sample-ids`**. So
an isolated EIS validation cannot pass a sample-id subset through calibration/ranking —
`04b`/`05b` operate on the **entire candidate set in the run directory**. The workflow is:
run `02`/`03` (optionally sample-id-scoped) into an isolated `--run-dir`, then run
`04 -> 04b --apply -> 05b --apply -> 05` against that whole run-dir. Do not expect
`--sample-ids` to thread through `04b`/`05b`; scope the sample at `02`/`03` instead.

Each EIS run should use an isolated run directory. After QA passes, publish by atomically
replacing only the EIS partition in production outputs.

The manifest must record:

- source parquet hashes;
- code commit;
- retrieval configuration;
- classifier, calibrator, and ranker versions;
- project ID set;
- row/project counts at every stage;
- stage start/end times and status.

Do not use append semantics to preserve old EIS candidates or selected flags across a
full EIS rebuild.

## 6. Implementation Phases

### Phase 0: freeze baseline and tests

Before behavior changes:

1. Materialize the current EIS project, packet, candidate, and selected-date metrics.
   Define `has_extractable_text` as at least one indexed EIS page with
   `length(trim(page_text)) > 100`. The current denominator is 4,119 projects, so the 99%
   retrieval gate requires at least 4,078 projects to emit a packet.
   **Also compute, before any behavior change, two numbers that gate later phases:**
   - **Joint source ceiling** (§1): count EIS projects with both a recognized endpoint
     document (`decision_doc_score >= 3.5`) and a recognized initiation-evidence source
     (NOI / scoping / application / register). This is the hard ceiling on local-text
     complete timelines. If it is below 2,891, the 70% target requires Phase 5; record
     the adjusted local-achievable target.
   - **`unknown`-role selection reliance** (§5.3, §5.6): count currently selected EIS
     decision dates whose winning candidate is `unknown`-role (baseline: 1,268). This is
     the regression-guard denominator for the Phase 4 pool redesign.
   - **Decision composition** (§5.6 preserved invariant): record the ROD vs FEIS-fallback
     split of current EIS decisions (baseline: 2,204 = 638 ROD + 1,564
     `decision_is_feis_fallback`; `final_eis_date` populated for 0). FEIS stays in
     `decision_date` (option a, locked) — this baseline is the regression denominator for
     the Phase 4 "usable ROD" gate, which must not drop net decision coverage below 2,204.
2. Freeze representative validation sets:
   - zero-candidate projects;
   - ROD projects;
   - FEIS-no-ROD projects;
   - DEIS-only projects;
   - projects with both candidate types but missing dates;
   - selected dates currently judged correct.
3. Add automated invariants:
   - 4,130 EIS output rows;
   - CE/EA substantive output equality (the dedup-rule guard from §5.2 — candidate counts
     and selected dates byte-identical for CE/EA);
   - no future dates;
   - selected initiation precedes selected endpoint when comparable;
   - at most one selected flag per project/event;
   - all selected IDs exist in the current candidate partition;
   - **decision coverage does not regress:** FEIS stays in `decision_date` (§5.6, option a),
     so completion is still `!is.na(initiation_date) & !is.na(decision_date)`; projects with
     a decision date must not fall below 2,204, and complete timelines must not fall below
     842, at any phase (guards the Phase 4 "usable ROD" gate edge case).

### Phase 1: state and orchestration correctness

Implement selected-flag reset, isolated EIS partitions, full stage ordering, manifest
versioning, and atomic publish. This phase should not intentionally change selected dates.

**Dead-code cleanup (do it here, while behavior is frozen).** Option (a) makes several
feature flags and one code path permanently dead; remove them so a later reader does not
mistake them for live options:

- `EIS_FINAL_EIS_ENABLED` (always `False`) and the `_select_eis_final_eis` function it
  gates — the separate-`final_eis_date` path is not used under option (a). **Sweep
  consumers first:** `08_analyze.R` references `final_eis_date` in its `endpoint_date`
  coalesce; either drop the dormant `final_eis_date` column and simplify that line to use
  `decision_date` directly, or keep the column explicitly documented as always-null. Pick
  one and leave no half-wired path.
- `EIS_DETERMINISTIC_DOC_ROD` (always `False`) if it has no live branch.
- Any `if not EIS_TIERED_DECISION:` reversible-fallback branch in `_select_eis_decision`
  once tiering is confirmed permanent — collapse to the tiered path.

Removal must be byte-equivalent on CE/EA and reproduce EIS dates (the flags are already
off, so deleting their dead branches changes nothing at runtime).

**Ship gate:** current EIS dates reproduce except for stale candidate flags and volatile
timestamps; no dead EIS feature flag or unreachable selection branch remains.

### Phase 2: retrieval recall

Implement the 12,000-character EIS cap in both Tier B and Tier D, longer-context-aware
deduplication, ROD full-read, FEIS role quotas, and no-packet text fallback.

Run `02 -> 03` only in an isolated EIS directory.

**Ship gates:**

- at least 99% of EIS projects with extractable text emit a packet;
- zero-candidate projects fall materially from 664;
- candidate precision on the frozen sample does not decline;
- all newly recovered candidates are attributable to a retrieval reason;
- packet volume and runtime remain within agreed limits.

### Phase 3: extraction locality and rejection audit

Implement local EIS exclusion windows and the rejection sidecar. Re-run `03` from the
Phase 2 packets.

**Cheap interim before the full sidecar.** The full
`timeline_candidate_rejections.parquet` (§5.3) is a Phase 3 deliverable, but the Phase 3
ship gate ("manually reviewed recovered mentions ≥90% valid", "every rejected date can be
assigned to a documented reason") needs rejection tracing to evaluate at all. Rather than
re-running `03` in debug mode per sample project, first add a single nullable
`rejection_reason` string to the `03` candidate output, populated from the existing
`_should_reject_date` return value, for every date mention that was considered but not
emitted. This is ~10 lines, produces exactly the data the gate needs, and the full
sidecar (with anchored context, matched keyword/regex, document role) expands on it
within the same phase.

**Ship gates:**

- manually reviewed recovered mentions are at least 90% valid date mentions;
- valid milestone recall improves on the frozen sample;
- citation and historical false positives do not materially increase;
- every rejected date can be assigned to a documented reason.

### Phase 4: selection refactor

Remove ranker score gates, create calibrated event pools, preserve ROD/FEIS semantics,
handle month granularity by event, and emit selection diagnostics.

Do not implement this as an unconditional deletion of the current ranker gates. The
replacement eligibility rules and thresholds must be frozen first. Otherwise the change
would convert weak candidates into dates merely because every project has a top-ranked
candidate.

**Entry prerequisite — expand the frozen event sets (this is the thing that currently
blocks Phase 4, so it is listed as a gate, not left implicit).** §2.4 shows the frozen
inventory is too thin for stable thresholds: 28 EIS projects with a decision pick but
only **13 with an initiation pick**, a 46-project ROD sample, a 25-project FEIS sample,
and **no FEIS threshold column in `ranker.csv`**. Concrete, bounded path to the
"30 verified positive + 30 verified negative per event" minimum:

- **Initiation:** review 30 of the **168 projects** that have `p_init_cal > 0.5` AND a
  chronologically valid initiation candidate but are blocked by the ranker gate (§2.3,
  §9). Inspecting `initiation_evidence_text` + the candidate `model_context` for these
  expands the frozen initiation set from 13 to ~43 verified picks and is the highest-value
  labeling because it both unblocks thresholds and directly validates the largest coverage
  lever.
- **FEIS:** the existing 25-project FEIS sample is near the floor; review ~5 more from the
  labeled FEIS set and **add the missing FEIS threshold column to `ranker.csv`**.
- **ROD:** the 46-project sample already meets the minimum; preserve it as the regression
  set.

Treat any threshold derived from a still-undersized event set as provisional and say so in
the run manifest. Do not start the ranker-gate removal until the initiation set is expanded.

**Ship gates:**

- recover the chronologically valid high-confidence initiation cases currently blocked
  by the ranker without reducing initiation precision below the frozen baseline;
- route or resolve the 182 missing-decision projects currently excluded from the EIS
  pool;
- ROD and FEIS precision each meet their frozen validation thresholds;
- no existing validated selection is lost without an explicit reason.

### Phase 5: model and adjudication improvements

Only after Phases 2–4:

- retrain/calibrate EIS event heads if candidate recall is adequate but probabilities are
  weak;
- retrain rankers on the expanded pool;
- add a project-level existence model if needed;
- run bounded LLM adjudication for remaining ambiguous cases;
- add OCR or external-source recovery for the documented source-gap cohort.

## 6.1 Near-term implementation slice

The practical first slice, suitable for an isolated EIS run, is:

1. reset stale selected-candidate flags;
2. use 12,000 characters in both EIS Tier B and Tier D;
3. prefer the longer packet when duplicate EIS document-text packets represent the same
   page;
4. emit first/last text fallback packets when an EIS project otherwise emits no packet;
5. run isolated `02 -> 03` validation and measure packet/candidate recovery.

Items 1–4 are implemented in code. They must be validated before production publication.

Ranker-gate removal is not part of this first slice. It is likely the largest final-date
coverage lever, but it is a selection-policy change rather than a mechanical bug fix.
Implement it in Phase 4 after event-specific eligibility thresholds and frozen samples
are ready.

## 7. Acceptance Metrics

Report all metrics for the full 4,130-project EIS universe and, separately, for projects
with recognized local endpoint documents.

### Retrieval

- projects with extractable text;
- projects with at least one packet;
- no-packet projects;
- packet cap hits;
- page recall by ROD/FEIS/DEIS/OTHER;
- truncation rate and retained-character quantiles.

### Extraction

- projects with any candidate;
- projects with initiation candidates;
- projects with ROD candidates;
- projects with FEIS candidates;
- projects with both initiation and endpoint candidates;
- rejection reasons and recovery source;
- frozen-sample date-mention recall and precision.

### Selection

- selected initiation, ROD, FEIS, endpoint, and complete timelines;
- abstentions by reason;
- chronology losses;
- calibrated-probability distribution for selected and rejected candidates;
- rank of the selected candidate within each eligible pool;
- ROD-versus-FEIS endpoint mix.

### Quality

- exact-date precision at day granularity;
- month/year precision at the stated granularity;
- invalid order;
- implausible duration;
- future dates;
- regression count among previously validated dates;
- CE/EA equality.

## 8. Decision Rules

1. Do not publish a phase that increases coverage but fails its precision gate.
2. Do not interpret retrieval recovery as final-date recovery.
3. Do not add estimated recovery counts across overlapping cohorts.
4. Do not lower calibrated thresholds merely to meet 70%.
5. If local-document candidate coverage remains below 70% after Phase 3, quantify the
   source gap before further model tuning.
6. If the all-EIS complete target is blocked by missing source documents, report:
   - local-text achievable coverage;
   - OCR-recoverable coverage;
   - external-register-recoverable coverage;
   - genuinely unsupported remainder.

## 9. Expected Impact

The currently measured direct opportunities are:

| Opportunity | Observed projects | Confidence |
|---|---:|---|
| No packets despite extractable text | 212 | high retrieval opportunity; date yield unknown |
| Truncation recovers any candidate | 40 | measured |
| Truncation + local exclusion recovers any candidate | 54 | measured |
| Truncation + local exclusion recovers both types | 9 | measured |
| Missing initiation, ranker blocks valid chronology | 645 | direct selection defect |
| High-confidence initiation blocked by ranker | 168 | high-priority subset |
| Missing decision due to EIS pool exclusion | 182 | direct selection/routing defect |
| Filename cue missed by index scoring | 9 | low-impact metadata fix |

These figures overlap and are not a forecast. The largest immediately demonstrated
coverage lever is the initiation selection gate. The largest demonstrated extraction
gap is projects that never receive packets despite having text. The largest strategic
constraint is source availability for projects without a recognized ROD or FEIS.

## 10. Explicit Non-goals

- No Phase 1-style generic first/last document-date fallback.
- No use of LambdaRank scores as probabilities.
- No conversion of FEIS publication dates into untyped ROD dates.
- No broad regex expansion without a rejection audit and frozen validation sample.
- No production mutation from an isolated experiment.
- No CE/EA behavior changes as part of EIS recovery.
- No claim that 70–80% complete coverage is achievable from local text until the
  source-ceiling audit demonstrates it.

## 11. Recommended Execution Order

1. Phase 0 baseline and validation fixtures — **including the joint source ceiling and the
   `unknown`-role reliance count (1,268).** If the joint ceiling is below 2,891, fix the
   adjusted local-text target now and mark Phase 5 mandatory.
2. Phase 1 orchestration/state correctness.
3. Phase 2 retrieval refactor (infrastructure/diagnostic milestone; ~1–2% of coverage).
4. Phase 3 extraction-window and rejection diagnostics.
5. Re-measure the 70% pre-selection candidate target.
6. **Expand and freeze the event-specific label sets (Phase 4 entry prerequisite):
   30 initiation reviews from the 168-project blocked cohort, ~5 FEIS reviews, add the
   FEIS threshold column to `ranker.csv`.**
7. Phase 4 selection refactor — the coverage lever (~985 of the 999-project gap). Preserve
   the 1,268 `unknown`-role decision selections.
8. Re-measure complete timelines and source ceilings.
9. Phase 5 model, OCR, external-source, and LLM work only for the remaining documented
   gaps.

This order keeps the work reversible and answers the key question after each phase:
whether the missing date is absent from the source, absent from retrieval, rejected by
extraction, misclassified, suppressed by selection, or correctly left missing.
