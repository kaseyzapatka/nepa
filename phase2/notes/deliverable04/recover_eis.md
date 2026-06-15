# D4 Timeline: EIS Recovery Refactor

**Status:** design only; no pipeline changes implemented.

**Baseline date:** 2026-06-15.

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
2. longer context;
3. higher retrieval priority;
4. higher role-specific score.

Do not hash full page text and then retain a shorter slice solely because its tier is
earlier.

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

#### Initiation eligibility

A candidate is initiation-eligible when any of the following holds:

- authoritative NOI/register initiation;
- authoritative initiation rule;
- `p_init_cal >= T_init`, where `T_init` is selected on frozen EIS validation data.

Then apply chronology and rank eligible candidates. Chronology failures should produce a
separate reason code; they must not be conflated with ranker rejection.

#### ROD decision eligibility

A candidate is ROD-eligible when:

- it is an authoritative register ROD;
- it has explicit local ROD issuance/signature evidence; or
- `p_dec_cal >= T_rod` and it is supported by ROD document evidence.

High-`p_dec_cal` candidates without document-role support should route to adjudication,
not disappear and not be auto-selected.

#### Final EIS eligibility

FEIS publication is a separate event selected from:

- explicit filing/publication/availability evidence;
- `p_feis_cal >= T_feis` within an FEIS-supported document.

Preserve `final_eis_date` separately. Define a reporting endpoint:

```text
endpoint_date = ROD date when present, otherwise Final EIS date
```

Do not silently call an FEIS publication date a ROD. If the existing `decision_date`
column must remain ROD-or-FEIS for backward compatibility, emit an explicit
`decision_event_type = rod | final_eis_fallback` and prohibit downstream analysis from
mixing the two without acknowledgement.

#### Month granularity

Do not globally discard month-level EIS events before event-specific selection.

- A month-only ROD remains too coarse for exact duration and should normally route to
  adjudication.
- A month-level FEIS publication can be a valid endpoint at month granularity.
- Never impute a day before selection; midpoint imputation is a reporting transformation,
  not evidence.

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
2. Freeze representative validation sets:
   - zero-candidate projects;
   - ROD projects;
   - FEIS-no-ROD projects;
   - DEIS-only projects;
   - projects with both candidate types but missing dates;
   - selected dates currently judged correct.
3. Add automated invariants:
   - 4,130 EIS output rows;
   - CE/EA substantive output equality;
   - no future dates;
   - selected initiation precedes selected endpoint when comparable;
   - at most one selected flag per project/event;
   - all selected IDs exist in the current candidate partition.

### Phase 1: state and orchestration correctness

Implement selected-flag reset, isolated EIS partitions, full stage ordering, manifest
versioning, and atomic publish. This phase should not intentionally change selected dates.

**Ship gate:** current EIS dates reproduce except for stale candidate flags and volatile
timestamps.

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

**Ship gates:**

- manually reviewed recovered mentions are at least 90% valid date mentions;
- valid milestone recall improves on the frozen sample;
- citation and historical false positives do not materially increase;
- every rejected date can be assigned to a documented reason.

### Phase 4: selection refactor

Remove ranker score gates, create calibrated event pools, preserve ROD/FEIS semantics,
handle month granularity by event, and emit selection diagnostics.

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

1. Phase 0 baseline and validation fixtures.
2. Phase 1 orchestration/state correctness.
3. Phase 2 retrieval refactor.
4. Phase 3 extraction-window and rejection diagnostics.
5. Re-measure the 70% pre-selection candidate target.
6. Phase 4 selection refactor.
7. Re-measure complete timelines and source ceilings.
8. Phase 5 model, OCR, external-source, and LLM work only for the remaining documented
   gaps.

This order keeps the work reversible and answers the key question after each phase:
whether the missing date is absent from the source, absent from retrieval, rejected by
extraction, misclassified, suppressed by selection, or correctly left missing.
