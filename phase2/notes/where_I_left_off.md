# D4 Timeline — Current State and Monday Plan

_Updated June 5, 2026. This is the current handoff for date selection, the learned ranker,
EIS coverage, and next steps. It replaces the previous pending-run narrative._

> **UPDATE 2026-06-08 — EIS Phase A/B/C landed; numbers below are partly stale.** The active EIS
> work and current status live in **`phase2/plans/eis_audit.md`** (see its "Progress / Status"
> section). What changed since this note: production `timeline_project_dates.parquet` is now
> **59,215 rows** (all 4,130 EIS present; was 58,551), with new `final_eis_*` columns; EIS confirmed
> ROD coverage 297→**542** and a flagged `final_eis_date` endpoint (1,952). Phase C is **implemented
> but UNVALIDATED** — ROD-promotion and FEIS precision gates are pending a Codex labeling pass
> (`eis_audit/eis_rod_promotion_sample.csv`, `eis_feis_sample.csv`). Backups in
> `timeline/_backup_phaseBC/`. The EIS coverage tables further down this file pre-date that run._

## Core Issues

1. Initiation selection is weakest, particularly EIS: held-out date accuracy is 0.412.

2. Role filtering blocks valid candidates. Many true initiation dates exist but are labeled unknown or clear_decision, so 05 never considers them.

3. The ranker cannot learn abstention because projects labeled none are excluded from LambdaRank training.

4. EIS decision coverage is extremely low: only 491 / 3,466 projects. This likely involves retrieval, extraction, and source gaps, not merely ranking.

5. The evaluation set is small and already consulted during development, creating overfitting risk. Further tuning should use cross-validation before touching the holdout.

6. Script 06 is stale, still using raw probabilities and only three candidates.

7. Six durations exceed 10,000 days and require review.

## Executive Summary

The current D4 pipeline run is complete and stable. The SetFit candidate classifier is strong,
the LightGBM ranker is trained and applied, and the ranker feature-scope bug has been fixed.
Decision selection is reasonably strong. Initiation selection, especially for EIS projects, is the
main remaining model weakness. At corpus scale, missing EIS decisions are an even larger coverage
problem than EIS initiation ranking.

Do not launch a full pipeline rebuild by default. The next work should first distinguish selection
failures from retrieval/extraction/source failures, test selector changes offline, and change
production code only if cross-validated results improve.

## Trusted Baseline

### Candidate classifier

The calibrated SetFit two-head classifier is no longer the principal bottleneck:

| Head | Frozen `test_v2` F1 |
|---|---:|
| Initiation | 0.896 |
| Decision | 0.892 |

### Learned candidate ranker

`05b_rank.py` trains separate LightGBM LambdaRank models for initiation and decision.

| Head | Held-out top-1 | MRR |
|---|---:|---:|
| Initiation | 0.605 | 0.727 |
| Decision | 0.791 | 0.873 |

Per-process held-out top-1:

| Process | Initiation | Decision |
|---|---:|---:|
| CE | 0.692 | 0.706 |
| EA | 0.692 | 0.889 |
| EIS | 0.471 | 0.750 |

### End-to-end selection

The current production configuration is Config D:

- Learned score replaces the heuristic `ranking_score`.
- Existing score gates act as abstention/confidence thresholds.
- Role tiers, chronology, granularity rules, and other hard constraints remain in `05`.
- `agreement_count` is computed per project consistently at train, evaluation, and apply time.

The 60-project deterministic ranker holdout is the honest quality estimate:

| Process | Init overall / date accuracy | Decision overall / date accuracy |
|---|---:|---:|
| ALL | **0.717 / 0.605** | **0.850 / 0.791** |
| CE | 0.850 / 0.769 | 0.850 / 0.824 |
| EA | 0.800 / 0.692 | 0.900 / 0.889 |
| EIS | **0.500 / 0.412** | 0.800 / 0.500 |

The full 300-project figures are useful for A/B debugging but include the 240 projects used to
train the ranker and are not unbiased:

| Head | Overall | Date accuracy when gold has a date |
|---|---:|---:|
| Initiation | 0.700 | 0.576 |
| Decision | 0.907 | 0.897 |

`_eval_selection_vs_gold.py` now reproduces the ranker's deterministic split and reports
`HOLDOUT`, `TRAIN`, and `ALL` separately.

## Current Full-Corpus Coverage

`timeline_project_dates.parquet` contains all 58,551 projects, has no duplicate project IDs,
and has fresh run timestamps for every row.

| Status | Projects |
|---|---:|
| `missing_initiation` | 28,623 |
| `complete_clear` | 10,015 |
| `missing_decision` | 7,211 |
| `missing_both` | 6,373 |
| `complete_with_proxy` | 6,315 |
| `manual_review` | 14 |

By process:

| Process | Total | Has initiation | Has decision |
|---|---:|---:|---:|
| CE | 52,093 | 20,583 | 42,468 |
| EA | 2,992 | 1,575 | 2,008 |
| EIS | 3,466 | 1,397 | **491** |

EIS statuses:

| Status | EIS projects |
|---|---:|
| `missing_both` | 1,839 |
| `missing_decision` | 1,136 |
| `missing_initiation` | 230 |
| `complete_clear` | 183 |
| `complete_with_proxy` | 73 |
| `manual_review` | 5 |

QA observations:

- No negative durations.
- No dates after the current date.
- Six durations exceed 10,000 days and need review.
- Initiation dates range from 1970-05-15 to 2025-04-15.
- Decision dates range from 1970-10-02 to 2026-04-30.

## What Was Fixed

### Real ranker bug

`05b_rank.py::build_features` previously calculated `agreement_count` over whichever frame it
received. Training/evaluation passed one project's candidates, while `--apply` passed all 414,637
candidates. Common dates therefore received pool-global counts during application that were far
outside the training distribution.

The feature is now grouped by `project_id`, so its meaning and magnitude are consistent in all
modes. The ranker was retrained and reapplied after this fix.

### Learned-score wiring

The negative learned scores and the existing `> 0` / `> -2` gates are intentional. They provide
the current abstention behavior. Separating heuristic eligibility from learned ranking caused
false positives to explode and reduced selection accuracy to roughly 0.31 initiation and 0.40
decision. The learned-score replacement should remain unless a tested alternative improves it.

Heuristic-only selection is also poor, roughly 0.30 initiation and 0.39 decision overall. The
learned ranker is providing real value.

## Core Remaining Issues

### 1. Initiation selection is the main model weakness

Initiation held-out date accuracy is 0.605 overall and 0.412 for EIS. The largest full-corpus status
is `missing_initiation`.

The failure is not purely an extraction gap. Among the 300-project gold errors, many true initiation
candidates already exist but have roles that `05` excludes:

- 38 missed gold initiation candidates are labeled `unknown`.
- 18 are labeled `clear_decision`.
- Only a small number of missed gold candidates are labeled `clear_initiation` or
  `proxy_initiation`.

`05` currently considers only `clear_initiation` and `proxy_initiation` for its normal initiation
pass. The learned ranker may score a cross-role or unknown candidate correctly but cannot select it.

### 2. Role tiers and learned ranking conflict

The ranker is trained over all candidates, while production selection first restricts the pool by
regex role. Removing role restrictions globally is not safe: simple global-argmax experiments
increase false positives and did not improve the held-out result.

The promising direction is a controlled fallback after the normal role-based path fails, not full
replacement of role tiers.

### 3. The ranker does not learn whether a date exists

Projects whose gold answer is `none` are excluded from LambdaRank training because the loss needs a
positive item in each group. The ranker learns which candidate is best conditional on a real answer,
not whether the project should have an answer.

The current score gates provide empirical abstention, but they are not trained or calibrated as a
project-level existence model. Any broader candidate fallback needs explicit false-positive
protection based on score thresholds, margins, classifier probability, and hard exclusions.

### 4. EIS has both model and source-coverage problems

There are two separate EIS problems:

1. EIS initiation ranking is weak on the holdout.
2. Only 491 of 3,466 EIS projects have a decision date in the full output.

The second problem cannot be solved by selector tuning if the decision evidence was never retrieved
or extracted. Before changing regexes or retrieval, construct an EIS failure funnel:

- No context packets.
- Packets exist but contain no surviving date candidate.
- Candidates exist but no role-eligible candidate.
- Eligible candidates exist but fail score thresholds.
- A candidate is selected and then rejected by chronology/month rules.
- Authoritative register/NOA/ROD source is absent.

This determines whether the correct intervention belongs in `02`, `03`, `05`, or register sourcing.

### 5. Evaluation risk

The honest holdout has only 60 projects, including 20 EIS projects. Some per-process date-accuracy
denominators are smaller still. The holdout has also already influenced debugging decisions.

Do not repeatedly optimize against it. Use stratified project-level cross-validation within the
240 training projects to choose a new selector configuration, and reserve the 60-project holdout
for final confirmation.

### 6. Script 06 remains stale

`06_adjudicate_llm.py` still:

- Uses raw classifier probabilities instead of `p_init_cal` / `p_dec_cal`.
- Uses `ROUTED_TOPK = 3`.
- Has comments and thresholds from the pre-calibration model.

It should eventually use calibrated probabilities and likely top five candidates. Do not launch a
paid API run unattended. Stabilize selection and inspect a dry-run queue before any real calls.

## Recommended Next Work

### Phase 1: Preserve and diagnose

1. Record the current parquet timestamps/checksums and baseline metrics.
2. Produce an EIS failure-funnel CSV for initiation and decision.
3. Produce an initiation error taxonomy by process, candidate role, score, classifier probability,
   chronology, and whether the gold candidate was eligible under current `05` rules.
4. Review the six durations over 10,000 days.

### Phase 2: Build an offline selector experiment harness

Experiments must not rewrite production parquets. Test:

1. Current role-tier baseline.
2. Normal role path plus high-confidence `unknown` initiation fallback.
3. Normal role path plus cross-role initiation fallback.
4. Fallback gated by both `learned_init_score` and `p_init_cal`.
5. Top-score versus second-score margin thresholds.
6. EIS-specific fallback versus a process-neutral fallback.
7. Variants of earliest-initiation selection among high-scoring candidates.
8. Global ranker selection only as a diagnostic baseline, not the expected solution.

Report:

- Overall accuracy.
- Date accuracy when gold has a date.
- False positives.
- Misses.
- Wrong-date mismatches.
- Results separately for CE, EA, and EIS.

### Phase 3: Cross-validation and decision gate

Use stratified five-fold project-level cross-validation over the 240 ranker-training projects.
Select a configuration only if it:

- Improves initiation date accuracy by at least 3 percentage points.
- Does not reduce decision accuracy by more than 1 point.
- Does not materially increase false positives.
- Improves more than one process type, or provides a clear EIS improvement without harming CE/EA.
- Can be expressed as a small, understandable production rule.

Then run the chosen configuration once on the untouched 60-project holdout.

### Phase 4: Conditional implementation and rerun

If a configuration passes:

1. Make the smallest possible change to `05_select_dates.py`.
2. Re-run `05_select_dates.py` only.
3. Run `_eval_selection_vs_gold.py`.
4. Run full QA on `timeline_project_dates.parquet`.
5. Compare status distributions and false-positive counts with Config D.

If no configuration passes, retain Config D. Do not force a production change.

### Phase 5: Address EIS coverage

Use the failure funnel to choose the next work:

- Retrieval gap: improve EIS decision-page/section retrieval in `02`.
- Extraction gap: target specific missed date contexts in `03`.
- Role gap: fix narrowly demonstrated prelabel patterns or use the validated fallback in `05`.
- Source gap: prioritize Federal Register NOA/ROD evidence and review-status BLM case numbers.
- Existing-candidate ambiguity: consider calibrated LLM adjudication after a dry run.

A full `02 -> 04` rebuild is justified only if the failure funnel shows a retrieval, extraction, or
classification change is necessary.

## Most Promising Production Design

Keep the existing role-based selector as the primary path. If it finds no initiation date, allow a
fallback candidate from the broader pool only when all of the following hold:

- `learned_init_score` exceeds a validated threshold.
- `p_init_cal` supports initiation.
- The candidate precedes the selected decision.
- The candidate is not historical, review, reject, or otherwise hard-excluded evidence.
- The top initiation score is clearly separated from the runner-up.

Mark such selections as fallback/proxy-derived so downstream analysis can exclude them in a
sensitivity check.

This design addresses true dates currently labeled `unknown` or cross-role without giving up the
false-positive protection of the current role tiers and score gates.

## Monday Deliverables

The next work session should produce:

- Baseline-versus-experiment comparison CSV.
- EIS failure-funnel CSV.
- Updated held-out evaluation and 50-project report.
- Full-corpus QA report.
- A decision to retain Config D or adopt one validated fallback.
- Exact rerun commands and remaining risks.

No paid API calls and no full pipeline rebuild should occur without an explicit decision after the
offline diagnostics.

## Current Files

| Purpose | Path |
|---|---|
| Selection | `phase2/code/deliverable04/05_select_dates.py` |
| LightGBM ranker | `phase2/code/deliverable04/05b_rank.py` |
| End-to-end evaluator | `phase2/code/deliverable04/_eval_selection_vs_gold.py` |
| Gold candidate picks | `phase2/output/deliverable04/project_gold_sample.csv` |
| Gold dates | `phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet` |
| Candidate pool | `phase2/data/analysis/timeline/timeline_candidates.parquet` |
| Project output | `phase2/data/analysis/timeline/timeline_project_dates.parquet` |
| Evaluation summary | `phase2/output/deliverable04/selection_eval_summary.csv` |
| Evaluation errors | `phase2/output/deliverable04/selection_eval_errors.csv` |
| Holdout visual report | `phase2/output/deliverable04/selection_eval_report.txt` |

## Uncommitted Code to Review

- `phase2/code/deliverable04/05_select_dates.py`
- `phase2/code/deliverable04/05b_rank.py`
- `phase2/code/deliverable04/_eval_selection_vs_gold.py`

Do not stage generated files under `phase2/data/` or `phase2/output/`.
