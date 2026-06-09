# D4 Timeline — Current State & Plan

_Updated 2026-06-09. Authoritative handoff. Supersedes all earlier narratives in this file and the
`eis_audit.md` progress section. Reflects the reorg, the classifier rebuild, the EIS labeling
round, the guardrails, and the full 05+08 run over the rebuilt pool._

---

## TL;DR — where we are

- The full pipeline ran end-to-end over the rebuilt pool: **`timeline_project_dates.parquet` =
  59,215 projects** (all 4,130 EIS reconciled in). **06 (LLM) has NOT been run** — deliberately
  deferred to first "kick the tires" on data quality.
- **Decision coverage: CE 82.2%, EA 67.0%, EIS 53.4%.** But coverage ≠ complete timelines (see below).
- The classifier is strong; the tiered EIS decision + FEIS-fallback improved EIS; **EA regressed**
  and needs investigation; **EIS is still ~22 pts under the Phase 1 baseline (75.2%)** and that gap
  is an **extraction-recall** problem, not a selection or LLM problem.

## Coverage — read carefully (decision-only vs complete)

| process | decision coverage | **complete** (both dates) | complete_clear | duration-usable |
|---|---:|---:|---:|---|
| CE | 82.2% (42,821) | **29.4%** (15,327) | 16.6% (8,634) | ~8,600 clean timelines |
| EA | 67.0% (2,004) | ~46% | 42.3% (1,265) | ~1,265 |
| EIS | 53.4% (2,207) | ~20% | 10.4% (430) | ~430 |

- **"Decision coverage" is decision-date-present, NOT a full timeline.** CE is 82.2% decision but
  only 29.4% have BOTH dates — CE **initiation** coverage is just 40.6% (structurally rare; only
  BLM register supplies CE start dates). Complete can't exceed the init ceiling.
- Duration analysis uses `complete_clear` only. Headline medians: **CE 18 d, EA 74 d, EIS 793 d (26 mo)**.

## Phase 1 baseline comparison (the goal: beat Phase 1 D3 coverage)

| process | D4 now | Phase 1 baseline | status |
|---|---:|---:|---|
| EIS | 53.4% | **75.2%** | ~22 pt gap → needs extraction recall |
| CE | 82.2% | (not yet located) | likely competitive |
| EA | 67.0% | (not yet located; prior D4 run 89.5%) | **regression — investigate** |

TODO: locate the Phase 1 CE/EA decision-coverage baselines so we know exactly where we stand.

## What's done (this cycle)

1. **Classifier rebuilt** — 3-head SetFit (initiation / decision / **final_eis**), document-type
   gated (final_eis confined to FEIS docs: precision 0.50→0.74), Platt-calibrated (3 heads).
   Frozen-test: init/decision F1 ~0.88; final_eis P0.50/R0.64. True ROD top-5 90%, FEIS top-5 95%.
   `num_iterations=12`; checkpoints pinned to `models/_setfit_checkpoints` (gitignored).
2. **Tiered EIS decision** in 05 — ROD-first, FEIS-fallback, per-project `has_rod` flag; ROD
   outranks FEIS by construction. New cols: `has_rod`, `decision_is_feis_fallback`.
3. **EIS labeling round** — verified EIS decision picks 38 → **78 positive** (36 ROD + 42 FEIS) +
   64 verified `none`. FEIS-fallback recovery recovered 26 ROD-first-suppressed projects.
4. **Guardrails** — frozen `split` on ranker.csv; `frozen_eval_ids.txt` registry (28 protected
   ids); `05b` hard-fails if a training project is in the registry; gold-rank check uses
   frozen-eval only. **A label is training XOR evaluation — never both.**
5. **06 routing gate** in 05 — `route_to_llm` + `decision_confidence_cal` per project
   (LLM_ROUTE_THRESHOLD=0.7). Confident deterministic picks are final; ambiguous / missing-with-
   candidates route. **~30,876 projects flagged route_to_llm** (~$1 for ambiguous-only, ~$37 if
   the coverage-recovery bucket is included; Haiku ~1,500 tok/project).
6. **Repo reorg** — labels are INPUTS under `training/`; outputs are regenerable under `output/`.

## NEXT STEPS (in order)

### 1. Kick the tires on data quality (NEXT — before any API/06 spend)
Manually inspect extracted dates before trusting them:
- Sample `complete_clear` projects per process; verify init/decision dates against the source
  context (`decision_evidence_text`, `initiation_evidence_text` in project_dates).
- Sanity-check duration outliers (flagged `implausible_duration_*`; 6 durations >10,000 days).
- Spot-check EIS ROD vs FEIS-fallback picks (`decision_is_feis_fallback`) — are FEIS dates real
  publication dates?
- Eyeball year-proxy CE decisions (do not over-trust them in headline numbers).

### 2. Investigate the EA regression (parallel)
EA decision coverage dropped 89.5% → 67%. Confirmed cause: **month-suppression rule** drops
month-granularity EA decisions (all 2,004 selected EA decisions are day-only; 381 missing-decision
projects have a month candidate; 473 have a confident candidate not selected). Likely also the
learned-ranker eligibility gate (`ranking_score > 0`). A Codex investigation prompt exists (see
chat 2026-06-09). Fix candidates: allow month-granularity EA decisions (flagged); revisit the gate.

### 3. EIS extraction recall — PARKED (the real coverage lever)
The ~22-pt EIS gap to Phase 1 is missing ROD/FEIS dates that exist in documents but the **regex
extraction (03)** never surfaced (Phase 1 used a fine-tuned BERT). Recovering the ~600 reconciled
date-less EIS + the document-text RODs is the only path past 75.2%. **06 will NOT close this** —
the LLM adjudicates *extracted* candidates; it can't find dates that aren't in the pool. Larger
effort; revisit after tire-kicking.

### 4. Wire & run 06 (DEFERRED until after tire-kicking)
`06_adjudicate_llm.py` is stale (raw probs, 3 candidates). When ready: build per-project packet
(top-k init+decision, ROD/FEIS for EIS, has_rod, scores), call Haiku per routed project, write
back chosen candidate_ids. Decide the routing policy first (ambiguous-only ~$1 vs +coverage-
recovery ~$37). Validate against `frozen_eval_ids` before a full run.

## Key paths (post-reorg)

- **Labels (INPUTS):** `phase2/training/deliverable04/`
  - `classifier.csv` (candidate-level, frozen split) · `ranker.csv` (project-level, frozen split)
  - `eis_validation/` (verified ROD/FEIS yardstick) · `frozen_eval_ids.txt` · `_backups/` (gitignored)
- **Outputs (regenerable):** `phase2/output/deliverable04/`
  - `diagnostics/` (d4_*.csv) · `figures/` (fig_*.png) · `reports/` · `review_queues/` (gitignored)
- **Data/models:** `phase2/data/analysis/timeline/` — `timeline_project_dates.parquet`,
  `timeline_candidates.parquet`, `models/` (gitignored: classifier, ranker, calibrators, checkpoints)
- **Pipeline:** `phase2/code/deliverable04/` 00→08, 04b, 05b; one-off tools prefixed `_`;
  superseded code in `_archived/`.

## Do NOT
- Do not launch a full 02→04 rebuild by default (slow; not needed for tire-kicking).
- Do not run 06 / spend API until data quality is checked and the routing policy is chosen.
- Do not fold a yardstick label into training (guardrail enforces this; respect it).
- Do not publish EA numbers until the regression is run down.

## Known caveats
- Classifier frozen-test is positive-heavy; deployment precision is lower than test.
- Ranker EIS held-out is small (frozen-eval n≈7 ROD) — ranker-vs-classifier is statistically
  inconclusive; classifier `p_dec_cal`/`p_feis_cal` shortlist (top-5 90–95%) is good enough for 06.
- `route_to_llm` bundles ambiguous picks AND missing-with-candidates (coverage recovery) — different
  value; split them when choosing the 06 policy.
- EA `process_type` count (2,992) is oddly below EIS — possible misclassification (separate from
  the coverage regression; both on the EA to-do).
