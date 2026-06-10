# D4 Timeline — Where I Left Off (combined handoff)

> **Last updated: 2026-06-10.** This is the single authoritative warm-start note for D4.
> It consolidates three earlier handoffs written at different points:
> - the **2026-06-09** full-pipeline handoff (cross-process coverage, classifier, guardrails, 06 gate);
> - the **2026-06-10** EA decision-coverage recovery note (EA 67% → 74.2%); and
> - the **2026-06-03 → 06-05** classifier-rebuild session narrative (now historical — see Appendix).
>
> Where numbers conflict, the **latest** wins: EA is now **74.2%** (the 67% in the 06-09 note was the
> pre-recovery state), and the "EA regression" that the 06-09 note listed as a next step is **resolved**
> (see the EA section). EIS and CE are unchanged from 06-09.

---

## TL;DR — where we are

- The full pipeline ran end-to-end over the rebuilt pool: **`timeline_project_dates.parquet` ≈
  59,215 projects** (all 4,130 EIS reconciled in). **06 (LLM) has NOT been run** — deliberately
  deferred to first "kick the tires" on data quality.
- **Decision coverage now: CE 82.2%, EA 74.2%, EIS 53.4%.** Coverage ≠ complete timelines (see below).
- The classifier is strong and calibrated. **EA** has been recovered from a 67% regression up to 74.2%
  (still below the old 89.5% D4 run — see EA section). **EIS** remains ~22 pts under the Phase 1
  baseline (75.2%); that gap is an **extraction-recall** problem, not a selection or LLM problem.

## Coverage — read carefully (decision-only vs complete)

| process | decision coverage | **complete** (both dates) | complete_clear | duration-usable |
|---|---:|---:|---:|---|
| CE | 82.2% (42,821) | **29.4%** (15,327) | 16.6% (8,634) | ~8,600 clean timelines |
| EA | **74.2% (2,220)** | **48%** (1,434) | ~42% | ~1,434 |
| EIS | 53.4% (2,207) | ~20% | 10.4% (430) | ~430 |

- **"Decision coverage" is decision-date-present, NOT a full timeline.** CE is 82.2% decision but
  only 29.4% have BOTH dates — CE **initiation** coverage is just 40.6% (structurally rare; only the
  BLM register supplies CE start dates). Complete can't exceed the initiation ceiling.
- The EA *complete-timeline* figure (1,434) moved less than EA decision coverage because most EA
  recoveries are **decision** dates (register/FONSI signatures) with no matching **initiation** —
  initiation is the separate, harder lever (see EA → deferred §4).
- Duration analysis uses `complete_clear` only. Headline medians: **CE 18 d, EA 74 d, EIS 793 d (26 mo)**.

## Phase 1 baseline comparison (the goal: beat Phase 1 D3 coverage)

| process | D4 now | Phase 1 baseline | status |
|---|---:|---:|---|
| EIS | 53.4% | **75.2%** | ~22 pt gap → needs extraction recall (open) |
| CE | 82.2% | (not yet located) | likely competitive |
| EA | **74.2%** | (not located; prior D4 run 89.5%) | recovered from 67%; still under 89.5% |

TODO: locate the Phase 1 CE/EA decision-coverage baselines so we know exactly where we stand.

---

## Cross-cutting pipeline state (all processes)

What's done this cycle, independent of process:

1. **Classifier rebuilt** — 3-head SetFit (initiation / decision / **final_eis**), document-type
   gated (final_eis confined to FEIS docs: precision 0.50→0.74), Platt-calibrated (3 heads).
   Frozen-test: init/decision F1 ~0.88; final_eis P0.50/R0.64. True ROD top-5 90%, FEIS top-5 95%.
   `num_iterations=12`; checkpoints pinned to `models/_setfit_checkpoints` (gitignored).
2. **Tiered EIS decision** in 05 — ROD-first, FEIS-fallback, per-project `has_rod` flag; ROD
   outranks FEIS by construction. Cols: `has_rod`, `decision_is_feis_fallback`.
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

---

## EA — status & what's left

### Where EA stands (74.2%, recovered)

EA **decision** coverage: **66.98% → 74.2%** (2,004 → 2,220 of 2,992). CE (82.2%) and EIS (53.4%)
were **byte-identical** throughout — every change was EA-gated and validated.

| Metric | Before | Now |
|---|---:|---:|
| EA decision coverage | 2,004 (67%) | **2,220 (74%)** |
| EA initiation coverage | ~1,620 (54%) | 1,675 (56%) |
| EA complete (both endpoints, the boxplot) | 1,371 (46%) | **1,434 (48%)** |

The **EA regression that the 06-09 handoff listed as an open investigation is now resolved.**
Root cause: `05_select_dates.py` used the LightGBM **ranker score as an absolute eligibility gate**
(`>0`), but the ranker is trained only on groups that contain a positive → the score has no
"decision exists" meaning, so valid decisions were dropped. The month-suppression rule compounded it
(dropped month-granularity EA decisions). Fix: **decouple eligibility from ordering** (the ranker
orders; cue/source decides eligibility).

### How EA selection works now

All in `05_select_dates.py::_select_ea_decision()` (EA-only branch, no feature flag — permanent).
Tier order:

1. **Cascade** (unchanged from CE/EA): clear_decision `ranking_score>0` → proxy → body.
2. **Tier EA-1 register gap-fill:** authoritative BLM/DOE Tier A *day* register date, bypasses the gate.
3. **Tier EA-2 strong-cue (Phase C):** `clear_decision` day with `role_confidence_score==5.0` (real
   FONSI / Field-Manager / digital signature), bypasses the gate, hard negatives via `EA_STRONG_NEG_RE`.
4. **No-FONSI Final-EA month proxy:** last resort when project has **no FONSI doc**; event-bound via
   `EA_MONTH_ISSUANCE_RE` + `EA_MONTH_NEG_RE`; stays `granularity="month"` (no duration), flagged
   `ea_decision_fea_month`.

**Phase C retrieval** (`02_retrieve.py::build_ea_decision_full_read_packets`): EA-only; reads EVERY
page of each `decision_doc_score>=4.5` document (the short FONSI/ROD, median 4pp) at an 8000-char
limit, tier `ea_decision_full_read`. This surfaces signature dates that first/last/cue sampling
missed (~half of endpoint dates, per the Phase 1 vs Phase 2 candidate comparison).

**Phase C labeling** (`03_extract_candidates.py`): EA-only escape in the specialist-sheet
disambiguation — when a decision-authority title (`EA_DECISION_AUTHORITY_RE`: field/district manager,
authorizing official, state director, …) is present, an EA signature date stays `clear_decision`
instead of being downgraded to `review`. `process_type` is plumbed into `_prelabel_role`.

**Commits on `desktop`:**

| Commit | Phase | Mechanism | Recovered |
|---|---|---|---:|
| `201fb4d` | **B-1** | Authoritative BLM/DOE register dates bypass the learned-ranker gate | +55 |
| `1c52aad` | **B-2m** | No-FONSI Final-EA month proxy (event-bound, flagged, midpoint→15th) | +18 |
| `8f4f106` | **2a** | Gate-decoupled strong-cue signature tier (role_confidence==5.0) | +45 |
| `deee8e8` | **Phase C** | Full-read of the EA decision document + EA-only signature labeling | +99 |
| `c94f9fe` | merge | merge into desktop | |
| `fd88cf5` | figures | regenerated D4 figures | |

**Infra:** `04b_calibrate.py` and `05b_rank.py` gained `--run-dir` for isolated re-runs.
`_audit_ea_decision_recall.py` reproduces the 988-project failure funnel (Phase A audit).

### ⚠️ EA data-provenance caveat (important for next session)

The desktop EA **data** was produced by **EA-only isolated runs + row merges**, not a single clean
`_run.py`. Specifically, **Phase C was run on still-missing EA only** (~869 projects) — the existing
2,122 EA decisions were NOT re-run through the full-read. Consequences:

- Code and data **agree** (data is the output of the merged code), but it's stitched, not one run.
- A few **existing** EA decisions might have a better date available that the full-read would find.
- **Do NOT** do a blanket full `02→08` rebuild to "clean this up" — it re-runs CE (52k projects,
  hours) and risks shifting the stable CE/EIS numbers (production data accumulated over many runs;
  a fresh rebuild may not reproduce them byte-identical). Keep EA changes EA-scoped.

Backups of every merge step: `phase2/data/analysis/timeline/_backups/ea_{b1,2a,2b}_*/`.

### EA — deferred recovery levers (none touched)

1. **All-EA full-read pass** *(highest value, moderate effort)* — re-run `02→03→04→04b→05b→05` over
   **all** EA (not just still-missing) so the full-read also corrects existing sub-optimal dates.
   ~30–60 min (04 SetFit is the long pole). Changes some existing dates → review them. Use
   `--process EA --run-dir <iso>` then merge EA rows (the subset guard forces the run-dir path).
2. **OCR (Phase D)** *(high effort, blocked)* — ~175 still-missing EAs have **image-only scanned
   FONSIs** (no text). Needs `documents.parquet.file_id` → source-PDF resolution, then OCR into an
   EA sidecar, then candidates with a distinct retrieval reason. ~175-project ceiling.
3. **Split-signal extraction** *(moderate)* — for FONSIs where the date and the authority title land
   in **different** candidate windows (e.g. "Recommended by /s/ X [date] … Approved by /s/ Field
   Manager"), widen the signature-block context window in `03` so one candidate captures
   date + authority + signature together. The `03` authority escape only helped 5 projects because
   of this; most recoveries came through the cascade instead.
4. **EA initiation recovery** *(separate effort)* — this is what moves the **complete-timeline**
   count (1,434), not decision coverage. NOI / scoping / application-received dates. Initiation
   coverage is only 56%; many recovered decisions lack a matching initiation.

Remaining ~770 still-missing EA breakdown: ~175 image-only (OCR), ~260 no decision doc in corpus
(source gap), rest genuinely weak/coarse or split-signal.

Master plan with the full audit + roadmap: `phase2/plans/ea_audit.md`.

---

## EIS — status & what's left

### Where EIS stands (53.4%, ~22 pt gap)

EIS **decision** coverage is **53.4% (2,207)**, complete ~20% (430 `complete_clear`). The tiered EIS
decision (ROD-first, FEIS-fallback) and the EIS labeling round improved selection, but EIS is still
**~22 pts under the Phase 1 baseline (75.2%)**.

**The gap is an extraction-recall problem, not selection or LLM.** The missing ROD/FEIS dates exist
in the documents but the **regex extraction (03)** never surfaced them (Phase 1 used a fine-tuned
BERT). Recovering the ~600 reconciled date-less EIS + the document-text RODs is the only path past
75.2%. **06 will NOT close this** — the LLM adjudicates *extracted* candidates; it can't find dates
that aren't in the pool.

### EIS — what's left (PARKED; the real coverage lever)

- **EIS extraction recall** is the headline EIS task and a larger effort — revisit after tire-kicking.
  Surface the date-less EIS ROD/FEIS dates that `03` misses (broaden retrieval / full-read RODs the
  way Phase C did for EA, or reintroduce a learned extractor).
- **EIS decision classifier soft spot:** EIS decision F1 was the weakest head (~0.70 on the older
  test); a few EIS `decision` labels are schedule / "Prepare ROD" / Gantt-milestone dates that should
  be `neither`. Worth a QC sweep if EIS decision underperforms after recall work.
- **Spot-check ROD vs FEIS-fallback picks** (`decision_is_feis_fallback`) — confirm FEIS dates are
  real publication dates, not draft/notice dates.

---

## Global next steps (in order)

### 1. Kick the tires on data quality (NEXT — before any API/06 spend)
Manually inspect extracted dates before trusting them:
- Sample `complete_clear` projects per process; verify init/decision dates against the source
  context (`decision_evidence_text`, `initiation_evidence_text` in project_dates).
- Sanity-check duration outliers (flagged `implausible_duration_*`; 6 durations >10,000 days).
- Spot-check EIS ROD vs FEIS-fallback picks; eyeball year-proxy CE decisions (don't over-trust them
  in headline numbers).

### 2. EIS extraction recall — the real coverage lever (parked; see EIS section)

### 3. Wire & run 06 (DEFERRED until after tire-kicking)
`06_adjudicate_llm.py` is stale (raw probs, 3 candidates). When ready: build per-project packet
(top-k init+decision, ROD/FEIS for EIS, has_rod, scores), call Haiku per routed project, write back
chosen candidate_ids. Decide the routing policy first (ambiguous-only ~$1 vs +coverage-recovery
~$37). Validate against `frozen_eval_ids` before a full run.

### 4. EA — optional all-EA full-read pass + initiation recovery (see EA deferred levers)

---

## How to run / validate (the workflow)

EA-scoped pipeline (all support `--process EA --sample-ids <ids> --run-dir <iso>`):

```
02_retrieve → 03_extract_candidates → 04_classify_candidates → 04b_calibrate --apply
→ 05b_rank --apply → 05_select_dates → (merge EA rows) → 05c_inject_ground_truth --scope all
→ 07_validate → 08_analyze.R
```

- **Always sanity-test on ~8–20 IDs first** (caught the split-signal issue this round).
- Validate by diffing isolated EA output vs production; review changed + added decisions for ≥95%
  precision; assert CE/EIS byte-identical.
- `05c --scope all` re-injects human-verified ranker.csv dates (run `05b --eval-output` *before* it
  for honest end-to-end metrics, or use `--scope train`).
- `08_analyze.R` regenerates all D4 tables + figures from `timeline_project_dates.parquet`.

## Key paths (post-reorg)

- **Labels (INPUTS):** `phase2/training/deliverable04/`
  - `classifier.csv` (candidate-level, frozen split) · `ranker.csv` (project-level, frozen split)
  - `eis_validation/` (verified ROD/FEIS yardstick) · `frozen_eval_ids.txt` · `_backups/` (gitignored)
- **Outputs (regenerable):** `phase2/output/deliverable04/`
  - `diagnostics/` (d4_*.csv) · `figures/` (fig_*.png) · `reports/` · `review_queues/` (gitignored)
- **Data/models:** `phase2/data/analysis/timeline/` — `timeline_project_dates.parquet`,
  `timeline_candidates.parquet`, `models/` (gitignored: classifier, ranker, calibrators, checkpoints),
  `_backups/ea_{b1,2a,2b}_*/` (EA merge-step backups)
- **Pipeline:** `phase2/code/deliverable04/` 00→08, 04b, 05b, 05c; one-off tools prefixed `_`;
  superseded code in `_archived/`.

## Do NOT

- Do not launch a full 02→04 rebuild by default (slow; not needed for tire-kicking; risks shifting
  stable CE/EIS numbers — see EA provenance caveat). Keep EA changes EA-scoped.
- Do not run 06 / spend API until data quality is checked and the routing policy is chosen.
- Do not fold a yardstick label into training (guardrail enforces this; respect it).
- Do not publish CE/EA/EIS numbers before tire-kicking the underlying dates.

## Known caveats / gotchas

- Classifier frozen-test is positive-heavy (~55%) vs the real pool (~10%); deployment precision is
  lower than test. Read **deployment** precision from the operating curve (full pool), not the test set.
- Ranker EIS held-out is small (frozen-eval n≈7 ROD) — ranker-vs-classifier is statistically
  inconclusive; the classifier `p_dec_cal`/`p_feis_cal` shortlist (top-5 90–95%) is good enough for 06.
- `route_to_llm` bundles ambiguous picks AND missing-with-candidates (coverage recovery) — different
  value; split them when choosing the 06 policy.
- **Scripts hard-require `CONDA_DEFAULT_ENV=nepa`.** Env python: `/opt/anaconda3/envs/nepa/bin/python`.
  `conda run` hit a permission error in an earlier session; call the python path directly with the env
  var set if `conda run` fails.
- **Digit-prefixed files** (`04_*`, `05_*`) can't be `import`ed — `04b`/`05b`/`05c`/`_diagnostics`
  load them via `importlib.util`. Follow that pattern for any new sibling script.

---

## Appendix — earlier session history (classifier rebuild, 2026-06-03 → 06-05)

*Condensed from the original `where_I_left_off_old.md`. Historical narrative of how the calibrated
classifier and selection rewrite came to be. The coverage numbers in this appendix are stale — see
the current state above — but the design decisions and rationale still hold.*

### The arc of that session
1. **Active-learning round 2** on SetFit → init F1 0.556→0.649, decision 0.647→0.737 (on the old
   154-row test). Declared the AL loop done (diminishing returns).
2. **Built calibration** (`04b_calibrate.py`): Platt calibrators + an operating curve (per-candidate
   AND per-project, classifier-`neither` candidates excluded).
3. **Headline finding:** `05_select_dates.py` originally **ignored the classifier entirely** — it
   ranked candidates on hand-weighted regex heuristics; `p_initiation`/`p_decision` were never read.
4. **Fixed `05`:** wired the classifier into `candidate_score_components()` (replacing the
   `classifier_signal = 0.0` stub) + role-aware page position, granularity, cross-candidate agreement,
   duration-plausibility flags, and 3 disambiguation rules (earliest-init, day>month decision, CE-only
   month-decision). Also fixed `06` to drop classifier-`neither` candidates before building packets.
5. **Corpus build-out** grew labels to ~1k/head, then an EA/EIS balance pass; final labeled set 4,471 rows.
6. **Re-froze the test set** to `test_v2` (one-time): 894 rows (247 init / 268 dec / 379 neither).
7. **Retrained** → **init F1 0.896, decision 0.892** on `test_v2`. Per-process: CE 0.96/0.95, EA
   0.82/0.86, **EIS decision weak (0.699)**.
8. **Wrote next-phase tooling:** `05b_rank.py` (LightGBM ranker) and the project-gold labeling spec.

### Key decisions locked (still in force)
- **Classifier drives selection** (was ignored). `candidate_score_components()` returns a named
  feature dict so the LightGBM ranker reuses the exact same features.
- **One-time test re-freeze** to `test_v2` is legitimate (labels are model-independent and drawn
  before the retrain → no leakage). Never re-draw again; new labels default to train.
- **SetFit now, DeBERTa later** — stay on SetFit for the cheap label-loop; graduate to DeBERTa only
  once labels + plateau justify it, decided on **end-to-end** date accuracy (`07_validate`), not
  candidate F1.
- **LightGBM (not XGBoost/ensemble)** for the ranker — native lambdarank, native categoricals,
  monotonic-constraint + SHAP interpretability.

### Companion labeling specs from that era
`training_steps.md`, `build_out_training.md`, `project_gold_labeling.md` (project-level labeling pass
that feeds the ranker).
