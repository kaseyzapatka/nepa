# D4 — Where I Left Off (session of 2026-06-03 → 06-05)

A narrative handoff of the work in this chat. Companion to `training_steps.md` (current state +
next steps), `build_out_training.md` and `project_gold_labeling.md` (labeling specs).

---

## The arc of this session

We started with the D4 two-head SetFit classifier at modest quality and ended with a strong,
calibrated classifier wired into selection — plus the tooling and specs to finish the pipeline.

1. **Active-learning round 2** on SetFit: emitted a `clear_initiation`-floored batch, Codex labeled
   it, retrained → init F1 0.556→0.649, decision 0.647→0.737 (on the old 154-row test). Declared the
   AL loop done (diminishing returns).
2. **Built calibration** (`04b_calibrate.py`): Platt calibrators + an operating curve (per-candidate
   AND per-project, with classifier-`neither` candidates excluded). First curve was disappointing
   (auto-resolve ~11%) — which exposed the real problems below.
3. **Headline finding:** `05_select_dates.py` **ignored the classifier entirely** — it ranked
   candidates on hand-weighted regex heuristics; `p_initiation`/`p_decision` were never read. The
   model we'd been improving was invisible to date selection.
4. **Fixed `05`:** wired the classifier into `candidate_score_components()` (replacing the
   `classifier_signal = 0.0` stub) + added role-aware page position, granularity, cross-candidate
   agreement, duration-plausibility flags, and 3 disambiguation rules (earliest-init, day>month
   decision, CE-only month-decision). Also fixed `06` to drop classifier-`neither` candidates before
   building LLM packets.
5. **Distributed diagnostics** into the workflow (`_diagnostics.py`): `04`/`04b` write numbered
   files `01–07` to `output/deliverable04/diagnostics/` at the right step (no standalone script).
6. **Corpus build-out** (overnight Codex, `build_out_training.md`): grew labels to ~1k/head, then a
   targeted EA/EIS pass for class balance, then dropped blanks. Final labeled set 4,471 rows.
7. **Re-froze the test set** (`_refreeze_test.py`) as `test_v2` — a deliberate one-time revamp: the
   old 18/18/118 was too small. New test = 894 rows (247 init / 268 dec / 379 neither), stratified.
8. **Retrained on the rebuilt corpus** → **init F1 0.896, decision 0.892** on `test_v2`. The target.
9. **Re-ran calibration** end to end (re-score pool → `--fit` → `--curve` → `--apply`). Calibrated
   probs now reach ~0.87; candidate precision ~0.88–0.91; `p_init_cal`/`p_dec_cal` written to all
   285,747 candidates.
10. **Wrote the next-phase tooling:** `05b_rank.py` (LightGBM ranker, complete) and
    `project_gold_labeling.md` (the project-level labeling pass that feeds it).

---

## Where things stand right now

- **Classifier:** `20260605T082856Z`, SetFit/MiniLM-L6, trained on 3,577 rows. `test_v2` F1
  init 0.896 / decision 0.892. Per-process: CE excellent (0.96/0.95), EA solid (0.82/0.86),
  **EIS decision weak (0.699)** — the one soft spot.
- **Pool:** re-scored; `p_init_cal`/`p_dec_cal` applied. `05`/`06` can consume calibrated confidence.
- **`05`:** classifier-integrated + new signals + 3 rules. Still uses the *heuristic* sum (now
  including `classifier_signal`); the *learned* ranker is the next upgrade.
- **Operating reality:** classifier is no longer the bottleneck. ~10–12% of projects auto-resolve
  both dates, ~65% at least one — the ceiling is **initiation date coverage**, not model quality.
- **Git:** 4 commits this session (`432fb4e`, `22708a9`, `66e2872`, + the AL-round-2 one), all
  **local/unpushed**. Push when ready.

---

## What's next (active = Phase 3, in `training_steps.md`)

1. **Project-gold labeling** — `project_gold_labeling.md`, ~300+ projects, true init/decision
   `candidate_id` each. Intensive judgment task; double-duty (ranker data + missing `07` gold).
2. **Train the LightGBM ranker** — `05b_rank.py` is written; `pip install lightgbm` first, then
   `--train`/`--eval`/`--apply`. Then wire `05` to prefer learned scores (4-line hook in its docstring).
3. **Retune `06`** against calibrated probs — point routing at `p_*_cal`, set `ROUTE_CONF_THRESHOLD`
   from the curve, raise `ROUTED_TOPK` 3→5. Steps in `training_steps.md` → "06 RETUNING STEPS".
4. **(Optional) EIS decision** targeted labeling if end-to-end EIS accuracy is poor.

---

## Things to know / gotchas

- **`test_v2` is positive-heavy (~55%)** vs the real pool (~10%). So its precision is optimistic —
  always read **deployment** precision from the operating curve (full pool), not the test set.
- **Old F1 numbers (0.649/0.737 etc.) are NOT comparable** to `test_v2` — different, larger test.
  The progression table restarts at `20260605T082856Z`.
- **Scripts hard-require `CONDA_DEFAULT_ENV=nepa`.** The env's python is `/opt/anaconda3/envs/nepa/bin/python`.
  `conda run` hit a permission error this session; call the python path directly with the env var set.
- **Digit-prefixed files** (`04_*`, `05_*`) can't be `import`ed — `04b`/`05b`/`_diagnostics` load them
  via `importlib.util`. Follow that pattern for any new sibling script.
- **LightGBM is not installed** in `nepa` yet.
- **`05b` needs project-gold to exist** (`project_gold_sample.csv`); it errors helpfully if missing.
- **Build-out QC residual:** a few EIS `decision` labels are schedule/"Prepare ROD"/Gantt-milestone
  dates that should be `neither` — small (~few %), worth a sweep if EIS decision underperforms.
- **`deliverable04.md` (architecture)** has an uncommitted change I did NOT make this session and left
  unstaged — worth a look. `review_sample20.R` has a stray pre-existing 1-char change.
- **Backups:** `labeling_sample.pre_refreeze.csv` (before test_v2), `labeling_sample.pre_al1.csv`.

---

## Key decisions locked this session

- **Classifier drives selection** (was ignored). `candidate_score_components()` returns a named feature
  dict specifically so the LightGBM ranker reuses the exact same features.
- **One-time test re-freeze** to `test_v2` is legitimate: labels are model-independent and drawn before
  the retrain, so no leakage. Never re-draw again; new labels default to train.
- **SetFit now, DeBERTa later** — stay on SetFit for the cheap label-loop; DeBERTa only once labels +
  plateau justify it, decided on **end-to-end** date accuracy (`07_validate`), not candidate F1.
- **LightGBM (not XGBoost/ensemble)** for the ranker — native lambdarank, native categoricals,
  monotonic-constraint + SHAP interpretability. Needs project-level labels (hence the gold pass).
