# D4 Full Recovery & Overnight Re-run Plan

**Date:** 2026-06-15 evening. **Goal:** make every safe, high-yield fix to candidate/timeline
coverage, then launch ONE clean full `02→08` re-run overnight on an isolated worktree, validate
in the morning, and merge to `desktop` — without re-running. `recover_eis.md` remains the EIS
deep-dive reference; this doc is the single actionable plan and folds in the remaining steps.

---

## MASTER REMAINING-WORK CHECKLIST (2026-06-16) — direct me here if we get lost

Locked design decisions: LLM pool ranked by the **learned ranker** (not classifier prob), **N=5 init /
10 decision**, **rule-inclusion** (register/ROD-sig/FEIS-pub/NOI always in pool), dedup-by-date keep-best,
300-char cap. Recall: init ~93% / decision ~86%@10. LLM cost measured ~$15.

**Phase A — build today (worktree, code only, preview-test each):**
- [ ] A1. Decision tiering: ROD A→B→C (incl. ROD signed/mentioned in FEIS) → Tier D = FEIS publication/cover
      date (`feis_publication`, month+yr ok) + `_DECISION_NEG_RE` + DEIS guard.  (partial 05 edits written)
- [ ] A2. Targeted FEIS cover re-pull (folded into A1): read FEIS cover pp.1–3 via the SAME 03 extraction,
      integrate via `04 --append → 04b → 05b`. (~404 projects today; full re-pull = weekend.)
- [ ] A3. LLM pool builder: learned-ranker rank, N=5 init/10 dec, rule-inclusion, dedup-by-date keep-best,
      surface granularity, 300-char cap.
- [ ] A4. route_to_llm / confidence-tier flag.
- [ ] A5. EIS init recovery: chronology fix + calibrated-prob admission (~187 strong inits + missed register inits).
- [ ] A6. ROD-pool tightening: precision-only; borderline falls through to Tier D (no coverage loss).
- [ ] A7. CE proxy guardrail: fire only with a genuine later decision-signature (earlier=init, signature=decision).
- [ ] A8. Prefer signature date over cover month for EA/CE decisions (Pedro Hill case).

**Phase B — run + validate (worktree):**
- [ ] B9. Run `05` with new logic (current project_dates still reflect OLD selection until this runs).
- [ ] B10. Re-audit: recall@N + precision spot-check.  [ ] B11. Pause for user review (after A1–A2 per request).

**Phase C — merge:** [ ] C12. commit night-run, merge → desktop (code/notes only).  [ ] C13. copy back gitignored
parquets + classifier.csv/ranker.csv by hand.

**Phase D — LLM run (desktop):** [ ] D14. rebuild `06_adjudicate_llm.py` (stale) for the new pool.
[ ] D15. pilot ~200 projects → confirm $/project → run ~10,097 routable (~$15–18, max_tokens≈150).
[ ] D16. merge LLM picks → finalize project_dates.

**Phase E — finalize:** [ ] E17. validate sample + confidence tiers.  [ ] E18. deliverable tables/figures (tiered).

**Phase F — weekend/deferred:** [ ] F19. full FEIS cover re-pull in `02_retrieve` (→ ~2,556 ceiling).
[ ] F20. classifier retrain (rebalance final_eis head, then 04→04b→05b w/ F1 gate; init F1 0.882→0.893 parked).
[ ] F21. held-out validation gold for `07` (project to-do #23).  [ ] F22. dedicated calibration split (to-do #24).

**Phase G — report:** [ ] G23. build `reports/deliverable04.qmd`, fold in `findings_for_report.md`.
[ ] G24. (standing) remind user to build validation gold.

**Loose flags:** `05 --process EIS`-only risks the CE/EA partition (old 1,211-CE loss) — run full 05 or guard it;
16 image-only EIS + ~10% EA-init cue = accept & disclose.

---

## 2026-06-16 PM — finalize selection on worktree, MERGE, then run LLM on DESKTOP (CURRENT PLAN)

Decision (user, 2026-06-16): do the selection/ranking tightening on the worktree, merge to desktop,
and run the LLM adjudication (06) on **desktop** — NOT on the worktree.

**Step 1 — Tighten DECISION selection (on worktree, code only).**
- Finish the FEIS-*publication*-date fallback for EIS-no-ROD, explicitly labeled
  (`decision_is_feis_fallback` / `feis_publication`), + the negative-cue filter (`_DECISION_NEG_RE`)
  + DEIS guard. (In `05_select_dates.py`; edits started.) Goal: deterministic decisions are correct
  and chronology windows are right so init isn't wrongly dropped.

**Step 2 — Build the 06 candidate pool / ranking (on worktree, code only).**
- For each routable project, assemble **top-5 initiation candidates by `p_init_cal` + top-5 decision
  candidates by `p_dec_cal` (plus the FEIS-publication date), across ALL regex roles**, deduped — the
  small, efficient, best-candidate set the LLM will read. Ranked by calibrated probability, NOT
  gated by regex role. This is the fix for "are the best candidates even being sent?".

**Step 3 — Merge worktree → desktop (code only; copy data/labels by hand).**
```bash
cd ../nepa-night && git add -A && git commit -m "[D4] decision-selection tightening + 06 candidate-pool builder + audit/findings"
cd /Users/Dora/git/consulting/nepa && git merge night-run        # brings tracked CODE/notes
# gitignored artifacts must be copied by hand (NOT carried by git):
cp ../nepa-night/phase2/data/analysis/timeline/timeline_{candidates,project_dates}.parquet phase2/data/analysis/timeline/
cp ../nepa-night/phase2/training/deliverable04/{classifier,ranker}.csv  phase2/training/deliverable04/
# git worktree remove ../nepa-night   # after verifying
```

**Step 4 — Run the LLM adjudication (06) on DESKTOP, not the worktree.**
- Rebuild/run `06_adjudicate_llm.py` on the merged desktop data, feeding the top-5+5 pool from Step 2.
- Routable set ≈ **10,097 projects** (CE 7,922, EA 897, EIS 1,278) = not-complete but has both
  candidate types. Cost (measured): ~$13 capped / ~$26 full context. Cap context to ~300 chars/candidate,
  `max_tokens`≈150, and **pilot ~200 projects first** to confirm cost-per-project before the full run.

**Note:** Steps 1–2 are code-only and safe to do on the worktree; the LLM run (Step 4) happens after
the merge, on desktop, so the worktree is purely for the selection/pool code.

---

## 2026-06-16 NIGHT 2 — AUTONOMOUS decoupled retrain + run (earlier; superseded by the PM plan above)

This supersedes the §2–§8 ordering for the night-2 run. The earlier sections remain valid
background (root causes, feasibility). **This is the runbook to resume from if context is lost.**

**Two locked decisions (user, 2026-06-16):**
1. **Decoupled + auto-fallback.** Attempt the classifier retrain, but the full coverage run does
   NOT depend on it. The run uses the retrained model ONLY if it passes the frozen-test F1 gate;
   otherwise it auto-falls back to the current committed model. A concrete coverage result is
   guaranteed by morning regardless of how the retrain goes.
2. **Stage only — NO merge, NO push.** Everything stays in the worktree. Morning deliverable =
   a written report + a coverage diff vs the prenight backup. The user merges after reviewing.

**Supervision:** none required. Bad labels cannot poison the morning numbers — the F1 gate
discards a regressing retrain and falls back to the current model. Claude QCs a label sample; the
gate is the structural backstop.

### Worktree + model isolation (CRITICAL: copy models, don't symlink)
```bash
MAIN=/Users/Dora/git/consulting/nepa
git worktree add ../nepa-night -b night-run desktop
cd ../nepa-night/phase2
mkdir -p data/processed data/analysis/timeline
# READ-ONLY inputs: symlink (no GB copies)
ln -s $MAIN/phase2/data/processed/ce  data/processed/ce
ln -s $MAIN/phase2/data/processed/ea  data/processed/ea
ln -s $MAIN/phase2/data/processed/eis data/processed/eis
cp   $MAIN/phase2/data/analysis/timeline/timeline_document_index.parquet data/analysis/timeline/
ln -s $MAIN/phase2/data/analysis/projects_combined.parquet data/analysis/projects_combined.parquet
# MODELS: COPY (retrain overwrites; a symlink would clobber desktop's model).
cp -R $MAIN/phase2/data/analysis/timeline/models data/analysis/timeline/models          # working dir (scripts' default MODEL_DIR)
cp -R $MAIN/phase2/data/analysis/timeline/models data/analysis/timeline/models_current   # FALLBACK (never touched by retrain)
```
The worktree writes its OWN candidates/project_dates parquets; main checkout is untouched.

### Prenight backup (in MAIN, before launch)
```bash
cd $MAIN/phase2/data/analysis/timeline; TS=$(date +%Y%m%dT%H%M%S)
cp timeline_project_dates.parquet timeline_project_dates.prenight_$TS.parquet
cp timeline_candidates.parquet    timeline_candidates.prenight_$TS.parquet
```

### Labeling (in-session, BEFORE launch; appended to worktree classifier.csv/ranker.csv, split BLANK→train)
Do #1,2,3,5 + targeted init. Never set split=test; never label a `frozen_eval_ids.txt` project as
ranker-train (05b --train hard-fails on that). Target volumes (quality > quantity; gate protects):
- **#3 active-learning** — `04 --emit-batch 150` (most-uncertain unlabeled; built-in).
- **#2 neither hard-negatives** — ~80 (cover months, citations, permit-expiration, consultation, historical RODs).
- **#5 final_eis** — ~40 EIS FEIS candidates.
- **targeted init** — ~120 low-`p_init_cal` EA/EIS init candidates (the bimodal-high cohort).
- **#1 ranker** — ~60 projects (true init + decision candidate_id; EA/EIS heavy), excluding frozen_eval ids.
Agents label candidate-level from the anchored `model_context` per `labeling_rules.md` (cover-month
asymmetry, activity-vs-milestone, only the `[[ ]]` date), MUST cite the evidence span. Claude QCs a
sample, writes labels back by candidate_id, commits in the worktree.

### Overnight driver (one backgrounded script: `code/deliverable04/_night_driver.py`)
Each stage wrapped in a per-stage `timeout` + logging; any failure logs and falls back; the RUN is
the priority. Sequence:
1. **Baseline F1**: `04 --eval --model-dir .../models_current/candidate_classifier` → record per-head F1.
2. **Retrain** (writes into working `models/`): `04 --train` → `04b --fit` → `05b --train`.
3. **GATE**: `04 --eval` (default = new `models/`) → per-head F1. PASS if init & decision F1 each
   ≥ baseline − 0.01. **PASS** → keep `models/` (new). **FAIL or any retrain step errored** →
   `rm -rf models && cp -R models_current models` (restore current).
4. **Full run**: `run_pipeline.py` (02→03→04→04b --apply→05b --apply→05→05c→07→08) using `models/`.
   NOTE: 05c `--scope all` still injects ranker.csv — fine for the run; validation gold is deferred.
5. **Validate + report**: read 08 coverage; diff worktree `timeline_project_dates` vs the prenight
   backup by process × coverage state; write `notes/deliverable04/morning_report_<date>.md` with
   coverage table, per-process deltas, which model won the gate (+ baseline vs new F1), proxy counts,
   and a 15–20-row sample of newly-covered dates per process.

### Wake-up reactions (Claude, on driver exit / scheduled heartbeat)
- Driver exited 0 → read report, confirm no process regressed unexpectedly, STAGE (no merge).
- Driver exited non-zero → read log, identify failed stage. Retrain failure must have already
  fallen back to current model; if the RUN failed, fix and relaunch from the failed stage
  (`run_pipeline.py` or `--select`). Never let a failure leave zero output — worst acceptable case
  is a current-model coverage run.
- Hang (no exit) → per-stage timeouts convert hangs to failures; if the whole driver hangs, the
  scheduled heartbeat re-checks and relaunches.

### Morning deliverable (STAGE ONLY)
`morning_report_<date>.md` + the worktree parquets left in place + the prenight backup for diffing.
Do NOT merge night-run into desktop or push. Present the report; user merges after review.

---

## 0. Honest feasibility (read first)

Target: tonight's run should capture **≥95% of Phase 1's final timelines** so a clean-vs-fossil
breakdown can resemble Phase 1. **Full 95% is not reachable tonight**, but the fixes below close
most of the gap. The hard limits (from the root-cause analysis):

| Gap | Size | Fixable tonight? |
|---|---|---|
| CE init candidates (Phase 2 has 45% of Phase 1's) | 4,577 CE | **Yes — Fix 1** (inferred-init proxy; mirrors Phase 1) |
| EIS decision candidates (64%) | 205 EIS: 117 have a ROD/FEIS doc, 88 don't | **Partly — Fix 2** recovers ~117; 88 are source-limited |
| EA both-candidates (82%) — candidates exist, not selected | ~80 EA | **Via Fix 3** (classifier retrain); not extraction |

Realistic post-run capture of Phase 1 finals: **CE ~85–90%, EA ~85%, EIS ~80%** — a major jump
from today (CE 41% / EA 82% / EIS 62% both-candidate), good enough for a defensible clean-vs-fossil
story, but be ready to say "the residual is source-limited (image-only / no decision doc)."

---

## 1. Root causes (established empirically 2026-06-15)

1. **CE stale `ranking_score`** — 4,384 CE candidates have NULL `ranking_score` (the truncation-fix
   recoveries) because `05b_rank --apply` was never re-run on them. Any `05` run without `05b`
   first drops their decisions (this caused the 1,211-CE loss). **Cause is run-order, not logic.**
2. **CE init gap = Phase 1's inferred application date.** 100% of the 4,577 CE init-gap projects got
   their Phase 1 init from `bert_inferred_application_date` = `application_date` if found, else the
   **earliest review date** (earliest dated mention). Phase 2 has no equivalent inference.
3. **EIS decision gap = truncation + source.** 117/205 have a ROD/FEIS doc (truncated away →
   recoverable by the 12k cap / full-read, already coded); 88/205 have no decision doc (source gap).
4. **EA = selection, not extraction.** Candidates exist for 82% of Phase 1's EA completes; they score
   low (bimodal `p_init_cal`). Lever is the classifier retrain, not more retrieval.

---

## 2. Fixes to make tonight (ordered by yield)

### Fix 0 — Run-order streamlining (DONE)  [addresses "don't make me remember"]

The pipeline MUST run `02 → 03 → 04 → 04b --apply → 05b --apply → 05 → 05c → 07 → 08`. Skipping
`04b`/`05b` is what corrupted CE. **`run_pipeline.py` is now the one orchestrator** —
`./run_pipeline.py` (full) or `./run_pipeline.py --select` (selection-only). `04b`/`05b`/`05c` are
baked in. `_run.py` (sharded runner, omitted those stages) is **retired to `_archived/`**.

### Fix 1 — CE inferred-init proxy (the biggest Phase-1-match lever, ~+4,000 CE init)

**Where:** `05_select_dates.py`, CE init selection, as a flagged last-resort fallback.
**What (mirror Phase 1):** when a CE project has a decision date but **no** initiation candidate
selected, set `initiation_date` = the **earliest candidate date strictly before the decision**
(prefer a candidate with an application/received cue; else the earliest dated mention). Flag it
`initiation_is_proxy = True` and add `ce_inferred_application` to `timeline_flags`. Only fires when
an earlier date exists (no zero-duration; matches Phase 1's `None` cases).
**Why it's defensible:** this IS Phase 1's method, and CE is a *build* issue where being less
conservative is acceptable (user call, 2026-06-15). Flagged so it can be included/excluded from
headline "clear" timelines. Closes most of the 4,577-CE init gap.
**Risk:** it's an inferred date. Keep it `is_proxy` + `ce_inferred_application`, reported separately;
never call it a clear date. **TODO (tomorrow): audit a sample of these inferred CE inits** — confirm
the earliest-date proxy is reasonable, not a stray figure/citation date.

### Fix 2 — EIS retrieval recall (ALREADY CODED; the full run applies it)

12k context cap (Tier B + D), longer-context dedup, `build_eis_text_fallback_packets`, windowed
exclusions. Recovers ~117 of the 205 missing EIS decision candidates on the full `02→03` run.
Nothing to add — it just needs the full re-run. The 88 source-gap projects stay uncovered (honest).

### Fix 3 — Tier 2: retrain classifier + ranker (ONLY if time; see §5 for split integrity)

Recovers the EA selection gap and lifts EIS/CE init scoring. Highest effort, highest risk. **Must run
BEFORE the overnight `02→08`** (the run's `04` uses the model). Hard gates in §5. If the frozen-test
F1 drops or time runs short, **skip it and run with the current model** — Fixes 0–2 stand alone.

---

## 3. Worktree + data isolation + backups  [addresses "branch + run + merge without re-running"]

The run depends on untracked data (huge `pages.parquet`, models). Strategy: isolate OUTPUTS in a
worktree, SYMLINK the big read-only INPUTS, back up production, copy outputs back in the morning.

**Setup (this evening):**
```bash
# 0. commit all code/notes changes to desktop first (so the worktree branches from them)
# 1. create the worktree off desktop
git worktree add ../nepa-night -b night-run desktop
cd ../nepa-night/phase2

# 2. symlink the large READ-ONLY inputs (no copying GBs); MAIN = absolute path to the main checkout
MAIN=/Users/Dora/git/consulting/nepa
mkdir -p data/analysis/timeline data/processed
ln -s $MAIN/phase2/data/processed/ce  data/processed/ce
ln -s $MAIN/phase2/data/processed/ea  data/processed/ea
ln -s $MAIN/phase2/data/processed/eis data/processed/eis
ln -s $MAIN/phase2/data/analysis/timeline/models data/analysis/timeline/models
cp   $MAIN/phase2/data/analysis/timeline/timeline_document_index.parquet data/analysis/timeline/   # small input from 01
ln -s $MAIN/phase1 ../phase1                     # for 08's Phase-1 comparison (if not already present)
ln -s $MAIN/phase2/data/analysis/projects_combined.parquet data/analysis/projects_combined.parquet  # energy join
# training labels are git-tracked → already present in the worktree
```
The worktree writes its OWN `data/analysis/timeline/timeline_{context_packets,candidates,project_dates}.parquet`.
**Main checkout production data is untouched during the run.**

**Backup (precaution, before morning copy):**
```bash
cd $MAIN/phase2/data/analysis/timeline
TS=$(date +%Y%m%dT%H%M%S)
cp timeline_project_dates.parquet timeline_project_dates.prenight_$TS.parquet
cp timeline_candidates.parquet    timeline_candidates.prenight_$TS.parquet
```

**Morning (after validating worktree outputs):**
```bash
# A. copy regenerated outputs from worktree -> main canonical location
cp ../nepa-night/phase2/data/analysis/timeline/timeline_{context_packets,candidates,project_dates}.parquet \
   $MAIN/phase2/data/analysis/timeline/
# B. merge the code to desktop
cd ../nepa-night && git add -A && git commit -m "[D4] full overnight re-run fixes" && \
  cd $MAIN && git merge night-run
# C. (optional) git worktree remove ../nepa-night
```
> If retraining (Fix 3), also copy back the new model artifacts from `../nepa-night/.../models/`
> (or, if models were symlinked, point the retrain output at a worktree-local models dir so the
> main models aren't overwritten mid-evening — see §5).

---

## 4. Run sequence (one command, in the worktree)

```bash
cd ../nepa-night/phase2/code/deliverable04
CONDA_DEFAULT_ENV=nepa ./run_pipeline.py   2>&1 | tee ../../notes/deliverable04/nightrun_$(date +%Y%m%d).log
```
Runs, in order: `02 → 03 → 04 → 04b --apply → 05b --apply → 05 → 05c --scope all → 07 → 08`.

**Realistic runtime ≈ 3–5 h** (NOT the earlier over-conservative 6–12 h, which was anchored on the
*stitched* >24h build — that was iterative debugging, not one clean pass). Grounded estimate:
- `02` retrieve: EIS pages are 5.5 GB (long pole, ~30–60 min); CE (350 MB) + EA (433 MB) are minutes.
- `04` classify: SetFit over ~420k candidates (CE 274k / EIS 112k / EA 36k) — the dominant cost, ~1.5–3 h.
- `03` ~30–60 min; `05` ~10–25 min (measured tonight); `04b`/`05b`/`05c`/`07`/`08` ~20 min.

**Speed option:** only EIS changed at the *retrieval* level — CE/EA changes are selection-only
(stale-rank fix + CE proxy + EA calibrated gate), which need just `05b --apply → 05` (minutes), NOT
`02→04`. So a *targeted* run (EIS full `02→05` + CE/EA `05b→05`) is **~2–3 h**. The tradeoff: the
full `02→08` gives clean single-provenance across all three (what you asked for); the targeted run is
faster but keeps CE/EA candidates from the existing pool. **If Tier 2 retrains the shared classifier,
the full run is required** (all processes must re-classify) and lands ~4–6 h.

---

## 5. Tier 2 retrain — classifier + ranker WITHOUT inflating numbers

**The danger:** there is NO populated project-level gold (`07`'s gold is empty), so coverage going
up does NOT prove accuracy. The ONLY objective guard is the frozen candidate-level test set. Protect
it absolutely.

**Split integrity rules (from `labeling_rules.md`):**
- The 154-row **test split is FROZEN** (seed 42). **Never** add to it, never relabel it.
- **New labels append to `classifier.csv` with `split` BLANK** → they default to `train`. Do not set
  `split` on new rows; do not edit existing rows.
- Ranker: new project-level labels append to `ranker.csv`; respect `frozen_eval_ids.txt`
  (`05b` hard-fails on train/eval contamination — do not bypass).
- A label is **training XOR evaluation, never both.**

**Labeling with AI agents (you have several):**
- Target the **hard, currently-near-zero cases**: EA initiation (source-limited but some real),
  EIS initiation/decision. Use `phase0/cohort_ranker_blocked_init.txt` (331 high-conf) + sample
  more low-`p_init_cal` EA/EIS init candidates.
- Agents MUST follow `labeling_rules.md` exactly (e.g. CX cover month = `decision`, EA/EIS cover
  month = `neither`; activity-vs-milestone). Mislabels poison training.
- Aim ~150–300 new init labels + enough decision/neither to keep heads balanced.

**Retrain sequence (evening, BEFORE the overnight run):**
1. Append new labels (blank split) → `classifier.csv`.
2. `04 --train` → retrain SetFit (write the new model to a **worktree-local** models dir so you can
   roll back; ~1–2 h).
3. **GATE:** `04 --eval` on the frozen test → per-head F1 must **hold or improve** vs the current
   model. If init/decision F1 drops, **discard the retrain** and run with the current model.
4. `04b --train`/recalibrate; `05b --train` (respect frozen_eval_ids) → eval on its frozen split.
5. Only if both gates pass: proceed to the overnight `02→08`, which will `04`-score with the new model.

**Anti-inflation checklist for the morning summary:**
- Report **frozen-test F1 per head** next to coverage. Coverage ↑ with F1 flat/↑ = real; coverage ↑
  with F1 ↓ = overfit, do not present.
- Sample-review 15–20 newly-covered dates per process against their evidence text.
- Keep `is_proxy` / `ce_inferred_application` dates reported **separately** from clear timelines.

> **Shared-classifier caveat:** one model serves CE/EA/EIS. Retraining shifts ALL three. The morning
> validation (diff vs the prenight backup) must cover CE/EA/EIS, not just the head you targeted.

---

## 6. Morning validation + merge

1. In the worktree: `08` already ran → read the new coverage table.
2. **Diff** worktree `project_dates` vs the prenight backup, by process: confirm CE recovered
   (≥ prior decision count — the stale-rank fix should make CE ≥ before), EIS up, EA up, **no
   process regressed** without an explained reason (proxy additions are expected).
3. Frozen-test F1 check (if retrained). Sample-review.
4. If clean → copy outputs back + merge code (§3 morning steps). If not → production is untouched;
   investigate before copying.

---

## 7. Risks / what else you're missing

- **Critical path / time:** retrain (~2 h) sits BEFORE the run (the run's `04` loads the model), and a
  retrain forces the *full* ~4–6 h run (all processes re-classify). Start labeling early; if it slips,
  drop Fix 3 and run Fixes 0–2 (the faster ~2–3 h targeted run still gets CE proxy + EIS retrieval).
- **95% is not fully reachable:** CE inferred-init proxy + EIS retrieval get most of it; ~88 EIS and
  the EA source-limited residual won't close tonight. Say "source-limited," not "worse than Phase 1."
- **The CE proxy is a proxy.** Flag it; decide before the talk whether headline numbers include it.
  (Including it matches Phase 1; excluding it is the stricter Phase 2 number — present both.)
- **`06_adjudicate_llm.py` is still stale** (top-3, raw probs). It is NOT part of tonight. Rebuild it
  Tuesday before the adjudication pass (cleanup_plan #4) — that's the separate accuracy lever.
- **`05c --scope all`** re-injects verified dates and runs `05b`-dependent logic — it's in the chain;
  don't run `05` alone again (that's what dropped CE).
- **Coverage-by-energy figure (a deliverable, build into `08`):** add
  `fig_d4_coverage_by_process_and_energy` — coverage (both/decision/init/none) split by
  Decarb/Fossil/Other within each process — so the **Decarb (clean) coverage is directly visible and
  comparable to Phase 1**. `08` has duration/FRA/year-by-energy but NOT coverage-by-energy today.
- **The 88 source-gap EIS (no decision doc):** Phase 1 read ~36 from narrative ROD/FEIS-date
  *mentions* inside non-decision docs, and ~52 from looseness (filing dates, consistency
  determinations, citations). Phase 2 can't reach them tonight (no decision doc to read). The ~36 need
  *narrative decision-date extraction* — a **tomorrow** recall item, not tonight; the ~52 are
  correctly declined.
- **Do NOT retrain on the frozen test, do NOT branch the run off uncommitted code** (commit first,
  then worktree).

---

## 8. Tonight's checklist (in order)

- [ ] Commit all current code/notes changes to `desktop` (#1).
- [ ] Write `run_pipeline.py` (Fix 0) + fix `_run.py`.
- [ ] Implement CE inferred-init proxy (Fix 1) in `05`; sample-test on ~200 CE.
- [ ] Create worktree + symlink inputs + back up production (§3).
- [ ] (If time) label hard cases → retrain `04`/`04b`/`05b` → **frozen-test F1 gate** (§5).
- [ ] Launch `run_pipeline.py` overnight in the worktree (§4).
- [ ] Morning: validate (§6) → copy outputs back + merge to `desktop`.
