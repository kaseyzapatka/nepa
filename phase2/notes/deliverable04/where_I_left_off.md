# D4 Timeline — Where I Left Off (combined handoff)

> **CURRENT as of 2026-06-17 — read the "06 LLM-adjudication prep" section immediately below first.**
> Older sections (2026-06-16 coverage cycle, 2026-06-10 banner) are retained as history; where
> numbers conflict, the newest section wins.

---

## 2026-06-17 — 06 LLM-adjudication prep + audit (READ FIRST)

**State:** selection (05) is final and on `desktop`. The next action is the **06 LLM adjudication
run** of the ~11,207 "send-set". Everything below is ready; the only gate is the API key (now in the
macOS Keychain, prompt-on-access).

### What we did this session
- **Recovered + hardened 05:** restored the candidates file (clobbered by a `--process EIS` subset
  write; Time Machine) and added **Guard 2** so a subset run can never overwrite the canonical file;
  fixed the O(n²) `build_review_queue`; **Variant B** CE register-init fix (authoritative metadata
  inits admitted/preferred over the learned-score gate — 13/13 prior regressions recovered);
  **month-decision sliver** routing (cued month ROD/FEIS dates → 06); `--workers` parallelized 05
  (proved byte-identical to serial). Re-ran 05 + 05c. → commit **bd1feec**.
- **06 hardened for the run** (→ commit **a875133**): credit-safety (incremental checkpoint every 50,
  fail-fast on billing errors, errored rows excluded from the resume cache → top-up + re-run resumes;
  retry+backoff on typed anthropic 429/overloaded/5xx); **`--workers N` concurrency** (ThreadPoolExecutor,
  verified ~20–25 min vs ~5 hr serial); **completable scope gate** = queue is the **11,207** missing-
  but-completable rows (`INCLUDE_RECHECKS` removed); **month picks stored as the 15th**; day-vs-month
  rule in the prompt; accurate per-model pricing; **`--sample-ids` / `--no-apply`**; **`timeline_llm_run_at`**
  per-row audit stamp on adjudicated rows.
- **Audit of resolved (non-LLM) dates** (→ commit **4fb3eeb**, `_audit_resolved.py`): non-register
  picks are mostly sound. **KEY finding:** all **1,196 EA register decisions are FONSIs** → those EAs
  did NOT escalate to EIS; the short EA median (105 d) is a register-anchor artifact (register start
  ≈ 60 d before the FONSI vs 364 d for document inits). Findings written to `findings_for_report.md`.
- **08 completion figure** → full 0–100% box + dot at the share (→ commit **2212fe1**). Models synced
  to desktop (old archived under `_archived/`); data parquets copied night→desktop.

### Numbers (pre-LLM floor; post Variant-B + sliver)
| | complete now (both) | send to 06 (completable) | structural (unrecoverable) |
|---|---|---|---|
| CE | 45.8% (24,741) | 8,625 | 20,674 |
| EA | 50.8% (1,534) | 901 | 582 |
| EIS | 24.5% (1,011) | 1,681 | 1,438 |
| **all** | **44.6% (27,286)** | **11,207** | 22,694 |

Honest post-LLM estimate: **CE ~49% · EA ~54% · EIS ~35%**. Decarb (clean) EIS = 753 (18.2% of EIS).
EIS *decisions* are ~85% month-granularity (FEIS-publication, stored as the 15th).

### NEXT (in order)
1. **100-sample A/B test** (Haiku vs Sonnet), then my reference adjudication → agreement + measured
   cost. Command below.
2. **Full 06 run** on the 11,207 send-set with `--workers 12` (likely Sonnet ≈ $33; $45 credit covers
   it; fail-fast/resume safe). Monitor live.
3. **Re-run 08** (figures → final numbers) + update the numbers in `findings_for_report.md`.
4. **Push** — commits bd1feec / 4fb3eeb / 2212fe1 / a875133 are LOCAL on `desktop`.
5. Deferred: 02/03 parallelization (todo #26); A2 FEIS cover full re-pull (todo #25).

### How to run 06 (API key is Keychain-gated, prompt-on-access)
Key stored via `security add-generic-password -a "$USER" -s nepa-anthropic -T "" -U -w`. Every run
fetches it transiently (fires a macOS Allow dialog = the authorization; key never printed/persisted):
```
ANTHROPIC_API_KEY=$(security find-generic-password -a "$USER" -s nepa-anthropic -w) \
PYTHONPATH=/Users/Dora/git/consulting/nepa conda run -n nepa \
  python code/deliverable04/_test_adjudication.py --models claude-haiku-4-5-20251001 claude-sonnet-4-6 --workers 12
```
Full run: `python code/deliverable04/06_adjudicate_llm.py --process CE EA EIS --workers 12 --model claude-sonnet-4-6`
(no `--no-apply` → writes the dates). All 06 runs need `PYTHONPATH=<repo root>`.

### Key files (this session)
`code/deliverable04/{06_adjudicate_llm.py, _test_adjudication.py, _audit_resolved.py}`,
`output/deliverable04/test_sample_100.csv`, `notes/deliverable04/{findings_for_report.md, coverage_constraints.md}`.

---

## 2026-06-16 — Coverage-recovery cycle (READ FIRST)

**Goal of this cycle:** raise the *full-timeline overlap* (initiation **and** decision present) toward
Phase-1 clean-energy rates (CE ~30%, EA ~62%, EIS ~48%), broken out by energy type. Driven by a
Phase-1-vs-Phase-2 candidate comparison. Root causes are documented in
[`coverage_constraints.md`](coverage_constraints.md); the working plan is
[`full_recover.md`](full_recover.md); the EIS deep-dive reference is [`recover_eis.md`](recover_eis.md).

### Current coverage (after this cycle's SELECTION-only fixes applied to production)

| | All projects (complete) | Decarb/clean (complete) | Phase 1 clean target |
|---|---:|---:|---:|
| CE  | 43.6% | **38.5%** w/ proxy (23.5% clear-only) | 30.4% |
| EA  | 48.8% | 35.5% | 62.0% |
| EIS | 23.3% | 33.6% | 48.1% |

> **CE now exceeds Phase 1** (the with-proxy 38.5% is apples-to-apples: Phase 1's CE init was *also*
> an inferred date). EA/EIS still below — see pending work. The numbers above reflect a **selection-only
> re-run** (`05b→05→05c`) on the EXISTING candidate pool. The **retrieval + new-cue fixes are committed
> but NOT yet reflected in production** — they need the full `02→08` re-run (see Pending).

### Fixes committed this session (on `desktop`)

| Commit | Fix | What it does | In production yet? |
|---|---|---|---|
| `acdd7ba` (prev) | CE truncation + EA tier_d=8000 | retrieval cap fixes | needs full run |
| `6a33d19` | EIS retrieval (12k cap, dedup, text-fallback), `03` EIS windowed exclusions, EA+EIS calibrated/additive init eligibility, `_phase0_baseline.py` | recover EIS candidates + role-agnostic-via-prob init selection | **selection part: yes**; retrieval part: needs full run |
| `d732f96` | **CE inferred-init proxy** (earliest cand date < decision, 5y cap; flagged `ce_inferred_application`+`is_proxy`), `run_pipeline.py` orchestrator (**`_run.py` retired**), `08` coverage-by-energy figure | close CE init gap (mirrors Phase-1 inferred-application) | yes |
| `93fdff9` | **CE "applied for" cue** (Fix B) + vectorized the CE proxy | precise CE application-date inits | needs full run (it's a `03` cue) |
| `62c5430` | **EA/EIS scoping/NOI init cue** (`SCOPING_NOI_INIT`) | re-role scoping/NOI dates the classifier scored high but regex left `unknown` | needs full run (`03` cue) |
| `a2dd7e3` | CE proxy **permit/compliance negative filter** | proxy skips "permit issued"/CFR/"must comply" dates | yes |

### How fixes map to the gaps (the mental model)
- **CE init** (was the gap): solved by the inferred-init proxy + "applied for" cue → CE now exceeds Phase 1.
- **EIS decision** (the gap): ~117 truncated ROD/FEIS dates → the **retrieval fix** recovers them (needs the full run). ~88 are source-gap (no decision doc; not recoverable cleanly).
- **EA/EIS initiation**: candidates often *exist* but were **role-gated out of selection** → the **calibrated/additive init eligibility** (union of legacy ranker gate + `p_init_cal ≥ 0.5` / authoritative, role-agnostic-ish) + the **scoping/NOI cue** recover them. EA is additionally **source-limited** (no register/NOI init for ~half; start often == decision).
- **CE "stale ranking_score" 1,211-loss** root cause = run order: `05b` must run before `05`. Now baked into `run_pipeline.py`. **Never run `05` alone.**

### Tooling added this session
- **`run_pipeline.py`** — the one orchestrator. `python run_pipeline.py` (full `02→08`) or `--select`
  (`05b→05→05c→08`, minutes). Bakes in `04b`/`05b`/`05c` (skipping `05b` is what corrupted CE).
- `_phase0_baseline.py` — baseline metrics + corrected source-ceiling (EIS local ceiling = **2,664 / 64.5%**; 70%+ needs OCR/external = out of scope).
- `09_sample_check.R` — pulls ~20 projects/type, lists every candidate + selected dates for eyeballing
  (outputs `output/deliverable04/sample_check_{candidates,projects}.csv`). *(untracked)*
- `README.md`, `clean_up_plan.md`, `coverage_constraints.md`, `full_recover.md`. *(some untracked)*

### Production data state + backups
Production `timeline_project_dates.parquet` = CE (preserved via partition merge) + EA/EIS (selection-fixed).
Backups in `data/analysis/timeline/`: `timeline_project_dates.{pre_selfix,preselect,pre_cemerge,pre_gt_inject}_*.parquet`.

### PENDING / next steps (not yet done)
1. **The full `02→08` overnight re-run** — the big unbanked win. Applies EIS retrieval (~117 decision
   recoveries) + the scoping/NOI + applied-for cues across the whole corpus. Plan: isolated **git
   worktree off `desktop`** with symlinked input data + COPIED `models/` + production backup, run
   `run_pipeline.py`, validate (diff vs backup, CE must not regress, frozen-test if retrained), then
   merge code + copy data back. Full setup/commands in [`full_recover.md`](full_recover.md) §3–§6.
2. **Tier 2 classifier retrain** (the bigger EA/EIS init lever beyond what's banked): label ~200–300
   hard EA/EIS init cases → retrain `04`/`04b`/`05b`, **gated on frozen-test F1 holding** (no project
   gold exists, so frozen-test F1 is the only anti-inflation guard). Best on the worktree with a
   worktree-local `models/`. Shared classifier → retrain shifts all 3 processes. See `full_recover.md` §5.
3. **Remaining regex lever (identified, NOT implemented):** extend the application/"applied for" cue to
   EA+EIS and add FERC **`pre-filing`/`pre-application`** for EIS (raw candidates ~EIS 396 applied-for +
   162 pre-filing; EA 176). Same Fix-B treatment (anchor + sample-test). Last productive cue; regex is
   otherwise exhausted (EIS ROD-narrative and CE submitted/filed were tested and rejected as noisy/low).
4. **`06_adjudicate_llm.py` rebuild** (stale: top-3, raw probs) — separate effort before any LLM pass.

### Critical gotchas
- **Run order is non-negotiable:** `02→03→04→04b --apply→05b --apply→05→05c→07→08`. Use `run_pipeline.py`.
  Running `05` without `05b` drops candidates with NULL `ranking_score` (this caused the CE 1,211-loss).
- **Proxy/inferred dates are flagged** (`ce_inferred_application`, `is_proxy`, `decision_is_feis_fallback`).
  Report **with-proxy AND clear-only** — with-proxy is Phase-1-comparable; clear-only is the strict floor.
- **`05` reconciles all 4,130 EIS into the output as stubs** — restrict to actual sample ids when measuring.
- **No project-level gold** (`07`'s gold sample is empty) → validate by diff-vs-backup + sampling, not accuracy.
- Scripts hard-require `CONDA_DEFAULT_ENV=nepa`; env python `/opt/anaconda3/envs/nepa/bin/python`.

---

> **Last updated: 2026-06-10 (post-investigation).** This is the single authoritative warm-start note for D4.
> It consolidates three earlier handoffs written at different points:
> - the **2026-06-09** full-pipeline handoff (cross-process coverage, classifier, guardrails, 06 gate);
> - the **2026-06-10** EA decision-coverage recovery note (EA 67% → 74.2%);
> - the **2026-06-10** missing-reviews investigation (1,376 clean reviews absent from Phase 2, root cause confirmed, code fixes applied — see §Missing-reviews investigation); and
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

---

## Missing-reviews investigation (2026-06-10)

**Finding:** 1,376 clean-energy projects that Phase 1 had are absent from Phase 2 (1,360 CE + 16 EA).
All are in `timeline_document_index.parquet` but have **zero rows** in `timeline_candidates.parquet`
— the failure is at candidate extraction (`02_retrieve.py` → `03_extract_candidates.py`), not adjudication.

**Root cause (CE):** DOE/BLM CE-determination forms are `priority_3` (bare `document_type_clean == "CE"`
scores 0 in `01_index.py`). `build_tier_d_packets` stored only 2,000 chars per page; the NEPA signature
date sits at the bottom of dense ~7k-char forms, truncated off. A secondary loss: `_should_reject_date`
scanned the whole block for exclusion keywords, killing real signature dates that shared a block with
`"expiration date"` / statute citations.

**Root cause (EA):** Mixed, and **only partly a real regression.** Of the 16 EA: ~4–6 are
truncation-recoverable (same mechanism on priority_3 EA narrative docs); **6 are image-only scanned
FONSIs/RODs** (empty `page_text` → unrecoverable without OCR); 3 are genuinely date-less; and several
Phase-1 EA "dates" were demonstrably **wrong** (a railroad timetable, a revision-date stamp, a
4(d)-rule citation) — so Phase 2's absence is partly *more correct*, not worse. EA Fix 1 (tier_d
8000) targets only the ~4–6 recoverable; keep-vs-revert is gated on the EA regression diff. Full
per-project verdict + decision rule: [`../missing_investigation_EAplan.md`](../missing_investigation_EAplan.md).

**Code fixes — APPLIED in `acdd7ba` (2026-06-10), validation run NOT yet performed:**

| Fix | File | Change | Recovers |
|---|---|---|---|
| **CE Fix 1** ✅ | `02_retrieve.py` `build_tier_d_packets` | `TIER_D_CONTEXT_CHARS = {"CE": 30_000, "EA": 8_000, "EIS": 2_000}` replaces hard-coded 2,000 | ~1,029 CE |
| **CE Fix 2** ✅ | `03_extract_candidates.py` `_should_reject_date` | `date_span` param added; exclusion keyword check windowed to ±60 chars for CE only | ~251 CE |
| **EA Fix 1** ✅ | `02_retrieve.py` (same dict) | EA `tier_d` cap raised from 2,000 → 8,000 (matches `build_ea_decision_full_read_packets`) | ~4–6 EA |
| **CE Fix 3** ❌ | `01_index.py` `_compute_scores` | Floor `decision_doc_score` on `document_type_category == "decision"` → promotes defer docs | ~30–40 CE (optional) |

**Next:** run the isolated validation recipe from `missing_investigation_CEplan.md §6` on the missing
cohort IDs (`missing_ce_ids.txt`, `missing_ea_ids.txt`) before running a full-process rebuild.

Full evidence: [`missing_investigation_findings.md`](missing_investigation_findings.md).
Fix details: [`../missing_investigation_CEplan.md`](../missing_investigation_CEplan.md), [`../missing_investigation_EAplan.md`](../missing_investigation_EAplan.md).

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
