# D4 Night-2 morning report (2026-06-16)

**Run:** full `02→08` on the `nepa-night` worktree, **current model** (the retrain was attempted,
improved init F1 0.882→0.893, but was auto-discarded on two driver bugs — see below — so the run
used the current committed model, exactly as the decoupled design intended). Pipeline rc=0.
**Nothing merged or pushed — staged only.**

## Complete-timeline coverage (initiation AND decision), ALL energy types

| process | before (prenight) | after (tonight) | Δ complete | Δ pp |
|---|---|---|---|---|
| CE | 23,225 / 53,315 (43.6%) | 26,387 / 54,040 (48.8%) | +3,162 | +5.2 |
| EA | 1,460 / 2,992 (48.8%) | 1,534 / 3,017 (50.8%) | +74 | +2.0 |
| EIS | 964 / 4,130 (23.3%) | 1,402 / 4,130 (33.9%) | +438 | +10.6 |

## Complete-timeline coverage, CLEAN energy only (the deliverable cohort)

| process | n (clean) | before | after | Δ complete | Δ pp | Phase-1 bench |
|---|---|---|---|---|---|---|
| CE | 19,261 | 38.5% | **43.2%** | +909 | +4.7 | ~30% (exceeds) |
| EA | 563 | 35.5% | **37.8%** | +15 | +2.3 | ~62% (source-limited gap) |
| EIS | 753 | 33.6% | **44.1%** | +79 | +10.5 | ~48% (approaching) |

## What drove the gains
- **EIS +10.6pp (all) / +10.5pp (clean):** the 12k-cap full-read + first/last-page retrieval recovered
  truncated ROD/FEIS dates (init_any 1,422→1,953 = +531; dec_any 2,204→2,418 = +214). The biggest win.
- **CE +5.2pp:** inferred-init proxy + the stale-`ranking_score` run-order fix (init_any +2,361, dec_any +2,646).
- **EA +2.0pp:** modest, as expected — EA initiation is genuinely source-limited (no NOI, faint text).
- **FR-NOI removal + calibrated-init eligibility** applied throughout.

## Caveats / honest notes
- **Current model, not the retrain.** The retrain succeeded and improved init F1 (0.882→0.893, decision
  flat), but was discarded because (1) the baseline eval crashed cosmetically writing a diagnostics CSV
  to a missing dir, and (2) the ranker retrain ran before the candidates parquet existed. Both driver
  bugs are now fixed; the retrain is de-risked and re-runnable as a deliberate pass.
- **08_analyze.R did not render figures** — the worktree was missing the `../phase1` symlink, so the
  Phase-1 comparison figures/tables didn't generate. Coverage data is unaffected (computed directly here).
  Re-runnable after symlinking phase1.
- **07_validate is a no-op** (project-level gold sample empty) — expected; that's the deferred
  validation-gold workstream.
- Labels preserved: `classifier.csv` 5,735 rows (+374), `ranker.csv` 363 (+6); originals byte-identical.

## Hand-back (STAGE ONLY — copy back if accepting; labels are gitignored, not a git merge)
- Outputs: `nepa-night/phase2/data/analysis/timeline/timeline_{candidates,project_dates}.parquet`
- Labels: `nepa-night/phase2/training/deliverable04/{classifier,ranker}.csv`
- Prenight backup for diffing: `timeline_*.prenight_20260616T013456.parquet` in the MAIN timeline dir.
