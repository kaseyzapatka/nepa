# Archived D4 labeling round — night-run 2026-06-16 (NOT incorporated)

Frozen snapshot of the active-learning labeling round produced in the `nepa-night`
worktree on 2026-06-16. **These labels are deliberately NOT in the live training set.**
The current/committed D4 classifier (`salvage_20260609`) is trained *without* them and
that is intentional — see the regression below. Archived here so the manual labeling work
isn't lost when the worktree is deleted.

## What's here

| File | What it is |
|---|---|
| `classifier.csv.gz` | Full merged candidate-classifier store, **5,735 labels** = the committed 5,361 **plus 374 new** rows from this round (gzipped; `gunzip` to restore native format). |
| `ranker.csv` | Full project-pairing store, **363 rows** = committed 357 **plus 6 new**. |
| `labeling/*.labeled.csv` | The human-labeled worksheets that fed the 374/6 new rows (active=148, init=118, neither=68, final_eis=40; ranker_answers=6). |
| `classifier_ids.txt` | Candidate-id manifest from the round. |
| `morning_report_20260616.md` | The night-2 run report (note: it oversold the result — omitted the final_eis collapse below). |

The new labels are purely **additive** — the committed label set is a byte-identical
subset, zero rows diverged. The 374 new classifier labels skew EIS (232 EIS / 80 CE /
62 EA) and are 307 `neither` / 49 `initiation` / 18 `decision`. All 374 have an empty
`split` (staged, never assigned train/test).

## Why NOT incorporated — the retrain regressed `final_eis`

Retraining with these labels (same 938-row frozen test):

| head | before (committed `salvage_20260609`) | after retrain (+374 labels) | Δ F1 |
|---|---|---|---|
| initiation | F1 = 0.882 | F1 = 0.893 | +0.011 (slight gain) |
| decision | F1 = 0.885 | F1 = 0.882 | −0.003 (flat) |
| **final_eis** | **F1 = 0.560** (R=0.636) | **F1 = 0.214** (R=0.136) | **−0.346 (collapse)** |

The `final_eis` head — a small, fragile head (~44 test positives, P≈0.500 at best) —
lost most of its recall (caught 6 of 44 vs 28 before). One small win, one wash, one big
regression. Keeping the committed model preserves the much healthier final_eis head and
keeps the pipeline reproducible.

## If revisiting later

The labels look correct; the collapse reads as a small-n / training-recipe instability on
the final_eis head, not bad labeling. Before folding back in:
1. `gunzip classifier.csv.gz`, copy `classifier.csv` + `ranker.csv` back to
   `phase2/training/deliverable04/`, assign a `split` to the 374 new rows.
2. Stabilize the final_eis head first (e.g. freeze/upweight final_eis, avoid adding EIS
   `neither` hard-negatives that starve it) and re-eval on the frozen test before trusting.
