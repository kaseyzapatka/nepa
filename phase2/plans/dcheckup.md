# Phase 2 stale-figure checkup — 2026-07-24

Read-only cleanup-agent audit of all six deliverables' figure closures (figures on disk vs.
generating code vs. report references vs. mtimes). One agent per deliverable. **No files were
modified by the audit.**

## Bottom line

**No stale figures in any deliverable's report closure.** All 144 report-referenced figures
across D1–D6 exist, are produced by live code, and were regenerated after the last edit of
their generating script. No orphaned figures, no missing references. The mtime inversions
caused by the uncommitted D3/D4 script edits were investigated and are benign (see below).
The only formal staleness is D5's dependency on the rebuilt D4 timeline parquet (impact
verified nil), plus genuinely stale **factsheet** outputs.

## Per-deliverable verdicts

| Deliverable | Verdict | Detail |
|---|---|---|
| D1 | CLEAN (15/15) | fig1–12 from `02_create_figures.R` (Jul 20 run, post-edit), fig13–15 from `04_secondary_review_crosstabs.R` (Jul 22). docs/ copies byte-identical. |
| D2 | CLEAN (26/26) | `significance_determinations.parquet` rewritten Jul 23 *after* figures, but verified manifest-only (mitigation aggregates recomputed from current parquet match figure-era CSV exactly: 258/193/162/0.628/149/0.578). |
| D3 | CLEAN (25/25) | Uncommitted `04_create_figures.R` diff removes only dead Section 6 (fig20 + 2 CSVs) gated on a nonexistent `timeline.parquet` — never produced output. Figures (Jul 23 10:03–10:05) postdate last committed script change (Jul 20). |
| D4 | CLEAN (28 report figs; 36 on disk) | Uncommitted `08_create_figures*.R` diffs are comment-only. Jul 24 `duration_days` finalize fix changed only that column (37,959 rows; non-null 23,922 → 27,046), which no figure script reads raw — all recompute from date columns. Report qmd never touches the parquet. |
| D5 | Consistent but FORMALLY STALE | See below. |
| D6 | CLEAN (20/20) | Figures Jul 22 16:28, post-script-edit, post-data. No BERTopic leftovers (only archived parquet in `_archived_v1/`). 2 unreferenced figures (`fig_d6_outcomes_waffle`, `fig_d6_states`) documented as intentionally kept in output README. |

`.rds` sidecars alongside PNGs in every deliverable are the deliberate cross-deliverable
retitle convention (factsheet pipeline) — not orphans, keep.

## D5 formal staleness

- Anchor input `phase2/data/analysis/timeline/timeline_project_dates.parquet` was rewritten
  Jul 23 17:53 (D4 tier-C re-run, 254 date refinements) — ~8h **after** the D5 build
  (Jul 23 10:08).
- D5's own freshness guard (`03_create_figures.R` lines 263–283) would `stop()` with
  "Stale citation scan" if run now.
- **Impact verified nil**: the post-rebuild re-render of `deliverable05.qmd` (commit
  e8f9a8c) is byte-identical in visible text to the pre-rebuild version — the 254
  refinements moved no published D5 number.
- **Action**: after the in-flight D4 finalize work is committed and the timeline parquet is
  final, re-run `01_extract_law_citations.py --source all` → `02_build_ce_categories.py` →
  `03_create_figures.R`, re-render the report. Regeneration is for formal reproducibility,
  not because anything shown is wrong.

## Action items

- [ ] **Factsheets stale (real numbers impact)** — `phase2/code/factsheet_figures.R`
  (stopgap block, lines 162–166) reads the parquet's **raw** `duration_days`, the exact
  column fixed on Jul 24 (non-null universe 23,922 → 27,046 → prior factsheet undercounted).
  `fs1_duration_by_technology.png`, `fs1_duration_by_technology.csv`,
  `fs1_duration_by_energy.csv`, `factsheet1_timelines.docx` (Jul 17) must be rebuilt.
  *(Being handled 2026-07-24.)*
- [ ] **D3 runbook stale** — `phase2/runbooks/deliverables/deliverable03.md` (lines ~18,
  101, 184) still describes deleted Section 6 / fig20 / `timeline_coverage.csv` /
  `duration_summary.csv`. Architecture doc already updated in the same uncommitted batch.
  *(Being handled 2026-07-24.)*
- [ ] **Commit the D4 finalize fix** — `05_select_dates.py`, `06_adjudicate_llm.py`,
  untracked `_finalize_duration.py`, comment-only `08_*.R` updates: functional pipeline
  code currently only on disk. Repair mode was already run (parquet Jul 24 15:06; backup
  `timeline_project_dates.pre_finalize_20260724T220619Z.parquet` kept).
- [ ] **D5 regeneration** — sequence after the D4 commit (see above).
- [ ] **D1 confirm-no-rebuild** — `01_extract_nepa_trigger.py` got a 7-line rule change in
  efe8be3 (Jul 23, publish-review remediation) *after* `projects_nepa_trigger.parquet` was
  built (Jul 20). Likely intentional code-to-shipped-state reconciliation, but no note says
  "no rebuild needed." Confirm; if a behavior change was intended, re-run 01 → 02 → 04 →
  render.
- [ ] **D3 re-render** — `docs/phase2/reports/deliverable03.html` (Jul 23 18:07) predates
  the Jul 24 qmd edit (removes one unused variable). Re-render before next publish.
- [ ] **Timeline dir hygiene** — backup/temp files accumulating in
  `phase2/data/analysis/timeline/` (`*.prefilter_bak`, `*.scrubbed.tmp`, `pre_adj`,
  `pre_gt_inject`, `pre_finalize` backups). Candidates for a data-cleanup pass.
- [ ] **D3 rds size** — `fig19_visual_section_length.rds` (83MB) and
  `fig19a_section_length_energy.rds` (84MB) embed full data; discuss before public-repo
  push.
- [ ] **D6 cosmetic** — script-number collisions (two `01_`, two `05_` scripts); committed
  `01_build_fonsi_inventory.py` / `05_build_fonsi_packets.py` postdate their Jul 1 output
  parquets (outputs built by pre-checkin versions; predates figures, no freshness issue).
- [ ] **D3 output CSVs from May** — Python-side outputs (`visual_topic_terms_detail.csv`
  — read by the qmd — `nmf_elbow_data.csv`, `visual_qa_sample.csv`) date to May 15–19;
  confirm currency during a full D3 pass.
