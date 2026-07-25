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

- [x] **Factsheets stale (real numbers impact)** *(DONE 2026-07-24, commit 0deeb3d)* — `phase2/code/factsheet_figures.R`
  (stopgap block, lines 162–166) reads the parquet's **raw** `duration_days`, the exact
  column fixed on Jul 24 (non-null universe 23,922 → 27,046 → prior factsheet undercounted).
  `fs1_duration_by_technology.png`, `fs1_duration_by_technology.csv`,
  `fs1_duration_by_energy.csv`, `factsheet1_timelines.docx` (Jul 17) must be rebuilt.
  *(Being handled 2026-07-24.)*
- [x] **D3 runbook stale** *(DONE 2026-07-24, commit 0deeb3d)* — `phase2/runbooks/deliverables/deliverable03.md` (lines ~18,
  101, 184) still describes deleted Section 6 / fig20 / `timeline_coverage.csv` /
  `duration_summary.csv`. Architecture doc already updated in the same uncommitted batch.
  *(Being handled 2026-07-24.)*
- [x] **Commit the D4 finalize fix** *(DONE 2026-07-24)* — `05_select_dates.py`, `06_adjudicate_llm.py`,
  untracked `_finalize_duration.py`, comment-only `08_*.R` updates: functional pipeline
  code currently only on disk. Repair mode was already run (parquet Jul 24 15:06; backup
  `timeline_project_dates.pre_finalize_20260724T220619Z.parquet` kept).
- [x] **D5 regeneration** *(DONE 2026-07-24)* — guard passed after fresh citation scan
  (8,741 law rows; 74,035 category rows). Deltas trivial: spike-window counts moved by
  1–5 projects from the timeline reshuffle (e.g. IRA-in-EIS 18.9% → 18.8%); all headline
  findings (DOE spike, 1.60× BIL, B5.1 surge) unchanged. Report re-rendered.
- [x] **D1 confirm-no-rebuild** *(RESOLVED 2026-07-24: zero blast radius, no rebuild)* —
  the efe8be3 rule edits were dead-code cleanup: the underlying patterns were deleted
  Apr 23 (`license_amendment`, ef481f8) and Apr 24 (`rmp`, 61d0a1d), months before the
  Jul 20 parquet build. Shipped parquet has 0 rows on either rule id; a re-run would be
  byte-identical for this diff. (`trigger_language_bank.csv` added in same commit is not
  a pipeline input.)
- [x] **D3 re-render** *(DONE 2026-07-24)* — re-rendered with system quarto; visible-text
  diff is date-stamp only; all 25 figures embedded.
- [x] **Timeline dir hygiene** *(DONE 2026-07-24)* — verified write-only (code creates
  them as pre-repair snapshots, nothing reads them); deleted all six (~55MB). The two
  Jul 23 tier-C backup *dirs* (`_pre_tierc_backup_20260723`, `_tierc_state_20260723`)
  were left — same character, deletable on user say-so.
- [x] **D3 rds size** *(RESOLVED 2026-07-24: non-issue for the repo)* — `*.rds` is
  globally gitignored (`.gitignore:45`), zero tracked; the 79–80MB fig19/fig19a sidecars
  (ggplot objects embedding the full per-section dataset) are disk-only and regenerate on
  every figure run. Optional local-disk fix: aggregate before plotting in those two figs.
- [x] **D6 cosmetic** *(CLOSED 2026-07-24, won't-fix)* — numbering collisions left as-is
  (pipeline complete + re-run; renaming would break runbook/doc references for no
  functional gain). Provenance note: `01_build_fonsi_inventory.py` /
  `05_build_fonsi_packets.py` were committed Jul 23 as whole-file check-ins of code
  already running since ~Jul 1; outputs built by pre-checkin versions, downstream verified
  consistent, re-verify only if D6 is touched again.
- [x] **D3 output CSVs from May** *(VERIFIED CURRENT 2026-07-24)* — generating scripts
  (`01_identify_visual_impact_candidates.py`, `03_inventory_visual_sections.py`) are
  unchanged since May (git diff to HEAD empty); the CSVs were written May 19 14:22,
  *after* the final commit (May 19 10:48). Current by construction; no re-run needed.
