# Phase 2 Factsheets — "NEPA by the Numbers"

Client-facing factsheets built from the Phase 2 deliverable reports, per
`admin/factsheets/phase2/outline.docx`. Styling follows the Phase 1 factsheet
(`admin/factsheets/phase1/Phase1_final.docx`, rendered from
`phase1/reports/key_insights.qmd`) via `catf-reference.docx`.

| File | Outline | Sources |
|---|---|---|
| `factsheet1_timelines.qmd` | Fact Sheet 1: Timelines (highest priority) | D4 report + Phase 1 D2/D3 recaps |
| `factsheet2_triggers.qmd` | Fact Sheet 2: What Triggers NEPA? | D1 report |
| `factsheet3_categorical_exclusions.qmd` | Fact Sheet 3: Categorical Exclusions | D6 report + D3 CE-authorities figure |
| `factsheet4_visual_impacts.qmd` | Fact Sheet 4: Visual Impacts | D3 report (visual module) |
| — | Fact Sheet 5: Determinations of Significance | TK (plan pending) |

## Build

```bash
# 1. Regenerate/refresh factsheet figures (headline-titled retitles + passthrough
#    copies of the deliverable PNGs) into phase2/output/factsheet/figures/
#    NOTE: run each deliverable's NN_create_figures.R first — they write the .rds
#    sidecars this script reads. (Rscript phase2/code/deliverable0X/...)
Rscript phase2/code/factsheet_figures.R

# 2. Render a factsheet to the client .docx (use the base env, NOT the nepa env)
quarto render phase2/factsheets/factsheet1_timelines.qmd --to docx
# .docx lands next to the .qmd (phase2/factsheets/)
```

## How figures work

- **Every deliverable figure script is named `NN_create_figures.R` and, for each
  figure, saves a `.rds` of the ggplot object right next to its `.png`** (same
  directory, same basename). That `.rds` is the reusable source for retitling —
  no re-computation downstream.
- All factsheet `.qmd`s read figures from **one directory**:
  `phase2/output/factsheet/figures/`.
- `phase2/code/factsheet_figures.R` fills that directory three ways:
  - **RETITLED** — `readRDS()` the deliverable's `.rds`, add a client-facing
    headline title with `labs()`, and `ggsave()` (the `fs1_*` / `fs2_*` figures).
    `labs()` overrides the upstream title and keeps the rest of the plot intact.
  - **FROM SCRATCH** — built here only when there is *no* upstream original to
    reuse (currently just `fs1_duration_by_technology`, which carries a TODO to
    convert once D4 adds that figure to its `08_create_figures.R`).
  - **COPIED** — passthrough copy of the deliverable's PNG (`fig_d4_*`,
    `fig_d6_*`, etc.), unchanged.
- To retitle a passthrough figure: promote it to a RETITLED block in
  `factsheet_figures.R` (`readRDS` its `.rds` + `labs()`); if the originating
  script does not yet write a `.rds`, add one there first.
- Inline numbers in the `.qmd`s are computed in each setup chunk from the same
  summary CSVs the reports use (`phase2/output/factsheet/tables/`), so they stay
  in sync with pipeline re-runs.

## Known TODOs (first-draft placeholders)

- FS1 intro: `[cite the Phase 1 report's timelines literature — RFF et al.]`
  needs the actual citations from CATF's Phase 1 report.
- FS1: CEQ-regulatory-regime duration segmentation is flagged "in progress"
  (D4 report section is a proposal, not yet built).
- FS1: outlier case studies (SunZia etc.) marked as CATF-to-add.
- Website publication (adding these to `_quarto.yml` render targets + navbar)
  is deliberately deferred — tbd.
