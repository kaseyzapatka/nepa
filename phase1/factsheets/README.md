# Phase 1 Factsheet — "NEPA by the Numbers"

Client-facing fact sheet mirroring the folder organization of
[`phase2/factsheets/`](../../phase2/factsheets/README.md), but sourced from the **final
submitted Phase 1 fact sheet** rather than rebuilt from a live pipeline.

## Source of truth

The user's final submitted Phase 1 fact sheet lives at
`admin/factsheets/phase1/Phase1_final.docx`. **`admin/` is gitignored — that .docx is not
tracked in this repository.** `factsheet1_key_insights.qmd` in this folder is a faithful
markdown/Quarto transcription of that document's text and table content, built for this
folder's organization to match Phase 2's conventions. It is not a live, data-driven rebuild:
see "How this differs from `phase2/factsheets/`" below.

## Naming convention

Files follow the Phase 2 pattern, `factsheetN_<topic>.qmd`. **Phase 1 only produced one
final fact sheet document** (unlike Phase 2's five separate topic fact sheets), so only
`factsheet1_key_insights.qmd` exists here. See "Structure of the source document" below for
why this wasn't split into multiple files.

## Structure of the source document

`Phase1_final.docx` is a single continuous document, not five separate topic fact sheets like
Phase 2. Its structure (preserved as section headings in `factsheet1_key_insights.qmd`):

- Title, authors/contributors
- Key Findings (callout box)
- Executive Summary + Recommendations
- Definitions
- Background (NEPA overview, the NEPA data challenge, a fresh perspective on NEPA)
- Finding 1: The NEPA process is categorically excluding small-scale projects and scrutinizing
  the impacts of larger ones (§1.1 CE share of energy reviews, §1.2 solar timelines and
  generation-capacity correlation)
- Finding 2: Effective agency implementation of existing NEPA options can speed review times
  (§2.1 programmatic/tiered reviews, §2.2 interagency coordination / bridge scores)
- Finding 3: FRA (2023) page limits are associated with a real, but modest, reduction in page
  lengths
- Caveats and Considerations
- Appendix: Scope of analysis (decarbonization/fossil technology tag classification, Table 1)

This mirrors `phase1/reports/key_insights.qmd` — the live-data Quarto source that was originally
rendered and then hand-finalized (title, executive summary, recommendations, definitions, and
caveats expanded) into `Phase1_final.docx`. `key_insights.qmd` still exists at that path and
remains the pipeline-driven version; it was NOT modified as part of building this folder.

## How this differs from `phase2/factsheets/`

- **One file, not five.** Phase 2 built five separate topic fact sheets (`factsheet1..5`) each
  keyed to a Phase 2 deliverable. Phase 1's final output was a single combined fact sheet
  covering three numbered findings — there is no clean way to split it into multiple documents
  without fragmenting cross-referenced content (e.g., the Key Findings callout box summarizes
  all three findings sections together).
- **Static transcription, not a live R/data build.** Phase 2's `.qmd`s compute every inline
  number from CSVs written by each deliverable's figure/table scripts, so the prose and figures
  stay in sync with pipeline re-runs (see `phase2/factsheets/README.md`, "How figures work").
  `factsheet1_key_insights.qmd` instead hardcodes the exact numbers and prose from
  `Phase1_final.docx` as static text — this is a faithful record of what was submitted, not a
  regenerable pipeline output. Do not add live R computation to this file without re-deriving
  and re-verifying every number against a rebuilt Phase 1 pipeline.
- **No verified figure files.** `Phase1_final.docx` embeds its own final chart images, which are
  not tracked in this repository (they only exist inside the gitignored .docx). Every figure in
  `factsheet1_key_insights.qmd` is a bracketed placeholder carrying the exact caption from the
  source docx, with a note pointing to the closest-content PNG already checked into
  `phase1/output/factsheet/figures/` (built by the `key_insights.qmd` pipeline) — but those
  candidate PNGs were **not verified pixel-identical** to the docx's embedded charts. Reconcile
  and wire up real `knitr::include_graphics()` calls before rendering this fact sheet to a
  client-facing document.
- **A few source-text issues are flagged inline, not corrected.** `Phase1_final.docx` contains a
  handful of internal inconsistencies and garbled sentences (a stat that appears to be missing a
  digit, a duplicated FRA-compliance statistic that disagrees between two locations in the same
  document, a dropped word in one sentence). These are preserved verbatim in
  `factsheet1_key_insights.qmd` with non-rendering `<!-- SIC: ... -->` HTML comments flagging the
  issue at that spot, rather than silently corrected — see the qmd source for the full list.

## Build

Rendering has not been run as part of building this folder (figure references are placeholders
— see above). Once figures are wired up, the render command follows the Phase 2 pattern:

```bash
# Render to the client .docx (use the base env, NOT the nepa env — see
# MEMORY.md "Render with base env, not nepa")
quarto render phase1/factsheets/factsheet1_key_insights.qmd --to docx
```

The `reference-doc` in the YAML front matter points to `phase1/reports/catf-reference.docx`
(the existing Phase 1 reference doc, itself gitignored like all `.docx` files but present on
disk) rather than a copy inside this folder, per the "no docx binaries in this folder"
constraint on how it was built.
