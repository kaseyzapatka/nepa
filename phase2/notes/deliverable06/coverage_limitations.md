---
title: "D6: FONSI-Patterns Coverage & Limitations"
---

*This page collects the caveats, assumptions, and sensitivity notes behind the
[D6 FONSI-patterns report](../../reports/deliverable06.html): what the headline numbers rest on,
how the facts were extracted and verified, and how the main conclusions would move under different
choices. All figures are current as of the committed pipeline outputs in
`phase2/data/analysis/deliverable06/` and `phase2/output/deliverable06/`.*

## Caveats & next steps

- **LLM-extracted, verified facts.** Action definitions, numeric limits, mitigation
  dependence, and significance thresholds come from a one-pass enrichment of all
  451 decarbonization FONSIs (Claude Sonnet); every quoted value was checked verbatim
  against its cited source page (97% verified). Values that did not verify are
  flagged, not shown.
- **CE matches are now adjudicated against the eCFR.** Each adopt/expand cell's top-5 CEs were
  checked against the current eCFR text (`ce_ecfr_verify.py` fetch → manual adjudication in the
  [eCFR verification worksheet](ce_ecfr_verification.html), wired into the verdicts by
  `07_classify_and_rank.py`): 10 cells *verified*, 12 *partial*, one flipped to *develop*, 1
  *unclear*. A structural limit surfaced by the fetch: of the 120 top-5 CE matches only 29 are
  codified in the eCFR — 72 live in agency NEPA-procedure documents and 19 use legacy eCFR URLs,
  so those are capped at *partial* (text-unverifiable), not *verified*.
- **Recommended next steps:** (1) for the 12 *partial* cells, confirm coverage against the agency
  procedure documents (outside the eCFR); (2) resolve the one *unclear* cell (Solar assessment);
  (3) the 92 *"other"*-action FONSIs are already clustered for within-cell sub-themes
  (`12_other_action_themes.py` — see the
  [supplementary themes table](../../reports/deliverable06.html#tbl-d6-other-themes) in the report).

## Assumptions & sensitivity

The conclusions rest on a few explicit assumptions — here is how each would move the result:

- **Every FONSI is placed in the grid** by its technology (from D3) and its LLM-labeled
  action verb — no candidate category is hand-picked, so the scan is exhaustive over the
  451 FONSIs. The one residual is the 92 FONSIs whose action the model could
  only label "other"; those cells exist in the grid but their internal structure is not
  decomposed here.
- **"CE-shaped" adds a transmission shape gate** on top of the LLM's `is_bounded_low_impact`
  judgment (transmission must be modify-existing within an existing ROW). Dropping that gate —
  the looser, LLM-bounded-only definition — would *raise* the set from 215 to
  293, mostly by admitting large within-ROW reconductors that exceed a CE's
  typical mileage cap.
- **A retrieval score ≥ 0.40 is treated as a close CE.** This threshold is what separates
  *adopt/expand* cells (a close CE exists) from *develop* cells (none does); the adopt/expand
  matches clear it comfortably (min 0.4005, most well above 0.50); the two expand-cell scores (0.41, 0.41) sit closer to the line and would flip under a 0.50 cut.
- **The expand test keys on numeric bounds.** Most matched CEs are qualitative, so no
  *numeric* expand fires for them — but transmission upgrades exceed CE #19's ~25-mile cap,
  so 2 cells resolve to expand. Qualitative coverage elsewhere remains unverified.
- **Ranks use fixed component weights.** A systematic sweep (2,000 Dirichlet-sampled weightings,
  the [rank-sensitivity table](../../reports/deliverable06.html#tbl-d6-rank-sensitivity) in the
  report) confirms the leaders are a *band*: for cells with a wide IQR, read priority bands, not
  exact ranks. The top develop cell (Solar — other) holds a top-3 rank in ~90% of draws;
  thinner-n cells swing widely.
- **CE coverage is now adjudicated, not assumed.** Each adopt/expand cell's top-5 CEs were checked
  against the current eCFR text: 10 cells are *verified*, 12 *partial*,
  one flipped to *develop* (Hydropower new-build — no covering CE), and 1 is *unclear*
  and flagged for review (Solar assessment). A hard limit remains: only CEs codified in the eCFR earn a
  clean *verified*; matches held in agency-procedure documents (not the eCFR) are capped at *partial*,
  so "partial" here means text-unverifiable, not necessarily narrow.
