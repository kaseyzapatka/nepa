---
title: "D2: Significance Coverage & Limitations"
---

*As of 2026-07-22, describing the published extraction runs (FONSI and EIS batch passes of
2026-07-08/09, schema `d2_v2_11`). All numbers on this page are recomputed from the committed
pipeline outputs in `phase2/data/analysis/deliverable02/`.*

This page collects the methods fine print behind the
[D2 significance report](../../reports/deliverable02.html): what is in scope, how far the two
extraction funnels reach, how the numbers were validated, and the caveats that should ride along
with any reuse of the data.

## Scope

- **Decarbonization only** (`project_energy_type = 'Clean'`).
- **Resource areas.** We read *"across resource areas"* as the standard NEPA environmental
  categories the documents are organized around — air quality; water (including wetlands and
  floodplains); biological resources; cultural / historic; visual; noise; soils / geology;
  socioeconomics (including environmental justice); transportation; land use; climate / greenhouse
  gases; and public health (including hazardous materials). Agencies judge significance separately
  for each, so this set defines the rows of every resource map in the report.
- **Agency scope differs by track.** The **FONSI** track is limited to **BLM + DOE**: of the 452
  decarbonization FONSI projects, 427 (94.5%) are BLM + DOE and carry the headline, while 25
  (5.5%, led by other agencies with only partial coverage or flagged for manual review) are held
  as context and never mixed into the primary rates. The **EIS** track instead covers **all
  agencies** — it is descriptive and does not rest on the completeness a rate comparison needs —
  with BLM + DOE isolated wherever a finding is agency-sensitive (e.g. the FONSI-vs-EIS
  comparison, which restricts both tracks to BLM + DOE).

## Coverage — two extraction funnels

- **FONSI.** Three filters take the 452-project corpus to the analyzed set (traced in the
  [report's funnel figure](../../reports/deliverable02.html#fig-fonsi-funnel)): **452 → 427**
  (agency scope), **427 → 261** (a dated, 2009-or-later decision: 105 projects have no
  machine-extractable decision date, 55 predate 2009, 6 sit on a regime boundary — every FONSI
  rate is read against the decision's regulatory era, so undated projects cannot be placed),
  and **261 → 193** (extraction coverage: finding sections are recognized in 224 of the 261 —
  the other 37 documents were parsed but phrase their finding in unrecognized wording — and in
  31 more the flagged text contains no codable determination). The analyzed set is **193
  projects / 258 decision documents**; the extraction gaps are a **coverage limit of the source
  extraction, not a sampling choice**. FONSI finding sections derive from D6 artifacts.
- **EIS.** Determinations are retrieved **live from ~45,000 document sections** by keyword and
  pattern (FONSIs instead ride on a curated span set). Of the 753 EIS corpus projects,
  significance sections were retrieved for 536 and **506 projects (1,082 documents, all
  agencies)** yielded a determination. Of those 506, only 239 have a firm in-window decision
  date; the rest (206 undated, 57 pre-2009, 4 boundary) are **kept**, because the EIS analysis
  is descriptive and never uses the date. The held-out test recovers **~77%** of the
  determinations a human finds — conclusions stated in summary-of-impacts tables and records of
  decision are the main miss. **Read the EIS rates as a well-grounded floor on where impacts
  cross the line, not an exhaustive census.**

## Validation

- **The answer key is AI-coded and human-audited.** Two AI coding agents (Claude and Codex)
  independently coded a stratified 400-window sample per track (each window read once; one label
  per resource conclusion); a human analyst audited both sets, accepting values where the coders
  agreed on the core fields and hand-adjudicating every disagreement (over half the rows required
  an adjudication decision on at least one field). "AI-human reviewed" in the report is shorthand
  for that process.
- **The held-out test.** A deterministic **30% of the windows was held out** — never used to
  build or tune the extraction — and the pipeline was scored separately on it. **The held-out
  scores are the ones reported.** Rare class/resource/threshold cells are reported descriptively,
  not as pass/fail macro-metrics.
- **EIS is validated lower than FONSI, for a real reason.** The EIS significance *class* agrees
  with the answer key at **0.69** (vs 0.81 for FONSI). Separating *significant* from *significant
  and unavoidable* is a genuinely fine judgment — the two AI coders themselves agreed only
  **~58%** on the EIS class (vs 68% for FONSI). The report therefore leads with the robust cut
  (*which resources cross the line*) and treats the adverse-vs-unavoidable split as directional.
- **EIS exploratory fields are not gold-validated.** The `alternative_name`,
  `significance_factor`, and `impact_type` fields are captured for the above-the-line analysis
  but sit **outside** the answer key (which validates class, resource, mitigation, and
  threshold). Treat the by-factor and by-pathway breakdowns as indicative, not as validated
  measurements.
- **Resource-level mitigation pairing: validated tags, aggregate-only claim.** The report's
  finding that ~23% of flagged significant / less-than-significant FONSI determinations are paired
  with a same-resource committed mitigation condition rests on condition→resource tags validated on
  an 80-project gold set (independent model labeler, plus an informal second-reviewer blind
  spot check that found no disagreements — not a persisted comparison artifact): tag-level
  **F1 0.83**, **any-overlap accuracy 0.89** (vs 0.46 for the prior keyword
  tags). The finding is reported **aggregate-only** and under **any-overlap** matching (a
  determination counts as paired if *any* same-resource commitment sits in its window). Two limits
  ride with it: (a) the gold's one known bias — inclusive multi-labeling — slightly *inflates*
  measured match rates, so the aggregate share is a mild over-count, **not a floor**; (b) the
  any-overlap rule is deliberately inclusive, so exact per-determination attribution is weaker
  (condition-tag precision ≈**0.76**) and the **per-resource splits are directional, not a census**.
  The separate `mitigation_dependent` flag (whether an *individual* conclusion legally depends on
  mitigation) is a labeled **screening** metric only — tightened this pass to require a real
  same-resource overlap plus ≥2 committed conditions, scoring **F1 0.62 / precision 0.53** (ceiling
  ≈0.53) — and is never a reported share. One taxonomy boundary is documented: **"sociocultural
  systems" commitments map to `socioeconomic`, not `cultural`**, under the taxonomy's narrow
  (historic / §106) cultural definition — a ratified reading that affects only the per-resource
  splits, never the aggregate.

## EIS extraction specifics

- **Window size.** EIS sections run longer than FONSI ones, so the EIS pass reads each section to
  **24,000 characters** (vs 16,000 for FONSI), capturing ~95% of sections in full; the ~5% that
  run longer can still lose a conclusion past the cap.
- **De-duplication.** A Draft and Final EIS repeat identical section text; such passages are
  counted **once per project**. Identical text across *different* projects is kept — it is
  legitimate per-project attribution.
- **EIS mitigation is the class signal only.** "Below the line only with committed mitigation" on
  the EIS side is the model's class judgment; there is **no** record-of-decision commitment
  linkage (the enforceable-condition artifacts are FONSI-only). ROD-commitment mitigation is out
  of scope for this pass.
- **Two dropped windows.** 2 of 21,854 EIS windows (0.009%) returned no result from the batch API
  (a transient error) and were recorded as missing rather than silently dropped — no material
  effect on any rate.

## Other caveats

- **D4 date/regime coverage.** Many projects lack a high-confidence decision date; regime cells
  are confidence-gated and thin post-2023. EIS decision dates are frequently proxies.
- **EPA review letters (Clean Air Act §309) are out of scope.** Under Section 309, the EPA
  independently reviews and *rates* other agencies' environmental impact statements. Those
  comment letters are not in this dataset, so the report makes no claim about EPA's own view of
  significance.
- **Technology comes from the dataset's `project_type` classification.** Each project carries a
  curated multi-tag project-type field; we assign one **primary** technology per project by
  priority (generation types first, then nuclear/CCS, then transmission). About **1 in 5
  EIS projects and 1 in 4 FONSI projects** resolve to an "Other / mixed" bucket, and the
  "Renewable (other)" catch-all is also
  set aside; both are excluded from the technology figures, which therefore describe the
  clearly-typed technologies.
- **Mitigation enforceability** is the model's read of whether a committed measure is tied to an
  enforceable permit condition; an "unmatched" conclusion may still be enforced through channels
  the extraction doesn't capture, so the enforceable shares are a floor.
- **Suppressed cells.** Any cell below 5 is suppressed.

## Where the data lives

Determination parquets and validation metrics: `phase2/data/analysis/deliverable02/`; derived CSV
tables and figures: `phase2/output/deliverable02/analysis/`; the answer keys (AI-coded,
human-adjudicated): `phase2/data/analysis/deliverable02/gold/`. Schemas for every emitted parquet
are documented in `phase2/notes/deliverable02/data_dictionary.md`, and full run provenance is
recorded in `significance_run_manifest.parquet`.
