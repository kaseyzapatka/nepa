---
title: "D3: Review-Process Coverage & Limitations"
---

*This page collects the methods fine print behind the
[D3 review-process report](../../reports/deliverable03.html): what is in scope, how far each
analysis reaches, and the caveats that should ride along with any reuse of the data. All numbers are
recomputed from the committed pipeline outputs in `phase2/data/analysis/deliverable03/` and
`phase2/output/deliverable03/`.*

## Scope

- **Energy projects only.** The analysis universe is the **31,508** NEPATEC 2.0 projects classified
  as **Clean** (20,725, "Decarbonization") or **Fossil** (10,783, "Fossil Fuel"). Projects
  classified as `Other` are excluded before analysis so the clean/fossil comparison is not diluted
  by unrelated federal actions.
- **Review process.** Every project carries a CE / EA / EIS process type read from
  `projects_combined.parquet`; the review-rate comparisons are complete for both portfolios.
- **Visual-impact analysis** is restricted to **EA and EIS** documents (CE forms are out of scope),
  and within those to projects whose documents expose a recognizable visual-resource section.

## Known gaps and cautions

**NEPA triggers are clean-energy only.** The CE-by-trigger analysis rests on Deliverable 1's trigger
classifications, which cover the **clean-energy** portfolio; fossil projects currently have no
trigger values. Trigger-based comparisons should therefore be read as clean-energy statements, not
clean-vs-fossil claims. Extending the trigger logic to the fossil portfolio is the prerequisite for
a symmetric comparison.

**Linear geometry is an advisory heuristic.** The `is_linear` flag is derived from the NEPATEC
`project_type` taxonomy — transmission, pipelines, and corridors are classed as linear, everything
else as point/area. A project carrying any linear label is treated as linear, since geometry is a
property of the built infrastructure. This is a label-based heuristic, not verified project geometry,
and should be treated as advisory.

**Geography counts are footprints, not unique projects.** State and county maps and tables count
**project-state** and **project-county** records: a project spanning multiple states or counties
contributes to each. This is the correct denominator for a geographic-footprint question but should
never be read as a unique-project count.

## Geothermal control caveats

The geothermal-vs-oil-and-gas comparison in the main report is an **all-agency blend** — it pools
BLM, USFS, DOE, and other lead agencies. Three limitations bound how far it should be read:

- **It mixes pathways.** The blend answers the broad technology question (does geothermal move
  through NEPA more like decarbonization or like oil and gas?) but does not isolate any single
  agency's permitting regime.
- **A BLM-only control was considered and deliberately not included.** A public-land (BLM-lead)
  geothermal-versus-oil-and-gas comparison would be the right way to test the public-land question,
  but it is out of scope for this version and is noted as a candidate follow-up rather than an
  available result.
- **Federal-land trigger share is not classified for oil and gas.** The comparison table's
  Federal-Land Trigger Share reads "Not yet classified" for the oil-and-gas rows because trigger
  classification was not extended to the fossil portfolio; only the clean-energy (including
  geothermal) side carries trigger values.

## Where the data lives

- Project-level review data: `phase2/data/analysis/deliverable03/projects_nepa_reviews.parquet`
  (one row per project — energy category, technology group, review process, lead agency, geography,
  geometry, and clean-energy trigger information).
- Normalized CE citations: `phase2/data/analysis/deliverable03/ce_citations.parquet`.
- Geothermal vs. oil-and-gas subset: `phase2/data/analysis/deliverable03/projects_geothermal_og.parquet`.
- Figures and CSV tables: `phase2/output/deliverable03/`.
- Build scripts: `phase2/code/deliverable03/02_build_nepa_reviews.py` (datasets) and
  `phase2/code/deliverable03/04_create_figures.R` (figures and tables).
