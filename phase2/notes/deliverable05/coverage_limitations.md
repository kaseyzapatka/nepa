---
title: "D5: CE-Spike Coverage & Limitations"
---

*This page collects the methods fine print behind the
[D5 CE-spike report](../../reports/deliverable05.html): what is in scope, how the analysis
separates a genuine policy signal from an artifact of the dataset's growth, and the caveats that
should ride along with any reuse of the numbers. All values are drawn from the committed pipeline
outputs in `phase2/data/analysis/deliverable05/` and `phase2/output/deliverable05/diagnostics/`.*

## Scope

The base population is every categorical exclusion that can be placed in time. Each CE is anchored to
the date its determination was issued — the **decision date** from the Deliverable 4 timeline — and
where that is absent the analysis falls back to the **initiation date** as a same-year proxy. The
fallback is safe for CEs specifically: their median duration is roughly three weeks, so the two dates
almost always fall in the same calendar year, and placing a CE in a year requires only one date rather
than the two that a duration measurement needs. This yields a base of **52,089 of 54,668 CE projects
(95.3%)** that can be located on the timeline — a far broader base than the complete-timeline set used
for the duration analysis in Deliverable 4.

The three named laws are the American Recovery and Reinvestment Act (ARRA, Feb 2009), the Bipartisan
Infrastructure Law / Infrastructure Investment and Jobs Act (BIL/IIJA, Nov 2021), and the Inflation
Reduction Act (IRA, Aug 2022). Citation detection scans CE, EA, and EIS document pages, so the
association layer reaches beyond the CE-only volume analysis; the category-mix analysis is CE-only.

## The coverage ramp, and how the analysis works around it

A raw count of CEs per year mixes three distinct things: genuine policy-driven surges, the growth of
NEPATEC's document coverage over time, and a drop-off in the most recent years caused by recent
documents not yet being ingested. The dataset is sparse before 2009, so a naive reading would credit
ARRA with a level shift that is partly just the corpus filling in. The report therefore does not lean
on aggregate counts to make the causal claim. **Two devices isolate the real signal.**

The first is *conditioning on agency*. The post-ARRA surge is a Department of Energy phenomenon: DOE
administered ARRA's energy grants, loan guarantees, and weatherization money, and it is DOE's CEs that
jump. The Bureau of Land Management — the only other agency issuing CEs at scale, and one drawing on the
same corpus subject to the identical coverage ramp — stays flat through the ARRA window. Because both
agencies experience the same dataset growth, the contrast between them cannot be explained by coverage
and is the cleanest available evidence of a policy effect.

The second device is *citation evidence*. A Recovery Act citation cannot appear in a document before the
Recovery Act existed, so an explicit citation is attribution that is robust to the coverage ramp
entirely. In the ARRA window, 59.7% of CEs cite the Recovery Act by name (n = 7,014 window CEs), versus
a negligible rate outside it. Volume tells us something happened; citations tell us it was tied to the
law.

## Caveats to keep stating

**Correlation is not attribution.** The volume spike is a correlational observation; the citation
analysis is the stronger attribution layer. The report presents both and rests the causal claim on the
citations rather than on the counts.

**Pre-2009 coverage is thin.** NEPATEC contains few CE documents before 2009, so ARRA's apparent *level*
shift is partly a coverage ramp. The ARRA claim rests on the DOE-vs-BLM contrast and the 59.7% citation
rate, not on raw counts, and the report reports no ARRA window-versus-baseline ratio for exactly this
reason.

**The BIL and IRA windows overlap.** The two laws passed nine months apart, so their post-law windows
cannot be cleanly separated by date alone. Where an action needs to be attributed to one law rather than
the other, the citation evidence does the work that the calendar cannot. Both laws show a real but far
more muted pattern than ARRA: DOE CE activity rose to 1.60× its pre-law monthly baseline after BIL, but
explicit CE citations to BIL and IRA are rare because these laws are seldom named in a one-page
categorical exclusion. They surface far more often in larger reviews — IRA is cited in 18.9% of post-IRA
EISs (versus 8.7% before) and BIL in 17.5% of post-BIL EISs — so the citation evidence for BIL and IRA
lives in the EIS record even though the volume signal lives in the DOE CEs.

**The most recent years are incomplete.** The decline visible in 2024–2025 reflects ingestion lag, not a
real fall in CE use, and those years should not be read as a downward trend.

**Dates are extracted, not authoritative.** The determination date comes from the Deliverable 4
extraction, which places 95.3% of CEs; the remaining few percent carry no usable date and are absent from
the time series. Coverage is not the same as accuracy — end-to-end date accuracy is reported in tiers in
Deliverable 4, and proxy and fallback dates are flagged there.

## A note on the category-mix evidence

The category-shift analysis compares the DOE categorical-exclusion mix inside the ARRA window against a
stable 2016–2019 baseline (there is no usable pre-ARRA baseline, again because of the coverage ramp). The
standout is category **B5.1, "Actions to conserve energy or water," which rises to 49.7% of DOE CEs in
the ARRA window from 1.2% at baseline** — the energy-efficiency stimulus showing up directly in the
categorical-exclusion mix. Because this is a within-window *share* rather than a raw count, it is far less
sensitive to the coverage ramp than the volume series: it describes how the composition of DOE CEs
changed, not how many there were.

## Where the data lives

- Normalized law citations: `phase2/data/analysis/deliverable05/law_citations.parquet` (project- and
  page-level ARRA / BIL / IRA citations, with context-based disambiguation of the ambiguous acronyms),
  built by `phase2/code/deliverable05/01_extract_law_citations.py`.
- Normalized CE categories: `phase2/data/analysis/deliverable05/ce_categories.parquet` (document
  `ce_category` metadata parsed to CFR codes — DOE 10 CFR 1021, DOI 516 DM 11, EPAct 2005 §390), built by
  `phase2/code/deliverable05/02_build_ce_categories.py`.
- CE dates: the decision and initiation dates come from the Deliverable 4 timeline,
  `phase2/data/analysis/timeline/timeline_project_dates.parquet`.
- Figures and diagnostic CSVs: `phase2/output/deliverable05/figures/` and
  `phase2/output/deliverable05/diagnostics/`, produced by
  `phase2/code/deliverable05/03_create_figures.R`.
