# Deliverable 04 — Timeline Coverage & Quality Facts

Empirical answers for kicking the tires on D4 data coverage/quality.

**Source run:** `phase2/data/analysis/timeline/timeline_project_dates.parquet`
(timeline_run_at max = `2026-06-10T08:25:53Z`; file mtime 2026-06-10 01:28 local)
**Rows:** 59,215 projects (CE 52,093 · EA 2,992 · EIS 4,130)
**Energy join:** `phase1/data/analysis/projects_combined.parquet` (`project_energy_type`); all 59,215 matched.
**Compiled:** 2026-06-10

Definitions (match `08_analyze.R`):
- *has initiation* = `initiation_date` present (any granularity)
- *has decision* = `decision_date` present (any granularity)
- *overlap / complete* = both present → `timeline_complete` (= the metric in `fig_d4_complete_timeline_share_boxplot.png`)

---

## 1. Initiation Decision overlap

### A. All projects (matches the current published figure)

| Review type | Total projects | Has initiation | Has decision | **Overlap (both)** |
|-------------|---------------:|---------------:|-------------:|-------------------:|
| CE          | 52,093         | 21,160 (40.6%) | 42,825 (82.2%) | **15,348 (29.5%)** |
| EA          | 2,992          | 1,675 (56.0%)  | 2,220 (74.2%)  | **1,434 (47.9%)**  |
| EIS         | 4,130          | 1,264 (30.6%)  | 2,204 (53.4%)  | **842 (20.4%)**    |
| **All 3**   | **59,215**     | 24,099 (40.7%) | 47,249 (79.8%) | **17,624 (29.8%)** |

Reading the overlap: completion is gated by **initiation**, not decision. Decision
coverage is high everywhere (53–82%); initiation is the scarce endpoint (31–56%).
For CE and EIS the overlap ≈ the initiation rate, i.e. nearly every project that has
an initiation date also has a decision date — the missing initiation is the whole story.
EA is the exception (overlap 47.9% < initiation 56.0% < decision 74.2%), so EA loses
projects on both ends.

### B. Clean-energy projects only (apples-to-apples with Phase 1; see Q3)

| Review type | Total projects | Has initiation | Has decision | **Overlap (both)** |
|-------------|---------------:|---------------:|-------------:|-------------------:|
| CE          | 18,039         | 5,389 (29.9%)  | 16,677 (92.4%) | **4,519 (25.1%)**  |
| EA          | 557            | 265 (47.6%)    | 350 (62.8%)    | **188 (33.8%)**    |
| EIS         | 753            | 312 (41.4%)    | 468 (62.2%)    | **213 (28.3%)**    |
| **All 3**   | **19,349**     | 5,966 (30.8%)  | 17,495 (90.4%) | **4,920 (25.4%)**  |

**Clean vs non-clean diverge sharply for EA.** All-projects EA overlap is 47.9%, but
clean-only EA overlap is just 33.8% — meaning non-clean EA projects complete at ~51%
(1,246/2,435). The headline 48% EA figure is buoyed by non-clean EAs; clean-energy EAs
are materially worse covered. For CE the clean subset has *higher* decision coverage
(92.4% vs 82.2%) but *lower* initiation coverage (29.9% vs 40.6%), so clean CE overlap
(25.1%) sits below the all-projects CE overlap (29.5%).

---

## 2. Phase 2 vs Phase 1 complete-timeline share

### ⚠️ The two published figures are NOT the same universe

- **Phase 1** (`phase1/output/deliverable3/figures/03_complete_timeline_share_boxplot.png`):
  denominators 19,399 + 573 + 753 = **20,725 = clean-energy projects only**.
- **Phase 2** (`phase2/output/deliverable04/figures/fig_d4_complete_timeline_share_boxplot.png`):
  denominators 52,093 + 2,992 + 4,130 = **59,215 = ALL projects** (clean + fossil + other).

Comparing the two figures directly is misleading. Two comparisons below.

### Table 1 — As published (what each figure actually shows)

| Review type | Phase 1 (clean only) | Phase 2 (all projects) | Δ pts |
|-------------|---------------------:|-----------------------:|------:|
| CE          | 5,899/19,399 (30%)   | 15,348/52,093 (29%)    | −1    |
| EA          | 355/573 (62%)        | 1,434/2,992 (48%)      | −14   |
| EIS         | 362/753 (48%)        | 842/4,130 (20%)        | −28   |

> Not apples-to-apples — Phase 2 dilutes clean projects with fossil/other and uses a
> much larger universe. Use Table 2 for the real trend.

### Table 2 — Apples-to-apples (clean-energy only, both phases)

| Review type | Phase 1 clean | Phase 2 clean | Δ pts |
|-------------|--------------:|--------------:|------:|
| CE          | 5,899/19,399 (30.4%) | 4,519/18,039 (25.1%) | −5.3 |
| EA          | 355/573 (62.0%)      | 188/557 (33.8%)      | −28.2 |
| EIS         | 362/753 (48.1%)      | 213/753 (28.3%)      | −19.8 |

**On a like-for-like clean-energy basis, Phase 2 completion is *lower* than Phase 1
across all three review types.** This is the opposite of the impression Table 1's
CE column gives, and it is the headline thing to investigate.

### Universe drift in the clean subset

- Clean projects in `projects_combined.parquet`: **20,725**.
- Clean projects present in the current Phase 2 timeline output: **19,349** (1,376 missing).
- Per-type clean counts: CE 19,399→18,039, EA 573→557, EIS **753→753 (identical)**.
- The 1,376 dropped clean projects are almost entirely CE.

So part of the apparent CE change is a shrinking clean CE universe, but EA (−28pts) and
EIS (−20pts) completion drops are real coverage regressions, not universe artifacts
(EIS denominator is literally unchanged at 753).

---

## Q4 — Are Phase 1 clean reviews present in Phase 2? (set membership)

Comparison is at the **project_id** level. Phase 1 clean review set = the union of the
three Phase 1 timeline source files (`projects_timeline_bert.parquet` 19,399 CE +
`projects_timeline_bert_ea_llm.parquet` 573 EA + `projects_timeline_bert_eis_llm.parquet`
753 EIS = **20,725**). Phase 2 set = `timeline_project_dates.parquet` (59,215) and its
candidate pool `timeline_candidates.parquet` (58,551 projects with ≥1 candidate).

### Direction that matters: Phase 1 → Phase 2

| Metric | Count | Share of P1 clean |
|--------|------:|------------------:|
| Phase 1 clean reviews (universe) | 20,725 | 100% |
| …also present in Phase 2 output  | **19,349** | **93.4%** |
| …**missing** from Phase 2 output | **1,376** | **6.6%** |
| …with ≥1 candidate pulled in Phase 2 | 19,290 | 93.1% |
| …with **zero candidates** pulled in Phase 2 | 1,435 | 6.9% |

**The 1,376 missing reviews were never ingested — all 1,376 have ZERO candidates in
Phase 2.** They were not lost at adjudication (a 06_-script problem); they were never
pulled at the candidate stage (a 03_/scan problem). Breakdown of the missing:

| Phase-1 process | Missing from Phase 2 |
|-----------------|---------------------:|
| CE  | 1,360 |
| EA  | 16 |
| EIS | 0 |

(The 1,435 zero-candidate total = the 1,376 never-ingested + 59 that got a row in Phase 2
but still have no candidate, i.e. register/proxy-only dates.)

### Reverse direction (your stated worry): Phase 2 → Phase 1

> **Phase 2 clean reviews not in Phase 1 = 0.**

Every one of the 19,349 clean reviews in Phase 2 was already in the Phase 1 set. Phase 2
clean is a **strict subset** of Phase 1 clean. So the concern as phrased — "clean projects
in Phase 2 that are not in Phase 1" — does not occur. The actual asymmetry is the opposite:
Phase 2 is *missing* 1,376 clean reviews (almost all CE) that Phase 1 had. This is consistent
with your read that Phase 2 has **not pulled as many candidates** — it just shows up as
Phase-1-only projects, not Phase-2-only.

**Bottom line:** the LLM adjudication (06_) can only make Phase 2 more robust on the 19,349
reviews it already covers; it cannot recover the 1,376 reviews that have no candidates. Those
need an upstream re-scan to reach parity with Phase 1.

### Side finding — EIS candidate gap (feeds the checklist)

Zero-candidate projects in the Phase 2 output, by process type:

| Process | Zero candidates (all) | Zero candidates (clean) |
|---------|----------------------:|------------------------:|
| CE  | 0 / 52,093 (0.0%) | 0 / 18,039 (0.0%) |
| EA  | 0 / 2,992 (0.0%)  | 0 / 557 (0.0%) |
| EIS | **664 / 4,130 (16.1%)** | 59 / 753 (7.8%) |

EIS is the only process with a candidate hole inside the Phase 2 output: **664 EIS projects
(59 of them clean) have no candidates at all.** This is the "~600 EIS with none" on the
checklist (precise: 664 all / 59 clean).

### Missing reviews vs NEPATEC 2.0 scale

The 1,376 missing clean reviews are a small slice of the full NEPATEC 2.0 project universe
(project counts: CE 54,668 + EA 3,083 + EIS 4,130 = 61,881 ≈ the paper's ">60,000 projects").

| Process | Missing from Phase 2 (vs Phase 1 clean) | NEPATEC 2.0 total projects | Missing as % of NEPATEC |
|---------|----------------------------------------:|---------------------------:|------------------------:|
| CE      | 1,360 | 54,668 | 2.5% |
| EA      | 16    | 3,083  | 0.5% |
| EIS     | 0     | 4,130  | 0.0% |
| **Total** | **1,376** | **61,881** | **2.2%** |

Diagnostic profile of the 1,376 missing (from `timeline_document_index.parquet`): **all
1,376 are in the Phase 2 document index** (ingested/scanned) but produced **zero candidates**
— so the failure is at candidate extraction (03_), not ingestion. They are short documents
(median **2 pages** vs 4 in the kept cohort), 96% flagged main-document, overwhelmingly
**`decision`-category CE determinations** (1,325/1,430 doc rows), and dominated by **DOE
(1,285) and BLM (73)** lead agencies. This matches the known CE "template-specific form
layout" coverage gap. Full list: [`missing.csv`](missing.csv). Investigation brief:
[`missing_investigation_prompt.md`](missing_investigation_prompt.md).

---

## Next-week checklist

- [ ] **Investigate EA completion drop** (clean-only EA 62%→34%; check `where_I_left_off.md`).
      Confirm whether the recent EA decision recovery (67%→74%) landed mostly on *non-clean*
      EAs, leaving clean EAs behind.
- [ ] **Recapture candidates for the 664 EIS projects with none** (59 clean / 605 non-clean).
      These have a Phase 2 row but zero candidates → re-scan upstream (03_).
- [ ] **Recover the 1,376 Phase-1 clean reviews missing from Phase 2** (1,360 CE + 16 EA, 0 EIS).
      Root cause confirmed (truncation + exclusion-keyword rejection on CE forms). Code fixes
      **applied** in `acdd7ba` (2026-06-10) — see `where_I_left_off.md §Missing-reviews investigation`.
      **Remaining:** run the isolated validation recipe (`missing_investigation_CEplan.md §6`) on
      `missing_ce_ids.txt` / `missing_ea_ids.txt`, then re-run full CE pipeline if validation passes.
- [ ] **Run the 06_ adjudication pass** to fill out Phase 2 on the 19,349 covered reviews,
      then re-cut Q2/Q3 tables to measure the lift over regex-only completion.
- [ ] **Decide the D4 reporting universe** — re-cut `fig_d4_complete_timeline_share_boxplot.png`
      to clean-only so it is comparable to Phase 1 and the rest of the project (current figure
      is all-projects; see Q3 caveat).
- [ ] **Target initiation extraction, not decision** — overlap ≈ initiation rate for CE/EIS;
      initiation is the binding constraint on completion (Q2).

---

## Duration outliers (post-LLM run, 2026-06-17)

Implausibly long init→decision spans (> 5,000 days ≈ 13.7 yr), `complete_clear` only:
**CE 27 · EA 4 · EIS 14.** Produced reproducibly by `code/deliverable04/10_outliers.R` →
`output/deliverable04/diagnostics/d4_duration_outliers.csv` (+ a client-facing EA/EIS subset).

These are a **mix** of two things, separable only by reading the evidence text (not by a year
cutoff):
- **Genuinely long NEPA processes** — e.g., SunZia Southwest Transmission (14.6 yr; ROW
  application Sep 2008 → ROD Apr 2023), Energia Sierra Juarez, Grain Belt Express, Cushman
  Hydroelectric. These are the client-investigable "where it went wrong" cases.
- **Extraction errors** where the "initiation" is a *different action's* date than the decision:
  a license-renewal application (Palisades), a RCRA permit (NRDWL landfill), a state-PUC filing,
  a park authorization, or a prior plan. Pattern = cross-action contamination on one `project_id`.

Related data-quality findings from the same pass:
- **~223 CE `complete_with_proxy` rows have NEGATIVE durations** (decision before initiation) yet
  are *not* flagged `invalid_order` — a proxy-completion ordering bug to fix.
- **Duplicate `project_id`s** for some long EIS (West Mojave ×2, Clearwater ×2, Roan Plateau ×2).

---

## Open questions / things to chase

1. **Why is clean-only completion lower in Phase 2 than Phase 1** for every review type,
   despite the recent EA decision-coverage recovery work (git: EA 67%→74%, 1,371→1,434)?
   The EA recovery shows up in *all-projects* but clean-only EA is only 33.8%. Are the
   recovered EA decisions disproportionately non-clean projects?
2. **1,376 clean projects missing** — **[where-did-they-go RESOLVED 2026-06-10]** root cause is
   candidate-stage extraction: CE forms (1,360) were `priority_3`, so `build_tier_d_packets`
   truncated each page to 2,000 chars and cut off the bottom-of-form signature date; a secondary
   loss came from whole-block exclusion-keyword rejection. EA (16) is a mix of truncation, image-only
   PDFs, and cases where Phase 1's date was wrong. Fixes applied in `acdd7ba`
   (see `where_I_left_off.md §Missing-reviews investigation`,
   [`missing_investigation_findings.md`](missing_investigation_findings.md)). **Still open:** whether
   to reconcile the D4 reporting universe back to the Phase 1 clean set of 20,725.
3. **Initiation is the binding constraint** on completion (overlap ≈ initiation rate for
   CE/EIS). Coverage gains should target initiation extraction, not decision.
4. **Decide the reporting universe for D4.** If the deliverable is about clean energy,
   the Phase 2 figure should likely be re-cut to clean-only to stay comparable to Phase 1
   and to the rest of the project.
