# D4 (Timeline) — Deferred Items

**As of 2026-07-15**, for the Thursday CATF presentation. Everything under "Fixed in this pass" is committed and merged to `desktop` (four commits, local / unpushed), and the corrected data is propagated to main — so main and `desktop` reproduce the committed report.

This is the internal record of what was consciously **deferred** in D4 and why. The reader-facing subset (the ones that bear on interpreting the numbers) is also summarized in the report's *Data Quality & Caveats → "Known issues, understood and deferred"* section. Nothing below changes the headline medians, which are robust to these tails.

## Fixed in this pass (committed to `desktop`)
- **CE register-vs-document decisions** — prefer the authoritative BLM/DOE register when a document-text CE decision disagrees by >~2 years (~650 CE decisions corrected). `55422f9`.
- **Negative-duration ordering** — reclassified to `invalid_order` at source (05 / 05c normalizer) with an assertion in 08; the old downstream stopgap was removed. `55422f9`.
- **06 LLM cache-replay** — on a cache hit, script 06 now re-applies the stored adjudication from `timeline_api_adjudications.parquet` (no API call), so a regenerated `project_dates` restores the full LLM layer deterministically for $0. `55422f9`.
- **Fossil-EA reporting** — corrected the "84% = well-abandonment" error (84% is the *register-anchoring* share; well-abandonment is 53%), and added a second "Fossil (doc-anchored)" row to Figure 6 (~5-month median) as the more defensible review-length view. `eb01523`.
- **Data-quality caveats + this note** — split the report's caveats into "resolved this pass" vs "deferred" and added this deferred-items note. `8e76622`.
- **EIS methodological note** — the Phase-1-gap / FEIS-proxy explanation at the bottom of this file. `9b911bf`.

## Deferred — documented, not in active development

None of the items below are being actively worked. They are recorded so the underlying issue is understood and can be picked up later if a future pass warrants it — not because any of them is queued or in progress.

- **EIS decision-coverage gap (biggest open issue; investigated and closed 2026-07-14).** EIS complete coverage is ~32%, below Phase 1's ~48% for decarb EIS. This is **not** in active development. We *did* build and test a targeted selection-only fix in an isolated worktree (route month-granularity dates on ROD/FEIS documents to the LLM; widen the FEIS cover-date regex, both in script 05); it recovered only **~57 projects (~+1.5 pts)**, which confirmed the real bottleneck is **upstream candidate generation, not selection**. The worktree was discarded — nothing was promoted and `desktop`/main were untouched. The full explanation of *why* EIS is lower than Phase 1 and why the recovery is hard is in the **methodological note at the bottom of this file**; a real fix would be a heavy Phase-2.1 re-extraction, only if someone decides to invest in it.

### Lower-priority / diagnostic (no deliverable-number impact)
- **#8 VALIDATION-GOLD (medium).** Build a disjoint, held-out project-level gold set (~180 projects, hand-verified from source docs) to produce an honest end-to-end precision/recall. Deferred because it is effortful (labeling) and **adds an accuracy statistic without changing any reported number**. The report carries the honest caveat ("end-to-end accuracy not yet formally validated against a held-out gold set"). Current project-level gold (`timeline_gold_projects.parquet`) is ~76% contaminated (overlaps ranker-train + 05c injection), so it is not a clean accuracy measure — hence the caveat rather than a stated number.
- **#10 YEAR-ANCHOR (medium).** Add a `nepa_case_year` sanity-anchor flag to selection/QA (flag/down-rank a selected date that sits >~2 years from the case-number year, e.g. `DOI-BLM-ID-B010-2022-0032` → 2022). Deferred: it is a **diagnostic flag** that does not change the selected dates; requires a case-year-derivation step first; and the main thing it would catch (historical-citation initiations) is already documented below.
- **#15 GOLD-CLEANUP (low).** Delete the retired candidate-level gold apparatus (`labeling/` gold-builder scripts + `data/analysis/timeline/gold/` + `output/deliverable04/gold/` candidate-level artifacts). **KEEP** the project-level `timeline_gold_projects.parquet` used by `07_validate.py`. Pure housekeeping — deferred, and to be done carefully (confirm the exact file list first) if/when #8 is picked up.
- **CE long-duration outliers (historical-citation initiations).** ~106 CE reviews (0.4%) show 13–45-year "durations" because the extractor selected a decades-old citation (original grant, covering PEIS, prior plan) as the initiation instead of the recent application/renewal date. The candidate-level `historical_gap_candidate` flag already exists but is under-weighted in ranking (plus a ~9-row bug where LLM adjudication overrode a correct ranker date). The CE median is 20 days with or without these rows. A fix (stronger `historical_gap_candidate` penalty in 05) is deferred because it is a **global CE-ranking change** — it would require a full re-run + verification for a 0.4%, no-headline-impact gain. These rows are flagged `suspect_error` in `10_outliers.R`.
- **Smaller register/document CE decision disagreements.** Sub-2-year disagreements are left to the ranker (the >2-year, decade-scale ones were fixed this pass).
- **Post-FRA CE durations.** Post-FRA CE runs modestly longer than pre-FRA (median ~37 vs ~19 days; ~1.9× shrinking to ~1.3× once proxy rows are excluded). The residual is spread across date sources, not a single bug. NOTE: the old "43 vs 1,005 days, n≈208" figure was from a stale pre-fix build (10× less coverage) and does not reproduce on current data.

## Deleted (not doing)
- **#9 CALIBRATION-SPLIT.** 04b calibrates the classifier confidence in-sample (on the frozen 938-row test split). A dedicated out-of-sample calibration split was **deleted from scope** — it affects only internal confidence scores (not the selected dates), moves report numbers, breaks comparability, and can trigger LLM re-routing. Revisit only if calibration is specifically challenged.
- **#13 BUCKET3-INIT.** ~959 projects produced no candidates at retrieval (scanned/corrupt PDFs, no extractable text). **Deleted from scope** — not recoverable without OCR reprocessing; ~1.5% of the universe.

---

## Methodological note: why EIS decision coverage is lower than Phase 1, and why the EIS dates are a best guess

Phase 1 reported ~48% complete timelines for **decarbonization EIS**; Phase 2 reaches **~32%** for **all-energy EIS** on a roughly 5× larger universe. This is a genuine difference, but it is explained by scope plus a structural data limitation — **not** a pipeline regression — and we confirmed the cause empirically (2026-07-14) before accepting it.

**The structural cause.** An EIS "decision" is a **Record of Decision (ROD)**, a *separate document* that in NEPATEC 2.0 is usually absent — only ~18% of EIS have a ROD in the corpus. Where there is no ROD, the pipeline falls back to the **Final-EIS publication date** as a **month-granularity proxy** for the decision (imputed to the 15th of the month). So a large share of the EIS decision dates we *do* report are **best-guess proxies** (FEIS publication ≈ decision), not authoritative ROD signature dates. That is why EIS coverage is both lower and lower-confidence than CE or EA.

**Why the gap can't be cheaply closed (tested and abandoned 2026-07-14).** We investigated the 2,388 EIS that lack a decision — 836 of which have a ROD/FEIS document in the corpus — and found the bottleneck is **candidate generation, not selection**:

- **33% (275 of 836) have no decision-date candidate extracted at all.** The ROD/FEIS cover date was never captured by retrieval/extraction, so there is nothing for the selection step to pick.
- **~60% were already sent to the LLM, which returned no usable decision.** Typically a month-only cover date ("June 2015") is extracted with the identifying "Final Environmental Impact Statement" title on a *different* page, so the date ranks below the top-3 candidate packets and the model never sees it.
- A **targeted selection-only fix** (routing month-on-ROD/FEIS-document dates to the LLM; widening the FEIS cover-date regex) recovered only **~57 projects (~+1.5 pts)** — confirming selection is not the wall.

**What a real fix would take (Phase 2.1).** Re-pull the FEIS/ROD cover pages to *generate* candidates for the ~275 with none, and improve how month cover dates are ranked into the LLM packets. That is a heavy re-extraction over the 5.5 GB EIS page corpus plus additional LLM spend — out of scope for this deliverable.

**Bottom line.** Present EIS coverage of ~32% as-is, with the caveat that a large share of EIS decisions are **Final-EIS-publication proxies (month-granularity)**, and that the shortfall versus Phase 1 is a **corpus / ROD-availability limitation, not an extraction error**. The EIS timelines are the best available estimate given what NEPATEC 2.0 contains.
