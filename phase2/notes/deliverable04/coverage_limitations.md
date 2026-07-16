---
title: "D4: Timeline Coverage & Limitations"
---

*As of 2026-07-15, computed from the current pipeline output (`timeline_run_at` 2026-07-14; full-universe reconciliation applied 2026-07-15). All numbers on this page are recomputed from the current `timeline_project_dates.parquet` / `timeline_candidates.parquet` or read from the report's diagnostics (`output/deliverable04/diagnostics/`).*

This page explains **why timelines are missing where they are missing**: what coverage each review process reaches, where the gaps come from, and which gaps are structural (the date is not in the corpus) versus recoverable. Companion pages: [Date Sourcing & Provenance](date_sourcing.html) (what the dates mean and where each one comes from) and [Known Issues & Deferred Items](known_issues.html) (what is known-broken and why it was deferred).

## Headline coverage

A project's timeline is **complete** when both an initiation and a decision date are present — the definition used in the report's coverage figures.

| Review type | Projects | Has initiation | Has decision | Complete (both) |
|---|---:|---:|---:|---:|
| CE | 54,668 | 32,612 (59.7%) | 49,392 (90.3%) | **29,915 (54.7%)** |
| EA | 3,083 | 2,053 (66.6%) | 2,455 (79.6%) | **1,794 (58.2%)** |
| EIS | 4,130 | 2,571 (62.3%) | 1,743 (42.2%) | **1,478 (35.8%)** |
| All | 61,881 | 37,236 (60.2%) | 53,590 (86.6%) | **33,187 (53.6%)** |

A small share of the "both dates" rows have the decision before the initiation (`invalid_order`: CE 1,412 · EA 58 · EIS 154, mostly month-imputation artifacts). They count as covered above but are excluded from every duration figure; restricting completeness to valid-order rows gives CE 52.0% · EA 56.3% · EIS 32.1%.

Reading the table: for CE and EA the **decision** is well covered (81–91%) and the **initiation** is the scarce endpoint; EIS is the reverse — initiations are well covered (a Notice of Intent is required) but decision coverage is only 42.2%. For duration context, the headline medians on complete timelines are CE 20 days (n = 27,278), EA 116 days (n = 1,730), EIS 1,008 days (n = 1,321); duration n's are slightly below the complete counts because year-granularity endpoints are excluded from durations (see the duration-frame definition in [Date Sourcing & Provenance](date_sourcing.html)).

### The denominator: the full inventory

The table's 61,881 projects are the **complete NEPATEC 2.0 project inventory** — coverage denominators and the report's headline project count are the same universe. This holds by construction since 2026-07-15: the pipeline's universe reconciliation (`reconcile_universe` in `05_select_dates.py`, EIS-only in earlier runs) now covers all three process types, so the 628 CE and 66 EA projects whose documents yield **zero date candidates** appear in the output as `missing_both` stubs rather than silently vanishing. (Earlier published figures used the 61,187-project pipeline universe; the harmonization added only dateless stubs, so every date, duration, and median is unchanged.)

## Main reason for low coverage, per review process

| Process | The cap is on… | One-line reason |
|---|---|---|
| **CE** | initiation | Most CEs are fast determinations with **no initiation date recorded** — structural. |
| **EA** | initiation | EAs have **no Notice-of-Intent requirement** and often skip scoping, so a start date is rarely documented; it exists mainly where the BLM/DOE register captured it — structural. |
| **EIS** | **decision** | EIS *initiations* are usually documented (an NOI is required), but the **Record of Decision is frequently a separate document not in the corpus**, so the Final-EIS publication date is used as a proxy decision; some EIS "projects" are also comment letters/fragments with no milestone. |

So CE and EA are initiation-limited (structural), and EIS is decision-document-limited — three different, defensible root causes, none of which is a pipeline defect.

## Where the no-dates come from

**Projects with no date at all** (`missing_both`): CE 4.7% · EA 12.0% · EIS 31.3% of each process's population. EIS is the structural outlier.

**Zero-candidate projects** — projects whose documents produced no date candidates at all, materialized as `missing_both`: EIS 403 (9.8%; 41 of the 753 clean-energy EIS), CE 628 (1.1%), EA 66 (2.1%). Reading the EIS set, they are **non-milestone documents** — EPA/agency comment letters, draft-review correspondence, short fragments, and a few OCR-garbled scans. A comment letter's date is not a NEPA milestone, so extracting nothing is the correct behavior.

**Structural share of the remaining gaps** — a missing endpoint is *structural* when the project has no candidate of that role anywhere in its documents (nothing for selection or LLM adjudication to pick):

| Process | Missing initiation | …with no initiation-role candidate | Missing decision | …with no decision-role candidate |
|---|---:|---:|---:|---:|
| CE | 22,056 | 20,583 (93%) | 5,276 | 1,030 (20%) |
| EA | 1,030 | 489 (47%) | 628 | 233 (37%) |
| EIS | 1,559 | 941 (60%) | 2,387 | 1,283 (54%) |

(An "initiation-role candidate" is an extracted date whose role cue is a clear or proxy initiation; a "decision-role candidate" is a clear/proxy decision or the body-text decision holding category.)

The **LLM adjudication layer has already been applied**: every completable ambiguous case was sent to Claude Haiku (11,216 cached adjudications; 10,330 projects in the current output carry an LLM-sourced date; the June 2026 full run cost $18.20). The remaining gaps are therefore not "waiting on the LLM" — they are dominated by the structural buckets above.

For EA specifically, a June 2026 full-text audit of 100 EA decision-only projects found **82 had no initiation signal anywhere** (no application-received, scoping, NOI, or pre-filing date); several of the remaining 18 signals were not true NEPA initiations ("external scoping was deemed unnecessary", water-permit applications). EA initiation is genuinely sparse at the source, not under-extracted.

## The EA register-anchor artifact

54% of EA initiation dates (1,112 of 2,053) come from the BLM/DOE register rather than the EA document. Those register "project start" entries are often **late administrative entries made near the decision**, which compresses the measured span. On complete EA timelines (headline duration frame), split by `initiation_source_type`:

- Register-anchored (`metadata`): 1,105 projects, **median 60 days** initiation→decision.
- Document-anchored (`document_text`): 414 projects, **median ~405 days (~13 months)**.
- The remaining ~211 complete EAs have LLM-adjudicated or hand-verified initiations (medians ~180 days, in between).

Register-based EA durations therefore **understate the true process length** by roughly a factor of six; the document-anchored figure is the more defensible review-length view (the report's Figure 6 shows both for fossil EAs, where the artifact is strongest).

## EIS: the FEIS-publication fallback and month granularity

- Only **724 of 4,130 EIS (17.5%) have a Record of Decision in the corpus**. The pipeline searches ROD-first (register ROD → ROD-typed document → ROD language in narrative), and only when no ROD exists anywhere falls back to the **Final-EIS publication date** as the decision, flagged `decision_is_feis_fallback`.
- In the current output, **760 of the 1,743 EIS decisions (44%) are FEIS-publication fallbacks**, and 502 EIS decisions (29%) are month-granularity dates imputed to the 15th (a further 4 are year-only and excluded from durations). Durations built on them are month-precision.
- Why the gap cannot be cheaply closed (a selection-only fix was tested and abandoned) is documented in [Known Issues & Deferred Items](known_issues.html).

## Coverage in the clean-energy subset

Complete-timeline coverage for clean-energy (decarbonization) projects: **CE 49.2% (9,554/19,399) · EA 47.6% (273/573) · EIS 42.4% (319/753)**. Note that decarb EIS (42.4%) sits **well above** the EIS aggregate (35.8%) — clean-energy EISs are better documented than the heterogeneous "Other" bucket that drags the aggregate down.

## Phase 1 vs Phase 2 — the two published figures are not the same universe

- **Phase 1**'s complete-timeline figure covered **clean-energy projects only** (denominators 19,399 CE + 573 EA + 753 EIS = 20,725).
- **Phase 2**'s figure covers **the entire database** (54,668 CE + 3,083 EA + 4,130 EIS = 61,881 projects: clean + fossil + other).

Comparing the two figures directly is misleading. Two comparisons:

**As published (different universes):**

| Review type | Phase 1 (clean only) | Phase 2 (all projects) |
|---|---:|---:|
| CE | 5,899/19,399 (30%) | 29,915/54,668 (55%) |
| EA | 355/573 (62%) | 1,794/3,083 (58%) |
| EIS | 362/753 (48%) | 1,478/4,130 (36%) |

**Apples-to-apples (clean-energy only, both phases):**

| Review type | Phase 1 clean | Phase 2 clean | Δ pts |
|---|---:|---:|---:|
| CE | 5,899/19,399 (30.4%) | 9,554/19,399 (49.2%) | +18.8 |
| EA | 355/573 (62.0%) | 273/573 (47.6%) | −14.4 |
| EIS | 362/753 (48.1%) | 319/753 (42.4%) | −5.7 |

On a like-for-like clean-energy basis, **Phase 2 CE completion is far above Phase 1** (authoritative register dates plus the recovered CE-form extraction), while **EA and EIS land below their Phase 1 rates** for the structural reasons above: EA initiation is source-sparse (Phase 1's higher EA rate leaned on noisier candidate dates that Phase 2 correctly declines), and EIS is capped by ROD availability in the corpus (the EIS shortfall is a corpus limitation, not an extraction regression — see the methodological note in [Known Issues & Deferred Items](known_issues.html)).

**There is no universe drift.** Every one of Phase 1's 20,725 clean projects is present in the Phase 2 output, and the two phases' clean-energy denominators are identical. (Before the June 2026 candidate-extraction fixes, 1,376 were missing; the last 148 were zero-candidate projects, materialized as `missing_both` by the 2026-07-15 universe reconciliation.)

## Caveats to keep stating

- **Coverage is not accuracy.** "Complete" means both dates were found; end-to-end date accuracy has not yet been formally validated against a held-out gold set (see [Known Issues & Deferred Items](known_issues.html) on why the current gold set cannot serve that role). Proxy and fallback dates are flagged and reported in tiers.
- **Register-anchored EA durations understate the true process length**; prefer document-anchored views where sample size allows.
- **A large share of EIS decisions are Final-EIS-publication proxies at month granularity**; EIS durations are best-available estimates given what NEPATEC 2.0 contains.
- **Clean-energy EA completion (47.6%) lands under Phase 1's 62%** because EA initiation is structurally sparse — this is expected, not a regression.
