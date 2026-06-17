# D4 Timeline — Coverage Constraints by Review Type

Why the full-timeline overlap (initiation **and** decision present) is low, per review type.
Compiled 2026-06-15 from the Phase-1-vs-Phase-2 candidate comparison and the EA/EIS/CE diagnostics.

## Where the no-dates actually come from — empirical decomposition (2026-06-16)

Don't just benchmark to Phase 1 — this is the *source* of the no-date rates, measured directly.
The "high no-date" rate is **not the same story** for EA and EIS.

**EIS (77% incomplete): largely a truncation/retrieval BUG, not a source limit.**
- 4-way split: complete 23% · decision-only 30% · init-only 11% · **neither 36%**.
- Of the 1,468 "neither" (no date at all): **664 have ZERO candidates**, but they **have indexed pages,
  were scanned (none deferred), and 1,295/1,344 of their documents have extractable text** — so the
  dates *are in the documents*. The old **2,000-char truncation cut them off** and retrieval didn't pull
  the date pages. **This is the surprising part: real EIS docs containing NOI/ROD/FEIS dates, truncated
  away.** Fixable by the 12k-cap + FEIS-full-read + text-fallback changes (applied by the full run).
- The other 804 "neither" have candidates but nothing selected → selection/scoring.
- So EIS no-dates ≈ **half retrieval/truncation (recoverable, coded)** + half selection.

**EA (51% incomplete): NOT a document/extraction failure — selection + init source sparsity.**
- 4-way split: complete 49% · decision-only 25% · init-only 9% · **neither 17%**.
- Of the 516 "neither": **all have candidates** (357 a decision candidate, 322 an init candidate, 336
  even a ROD/FEIS doc) — **zero are zero-candidate.** The dates were extracted but **not selected** —
  the candidates are weak and don't pass the register/FONSI-anchored EA decision tiers or the init
  thresholds.
- The 25% decision-only is the **init source gap**: no register/NOI init (EAs don't publish an NOI),
  faint document init signal; often the register's start date *equals* the decision date (no span).
- So EA no-dates ≈ **weak-candidate selection + genuinely thin initiation signal.**

**Takeaway:** EIS has lots of *recoverable* dates (truncation + role-gating + cues); EA is the
genuinely harder one (faint/absent initiation, weak candidates), only partly recoverable.

## Complete / LLM-recoverable / structural decomposition (2026-06-16, post Variant-B + sliver)

Every project falls in exactly one of three buckets. **Complete** = already has both dates
(register / regex / proxy). **Send to LLM** = currently missing ≥1 slot but *can* complete — each
missing slot has a candidate to pick from (the 06 send-set). **Structural** = missing a slot with
**no candidate** for it — unrecoverable by any method (the date isn't in the corpus).

| process | already complete | + send to LLM (recoverable) | + structural (unrecoverable) | total |
|---------|-----------------:|----------------------------:|-----------------------------:|------:|
| **CE**  | 24,741           | 8,625                       | 20,674                       | 54,040 |
| **EA**  | 1,534            | 901                         | 582                          | 3,017 |
| **EIS** | 1,011            | 1,681                       | 1,438                        | 4,130 |
| **all** | 27,286           | 11,207                      | 22,694                       | 61,187 |

Notes for the report/architecture:
- **CE & EA** are mostly *already complete* (decisions well-covered by register/FONSI); the LLM adds
  a modest increment. CE's large structural bucket = missing-init CEs with no init candidate.
- **EIS is the exception**: "send" (1,681) > "already complete" (1,011) — EIS decisions are
  ROD-sparse, so the LLM is where most EIS completion comes from. Includes the month-sliver.
- The LLM send-set is gated to *completable* projects only; structurally-missing slots are not sent
  (no candidate to choose), which is why the send-set (11,207) ≠ the broader incomplete count.

## Code-review findings (2026-06-16, pre-overnight-run)

**(0) All current parquets are PRE-FIX — the headline caveat.** The candidates were last
extracted 2026-06-09/10 (zero instances of any new cue: `scoping_noi_init`,
`application_prefiling_init`, `applied_for_application`, `ce_inferred_application`) and the
retrieval packets are all from a single 2026-05-30 run (no `eis_text_fallback` tier present). So
the 664 zero-candidate EIS / 516 EA-neither figures are **pre-fix ceilings measured on the OLD
pipeline** (2k truncation, no EIS fallback, no calibrated init). The committed June-15/16 fixes are
NOT yet in the data — the overnight run is what tests their recovery. Do not quote current
coverage as the post-fix result.

**(1) FOUND + FIXED — stale FR NOI API source leaking into selection.** `noi_publication_date`
comes from the Phase-1 Federal Register NOI **API match** (00_sample renames it
`fr_noi_publication_date`; `noi_match_status` has only **93 "accepted"** EIS matches, just **6** of
which agree with a BLM/DOE register date). It fed a Tier-A initiation candidate
(`candidate_source_type="noi_notice"`, 94 rows) and was selected for **12 EIS initiations**. This
source is unreliable. **Disabled** via `FR_NOI_TIER_A_ENABLED = False` in `02_retrieve.py`
(reversible flag). Footprint: 11 of the 12 have a non-NOI backup candidate → ~1 net EIS init lost.
**BLM/DOE register dates (`candidate_source_type="metadata"`) are GOLD and untouched** — they come
from separate Tier-A paths and remain the authoritative initiations in `_calibrated_init_eligible`.

**(2) Verified clean:** run order (`run_pipeline.py` runs 02→03→04→04b→05b→05→05c, preventing a
repeat of the stale-`ranking_score` CE loss); the new init cues (correctly anchored to the date
clause, never steal a decision role, `_ms`/`_me`/`block` in scope); calibrated-init eligibility
(strictly additive union, OMB boilerplate excluded, per-process duration guard).

## CE — the constraint was *initiation* candidates (now fixed)

- Decision candidates were never the problem (Phase 2 had **98%** of Phase 1's).
- Initiation was: Phase 2 had only **45%** of Phase 1's CE init candidates. Phase 1 derived CE
  init from `bert_inferred_application_date` = the application date if found, **else the earliest
  dated mention** — an *inference* Phase 2 didn't replicate.
- **Fixed (2026-06-15):** inferred-init proxy (Fix 1, earliest candidate date < decision, 5y cap,
  flagged `ce_inferred_application`) + the "applied for" application cue (Fix B). CE Decarb now
  *exceeds* Phase 1 (38.5% w/ proxy vs 30.4%).

## EA — the constraint is *initiation*, mostly a *source* limit

- EA decision is fine (~74%, register-backed).
- EA init is **register-dependent**: 67% of selected EA init dates come from the BLM/DOE register.
  The uncovered cohort (~476 projects with a decision but no init) has register init for only
  **11 of them, and zero NOIs** — EAs are not required to publish a Federal Register Notice of
  Intent (that's an EIS requirement).
- Their document-text init candidates are **weak**: 84% have best `p_init_cal` < 0.1. EA
  "initiation" is fuzzy in the text (informal scoping, application receipt), rarely a clean date.
- Net: **source-limited** (register lacks it + text doesn't state it cleanly). Partly recoverable
  by the classifier retrain (Tier 2) for candidates that *do* exist, but capped by source.

## EIS — *both* endpoints are constrained

- **Decision (bigger gap):** Phase 2 has only **64%** of Phase 1's EIS decision candidates. Of the
  205 missing: **~117 are truncated-away ROD/FEIS dates** (date sat past the old 2k-char cut) →
  **recoverable by the retrieval fix** (12k cap + FEIS full-read, applied by the full run). The
  other **88 have no decision doc at all**.
- **Initiation:** candidates mostly exist (85% of Phase 1's) but the **classifier scores half
  <0.1** (the bimodal problem) → a selection/classifier issue, not extraction.

## "Phase 1 pulled dates from other documents" (tested, marginal)

Of EIS's 88 no-decision-doc projects, **~36** came from Phase 1 reading the ROD/FEIS date
*mentioned in narrative* inside a non-decision document; ~52 from looser/incidental dates (filing
dates, consistency determinations, citations). A cue to replicate this (`Record of Decision …
signed/issued`) was prototyped: ~22 tight hits at only ~60–70% precision, leaking historical
citations, programmatic references, and *anticipated* future dates. That is *why* Phase 1 was less
accurate — those narrative mentions are often references to **other/prior** RODs. **Marginal,
noisy — not pursued.**

## What can still move the needle (real, unbanked)

1. **EIS decision — the overnight retrieval run.** Biggest unbanked win: ~117 truncated EIS
   decisions recovered by the 12k-cap / FEIS full-read code (written, not yet run).
2. **EA + EIS initiation — Tier 2 classifier retrain.** Recovers the bimodal-low init candidates
   that already exist in the pool (the lever for EA within its source cap, and EIS init).

## Genuinely capped / not worth chasing

- EA init beyond its source limit (no register/NOI, weak text).
- The 52 "looser" EIS decisions (Phase 1 noise Phase 2 correctly declines).
- The narrative-ROD cue (noisy).
- Image-only PDFs (OCR — out of scope).
