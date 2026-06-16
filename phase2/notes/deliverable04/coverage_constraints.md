# D4 Timeline — Coverage Constraints by Review Type

Why the full-timeline overlap (initiation **and** decision present) is low, per review type.
Compiled 2026-06-15 from the Phase-1-vs-Phase-2 candidate comparison and the EA/EIS/CE diagnostics.

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
