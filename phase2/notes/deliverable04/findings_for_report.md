# D4 Timeline — Findings for the report & Thursday presentation

Canonical, verified findings from the 2026-06-16 coverage/quality analysis. This is the durable
talking-points doc: recount from here for the presentation, and fold into `reports/deliverable04.qmd`
when it is built. All numbers measured from tonight's run (current-model) vs the prenight backup and
the Phase-1 regex candidates.

## Headline: main reason for low complete-coverage, per review process (CLIENT-READY)

| process | the cap is on… | one-line reason |
|---|---|---|
| **CE** | initiation | Most CEs are fast determinations with **no initiation date recorded** — structural. |
| **EA** | initiation | EAs have **no Notice-of-Intent requirement** and often skip scoping, so a start date is rarely documented; it exists only when the BLM/DOE register captured it — structural. |
| **EIS** | **decision** | EIS *initiations* are usually documented (an NOI is required), but the **ROD (the decision) is frequently a separate document not in the provided files**, so the FEIS publication date is used as the decision; plus some EIS "documents" are comment letters/fragments with no milestone. |

**So: CE and EA are initiation-limited (structural); EIS is decision-document-limited.** Three
different, defensible root causes — none is "the pipeline is broken."

## Additional findings (2026-06-17, post Variant-B + month-sliver)

**1. EA timelines are short because of a register-anchor artifact, NOT EA→EIS escalation (KEY).**
All **1,196** EA projects resolved via the BLM/DOE register end in a **FONSI** (Finding of No
Significant Impact) — zero non-FONSI. A FONSI means "no EIS needed," so these EAs did **not** escalate
to a full EIS (escalation produces a notice-of-intent, not a FONSI). The short EA median (105 d) is
driven by the register "project start date" being a **late/administrative anchor**: register-init→
FONSI span ≈ **60 d** median vs **364 d** for document-based inits. *Caveat to state:* register-based
EA durations understate the true process length; document-init EAs (~12 mo) are closer.

**2. The deterministically-resolved (non-LLM) dates are mostly sound.** Audit of resolved projects
not sent to the LLM (`code/deliverable04/_audit_resolved.py`): non-register picks are largely correct
— many CE `document_text` decisions are the *same date* as the DOE/BLM register (the form's
determination date appears in both), or legitimate signature/approval dates; competitors are usually
the same date from another source. A minority are marginal (an init labeled from a "determined"
context). No systematic precision problem.

**3. Complete / LLM-recoverable / structural decomposition** (full table in `coverage_constraints.md`):
already-complete CE 24,741 / EA 1,534 / EIS 1,011; LLM-recoverable (sent to 06) CE 8,625 / EA 901 /
EIS 1,681 (**11,207** total); structural (no candidate, unrecoverable) CE 20,674 / EA 582 / EIS 1,438.

**4. Clean-energy (decarb) subset:** **18.2%** of EIS are decarb (753 of 4,130). Honest complete-
timeline estimate after the LLM ≈ **CE 49% · EA 54% · EIS 35%** (vs pre-LLM floors CE 45.8% · EA
50.8% · EIS 24.5%). EIS is the process where the LLM matters most (send-set > already-complete).

**5. ~85% of EIS *decisions* are month-granularity** (FEIS-publication date, stored as the 15th),
not day-precise RODs — because RODs are frequently a separate document not in the corpus.

## Supporting evidence (verified this session)

**1. Candidate extraction is in good shape; P2 candidates are *cleaner* than P1.**
- Totals comparable: P2 ~689k candidates vs P1 ~645k. Per-project density: CE 7.4 (P1 4.0, P2 richer),
  EA 14 (P1 177), EIS 66 (P1 646).
- The P1 high density is **mostly noise** — spot-check showed page-header cover months
  ("3-268 December 2010"), citations ("Accessed July 4, 2013"), figure dates, EO citations. P2's
  targeted retrieval extracts fewer but cleaner candidates. Lower density ≠ lost signal.

**2. Zero-candidate projects: only EIS, and they are correctly excluded.**
- CE 0, EA 0, **EIS 403 (9.8%)** — down from 664 prenight after the full-read retrieval fix.
- Of the 403: 392 have extractable text, only 16 are image-only. Reading them: they are
  **non-milestone documents** — EPA/agency **comment letters**, draft-review correspondence, short
  fragments, and a few OCR-garbled old scans. A comment letter's date is not a NEPA milestone, so the
  workflow is correctly extracting nothing.

**3. The candidates exist — the gap is selection, not extraction.**
- % of projects with **both** an initiation and a decision *candidate*: CE 42%, **EA 80%, EIS 63%** —
  the EA/EIS ceilings **exceed** Phase 1's achieved complete rates (62% / 48%).
- But we currently *select* both for only CE 49% / EA 51% / EIS 34%. The ~30-point EA/EIS gap is
  candidates we extracted but didn't confidently select → the LLM adjudication step (06) resolves these.

**4. EA initiation is structurally missing (audit of 100 EA decision-only projects, full-text scan).**
- **82 of 100 have NO initiation signal anywhere** (no application-received, scoping, NOI, or
  pre-filing date). Of the 18 with a signal, several aren't real NEPA inits ("external scoping was
  *deemed unnecessary*", internal meetings, *water-permit* applications). ~8–10/100 truly recoverable.
- Mechanism: no NOI requirement + scoping often skipped + start date only in the BLM/DOE register.

**5. EIS decision uses a FEIS-publication-date fallback (methodology note — state this plainly).**
- Only a minority of EIS projects have a ROD in the provided files. The pipeline searches **ROD-first**
  (register ROD → ROD-typed document → ROD signed/mentioned in narrative *including inside the FEIS*),
  and **only when no ROD exists anywhere** falls back to the **Final-EIS publication date** as the
  decision, clearly flagged (`decision_is_feis_fallback` / `feis_publication`). Intentional and disclosed.
- **Granularity caveat (say this):** because of source-data limits, the FEIS publication date is in
  **month+year granularity in some cases** (the title-page month, e.g., "Final EIS — June 2015"), not
  a day. Durations using these are month-precision.
- **KNOWN GAP / TODO (revisit after Thursday):** some FEIS publication cover dates were **skipped at
  retrieval** (the cover page wasn't pulled), so a subset of FEIS-fallback decisions are missing a
  date they should have. **We need to re-pull FEIS publication/cover dates** (target the document
  title-page month / filename metadata date) to lift EIS decision coverage toward its ~62% ceiling.
  Tracked in the to-do list.

## Why initiation is selected more reliably than the decision (1-liner for the deck)

Initiation is a single, distinctive event (one NOI / application / scoping date), so the ranking
model isolates it cleanly; a decision is a cluster of look-alike dates — a document's cover month vs.
its signature date, plus citations to *other* RODs — so the model must disambiguate the operative
decision from several decoys, which is why decision selection needs more candidates + authoritative-
source rules.

## Numbers to quote (FINAL — post-LLM adjudication run, 2026-06-17)

- **Complete-timeline coverage (post-LLM, all projects): CE 52.6% · EA 57.6% · EIS 32.3%.**
  Gains over the pre-LLM deterministic pipeline: CE +7.0pp (45.6→52.6, +3,759 projects) · EA +6.8pp
  (50.8→57.6, +203) · EIS +7.9pp (24.4→32.3, +326). CE/EA landed in/near projection (CE ~53–58%,
  EA ~56–62%); **EIS came in below the optimistic ~43–50% projection** — it is decision-document-
  limited (RODs frequently absent; EIS decision coverage only 42%), and the FEIS cover-date re-pull
  that would lift it further is deferred (todo).
- **Clean-energy (Decarb) complete coverage (post-LLM): CE 48.9% · EA 48.5% · EIS 42.4%.** Note EIS
  Decarb (42.4%) is *well above* EIS overall (32.3%) — clean-energy EISs are better-documented than
  the "Other" bucket that drags the EIS aggregate down.
- **"No date at all" (missing_both), population share: CE 3.6% · EA 10.0% · EIS 31.3%.** EIS is the
  structural gap (RODs absent + comment-letter/fragment documents with no extractable milestone).
- Sub-coverage: initiation CE 60% · EA 68% · EIS 62%; decision CE 91% · EA 81% · **EIS 42%** (the
  decision is the EIS bottleneck, not the initiation).
- **LLM adjudication cost (actual): $18.20** for the full 11,207-project send-set (Haiku 4.5,
  workers=24, Tier 2, 0 errors, 0 non-haiku rows). ≈ $0.0016/project.
- Headline durations (complete_clear only): CE 21 d (0.7 mo, n=9,272) · EA 86 d (2.8 mo, n=1,355) ·
  EIS 1,077 d (35.4 mo, n=213).

## Concrete examples to cite (verified — pull 1–2 into the report)

**Zero-candidate EIS — non-milestone documents (correctly extract nothing).** These "EIS projects"
are represented in the corpus only by comment letters / draft-review correspondence / fragments, so
there is no NEPA milestone date to pull. *Illustrative read:* an EPA Region 2 letter reviewing the
NY Route 17 DEIS/FEIS (`3546.pdf`) — its date is a *comment* date, not a milestone. 10 examples:
```
ca6e19ce10cb1a90147ef3ea1a7b2edf  67feb0d30d62fc8b581023243a9a6d87
6ded7a251038936ee753adc06b770888  8987cf33dc8ea8be2c02bd38bd0caf91
ec22620fe3b06646ec8f1e763a411027  7493424e4d4f724a97312eae18c16817
0e3185d181c81e468d6e92ee785a4e69  f77173e76bb8e6cada9a9e89ddf16cef
6e0d830c383cc7c79f87263eafaaa15f  f9d19387bc8bf54fcaf9d3ffde9bfc6f
```

**EA decision-only — clear decision, no documented initiation (the structural EA init gap).** These
EAs have a clean FONSI / Decision Record (so a decision date) but **no application-received, scoping,
or NOI date** in the record. *Illustrative read:* the Pedro Hill Water Trough EA (CA-180-16-03) — a
FONSI and Decision Record both signed **June 1, 2016** (decision), but the document states no project
initiation date; it only cites *other* plans' dates (Sierra RMP 2008, Cronan Ranch Plan 2007). 10 examples:
```
fb8058571842f3c85381806751d1f418  597c4f0a1b64fe720b8d45d4f929e77d
27dd86ef5378ba8c42f1b0ffb7876eab  8e27690e4f660a47e4c892488a2ab6c5
ff76899954500f07ec4b2988863df040  f29445bad4bc11fd25d0527c50d75fce
6569c2b2f4cc8947a03102d029e78134  8fd4998c67429ef48f66826e21f9b4b3
b7df31da5fc4132231a2f1c517b73530  ccf596c08a843cc49f2f6894c1bd421c
```

## Caveats to keep saying
- "Complete coverage up" is a coverage statement; date *accuracy* is validated separately (held-out
  gold is a pending workstream). The FEIS-publication fallback and proxy dates are reported tiered.
- EA will likely land slightly *under* P1's 62% because EA initiation is structurally sparse.
