# Timeline BERT Quality Review Samples

**Purpose:** quick reference set of samples for reviewing BERT timeline performance (decision/initiation/review).  
**Last updated:** 2026-02-02

## Sample list (project IDs + notes)

### 2f27a6b9-3588-3dc6-2bdd-fde29655e2e7
- **Observed (BERT 2026-01-30 test20):**
  - 2022-11-17 labeled decision (McClellan review completed)
  - 2022-11-23 labeled decision (signature)
- **Expected behavior:**
  - 2022-11-17 should be **review** (interim step), and only used as initiation if no explicit initiation date exists.

### 3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0
- **Observed:**
  - 2021-01-05 labeled decision (“decision in principle” wording)
  - 2021-02-01 and 2021-02-09 labeled decision (signature block)
- **Expected behavior:**
  - 2021-01-05 should be **review** (interim / future actions language).
  - Missing review dates in text: 2021-01-07, 2021-01-11, 2021-01-26.
  - Context window should expand for review cues but stay tight for signature cues.

### 5418c75b-f493-d342-40c5-cd4f8acaca5a
- **Notes:** no issues recorded.

### 7ae7b22f-eaee-bdc1-f425-8dd87daaeb05
- **Observed:**
  - BERT captured only header boilerplate (Revised/Reviewed) as decision.
- **Expected behavior:**
  - Initiator and decision dates at bottom of document should be prioritized.
  - Header dates could be **review** if context supports it.
  - Footer dates should outrank header dates only when cue strengths are close and positions disagree.

### 824ba268-8ddf-a34f-f9a7-625e7727c242
- **Observed:**
  - 2011-11-30, 2012-07-16, 2012-07-19 all labeled decision.
- **Expected behavior:**
  - 2011-11-30 = **review** (Phase 1 approval).
  - 2012-07-16 = **review** (Phase 2 NEPA first review date).
  - 2012-07-19 = **decision** (signature).

### 90133a0a-6dee-eb4c-1b07-cb16ab318599
- **Notes:** no issues recorded.

### bb5f234b-21a8-9741-58f0-276c30c8fcd1
- **Notes:** no issues recorded.

### c0a5a0de-acea-0b84-f727-029167677961
- **Notes:** no issues recorded.

### f73a267f-0e1f-438d-61c9-57e17877e473
- **Observed:**
  - BERT found only an `other` boilerplate context.
- **Question:** did BERT miss a decision/initiation date? If yes, check source docs and add to review set.

## Review checklist (use when new model runs finish)
- **Decision accuracy:** signature blocks should win over headers and boilerplate.
- **Review classification:** interim approvals and phase approvals should be **review**, not decision.
- **Initiation backfill:** if initiation missing, first review date becomes initiation.
- **Context windows:** review contexts can be larger, signature contexts should stay tight.
- **Footer vs header:** use position only when strong cues are similar and positions disagree.
