# Tier 4 Notes

This file holds tactical notes that are useful while working on `01_extract_nepa_trigger.py`. It is intentionally more disposable than the architecture doc. Keep it if it is helpful, or delete it later.

## Review scope and source of truth

For this refactor, the working reference files are:

- `tier4_refactor_spec.md`
- `tier4_implementation_checklist.md`
- `_legend.md`
- `_notes.md`
- `_example_bank.md`

`phase2/architecture/deliverables/deliverable01.md` is not being updated in this pass and should not be treated as the current implementation source of truth for Tier 4.


## Model selection rationale

### Tier 4: `cross-encoder/nli-MiniLM2-L6-H768`

We use a cross-encoder NLI model instead of a fine-tuned BERT classifier because **Tier 4 is a data scarcity problem, not a label quality problem**.

A fine-tuned BERT classifier requires hundreds of labeled examples per class to generalize. We do not have that for NEPA trigger classification — the classes are nuanced (`federal_funding` vs `federal_action` boundary cases) and labeling at scale is expensive.

NLI sidesteps training entirely. You feed the model two texts — a document chunk and a natural language hypothesis — and it returns an entailment probability. The model was pre-trained on millions of general NLI pairs (MNLI, SNLI) and transfers to "does this text show federal involvement" without any fine-tuning.

```
Input:  [chunk text] + "This project involves federal funding or financial assistance"
Output: entailment=0.87 → class score for federal_funding
```

The example bank (`_example_bank.md`) is used to **calibrate hypothesis wording** (not to train the model). Run each positive example through the model; the correct class hypothesis should score ≥ 0.75.

**Why MiniLM2-L6-H768 specifically:**
- ~67M parameters — fast enough for thousands of chunks on CPU
- 3-class output: contradiction / neutral / entailment (we use entailment score)
- `facebook/bart-large-mnli` (~400M params) is available as fallback if precision is insufficient

**Why not a bi-encoder (e.g., `all-MiniLM-L6-v2`):**
- Bi-encoders encode text and hypothesis independently and compare vectors
- Cross-encoders read both together in one pass — better at catching subtle entailment cues
- `all-MiniLM-L6-v2` is still used as the embedding fallback when the NLI model is unavailable

### Timeline pipeline: fine-tuned BERT

The timeline extraction task (`extract_timeline.py`) uses a fine-tuned BERT classifier because it has the **opposite profile** from Tier 4 — enough labeled examples exist (from regex pre-screening and manual corrections), but label quality is the challenge.

BERT fine-tuning works well when you can curate clean training examples and the label scheme is stable. The timeline classes (initiation, decision, etc.) are well-defined and the training set is large enough to generalize.

**Summary:** use NLI zero-shot when you can articulate what a class means in natural language but lack training data; use fine-tuned BERT when you have clean labeled examples and a stable label scheme.

## What improves Tier 4 fastest

1. Shrink the Tier 4 queue before changing the model.
- stop silently finalizing DOE Tier 1a rows
- expand the auto-accept whitelist only after audit
- fix validation so big deterministic rules are visible

2. Give Tier 4 better evidence, not just a better model.
- retrieve chunks from page text
- split CE differently from EA/EIS
- suppress boilerplate before classification

3. Build a small chunk example bank.
- a handful of good positive chunks per class
- a handful of strong negatives from CE forms and compliance lists

4. Restrict candidate classes per project.
- this is one of the cheapest and highest-value improvements

5. Let Tier 4 abstain.
- high precision matters more than forced coverage

## A good first routing scheme

### Deterministic metadata can stay final

Examples:

- FERC-like permit agencies -> `federal_permit`
- FAA-like permit agencies -> `federal_permit`
- FCC-like permit agencies -> `federal_permit`

### Ambiguous metadata should route into Tier 4

Examples:

- DOE -> usually `federal_funding` vs `federal_action` vs `unknown`
- USACE -> usually `federal_permit` vs `federal_land` vs `unknown`
- some BLM / USFS cases -> `federal_land` vs `federal_action`

### Cue-driven candidate class restriction

- DOE rows:
  - `federal_funding`
  - `federal_action`
  - `unknown`

- BLM / USFS rows:
  - `federal_land`
  - `federal_action`
  - `federal_program`
  - `unknown`

- permit-heavy rows:
  - `federal_permit`
  - `federal_land`
  - `unknown`

- programmatic title rows:
  - `federal_program`
  - `unknown`

## How to keep Tier 5 cheap

Tier 5 should be a narrow adjudication queue, not a general fallback.

Send to Tier 5 only when:

- top Tier 4 score is weak
- top-2 classes are too close
- chunks disagree
- no affirmative evidence chunk exists
- the only evidence is noisy or template-heavy

Do not send to Tier 5 when:

- deterministic whitelist already resolved the row
- Tier 4 has strong chunk support and a clean margin
- evidence is missing and `unknown` is the honest answer

## Highest-value example families

If collecting examples for review or for calibrating Tier 4, start here:

1. `T1a_DOE_action`
2. `T1a_DOE_funding`
3. `T3_sec404`
4. `T3_arra`
5. `T1b_arra`
6. `T1a_BLM_land`
7. `T1a_BLM_action`
8. `T3_rmp`
9. `T2_doc_title_peis`

## Best chunk examples to save

### Positive examples

- `federal_funding`
  - grant
  - loan guarantee
  - cooperative agreement
  - financial assistance

- `federal_action`
  - agency proposes to construct
  - agency will implement
  - agency will install
  - agency will restore

- `federal_land`
  - right-of-way grant
  - special use permit
  - crosses federal land
  - administered by BLM / USFS

- `federal_permit`
  - permit application
  - authorization required
  - Corps permit
  - FERC approval

- `federal_program`
  - PEIS
  - plan revision
  - leasing framework

- `federal_property_transaction`
  - land exchange
  - conveyance
  - disposal

### Negative examples

- CE checklist boilerplate
- generic compliance lists
- legal citation lists
- Section 404 mention without actual permit nexus
- ARRA mention without actual funding nexus
- plan-conformance text without programmatic action

## Practical threshold notes

Good conservative starting thresholds:

- top class score >= `0.90`
- top-minus-second margin >= `0.15`
- at least one affirmative supporting chunk
- no contradictory class within `0.10`

These are starting points, not fixed truths.

## What the first successful version should look like

- some audited Tier 1-3 rules are auto-accepted
- DOE and CE-heavy ambiguous rows go into Tier 4
- Tier 4 resolves a large local share using retrieved chunks
- Tier 5 sees only a small uncertain queue
- residual `unknown` is acceptable where evidence is weak
