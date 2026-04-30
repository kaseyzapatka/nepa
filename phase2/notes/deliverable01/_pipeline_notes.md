# Pipeline Notes: `01_extract_nepa_trigger.py`

Developer working notes for building and debugging the trigger classification pipeline. More disposable than the architecture doc; update freely.

---

## Model selection rationale

### Tier 4: `cross-encoder/nli-MiniLM2-L6-H768`

Tier 4 uses a cross-encoder NLI model instead of a fine-tuned BERT classifier because **Tier 4 is a data scarcity problem, not a label quality problem**.

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

### Why not fine-tuned BERT for Tier 4?

The timeline extraction task (`extract_timeline.py`) uses a fine-tuned BERT classifier because it has the **opposite profile** from Tier 4 — enough labeled examples exist (from regex pre-screening and manual corrections), but label quality is the challenge.

**Summary:** use NLI zero-shot when you can articulate what a class means in natural language but lack training data; use fine-tuned BERT when you have clean labeled examples and a stable label scheme.

---

## Core problem

At a high level, the task is harder than it first looks because the trigger is often **implicit** rather than stated cleanly.

A document may:
- clearly say DOE is issuing a loan guarantee
- clearly say a Section 404 permit is required
- or it may only say DOE is the lead agency, while the actual nexus is buried in the project description or page text

The corpus is heavily dominated by **DOE-led projects**. That means metadata alone is not enough, because DOE can mean federal funding, direct federal action, both, or sometimes neither is stated clearly enough to classify safely.

CE documents contain a lot of **template language and checklist boilerplate**. That creates false positives when a rule sees phrases like `Section 404` or `ARRA` in a generic form rather than in real project-specific trigger language.

The basic failure mode is: **mention gets mistaken for evidence.**

---

## What I learned from the current output

### 1. Metadata is useful, but only for deterministic cases

Metadata works when the agency-to-trigger mapping is basically one-to-one:
- FERC-like agency → usually `federal_permit`
- a land exchange title → usually `federal_property_transaction`

Metadata fails when the agency is too broad:
- DOE can mean funding or action
- BLM / USFS can mean land authorization or agency action

Right framing: deterministic metadata can stay final; ambiguous metadata should become a routing hint, not a final answer.

### 2. CE documents need special handling

CEs are where most false positives come from:
- `Section 404` may appear inside an unchecked CE checklist item
- `ARRA` may appear in a generic DOE categorical exclusion form
- a resource management plan may appear only in a conformance block

These are realistic traps for a classifier — why the new approach needs stronger boilerplate suppression and document chunking.

### 3. Current Tier 4 is too weak

The current Tier 4 compares `project_title + project_description` to a few prototype sentences. That is too shallow. It misses trigger language in page text, differences between affirmative evidence and template language, and differences between CE forms and EA/EIS narrative sections. The next version of Tier 4 must become a **retrieval-first local adjudication stage**, not just an embedding fallback.

---

## What improves Tier 4 fastest

1. **Shrink the Tier 4 queue before changing the model.**
   - Stop silently finalizing DOE Tier 1a rows
   - Expand the auto-accept whitelist only after audit
   - Fix validation so big deterministic rules are visible

2. **Give Tier 4 better evidence, not just a better model.**
   - Retrieve chunks from page text
   - Split CE differently from EA/EIS
   - Suppress boilerplate before classification

3. **Build a small chunk example bank.**
   - A handful of good positive chunks per class
   - A handful of strong negatives from CE forms and compliance lists

4. **Restrict candidate classes per project.**
   - One of the cheapest and highest-value improvements

5. **Let Tier 4 abstain.**
   - High precision matters more than forced coverage

---

## Routing scheme

### Deterministic metadata can stay final

- FERC-like permit agencies → `federal_permit`
- FAA-like permit agencies → `federal_permit`
- FCC-like permit agencies → `federal_permit`

### Ambiguous metadata should route into Tier 4

- DOE → usually `federal_funding` vs `federal_action` vs `unknown`
- USACE → usually `federal_permit` vs `federal_land` vs `unknown`
- Some BLM / USFS cases → `federal_land` vs `federal_action`

### Cue-driven candidate class restriction

| Agency/cue type | Candidate classes |
|---|---|
| DOE rows | `federal_funding`, `federal_action`, `unknown` |
| BLM / USFS rows | `federal_land`, `federal_action`, `federal_program`, `unknown` |
| Permit-heavy rows | `federal_permit`, `federal_land`, `unknown` |
| Programmatic title rows | `federal_program`, `unknown` |

---

## How to keep Tier 5 cheap

Send to Tier 5 **only** when:
- top Tier 4 score is weak
- top-2 classes are too close
- chunks disagree
- no affirmative evidence chunk exists
- the only evidence is noisy or template-heavy

Do **not** send to Tier 5 when:
- a trusted deterministic rule already resolved the row
- Tier 4 has strong chunk support and a clean margin
- evidence is missing and `unknown` is the honest answer

---

## Threshold calibration targets

These are starting points, not fixed truths:

- top class score ≥ **0.90**
- top-minus-second margin ≥ **0.15**
- at least one affirmative supporting chunk
- no contradictory class within 0.10

---

## What the first successful version should look like

- audited Tier 1-3 rules are auto-accepted
- DOE and CE-heavy ambiguous rows go into Tier 4
- Tier 4 resolves a large local share using retrieved chunks
- Tier 5 sees only a small uncertain queue
- residual `unknown` is acceptable where evidence is weak
- overall accuracy improves because the system stops over-reading boilerplate

---

## One-paragraph summary

> I'm classifying the federal nexus that actually triggered NEPA, not just any federal mention in the file. The pipeline is tiered: deterministic rules handle the obvious cases first, then a new Tier 4 retrieves the best document chunks and uses a local model to adjudicate ambiguous cases like DOE funding vs DOE action or real permit language vs CE boilerplate. Only the small residual that is still unclear goes to the enterprise LLM. The key lesson from the current run is that accuracy depends less on adding a bigger model and more on routing the right cases, suppressing boilerplate, and classifying on affirmative project-specific evidence rather than on generic mentions.
