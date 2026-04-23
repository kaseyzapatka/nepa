# NEPA Trigger Classification Strategy

This is a meeting-ready explanation of the current trigger classification strategy and the planned Tier 4 refactor. It is written to be detailed enough to be useful without getting lost in implementation detail.

## What I am trying to classify

The goal is to classify the **primary federal trigger** for why a project is in NEPA review.

The main trigger classes are:

- `federal_funding`
- `federal_action`
- `federal_land`
- `federal_permit`
- `federal_program`
- `federal_property_transaction`
- `unknown`

The key distinction is that I am trying to identify the **actual federal nexus**, not just any federal agency mention or any environmental law mention in the document.

---

## Core problem

At a high level, the task is harder than it first looks because the trigger is often **implicit** rather than stated cleanly.

A document may:

- clearly say DOE is issuing a loan guarantee
- clearly say a Section 404 permit is required
- or it may only say DOE is the lead agency, while the actual nexus is buried in the project description or page text

The biggest structural issue in this corpus is that it is heavily dominated by **DOE-led projects**. That means metadata alone is not enough, because DOE can mean:

- federal funding
- direct federal action
- both
- or sometimes neither is stated clearly enough to classify safely

The other major problem is that CE documents contain a lot of **template language and checklist boilerplate**. That creates false positives when a rule sees phrases like `Section 404` or `ARRA` in a generic form rather than in real project-specific trigger language.

So the basic failure mode is:

- **mention** gets mistaken for **evidence**

---

## Current strategy: tiered classification

The pipeline is intentionally tiered so the easy cases can be handled cheaply and deterministically, and only the ambiguous cases move to more expensive stages.

### Tier 1-3: deterministic extraction

The first three tiers use rules and metadata to capture cases where the trigger is relatively explicit.

These tiers use:

- agency metadata
- project title and description cues
- document title cues
- page-text cues such as permit language, right-of-way language, and programmatic titles

This stage is good at obvious cases like:

- DOE loan guarantee titles -> `federal_funding`
- special use permit / ROW grant language -> `federal_land`
- explicit NPDES or Corps permit language -> `federal_permit`
- programmatic EIS / EA titles -> `federal_program`
- land exchange language -> `federal_property_transaction`

The important point is that Tier 1-3 should handle **clear cases only**.

### Why Tier 1-3 is not enough

Tier 1-3 breaks down in two places:

1. ambiguous agencies, especially DOE
2. CE boilerplate that creates false positives

So the current issue is not that rules are useless. The issue is that some rules are genuinely high-value and some are over-broad, and right now they are not being separated cleanly enough.

---

## What I learned from the current output

There are three main takeaways from the current run:

### 1. metadata is useful, but only for deterministic cases

Metadata works when the agency-to-trigger mapping is basically one-to-one.

Examples:

- FERC-like agency -> usually `federal_permit`
- FAA-like agency -> usually `federal_permit`
- a land exchange title -> usually `federal_property_transaction`

Metadata fails when the agency is too broad.

Examples:

- DOE can mean funding or action
- BLM / USFS can mean land authorization or agency action
- USACE can mean permit, land, or broader federal involvement

So the right framing is:

- deterministic metadata can stay final
- ambiguous metadata should become a routing hint, not a final answer

### 2. CE documents need special handling

CEs are where a lot of false positives come from.

For example:

- `Section 404` may appear inside an unchecked CE checklist item
- `ARRA` may appear in a generic DOE categorical exclusion form
- a resource management plan may appear only in a conformance block

Those are not random mentions. They are realistic traps for a classifier. That is why the new approach needs stronger boilerplate suppression and document chunking.

### 3. current Tier 4 is too weak

The current Tier 4 does not retrieve page text or chunk documents. It compares `project_title + project_description` to a few prototype sentences.

That is too shallow for this problem.

It misses:

- trigger language in page text
- differences between affirmative evidence and template language
- differences between CE forms and EA/EIS narrative sections

So the next version of Tier 4 has to become a **retrieval-first local adjudication stage**, not just an embedding fallback.

---

## Refined strategy going forward

The strategy is:

1. keep deterministic rules for clear cases
2. stop silently finalizing the ambiguous cases
3. send only the ambiguous queue into a stronger local Tier 4
4. send only a very small residual to Tier 5

### Step 1: keep safe deterministic cases final

If the evidence is unusually explicit, I do not want to reprocess it.

Examples:

- loan guarantee titles
- special use permit / ROW grant text
- permit application titles
- PEIS / programmatic titles
- land exchange text

These can usually bypass the expensive adjudication stages.

### Step 2: force ambiguous cases into Tier 4

The main examples are:

- DOE metadata rules
- CE `Section 404` hits
- CE `ARRA` hits
- some BLM / USFS cases
- program / plan-conformance edge cases

This is where the classifier should slow down and ask:

- what is the actual evidence?
- is it affirmative?
- is it project-specific?
- is it boilerplate?

---

## What the new Tier 4 will do

The improved Tier 4 is designed around **better evidence**, not just a better model.

### Tier 4 will do three things

1. retrieve the best chunks of text
2. score a small set of plausible classes locally
3. abstain if the evidence is weak or contradictory

### Why retrieval matters

Instead of asking a model to classify a whole document or a short title/description summary, Tier 4 will try to pull the small amount of text that actually matters.

For example:

- `DOE would provide approximately 50 percent of the funding`
- `the USFS purpose and need is to determine whether to issue a special use permit`
- `a NPDES permit must be obtained`
- `Programmatic Environmental Impact Statement`

That is much more useful than sending an entire CE form or an entire EA abstract to a model.

### Why class restriction matters

Tier 4 should not score all classes for every project.

It should use metadata and cue types to narrow the candidate set.

Examples:

- DOE rows -> usually `federal_funding`, `federal_action`, or `unknown`
- BLM / USFS rows -> usually `federal_land`, `federal_action`, `federal_program`, or `unknown`
- permit-heavy rows -> usually `federal_permit`, `federal_land`, or `unknown`
- programmatic title rows -> usually `federal_program` or `unknown`

That improves both efficiency and accuracy.

### Why abstention matters

To get high accuracy, Tier 4 needs permission to say:

- `unknown`
- or `send to Tier 5`

If the model is forced to classify every ambiguous case, precision will collapse.

So the target is not “classify everything locally.”

The target is:

- auto-classify the clear cases with high precision
- abstain honestly on the hard cases

---

## How I plan to keep Tier 5 cheap

Tier 5 is the enterprise LLM stage, so I want it to be the exception, not the default.

A case should only go to Tier 5 when:

- the top Tier 4 score is weak
- the top two classes are too close
- different chunks support different classes
- there is no affirmative evidence chunk
- the only text available is noisy or template-heavy

A case should **not** go to Tier 5 when:

- a trusted deterministic rule already resolved it
- Tier 4 has strong chunk-level support
- `unknown` is the honest answer

The operational goal is:

- Tier 1-3 handle the obvious cases
- Tier 4 handles most of the ambiguous cases locally
- Tier 5 only sees a narrow adjudication queue

---

## What improves accuracy fastest

There are four practical levers that matter most right now.

### 1. shrink the Tier 4 queue before changing the model

The first improvement is not model-related. It is routing-related.

That means:

- stop silently finalizing DOE metadata cases
- audit and expand the safe auto-accept whitelist
- make sure known risky rule families are routed for adjudication

### 2. improve the evidence before improving the model

The biggest quality gain comes from:

- chunk retrieval
- better CE handling
- boilerplate suppression

This matters more than swapping one generic classifier for another.

### 3. use a small example bank to calibrate the classifier

I am building an example bank with:

- clean positives for each class
- ambiguous boundary cases
- hard negatives from CE boilerplate and compliance text

That helps define what real evidence looks like and what should be ignored.

### 4. allow `unknown`

Residual `unknown` is acceptable when the evidence is genuinely weak.

That is better than forcing a wrong class.

---

## Simple way to explain the strategy in one sentence

The strategy is to use deterministic rules for obvious triggers, then use retrieval plus local adjudication for the ambiguous cases, and reserve the enterprise LLM for only the small residual set where the evidence is still unclear.

---

## Simple way to explain the quality issue

The main quality issue is not lack of text. It is that the corpus contains a lot of text that looks relevant but is not actually trigger evidence, especially in DOE-led CE documents. So the refactor is mainly about routing, retrieval, and evidence quality, not just “using a smarter model.”

---

## What success looks like

The first successful version should look like this:

- trusted Tier 1-3 rules are auto-accepted
- DOE and CE-heavy ambiguous rows go into Tier 4
- Tier 4 resolves a large share locally using retrieved chunks
- Tier 5 sees only a small, expensive-to-review residual queue
- some rows remain `unknown`, which is acceptable
- overall accuracy improves because the system stops over-reading boilerplate

---

## Short meeting version

If I need to explain it quickly, I would say:

> I’m classifying the federal nexus that actually triggered NEPA, not just any federal mention in the file. The pipeline is tiered: deterministic rules handle the obvious cases first, then a new Tier 4 retrieves the best document chunks and uses a local model to adjudicate ambiguous cases like DOE funding vs DOE action or real permit language vs CE boilerplate. Only the small residual that is still unclear goes to the enterprise LLM. The key lesson from the current run is that accuracy depends less on adding a bigger model and more on routing the right cases, suppressing boilerplate, and classifying on affirmative project-specific evidence rather than on generic mentions.
