# Rule ID Legend

This file is a compact reference for the rule IDs used in `01_extract_nepa_trigger.py`.

## General format

Most rule IDs follow:

`T{tier}{subtier}_{source-or-agency}_{rule-slug}`

Examples:

- `T1a_BLM_land`
- `T1a_DOE_funding`
- `T1b_special_use`
- `T2_doc_title_peis`
- `T3_sec404`
- `T4_embed_below_threshold`
- `T5_llm`

## Tier prefixes

- `T1a`: Tier 1a, agency metadata heuristics
- `T1b`: Tier 1b, title + project description pattern matching
- `T2`: document title scan
- `T3`: page text / document text / Purpose and Need text
- `T4`: embedding or local fallback stage
- `T5`: enterprise LLM fallback

## Middle segment

The middle segment usually indicates either:

- an agency code: `BLM`, `DOE`, `FERC`, `USFS`
- or a source: `doc_title`, `embed`

## Final segment

The final segment usually indicates either:

- the assigned trigger class: `land`, `funding`, `action`
- or the specific cue that fired: `sec404`, `loan_guarantee`, `special_use`, `peis`, `rmp`

## Common examples

- `T1a_BLM_land`
  - Tier 1a used BLM metadata and assigned `federal_land`

- `T1a_DOE_action`
  - Tier 1a used DOE metadata and assigned `federal_action`

- `T1a_DOE_funding`
  - Tier 1a used DOE metadata and assigned `federal_funding`

- `T1b_special_use`
  - Tier 1b matched `special use permit` or similar language

- `T1b_ferc_license`
  - Tier 1b matched FERC licensing or approval language

- `T2_doc_title_peis`
  - Tier 2 matched a programmatic EIS title

- `T2_doc_title_loan_guarantee`
  - Tier 2 matched a loan guarantee title

- `T3_sec404`
  - Tier 3 matched `Section 404` in page or document text

- `T3_arra`
  - Tier 3 matched ARRA text in page or document text

- `T3_rmp`
  - Tier 3 matched `resource management plan` text

- `T4_embed_below_threshold`
  - Tier 4 could not classify confidently

- `T5_llm`
  - final classification came from the LLM fallback

## Highest-value rule families for examples

These are the rule families where example collection is most useful for auditing and improving Tier 4.

### 1. DOE boundary cases

- `T1a_DOE_action`
- `T1a_DOE_funding`

Why:

- DOE dominates the corpus
- this is the biggest `federal_action` vs `federal_funding` ambiguity

Good examples should show:

- true DOE-as-funder language
- true DOE-as-actor language
- weak metadata-only cases that should probably not finalize

### 2. Section 404 cases

- `T3_sec404`

Why:

- likely mixes real permit triggers with CE checklist boilerplate

Good examples should include:

- positive examples: real permit application / authorization language
- negative examples: generic compliance checklists and form mentions

### 3. ARRA cases

- `T3_arra`
- `T1b_arra`

Why:

- likely mixes real funding-trigger language with generic stimulus-era form text

Good examples should include:

- true funding-trigger mentions
- generic law mentions with no actual nexus evidence

### 4. Land vs action cases

- `T1a_BLM_land`
- `T1a_BLM_action`

Why:

- useful for distinguishing project-on-federal-land from direct federal agency action

### 5. Programmatic cases

- `T3_rmp`
- `T2_doc_title_peis`

Why:

- `T2_doc_title_peis` is usually clean
- `T3_rmp` may be either strong programmatic evidence or generic plan-conformance language

## Best example bank to build for Tier 4

If building a small chunk-level example bank, prioritize:

- `federal_funding`
  - loan guarantee
  - cooperative agreement
  - federal grant
  - financial assistance

- `federal_action`
  - agency proposes to construct
  - agency will install / implement / restore

- `federal_land`
  - right-of-way grant
  - special use permit
  - crosses federal land
  - land administered by BLM / USFS

- `federal_permit`
  - permit application
  - authorization required
  - Corps permit
  - FERC approval

- `federal_program`
  - PEIS
  - resource management plan revision
  - leasing framework

- `federal_property_transaction`
  - land exchange
  - conveyance
  - disposal

- `unknown`
  - checklist boilerplate
  - legal compliance lists
  - generic plan-conformance language
  - stray statutory mentions

## Practical use

Use this legend when:

- sampling rule families for QA
- deciding which Tier 1-3 outputs should bypass Tier 4
- building positive and negative chunk examples for local Tier 4 adjudication
- explaining outputs to reviewers
