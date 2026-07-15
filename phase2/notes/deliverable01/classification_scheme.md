---
title: "D1: Classification Scheme"
---

This document covers the schema design, priority logic, confidence framework, keyword dictionary, and example coding scenarios for the NEPA trigger classification pipeline.

For class definitions and NLI hypothesis statements, see [Trigger Classification](trigger_classification.md).

---

## Suggested data structure

A project-level schema could look like this:

| Variable | Type | Description |
|---|---|---|
| `nepa_trigger_primary` | string | Main federal nexus category |
| `nepa_trigger_secondary` | list/string | Additional trigger categories present |
| `nepa_trigger_multi` | list/string | All categories detected |
| `nepa_trigger_evidence_text` | string | Short snippet supporting classification |
| `nepa_trigger_evidence_source` | string | Where evidence came from |
| `nepa_trigger_confidence` | integer or string | Confidence in classification |
| `nepa_trigger_manual_review` | boolean | Whether human review is recommended |
| `nepa_trigger_notes` | string | Analyst notes |

### Recommended values

#### `nepa_trigger_primary`
- `federal_funding`
- `federal_land`
- `federal_permit`
- `federal_direct_action`
- `federal_program`
- `federal_property_transaction`
- `pma` *(Power Marketing Administration (PMA) + Tennessee Valley Authority (TVA))*
- `unknown`

#### `nepa_trigger_evidence_source`
- `project_title`
- `project_description`
- `agency_metadata`
- `document_title`
- `document_text`
- `multiple_sources`

#### `nepa_trigger_confidence`
- `high`
- `medium`
- `low`

---

## Primary vs. secondary trigger logic

Many projects have multiple forms of federal involvement. For example:
- A transmission line may cross federal land **and** require a permit
- A transit project may receive federal funding **and** require a USACE permit
- A renewable project may be on BLM land and involve a federal right-of-way grant

### Rule: assign the primary trigger as the federal nexus most directly responsible for the agency's NEPA decision.

### Priority order for primary trigger assignment (as implemented)

Use this only when the text is ambiguous. This is the order implemented in
`TRIGGER_HIERARCHY` in `01_extract_nepa_trigger.py`:

1. `federal_program`
2. `federal_direct_action`
3. `pma` *(Power Marketing Administration (PMA) + Tennessee Valley Authority (TVA))*
4. `federal_property_transaction`
5. `federal_land`
6. `federal_permit`
7. `federal_funding`
8. `unknown`

**Why this order:** A programmatic umbrella outranks everything because it changes the level
of the NEPA review itself (see the example below). Below that, priority goes to the federal
role when the agency is the principal actor or decision-maker, rather than simply one source
of support or one regulatory checkpoint. PMA/TVA projects are elevated above land and permit
because the agency identity is the clearest nexus signal for those entities.

**Note (verified May 2026 output):** the ordering is empirically inert for `federal_program` —
no project in the current output carries `federal_program` together with another class in
`nepa_trigger_multi`, so its position in the hierarchy does not change any project's primary
classification.

Examples:
- If BLM is approving a right-of-way across federal land, `federal_land` is often more informative than `federal_permit`
- If DOE is building or directly implementing the action, `federal_direct_action` should outrank `federal_funding`
- If the document is a programmatic plan, `federal_program` should outrank narrower project features
- If BPA, WAPA, SEPA, SWPA, or TVA is the lead agency, assign `pma` as primary even when the project also crosses federal land or requires a permit

---

## Keyword dictionary

Keywords are a first-pass filter only — see the "Important caution" note below.

### `federal_funding`
Indicative terms: federal funding, federally funded, federal financial assistance, grant, cooperative agreement, loan guarantee, cost share, reimbursable agreement, funded through

Agencies often associated: DOE, DOT, HUD, EPA, USDA, FEMA

### `federal_land`
Indicative terms: federal land, public lands, right-of-way, ROW grant, special use permit, easement, crossing federal land, National Forest System lands, BLM land, Bureau of Land Management land, Forest Service land, military reservation

Agencies often associated: BLM, USFS, NPS, Bureau of Reclamation, DoD, USACE (when property control is central)

### `federal_permit`
Indicative terms: permit, license, authorization, approval, certification, permit application, jurisdictional waters, Section 404, incidental take permit, license amendment

Agencies often associated: USACE, FERC, FAA, FCC, NOAA/NMFS, USFWS

### `federal_action`
Indicative terms: agency proposes to, the Bureau proposes, the Department proposes, federal action consists of, construct, install, upgrade, operate, implement, federal facility, base operations

Agencies often associated: DoD, VA, USFS, BLM, Bureau of Reclamation, DOE, USACE

### `federal_program`
Indicative terms: programmatic EIS, PEIS, site-wide EIS (SWEIS), Tier 1 review, policy, rulemaking, integrated resource plan, program-wide, nationwide, regional plan

Note: land-management programmatic reviews (e.g., vegetation management PEAs, leasing program PEIS on federal lands, BLM wind/solar PEIS, Western Solar Plan, Section 368 corridor PEIS) are classified as `federal_land`, not `federal_program`.

### `pma` — Power Marketing Administration (PMA) + Tennessee Valley Authority (TVA)
Indicative terms: Bonneville Power Administration, BPA, Western Area Power Administration, WAPA, Southeastern Power Administration, SEPA, Southwestern Power Administration, SWPA, Power Marketing Administration, Tennessee Valley Authority, TVA

Agencies: BPA, WAPA, SEPA, SWPA, TVA, and any generic PMA. Assign `pma` as the primary trigger whenever one of these entities is the lead or sponsoring agency, even when the project also involves federal land (e.g., transmission line ROW grants) or permits. Add `federal_land` or `federal_permit` as secondary triggers when applicable.

### `federal_property_transaction`
Indicative terms: land exchange, conveyance, disposal, transfer, sale of federal land, acquisition, parcel transfer

### Important caution

Do not classify solely on keyword presence. For example:
- "permit" may appear in a background section even if the true trigger is federal land
- "funding" may be mentioned historically without being the current federal nexus
- "right-of-way" can be central or merely descriptive

Recommended rules engine:
1. detect candidate categories
2. extract surrounding evidence text
3. rank likely categories
4. assign confidence
5. send low-confidence cases to review

---

## Confidence framework

### High confidence
Assign `high` when:
- the text explicitly states the federal action or approval that triggers review
- multiple sources agree
- the language is direct and project-specific

Example: "The Bureau of Land Management must approve a right-of-way grant across federal lands" → high confidence `federal_land`

### Medium confidence
Assign `medium` when:
- the project strongly implies a trigger
- one source gives clear but indirect evidence
- multiple candidate triggers exist but one appears most likely

Example: transmission project with BLM lead agency and repeated ROW language, but no explicit sentence stating the nexus

### Low confidence
Assign `low` when:
- the project is clearly under NEPA review but the trigger is unclear
- only weak keyword matches are found
- several categories are plausible with no dominant one

Flag these for review.

---

## Recommended manual review flags

Set `nepa_trigger_manual_review = TRUE` when any of the following occur:
- more than one category receives similar evidence strength
- only low-confidence text was found
- the agency role is unclear
- the project appears to involve both federal land and federal permit issues
- the project is highly programmatic or unusual
- the project has sparse metadata

---

## Example coding scenarios

### Example 1: Transmission line across BLM land
Evidence: Project title mentions new 230-kV transmission line; description says the project crosses BLM-administered land and needs a right-of-way grant.

- `nepa_trigger_primary = federal_land`
- `nepa_trigger_secondary = federal_permit` only if a separate federal permit is clearly required
- `confidence = high`

### Example 2: Wetland fill permit for private development
Evidence: Private developer; USACE permit under Section 404 required.

- `nepa_trigger_primary = federal_permit`
- `confidence = high`

### Example 3: Federally funded transit improvement
Evidence: DOT grant funding is central to the project; no federal land use mentioned; NEPA document tied to funding approval.

- `nepa_trigger_primary = federal_funding`
- `confidence = high`

### Example 4: Forest Service vegetation management project
Evidence: Forest Service proposes thinning and fuel reduction on National Forest lands.

- `nepa_trigger_primary = federal_land`
- `nepa_trigger_secondary = federal_direct_action` optional if you want to separately record that the agency is the direct actor
- `confidence = high`

### Example 5: Programmatic leasing framework
Evidence: Document is a Programmatic EIS for a regional leasing strategy.

- `nepa_trigger_primary = federal_program`
- `confidence = high`

### Example 6: BPA transmission line with ROW on federal land
Evidence: Bonneville Power Administration proposes to rebuild a 230-kV transmission line; project requires a right-of-way grant across BLM-administered land.

- `nepa_trigger_primary = pma`
- `nepa_trigger_secondary = federal_land` (ROW grant on BLM land recorded as secondary nexus)
- `confidence = high`

Note: `pma` takes priority over `federal_land` because BPA is the lead agency and primary actor. The land nexus is real but secondary.

### Example 7: Ambiguous energy project
Evidence: DOE appears in metadata; document mentions funding and approvals but not clearly; no explicit statement of the federal nexus.

- `nepa_trigger_primary = unknown` or best inferred category
- `confidence = low`
- `manual_review = TRUE`

---

## Implementation options

### Minimal viable
Start with just these variables:
- `nepa_trigger_primary`
- `nepa_trigger_confidence`
- `nepa_trigger_evidence_text`
- `nepa_trigger_manual_review`

### Expanded (for research use)
- `nepa_trigger_primary`
- `nepa_trigger_secondary`
- `nepa_trigger_multi`
- `nepa_trigger_evidence_text`
- `nepa_trigger_evidence_source`
- `nepa_trigger_confidence`
- `nepa_trigger_manual_review`
- `nepa_trigger_notes`
- `nepa_trigger_rule_id` (which rule fired)
- `nepa_trigger_model_label` (if using ML/LLM adjudication)
