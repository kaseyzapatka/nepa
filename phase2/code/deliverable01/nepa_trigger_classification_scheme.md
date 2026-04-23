# NEPA Trigger Classification Scheme

This document translates the general idea of “what triggers NEPA review” into an implementable classification scheme for project-level research. It is designed for use in a structured dataset, document-mining pipeline, or mixed rules + manual review workflow.

## Purpose

The goal is to classify the **federal nexus** that appears to trigger NEPA review for a project. In practice, a project can have more than one trigger, so this scheme distinguishes between:

1. **Primary trigger**: the most central federal action driving NEPA review
2. **Secondary triggers**: other federal connections present in the project
3. **Evidence source**: where the trigger was identified
4. **Confidence**: how certain the classification is

This structure should work well for research on timelines, agency behavior, review intensity, and project type differences.

---

## Core concept

A project enters NEPA because there is some kind of **federal action** or **federal decision point**. For coding purposes, the cleanest umbrella concept is:

> **Federal nexus** = the specific type of federal involvement that appears to require or motivate NEPA review.

Instead of trying to code a single undifferentiated “reason for NEPA,” code the type of federal nexus.

---

## Recommended top-level classes

Use the following mutually intelligible categories. These can be implemented as a single primary category plus multi-label secondary categories.

### 1. Federal funding or financial assistance
**Code:** `federal_funding`

Use when the project is undergoing NEPA because it receives direct or indirect federal financial support.

Includes:
- Federal grants
- Cooperative agreements
- Federal loans or loan guarantees
- Federal cost sharing
- Federal reimbursement programs
- Formula funding where agency approval is tied to the project

Typical language:
- “funded by”
- “federal grant”
- “financial assistance”
- “loan guarantee”
- “cost share”
- “recipient of DOE/DOT/HUD funding”

Examples:
- Transit project using Federal Transit Administration money
- Grid project using DOE funding
- Housing or infrastructure project using HUD funds

**Coding rule:**
Assign `federal_funding` when the project description or NEPA document indicates that federal financial support is a reason the agency is involved in project approval or review.

---

### 2. Federal land, right-of-way, easement, or land management involvement
**Code:** `federal_land`

Use when the project is located on, crosses, uses, or requires access to federally managed land or federally controlled property interests.

Includes:
- Project located on federal land
- Transmission line, road, pipeline, or trail crossing federal land
- Right-of-way grant across BLM, USFS, NPS, DoD, Reclamation, etc.
- Easements or land-use authorizations on federal property
- Site-specific land management approvals

Typical language:
- “located on federal land”
- “crosses BLM land”
- “right-of-way grant”
- “special use permit”
- “land management plan area”
- “National Forest System lands”

Examples:
- Transmission line across BLM land
- Solar project on Bureau of Land Management land
- Recreation or mining access authorization on Forest Service land

**Coding rule:**
Assign `federal_land` when the federal nexus arises from land ownership, land management authority, easement authority, or right-of-way control.

---

### 3. Federal permit, license, approval, or authorization
**Code:** `federal_permit`

Use when the project needs a federal permit, license, certification, or approval, even if it is otherwise private or state-led.

Includes:
- U.S. Army Corps permits
- FERC approvals
- FAA approvals
- FCC licenses
- Federal siting or operational approvals
- Incidental take permits or other federal environmental permits
- Any other federal authorization needed before the project can proceed

Typical language:
- “requires a federal permit”
- “application for permit”
- “license amendment”
- “authorization requested”
- “approval by [federal agency]”
- “jurisdictional waters permit”

Examples:
- Wetland fill permit from USACE
- Hydropower licensing action at FERC
- FAA action related to airport expansion

**Coding rule:**
Assign `federal_permit` when the federal nexus is primarily a regulatory authorization, license, permit, or approval.

---

### 4. Direct federal agency action or federal project sponsor
**Code:** `federal_action`

Use when a federal agency is itself carrying out, constructing, adopting, managing, or directly sponsoring the action.

Includes:
- Federal construction projects
- Federal facility upgrades
- Military base actions
- Federal land management actions
- Agency-run restoration or infrastructure actions
- Federal leasing decisions when the agency is the primary actor

Typical language:
- “the agency proposes to”
- “federal action by”
- “the Bureau proposes”
- “the Department will construct”
- “the Forest Service proposes”

Examples:
- VA hospital expansion
- Army Corps channel dredging
- USFS vegetation management project

**Coding rule:**
Assign `federal_action` when the agency is not just approving someone else’s project but is the primary actor proposing or implementing the action.

---

### 5. Federal plan, program, rulemaking, or policy decision
**Code:** `federal_program`

Use when the action is not primarily a site-specific project, but a broader federal planning or policy action.

Includes:
- Programmatic EIS
- Resource management plan revisions
- National or regional strategy documents
- Program-wide leasing frameworks
- Rulemaking or regulation changes
- Corridor designations or similar planning actions

Typical language:
- “programmatic environmental impact statement”
- “resource management plan amendment”
- “rulemaking”
- “leasing program”
- “policy framework”
- “regional plan”

Examples:
- Offshore wind leasing program
- BLM land-use planning revision
- Nationwide programmatic review for infrastructure corridor designations

**Coding rule:**
Assign `federal_program` when the NEPA action is a broader plan, rule, policy, or program rather than a single facility or construction project.

---

### 6. Federal property transfer, disposal, acquisition, or conveyance
**Code:** `federal_property_transaction`

Use when the federal action involves transfer or disposition of land or property interests.

Includes:
- Sale of federal land
- Land exchange
- Transfer of administrative control
- Acquisition of land by federal agency when NEPA review is tied to that action
- Easement conveyance or related property transaction

Typical language:
- “land exchange”
- “disposal”
- “conveyance”
- “property transfer”
- “sale of federal parcel”

Examples:
- Federal land disposal for development
- Land exchange involving public lands

**Coding rule:**
Assign `federal_property_transaction` when the trigger is fundamentally about a transfer or disposition of federally controlled property.

---

### 7. Unknown or unclear federal nexus
**Code:** `unknown`

Use when the project clearly underwent NEPA review, but the available materials do not allow reliable identification of the triggering nexus.

**Coding rule:**
Assign `unknown` only after attempting structured extraction from title, description, process metadata, agencies, and document text.

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
- `federal_action`
- `federal_program`
- `federal_property_transaction`
- `unknown`

#### `nepa_trigger_evidence_source`
- `project_title`
- `project_description`
- `agency_metadata`
- `document_title`
- `document_text`
- `multiple_sources`

#### `nepa_trigger_confidence`
Either use a numeric scale or label scale.

Example label scale:
- `high`
- `medium`
- `low`

Example numeric scale:
- `3 = high`
- `2 = medium`
- `1 = low`

---

## Primary vs. secondary trigger logic

Many projects have multiple forms of federal involvement. For example:
- A transmission line may cross federal land **and** require a permit
- A transit project may receive federal funding **and** require a USACE permit
- A renewable project may be on BLM land and involve a federal right-of-way grant

To make the coding analytically useful, use this rule:

### Assign the **primary trigger** as the federal nexus most directly responsible for the agency’s NEPA decision.

### Assign **secondary triggers** for any additional federal nexuses clearly present.

A practical prioritization rule can help when documents mention several nexuses but do not clearly state which one is primary.

## Suggested priority order for primary trigger assignment

Use this only when the text is ambiguous.

1. `federal_action`
2. `federal_program`
3. `federal_property_transaction`
4. `federal_land`
5. `federal_permit`
6. `federal_funding`
7. `unknown`

### Why this order?
This ordering gives priority to the federal role when the agency is the principal actor or decision-maker, rather than simply one source of support or one regulatory checkpoint.

For example:
- If BLM is approving a right-of-way across federal land, `federal_land` is often more informative than `federal_permit`
- If DOE is building or directly implementing the action, `federal_action` should generally outrank `federal_funding`
- If the document is a programmatic plan, `federal_program` should outrank narrower project features

You can revise this ordering depending on your research question.

---

## Implementation workflow

A workable coding pipeline would usually have four stages.

### Stage 1: Structured metadata pass
Use project-level fields first.

Useful fields may include:
- lead agency
- cooperating agencies
- project title
- project description
- project type
- location
- any permit, funding, or right-of-way metadata

#### Examples of easy structured cues
- DOE, DOT, HUD funding language → likely `federal_funding`
- BLM/USFS land crossing or ROW language → likely `federal_land`
- “permit,” “license,” “authorization,” “Section 404” → likely `federal_permit`
- “agency proposes to construct” → likely `federal_action`
- “programmatic EIS” or “resource management plan” → likely `federal_program`
- “land exchange” or “conveyance” → likely `federal_property_transaction`

### Stage 2: Document-title pass
If metadata is not enough, scan NEPA document titles.

Useful title phrases:
- “Programmatic Environmental Impact Statement” → `federal_program`
- “Land Exchange” → `federal_property_transaction`
- “Right-of-Way” → often `federal_land`
- “Permit Application” or “License Amendment” → `federal_permit`

### Stage 3: Document-text pass
If still unresolved, search document text for trigger phrases and contextual evidence.

Use excerpt extraction around terms like:
- funding
- grant
- right-of-way
- permit
- authorization
- federal lands
- proposes to
- resource management plan
- conveyance

Capture snippets with enough context to code the trigger.

### Stage 4: Manual review or model adjudication
Use manual review or an LLM/BERT adjudication step for:
- projects with multiple conflicting signals
- projects where only weak evidence is found
- projects where category ranking is ambiguous

---

## Rules-based coding dictionary

Below is a practical keyword dictionary to start with. This is not enough by itself, but it is a good first-pass rules layer.

## `federal_funding`
Indicative terms:
- federal funding
- federally funded
- federal financial assistance
- grant
- cooperative agreement
- loan guarantee
- cost share
- reimbursable agreement
- funded through

Agencies often associated with this trigger:
- DOE
- DOT
- HUD
- EPA
- USDA
- FEMA

## `federal_land`
Indicative terms:
- federal land
- public lands
- right-of-way
- ROW grant
- special use permit
- easement
- crossing federal land
- National Forest System lands
- BLM land
- Bureau of Land Management land
- Forest Service land
- military reservation

Agencies often associated with this trigger:
- BLM
- USFS
- NPS
- Bureau of Reclamation
- DoD
- USACE when property control is central

## `federal_permit`
Indicative terms:
- permit
- license
- authorization
- approval
- certification
- permit application
- jurisdictional waters
- Section 404
- incidental take permit
- license amendment

Agencies often associated with this trigger:
- USACE
- FERC
- FAA
- FCC
- NOAA/NMFS
- USFWS

## `federal_action`
Indicative terms:
- agency proposes to
- the Bureau proposes
- the Department proposes
- federal action consists of
- construct
- install
- upgrade
- operate
- implement
- federal facility
- base operations

Agencies often associated with this trigger:
- DoD
- VA
- USFS
- BLM
- Bureau of Reclamation
- DOE
- USACE

## `federal_program`
Indicative terms:
- programmatic EIS
- PEIS
- resource management plan
- land use plan
- policy
- rulemaking
- corridor designation
- leasing program
- program-wide
- nationwide
- regional plan

## `federal_property_transaction`
Indicative terms:
- land exchange
- conveyance
- disposal
- transfer
- sale of federal land
- acquisition
- parcel transfer

---

## Important caution: keywords are not enough

You should not classify solely on the presence of a keyword. For example:
- “permit” may appear in a background section even if the true trigger is federal land
- “funding” may be mentioned historically without being the current federal nexus
- “right-of-way” can be central or merely descriptive

So the recommended rules engine is:

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

Example:
- “The Bureau of Land Management must approve a right-of-way grant across federal lands” → high confidence `federal_land`

### Medium confidence
Assign `medium` when:
- the project strongly implies a trigger
- one source gives clear but indirect evidence
- multiple candidate triggers exist but one appears most likely

Example:
- transmission project with BLM lead agency and repeated ROW language, but no explicit sentence stating the nexus

### Low confidence
Assign `low` when:
- the project is clearly under NEPA review but the trigger is unclear
- only weak keyword matches are found
- several categories are plausible with no dominant one

These cases should usually be flagged for review.

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

## Example coding rules

### Example 1: Transmission line across BLM land
Evidence:
- Project title mentions new 230-kV transmission line
- Description says the project crosses BLM-administered land and needs a right-of-way grant

Coding:
- `nepa_trigger_primary = federal_land`
- `nepa_trigger_secondary = federal_permit` only if a separate federal permit is clearly required
- `confidence = high`

### Example 2: Wetland fill permit for private development
Evidence:
- Private developer
- USACE permit under Section 404 required

Coding:
- `nepa_trigger_primary = federal_permit`
- `nepa_trigger_secondary = none`
- `confidence = high`

### Example 3: Federally funded transit improvement
Evidence:
- DOT grant funding is central to the project
- No federal land use mentioned
- NEPA document tied to funding approval

Coding:
- `nepa_trigger_primary = federal_funding`
- `confidence = high`

### Example 4: Forest Service vegetation management project
Evidence:
- Forest Service proposes thinning and fuel reduction on National Forest lands

Coding:
- `nepa_trigger_primary = federal_action`
- `nepa_trigger_secondary = federal_land` is optional depending on whether you want to separately record land context
- `confidence = high`

### Example 5: Programmatic leasing framework
Evidence:
- Document is a Programmatic EIS for a regional leasing strategy

Coding:
- `nepa_trigger_primary = federal_program`
- `confidence = high`

### Example 6: Ambiguous energy project
Evidence:
- DOE appears in metadata
- document mentions funding and approvals but not clearly
- no explicit statement of the federal nexus

Coding:
- `nepa_trigger_primary = unknown` or best inferred category
- `confidence = low`
- `manual_review = TRUE`

---

## Minimal viable implementation

If you want a lighter-weight first pass, start with just these variables:

- `nepa_trigger_primary`
- `nepa_trigger_confidence`
- `nepa_trigger_evidence_text`
- `nepa_trigger_manual_review`

That gives you a strong starting point without overcomplicating the pipeline.

---

## Expanded implementation for research use

If you want stronger downstream analysis, include:

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

This makes it easier to audit classifications later.

---

## Suggested research framing

This scheme is especially useful if your broader question is not just “what type of project is this?” but:

- What type of **federal nexus** is bringing the project into NEPA?
- Do projects triggered by permits move faster than those triggered by federal land use?
- Are programmatic reviews longer or broader than funding-based reviews?
- Are some agencies associated with certain trigger types?

That is the main conceptual advantage of coding NEPA triggers this way.

---

## Final recommendation

For most projects, implement this as a **hybrid classification system**:

1. **Rules-based first pass** using metadata and keyword dictionaries
2. **Evidence snippet extraction** from project text and documents
3. **Confidence scoring**
4. **Manual or model-assisted adjudication** for ambiguous cases

That approach should be both scalable and defensible.
