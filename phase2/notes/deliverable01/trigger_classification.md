---
title: "D1: NEPA Trigger Classification"
---

This is an explanation of the trigger definitions we want to use for this deliverable. 

## Classification Schema

Each of these 6 classes triggers NEPA review. The main trigger classes are:

- `Federal Funding`
- `Federal Action`
- `Federal Land`
- `Federal Permit`
- `Federal Program`
- `Federal Property Transaction`

### 1. Federal funding or financial assistance
**Code:** `federal_funding`

Use when the project is undergoing NEPA review because it receives direct or indirect federal financial support.

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

### 7. Unknown or unclear federal nexus
**Code:** `unknown`

Use when the project clearly underwent NEPA review, but the available materials do not allow reliable identification of the triggering nexus.

**Coding rule:**
Assign `unknown` only after attempting structured extraction from title, description, process metadata, agencies, and document text.

## Class Hypotheses

These are the class-specific hypothesis statements used in Tier 4 to determine whether a document belongs to a class.

- `federal_funding`
  - This text shows that a federal agency is funding, financing, or providing financial assistance, a grant, or a loan guarantee for this project.

- `federal_action`
  - This text shows that a federal agency is directly implementing, constructing, installing, operating, or restoring this project.

- `federal_land`
  - This text shows that the project is located on or crosses federal land, or requires a right-of-way grant or special use permit on federal land.

- `federal_permit`
  - This text shows that a federal permit, license, or authorization is required for this project.

- `federal_program`
  - This text shows that this is a programmatic environmental review, a resource management plan revision, or a land use plan covering a class of actions.

- `federal_property_transaction`
  - This text shows that this involves a federal land exchange, conveyance, or disposal.