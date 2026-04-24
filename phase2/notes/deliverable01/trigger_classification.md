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

- Federal grants (DOE, DOT, HUD, USDA)
- Cooperative agreements with partial federal funding
- Federal loans or loan guarantees (including DOE Title XVII loan guarantees)
- Federal cost sharing arrangements
- Formula-based awards (EECBG, State Energy Program, WAP)
- Bipartisan Infrastructure Law or Inflation Reduction Act funding
- ARRA / Recovery Act funding

Typical language (these are the exact phrases the extraction script detects):

**Grants and direct funding:**
- “federal grant” / “DOE grant” / “DOT grant” / “HUD grant” / “USDA grant” / “grant funding”
- “federal funding” / “federal financial assistance”
- “DOE/DOT/HUD/USDA grant” / “DOE/DOT/HUD/USDA funding”
- “provide federal funding”
- “DOE Funding” (in dollar-amount context)
- “DOE/Department of Energy ... would provide ... funds/funding/grant/awards/cost-share”
- “total award value” / “Total Project Value”

**Loans and loan guarantees:**
- “loan guarantee”
- “Title XVII”

**Cooperative agreements:**
- “through a cooperative agreement ... partially fund”
- “providing financial assistance to ... cooperative agreement”
- “awarding a grant ... partially fund”

**Cost sharing:**
- “cost share”
- “DOE Funding = $X / Cost Share = $X”
- “cost-shared arrangement”
- “Federal Cost Share ... Total Project Value”

**Legislative funding authorities:**
- “Inflation Reduction Act”
- “Bipartisan Infrastructure Law” / “Bipartisan Infrastructure Act”
- “Title XVII”
- “ARRA” / “Recovery Act” (detected via Tier 1a DOE agency routing)

**Formula-based programs:**
- “formula awards” / “formula-based awards” / “formula-based grants”
- “EECBG funding” / “DOE EECBG funding”
- “State Energy Program (SEP) ... formula awards”
- “WAP ... formula awards”
- “Administrative and Legal Requirements Document (ALRD) ... formula awards”

Examples:

- Transit project using Federal Transit Administration money
- Grid project using DOE Title XVII loan guarantee
- Housing or infrastructure project using HUD funds
- Energy efficiency project funded through an EECBG formula grant

**Coding rule:**
Assign `federal_funding` when the project description or NEPA document indicates that federal financial support is a reason the agency is involved in project approval or review.

### 2. Federal land, right-of-way, easement, or land management involvement
**Code:** `federal_land`

Use when the project is located on, crosses, uses, or requires access to federally managed land or federally controlled property interests.

Includes:

- Right-of-way grants, renewals, and amendments on BLM, USFS, NPS, Bureau of Reclamation, or tribal lands
- Special use permits on federal land
- Temporary and permanent easements across federal land, including expired easement renewals
- 2920 Land Use Authorizations (Bureau of Reclamation)
- Bureau of Indian Affairs right-of-way grants
- Amendments to existing ROW grants or land use authorizations
- Federal Land Policy and Management Act (FLPMA) authorizations for rights-of-way across public lands

Typical language (these are the exact phrases the extraction script detects):

**ROW applications and grants on BLM / public land:**
- “application for a right-of-way grant”
- “30-year right-of-way grant” + “BLM-administered lands” (within same passage)
- “right-of-way (ROW)” + “public land administered by the Bureau of Land Management” (within same passage)
- “request for a right-of-way” + “public land managed by BLM” (within same passage)
- “public lands managed by the Bureau of Land Management”
- “grant a perpetual ROW on BLM managed public land”
- “perpetual right-of-way grant”
- “rights-of-way over, upon, under, or through public lands”

**ROW renewals and amendments:**
- “right-of-way renewal applications”
- “right-of-way renewal and amendment”
- “Request to Amend Existing Authorization”
- “amend its ROW grant”

**Special use permits:**
- “special use permit”
- “current authorization with a defined ROW” + “Operation and Maintenance Plan” (either order, within same passage)

**Easements:**
- “temporary and permanent easements”
- “easement has expired”
- “easement” + “right-of-way” (within same passage; includes “easement for the right-of-way”)

**Bureau of Indian Affairs:**
- “Bureau of Indian Affairs is requesting a new right-of-way (ROW)”

**Bureau of Reclamation:**
- “lands administered by the Bureau of Reclamation” + “permissions must be sought” (within same passage)
- “2920 Land Use Authorization”

**FLPMA statutory authority:**
- “Title V of the Federal Land Policy and Management Act” + “respond to requests for rights-of-way across public lands” (within same passage)

**Document title (Tier 2 scan):**
- “right-of-way” (any hyphen/spacing variant) appearing in the document title

**Agency metadata (Tier 1a — detected without text cues):**
- Agency is BLM, Bureau of Land Management, USFS, Forest Service, NPS, National Park Service, FWS / USFWS, Fish and Wildlife Service, BOR / USBR, Bureau of Reclamation

Examples:

- Transmission line across BLM land requiring a 30-year ROW grant
- Solar project requiring a special use permit on National Forest land
- Expired BPA easement renewed through Bureau of Indian Affairs
- Pipeline ROW amendment on Bureau of Reclamation land

**Coding rule:**
Assign `federal_land` when the federal nexus arises from land ownership, land management authority, easement authority, or right-of-way control.

### 3. Federal permit, license, approval, or authorization
**Code:** `federal_permit`

Use when the project needs a federal permit, license, certification, or approval, even if it is otherwise private or state-led.

Includes:

- U.S. Army Corps Section 404 permits (individual, standard, nationwide) and Section 10 Rivers and Harbors Act permits
- NPDES permits (construction stormwater general permit, point source discharge)
- FERC hydropower licenses, relicensing, and certificates of public convenience and necessity
- NRC early site permits, combined licenses, license renewals (including subsequent license renewals), and license amendments
- Incidental Take Permits (ITP) under ESA Section 10(a)(1)(B), including renewed/amended ITPs and Habitat Conservation Plan-linked permits
- Presidential Permits (cross-border infrastructure)
- FAA and FCC approvals — detected via agency metadata (Tier 1a), not document text cues

Typical language (these are the exact phrases the extraction script detects):

**Army Corps / water permits:**
- “Standard Individual Permit Application” / “Individual Permit Application”
- “Section 404 permit application”
- “applied for an individual permit under Section 404”
- “Department of Army permit pursuant to Section 404 ... Section 10 of the Rivers and Harbors”
- “Section 10 ... Rivers and Harbors”
- “Nationwide Permit (NWP) Verification”
- “Nationwide, Regional General, or Standard Individual Permit may be required”

**NPDES:**
- “National Pollutant Discharge Elimination System (NPDES) permit”
- “NPDES permit must be obtained”
- “NPDES ... permitting decision”
- “Construction Storm Water General Permit is required”

**Incidental take / ESA:**
- “incidental take permit application”
- “Incidental Take Permit (ITP) under Section 10(a)(1)(B)”
- “Renewed/Amended ITP is needed”
- “Habitat Conservation Plan and Incidental Take Permit”

**FERC:**
- “hydropower license”
- “relicense” / “relicensing”
- “application for a certificate of public convenience and necessity”
- “NRC/FERC license amendment” (also detected near “10 CFR 50.90” or “FERC order”)

**Presidential permits:**
- “Amendment to Presidential Permit”
- “Issuance of Presidential Permit PP-[number]”
- “Presidential Permit Application Review”
- “Presidential Permit”

**NRC nuclear licenses:**
- “Early Site Permit”
- “Combined License”
- “License Renewal” / “Subsequent License Renewal”
- “issuance of renewed facility operating licenses”

Examples:

- Wetland fill requiring an individual Section 404 permit from USACE
- Hydropower relicensing action at FERC
- Wind project requiring an Incidental Take Permit and Habitat Conservation Plan
- Nuclear plant combined license or subsequent license renewal from NRC
- Cross-border transmission line requiring a Presidential Permit

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