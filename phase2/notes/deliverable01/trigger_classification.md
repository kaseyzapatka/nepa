---
title: "D1: NEPA Trigger Classification"
---

This is an explanation of the trigger definitions we want to use for this deliverable. 

## Classification Schema

Each of these 6 classes triggers NEPA review. The main trigger classes are:

- `Federal Funding`
- `Federal Land`
- `Federal Permit`
- `Federal Action`
- `Federal Program`
- `Federal Property Transaction`

### 1. Federal funding 

--- 

**Coding rule:** `Assign federal funding classification when the project receives direct or indirect federal financial support or when federal financial support is the reason the agency is involved in project approval or review`.

**Grants and direct funding:**

- “federal grant” / “DOE grant” / “DOT grant” / “HUD grant” / “USDA grant” / “grant funding”
- “federal funding” / “federal financial assistance”
- “DOE grant” / “DOE funding” / “DOT grant” / “DOT funding” / “HUD grant” / “HUD funding” / “USDA grant” / “USDA funding” (Tier 1b agency+type pattern)
- “provide federal funding”
- “DOE Funding” (standalone; Tier 4 context scan)
- “DOE/Department of Energy” + “would provide” + “funds/funding/grant/awards/cost-share” (within same passage)
- “total award value”
- “Total Project Value”

**Loans and loan guarantees:**

- “loan guarantee”
- “Title XVII”

**Cooperative agreements:**

- “through a cooperative agreement ... partially fund” (within same passage)
- “providing financial assistance to ... under/through a cooperative agreement” (within same passage)
- “awarding a grant ... partially fund” (within same passage)

**Cost sharing:**

- “cost share”
- “DOE Funding = $[amount] ... Cost Share = $[amount]” (Tier 1b; requires actual dollar figures)
- “cost-shared arrangement”
- “DOE's (proposed) action is to provide ... cost-shared arrangement” (within same passage; Tier 1b only)
- “Federal Cost Share” + “Total Project Value” within same passage (Tier 1b)
- “Federal Cost Share” standalone (Tier 4 context scan — lower specificity)

**Legislative funding authorities:**

- “Inflation Reduction Act”
- “Bipartisan Infrastructure Law” / “Bipartisan Infrastructure Act”
- “Title XVII”

**Formula-based programs:**

- “formula awards” / “formula-based awards” / “formula grants” / “formula-based grants”
- “EECBG funding” / “DOE EECBG funding”
- “State Energy Program” / “SEP” / “WAP” + “formula awards/grants” within same passage
- “Administrative and Legal Requirements Document” / “ALRD” + “formula awards/grants” within same passage (medium confidence)

**Document title (Tier 2 scan):**

- “loan guarantee” appearing in the document title

**Agency metadata (Tier 1a — detected without text cues):**

- Agency is DOT / Department of Transportation → auto-assigned `federal_funding`
- Agency is HUD / Department of Housing and Urban Development → auto-assigned `federal_funding`
- Agency is FTA / Federal Transit Administration → auto-assigned `federal_funding`
- Agency is FHWA / Federal Highway Administration → auto-assigned `federal_funding`
- Note: DOE is ambiguous — it can indicate either `federal_funding` or `federal_action` depending on verb context; DOE projects are routed to Tier 4 for adjudication


### 2. Federal land

---

**Coding rule:** `Assign federal land classification when the project is located on, crosses, uses, or requires access to federally managed land or federally controlled property interests or when the project arises from federal land ownership, land management authority, easement authority, or right-of-way control`.

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
- “easement for the right-of-way” (Tier 1b; exact phrase)
- “easement” + “right-of-way” within same passage (~120 chars) (Tier 4; broader proximity match)

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


### 3. Federal permit

---

**Coding rule:** `Assign federal permit classification when the  project needs a federal permit, license, certification, or approval--even if it is otherwise private or state-led--and the action is primarily a regulatory authorization, license, permit, or approval.`

**Army Corps / Section 404 permits:**

- “Department of the Army Environmental Assessment and Statement of Finding/Findings” + “(Standard) Individual Permit Application” within same passage (Tier 1b only)
- “Standard Individual Permit Application” / “Individual Permit Application”
- “Section 404 permit application”
- “applied for an individual permit under Section 404”
- “Department of Army permit” / “Department of the Army (DA) permit” + “Section 404” within same passage (180 chars)
- “Section 10” + “Rivers and Harbors” within same passage (80 chars)
- “Nationwide Permit (NWP) Verification” (medium confidence)
- “Nationwide, Regional General, or Standard Individual Permit may be required” (medium confidence)

**NPDES:**

- “issuance of a National Pollutant Discharge Elimination System permit” (Tier 1b; without NPDES acronym)
- “National Pollutant Discharge Elimination System (NPDES) permit” (Tier 1b + Tier 4; with NPDES acronym)
- “National Pollutant Discharge Elimination System (NPDES) Construction Storm Water General Permit is required” (full phrase, Tier 1b)
- “NPDES permit must be obtained”
- “NPDES” + “permitting decision” within same passage (80 chars)
- “Construction Storm Water General Permit is required”

**Incidental take / ESA:**

- “incidental take permit application”
- “Incidental Take Permit (ITP) under Section 10(a)(1)(B)”
- “Renewed/Amended ITP is needed”
- “Habitat Conservation Plan and Incidental Take Permit”

**FERC:**

- “hydropower license”
- “relicense” / “relicensing” (medium confidence)
- “application for a certificate of public convenience and necessity”

**Presidential permits:**

- “Amendment to Presidential Permit”
- “(Amendment to) Presidential Permit” (Tier 4 broader form; covers both standalone and amendment)
- “Issuance of Presidential Permit PP-[number]”
- “Presidential Permit Application Review” (Tier 1b only)

**NRC nuclear licenses:**

- “Early Site Permit”
- “Combined License”
- “License Renewal” / “Subsequent License Renewal”
- “issuance of renewed facility operating licenses”
- “NRC/FERC” + “license amendment” within same passage (80 chars) — OR — “license amendment” + “NRC/FERC/10 CFR 50.90/FERC order” within same passage (Tier 4 only)

**Document title (Tier 2 scan):**

- “(Standard) Individual Permit Application”
- “Hydropower License”
- “Incidental Take Permit”
- “Presidential Permit”
- “Early Site Permit”
- “Combined License”
- “(Subsequent) License Renewal”
- “certificate of public convenience and necessity”

**Agency metadata (Tier 1a — detected without text cues):**

- Agency is FERC / Federal Energy Regulatory Commission → auto-assigned `federal_permit`
- Agency is FAA / Federal Aviation Administration → auto-assigned `federal_permit`
- Agency is FCC / Federal Communications Commission → auto-assigned `federal_permit`


### 4. Federal Action 

---

**Coding rule:** `Assign federal action coding when the federal agency is not just approving someone else’s project but is the primary actor proposing, implementing, constructing, adopting, managing, or directly sponsoring the action.`

**Recognized federal actors** — the following agency names are required to appear in any compound pattern below:

- DOE / Department of Energy
- NNSA / National Nuclear Security Administration
- BPA / Bonneville / Bonneville Power Administration
- WAPA / Western / Western Area Power Administration
- Reclamation / Bureau of Reclamation / USBR
- CBP / U.S. Customs and Border Protection
- Forest Service / U.S. Forest Service / USFS
- Bureau of Land Management / BLM
- NPS / National Park Service
- PNNL / Pacific Northwest National Laboratory

**Compound patterns (actor + intent verb + action verb — all must appear in same passage):**

Intent verbs recognized: “proposes to” / “propose to” / “is proposing to” / “will” / “would” / “would be to” / “is to”

Action verbs recognized: construct, install, build, operate, implement, manage, restore, undertake, develop, upgrade, expand, demolish, replace, retrofit, rebuild, reconductor, renovate, refurbish, relocate, repair, reconfigure, dismantle, modernize, improve

The compound patterns are:

- [ACTOR] + (within 80 chars) + [INTENT VERB] + [ACTION VERB]
- [ACTOR] + (within 80 chars) + [INTENT VERB] + “remove and replace”
- [ACTOR] + (within 120 chars) + “construct, own, operate, and maintain”
- [ACTOR] + (within 160 chars) + “constructed and operated” / “would be constructed and operated”
- [ACTOR] + (within 160 chars) + “continue to occupy and maintain existing facilities” + (within 180 chars) + “refurbish existing facilities”
- [ACTOR] + (within 160 chars) + “would functionally replace”
- [ACTOR] + (within 520 chars) + “rebuild the existing”
- [ACTOR] + (within 200 chars) + “upgrade/rebuild” + (within 160 chars) + “by removing” + (within 160 chars) + “and installing”

**Specific literal pattern (no verb structure required):**

- “now that DOE has acquired ownership of the parcel, DOE proposes to operate and maintain the site”

**Specific literal pattern (no verb structure required):**

- “now that DOE has acquired ownership of the parcel, DOE proposes to operate and maintain the site”

**Standalone patterns (no actor name required):**

- “federal construction” / “federal facility” / “federal installation”
- “military installation” / “military base” / “military facility” / “military construction”
- “federal facility upgrade” / “federal facility expansion” / “federal facility construction” (Tier 1b; more specific form)
- “vegetation management” + “National Forest” within same passage (~50 chars) (Tier 1b only)

**Agency metadata (Tier 1a — detected without text cues):**

- **`AGENCY_ACTION_PRIOR_MAP`** — sets a prior toward `federal_action`; continues through Tier 1b/2/3/4 for confirmation:
  - Power Marketing Administration
  - Bonneville Power Administration / BPA
  - Western Area Power Administration / WAPA

- **`AGENCY_ACTION_ONLY_MAP`** — border infrastructure is always direct federal construction; continues through tiers:
  - CBP / U.S. Customs and Border Protection

- Note: DOE and USACE appear in `FEDERAL_ACTION_ACTOR_PATTERN` (used in compound text patterns above) but are classified as `AGENCY_AMBIGUOUS` in Tier 1a — they are NOT auto-assigned to `federal_action` from metadata alone; verb context from document text is required to distinguish `federal_action` from `federal_funding` (DOE) or `federal_permit` (USACE)


### 5. Federal programs

---

**Coding rule:** `Assign the federal program classification when the NEPA action is not primarily a site-specific project, but a broader plan, rule, policy, or program rather than a single facility or construction project.`

**Programmatic document identifiers (full phrases):**

- “programmatic environmental impact statement”
- “programmatic environmental assessment”
- “Draft Programmatic Environmental Impact Statement” / “Final Programmatic Environmental Impact Statement” / “Supplemental Programmatic Environmental Impact Statement”
- “this programmatic EIS” / “this programmatic EA” / “this programmatic Environmental...”

**Programmatic acronyms and abbreviations:**

- “PEIS” / “DPEIS” / “FPEIS” / “SPEIS”
- “PEA” (programmatic environmental assessment)
- “SWEIS” (site-wide environmental impact statement)
- “SWEA” (site-wide environmental assessment)
- “program-wide” / “programwide”

**Generic and tiered review types:**

- “generic environmental impact statement” / “generic environmental assessment” / “generic EIS” / “generic EA”
- “Tier 1 review” / “Tier 1 NEPA review” / “Tier 1 environmental impact statement” / “Tier 1 EIS” / “Tier 1 EA” / “Tier i review” / “Tier one review”
- “Environmental Impact Statement Tier 1” / “EIS Tier 1” / “EA Tier 1” (reversed order)
- “site-wide environmental impact statement” / “site-wide environmental assessment” / “sitewide environmental assessment”

**Resource and land management plans:**

- “resource management plan amendment” / “resource management plan revision”
- “resource management plan” (standalone, in title or description)
- “revision of the ... land and resource management plan” (within same passage)
- “final ... land and resource management plan” / “proposed ... land and resource management plan” (medium confidence)

**Other program types:**

- “leasing program” / “leasing framework”
- “corridor designation”
- “rulemaking”
- “policy framework”

**Named federal energy programs (compound patterns — both terms must appear in same passage):**

- “integrated resource plan” + “programmatic environmental impact statement” / “supplemental environmental impact statement” / “draft EIS”
- “integrated vegetation management program” + “programmatic environmental assessment”
- “system-wide operations and maintenance” + “programmatic environmental assessment”
- “uranium leasing program” + “programmatic environmental assessment”
- “outer continental shelf oil and gas leasing program” + “programmatic environmental impact statement”
- “solar energy development in six southwestern states” + “programmatic environmental impact statement”
- “wind energy development on Bureau of Land Management-administered lands” + “programmatic environmental impact statement”
- “updates to the western solar plan” + “solar PEIS” / “programmatic environmental impact statement” — OR — “2023 draft solar PEIS”
- “designation of energy corridors on federal land” + “programmatic environmental impact statement”
- “section 368 energy corridor revisions” + “resource management plan amendment” / “environmental impact statement”
- “long-term experimental and management plan” + “environmental impact statement”

**Document title (Tier 2 scan):**

- “programmatic” / “program-wide” / “PEIS” / “PEA” appearing in the document title (with exclusion check applied)

**Excluded patterns:**

- “programmatic agreement” → refers to a Section 106 consultation agreement, not a NEPA program review
- “programmatic biological opinion” → a Fish and Wildlife consultation document, not a NEPA program review
- “programmatic consultation” → inter-agency coordination, not NEPA
- “programmatic collaboration” → not NEPA

### 6. Federal property transaction 

---

**Coding rule:** `Assign federal property transaction coding when the federal action involves transfer or disposition of federally controlled land or property interests`.

**Land exchange:**

- “land exchange”
- “fee-for-fee land exchange”
- “exchange property with”
- “asset exchange” + “rights-of-way/easements/line easements” within same passage (medium confidence)
- “easement exchange” (document title only)

**Disposal:**

- “dispose of land rights” / “dispose of the underlying land rights”
- “land disposal”
- “disposal of federal land” / “disposal of federal property” (Tier 1b: requires “of federal” qualifier)
- “disposal” standalone (Tier 4 context scan — lower specificity)

**Sale:**

- “sale of land rights”
- “sell in fee”

**Conveyance:**

- “conveyance of federal land” / “conveyance of federal property” (Tier 1b: requires “of federal” qualifier)
- “conveyance” standalone (Tier 4 context scan — lower specificity)

**Acquisition of land or easements:**

- “acquire land rights”
- “acquire several road easements”
- “acquire access road rights”
- “acquire and release access road rights”
- “purchase lots and easements” / “purchase two lots and easements” / “purchase lots and line easements”
- “land purchase and easement acquisition” (document title only)
- “land rights acquisition” (document title only)

**Transfer of ownership or title:**

- “transfer ownership” + “easements/rights-of-way/associated easements/land rights” within same passage
- “title transfer” + “easements/rights-of-way/land rights” within same passage
- “title transfer” (document title — standalone)
- “parcel transfer” (medium confidence)
- “property transfer” near “transmission line” or “substation” in document title (contextual; document title only)

**Document title (Tier 2 scan):**

- “land exchange”
- “land disposal”
- “sale of land rights”
- “land purchase and easement acquisition”
- “land rights acquisition”
- “easement exchange”
- “title transfer”
- “transmission line ... property transfer” / “substation ... property transfer” (within ~40 characters in title)

**Excluded patterns:**

- “acquired the property as part of ... land exchange” — past acquisition context, not the current action
- “completed a NEPA review of the land exchange” — historical reference to a prior review
- “no transfer of land ownership” — explicitly negates a transfer
- “only change would be in ownership of assets” — asset change without land conveyance
- “land exchanges, withdrawals, and the implementation of RMP” — background policy mention in an RMP review
- “disposal of land parcels” / “disposals of land parcels” — generic land management language, not a discrete transaction
- “land exchange could lower/allow/play a role/be considered” — conditional future reference, not a proposed action
- “land exchanges are considered on a case-by-case basis” — policy description, not a specific proposed action


### 7. Unknown or unclear 

--- 

**Coding rule:** `Assign the unknown classification only after attempting structured extraction from title, description, process metadata, agencies, and document text and the available materials do not allow reliable identification of the triggering nexus.`

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