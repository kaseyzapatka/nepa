---
title: "D1: Examples"
---

Use this file to collect a small set of representative text chunks for Tier 4 calibration.

## How to use these examples

These examples are **not** training data — the NLI model is zero-shot and runs without them. They serve two specific purposes:

### 1. Validate NLI hypothesis templates

Before running at scale, score each positive example through the NLI model using the hypothesis templates defined in `tier4_refactor_spec.md`. The correct class hypothesis should score **≥ 0.75**. If a positive example scores below this on its correct class, adjust the hypothesis wording before proceeding.

```python
# Quick validation loop (run once before full corpus)
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")

chunk = "DOE is proposing to provide federal funding..."
hypothesis = "This text shows that a federal agency is funding this project."
score = model.predict([[chunk, hypothesis]])[0]  # entailment score
assert score >= 0.75, f"Positive example scored too low: {score:.3f}"
```

### 2. Calibrate auto-accept thresholds

Run hard negative examples (boilerplate chunks under "Hard negatives") through the NLI model. All class hypotheses should score **≤ 0.50** for a well-calibrated negative. If a boilerplate chunk scores above 0.50 on any class, either tighten the hypothesis wording or raise the auto-accept threshold in the spec (default 0.90).

**Do this calibration on the example bank before any full-corpus run.** It takes minutes and prevents silent miscalibration at scale.

## Targets

- at least `3-5` clean positives for each positive class
- at least `5-10` entries across the ambiguous rule families
- at least `10` hard negatives

## Required fields for every example

- `Rule family`
- `Project ID`
- `Suggested class`
- `Why`
- `Chunk`

## Positive classes

### federal_funding

#### Example 1

```text
Rule family: T2_doc_title_loan_guarantee
Project ID: 3153f73d56c98c8ccf40db525981fe01
Suggested class: federal_funding

Why: The document title explicitly says the NEPA review is for a Department of Energy loan guarantee. This is a clean funding-trigger example and useful as a high-confidence positive.

Chunk: FINAL ENVIRONMENTAL ASSESSMENT Volume I for Department of Energy Loan Guarantee to High Plains II, LLC for the California Valley Solar Ranch Project
```

#### Example 2

```text
Rule family: T3_agency_grant
Project ID: 251d4f28-0319-d49b-7044-ccb53c78e79d
Suggested class: federal_funding

Why: The award language directly identifies DOE funding and cost share for the project. This is strong affirmative federal_funding evidence.

Chunk: NEPA PROVISION
DOE has made a final NEPA determination for this award

DOE Funding = $6,999,959
Cost Share = $20,999,876
Total Project Cost = $27,999,835
```

#### Example 3

```text
Rule family: T1b_agency_grant
Project ID: f5f17760-cb4b-6fb3-2d6c-b0bb1d393c58
Suggested class: federal_funding

Why: The text says DOE is proposing to provide federal funding and then explains what the DOE funding would be used for. That is exactly the kind of project-specific funding language Tier 4 should treat as positive evidence.

Chunk: Rational for determination:
DOE is proposing to provide federal funding to the Contra Costa Economic Partnership to support local and regional efforts to address and achieve measurable improvements in market conditions for both commercial and residential rooftop photovoltaic (PV) solar arrays.

DOE funding would be used to develop and implement a transparent, consistent, and expedient permitting and interconnection process for residential and small commercial rooftop PV systems throughout all participating jurisdictions.
```

### federal_action

#### Example 1

```text
Rule family: T1a_DOE_action
Project ID: 55785ab38769051f617b47da1931e34f
Suggested class: federal_action

Why: The description presents DOE as the proposing agency for direct construction and operation work at NREL. This is the type of DOE-led facility action that should support a true federal_action label.

Chunk: The Department of Energy (DOE) prepared this Final Supplemental EA to assess the potential environmental effects resulting from the proposed improvements to the RFHP. Specifically, the DOE proposes to develop, construct and operate a woodchip fuel storage silo at the National Renewable Energy Laboratory’s (NREL) South Table Mountain (STM) site in Golden, Colorado.
```

#### Example 2

```text
Rule family: T1a_DOE_action
Project ID: 05233996-c065-d1d6-4b6a-aebd57c4282f
Suggested class: federal_action

Why: Western is not just funding or authorizing someone else here. The text says Western itself will construct, demolish, and install substation infrastructure. That is clean direct-action language.

Chunk: Western Area Power Administration (Western) will construct a new control building at the Lusk Rural Substation (LRS) located in Niobrara County, Wyoming. The proposed work at the LRS control building consists of the following; construct a new control building and associated foundation, demolish existing 69-kV switch, construct new Fault Interrupter foundations and install steel support structure and fault interrupter, and demolish existing control building.
```

#### Example 3

```text
Rule family: T1a_DOE_action
Project ID: cd41bb62-9377-d1e4-8038-0d32c2001811
Suggested class: federal_action

Why: This is another clean direct-action example because Western is the entity that will construct the communications building and do the associated site work.

Chunk: Western Area Power Administration (Western) will construct a new communications building on the Archer Microwave Site (ARW). This project will have the following components:
* Construct a new communications building
* Rebuild the fence along the existing fence line
* Conduct the site work necessary to improve the driveway
```

### federal_land

#### Example 1

```text
Rule family: T1b_special_use
Project ID: 8a1ff4621f3b80f0a5c2be15f3649717
Suggested class: federal_land

Why: The context says “The USFS purpose and need is to determine whether to issue a special use permit” and that “the USFS would bring Western's facilities under a current authorization with a defined ROW.” That is federal land access / right-of-way authorization language, which is why this is a good federal_land example.

Chunk: Forest Service Purpose and Need The USFS purpose and need is to determine whether to issue a special use permit for the proposed transmission lines upgrade and rebuild. In conjunction with the issuance, the USFS would bring Western's facilities under a current authorization with a defined ROW and an Operation and Maintenance Plan.
```

#### Example 2

```text
Rule family: T1b_row_grant
Project ID: 74d906df-095c-24b1-3055-8b96bcdcc5b8
Suggested class: federal_land

Why: The project-specific language is about acquiring a perpetual right-of-way grant across a parcel now owned by BLM. This is a clean federal land authorization case.

Chunk: BPA proposes to acquire a perpetual right-of-way grant for BPA's existing Wautoma-Rock Creek transmission line across a parcel of land in Klickitat County, Washington. Originally BPA had acquired a 50-year easement for the right-of-way from Yakama Tribal Allottees. However, the easement has expired and the Bureau of Land Management now owns the parcel.
```

#### Example 3

```text
Rule family: T1b_row_grant
Project ID: b2b71bb4-5152-128c-139f-67a63a83ef3b
Suggested class: federal_land

Why: This is a straightforward right-of-way renewal on tribal land through a federal land-management authority. The trigger is land access/renewal, not federal funding.

Chunk: A new right-of-way grant from the Bureau of Indian Affairs for a 25 year term from March 29, 2016 through March 28, 2041, with the right to extend the right-of-way for an additional 25 years thorough March 28, 2066. The line has not moved since constructed in the early 1960's, this is just a right-of-way renewal.
```

### federal_permit

#### Example 1

```text
Rule family: T3_npdes
Project ID: 2dbec454-c1d5-a8fa-4819-0fc9d6a646c4
Suggested class: federal_permit

Why: The evidence directly says an NPDES permit for construction activities must be obtained. This is affirmative permit-trigger language and a strong positive example.

Chunk: This would entail obtaining the National Pollutant Discharge Elimination System (NPDES) permit for Construction activities. Develop and implement a Stormwater Pollution Prevention (SWPP) Plan/Temporary Erosion Sediment (TESC) Plan to limit project impacts.
```

#### Example 2

```text
Rule family: T3_npdes
Project ID: da5f8822-d72d-a8ed-cc60-106633f5bff9
Suggested class: federal_permit

Why: This is explicit conditional permit language tied to the project work. It names the permit and the legal authority.

Chunk: If Task 7.2 "Perform Data Acquisition" requires discharge into that stilling basin, a NPDES permit must be obtained to comply with Section 402 of the Clean Water Act.
```

#### Example 3

```text
Rule family: T2_doc_title_permit_app
Project ID: d8ce951d6f48a7914a867e2ee6f6fa88
Suggested class: federal_permit

Why: The document title itself is a permit application document, which is a strong permit-trigger cue.

Chunk: Department of the Army Environmental Assessment and Statement of Findings for the Above-Referenced Standard Individual Permit Application
```

### federal_program

#### Example 1

```text
Rule family: T2_doc_title_peis
Project ID: debe659941dc65ed630daab88d5fbf81
Suggested class: federal_program

Why: The title explicitly says “Programmatic Environmental Assessment,” which is the clearest possible program-level cue.

Chunk: Parker-Davis Transmission System Routine Operation and Maintenance Project and Proposed Integrated Vegetation Management Program Programmatic Environmental Assessment
```

#### Example 2

```text
Rule family: T1b_programmatic_title
Project ID: faec045d0d990209b8aada96a616548f
Suggested class: federal_program

Why: This is another direct programmatic title, centered on system-wide operations and maintenance plus an integrated vegetation management program.

Chunk: Programmatic Environmental Assessment for System-wide Operations and Maintenance Activities and Integrated Vegetation Management Program
```

#### Example 3

```text
Rule family: T2_doc_title_peis
Project ID: 19c9535e21071f0bcf98e55248a4025d
Suggested class: federal_program

Why: The title identifies the document as a Draft Programmatic Environmental Impact Statement, which is a clean federal_program example.

Chunk: Upper Great Plains Wind Energy Draft Programmatic Environmental Impact Statement
```

### federal_property_transaction

#### Example 1

```text
Rule family: T1b_land_exchange
Project ID: aad5c0f845b958c77a8bfaa9bb88d1e7
Suggested class: federal_property_transaction

Why: The project title itself explicitly identifies a land exchange. That is direct property-transaction language.

Chunk: Falls Creek Hydroelectric Project and Land Exchange
```

#### Example 2

```text
Rule family: T1b_land_exchange
Project ID: a0acb155-1e65-6f3b-a54c-e894e735429c
Suggested class: federal_property_transaction

Why: This example combines the generic property-transfer category language with an explicit DOE land exchange. It is a strong property-transaction positive.

Chunk: B1.24 Property transfers Transfer, lease, disposition, or acquisition of interests in personal property (including, but not limited to, equipment and materials) or real property (including, but not limited to, permanent structures and land).

Rationale for determination:
The U.S. Department of Energy (DOE) is proposing to conduct a multi-party land exchange with Jefferson County Open Space (JCOS) and the State of Colorado (the State).
```

#### Example 3

```text
Rule family: T1b_land_exchange
Project ID: daa6b3b3dc506abd01afbbfaf6ca0ffb
Suggested class: federal_property_transaction

Why: The evidence ties the selected village site to a prior land exchange approved by Congress. That is a clean property-transaction cue.

Chunk: This final Environmental Impact Statement (EIS) describes a number of alternatives in a historical context for the purpose of illustrating how the long-term evolution of the project led to the selection of a new village site to be constructed at Mertarvik on Nelson Island, a site granted to the village in a land exchange approved by the U.S.
```

## Ambiguous / boundary rule families

These are not simple positives or negatives. Save chunks that help distinguish between plausible classes.

### DOE funding vs federal_action

#### Example 1

```text
Rule family: T1a_DOE_funding
Project ID: 97fca3873618664c1ca208c1874ebff7
Suggested class: boundary case between federal_funding and federal_action

Why: The title and opening language frame the project as DOE funding, but the description also says DOE’s proposed action under NEPA is the design, construction, and operation of the FRIB. This is a useful example of why DOE metadata alone is not enough.

Chunk: DOE published a “funding opportunity announcement” (FOA) on May 20, 2008, seeking applications for the conceptual design and establishment of a particle acceleration facility—the FRIB. Subsequent to construction, funding for operations would be allocated annually based on Congressional appropriations. DOE’s proposed action under NEPA is the design, construction, and operation of the FRIB.
```

#### Example 2

```text
Rule family: T1a_DOE_funding
Project ID: 77eb1b37de8501f37cc21e7990aff341
Suggested class: boundary case leaning federal_funding

Why: This is still mostly a funding case, but the text is full of plant-construction detail. It is useful for teaching Tier 4 the difference between DOE financial assistance and DOE direct construction.

Chunk: The U.S. Department of Energy (DOE) proposes, through a cooperative agreement with Honeywell International Inc. (Honeywell), to partially fund the construction of a manufacturing plant to produce a critical battery material, lithium hexafluorophosphate (LiPF6). If approved, DOE would provide approximately 50 percent of the funding for the project.
```

#### Example 3

```text
Rule family: T1a_DOE_funding
Project ID: 00b2f4e33f9e91f7ffe1401338859099
Suggested class: boundary case leaning federal_funding

Why: The project includes design, installation, and demonstration of a biomass boiler center, but DOE’s role is still framed as partial funding through a cooperative agreement.

Chunk: The United States Department of Energy (DOE) proposes through a cooperative agreement with Burns & McDonnell Engineering, to partially fund project activities to design, install, and demonstrate an innovative biomass boiler pilot project. Under the terms of the cooperative agreement, DOE would provide $1,655,945 for Burns & McDonnell Engineering to facilitate the development and demonstration of a biomass energy center at the Frito-Lay manufacturing plant.
```

### Section 404: true trigger vs boilerplate

#### Example 1

```text
Rule family: T1b_sec404
Project ID: 56f87d34b0a565f24bcdb727fba09c18
Suggested class: federal_permit

Why: This is a true Section 404 trigger. The text explicitly says USACE would issue a permit pursuant to Section 404 and Section 10 for site dredge and fill activities.

Chunk: The proposed actions related to the PSEG application are (1) the NRC issuance of an ESP for the PSEG Site and (2) the USACE issuance of a permit pursuant to Section 404 of the Federal Water Pollution Control Act [Clean Water Act (CWA)] and Section 10 of the Rivers and Harbors Appropriation Act of 1899, as amended, to perform certain dredge and fill activities on the site.
```

#### Example 2

```text
Rule family: T1b_sec404
Project ID: 20871b5a-ddcb-9303-27d5-fea2681d67c0
Suggested class: boundary case leaning unknown

Why: This example mentions Section 404 in a real project context, but the text is about reviewing applicability, and the desc later says USACE determined a 404 permit was not required. It is useful for separating mention from actual permit trigger.

Chunk: DOE submitted the project file and the "limited environmental assessment" to the Army Corps of Engineers (Mobile District) for their review and determination of the applicability of Section 404 of the Clean Water Act to the proposed project. On a letter dated Oct. 22nd 2010, USACE determined that the proposed project would not require a 404 permit.
```

### ARRA: true funding trigger vs generic mention

#### Example 1

```text
Rule family: T3_arra
Project ID: 5d988327-5b0c-2e21-8c87-12c04a3f2cb5
Suggested class: federal_funding

Why: This is a useful positive comparator because the text does not merely mention ARRA. It says ARRA appropriates funding for DOE to issue grants under EECBG.

Chunk: The American Recovery and Reinvestment Act of 2009, Public Law 111-5, appropriates funding for the Department of Energy (DOE) to issue/award formula-based grants to states, U.S. territories, units of local government, and Indian tribes under the Energy Efficiency and Conservation Block Grant (EECBG) Program.
```

#### Example 2

```text
Rule family: T3_arra
Project ID: 445a359b-b00d-f375-2e97-44953a83665b
Suggested class: boundary case leaning unknown

Why: This is mostly form metadata. It has a checked ARRA box, but the chunk does not itself explain the federal funding nexus for this specific project.

Chunk: Department of Energy
Categorical Exclusion Determination Form

Program or Field Office: Energy Efficiency and Conservation Block Program
Project Title: Energy efficiency lighting and plumbing retrofits for several City facilities.

American Recovery and Reinvestment Act: [x]
```

### BLM / USFS land vs action

#### Example 1

```text
Rule family: T1a_BLM_action
Project ID: 6997b8a719abdf2633b890254d964252
Suggested class: boundary case between federal_land and federal_action

Why: This text is why BLM is difficult. It is about federal conveyances and perpetual ROWs, so it has both land-authorization and federal-decision language.

Chunk: In some of the conveyances, the BLM may establish terms and conditions, reserve access through the issuance of perpetual rights of way (ROWs), and/or convert ROWs to perpetual terms for the conveyance parcels. The purpose of the proposed action is to analyze whether or not to 1) establish terms and conditions, 2) reserve public access through the issuance of perpetual ROWs, and/or 3) convert several existing ROWs to perpetual terms.
```

#### Example 2

```text
Rule family: T1a_BLM_land
Project ID: a5ee94b8e14073d5ca1ae39925d551c6
Suggested class: federal_land

Why: This is the cleaner side of the same boundary. The excerpt is centered on granting a new right-of-way and analyzing the use of that ROW.

Chunk: includes granting PG&E a new right-of-way for the existing distribution line and updating the underground alignment. As per 40 CFR 1501.3, this Environmental Assessment (EA) has been prepared to disclose and analyze the environmental consequences of the use of a ROW proposed by PG&E.
```

### RMP: programmatic action vs plan-conformance

#### Example 1

```text
Rule family: T3_rmp
Project ID: 481ef767-d681-a1f2-fbe3-d50199c5c336
Suggested class: boundary case between federal_program and federal_land, leaning federal_land

Why: The chunk looks like a programmatic RMP hit, but it is really plan-conformance language supporting a transmission line amendment and associated rights-of-way. This is a good cautionary example.

Chunk: Land Use Plan Conformance

Land Use Plan Name: Spokane Resource Management Plan (RMP)

The proposed action is in conformance with the Spokane RMP because it is clearly consistent with the following RMP objectives, terms, and conditions: Keep public lands open for exploration/development of mineral resources, rights-of-way, access, and other public purposes.
```

## Hard negatives

Use this section for text that looks relevant but should not drive classification.

### Boilerplate / checklist negatives

#### Example 1

```text
Rule family: T3_sec404
Project ID: 98eb050f-8691-cb6b-0df5-f5f546ed1599
Suggested class: unknown

Why: This is CE checklist boilerplate from a DOE form. It mentions “Section 404 permit,” but only as an unchecked checklist item rather than affirmative evidence that Section 404 permitting triggered NEPA for this project.

Chunk: Conservation, Fossil, and Renewable Energy Activities
[ ] B5.3 - Modification (not expansion)/abandonment of oil storage access/
brine injection/gas/geothermal wells; no site closure
[ ] B5.4 - Repair/replacement of pipeline sections within maintenance
provisions of a Section 404 permit
[ ] B5.5 - Short crude oil/gas/steam/geothermal pipeline const/oper within a
single industrial complex/existing right-of-way
```

#### Example 2

```text
Rule family: T3_sec404
Project ID: 66feacb8-3138-57e0-2755-4725e7d53625
Suggested class: unknown

Why: Same false-positive pattern as Example 1. The project is about developing Li-ion cells, but the hit comes from recycled CE checklist text rather than project-specific permit language.

Chunk: Conservation, Fossil, and Renewable Energy Activities
[ ] B5.4 - Repair/replacement of pipeline sections within maintenance
provisions of a Section 404 permit
```

#### Example 3

```text
Rule family: T3_sec404
Project ID: 2b4fa793-82cb-a468-4170-ce11f7b2b84c
Suggested class: unknown

Why: This one is especially useful because one CE category is checked, which makes the form look more “real,” but the Section 404 line is still unchecked boilerplate and should not drive classification.

Chunk: **Conservation, Fossil, and Renewable Energy Activities**
[x] B5.1 - Actions to conserve energy, no indoor air quality degradation
[ ] B5.4 - Repair/replacement of pipeline sections within maintenance
provisions of a Section 404 permit
```

#### Example 4

```text
Rule family: T3_sec404
Project ID: 262cc267-a46e-6a84-b1e0-fac687f38484
Suggested class: unknown

Why: Another unchecked CE checklist hit. The project is about subsurface microbial characterization, so the Section 404 line is clearly non-evidence.

Chunk: Conservation, Fossil, and Renewable Energy Activities
B5.4 - Repair/replacement of pipeline sections within maintenance provisions of a Section 404 permit [ ]
```

#### Example 5

```text
Rule family: T3_sec404
Project ID: 8a2aace3-98a0-e6e2-23a7-7c542626a767
Suggested class: unknown

Why: The project is about redundant booster pumps. The Section 404 phrase appears only because the CE form lists a boilerplate category.

Chunk: **Conservation, Fossil, and Renewable Energy Activities**
[ ] B5.4 - Repair/replacement of pipeline sections within maintenance provisions of a Section 404 permit
```

### Generic grant / ARRA form negatives

#### Example 1

```text
Rule family: T3_arra
Project ID: 9929ff24-2841-aeb9-4c3a-1b644ead67c0
Suggested class: unknown

Why: This is form scaffolding, not project-specific funding evidence. The ARRA box is blank and the chunk is just a generic EECBG form header.

Chunk: Department of Energy
Categorical Exclusion Determination Form

Program or Field Office: Energy Efficiency and Conservation Block Grant Program
Project Title MD-City-Bowie

Proposed Action or Project Description
American Recovery and Reinvestment Act: [ ]
```

#### Example 2

```text
Rule family: T3_arra
Project ID: 445a359b-b00d-f375-2e97-44953a83665b
Suggested class: unknown

Why: Even with a checked ARRA box, this chunk is still mostly template text. It does not state that ARRA or DOE funding is the specific trigger for NEPA review.

Chunk: Department of Energy
Categorical Exclusion Determination Form

Program or Field Office: Energy Efficiency and Conservation Block Program
Project Title: Energy efficiency lighting and plumbing retrofits for several City facilities.

American Recovery and Reinvestment Act: [x]
```

### Plan / compliance negatives

#### Example 1

```text
Rule family: T3_rmp
Project ID: 152b468c-9cab-767d-3fdb-150ce1e2e8d1
Suggested class: unknown

Why: This is plan-conformance review text, not evidence that the project itself is a federal program. The real project is a temporary ROW / permit action.

Chunk: PART II – PLAN CONFORMANCE REVIEW
This proposed action is subject to the following land use plan(s):
Safford District Resource Management Plan (RMP and Record of Decision (September 1992).

Land use authorizations (rights-of-way, leases, permits, easements) will continue to be issued on a
case-by-case basis.
```

#### Example 2

```text
Rule family: T3_rmp
Project ID: 2c092490-2876-4923-a6b7-ef663139d05d
Suggested class: unknown

Why: “Consistent with the Forest Plan” is generic compliance language. It should not be treated as a federal_program trigger.

Chunk: COMPLIANCE WITH FOREST PLAN

The proposal is consistent with the approved Forests' Land and Resource Management Plan
(As Amended January 2012).
```

#### Example 3

```text
Rule family: T3_rmp
Project ID: 0535c33e-cfaa-48c6-1225-f6d1b007a73e
Suggested class: unknown

Why: This hit is purely from the phrase “Cultural Resource Management Plan,” which is not the kind of programmatic trigger the classifier is trying to detect.

Chunk: Idaho National Laboratory Cultural
Resource Management Plan.
```

#### Example 4

```text
Rule family: T3_rmp
Project ID: 75d0b415-8dcb-81ec-a597-0bfc603d6b49
Suggested class: unknown

Why: This is another misleading management-plan mention. It is about exemption from cultural resource review, not a federal programmatic action.

Chunk: MFC-774 is eligible for nomination to the National Register of Historic Places as a Category 3 historic property; however, the activities described in this Environmental Checklist (EC) are exempted from cultural resource review ("Idaho National Laboratory [INL] Cultural Resource Management Plan" Table 2, exemptions 2 and 8.
```

## Suggested collection order

1. DOE funding vs action
2. Section 404 positives and negatives
3. ARRA positives and negatives
4. BLM / USFS land vs action
5. RMP / programmatic cases
