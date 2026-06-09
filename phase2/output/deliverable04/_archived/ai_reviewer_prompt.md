# AI Reviewer Prompt — D4 Timeline QC

Paste this prompt (with the CSV data appended) into Claude or Codex.

---

You are acting as a careful human quality-control reviewer for NEPA project timeline extraction. Your reviewer ID is **reviewer_3**.

For each project in the CSV data below, determine the best-supported **initiation date** and **decision date** by independently evaluating all available evidence. Do not simply accept the suggested dates.

---

## Definitions

**Initiation date** — the earliest date the project formally entered the NEPA review process:
- For EIS: Federal Register Notice of Intent (NOI) publication date
- For EA: first scoping notice, draft EA publication, or formal agency request received
- For CE: agency NEPA Register start date, or application/request received date

**Decision date** — the date of the final NEPA determination:
- For EIS: Record of Decision (ROD) signing date
- For EA: FONSI or Decision Notice signing date
- For CE: authorizing official signature date on the CE determination

---

## Evidence Priority Rules (apply in order — stop at first match)

### Initiation (highest → lowest priority)
1. Federal Register NOI publication date
2. BLM / DOE agency NEPA Register project start date
3. Formal public scoping notice or scoping period start date
4. Draft EA / scoping letter publication date
5. Formal application or request received date (stated in the NEPA document)
6. Document body date referencing EA/EIS preparation start
7. Document cover/header month only → use YYYY-MM-01

### Decision (highest → lowest priority)
1. Authorizing official digital or wet signature with explicit date
2. BLM / DOE agency NEPA Register decision date
3. DOE CX Register or NEPA Workflow determination date
4. ROD / FONSI / Decision Notice signing date stated in document body
5. Federal Register Notice of Availability for ROD or FONSI
6. 30-day post-FEIS waiting period expiry (= FEIS NOA date + 30 days)
7. Final EA / FEIS publication month → use YYYY-MM-01 (note: this precedes the ROD)
8. Document cover/header month only → use YYYY-MM-01

---

## Hard Rules

- **Never use** Section 106 (cultural resources) or Section 7 (ESA) consultation dates as NEPA initiation or decision dates.
- **Never use** state permits, CEQA decisions, court orders, or legislative authorizations as NEPA decisions.
- **Never use** a DEIS publication date as a decision date.
- For multi-agency projects, record the **lead agency's** decision, not a cooperating agency's parallel action.
- If a document header says "July 2019" but a signature in the body says "2019.07.16," use 2019-07-16.
- If only a month is known (no day in text), record YYYY-MM-01.
- **Leave blank** only when all available evidence is pre-NEPA activity (surveys, proposals, state permits). If Tier 5–7 evidence exists, record it and note the uncertainty.

---

## Candidate String Format

Each candidate in `top_initiation_candidates` and `top_decision_candidates` looks like:

```
YYYY-MM-DD [type|confidence|score=N] doc=DOCTYPE | supporting text
```

- `clear_initiation` / `clear_decision` = strong signal; prefer over proxy
- `proxy_initiation` / `proxy_decision` = weak signal; use only if no clear candidates
- `confidence`: high > medium > low
- `doc=None` = date came from a register (most authoritative); other doc types are from extracted text

---

## Output Format

Return a JSON array. One object per project. Include all 100 projects — do not skip any.

```json
[
  {
    "sample_id": "1",
    "review_initiation_date": "YYYY-MM-DD or empty string",
    "review_initiation_source": "Direct quote or register reference that supports the date",
    "review_initiation_notes": "1-2 sentences: what evidence you used, why you rejected alternatives",
    "review_decision_date": "YYYY-MM-DD or empty string",
    "review_decision_source": "Direct quote or register reference that supports the date",
    "review_decision_notes": "1-2 sentences: what evidence you used, why you rejected alternatives",
    "reviewer": "reviewer_3"
  },
  ...
]
```

---

## Calibration Examples

Study these before starting. They show correct application of the rules.

**[SID 44] CE — L18K Earthscope Seismic Station Assignment (BLM)**
- Init: `2019-03-18` — BLM NEPA Register start date (doc=None, Tier 2 rule)
- Dec:  `2019-04-16` — "Bonnie Million Anchorage Field Manager -4/16/2019 Date" (authorizing official signature, Tier 1). The July 2019 proxy candidates are the document header month — overridden by the explicit Field Manager date in the body.

**[SID 12] EIS — Durham-Orange Light Rail Transit (FTA)**
- Init: `2012-04-03` — "A Notice of Intent (NOI) was published in the Federal Register on April 3, 2012" (Tier 1)
- Dec:  *(blank)* — project cancelled before ROD; DEIS candidates are not decision dates

**[SID 38] EA — Beale WAPA Interconnection (Air Force / DOE)**
- Init: `2016-03-08` — "On March 8, 2016, Beale AFB submitted an interconnection request to WAPA" (Tier 5 — formal request documented in NEPA record as the trigger)
- Dec:  `2020-11-30` — "SONJA ANDERSON Digitally signed Date: 2020.11.30 12:14:03" (Tier 1 authorizing official signature). Ignore the 1977 EO dates — those are cited regulations, not decision dates.

**[SID 40] EA — East Fork Irrigation District (NRCS)**
- Init: `2019-03-12` — consultation letters sent to SHPO and tribes March 12, 2019 (Tier 4 — for NRCS EAs, tribal/SHPO consultation letters are the formal NEPA scoping start)
- Dec:  `2020-11-03` — "SCOTT ARMENTROUT Digitally signed Date: 2020.11.03 13:37:34" (Tier 1)

**[SID 29] CE — BH Buildings 802 and 812 Damage Inspection (DOE)**
- Init: *(blank)* — no application receipt or register start date in packet; CE has no identifiable initiation separate from the decision
- Dec:  `2010-01-07` — "Approved by SPRPMO NEPA Compliance Officer 01/07/10 Determination Date" (Tier 3)

---

## Project Data

```json
[
  {
    "sample_id": "1",
    "project_title": "Expanding Capabilities at the Power Grid Test Bed at Idaho National Laboratory",
    "process_type": "EA",
    "lead_agency": "Energy Programs",
    "suggested_initiation_date": "2017-05-15",
    "suggested_initiation_evidence": "DOE briefed the Heritage Tribal Office on the cultural resource evaluation for the PGTB Project during several regularly scheduled Cultural Resource Working Group meetings from May 2017 through April 2019.",
    "top_initiation_candidates": "2017-05-01 [proxy_initiation|low|score=8.0] doc=None | DOE briefed the Heritage Tribal Office on the cultural resource evaluation for the PGTB Project during several regularly scheduled Cultural Resource W ||| 2017-05-01 [proxy_initiation|low|score=8.0] doc=None | DOE briefed the Heritage Tribal Office on the cultural resource evaluation for the PGTB Project during several regularly scheduled Cultural Resource W ||| 2019-04-01 [proxy_initiation|low|score=6.2] doc=None | DOE briefed the Heritage Tribal Office on the cultural resource evaluation for the PGTB Project during several regularly scheduled Cultural Resource W ||| 2019-04-01 [proxy_initiation|low|score=6.2] doc=None | DOE briefed the Heritage Tribal Office on the cultural resource evaluation for the PGTB Project during several regularly scheduled Cultural Resource W ||| 2019-05-01 [proxy_initiation|low|score=4.2] doc=DEA | DOE/EA-2097 Draft Environmental Assessment for Expanding Capabilities at the Power Grid Test Bed at Idaho National Laboratory Draft May 2019",
    "suggested_decision_date": "2019-06-26",
    "suggested_decision_evidence": "These measures have been formalized in a memorandum of agreement (MOA) signed by DOE and Idaho SHPO June 26, 2019 (see Appendix C).",
    "top_decision_candidates": "2019-06-26 [clear_decision|medium|score=6.2] doc=EA | These measures have been formalized in a memorandum of agreement (MOA) signed by DOE and Idaho SHPO June 26, 2019 (see Appendix C)."
  },
  {
    "sample_id": "2",
    "project_title": "Coyote Springs Cogeneration Project",
    "process_type": "EIS",
    "lead_agency": "Power Marketing Administration",
    "suggested_initiation_date": "1993-02-19",
    "suggested_initiation_evidence": ".\u201d Request for Transmission from Portland General Electric Company On February 19, 1993, PGE submitted a request for transmission wheeling services from its proposed Coyote Springs Cogeneration Plant in Boardman, Oregon.",
    "top_initiation_candidates": "1993-02-19 [clear_initiation|high|score=13.9] doc=ROD | .\u201d Request for Transmission from Portland General Electric Company On February 19, 1993, PGE submitted a request for transmission wheeling services fr ||| 1993-07-01 [clear_initiation|high|score=11.3] doc=None | In July 1993, BPA published a Notice of Intent to prepare an environmental impact statement (EIS) to help decide whether to wheel power from PGE's pro ||| 1993-02-01 [proxy_initiation|low|score=10.2] doc=ROD | In February 1993, Portland General Electric Company (PGE) requested that BPA transmit power from its Coyote Springs development over the FCRTS to its ||| 1993-07-01 [proxy_initiation|low|score=8.3] doc=None | BPA conducted an EIS scoping process in June and July 1993, issued a Draft EIS in January 1994, and distributed the Final EIS in July 1994. ||| 1993-07-01 [clear_initiation|high|score=8.3] doc=FEIS | In July 1993, BPA published a Notice of Intent to prepare an environmental impact statement (EIS) to help decide whether to wheel power from PGE's pro",
    "suggested_decision_date": "1994-01-01",
    "suggested_decision_evidence": "Document filename date (rod): 1994-01-01",
    "top_decision_candidates": "1994-01-01 [clear_decision|medium|score=14.0] doc=ROD | Document filename date (rod): 1994-01-01 ||| 1994-07-29 [proxy_decision|medium|score=11.0] doc=ROD | A Notice of Availability for the Final EIS was published in the Federal Register on July 29, 1994. ||| 1994-07-01 [clear_decision|low|score=10.0] doc=ROD | Coyote Springs Cogeneration Project Morrow County, Oregon Record of Decision DOE/FEIS-0201 July 1994 ||| 1994-09-01 [clear_decision|low|score=9.8] doc=ROD | DEPARTMENT OF ENERGY September 1994 ||| 1994-09-01 [clear_decision|low|score=9.8] doc=ROD | DOE/BP-2456 September 1994 550 Final Environmental Impact Statement Coyote Springs Cogeneration Project"
  },
  {
    "sample_id": "3",
    "project_title": "Visual Impact Assessment of a Turbine Project",
    "process_type": "EA",
    "lead_agency": "Other Commissions and Boards",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "4",
    "project_title": "Upper Colorado River Special Recreation Management Area (SRMA) Withdrawal Rights-of-Way (ROWs)",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2020-01-01",
    "suggested_decision_evidence": "Document filename date (ce): 2020-01-01",
    "top_decision_candidates": "2020-01-01 [clear_decision|medium|score=8.0] doc=CE | Document filename date (ce): 2020-01-01 ||| 2021-07-01 [proxy_decision|low|score=4.0] doc=CE | Park Ave., PO Box 68 Kremmling, CO 80459 CATEGORICAL EXCLUSION Upper Colorado River Special Recreation Management Area Withdrawal Rights-Of-Way DOI-BL ||| 2020-10-15 [clear_decision|low|score=4.0] doc=CE | The existing DOI-BLM-N020-2021-0005-CX 1 withdrawal expired on October 15, 2020; The Proposed Action would close to oil and gas leasing, non-energy so ||| 2000-10-16 [clear_decision|low|score=2.5] doc=CE | 200/Monday, October 16, 2000/Notices Dated: October 3. ||| 2000-10-16 [clear_decision|low|score=2.5] doc=CE | EFFECTIVE DATE: October 16, 2000."
  },
  {
    "sample_id": "5",
    "project_title": "Heinzer Road ROW Assignment",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2017-11-27",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2017-11-27",
    "top_initiation_candidates": "2017-11-27 [clear_initiation|high|score=10.1] doc=None | BLM NEPA Register project start date: 2017-11-27 ||| 2017-03-01 [clear_initiation|high|score=8.9] doc=CE | Description of Proposed Action: In March 2017, the BLM received an application to assign AZA-027957 to Christoph V. ||| 2007-05-01 [proxy_initiation|low|score=6.0] doc=CE | Upon Raymond's death, his wife, Janet Schock, was recognized as the sole holder of the ROW in May 2007.",
    "suggested_decision_date": "2018-01-10",
    "suggested_decision_evidence": "Lopez TITLE: Field Manager DATE: 1/10/2018 Note: The signed conclusion on this compliance record is part of an interim step in the BLM's internal decision process and does not constitute an appealable decision.",
    "top_decision_candidates": "2018-01-10 [clear_decision|medium|score=6.0] doc=CE | Lopez TITLE: Field Manager DATE: 1/10/2018 Note: The signed conclusion on this compliance record is part of an interim step in the BLM's internal deci ||| 2013-08-01 [clear_decision|low|score=4.5] doc=CE | Preparer's Initials MAH AZ-1790-1 August 2013 (b) Have significant impacts on such natural resources and unique geographic characteristics as historic ||| 2013-08-01 [clear_decision|low|score=4.5] doc=CE | No new construction is Preparer's Initials MAH AZ-1790-1 August 2013 planned. ||| 2013-08-01 [clear_decision|low|score=4.5] doc=CE | AZ-1790-1 August 2013 ||| 2018-07-01 [proxy_decision|low|score=4.0] doc=CE | NEPA No.: DOI-BLM-AZ-G020-2018-0008-CX Case File No.: AZA-027957 This is an unpaved road that travels south from Hwy 82 to a private residence."
  },
  {
    "sample_id": "6",
    "project_title": "Powder River Basin Oil and Gas Project",
    "process_type": "EIS",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2002-07-15",
    "suggested_initiation_evidence": "The FS released the ROD on the FEIS and proposed LRMP for the TBNG in July 2002 to replace the 1985 plan, as amended.",
    "top_initiation_candidates": "2002-07-01 [proxy_initiation|low|score=5.3] doc=FEIS | The FS released the ROD on the FEIS and proposed LRMP for the TBNG in July 2002 to replace the 1985 plan, as amended. ||| 2002-07-01 [proxy_initiation|low|score=5.3] doc=FEIS | In the July 2002 ROD for the FEIS and LRMP revision for the TBNG, these decisions were deferred pending completion of this FEIS.",
    "suggested_decision_date": "2003-01-01",
    "suggested_decision_evidence": "Document filename date (rod): 2003-01-01",
    "top_decision_candidates": "2003-01-01 [clear_decision|medium|score=13.0] doc=ROD | Document filename date (rod): 2003-01-01 ||| 2003-01-17 [proxy_decision|medium|score=11.0] doc=ROD | On January 17, 2003, BLM released the FEIS and Proposed Plan Amendments for the Powder River Basin Oil and Gas Project. ||| 2002-01-18 [clear_decision|low|score=9.0] doc=ROD | On January 18, 2002, the Bureau of Land Manage\u00ad ment (BLM) released the Draft Environmental Impact Statement (DEIS) for the project. ||| 2002-12-20 [clear_decision|low|score=8.0] doc=ROD | Forest Service in a letter, both dated December 20, 2002, to address concerns regarding air quality impacts. ||| 2002-07-01 [proxy_decision|low|score=4.8] doc=FEIS | The FS has released a ROD, Revised LRMP, and FEIS for the TBNG (July 2002)."
  },
  {
    "sample_id": "7",
    "project_title": "Suniva Solar Project Site",
    "process_type": "EA",
    "lead_agency": "Department of Housing and Urban Development",
    "suggested_initiation_date": "2009-11-30",
    "suggested_initiation_evidence": "Cultural Resources Upon request of the State of Michigan\u2019s Historic Preservation Officer (SHPO), a cultural resources survey was completed on November 30, 2009 for the property on which the ARTisun facility would be built.",
    "top_initiation_candidates": "2009-11-30 [clear_initiation|medium|score=11.1] doc=FONSI | Cultural Resources Upon request of the State of Michigan\u2019s Historic Preservation Officer (SHPO), a cultural resources survey was completed on November ||| 2010-02-01 [clear_initiation|medium|score=6.8] doc=FONSI | As the responsible entity for completing the NEPA process, the County of Saginaw issued a Finding of No Significant Impact (FONSI) and Request for Rel ||| 2010-02-01 [clear_initiation|medium|score=6.8] doc=FONSI | 2 Thomas Township has amended its zoning ordinance, zoning map, master plan, and future land use map to create a Solar Technology and Renewable Energy ||| 2010-02-10 [clear_initiation|medium|score=6.2] doc=FONSI | The County of Saginaw posted a combined Notice to Public of No Significant Impact on the Environment and Notice to Public of Request for Release of Fu ||| 2010-02-10 [clear_initiation|medium|score=0.2] doc=EA | ITEM COMMENCE MO/DAYIYR EXPIRE MOIDAYIYR Notice of Finding of No Significant Impact (FONS!) Publication Notice of Intent to Request a Release of Funds",
    "suggested_decision_date": "2010-01-01",
    "suggested_decision_evidence": "Document filename date (fonsi): 2010-01-01",
    "top_decision_candidates": "2010-01-01 [clear_decision|medium|score=13.0] doc=FONSI | Document filename date (fonsi): 2010-01-01 ||| 2010-02-01 [proxy_decision|medium|score=11.8] doc=FONSI | The HUD EA was completed in February 2010 and analyzed the potential environmental impacts associated with the construction of Suniva, Inc.\u2019s (Suniva) ||| 2010-05-24 [clear_decision|low|score=9.0] doc=FONSI | The Tribe responded on May 24, 2010 that the area of potential effect is close to an area in which they have information indicating the presence of an ||| 2010-02-08 [clear_decision|low|score=9.0] doc=FONSI | Public Involvement in the EA Process The County of Saginaw sent a copy of the completed HUD EA and all supporting documentation to the State of Michig ||| 2010-02-01 [proxy_decision|medium|score=7.2] doc=None | The HUD EA was completed in February 2010 and analyzed the potential environmental impacts associated with the construction of Suniva, Inc.\u2019s (Suniva)"
  },
  {
    "sample_id": "8",
    "project_title": "License Renewal for Davis-Besse Nuclear Power Station, Unit 1",
    "process_type": "EIS",
    "lead_agency": "Nuclear Regulatory Commission",
    "suggested_initiation_date": "2010-11-23",
    "suggested_initiation_evidence": "ML102980688) November 23, 2010 Letter to Brian Mitch, Environmental Review Manager, OHDNR, \u201cRequest for List of Protected Species Within the Area Under Evaluation for the Davis-Besse Nuclear Power Station License Renewal Application Review\u201d (ADAMS Accession No.",
    "top_initiation_candidates": "2010-11-23 [clear_initiation|medium|score=7.5] doc=OTHER | ML102980688) November 23, 2010 Letter to Brian Mitch, Environmental Review Manager, OHDNR, \u201cRequest for List of Protected Species Within the Area Unde ||| 2010-11-23 [clear_initiation|medium|score=7.5] doc=OTHER | ML102980430) November 23, 2010 Letter to Edgar L, French, Delaware Nation, \u201cRequest for Scoping Comments Concerning the Davis-Besse Nuclear Power Plan ||| 2010-11-23 [clear_initiation|medium|score=7.5] doc=OTHER | ML1030001644) November 23, 2010 Letter to Kenneth Meshiguad, Hannahville Indian Community Council, \u201cRequest for Scoping Comments Concerning the Davis- ||| 2010-11-23 [clear_initiation|medium|score=7.5] doc=OTHER | ML1030001644) November 23, 2010 Letter to Ron Sparkman, Shawnee Tribe, \u201cRequest for Scoping Comments Concerning the Davis-Besse Nuclear Power Plant, U ||| 2010-11-23 [clear_initiation|medium|score=7.5] doc=OTHER | ML1030001644) November 23, 2010 Letter to Leaford Bearskin, Wyandotte Nation, \u201cRequest for Scoping Comments Concerning the Davis-Besse Nuclear Power P",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "9",
    "project_title": "Rulemaking for Colorado Roadless Areas",
    "process_type": "EIS",
    "lead_agency": "Forest Service",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2015-11-20 [clear_initiation|medium|score=2.5] doc=FEIS | Content Analysis Process The SDEIS comment period opened on Friday, November 20, 2015, and closed on Friday, January 15, 2016. ||| 2016-01-15 [clear_initiation|medium|score=2.5] doc=FEIS | Content Analysis Process The SDEIS comment period opened on Friday, November 20, 2015, and closed on Friday, January 15, 2016. ||| 2015-11-20 [clear_initiation|medium|score=2.5] doc=FEIS | Rulemaking for Colorado Roadless Areas E-1 Response to Comments Public involvement is critical in shaping public land management policy. Public commen ||| 2016-01-15 [clear_initiation|medium|score=2.5] doc=FEIS | Rulemaking for Colorado Roadless Areas E-1 Response to Comments Public involvement is critical in shaping public land management policy. Public commen ||| 2015-12-30 [clear_initiation|medium|score=2.5] doc=FEIS | Rulemaking for Colorado Roadless Areas E-1 Response to Comments Public involvement is critical in shaping public land management policy. Public commen",
    "suggested_decision_date": "2014-06-15",
    "suggested_decision_evidence": "Issues The June 2014 District Court of Colorado\u2019s opinion in High Country Conservation Advocates v.",
    "top_decision_candidates": "2014-06-01 [proxy_decision|low|score=4.2] doc=FEIS | Issues The June 2014 District Court of Colorado\u2019s opinion in High Country Conservation Advocates v."
  },
  {
    "sample_id": "10",
    "project_title": "Keswick, Shasta, and Whiskeytown Fault Trenching Study \u2013 Group 1",
    "process_type": "EA",
    "lead_agency": "Bureau of Reclamation",
    "suggested_initiation_date": "2021-07-15",
    "suggested_initiation_evidence": "Department of Energy, Western Area Power Administration, Sierra Nevada Region ACTION: Finding of No Significant Impact (FONSI) In July 2021 the Bureau of Reclamation (Reclamation) prepared a draft Environmental Assessment (EA) for the Keswick, Shasta, and Whiskeytown Fault Trenching Study \u2013 Group 1",
    "top_initiation_candidates": "2021-07-01 [proxy_initiation|low|score=9.1] doc=FONSI | Department of Energy, Western Area Power Administration, Sierra Nevada Region ACTION: Finding of No Significant Impact (FONSI) In July 2021 the Bureau ||| 2020-11-01 [proxy_initiation|low|score=4.0] doc=EA | Reclamation chose trenching sites specifically to avoid wetlands and other waters of the U.S., using a combination of field surveys conducted in Novem ||| 2020-11-01 [proxy_initiation|low|score=4.0] doc=EA | 15 removing habitat from any species utilizing the tree. Reclamation has utilized on-site surveys and GIS maps to choose trenching sites with minimal",
    "suggested_decision_date": "2021-08-09",
    "suggested_decision_evidence": "Department of Energy Date SONJA ANDERSON Digitally signed by SONJA ANDERSON Date: 2021.08.09 12:22:56 -07'00'",
    "top_decision_candidates": "2021-08-09 [clear_decision|high|score=13.0] doc=FONSI | Department of Energy Date SONJA ANDERSON Digitally signed by SONJA ANDERSON Date: 2021.08.09 12:22:56 -07'00' ||| 2021-08-03 [proxy_decision|medium|score=11.0] doc=FONSI | Reclamation approved the Final EA and Finding of No Significant Impact (FONSI) on August 3, 2021. ||| 2021-06-07 [clear_decision|low|score=9.2] doc=FONSI | 1531 et seq.), and on June 7, 2021 received a concurrence letter on Reclamation\u2019s determination that the project may affect, but is not likely to adve ||| 2021-06-07 [clear_decision|low|score=9.2] doc=FONSI | Reclamation completed informal consultation with the USFWS and received a letter of concurrence on June 7, 2021. ||| 2021-07-09 [clear_decision|low|score=9.0] doc=FONSI | Department of Energy, Western Area Power Administration, Sierra Nevada Region ACTION: Finding of No Significant Impact (FONSI) In July 2021 the Bureau"
  },
  {
    "sample_id": "11",
    "project_title": "Proposed Change in Management of Paria Canyon-Vermilion Cliffs Wilderness",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2019-05-08",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2019-05-08",
    "top_initiation_candidates": "2019-05-08 [clear_initiation|high|score=11.6] doc=None | BLM NEPA Register project start date: 2019-05-08",
    "suggested_decision_date": "2020-12-18",
    "suggested_decision_evidence": "BLM NEPA Register decision date (fonsi): 2020-12-18",
    "top_decision_candidates": "2020-12-18 [clear_decision|high|score=10.0] doc=None | BLM NEPA Register decision date (fonsi): 2020-12-18"
  },
  {
    "sample_id": "12",
    "project_title": "Durham-Orange Light Rail Transit Project",
    "process_type": "EIS",
    "lead_agency": "",
    "suggested_initiation_date": "2012-04-03",
    "suggested_initiation_evidence": "Notification Methods Notification methods of the scoping process are listed below: Federal Register/Notice of Intent (NOI) A Notice of Intent (NOI) was published in the Federal Register on April 3, 2012, indicating that the Federal Transit Administration and Triangle Transit will be preparing an Env",
    "top_initiation_candidates": "2012-04-03 [clear_initiation|high|score=9.8] doc=OTHER | Notification Methods Notification methods of the scoping process are listed below: Federal Register/Notice of Intent (NOI) A Notice of Intent (NOI) wa ||| 2012-04-03 [clear_initiation|high|score=9.8] doc=OTHER | Scoping Report Durham-Orange Light Rail Transit Project | September 2012 | 6-2 6.1.1. Notification Methods Notification methods of the scoping process ||| 2012-09-01 [clear_initiation|high|score=9.8] doc=OTHER | Scoping Report Durham-Orange Light Rail Transit Project | September 2012 | 6-2 6.1.1. Notification Methods Notification methods of the scoping process ||| 2012-09-01 [clear_initiation|high|score=9.8] doc=OTHER | Scoping Report Durham-Orange Light Rail Transit Project | September 2012 | 2-2 In May 2012, following publication in the Federal Register of a Notice ||| 2012-08-01 [clear_initiation|high|score=9.2] doc=OTHER | Scoping Report Durham-Orange Light Rail Transit Project | September 2012 | 2-2 In May 2012, following publication in the Federal Register of a Notice",
    "suggested_decision_date": "2015-01-06",
    "suggested_decision_evidence": "The archaeological APE was determined by the FTA in consultation with the SHPO (see SHPO letter of January 6, 2015 included in appendix G).",
    "top_decision_candidates": "2015-01-06 [clear_decision|medium|score=8.0] doc=DEIS | The archaeological APE was determined by the FTA in consultation with the SHPO (see SHPO letter of January 6, 2015 included in appendix G). ||| 2015-01-06 [clear_decision|medium|score=8.0] doc=DEIS | D-O LRT Project DEIS/Draft Section 4(f) Evaluation 4-104 and 100 feet from the southern edge of the ROMF. \uf0a7 The APE extends an additional 100 feet out ||| 2014-06-01 [clear_decision|medium|score=6.2] doc=DEIS | The Development Agreement was approved by the Town Council in June 2014 and will incorporate mixed-use office, retail, and restaurants into six new ap ||| 2014-06-01 [clear_decision|medium|score=6.2] doc=DEIS | D-O LRT Project DEIS/Draft Section 4(f) Evaluation 4-29 the redevelopment of the Glen Lennox area to increase the number of residential units and add ||| 1994-04-11 [clear_decision|medium|score=6.0] doc=DEIS | 5Environmental Justice Executive Order (EO) 12898, Federal Actions to Address Environmental Justice in Minority Populations and Low-Income Populations"
  },
  {
    "sample_id": "13",
    "project_title": "New Authorization for Poland Jct to Bumble Bee 12kV Line",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2017-02-22",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2017-02-22",
    "top_initiation_candidates": "2017-02-22 [clear_initiation|high|score=10.0] doc=None | BLM NEPA Register project start date: 2017-02-22",
    "suggested_decision_date": "2017-03-04",
    "suggested_decision_evidence": "AUTHORIZING OFFICIAL: NAME: Rem Hawes Rangas DATE: 3/4/2017 TITLE: Field Manager, Hassayampa Field Office Note: The signed conclusion on this compliance record is part of an interim step in the BLM's internal decision process and does not constitute an appealable decision.",
    "top_decision_candidates": "2017-07-01 [proxy_decision|low|score=4.0] doc=CE | - PROPOSED ACTION BLM Office: Hassayampa Field Office NEPA No.: DOI-BLM-AZ-P010-2017-0009-CX Case File No.: AZA-036891 Proposed Action Title/Type: To ||| 2017-03-04 [clear_decision|low|score=4.0] doc=CE | AUTHORIZING OFFICIAL: NAME: Rem Hawes Rangas DATE: 3/4/2017 TITLE: Field Manager, Hassayampa Field Office Note: The signed conclusion on this complian ||| 2013-08-01 [clear_decision|low|score=2.5] doc=CE | Attachment 4-1 AZ-1790-1 August 2013 PART III. ||| 2013-08-01 [clear_decision|low|score=2.5] doc=CE | [x] Preparer's Initials CC Attachment 4-4 AZ-1790-1 August 2013 PART V.-COMPLIANCE REVIEW CONCLUSION This categorical exclusion is appropriate in this ||| 2013-08-01 [clear_decision|low|score=2.5] doc=CE | Attachment 4-5 AZ-1790-1 August 2013"
  },
  {
    "sample_id": "14",
    "project_title": "Alton Coal Tract Lease by Application",
    "process_type": "EIS",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2006-11-28",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2006-11-28",
    "top_initiation_candidates": "2006-11-28 [clear_initiation|high|score=12.0] doc=None | BLM NEPA Register project start date: 2006-11-28",
    "suggested_decision_date": "2018-08-28",
    "suggested_decision_evidence": "BLM NEPA Register decision date (rod): 2018-08-28",
    "top_decision_candidates": "2018-08-28 [clear_decision|high|score=10.0] doc=None | BLM NEPA Register decision date (rod): 2018-08-28"
  },
  {
    "sample_id": "15",
    "project_title": "Withdrawal Revocation of Lands Segregated for a Geophysical Observatory Under Public Land Order 5275",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2014-01-01",
    "suggested_decision_evidence": "Document filename date (ce): 2014-01-01",
    "top_decision_candidates": "2014-01-01 [clear_decision|medium|score=8.0] doc=CE | Document filename date (ce): 2014-01-01 ||| 2014-04-21 [clear_decision|medium|score=6.5] doc=CE | Stiewig / Field Office Manager DATE 4/21/14 Chapter 2 Extraordinary Circumstances Worksheet ||| 1983-10-17 [clear_decision|low|score=4.2] doc=CE | Description of Proposed Action: On October 17, 1983, the National Science Foundation requested that the withdrawal for the Seismological Observatory ( ||| 2004-06-29 [clear_decision|low|score=4.2] doc=CE | Field inspections were completed to determine any residual hazmat evidence on June 29, 2004. ||| 2004-05-27 [clear_decision|low|score=4.2] doc=CE | The applicable Categorical Exclusion, effective May 27, 2004, reference in 516 DM 11.5 E (3)."
  },
  {
    "sample_id": "16",
    "project_title": "Houston Ship Channel Expansion Channel Improvement Project, Harris, Chambers, and Galveston Counties, Texas",
    "process_type": "EIS",
    "lead_agency": "Corps of Engineers--Civil Works",
    "suggested_initiation_date": "2008-08-15",
    "suggested_initiation_evidence": "The following sections must be completed prior to field sampling or laboratory analysis: Data Review Document Page 2 of 31 August 2008 Page 98 of 172",
    "top_initiation_candidates": "2008-08-01 [proxy_initiation|low|score=7.0] doc=OTHER | The following sections must be completed prior to field sampling or laboratory analysis: Data Review Document Page 2 of 31 August 2008 Page 98 of 172 ||| 2017-05-01 [proxy_initiation|low|score=5.0] doc=OTHER | It is anticipated that an integrated Draft Feasibility Report/ EIS will be made available for public review approximately May 2017. ||| 2019-12-01 [clear_initiation|medium|score=1.8] doc=OTHER | 30 HSC ECIP December 2019 FINAL General Conformity Determination 5 DRAFT GCD COMMENTS AND RESPONSES The USACE submitted the Draft GCD, and issued a pu ||| 2019-10-18 [clear_initiation|medium|score=1.8] doc=OTHER | 30 HSC ECIP December 2019 FINAL General Conformity Determination 5 DRAFT GCD COMMENTS AND RESPONSES The USACE submitted the Draft GCD, and issued a pu ||| 2019-12-04 [clear_initiation|medium|score=1.8] doc=OTHER | 30 HSC ECIP December 2019 FINAL General Conformity Determination 5 DRAFT GCD COMMENTS AND RESPONSES The USACE submitted the Draft GCD, and issued a pu",
    "suggested_decision_date": "2017-11-13",
    "suggested_decision_evidence": "James Prazak Chair, Lone Star Harbor Safety Committee 13-Nov Abrupt mixing of deep draft and shallow draft vessel traffic below Morgans Point: Currently the barge lanes terminate below Morgans Point. This results in a more congested maritime space for both deep draft and shallow draft vessel traffic",
    "top_decision_candidates": "2017-11-13 [clear_decision|medium|score=7.5] doc=OTHER | James Prazak Chair, Lone Star Harbor Safety Committee 13-Nov Abrupt mixing of deep draft and shallow draft vessel traffic below Morgans Point: Current ||| 2008-08-31 [clear_decision|medium|score=7.0] doc=OTHER | DRAFT Submitted by: Date submitted: Approved by: Project Review The SAP/QAPP was prepared and submitted for approval by the Corps of Engineers Distric ||| 2008-08-01 [clear_decision|medium|score=7.0] doc=OTHER | DRAFT Submitted by: Date submitted: Approved by: Project Review The SAP/QAPP was prepared and submitted for approval by the Corps of Engineers Distric ||| 2008-08-31 [clear_decision|medium|score=7.0] doc=OTHER | Data Review Document Page 2 of 31 August 2008 Submitted by: Date submitted: Approved by: Project Review The SAP/QAPP was prepared and submitted for ap ||| 2008-08-01 [clear_decision|medium|score=7.0] doc=OTHER | Data Review Document Page 2 of 31 August 2008 Submitted by: Date submitted: Approved by: Project Review The SAP/QAPP was prepared and submitted for ap"
  },
  {
    "sample_id": "17",
    "project_title": "Sutter Co. CO2 Capture and Storage Project, Northern California",
    "process_type": "CE",
    "lead_agency": "Department of Energy",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2023-02-22",
    "suggested_decision_evidence": "DOE Initiator Signature: NATALIE IANNACCHIONE Digitally signed by NATALIE IANNACCHIONE Date: 2023.02.22 13:28:52 -05'00' Date: 02 / 22 / 2023 month day year NEPA Compliance Officer: JILL TRIULZI Digitally signed by JILL TRIULZI Date: 2023.03.02 13:58:15-05'00' Date: 03 / 02 / 2023 month day year The",
    "top_decision_candidates": "2023-02-22 [clear_decision|high|score=8.0] doc=CE | DOE Initiator Signature: NATALIE IANNACCHIONE Digitally signed by NATALIE IANNACCHIONE Date: 2023.02.22 13:28:52 -05'00' Date: 02 / 22 / 2023 month da ||| 2023-03-02 [clear_decision|high|score=8.0] doc=CE | DOE Initiator Signature: NATALIE IANNACCHIONE Digitally signed by NATALIE IANNACCHIONE Date: 2023.02.22 13:28:52 -05'00' Date: 02 / 22 / 2023 month da"
  },
  {
    "sample_id": "18",
    "project_title": "",
    "process_type": "EA",
    "lead_agency": "United States Geological Survey",
    "suggested_initiation_date": "2025-04-15",
    "suggested_initiation_evidence": "Campbell County Wind Farm 2 Interconnection Request Finding of No Significant Impact and Decision Campbell County, South Dakota DOE/EA-2062 April 2025",
    "top_initiation_candidates": "2025-04-01 [clear_initiation|medium|score=11.3] doc=FONSI | Campbell County Wind Farm 2 Interconnection Request Finding of No Significant Impact and Decision Campbell County, South Dakota DOE/EA-2062 April 2025 ||| 2023-07-20 [clear_initiation|medium|score=8.0] doc=DEA | 5.3 Native American Tribes and Associated Bodies Pursuant to Section 106 of the NHPA, WAPA initiated tribal consultations with the following Tribes by ||| 2023-07-20 [clear_initiation|medium|score=8.0] doc=DEA | Campbell County Wind Farm 2 Draft Environmental Assessment 102 \uf0b7 U.S. House of Representatives The NRCS - South Dakota Office, USFWS South Dakota Ecol ||| 2022-11-02 [clear_initiation|medium|score=8.0] doc=EA | Campbell County Wind Farm 2 Final Environmental Assessment 106 5.0 Consultation and Coordination WAPA held a public scoping comment period from Novemb ||| 2025-04-01 [proxy_initiation|low|score=7.8] doc=None | Campbell County, South Dakota DOE/EA-2062 April 2025 Campbell County Wind Farm 2 Project Finding of No Significant Impact and Decision Document, Campb",
    "suggested_decision_date": "2025-04-25",
    "suggested_decision_evidence": "April 25 Name ______________________________________ Date __________________ LLOYD LINKE Digitally signed by LLOYD LINKE Date: 2025.04.25 12:27:00 -05'00' April 25, 2025 Lloyd A.",
    "top_decision_candidates": "2025-04-25 [clear_decision|high|score=13.2] doc=FONSI | April 25 Name ______________________________________ Date __________________ LLOYD LINKE Digitally signed by LLOYD LINKE Date: 2025.04.25 12:27:00 -05 ||| 2025-04-25 [clear_decision|high|score=13.2] doc=FONSI | x Campbell County Wind Farm 2 Project Finding of No Significant Impact and Decision Document, Campbell County, South Dakota The Project itself is typi ||| 2022-11-02 [clear_decision|low|score=10.0] doc=FONSI | Both an agency scoping meeting and public scoping meeting were held at separate times on November 2, 2022. ||| 2022-12-02 [clear_decision|low|score=9.5] doc=FONSI | PUBLIC INVOLVEMENT: WAPA held a public scoping comment period from November 2 \u2013 December 2, 2022, to provide the general public, government agencies, ||| 2024-09-20 [clear_decision|low|score=9.0] doc=FONSI | WAPA circulated the draft EA for public review and comment for 30 days, ending on September 20, 2024."
  },
  {
    "sample_id": "19",
    "project_title": "SR 86: Sandario Road to Kinney Road",
    "process_type": "EA",
    "lead_agency": "Federal Highway Administration",
    "suggested_initiation_date": "2013-07-01",
    "suggested_initiation_evidence": "Department of the Interior Bureau of Land Management Tucson Field Office Pima County; State Route 86 Expansion, Assignment of portion of AZA-17485, Partial Relinquishment of AZA-18432 and AZA-35322 DOI-BLM-AZ-G020-2013-0034-EA Background On December 5, 2014 Arizona Department of Transportation filed",
    "top_initiation_candidates": "2013-07-01 [clear_initiation|medium|score=14.2] doc=ROD | Department of the Interior Bureau of Land Management Tucson Field Office Pima County; State Route 86 Expansion, Assignment of portion of AZA-17485, Pa ||| 2013-07-01 [clear_initiation|medium|score=14.2] doc=ROD | DECISION RECORD U.S. Department of the Interior Bureau of Land Management Tucson Field Office Pima County; State Route 86 Expansion, Assignment of por ||| 2013-07-01 [clear_initiation|medium|score=14.2] doc=ROD | Department of the Interior Bureau of Land Management Tucson Field Office Pima County; State Route 86 Expansion, Right-of-Way Grant DOI-BLM-AZ-G020-201 ||| 2013-07-01 [clear_initiation|medium|score=14.2] doc=ROD | Department of the Interior Bureau of Land Management Tucson Field Office Pima County; State Route 86 Expansion, Rights-of-Way Grant DOI-BLM-AZ-G020-20 ||| 2013-03-27 [clear_initiation|medium|score=13.5] doc=ROD | Department of the Interior Bureau of Land Management Tucson Field Office Pima County; State Route 86 Expansion, Rights-of-Way Grant DOI-BLM-AZ-G020-20",
    "suggested_decision_date": "2015-03-18",
    "suggested_decision_evidence": "/s/ Bruce Sillitoe 03/18/2015 Bruce Sillitoe, Acting Tucson Field Manager Date Attachments: Finding of No Significant Impact dated Environmental Assessment \u2013 Federal Highway Administration (FHWA) environmental assessment (EA) numbered STP-086-A (APA) 086 PM...",
    "top_decision_candidates": "2015-03-18 [clear_decision|high|score=14.0] doc=ROD | /s/ Bruce Sillitoe 03/18/2015 Bruce Sillitoe, Acting Tucson Field Manager Date Attachments: Finding of No Significant Impact dated Environmental Asses ||| 2014-08-14 [clear_decision|high|score=14.0] doc=ROD | Attachments: Finding of No Significant Impact dated Environmental Assessment \u2013 Federal Highway Administration (FHWA) environmental assessment (EA) num ||| 2014-08-14 [clear_decision|high|score=14.0] doc=ROD | /s/ Bruce Sillitoe 03/18/2015 Bruce Sillitoe, Acting Tucson Field Manager Date Attachments: Finding of No Significant Impact dated Environmental Asses ||| 2015-03-18 [clear_decision|high|score=14.0] doc=ROD | /s/ Bruce Sillitoe 03/18/2015 Bruce Sillitoe, Acting Tucson Field Manager Date Attachments: Finding of No Significant Impact dated Environmental Asses ||| 2014-08-14 [clear_decision|high|score=14.0] doc=ROD | /s/ Bruce Sillitoe 03/18/2015 Bruce Sillitoe, Acting Tucson Field Manager Date Attachments: Finding of No Significant Impact dated Environmental Asses"
  },
  {
    "sample_id": "20",
    "project_title": "Dog River Pipeline Replacement",
    "process_type": "EA",
    "lead_agency": "Forest Service",
    "suggested_initiation_date": "1999-01-21",
    "suggested_initiation_evidence": "City of The Dalles Application Permit Certificate Claim, Decree, or Transfer Priority date Type of Beneficial Use Authorized Rate or Annual Volume Dog River 14954 Hood River decree 8/1/1870 Municipal \u201cAll the water in stream at point of diversion\u201d South Fork Mill Creek 5691 Mill Creek decree 1862 Mu",
    "top_initiation_candidates": "1999-01-21 [clear_initiation|medium|score=6.5] doc=EA | City of The Dalles Application Permit Certificate Claim, Decree, or Transfer Priority date Type of Beneficial Use Authorized Rate or Annual Volume Dog ||| 2016-08-01 [proxy_initiation|low|score=-1.8] doc=EA | In August 2016, a field trip to the project area included Forest Service staff and representatives from the NOAA National Marine Fisheries Service (NM ||| 2016-03-01 [proxy_initiation|low|score=-1.8] doc=EA | Dog River Pipeline Replacement | Environmental Assessment 17 1912 Cooperative Agreement & 1972 Memorandum of Understanding Because much of the municip ||| 2016-08-01 [proxy_initiation|low|score=-1.8] doc=EA | Dog River Pipeline Replacement | Environmental Assessment 17 1912 Cooperative Agreement & 1972 Memorandum of Understanding Because much of the municip",
    "suggested_decision_date": "2016-03-15",
    "suggested_decision_evidence": "A second scoping letter was sent to the public in March 2016.",
    "top_decision_candidates": "2016-03-01 [proxy_decision|low|score=3.2] doc=EA | A second scoping letter was sent to the public in March 2016."
  },
  {
    "sample_id": "21",
    "project_title": "Kachess Drought Relief Pumping Plant and Keechelus Reservoir-to-Kachess Reservoir Conveyance",
    "process_type": "EIS",
    "lead_agency": "",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2013-10-30 [clear_initiation|high|score=5.5] doc=FEIS | 5.2.2 Scoping Comments Received from the Public The scoping period began October 30, 2013, and concluded December 16, 2013, during which time the agen ||| 2013-12-16 [clear_initiation|high|score=5.2] doc=FEIS | 5.2.2 Scoping Comments Received from the Public The scoping period began October 30, 2013, and concluded December 16, 2013, during which time the agen ||| 2013-10-30 [clear_initiation|high|score=4.5] doc=OTHER | KDRPP and KKC SDEIS 1.2.5 Release of 2015 KDRPP-KKC DEIS On October 30, 2013, Reclamation published in the Federal Register a Notice of Intent (NOI) t ||| 2013-11-04 [clear_initiation|high|score=4.5] doc=OTHER | KDRPP and KKC SDEIS 1.2.5 Release of 2015 KDRPP-KKC DEIS On October 30, 2013, Reclamation published in the Federal Register a Notice of Intent (NOI) t ||| 2015-03-10 [clear_initiation|high|score=4.5] doc=OTHER | KDRPP and KKC SDEIS 1.2.5 Release of 2015 KDRPP-KKC DEIS On October 30, 2013, Reclamation published in the Federal Register a Notice of Intent (NOI) t",
    "suggested_decision_date": "1979-12-28",
    "suggested_decision_evidence": "YRBWEP was authorized on December 28, 1979 (93 Stat.",
    "top_decision_candidates": "1979-12-28 [clear_decision|medium|score=7.5] doc=FEIS | YRBWEP was authorized on December 28, 1979 (93 Stat. ||| 2019-03-01 [proxy_decision|low|score=6.5] doc=FEIS | KDRPP and KKC FEIS Page 5-2 5.2 \u2013 Public Involvement March 2019 the evening. ||| 2015-01-01 [proxy_decision|low|score=6.5] doc=FEIS | Reclamation and Ecology issued the DEIS in January 2015. ||| 2015-01-01 [proxy_decision|low|score=6.5] doc=FEIS | The DEIS was released for public and agency review in January 2015."
  },
  {
    "sample_id": "22",
    "project_title": "Clayhole Allotment Pipeline Installation & Water Development",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2022-06-15",
    "suggested_initiation_evidence": "United States Department of the Interior Bureau of Land Management Finding of No Significant Impact Clayhole Allotment Pipeline Installation & Water Developments DOI-BLM-AZ-A010-2021-0008-EA Mohave County, Arizona Applicant/Address: Heaton Cattle Company c/o Kelly Heaton P. O. Box 910088 St. George,",
    "top_initiation_candidates": "2022-06-01 [proxy_initiation|low|score=10.3] doc=FONSI | United States Department of the Interior Bureau of Land Management Finding of No Significant Impact Clayhole Allotment Pipeline Installation & Water D ||| 2022-02-01 [proxy_initiation|low|score=9.4] doc=FONSI | FONSI Under the Proposed Action of Clayhole Allotment Pipeline Installation & Water Developments Environmental Assessment (EA) \u2013 DOI-BLM-AZ-A010-2021- ||| 2022-06-01 [proxy_initiation|low|score=5.3] doc=None | Clayhole Allotment Pipeline Installation & Water Developments DOI-BLM-AZ-A010-2021-0008-EA Mohave County, Arizona Applicant/Address: Heaton Cattle Com",
    "suggested_decision_date": "2022-06-28",
    "suggested_decision_evidence": "Christian Field Manager Arizona Strip Field Office BRANDON BOSHELL Digitally signed by BRANDON BOSHELL Date: 2022.06.28 14:03:00 -06'00'",
    "top_decision_candidates": "2022-06-28 [clear_decision|high|score=13.2] doc=FONSI | Christian Field Manager Arizona Strip Field Office BRANDON BOSHELL Digitally signed by BRANDON BOSHELL Date: 2022.06.28 14:03:00 -06'00' ||| 2022-06-28 [clear_decision|high|score=13.2] doc=FONSI | 7 The proposed action does not violate any known federal, state, local or tribal law or requirement imposed for the protection of the environment. Sta ||| 2021-07-01 [proxy_decision|low|score=10.5] doc=FONSI | United States Department of the Interior Bureau of Land Management Finding of No Significant Impact Clayhole Allotment Pipeline Installation & Water D ||| 2022-06-01 [clear_decision|low|score=10.2] doc=FONSI | George, Utah 84791 June 2022 U.S. ||| 2021-07-01 [proxy_decision|low|score=10.0] doc=FONSI | 1 FINDING OF NO SIGNIFICANT IMPACT Clayhole Allotment Pipeline Installation & Water Developments NEPA # DOI-BLM-AZ-A010-2021-0008-EA INTRODUCTION & BA"
  },
  {
    "sample_id": "23",
    "project_title": "Renewal of an existing 633' length, 50' width 6\u201d sour gas pipeline",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2018-01-19",
    "suggested_decision_evidence": "Boruch Date: Jan 19, 2018 Reviewed By: Randy Verett Date: 1/23/18",
    "top_decision_candidates": "2018-07-01 [proxy_decision|low|score=4.2] doc=CE | UNITED STATES DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT Rock Springs Field Office CATEGORICAL EXCLUSION REVIEW Extraordinary Circumstances ||| 2018-07-01 [proxy_decision|low|score=4.2] doc=CE | Background BLM Office: Rock Springs Field Office Lease/Serial/Case File No.: WYW59095 CX Number: DOI-BLM-WY-D040-2018-0045-CX Right-of-Way Applicant/H ||| 2018-01-19 [clear_decision|low|score=4.0] doc=CE | Boruch Date: Jan 19, 2018 Reviewed By: Randy Verett Date: 1/23/18 ||| 2018-01-23 [clear_decision|low|score=4.0] doc=CE | Boruch Date: Jan 19, 2018 Reviewed By: Randy Verett Date: 1/23/18"
  },
  {
    "sample_id": "24",
    "project_title": "Grazing Permit Renewal for Childs, Coyote Flat #2, and Sentinel Allotments",
    "process_type": "EA",
    "lead_agency": "Department of the Interior",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2021-12-01 [proxy_initiation|low|score=4.8] doc=FONSI | Department of the Interior Bureau of Land Management Lower Sonoran Field Office 21605 North 7th Avenue Phoenix, Arizona 85027 623-580-5500 December 20 ||| 2021-12-01 [proxy_initiation|low|score=4.8] doc=FONSI | Grazing Permit Renewal for Childs, Coyote Flat #2, and Sentinel Allotments Finding of No Significant Impact DOI-BLM-AZ-P020-2021-0013-EA U.S. Departme ||| 1997-04-01 [proxy_initiation|low|score=-3.0] doc=OTHER | RATIONALE The Secretary of the Interior approved Arizona Standards for Rangeland Health and Guidelines for Grazing Administration in April 1997. ||| 1997-04-01 [proxy_initiation|low|score=-3.0] doc=OTHER | 5 5 The Secretary of the Interior approved Arizona Standards for Rangeland Health and Guidelines for Grazing Administration in April 1997. ||| 1997-04-01 [proxy_initiation|low|score=-3.0] doc=OTHER | 5 5 RATIONALE The Secretary of the Interior approved Arizona Standards for Rangeland Health and Guidelines for Grazing Administration in April 1997.",
    "suggested_decision_date": "1997-04-15",
    "suggested_decision_evidence": "The Standards and Guidelines Environmental Assessment Decision Record, signed by the BLM State Director in April 1997, provides for full implementation of the Standards and Guidelines in all Arizona BLM land use plans.",
    "top_decision_candidates": "2021-07-01 [proxy_decision|low|score=9.5] doc=FONSI | Grazing Permit Renewal for Childs, Coyote Flat #2, and Sentinel Allotments Finding of No Significant Impact DOI-BLM-AZ-P020-2021-0013-EA U.S. ||| 1997-04-01 [clear_decision|medium|score=4.0] doc=OTHER | The Standards and Guidelines Environmental Assessment Decision Record, signed by the BLM State Director in April 1997, provides for full implementatio ||| 1997-04-01 [clear_decision|medium|score=4.0] doc=OTHER | The Standards and Guidelines Environmental Assessment Decision Record, signed by the BLM State Director in April 1997, provides for full implementatio ||| 1995-08-21 [clear_decision|medium|score=4.0] doc=OTHER | 6 6 this part. These changes must be supported by monitoring, field observations, ecological site inventory, or other data acceptable to the authorize ||| 1995-08-21 [clear_decision|medium|score=4.0] doc=OTHER | 6 6 \u00a74110.4(a) Where there is a decrease in public land acreage available for livestock grazing within an allotment: (1) Grazing permits or leases may"
  },
  {
    "sample_id": "25",
    "project_title": "Howard T. Ricketts (HTR) Regional Biocontainment Laboratory",
    "process_type": "EA",
    "lead_agency": "National Institutes of Health",
    "suggested_initiation_date": "2003-02-15",
    "suggested_initiation_evidence": "In February 2003, the University submitted a proposal to the NIH to construct the HTRL on the grounds of Argonne.",
    "top_initiation_candidates": "2003-02-01 [proxy_initiation|low|score=3.0] doc=EA | In February 2003, the University submitted a proposal to the NIH to construct the HTRL on the grounds of Argonne. ||| 2003-09-01 [proxy_initiation|low|score=3.0] doc=EA | In September 2003, the NIH announced that it would award a grant to the University to construct the HTRL at Argonne. ||| 2002-08-01 [proxy_initiation|low|score=3.0] doc=EA | In August 2002, NIAID had solicited applications from research institutions across the United States to develop RCEs for the scientific study of disea",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "26",
    "project_title": "WYWY 106322307 Chipcore 2\" Water Pipeline Right-of-Way",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2023-10-03",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2023-10-03",
    "top_initiation_candidates": "2023-10-03 [clear_initiation|high|score=10.2] doc=None | BLM NEPA Register project start date: 2023-10-03",
    "suggested_decision_date": "2023-12-15",
    "suggested_decision_evidence": "Signature DARCI NATION Digitally signed by DARCI NATION Authorizing Official: Date: 2023.12.15 14:05:38 -07'00' Darci N.",
    "top_decision_candidates": "2023-12-15 [clear_decision|high|score=8.0] doc=CE | Signature DARCI NATION Digitally signed by DARCI NATION Authorizing Official: Date: 2023.12.15 14:05:38 -07'00' Darci N."
  },
  {
    "sample_id": "27",
    "project_title": "Exploration Plan of Operations for the Antler Operations Project",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2022-02-04",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2022-02-04",
    "top_initiation_candidates": "2022-02-04 [clear_initiation|high|score=10.8] doc=None | BLM NEPA Register project start date: 2022-02-04",
    "suggested_decision_date": "2022-11-28",
    "suggested_decision_evidence": "BLM NEPA Register decision date (fonsi): 2022-11-28",
    "top_decision_candidates": "2022-11-28 [clear_decision|high|score=10.0] doc=None | BLM NEPA Register decision date (fonsi): 2022-11-28"
  },
  {
    "sample_id": "28",
    "project_title": "Right-of-way renewal",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2022-07-13",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2022-07-13",
    "top_initiation_candidates": "2022-07-13 [clear_initiation|high|score=5.0] doc=None | BLM NEPA Register project start date: 2022-07-13",
    "suggested_decision_date": "2022-07-01",
    "suggested_decision_evidence": "Preparer/s DENISE BOUDREAULT Project Lead Digitally signed by DENISE BOUDREAULT Date: 2022.08.05 06:34:18 -06'00' RANDY VERRET Environmental Reviewer Digitally signed by RANDY VERRET Date: 2022.08.05 07:57:13 -06'00' DOI-BLM-WY-D040-2022-0115-CX Page 2 of 2",
    "top_decision_candidates": "2022-07-01 [clear_decision|high|score=8.5] doc=CE | Preparer/s DENISE BOUDREAULT Project Lead Digitally signed by DENISE BOUDREAULT Date: 2022.08.05 06:34:18 -06'00' RANDY VERRET Environmental Reviewer ||| 2022-08-05 [clear_decision|high|score=8.0] doc=CE | Preparer/s DENISE BOUDREAULT Project Lead Digitally signed by DENISE BOUDREAULT Date: 2022.08.05 06:34:18 -06'00' RANDY VERRET Environmental Reviewer ||| 1997-08-08 [clear_decision|medium|score=6.0] doc=CE | Date Approved/Amended: August 8, 1997. ||| 2022-07-01 [proxy_decision|low|score=4.5] doc=CE | BUREAU OF LAND MANAGEMENT Rock Springs Field Office DOI-BLM-WY-D040-2022-0115-CX CATEGORICAL EXCLUSION A. ||| 2022-07-01 [proxy_decision|low|score=4.5] doc=CE | \"Renewals and assignments of leases, permits, or rights-of-way where no additional rights are conveyed beyond those granted by the original authorizat"
  },
  {
    "sample_id": "29",
    "project_title": "BH Buildings 802 and 812 Damage Inspection",
    "process_type": "CE",
    "lead_agency": "Department of Energy",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2010-01-07",
    "suggested_decision_evidence": "Signature on file Approved by SPRPMO NEPA Compliance Officer 01/07/10 Determination Date",
    "top_decision_candidates": "2010-01-07 [clear_decision|high|score=8.0] doc=CE | Signature on file Approved by SPRPMO NEPA Compliance Officer 01/07/10 Determination Date"
  },
  {
    "sample_id": "30",
    "project_title": "RED WASH SUBSTATION EXPANSION",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2019-07-22 [clear_initiation|high|score=5.0] doc=None | BLM NEPA Register project start date: 2019-07-22 ||| 2019-08-01 [proxy_initiation|low|score=-1.0] doc=CE | UNITED STATES DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT CATEGORICAL EXCLUSION RED WASH SUBSTATION EXPANSION DOI-BLM-UT-G010-2019-0073-CX Au",
    "suggested_decision_date": "2019-07-16",
    "suggested_decision_evidence": "Signature and Date for Migratory Birds: Chris Perkins, 7/16/2019 3.",
    "top_decision_candidates": "2019-07-01 [proxy_decision|low|score=4.2] doc=CE | UNITED STATES DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT CATEGORICAL EXCLUSION RED WASH SUBSTATION EXPANSION DOI-BLM-UT-G010-2019-0073-CX Au ||| 2019-07-01 [proxy_decision|low|score=4.2] doc=CE | DEPARTMENT OF THE INTERIOR RED WASH SUBSTATION EXPANSION CATEGORICAL EXCLUSION DOI-BLM-UT-G010-2019-0073-CX INTRODUCTION Office: Vernal Field Office L ||| 2019-07-16 [clear_decision|low|score=4.2] doc=CE | Signature and Date for Migratory Birds: Chris Perkins, 7/16/2019 3. ||| 2019-07-16 [clear_decision|low|score=4.2] doc=CE | Signature and Date for Wildlife: Chris Perkins, 7/16/2019 9."
  },
  {
    "sample_id": "31",
    "project_title": "Proposed Currency Production Facility",
    "process_type": "EIS",
    "lead_agency": "Bureau of Engraving and Printing",
    "suggested_initiation_date": "2021-06-04",
    "suggested_initiation_evidence": "At this time, the City restates its strong opposition to Treasury's Preferred Alternative and support for the No Build Alternative. We urge Treasury to reconsider the Purpose and Need of the Project and the alternatives under consideration. We also request that a more complete investigation of all a",
    "top_initiation_candidates": "2021-06-04 [clear_initiation|medium|score=10.5] doc=None | At this time, the City restates its strong opposition to Treasury's Preferred Alternative and support for the No Build Alternative. We urge Treasury t ||| 2019-12-01 [proxy_initiation|low|score=7.8] doc=None | A stakeholder meeting in December 2019 was not publicized (maybe by intent), and despite COVID-19, the EIS timeline was not changed. ||| 2019-12-01 [proxy_initiation|low|score=7.8] doc=None | John Lipart Chair of the Green Team Section 1.10 - Public Participation 4) It seems like there are deliberate attempts to keep the voices of Greenbelt ||| 2021-07-01 [proxy_initiation|low|score=6.0] doc=None | The Proposed Action would be implemented over an estimated nine-year period, after completion of the NEPA analysis and signing of the Record of Decisi ||| 2020-01-01 [proxy_initiation|low|score=5.8] doc=None | While the DEIS is responsive to EPA scoping comments (see January 2020 BEP Scoping Report) to include a list of sites examined for the facility reloca",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "32",
    "project_title": "Van Ness Avenue Bus Rapid Transit Project",
    "process_type": "EIS",
    "lead_agency": "Department of Transportation",
    "suggested_initiation_date": "2008-03-15",
    "suggested_initiation_evidence": "2.1.371B\uf0bdAlternatives Screening/Analysis To identify the limited set of build alternatives to be analyzed in the Draft EIS/EIR, the Authority prepared an Alternatives Screening Report (March 2008).",
    "top_initiation_candidates": "2008-03-01 [proxy_initiation|low|score=5.6] doc=FEIS | 2.1.371B\uf0bdAlternatives Screening/Analysis To identify the limited set of build alternatives to be analyzed in the Draft EIS/EIR, the Authority prepared ||| 2008-03-01 [proxy_initiation|low|score=5.6] doc=FEIS | Chapter 2: Project Alternatives Van Ness Avenue Bus Rapid Transit Project Final Environmental Impact Statement/ Environmental Impact Report 2-2 San Fr ||| 2006-12-01 [proxy_initiation|low|score=5.0] doc=OTHER | Previous studies and documents relevant to this action include the recently completed Van Ness Avenue BRT Feasibility Study (December 2006); 2005 Prop ||| 2005-03-01 [proxy_initiation|low|score=5.0] doc=OTHER | Previous studies and documents relevant to this action include the recently completed Van Ness Avenue BRT Feasibility Study (December 2006); 2005 Prop ||| 2013-07-01 [clear_initiation|medium|score=2.5] doc=FEIS | Van Ness Avenue Bus Rapid Transit Project Chapter 2: Project Alternatives Final Environmental Impact Statement/ Environmental Impact Report San Franci",
    "suggested_decision_date": "2009-04-15",
    "suggested_decision_evidence": "Fieldwork occurred in March and April 2009.",
    "top_decision_candidates": "2009-04-01 [proxy_decision|low|score=4.8] doc=FEIS | Fieldwork occurred in March and April 2009. ||| 2009-02-01 [proxy_decision|low|score=4.8] doc=FEIS | In addition, the CHRIS was reviewed and a records search was conducted for the project in February 2009. ||| 2008-03-01 [proxy_decision|low|score=4.5] doc=FEIS | March 2008."
  },
  {
    "sample_id": "33",
    "project_title": "L&C County Comm. Facilities ROW Renewal, MTM-4101",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2019-08-05",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2019-08-05",
    "top_initiation_candidates": "2019-08-05 [clear_initiation|high|score=5.0] doc=None | BLM NEPA Register project start date: 2019-08-05",
    "suggested_decision_date": "2019-07-01",
    "suggested_decision_evidence": "CATEGORICAL EXCLUSION L&C County Communication Use Lease Renewal, MTM-4101 DOI-BLM-MT-B070-2019-0031-CX Description of the Proposed Action and any Applicable Mitigation Measures BLM Office Butte Field Office NEPA PROJECT NUMBER DOI-BLM-MT-B070-2019-0031-CX PROPOSED ACTION TITLE L&C County Comm.",
    "top_decision_candidates": "2019-07-01 [proxy_decision|low|score=4.0] doc=CE | CATEGORICAL EXCLUSION L&C County Communication Use Lease Renewal, MTM-4101 DOI-BLM-MT-B070-2019-0031-CX Description of the Proposed Action and any App"
  },
  {
    "sample_id": "34",
    "project_title": "Uncompahgre Proposed Resource Management Plan Revision",
    "process_type": "EIS",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2013-02-26 [clear_initiation|high|score=4.5] doc=None | 5.2.5 North Fork Advocacy Group On February 26, 2013, the BLM received a letter from an advocacy group with preliminary documents depicting the \u201cNorth ||| 2013-04-02 [clear_initiation|high|score=4.5] doc=None | 5.2.5 North Fork Advocacy Group On February 26, 2013, the BLM received a letter from an advocacy group with preliminary documents depicting the \u201cNorth ||| 2016-11-01 [clear_initiation|medium|score=2.5] doc=None | In response to public requests, the comment period was extended for an additional 60 days, to end on November 1, 2016. ||| 2016-07-21 [clear_initiation|medium|score=2.5] doc=None | The extension of the comment period was announced via a press release on July 21, 2016. ||| 2013-03-01 [proxy_initiation|low|score=0.5] doc=None | The BLM invited one additional agency in March 2013.",
    "suggested_decision_date": "1989-07-15",
    "suggested_decision_evidence": "July 1989.",
    "top_decision_candidates": "1989-07-01 [proxy_decision|low|score=5.5] doc=FEIS | July 1989. ||| 2008-02-01 [proxy_decision|low|score=5.5] doc=FEIS | Revised February 2008."
  },
  {
    "sample_id": "35",
    "project_title": "Install Packaged Central HVAC Units for BC Building 423",
    "process_type": "CE",
    "lead_agency": "Department of Energy",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2015-08-12",
    "suggested_decision_evidence": "DOE CX Register determination date (cx-14103.0): 2015-08-12",
    "top_decision_candidates": "2015-08-12 [clear_decision|high|score=15.2] doc=categorical exclusion determin | DOE CX Register determination date (cx-14103.0): 2015-08-12 ||| 2015-08-12 [clear_decision|high|score=8.2] doc=CE | Electronic approval via NEPA Workflow Approved by SPRPMO NEPA Compliance Officer 8/12/15 Determination Date"
  },
  {
    "sample_id": "36",
    "project_title": "New Pueblo Freeway project",
    "process_type": "EIS",
    "lead_agency": "Federal Aviation Administration",
    "suggested_initiation_date": "2010-03-15",
    "suggested_initiation_evidence": "These properties are considered mutually beneficial, and the MOU between CDOT and the City (March 2010) specifies the future land exchange, ownership, and maintenance responsibilities (see Appendix F).",
    "top_initiation_candidates": "2010-03-01 [proxy_initiation|low|score=4.9] doc=FEIS | These properties are considered mutually beneficial, and the MOU between CDOT and the City (March 2010) specifies the future land exchange, ownership, ||| 2010-03-01 [proxy_initiation|low|score=4.9] doc=FEIS | CHAPTER 11 SUMMARY OF MITIGATION COMMITMENTS Colorado Department of Transportation Mitigation Commitment Monitoring and Reporting FINAL ENVIRONMENTAL ||| 2010-12-01 [proxy_initiation|low|score=3.9] doc=FEIS | EXECUTIVE SUMMARY FINAL ENVIRONMENTAL IMPACT STATEMENT AND SECTION 4(f) EVALUATION FOR I-25 IMPROVEMENTS THROUGH PUEBLO ES-11 In December 2010, the U.",
    "suggested_decision_date": "2011-11-15",
    "suggested_decision_evidence": "Publication of the DEIS The DEIS was released in November 2011 for public and agency review and comment.",
    "top_decision_candidates": "2011-11-01 [proxy_decision|low|score=3.0] doc=FEIS | Publication of the DEIS The DEIS was released in November 2011 for public and agency review and comment."
  },
  {
    "sample_id": "37",
    "project_title": "Skookumchuck Wind Energy Project Proposed Habitat Conservation Plan and Incidental Take Permit for Marbled Murrelet, Bald Eagle, and Golden Eagle",
    "process_type": "EIS",
    "lead_agency": "United States Fish and Wildlife Service",
    "suggested_initiation_date": "2018-05-03",
    "suggested_initiation_evidence": "The 30-day public scoping period began on May 3, 2018, and lasted through June 4, 2018.",
    "top_initiation_candidates": "2018-05-03 [clear_initiation|high|score=10.5] doc=OTHER | The 30-day public scoping period began on May 3, 2018, and lasted through June 4, 2018. ||| 2018-05-03 [clear_initiation|high|score=10.5] doc=OTHER | Skookumchuck Wind Energy Project Draft Environmental Impact Statement 137 November 2018 not be expected to result in cumulatively significant impacts ||| 2018-06-04 [clear_initiation|high|score=10.4] doc=OTHER | The 30-day public scoping period began on May 3, 2018, and lasted through June 4, 2018. ||| 2018-06-04 [clear_initiation|high|score=10.4] doc=OTHER | Skookumchuck Wind Energy Project Draft Environmental Impact Statement 137 November 2018 not be expected to result in cumulatively significant impacts ||| 2019-01-14 [clear_initiation|medium|score=8.5] doc=FEIS | The 45-day public comment period began on November 30, 2018, and lasted through January 14, 2019.",
    "suggested_decision_date": "2019-05-15",
    "suggested_decision_evidence": "Skookumchuck Wind Energy Project Final Environmental Impact Statement i May 2019 TABLE OF CONTENTS Summary .....................................................................................................................................",
    "top_decision_candidates": "2019-05-01 [proxy_decision|low|score=5.0] doc=FEIS | Skookumchuck Wind Energy Project Final Environmental Impact Statement i May 2019 TABLE OF CONTENTS Summary ........................................... ||| 2019-05-01 [proxy_decision|low|score=4.0] doc=FEIS | Skookumchuck Wind Energy Project Final Environmental Impact Statement 2 May 2019 minimize and mitigate those impacts."
  },
  {
    "sample_id": "38",
    "project_title": "Beale WAPA Interconnection Project",
    "process_type": "EA",
    "lead_agency": "Air Force",
    "suggested_initiation_date": "2016-03-08",
    "suggested_initiation_evidence": "Department of Energy, Western Area Power Administration, Sierra Nevada Region ACTION: Finding of No Significant Impact On March 8, 2016, Beale Air Force Base (AFB) submitted an interconnection request to Western Area Power Administration (WAPA) to provide an interconnection from WAPA\u2019s existing Cott",
    "top_initiation_candidates": "2016-03-08 [clear_initiation|medium|score=13.0] doc=FONSI | Department of Energy, Western Area Power Administration, Sierra Nevada Region ACTION: Finding of No Significant Impact On March 8, 2016, Beale Air For ||| 2013-12-01 [proxy_initiation|low|score=12.0] doc=FONSI | Purpose and Need The project is needed because the Department of Defense (DoD) issued an Electric Power Resilience (ERP) memorandum in December 2013 t ||| 2017-04-01 [clear_initiation|medium|score=11.0] doc=None | The results of the System Impact Study Report dated April 2017 indicated that no mitigation or system improvement of the existing system is required t ||| 2017-04-01 [clear_initiation|medium|score=11.0] doc=None | WAPA\u2019s purpose and need is to consider and respond to Beale AFB\u2019s interconnection request submitted in accordance with WAPA\u2019s General Requirements for ||| 2013-12-01 [proxy_initiation|low|score=9.5] doc=None | The project is needed because the Department of Defense (DoD) issued an Electric Power Resilience (ERP) memorandum in December 2013 that documented ke",
    "suggested_decision_date": "2020-11-30",
    "suggested_decision_evidence": "Department of Energy Date SONJA ANDERSON Digitally signed by SONJA ANDERSON Date: 2020.11.30 12:14:03 -08'00' 11/30/2020",
    "top_decision_candidates": "2020-11-30 [clear_decision|high|score=13.0] doc=FONSI | Department of Energy Date SONJA ANDERSON Digitally signed by SONJA ANDERSON Date: 2020.11.30 12:14:03 -08'00' 11/30/2020 ||| 1977-05-24 [clear_decision|low|score=8.0] doc=FONSI | Similarly, EO 11988, Floodplain Management (May 24, 1977), requires Federal agencies to avoid to the extent possible the long and short-term adverse i ||| 1977-05-24 [clear_decision|low|score=8.0] doc=FONSI | BWIP FONSI/FONPA Beale AFB, CA 6 provide for early public review of plans for construction in wetlands. In accordance with EO 11990 and 32 CFR \u00a7 989,"
  },
  {
    "sample_id": "39",
    "project_title": "Bauer Access Road Right-of-Way",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2018-10-03",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2018-10-03",
    "top_initiation_candidates": "2018-10-03 [clear_initiation|high|score=10.5] doc=None | BLM NEPA Register project start date: 2018-10-03 ||| 2018-11-01 [proxy_initiation|low|score=4.2] doc=CE | United States Department of the Interior Bureau of Land Management Categorical Exclusion for Bauer Access Road Right-of-Way Grand Junction Field Offic",
    "suggested_decision_date": "2019-01-01",
    "suggested_decision_evidence": "Document filename date (ce): 2019-01-01",
    "top_decision_candidates": "2019-01-01 [clear_decision|medium|score=8.0] doc=CE | Document filename date (ce): 2019-01-01 ||| 2019-07-01 [proxy_decision|low|score=4.2] doc=CE | United States Department of the Interior Bureau of Land Management Categorical Exclusion for Bauer Access Road Right-of-Way Grand Junction Field Offic ||| 2002-01-09 [clear_decision|low|score=2.0] doc=CE | The BLM issued COC 65688 on January 9, 2002."
  },
  {
    "sample_id": "40",
    "project_title": "East Fork Irrigation District Infrastructure Modernization Project",
    "process_type": "EA",
    "lead_agency": "Natural Resources Conservation Service",
    "suggested_initiation_date": "2019-03-12",
    "suggested_initiation_evidence": "Kate Valdez, Confederated Tribes and Band of the Yakama Nation Cultural resources consultation letters sent to SHPO, CTWS, Confederated Tribes of the Umatilla Indian Reservation, and Yakama Tribes March 12, 2019 Rachel Gebauer, NRCS Cindy Thieman, Hood River Watershed Group Blayne Eineichner, CTWS T",
    "top_initiation_candidates": "2020-11-01 [proxy_initiation|low|score=9.0] doc=FONSI | East Fork Irrigation District Modernization Project Page 1 Finding of No Significant Impact East Fork Irrigation District Modernization Project Findin ||| 2019-03-12 [clear_initiation|medium|score=6.9] doc=EA | Kate Valdez, Confederated Tribes and Band of the Yakama Nation Cultural resources consultation letters sent to SHPO, CTWS, Confederated Tribes of the ||| 2019-03-12 [clear_initiation|medium|score=6.9] doc=EA | East Fork Irrigation District Infrastructure Modernization Project Final Watershed Plan \u2013 Environmental Assessment USDA-NRCS 119 July 2020 Date Contac ||| 2019-03-15 [clear_initiation|medium|score=6.9] doc=EA | Kate Valdez, Confederated Tribes and Band of the Yakama Nation Cultural resources consultation letters sent to SHPO, CTWS, Confederated Tribes of the ||| 2019-03-15 [clear_initiation|medium|score=6.9] doc=EA | East Fork Irrigation District Infrastructure Modernization Project Final Watershed Plan \u2013 Environmental Assessment USDA-NRCS 119 July 2020 Date Contac",
    "suggested_decision_date": "2020-11-03",
    "suggested_decision_evidence": "ARMENTROUT Executive Vice President Environment, Fish and Wildlife SCOTT ARMENTROUT Digitally signed by SCOTT ARMENTROUT Date: 2020.11.03 13:37:34 -08'00'",
    "top_decision_candidates": "2020-11-03 [clear_decision|high|score=13.2] doc=FONSI | ARMENTROUT Executive Vice President Environment, Fish and Wildlife SCOTT ARMENTROUT Digitally signed by SCOTT ARMENTROUT Date: 2020.11.03 13:37:34 -08 ||| 2020-11-03 [clear_decision|high|score=13.2] doc=FONSI | East Fork Irrigation District Modernization Project Page 5 Finding of No Significant Impact Issued in Portland, Oregon. ____________________________ S ||| 2020-07-01 [clear_decision|low|score=10.0] doc=FONSI | The final Plan-EA and NRCS\u2019 Finding of No significant Impact (FONSI) were released in July 2020. ||| 2020-11-18 [clear_decision|low|score=9.0] doc=FONSI | East Fork Irrigation District Modernization Project Page 1 Finding of No Significant Impact East Fork Irrigation District Modernization Project Findin ||| 2020-01-01 [clear_decision|low|score=9.0] doc=FONSI | The draft Plan-EA was released for public review and comment in January 2020."
  },
  {
    "sample_id": "41",
    "project_title": "Sunnydale-Velasco HOPE SF Master Plan Project",
    "process_type": "EIS",
    "lead_agency": "Department of Housing and Urban Development",
    "suggested_initiation_date": "2008-12-15",
    "suggested_initiation_evidence": "Pursuant to AB 32, ARB adopted a Scoping Plan in December 2008, outlining measures to meet the 2020 GHG reduction limits.",
    "top_initiation_candidates": "2008-12-01 [proxy_initiation|low|score=7.5] doc=DEIS | Pursuant to AB 32, ARB adopted a Scoping Plan in December 2008, outlining measures to meet the 2020 GHG reduction limits. ||| 2013-03-01 [proxy_initiation|low|score=3.8] doc=OTHER | Sunnydale\u2010Velasco HOPE SF Master Plan EIR/EIS 13 ESA / 210039 Scoping Report March 2013 2. ||| 2013-03-01 [proxy_initiation|low|score=3.8] doc=OTHER | Sunnydale\u2010Velasco HOPE SF Master Plan EIR/EIS 1 ESA / 210039 Scoping Report March 2013 1. ||| 2014-12-01 [proxy_initiation|low|score=1.5] doc=DEIS | 2010.0305E Draft EIR/EIS December 2014 SWRCB State Water Resources Control Board TAAS Theoretically Available Annual Sunlight TeNS technical noise sup ||| 2014-12-01 [proxy_initiation|low|score=1.5] doc=DEIS | 2010.0305E Draft EIR/EIS December 2014 8.2 Glossary of Terms CEQA (California Environmental Quality Act.",
    "suggested_decision_date": "2013-09-27",
    "suggested_decision_evidence": "SB 743, which amended the Public Resources Code to add section 21099, was signed by Governor Brown on September 27, 2013.",
    "top_decision_candidates": "2013-09-27 [clear_decision|medium|score=7.5] doc=DEIS | SB 743, which amended the Public Resources Code to add section 21099, was signed by Governor Brown on September 27, 2013."
  },
  {
    "sample_id": "42",
    "project_title": "Pokagon Band of Potawatomi Indians Fee-to-Trust Transfer for Tribal Village and Casino City of South Bend, Indiana",
    "process_type": "EIS",
    "lead_agency": "Bureau of Indian Affairs",
    "suggested_initiation_date": "2012-05-14",
    "suggested_initiation_evidence": "Bureau of Indian Affairs (BIA) received an application for the conveyance into trust of \u00b1165.81 acres of land currently held by the Pokagon Band of Potawatomi Indians in the City of South Bend, Saint Joseph County, Indiana on May 14th, 2012 with amendments on March 5, 2015.",
    "top_initiation_candidates": "2012-05-14 [clear_initiation|medium|score=8.2] doc=FEIS | Bureau of Indian Affairs (BIA) received an application for the conveyance into trust of \u00b1165.81 acres of land currently held by the Pokagon Band of Po ||| 2012-05-14 [clear_initiation|medium|score=8.2] doc=FEIS | FINAL ENVIRONMENTAL IMPACT STATEMENT Pokagon Band of Potawatomi Indians Fee\u2010to\u2010Trust Transfer for Tribal Village and Casino City of South Bend, Indian ||| 2015-03-05 [clear_initiation|medium|score=7.5] doc=FEIS | Bureau of Indian Affairs (BIA) received an application for the conveyance into trust of \u00b1165.81 acres of land currently held by the Pokagon Band of Po ||| 2015-03-05 [clear_initiation|medium|score=7.5] doc=FEIS | FINAL ENVIRONMENTAL IMPACT STATEMENT Pokagon Band of Potawatomi Indians Fee\u2010to\u2010Trust Transfer for Tribal Village and Casino City of South Bend, Indian ||| 2015-04-30 [clear_initiation|medium|score=6.3] doc=FEIS | Final EIS, Pokagon Band of Potawatomi Indians Fee-to-Trust Transfer for Tribal Village and Casino, South Bend, Indiana 1: Purpose and Need 1-7 June 20",
    "suggested_decision_date": "2016-06-15",
    "suggested_decision_evidence": "2000 Cliff Mine Drive, Suite 420 Pittsburgh, PA 15275 June 2016",
    "top_decision_candidates": "2016-06-01 [proxy_decision|low|score=5.0] doc=FEIS | 2000 Cliff Mine Drive, Suite 420 Pittsburgh, PA 15275 June 2016"
  },
  {
    "sample_id": "43",
    "project_title": "South Powder River Basin Coal",
    "process_type": "EIS",
    "lead_agency": "",
    "suggested_initiation_date": "1990-01-15",
    "suggested_initiation_evidence": "These federal coal lands are located within the Powder River Federal Coal Region, which was decertified in January 1990.",
    "top_initiation_candidates": "2000-10-25 [clear_initiation|high|score=9.0] doc=ROD | PUBLIC INVOLVEMENT On September 12, 2000, the BLM published notice of the receipt of this lease application in the Federal Register. Copies of this no ||| 2001-10-03 [clear_initiation|high|score=9.0] doc=ROD | PUBLIC INVOLVEMENT On September 12, 2000, the BLM published notice of the receipt of this lease application in the Federal Register. Copies of this no ||| 2001-09-25 [clear_initiation|high|score=9.0] doc=ROD | PUBLIC INVOLVEMENT On September 12, 2000, the BLM published notice of the receipt of this lease application in the Federal Register. Copies of this no ||| 2001-10-02 [clear_initiation|high|score=9.0] doc=ROD | PUBLIC INVOLVEMENT On September 12, 2000, the BLM published notice of the receipt of this lease application in the Federal Register. Copies of this no ||| 2001-10-10 [clear_initiation|high|score=9.0] doc=ROD | PUBLIC INVOLVEMENT On September 12, 2000, the BLM published notice of the receipt of this lease application in the Federal Register. Copies of this no",
    "suggested_decision_date": "2000-03-10",
    "suggested_decision_evidence": "U.S. DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT RECORD OF DECISION NARO NORTH FEDERAL COAL LEASE APPLICATION WYW150210 CAMPBELL COUNTY, WYOMING INTRODUCTION Powder River Coal Company filed an application with the Bureau of Land Management (BLM) to lease two tracts of Federal coal as mainte",
    "top_decision_candidates": "2000-03-10 [clear_decision|medium|score=12.5] doc=ROD | U.S. DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT RECORD OF DECISION NARO NORTH FEDERAL COAL LEASE APPLICATION WYW150210 CAMPBELL COUNTY, WYOM ||| 2003-12-24 [proxy_decision|medium|score=11.8] doc=ROD | The BLM and EPA each published a Notice of Availability of the Final EIS in the Federal Register on December 24, 2003. ||| 2003-12-24 [proxy_decision|medium|score=11.8] doc=ROD | The BLM and EPA each published a Notice of Availability of the Final EIS in the Federal Register on December 24, 2003. ||| 2004-04-23 [clear_decision|medium|score=11.5] doc=ROD | The Decision Notice for this EA was signed by the Medicine Bow-Routt National Forests and Thunder Basin National Grassland Forest Supervisor on April ||| 2004-05-04 [clear_decision|medium|score=11.2] doc=ROD | The Forest Service provided consent to lease the lands with NFS surface in the NARO North LBA Tract in a decision signed on May 4, 2004."
  },
  {
    "sample_id": "44",
    "project_title": "L18K Earthscope Seismic Station Assignment",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2019-03-18",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2019-03-18",
    "top_initiation_candidates": "2019-03-18 [clear_initiation|high|score=10.1] doc=None | BLM NEPA Register project start date: 2019-03-18",
    "suggested_decision_date": "2019-04-16",
    "suggested_decision_evidence": "-4/16/2019 Date Bonnie Million Anchorage Field Manager 18K Earthscope Seismic Station Assignment DOI-BLM-AK-A010-2019-0012-CX 3",
    "top_decision_candidates": "2019-07-01 [proxy_decision|low|score=4.5] doc=CE | BACKGROUND Project Name / Type: NEPA Register Number: Case File Number: Location / Legal Description: Applicant (if any): Description of Proposed Acti ||| 2019-07-01 [proxy_decision|low|score=4.5] doc=CE | 18K Earthscope Seismic Station Assignment DOI-BLM-AK-A010-2019-0012-CX 2 EXTRAORDINARY CIRCUMSTANCES 1. ||| 2019-07-01 [proxy_decision|low|score=4.5] doc=CE | -4/16/2019 Date Bonnie Million Anchorage Field Manager 18K Earthscope Seismic Station Assignment DOI-BLM-AK-A010-2019-0012-CX 3 ||| 2019-04-16 [clear_decision|low|score=4.0] doc=CE | -4/16/2019 Date Bonnie Million Anchorage Field Manager 18K Earthscope Seismic Station Assignment DOI-BLM-AK-A010-2019-0012-CX 3"
  },
  {
    "sample_id": "45",
    "project_title": "Bonneville Power Administration South Oregon Coast Reinforcement Project",
    "process_type": "EIS",
    "lead_agency": "Power Marketing Administration",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "46",
    "project_title": "Petroleum Production at Maximum Efficient Rate Naval Petroleum Reserve No. 1 (Elk Hills) Kern County, California",
    "process_type": "EIS",
    "lead_agency": "Department of Energy",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "1979-08-01 [proxy_initiation|low|score=-0.8] doc=FEIS | DOE/EIS-0012 Final Environmental lmpa t-Statement . ---- - Ker.n10ounty, California- 1-...---1--- b ..... 4' \u2022 .. - - - \u2022 \u2022 - -- - - - U.S. Department",
    "suggested_decision_date": "1979-08-15",
    "suggested_decision_evidence": "Department of Energy August 1979",
    "top_decision_candidates": "1979-08-01 [proxy_decision|low|score=4.2] doc=FEIS | Department of Energy August 1979 ||| 1975-12-01 [proxy_decision|low|score=4.0] doc=FEIS | Sei smi c Engi neeri ng Program Report , October - December 1975 , U ."
  },
  {
    "sample_id": "47",
    "project_title": "Phase III Moss Mine Expansion and Exploration Project",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2006-09-15",
    "suggested_initiation_evidence": "Fonn 1!!42-1 (September 2006) .J3 CFR SUBPART l82t-GENERAL tNFORi\\lATION Sec. I !!21.10 Where are BLM offices localed'! (a) In addition lo lhc Headquarters Office in Washington, D.C. and seven national k\\'el support and service centers, BU,,1 operates 12 Stale Offices each ha\\'ing se\\\u00b7eral subsidiar",
    "top_initiation_candidates": "2006-09-01 [proxy_initiation|low|score=9.2] doc=ROD | Fonn 1!!42-1 (September 2006) .J3 CFR SUBPART l82t-GENERAL tNFORi\\lATION Sec. I !!21.10 Where are BLM offices localed'! (a) In addition lo lhc Headqua ||| 2020-03-01 [proxy_initiation|low|score=5.5] doc=FONSI | Golden Vertex Corp. Phase III Moss Mine Expansion and Exploration Project Finding of No Significant Impacts DO1-BLM-AZ-C0l 0-2019-0033-EA March 2020 ||| 2020-03-01 [proxy_initiation|low|score=0.5] doc=EA | Department of the Interior Bureau of Land Management Colorado River District Kingman Field Office 2755 Mission Boulevard Kingman, Arizona 86401 (928) ||| 2020-03-01 [proxy_initiation|low|score=0.5] doc=EA | United States Department of the Interior Bureau of Land Management Mine Plan of Operations for the Phase III Moss Mine Expansion and Exploration Proje ||| 2020-03-01 [proxy_initiation|low|score=0.0] doc=EA | Phase III Moss Mine Expansion and Exploration Project Final Environmental Assessment Mohave County, Arizona March 2020 Page 48 Under the No Action alt",
    "suggested_decision_date": "2020-03-15",
    "suggested_decision_evidence": "Phase III Moss Mine Expansion and Exploration Project Finding of No Significant Impacts DO1-BLM-AZ-C0l 0-2019-0033-EA March 2020",
    "top_decision_candidates": "2020-03-01 [clear_decision|low|score=10.5] doc=FONSI | Phase III Moss Mine Expansion and Exploration Project Finding of No Significant Impacts DO1-BLM-AZ-C0l 0-2019-0033-EA March 2020 ||| 2019-07-01 [proxy_decision|low|score=10.0] doc=ROD | PHASE III MOSS MINE EXPANSION AND EXPLORATION PROJECT MOHAVE COUNTY, ARIZONA Environmental Assessment DOI-BLM-AZ-C0l0-2019-0033-EA INTRODUCTION/BACKGR ||| 2019-08-29 [clear_decision|low|score=9.2] doc=ROD | period at the Laughlin Ranch Golf Club, Bullhead City, Arizona, on August 29, 2019. ||| 2019-07-12 [clear_decision|low|score=9.0] doc=ROD | Tribal consultation letters for the proposed Moss Mine Expansion and Exploration Project were issued by BLM to potentially affected tribes on July 12, ||| 2019-12-10 [clear_decision|low|score=9.0] doc=ROD | The environmental assessment (EA) was placed on the project webpage for a public comment and review period from December 10, 2019 through January 10,"
  },
  {
    "sample_id": "48",
    "project_title": "Restoration Handbook for Sagebrush Steppe Ecosystems with Emphasis on Greater Sage-Grouse Habitat\u2014 Part 3. Site Level Restoration Decisions",
    "process_type": "EA",
    "lead_agency": "United States Geological Survey",
    "suggested_initiation_date": "2022-07-01",
    "suggested_initiation_evidence": "U.S. Department of the Interior Bureau of Land Management Uncompahgre Field Office 2465 S. Townsend Ave. Montrose, CO 81401 Finding of No Significant Impact (FONSI) Programmatic Ecological Restoration DOI-BLM-CO-S050-2022-0035 EA BACKGROUND A megadrought lasting 22 years is impacting soil and vegeta",
    "top_initiation_candidates": "2022-07-01 [clear_initiation|medium|score=12.2] doc=FONSI | U.S. Department of the Interior Bureau of Land Management Uncompahgre Field Office 2465 S. Townsend Ave. Montrose, CO 81401 Finding of No Significant",
    "suggested_decision_date": "2022-12-19",
    "suggested_decision_evidence": "Signature of Authorized Officer Suzanne Copping Uncompahgre Field Manager SUZANNE COPPING Digitally signed by SUZANNE COPPING Date: 2022.12.19 15:38:16 -07'00'",
    "top_decision_candidates": "2022-12-19 [clear_decision|high|score=13.5] doc=FONSI | Signature of Authorized Officer Suzanne Copping Uncompahgre Field Manager SUZANNE COPPING Digitally signed by SUZANNE COPPING Date: 2022.12.19 15:38:1 ||| 2022-12-19 [clear_decision|high|score=13.5] doc=FONSI | 5 4. Effects that would violate Federal, State, Tribal, or local law protecting the environment. Degree to which the possible effects on the quality o ||| 2022-07-01 [proxy_decision|low|score=9.8] doc=FONSI | Montrose, CO 81401 Finding of No Significant Impact (FONSI) Programmatic Ecological Restoration DOI-BLM-CO-S050-2022-0035 EA BACKGROUND A megadrought ||| 2022-12-19 [clear_decision|high|score=7.5] doc=EA | SIGNATURE OF AUTHORIZED OFFICER: Suzanne Copping Uncompahgre Field Manager ATTACHMENTS Attachment A: Decision Area Map Attachment B: Ecological Restor"
  },
  {
    "sample_id": "49",
    "project_title": "East Altamont Energy Center",
    "process_type": "EA",
    "lead_agency": "Power Marketing Administration",
    "suggested_initiation_date": "2001-03-20",
    "suggested_initiation_evidence": "BIOLOGICAL RESOURCES 5.2-54 September, 2002 CEC (California Energy Commission). 2001h. East Altamont Energy Center Third Set of Data Requests. Dated and docketed September 25, 2001. CEC (California Energy Commission). 1999. Avian Collision and Electrocution: An Annotated Bibliography. CEC, Sacrament",
    "top_initiation_candidates": "2001-03-20 [clear_initiation|medium|score=6.3] doc=EA | BIOLOGICAL RESOURCES 5.2-54 September, 2002 CEC (California Energy Commission). 2001h. East Altamont Energy Center Third Set of Data Requests. Dated a ||| 2001-03-29 [clear_initiation|medium|score=6.3] doc=EA | BIOLOGICAL RESOURCES 5.2-54 September, 2002 CEC (California Energy Commission). 2001h. East Altamont Energy Center Third Set of Data Requests. Dated a ||| 2001-05-01 [clear_initiation|medium|score=6.2] doc=EA | BIOLOGICAL RESOURCES 5.2-54 September, 2002 CEC (California Energy Commission). 2001h. East Altamont Energy Center Third Set of Data Requests. Dated a ||| 2001-05-03 [clear_initiation|medium|score=6.2] doc=EA | BIOLOGICAL RESOURCES 5.2-54 September, 2002 CEC (California Energy Commission). 2001h. East Altamont Energy Center Third Set of Data Requests. Dated a ||| 2001-07-09 [clear_initiation|medium|score=6.0] doc=EA | BIOLOGICAL RESOURCES 5.2-54 September, 2002 CEC (California Energy Commission). 2001h. East Altamont Energy Center Third Set of Data Requests. Dated a",
    "suggested_decision_date": "2002-04-03",
    "suggested_decision_evidence": "Any changes to the April 3, 2002 landscaping plan and management practices should be approved by the USFWS, CDFG, and Western in consultation with Energy Commission staff.",
    "top_decision_candidates": "2002-04-03 [clear_decision|medium|score=6.0] doc=EA | Any changes to the April 3, 2002 landscaping plan and management practices should be approved by the USFWS, CDFG, and Western in consultation with Ene ||| 2002-04-03 [clear_decision|medium|score=6.0] doc=EA | BIOLOGICAL RESOURCES 5.2-38 September, 2002 Mitigation for Landscaping and Visual Screening (Condition of Certification BIO-14) Biology staff prefers"
  },
  {
    "sample_id": "50",
    "project_title": "Condit Hydroelectric Project",
    "process_type": "EIS",
    "lead_agency": "Department of Energy",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "51",
    "project_title": "Proposed Resource Management Plan for the Buffalo Field Office Planning Area",
    "process_type": "EIS",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2008-11-15",
    "suggested_decision_evidence": "Well Statistics for Campbell, Johnson, and Sheridan Counties, November 2008 ................",
    "top_decision_candidates": "2008-11-01 [proxy_decision|low|score=4.0] doc=FEIS | Well Statistics for Campbell, Johnson, and Sheridan Counties, November 2008 ................ ||| 2015-05-01 [proxy_decision|low|score=4.0] doc=FEIS | Department of the Interior Bureau of Land Management Buffalo Field Office, Wyoming May 2015"
  },
  {
    "sample_id": "52",
    "project_title": "Noyes Apiaries Leases",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2017-05-18 [clear_initiation|high|score=5.2] doc=None | BLM NEPA Register project start date: 2017-05-18",
    "suggested_decision_date": "2017-05-18",
    "suggested_decision_evidence": "/s/ Richard White Field Manager 5/18/17 Date 1-2 Categorical Exclusions: Extraordinary Circumstances Worksheet NOTE: Italicized prompts (in \u201cRationale\u201d) should not be visible in the final document.",
    "top_decision_candidates": "2017-05-18 [clear_decision|high|score=8.2] doc=CE | /s/ Richard White Field Manager 5/18/17 Date 1-2 Categorical Exclusions: Extraordinary Circumstances Worksheet NOTE: Italicized prompts (in \u201cRationale ||| 2017-02-15 [clear_decision|low|score=5.0] doc=CE | Sisson 2/15/17 8. ||| 2017-02-15 [clear_decision|low|score=5.0] doc=CE | Sisson 2/15/17 3. ||| 2017-07-01 [proxy_decision|low|score=4.0] doc=CE | Department of the Interior Bureau of Land Management Cottonwood Field Office 1 Butte Drive, Cottonwood, ID 83522 Categorical Exclusion Documentation N ||| 2017-04-20 [clear_decision|low|score=4.0] doc=CE | The Nez Perce tribe was consulted about these apiaries on April 20, 2017 and raised no objections to the re-issuance of these permits."
  },
  {
    "sample_id": "53",
    "project_title": "Cooley Road Free Use Permit",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2020-01-01",
    "suggested_decision_evidence": "Document filename date (ce): 2020-01-01",
    "top_decision_candidates": "2020-01-01 [clear_decision|medium|score=8.0] doc=CE | Document filename date (ce): 2020-01-01 ||| 2020-09-08 [clear_decision|low|score=4.8] doc=CE | 1849 United States Department of the Interior - Bureau of Land Management Cooley Lake Road DeBaca County Road Department Free Use Permit T 02 S - R 25 ||| 2020-09-08 [clear_decision|low|score=4.8] doc=CE | Page 4 of 4 United States Department of the Interior - Bureau of Land Management Cooley Lake Road DeBaca County Road Department Free Use Permit T 02 S ||| 2020-09-08 [clear_decision|low|score=4.8] doc=CE | Stipulations/Mitigations: See Exhibit A, Attached Condition of Approval Authorized Official: Ruben Sanchez Roswell Field Office Assistant Field Manage ||| 2020-09-08 [clear_decision|low|score=4.8] doc=CE | Page 4 of 4 Legend County Road Main Road United St MC Department of the Interior - Bureau of Land Mement Cooley Lake Road DeBaca County Road Departmen"
  },
  {
    "sample_id": "54",
    "project_title": "HMIS Annual Categorical Exclusion (CX) B2.5, Facility Safety and Environmental Improvements for CY 2022",
    "process_type": "CE",
    "lead_agency": "Department of Energy",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2021-10-06",
    "suggested_decision_evidence": "DOE CX Register determination date (cx-24727.0): 2021-10-06",
    "top_decision_candidates": "2021-10-06 [clear_decision|high|score=15.0] doc=categorical exclusion determin | DOE CX Register determination date (cx-24727.0): 2021-10-06"
  },
  {
    "sample_id": "55",
    "project_title": "Valley View Trail Rehabilitation",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2015-11-06",
    "suggested_decision_evidence": "DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT United States Department of the Interior Bureau of Land Management Date 11/06/2015 Categorical Exclusion DOI-BLM-CA-C05000-2016-005 Valley View Trail Rehabilitation North Cow Mountain Recreation Area Mendocino County, CA U.S.",
    "top_decision_candidates": "2015-11-06 [clear_decision|low|score=4.2] doc=CE | DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT United States Department of the Interior Bureau of Land Management Date 11/06/2015 Categorical Ex ||| 2015-11-06 [clear_decision|low|score=4.2] doc=CE | Department of the Interior Bureau of Land Management Ukiah Field Office 2550 North State Street Ukiah, CA 95482 Phone: (707) 468-4000 FAX: (707) 468-4 ||| 2015-12-19 [clear_decision|low|score=4.0] doc=CE | 2 Valley View Trail Rehabilitation DOI-BLM-CA-C05000-2016-005 Background Information: On Friday December 19th, 2015 the Ukiah Valley Trails group prop"
  },
  {
    "sample_id": "56",
    "project_title": "South Fort Meade Phosphate Mine",
    "process_type": "EA",
    "lead_agency": "Corps of Engineers--Civil Works",
    "suggested_initiation_date": "2007-05-25",
    "suggested_initiation_evidence": "Army Corps of Engineers 10117 Princess Palm Drive, Suite 120 Tampa, Florida336lO-8300 Service Federal Activity Code: 4l420-2007-FA-1006 Service Consultation Code: 41420-2007-F-i 189 Corps Application No.: SAJ-1997-4099 (IP-MGH) Date Received: May 25, 2007 Formal consultation initiation date: May 13,",
    "top_initiation_candidates": "2007-05-25 [clear_initiation|medium|score=5.5] doc=OTHER | Army Corps of Engineers 10117 Princess Palm Drive, Suite 120 Tampa, Florida336lO-8300 Service Federal Activity Code: 4l420-2007-FA-1006 Service Consul ||| 2010-05-13 [clear_initiation|medium|score=5.5] doc=OTHER | Army Corps of Engineers 10117 Princess Palm Drive, Suite 120 Tampa, Florida336lO-8300 Service Federal Activity Code: 4l420-2007-FA-1006 Service Consul ||| 2009-10-09 [clear_initiation|medium|score=5.2] doc=OTHER | On October 6, 2009, the applicant, his environmental consultant, and a Service biologist conducted a site visit of the proposed phosphate mine parcel. ||| 2009-10-28 [clear_initiation|medium|score=5.2] doc=OTHER | On October 6, 2009, the applicant, his environmental consultant, and a Service biologist conducted a site visit of the proposed phosphate mine parcel. ||| 2009-10-30 [clear_initiation|medium|score=5.2] doc=OTHER | On October 6, 2009, the applicant, his environmental consultant, and a Service biologist conducted a site visit of the proposed phosphate mine parcel.",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "57",
    "project_title": "Peregrine Exploration Program",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2021-09-17",
    "suggested_initiation_evidence": "On September 17, 2021, prior to Emerald House submitting a Plan of Operations for its 2021/2022 proposed operations, the BLM received a letter signed by a group of 14 environmental organizations regarding the five-year Peregrine Exploration Program that began in early 2021.",
    "top_initiation_candidates": "2021-09-17 [clear_initiation|high|score=14.1] doc=ROD | On September 17, 2021, prior to Emerald House submitting a Plan of Operations for its 2021/2022 proposed operations, the BLM received a letter signed ||| 2021-09-17 [clear_initiation|high|score=14.1] doc=ROD | On December 21, 2021, BLM received a letter signed by a group of 8 environmental organizations, who had been included on the September 17, 2021 letter ||| 2021-12-21 [clear_initiation|high|score=13.1] doc=ROD | On December 21, 2021, BLM received a letter signed by a group of 8 environmental organizations, who had been included on the September 17, 2021 letter ||| 2021-09-01 [proxy_initiation|low|score=9.7] doc=ROD | During discussions between the USFWS and Emerald House in September 2021, the USFWS described very low polar bear densities (and even lower likelihood ||| 2021-09-01 [proxy_initiation|low|score=9.7] doc=ROD | Page 6 of 8 Informal consultation between BLM and the U.S. Fish and Wildlife Service (USFWS) concluded November 4, 2020, covering a 5-year period (202",
    "suggested_decision_date": "2022-02-07",
    "suggested_decision_evidence": "/s/ Nichelle Jones Date: February 7, 2022 Nichelle Jones Arctic District Manager /s/ Robert Brumbaugh Date: February 7, 2022 Robert Brumbaugh acting for Wayne M.",
    "top_decision_candidates": "2022-02-07 [clear_decision|high|score=13.2] doc=ROD | /s/ Nichelle Jones Date: February 7, 2022 Nichelle Jones Arctic District Manager /s/ Robert Brumbaugh Date: February 7, 2022 Robert Brumbaugh acting f ||| 2022-02-07 [clear_decision|high|score=13.2] doc=ROD | Page 8 of 8 If you wish to file a petition for stay pursuant to 43 CFR Part 4.21(b), the petition for stay should accompany your notice of appeal and ||| 2021-09-17 [clear_decision|medium|score=11.8] doc=ROD | The BLM also received a letter on January 7, 2022, signed by a group of 9 environmental organizations (the same organizations included on the Septembe ||| 2022-01-07 [clear_decision|medium|score=11.5] doc=ROD | The BLM also received a letter on January 7, 2022, signed by a group of 9 environmental organizations (the same organizations included on the Septembe ||| 2022-07-01 [proxy_decision|low|score=10.2] doc=ROD | Decision Record ADDENDUM Peregrine Exploration Program Environmental Assessment: DOI-BLM-AK-R000-2022-0004-EA Emerald House, LLC."
  },
  {
    "sample_id": "58",
    "project_title": "Nez Perce Tribe Integrated Resource Management Plan",
    "process_type": "EIS",
    "lead_agency": "Bureau of Indian Affairs",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2023-02-01 [proxy_initiation|low|score=0.2] doc=OTHER | Appendix D \u2013 Regulatory Setting February 2023 185 Nez Perce Tribe IRMP Draft Programmatic Environmental Impact Statement shortcomings, such as greater ||| 2023-07-01 [proxy_initiation|low|score=0.2] doc=FEIS | Section 6.0 Consultation & Coordination July 2023 136 Nez Perce Tribe IRMP Final Programmatic Environmental Impact Statement SECTION 6 \u2013 CONSULTATION ||| 2023-07-01 [proxy_initiation|low|score=0.2] doc=FEIS | Appendix D \u2013 Regulatory Setting July 2023 186 Nez Perce Tribe IRMP Final Programmatic Environmental Impact Statement shortcomings, such as greater slo ||| 2016-01-01 [proxy_initiation|low|score=-0.5] doc=FEIS | \u2022 Public service announcements aired on KIYE 98.1 FM (January 2016) \u2022 Nez Perce Tribe Helpdesk notifications (January 2016) \u2022 Newspaper articles publi ||| 2016-02-01 [proxy_initiation|low|score=-0.5] doc=FEIS | \u2022 Public service announcements aired on KIYE 98.1 FM (January 2016) \u2022 Nez Perce Tribe Helpdesk notifications (January 2016) \u2022 Newspaper articles publi",
    "suggested_decision_date": "2015-12-07",
    "suggested_decision_evidence": "Section 1.0 Introduction February 2023 3 Nez Perce Tribe IRMP Draft Programmatic Environmental Impact Statement of public hearing(s) to receive comments from the public concerning this Draft EIS. Substantive comments received on the Draft EIS during the comment period, including those submitted or r",
    "top_decision_candidates": "2015-12-07 [clear_decision|high|score=8.0] doc=OTHER | Section 1.0 Introduction February 2023 3 Nez Perce Tribe IRMP Draft Programmatic Environmental Impact Statement of public hearing(s) to receive commen ||| 2015-12-07 [clear_decision|high|score=8.0] doc=FEIS | Section 1.0 Introduction July 2023 3 Nez Perce Tribe IRMP Final Programmatic Environmental Impact Statement of public hearing(s) to receive comments f"
  },
  {
    "sample_id": "59",
    "project_title": "Marine Geophysical Survey (MATRIX) by the US Geological Survey",
    "process_type": "EA",
    "lead_agency": "United States Geological Survey",
    "suggested_initiation_date": "2016-07-15",
    "suggested_initiation_evidence": "In July 2016, the National Oceanic and Atmospheric Administration\u2019s (NOAA) National Marine Fisheries Service (NMFS) released new technical guidance for assessing the effects of anthropogenic sound on marine mammal hearing (NMFS 2016a).",
    "top_initiation_candidates": "2016-07-01 [proxy_initiation|low|score=5.2] doc=EA | In July 2016, the National Oceanic and Atmospheric Administration\u2019s (NOAA) National Marine Fisheries Service (NMFS) released new technical guidance fo ||| 2016-07-01 [proxy_initiation|low|score=5.2] doc=EA | In July 2016, the National Oceanic and Atmospheric Administration\u2019s (NOAA) National Marine Fisheries Service (NMFS) released new technical guidance fo ||| 2018-07-01 [proxy_initiation|low|score=-1.5] doc=EA | Ms. Jolie Harrison 2 July 2018 Page 3 readily estimate ranges to the cumulative sound exposure level (SELcum) thresholds. In the absence of such a mod ||| 2018-07-01 [proxy_initiation|low|score=-1.5] doc=EA | Proposed Authorization As a result of these preliminary determinations, NMFS proposes to issue an IHA to SIO for conducting a low- energy seismic surv",
    "suggested_decision_date": "2018-07-15",
    "suggested_decision_evidence": "Jolie Harrison 2 July 2018 Page 3 readily estimate ranges to the cumulative sound exposure level (SELcum) thresholds.",
    "top_decision_candidates": "2018-07-01 [proxy_decision|low|score=3.5] doc=EA | Jolie Harrison 2 July 2018 Page 3 readily estimate ranges to the cumulative sound exposure level (SELcum) thresholds. ||| 2018-06-01 [proxy_decision|low|score=3.0] doc=EA | The guidance was updated, but effectively remained the same, in June 2018 (NMFS 2018)."
  },
  {
    "sample_id": "60",
    "project_title": "Yellowstone Pipe Out-of-Service Abandonment",
    "process_type": "EA",
    "lead_agency": "Forest Service",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "61",
    "project_title": "Grand Coulee's Third Powerplant 500-kilovolt Transmission Line Replacement Project",
    "process_type": "EA",
    "lead_agency": "Bureau of Reclamation",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2011-05-01 [proxy_initiation|low|score=-1.0] doc=EA | CHAPTER 3 AFFECTED ENVIRONMENTS, ENVIRONMENTAL CONSEQUENCES, AND MITIGATION MEASURES 4-4 Preliminary Environmental Assessment\u2014May 2011 Agency Law Comp ||| 2011-05-01 [proxy_initiation|low|score=-1.0] doc=EA | CHAPTER 2 NEED AND PURPOSE FOR ACTION 2-2 Preliminary Environmental Assessment\u2014May 2011 \uf0a7 Reuse the Spreading Yard take off structure. ||| 2011-05-01 [proxy_initiation|low|score=-1.0] doc=EA | CHAPTER 3 AFFECTED ENVIRONMENTS, ENVIRONMENTAL CONSEQUENCES, AND MITIGATION MEASURES 3-126 Preliminary Environmental Assessment\u2014May 2011 need to be up ||| 2011-05-01 [proxy_initiation|low|score=-1.0] doc=EA | Preliminary Environmental Assessment\u2014May 2011 3-1 Chapter 3 Affected Environments, Environmental Consequences, and Mitigation Measures 3.1 Introductio ||| 2011-05-01 [proxy_initiation|low|score=-1.0] doc=EA | AIR QUALITY Preliminary Environmental Assessment\u2014May 2011 4-7 The proposed project could potentially impact birds through collisions with power lines",
    "suggested_decision_date": "2011-01-01",
    "suggested_decision_evidence": "Document filename date (fonsi): 2011-01-01",
    "top_decision_candidates": "2011-01-01 [clear_decision|medium|score=13.0] doc=FONSI | Document filename date (fonsi): 2011-01-01 ||| 2009-03-20 [clear_decision|low|score=9.2] doc=FONSI | Notice of floodplain and wetlands involvement was included in the letter sent to the project mailing list announcing the availability of the Prelimina ||| 2009-03-20 [clear_decision|low|score=9.2] doc=FONSI | Grand Coulee\u2019s Third Powerplant 500-kV Transmission Line Replacement Project BPA\u2019s Finding of No Significant Impact 8 \uf0b7 Minor increases in dust and ex ||| 2011-05-01 [proxy_decision|low|score=4.0] doc=EA | PURPOSES Preliminary Environmental Assessment\u2014May 2011 1-7 BPA Environmental Assessment Determination (DOE/EA-1679). ||| 2011-02-01 [proxy_decision|low|score=3.0] doc=EA | COMPARISON OF ALTERNATIVES Preliminary Environmental Assessment\u2014February 2011 2-9 facilities)."
  },
  {
    "sample_id": "62",
    "project_title": "Stud Horse Butte (SHB) 213-08A Natural Gas Well",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2016-10-26",
    "suggested_initiation_evidence": "Hiner Field Manager 10/26/2016 Date Administrative Review or Appeal Opportunities Any party adversely affected by this decision may request a State Director Review in accordance with 43 CFR \u00a7 3165.3.",
    "top_initiation_candidates": "2016-10-26 [clear_initiation|medium|score=6.0] doc=CE | Hiner Field Manager 10/26/2016 Date Administrative Review or Appeal Opportunities Any party adversely affected by this decision may request a State Di",
    "suggested_decision_date": "2020-10-23",
    "suggested_decision_evidence": "In order to meet this condition, the proposed natural gas well must be spudded on or before October 23, 2020.",
    "top_decision_candidates": "2020-10-23 [clear_decision|low|score=4.0] doc=CE | In order to meet this condition, the proposed natural gas well must be spudded on or before October 23, 2020. ||| 2016-12-01 [clear_decision|low|score=2.2] doc=CE | The proposed spud date for this proposed natural gas well is about December 1, 2016. ||| 2016-12-01 [clear_decision|low|score=2.2] doc=CE | The proposed spud date for this well is about December 1, 2016. ||| 2017-07-01 [proxy_decision|low|score=2.0] doc=CE | DOI-BLM-WY-0010-2017-0016-CX A. ||| 2015-10-23 [clear_decision|low|score=2.0] doc=CE | NEPA document (WY-100-EA15-182) was finalized (10/23/15) within the last five years."
  },
  {
    "sample_id": "63",
    "project_title": "Cuyahoga County Agricultural Society Wind Energy Project",
    "process_type": "EA",
    "lead_agency": "Energy Programs",
    "suggested_initiation_date": "2008-10-29",
    "suggested_initiation_evidence": "The Agricultural Society has provided opportunities for public involvement since July 4, 2008, in an attempt to educate the public about this project and to provide an opportunity for public comment. These opportunities have included public engagement by the City of Berea, the City of Middleburg Hei",
    "top_initiation_candidates": "2008-10-29 [clear_initiation|medium|score=8.5] doc=None | The Agricultural Society has provided opportunities for public involvement since July 4, 2008, in an attempt to educate the public about this project ||| 2008-11-24 [clear_initiation|medium|score=8.5] doc=None | The Agricultural Society has provided opportunities for public involvement since July 4, 2008, in an attempt to educate the public about this project ||| 2008-12-02 [clear_initiation|medium|score=8.2] doc=None | Zoning Approved 12/2/08 Council & Mayor Final Approved 9/16/09 Building & Zoning Permit end date Expired 12/16/09 Building & Zoning 280\u2019 Zoning Approv ||| 2008-05-22 [clear_initiation|medium|score=8.2] doc=None | The Agricultural Society has provided opportunities for public involvement since July 4, 2008, in an attempt to educate the public about this project ||| 2008-09-17 [clear_initiation|medium|score=8.2] doc=None | The Agricultural Society has provided opportunities for public involvement since July 4, 2008, in an attempt to educate the public about this project",
    "suggested_decision_date": "2011-01-01",
    "suggested_decision_evidence": "Document filename date (fonsi): 2011-01-01",
    "top_decision_candidates": "2011-01-01 [clear_decision|medium|score=13.2] doc=FONSI | Document filename date (fonsi): 2011-01-01 ||| 2011-01-01 [clear_decision|high|score=8.2] doc=EA | and Woodland Ave, Cleveland, Ohio 44104 Archbold Area Schools Wind Energy Project \u2013 DOE/EA-1820 (Draft EA issued January 2011) 500-kilowatt wind turbi ||| 2010-08-01 [clear_decision|high|score=8.2] doc=EA | and Woodland Ave, Cleveland, Ohio 44104 Archbold Area Schools Wind Energy Project \u2013 DOE/EA-1820 (Draft EA issued January 2011) 500-kilowatt wind turbi ||| 2010-02-01 [clear_decision|high|score=8.0] doc=EA | and Woodland Ave, Cleveland, Ohio 44104 Archbold Area Schools Wind Energy Project \u2013 DOE/EA-1820 (Draft EA issued January 2011) 500-kilowatt wind turbi ||| 2009-12-16 [clear_decision|medium|score=6.0] doc=EA | On December 16, 2009, an application for zoning variance for the height of the proposed turbine was submitted to and approved by the Middleburg Board"
  },
  {
    "sample_id": "64",
    "project_title": "Right-of-Way for 8-inch Buried Gas Lift Line",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2022-08-09",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2022-08-09",
    "top_initiation_candidates": "2022-08-09 [clear_initiation|high|score=5.0] doc=None | BLM NEPA Register project start date: 2022-08-09",
    "suggested_decision_date": "2022-07-01",
    "suggested_decision_evidence": "NEPA COMPLIANCE RECORD FOR CATEGORICAL EXCLUSION (CX) Bureau of Land Management (BLM) Office: Carlsbad Field Office (CFO) DOI-BLM-NM-P020-2022-0956-CX IT4RM CX-2022-0000 XTO Holding, LLC Serial Number: NM-144885 Proposed Action: XTO Holding, LLC, is requesting a right-of-way for an 8-inch buried MAT",
    "top_decision_candidates": "2022-07-01 [proxy_decision|low|score=4.2] doc=CE | NEPA COMPLIANCE RECORD FOR CATEGORICAL EXCLUSION (CX) Bureau of Land Management (BLM) Office: Carlsbad Field Office (CFO) DOI-BLM-NM-P020-2022-0956-CX ||| 2022-07-01 [proxy_decision|low|score=4.2] doc=CE | NEPA COMPLIANCE RECORD FOR CATEGORICAL EXCLUSION (CX) Bureau of Land Management (BLM) Office: Carlsbad Field Office (CFO) DOI-BLM-NM-P020-2022-0956-CX"
  },
  {
    "sample_id": "65",
    "project_title": "Dog River Pipeline Replacement Project",
    "process_type": "EA",
    "lead_agency": "Forest Service",
    "suggested_initiation_date": "2019-02-15",
    "suggested_initiation_evidence": "After the comment period ended, Forest Service staff met with the Confederated Tribes of Warm Springs (February 2019) and a representative from the City of The Dalles (March 2019).",
    "top_initiation_candidates": "2019-02-01 [clear_initiation|medium|score=12.6] doc=FONSI | After the comment period ended, Forest Service staff met with the Confederated Tribes of Warm Springs (February 2019) and a representative from the Ci ||| 2019-02-01 [clear_initiation|medium|score=12.6] doc=FONSI | Dog River Pipeline Replacement Project Decision Notice Page 5 of 11 A Preliminary Environmental Assessment was published on November 10, 2018 and a 30 ||| 2018-11-10 [clear_initiation|medium|score=12.6] doc=FONSI | Dog River Pipeline Replacement Project Decision Notice Page 5 of 11 A Preliminary Environmental Assessment was published on November 10, 2018 and a 30 ||| 2019-03-01 [clear_initiation|medium|score=12.5] doc=FONSI | After the comment period ended, Forest Service staff met with the Confederated Tribes of Warm Springs (February 2019) and a representative from the Ci ||| 2019-03-01 [clear_initiation|medium|score=12.5] doc=FONSI | Dog River Pipeline Replacement Project Decision Notice Page 5 of 11 A Preliminary Environmental Assessment was published on November 10, 2018 and a 30",
    "suggested_decision_date": "2020-06-02",
    "suggested_decision_evidence": "The legal notice of the opportunity to object was published in The Oregonian newspaper on June 2, 2020.",
    "top_decision_candidates": "2020-06-02 [clear_decision|low|score=9.0] doc=FONSI | The legal notice of the opportunity to object was published in The Oregonian newspaper on June 2, 2020. ||| 2020-09-03 [clear_decision|low|score=9.0] doc=FONSI | An objection resolution meeting was held on September 3, 2020 with the Deputy Regional Forester who was the Objection Reviewing Official. ||| 2020-09-30 [clear_decision|low|score=9.0] doc=FONSI | In letters dated September 30, 2020, the Objection Reviewing Official documented the following: \u2022 The draft decision clearly describes the actions to"
  },
  {
    "sample_id": "66",
    "project_title": "Boise District Noxious Weed and Invasive Plant Management",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2016-07-01",
    "suggested_initiation_evidence": "DOI-BLM-ID-B000-2016-0002-EA Decision Record 10 Resource Treatment Type Protection Measures Authority/ Source Herbicide Treatments EOs, Occupied Habitat, PCH, and Slickspot Peppergrass Habitat \uf0b7 Aerial application of herbicide would not be authorized under the current proposed action in any habitat",
    "top_initiation_candidates": "2016-07-01 [clear_initiation|medium|score=6.1] doc=OTHER | DOI-BLM-ID-B000-2016-0002-EA Decision Record 10 Resource Treatment Type Protection Measures Authority/ Source Herbicide Treatments EOs, Occupied Habit ||| 2004-12-01 [proxy_initiation|low|score=4.5] doc=EA | Letter of Concurrence for the Addendum to the December 2004 Biological Assessment for the Normal Fire Emergency Stabilization and Rehabilitation Plan ||| 2018-06-01 [proxy_initiation|low|score=-1.5] doc=OTHER | Finding of No Significant Impact DOI-BLM-ID-B000-2016-0002-EA 1 United States Department of the Interior BUREAU OF LAND MANAGEMENT Boise District Offi",
    "suggested_decision_date": "2016-07-29",
    "suggested_decision_evidence": "The ROD was signed on July 29, 2016.",
    "top_decision_candidates": "2016-07-29 [clear_decision|high|score=7.2] doc=EA | The ROD was signed on July 29, 2016. ||| 2016-07-29 [clear_decision|high|score=7.2] doc=EA | Boise District Noxious Weed and Invasive Plant Management Environmental Assessment Purpose and Need 4 1.4 TIERING AND INCORPORATION BY REFERENCE The E ||| 2016-07-01 [clear_decision|medium|score=6.5] doc=OTHER | DOI-BLM-ID-B000-2016-0002-EA Decision Record 1 United States Department of the Interior BUREAU OF LAND MANAGEMENT Boise District Office 3948 Developme ||| 2018-06-04 [clear_decision|medium|score=5.5] doc=OTHER | DOI-BLM-ID-B000-2016-0002-EA Decision Record 1 United States Department of the Interior BUREAU OF LAND MANAGEMENT Boise District Office 3948 Developme ||| 2007-09-29 [clear_decision|high|score=5.2] doc=EA | The ROD was signed on September 29, 2007."
  },
  {
    "sample_id": "67",
    "project_title": "Belloq 2 State #2H, #5H, and #6H SWD Pipeline",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2017-03-06",
    "suggested_decision_evidence": "Authorized Official: /s/ Kari Vasenden 03/06/2017 Date Carlsbad Field Office Manager",
    "top_decision_candidates": "2017-03-06 [clear_decision|high|score=8.2] doc=CE | Authorized Official: /s/ Kari Vasenden 03/06/2017 Date Carlsbad Field Office Manager"
  },
  {
    "sample_id": "68",
    "project_title": "North Faber Livestock Exclosure Realignment",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2022-06-03",
    "suggested_decision_evidence": "BENJAMIN HILEMAN HILEMAN Digitally signed by BENJAMIN Date: 2022.06.03 10:28:02 -06'00' Authorized Officer/Date 7| Page",
    "top_decision_candidates": "2022-06-03 [clear_decision|high|score=8.0] doc=CE | BENJAMIN HILEMAN HILEMAN Digitally signed by BENJAMIN Date: 2022.06.03 10:28:02 -06'00' Authorized Officer/Date 7| Page ||| 2022-05-23 [clear_decision|low|score=4.0] doc=CE | Department of the Interior Bureau of Land Management Categorical Exclusion (CX) North Central Montana District Havre Field Office 3990 HWY 2 West Havr ||| 2022-07-01 [proxy_decision|low|score=4.0] doc=CE | Department of the Interior Bureau of Land Management Categorical Exclusion (CX) North Central Montana District Havre Field Office 3990 HWY 2 West Havr"
  },
  {
    "sample_id": "69",
    "project_title": "New Federal Courthouse in Eugene/Springfield, Lane County, Oregon",
    "process_type": "EIS",
    "lead_agency": "Real Property Activities",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "70",
    "project_title": "CVEA \u2013 Temporary electric line for Sourdough Bridge replacement",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2022-07-01",
    "suggested_decision_evidence": "ALYSIA HANCOCK Digitally signed by ALYSIA HANCOCK Date: 2022.08.12 15:27:23 -08'00' Marnie Graham Date Glennallen Field Manager DOI-BLM-AK-A020-2022-0017-CX 3",
    "top_decision_candidates": "2022-07-01 [clear_decision|high|score=14.0] doc=ROD | ALYSIA HANCOCK Digitally signed by ALYSIA HANCOCK Date: 2022.08.12 15:27:23 -08'00' Marnie Graham Date Glennallen Field Manager DOI-BLM-AK-A020-2022-0 ||| 2022-07-01 [clear_decision|high|score=14.0] doc=ROD | DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT CVEA \u2013 Temporary electric line for Sourdough Bridge replacement Categorical Exclusion, DOI-BLM-AK ||| 2022-07-01 [clear_decision|high|score=14.0] doc=ROD | ALYSIA HANCOCK Digitally signed by ALYSIA HANCOCK Date: 2022.08.12 15:28:42 -08'00' Marnie Graham Glennallen Field Manager Attachments Categorical Exc ||| 2022-08-12 [clear_decision|high|score=13.2] doc=ROD | ALYSIA HANCOCK Digitally signed by ALYSIA HANCOCK Date: 2022.08.12 15:27:23 -08'00' Marnie Graham Date Glennallen Field Manager DOI-BLM-AK-A020-2022-0 ||| 2022-08-12 [clear_decision|high|score=13.2] doc=ROD | ALYSIA HANCOCK Digitally signed by ALYSIA HANCOCK Date: 2022.08.12 15:28:42 -08'00' Marnie Graham Glennallen Field Manager Attachments Categorical Exc"
  },
  {
    "sample_id": "71",
    "project_title": "Grand Ditch Breach Restoration",
    "process_type": "EIS",
    "lead_agency": "National Park Service",
    "suggested_initiation_date": "2012-05-25",
    "suggested_initiation_evidence": "A 60-day public comment period was opened for the draft environmental impact statement that extended to May 25, 2012.",
    "top_initiation_candidates": "2012-05-25 [clear_initiation|medium|score=7.6] doc=FEIS | C-1 INTRODUCTION A notice of availability for the draft environmental impact statement for the Grand Ditch breach res- toration was published in the F ||| 2012-05-25 [clear_initiation|medium|score=7.6] doc=FEIS | A 60-day public comment period was opened for the draft environmental impact statement that extended to May 25, 2012. ||| 2010-06-01 [proxy_initiation|low|score=7.5] doc=FEIS | CONSULTATION AND COORDINATION 424 Two pieces of correspondence were received on the first newsletter (Spring 2010) and approximately 110 comments were ||| 2013-04-01 [proxy_initiation|low|score=0.2] doc=FEIS | Department of the Interior Rocky Mountain National Park Colorado GRAND DITCH BREACH RESTORATION GRAND DITCH BREACH RESTORATION FINAL Environmental Imp ||| 2013-04-01 [proxy_initiation|low|score=0.2] doc=FEIS | National Park Service U.S. Department of the Interior Rocky Mountain National Park Colorado GRAND DITCH BREACH RESTORATION GRAND DITCH BREACH RESTORAT",
    "suggested_decision_date": "2013-04-15",
    "suggested_decision_evidence": "NPS 121/112690 / April 2013",
    "top_decision_candidates": "2013-04-01 [proxy_decision|low|score=6.2] doc=FEIS | NPS 121/112690 / April 2013"
  },
  {
    "sample_id": "72",
    "project_title": "Green River Diversion Rehabilitation Project",
    "process_type": "EIS",
    "lead_agency": "Natural Resources Conservation Service",
    "suggested_initiation_date": "2013-05-29",
    "suggested_initiation_evidence": "The 2nd scoping period opened on May 29, 2013 and ended on July 2, 2013 for a total of 35 days.",
    "top_initiation_candidates": "2013-05-29 [clear_initiation|high|score=14.5] doc=None | The 2nd scoping period opened on May 29, 2013 and ended on July 2, 2013 for a total of 35 days. ||| 2013-07-02 [clear_initiation|high|score=14.4] doc=None | The 2nd scoping period opened on May 29, 2013 and ended on July 2, 2013 for a total of 35 days. ||| 2012-10-30 [clear_initiation|high|score=12.1] doc=FEIS | NRCS Green River Diversion Rehabilitation FEIS Page 5-4 June 2014 \uf0b7 Evaluate the effectiveness of public participation activities on a continual basis ||| 2012-11-30 [clear_initiation|high|score=12.0] doc=FEIS | NRCS Green River Diversion Rehabilitation FEIS Page 5-4 June 2014 \uf0b7 Evaluate the effectiveness of public participation activities on a continual basis ||| 2013-05-29 [clear_initiation|high|score=11.5] doc=FEIS | The 2nd scoping period opened on May 29, 2013 and ended on July 2, 2013 for a total of 35 days.",
    "suggested_decision_date": "2014-06-15",
    "suggested_decision_evidence": "NRCS Green River Diversion Rehabilitation FEIS Page 5-7 June 2014 5.8. Final EIS A public notice providing notice of availability of the FEIS will be distributed to interested parties (Chapter 7, Distribution) on June 20, 2014, published in local newspapers (The Sun Advocate, Moab Times-Independent,",
    "top_decision_candidates": "2014-06-01 [clear_decision|high|score=10.5] doc=FEIS | NRCS Green River Diversion Rehabilitation FEIS Page 5-7 June 2014 5.8. Final EIS A public notice providing notice of availability of the FEIS will be ||| 2014-06-20 [clear_decision|high|score=10.0] doc=FEIS | NRCS Green River Diversion Rehabilitation FEIS Page 5-7 June 2014 5.8. Final EIS A public notice providing notice of availability of the FEIS will be ||| 2014-07-19 [clear_decision|high|score=10.0] doc=FEIS | NRCS Green River Diversion Rehabilitation FEIS Page 5-7 June 2014 5.8. Final EIS A public notice providing notice of availability of the FEIS will be ||| 2014-06-07 [clear_decision|high|score=10.0] doc=FEIS | NRCS Green River Diversion Rehabilitation FEIS Page 5-7 June 2014 5.8. Final EIS A public notice providing notice of availability of the FEIS will be ||| 2014-06-17 [clear_decision|high|score=9.8] doc=FEIS | NRCS Green River Diversion Rehabilitation FEIS Page 5-7 June 2014 5.8. Final EIS A public notice providing notice of availability of the FEIS will be"
  },
  {
    "sample_id": "73",
    "project_title": "Shu1uuk Wind Project, Campo Indian Reservation, San Diego County, California",
    "process_type": "EIS",
    "lead_agency": "Bureau of Indian Affairs",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "74",
    "project_title": "Powder Horn Outfitters, LLC IOC",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2020-07-01",
    "suggested_initiation_evidence": "Department of the Interior Bureau of Land Management Powder Horn Outfitters, LLC IOC Categorical Exclusion (CX) BLM Wyoming \u2013 Lander Field Office January 2021 DOI-BLM-WY-R050-2021-0002-CX Lander Field Office 1335 Main Street Lander, Wyoming 82520 307-332-8400 307-332-2318 (FAX) Categorical Exclusion",
    "top_initiation_candidates": "2020-07-01 [clear_initiation|medium|score=6.5] doc=CE | Department of the Interior Bureau of Land Management Powder Horn Outfitters, LLC IOC Categorical Exclusion (CX) BLM Wyoming \u2013 Lander Field Office Janu ||| 2021-01-01 [clear_initiation|medium|score=1.0] doc=CE | Department of the Interior Bureau of Land Management Powder Horn Outfitters, LLC IOC Categorical Exclusion (CX) BLM Wyoming \u2013 Lander Field Office Janu ||| 2021-07-01 [clear_initiation|medium|score=1.0] doc=CE | Department of the Interior Bureau of Land Management Powder Horn Outfitters, LLC IOC Categorical Exclusion (CX) BLM Wyoming \u2013 Lander Field Office Janu",
    "suggested_decision_date": "2020-12-30",
    "suggested_decision_evidence": "Signature Authorizing Official: /s/Johanna Blanchard (acting FM) Date: December 30, 2020 John R.",
    "top_decision_candidates": "2020-12-30 [clear_decision|high|score=8.0] doc=CE | Signature Authorizing Official: /s/Johanna Blanchard (acting FM) Date: December 30, 2020 John R."
  },
  {
    "sample_id": "75",
    "project_title": "Perro Loco 27-22 Powerline/OHEL",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2023-07-01",
    "suggested_decision_evidence": "UNITED STATES DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT Pecos District Carlsbad Field Office 620 E Greene Street Carlsbad, NM 88220 CATEGORICAL EXCLUSION DOI-BLM-NM-P020-2023-0447-CX A.",
    "top_decision_candidates": "2023-07-01 [proxy_decision|low|score=5.0] doc=CE | UNITED STATES DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT Pecos District Carlsbad Field Office 620 E Greene Street Carlsbad, NM 88220 CATEGOR ||| 2023-07-01 [proxy_decision|low|score=5.0] doc=CE | [Image of a map with \"NWNE\" and \"SWNE\" labels, and a blue line indicating a path] Location of Proposed Action: New Mexico Principle Meridian, Lea Coun ||| 2023-07-01 [proxy_decision|low|score=5.0] doc=CE | DOI-BLM-NM-P020-2023-0311-CX E. ||| 2023-07-01 [proxy_decision|low|score=5.0] doc=CE | SIGNATURE Authorized Officer: __________________________ Date: _______________ ATTACHMENTS: 1 \u2013 Extraordinary Circumstances Review DOI-BLM-NM-P020-202 ||| 2023-07-01 [proxy_decision|low|score=5.0] doc=CE | DOI-BLM-NM-P020-2023-0311-CX"
  },
  {
    "sample_id": "76",
    "project_title": "Gemini Solar Project",
    "process_type": "EIS",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2018-07-13",
    "suggested_initiation_evidence": "GEMINI SOLAR PROJECT DRAFT EIS 4 Consultations, Coordination, and Public Involvements Chapter 4 Consultations, Coordination, and Public Involvement 4.1 Introduction This chapter summarizes the consultation and coordination activities conducted with interested agencies, organizations, tribes, and ind",
    "top_initiation_candidates": "2018-07-13 [clear_initiation|high|score=10.5] doc=DEIS | GEMINI SOLAR PROJECT DRAFT EIS 4 Consultations, Coordination, and Public Involvements Chapter 4 Consultations, Coordination, and Public Involvement 4. ||| 2018-08-27 [clear_initiation|high|score=10.5] doc=DEIS | GEMINI SOLAR PROJECT DRAFT EIS 4 Consultations, Coordination, and Public Involvements Chapter 4 Consultations, Coordination, and Public Involvement 4. ||| 2018-08-01 [clear_initiation|high|score=10.5] doc=DEIS | GEMINI SOLAR PROJECT DRAFT EIS 4 Consultations, Coordination, and Public Involvements Chapter 4 Consultations, Coordination, and Public Involvement 4. ||| 2018-08-02 [clear_initiation|high|score=10.5] doc=DEIS | GEMINI SOLAR PROJECT DRAFT EIS 4 Consultations, Coordination, and Public Involvements Chapter 4 Consultations, Coordination, and Public Involvement 4. ||| 2017-09-01 [clear_initiation|high|score=10.0] doc=DEIS | GEMINI SOLAR PROJECT DRAFT EIS 4 Consultations, Coordination, and Public Involvements Chapter 4 Consultations, Coordination, and Public Involvement 4.",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "77",
    "project_title": "Apalachicola-Chattahoochee-Flint Master Water Control Manual Update",
    "process_type": "EIS",
    "lead_agency": "Corps of Engineers--Civil Works",
    "suggested_initiation_date": "2000-05-16",
    "suggested_initiation_evidence": "On May 16, 2000, the Governor of Georgia submitted a formal request to the Assistant Secretary of the Army (Civil Works) to adjust the operation of Lake Lanier, and to enter into agreements with the State, or water supply providers, to accommodate increases in water supply withdrawals from Lake Lani",
    "top_initiation_candidates": "2000-05-16 [clear_initiation|medium|score=11.5] doc=None | On May 16, 2000, the Governor of Georgia submitted a formal request to the Assistant Secretary of the Army (Civil Works) to adjust the operation of La ||| 1988-03-01 [proxy_initiation|low|score=9.5] doc=None | When submitted to the South Atlantic Division in March 1988, the draft GDM confirmed that the plan recommending the reregulation dam still had the gre ||| 1988-10-01 [proxy_initiation|low|score=9.5] doc=None | Due to the closeness of the net benefits of the two plans, a more detailed evaluation of the hydropower impacts was conducted, and in October 1988, th ||| 1988-11-01 [proxy_initiation|low|score=9.5] doc=None | This result was coordinated with the Governor of Georgia who, in November 1988, concurred with the recommendation. ||| 1989-03-01 [proxy_initiation|low|score=9.5] doc=None | The appropriate procedure to pursue implementation of the reallocation plan was determined to be preparation of a Post-Authorization Change (PAC) repo",
    "suggested_decision_date": "2006-03-15",
    "suggested_decision_evidence": "Court-ordered mediation between the parties was initiated in March 2006 for both the ACT and ACF litigation.",
    "top_decision_candidates": "2006-03-01 [proxy_decision|low|score=5.5] doc=FEIS | Court-ordered mediation between the parties was initiated in March 2006 for both the ACT and ACF litigation. ||| 2007-03-01 [proxy_decision|low|score=5.5] doc=FEIS | The mediation expired in March 2007 (ACF Basin) and September 2007 (ACT Basin). ||| 2007-09-01 [proxy_decision|low|score=5.5] doc=FEIS | The mediation expired in March 2007 (ACF Basin) and September 2007 (ACT Basin). ||| 2008-10-01 [proxy_decision|low|score=5.5] doc=FEIS | The scoping meetings were held in October 2008 at five locations throughout the ACF Basin. ||| 2009-01-01 [proxy_decision|low|score=5.5] doc=FEIS | The results of this scoping were published by USACE in a final scoping report in January 2009."
  },
  {
    "sample_id": "78",
    "project_title": "Modelo 3 Fed Com 526H Gas Pipelines",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2022-04-19",
    "suggested_decision_evidence": "DATE REVISION MODELO 3 IRON 04/19/2022 FILE:SK MODELO_3_IRON DRAWN BY: BSZ SHEET 1 OF 1 SITE PEGASUS 3FED COM EOL 452419.97 3 SITE 10 U.S.A.: 938.96 FEET/56.91 RODS TOTAL LINE: 938,96 FEET/56,91 RODS SURVEY SECTION LINE PROPOSED 1 LAYPLAT TEMP WATERLINE FENCE ROAD WAY 749356.85 45347988 TE SITE MODE",
    "top_decision_candidates": "2022-07-01 [proxy_decision|low|score=3.0] doc=CE | DOI-BLM-NM-P020-2022-0753-CX Project Name: Modelo 3 Fed Com 526H Gas Pipelines Preparer: Dellen Chaz Sartin Original APD/EA NEPA No. ||| 2020-07-01 [proxy_decision|low|score=3.0] doc=CE | DOI-BLM-NM-P020-2020-0377-EA A. ||| 2022-04-19 [clear_decision|low|score=3.0] doc=CE | DATE REVISION MODELO 3 IRON 04/19/2022 FILE:SK MODELO_3_IRON DRAWN BY: BSZ SHEET 1 OF 1 SITE PEGASUS 3FED COM EOL 452419.97 3 SITE 10 U.S.A.: 938.96 F"
  },
  {
    "sample_id": "79",
    "project_title": "2021 Rainy River Watershed Withdrawal Application and Environmental Assessment",
    "process_type": "EA",
    "lead_agency": "Forest Service",
    "suggested_initiation_date": "2018-09-06",
    "suggested_initiation_evidence": "On September 6, 2018, the Forest Service submitted a letter to BLM cancelling the withdrawal application and associated environmental assessment.",
    "top_initiation_candidates": "2018-09-06 [clear_initiation|medium|score=9.0] doc=EA | On September 6, 2018, the Forest Service submitted a letter to BLM cancelling the withdrawal application and associated environmental assessment. ||| 2018-01-26 [clear_initiation|medium|score=8.5] doc=EA | A press release was issued on January 26, 2018, announcing the change to an environmental assessment and an additional 30-day comment period. ||| 2018-09-06 [clear_initiation|medium|score=7.5] doc=EA | On September 6, 2018, the Forest Service submitted a letter to the BLM cancelling the withdrawal application. ||| 2018-09-06 [clear_initiation|medium|score=7.5] doc=EA | Rainy River Withdrawal Environmental Assessment Superior National Forest 3 from disposal under mineral and geothermal leasing laws, initiating a 90-da ||| 2017-01-13 [clear_initiation|medium|score=7.2] doc=EA | Rainy River Withdrawal Environmental Assessment Superior National Forest 3 from disposal under mineral and geothermal leasing laws, initiating a 90-da",
    "suggested_decision_date": "2022-06-24",
    "suggested_decision_evidence": "#11577) Eastern States Office ________________________________ (Signature) ________________________________ (Date) ERIC WIRZ Digitally signed by ERIC WIRZ Date: 2022.06.24 12:17:00 -05'00' KEVIN JOHNSON Digitally signed by KEVIN JOHNSON Date: 2022.06.24 10:26:15 -07'00' CONSTANCE CUMMINS Digitally s",
    "top_decision_candidates": "2022-06-24 [clear_decision|high|score=7.5] doc=OTHER | #11577) Eastern States Office ________________________________ (Signature) ________________________________ (Date) ERIC WIRZ Digitally signed by ERIC ||| 2022-06-27 [clear_decision|high|score=7.5] doc=OTHER | #11577) Eastern States Office ________________________________ (Signature) ________________________________ (Date) ERIC WIRZ Digitally signed by ERIC"
  },
  {
    "sample_id": "80",
    "project_title": "Dare to Dream Special Recreation Permit (SRP) Spring 2023",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2023-03-29",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2023-03-29",
    "top_initiation_candidates": "2023-03-29 [clear_initiation|high|score=10.0] doc=None | BLM NEPA Register project start date: 2023-03-29 ||| 2023-04-01 [proxy_initiation|low|score=4.0] doc=CE | Department of the Interior Bureau of Land Management DOI-BLM-CO-S054-2023-0002 CX Attachment 1: Categorical Exclusion Review Dare to Dream Special Rec",
    "suggested_decision_date": "2023-04-10",
    "suggested_decision_evidence": "SIGNATURE OF AUTHORIZED OFFICER Digitally signed by SHERMAN SHERMAN FRANZ FRANZ Date: 2023.04.10 14:53:46 -06'00' S.",
    "top_decision_candidates": "2023-04-10 [clear_decision|high|score=8.2] doc=CE | SIGNATURE OF AUTHORIZED OFFICER Digitally signed by SHERMAN SHERMAN FRANZ FRANZ Date: 2023.04.10 14:53:46 -06'00' S. ||| 2023-04-10 [clear_decision|high|score=8.2] doc=CE | NAME OF PREPARER Tatyana Sukharnikova Outdoor Recreation Planner SIGNATURE OF AUTHORIZED OFFICER Digitally signed by SHERMAN SHERMAN FRANZ FRANZ Date: ||| 2023-07-01 [proxy_decision|low|score=5.0] doc=CE | DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT DEPARTMENT OF THE INTERIOR MARCH 3, 1849 United States Department of the Interior BUREAU OF LAND ||| 2023-07-01 [proxy_decision|low|score=5.0] doc=CE | Department of the Interior Bureau of Land Management DOI-BLM-CO-S054-2023-0002 CX Attachment 1: Categorical Exclusion Review Dare to Dream Special Rec ||| 2023-07-01 [proxy_decision|low|score=5.0] doc=CE | 1 DOI-BLM-CO-S054-2023-0002 CX Attachment 1: Categorical Exclusion Review Permit Conditions and Stipulations Over the next 15 years, beginning in the"
  },
  {
    "sample_id": "81",
    "project_title": "Triumvirate LLC Commercial Heli-skiing Special Recreation Permit",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2014-02-20",
    "suggested_decision_evidence": "/s/ Alan Bittner 02/20/2014 Alan Bittner Date Anchorage Field Manager Attachments 1.",
    "top_decision_candidates": "2014-02-20 [clear_decision|high|score=13.2] doc=FONSI | /s/ Alan Bittner 02/20/2014 Alan Bittner Date Anchorage Field Manager Attachments 1. ||| 2014-02-20 [clear_decision|high|score=10.2] doc=None | BLM NEPA Register decision date (fonsi): 2014-02-20 ||| 2013-07-01 [proxy_decision|low|score=10.0] doc=FONSI | Triumvirate LLC Commercial Heli-skiing Special Recreation Permit Environmental Assessment (DOI-BLM-AK-A010-2013-0008-EA), February 3, 2014 (Public Rel ||| 2013-07-01 [proxy_decision|low|score=10.0] doc=FONSI | Finding of No Significant Impact (FONSI), Triumvirate LLC Commercial Heli-skiing Special Recreation Permit Environmental Assessment (DOI-BLM-AK-A010-2 ||| 2013-07-01 [proxy_decision|low|score=10.0] doc=FONSI | Finding of No Significant Impact (FONSI), Triumvirate LLC Commercial Heli\u00ad skiing Special Recreation Permit Environmental Assessment (DOI-BLM-AK-A010-"
  },
  {
    "sample_id": "82",
    "project_title": "Alaska Outer Continental Shelf Cook Inlet Planning Area Oil and Gas Lease Sale 244 In the Cook Inlet, Alaska",
    "process_type": "EIS",
    "lead_agency": "Bureau of Ocean Energy Management",
    "suggested_initiation_date": "2014-12-08",
    "suggested_initiation_evidence": "The public comment period closed on December 8, 2014.",
    "top_initiation_candidates": "2014-12-08 [clear_initiation|medium|score=9.0] doc=OTHER | The public comment period closed on December 8, 2014. ||| 2014-12-08 [clear_initiation|medium|score=8.5] doc=FEIS | The public comment period closed on December 8, 2014. ||| 2012-08-01 [proxy_initiation|low|score=7.0] doc=OTHER | In August 2012, the Secretary of the Interior issued the Final OCS Oil and Gas Leasing Program for 2012-2017. ||| 2012-08-01 [proxy_initiation|low|score=6.5] doc=FEIS | In August 2012, the Secretary of the Interior issued the Final OCS Oil and Gas Leasing Program for 2012-2017. ||| 2016-11-01 [proxy_initiation|low|score=5.6] doc=FEIS | OCS Oil and Natural Gas: Potential Lifecycle Greenhouse Gas Emissions and Social Cost of Carbon November 2016 42 Table A-4.",
    "suggested_decision_date": "2016-12-15",
    "suggested_decision_evidence": "Chapters 6-7, Appendices A-G December 2016",
    "top_decision_candidates": "2016-12-01 [proxy_decision|low|score=5.0] doc=FEIS | Chapters 6-7, Appendices A-G December 2016 ||| 2016-12-01 [proxy_decision|low|score=4.5] doc=FEIS | Department of the Interior Bureau of Ocean Energy Management Alaska OCS Region December 2016"
  },
  {
    "sample_id": "83",
    "project_title": "Long-Term Operation of the Central Valley Project and State Water Project",
    "process_type": "EIS",
    "lead_agency": "Bureau of Reclamation",
    "suggested_initiation_date": "2024-11-15",
    "suggested_initiation_evidence": "Department of the Interior November 2024 Final Environmental Impact Statement Long-Term Operation of the Central Valley Project and State Water Project Central Valley Project, California Interior Region 10 - California Great-Basin",
    "top_initiation_candidates": "2024-11-01 [proxy_initiation|low|score=4.8] doc=FEIS | Department of the Interior November 2024 Final Environmental Impact Statement Long-Term Operation of the Central Valley Project and State Water Projec ||| 2024-11-01 [proxy_initiation|low|score=4.8] doc=FEIS | U.S. Department of the Interior November 2024 Final Environmental Impact Statement Long-Term Operation of the Central Valley Project and State Water P",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "84",
    "project_title": "Haiti Renewable Resource Study",
    "process_type": "CE",
    "lead_agency": "Department of Energy; Department of State; International Assistance Programs",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2012-07-23",
    "suggested_decision_evidence": "DOE CX Register determination date (cx-8556.0): 2012-07-23",
    "top_decision_candidates": "2012-07-23 [clear_decision|high|score=15.0] doc=categorical exclusion determin | DOE CX Register determination date (cx-8556.0): 2012-07-23"
  },
  {
    "sample_id": "85",
    "project_title": "St. Jude Mine Area Reclamation",
    "process_type": "EA",
    "lead_agency": "",
    "suggested_initiation_date": "2009-01-15",
    "suggested_initiation_evidence": "The Topaz POO and associated Environmental Assessment (EA) were approved with a Finding of No Significant Impact (FONSI) in January 2009.",
    "top_initiation_candidates": "2009-01-01 [proxy_initiation|low|score=3.0] doc=OTHER | The Topaz POO and associated Environmental Assessment (EA) were approved with a Finding of No Significant Impact (FONSI) in January 2009.",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "86",
    "project_title": "Ethen Roberts Film Permit",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2016-07-25",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2016-07-25",
    "top_initiation_candidates": "2016-07-25 [clear_initiation|high|score=10.0] doc=None | BLM NEPA Register project start date: 2016-07-25 ||| 2016-07-01 [proxy_initiation|low|score=4.1] doc=CE | United States Department of the Interior Bureau of Land Management Categorical Exclusion Not Established By Statute DOI-BLM-UT-CO30-2016-0041-CX July",
    "suggested_decision_date": "2016-07-26",
    "suggested_decision_evidence": "Corry | 7/26/16 Areas of Critical Environmental Concern | No | J.",
    "top_decision_candidates": "2016-07-26 [clear_decision|low|score=5.0] doc=CE | Corry | 7/26/16 Areas of Critical Environmental Concern | No | J. ||| 2016-07-26 [clear_decision|low|score=5.0] doc=CE | Hunsaker | 7/26/16 Environmental Justice | No | J. ||| 2016-07-26 [clear_decision|low|score=5.0] doc=CE | Corry | 7/26/16 Floodplains | No | D. ||| 2016-07-26 [clear_decision|low|score=5.0] doc=CE | Corry | 7/26/16 Invasive Species/Noxious Weeds | No | R. ||| 2016-07-26 [clear_decision|low|score=5.0] doc=CE | Reese | 7/26/16 Migratory Birds | No | R."
  },
  {
    "sample_id": "87",
    "project_title": "Prospect and Janus Solar + Storage Projects",
    "process_type": "EA",
    "lead_agency": "",
    "suggested_initiation_date": "2023-12-04",
    "suggested_initiation_evidence": "Dear Mark Suchy: This email is to inform you that we received your email or le\u01a9er request on 4 December 2023.",
    "top_initiation_candidates": "2023-12-04 [clear_initiation|medium|score=5.0] doc=DEA | Dear Mark Suchy: This email is to inform you that we received your email or le\u01a9er request on 4 December 2023. ||| 2023-12-01 [clear_initiation|medium|score=5.0] doc=DEA | Dear Mark Suchy: This email is to inform you that we received your email or le\u01a9er request on 4 December 2023.",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "88",
    "project_title": "Behm 3D \u2013 Additional Lands Modification to Geophysical Exploration MTM111417",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2023-10-27",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2023-10-27",
    "top_initiation_candidates": "2023-10-27 [clear_initiation|high|score=10.4] doc=None | BLM NEPA Register project start date: 2023-10-27",
    "suggested_decision_date": "2024-01-01",
    "suggested_decision_evidence": "Document filename date (ce): 2024-01-01",
    "top_decision_candidates": "2024-01-01 [clear_decision|medium|score=8.0] doc=CE | Document filename date (ce): 2024-01-01 ||| 2023-10-31 [clear_decision|medium|score=6.0] doc=CE | Authorizing Official: Ben Hileman, Field Manager Date: 10/31/2023 CONTACT: For additional information concerning this CX review, contact Kirsten Boyle ||| 2023-10-27 [clear_decision|low|score=4.2] doc=CE | 32: NE1/4, NW1/4 Applicant/Address: Operator: Behm Energy, Inc., P O Box 1060, Minot, ND, 58702 Geophysical Co: Echo Seismic USA, 4833 Front Street, U ||| 2024-07-01 [proxy_decision|low|score=4.0] doc=CE | Department of the Interior Bureau of Land Management North Central Montana District CATEGORICAL EXCLUSION BLM Office: Havre Field Office NEPA Project ||| 2023-12-31 [clear_decision|low|score=4.0] doc=CE | MTM111417 was approved in 2020 and will expire on December 31, 2023."
  },
  {
    "sample_id": "89",
    "project_title": "Patos Lightstation Amateur Radio Educational Events",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2017-04-10",
    "suggested_decision_evidence": "Signature /s/ Marcia deChadenedes Marcia deChadenedes Monument Manager 4/10/17 Date E.",
    "top_decision_candidates": "2017-04-10 [clear_decision|high|score=8.2] doc=CE | Signature /s/ Marcia deChadenedes Marcia deChadenedes Monument Manager 4/10/17 Date E. ||| 2017-04-10 [clear_decision|low|score=4.2] doc=CE | Reviewers | Resource | Initials | Date ---|---|---|--- Erik Ellis | Wildlife, Special Status Wildlife | EDE | 3/14/2017 Molly Boyter | Botany, Special ||| 2017-07-01 [proxy_decision|low|score=4.0] doc=CE | Background BLM Office: San Juan Islands National Monument NEPA Log Number: DOI-BLM-ORWA-W040-2017-0003-CX Proposed Action Title: Patos Lightstation Am ||| 2016-09-01 [clear_decision|low|score=4.0] doc=CE | The gallery was approved for safety of human activity in September 2016. ||| 2017-03-14 [clear_decision|low|score=4.0] doc=CE | Reviewers | Resource | Initials | Date ---|---|---|--- Erik Ellis | Wildlife, Special Status Wildlife | EDE | 3/14/2017 Molly Boyter | Botany, Special"
  },
  {
    "sample_id": "90",
    "project_title": "Interstate 290 Eisenhower Expressway",
    "process_type": "EIS",
    "lead_agency": "Department of Transportation",
    "suggested_initiation_date": "2016-12-30",
    "suggested_initiation_evidence": "I-290 Eisenhower Expressway S-6 Final Environmental Impact Statement communication is described in more detail in Section 4.0 of the Final Environmental Impact Statement (FEIS) and the Stakeholder Involvement Plan.2 Six NEPA/404 Merger team coordination meetings were held, and more than 150 meetings",
    "top_initiation_candidates": "2016-12-30 [clear_initiation|medium|score=6.2] doc=FEIS | I-290 Eisenhower Expressway S-6 Final Environmental Impact Statement communication is described in more detail in Section 4.0 of the Final Environment ||| 2009-10-01 [clear_initiation|medium|score=6.2] doc=FEIS | I-290 Eisenhower Expressway S-6 Final Environmental Impact Statement communication is described in more detail in Section 4.0 of the Final Environment ||| 2017-04-01 [clear_initiation|medium|score=6.2] doc=FEIS | I-290 Eisenhower Expressway S-6 Final Environmental Impact Statement communication is described in more detail in Section 4.0 of the Final Environment ||| 2009-10-01 [proxy_initiation|low|score=4.2] doc=FEIS | As a result of this collective outreach and community involvement, more than 1,700 public comments were received and considered from October 2009 to A ||| 2017-04-01 [proxy_initiation|low|score=4.2] doc=FEIS | As a result of this collective outreach and community involvement, more than 1,700 public comments were received and considered from October 2009 to A",
    "suggested_decision_date": "",
    "suggested_decision_evidence": "",
    "top_decision_candidates": ""
  },
  {
    "sample_id": "91",
    "project_title": "Downeast Liquified Natural Gas, Washington County, Maine",
    "process_type": "EIS",
    "lead_agency": "Department of Energy",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2008-12-29",
    "suggested_decision_evidence": "The facility may require various MEPDES permits depending on the final design and operational scheme for the facility. Such permits may include the following: a. A Construction General Permit will be required during all proposed construction activities (Section 2.3) as Downeast proposes to clear and",
    "top_decision_candidates": "2008-12-29 [clear_decision|medium|score=5.0] doc=OTHER | The facility may require various MEPDES permits depending on the final design and operational scheme for the facility. Such permits may include the fo"
  },
  {
    "sample_id": "92",
    "project_title": "NOAA Climate Monitoring Station Grant Issuance",
    "process_type": "CE",
    "lead_agency": "Department of Commerce",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "",
    "suggested_decision_date": "2014-06-23",
    "suggested_decision_evidence": "/s/ Karen Simms, Acting Tucson Field Office Manager 06/23/2014 APPROVING OFFICIAL: DATE: TITLE: Note: The signed conclusion on this compliance record is part of an interim step in the BLM's internal decision process and does not constitute an appealable decision.",
    "top_decision_candidates": "2014-06-23 [clear_decision|high|score=8.0] doc=CE | /s/ Karen Simms, Acting Tucson Field Office Manager 06/23/2014 APPROVING OFFICIAL: DATE: TITLE: Note: The signed conclusion on this compliance record ||| 2014-07-01 [clear_decision|high|score=8.0] doc=CE | DECISION MEMORANDUM NOAA Climate Monitoring Station Grant Issuance DOI-BLM-AZ-G020-2014-0016-CX U.S. ||| 2014-06-24 [clear_decision|high|score=8.0] doc=CE | /s/ Karen Simms, Acting 6/24/2014 Viola Hillman, Tucson Field Office Manager Date Attachment: Stipulations 2 of 4 STIPULATIONS AZA-036518 Grant for NO ||| 2013-08-01 [clear_decision|low|score=5.0] doc=CE | Attachment 4-1 AZ-1790-1 August 2013 Part II. ||| 2013-08-01 [clear_decision|low|score=5.0] doc=CE | Attachment 4-2 AZ-1790-1 August 2013 Part IV."
  },
  {
    "sample_id": "93",
    "project_title": "Revision of 9B Regulations Governing Non-Federal Oil and Gas Activities",
    "process_type": "EIS",
    "lead_agency": "National Park Service",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2015-12-28 [clear_initiation|medium|score=1.5] doc=FEIS | Following the release of the draft plan/EIS and Proposed Rule, a 60-day public comment period was open that ended on December 28, 2015. ||| 2013-01-01 [proxy_initiation|low|score=-0.2] doc=FEIS | INTERNAL SCOPING Internal scoping for the proposed rule revisions/EIS began in January 2013 with the establishment of an interdisciplinary team compri ||| 2009-07-01 [proxy_initiation|low|score=-1.0] doc=FEIS | The scoping process began in July 2009 with the establishment of an interdisciplinary team comprised of NPS subject matter experts, practitioners, and ||| 2013-01-01 [proxy_initiation|low|score=-1.8] doc=OTHER | Internal scoping for the EIS began in January 2013 with the establishment of an interdisciplinary team comprising Service subject matter experts, prac",
    "suggested_decision_date": "1978-01-08",
    "suggested_decision_evidence": "Appendix B: 9B Regulations Revision of 9B Regulations Governing Non-Federal Oil and Gas Activities / EIS B-15 \u00a7 9.50 Use of roads by commercial vehicles. (a) After January 8, 1978, no commercial vehicle shall use roads administered by the National Park Service without being registered with the Super",
    "top_decision_candidates": "1978-01-08 [clear_decision|medium|score=7.8] doc=FEIS | Appendix B: 9B Regulations Revision of 9B Regulations Governing Non-Federal Oil and Gas Activities / EIS B-15 \u00a7 9.50 Use of roads by commercial vehicl"
  },
  {
    "sample_id": "94",
    "project_title": "Inyo-Barren Ridge Transmission Line Clearance Grading",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2016-06-15",
    "suggested_initiation_evidence": "[ ] Yes [x] No Comments: A project specific cultural resource survey was completed by DUDEK Corp and documented in the Archaeological Survey Report for the Barren Ridge Transmission Line Clearance Project (June 2016).",
    "top_initiation_candidates": "2017-08-01 [clear_initiation|high|score=5.2] doc=None | BLM NEPA Register project start date: 2017-08-01 ||| 2016-06-01 [proxy_initiation|low|score=4.6] doc=CE | [ ] Yes [x] No Comments: A project specific cultural resource survey was completed by DUDEK Corp and documented in the Archaeological Survey Report fo",
    "suggested_decision_date": "2017-01-01",
    "suggested_decision_evidence": "Document filename date (ce): 2017-01-01",
    "top_decision_candidates": "2017-01-01 [clear_decision|medium|score=8.0] doc=CE | Document filename date (ce): 2017-01-01 ||| 2017-07-01 [proxy_decision|low|score=4.2] doc=CE | **United States Department of the Interior** **BUREAU OF LAND MANAGEMENT** Bishop Field Office 351 Pacu Lane, Suite 100 Bishop, California 93514 www.b ||| 2017-07-01 [proxy_decision|low|score=4.2] doc=CE | Background** **BLM Office:** Bishop Field Office, 351 Pacu Lane, Suite 100, Bishop, CA 93514 **Lease/Serial/Case File Number:** CALA 0 088876 (ROW), C ||| 2017-08-01 [clear_decision|low|score=4.2] doc=CE | The cultural resources effects determination for the proposed action was posted on the BLM NEPA Project Register on August 1, 2017. ||| 2018-12-31 [clear_decision|low|score=4.0] doc=CE | It is expected that construction work would take about three weeks, beginning after the ROW/TUP is authorized and ending before December 31, 2018."
  },
  {
    "sample_id": "95",
    "project_title": "Reasonably Foreseeable Development Scenario",
    "process_type": "EA",
    "lead_agency": "United States Geological Survey",
    "suggested_initiation_date": "2021-11-15",
    "suggested_initiation_evidence": "4.2.1 Section 208 Report In November 2021, the Department of the Interior released a Report on the Federal Oil and Gas Leasing Program (Report).",
    "top_initiation_candidates": "2021-11-01 [proxy_initiation|low|score=5.7] doc=EA | 4.2.1 Section 208 Report In November 2021, the Department of the Interior released a Report on the Federal Oil and Gas Leasing Program (Report). ||| 2023-12-01 [proxy_initiation|low|score=5.5] doc=FONSI | December 2023 Oil and Gas Lease Parcel Sale DOI-BLM-MT-0000-2023-0003-EA December 2023 U.S. Department of the Interior Bureau of Land Management Monta ||| 2023-12-01 [proxy_initiation|low|score=5.0] doc=FONSI | 4 The BLM prepared an EA to disclose and analyze the potential environmental consequences from offering the 14 parcels1 in a competitive oil and gas l ||| 2023-12-01 [proxy_initiation|low|score=5.0] doc=FONSI | As per 43 CFR \u00a73120.1-1(d), the BLM has identified that the lands associated with the parcel are subject to drainage; therefore, the BLM has self-nomi ||| 2023-12-01 [proxy_initiation|low|score=5.0] doc=FONSI | 4 ND-2023-12-6869 North Dakota Field Office, COE, ND Pending Litigation The BLM prepared an EA to disclose and analyze the potential environmental con",
    "suggested_decision_date": "2023-01-01",
    "suggested_decision_evidence": "Document filename date (fonsi): 2023-01-01",
    "top_decision_candidates": "2023-01-01 [clear_decision|medium|score=13.0] doc=FONSI | Document filename date (fonsi): 2023-01-01 ||| 2023-12-01 [clear_decision|low|score=10.5] doc=FONSI | December 2023 Oil and Gas Lease Parcel Sale DOI-BLM-MT-0000-2023-0003-EA December 2023 U.S. ||| 2023-12-07 [clear_decision|low|score=10.0] doc=FONSI | Following additional review, BLM has also deleted one parcel (ND-2023-12-0715) because BLM has determined it does not have jurisdiction over the miner ||| 2023-12-07 [clear_decision|low|score=10.0] doc=FONSI | Furthermore, the BLM added parcel ND-2023-12-0721 to the December 2023 lease sale. ||| 2023-12-01 [clear_decision|low|score=10.0] doc=FONSI | Furthermore, the BLM added parcel ND-2023-12-0721 to the December 2023 lease sale."
  },
  {
    "sample_id": "96",
    "project_title": "Sierra Scenic Byway Roadside Hazard Abatement Project",
    "process_type": "EA",
    "lead_agency": "Forest Service",
    "suggested_initiation_date": "2021-10-28",
    "suggested_initiation_evidence": "Public Involvement Project scoping was initiated on October 28, 2021.",
    "top_initiation_candidates": "2021-10-28 [clear_initiation|high|score=13.8] doc=FONSI | Public Involvement Project scoping was initiated on October 28, 2021. ||| 2022-01-26 [clear_initiation|medium|score=11.5] doc=FONSI | The public comment period began on January 26, 2022, with a public notice published in the Fresno Bee, the newspaper of r... ||| 2022-08-01 [proxy_initiation|low|score=9.0] doc=FONSI | Final Decision Notice and Finding of No Significant Impact \u2013 Sierra Scenic Byway Roadside Hazard Abatement Project 8 conservation measures to ensure t ||| 2022-08-05 [clear_initiation|medium|score=6.2] doc=FONSI | Final Decision Notice and Finding of No Significant Impact \u2013 Sierra Scenic Byway Roadside Hazard Abatement Project 8 conservation measures to ensure t",
    "suggested_decision_date": "2022-08-05",
    "suggested_decision_evidence": "08/05/2022 Dean A.",
    "top_decision_candidates": "2022-08-05 [clear_decision|low|score=9.2] doc=FONSI | 08/05/2022 Dean A. ||| 2021-02-23 [clear_decision|low|score=9.0] doc=FONSI | may affect, but is not likely to adversely affect the Sierra Nevada yellow-legged frog (SNYLF) The US Fish and Wildlife Service (USFWS) issued a Biolo"
  },
  {
    "sample_id": "97",
    "project_title": "Paul 27X Well Abandonment",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2017-10-12",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2017-10-12",
    "top_initiation_candidates": "2017-10-12 [clear_initiation|high|score=10.1] doc=None | BLM NEPA Register project start date: 2017-10-12",
    "suggested_decision_date": "2017-11-13",
    "suggested_decision_evidence": "BLM NEPA Register decision date (fonsi): 2017-11-13",
    "top_decision_candidates": "2017-11-13 [clear_decision|high|score=10.0] doc=None | BLM NEPA Register decision date (fonsi): 2017-11-13"
  },
  {
    "sample_id": "98",
    "project_title": "Maximus Operating Limited's R&R Fed 34-17 Application for Permit to Drill",
    "process_type": "CE",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "",
    "suggested_initiation_evidence": "",
    "top_initiation_candidates": "2019-07-01 [clear_initiation|medium|score=6.5] doc=ROD | Department of the Interior Bureau of Land Management Decision Record: Maximus Operating Limited R&R Fed 34-17 Application for Permit to Drill DOI-BLM- ||| 2019-07-01 [clear_initiation|medium|score=6.5] doc=ROD | Department of the Interior Bureau of Land Management 390 CX#3: Maximus Operating Limited R&R Fed 34-17 Application for Permit to Drill DOI-BLM-WY-P070",
    "suggested_decision_date": "2019-06-06",
    "suggested_decision_evidence": "Public Involvement A project summary was posted to BLM's national ePlanning website on June 6, 2019.",
    "top_decision_candidates": "2018-07-01 [proxy_decision|medium|score=11.0] doc=ROD | The tiered Laurel Fed 23-7 NEPA analysis #DOI-BLM-WY-P070-2018-0049-EA was finalized or supplemented within five years of spudding (drilling) the prop ||| 2019-07-01 [proxy_decision|low|score=9.5] doc=ROD | Project Summary The CX3, DOI-BLM-WY-P070-2019-0035-CX, includes the project description, including site-specific mitigation measures which are incorpo ||| 2019-06-06 [clear_decision|low|score=9.0] doc=ROD | Public Involvement A project summary was posted to BLM's national ePlanning website on June 6, 2019. ||| 2018-03-26 [clear_decision|low|score=9.0] doc=ROD | The decision complies with the March 26, 2018, U.S. ||| 2019-10-11 [clear_decision|low|score=9.0] doc=ROD | Yeager Field Manager 10/11/2019 Date"
  },
  {
    "sample_id": "99",
    "project_title": "Mariana Islands Training and Testing",
    "process_type": "EIS",
    "lead_agency": "Navy, Marine Corps",
    "suggested_initiation_date": "2010-02-22",
    "suggested_initiation_evidence": "We appreciate the additional information and herein provide our response to your requests: \u2022 We acknowledge your request is now to reinitate consultation on MIRC, as the MITT action is a continuation of the MIRC action (biological opinion 2009-F-0345; dated February 22, 2010).",
    "top_initiation_candidates": "2010-02-22 [clear_initiation|medium|score=8.3] doc=FEIS | We appreciate the additional information and herein provide our response to your requests: \u2022 We acknowledge your request is now to reinitate consultat ||| 2010-02-22 [clear_initiation|medium|score=8.3] doc=FEIS | MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 APPENDIX C AGENCY CORRESPONDENCE C-62 Mr. L.M. Foster Service File No. 2014-F-0262 In a s ||| 2008-09-01 [proxy_initiation|low|score=7.5] doc=FEIS | MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 4.3.3 OTHER MILITARY ACTIONS 4.3.3.1 Army and Air Force Exchange Service on Guam In Septe ||| 2011-09-16 [clear_initiation|high|score=6.0] doc=None | FR NOI publication date: 2011-09-16 ||| 2011-09-16 [clear_initiation|high|score=5.5] doc=OTHER | E.2 GENERAL SUMMARY OF THE SCOPING PERIOD The public scoping period began with the issuance of the Notice of Intent in the Federal Register on 16 Sept",
    "suggested_decision_date": "2010-09-15",
    "suggested_decision_evidence": "MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 4.3.3 OTHER MILITARY ACTIONS 4.3.3.1 Army and Air Force Exchange Service on Guam In September 2008, the Army and Air Force Exchange Service opened a 181,000-square-foot (ft.2) (16,815.4-square-meter [m2]) Shopping Complex on Andersen Air F",
    "top_decision_candidates": "2010-09-01 [clear_decision|high|score=9.8] doc=FEIS | MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 4.3.3 OTHER MILITARY ACTIONS 4.3.3.1 Army and Air Force Exchange Service on Guam In Septe ||| 2012-02-01 [clear_decision|high|score=9.8] doc=FEIS | MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 4.3.3 OTHER MILITARY ACTIONS 4.3.3.1 Army and Air Force Exchange Service on Guam In Septe ||| 2015-05-01 [proxy_decision|low|score=6.5] doc=FEIS | MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 APPENDIX C AGENCY CORRESPONDENCE C-62 Mr. ||| 2015-05-01 [proxy_decision|low|score=6.5] doc=FEIS | MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 APPENDIX C AGENCY CORRESPONDENCE C-109 Frank M. ||| 2015-05-01 [proxy_decision|low|score=6.5] doc=FEIS | MARIANA ISLANDS TRAINING AND TESTING FINAL EIS/OEIS MAY 2015 APPENDIX C AGENCY CORRESPONDENCE C-111 Frank M."
  },
  {
    "sample_id": "100",
    "project_title": "Dillon Road at Interstate 10 Multi-Tenant Wireless Broadband Communications Site",
    "process_type": "EA",
    "lead_agency": "Bureau of Land Management",
    "suggested_initiation_date": "2019-10-18",
    "suggested_initiation_evidence": "BLM NEPA Register project start date: 2019-10-18",
    "top_initiation_candidates": "2019-10-18 [clear_initiation|high|score=10.8] doc=None | BLM NEPA Register project start date: 2019-10-18",
    "suggested_decision_date": "2020-08-07",
    "suggested_decision_evidence": "BLM NEPA Register decision date (fonsi): 2020-08-07",
    "top_decision_candidates": "2020-08-07 [clear_decision|high|score=10.0] doc=None | BLM NEPA Register decision date (fonsi): 2020-08-07"
  }
]
```
