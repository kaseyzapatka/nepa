import pandas as pd


LABELS = [
    {
        "candidate_id": "02045845ec5dbca2c4e2",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote 'Corpus Christi, Texas (Liquefaction Project), and the Commission\u2019s November 22, 2019 Order (2019 Order), which authorized the addition'."
    },
    {
        "candidate_id": "9706763116a7b61e9dc5",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'the Point No Point Treaty Council letter on October 22, 2012. On August 6, 2012, Commission staff held'."
    },
    {
        "candidate_id": "98c127d828da859654fa",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'March 2, 2022, and the Georgia SHPO on March 21, 2022. Commission staff addressed the administrative changes and'."
    },
    {
        "candidate_id": "2c1d59f5016eec9e1bea",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'of analysis. A press release was issued on January 26, 2018, announcing the change to an environmental assessment'."
    },
    {
        "candidate_id": "6bbe2967d0316f04e643",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Coordination Meeting. May 4,2006: The Service received the May 2006 Draft EWEIS, EWEIS Executive Summary, ASIP, and a'."
    },
    {
        "candidate_id": "8a854a5df7dab8663d27",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'to his application, received by the BLM on July 29, 2013. The revised application modified the earlier application'."
    },
    {
        "candidate_id": "97537b9d551bd9af07ed",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'No comments on the Draft Plan-EA were received. March 26, 2020 Rod French, ODFW Confirmation of statement concerning'."
    },
    {
        "candidate_id": "ee4cc6caa2f78ed2a091",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Air Pollution Control Permit to Construct (PTC) from September 21, 2023, through October 21, 2023. The comments received'."
    },
    {
        "candidate_id": "c32d1bcb6efd3c903139",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'comment period on the preliminary EA occurred from December 7, 2018 to January 9, 2019. Three comment letters'."
    },
    {
        "candidate_id": "f7ef5f392f712ad04e45",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'the cancellation of the 2017 withdrawal application, on May 15, 2019, the BLM renewed two hardrock mineral leases'."
    },
    {
        "candidate_id": "9342dc268353a1b8a15b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'EGLE Air Quality Division, Bay City District Office 4/20/23 Phone call regarding respective project and air permitting'."
    },
    {
        "candidate_id": "ca68e6c2f53ab92a9e78",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Joint Permit Application submitted to the USACE in January 2016) which includes procedures to monitor for inadvertent fluid'."
    },
    {
        "candidate_id": "57418c5a9cc4a92fe935",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'On September 26, 2022, the intervenors supplemented their November 2021 and April 2022 comments and protests. The Bureau'."
    },
    {
        "candidate_id": "477a9d06367013759c2e",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'the local newspaper. The comment period ran from December 24, 2014 through February 9, 2015, and four comments'."
    },
    {
        "candidate_id": "8abcdcba84c5fe7ac534",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'decision. The EA that was prepared for the March 2019 oil and gas lease sale considers effects that'."
    },
    {
        "candidate_id": "6884fc47421c5b1b4b7b",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'BLM initiated internal scoping on this project on September 7, 2023, with an interdisciplinary team meeting. The interdisciplinary'."
    },
    {
        "candidate_id": "0d6b1e394e13984c78b7",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'RA (Rural Residential/Agricultural) to M2 (Light Manufacturing) on October 13, 2010, which was approved by a unanimous vote'."
    },
    {
        "candidate_id": "baa199c5863b7d2de964",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'encouraged comments and information to be received by February 25, 2013, for each group of allotments but did'."
    },
    {
        "candidate_id": "543df146f57d9462a974",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'lethal removal as proposed in the eligible entities June 13, 2019, application. Bonneville would fund the request from'."
    },
    {
        "candidate_id": "6c2e63ff31a77c1bdf78",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'resources would be affected at the location. On May 8, 2012, MDAH issued a clearance letter with its'."
    },
    {
        "candidate_id": "ef1761ab26bbbaebbdf8",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Introduction: Elmore County, Idaho filed an application on October 16, 2017, requesting a right-of-way (ROW) from the Bureau'."
    },
    {
        "candidate_id": "266266a55f7ea3522b7a",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'a representative from the City of The Dalles (March 2019). The purpose of these meetings was to discuss'."
    },
    {
        "candidate_id": "de5a2bec46b23cba9563",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Oficina de Geolog\u00eda e Hidrogeolog\u00eda, Junta de Planificaci\u00f3n 02/02/2024 Federal Consistency Review Process Meeting 02/05/2024 Federal Consistency'."
    },
    {
        "candidate_id": "e0e880e87fa379c683d7",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'agencies, organizations, and members of the public on January 9, 2004. A scoping notice was published on the'."
    },
    {
        "candidate_id": "4ede0d19b4b198582db9",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'transmission line for the Tribes to tie into. July 11, 2019 Letter from WAPA to the BLM Lake'."
    },
    {
        "candidate_id": "90ff8992aca450179b2a",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'DOE NEPA Register NOI date (DOE/EA-1977): 2014-06-04'."
    },
    {
        "candidate_id": "dd0e38a3b3c3545fcce9",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Energy Technology Laboratory Final Environmental Assessment Appendix I 122 December 2011 >>> <West.Norman@epamail.epa.gov> 10/18/2011 6:49 PM >>> Greetings,'."
    },
    {
        "candidate_id": "3224a4ab753b7a7dd8f4",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote 'Rails to Trails Act Macon County, Illinois Date: June 2018 Type of Action: Railroad Right-of-Way Conveyance under the'."
    },
    {
        "candidate_id": "c01d81e86484c5271268",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'scheduled to last 15 days and end on September 28, 2010. DOE received a request to extend the'."
    },
    {
        "candidate_id": "5adcaab8a1cbf17020e1",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'and Approvals: 3/26/10 Mayor & Legal Introduction Application 4/21/10 Arch & Building Architectural Approved 5/6/10 Planning Project'."
    },
    {
        "candidate_id": "f0c6dd5db0b032b593a1",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'a draft BA to NMFS for comment in December 2010 as part of the preconsultation process and is'."
    },
    {
        "candidate_id": "a66990c60e2e0b17108f",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote 'from which it could extrapolate relevant data. On June 25, 2013 DOE issued a request for information (RFI)'."
    },
    {
        "candidate_id": "37a4df9ede6cbab0a4dd",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'EA process. The public comment period began on March 11, 2019, and Bonneville accepted comments on the program'."
    },
    {
        "candidate_id": "ad4e7b055c7f14e887ad",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Annotated (MCA). MATL submitted its MFSA application in December 2005. For DOE, the initial step was MATL\u2019s submission'."
    },
    {
        "candidate_id": "2af85b1698f0abbca5ae",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'scoping meeting was planned for this project. On April 2, 2007, DOE posted a Request for Public and'."
    },
    {
        "candidate_id": "a22819c287a8e1b64b10",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Avian and Bird Protection Plan for Perrin per 4/15/11 phone call. [SWCA reported that Hopi received the'."
    },
    {
        "candidate_id": "855924eaa8d0a3516e0b",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'application and approved the special use permit on October 28, 2010, recommending approval by the governing body (City'."
    },
    {
        "candidate_id": "39767d02564959778cdb",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote '4, 2012), reh\u2019g denied DOE/FE Order No. 2961-B (Jan. 25, 2013), amended by DOE/FE Order No. 2961-C (May'."
    },
    {
        "candidate_id": "58947dc9ade7e6c02e2c",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'during a 30-day public review period that ended September 30, 2024. Nine comment submittals (written and by voicemail)'."
    },
    {
        "candidate_id": "a430ae4b906f40c81ec4",
        "label": "neither",
        "notes": "Neither: comment/review date, quote '7, 2010 U.S. Department of Agriculture, Forest Service May 10, 2010 Oregon Department of Fish and Wildlife May'."
    },
    {
        "candidate_id": "e97c1af9b35f6eafc70c",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'AZ SOLAR 1 INTERCONNECTION PROJECT (DOE/EA-2098) Scoping Summary January 2019 Western Area Power Administration (WAPA) is responding to'."
    },
    {
        "candidate_id": "695c8f98caeb1a6f1d32",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'of the project site with Maricopa County in June 2008 for approval to amend an existing land use'."
    },
    {
        "candidate_id": "bbf23810dcc28777e8e8",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'Courtesy notice re: pending loan application Date: Wednesday, January 31, 2024 10:18:00 AM Attachments: image001.png Locator Map.png Project'."
    },
    {
        "candidate_id": "efa76085f6ac2cffa6e3",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'migratory birds, range, soils, socioeconomic values, and wildlife. May 9, 2014 \u2013 The permittee was emailed to inform'."
    },
    {
        "candidate_id": "b54a0f48011efd0f03f1",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'The draft EA was distributed via letter on November 13, 2006, and the comment period extended through December'."
    },
    {
        "candidate_id": "6824408dd48bfa4202c5",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'at regional aquifer monitoring well R-28. Subsequently, on January 7, 2014, we submitted an application amendment to broaden'."
    },
    {
        "candidate_id": "86a5684d333f41249133",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'Application OGPe Final project design (100%) Expected in July 2024 Expected in July 2024 Fire department and Department'."
    },
    {
        "candidate_id": "c0882bad526bf97047e8",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Suquamish Tribe March 3, 2010 American Waterways Operators May 6, 2010, and June 1, 2010 Based on the'."
    },
    {
        "candidate_id": "ee678f49889b1ecbfe9c",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'proposed mitigation plan provided by the applicant. On April 2, 2010, the applicant provided a revised version of'."
    },
    {
        "candidate_id": "d1e0a2ca9fd44a6a98f1",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'by: Erin Julianus and David G. Parker Date: March 30, 2015 and April 1, 2015 Type of Assessment/Sources:'."
    },
    {
        "candidate_id": "860041ca7e6f4326de32",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'to this, through pre- application meetings beginning in January 2008, the Corps had been made aware of the'."
    },
    {
        "candidate_id": "add38ec1dbbfa5a4cf04",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Proposed Action and Alternatives DOE EA-1789 21 October 2010 Activity Permit, Plan or Approval Parties Involved Completed'."
    },
    {
        "candidate_id": "0562454afe98fa2d0a55",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'about 2.25 acres in Butte County, Idaho in September 2014. In addition to the authorization of the ROW,'."
    },
    {
        "candidate_id": "0778cef79e455f453e4a",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'draft EA was distributed for public comment in August 2001, and the public comment period ended on October'."
    },
    {
        "candidate_id": "01f35551c19c23def588",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'permit was prepared and submitted to TDEC in May 2016. The currently-expired NPDES permit continues in effect until'."
    },
    {
        "candidate_id": "8a9eec7ce2289c1ecdc0",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'lethal removal as proposed in the eligible entities June 13, 2019, application. Bonneville would fund the request from'."
    },
    {
        "candidate_id": "b289c84a8689080dc6ee",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'Press published an article about the proposal. On August 13, 2015, the BLM distributed a news release that'."
    },
    {
        "candidate_id": "250909e0ed7121765923",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'EA was made available for public comment beginning December 21, 2015. Letters were sent to the BLM\u2019s interested'."
    },
    {
        "candidate_id": "069d56f1eb3881406220",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'include the following: 1. Executive Order 12312, dated May 18, 2001, which mandates that agencies act expediently and'."
    },
    {
        "candidate_id": "6bb8426b7a798dffade9",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Appendix D to the licensee\u2019s amendment application on May 17, 2019, is approved. The licensee must implement the'."
    },
    {
        "candidate_id": "ce64c36629d2660644ee",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'State Department of Natural Resources was consulted on May 14, 2010 and a follow-up request was placed to'."
    },
    {
        "candidate_id": "9f538287e8e6ee678c4c",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'letters were mailed to all listed persons on April 24, 2012. At this tim...'."
    },
    {
        "candidate_id": "2f60c27a3e2c7433a9c7",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'response to the intervenors\u2019 April 2022 filing. On September 26, 2022, the intervenors supplemented their November 2021 and'."
    },
    {
        "candidate_id": "48c9bc97d6c9361eebe9",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Plan of Development Table Mesa. at I-17 August 10, 2020 Page 6 ICT is filing the SF299 application'."
    },
    {
        "candidate_id": "1197e76f33b6c081c4d6",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'was hosted by the BLM for residents on February 22, 2016. Approximately 260 people were in attendance at'."
    },
    {
        "candidate_id": "23bab19e9fce670a1b03",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Community and the Forest County Potawatomi Community on October 23, 2009, because these communities have expressed a historical'."
    },
    {
        "candidate_id": "8295b05fe955ee746d96",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote 'No. 2-Alt Application for Permit to Drill (APD) March 2024 U.S. Department of the Interior Bureau of Land'."
    },
    {
        "candidate_id": "ec7e8b9c345172edd021",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'of treated groundwater was originally issued on 1 July 27, 2015. On February 6, 2020, NMED approved the'."
    },
    {
        "candidate_id": "ac6d3659901399b27f79",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'DOI-BLM-AKF01000-2017-0041EA Page 50 of 53 December 8, 2017 APPENDIX C BLM Alaska Seismic Conditions of Approval'."
    },
    {
        "candidate_id": "62dc35803540f556cdb7",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'EA for an additional 30-day period beginning on December 17, 2001. In both instances LANL stakeholders and members'."
    },
    {
        "candidate_id": "471e7ee4201bb508d86a",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote '90913 or 90918 CERTIFIED - RETURN RECEIPT REQUESTED June 15, 2012 Sheephook Cattle Grazing Association, LLC c/o Skip'."
    },
    {
        "candidate_id": "a4e977f02f204623d86b",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote '30-day public scoping period for the project on October 24, 2018, ending on November 26, 2018. Scoping letters'."
    },
    {
        "candidate_id": "4404904c946773227381",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'expressed interested in management of the WFCA. On August 5, 2015, the Coeur d\u2019Alene Press published an article'."
    },
    {
        "candidate_id": "b76f30df692e14f55b67",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'Affected Environment and Environmental Consequences DOE/EA 1787 24 December 2010 \uf0b7 Terral River Service operates a dock for'."
    },
    {
        "candidate_id": "cebaa4350fa1b8caabe2",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'that funds must be obligated to sub-recipients by September 30, 2010, and spent by March 2012; therefore, all'."
    },
    {
        "candidate_id": "d8a97ca0803b9b9364d4",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'outlined in the letters dated March 6, and July 1, 2002, to Mr. Frederick Johnson (copies enclosed) to'."
    },
    {
        "candidate_id": "12763f73349d7a74864d",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote '2 began with initiating a scoping letter on January 11, 2013. The letter requested comments and information be'."
    },
    {
        "candidate_id": "cb68ce2f8bdf9a14ba0f",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote '(DRC2008-00097, DRC2009-00004) to San Luis Obispo County on January 13, 2009. The CVSR CUP is needed to allow'."
    },
    {
        "candidate_id": "7b42e0e57175ab07b564",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote 'Description of Current Conditions US Steel Fairless Works (July 1993, Revised March 1994); \uf0b7 Technical Approach to the'."
    },
    {
        "candidate_id": "ec99110211dc39b8907f",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'Hazardous Materials Safety Administration (PHMSA) siting regulations. On August 31, 2018, USDOT PHMSA and FERC signed a Memorandum'."
    },
    {
        "candidate_id": "50858fe17fcfcd56fd3b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'regarding the proposed project have been identified. On November 12, 2009, DOE sent a request to seven separate'."
    },
    {
        "candidate_id": "0c0c5ea7e7171f86e780",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'In response to your request, received on Monday, October 5, 2009, we have reviewed the documents you submitted'."
    },
    {
        "candidate_id": "04669f99c8f6898c5489",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Office held a community scoping meeting on Tuesday, February 10, 2015, from 6:30 to 7:30 pm, to discuss'."
    },
    {
        "candidate_id": "08fd443a9f80bf84a67d",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'by the BLM\u2019s Central Coast Field Office on June 4, 2021 and ended on July 6, 2021. The'."
    },
    {
        "candidate_id": "89c4148355a76408ad76",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'with several members of the equestrian community on December 8, 2015 to develop alternatives with regards to trail'."
    },
    {
        "candidate_id": "516a63a11b7e5056b102",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'about the initiation of the scoping period. On August 23, 2022, the BLM sent the Tribes a letter'."
    },
    {
        "candidate_id": "8128591694f5cfc748e1",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Title: DOE/EA- Subject: Author: Keywords: Comments: Creation Date: 9/10/2008 10:43:00 AM Change Number: 2 Last Saved On:'."
    },
    {
        "candidate_id": "f2f7369a19a0a50fbad3",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Applicant\u2019s Proposed Action BLM received an application on August 21, 2014, for renewal of the grazing permit from'."
    },
    {
        "candidate_id": "f30665dca667951dcdc8",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'Improvement Authority was issued a Tidelands Grant on January 23, 2014 (0814- 08-0002.1 TDG100001) for the upland portions'."
    },
    {
        "candidate_id": "726494ffaff0b18b1dd8",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Treaty Council letter on October 22, 2012. On August 6, 2012, Commission staff held a technical conference to'."
    },
    {
        "candidate_id": "18fb7a3248bfbcf22c41",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote 'No. 1 Application for Permit to Drill (APD) October 2021 U.S. Department of the Interior Bureau of Land'."
    },
    {
        "candidate_id": "53ef8a44edb19782ee65",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote '-----Original Message----- From: Dave Plagge [mailto:DPlagge@fageneng.com] Sent: Thursday, July 25, 2013 10:00 AM To: Mehls, Casey Cc: silka.kempema@state.se.us'."
    },
    {
        "candidate_id": "452e8c1c854815bf36dc",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'Denied Berea City Department Public Meetings and Approvals: 3/26/10 Mayor & Legal Introduction Application 4/21/10 Arch &'."
    },
    {
        "candidate_id": "2185102aecaf18506eaf",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'Evaluation by: Erin Julianus and Carl Kretsinger Date: 5/7/2015 Type of Assessment/Sources: Effect of the proposal on'."
    },
    {
        "candidate_id": "78153275239d7b9bc8be",
        "label": "neither",
        "notes": "Neither: comment/review date, quote '\u2013 Office of Environmental Impact Review (OEIR) on July 21, 2010 for a federal consistency determination. As a'."
    },
    {
        "candidate_id": "906cbc33ca5850006e03",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'Meeting 12/22/2023 AD1006 Comments Received \u2013 Both sites 01/22/2024 Notice of Intent to Prepare an Environmental Assessment'."
    },
    {
        "candidate_id": "3db19976951fd23d7603",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Wind Energy Interconnection Project Public Draft EA Chapter 1 July 2011 3 1.2.1 Western Area Power Administration Perrin'."
    },
    {
        "candidate_id": "164808442202bbf5ba88",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 't (360) 753-91 16. < tnvi ronmental Assessment B-3 October 1996'."
    },
    {
        "candidate_id": "3f8006b3aafec3ad0e22",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Mr. Strickland: I have received your application, dated October 28, 2011, and January 11, 2011, for a crossing'."
    },
    {
        "candidate_id": "41fdc12d594aa72c6586",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2017-06-07'."
    },
    {
        "candidate_id": "1be694e5873e8b0f681d",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'written comments on the Draft version. Beginning in November 2002, Ruby English of Active Citizens for Truth (ACT)'."
    },
    {
        "candidate_id": "764777ea01c0aa7d5e27",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'agency and public comment between August 31 and November 29, 2018; those comments and the BLM\u2019s responses are'."
    },
    {
        "candidate_id": "ea3bc210cf6b8dd75ece",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'for revision of that application was received on July 21, 2012, along with comments following review of the'."
    },
    {
        "candidate_id": "78d7f082a64232f0f551",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'BLM received a concurrence letter from SHPO on February 20, 2019 (EOI ES3320 [2258]) and March 6, 2019'."
    },
    {
        "candidate_id": "6d4a1ee8028e0100e92d",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote 'Campbell Tract Integrated Pest Management February 2024 Environmental Assessment 34 3.5.2.2. Impacts of the Alternative'."
    },
    {
        "candidate_id": "c52b868d54781daa425f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2020-07-01'."
    },
    {
        "candidate_id": "af34b5d5eefc5dc32ef0",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'application was revised in October 2007, August 2008, May 2009, and January 2010 to reflect changes and refinements'."
    },
    {
        "candidate_id": "811e7f3c05d1eaea7c98",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-09-12'."
    },
    {
        "candidate_id": "6054e0c343cc28952195",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'and comment was given to the BLM on August 19, 2010. Scoping comments were received from the IDFG'."
    },
    {
        "candidate_id": "30288992f1b593503dd4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2024-08-19'."
    },
    {
        "candidate_id": "e386e9e395d41193d71e",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Eastport, Maine September 8, 2009 Washington County Commissioners September 9, 2009 Alan Stein September 11, 2009 U.S. Fish'."
    },
    {
        "candidate_id": "c61509bf93dcd36b7a8e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-07-13'."
    },
    {
        "candidate_id": "c7ca0347103ad5699a17",
        "label": "neither",
        "notes": "Neither: historical authorization or reference, quote 'Denver, CO 80203 303-866-3395 _______________________________________________________________________ File Access Request July 10, 2008 A completed and signed copy of this'."
    },
    {
        "candidate_id": "ceea02d5936f5a6a6577",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'response to Western's Request for Proposal issued on September 30, 2014, Sempra proposed to construct generation of up'."
    },
    {
        "candidate_id": "20feada881726bf0d99b",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'period, which occurred from March 22, 2019 to April 21, 2019. Comments received during the 30-day comment period'."
    },
    {
        "candidate_id": "e97e85508dedec1b0f66",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote '(NSB). The BLM ROW application was submitted on April 20, 2018, by the Applicant to the BLM Arctic'."
    },
    {
        "candidate_id": "5d6e2db9bca1dd5804ea",
        "label": "neither",
        "notes": "Neither: EA document or mid-process date, quote '1998); \uf0b7 Request for Non-Use Aquifer Determination dated January 1999; and \uf0b7 Non-use aquifer determination issued by PADEP'."
    },
    {
        "candidate_id": "cfcfb64af12a2316beab",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Cameron Parish, Louisiana (Expansion Project). SUPPLEMENTARY INFORMATION: On September 30, 2013, Sabine Pass filed an application with FERC'."
    },
    {
        "candidate_id": "a2d229b4dc71f490f868",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-06-20'."
    },
    {
        "candidate_id": "cdf8817a39fac982d2a9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-07-12'."
    },
    {
        "candidate_id": "bf55586045ac95944ea5",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'draft EA was distributed for public comment in August 2001, and the public comment period ended on October'."
    },
    {
        "candidate_id": "3cea633aa9e7f4b59406",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2020-01-14'."
    },
    {
        "candidate_id": "fd4214a5711cdbe84869",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Corps Public Notice (\"CPN\") in a letter Dated March 11, 2009. In their letter, the EP A reiterated'."
    },
    {
        "candidate_id": "aae99a2213daf70fc74a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-02-17'."
    },
    {
        "candidate_id": "3ceb7ce5c076fe8efb09",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'from Western Watersheds Project and WildEarth Guardians on February 2, 2015. The following is a summary of the'."
    },
    {
        "candidate_id": "c14b27e0a2d5ede7d5f3",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'attended a meeting of the Seth-De-Ya-Ah Corporation in September 2015 and provided an update on the Tolovana Hot'."
    },
    {
        "candidate_id": "b6622f8a1d6bde7b3431",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'November 2024 DOI-BLM-ORWA-M000-2024-0001-EA 150 This issue was considered but not'."
    },
    {
        "candidate_id": "86ba558602fba63be7fe",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'result in any adverse air quality impact. In August 2010, VSCR submitted a PSD permit application to LDEQ'."
    },
    {
        "candidate_id": "65696cf0a7ef9f58b425",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-05-03'."
    },
    {
        "candidate_id": "2c910028cda0ac16e5ba",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'the project on October 24, 2018, ending on November 26, 2018. Scoping letters were mailed to 59 agencies,'."
    },
    {
        "candidate_id": "f0abcce1898e06f97b10",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'B). The BLM received confirmation from SHPO on February 19, 2020, that the BLM could proceed with the'."
    },
    {
        "candidate_id": "85d3474b971ab6bce86e",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'response to Western's Request for Proposal issued on September 30, 2014, Sempra proposed to construct generation of up'."
    },
    {
        "candidate_id": "fad726551d4aa1674c54",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2022-01-05'."
    },
    {
        "candidate_id": "e92f5b5187a2a5974d6b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Bond Release Approval, Supplemental Order No. LSM-1-A (08-1), May 9, 2008. o Land Use Change Approval, Supplemental Order'."
    },
    {
        "candidate_id": "61ac2fe1dcb34925c8c4",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'at http://www.blm.gov/id/st/en/fo/pocatello/travel_management.html. The 90-day scoping period started on May 31, 2012 and ended on August 31, 2012. The'."
    },
    {
        "candidate_id": "f410ab7fbf4f0a86caf7",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'scoping period for the EA that began on October 6, 2023, and ended on November 6, 2023. The'."
    },
    {
        "candidate_id": "f92639eb0009acf8019a",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'of DOI-BLM-AZ-C030-2007-0050- EA and the Bullhead TMP. On December 2, 2008, the EA and the associated draft implementation'."
    },
    {
        "candidate_id": "79798b70b246cb5b0165",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'A. The public scoping comment period closed on February 4, 2015. During the scoping period, DOE received three'."
    },
    {
        "candidate_id": "98c31af12948b8e6141b",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Decision was not sent out for service until February 13, 2013. BLM: Although, this is more of a'."
    },
    {
        "candidate_id": "f68f53ac3a429ec84787",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2024-07-16'."
    },
    {
        "candidate_id": "cf8bcdb119a050cc4b28",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-12-13'."
    },
    {
        "candidate_id": "42a8dafe16ed13935bc8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2012-01-05'."
    },
    {
        "candidate_id": "0ea3bf665cddc06635a2",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'biological assessment was submitted for their review on December 12, 2011 resulting in receipt of a biological opinion'."
    },
    {
        "candidate_id": "8209c67b82d03b1f11be",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'on July 20, 2011. The appeal was dated July 29, 2011. 1.5 Comments Received and Issues Identified During'."
    },
    {
        "candidate_id": "9fe96873aaac638b3978",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Resources database was conducted by BLM\u2019s archaeologist on Jan. 20th, 2021. There are surveys and sites recorded. Some'."
    },
    {
        "candidate_id": "5b008725c3085bac3edb",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Four Southern Tribes Cultural Resources Working Group on January 21, 2022. Additionally, the BLM attended two virtual public'."
    },
    {
        "candidate_id": "9d2936edec43e73e1125",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or milestone, quote 'David G. Parker Date: March 30, 2015 and April 1, 2015 Type of Assessment/Sources: Review of application materials,'."
    },
    {
        "candidate_id": "566dd6efa397c93aeef6",
        "label": "neither",
        "notes": "Neither: comment/review date, quote 'Separation and Purification Plant Oliver County, North Dakota December 2023 A public comment period was held regarding the'."
    }
]


def main():
    path = "phase2/output/deliverable04/labeling_sample.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab = pd.DataFrame(LABELS)
    merged = df.merge(lab, on="candidate_id", how="left", suffixes=("", "_new"))
    blank = merged["label"].astype(str).str.strip().eq("")
    non_test = merged["split"].astype(str).str.strip().ne("test")
    has_new = merged["label_new"].notna()
    apply = blank & non_test & has_new
    merged.loc[apply, "label"] = merged.loc[apply, "label_new"]
    merged.loc[apply, "notes"] = merged.loc[apply, "notes_new"]
    merged[df.columns].to_csv(path, index=False)
    print(f"Applied {int(apply.sum())} labels")


if __name__ == "__main__":
    main()
