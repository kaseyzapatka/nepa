import pandas as pd


LABELS = [
    {
        "candidate_id": "656d0a3c4712848332d9",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Reviewed By: Assistant Field Manager Minerals & Lands Date Authorizing Official: /s/ Doug'."
    },
    {
        "candidate_id": "81e03cf8ce443469aa17",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Lorraine M. Christian | Field Manager - ASFO | /s/ L. M. Christian 10/23/2023 | | Darrel'."
    },
    {
        "candidate_id": "b122361d8b01c9aaba33",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: C. Elaine Everitt Date: 08 / 24 / 2015 month'."
    },
    {
        "candidate_id": "3ace02867a2624bba5c8",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'NETL GOV c US Date NEPA Compliance Officer john ganz Digitally signed by john ganz DN'."
    },
    {
        "candidate_id": "c45c7275dc9bb44996f9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Bruce W. Lani Date: 09 / 14 / 2012 month day year'."
    },
    {
        "candidate_id": "a0fd2c3bd4e1c2373b77",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Coordinator SIGNATURE OF AUTHORIZED OFFICER SUZANNE COPPING Digitally signed by SUZANNE COPPING'."
    },
    {
        "candidate_id": "a567c4a8dc9edde1523e",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2023 month day year NEPA Compliance Officer: Pierina Fayish Digitally signed by Pierina Fayish'."
    },
    {
        "candidate_id": "8846a948ae8a71bfcdcf",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date: 2021.05.12 Authorized Officer: 09:55:30 -07'00' Jason West Field Manager, Lake Havasu'."
    },
    {
        "candidate_id": "ac1c35ad37c9317cad53",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: James C. Robinson Date: 07 / 27 / 2022 month day'."
    },
    {
        "candidate_id": "e345bdc6e417e1dff1d5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: B. Andrew O'Palko Digitally signed by B. Andrew O'."
    },
    {
        "candidate_id": "1f2a786c7479cfd9adab",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: Date Determined: 05Nov09'."
    },
    {
        "candidate_id": "e2ab06687dd30876dd3e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Assistant Shoshone Field Manager | HC | 12/20/17 James D. Barnum | Supervisory'."
    },
    {
        "candidate_id": "83571833c9b5faa00451",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 03/21/2013'."
    },
    {
        "candidate_id": "6a9af85b6a0f5ffff944",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '21 09:40:48 -07'00' Authorizing Official: Cody R. Layton for Carlsbad Field Office Manager'."
    },
    {
        "candidate_id": "2111f1196e61a8dba211",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 10/13/2010 Comments: Webmaster: Record ID: 1025'."
    },
    {
        "candidate_id": "78a45317b75a15dedc0f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Summer Lee DATE: 06 /28 / 2010 month day year NEPA'."
    },
    {
        "candidate_id": "0338e2b1f46212aa41b4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'NEPA Compliance Officer Date Determined: 03/03/2016'."
    },
    {
        "candidate_id": "2bd38f87aa7e7607001d",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '-07'00' Jared Bybee Field Manager Bristlecone Field Office 6/29/2022 Date Contact Person'."
    },
    {
        "candidate_id": "51704633e00fadbdee27",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '9/16/2016 Assistant Field Manager Authorizing Official: William A. Mier Date: 9/16'."
    },
    {
        "candidate_id": "b8ceaadbbd938daa9ac5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: William W. Aljoe Digitally signed by William W. Aljoe'."
    },
    {
        "candidate_id": "b3d75a9633a8c9ae6bec",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Comments: WS - B - 2010 - 001, Rev.0 Digitally signed'."
    },
    {
        "candidate_id": "02e9588fd44fdaabcefe",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Date: October 21, 2015 Attachment(s): Environmental'."
    },
    {
        "candidate_id": "c9b91efe73ed5609f799",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Curtis 44-9TFH. The FONSI was signed on June 12, 2019. Persons and Agencies'."
    },
    {
        "candidate_id": "a792004a9b335422a0e1",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: Digitally signed by'."
    },
    {
        "candidate_id": "0ad7880ccbb145e0469f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'further NEPA analysis. Authorized Officer: Rhonda Karges, Andrews/Steens Resource Area Field'."
    },
    {
        "candidate_id": "1e10ed0152a477dc5373",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'March 14 1996 and a Finding of No Significant Impact FONSI issued on June 20 1997'."
    },
    {
        "candidate_id": "c65c54370d5cbab57661",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'the project feature prior to construction Task 2 Results CEMVK-EC-H personnel Mr Brian S Johnson made'."
    },
    {
        "candidate_id": "9adf8c9d5054fdcb25ae",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '28 10:24:46 -04'00' Date Determined: 10/03/2019 Comments: TC-A-2018-0083, Rev. 1'."
    },
    {
        "candidate_id": "fcec434f932d1ecb62c8",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: H. Raymond Pratt Date: 09 / 03 / 2015 month day'."
    },
    {
        "candidate_id": "66a183cc9966ba92119a",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2017.12.05 17:29:20-05'."
    },
    {
        "candidate_id": "3a5474306026fddc4292",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '10 18:41:34 -04'00' Date Determined: 09/04/2018 Comments: OBU-G-2018-0225, Rev. 0'."
    },
    {
        "candidate_id": "62e2c40229e80a4afd9f",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'ement Plan Revision Record of Decision and Approved Resource Manag ement Plan \u2013 June 2014'."
    },
    {
        "candidate_id": "730e6d055cfe1fcc1d67",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2019 month day year NEPA Compliance Officer: Mark Lusk Digitally signed by Mark Lusk Date: 2019'."
    },
    {
        "candidate_id": "8254ea406e7d1264ffca",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'interested public on February 27 2004 initiating a 30-day comment period Two public meetings were held'."
    },
    {
        "candidate_id": "5e1dfa5c8a67702a0957",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for: Keith E. Berger, Field Manager DATE SIGNED: 4/19/17'."
    },
    {
        "candidate_id": "f65b8869f876422f1d61",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Carrie Stewart Date Determined: May 8, 2017'."
    },
    {
        "candidate_id": "4739fb513a3d342fa8c2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '15 11:48:00-06'00' Authorizing Official: Date: Keith Rigtrup Field Office Manager Contact'."
    },
    {
        "candidate_id": "6c7e2d072aaad804fa5a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '10 13 12 31 01 04 00 Date Determined 10/13/2023 Comments EEC No TC-A-2023-00064 Rev No'."
    },
    {
        "candidate_id": "9dd8d4da1a41520112f3",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Gary L. Covatch Date: 04 / 15 / 2014 Digitally signed'."
    },
    {
        "candidate_id": "38f1fa125c8234e54232",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'analysis is required. Approving Official(s): LORRAINE Digitally signed by LORRAINE CHRISTIA'."
    },
    {
        "candidate_id": "0e4dc534ed0128833605",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Information Bryant D. Smith Field Manager Date: 12/21/16'."
    },
    {
        "candidate_id": "1959ac762f80c976f9a1",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Junction Field Office Record of Decision and Approved Resource Management Plan (ROD/RMP). Date'."
    },
    {
        "candidate_id": "b2812e4e5d596665c2f5",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'temporary and confined to existing routes and ways No impact to geological resources mineral or energy'."
    },
    {
        "candidate_id": "871dd98552f2eddab663",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Reviewed By: Assistant Field Manager Minerals & Lands Date Authorizing Official: /s/ Doug'."
    },
    {
        "candidate_id": "f11fb48837efb7ed6653",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'review. Tracy Ribeiro NEPA Compliance Officer: Tracy Ribeiro 2016.07.13 12:22:01 -06'00' Date'."
    },
    {
        "candidate_id": "8e437753eff02e3063d4",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Kenneth J. Crane, Burley Field Manager Date: 1-2-24 Contact Person For additional information'."
    },
    {
        "candidate_id": "51ca00bd84e872c33783",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "8d907891d3add2f447c4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '00') **Comments:** **Date Determined:** Sep 9, 2011 LWO-H-2011-0090, Rev.1 Submit via Email Submit to'."
    },
    {
        "candidate_id": "afb0b229dbaf2c1a800f",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'administrative record, a Record of Decision (ROD) will be signed by the Secretary of the Navy'."
    },
    {
        "candidate_id": "1b8a6d5f31a892fbffba",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'and as noticed in the April 15 2011 Federal Register This decision approves the DSSF Agency'."
    },
    {
        "candidate_id": "d31bc89f4806e3d15e6d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Alan L. Blosser DATE: 04 /20 / 2010 month day'."
    },
    {
        "candidate_id": "aa3d3906b5bc566f12c2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2021-0073 Rev No: 1 Date Determined: 12/27/2022'."
    },
    {
        "candidate_id": "e2938c4ed50d5a3a6c8e",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'accompanying CD lanes. EA/FONSI approved on 05/19/15; construction began in 2017'."
    },
    {
        "candidate_id": "de287e68ad3a24f4d8fa",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2014 month day year NEPA Compliance Officer: John Ganz Digitally signed by John Ganz Date: 2014'."
    },
    {
        "candidate_id": "5b000436d1ff2da6f4d1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Lucy A. Ribeau Date Determined: 12/13/2011'."
    },
    {
        "candidate_id": "704452ba51c6a5305467",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'forth in the Tribe s October 6 2017 letter to the Bureau of Reclamation provided as'."
    },
    {
        "candidate_id": "74eb2943a28132ba68f4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer James L. Elmore Date Determined: 9/17/2010 Record ID: 27'."
    },
    {
        "candidate_id": "185dbb713e9ef3374486",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Stephen A. Danker Date Determined: 10/13/2015 Comments: OBU-L-2015-0111, Rev. 0'."
    },
    {
        "candidate_id": "95f0e90cb4b661c38054",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Adam Payne Date: 01 / 13 / 2021 month day year NEPA'."
    },
    {
        "candidate_id": "1b521228ed4ed45b704b",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Environmental Assessment Decision Record. Decision Record, Environmental Assessment. Arcata'."
    },
    {
        "candidate_id": "847c4d61aa7260dd6706",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 13:44:47 -04'00' Date Determined: 08/10/2021 Comments: OBU-G-2021-0196, Rev. 0'."
    },
    {
        "candidate_id": "21a2b98b21da1a03fef1",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'was prepared, and a Decision Record was signed July 2013: Kerr McGee Oil & Gas Onshore'."
    },
    {
        "candidate_id": "93d73f4a405fe06ca69a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Loren C. Wickstrom Field Manager North Dakota Field Office 5 ATTACHMENT 1 \u2013 STIPULATIONS'."
    },
    {
        "candidate_id": "25a693986322f0339e2a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 6/10/2010 Comments: Webmaster: Record ID: 125'."
    },
    {
        "candidate_id": "96425d4c098342f093a0",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Reservation Yankton Sioux Tribe In follow up to the tribal consultation letters BLM cultural resource specialists'."
    },
    {
        "candidate_id": "fc986a06df59d99eb2ee",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Education Meetings In February 2016 following publication of the NOI for the SDNM RMPA/EIS the BLM'."
    },
    {
        "candidate_id": "b27f6ca642d11225b902",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'and the overall DMMP September 2001 Consultation was initiated with NMFS May 2002 Consultation was reinitiated'."
    },
    {
        "candidate_id": "34045551ce6448105788",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'River Complex EIS Draft April 2021 6-7 Public Involvement and Distribution 6 6 2 3 Press'."
    },
    {
        "candidate_id": "ccc7a45b48cea55a11e4",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'ASSESSMENT Executive Summary JUNE 2014 ES-3 ES-3 PUBLIC SCOPING Consistent with CEQA and NEPA requirements public'."
    },
    {
        "candidate_id": "eac0a2971360a4cbe791",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Plants Draft EIS/OEIS August 2022 ES-6 Executive Summary and public scoping meetings A project website was'."
    },
    {
        "candidate_id": "122ca5b7e3888b273bc0",
        "label": "neither",
        "notes": "Neither: construction/activity period date, quote 'planning stage are under construction or that have been constructed Construction of this EIP was completed'."
    },
    {
        "candidate_id": "585a2a91a9cb22a8e63c",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Corridor Project S-10 July 2021 Draft EIS/EIR Executive Summary S 4 Affected Environment and Environmental Consequences'."
    },
    {
        "candidate_id": "97f58c7db44340fd8968",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'September 2016 D-5 Final EIS the low, intermediate, and high predictions of sea'."
    },
    {
        "candidate_id": "34a78c28a330a343f987",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'CO-G010-2023-0024-CX SEPTEMBER 2023 1 BLM U S DEPARTMENT OF THE INTERIOR BUREAU OF LAND MANAGEMENT INTRODUCTION'."
    },
    {
        "candidate_id": "5b558f4ed6f1a9c03031",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Corridor Project 4-272 | July 2021 Draft EIS/EIR Chapter 4: Affected Environment and Environmental'."
    },
    {
        "candidate_id": "db6b0673175e48edbe23",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Service project website Scoping comments were accepted through the project website email hard copy or fax'."
    },
    {
        "candidate_id": "7e534245f08e363a0f58",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'SURTASS LFA Sonar Final June 2017 1-12 Purpose of and Need for the Proposed Action 1'."
    },
    {
        "candidate_id": "b98f9477e4bbf0e7ea0f",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'and Wildlife Service s January 2025 Final Environmental Impact Statement for the Elliott State Research Forest'."
    },
    {
        "candidate_id": "5d45df0b7851e3c9b9bc",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Amendment/Draft EIS was published in the Federal Register in March 2009 The NOI initiated a 90-day'."
    },
    {
        "candidate_id": "ca30ac873bf026cfe0b4",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Battle Creek Allotment Map 27 Bighorn sheep and cattle sightings from aerial observations Apr-June 1988-91 Compiled'."
    },
    {
        "candidate_id": "cc22edf849fb8cf2f2f2",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Johnson unpubl rep March 2003 in cooperation with the USFWS Each Forest was asked to determine'."
    },
    {
        "candidate_id": "4598cc4b872682115567",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'APPROVED OMB NO 1004-0137 Expires January 31 2018 Form 3160-5 June 2015 UNITED STATES DEPARTMENT OF'."
    },
    {
        "candidate_id": "722b08158f42e23379e4",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '2009 October 2009 and April 2010 including 14 separate scoping meetings and over 1 400 public'."
    },
    {
        "candidate_id": "15c1bb0556ff66eab532",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'November 2023 | 6-5 Final EIS for KC-46A MOB 6 Beddown GLOSSARY Operation: An aircraft'."
    },
    {
        "candidate_id": "7bfb5dbc7c11718771e2",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Office June 2018 DECISION RECORD Wamsutter LLC DOI-BLM-WY-D030-2015-0086-CX Latham'."
    },
    {
        "candidate_id": "ed4220100a4d9ff2e4a2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Sincerely, Paul Briggs Field Manager Enclosures: 1. DOI-BLM-UT-2018-0024 2. DOI-BLM-UT'."
    },
    {
        "candidate_id": "4db8a102991c4d7d7b1b",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Mine Complex Project September 2016 Draft Environmental Impact Statement 1 10 2 Public Scoping Meetings 1'."
    },
    {
        "candidate_id": "a6f8f2505700bb86610f",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Final EIS Fact Sheet August 2002 Page 1 Wallula Power Project and Wallula-McNary Transmission Line Project'."
    },
    {
        "candidate_id": "fd923ce31554e1bba593",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'December 20 2022 and ended on January 19 2023 The BLM received 14 comment submissions from'."
    },
    {
        "candidate_id": "c982d338b42b30c5cc77",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Tribal governments In July 2010 the Payette National Forest and Nez Perce Tribe accepted invitations to'."
    },
    {
        "candidate_id": "db0f803844b5d6ff22f4",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'and attend an agency scoping meeting held in July 2011 Participating agency meetings were also held'."
    },
    {
        "candidate_id": "ea5f128a9c0cb3cc1c81",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Section 3 8 1 and at the October 2019 public hearings MDOT coordinated with the officials'."
    },
    {
        "candidate_id": "4e0c10deff8bd728d364",
        "label": "neither",
        "notes": "Neither: construction/activity period date, quote 'Statement for the Siting Construction and Operation of New Production Reactor Capacity Volume 3 Sections 7-12'."
    },
    {
        "candidate_id": "caa6991ffc7bf28fbebe",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'and Gravina Island In July 2004 FHWA and DOT PF issued a Final Environmental Impact Statement'."
    },
    {
        "candidate_id": "4659125a3a4e905f4f65",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote '(See Attachment A \u2013 Map) In November 2020, Louis and Patricia Von Proksch purchased the property'."
    },
    {
        "candidate_id": "aeb1972e281f5dd4a876",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'government officials in February 2010 The scoping meeting itself took place on February 16 2010 The'."
    },
    {
        "candidate_id": "a31561281e77e986dde3",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'the EIS The 30-day public scoping period for the EIS formally began on April 13 2010'."
    },
    {
        "candidate_id": "257496ff4e81a6bcbe43",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '2018 and Scoping in April 2019 Sound Transit held large open houses in three different locations'."
    },
    {
        "candidate_id": "14d7d1849364c25d98b5",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Programmatic EIS USDOI July 2012 BOEM Consultation and Coordination 8-2 weather and subsequently could'."
    },
    {
        "candidate_id": "7fc49b654a3df6818071",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'from July 2010 through July 2011 In February 2011 AIRPA held a public meeting to introduce'."
    },
    {
        "candidate_id": "5eb0f565a4930478c2b8",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Register and will continue until December 29 2005 DOE will consider all comments received or postmarked'."
    },
    {
        "candidate_id": "236e556b005242c6ae89",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Plant Stabilization May 1996 file I Data 20Migration 20Task/EIS-0244-FEIS-1996/eis0244f f html 6/27/2011 2 33 49 PM'."
    },
    {
        "candidate_id": "31b1b86be5a94650e231",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'information added in response to comments to incorporate new/updated information not available when the Draft EIS'."
    },
    {
        "candidate_id": "8173e7d1da37fe8fb268",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Chalkyitsik Village tribal governments copies of the preliminary alternatives for the Proposed RMP/Final EIS for their'."
    },
    {
        "candidate_id": "3ce4cb04eca25f361b4e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'comment periods conducted between August 2003 and February 2004 were open for over 30 days and'."
    },
    {
        "candidate_id": "f049250c477e63be3420",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Office February 2018 DECISION RECORD DOI-BLM-WY-D030-2018-0048-CX Forster, Reg & Laurie'."
    },
    {
        "candidate_id": "f50141df781ea43d5f6a",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Executive Summary November 2016 Final EIS for Eagle Take Permits for the CCSM Phase I Project'."
    },
    {
        "candidate_id": "0f434a93a72d700c5ccb",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'housing complex during site visits conducted in February 2023 the closest viewpoint from the property was'."
    },
    {
        "candidate_id": "8ffc813832c848177ede",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Project 1. INTRODUCTION March 2016 1-7 Final EIS/EIR A total of 21 unique commenters (8 individuals'."
    },
    {
        "candidate_id": "00571cb413c899a64b53",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'day public review and comment period that opened on February 15 2019 and closed on March'."
    },
    {
        "candidate_id": "a705a1c8044fdebf05a4",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'FERC/FEIS-0333 July 2023 CP2 LNG and CP Express Project FINAL ENVIRONMENTAL IMPACT STATEMENT Venture Global CP2'."
    },
    {
        "candidate_id": "0232667c6680aaeaa579",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Fort Worth District September 2019 Final Environmental Impact Statement Lake Ralph Hall Regional Water Supply Reservoir'."
    },
    {
        "candidate_id": "b883e20d01ffbb1c9046",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'on the draft EIS in September 2005 EPA requested that the final EIS include information about'."
    },
    {
        "candidate_id": "7d40ddc38c851d030662",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'the property at the November 2018 public meeting see Section 3 8 1 and at the'."
    },
    {
        "candidate_id": "1861562649451c7a4bbe",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'qualification plant In late December 2021 Syrah completed the development of its Stormwater Pollution Prevention Plan'."
    },
    {
        "candidate_id": "419ae874cac15d711aa0",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'September 2024 Draft EIS for the Proposed Modernization of the Bridge of the'."
    },
    {
        "candidate_id": "6b226b6425fe936e12aa",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Environmental Consequences April 2011 Desert Sunlight Solar Farm Project Final EIS and CDCA Plan Amendment 4'."
    },
    {
        "candidate_id": "c5a605c9d1017021a14e",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'have occ rred between October 2008 when'EP A as a member of the Core Team signed'."
    },
    {
        "candidate_id": "2346cba12b9c542e579e",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'Glen Canyon Dam Long-Term Experimental and Management Plan October 2016 Final Environmental Impact Statement lxv HRR'."
    },
    {
        "candidate_id": "409e8fe537c9c341ac05",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Action and Alternatives July 2015 Coeur Rochester Mine Plan of Operations Amendment 10 and Closure Plan'."
    },
    {
        "candidate_id": "bb01006cd5ea47521c1d",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'interested parties in late January 2003 On February 7 2003 the Environmental Protection Agency EPA published'."
    },
    {
        "candidate_id": "0372572ffc97bd18c23c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'unsatisfactory to the authorized officer, the holder, shall within 30 days of demand, furnish'."
    },
    {
        "candidate_id": "2da49205d7cca76e4400",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '21/2017 Amanda Dodson Field Manager, Kingman Field Office Date Attachment: Form 1842-1'."
    },
    {
        "candidate_id": "3943a5e37ba302649884",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Section 4 f Evaluation September 2019 24 11 Final Section 4 f Evaluation 1082 FRA will'."
    },
    {
        "candidate_id": "8f2daa9d97c405ae6082",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'decision as the Worland Field Office Manager to approve the associated amendment right-of- way'."
    },
    {
        "candidate_id": "429ec6981bff528decca",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Table of Contents November 2018 5 Ho-Chunk Nation Beloit Fee-to-Trust and Casino Project Draft Environmental Impact'."
    },
    {
        "candidate_id": "f60bb854f77c211e6fde",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'and in a letter dated January 8 2004 BLM initiated consultation with the following Native American'."
    },
    {
        "candidate_id": "abccc7cd3073ba293e55",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'District anticipates the public meeting will be held in January 2019 The public review meeting will'."
    },
    {
        "candidate_id": "feadf53c4b9e45ab9bc2",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'District is a descriptive term F inal August 1995 encompassing all features on top of Foote'."
    },
    {
        "candidate_id": "bd4a2e4acac19e45020e",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Development Project EIS ROD was approved in September 2016. The CD-C EIS was'."
    },
    {
        "candidate_id": "8172041eee1741668fa6",
        "label": "initiation",
        "notes": "Initiation: posted to ePlanning/NEPA Register, quote 'Kingman Field Office, May 2015. Available at https://eplanning.blm.gov/epl-front\u00ad office/eplanning/planAndProjectSite.do?methodName=renderDefaultPlanOrProjectSite'."
    },
    {
        "candidate_id": "7ae30fa8f34323bd8cf6",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Office June 2016 DECISION RECORD Iron Bar Holdings LLC. \u2013 McKee Ranch to Tetrad Corp'."
    },
    {
        "candidate_id": "facf5ad97358f1366292",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'accepted DON s offer in September 2016 Public Review of the DEIS The next step of'."
    },
    {
        "candidate_id": "d48ea21544ce50d406de",
        "label": "neither",
        "notes": "Neither: consultation date, quote '5 will take effect USFWS letter of concurrence March 2018 see Mitigation section The Monarch Butterfly'."
    },
    {
        "candidate_id": "1be59753a2d9e9496daa",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'in the RMPA/FEIS and ROD announced in the Federal Register, Volume 72, Number'."
    },
    {
        "candidate_id": "cf2d678312df75f3db52",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Impact Statement Volume I March 2021 Estimated Lead Agency Costs Associated with Developing and Producing this'."
    },
    {
        "candidate_id": "fd627f650864e1a405d2",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'the dra DARHT EIS in May 1995 and the nal classi ed supplement was completed concurrently'."
    },
    {
        "candidate_id": "c6dbea0b97826bd4a6a8",
        "label": "neither",
        "notes": "Neither: survey/inspection/activity date, quote 'A cultural resource survey was conducted for the proposed action in the February 2012 which revealed'."
    },
    {
        "candidate_id": "e25ab6ab203c2791d869",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'December 2024 Final EIS for the Proposed Modernization of the Bridge of the'."
    },
    {
        "candidate_id": "e2704509c431d06d50c3",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'in November 2017 six Open House Meetings were held between May 8 and May 22 2018'."
    },
    {
        "candidate_id": "f92fe3da22d7d5441aa8",
        "label": "neither",
        "notes": "Neither: construction/activity period date, quote 'per year BLM 2012 Operation of the Ruby Hill Mine was temporarily suspended in November 2013'."
    },
    {
        "candidate_id": "7eccb3357496662bb9be",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Code 2023-0058024 June 2023 WisDOT completed the FHWA FRA FTA Programmatic Consultation for Transportation Projects affecting'."
    },
    {
        "candidate_id": "29b674d1056bfdddb7c2",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Supplemental Draft EIS for public comment accepting comments until March 2005 The Supplemental Draft EIS identified'."
    },
    {
        "candidate_id": "ea56076167c56469ee45",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Meetings on Scoping March 2011 Draft Scoping Report July 2011 Public Meetings on Draft Scoping Report'."
    },
    {
        "candidate_id": "f5c7595e41b1254cce03",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'AVIATION ADMINISTRATION MARCH 2021 LGA Access Improvement Project Final EIS | ES-2 | Executive Summary the guiding principles'."
    },
    {
        "candidate_id": "3b3170379e213d2fe233",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Geothermal Leasing Project March 2017 Environmental Impact Statement Draft Scoping meetings conducted in 2012Geothermal flyer posted'."
    },
    {
        "candidate_id": "11b7e41ab1bd30668e2f",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'the BLM decided in December 2009 to prepare an EIS instead of an EA The EIS'."
    },
    {
        "candidate_id": "a37c7e7963d98c8b8b3e",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'draft was released in June 2024 The purpose of this RDEIS is to document the existing'."
    },
    {
        "candidate_id": "baaa26dafa759f04a179",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Section 1 0 Introduction September 2013 1-11 Soboba Band Of Luise o Indians Final EIS This'."
    },
    {
        "candidate_id": "3e07102f0d39feac2b0b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'staff for comment in October 2010 Consultation was concluded in April 2011 The Forest met again'."
    },
    {
        "candidate_id": "52130ef86093091a6aa9",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Gunnison Montrose This map was produced by the BLM Grand Junction Field Office February 2023 Document'."
    },
    {
        "candidate_id": "b0147fc3a7c0fc6f251f",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Mitigation and Monitoring July 2015 Coeur Rochester Mine Plan of Operations Amendment 10 and Closure Plan'."
    },
    {
        "candidate_id": "2903afe69a5382bbe6a6",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Vernal Field Office. Record of Decision and Approved Resource Management Plan. Prepared by'."
    },
    {
        "candidate_id": "d47edd0d0efec4398d9a",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Vegetation Resources February 2013 4 17-27 Alta East Wind Project AEWP Final EIS Fish and Wildlife'."
    },
    {
        "candidate_id": "85c0c2ba3601ada0fca5",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'In addition to the scoping meetings the BLM hosted socio-economic workshops in Cedar City and Beaver'."
    },
    {
        "candidate_id": "9726dac45076e548cf80",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Grazing Lease Renewal DOI-BLM-ORWA-M050-2021-0002-CX 1. Background The BLM is proposing to renew the'."
    },
    {
        "candidate_id": "4d2a9cdfa5427d558fe5",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'environment. The proposed DOI-BLM-ID-T020-2023-0009-CX A-3 action has been reviewed, and none of the'."
    },
    {
        "candidate_id": "c624396a6bc7ccd02545",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Random Investments_CX DOI-BLM-ID-I040-2018-0008-CX Page 3 Extraordinary Circumstance The Holder'."
    },
    {
        "candidate_id": "fde03ead8c4fb10e9746",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'NEPA Register Number: DOI-BLM-AK-A020-2020-0002-CX Case File Number: AA 81315 Location / Legal Description'."
    },
    {
        "candidate_id": "34b6798ff31000063d65",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Flood Project No.: DOI-BLM-CA-N050-2017-0011-CX Project Location Bizz Johnson Trail within the'."
    },
    {
        "candidate_id": "fbd78bb53b0ab485289b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'food, shelter, and 6 DOI-BLM-CO-N040-2019-0002-CX | BLM - Colorado River Valley Field Office security'."
    },
    {
        "candidate_id": "5c7a4fb53a12ae1bad48",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'FLEXPIPE INJECTION LINE DOI-BLM-UT-G010-2019-0088-CX September 2019 Location: Sec 29, T 8S, R 23E'."
    },
    {
        "candidate_id": "af29a5b87c28310a9dff",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CO-F020-2022-0013-CX i. The duty to provide the BLM with reasonable'."
    },
    {
        "candidate_id": "905118418b849ba5626e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Office April 2021 DOI-BLM-WY-R050-2021-0008-CX Lander Field Office 1335 Main Street Lander,'."
    },
    {
        "candidate_id": "1984543aaba2219eacc3",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DOI-BLM-AZ-C030-2013-0023-EA Page 15 Technical Review services in advance'."
    },
    {
        "candidate_id": "ccc7feb8f8255eb048e6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'covered by the CX. DOI-BLM-NV-B020-2020-0024-CX N-53136 RMG Renewal 3 Screening for Extraordinary'."
    },
    {
        "candidate_id": "2686e3cda2f0713e6bd4",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DOI-BLM-AZ-C030-2013-0023-EA Decision Record Page 4 9. DECISION RECORD DECISION'."
    },
    {
        "candidate_id": "cfbd63a22e29bde74a26",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '100' NORTH R 100' DOI-BLM-CO-N05-2018-0077-CX 4+40 BB Proposed Edge of Disturbance Edge of'."
    },
    {
        "candidate_id": "fd0ccdd8a4ae6b9186be",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'not be used. CX#: DOI-BLM-NV-W030-2012-0018-CX Applicant: David R. Vixie Project Title: Paradise'."
    },
    {
        "candidate_id": "61ecde7c8990ce5d1491",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '26 12:52:40 -07'00' DOI-BLM-WY-D040-2024-0017-CX Page 2 of 2'."
    },
    {
        "candidate_id": "452a81a164ff6924a138",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'considered.) Yes, DOI-BLM-CO-N050-2019-0024, the EA for the ELU A24 pad includes wells producing'."
    },
    {
        "candidate_id": "c462e0c90e646fccfa78",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'section d. p. A-1 CX#: DOI-BLM-OR-S000-2014-0001-CX Revised Salem District Special Forest Products'."
    },
    {
        "candidate_id": "9eb5006f16da52426fff",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Field Office (CFO) DOI-BLM-NM-P020-2021-0534-CX IT4RM-P020-2021-0521-CX Right Meow 31 Primary'."
    },
    {
        "candidate_id": "995f9ae5a04e0a07dec7",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DECISION RECORD FOR DOI-BLM-MT-L030-2020-0015-CX BEHM 3D SEISMIC Decision The Bureau of Land Management'."
    },
    {
        "candidate_id": "31a0d352c589e96840f2",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2023-0006-CX Categorical Exclusion Documentation I. Background'."
    },
    {
        "candidate_id": "802484e69dc6ff9ddaf4",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'monitored successfully. DOI-BLM-UT-Y010-2022-0048-CX Extraordinary Circumstances Review 4. Have highly'."
    },
    {
        "candidate_id": "4baf008aace44c77292b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '2 Decision Record DOI-BLM-CA-D060-2021-0036-EA CV Link Project INTRODUCTION The Bureau of Land'."
    },
    {
        "candidate_id": "b4c3f637cd082633294d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Act of 2005 NEPA No. DOI-BLM-NM-P020-2019-0684-CX Project Name: Green Drake Overhead Electric Line Original'."
    },
    {
        "candidate_id": "dfa69236c27bdf4ce678",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Part 46.215 apply. DOI-BLM-ID-B010-2019-0023-CX 1 The following list of Extraordinary Circumstances'."
    },
    {
        "candidate_id": "dcc3eaaf907b3d02ad59",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Forester, 12-11-2019 DOI-BLM-ID-B010-2020-0012-CX 2 3. Have highly controversial environmental'."
    },
    {
        "candidate_id": "c2a70582aa1907f679ab",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Environmental Assessment 15 DOI-BLM-ID-C010-2015-0003-EA Tree mortality is used as a measure of fire severity'."
    },
    {
        "candidate_id": "2f4f5be8a7bdcbf70fee",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Interpretive Site DOI-BLM-ID-B011-2015-0001-EA Page 17 Stillman, A.J. 2006. Population genetics'."
    },
    {
        "candidate_id": "e57f0a4cd430a3868993",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'this unique habitat DOI-BLM-ORWA-L040-2021-0003-CX (Stukel Juniper Treatment) Page 2 feature within'."
    },
    {
        "candidate_id": "a064fddbd923a24f6d70",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '106H NEPA Log number DOI-BLM-NM-P020-2018-0375. Plats and imagery of the lines and area are included'."
    },
    {
        "candidate_id": "8fe3ad03e9e8659e49a6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Nominee Corporation DOI-BLM-NM-F010-2017-0116-CX 1 DECISION I have decided to implement the assignment'."
    },
    {
        "candidate_id": "f2da5e7d4cdc38a3a022",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CO-F020-2020-0063 CX REMARKS: Cultural Resources: Although cultural'."
    },
    {
        "candidate_id": "db4b65c11a1fb1741220",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'RECREATION PERMITS DOI-BLM-UT-G020-2023-0012-CX 1.0 BACKGROUND Bureau of Land Management (BLM'."
    },
    {
        "candidate_id": "3fb981d0f73942454738",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'termination activities. DOI-BLM-NV-B020-2020-0020-CX N-52438 & N-52439 Beatty Renewals 11'."
    },
    {
        "candidate_id": "8927055d29246518e4b2",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established by Statute DOI-BLM-WY-D030-2017-0194-CX Samson Resources Company A. Background BLM Office'."
    },
    {
        "candidate_id": "9efaa925d617373d5b5b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Action allows for sheep DOI-BLM-ID-B010-2023-0018-CX (Livestock Trailing Permit Shirts 101023038)'."
    },
    {
        "candidate_id": "6d1aa9e4935dc29a862d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established By Statute** **DOI-BLM-UT-Y010-2016-0188 CX** **Wildlife Structures - Guzzler Upgrade** It'."
    },
    {
        "candidate_id": "5dc0ba3735231dc57f73",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'for this project (DOI-BLM-MT-C030-2016-0237-CX) and tiered to EA (DOI-BLM-MT-C030- 2015-170'."
    },
    {
        "candidate_id": "304f4312dc4970f064c7",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Field Office NEPA No.: DOI-BLM-AZ-P020-2017-0001-CX Case File No.: AZAR-034883 Proposed Action Title'."
    },
    {
        "candidate_id": "7a856fca003de0be5145",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'NEPA Project Number: DOI-BLM-MT-070-2018-0007-CX Project Title: Fort Benton Contact Station Site'."
    },
    {
        "candidate_id": "84ea0ed420ab2ab74a96",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'ESTABLISHED BY STATUTE DOI-BLM-WY-D090-2024-0015-CX A. Background BLM Office: Bureau of Land Management'."
    },
    {
        "candidate_id": "39f49a51018a03a74e9a",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CO-SO50-2016-0037 CX October, 2016 Bradly Burch to Burch Family Ranch'."
    },
    {
        "candidate_id": "72ab0a8cc620ace231ea",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion Documentation DOI-BLM-CA-N030-2016-0010-CX Case File #: CACA 56756 Applicant: Educational'."
    },
    {
        "candidate_id": "b7d748f0270c6064125d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '[ ] [x] [ ] [x] 6 DOI-BLM-CO-G020-2023-0047-CX | BLM - Colorado River Valley Field Office INTERDISCIPLINARY'."
    },
    {
        "candidate_id": "aad27526b6347590f0cc",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Inch Lay Flat Hose. DOI-BLM-WY-030-2016-0172-CX ROW: WYW-185515 Decision I have reviewed the'."
    },
    {
        "candidate_id": "65cad2cb9969f1b99ab1",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Habitat Conservation DOI-BLM-ORWA-N000-2019-0002-CX Page 2 of 16 cleaning (unless augmentation occurs'."
    }
]


def main():
    path = "phase2/output/deliverable04/labeling_sample.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab = pd.DataFrame(LABELS)
    merged = df.merge(lab, on="candidate_id", how="left", suffixes=("", "_new"))
    blank = merged["label"].astype(str).str.strip().eq("")
    has_new = merged["label_new"].notna()
    apply = blank & has_new
    merged.loc[apply, "label"] = merged.loc[apply, "label_new"]
    merged.loc[apply, "notes"] = merged.loc[apply, "notes_new"]
    merged[df.columns].to_csv(path, index=False)
    print(f"Applied {int(apply.sum())} labels to labeling_sample.csv")


if __name__ == "__main__":
    main()
