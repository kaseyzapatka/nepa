import pandas as pd


LABELS = [
    {
        "candidate_id": "479109d901c8d3245c05",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Exhibit A A Background On March 26 2018 the Bureau of Land Management BLM received an'."
    },
    {
        "candidate_id": "b47ba6822072ae70a7f0",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Land Management BLM received the application for permit to drill APDs from Lime Rock Resources III'."
    },
    {
        "candidate_id": "cad30c1c9718f9a2154c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3cae5cc4db5055266bbf",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7dcaf584416bc089c53e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "63d1ed643dc43228b9f6",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Licensing Board ML23038A210 2/8/2023 Memorandum and Order (Initial Prehearing Order) ML23039A158 2/10/2023 Joint Unopposed Motion'."
    },
    {
        "candidate_id": "7fe1ca962274de15f0a4",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'February 1 2012 On May 29 2014 Juneau Hydro filed its license application On November 17'."
    },
    {
        "candidate_id": "206ff990b2c417e47fe5",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'answer period no oral comments were made The scoping period extended from February 1 through April'."
    },
    {
        "candidate_id": "b0d282c253b6b42366a3",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'JLA 7/21/16 C Drake 7/28/16 EMCK 7/21/16 LP 07/21/2016 CSB 7/25/2016 To be filled out during'."
    },
    {
        "candidate_id": "90439a9ef4f7a7f87549",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "eecd3df71f6d16c3690a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bd5e246641619c7b0f4b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Deborah Buterbaugh DATE: 05 / 14 / 2010 month'."
    },
    {
        "candidate_id": "74a7c9e8743b6138a60e",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'fire-threat areas In August 2023 the BLM Ukiah Field Office UKFO received an application from PG'."
    },
    {
        "candidate_id": "e52c653764507db2cda0",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'TGL application The NOI was published July 3 2019 and included information on a public scoping'."
    },
    {
        "candidate_id": "62333a384faacb1a6ef8",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BRIAN MOLLOHAN Digitally signed by BRIAN MOLLOHAN'."
    },
    {
        "candidate_id": "bdb558f923b7b91b750a",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Shoshone Field Office received an application from Qwest Communications DBA CenturyLink QC Qwest on May 20'."
    },
    {
        "candidate_id": "94f7f766b31e25203a5b",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote '18, 2021 FAA Issues Record of Decision April 19, 2021 Wetland Permit Issued (if needed)'."
    },
    {
        "candidate_id": "e9707f0dea401aa474bd",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'and conditions The term of the McDowell Allotment grazing permit would run from March 1 2021'."
    },
    {
        "candidate_id": "87385f0074309fd72bd7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e422ad9e773a873c94cc",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'EIS and request for comments in the Federal Register The comment period for this notice ended'."
    },
    {
        "candidate_id": "e88622738c58a8b54691",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Scarpinm DATE: 03 /23 / 2010 month day year NEPA'."
    },
    {
        "candidate_id": "5eb002f180e98b9e6ac6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6c00dc7c76b4959681ad",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Sandy Napolitano Date: 02 / 07 / 2024 month day'."
    },
    {
        "candidate_id": "cc171a0ba09f011b4dc8",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Patcharin Burke Digitally signed by Patcharin Burke'."
    },
    {
        "candidate_id": "7fb8581d440f50a12fe7",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'December 21 2012 The comment period closed on February 4 2013 During the comment period the'."
    },
    {
        "candidate_id": "1d9857490aea73220fed",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "801e129b9c40a40f32c6",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BENJAMIN MAY Date: 11 / 10 / 2022 month day year'."
    },
    {
        "candidate_id": "dc7a6ab2a0b34907016f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Katharina Daniels Digitally signed by Katharina Daniels'."
    },
    {
        "candidate_id": "768a7c811efa97f12eae",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7298ad3cfbf18b31316c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "aef564030189388cb98f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5210aa1a215fd980249a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "92b5b1c8a839854d2da2",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'submitted no later than December 28th 2009 to allow for meeting planning DOE also anticipates holding'."
    },
    {
        "candidate_id": "965cc973c5f2d133bc57",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Notice of Availability on September 23 2014 in the Federal Register announcing the availability of this'."
    },
    {
        "candidate_id": "18eb247cfb400280f189",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: ZACHARY ROBERTS Date: 04 / 15 / 2022 month day'."
    },
    {
        "candidate_id": "39ffb575971442a663f2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ef02d1c3577fd43656aa",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'of this EIS A public scoping meeting was held on November 16 2010 at the Carrisa'."
    },
    {
        "candidate_id": "d5bea55ce62155e558ea",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'February 13 2015 and March 26 2015 During this time six public scoping open houses were'."
    },
    {
        "candidate_id": "a118c7339d740cd8e20e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "05d355dca2a5d6147cc6",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Timothy Fout Date: 08 / 14 / 2012 month day year'."
    },
    {
        "candidate_id": "453b5c539bbaf163c408",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5a36e4970af0b354f39a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: STEPHEN HENRY Date: 03 / 16 / 2020 month day'."
    },
    {
        "candidate_id": "a6e0e1376d8c8058215c",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Cliff Whyte DATE: 12 /17 /10 month day year NEPA'."
    },
    {
        "candidate_id": "ff6d03c8e7fa421f51bc",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d7f06b729678e2f3ad92",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1457cf8f6504bd696c8b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ac823812e70c8b2a3fd2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cfa666de1b62357d8c87",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "21d7acee3366740ff092",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f359ea8d33fde5eb6bdd",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote '1 Scoping The formal scoping comment period started with publication of the Notice of Intent NOI'."
    },
    {
        "candidate_id": "881d992d0218feda51c0",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'County Public Works submitted an SF-299 application N-100111 in November 2020 to utilize public lands administered'."
    },
    {
        "candidate_id": "84d4478ccf7862d33ece",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Robert Noll Date: 08 / 08 / 2012 month day year'."
    },
    {
        "candidate_id": "9806113218afdae8c9da",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9766ed84ece3deab75b3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c627c9658d2d4c2c7c4a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "db961a9e8a9129756d26",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'meeting in Coos Bay on March 27 2013 The Open House was advertised to the public'."
    },
    {
        "candidate_id": "37c532e4ab3849af11a8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b9fd413dd68ca366498d",
        "label": "neither",
        "notes": "Neither: construction/activity period date, quote 'property The proposed construction start date is June 2018 or as early as application processing and'."
    },
    {
        "candidate_id": "a67171de243bc589de9f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "60a539096276d3b35fc6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d37cfa1821593ae54746",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'information meeting and site visit held on January 25 2009 Letters to request consultation to develop'."
    },
    {
        "candidate_id": "88f79a5d6196d22ff889",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: William Aljoe DATE: 11 /19 /2009 month day year'."
    },
    {
        "candidate_id": "079b1e27bed3baf2ff38",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Draft EIS 6/27/23 Consultation considered complete Kickapoo Tribe in Kansas 10/31/22 Emailed 3/6/23 Confirmed receipt Tribal'."
    },
    {
        "candidate_id": "8e08646d95bbdcf0ed24",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8da829e6ab1215694f2a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "fa698d799f23352ea29d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: RICHARD CHINN Digitally signed by RICHARD CHINN Date'."
    },
    {
        "candidate_id": "2cc3d573266794d2f848",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'shown on the attached Figure labeled as SWLRT Delineation Concurrence and PJD 2/18/2015 Figure I The'."
    },
    {
        "candidate_id": "fe206cc9c61752322ef9",
        "label": "neither",
        "notes": "Neither: consultation date, quote '- 3/16/23 \u2022 Phone - 4/3/23 \u2022 Response received - 4/6/23 \u2022 Consultation is complete. Forest County Potawatomi No Historic'."
    },
    {
        "candidate_id": "6e886b12e26c93512a79",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Kristen Kief DATE: 08 /02 / 2010 month day year'."
    },
    {
        "candidate_id": "10e5d7a9bd6c767bae83",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Robert Gormley DATE: 03 / 11 / 2010 month day'."
    },
    {
        "candidate_id": "148aadba9007fc946c13",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Federal Register on May 22 2014 and an open house and three public meetings were held'."
    },
    {
        "candidate_id": "938437d200819364a14d",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Linn Caleb M. Hiner Field Manager Pinedale Field Office 10/24/2017 Date 3 Administrative'."
    },
    {
        "candidate_id": "59946f8b4f9defbdf86b",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'materials For NEPA a Notice of Intent to prepare an EIS was published in the Federal'."
    },
    {
        "candidate_id": "ff6c13e7918780bef812",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'and provide input The NOI was published on December 17 2012 in the Federal Register to'."
    },
    {
        "candidate_id": "c268b191de7ee233d9f6",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'NOA initiated a 45-day public comment period during which the Service solicited comments regarding the Proposed'."
    },
    {
        "candidate_id": "1b3e75748e43d55dd8fa",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a3a21d2000b9a07bf0cd",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'No ML17088A264 25 April 20 2017 Letter from NRC to Reid Nelson Advisory Council on Historic'."
    },
    {
        "candidate_id": "14b4fd4ec336e87c8cd1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "12cd92e9c0f26f8f1cc4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing permit On October 2 2017 the Lewistown BLM Field Office received completed transfer of'."
    },
    {
        "candidate_id": "435129cbd6c538338c3f",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On October 18 2018 the Town of Quartzsite submitted a right-of-way ROW application for'."
    },
    {
        "candidate_id": "10d9c251e624107861fa",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'February 1 March 17 2013 Scoping Meeting Dates Poulsbo WA February 21 2013 Chimacum WA February'."
    },
    {
        "candidate_id": "7f656daa711fa72dbd66",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Seth Lawson Date: 05 / 18 / 2021 Digitally signed'."
    },
    {
        "candidate_id": "4f57a5c98b587b5451b4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'natural gas pipeline On October 7 2022 ONEOK submitted a SF-299 AND/OR MITIGATION application to the'."
    },
    {
        "candidate_id": "af71cd97514d41a4883e",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Otis Mills DATE: 12 / 08 / 2010 month day year'."
    },
    {
        "candidate_id": "572a674a75e0e12fd49e",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Resource Specialist Date Prepared: 2/20/15 D. Implementation Date The following is a COA for'."
    },
    {
        "candidate_id": "cf9a5bac9fc13ced33ad",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7d4d1a9a4f9fe65523d4",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: PATCHARIN BURKE Date: 12 / 12 / 2019 month day'."
    },
    {
        "candidate_id": "b68005cf413ab8b66c12",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "014c28a0e01d84559175",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'described in the report the scoping period began on September 10 2020 and ended on November'."
    },
    {
        "candidate_id": "af7271eb90bee2b155a0",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'approved the assignment on December 4 2018 2019 On April 29 2019 BOEM received an application'."
    },
    {
        "candidate_id": "29aebe74204964d08fb4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On November 13 2019 Dale Kirham of GO LLC submitted an application for a'."
    },
    {
        "candidate_id": "cb5bba47c65d79c20fd9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Colleen Butcher Date: 08/12/2015 (Digitally signed'."
    },
    {
        "candidate_id": "71821e5495c3d7905409",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'publication started the 90-day comment period that ended November 20 2013 However this comment period was'."
    },
    {
        "candidate_id": "be3d3ccdcd653f0e682e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Archeologist Assistant Field Manager, Minerals and Lands Project Lead Preparer Preparer'."
    },
    {
        "candidate_id": "061f975f2dbf3ab5f2b3",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'DATE: 11 / 19 / 2009 DOE INITIATOR SIGNATURE: Karen Cohen month day year DATE: 01 / 22 / 2010'."
    },
    {
        "candidate_id": "32ac63efa67f22004c3f",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2009 FHWA letter to SHPO Invite to serve as historic consulting party May 18 2011 FHWA'."
    },
    {
        "candidate_id": "23dba7b0fc7eacd94240",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Patcharin Burke Date: 06 / 06 / 2013 month day'."
    },
    {
        "candidate_id": "bd217f0c8008b0984ed1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f5771f23aa0ff68e1b69",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5d2c9b561266a503b2bd",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Burke Date that any scoping meeting was conducted N/A Date that concurrent electronic distribution for review'."
    },
    {
        "candidate_id": "f77d4f30f12ab4b78eb2",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'the BLM and USFS in September 2006 On February 12 2007 LADWP officially submitted a ROW'."
    },
    {
        "candidate_id": "b6a411ea46073f7a9aaa",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'planning effort The scoping period began on March 12 2001 and ended December 31 2001 In'."
    },
    {
        "candidate_id": "202130da06a71597c157",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'period which concluded on February 18 2020 The availability of the Draft EIS was announced in'."
    },
    {
        "candidate_id": "3fd45f99c35bbe2d9440",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'and climate change On May 27 2021 the Commission issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "2a2fd0d2b0d6293d852d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a5540a4919c2fc3ee524",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "045b10127bba10f5221d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: James Briones DATE: 01 / 27 / 2011 month day'."
    },
    {
        "candidate_id": "f364b1a28b9487ef0f88",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Advertisements The BLM published a Federal Register Notice of Intent Notice on December 21 2012 Federal'."
    },
    {
        "candidate_id": "29e9fdad48b44d45faad",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'submitted a revised BA February 24 2016 The Service initiated formal consultation by letter to the'."
    },
    {
        "candidate_id": "0f1d1bd44687980b3126",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f6c7f43e7b020ed4ab00",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'intervene Intervenors Date Filed U S Department of the Interior October 25 2011 National Marine Fisheries'."
    },
    {
        "candidate_id": "80c8c233ec0784e46bb8",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'the Proposed Action On November 25 2016 Anne Ousley of Swiftwater RV Park submitted an SF-2920'."
    },
    {
        "candidate_id": "c32f9930ac52e28b70e7",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'PROPOSED ACTION On April 4 2016 Monte MacConnell of Triple M Land and Cattle LLC submitted'."
    },
    {
        "candidate_id": "bcebc30907023487c493",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Colville and Spokane in September 2008 and collaboration workshops were held in September October and November'."
    },
    {
        "candidate_id": "5a74ba3bf475df420f35",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'informational open houses in February 2012 September 2012 and May 2013 The purpose of the open'."
    },
    {
        "candidate_id": "f97cd2c04f6b266691e7",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'obtained as required Consultation with the U S Fish and Wildlife Service USFWS was initiated on'."
    },
    {
        "candidate_id": "ca854999526a9084a7fc",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0efb06e1c3d5cfb5ef39",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'In Reply Refer To On April 22 2014 the request of the Bureau of Land Management'."
    },
    {
        "candidate_id": "dcf9323535c9e29d1fba",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a40728323ba94c7a601e",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: RONALD TATOM Date: 07/02/2019 month day year'."
    },
    {
        "candidate_id": "dc09607e3f91495230b0",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JOSEPH STOFFA Date: 09/05/2017 NEPA Compliance'."
    },
    {
        "candidate_id": "7cfcceaf83a31edbef03",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Glades Reservoir DEIS October 30 2015 Draft Environmental Impact Statement U S Army Corps of Engineers'."
    },
    {
        "candidate_id": "89c4684f54d91d4bfd77",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'No ML102990090 15 October 26 2010 Letter to Taino Tribal Council of Jatibonicu NJ US Taino'."
    },
    {
        "candidate_id": "d2be4417b34e751ecbeb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0fee4678172a5de874f5",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'environmental review process On August 24 2016 the FERC issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "ed99d9e64831a3477f04",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b86fc5a3b3f5020ac92b",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Solar and Wind Energy Applications Pre-Application and 13 Screening dated February 7 2011 establishes process for'."
    },
    {
        "candidate_id": "e376d40dd23b83c50aad",
        "label": "initiation",
        "notes": "Initiation: FERC/application notice, quote 'review of the Project August 5 2022 FERC resends Notice of Application and Establishing Intervention Deadline'."
    },
    {
        "candidate_id": "db49979549d6478f5494",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'BEGINNING BACKGROUND On September 6 2019 the BLM Tres Rios Field Office received a Special Recreation'."
    },
    {
        "candidate_id": "f2cdc70551d63e0388df",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '16 LP 06/09/2016 CSB 6/8/2016 To be filled out during scoping meeting and for Admin Record'."
    },
    {
        "candidate_id": "6966fe3260865aaa117c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cdc07ed86d427b46a487",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c5c4c8ff2922863c319b",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'to announce a 15-day public comment period on the Draft EA A Notice of Availability was'."
    },
    {
        "candidate_id": "79be0f94c46f69464068",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a020fb24e87fb981ee5f",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'January 2021 1 6 1 4 Scoping Comments The 30-day scoping comment period began on August'."
    },
    {
        "candidate_id": "fdef0b83926b02626dd1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "951818653756da075d8d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ad63ea0b865e99a8f429",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5d36107de965f80e64ef",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'ca.blm.gov/motherlode Date Prepared: 11/1/2016 Project: TigerCreekRoadside_Topo -3000- 3200 HIDDEN'."
    },
    {
        "candidate_id": "d51fbcbbe0fda2fed8bb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6c271f0790ee9c78c54c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "118152d85ca6fd8e31ba",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'in June 2022 and the NOI issued in July 2022 In addition we conducted a virtual'."
    },
    {
        "candidate_id": "1874fa9a838a2fa2cfc8",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'affected by the project Consultation with tribes and the Idaho State Historic Preservation Office SHPO was'."
    },
    {
        "candidate_id": "e1f7c08a98545c2db689",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Response IR 217 received February 13 2024 BMOP Blue Marlin Offshore Port LLC 2024c Blue Marlin'."
    },
    {
        "candidate_id": "0f1c984d7e600708f6d7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "173faceece02bade45ab",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Effect Archaeology On August 19 2009 ODOT provided to SHPO a Finding of No Historic Properties'."
    },
    {
        "candidate_id": "8b46208552d7639f527d",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On March 28 2017 Mark Finley on behalf of Finley-Holiday Film Corp filed film'."
    },
    {
        "candidate_id": "67f307fdfaad859ffe2d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c19b1616a8a155af2c68",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Request to engage with the tribe and follow-up from 9/30/2021 letter sent to Tribal Council Email'."
    },
    {
        "candidate_id": "6ea1f092c139e87819bf",
        "label": "initiation",
        "notes": "Initiation: FERC/application notice, quote 'review of the Project On April 6 2017 the FERC issued a Notice of Application announcing'."
    },
    {
        "candidate_id": "1aff09515e147ba382c3",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'document in the EIS 6 2 5 Scoping Meetings The following three scoping meetings were held'."
    },
    {
        "candidate_id": "aa125bce253482b6f341",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'Agency and Other Entity Date Filed California DFW July 21 2014 FWS July 22 2014 Conservation'."
    },
    {
        "candidate_id": "39151e02707b40ad8c42",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'permit 2506520 On May 8 2020 the Lewistown BLM Field Office received completed transfer of grazing'."
    },
    {
        "candidate_id": "e0c1bf35495552783af9",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'of the FEIS formal consultation was initiated on January 23 2007 The response and Biological Opinion'."
    },
    {
        "candidate_id": "c673f4d9a2b318a6ad21",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'review ML14157A354 July 8 2014 D Wrona NRC to R Sparkman Chief Shawnee Tribe Request for'."
    },
    {
        "candidate_id": "f7d0e913558eb82f4406",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8db01e44b020d48d421f",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Furthermore the Bishop Pauite Tribe raised a concern that the Cultural Resources Technical Report did not'."
    },
    {
        "candidate_id": "124b1beed648e11c6a05",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'meeting was held on October 19 1999 in Refer to page viii for a list of'."
    },
    {
        "candidate_id": "dc1059c25ecdef6e3b34",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c109fe1f4491c0fb3ef7",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Accession No ML103010069 November 4 2010 Transcript Davis-Besse License Renewal Public Meeting Afternoon Session pages 1'."
    },
    {
        "candidate_id": "2df98d81fcc6179b789e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c21c274b58b71759b11c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1ab3b0ff44aa27513ae3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cafcd799514dd5fe7d88",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8acc9925cab80d8623a8",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'the Texas Register on August 11 2006 and in the Federal Register on August 18 2006'."
    },
    {
        "candidate_id": "ad80a3588cc214d484b7",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Tract and that CMC had filed a lease application for the Maysdorf II LBA Tract The'."
    },
    {
        "candidate_id": "c905ead25c86122aa45d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ce5293ef1c2b116c731d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Date: 9 /10/2013 month day year NEPA Compliance'."
    },
    {
        "candidate_id": "fe03316fde155e47330a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "35752dde6a8e65314b66",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'public comment during a scoping period to help identify issues and concerns that should be considered'."
    },
    {
        "candidate_id": "1beabaada0e89efa2d48",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'www blm gov/california Date Prepared 4/3/2018 Project Hazard Tree Removal xd N W E S Bull'."
    },
    {
        "candidate_id": "8d2d80fb5dd0ecf36690",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: GARY COVATCH Digitally signed by GARY COVATCH Date'."
    },
    {
        "candidate_id": "6512e9738c61d3685643",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6fb867a8bd2b5f5cb818",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'TGL application. The NOI was published July 3, 2019, and includ...'."
    },
    {
        "candidate_id": "6f47a1285a40d4eed644",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d0af4ce03a54eede9504",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Kalamazoo County The public comment period ran from May 31 through August 17 2016 during which'."
    },
    {
        "candidate_id": "e7249513022098af839d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "708ffc52c275dea2941c",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'with the Commission On March 13 2006 the FERC issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "b2e03bc984aee196ec92",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4363669b0c85b7415389",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: VITO CEDRO Date: 09/08/2017 month day year NEPA'."
    },
    {
        "candidate_id": "9a0fd7b5be1e8a5fe930",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e235ea9b525d1fab9fe9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: BRIAN MOLLOHAN Date: 05 / 09 / 2014 month day'."
    },
    {
        "candidate_id": "bc86a63eeb195128f7ba",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'EIS website when the NOI was published on July 23 2021 and was available for the'."
    },
    {
        "candidate_id": "b9d72ddc393f238d52e5",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Hawksbill Sea Turtles July 8 2015 B-109 USFWS Letter to USAF Concurring with the Not Likely'."
    },
    {
        "candidate_id": "0ca1f64205d724e099fd",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On November 19 2014 the Bureau of Land Management BLM received a ROW renewal'."
    },
    {
        "candidate_id": "c4b18cf6f5217b7f3775",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Richard Baker Digitally signed by Richard Baker Date'."
    },
    {
        "candidate_id": "81b2f4e8141baad7fcc9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Date: 07 / 24 / 2013 month day year NEPA Compliance'."
    },
    {
        "candidate_id": "9929f2a729261cda0a73",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Andrea McNemar Date: 07 / 24 / 2012 month day'."
    },
    {
        "candidate_id": "fb357e10be9647eef2da",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "33e6c4d0a870a5a8770b",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Response IR 167 received October 20 2022 BMOP Blue Marlin Offshore Port LLC 2024 Blue Marlin'."
    },
    {
        "candidate_id": "8930a1b6a5f888debbe0",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Karen Kluger Date: 07 / 30 / 2012 month day year'."
    },
    {
        "candidate_id": "4e38a6900e7a655377dc",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'project is in progress a notice of intent to prepare an environmental impact statement was published'."
    },
    {
        "candidate_id": "a984f17c2f6b6bf4839c",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'media announced the scoping comment period and virtual public scoping meeting which was held on March'."
    },
    {
        "candidate_id": "0ca073a129be848e8e18",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '2023 BACK GROUND On March 30 2008 PacifiCorp d/b/a Rocky Mountain Power submitted an application SF-299'."
    },
    {
        "candidate_id": "f978aa5cf761f37b6b80",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "21aa98b1af1d003a9410",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "740aacb87ae634d48768",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'under this SRP This consultation was conducted in-person or project updates were sent per the tribes'."
    },
    {
        "candidate_id": "8d4a7a543c5bf82aa39f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Doug D. Linn Acting Field Manager Pinedale Field Office 01/09/2020 Date 4 Administrative'."
    },
    {
        "candidate_id": "05d84b8e83dba1902b65",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote '2011 Page 75554 The public comment period ended on March 1 2012 Public meetings were held'."
    },
    {
        "candidate_id": "f1c0dc6df4c35e2251c4",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'December 23 2013 a formal public comment period commenced This public comment period began on December'."
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
