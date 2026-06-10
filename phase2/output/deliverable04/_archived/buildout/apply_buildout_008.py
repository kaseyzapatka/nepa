import pandas as pd


LABELS = [
    {
        "candidate_id": "d19c8a01a3ae8c5174bf",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'ESTABLISHED BY STATUTE DOI-BLM-WY-D090-2019-0016-CX A. Background Office: Bureau of Land Management'."
    },
    {
        "candidate_id": "6f403e6113a691cc52ff",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'TRC Well Abandonment DOI-BLM-CA-C060-2024-0031-EA Introduction TRC Cypress Group, the Operator'."
    },
    {
        "candidate_id": "6546a94ab9f5cc0e9f9e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion (CX) # DOI-BLM-NV-C010-2016-0011-CX, it is my decision to implement the Proposed'."
    },
    {
        "candidate_id": "40d5446e2293b1111539",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established By Statute DOI-BLM-UT-C030-2022-0013-CX March 2022 Rim Tours Special Recreation Permit'."
    },
    {
        "candidate_id": "6fb84efe3a877e97de24",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '37533, Land Use Permit DOI-BLM-ID-B011-2020-0019-CX previously recorded populations and thus do not'."
    },
    {
        "candidate_id": "4621ff2f9f2cbb67fe78",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2021-0081-CX promote the introduction, growth, or expansion'."
    },
    {
        "candidate_id": "784e612c0ce787662b4d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'exclusion review (CX; DOI-BLM-ORWA-R000-2019-0003-CX). Based on my review of the attached CX, I have'."
    },
    {
        "candidate_id": "66b7cfa06e7919a65536",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '2 5 Wanrack Carlin DOI-BLM-NV-E020-2018-0020-CX 3. Maps, Stipulations and Documentation 3.1.'."
    },
    {
        "candidate_id": "550e8c86502742ba13a8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'road clearing. CX# DOI-BLM-CA-N020-2017-0016-CX Cove Fire ES&R Plan Page 2 b. May include temporary'."
    },
    {
        "candidate_id": "005108eb7e3b7cbf92da",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'conditions arise which DOI-BLM-CO-N050-2021-00013-CX Decision Record 2 result in the approved terms'."
    },
    {
        "candidate_id": "a1ca0e905eefd01bbce5",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Film Permit UTU-95284 DOI-BLM-UT-Y010-2021-0002-CX Nocturne The following elements are not present'."
    },
    {
        "candidate_id": "7ff9436a9a8178fcd7ec",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S030-2024-0005-CX Southern Nevada District Office NV Energy NVN'."
    },
    {
        "candidate_id": "8e099650b447a9069b52",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'EXCLUSION NUMBER: DOI-BLM-NM-A010-2023-0003-CX DECISION It is my decision to implement the Proposed'."
    },
    {
        "candidate_id": "3f3543c663698a5bcbf6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Wild Horse and Burro DOI-BLM-AZ-C030-2017-0033-CX ALERT Precipitation/Stream Stage Station 4 of'."
    },
    {
        "candidate_id": "1d114e0803afcc6684ff",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'with Pipeline Loop DOI-BLM-CO-N050-2018-0110-CX Identifying Information Project Title: Enterprise'."
    },
    {
        "candidate_id": "25666fb2fb68080b6fd1",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Wyoming Women Anglers DOI-BLM-WY-D030-2018-0180-CX Decision: I have reviewed the attached Categorical'."
    },
    {
        "candidate_id": "f5c24919c240333554fd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'future generations. DOI-BLM-WY-D010-2018-0132-CX March 9, 2018'."
    },
    {
        "candidate_id": "e046c17487f08ab56909",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Foster Field Manager DOI-BLM-WY-D040-2022-0014-CX Page 1 of 1'."
    },
    {
        "candidate_id": "bad7279cf324d27ef7b1",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DOI-BLM-ID-B010-2023-0030-EA Decision Record 8 reasonable to prevent wildfires'."
    },
    {
        "candidate_id": "00d2e002894ee9f278aa",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Thinning and Salvage DOI-BLM-ORWA-N020-2018-0011-CX 5 \u2022 Trees will be felled to lead, generally away'."
    },
    {
        "candidate_id": "82a172fc1d3e4a08e946",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'authority over the scene. DOI-BLM-ID-B010-2022-0032-CX 11 Cultural 44. Pursuant to 43 CFR 10, the Holder'."
    },
    {
        "candidate_id": "dbb7a7d52a3b29a404af",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'WYW121481 CX Number: DOI-BLM-WY-D040-2020-0111-CX Right-of-Way Applicant/Holder: QWEST Proposed'."
    },
    {
        "candidate_id": "230407a15937fc50aa28",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Trail Signage NEPA # DOI-BLM-AZ-C010-2020-0015-CX A. Background The project proposal is to install'."
    },
    {
        "candidate_id": "17d07b26ee6abdb96395",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CO-F020-2020-0001 CX isita 736 31 36 31 1 6 31 450 2200 41 Baditos'."
    },
    {
        "candidate_id": "8eb4d0a29125e75fbef8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'interdisciplinary team prepared CX-DOI-BLM-CO-N040-2016-0040 for the proposed permit. My proposed decision is'."
    },
    {
        "candidate_id": "1f9cb8a31d7b803a6750",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Road Right-of-Way DOI-BLM-WY-P070-2024-0051-CX Date & Time: Wed, Aug 31, 2022, 08:26:04 MDT'."
    },
    {
        "candidate_id": "cf6f227916be5974f17c",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'OAHP RB.LM.R441) DOI-BLM-CO-N05-2015-0094-CX thru -0111-CX 8 O'Neil, Brian 1995 Cultural Resources'."
    },
    {
        "candidate_id": "748ac26b6670449ff96d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'NEPA Document Number: DOI-BLM-WY-R050-2020-00003-CX Description of Proposed Action: The Lander Field'."
    },
    {
        "candidate_id": "1f184de4af9d02693faf",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'final (in the event no DOI-BLM-CO-S010-2021-0003-CX Proposed Grazing Decision - Marcum 3 protests'."
    },
    {
        "candidate_id": "402b61a901040de383e6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'covered by this CX. DOI-BLM-ORWA-W020-2016-0012-CX, Colockum Creek Grazing Lease Renewal 6 Badger'."
    },
    {
        "candidate_id": "91fbbf0b05b48bca19dc",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'scale) CX Number: DOI-BLM-OR-S050-2012-0001-CX Project: South Fork Lobster Creek Culvert Replacement'."
    },
    {
        "candidate_id": "3266b47c3431deadad9f",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DECISION NEPA Log Number: DOI-BLM-NM-L000-2017-00112-CX Lease/Serial/Case File No.: A. Background Title'."
    },
    {
        "candidate_id": "6121fa825a8309a56e10",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion - DOI-BLM-AK-A020-2020-0011-CX Permit Stipulations DOI-BLM-AK-A020-2019-0030-CX 2'."
    },
    {
        "candidate_id": "29570854d2b6ce5571a1",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Revised) p. 2 CX#: DOI-BLM-ORWA-S060-2016-0006-CX Project: JTF, Inc. O&C Road Use Permit CX and'."
    },
    {
        "candidate_id": "76675b9308bf8e02d118",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'ESTABLISHED BY STATUTE DOI-BLM-WY-D090-2024-0007-CX A. Background BLM Office: Bureau of Land Management'."
    },
    {
        "candidate_id": "b4cb2dcd79f8ed6bf1c4",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'where construction DOI-BLM-ID-B020-2021-0010-CX 8 equipment and vehicles are not allowed will'."
    },
    {
        "candidate_id": "ddb8056474a1c87fab4f",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established By Statute DOI-BLM-UT-C030-2020-0031-CX February 2020 Leavitt-Blue Sky Road ROW Assignment'."
    },
    {
        "candidate_id": "280de75d419b316677e0",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'drill Project, CX# DOI-BLM-NV-C020-2023-0021-CX is approved for implementation. This decision'."
    },
    {
        "candidate_id": "2fb8a2c95ef12a363d74",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '(E)]. Rationale: DOI-BLM-AK-A020-2023-0018-CX 2 This action is not controversial, nor does'."
    },
    {
        "candidate_id": "1101b05761116a1dfcf9",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DOI-BLM-ID-B030-2020-0012-EA 9 3.1.3.2 Past, Present and Reasonably Foreseeable'."
    },
    {
        "candidate_id": "de4464e24667cc2400ad",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '2 Decision Record DOI-BLM-ID-I020-2023-0001-CX **Page 3** Standards for Obtaining a Stay Except'."
    },
    {
        "candidate_id": "34645e5d7e26265a6259",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Guide A. Background DOI-BLM-NV-B000-2016-0003 BLM Office: Battle Mountain District LLNVB0000 Lease'."
    },
    {
        "candidate_id": "736bcfb8ca60d094e139",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Valley Field Office DOI-BLM-CO-F030-2023-0002 CX ROW Grant \u2013 Road Access- Coolbroth Cabin INTERDISCIPLINARY'."
    },
    {
        "candidate_id": "3e496cd97b94b103c6c8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'WYW96310 CX Number: DOI-BLM-WY-D040-2016-0121CX Right-of-Way Applicant/Holder: QEPM Gathering'."
    },
    {
        "candidate_id": "9dc954aaf678890441b7",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'and approximately DOI-BLM-CO-N05-2018-0088-0095-CX 2 200,000 bbls of produced/recycled water'."
    },
    {
        "candidate_id": "51dd2532b75f4cd7b992",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '2019 Attachment 1 DOI-BLM-CO-N010-2019-0027-CX South Cole Gulch Allotment #04086 6630- 6800'."
    },
    {
        "candidate_id": "ad353e842d9ab5c3aa3e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'environmental assessments (see DOI-BLM-OR-L050-2014-0021-EA and OR 14-93-09). These existing weed/invasive'."
    },
    {
        "candidate_id": "a184d93ee1aa3c2e54c5",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'WYW133162 CX Number: DOI-BLM-WY-D040-2018-0158-CX Right-of-Way Applicant/Holder: Wexpro II Company'."
    },
    {
        "candidate_id": "713704cb6d2088a71a1e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Force Base Assignment DOI-BLM-AZ-G020-2020-0016-CX U.S. Department of the Interior Bureau of Land'."
    },
    {
        "candidate_id": "2d6f35df2d090277fd0e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Office NEPA Number: DOI-BLM-UT-C030-2023-0026CX Lease/Serial/Case File No: UTU-96266, UTU -96276'."
    },
    {
        "candidate_id": "806cc80e201891891cf1",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'approved the TSP in July 2020 The Recommended Plan in this Final EIS is the Levee'."
    },
    {
        "candidate_id": "b46f7727b42e85735ce7",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'NEPA Project Number DOI-BLM-MT-B010-2019-0018-CX Proposed Action Title Brandon-Crean Right of'."
    },
    {
        "candidate_id": "2d6063f4962da3e1f7a1",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'average elevation of DOI-BLM-CO-N050-2020-0030-CX 7 5,400 ft. If an alternate date of seeding is'."
    },
    {
        "candidate_id": "dac27c3077fc59758a9e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Pipelines NEPA No. DOI-BLM-NM-P020-2020-0595-CX Project Name: PAPAS FRITAS 27-22 FED COM SWD'."
    },
    {
        "candidate_id": "d965828f89cf7a7cc442",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Officer 9-26-16 Date DOI-BLM-UT-G020-2016-0048-CX, Decision Record Page 3 NATIONAL SYSTEM OF PUBLIC'."
    },
    {
        "candidate_id": "00e0acc8b9f54602f277",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-ID-B010-2019-0016-CX 1 facilities for private mobile radio service'."
    },
    {
        "candidate_id": "8b999a6e2d9f4373c1b2",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'interdisciplinary team prepared CX-DOI-BLM-CO-G020-2021-0022 for the proposed permit. My proposed decision is'."
    },
    {
        "candidate_id": "0e16a8520095db754dff",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CA-N050-2019-0003-CX PAGE 8 [Map - Title: Charter Communications Right'."
    },
    {
        "candidate_id": "1b49cb63938a6e2c16f6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'ROW- Sacramento Wash DOI-BLM-AZ-C030-2014-0015-CX 7 disposed of, or stored within the lease area'."
    },
    {
        "candidate_id": "1deada33486b162f14ed",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Proposed Activity NEPA #: DOI-BLM-CA-C090-2018-0036-CX Lease/Serial/Case File No: CACA 058044 Description'."
    },
    {
        "candidate_id": "125250bdf9a63fdc1f11",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'acts under\u00a74770.1 may DOI-BLM-AZ-C010-2020-0013-CX Attachment 3 Page 8 of 12 result in suspension'."
    },
    {
        "candidate_id": "557b9c0528a6f6b8ea28",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'or water samples. DOI-BLM-AZ-C030-2018-0012-CX AZA 37386 AZ Game & Fish Guzzler Monitor 2 of'."
    },
    {
        "candidate_id": "451e34377e9a229b1219",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'CATEGORICAL EXCLUSION DOI-BLM-CO-S050-2022-0005-CX IDENTIFYING INFORMATION PROJECT NAME: COC-60133'."
    },
    {
        "candidate_id": "7976180e30a8515fdacf",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'ESTABLISHED BY STATUTE DOI-BLM-WY-D090-2017-0018-CX A. Background Bureau of Land Management Kemmerer'."
    },
    {
        "candidate_id": "f9c01fe029e74dc78b18",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Rhyne Appendix B DOI-BLM-CO-N010-2020-0033 CX Standard Terms and Conditions 1) Grazing permit'."
    },
    {
        "candidate_id": "1d7ac5918eb41d3c0c80",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2023-0047-CX 9.2. If construction activities require that'."
    },
    {
        "candidate_id": "64fe023d14c700c59ab5",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2021-0044-CX effects. 6. Have a direct relationship to other'."
    },
    {
        "candidate_id": "11201bfcd232c6a08e03",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'of Land Management DOI-BLM-MT-C020-2016-0098-CX May 19, 2016 RIEGER ALLOTMENT GRAZING TRANSFER'."
    },
    {
        "candidate_id": "15cf591db521bf847d1f",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CA-N020-2021-0014-CX PAGE 3 2021 SCREENING FOR CATEGORICAL EXCLUSIONS'."
    },
    {
        "candidate_id": "db4360833a354c6d63db",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'CIRCUMSTANCES CHECKLIST DOI-BLM-ORWA-N050-2017-0008-CX O. & C. Logging Road Right-Of-Way Permit; EWEB'."
    },
    {
        "candidate_id": "c95f038a8b6ba49349e6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'appraiser Decision Record DOI-BLM-ID-B011-2022-0002-CX 7 Exhibit C IDI-37851 Page 2 of 11 STIPULATIONS'."
    },
    {
        "candidate_id": "949222b08517b8c59826",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'RECORD for the NEPA No. DOI-BLM-NM-P020-2023-0200-CX EOG Resources Incorporated Shiprock 5 Fed Com'."
    },
    {
        "candidate_id": "b9b164e9b8530dbbe00d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'WYW-74570 Renewal DOI-BLM-WY-D030-2017-0091-CX Decision I have reviewed the attached Categorical'."
    },
    {
        "candidate_id": "2cf1a75ec59eb2dee872",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'advertising. XIII. CAMPSITES DOI-BLM-CO-S010-2020-0016-CX 13 **Page 14** A. Camps may be set up for no'."
    },
    {
        "candidate_id": "0b3d50e4db65d31cd56d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'neither establishes a DOI-BLM-ID-B010-2020-0017-CX 2 precedent for future actions nor represents'."
    },
    {
        "candidate_id": "3c9943418c02f1917e4b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Permit H-1790-1 CX: DOI-BLM-ORWA-S060-2016-0008-CX (March 2011 Revised) Page 1 of 7 once that day'."
    },
    {
        "candidate_id": "8b2469f104175998ddb3",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Trail Race Series DOI-BLM-CO-N010-2019-0019-CX Identifying Information Project Title: Dinosaur'."
    },
    {
        "candidate_id": "ba69988f2e21338e8e0d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'the project area. DOI-BLM-CO-N02-2017-0038-CX Decision Record 4 Categorical Exclusion Review'."
    },
    {
        "candidate_id": "0be7d19b876a09aeaca6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Electric NEPA No. DOI-BLM-NM-P020-2022-1102-CX Project Name: Hawk 35 Fed 506H Power Line Reroute'."
    },
    {
        "candidate_id": "cadba58ad34c17cbc765",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'A NEPA Log Number: DOI-BLM-ORWA-W020-2017-0010-CX Proposed Action Title: First Creek Fire Osprey'."
    },
    {
        "candidate_id": "9e3287a95acafa92fe73",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2021-0047-CX Mitigation Measures and Stipulations The attached'."
    },
    {
        "candidate_id": "16056405181524761e10",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Survey Monuments DOI-BLM-NV-S010-2018-0059 6.1. Holder shall protect all survey monuments found'."
    },
    {
        "candidate_id": "a8001390b76c1ff24a76",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Phone: (707) 468-4000 DOI-BLM-CA-C050-2020-0017-CX Land Trust of Napa County Request for Right-of'."
    },
    {
        "candidate_id": "9bbd0901949f683c471f",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'confirmation within 48 DOI-BLM-NV-B020-2021-0050-CX N-100493 AT&T 7 hours. The grant Holder shall'."
    },
    {
        "candidate_id": "d7f77a6ea9781db2f6c4",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Inventory and Monitoring DOI-BLM-ID-B011-2018-0015-CX March 23, 2018 I have reviewed the plan conformance'."
    },
    {
        "candidate_id": "b72fe97f4ff143a57443",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'RECREATION PERMITS DOI-BLM-UT-G020-2023-0011-CX July 2023 Location: BLM Managed Lands in Emery'."
    },
    {
        "candidate_id": "aa8e33824765ee8b5748",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Project Summary The CX3, DOI-BLM-WY-P070-2019-0048-CX, includes the project description, including'."
    },
    {
        "candidate_id": "bcbf018d69749b65471b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established by Statute DOI-BLM-WY-D040-2018-0102-CX A. Background BLM Office: Rock Springs Field'."
    },
    {
        "candidate_id": "1b64930d6d45bcd920b0",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'knowingly disturbing DOI-BLM-CO-N020-2021-0029-CX_Appendix 3 historic or archaeological sites,'."
    },
    {
        "candidate_id": "712e5e75976804eeccdb",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Act of 2005 NEPA No. DOI-BLM-NM-P020-2023-0856-CX Project Name: Turkey Track to Auto State DCP'."
    },
    {
        "candidate_id": "ac323b56acdf4d398d97",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'NEPA Document Number: DOI-BLM-WY-R020-2020-0003-CX Description of Proposed Action: Two permanent'."
    },
    {
        "candidate_id": "44cd82f60694cb172f25",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'apply. 1 NEPA No. DOI-BLM-ID-T020-2014-0031-CX Consideration of Extraordinary Circumstances'."
    },
    {
        "candidate_id": "39e9ea26552206b110f5",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'calendar year that covers DOI-BLM-AZ-C030-2018-0004-CX AZA 30318 UNS Electric Inc., Renewal 8 of 9 the'."
    },
    {
        "candidate_id": "7d665f63742e9f957c57",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion (CX) # DOI-BLM-NV-C020-2022-0011-CX, it is my decision to implement the Valley Off'."
    },
    {
        "candidate_id": "145dd09cda1bf1627092",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-WY-D090-2019-0080-CX'."
    },
    {
        "candidate_id": "8e3a00637725a0907d5a",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'ROD/RMPA, page 2-39) DOI-BLM-CO-N050-2021-0028-CX 1 Proposed Action Background TEP Rocky Mountain'."
    },
    {
        "candidate_id": "dabe27d76d53250995da",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CO-F020-2020-0010 CX ALLOTMENT/ NUMBER KIND GRAZING GRAZING % TYPE'."
    },
    {
        "candidate_id": "bafa4609cf95d993b549",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2020-0051-CX safety, and to avoid disruption or corrosion effects'."
    },
    {
        "candidate_id": "71da642874adc285563a",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established By Statute DOI-BLM-UT-Y010-2016-0193-CX May 2016 Film Permit UTU-91720 Location: White'."
    },
    {
        "candidate_id": "99b1408af51bb1b0a1ac",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '14 DOI-BLM-CA-C090-2021-0014-EA MA-REC-OHV-015: Install one (new) gate at the'."
    },
    {
        "candidate_id": "d36064b622e629a5df20",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Assessment Number: DOI-BLM-AK-R000-2018-0007-EA Applicant: North Slope Borough P.O. Box 69 Utqiagvik'."
    },
    {
        "candidate_id": "6b982d3d216ef63b4edf",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'rwalthour@blm.gov DOI-BLM-AK-F030-2017-0030-CX F-97345'."
    },
    {
        "candidate_id": "d6fd746c83742e39f28d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion Number: DOI-BLM-OR-S000-2014-0001-CX Date: 9/21/2015 Proposed Action Title/Type: Special'."
    },
    {
        "candidate_id": "0fa595b8c7767cc19a74",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion CE Number: DOI-BLM-ID-I030-2013-0005-CE Title of Action: Challis Field Office Abandoned'."
    },
    {
        "candidate_id": "d1e44203f1a9d448eaa4",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'and Rationale for DOI-BLM-NV-S010-2023-0089-CX Preliminary Geotechnical Investigation for the'."
    },
    {
        "candidate_id": "28867e3373f7f87c14aa",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CO-SO50-2017-0028 CX August 2017 AJ Mueller Photography Film Permit'."
    },
    {
        "candidate_id": "c7b551bf50e5f67804e3",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'ENVIRONMENTAL ASSESSMENT DOI-BLM-CA-D060-2020-0040-EA earthen mound. The roads include segments of'."
    },
    {
        "candidate_id": "b30fa3921543cb152e62",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-ORWA-C030-2021-0005-CX Decision: It is my decision to allow fireline'."
    },
    {
        "candidate_id": "9764b2057a6d20e20d53",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '84 84 167 33 29 1 DOI-BLM-CO-N040-2020-0057-CX | BLM - Colorado River Valley Field Office Terms'."
    },
    {
        "candidate_id": "a9e6bac785a4e5a1d6d4",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Any excavations into DOI-BLM-CO-N05-2015-0068-CX thru 0081 CX 5 the underlying sedimentary rock'."
    },
    {
        "candidate_id": "400dee10b5891a1cdb2c",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Line Rights-of-Way DOI-BLM-ORWA-B060-2024-0002-CX Authority: Rights-of-Way 2800 Authority for rights'."
    },
    {
        "candidate_id": "a2db992500a7ec0aae6b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'consulting with the holder. DOI-BLM-ID-B010-2022-0015-CX 8 Access 32. New access roads or cross-country'."
    },
    {
        "candidate_id": "8ec052f9fd3e842b73ad",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion CX#: DOI-BLM-NV-W010-2022-0007-CX Date: 12/16/2021 Lease / Case File / Serial'."
    },
    {
        "candidate_id": "16195bdeeb9526294f3f",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DECISION DOI-BLM-MT-B010-2020-0027-CX Sawpit Allotment Grazing Lease Transfer Decision'."
    },
    {
        "candidate_id": "af92e13621596f2f9756",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established By Statute DOI-BLM-UT-Y010-2017-0184-CX March 2017 Film Permit UTU-92377, Finley-Holiday'."
    },
    {
        "candidate_id": "cddf47028bc32bc40c32",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'may be applicable. DOI-BLM-UT-Y020-2019-0030-CX 8 THIS PERMIT: 1. CONVEYS NO RIGHT, TITLE OR'."
    },
    {
        "candidate_id": "42585d6bbc7128551f45",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Springs Field Office DOI-BLM-WY-D040-2023-0006-CX CATEGORICAL EXCLUSION A. Background Lease/Serial'."
    },
    {
        "candidate_id": "0b0c58a80c9eb8d3aaca",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Existing Fiber Optic Line DOI-BLM-CO-N05-2017-0001-CX Identifying Information Project Title: ROW For'."
    },
    {
        "candidate_id": "9303d3bcd28c8326dee8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-WY-D040-2021-0079-CX AUGUST 2021 BUREAU OF LAND MANAGEMENT Rock Springs'."
    },
    {
        "candidate_id": "b158334d0d0f07d5b233",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DOI-BLM-CA-C090-2019-0024-EA Page 17 Potential indirect adverse effects that'."
    },
    {
        "candidate_id": "29caaf3702b26e47c462",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S020-2020-0006-CX -------------Page intentionally left blank--'."
    },
    {
        "candidate_id": "92a1f0e813f1ceb5080d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Creek Fire Salvage DOI-BLM-ORWA-N020-2019-0002-CX 10 Will the Proposed Action documented in this'."
    },
    {
        "candidate_id": "5d2248bcad558db02f08",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2022-0054-CX Categorical Exclusion Documentation I. Background'."
    },
    {
        "candidate_id": "1313c6e0d280969cf3b0",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'EXCLUSION NUMBER: DOI-BLM-CO-F020-2021-0017-CX CASEFILE/PROJECT NUMBER (optional): COC-80096'."
    },
    {
        "candidate_id": "a0c69ac6571e12716d8e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Grazing Lease Renewal DOI-BLM-WY-P070-2019-0133-CX EXTRAORDINARY CIRCUMSTANCES: 1. Have significant'."
    },
    {
        "candidate_id": "29606c4abafd33935462",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Field Office NEPA No.: DOI-BLM-AZ-A010-2015-0002-CX Case File No.: AZA 036624, UTU 090985 Proposed'."
    },
    {
        "candidate_id": "a8c60834dafd67db69dd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Springs Field Office DOI-BLM-WY-D040-2023-0045-CX CATEGORICAL EXCLUSION A. Background Lease/Serial'."
    },
    {
        "candidate_id": "e62776467771ecee496b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'area are proposed. DOI-BLM-CO-N020-2016-0042-CX Decision Record 1 Exploration activities would'."
    },
    {
        "candidate_id": "67ab82b5d91d9ff49628",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Parachute Testing DOI-BLM-ID-B010-2022-0003-CX 5 General Terms 1. The Holder will indemnify'."
    },
    {
        "candidate_id": "b57e5e71ff944ec14120",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion (CX) number DOI-BLM-AZ-P010-2023-0022-CX. I find this action conforms to BLM CX (E.11'."
    },
    {
        "candidate_id": "ed23787488c7036ad6df",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CA-D060-2021-00011-CX AM Wind Repower PREPARING OFFICE U.S. Department'."
    },
    {
        "candidate_id": "12b810d6ee6ae37f5681",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Field Office (CFO) DOI-BLM-NM-P020-2020-0875-CX IT4RM EA-2020-0000 EOG Resources, Inc. Serial'."
    },
    {
        "candidate_id": "e1d537cf3bb76cfb2876",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion Documentation DOI-BLM-NV-S010-2022-0095-CX Kyocera Laser Product Demonstration Event Special'."
    },
    {
        "candidate_id": "6abd1947b755004db05a",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion \u2013 Appendix A DOI-BLM-NV-S0X0-2021-0004-CX environmental effects. 6. Have a direct relationship'."
    },
    {
        "candidate_id": "95dad7bd3fd3bdf2e40a",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'forested location DOI-BLM-OR-S050-2011-0007-CX Project: Special Forest Products Program H-1790'."
    },
    {
        "candidate_id": "aa944e28ff2edeae7544",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '83639. (208) 896-5917. DOI-BLM-ID-B030-2016-0008-CX Divide Top FUP Renewal 6 FREE USE PERMIT STANDARD'."
    },
    {
        "candidate_id": "c226a9ab90e4980a5fb8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Environmental Assessment (DOI-BLM-AZ-G010-2014-0009-EA), and have made a Finding of No Significant Impact'."
    },
    {
        "candidate_id": "6e35696fadf1d1e7768c",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Resource Specialist) DOI-BLM-ID-C020-2021-0014-CX 4 N W E S Salmon River 95 Fiddle Creek fruit'."
    },
    {
        "candidate_id": "18c782e894763951c2d3",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'populations. | [ ] | [x] DOI-BLM-CO-N050-2023-0005-CX 9 Extraordinary Circumstance | YES | NO k) Limit'."
    },
    {
        "candidate_id": "b0f2c5a7e6900e3c397a",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2019-0092-CX project site. Depending on what time in the bird'."
    },
    {
        "candidate_id": "41ec65c89f7c2d59aa60",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'animals on severe DOI-BLM-CO-N050-2019-0072-CX 4 winter range. Exceptions and modifications'."
    },
    {
        "candidate_id": "8ec958ee1431ce121f97",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Authorized Officer. DOI-BLM-ID-B011-2023-0010-CX IDI-35867& IDI-34219 8 19. Following construction'."
    },
    {
        "candidate_id": "71d227cb1295399120cf",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Springs Transfer #DOI-BLM-AZ-G010-2018-00004-CX Date Internal Scoping Initiated: Date Internal'."
    },
    {
        "candidate_id": "9165614342daab2f9c1d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'significant impacts. DOI-BLM-ID-I040-2024-0006-CX A-3 (e) Establish a precedent for future action'."
    },
    {
        "candidate_id": "9c3f1027e720134e1cc8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DOI-BLM-AK-F030-2016-0007-EA F-97202 Public Involvement It was determined'."
    },
    {
        "candidate_id": "c0e3046c382f1ca45acd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '6 DOI-BLM-AZ-C010-2013-0018-EA Whiskey Basin Trail Maintenance FINDING OF NO'."
    },
    {
        "candidate_id": "c4e5db4b630834c78ebd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Decision Documentation DOI-BLM-ES-0030-2019-0001-CX Rolland A Unit Communitization Agreement Date'."
    },
    {
        "candidate_id": "ba0d37e66cf5ff22e5dd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CO-SO50-2019-0024-CX June, 2019 FY 2019 Right-of-Way Renewals Location'."
    },
    {
        "candidate_id": "a25b754b9ac05b1de7a5",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'included in Appendix 1 of DOI-BLM-UT-C010-2018-0087-CX. Plan Conformancy and Consistency The proposed'."
    },
    {
        "candidate_id": "1c8da7e1558cc9484a04",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Appendix B: Stipulations DOI-BLM-NV-S020-2020-0006-CX 6.2.7. Report weed populations they encounter'."
    },
    {
        "candidate_id": "cf1bd4d223bdf7c51dd3",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'County. NEPA Number: DOI-BLM-UT-G010-2015-0111-CX Lead Preparer: Bill Civish, Vernal Field Office'."
    },
    {
        "candidate_id": "bc647572ea2902abd754",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Assessment NEPA # - DOI-BLM-ORWA-P000-2013-0017- EA. The BPA Glass Butte Radio Station project was'."
    },
    {
        "candidate_id": "58e8cc9fcb0b1469eab7",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'NEPA Document Number: DOI-BLM-NV-C020-2023-0014-CX Categorical Exclusion Review Background BLM Office'."
    },
    {
        "candidate_id": "110ed822f37f53663a6d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Impact Statement (EIS), DOI-BLM-WY-P060-2014-0135-EIS, Record of Decision was approved December 23'."
    },
    {
        "candidate_id": "0bd778667010683a35f6",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'or other resource DOI-BLM-CO-N05-2016-0074-CX 8 surveys. Additional measures may be required'."
    },
    {
        "candidate_id": "63151290a615cbdca33b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Established By Statute DOI-BLM-UT-Y010-2018-0085-CX March 2018 Film Permit UTU-93180, Heavy Metal'."
    },
    {
        "candidate_id": "c9c68a9540ef24bce920",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'County. I have reviewed DOI-BLM-WY-P060-2022-0066-CX for offering Qwest Corporation dba CenturyLink'."
    },
    {
        "candidate_id": "7eec777b95e3c7bff8c5",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'maps, drawings, and DOI-BLM-CO-N020-2021-0016-CX 4 photographs. The BLM will forward documentation'."
    },
    {
        "candidate_id": "6ba775a7b594a835f04c",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Communication Site Assignments DOI-BLM-CO-N010-2020-0059-CX Identifying Information Project Title: Cedar'."
    },
    {
        "candidate_id": "727b6ba49ea805a5a74d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'NO [ ] X [ ] X 10 DOI-BLM-CO-G020-2022-0030-CX | BLM - Colorado River Valley Field Office EXTRAORDINARY'."
    },
    {
        "candidate_id": "27691241db042c510613",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Date: August 24, 2020 DOI-BLM-Y020-2020-0045-CX CATEGORICAL EXCLUSION DECISION DOCUMENT DOI-BLM'."
    },
    {
        "candidate_id": "5721b006bd058e956f33",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Carlsbad Field Office DOI-BLM-NM-P020-2018-0046-CX Pedro C Franco Driveway Easement NM-137227 Proposed'."
    },
    {
        "candidate_id": "a4f4dbc163218d974fb8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Off Road Warriors 3 DOI-BLM-AK-A020-2014-0016-EA T. TRAVEL MANAGEMENT AND OHV USE T-5 Management'."
    },
    {
        "candidate_id": "7a155962864e9d3684c1",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'covered by the CX. DOI-BLM-AZ-C010-2016-0032-CX 2017 Abandoned Mine Lands Remediation: Bat Cupolas'."
    },
    {
        "candidate_id": "a89e9eac7f7792235635",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Environmental Assessment #DOI-BLM-ID-B020-2012-0005. The actions analyzed in the Environmental Assessment'."
    },
    {
        "candidate_id": "e16901868b6598701a14",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DECISION NEPA Log Number: DOI-BLM-NM-L000-2016-0114-CX Lease/Serial/Case File No.: NMNM 095055 A. Background'."
    },
    {
        "candidate_id": "eaa8665522932c0055ce",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion (CX) Number: DOI-BLM-ORWA-B060-2017-0002-CX Date: June 21, 2017 Case File/Serial Number:'."
    },
    {
        "candidate_id": "cc408ad6130aa3a2df73",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Field Office (CFO) DOI-BLM-NM-P020-2021-0965-CX IT4RM EA-2021-0000 OXY USA Inc. Serial Number'."
    },
    {
        "candidate_id": "783fbe151134d452c4ed",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Number (if applicable): DOI-BLM-MT-C020-2017-0016-CX Proposed Action Title/Type: Assignment of a Land'."
    },
    {
        "candidate_id": "c9c7e04f2ae334122cd8",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'WYW182926 CX Number: DOI-BLM-WY-D040-2016-0130-CX Right-of-Way Applicant/Holder: PacifiCorp Proposed'."
    },
    {
        "candidate_id": "d6d31ca0acb44be011cd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Field Office NEPA #: DOI-BLM-AZ-G020-2013-0036-CX Serial / Case File No. N/A Proposed Action Title'."
    },
    {
        "candidate_id": "00d3b8d28423f387d56c",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion/Decision Record DOI-BLM-OR-S050-2012-0005-CX 1 streamside topography, and vegetation. Susceptibility'."
    },
    {
        "candidate_id": "2d5d26f8be17e208a934",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Allotment Transfer CX DOI-BLM-MT-C020-2024-0038-CX February 2024 Miles City Field Office 111 Garryowen'."
    },
    {
        "candidate_id": "0c93d2db7baa0b2fd9af",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'DOI-BLM-AZ-C010-2013-0051-EA EAFreeUsePermitBlackMountain.doc 11 the BLM Kingman'."
    },
    {
        "candidate_id": "af409f5e724193b765e9",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'status during the DOI-BLM-CO-N050-2019-0072-CX 5 Specialist/Realty Specialist prior to seeding'."
    },
    {
        "candidate_id": "c48e0ff9e2d510ef96d3",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'CIRCUMSTANCES CHECKLIST DOI-BLM-OR-E060-2016-0014-CX Right-of-Way Grant OR 68745 Review the proposed'."
    },
    {
        "candidate_id": "963ee3114eea2715c970",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Record NEPA Project No. DOI-BLM-ID-I030-2022-0025-CX Ride the Bayhorse DECISION AND RATIONALE FOR'."
    },
    {
        "candidate_id": "2d9ac43d1a6eea169478",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Communications Facility DOI-BLM-CA-C050-2020-0018-CX Additional on/in building installations: Metal'."
    },
    {
        "candidate_id": "2ad894d8a2159302de21",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Junction, Colorado 81506 DOI-BLM-CO-N030-2015-0009-CX May 2015 BLM US. DEPARTMENT OF THE INTERIOR BURSAU'."
    },
    {
        "candidate_id": "3547408f9b0a6fe6ad4d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '[ ] [x] [ ] [x] 3 DOI-BLM-CO-N040-2019-0080-CX | BLM - Colorado River Valley Field Office EXTRAORDINARY'."
    },
    {
        "candidate_id": "f32a0670c56404063e4e",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Areas NEPA Number DOI-BLM-AZ-G010-2023-0024-CX Bureau of Land Management - Safford Field Office'."
    },
    {
        "candidate_id": "6c8d75b7232fbc2bf345",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Rivers Field Manager DOI-BLM-ID-B010-2010-0083-CX Page 4 Agua Caliente Sandy Spring Road ROW Exhibit'."
    },
    {
        "candidate_id": "58888c4d6ab7b48f390c",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Transmission Project CONTENTS March 2016 i Final EIS/EIR Contents Executive Summary ..................'."
    },
    {
        "candidate_id": "569bd8040b616a79f151",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'contractor. 6 CX#: DOI-BLM-NV-W030-2022-0006-CX Applicant: Ormat Project Title: Pinto Temperature'."
    },
    {
        "candidate_id": "9dc1139e2972c1d71b38",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote '4 and Form 1842-1. DOI-BLM-ID-B011-2020-0023 Decision Record 1 **Page 2:** The appeal must be'."
    },
    {
        "candidate_id": "2c858d35bf62b34d3d5c",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Exclusion Review Record DOI-BLM-UT-Y020-2017-0303 UTU-92749 Matador Network Online Film Sponsored'."
    },
    {
        "candidate_id": "504df449d80e8e24921f",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2021-0081-CX birds. If nesting birds are found, methods to'."
    },
    {
        "candidate_id": "f1b5b403f2336c226dfd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'environment. Attachment #2 DOI-BLM-CO-N010-2016-0017-CX Area Map Draw Monument Hill 1812 Sand wring Maligon'."
    },
    {
        "candidate_id": "c4d3132a43f38140e52a",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'removal of facilities, DOI-BLM-CO-N05-2016-0115-CX 4 drainage structures, and surface material;'."
    },
    {
        "candidate_id": "4d0a9a428048bfc7e97c",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'BLM ASDO NEPA No. DOI-BLM-AZ-A010-2017-0041-EA 21 immediately notified. The immediate area of'."
    },
    {
        "candidate_id": "5b13362a2b6d398a46a7",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Monument NEPA No.: DOI-BLM-AZ-A030-2015-0001-CX Case File Nos.: AZA 026544 & AZA 033641 Proposed'."
    },
    {
        "candidate_id": "962eaa48200e2e934487",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'EXCLUSION NUMBER: DOI-BLM-CO-N010-2014-0032-CX CASEFILE/PROJECT NUMBER (optional): COC076517'."
    },
    {
        "candidate_id": "69db409f9d525127a92b",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Field Office NEPA No.: DOI-BLM-AZ-P020-2015-0005-CX Case File No.: AZA-19290, AZA-19291, AZA-19292'."
    },
    {
        "candidate_id": "bab109c5ee2b6933267d",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Communication Site Renewal DOI-BLM-AZ-C030-2017-0041-CX A. Background The Lake Havasu Field Office received'."
    },
    {
        "candidate_id": "08d6912c450690217d62",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'PSoC Line Maintenance DOI-BLM-CO-F020-2018-0071 CX 0 0 5 1 2 3 4 Miles NOTE TO MAP'."
    },
    {
        "candidate_id": "7d30f849e675b87e6a17",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'authorized officer. 2 DOI-BLM-CO-G020-2023-0007-CX | BLM - Colorado River Valley Field Office The'."
    },
    {
        "candidate_id": "9820c13d190fa51d0dcd",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'categorical exclusion # DOI-BLM-ID-I010-2019-00184-CX has been reviewed to determine that none of the'."
    },
    {
        "candidate_id": "df0da9ed601efb17a511",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Hazard Tree Cutting DOI-BLM-ORWA-N020-2017-0001-CX 6 U.S. DEPARTMENT OF INTERIOR BUREAU OF LAND'."
    },
    {
        "candidate_id": "083f806bbe54f5e3c414",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-CA-D070-2018-0074-CX Pioneer Productions Motion Picture Filming ISDRA'."
    },
    {
        "candidate_id": "64b20326ee975568caaf",
        "label": "neither",
        "notes": "Neither: NEPA case number, quote 'Categorical Exclusion DOI-BLM-NV-S010-2020-0066-CX 7.2. The use of pesticide treatment requires'."
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
