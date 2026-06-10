import pandas as pd


LABELS = [
    {
        "candidate_id": "72a26279ffecfaa3c5dd",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d0f050722af5f37c19af",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9577356914fb2855433b",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote '0 125 0 25 0 5 Miles Date Created 1/29/2018 Created By mpereira NAD 1983 UTM'."
    },
    {
        "candidate_id": "a26cb6d3221291c603bc",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: OMER BAKSHI Digitally signed by OMER BAKSHI Date'."
    },
    {
        "candidate_id": "9192a55e81ad02416854",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ef21f3bfc7b9ba352b57",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'site visit and written comments filed with the Commission staff issued a revised scoping document on'."
    },
    {
        "candidate_id": "9ad0b52f00300dd12d00",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a9b71f4fa64549457fb7",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Consultants 47 May 2011 7 0 CONSULTATION The NNDFW was consulted on this project A biological'."
    },
    {
        "candidate_id": "d43ed71e979826058bfd",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'review ML14157A354 July 8 2014 D Wrona NRC to A Payment Tribal Chairperson Sault Ste Marie'."
    },
    {
        "candidate_id": "c43a9da007ade974ddb2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a79688b7333afcd51981",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'intervene Intervenor Date Filed California State Water Resources Control Board September 9 2014 National Oceanic and'."
    },
    {
        "candidate_id": "56e51c09a98be6fc1509",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'THE PROPOSED ACTION On December 19 2017 The Billings Field Office received an application from Marathon'."
    },
    {
        "candidate_id": "fccb9548be63e540f8f6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "eb28a7e3c8a2f831dcce",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'and comment period on March 9 2012 8 4 PUBLIC COMMENTS ON THE SFEIS LADOTD made'."
    },
    {
        "candidate_id": "f57ec88f035bb4a3066a",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'submitted to NMFS and USFWS on May 18 2012 NMFS provided a request for additional information'."
    },
    {
        "candidate_id": "7e23719c931005bc48fa",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "fbaa5b0c5fa14efaef83",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: RICHARD BAKER Date: 03 / 10 / 2017 month day'."
    },
    {
        "candidate_id": "49e00fbef819d1154de3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cc94b10b4979200c76fa",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Intent NOI and was published in the Federal Register on January 24 2018 The NOI listed'."
    },
    {
        "candidate_id": "e3d7c2fbe40c8cfbbba2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Nancy Baker Assistant Field Manager -- Minerals and Lands Tim Novotny Assistant Field'."
    },
    {
        "candidate_id": "e3a9617d370cc18f8f4f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ad1d0087dbcd52370743",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3148a78c1b7f1d9bb609",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Shoshone Field Office received an application from the City of Gooding on April 16 2013 for'."
    },
    {
        "candidate_id": "0461fd36616c36da59ec",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bf1dbd5a4b6fd55d45c9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: David Ollett Date: 8 / 26 / 2014 month day year'."
    },
    {
        "candidate_id": "e44b5f2d81aca9b9a60b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Stephen Henry Date: 09 / 26 / 2022 month day'."
    },
    {
        "candidate_id": "f4981d37f7bba603a287",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'environmental review process On January 13 2015 FERC issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "73ae7e1e57f8483da832",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9f901c16299ec04e96e4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "32901408a358ade9840a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0c1507f70032dffba15a",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'the SEPA Register on February 15 2019 These notices initiated formal scoping and started a required'."
    },
    {
        "candidate_id": "999530b6a466a7c79ce4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a6790f523eeea811f341",
        "label": "neither",
        "notes": "Neither: consultation date, quote '22 and 23 1996 On April 12 1996 the Tribe filed a petition requesting that the'."
    },
    {
        "candidate_id": "f09d633a33eb2d91d40d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0416b925bb02a5a03582",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "99a734857cf36c8d43b8",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'of the business on February 13 2020 the BLM Tres Rios Field Office received a signed'."
    },
    {
        "candidate_id": "126a6b2d7e02861635ef",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'Comment Period 22 The scoping period for the DRS was initiated on December 8 2014 with'."
    },
    {
        "candidate_id": "f23aa994c53613f8319e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "95570a66861a34754c7d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Ryan Watson DATE: 09 /17 / 2010 month day year'."
    },
    {
        "candidate_id": "d040959491fb1f7db20a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e1e493a317f8952fb7ee",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "56e91253ff4e88e6fc7e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "09631dccc95fe08347e0",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "dee23887ad5967c4f029",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: ZACHARY ROBERTS Digitally signed by ZACHARY ROBERTS'."
    },
    {
        "candidate_id": "8b0b8642434074bffb3c",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'environmental review process On November 4 2014 the Commission issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "bd09704045c723280953",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'that day and continued through January 31 2006 Note In the NOI the public scoping comment'."
    },
    {
        "candidate_id": "c6a3c73038015c01f0f2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "80abf301d2ea9508cb1a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BARBARA NOLL Digitally signed by BARBARA NOLL Date'."
    },
    {
        "candidate_id": "9d27e5cdb15e86eafc94",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3f133f85f2d74bcbdc53",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c5b19b529df60875df25",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "80f8b93931527c5d873b",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'hearings or mail written comments on or before the comment period closing date of November 15'."
    },
    {
        "candidate_id": "f3c5fd00ea89bea25c17",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "2e2d273991adc32fc8fc",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3a013b5b865689c64d1d",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'was established from April 10 2012 to May 12 2012 Two scoping meetings were scheduled during'."
    },
    {
        "candidate_id": "34df823992e37cbd00ea",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Application ML20294A483 11/02/2020 B. Obermeyer, Delaware Tribe Historic Preservation Office, to R. Hoffman (NRC)'."
    },
    {
        "candidate_id": "e326a0cff06e06b3e299",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'comment period began on December 21 2013 The comment period for the Draft EIS began December'."
    },
    {
        "candidate_id": "9d521fab50980c4a0ac7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5834dbb613fbd2054678",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JAMES POSTON Digitally signed by JAMES POSTON Date'."
    },
    {
        "candidate_id": "e1db6087dc042eabdf6f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "04d4c19ba65253ee1026",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7988d81d6d7e6db71a43",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "28e048accdaa9b84573c",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'on December 5 2019 Consultation with the SHPO and coordination with the tribes occurred on February'."
    },
    {
        "candidate_id": "e4749ce0b9537305d104",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Joseph Quaranta DATE: 07 / 30 / 2010 month day'."
    },
    {
        "candidate_id": "965c1553181a3db96a58",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On March 13 2018 YETI Coolers filed film permit application UTU-93206 proposing 3 days'."
    },
    {
        "candidate_id": "3480d107ca914d6a7398",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d90e49da05aedbef0f4a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5e755c4b4398c0fefb97",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3f3a636c01b174de8747",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'agencies and State Tribal and local governments to consider becoming cooperating agencies in the preparation of'."
    },
    {
        "candidate_id": "746699e3939dcff11f71",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'scoping period ended on February 4 2005 during which time BLM received eleven written comment letters'."
    },
    {
        "candidate_id": "3bc2fdcb40fed70a2ca2",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Confirmed receipt Phone 3/16/23 Letter mailed 3/21/23 Email 4/20/23 Kickapoo Tribe in Kansas 10/31/22 Emailed 3/6/23'."
    },
    {
        "candidate_id": "537ca326594ca65ab2e5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a6d9d9fc0ad9a385f835",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'on these documents August 16 2006 T 1 1 e Service received a request for fomlal'."
    },
    {
        "candidate_id": "f4e91eb8fadf91568d73",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'No PF15-15-000 On August 12 2015 the FERC s staff participated in a site visit of'."
    },
    {
        "candidate_id": "cb00c26ecc1549e67056",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b7c447a1dbdbecac1fe5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "60808f59d456d35c436c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5e8a8a892ee7e0a84195",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'the EIS for filing public comments are actually not due until August 18 2008 We therefore'."
    },
    {
        "candidate_id": "d48921d8cdc7698db77c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "fc887d8c1dd20ea3389d",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'meeting was mailed on June 4 2002 to various media contacts A letter describing the LBA'."
    },
    {
        "candidate_id": "ea0441cd1953ff656214",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "414dca1eb6a366fe4c77",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Federal Register on January 17 2025 Public meetings were held to inform the public of the'."
    },
    {
        "candidate_id": "026b91ac7261a6cbb09c",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'the same month The public comment period closed on July 29 2005 A transcript of the'."
    },
    {
        "candidate_id": "83f629345e9bcb4db496",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'and discussed at PRRCT public meetings held on April 19 2006 in Casper Wyoming The applicant'."
    },
    {
        "candidate_id": "232692ec078ab096875e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bd33918d3d3b05dd0680",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2 and 3 ML24303A015 11/01/2024 NRC letter to M J Wesaw Chair Pokagon Band of Potawatomi'."
    },
    {
        "candidate_id": "6f13d21ae866065a666c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "292c095fc4473e35d086",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f3bc98906235b45ab5fe",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing permit On April 14 2020 the Lewistown BLM Field Office received completed of transfer'."
    },
    {
        "candidate_id": "55ea49f4f82c9f390327",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "33e89ce6789695605a45",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'managed public lands On February 1 2022 Powder River County submitted a SF-299 application to the'."
    },
    {
        "candidate_id": "506ad472cd593ff6f2a5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5bdca5cbe3dec4a7b7ab",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Standing Rock Sioux Tribe 15 Feb 2008 Letter Col Vander Hamm to Chairman His Horse Is'."
    },
    {
        "candidate_id": "3a462b92a03d64cade6d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f847a36c929bcdc99d26",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Scoping Meeting NOI On August 2 2016 the FERC issued a Supplemental Notice of Intent to'."
    },
    {
        "candidate_id": "7d116982010d32dc514c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bb4b7ced5da577085363",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'entailed publishing a Notice of Intent to prepare an Environmental Impact Statement in the Federal Register'."
    },
    {
        "candidate_id": "69b82c6814e7859fc092",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'comments in a public forum May 3 2012 Notice of Intent NOI for NEPA NOI published'."
    },
    {
        "candidate_id": "20302b82a815d3c5de86",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c11ed88f2eab11f18cd7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4fa069016f19e85c6075",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0d555aa9cd38432bbce6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6fd9906d34920fd4f322",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7512ae63aad33dbf85a4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8a49030f07c869325f60",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a138131c93c042131a10",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Letters ML24103A114 05/06/2024 Clinton Power Station Unit 1 License Renewal Public Scoping Meeting May 2024 Meeting'."
    },
    {
        "candidate_id": "fd6b95065cb7520138e8",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'years. The original ROW grant was issued for a 30-year term in July of 1989.'."
    },
    {
        "candidate_id": "71ed683e944aeced30b3",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Gulf XPress Project On June 2 2016 the Commission issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "876c277bcb8020a050e9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: SETH LAWSON Date: 11 / 17 / 2022 month day year'."
    },
    {
        "candidate_id": "f2e7e72c59d6495a0c3a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "251a1036392b75b9d54c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a439da5988ec0defdf09",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "fdcf9c475fb954529f57",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'submitted a 9 Part 2 Application in January 2015 As requested by DOE in a letter'."
    },
    {
        "candidate_id": "f146f0d05ede3b38a5c3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5ce1b9c39bfe594bdbdb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "276258462d3aa61d8add",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f94404e74b1e598f7e5c",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Unanticipated Cultural Resources July 2022 Page 9 of 22 REDACTED organization or in the case of'."
    },
    {
        "candidate_id": "1b8adbc85e1246eb97a2",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: ANTHONY ZINN Digitally signed by ANTHONY ZINN Date'."
    },
    {
        "candidate_id": "dd0a41c68c64a54903c0",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Adrienne Riggi Date: 03/16/2012 month day year'."
    },
    {
        "candidate_id": "2510f1983a065eef8e99",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Application ML23069A280 03/13/2023 T Smith NRC to K Jensvold Tribal Chairman Upper Sioux Community Request for'."
    },
    {
        "candidate_id": "c530986c579c96763f45",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2020 After subsequent consultation with USACE on May 6 2020 the FAA submitted a request for'."
    },
    {
        "candidate_id": "1ad26d9e080a6dc2fc8a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a75e12f168083a72eac1",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'McGrew, Realty Specialist Date Prepared: June 10, 2015 D. Decision and Rationale for Action I considered'."
    },
    {
        "candidate_id": "168e739bb8a876cb3a92",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'and Wildlife Service USFWS in a consultation letter dated September 29 2010 Please see the attached'."
    },
    {
        "candidate_id": "1d29a7208369c692712d",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'for Leave to Answer on October 20 2020 13 followed by an Order on Rehearing DOE/FE'."
    },
    {
        "candidate_id": "67d378e9596038575da0",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Kansas 10/31/22 Emailed 3/6/23 Confirmed receipt; \u2022 Tribal website email - 3/16/23 \u2022 Letter mailed - 3/21/23'."
    },
    {
        "candidate_id": "62a971acb27de9b3da99",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0f98b4e458e06bc0fa08",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9e904ef71afa667ebbfc",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'PROPOSED ACTION On March 22 2022 the Bureau of Land Management Royal Gorge Field Office received'."
    },
    {
        "candidate_id": "e0b06af6fcab25339d35",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'interested stakeholders On July 23 2015 the FERC issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "57f6092043c8d35d64ac",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1294112acb33f98181eb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "43060930fe8bc3eebc44",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "56d5f463a0694964098e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f8183edc74855fee37e4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '4 34 BACKGROUND On October 30 2018 the Bureau of Land Management BLM Tres Rios Field'."
    },
    {
        "candidate_id": "81ca6e00963356473e02",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Public Involvement A Notice of Intent NOI advertising the scoping period was originally published in the'."
    },
    {
        "candidate_id": "19fd0735798f01019373",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'requests in 2016 In July 2016 a mechanical spray evaporator was taken out of service and'."
    },
    {
        "candidate_id": "4a5a4da92895822e1133",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Otis Mills DATE: 12 / 13 / 2010 month day year'."
    },
    {
        "candidate_id": "de31cf0b800d03598444",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0b579812295c7d85b530",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "395953db01673a476754",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'sunflower However the USFWS subsequently rescinded their not likely to adversely affect concurrence for the USACE'."
    },
    {
        "candidate_id": "1d263e38793a759ea6cc",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '1 I INTRODUCTION On August 14 2020 Golden Pass LNG Terminal LLC Golden Pass LNG filed'."
    },
    {
        "candidate_id": "63b59fa1a70038595aec",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d155df66f66aa144027e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "841accf80f77b991a612",
        "label": "initiation",
        "notes": "Initiation: posted to ePlanning/NEPA Register, quote 'Livestock Company on September 13 2013 The project was posted on the BLM ePlanning website on'."
    },
    {
        "candidate_id": "2ed561472f7d9aff3250",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote '2 Scoping Period The notice of intent to prepare an EIS was published in the Federal'."
    },
    {
        "candidate_id": "f77002eeccf200d63be8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "60b20200fe0c968a8545",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1cb3b421148034692662",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "564f83f87fb3999eb781",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JESSICA MULLEN Digitally signed by JESSICA MULLEN'."
    },
    {
        "candidate_id": "6bcde45834342043702d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4d4350f93a6d4661cf50",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ca0e9c62e38a9dc519ab",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d69216bf8009719af325",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Connector 2020 2022 Notice of Intent to Prepare an EIS published in the Federal Register December'."
    },
    {
        "candidate_id": "e51dfb3c017bc96a2fa0",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'any effect on Indian tribal self-governance or sovereignty tribal treaties or other rights Consistent with EO'."
    },
    {
        "candidate_id": "c7d92444a150f4b1fdfa",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Contact/Correspondence* 02/12/2024 Formal Consultation under Programmatic Biological Opinion \u2013 Salinas 02'."
    },
    {
        "candidate_id": "3ec3ad03751392683835",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'ENVIRONMENTAL REQUIREMENTS The Notice of Intent to prepare an Environmental Impact Statement was published in the'."
    },
    {
        "candidate_id": "9c70d9d307681b5dca68",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "896d5575b99ddf20bfe1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7f735c154c5cfa7a7334",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ffe64656fc2c30a455c6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "72d365e8a17e08af5534",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'emails 3/30/23 Email 4/20/23 Postcard notification of the Draft EIS 6/27/23 Consultation considered complete Miami Tribe'."
    },
    {
        "candidate_id": "2926ee4982c25a6780dc",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'responses Wednesday April 27 2016 ID Team/Required Reviewers will be determined at scoping meeting or as'."
    },
    {
        "candidate_id": "c055240665f6826083d1",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: LEE JENSEN Digitally signed by LEE JENSEN Date: 2023'."
    },
    {
        "candidate_id": "0f50d4d4a592c46f8336",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3a1a1cd1e1b91c93dc10",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "303f623f2300a1c0cdc7",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'application to renew an existing buried copper phone line ROW along the south side of Bridgeport'."
    },
    {
        "candidate_id": "9793a55794e4781c42cb",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2011 FHWA letter to SHPO Request for concurrence on APE September 8 2011 FHWA letter to'."
    },
    {
        "candidate_id": "9d9f4db31f63b8f9fb6c",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Jasper Date that any scoping meeting was conducted N/A Date that concurrent electronic distribution for review'."
    },
    {
        "candidate_id": "f2d5e826a75dd78161da",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9e3739c33930eb589310",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'comments/notification of Section 106 review ML13186A174 8/13/13 M Wong NRC to J Greendeer Ho-Chunk Nation Request'."
    },
    {
        "candidate_id": "6adfa3ea888b1387432a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Ryan Egidi DATE: 02 / 26 / 2010 month day year'."
    },
    {
        "candidate_id": "ed853222ab4e43178316",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Circuit also granted the USFWS s request to suspend the legal challenge until the USFWS had'."
    },
    {
        "candidate_id": "c7f007c40551ae010667",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f16f1da575981598d171",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'development The official scoping period that kicked off the NEPA process began with publication of the'."
    },
    {
        "candidate_id": "4ac0e4d02e9559ca92ce",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Federal Register on June 10 2004 the BLM published a Notice of Availability of the Final'."
    },
    {
        "candidate_id": "ddac0d77ff9ce4d85387",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Information RAI from USFWS on September 4 2013 The Corps provided a Supplemental Technical Analysis in'."
    },
    {
        "candidate_id": "2458d4dc4ac91481e12e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote '2020 which opened a comment period through January 5 2021 on the analysis related to the'."
    },
    {
        "candidate_id": "3dc0a9b75af32c5d5881",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'how it addressed the March 30 2009 petition to close roads and trails in the Jemez'."
    },
    {
        "candidate_id": "fd1d7057f9760f702395",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "2f404d8e78826797cefa",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b4abd39877849995a990",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'environmental review process On January 27 2017 the Commission issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "29d25183277ba16a22f5",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'comments/notification of Section 106 review ML18114A381 May 24 2018 B Beasley NRC to J Floyd Muscogee'."
    },
    {
        "candidate_id": "72dfdca2661cf882421b",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Travel Plan and on March 31 2004 it published an official Notice of intent to prepare'."
    },
    {
        "candidate_id": "00b228b9df2128030ed2",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote '9 c 4 A revised notice of intent was published on August 8 2015 with a'."
    },
    {
        "candidate_id": "1ef1cb1931004f215242",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: SAI GOLLAKOTA Date: 09 / 06 / 23 month day year'."
    },
    {
        "candidate_id": "ea1542cdb10e9e1d4d2d",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'of Zuni was held on March 16 2020 Scoping Process The Proponent submitted its initial ROW'."
    },
    {
        "candidate_id": "d4ed77aad43a1b8784df",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2016 month day year NEPA Compliance Officer: Fred E. Pozzuto Date: 09 / 09 / 2016 month day'."
    },
    {
        "candidate_id": "7f353919ecbaff5994bb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bcced25865eef17eede3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "980962d945e400b01da9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e7824c95f9e6bf66f837",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'site visit and written comments filed with the Commission we issued a second scoping document SD2'."
    },
    {
        "candidate_id": "c96cffade06a415ee0b1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cfdc3f59ae50f3c5d698",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'under review filed February 2015 Not Applicable Not Applicable Ohio Department of Natural Resources State-listed species'."
    },
    {
        "candidate_id": "5608513e9a4fcf044b96",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "69917bda9c8884b78143",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'current renewable energy right-of-way policy guidance (WO-IM-2011-061, issued February 7, 2011). \u2022 For pending applications'."
    },
    {
        "candidate_id": "7975ed59cf924538cb6a",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Laurie Ford Date that any scoping meeting was conducted N/A Date that concurrent electronic distribution for'."
    },
    {
        "candidate_id": "c287886036c39e29d402",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6687b80d68866a182162",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Pumped Storage Project On April 29 2016 FirstLight Hydro Generating Company filed an application for a'."
    },
    {
        "candidate_id": "b83b87191880a0cbf8da",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
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
