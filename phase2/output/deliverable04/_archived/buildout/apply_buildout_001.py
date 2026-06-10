import pandas as pd


LABELS = [
    {
        "candidate_id": "035034ebcbc373062c96",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a3d71eef3f65ca55ec45",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: MARIA REIDPATH Digitally signed by MARIA REIDPATH'."
    },
    {
        "candidate_id": "37385f3eb4f26ff7848f",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'geology and soils On September 16 2016 the Notice of Availability for the DEIS was published'."
    },
    {
        "candidate_id": "02a00775b21e0803e88d",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'followed He explained how comments could be submitted verbally at the hearing in writing at the'."
    },
    {
        "candidate_id": "24cbebc284b35e72ec96",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'were invited to submit comments on or before November 4 2011 The comment period was later'."
    },
    {
        "candidate_id": "d8a468f10151af80dc79",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Region in Nebraska May 2012 Keystone filed a new application for a Presidential Permit for the'."
    },
    {
        "candidate_id": "fe81821fa37d45e78264",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2 1 2 24 APRIL 2019 USFWS COOPERATING AGENCY RESPONSE LETTER 17 2 1 3 24'."
    },
    {
        "candidate_id": "dd70ca2d8b9f7d890052",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'the open houses The comment period closed on June 16 2003 A total of 642 written'."
    },
    {
        "candidate_id": "7fe32f41aebe49a8df74",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "eecdb92358b0b5ef2ade",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BENJAMIN CHORPENING Digitally signed by BENJAMIN'."
    },
    {
        "candidate_id": "75507de159cf8bc14244",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "377c0f162f8f6a193fb9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ded2daf2ae61917e0c07",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: RICHARD BERGEN Date: 10 / 11 / 2023 month day'."
    },
    {
        "candidate_id": "301ee2672cfc74b78ffb",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Public Engagement The Notice of Intent to prepare an EIS was published in the Federal Register'."
    },
    {
        "candidate_id": "bac426b9a5b6f0d90e72",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'INTRODUCTION and BACKGROUND In August 2023 the BLM Ukiah Field Office UKFO received an application from'."
    },
    {
        "candidate_id": "0c688629b51f4ce95cf1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8d6739114973d9b32498",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "483712b2ecb7ad93258d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4eedb07be37a39b93d78",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "727d20d47245fb48b704",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'informational open houses in February 2012 September 2012 and May 2013 The purpose of the open'."
    },
    {
        "candidate_id": "bc555189b339f4c787e8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a2ca2b02365a496040f3",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'cultural resources In January 2011 Boise District received an application for a Special Recreation Use Permit'."
    },
    {
        "candidate_id": "f9c6e719d8873fadc8f3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d941d81260f542d88d0d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1c9a7ad8ad9b16868e77",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "168814549e86d6eb9874",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '27 2022 Zoom meeting December 13 2022 Zoom meeting March 29 2023 Zoom meeting Public scoping'."
    },
    {
        "candidate_id": "995020549240b1e03be9",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing permit On July 13 2018 the Lewistown BLM Field Office received completed transfer of'."
    },
    {
        "candidate_id": "0449332616dee145b344",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b2f09ed1f6d2ff3a69c4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "104bae5e20bd400ca3c3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9e69786c56c1e732dd19",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Cooperating Agency 125 January 7 2022 Email from USFWS Declining Offer to Become a Cooperating Agency'."
    },
    {
        "candidate_id": "3780d482fcbda26dedfb",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Land Management BLM received an amendment Right-of-Way ROW application N-91801 for American Tower Corporation's Lockes NV'."
    },
    {
        "candidate_id": "e1dcadf502c8b65ae1b2",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Pipeline Project On July 28 2020 FERC issued a Notice of Intent to Prepare an Environmental'."
    },
    {
        "candidate_id": "4ffdd7451ba59493e10e",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'of responses Friday October 20 2017 ID Team/Required Reviewers will be determined at scoping meeting or'."
    },
    {
        "candidate_id": "f59531285b4f6fb8af07",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'of change of public hearing date which was originally scheduled for May 24 2005 to May'."
    },
    {
        "candidate_id": "c3973c665f867210db31",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Preservation Officer dated March 4 2022 14 regarding Request to Initiate Section 106 Consultation and Scoping'."
    },
    {
        "candidate_id": "4a6cce8c3a64f70364f5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: DATE: 11 / 19 / 2010 NEPA Compliance Officer'."
    },
    {
        "candidate_id": "306d32c461304e5a5054",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'would require an EIS On July 19 2012 we issued a Supplemental Notice of Intent to'."
    },
    {
        "candidate_id": "6188da1d05f63db12c3a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: PING WANG Date: 01 / 26 / 2024 month day year'."
    },
    {
        "candidate_id": "39a3a322a012980745f7",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'comments on the draft biological assessment from USFWS 10-11-2019 Regulatory Decision NCMRWC approved resolution affirming commitment'."
    },
    {
        "candidate_id": "f895264452dbbc7ca791",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5be5bcc56ed17667f651",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Canal 2 3 4 1 138 565 Date Created 3/28/2018 Created By kprestwich NAD 1983 UTM'."
    },
    {
        "candidate_id": "1511c53443c201792e6e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "2ee4efa447d18817855b",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Spring GEIKO Humboldt Na Sources Esri Garmin USGS NPS Date 9/6/2023 6 CATEGORICAL EXCLUSION WORKSHEET STIPULATIONS'."
    },
    {
        "candidate_id": "cc06e18b896d5d02cf50",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'meeting and announced that public comments were requested to be received within 35 days no later'."
    },
    {
        "candidate_id": "60a35fd512318d2a77dd",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'scoping period that expired on October 15 2024 We received comments addressing stakeholder accessibility to the'."
    },
    {
        "candidate_id": "afbb284f703167bb4c3b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "646be929a0bc14feeb7f",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Scoping Activities A notice of intent to develop a CCP and a request for comments was'."
    },
    {
        "candidate_id": "5b8e587ba0260ea87f72",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'decision was signed by Forest Supervisor Kevin Martin on May 18, 2009. In July 2009, the'."
    },
    {
        "candidate_id": "2db4741886a5b44de9cf",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Application filed on December 23 2015 FTA Authorization received on August 17 2016 Non-FTA Authorization is'."
    },
    {
        "candidate_id": "0532589a5b354aa56c8e",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'public requesting Project public meetings Commonwealth held an initial open house meeting on October 23 2017'."
    },
    {
        "candidate_id": "17f00a4e96bb3b2b37ce",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Forest Service was open between April 29 2016 and May 31 2016 and for Placer County'."
    },
    {
        "candidate_id": "f6e3ba2629d2a2a02830",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Elevation Solar C LLC Amended SFA to be effective 8/15/2017 Filed Date 8/14/17 Accession Number 20170814'."
    },
    {
        "candidate_id": "f62cc173fc8a36e64c2d",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'proceeded with the PEIS On March 12 1999 DOE submitted the plan to Congress no legislation'."
    },
    {
        "candidate_id": "05b59cd805038d7eec3b",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'Agency and Other Entity Date Filed California DFW July 21 2014 FWS July 22 2014 Conservation'."
    },
    {
        "candidate_id": "bac7e4233612d429b36b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7d841ba503f26db66e0c",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'a Notice of Intent NOI in the Federal Register to inform the public of the planning'."
    },
    {
        "candidate_id": "6bb34a7668b126fe8fc3",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Patcharin Burke Digitally signed by Patcharin Burke'."
    },
    {
        "candidate_id": "45f17399596a121ca18f",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'January 7 2020 On November 15 2019 Mark Sharp on behalf of Idaho State University ISU'."
    },
    {
        "candidate_id": "57091e63d27606a182bc",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'dated April 9 2021 the USFWS added one additional species but gave concurrence to the rest'."
    },
    {
        "candidate_id": "953de09f34870f50a7a5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: (Signature present) DATE: 12 / 30 / 2010 month'."
    },
    {
        "candidate_id": "5bcd0ecd89a863dffb98",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Gary Covatch Digitally signed by Gary Cavatch DATE'."
    },
    {
        "candidate_id": "6ea99dd9b2621defeaa6",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: LEE JENSEN Digitally signed by LEE JENSEN Date: 2018'."
    },
    {
        "candidate_id": "306dd0f858b143efbcf0",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Application ML20294A491 10/30/2020 R Elliott NRC to W Brown Chief Cheroenhaka Nottoway Tribe Request for Scoping'."
    },
    {
        "candidate_id": "966951c73d2d069f32f9",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'must either modify its existing permit or apply for a new MPDES permit for the project'."
    },
    {
        "candidate_id": "f9de12d716a58bfd70af",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5c6b9af3d40fe54ca0f1",
        "label": "initiation",
        "notes": "Initiation: FERC pre-filing approved, quote 'PUBLIC INVOLVEMENT On October 10 2014 FERC accepted Venture Global s request to begin pre-filing and'."
    },
    {
        "candidate_id": "b43cdffa60f62beeb163",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4f75e61c2d94f9345f53",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Critical Habitat on November 22 2017 Preparer's Initials DT i Violate a Federal law or a'."
    },
    {
        "candidate_id": "033cbe44290eda4bb515",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "661f40248308d046da99",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "54dec4a6e68568560f01",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On April 18 2018 Northwest Pipeline filed an application for a Temporary Use Permit'."
    },
    {
        "candidate_id": "c4bbc8d707b77e4f7d9a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b2c316b7ff4876bff715",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Assessment to Initiate EFH Consultation October 14 2019 NMFS Issues a Response to the EFH Consultation'."
    },
    {
        "candidate_id": "f88c3ef3761835d9150d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "da10c875d6fb1c2538e4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'conceptual mitigation plan filed on July 27 2016 Approved jurisdictional determination received March 6 2018 Updated'."
    },
    {
        "candidate_id": "c5be5294d6dcf12f61f1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "73ddb99147194e5b0214",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "091971875731fe14c52c",
        "label": "initiation",
        "notes": "Initiation: FERC/application notice, quote 'environmental surveys On October 22 2020 the FERC issued a Notice of Application NOA The NOA'."
    },
    {
        "candidate_id": "720fd96b310b593fe7b7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1039928a4ff65be8e44b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9783531eb428f1e8551c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "eaedcc697f90091d520c",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Technical Review Initial consultation letter sent 8/21/14 Pre- application meeting held 7/13/15 and 12/17/15 Updated route'."
    },
    {
        "candidate_id": "47194006cfb1fb9f72e4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f884e2ddddd8abf93866",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Certification process PALNG submitted its application and request for consistency review as part of its USACE'."
    },
    {
        "candidate_id": "edea5f068718270fedfc",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f41a7b91e01af5c1d40b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: ERIK ALBENZE Date: 7 / 1 / 2019 month day year'."
    },
    {
        "candidate_id": "dc2fdc6e0a9676b50b26",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "73a78b09d4fa8ac04345",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c8a986793230e1e4b123",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "559c759f8531c280a8d7",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'Pipeline Pipeline Pipeline Expires 12/31/2020 12/31/2020 12/31/2019 3/19/2020 4/12/2020 5/20/2020 11/5/2020 12/31/2019 12/31/2020'."
    },
    {
        "candidate_id": "50c54c04a745e18272d8",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'From To Description May 10 2019 Letter FTA Muckleshoot Indian Tribe Snoqualmie Indian Tribe Stillaguamish Tribe'."
    },
    {
        "candidate_id": "46132d15107eada73c12",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'MITIGATION PROPOSAL On January 28 2010 Compact Power Inc submitted a Part 303 Wetland Permit Application'."
    },
    {
        "candidate_id": "a8db322144eacb703957",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '62 No 211 Friday October 31 1997 Notices 58979 DEPARTMENT OF THE INTERIOR Bureau of Land'."
    },
    {
        "candidate_id": "d3c21e12cfedcaf882d9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bf608236383300fb0694",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c610e86642692b856f49",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Richard Baker Date: 05 / 10 / 2021 month day'."
    },
    {
        "candidate_id": "7f43ffee922ef08f7374",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f04e60d0ecf7d2c1dbec",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c180c35050ac5c21f89c",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'a public notice on 3 July 22 2015 providing notice of public meetings held on July'."
    },
    {
        "candidate_id": "e6d033ef2a8091d306d2",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Scoping Report The scoping comment period ended on December 27 2017 In total 13 comment letters'."
    },
    {
        "candidate_id": "5d54a7f33165faeb5c29",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'Comment Period 22 The scoping period for the DRS was initiated on December 8 2014 with'."
    },
    {
        "candidate_id": "6b6ec77dd3fe90222e4f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "200bf85661f243cc04c0",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '0003-CX BACKGROUND On May 9 2019 the Bureau of Land Management BLM Tres Rios Field Office'."
    },
    {
        "candidate_id": "fba1de3aaa39157d3faa",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ef29982df4a16e612b19",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Renewal ML12128A093 June 6 2012 Letter from the NRC to Dr Andrea A Hunter Tribal Historic'."
    },
    {
        "candidate_id": "617657c76155d364e647",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Private other 1 141 168 Date Created 3/28/2018 Created By kprestwich NAD 1983 UTM Zone 11N'."
    },
    {
        "candidate_id": "396c54d6db03e79bc404",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'meetings and written comments filed with the Commission we issued a revised scoping document SD2 on'."
    },
    {
        "candidate_id": "0356d7bfb264a18fc385",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'announced the 32-day public scoping period solicited public comment and announced scheduled scoping meetings The scoping'."
    },
    {
        "candidate_id": "31390300fb1573653a12",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Consulting Parties January 9 2018 Meeting with DAHP and Consulting Parties to Review draft Programmatic Agreement'."
    },
    {
        "candidate_id": "b9902390406c2b5cb192",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "45b56ec325e96cf10182",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: John Jason Conley DATE: 08 /26 / 2011 month day'."
    },
    {
        "candidate_id": "bca713652a1074c885e6",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'comments/notification of Section 106 review ML13191B089 8/8/13 M Wong NRC to R Nelson ACHP Request for'."
    },
    {
        "candidate_id": "bae5c776ff7b6d07f6c1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "740ebb9b345ed4dc8113",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'prepared in response to an application submitted on October 10 2008 to the U S Nuclear'."
    },
    {
        "candidate_id": "c1a0c06196adc5bafaa4",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'of the initial 45-day comment period was granted resulting in a 90-day comment period that remained'."
    },
    {
        "candidate_id": "82dac2ef08b1958c22b6",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: LEE JENSEN Date: 03 / 26 / 2020 month day year'."
    },
    {
        "candidate_id": "681d70a15c1b4f576f59",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'published in October 2014 Public meetings were held in Dayton Washington on October 29 2014 and'."
    },
    {
        "candidate_id": "3e185264619cbe76dac3",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'on March 12 2015 On March 25 2015 the Commission issued a supplemental NOI to extend'."
    },
    {
        "candidate_id": "810b278dc9f695e57de8",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Species Act Section 7 consultation Concurrence received December 1 2022 NOAA/NMFS review of capacity amendment Request'."
    },
    {
        "candidate_id": "ea1ba067ec59ec9f9d0b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "eb2a3bd702807e1fe757",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'application for the MVP on October 23 2015 and Equitrans had filed its formal application for'."
    },
    {
        "candidate_id": "d25becb6b8c63669b6ec",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: NATALIE IANNACCHIONE Digitally signed by NATALIE'."
    },
    {
        "candidate_id": "8d45a6456f29078b2129",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3d3aea74439795075729",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Neil Kirschner Date: 2 / 2 / 2012 month day year'."
    },
    {
        "candidate_id": "9173a4dc04e00f58b665",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'analysis process A notice of intent to prepare an environmental impact statement was published in the'."
    },
    {
        "candidate_id": "00e3302f6148f1cf02c0",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "83164787ce0d2e60454d",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'public comment during a scoping period to help identify issues and concerns that should be considered'."
    },
    {
        "candidate_id": "df19401b1c5772597d6f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "288cf628cba6a752b131",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'of the United States. Application submitted on 04/18/2017 File closed by the COE on 07/28/2017. Coordination'."
    },
    {
        "candidate_id": "fd34d2f65df2d08131fa",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Rebecca Ahlgren On January 29 2016 the Lewistown BLM Field Office received completed transfer of grazing'."
    },
    {
        "candidate_id": "f808acf4ae194178ebc0",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'the Permit Term On October 1 2018 the Service received an application from LCRA TSC for'."
    },
    {
        "candidate_id": "fbbf4122e43d866d53a2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "91c5c12ce13008645794",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "628974189a6d0192e156",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "72df742473ca8a25a9f0",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "2a19e5c48ba13680d6d3",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Flagstaff Arizona and the deadline for submitting comments as August 28 2009 It included a description'."
    },
    {
        "candidate_id": "6934d7b51b2cb25d6850",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c0c6f89704238d84e62a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "167fab90865bd3158715",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0346cae394aa86272a9c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "067317f511b768ec1ec9",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing lease On October 26 2020 the Lewistown BLM Field Office received completed transfer of'."
    },
    {
        "candidate_id": "d98443931c916cc3c17b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5be660586e82f5840324",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cf89fe82178a8ecc7d71",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'letter was also sent on July 30 2010 to those on the project mailing list updating'."
    },
    {
        "candidate_id": "99cb16aca2507ed7fb56",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Gary Covatch Digitally signed by Gary Covatch Date'."
    },
    {
        "candidate_id": "9faa0d22fba5a628946e",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Gary Covatch Digitally signed by Gary Cavatch DATE'."
    },
    {
        "candidate_id": "4cf216385c08f2dbf2a5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'decision was signed by Forest Supervisor Kevin Martin on May 18 2009 In July 2009 the'."
    },
    {
        "candidate_id": "c9701e8495a8c9819989",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'public meetings and extended comment period was mailed on 28 October 2013 The postcards also included'."
    },
    {
        "candidate_id": "aff70fc9a5158fe9930a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5f9b078319d2cbcc1d23",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b00e722d21b29791b65c",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'water contamination and tribal treaty rights Duckwater Shoshone 02/24/2023 Email from BLM to Chairman Warren Graham'."
    },
    {
        "candidate_id": "93a3492d5fc35524723d",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Forest Service was open between April 29 2016 and May 31 2016 and for Placer County'."
    },
    {
        "candidate_id": "bbcba1455a9b00a788f2",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Paiutes Pahrump Paiute Tribe non- federally recognized Timbisha Shoshone Tribe The BLM consultation for the Searchlight'."
    },
    {
        "candidate_id": "2ff9ef6364119c133b6f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: RICHARD BAKER Date: 8 / 13 / 2019 month day year'."
    },
    {
        "candidate_id": "fa557e73b62a992ec55d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ecb88f206a25ac4f75b5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: NEIL KIRSCHNER Date: 02 / 01 / 2021 month day'."
    },
    {
        "candidate_id": "f124aa6f3d78a4eaa319",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c68cded77dd15801ae73",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'notice published) 25 Record of Decision rescinded FHWA issued notice that the ROD is rescinded'."
    },
    {
        "candidate_id": "fa075ef2fffab541078c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ec7b80d6b92d742daf30",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "550b1691b4bef842f372",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "dc1ae3f370331fb9d9a8",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'extension of the review and comment period and the public review/comment period was extended to September'."
    },
    {
        "candidate_id": "8e67ab7994ae5c6cfe78",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'and Department Actions September 2008 Keystone filed an initial Presidential Permit application requesting authorization to build'."
    },
    {
        "candidate_id": "78ab89714613cf20a90e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c5ea5918ac89e6e86270",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'letter to Delfin LNG on September 18 2015 That letter commenced a regulatory stop-clock effective September'."
    },
    {
        "candidate_id": "262c8a1f6041ab716586",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cde322981a019e85a211",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'across Federal lands On September 2 2021 Neil Baumann FEATURES AND/OR submitted a SF-299 application to'."
    },
    {
        "candidate_id": "39cad87ba72d483896cf",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Gregg Sawl Digitally signed by Gregg Sawl cn=Gregg'."
    },
    {
        "candidate_id": "2dc056c1983d8e284c81",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ffeee86ff622955ac051",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'river In response to a June 29 2023 follow up telephone call from Commission staff the'."
    },
    {
        "candidate_id": "c1cb8f19e516feabc8b9",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Statement 244 The initial public comment period ran from November 10 2016 until January 13 2017'."
    },
    {
        "candidate_id": "adb5d36fb243920692a2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cb75c452af8b35b42e58",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7bd14746967fa160be2d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "dd433d7d63e052f45d06",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7444a75c649a667b3d57",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Federal Register on June 15 2018 The comment period closed on July 30 2018 All comments'."
    },
    {
        "candidate_id": "50d1741156e655a1b3b2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f34d42a013044942076e",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'status. Timbisha Shoshone 11/09/2023 Field consultation/Project Area visit including BLM and Ioneer 5.3 Cooperating'."
    },
    {
        "candidate_id": "fadecebb7051766ca69e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'study area The official scoping comment period ended on November 11 2010 however comments received after'."
    },
    {
        "candidate_id": "95908e75e46ede7a7784",
        "label": "neither",
        "notes": "Neither: consultation date, quote '106 review ML13191B089 8/8/13 M Wong NRC to R Nelson ACHP Request for scoping comments/notification of'."
    },
    {
        "candidate_id": "662009608a4d5b74a425",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "657eb3b0e12111a72eae",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e5b6982c5c49550d7a2d",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'provided written comments Commenting Entity Date Filed National Marine Fisheries Service December 4 2014 Pacific Gas'."
    },
    {
        "candidate_id": "2638922c53e00303dfa6",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'for the submission of comments concerns and issues related to the environmental aspects of the Project'."
    },
    {
        "candidate_id": "758415afd9d30ed9e6d9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c6db88cffb3cee093b7b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Fisheries Review and consultation regarding state-listed threatened and endangered species Consultation initiated May 1 2020 and'."
    },
    {
        "candidate_id": "188122886e6ec108972f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BRIAN O'PALKO Digitally signed by BRIAN O'PALKO Date'."
    },
    {
        "candidate_id": "bf073b3a20d08421fd96",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'responses Wednesday August 02 2017 ID Team/Required Reviewers will be determined at scoping meeting or as'."
    },
    {
        "candidate_id": "43f80d67ef3f8b7d7653",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "eb34e08d8b5f95188732",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "42e5e8a81b5d600764a3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "54ae7d46d2d49cc9a8d5",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Scoping Meeting 1 March 2021 Scoping Meeting 2 and August 2021 Public Meeting This was an'."
    },
    {
        "candidate_id": "1cf6606bb81298e58cca",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Kirk Gerdes Date: 08/20/2015 month day year NEPA'."
    },
    {
        "candidate_id": "f168093df1dfbaed754b",
        "label": "neither",
        "notes": "Neither: construction/activity period date, quote 'Facilities during Project operation We discuss these facilities and activities in our cumulative impacts analysis in'."
    },
    {
        "candidate_id": "011e7b1e9f9e185f0976",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "65d6adabc98a6e3bf665",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JASON HISSAM Date: 04 / 14 / 2022 month day year'."
    },
    {
        "candidate_id": "a4c934a08966b4c86e6e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ee39fa4c199d909dc47f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "44c22e67a08599e87060",
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
