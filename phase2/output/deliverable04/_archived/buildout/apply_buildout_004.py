import pandas as pd


LABELS = [
    {
        "candidate_id": "abc6b5d96b3f62c5c09a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9e0e91ee5aae90847b09",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote '0 75 Miles 1 40 000 Date Created 1/23/2018 Created By mpereira NAD 1983 UTM Zone'."
    },
    {
        "candidate_id": "44c2ec4a237c128e214d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "782d16acfb9e6919bbd9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c4d3f36789b2ee47343b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "2cd1a72e513d919c1648",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "2d39e1df3b4ee78e7372",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Vito Cedro III DATE: 02 / 09 / 2011 month day'."
    },
    {
        "candidate_id": "b1b238377154f5e8efc2",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'EREF had changed On April 16 2010 Argonne National Laboratory Argonne on behalf of the NRC'."
    },
    {
        "candidate_id": "5f538332f26f99a893ed",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6e2c646566a158753389",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'received in response to the NOI and prior to issuance of the EA were addressed in'."
    },
    {
        "candidate_id": "7ea1883d8447aca8a205",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'consulting parties Agency Scoping Meeting An agency scoping meeting was held on March 17 2015 in'."
    },
    {
        "candidate_id": "c74472decfe98ec35a70",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5f9aa2384255ddb40c86",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "acca4e967efcf9a0efe4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "91bfd483fc365fd2b0cf",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "fa6ee42dd5ba6a8cce7a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8bf1af8d3338b1242723",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'interested parties in late January 2003 On February 7 2003 the Environmental Protection Agency EPA published'."
    },
    {
        "candidate_id": "9e0418e8992e192cfdde",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'NESE 1 10 000 Miles 2 Date Created 5/5/2017 Created By mpereira NAD 1983 UTM Zone'."
    },
    {
        "candidate_id": "a4d48457d13397b1c944",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: RICHARD DUNST Digitally signed by RICHARD DUNST Government'."
    },
    {
        "candidate_id": "0d52687c53ff4b97fec7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1790f7494e48d6a91340",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'a second request for public comment This second comment period was open through February 15 2000'."
    },
    {
        "candidate_id": "5b9b9698d3df4b115957",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'requesting that the previous suspension of operations and production SOP on the above referenced leases be'."
    },
    {
        "candidate_id": "a225ebfaf00804335621",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Protection Agency EPA Notice of intent NOI to prepare EIS published April 21 2020 Draft environmental'."
    },
    {
        "candidate_id": "e0257ff38e8a1e48c92b",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Meeting \u2013 June 14, 2018 \u2022 Forest Supervisor call to Nez Perce Tribe Chairman \u2013 April 26, 2024'."
    },
    {
        "candidate_id": "12c9c5d7906a5f11a7b3",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'February 22 2008 Peoria Tribe of Indians of Oklahoma Response to notification of EO-WB project B-14'."
    },
    {
        "candidate_id": "93ee0f0b38073212218f",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Positive Declaration on September 18 2015 Scoping Along with its issuance of a Positive Declaration HPD'."
    },
    {
        "candidate_id": "037a677f5ef5d1b2239b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: DATE: 11 /06 / 2009 month day year NEPA Compliance'."
    },
    {
        "candidate_id": "193425b341bfa03bf3fa",
        "label": "initiation",
        "notes": "Initiation: FERC/application notice, quote 'and other meetings On October 10 2014 the FERC issued a Notice of Application NOA announcing'."
    },
    {
        "candidate_id": "4f510f3a7cfd91a88327",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'RSFO Date: 5/17/17 Decision Record Bureau of Land Management Rock Springs Field Office'."
    },
    {
        "candidate_id": "29891ee3fa854445f290",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b64d858b52db7a6b6380",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'submitted structures The SHPO indicated that a letter would be sent WAPA in the near future'."
    },
    {
        "candidate_id": "9638f1fd0e04a2b7c529",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On November 29 2016 Skylar Nielsen on behalf of Bio Lite Camp stoves filed'."
    },
    {
        "candidate_id": "487f204199114fd70411",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d219af1d504aa40af211",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date 1/29/24 Comments Authorized Officer [ ] 1/30/24 -Acting ==End of OCR for page 9=='."
    },
    {
        "candidate_id": "d01e91802ee0410845d7",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'concerns raised during the scoping process Review of Draft EIR/EIS A Notice of Availability NOA for'."
    },
    {
        "candidate_id": "68fc76dbfa939e9f84d2",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'environmental surveys On January 31 2020 the Commission issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "a267a63b38b76473fc80",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JAMES POSTON Date: 02 / 13 / 2023 month day year'."
    },
    {
        "candidate_id": "2f3834dd6205f1c2e2fe",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: DATE: 03 /25 / 2010 month day year NEPA Compliance'."
    },
    {
        "candidate_id": "64c878d59d4732180b5a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: MARY SULLIVAN Date: 04 / 04 / 2018 month day'."
    },
    {
        "candidate_id": "93c6f194042b607445e3",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote '4400 www.ca.blm.gov Date Prepared: 5/1/2009 Project: Willis Ridge Review of Extraordinary Circumstances'."
    },
    {
        "candidate_id": "cc682ec3217f0b2a7490",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: STEPHEN HENRY Date: 09 / 15 / 2016 month day'."
    },
    {
        "candidate_id": "9d1f1461276e188e8f67",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '72827 Proposed Action On 1/31/2019 the Bureau of Land Management BLM received a right-of-way ROW application'."
    },
    {
        "candidate_id": "cf114ed1ad7f20b8baa9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c5a43b7400e08123a084",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: MARIA REIDPATH Date: 08 / 25 / 2022 month day'."
    },
    {
        "candidate_id": "65cc6bef93c37b198a84",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c20b877d4a4e477b84f6",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: ADAM GOLUBSKI Digitally signed by ADAM GOLUBSKI Date'."
    },
    {
        "candidate_id": "841e9e43c87e85a3d517",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'www.blm.gov/california Date Prepared: 6/27/2017 Project: Base Map Exhibit A: Imperial Irrigation District'."
    },
    {
        "candidate_id": "4d8e6a73efe647bb88ea",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ee8d2b1e01d3211548c3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6cdfc3186db6c9806a0e",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'comments with FERC On July 23 2015 FERC issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "3a593f5579ce3d7990a0",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'submitted to the BLM Authorized Officer, and will include a detailed summary of the number'."
    },
    {
        "candidate_id": "3244f02cc710a2c2106a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6dd6713d23a0caf42ad0",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ecb95d99fd40b8026456",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Draft EIS ML24115A306 04/25/2024 J Moses NRC to A Reider Flandreau Santee Sioux Tribe Request for'."
    },
    {
        "candidate_id": "fb36476af0d70450c927",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7aef3fa15cee51e6cdad",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Culture and History NHPA Section 106 Review and comment on the project and its effects on'."
    },
    {
        "candidate_id": "18316036a4b4465e0b17",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'on May 27 2022 On July 7 2022 FERC issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "77eed97676fdf225c368",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "83dff11864fd59813239",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1b0fc1dcf132ee217c8e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d09a7e5ac94e0f3a53f4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1343aa351180ae468567",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'archaeology survey On October 17 2018 the WV SHPO provided comments related to archaeological and architectural'."
    },
    {
        "candidate_id": "c7b59eb7d983cd3b5d39",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: DATE: 09 / 03 / 2010 month day year NEPA Compliance'."
    },
    {
        "candidate_id": "8ab5afd3b8cd83d443e0",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8798b7272c514776c287",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Impact Statement On April 28 2021 APHIS published a Notice of Intent NOI to prepare an'."
    },
    {
        "candidate_id": "d8d2e0b5afd90322dc75",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ed7ad2609f298d8fa263",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote '2006 Page 6 Company Notice of Intent Submitted Project Order Issued Application Submitted Filing Date Application'."
    },
    {
        "candidate_id": "c3b8447e6f6839a5469f",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'was invited to submit comments on the scope of the planning process and potential alternatives through'."
    },
    {
        "candidate_id": "3b82fd79d86fa2594579",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8f12ba2d377425dd1112",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'comment period began on December 29 2012 and BPA accepted comments on the project until March'."
    },
    {
        "candidate_id": "dd5aab8534ec51641c33",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bf6ef22feb6ad1722373",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'form February 10 to April 27 2006 on the DEIS Seven comment letters were received A'."
    },
    {
        "candidate_id": "67f6f85668ff381599ba",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Brett Aristegui DATE: 11 /16 / 2009 month day'."
    },
    {
        "candidate_id": "76d355ee0140a136608c",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'phase of the project On March 20 2007 a revised NOI was published to advise the'."
    },
    {
        "candidate_id": "b2f91235382c5dee53b8",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'intervene Intervenors Date Filed Washington Ecology May 1 2012 Washington DFW May 4 2012 Washington DNR'."
    },
    {
        "candidate_id": "cc08cd6224d5ba76d3ed",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0e996736fda0e40d358e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Federal Register The public comment period was extended to August 31 2015 based on a request'."
    },
    {
        "candidate_id": "519834e121bfff51e77d",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'of an existing road right-of-way originally granted on February 10, 1999 under the Federal Land Policy'."
    },
    {
        "candidate_id": "fe54cc76941e6c1840b5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "420ce8dc0c84eb18d0d3",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'www.blm.gov/california Date Prepared 09/25/2023 Project: MyProject No warranty is made by the Bureau'."
    },
    {
        "candidate_id": "a9f0523d09e64c47e815",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2020 After subsequent consultation with USACE on May 6 2020 the FAA submitted a request for'."
    },
    {
        "candidate_id": "a5ad8663117880fa31db",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "055166b3aca4a70de620",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b12601dab24645b359a9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "62eef14835949186ee6d",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'BUREAU OF LAND MANAGEMENT Decision Record for Categorical Exclusion FLPMA Road ROW MTM-109823'."
    },
    {
        "candidate_id": "d81b7096d9bab891cfcf",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Accession No ML110060289 December 18 2010 Transcript and Video Recording of the People s Hearing on'."
    },
    {
        "candidate_id": "157a7602cbb5418df7de",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'environmental review process On January 27 2017 the Commission issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "ebd8098d92b8431d14ac",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "03883f5091f2aa09e6cb",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'transmission line design August 4 2008 DOE issued Federal Register Notice of Intent NOI to Prepare'."
    },
    {
        "candidate_id": "d7f8107dd3d8a2085fb1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e1ff0b933b41f9e1b8ea",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Aaron Yocum Date: 09 / 10 / 2013 month day year'."
    },
    {
        "candidate_id": "aca3af9578b05849406f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "32d9989bf703d3a3f955",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'Notifications 2 4 Scoping Period and Meetings The scoping process was conducted in accordance with NEPA'."
    },
    {
        "candidate_id": "d7adb18c795d608f774c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "36f545932efdca9ebd2a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "541822548c110d622662",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "19ca1aa171cac5a4de09",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: William O'Dowd Digitally signed by William O'Dowd'."
    },
    {
        "candidate_id": "d3f97d2e4d38814fa7a5",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Solar The Applicant submitted an original ROW application to BLM on October 24 2019 as Taurus'."
    },
    {
        "candidate_id": "f305b0e8766bccb80b33",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "47418c20f4ed35ef073d",
        "label": "initiation",
        "notes": "Initiation: FERC/application notice, quote 'workplans from Tennessee June 14 2022 Letter from FERC inviting tribe to participate in FERC s'."
    },
    {
        "candidate_id": "4aebaf2ac3062e7b25ca",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e3f2efe5767ddfdaddc5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a60a1c5df6948155300f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "42e9a9329af80d69790c",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'the comment period was January 18 2007 In preparing the Final EIS DOE considered all comments'."
    },
    {
        "candidate_id": "4b6418bfdd16e131ae74",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e9b68bb23ddbc8fc05de",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '2 The Applicant then submitted a revised right-of-way application and preliminary POD on June 21 2010'."
    },
    {
        "candidate_id": "579a43c7b8fafc4e8772",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'application is acceptable October 5 2018 and 15 days after the Final EIS is published DEQ'."
    },
    {
        "candidate_id": "cac435020b2f350b67cc",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Hunter Seim Assistant Field Manager Special Status Plant Species 11/17/2022 Brandon Voegtle'."
    },
    {
        "candidate_id": "cea01923f717132781de",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d3dfc8a622693862c14b",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'from Fort Belvoir The comment period ran from September 12 through November 11 2014 On Tuesday'."
    },
    {
        "candidate_id": "ade243853506e3017912",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: sricha DATE: 12 /03 / 2009 month day year NEPA'."
    },
    {
        "candidate_id": "9e3af091adbf426cf948",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'next 180 days from November 3 2015 This further direction will be incorporated into the Final'."
    },
    {
        "candidate_id": "4d661b54347602c18507",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: WILLIAM O'DOWD Date: 09/13/2017 month day year'."
    },
    {
        "candidate_id": "121293fd2ec64c981f88",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ebd84193d8edc3be6203",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'issued June 26 2018 and expires on December 26 2019 An Amended Request for temporary structures'."
    },
    {
        "candidate_id": "81077a369fd68e103e71",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'in the East Oregonian September 2015 A public meeting with responsible officials was held in November'."
    },
    {
        "candidate_id": "8e79e399ed5d08bfd104",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Description of Project On July 21 2017 NV Energy submitted an application for a 10' wide'."
    },
    {
        "candidate_id": "bd81bcd75fb1922d8f3a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bf264ad680950e4f2adf",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'an extension to the scoping comment period USACE extended the comment period to May 31 2016'."
    },
    {
        "candidate_id": "be360b1418dcf4b15f4d",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Office ND SHPO on March 18th 2020 with request for concurrence on a determination of No'."
    },
    {
        "candidate_id": "6a4fc338e133b7e1c30d",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing permit On April 18 2021 the Lewistown BLM Office received completed transfer of grazing'."
    },
    {
        "candidate_id": "1debb9fc722b21a45afd",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: ROBERT VAGNETTI Date: 07/24/2018 NEPA Compliance'."
    },
    {
        "candidate_id": "e0619765b257290369e6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d214373d086620b3e72c",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: KEITH DODRILL Digitally signed by KEITH DODRILL Date'."
    },
    {
        "candidate_id": "05eeaa63a955932fd351",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Federal Register on August 5, 2021. On October 7, 2021, the FERC issued an NOI which...'."
    },
    {
        "candidate_id": "45789f83dc9659ed445e",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'reviewed and discussed at public meetings held on May 30 2002 in Casper Wyoming and on'."
    },
    {
        "candidate_id": "0a9c8eb2d9ed8fc6be90",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "693cca003918c5e21d54",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "10d2e3da95a44737d1b4",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Consequences Renewable Energy June 2015 North Dakota Greater Sage-Grouse Proposed RMPA/Final EIS 4-191 Alternatives were evaluated'."
    },
    {
        "candidate_id": "df48df7a169c9991d2e0",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'xxi comments filed with the Commission we issued a revised scoping document SD4 on December 7'."
    },
    {
        "candidate_id": "9af6cac1603de435d3b1",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'on June 11 2001 The comment period closed on July 20 2001 Four scoping meetings were'."
    },
    {
        "candidate_id": "1542f65c96b91d496f95",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'effects of the undertaking through the proposed mitigation measures On April 26 2022 the Georgia SHPO'."
    },
    {
        "candidate_id": "944f7a40b8b792bab422",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'to NMFS oin January February 7 2011 with a request to enter into formal consultation Based'."
    },
    {
        "candidate_id": "22037d75881581025948",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'resource concerns The NOI announcing the preparation of an EIS was published in the Federal Register'."
    },
    {
        "candidate_id": "6160a992b14b7d951f90",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c102f20d16729ccb4636",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Erin Russell-Story Digitally signed by Erin Russell'."
    },
    {
        "candidate_id": "55d2387a47cc80f7bedd",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'sponsored by Transco and on July 24 2020 we issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "6926dd2a882d19d3c0b0",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'transpired from January through March 2012 During the scoping process 13 federal State and local agencies'."
    },
    {
        "candidate_id": "ecf8eda34104d1b4a073",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'phase of the public scoping process including a call for resource information and the identification of'."
    },
    {
        "candidate_id": "f8ef006329bf9c6474be",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ad38eabc341c43167756",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'reopened scoping period closed on September 10 2012 Federal State and local governments along with other'."
    },
    {
        "candidate_id": "e34a880581aac896a00a",
        "label": "initiation",
        "notes": "Initiation: posted to ePlanning/NEPA Register, quote 'area This project was posted to the NEPA Register on 4/26/2016 Concerns or comments from the'."
    },
    {
        "candidate_id": "f572e30c908c65094d4e",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Exhibit A for a location map Description of Proposed Action On February 16 2021 this office'."
    },
    {
        "candidate_id": "0bdb0b699f9dc1832b14",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Application ML20294A483 10/30/2020 R Elliott NRC to C Bullock Chief Patawomeck Tribe Request for Scoping Comments'."
    },
    {
        "candidate_id": "bbb1bbcdb2bb02f11df7",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Scott Beautz Date: 03/26/2024 month day year'."
    },
    {
        "candidate_id": "49798fcb7865a8c23405",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'information October 4 2011 USFWS email to Jacobs species list update November 17 2011 USFWS letter'."
    },
    {
        "candidate_id": "aaebe42443fc6c39ef22",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'MARCH 3 1849 1 24 000 Date Created 1/23/2018 Created By mpereira NAD 1983 UTM Zone'."
    },
    {
        "candidate_id": "5673e151c21d41f1fb62",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: James Briones DATE: 02 / 16 / 2011 month day'."
    },
    {
        "candidate_id": "d4fa470353eabbc717b3",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'this application, which expires May 2021 and a business license from the State of Alaska. Arnie'."
    },
    {
        "candidate_id": "fd040823abd3a9a52e5c",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'lease application at a public meeting held on January 18 2007 in Casper Wyoming Each of'."
    },
    {
        "candidate_id": "44a6ef40b681655cd930",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c10e51d05e17a29fd05b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'on these documents August 16 2006 T 1 1 e Service received a request for fomlal'."
    },
    {
        "candidate_id": "8134cb964bbb19154524",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: John Terneus Digitally signed by John Terneus DN'."
    },
    {
        "candidate_id": "25f9d2bd42305d609975",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Gary Covatch Digitally signed by Gary Covath DATE'."
    },
    {
        "candidate_id": "324a4b4eac1ffb61790b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e4fc2476cdfda9e40688",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'comments/notification of Section 106 review ML13227A388 8/13/13 M Wong NRC to H Frank Forest County Potawatomi'."
    },
    {
        "candidate_id": "fab545a1c6f6544b3452",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'transportation service On July 20 2020 the Commission issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "b6725d7e1559f0841c94",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "488fbc5de27b593bb3de",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0efff5c7d1be6e87b713",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'for this project was published in the Federal Register on June 4 2021 notifying the public'."
    },
    {
        "candidate_id": "c9e21b74b18fd7634403",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing permit On March 10 2022 the Lewistown BLM Field Office received completed transfer of'."
    },
    {
        "candidate_id": "05163bf227015fded0e5",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'would be issued for a term of 30 years v1-3-2020 On February 18 2020 Qwest filed'."
    },
    {
        "candidate_id": "679736576e43e93698ff",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JESSICA MULLEN Digitally signed by JESSICA MULLEN'."
    },
    {
        "candidate_id": "aec6efbaa9f62ebe8990",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Date: 07 / 16 / 2015 month day year NEPA Compliance'."
    },
    {
        "candidate_id": "2a85a7b6739dd582acd9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "570fbe55f46584ae125f",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'Management BLM in an existing right-of-way ROW held by Reclamation On May 18 2018 AZ Solar'."
    },
    {
        "candidate_id": "afe0ead0c520b850e833",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'decisions that existed prior to the adoption of the Western Solar Plan ROD The Palen Solar'."
    },
    {
        "candidate_id": "11d3e0e485ccc350d886",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Casper Field Office May 2000 FINAL Environmental Impact Statement for the Horse Creek Coal Lease Application'."
    },
    {
        "candidate_id": "44a59f8f042fb7202afe",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1a9e7bb653d2b326121a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "33a15a98a2bca5b088e6",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'EnergySolutions submitted a license application to the State of Utah to allow permanent disposal of DOE'."
    },
    {
        "candidate_id": "322b02e560ac7f86975a",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'the public to attend a public meeting to learn more about these alternatives A public scoping'."
    },
    {
        "candidate_id": "91075f23d41cb559e01e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ae91e6454fcfef3f96b9",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Lateral is not needed On August 2 2016 the FERC issued a Supplemental Notice of Intent'."
    },
    {
        "candidate_id": "c9e4060d6a5537301df7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d4c1ce994c4ab960297f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "298f6bb9aed980eedf44",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'and NEPA process On June 9 2017 the FERC issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "49b71b1bebbe1cf2618f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Gary Covatch Digitally signed by Gary Cavatch DATE'."
    },
    {
        "candidate_id": "36d863f93b145d474737",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Accordance with the 17 PEIS ROD \u2013 May 14, 2014 update, 2016 Final PEIS for Vegetation'."
    },
    {
        "candidate_id": "d4a32d1220c2a9cfef66",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "707d845dd0551ee6d1d3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "de6c5f63d94602d53438",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c649c5b9ce0fcdfab851",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Scoping Meeting NOI On August 2 2016 the FERC issued a Supplemental Notice of Intent to'."
    },
    {
        "candidate_id": "b631f69d0407aca739c8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "fc883f80927cc8331f05",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7c2a7c2b562df9ac98d2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "74afdbad41a018710b20",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "591f34822aee3f68a7d6",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'received in response to the NOI and prior to issuance of the EA were addressed in'."
    },
    {
        "candidate_id": "79ae4f79c30322c206c5",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'received 2013 for a term through December 31 2013 Permit to Mine Application submitted January 2011'."
    },
    {
        "candidate_id": "94a83e979c449cafded8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6386528c8b96258d1c40",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "db157c97fef79284ca86",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BRIAN O'PALKO Date: 09 / 23 / 2022 month day'."
    },
    {
        "candidate_id": "2688ed5946462875e125",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'brief summary of the public scoping meetings The scoping comment period began on December 21 2012'."
    },
    {
        "candidate_id": "5c16564e9ec7847fcce2",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'this application were previously completed under Nevada State Water Rights Permit 48622 filed December 6 1984'."
    },
    {
        "candidate_id": "e02ad87b87279924cad4",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Islands to the Guam SHPO 8/26/2022 The Guam SHPO submitted a response to the DAF s'."
    },
    {
        "candidate_id": "8b5cadc213806c93bfcb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e88820eb3bec22b217f4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "618d0e32b7e696f2a470",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'C Drake 4/16/18 JA 4/21/16 CSB 4/19/2016 LP 4/18/2016 EM 4/27/16 To be filled out during'."
    },
    {
        "candidate_id": "02df17b05f334498d6ba",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3ac3f6fc51a9d2092d82",
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
