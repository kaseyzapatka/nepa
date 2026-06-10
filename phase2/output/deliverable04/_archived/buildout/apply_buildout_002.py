import pandas as pd


LABELS = [
    {
        "candidate_id": "81bebf1d61b29c2e1204",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "378f6722fa4137414f74",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Brian Dotson DATE: 02 /12 / 2010 month day year'."
    },
    {
        "candidate_id": "aab0556b9839b7525f9f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c3749d18ae11eeeabe66",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'and Wildlife Service USFWS U S Environmental Protection Agency EPA the Sierra Club and Healthy Gulf'."
    },
    {
        "candidate_id": "6b71ce76ac5f44cf1fdb",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Louisiana Thursday December 16 2004 NOAA Science Center 1301 East-West Highway Silver Spring Maryland Public scoping'."
    },
    {
        "candidate_id": "9a082cd660bc290b1fd4",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'during this timeframe In October 2021 FTA invited the involved tribes the Oregon SHPO and the'."
    },
    {
        "candidate_id": "60a881e0b74274c2585b",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Forest County Beacon published the scoping on May 18 2015 and DEIS on January 8 2018'."
    },
    {
        "candidate_id": "2358180db5e69bf647a7",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'protection project a Notice of Intent was published in the December 31 1987 Federal Register The'."
    },
    {
        "candidate_id": "399e6e0289a338a68e85",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "410be25931162aac6b66",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Francisco District Corps received a Department of the Army permit application to construct a solar photovoltaic'."
    },
    {
        "candidate_id": "6925fc74979f6236ce5b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d7c32c0d4e8ef89db763",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a69208fdb2ef79d1d082",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Arun Bose DATE: 03 / 17 / 2010 month day year'."
    },
    {
        "candidate_id": "56005742b3e0987b1aaa",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'December 30 2021 when the Notice of Intent to prepare the EIS was published in the'."
    },
    {
        "candidate_id": "f47bfd0491bf53747754",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e56dafde68a9ca8dbf5d",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2009 Outgoing Letter SHPO Archaeologist Phase I No Adverse Effect SHPO Letter 5/5/2009 Incoming Letter SHPO'."
    },
    {
        "candidate_id": "addf2eadc2a85d765e38",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'USDA-FS 2013b A Notice of Intent was published in the Redding Record Searchlight on February 27'."
    },
    {
        "candidate_id": "b5c96d7235a86ffbc336",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Permit Approval Consultation Status Federal FERC Certificate of Public Convenience and Necessity Application submitted February 2020'."
    },
    {
        "candidate_id": "48d5c1eb94ab51ae4b13",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "dd39f2dfbff74af53ab8",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'PROPOSED ACTION On January 9 2010 PacifiCorp submitted an application SF-299 requesting a renewal of an'."
    },
    {
        "candidate_id": "a8bbc408cb5ce00b444f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Linn Caleb M. Hiner Field Manager 11/07/2016 Date Administrative Review or Appeal'."
    },
    {
        "candidate_id": "5a935dadc5ace3061f3e",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'overhead transmission line Right-of-Way ROW The ROW is held by CS Mining LLC and encompasses 28'."
    },
    {
        "candidate_id": "18359d23901c6cda88a0",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'Project The 30-day public scoping period began on May 3 2018 and lasted through June 4'."
    },
    {
        "candidate_id": "449ee7a4498ad8e017a4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7c3f63b59708c4481099",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "330c43120511ccc83a65",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'declined participation. 2/17/2016 Public Scoping Meeting Mountain, ND 13 Comments/comment forms received from'."
    },
    {
        "candidate_id": "98ba83b3a5e0bfd43cbf",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "98c9dc258235c5375148",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a0a0954b0774375d8be1",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'base is available 2 1 Comment Period Extension Let me take this opportunity to formally request'."
    },
    {
        "candidate_id": "d451bdefb45767102773",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'copy of the Final EA until May 28 2003 even though we submitted written comments on'."
    },
    {
        "candidate_id": "e172a51f9179792e2dc4",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'August 18 2023 On September 18 2023 we issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "f11fac572f91a0996b88",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e5e533209e76a182d3e1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ffe6d9c2315ca710c7c4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "674777ace6cdb90bb48a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BARBARA CARNEY Date: 08/05/2019 month day year'."
    },
    {
        "candidate_id": "ae1b72d1267a3333415e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'scoping Period The comment period began on November 24th 2017 and ended January 23rd 2018 A'."
    },
    {
        "candidate_id": "f78c7ff1f1f1c4551c69",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d5361a7d10ec08117439",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Erin Russell-Story DATE: 09 /30 /2011 month day'."
    },
    {
        "candidate_id": "7f1dcbdbd81fc57f2422",
        "label": "neither",
        "notes": "Neither: construction/activity period date, quote 'sites are located during construction Tribal consultation will continue throughout the NEPA and Section 106 compliance'."
    },
    {
        "candidate_id": "d8fb53bf9b11eab0b850",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ba53f5c28eb40467f7e7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "24cd8b57a963e23385e7",
        "label": "neither",
        "notes": "Neither: comment filing date, quote 'intervene: Intervenor Date Filed Oneida Narrows Organization October 22, 2014* Greater Yellowstone Coalition November 14, 2014*'."
    },
    {
        "candidate_id": "befb7fd0efcf9979e7ce",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'information to POW LAT December 12 2016 Public scoping meeting in Thorne Bay 4 30-6 00'."
    },
    {
        "candidate_id": "38855a5cad86115952f4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1d98a157a4074016c976",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4fab1e73701fef6df8be",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote '2015 respectively The scoping comment period closed on August 28 2015 At this time MARAD and'."
    },
    {
        "candidate_id": "afd27ed1a237a82846c2",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'IR 242 and IR 245 received December 13 2024 BMOP Blue Marlin Offshore Port LLC 2024g'."
    },
    {
        "candidate_id": "1b6e82b457e359e67e99",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: MICHAEL FASOULETOS Date: 08 / 29 / 2019 month'."
    },
    {
        "candidate_id": "410c25dced23f93ff390",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Andrea McNemar Date: 04/23/2013 month day year'."
    },
    {
        "candidate_id": "a8f0fbedbf7bac1b5081",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6d68e615cf8109b998de",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c880f960e3d686393f23",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Yuma Quechan Indian Tribe to visit potential sites of concern that were identified within the APEs'."
    },
    {
        "candidate_id": "558b1dbe1929f1bb5324",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ac8f4935bc44a8f2a61a",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'RSFO Date 3/14/2018 Decision Record Bureau of Land Management Rock Springs Field Office'."
    },
    {
        "candidate_id": "5adb0138eafa15e159a4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'U S c 824a e On December 14 2005 the Department of Energy DOE received en'."
    },
    {
        "candidate_id": "fbc8a0f47c5b376fcbfb",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'PUBLIC INVOLVEMENT Scoping Period Public scoping for WAPA s Proposed Action was initiated on January 12'."
    },
    {
        "candidate_id": "5fa85feb71b927f9cef3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6125da472876d405fe53",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6f6a045b72cb217e5025",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6fd8690f2b87ae10cdd3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c259e5eeb0579bbb2a27",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'National Park Service State Date Created: 2/21/2019 Created By: mpereira Vicinity Map Private NAD 1983'."
    },
    {
        "candidate_id": "1d41af518c1b58a46103",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'the Scoping Period was published in the Federal Register on March 8 2013 extending the scoping'."
    },
    {
        "candidate_id": "51cb38b83e12b456d0a6",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'Idaho Power Company Date created 9 May 2023 Created by Matt Kohtz L2 L1 6N 1W'."
    },
    {
        "candidate_id": "0487f11b0fe1aa85994c",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'pre-filing period On September 9 2013 the FERC issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "4a7131fc69ac85037657",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f1510497ce65f09a92d4",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Patcharin Burke Digitally signed by Patcharin Burke'."
    },
    {
        "candidate_id": "f4b795ac7e6b38fa6c14",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "77e1b2fd446a06c153a5",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'comment period opened October 26 2004 and scoping comments were formally received though March 2005 USACE'."
    },
    {
        "candidate_id": "21340c254ce3ec2d5a45",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5ef54991835abf913cc3",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'McDermitt Paiute-Shoshone Tribe the Burns Paiute Tribe and the Nez Perce Tribe of Idaho Consultation with'."
    },
    {
        "candidate_id": "4a68bb2909bcad74137f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e45f7d9f8a526b445042",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9c114702810c213928d1",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Development Team Trip Report 9/30/2010 Advisory Council on Historic Preservation Letter of Consultation 9/11/2012 NRCS Letter'."
    },
    {
        "candidate_id": "948f2e94a3e743599f9c",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'began with issuing a Notice of Intent NOI to prepare an EIS The NOI was published'."
    },
    {
        "candidate_id": "fdd56578c518563c35a8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5ea6dc89facdd2a23bc3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6edec047c17700b927c2",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'November 10 2022 and December 14 2022 During this time four public scoping meetings were held'."
    },
    {
        "candidate_id": "68267b62b607b2afcc9a",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'location of a public scoping meeting EPA published a Notice of Availability of a Draft Environmental'."
    },
    {
        "candidate_id": "f65df5cca7e5033151dd",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "418b7b7ed3f4e7fa6c1f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b1013f3aad5fd4866635",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'the closing date was extended to 25 October 2013 due to the partial government shutdown which'."
    },
    {
        "candidate_id": "af7f2f88e8faabf9036b",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Submitted During Scoping Between September and December 2021 the BLM received 26 pre-scoping submissions in the'."
    },
    {
        "candidate_id": "b71dfc515acf6dd5be82",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Biological Opinion BO On August 28 2019 FERC requested reinitiation of Section 7 consultation due to'."
    },
    {
        "candidate_id": "85ea025abb2655e40e67",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'would be used That scoping period was initiated on May 23 2014 Due to public comments'."
    },
    {
        "candidate_id": "6ecbc0e6b56538cf7218",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Service initiated formal consultation by letter to the Corps The date for the biological opinion was'."
    },
    {
        "candidate_id": "be5445586678d92dff36",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: David Szucs DATE: 08 /20 / 2010 month day year'."
    },
    {
        "candidate_id": "8ecc2933957814fb305c",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: MARY SULLIVAN Date: 10 / 10 / 2019 month day'."
    },
    {
        "candidate_id": "506f67996c573e3e38b1",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JASON HISSAM Digitally signed by JASON HISSAM Date'."
    },
    {
        "candidate_id": "cfea4666fb130662578d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8b0f8694d3e8eee69e24",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7660e4939c3960d862ee",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4e62d70169b5fcebbdd9",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'ACTIVITIES Activity Date Notice of Intent published in the national register scoping period begins May 29'."
    },
    {
        "candidate_id": "899ca44bb7aae06ef9f9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e317ebdc49dc29617202",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "924c153858ee17e6c95e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'approved in writing by the authorized officer prior to conducting any surface disturbing activities'."
    },
    {
        "candidate_id": "b87112fdb482cc53b733",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a2da971c4a2dfba77d1f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Congjun Wang Digitally signed by Congjun Wang Date'."
    },
    {
        "candidate_id": "73df81b3a1abc5743213",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'environmental review process On June 18 2013 the FERC issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "6c2665a663848a9cd36f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9d727b8e0d5d39784e94",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0a3c4535b01cba911008",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f52b6e86ef37b959670e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3cc5da08d91d3ef9b030",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8aaf69c2fc320632bea1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "26860afe779d8b974bc6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d741b04da67cae443c9a",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '13 2022 Zoom meeting March 29 2023 Zoom meeting Public scoping meetings June 12 2023 Zoom'."
    },
    {
        "candidate_id": "51c1f3cde1bddb44ee8d",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'meeting for the Project on October 3 2012 in Baton Rouge Louisiana We also conducted an'."
    },
    {
        "candidate_id": "6e6f3721039445d4707b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Request to engage with the tribe and follow up from 9/30/2021 letter sent to Tribal Council'."
    },
    {
        "candidate_id": "93808db16e116dcb8233",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4fb8ec64c10f696e3af6",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'project. The second NOI published on October 15, 2014, established a scoping process to determine whether'."
    },
    {
        "candidate_id": "d914a77b584b2215dc73",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "14780d4371fb38d98ca3",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Teresa Jones DATE: 11 / 1 / 2010 month day year'."
    },
    {
        "candidate_id": "9d53cfaf119d3b98e8eb",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote '43 U S C 1761 as amended This project was posted to the NEPA Register on'."
    },
    {
        "candidate_id": "0c5b00ad6257f34447ae",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Shoshone 06/20/2023 Field consultation/Project Area visit including BLM and Ioneer Timbisha Shoshone 06/26/2023 Meeting with the'."
    },
    {
        "candidate_id": "c9daa01dcd554d5fa82d",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'tribes regarding the BLMs consultation request in May 2013 The BLM sent letters to the tribal'."
    },
    {
        "candidate_id": "2e8eaa572390733b082e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b428cd4ee42df9be0df7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "874d364a6ca70764728e",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: VITO CEDRO Digitally signed by VITO CEDRO Date: 2023'."
    },
    {
        "candidate_id": "22eaba7ce53ee28d86f5",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '1366 loan guarantee application submitted December 30 2010 supplemental Environmental Overview documents submitted on April 18'."
    },
    {
        "candidate_id": "9975861a9ed2e1efcc1a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e7486b556e4a7a7f00f8",
        "label": "initiation",
        "notes": "Initiation: FERC pre-filing approved, quote 'PUBLIC INVOLVEMENT On August 15 2017 FERC accepted Commonwealth s request to begin pre-filing and Docket'."
    },
    {
        "candidate_id": "fed1d66c48fa1f25f7bc",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "09c1604ed62979a65ffa",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Appendix E E-3 October 28 2010 Press Release NRC to Conduct Environmental Scoping Meeting as Part'."
    },
    {
        "candidate_id": "b545ac2e0f2054963dc1",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'that time A 60-day comment period on the Draft EIS commenced with publication of the EPA'."
    },
    {
        "candidate_id": "cb39a851e3e3916326f4",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'of the Draft EIS on May 11 2018 with public meetings held on June 4 2018'."
    },
    {
        "candidate_id": "75ee00c7280a9eb5568c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "af1bb52c559456502006",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'continued with the NOS the NOI and the Supplemental NOI which were issued in September 2021'."
    },
    {
        "candidate_id": "707c695f90621655d273",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'and conditions The term of the Sand Creek Common Allotment grazing permit would run from March'."
    },
    {
        "candidate_id": "5c2153302cee5616b463",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'was well-attended On September 18 2008 the Sponsor held its regular public meeting and selected the'."
    },
    {
        "candidate_id": "913df8e83f12d8af3ccb",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Dawn Deel Digitally signed by Dawn Deel DN: cn=Dawn'."
    },
    {
        "candidate_id": "ea335c44852dd8d14da4",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Brian Strazisar DATE: 05 /06 / 2010 month day'."
    },
    {
        "candidate_id": "dfd8b1ec6e96ca8d3c21",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: MARK RENDER Digitally signed by MARK RENDER Date'."
    },
    {
        "candidate_id": "75c8d6e8229c80acbf3e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'agencies The official comment period for the supplemental notice formally closed on September 25 2016 On'."
    },
    {
        "candidate_id": "001ed4f3ba0466f596fc",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote 'shown on the attached Figure labeled as SWLRT Delineation Concurrence and PJD 2/18/2015 Figure I The'."
    },
    {
        "candidate_id": "d7c20c6d99387af206e8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "77584bd073c45dd2ad05",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'information about the public consultation process including dates meeting notes attendees count Response NOAA published a'."
    },
    {
        "candidate_id": "9bcfa79eddab8a8d0c58",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d333b172ae256532b559",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'letter to FERC dated February 13 2013 the Swinomish Tribal Community indicated that it would participate'."
    },
    {
        "candidate_id": "cc5f2d0a9121ca0b1b02",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'agencies and others on August 5 2008 It was noticed in the Federal Register on August'."
    },
    {
        "candidate_id": "e5bdf1e4adebb33d9740",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: LEI HONG Digitally signed by LEI HONG Date: 2023'."
    },
    {
        "candidate_id": "a2c050d564ae08c65915",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "76618fd93f2541f4d8a6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "66721f1b1f8a6943a009",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'number of topics On August 1 2014 the Service issued the FEIS in the Federal Register'."
    },
    {
        "candidate_id": "c8f9e69be5ffc4a6a083",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7c1440a3cc6346a435e7",
        "label": "neither",
        "notes": "Neither: consultation date, quote '17B0532-17F1029 Section 7 consultation was initiated with the FWS on March 3 2021 by submitting an'."
    },
    {
        "candidate_id": "73b68fa400789a7d33a0",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'per WDEQ-AQD AP-4809 Application Analysis for the Antelope Coal Company Antelope Mine dated February 1 2007'."
    },
    {
        "candidate_id": "b4b8ee638f1064a7fd6e",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '1366 loan guarantee application submitted December 30 2010 supplemental Environmental Overview documents submitted on April 18'."
    },
    {
        "candidate_id": "3c574977886ada21abe9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8a2eac090dd25ab1831e",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'on April 27 2017 On May 3 2017 the BLM received response from the Nisqually Indian'."
    },
    {
        "candidate_id": "218360ba34d492f78a2a",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'Received During the Scoping Period The scoping process for the environmental review of the license renewal'."
    },
    {
        "candidate_id": "cf6778f0d7772e3bdf9d",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote '7100 www.blm.gov/seria Date Prepared: 2/13/2018 Project: Cattle Capture.mxd'."
    },
    {
        "candidate_id": "adee58062670a7de24bc",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f548e949740d14b5e9c8",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: **William O'Dowd** DATE: **12 / 23 / 2011** month'."
    },
    {
        "candidate_id": "18151789f43c2a6b3b3b",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'The original 45 day comment period is due to end on December 29 2014 and I'."
    },
    {
        "candidate_id": "a036951760038f5f59fa",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'DEIS/Amendment 9 on May 21 2018 with the comment period ending on July 5 2018 However'."
    },
    {
        "candidate_id": "81b029573085863faa01",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '2019 BACK GROUND On August 7 2018 Dee Conger Revocable Trust submitted an application SF-299 requesting'."
    },
    {
        "candidate_id": "4d6d4a89dcb7e9a99c09",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'provide comments by July 14 2023 On August 4 2023 Commission staff reached out to the'."
    },
    {
        "candidate_id": "a8542c4d9bf9641cd67d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "47ed66159379fb83941f",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'State NYSDOS in its June 8 2011 Coastal Zone Conditional Consistency Certification February 28 2012 Amendment'."
    },
    {
        "candidate_id": "6314111122b2906f23aa",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3c9bd345a930c6ce1fd9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0eb4d3421b3bd18cd576",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9f608b48d4a1555df96a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d78c769418c0f5980890",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a6167aa5115561a7d170",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7c08f5d46b1056ac9513",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'regarding signed Georgia HPD Section 106 letter October 25 2022 received signed Georgia HPD Section 106'."
    },
    {
        "candidate_id": "170b5f9064ba0297130a",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'at Hult Reservoir in June 2018 with few attendees The BLM then put the project on'."
    },
    {
        "candidate_id": "e08ab50da1ada3331c8d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5e6a568bbcdf4d9a61dd",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Trevelyn Hall Digitally signed by Trevelyn Hall Department'."
    },
    {
        "candidate_id": "acfe2c128fd22bae2a50",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "98909ad78d05432e551c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "05716b73d7b8782f9a3e",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '2006 An agency/public scoping meeting was held on September 25 2006 at the Lindale High School'."
    },
    {
        "candidate_id": "d53bbcba38e28d33c34d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b543b01e933bd5a9ca96",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Raymond Lopez Digitally signed by Raymond Lopez DN'."
    },
    {
        "candidate_id": "b05c7312aa716199fa55",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "963cc71cb0923c0e63ac",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Peninsula Daily News and comments were solicited through September 6 2010 September 3 2010 The City'."
    },
    {
        "candidate_id": "ed5e8b33eaf5944702d9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Brad DATE: 02 /22 / 2010 month day year NEPA'."
    },
    {
        "candidate_id": "3b82f0b92c87962dad96",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Steven Markovich Digitally signed by Steven Markovich'."
    },
    {
        "candidate_id": "17848a567e0ca0bacb8f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "97ab40c079eaa30c2e2a",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Date of Preparation 3/18/2015 BACKGROUND On February 17 2015 Lisa Davis submitted an application SF-299 requesting'."
    },
    {
        "candidate_id": "2af5fde3304a8dd461bb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b87a42585ad5b4ec2d9a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "217213a64abe13b26ba7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e014deb121c5c3559d82",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Bayou Crossing R-77 7/23/2019 FDOT Response to USCG comments on DEIS R-78 8/29/2019 USCG Concurrence with'."
    },
    {
        "candidate_id": "a6e058af3ad07f4368c8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "45153f78c8160a7d30ba",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Organization RE Notes 3/16/2015 3/30/2015 Incoming and Outgoing email Amishi Castelli Andrew Lewis DC SHPO cc'."
    },
    {
        "candidate_id": "4966dd4f2cf47a4e2993",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Christian L Robinson Date: 07 / 30 / 2021 month'."
    },
    {
        "candidate_id": "d129bf13f880c04f4c6d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Carl Maronde Digitally signed by Carl Maronde DN'."
    },
    {
        "candidate_id": "b99acd1ec1d62ee6da99",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b53d4357924880d5166a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Brian Dressel Digitally signed by Brian Dressel DN'."
    },
    {
        "candidate_id": "7034a0736bf392069d5c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "60e004730a43b4995184",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'period which ended on July 7 2003 These public meetings were held in La Grande Oregon'."
    },
    {
        "candidate_id": "0cf1305c764201e46d2b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c5d4203112e7f9282508",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "09bbfa9a70e4fac5d2fa",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'ML24283A161 10/24/2024 NHPA Section 106 Letters to Consulting Tribes for Clinton Power Station LR ML24285A138 11/21/2024'."
    },
    {
        "candidate_id": "c1f0248c0a1dee720a40",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "972dd023d32fbe171c41",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'a map titled SF 299 APPLICATION Location of Project Proposal for Peak Power Wind LLC in'."
    },
    {
        "candidate_id": "f918fef50b0c3c8ad4a0",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'INFORMATION: Background On September 22, 2020, NV Energy filed an Application for Transportation, Utility Systems, Telecommunications'."
    },
    {
        "candidate_id": "b883326a42f105edff4e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'the EA process The public comment period began on August 23 2018 and BPA accepted comments'."
    },
    {
        "candidate_id": "954438ea79e43d567f7c",
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
