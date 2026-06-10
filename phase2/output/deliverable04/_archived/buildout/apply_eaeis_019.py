import pandas as pd


LABELS = [
    {
        "candidate_id": "558e2e3e1b0a14583ba3",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'approval by the authorized officer. ____/s/ John Hodge____________ ___2/12/2018______________________ John Hodge D...'."
    },
    {
        "candidate_id": "7359ec4ead16e4b8f943",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'SCOTT ARMENTROUT Digitally signed by SCOTT ARMENTROUT Date: 2020.04.15 08:06:09 -07'00''."
    },
    {
        "candidate_id": "29c9e52b38c741de0ce5",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-05-15'."
    },
    {
        "candidate_id": "cfc3f0275bff0a50d48c",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2014-04-22'."
    },
    {
        "candidate_id": "7caa0d4f73aae7f12c69",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2018-12-07'."
    },
    {
        "candidate_id": "b2c6692e790e9e0d41e2",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Finding of No Significant Impact 2 12/29/2014 DOI-BLM-ID-B030-2014-001-EA 1. The proposed fire will burn across'."
    },
    {
        "candidate_id": "b2d61c2b42ebceb2b315",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-11-01'."
    },
    {
        "candidate_id": "ccbf62a5feeef09e292d",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Significant Impact (FONSI) for the Proposed Action. On July 8, 2015, the Corps issued its own agency-specific FONSI'."
    },
    {
        "candidate_id": "977c99309705f70293dc",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-11-17'."
    },
    {
        "candidate_id": "868c45b640517a7141ae",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-01-20'."
    },
    {
        "candidate_id": "a4591fcfe6b6bf827b78",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Test Facility EA Finding of No Significant Impact May 2018 iv Spill Prevention, Control, and Countermeasure (SPCC) Plan)'."
    },
    {
        "candidate_id": "65a2b77a13d271a43c8b",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-10-16'."
    },
    {
        "candidate_id": "34803914477dc040d675",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-08-30'."
    },
    {
        "candidate_id": "4ff49e6774704d864093",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-06-11'."
    },
    {
        "candidate_id": "27fcd466385a1750e378",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-08-06'."
    },
    {
        "candidate_id": "0b95e7666d228b232fe8",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2013-01-24'."
    },
    {
        "candidate_id": "d6038d005d624304925e",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2018-08-10'."
    },
    {
        "candidate_id": "51a527f8725d36099b24",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2017-04-13'."
    },
    {
        "candidate_id": "fee9e2c1d20e54b36a95",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Proposed Action. The Corps verified by letter dated August 28, 2015 that the County\u2019s proposed temporary discharge of'."
    },
    {
        "candidate_id": "441aa93742d9358df366",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Management Finding of No Significant Impact Environmental Assessment DOI-BLM-CO-SOl0-2014-0025 June 2016 GCC E11ergy Exploratio11 License Applicatio11 coc 76563'."
    },
    {
        "candidate_id": "c0b9892c25f3813ec971",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2018-11-07'."
    },
    {
        "candidate_id": "bc4725959c23d8601e64",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2023-09-08'."
    },
    {
        "candidate_id": "b2788dcb6902fdaf5e92",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-11-04'."
    },
    {
        "candidate_id": "b5b986e0e080b32b67ca",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-04-14'."
    },
    {
        "candidate_id": "edfce5e37b36808dbc31",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2023-09-05'."
    },
    {
        "candidate_id": "52b98426f2cdabce9ead",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'IMPACT Simco Expansion Training Area Environmental Assessment DOI-BLM-ID-B011-2021-0007-EA June 17, 2022 INTRODUCTION: The Bureau of Land Management (BLM)'."
    },
    {
        "candidate_id": "40c0e0b33510f7fdcbf2",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'management plans, policies, and programs. /s/ Amanda Hoffman 12/4/2017________ Authorized Officer Date Amanda Hoffman Morley Nelson Snake'."
    },
    {
        "candidate_id": "c156181dd4dde542de8a",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-2197): 2024-09-13'."
    },
    {
        "candidate_id": "df1ae932d4723c8167e5",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1715): 2010-09-22'."
    },
    {
        "candidate_id": "8ab0e62d5deebbec79e8",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-01-20'."
    },
    {
        "candidate_id": "90ba73ed43af57c5bc56",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-06-12'."
    },
    {
        "candidate_id": "2503bbce8edd0f02c96b",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Implementation Plan Finding of No Significant Impact DOI-BLM-CA-C05000-2016-0006-EA November 2016 FINDING OF NO SIGNIFICANT IMPACT The Environmental Assessment'."
    },
    {
        "candidate_id": "7300c3ad5fcddd9c480b",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'interest favors granting the stay. /s/ Viola Hillman 06/19/2014 Viola Hillman, Tucson Field Manager Date Attachments: Finding'."
    },
    {
        "candidate_id": "e6128752cea14ac06ade",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2023-08-23'."
    },
    {
        "candidate_id": "516a3f18a941eb56fd6d",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-03-21'."
    },
    {
        "candidate_id": "9399e63b572654d6e5ea",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2024-06-03'."
    },
    {
        "candidate_id": "06b9dded8bfd5ed04e14",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote '/s/ Jim Tharp for Michael Courtney Field Manager 1/23/2013 Date 5'."
    },
    {
        "candidate_id": "0b473b746bdccc113f55",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'To: 4100 (IDT030) P 91013 CERTIFIED-RETURN RECEIPT REQUESTED October 9, 2015 James Grant 1934 East 400 South Hazelton,'."
    },
    {
        "candidate_id": "922e55b7f440767a5938",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote '1201 Bird Center Drive Palm Springs, CA 92262 July 6, 2022 TIMOTHY GILLOON Digitally signed by TIMOTHY GILLOON'."
    },
    {
        "candidate_id": "0481352716efd484fb18",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-04-12'."
    },
    {
        "candidate_id": "6d1c17280ac4c62ed682",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-04-24'."
    },
    {
        "candidate_id": "1edb167ad0a681418a89",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2024-02-01'."
    },
    {
        "candidate_id": "f1b1df79ee245dc69ef5",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2018-08-27'."
    },
    {
        "candidate_id": "a3dfa4649a2ada66e9e1",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2023-06-20'."
    },
    {
        "candidate_id": "69a592180e3cf548c199",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'is in compliance with the 1978 NEPA Regulations. January 2021'."
    },
    {
        "candidate_id": "920f09138c0f5aeb97f5",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1690): 2010-04-20'."
    },
    {
        "candidate_id": "a8ffc2d241bb879dd58c",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'LAND MANAGEMENT Finding of No Significant Impact DOI-BLM-CO-S012-2023-0023-EA January 2024 Canyons of the Ancients Hazardous Fuels Treatments Applicant:'."
    },
    {
        "candidate_id": "bc1afcf64557544124d9",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2017-04-30'."
    },
    {
        "candidate_id": "b2aaf0ce1fdb28f1f6ce",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Department of Energy Finding of No Significant Impact 8 November 2002'."
    },
    {
        "candidate_id": "59a2a73f280fcfa883d1",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-04-03'."
    },
    {
        "candidate_id": "0b000dad0fa43ab130ee",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-07-03'."
    },
    {
        "candidate_id": "2159e0b55c2dd0501488",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-11-12'."
    },
    {
        "candidate_id": "231a8ef9414ac2b315ae",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'FONSI for the ULP 2 July 2007 On the basis of the information and analyses'."
    },
    {
        "candidate_id": "6be91556a02ea4d987ca",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-06-24'."
    },
    {
        "candidate_id": "34181d7f004a275b24f0",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Finding of No Significant Impact 3 10/23/2014 DOI-BLM-ID-B030-2013-009-EA The actions and practices analyzed in the'."
    },
    {
        "candidate_id": "6bf73ae8638b3a12517a",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2018-07-18'."
    },
    {
        "candidate_id": "141c66c11deeb967fdc9",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'No Significant Impact for the Uranium Leasing Program July 2007 U.S. Department of Energy Office of Legacy Management'."
    },
    {
        "candidate_id": "36f02974d7288eacc5c1",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Lead \u2013 KEC CONCUR: /s/Thomas C. McKinney__ DATE: April 7, 2003 Thomas C. McKinney NEPA Compliance Officer Attachment:'."
    },
    {
        "candidate_id": "abc83ab922e814c40d47",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1607): 2009-07-01'."
    },
    {
        "candidate_id": "ab1aae11bc12c1a5a785",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-03-05'."
    },
    {
        "candidate_id": "19eaed0f58337935ac5c",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-03-24'."
    },
    {
        "candidate_id": "a07e88f52f78da12c197",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-04-17'."
    },
    {
        "candidate_id": "185391fc4e31952e4494",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-06-02'."
    },
    {
        "candidate_id": "c09ff7d8eb0047862fe6",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-03-31'."
    },
    {
        "candidate_id": "90bfa74c92b356652703",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1255): 1998-04-30'."
    },
    {
        "candidate_id": "4ebe6de02c6e1ebf0261",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1717): 2010-03-25'."
    },
    {
        "candidate_id": "618eb4a3c83a20b4a47d",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-04-26'."
    },
    {
        "candidate_id": "86ba54f12f69634e94dc",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote '10 CFR 435 to incorporate that standard. FONSI December 2016'."
    },
    {
        "candidate_id": "9ad6ae0439aa2cc8fe43",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2011-02-04'."
    },
    {
        "candidate_id": "66dc0c7e2c9295ed9e7b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Project 7 Record of Decision Mitigation Measures Table March 2003 Fish Resources (continued) \u2022 All construction equipment and'."
    },
    {
        "candidate_id": "1c695f0ee13d20eed990",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Register / Vol. 75, No. 146 / Friday, July 30, 2010 / Notices DEPARTMENT OF THE INTERIOR Bureau'."
    },
    {
        "candidate_id": "aab9887191756e18a041",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'BASIN FISH ACCORDS MOA WITH THE SHOSHONE-BANNOCK TRIBES November 6, 2008'."
    },
    {
        "candidate_id": "f78449b3272687cfdcfb",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'roved Resource Ma na gement Pla n \u2013 June 2014 Up da ted June 2017 Willow Creek'."
    },
    {
        "candidate_id": "945dc3cc1c4311c2ffb3",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Decision for Fuel Breaks in the Great Basin March 2020 Conservation Measure Number Conservation Measure Text Do not'."
    },
    {
        "candidate_id": "12fb025d0d2d5996052f",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (rod): 2019-08-21'."
    },
    {
        "candidate_id": "43566684dd4fb1e3c15c",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'February 2013 Chapter 4: Public Involvement 37 NPR-A IAP Record'."
    },
    {
        "candidate_id": "3d4e2036fa0f3b246868",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Amendment to the California Desert Conservation Area Plan March 2006 BLM California Desert District BLM Logo Public Lands'."
    },
    {
        "candidate_id": "7352fa69f2d96ee7f4ac",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'IS McCoy Solar Energy Project Record of Decision March 2013'."
    },
    {
        "candidate_id": "28ff4cdf9b03683ffd10",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '2018 Release of FDR/FEIS & Record of Decision March 2019 Design Approval March 2019 Right of Way Acquisition'."
    },
    {
        "candidate_id": "121bdda4e1c044c91951",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (rod, DOE/EIS-0444): 2011-09-29'."
    },
    {
        "candidate_id": "7bd5b10a965a0a6e8ee0",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Blythe Solar Power Project Record of Decision A5-35 August 2014 Mitigation Measure (MM) or Design Feature (DF) Compliance'."
    },
    {
        "candidate_id": "91f59444f90065e124e6",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Sage-Grouse Bi-State Distinct Population Segment Record of Decision May 2016 Concurrent land use planning efforts for the Carson'."
    },
    {
        "candidate_id": "7a0d73de1349531d1952",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Tier 1 Record of Decision August 23, 2023 Page 5 of 35 BACKGROUND 1.1 Previous Studies'."
    },
    {
        "candidate_id": "bc03215e46578d976e2a",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'RECORD OF DECISION 2008 COLUMBIA BASIN FISH ACCORDS May 2, 2008'."
    },
    {
        "candidate_id": "b4d8991f1d0065e5bc6e",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'BLM/NM/PL-16-06-1610 Southline Transmission Line Project Record of Decision April 2016'."
    },
    {
        "candidate_id": "bcd8932100baa30ed119",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'of no significant impact and decision record in July 2021. Battle Mountain District Programmatic Oil and Gas Amendment2'."
    },
    {
        "candidate_id": "ca750857d02475c776f0",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'W. Keys III Pump-Generating Plant Modernization Project On March 12, 2012, a FONSI was signed authorizing the overhaul'."
    },
    {
        "candidate_id": "70d49c7fdfcd2f3f4cbc",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'the Eagle Butte West Coal Lease Application WYW155132 October 2007 BLM Casper Field Office'."
    },
    {
        "candidate_id": "61e2e8cf594c8bec0d43",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Project Plan Amendment and Record of Decision i May 2013 TABLE OF CONTENTS 1.0 Introduction ....................................................................................................................... 1 1.1'."
    },
    {
        "candidate_id": "a948c97d33be731a29b8",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (rod, DOE/EIS-0270): 1999-05-14'."
    },
    {
        "candidate_id": "5d27a921fa4099bcb622",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Hooper Springs Transmission Project 11 Record of Decision March 2015 Option 3A will impact approximately 20 acres of'."
    },
    {
        "candidate_id": "b8e53bd8d4558ca6758c",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Falls District 1405 Hollipark Drive Idaho Falls, Idaho 83401 April 2020'."
    },
    {
        "candidate_id": "b83666d15bae97b92257",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Soda Mountain Solar Project 4-1 January 2016 Record of Decision APPENDIX 4 Adopted Mitigation Measures'."
    },
    {
        "candidate_id": "018fb4cf5205350c1476",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'A-2 SunZia Southwest Transmission Project Record of Decision May 2023 Table A-1. Design Features for the Proposed Project'."
    },
    {
        "candidate_id": "ac92d74bff9e17602a3c",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Laura Daniel-Davis Digitally signed by Laura Daniel-Davis Date: 2022.04.25 16:46:53 -04'00''."
    },
    {
        "candidate_id": "dffd726b0040ecc3b518",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '4 Adopted Mitigation Measures McCoy Solar Energy Project 4-20 March 2013 Record of Decision Mitigation Measure Timing for'."
    },
    {
        "candidate_id": "41d01cb8e07008debe60",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'February 2013 Chapter 3: Management Considerations 15 NPR-A IAP Record'."
    },
    {
        "candidate_id": "5bf3fe688af8299b0563",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (rod): 2014-02-13'."
    },
    {
        "candidate_id": "230e3f076961adf90f8b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Natural Gas Development Project Record of Decision \u0001 September 2016 This page is blank for 2-sided printing.'."
    },
    {
        "candidate_id": "01271f52ffd2d9bcfc2c",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'El Centro Field Office El Centro. California www.blm.gov/elcentro August 2011 1.0 Introduction It is the decision of the'."
    },
    {
        "candidate_id": "833f541e203334583043",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'October 2017 Bull Mountain Unit Master Development Plan Record of'."
    },
    {
        "candidate_id": "a387e20826ff284ea446",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Natural Gas Development Project Record of Decision \u0001 September 2016 participate in reviewing crediting methodologies developed by the'."
    },
    {
        "candidate_id": "a03f29dfe862f4722dc3",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'River Valley Rehabilitation Project 3 Record of Decision August 2015 \u2022 Temporary Bypass Channel and Access Road/Levee. A'."
    },
    {
        "candidate_id": "5f41d9aea3e5a1ef3cc4",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Decision for Geothermal Leasing in the Western US December 2008'."
    },
    {
        "candidate_id": "703aa17f174e8c0a04fe",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Natural Gas Development Project Record of Decision \u0001 September 2016 7-7 Appropriate native plant materials should be selected'."
    },
    {
        "candidate_id": "c775374b4958ada32031",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Natural Gas Development Project Record of Decision \u0001 September 2016 priority/core habitat. The third priority for annual grassland'."
    },
    {
        "candidate_id": "62d38a033f1d6475b9ab",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'TransWest Express Transmission Project December 2016 Record of Decision 47 Executive Office of the'."
    },
    {
        "candidate_id": "612e78a9fcd3f0d55e78",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (rod): 2019-10-17'."
    },
    {
        "candidate_id": "7d4f61ebe2938563ca3d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Natural Gas Development Project Record of Decision \u0001 September 2016 4-i ATTACHMENT 4: MUDDY CREEK WATERSHED MONITORING PLAN'."
    },
    {
        "candidate_id": "d6f983d176434ab60f9a",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'and the associated Record of Decision (ROD) on February 23, 2022. In the NGDV ROD, the Postal Service'."
    },
    {
        "candidate_id": "8943f4b84de5b425e1c8",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'III Digitally signed by THERESA GARCIA CREWS Date: 2024.03.12 09:52:13 -04'00''."
    },
    {
        "candidate_id": "1e4ad073bf951746807b",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (rod): 2020-07-23'."
    },
    {
        "candidate_id": "a21a8e6284bc4cc95e3b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Issued in Portland, Oregon. /s/ Elliot E. Mainzer August 13, 2015 Elliot E. Mainzer Date Administrator and Chief'."
    },
    {
        "candidate_id": "7afbe7db37dd90b28ebb",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '4 Adopted Mitigation Measures McCoy Solar Energy Project 4-28 March 2013 Record of Decision Mitigation Measure Timing for'."
    },
    {
        "candidate_id": "867124b28e0dfd2b1506",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (rod, DOE/EIS-0414-S1): 2023-10-06'."
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
