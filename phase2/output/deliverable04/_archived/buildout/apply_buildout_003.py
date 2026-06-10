import pandas as pd


LABELS = [
    {
        "candidate_id": "ccf848459a8ad6593257",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ea1e08707fe27450b66c",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'proposed Project In March 2014 the 4 BLM received a ROW application from APS and must'."
    },
    {
        "candidate_id": "c4ebbacfa731aae5e84d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "85cd98f7899da1475478",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "90cbff4a520f0b89f0b4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b687efffba823222d4ed",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'for 2 to 3 years. A ROD approving development and operation of a modified'."
    },
    {
        "candidate_id": "c975e1d4e55a9de69373",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c0201d8a761503469e0a",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Register Summer 2016 Record of Decision (ROD) signed for LWI only2 Summer 2016 2 Military'."
    },
    {
        "candidate_id": "1b5b8c8ea8e70fc92052",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'scoping period a public open house was held by the Inyo National Forest on September 8'."
    },
    {
        "candidate_id": "98753f7abf3c40928557",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d445ba83b1a40a89a6df",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1f1249f99b0d7cd8115d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0a31ade748799cf61264",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Federal Register on 15 July 2010 This notice included a project description and scoping meeting dates'."
    },
    {
        "candidate_id": "1ab360f64fcdaadc7dd1",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On May 2 2018 Dustin Lyons on behalf of Prospect Bomb filed film permit'."
    },
    {
        "candidate_id": "f9a310d334887f1a2f43",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Frank Chynoweth DATE: 05 / 14 / 2010 month day'."
    },
    {
        "candidate_id": "a3e251d3e4c4b9e3a5bb",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Monger Date that any scoping meeting was conducted N/A Date that concurrent electronic distribution for review'."
    },
    {
        "candidate_id": "32d62f44eceab57dddfe",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "08d923299e1fc0ac5940",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'libraries and newspapers Between January 21 2022 and February 21 2022 the Commission received comments from'."
    },
    {
        "candidate_id": "3f4d7c69da9d1d06b489",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Supplemental Filing IR 173 received April 14 2023 BMOP Blue Marlin Offshore Port LLC 2023b Blue'."
    },
    {
        "candidate_id": "aeebf24fe6adcff64e66",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'and NEPA process On June 9 2017 the FERC issued a Notice of Intent to Prepare'."
    },
    {
        "candidate_id": "ced714a4182841d56c7f",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Jennifer Fox Date that any scoping meeting was conducted N/A Date that concurrent electronic distribution for'."
    },
    {
        "candidate_id": "8cc02da4003aedb8106f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Adrienne Riggi DATE: 09 /23 / 2010 month day'."
    },
    {
        "candidate_id": "25241922e9e3210c1cf9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ccb5fea965d3832d4cfe",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "32a1177ffee85d62ae43",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e4e2aaae38e7cbf4415c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "820fa30276318dc75588",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "15fca1764e9e22e45867",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8b6bae8494dbddfbdfd4",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'contains a copy of these consultation letters and all responses DOE issued the Draft EA for'."
    },
    {
        "candidate_id": "3f9520ab51e676b54d91",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'Summary Report Formal public scoping for the Steigerwald Floodplain Restoration Project EA was initiated on December'."
    },
    {
        "candidate_id": "93319036e8ebc2183159",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Development POD dated November 16 2016 The Applicant has also filed a Conditional Use Permit Application'."
    },
    {
        "candidate_id": "df2e15e643665a35163b",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'publication date is 2018 NOI published May 16 2014 Approximately 80 miles east of the Moneta'."
    },
    {
        "candidate_id": "d4dad9e9c4b91924a4dd",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'cumulative effects The Notice of Intent to prepare this Supplemental EIS was published in the Federal'."
    },
    {
        "candidate_id": "93d8aab77b5669d9f4c3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "abd04fff0f0cdb6f618d",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'staff stated that the public comment period closed August 22 2016 which is the minimum 45-day'."
    },
    {
        "candidate_id": "3cf2a7e2815160d10599",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Shoshone Paiute Tribe on December 16th 2010 in a government to government consultation process and provided'."
    },
    {
        "candidate_id": "8f86916335a8ebed23e8",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Date of Preparation November 20 2019 BACK GROUND On April 24 2019 Nikki Engkraf on behalf'."
    },
    {
        "candidate_id": "fda6e2d6b4011a6b3de8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "30919c99c7c110fe5ce9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f32597317fa2d473519e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "34b0cf13ad2c985498ea",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "132a5d8e0adf5c4772ad",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a88ae2261ca9b729c8d5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "01d4875b1a9a7cf6c655",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'contained therein On January 11 2013 the Owyhee Field Office initiated the public scoping process for'."
    },
    {
        "candidate_id": "38a9e739baafebc16ede",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'proposed Project On September 24 2021 the Commission issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "560d153d9e46c0c9f42b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'list was requested on 2 September 2020 from the USFWS Information Planning and Consultation website per'."
    },
    {
        "candidate_id": "4811d89026f04cd093b4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d660cc63be05534369e0",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e7cccdd764979f3c0f13",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b63d6275988449278073",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Filing July 2023 On July 18 2023 ETNG filed Final Resource Reports with an Abbreviated Application'."
    },
    {
        "candidate_id": "e863e45f2051386cacac",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'were extended to the USFWS and to the NOAA NMFS NMFS accepted the request to participate'."
    },
    {
        "candidate_id": "d8ba581232e04e50cfb8",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION.** **DOE Initiator Signature:** JOHN BALTRUS Digitally signed by JOHN BALTRUS Date'."
    },
    {
        "candidate_id": "2b518c4a84daac178191",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '2016 BACKGROUND On May 7 2015 Qwest Corporation d/b/a Century Link submitted an application SF-299 requesting'."
    },
    {
        "candidate_id": "c5cb46578f9d2ad461a8",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote '2014 actual amended January 2 2015 18 Draft EIS circulated for review and comment Draft EIS'."
    },
    {
        "candidate_id": "84a35f42c8725ba27ca9",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Project an NOI was published in the Federal Register that initiated a formal public scoping period'."
    },
    {
        "candidate_id": "f9ebec95b6bf2f19b5ee",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '1 0 INTRODUCTION In March 2013 ASARCO LLC Asarco submitted a Section 404 permit application to'."
    },
    {
        "candidate_id": "d2f5027e250657e4fcf9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4fdac208cf6a4f8b9df4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'the laws of Delaware filed an application Application with the Office of Fossil Energy FE Department'."
    },
    {
        "candidate_id": "293c1f4bd930a1ad5613",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'for the Approved JD 11/9/16 USACE Email Tania Asef sent Approved JD package to Veronica Li'."
    },
    {
        "candidate_id": "13435b6751ba606adc6f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Mike Hayes Date: 03 / 25 / 2015 month day year'."
    },
    {
        "candidate_id": "95ff2e7a3e022cc2881a",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On March 7 2017 Justin Montesalvo on behalf of Patriot Campers filed film permit'."
    },
    {
        "candidate_id": "361f0ff2b335c9912fec",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1b5ab334ca5f8a2ad85d",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'cross state lands; this right-of-way was approved in October 2010. Sempra executed an easement agreement'."
    },
    {
        "candidate_id": "743a9dc33d139eb152da",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f90f06b091bdb0665c67",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Judith Dyer DATE: 05 /21 / 2010 month day year'."
    },
    {
        "candidate_id": "c40137e4218721952fa7",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Briggs White DATE: 08 / 30 / 2011 month day year'."
    },
    {
        "candidate_id": "9cfda7018bb3764a8b23",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b87f46934e74eafe6043",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Linn Caleb M. Hiner Field Manager 3/14/2017 Date 3 Administrative Review or Appeal'."
    },
    {
        "candidate_id": "201b47b50186b36a6841",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote '28 1997 and written public comments were solicited through October 1 1997 Notices of the availability'."
    },
    {
        "candidate_id": "fa235e0c51d3f68fb748",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5dfc7a552691770f26e3",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'an amended NOI was published on 30 March 2020 85 Federal Register 17544 to announce cancellation'."
    },
    {
        "candidate_id": "6501fe9a804dc6fa20ad",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e8b7c91e6c420c679668",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "378a26561f16372e78b3",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: SYDNI CREDLE Digitally signed by SYDNI CREDLE Date'."
    },
    {
        "candidate_id": "c572844580880974a18f",
        "label": "initiation",
        "notes": "Initiation: posted to ePlanning/NEPA Register, quote 'BLM Field Office on December 5th 2014 for a minimum 30 day public viewing The Dash'."
    },
    {
        "candidate_id": "631cbe080033b3aea15b",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'Brownsville Texas on April 21 2015 The purpose of the open house was to provide the'."
    },
    {
        "candidate_id": "c416aad35784911ab50b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: PATCHARIN BURKE Date: 08 / 28 / 2017 month day'."
    },
    {
        "candidate_id": "f337cb93b6f6c6b064d3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "918ded2a7ca4cb1b04a9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f5dfe85969a48db37c50",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'percent interest. 2021 On September 30, 2021, BOEM published a Notice of Intent to Pre...'."
    },
    {
        "candidate_id": "5f737a31fda3e99547e9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6979d4ecd9dbedd0614b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Amy Tomer De & Efficiency Technologies Division,'."
    },
    {
        "candidate_id": "23ccadecda714dd379bf",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "fed0d1c3821a4bcb487d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "389c9bdf168b78478f06",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'substation An additional application was submitted on August 27 2013 to separate the transmission line from'."
    },
    {
        "candidate_id": "322a018fe3666849d401",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8408ee7409b55c74c740",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'brief summary of the public scoping meetings The scoping comment period began on December 21 2012'."
    },
    {
        "candidate_id": "6e33012591bf063602c4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bd4632892f8fc9a39a2c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "434c1a9f8c0598ad298b",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'is in response to the right-of-way application submitted on April 29 2010 to the BLM by'."
    },
    {
        "candidate_id": "0fc66c8f62e86be26e21",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Proposed Action On January 08 2020 the Ute Mountain Ute Tribe UMUT Environmental Programs Department EPD'."
    },
    {
        "candidate_id": "1126fff2062056209b41",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Katherine Kweder DATE: 09 /23 / 2010 month day'."
    },
    {
        "candidate_id": "d80ba9cb75036f752068",
        "label": "neither",
        "notes": "Neither: construction/activity period date, quote '0 Georgia SIP Air Construction Permit Modification Permit Issued June 27 2007 Permit No 2869-283-0005-S-01-0 Application'."
    },
    {
        "candidate_id": "03ac80863331f3da9501",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Steven Richardson Date: 9 / 14 / 2012 month day'."
    },
    {
        "candidate_id": "cad077e1763704cbf13d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f3fee5791faa28fa334f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a36cf9a05834eb29edf7",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: DUSTIN MCINTYRE Digitally signed by DUSTIN MCINTYRE'."
    },
    {
        "candidate_id": "424a30cbc74218b2a146",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6f38a7beb257af35573e",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Brad DATE: 03 /25 / 2010 (month day year) NEPA'."
    },
    {
        "candidate_id": "eec2c4e617c8d38740ff",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "67bc96c2ad26921f2658",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'specifically comments on the notice of intent published in the Federal Register on May 30 2019'."
    },
    {
        "candidate_id": "48c7486eabae8ac6a8de",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: DONALD FERGUSON Date: 07 / 25 / 2016 month day'."
    },
    {
        "candidate_id": "39dbf32f280d9e1dbd2a",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'and 9 BACKGROUND On February 18 2019 the Bureau of Land Management BLM Tres Rios Field'."
    },
    {
        "candidate_id": "865b446d2ffdb6fe4d11",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'The scoping period closed on March 9 2012 but some relevant comments were submitted after the'."
    },
    {
        "candidate_id": "d61a2f96fc622cf9208d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Donald Krastman DATE: 05 / 12 / 2010 month day'."
    },
    {
        "candidate_id": "c01ab4c173086892405d",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Comment Period Ends 09/08/2011 Contact Penny Woods 775 861 6466 EIS No 20110177 Draft EIS NOAA'."
    },
    {
        "candidate_id": "8d78482650619b69e1f4",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'lands NorthWestern filed an application SF-299 to amend their existing Special Use Permit SUP in May'."
    },
    {
        "candidate_id": "b23e559cd520f89fead7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "098722ff4558e716e4e7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c926901225212f320d3f",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'of the formal 45-day public comment period 80 Federal Register 2438-2439 January 16 2015 On January'."
    },
    {
        "candidate_id": "5af0c242325d8291253b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: KARA ZABETAKIS Digitally signed by KARA ZABETAKIS'."
    },
    {
        "candidate_id": "fc2a5e9a359cbfc80e7e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "be3b2611de42b8b5afc0",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'the Lower Brule Sioux Tribe regarding potential corridors and routes for the Lower Brule to Witten'."
    },
    {
        "candidate_id": "ca6ac2c1920e5ce13298",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'public comments On October 22 2010 DOE published a Notice of Intent NOI to prepare this'."
    },
    {
        "candidate_id": "fe7898d267616e9bb0b3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "bc4e153a58fa523022f9",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'established a closing date of March 13 2023 for receiving comments on the draft EIS On'."
    },
    {
        "candidate_id": "f7e77c736695593c1076",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "337ba8c0cadfcec3390f",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'established a closing date of August 30 2021 for receiving comments on the draft EIS The'."
    },
    {
        "candidate_id": "7c9239ffb52fae2ecd69",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Brian Mollohan DATE: 04 / 07 / 2011 month day'."
    },
    {
        "candidate_id": "0a76fd1f03db0f70e2e1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "886f4909e55beb16eb4e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0215813fb537d0dd171a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7c48686a5e73855ce67e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'October 3 2008 The scoping comment period for the Proposed Project ended on October 24 2008'."
    },
    {
        "candidate_id": "5cdd978ea131f4246059",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "228495e5cb30ac97918f",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Actions SOPA since May 2010 The final comment period began on December 21 2013 The comment'."
    },
    {
        "candidate_id": "d8899c3c9aa5e97e1276",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'comment period began on February 16 2018 with publication of a Notice of Availability NOA of'."
    },
    {
        "candidate_id": "8b114b8694b3f0e0ec99",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'the Forest Plan On September 23 2016 a new NOI was published in the Federal Register'."
    },
    {
        "candidate_id": "d4e78a68d423648b0bcf",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On September 21 2020 Jonathan Paltin on behalf of 495 Productions filed film permit'."
    },
    {
        "candidate_id": "fbf51985a85f96a210c9",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: DATE: 07 / 19 / 2010 month day year NEPA Compliance'."
    },
    {
        "candidate_id": "072cb533b8cb6dcd0095",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Jesperson Swan Lake Inc September 12 2016 U S Bureau of Reclamation September 12 2016 Dan'."
    },
    {
        "candidate_id": "3ee70511ea4d56fdd6b6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c820fcb7a0f44b8c5746",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'report was provided on 4/5/23 10/31/22 Emailed 2/7/23 Mailed 2/9/23 Sent SHPO report and working on'."
    },
    {
        "candidate_id": "ce70c56c37d5dd13e856",
        "label": "initiation",
        "notes": "Initiation: FERC/application notice, quote 'Notice of Application on January 22 2013 assigning Docket Number CP13-36-000 to this project The Notice'."
    },
    {
        "candidate_id": "b3dd765a54011bc8ce08",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "de8a449321a2bd360341",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JESSICA MULLEN Date: 06 / 19 / 2017 NEPA Compliance'."
    },
    {
        "candidate_id": "d85531dd4e37ac79186a",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'public record 125 On March 13 2015 FERC issued a Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "46c59a37021334bad1f8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "cc54402dae3b4bea12f5",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On September 16 2015 Larry Campbell on behalf of White Falcon Studios filed film'."
    },
    {
        "candidate_id": "7547797552d7ab0bfb8d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "415ae8d3b3383ab868c8",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'mitigated Resource Concerns April 11 2014 Boeing letter to NASA with comments on the FEIS October'."
    },
    {
        "candidate_id": "c3e2d89501742a3bbed0",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'd to the public in April 1997 Federal Register Notices announcing the availability of the final'."
    },
    {
        "candidate_id": "ddbeb532112390e0a40e",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Agreement Countries Authorization granted January 17, 2012 (DOE/FE Order No. 3059) Not applicable'."
    },
    {
        "candidate_id": "8d8b03d90f5772c51db3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "69ae4eb98aaeedf929a1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7a15131326d90bdb0fd3",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote '13429 1 August 2021 1 Introduction 1 1 Background On December 4 2020 EDFR submitted an'."
    },
    {
        "candidate_id": "ce5e3f192ba9912dc27a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "e18d03f1b7c0ba3b8eed",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0ee1d53109cefe9990f7",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'determination April 3 2013 SHPO letter to FHWA Concurrence on eligibility determination for Old Pyramid Highway'."
    },
    {
        "candidate_id": "182ba82079da115d87b3",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'concurrence from the Texas SHPO for a small portion of the marine area of potential affect'."
    },
    {
        "candidate_id": "55ae8ce8fea9d1d34ff5",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'Register starting a 45-day comment period which was later extended 15 additional days On April 14'."
    },
    {
        "candidate_id": "4afe322146af09a3bd4d",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'the lead agency for consultation with the USFWS for the FERC FEIS On November 21 2017'."
    },
    {
        "candidate_id": "1b4f9ed1ebbd3a2dc65f",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Cooperating Agency Request July 26 2016 Page A-22 Agency Review of Pre-Draft EIS Request Letter Jan'."
    },
    {
        "candidate_id": "fb8c48a4b7cc0dda1257",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Proposed Action On August 23 2019 Grant B Boring with Worldwide Trophy Adventures filed land use'."
    },
    {
        "candidate_id": "c37cd88be252584d8936",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "c715c9cf454990218f8f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "7ccd19070e07efe85b68",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "616af85bb0b7861eb00e",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'a scoping meeting on September 11 2008 Based on discussions during the site visit comments at'."
    },
    {
        "candidate_id": "5a666391ac0021d859e5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1e72df6ddb04db1a5b0a",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'in Newington, CT. \uf0b7 Jan. 9, 2013: Public scoping meeting was held at Suffolk Community College, Riverhead,'."
    },
    {
        "candidate_id": "d8cb8e451f56b87629a6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a3e182e71598920f793a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date Amanda James, Field Manager Ryan Couper Acting Field Manager Contact Person: Alice'."
    },
    {
        "candidate_id": "c05771c21c25e0432d03",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'of the draft EIS in June 2021 FERC staff estimated that the issuance of Notice of'."
    },
    {
        "candidate_id": "a12da150bd6e9598711e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "2d0eb1c2e6beb57fb07b",
        "label": "neither",
        "notes": "Neither: survey/inspection/activity date, quote 'See above entry for sampling scheduled to be conducted on 4/10/07 An extension for Request dated'."
    },
    {
        "candidate_id": "4a8f2f6389db12449f37",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f6d799a76e5a81201d84",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote '07 see Exhibit B Map 2 Description of Proposed Action By renewal application received September 8'."
    },
    {
        "candidate_id": "508d8051ce0795b9176f",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing permit On February 26 2018 the Lewistown BLM Field Office received completed transfer of'."
    },
    {
        "candidate_id": "b4387b8b4ba6f3095531",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "91e2d8b93afa17eda3be",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'scheduled to close on January 9 2006 Response See response to Comment EMC-0025-001 under PEIS Consultation'."
    },
    {
        "candidate_id": "117063e76e7e3cc11dc7",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Theodore McMahon Date: 03 / 28 / 2018 month day'."
    },
    {
        "candidate_id": "8b4a501cb7e37ae3c54e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0ec94341c126b05f5344",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Scoping period ended 9/3/2007 DEIS in Preparation Scoping period ended 9/3/2007 DEIS in Preparation Scoping period'."
    },
    {
        "candidate_id": "087b8dc1364240ca71a8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b256a745293b3365333b",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'from the public The NOI was published in the Federal Register FR on April 7 2023'."
    },
    {
        "candidate_id": "4b10dce49ee09657af80",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d1b5b1683871c83f0c40",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Teresa Jones DATE: 5 /17 /2011 month day year'."
    },
    {
        "candidate_id": "c6dae0d3be3005ac50f3",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: VITO CEDRO Digitally signed by VITO CEDRO Date: 2019'."
    },
    {
        "candidate_id": "26466ae5361ebeb8f841",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "4aec500487f426011f93",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JOSEPH HANNA Digitally signed by JOSEPH HANNA Date'."
    },
    {
        "candidate_id": "435926180e9792b35aa7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f68034ef87930c007378",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "87e84ad960cb3d36cd2f",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'business in Houston Texas filed an application Application with the Office of Fossil Energy FE on'."
    },
    {
        "candidate_id": "e1dda5ff0275d85606f1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "47000fb597ab15c8204a",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'public during a 30-day scoping comment period which was extended upon request from the public The'."
    },
    {
        "candidate_id": "cdd3bf35e65d3af0aef9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0de26d727604893b36d2",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "3bd1f2fe36bb428c4f03",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "9c407b0fd5b23e61ad7d",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '2012, the Owyhee Field Manager issued the scoping document for the Castlehead-Lambert'."
    },
    {
        "candidate_id": "6f4369b1a02d1dc57399",
        "label": "initiation",
        "notes": "Initiation: posted to ePlanning/NEPA Register, quote 'material sale On March 1 2012 the BLM posted this project on the NEPA website ePlanning'."
    },
    {
        "candidate_id": "51e6a3159f5f0dcc42bb",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0b22993b9ec8a6e00cea",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "d1deccfb94b201f8326e",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'to ROW UTU-95961 On February 21 2023 PacifiCorp filed an SF-299 amendment application to amend ROW'."
    },
    {
        "candidate_id": "5c216c3c296f9e6c543f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "ef0150fb6a29a9a88767",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "55d68d1ae36e092f4592",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "5da09cb4b376d2c6c769",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "05a1855c2ed834396784",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b2dea5b5254b7f2a0c58",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '2003 and in Lyons on October 2 2003 Reclamation conducted the scoping meetings in both an'."
    },
    {
        "candidate_id": "952c1dd0c843ee4da87e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "8d4b09dcba12195709e8",
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
