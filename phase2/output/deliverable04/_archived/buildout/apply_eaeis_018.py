import pandas as pd


LABELS = [
    {
        "candidate_id": "ed92a08c5507e60952e1",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-12-12'."
    },
    {
        "candidate_id": "17d7bc2850ca595ae27c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2017-09-07'."
    },
    {
        "candidate_id": "dfd2c6ee55da3b9b0e46",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'opened on May 3, 2019 and closed on June 3, 2019. BLM received three letters and emails from'."
    },
    {
        "candidate_id": "4e3f625b7f5474fdb986",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'the SSDO\u2019s on-line NEPA register in ePlanning on August 30, 2021. The project descriptions and maps of the'."
    },
    {
        "candidate_id": "7e0c93eaeb1d50128c3d",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'your grazing permit, received by the BLM on June 24, 2011. John Edwards, a second permittee authorized to'."
    },
    {
        "candidate_id": "0a3daf954308c0844fa5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-05-11'."
    },
    {
        "candidate_id": "3e72af3ef74330f5a0df",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-01-09'."
    },
    {
        "candidate_id": "3424183edfdbda80fd63",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'File Numbers). A ROW application was submitted on October 9, 2015, by the Applicant to the BLM Arctic'."
    },
    {
        "candidate_id": "9baac38869ef4437a821",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'the public comment period was being extended until January 30, 2008. On January 17, 2008, the federal agencies'."
    },
    {
        "candidate_id": "469303208d984351801d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-09-27'."
    },
    {
        "candidate_id": "35019b869ea937bd60af",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2020-12-07'."
    },
    {
        "candidate_id": "0caf1cc8a5b946fcaf78",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'workshops, with the first workshop being held on January 22, 2003, at the Walnut Recreation Center in Las'."
    },
    {
        "candidate_id": "d66c796be172030bade3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-09-26'."
    },
    {
        "candidate_id": "07088c3ea8e870c12831",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'The first rotation occurred from July 27 to July 31, 2020 with qualified archaeologists. The second rotation occurred'."
    },
    {
        "candidate_id": "2cbcf708c15738efbac0",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-08-06'."
    },
    {
        "candidate_id": "797faaec9fe1705beca1",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Arkansas Ecological Services Office (ARESO) was initiated on August 25, 2017. A response letter was received on October'."
    },
    {
        "candidate_id": "dcf96a28db65fba5799c",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'The Portsmouth Site-wide Waste Disposition ROD, approved in June 2015, identifies the selected alternative for disposing of waste'."
    },
    {
        "candidate_id": "3cd0ab76310b3d464c98",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Cooling Towers \u2013 Additions and Replacements FONSI issued October 2010 Action was to replace four original cooling towers'."
    },
    {
        "candidate_id": "1b90eeffd9a977f2088f",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '(ROD) for the Betzc Project were issued on June 10, 1991. Most changes since 1991 have occurred on'."
    },
    {
        "candidate_id": "a6a814c7e1b92cbe67a3",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'July 7 to August 28, 2023 Scoping Period July 27, 2023 Virtual Scoping Meeting February 2024 Draft EIS'."
    },
    {
        "candidate_id": "a4add3831a441394742d",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'EIS was published in the Federal Register on August 21, 2020. The NOA described the Proposed Project, provided'."
    },
    {
        "candidate_id": "e2932b99ec68f621b2f9",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'o During the Final EIS comment period ending February 2, 2007; and o During the preparation of this'."
    },
    {
        "candidate_id": "537502db085f0959afc5",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'statement. Appeal of the Record of Decision On September 12, 2005, the Carson and Santa Fe National Forest'."
    },
    {
        "candidate_id": "c14d35496f6f1dd09e52",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '2013 2nd Scoping Period Close Comment Period Close March 14, 2014 DEIS Public Comment Period Open Comment Period'."
    },
    {
        "candidate_id": "74949458dffd5f69b726",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Kern County Oct. 2010; Approved by Kern County April 2011 PdV Wind Energy Project; PdV Addendum Wind Turbine'."
    },
    {
        "candidate_id": "ed7a162d3c7aafe3fd1b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'December 31, 2019. Because the Secretary decided on December 20, 2019 not to extend the deadline, those funds'."
    },
    {
        "candidate_id": "da8efb2a7b4830acc09b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '130 660 \u2022 PV \u2022 ROW grant issued December 2019 \u2022 Pending construction Elizabeth Solar I Yuma Arizona'."
    },
    {
        "candidate_id": "5e540e2c4f6a176e43e7",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'issued on December 12, 2012 (FHWA and FAA, December 2012). \u2022 Council Bluffs Interstate System (CBIS) Improvements Project'."
    },
    {
        "candidate_id": "7f5b2eb309d07dd16a74",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '10/19/12 21 Prepare Biological Assessment Wed 2/1/12 Fri 3/30/12 22 Insert Butterflly data Fri 3/16/12 Fri 3/16/12'."
    },
    {
        "candidate_id": "035267e0027b2c87c2db",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Ivanpah Transmission Line Project\u201d, as noticed in the December 17, 2010, Federal Register. This approval will take the'."
    },
    {
        "candidate_id": "7093b2f6214cc6a0bebd",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Impact Statement (SEIS) addresses proposed changes since the September 2016 completion of the Environmental Impact Statement (EIS) for'."
    },
    {
        "candidate_id": "dde773b7ab4974496c71",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'occupy the site. A FONSI was issued on October 8, 1997, although no action has yet taken place.'."
    },
    {
        "candidate_id": "59131538a34e932efa20",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 're-export of foreign-sourced LNG in 2009. Approved On September 7, 2010, Sabine Pass LNG received approval from the'."
    },
    {
        "candidate_id": "1fe68c647cc22c261c65",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'INTRODUCTION BLM Vegetation Treatments Three New Herbicides 1-1 August 2016 Final Programmatic EIS Record of Decision CHAPTER 1'."
    },
    {
        "candidate_id": "1a7607ed7f428ca4e72e",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'open house at the next council meeting on May 3, 2013. August 16, 2013: Mr. Bungart sent a'."
    },
    {
        "candidate_id": "3586854cf5c466118b78",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'and Final Supplemental Environmental Impact Statement (\u201cROD/FSEIS\u201d) on September 12, 2007. The NOA was published in the Federal'."
    },
    {
        "candidate_id": "f1d901d0676b3aa6aef2",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2. Water quantity 3. Hydrology 4. Riparian communities 2024.08.02 Areas of NPS Special Coordination (ASC) for BLM'."
    },
    {
        "candidate_id": "8d59246e5c49f19be45a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'local agencies as well as to Tribes on October 4, 2011 for an agency scoping meeting on October'."
    },
    {
        "candidate_id": "c8a69053ec38d8da8652",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'need for the project as described in the June 2015 Final EIS (Chapter 1, pages 1-1 to 1-5).'."
    },
    {
        "candidate_id": "aa1760a36e93b8c848da",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'air quality impacts on residents and wildlife. On March 14, 2014, we issued a Notice of Availability (NOA)'."
    },
    {
        "candidate_id": "4af8fd6cf0507df52248",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'list can be found in Appendix I. On October 10, 2012, BLM held a meeting of the consulting'."
    },
    {
        "candidate_id": "8cd6a2ed0ff2528b5369",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Decision approving the Proposed Plan was signed on January 16, 1986 by the BLM New Mexico State Director'."
    },
    {
        "candidate_id": "d6f5e8927395b5762ab1",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '2022 (NTMWD, 2017). The ROD was signed in January 2018. The LBCR FEIS states that construction of the'."
    },
    {
        "candidate_id": "85e6fe5547ad80094b0d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'The BLM issued a record of decision in September 2021. The proposed expansion would extend the operational life'."
    },
    {
        "candidate_id": "3cac713bba717f6f5c43",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'subset of the connectivity emphasis area options. In October 2008, FHWA issued the Record of Decision (ROD), which'."
    },
    {
        "candidate_id": "86c75941d2abadea1001",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'on up to 50 acres. Approved September 2004; October 2004 HC/CUEP II EA; DR/FONSI (NV063- EA04- 61) N-66621'."
    },
    {
        "candidate_id": "1d71a3d587bb9219b973",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'EIS was published in the Federal Register on July 29, 2013. The NOI asked for public comment on'."
    },
    {
        "candidate_id": "45aa5da328011229b9ef",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'Environmental Impact Statement for the TMFEIS and ROD, March 2008, is to reconsider in light of the applicable'."
    },
    {
        "candidate_id": "33cc85fa1dbb1d9623e9",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Man Event Special Recreation Permit Record of Decision July 2019 This page intentionally left blank.'."
    },
    {
        "candidate_id": "828204429133a847b2f2",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote '7, 2013. The BLM approved this relinquishment on May 9, 2013. The Grant Holder submitted a Plan of'."
    },
    {
        "candidate_id": "4b0c18493da810a3f148",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Land and Minerals Management signed the ROD on February 14, 2014, which constitutes the final decision of the'."
    },
    {
        "candidate_id": "0c935a29e74f0725f79c",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'Glades Reservoir DEIS October 30, 2015 Draft Environmental Impact Statement U.S. Army Corps of'."
    },
    {
        "candidate_id": "514d91cb10f3e7ba3399",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'of no significant impact and decision record in April 2020. Sagebrush Focal Area Withdrawal1 Wildlife The BLM intends'."
    },
    {
        "candidate_id": "d68ee2286e1970d116a2",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Finding of No Significant Impact was signed in December 2005. The M&I WSP currently being implemented by Reclamation'."
    },
    {
        "candidate_id": "0e362a7f3e868fa7d66d",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Draft EIS for a 60-day public review on December 9, 2011, held a public hearing to receive comments'."
    },
    {
        "candidate_id": "bc8973b9bfc429b8bb24",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Waste Storage, Savannah River Plant, Aiken, S.C. Wednesday, July 9, 1980 *46154 Record of Decision Decision. The decision'."
    },
    {
        "candidate_id": "eb8410c2b3db439b2ccf",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'The record of decision (ROD) was signed in July 2019 initiating the next phase of the project, Preconstruction'."
    },
    {
        "candidate_id": "a6f08ad3696d429f1bf3",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'SALEM DISTRICT December 2005 ROD Record of Decision Integrated Pest Management Walter'."
    },
    {
        "candidate_id": "08e33f5f6f9d62ef4647",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'CEMVN issued a record of decision (ROD) on June 7, 2012, identifying Alternative Q as the least environmentally'."
    },
    {
        "candidate_id": "4aaf8cae04da1c1e55b7",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'between May 2015 and May 2016, (page ES-43, November 2008 plus 6.5 to 7.5 years), this project is'."
    },
    {
        "candidate_id": "2adeaf39cac37b16fbcd",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'and comments were received between January 22 and February 22, 2016. A revised Final EIS/EIR was issued in'."
    },
    {
        "candidate_id": "b87622a968fe9222ff3e",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '1-11 March 2014 was certified by CVAG on September 10, 2007, and a Record of Decision was signed'."
    },
    {
        "candidate_id": "b8b5229f1c1d682b5b9d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Enhancements Project Record of Decision (ROD), signed on October 16, 2014, approved educational and interpretive programs, two zip'."
    },
    {
        "candidate_id": "1d6d223b8682d6d7cc06",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'The BLM issued a record of decision in March 2020. The estimated time frame for project implementation is'."
    },
    {
        "candidate_id": "2fe8063844dc7304ee64",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'groundwater cleanup at SSFL was subsequently signed on October 4, 2018 (NASA, 2018b). For soil cleanup, significant new'."
    },
    {
        "candidate_id": "4e40d4fb77d473e4f312",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Environmental Impact Statement (2010 FEIS) was published in March 2010 and the Record of Decision (2010 ROD) was'."
    },
    {
        "candidate_id": "8af5a758e30863273761",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'the NPS with a draft biological opinion on April 9, 2014, and a final biological opinion on April'."
    },
    {
        "candidate_id": "be5e9fea720aac773722",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'the EIS. \u2022 A Draft EIS was published February 5, 2021. The Draft EIS identified a preferred alternative'."
    },
    {
        "candidate_id": "020bf7c647f5f2a8ea94",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'and associated summary table indicate that, as of December 12, 2011, 12 wind projects had been approved by'."
    },
    {
        "candidate_id": "acbcb3bc50660b5e3dde",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'biological assessment was submitted to the USFWS on December 8, 2009. The USFWS responded on January 11, 2010,'."
    },
    {
        "candidate_id": "9a93a795a653cf851ee2",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'would be required to evaluate potential secondary impacts. 5 February 2007: USACE and NMFS met to discuss their'."
    },
    {
        "candidate_id": "541819b338b7b5f40dfd",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'The Texas Historical Preservation Officer provided concurrence on July 23, 2014. See attachment 1 of appendix B. Tribal'."
    },
    {
        "candidate_id": "e120c02fa5f95f90eccf",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'the Record of Decision (ROD) was signed in July 1983. Construction was initiated in 1986 and the inlet'."
    },
    {
        "candidate_id": "4bf97573ab97018a7a07",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'February 2006 5 Record of Decision In addition, many decisions'."
    },
    {
        "candidate_id": "7a0dda311d9500727687",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'a record of decision will be signed by November 2024. DEIS public comment: November 9,...'."
    },
    {
        "candidate_id": "e57a75cb6449b56b77ab",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'submitted its initial application to the BLM on December 19, 2007 (Idaho Power Company 2007) and to the'."
    },
    {
        "candidate_id": "dfa213ecccdbead43bcc",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '1.0 Introduction May 2018 1-5 Tule River Tribe Fee-to-Trust and Casino Relocation'."
    },
    {
        "candidate_id": "fbb856124c4687caf0e4",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'in the \u201cDear Reader\u201d letter to the public March 2, 1995). In areas open to off-road vehicle use,'."
    },
    {
        "candidate_id": "36749fb1281baed81bb7",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'I-465 in Indianapolis.5 In the NOI published on October 15, 2014, to advise the public and resource agencies'."
    },
    {
        "candidate_id": "faa987b0c0c173d17fa3",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'reinstated the threatened status of Slickspot peppergrass, effective September 16, 2016 (81 Federal Register 55058\u201355084). At the time'."
    },
    {
        "candidate_id": "5c77959fa92bdb83b2ed",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'for documentation). Public Comments on Draft EIS On June 10, 2013, the DEIS was mailed to all required'."
    },
    {
        "candidate_id": "a3ce2531ae4a11db8d85",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Federal Coal Lease Tract (UTU-84102r' was signed on October 5, 2015 by the Responsible Officials for the Manti-La'."
    },
    {
        "candidate_id": "4c9beae3b3fa0bc14b01",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '2016. A revised Final EIS/EIR was issued in May 2016. The Record of Decision (ROD) for the plan'."
    },
    {
        "candidate_id": "4a878a6df57c9bb490dc",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'asked for public comment on the proposal by September 17th, 2012. The Forest received comments from five individuals'."
    },
    {
        "candidate_id": "45167030727c25868cce",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'ROW grant for the DPV2 transmission line in May 2005, which would commence a new environmental review by'."
    },
    {
        "candidate_id": "644654082566031129e9",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'facility to the Resource Agencies and FERC by October 15, 2014; 2. Hold preconstruction meeting with MD and'."
    },
    {
        "candidate_id": "a9f24249598991396d61",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'report. Copies were sent on August 2, 2012, June 11, 2012, and October 22, 2012 respectively. No comments'."
    },
    {
        "candidate_id": "7d5013e8c980a7b37090",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'from renewable energy sources. S.4 COOPERATING AGENCY On February 1, 2010, the County of San Diego accepted DOE's'."
    },
    {
        "candidate_id": "2d0272b9767afe3821d5",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'the modified LBA, as modified by BLM, on March 2, 2007. The only comment received during the 30-day'."
    },
    {
        "candidate_id": "8f35692bd3463dd7d5b9",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'scoping period extended from February 17, 2017 until April 3, 2017 and included six scoping hearings. Scoping Comments'."
    },
    {
        "candidate_id": "74dade3519150d5c5afe",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'A Record of Decision (ROD) was signed on January 13, 2010. The ROD selected Build Alternative 2 for'."
    },
    {
        "candidate_id": "384159bae1be2c33889d",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'scoping process. The Federal Register notice issued on March 21, 2014, established a 45-day comment period ending on'."
    },
    {
        "candidate_id": "8b75b1d468e9b8cb371a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '(NOA) was published in the Federal Register on February 5, 2021. Multnomah County (County) held a live Draft'."
    },
    {
        "candidate_id": "0cfc19b409f1a1d08d25",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'statement. Appeal of the Record of Decision On September 12, 2005, the Carson and Santa Fe National Forest'."
    },
    {
        "candidate_id": "a1cb64c4f5da2c2e86ef",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'c). These ESP license amendments were issued in May 2010 (NRC 2010d), June 2010 (NRC 2010e), and July'."
    },
    {
        "candidate_id": "1aa76401eb3b6ba430c7",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'concerns relating to the Ballville Dam EIS. On January 24, 2014, the Environmental Protection Agency published the Notice'."
    },
    {
        "candidate_id": "eb1c867cd1eea41e426a",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '300 2,254 \u2022 PV \u2022 ROW grant issued September 2022 Gemini Clark Nevada 690 7,063 \u2022 PV \u2022'."
    },
    {
        "candidate_id": "cda5a69f2dcc4708678f",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'issued a Notice of Decision (NOD), both dated January 16, 2023. The ROD included an errata sheet as'."
    },
    {
        "candidate_id": "93a47ab8dccf511e6870",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '8, 2017, and a ROD was signed on January 16, 2018. Public comments and TVA\u2019s responses are included'."
    },
    {
        "candidate_id": "e89a7c9e0c407e2389b5",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'An updated POD was provided by MVP in June 2022. The BLM is required to obtain the concurrence'."
    },
    {
        "candidate_id": "33d41e4f5e423908eb34",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'DATE DEIS Miller West Fisher Project Draft EIS February 2009 FEIS Miller West Fisher Project Final EIS May'."
    },
    {
        "candidate_id": "0ebcd118b1da9b7ab05e",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote '26, 1997); 5. EA #127A\u2013Wine Island (FONSI signed August 20, 2001); and 6. Continued Maintenance of the Houma'."
    },
    {
        "candidate_id": "2ec8d085c3cd3b1f3d84",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote '(FONSI) relative to the proposed exploration work in July 2008. During 2007, as the EA was being prepared,'."
    },
    {
        "candidate_id": "e88070b7ab0dd8ac09da",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Cooling Towers \u2013 Additions and Replacements FONSI issued October 2010 Action was to replace four original cooling towers'."
    },
    {
        "candidate_id": "5ea27fec18de52045b57",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'with a signed Record of Decision (ROD) in June 2010, and approved the selection of the preferred type'."
    },
    {
        "candidate_id": "1d27f60bf0ca131e73d4",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Land Port of Entry in Douglas, Arizona in April 2024 (herein referred to as the 2024 Final Environmental'."
    },
    {
        "candidate_id": "86482512a2e2ee3f08f0",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'also issued a FONSI for this action on October 24, 2003. On March...'."
    },
    {
        "candidate_id": "c250421af96f0e0a1854",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'PEIS Record of Decision (ROD) was signed on December 15, 2005. The ROD implemented a comprehensive BLM Wind'."
    },
    {
        "candidate_id": "66b19c50274a54bebad8",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'and a 100 percent PDS was issued in December 2023 that continued to develop and refine the selected'."
    },
    {
        "candidate_id": "38e8738b1cc19374e82c",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'provided by the USFWS in a letter dated August 11, 2009. A biological assessment was prepared to determine'."
    },
    {
        "candidate_id": "c7f052aaeedc8432c2e1",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '(BLM 1997) 1.8.5 National Grasslands Leasing Decisions In June 2003, the Dakota Prairie Grasslands/Montana State Office Oil and'."
    },
    {
        "candidate_id": "20a7f5dee46ee52da3df",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'EIS publication and 45-day public comment period \u2022 August 2023 \u2013 Final EIS publication \u2022 October 2023 \u2013'."
    },
    {
        "candidate_id": "dbbb616801a099eecb8c",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'FEIS/ROD was published in the Federal Register on June 5, 2017. The Supplemental DEIS was prepared primarily to'."
    },
    {
        "candidate_id": "e0e9fa9b39a6575aceca",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'proposed Lease Sale 251, which is scheduled for August 2018, to consider any relevant new information; a second'."
    },
    {
        "candidate_id": "192e4039b3b0e8cfc662",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Natural Gas Development Project Record of Decision \u0001 September 2016 Comment Number Comment Response 123 Appendix S requires'."
    },
    {
        "candidate_id": "567622435db9e8994e7a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'The Lead Agencies held public scoping meetings on March 14, 2013, in the cities of West Sacramento and'."
    },
    {
        "candidate_id": "4cc49a243f8604c6ac1d",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'began on March 6th 2015, and ended on April 20th, 2015. Each comment received was carefully reviewed and'."
    },
    {
        "candidate_id": "421c1114c71332b346ac",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Air Force encourages submitting comments no later than September 25, 2017 to ensure comments are given full consideration'."
    },
    {
        "candidate_id": "0a02482f632a0af2174e",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'EIS Central Corridor LRT Project Supplemental Final EIS 1 June 2013 1 PURPOSE OF THE SUPPLEMENTAL FINAL EIS'."
    },
    {
        "candidate_id": "377bb7325905fdd268de",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '31 Prepare Record of Decision Tue 10/30/12 Tue 2/5/13 32 ROD Signed Wed 2/6/13 Tue 2/12/13 33'."
    },
    {
        "candidate_id": "d476ed7b4c8168cabc99",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'the human environment and issued a FONSI on June 15, 2000. 17 18 \u2022 Environmental Assessment \u2013 Use'."
    },
    {
        "candidate_id": "22637cd75e35eda41452",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'jointly published the ARCF GRR Draft EIS/EIR in March 2015, in accordance with the requirements of NEPA and'."
    },
    {
        "candidate_id": "021e1d4e633329d865de",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Forest MIS Amendment Record of Decision (ROD) signed December 14, 2007. Guidance regarding MIS set forth in the'."
    },
    {
        "candidate_id": "c64c36120fdf0f7d9cdd",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote '(FONSI) and Presidential permits for both projects on December 5, 2001. The Presidential permits authorized each company to'."
    },
    {
        "candidate_id": "b7ea4e91c25fc3e83137",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'additional 60 days. The second comment period ended June 15, 2015. Reclamation conducted numerous community outreach events and'."
    },
    {
        "candidate_id": "e75631cdc123d6c54395",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '25-12 Chapter 25: Public Involvement and Agency Coordination September 2019 Milestone Coordination Points Timeframe Administrative DEIS Provided for'."
    },
    {
        "candidate_id": "9578a209364f53f4f08c",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Final Notice to Proceed for full construction issued June 2022 \u2022 Pending construction Arlington Solar Energy Center Riverside'."
    },
    {
        "candidate_id": "2cd7bc11a05ad5d9de98",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'SunZia Southwest Transmission Project January 2015 Record of Decision 4 1.3 SELECTED ALTERNATIVE The'."
    },
    {
        "candidate_id": "130c976763c12f4f63e1",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Nation TVA initiated consultation with these tribes on November 8, 2019. To date, two responses have been received,'."
    },
    {
        "candidate_id": "802c047287d9ac86fdb0",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'to a meeting held in Yuba City on November 9, 2012. Following receipt of the comment letter on'."
    },
    {
        "candidate_id": "7148686dea639c3795db",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'File - A-7 Official File - KEC-4 (EQ-14) SMason:5455:sm:10/25/02 W:KEC\\EISs-EQ-14\\McNary-John Day\\ROD\\ROD Final.doc McNary-John Day Transmission Line Project'."
    },
    {
        "candidate_id": "4d43623340820f9cfb95",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'public comment period on the Draft SPDP EIS (December 16, 2022 through March 16, 2023), and late comments'."
    },
    {
        "candidate_id": "50567c20ee38d1f98935",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'CFR 3165.4. Field Manager: /s/ Duane Spencer Date: 01/08/2016'."
    },
    {
        "candidate_id": "aa2c6efd1e0aca9248fe",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'APPROVED DRAWING KAI TEXAS ID # 9010 ON 02/02/2018 DARREN L. JAMES, R.A. 18748 02 FEB 18'."
    },
    {
        "candidate_id": "677291e67d13fd5c5e89",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'May 29, 2013 Scoping Meeting Notice Local Newspapers May 30, 2013 Scoping Meeting Notice Local Newspapers June 3,'."
    },
    {
        "candidate_id": "86235ca51f294594555d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Project. After the right-of-way grant was issued in September 2016 and pursuant to the requirements in the BLM\u2019s'."
    },
    {
        "candidate_id": "c55ebdcbfcc6eedf8d80",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'NOA was published in the Federal Register on October 12, 2007. The ROD/FSEIS replaced discrete sections of the'."
    },
    {
        "candidate_id": "1c33ccfb122921263c8b",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '28, 2017. A Public Hearing was held on February 7, 2017 to inform the public of the SDEIS'."
    },
    {
        "candidate_id": "c7cebe14a33dd9b2f64b",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'open houses held on July 25, 2007 and January 29, 2009; public hearings held on June 15, 2011'."
    },
    {
        "candidate_id": "406264140c0b9efc0d63",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Kenny Anderson with a copy of the initial November 2011 LTEMP letter asking if the Tribe wished to'."
    },
    {
        "candidate_id": "da724352df5916a38790",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'La Grange Project final license application. In its October 27, 2017, AIRs for each project, staff requested that'."
    },
    {
        "candidate_id": "21a0a57cfce5697627f4",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'to Kaibab Band of Paiute Indians for review. April 15, 2015: Preliminary Chapter 3 sent to Kaibab Band'."
    },
    {
        "candidate_id": "75a8c6ec1c23906854a1",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'of Idaho, Nevada, and Utah by letters dated September 16, 2015, before this ROD was issued. The BLM'."
    },
    {
        "candidate_id": "8a33ed33bb31a30887d0",
        "label": "neither",
        "notes": "Neither: consultation date, quote '2. Water quantity 3. Hydrology 4. Riparian communities 2024.08.02 Areas of NPS Special Coordination (ASC) for BLM'."
    },
    {
        "candidate_id": "56383da34b7a9af78f1c",
        "label": "neither",
        "notes": "Neither: consultation date, quote '(FONSI) for the Visitor Center and Parking Facilities (September 20, 1979) concluded that there would be slight social'."
    },
    {
        "candidate_id": "383fcae067a53403ee00",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'issued September 24, 1993; certified remedial action - July 18 1996; delisted from National Priorities List (NPL) but'."
    },
    {
        "candidate_id": "413662e6a6a9735d61a8",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '\u2022 The Loop 9, Segment A Final EIS (September 2023) All technical reports and supporting documentation incorporated by'."
    },
    {
        "candidate_id": "8c66059d5176406247d3",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'and issued a Record of Decision (ROD) on February 13, 2015. Demolition of the Building 34 Complex The'."
    },
    {
        "candidate_id": "5d977bce8e1a2afb3594",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Forest Supervisor withdrew the Record of Decision in June 2013 after review of the analysis record found some'."
    },
    {
        "candidate_id": "5b7ea44ce706147eda80",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '45-day public comment period between June 21 and August 6, 2024. The Supplemental Draft EIS evaluated the preliminary'."
    },
    {
        "candidate_id": "5b5e226d5d7707217d66",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Impact Statement (FEIS) prepared for the project, dated March 2001, and the Record of Decision (ROD) issued by'."
    },
    {
        "candidate_id": "e3f3c34740d3d22ea73b",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Schedule Activity Date Occurred/Tentative Release of Scoping Report April 2018 Notice of Availability of DDR/DEIS Winter 2019 DDR/DEIS'."
    },
    {
        "candidate_id": "8a9b0a5a7417eb257bbd",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'STEVEN COHN Digitally signed by STEVEN COHN Date: 2024.02.12 10:49:40 -09'00''."
    },
    {
        "candidate_id": "210492376c7dc3a68537",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Section 2.5.2). After the ROD was signed in 16 December 2016, USAF conducted further evaluation of fuel transfer'."
    },
    {
        "candidate_id": "b5f6c6739e25c449c303",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Notice of Availability published in Federal Register on May 24, 2002. A 60-day comment period on the Draft'."
    },
    {
        "candidate_id": "233f0a9e2f55a7c2ec0a",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '2019. Thereafter, the BLM signed the ROD on November 25, 2019. On August 27, 2020, Western Organization of'."
    },
    {
        "candidate_id": "35f9cd35c71afb0ff14b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'the Arizona Record of Decision (Page 3, ROD, July 1991). These herbicides are Atrazine; Bromacil; Bromacil + Diuron;'."
    },
    {
        "candidate_id": "c389ddb501cd71f81fa3",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'June 5, 2013 Scoping Meeting Notice Local Newspapers June 6, 2013 Scoping Meeting Notice Local Newspapers June 12,'."
    },
    {
        "candidate_id": "46f291dd40a90341b680",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Boardman to Hemingway Transmission Line Project A-2 November 2017 Record of Decision Table A-1. Comments Received on'."
    },
    {
        "candidate_id": "9d395fe581f7e735360f",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'described in the Supplemental Draft EIS issued on April 29, 2022, and as described in the preliminary Final'."
    },
    {
        "candidate_id": "49c45cb1966015942c6b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Boardman to Hemingway Transmission Line Project B-2 November 2017 Record of Decision conditions arise that result in'."
    },
    {
        "candidate_id": "98d8ca53e5ef17a52b45",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'November 2019 Buffalo Field Office Record of Decision and Approved'."
    },
    {
        "candidate_id": "a7e5a0b2d317333f8450",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'public hearings held on June 15, 2011 and October 9, 2014; a public informational meeting held on March'."
    },
    {
        "candidate_id": "a7ca76e59b86f9c7f07a",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'and the Record of Decision was issued on December 12, 2012 (FHWA and FAA, December 2012). \u2022 Council'."
    },
    {
        "candidate_id": "bc264e748506aabe8f81",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'revised RMPs approved by the single ROD on September 21, 2015. This ROD spanned nearly 10 million acres.'."
    },
    {
        "candidate_id": "e0e1611a4dfd77d881c9",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'on August 5, 2015 in Roseau, Minnesota; on August 6, 2015 in Baudette, Minnesota and Littlfork, Minnesota; on'."
    },
    {
        "candidate_id": "f3e58d9dd31c5a193a64",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Finding of No Significant Impact was signed in December 2005. The M&I WSP currently being implemented by Reclamation'."
    },
    {
        "candidate_id": "4c58b7d2b36abfb6dfcd",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'to Arizona was originally approved by CPUC in June 2007but not pursued by SCE after 2009. BLM ROD'."
    },
    {
        "candidate_id": "644ec6274be8d303921d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Field Office Record of Decision and Approved RMPA November 2019 addresses the planning issues, within the parameters of'."
    },
    {
        "candidate_id": "f6a8f07ff3d7165d5689",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'to review the DEIS and provide comments until December 4, 2017. Following the close of the comment period'."
    },
    {
        "candidate_id": "c1c932a3f818634d1ac7",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'The record of decision (ROD) was signed in July 2019 initiating the next phase of the study, Preconstruction'."
    },
    {
        "candidate_id": "69658300bac1765c0005",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'and public scoping meetings for the Project on October 28, 2014 at the Center for Visual and Perfor...'."
    },
    {
        "candidate_id": "3f94a7be3e75fdf29794",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'the court\u2019s decision. The SEIS was completed in March 2009 and on May 20, 2009 a Record of'."
    },
    {
        "candidate_id": "e78084b0cb3ce0594a5c",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'of Proposed Construction or Alteration \u2013 Form 7460 July 16, 2021 Milestones are based on One Federal Decision'."
    },
    {
        "candidate_id": "6f43fedabc0410692fb8",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'are unchanged from what was presented in the July 28, 2023 final EIS for the Project, and the'."
    },
    {
        "candidate_id": "663d9a2abe7a7746041b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '8, 2017, and a ROD was signed on January 16, 2018. Public comments and TVA\u2019s responses are included'."
    },
    {
        "candidate_id": "bd6e59b2ba31da033b2f",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'that were subsequently authorized under PL 110-28, signed May 25, 2007. Project status based on publicly available information'."
    },
    {
        "candidate_id": "b356bb6f5bbd934a8510",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Divert EIS\u201d) and Record of Decision (ROD), signed December 7, 2016. The ROD announced the USAF decision to'."
    },
    {
        "candidate_id": "d636b574b8ff08e4cb5e",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'period on the original Draft EIS. \u2022 In November 2009, the Forest released a Final EIS (FEIS) and'."
    },
    {
        "candidate_id": "ef3d9aa40c859a4320ce",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'of Scoping and Intent to Prepare EIS \u2013 October 2011 \uf0ea \u202230 Day Public Scoping Period \u2013 October'."
    },
    {
        "candidate_id": "cbecbe7d8ac9a8b66760",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Illinois DOT, and Illinois State Toll Highway Authority, October 2012). The Tier 2 Final EIS was signed by'."
    },
    {
        "candidate_id": "1dd428b441fa45d1adba",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'oral comments on the draft EIS: Moab, Utah, October 12, 2004; Salt Lake City, Utah, October 13, 2004;'."
    },
    {
        "candidate_id": "bfffc07c7d7611b1bd85",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '2022 and the Second Draft EIS published in April 2024, and the environmental analysis associated with each alternative.'."
    },
    {
        "candidate_id": "e1002d062bca8dc9545d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'herbicide active ingredients. The Record of Decision, signed December 22, 2016, approved the new herbicide active ingredients evaluated'."
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
