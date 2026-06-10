import pandas as pd


LABELS = [
    {
        "candidate_id": "d0223ec8eb3568395ad5",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'MANUFACTURING ~,.15., ~t~a ~.~ National Nuclear Security Administration November 2024'."
    },
    {
        "candidate_id": "77eb6b40c33b79ab4a06",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'EIS process. These took place in Pocatello, Idaho (January 8, 2019); Georgetown, Idaho (January 9, 2019); and Soda'."
    },
    {
        "candidate_id": "dd385e875e428fcbbcc4",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Digitally signed by Teresa M. (6TR) Robbins Date: 2020.11.04 18:28:12 -05'00''."
    },
    {
        "candidate_id": "4c9f174497c8f525778f",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Environmental Impact Statement (FEIS) for this Project on April 26, 2013 (BLM 2013a), and a Record of Decision'."
    },
    {
        "candidate_id": "fcade8dcf4ff0e0c89ac",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2024-12-06'."
    },
    {
        "candidate_id": "f7e3845153ea7425b81b",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote 'the Albany County Board of County Commissioners on July 16, 2021 (Albany County Planning Office 2021), and was'."
    },
    {
        "candidate_id": "6eb1394bf16a9cab1a38",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote '2-47 August 2008 2.3.6.6 Air Quality \u2022 Fugitive dust controls, including'."
    },
    {
        "candidate_id": "8635ffad903c4c220891",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-11-05'."
    },
    {
        "candidate_id": "c4aef16ada6a6776f3e0",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'was distributed to the entire mailing list on April 13, 2001. Copies were also provided to anyone expressing'."
    },
    {
        "candidate_id": "f3d18cac1e91813119d1",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'comment period. That comment letter was provided on August 21, 2019. Those comments, particularly the ones about the'."
    },
    {
        "candidate_id": "0dd346bf618f5f974427",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'approved by the Secretary of the Interior on July 30, 1992, authorizes the Nation to conduct Class III'."
    },
    {
        "candidate_id": "b0811c4cb91545cfd9e1",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'DRECP BLM Record of Decision Executive Summary September 2016 Page ES-2 PART TWO: DECISION 2.1 DESCRIPTION OF'."
    },
    {
        "candidate_id": "5d2dd519fd1f65bdcdf2",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'Rail Extension - FEIS ROD-3 of 16 On June 27, 2013, FTA and SEPTA formally initiated the NEPA'."
    },
    {
        "candidate_id": "5e2f17d2ade439afc6d4",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-08-06'."
    },
    {
        "candidate_id": "88789150b8f0f00a59ea",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Register / Vol. 80, No. 145 / Wednesday, July 29, 2015 / Notices County, Pa.; Consumptive Use of'."
    },
    {
        "candidate_id": "7b188eaef13a08766fbb",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'the Sun Advocate newspaper in Price, Utah on October 8, 2015. Appeal Procedures This decision may be appealed'."
    },
    {
        "candidate_id": "8f1d21850da2d170252e",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'about 70,000 residences in St. Charles County on February 7, 1993. \u2022A detailed response to the comments received'."
    },
    {
        "candidate_id": "ddc2b8690f7d14dfd754",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'was issued in May 2008 (DOE 2008a). In September 2008, DOE/NNSA issued the first ROD for the 2008'."
    },
    {
        "candidate_id": "5810c73437f3149cf996",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'Geothermal Leasing in the Western United States (PEIS), October 2008, along with Standard Stipulations on Form 3200-24a, are'."
    },
    {
        "candidate_id": "79c3de015008dc1904cd",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'State Office published Supplemental Environmental Assessment for the February 2015 - May 2016 Sold and Issued Leases, DOI-BLM-CO-0000-2019-0011-EA'."
    },
    {
        "candidate_id": "564d52d375f9f40aa1e5",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'update of the ACT Water Control Manuals, issued March 2013, incorporated ADROP in its Drought Contingency Plan, and'."
    },
    {
        "candidate_id": "ba2044e54a48dc48334c",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'BLM x \uf0b7 BLM (Approval) \uf0b7 BLM (Approval) December 2012 Ap.4\u201017 Record of Decision'."
    },
    {
        "candidate_id": "7a1c067414b500392d46",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'such sacred sites (EO 13007). /s/ George Maloof 06/07/2021 \u2610 \u2612 Emily Burke \u00a746.215(h) Have significant impacts'."
    },
    {
        "candidate_id": "b5a0ffb216b78b38e1e3",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'Casa Diablo IV Geothermal Project Record of Decision 15 August 2013 6.4 Availability of the Record of Decision'."
    },
    {
        "candidate_id": "6e0023b0592116a9f028",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'Tier 1 Record of Decision August 23, 2023 Page 21 of 35 \u2022 Project Website. The'."
    },
    {
        "candidate_id": "2a747a49bb59ca496a04",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'of the Department of Energy was signed on May 18, 2021, by Manny Oliver, Director, Office of Small'."
    },
    {
        "candidate_id": "5dd31c63b14f2670329c",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'name: SHELL OPUS PUGET SOUND REFINERY Facility name: 03/01/1996 Date form received by agency: Large Quantity Generator'."
    },
    {
        "candidate_id": "c94467c434bfda1a83e7",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-11-29'."
    },
    {
        "candidate_id": "303ae636772089af5110",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'that exist on the Camp Tatiyee Parcel. In December 2005, Page Land & Cattle Co. submitted a revised'."
    },
    {
        "candidate_id": "99d46b22d5a965b2d92c",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (rod, DOE/EIS-0166): 1996-01-01'."
    },
    {
        "candidate_id": "d1165cba70b1329e20f8",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2017-11-13'."
    },
    {
        "candidate_id": "36e69334764bf9398f36",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'Decision for Geothermal Leasing in the Western US December 2008'."
    },
    {
        "candidate_id": "031f1d5370aa941546f0",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2023-08-30'."
    },
    {
        "candidate_id": "5619ff5d7350b8b3fc48",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Service with comments relating to treaty rights. On July 11, 2016, the Forest Service mailed revised cultural resources'."
    },
    {
        "candidate_id": "1e1ddc526a66192ab873",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Phoenix Solar Facility EA and FONSI (2024) In March 2024, TVA completed an EA for its pilot proposal'."
    },
    {
        "candidate_id": "74d1f385319dd9a0693b",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'FF095664 SRP Valid from: April 1, 2017 \u2013 April 1, 2022 Activities authorized by this SRP: Guided Hunting'."
    },
    {
        "candidate_id": "3795522fdf4bdf97ee80",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote '2005. Delivery of the Canadian Entitlement will begin April 1, 1998. The Treaty, signed in 1961, led to'."
    },
    {
        "candidate_id": "872aede68f42869206de",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '2018 45-day public comment period: March 20\u2013May 4, 2020 September 2020 We are Here'."
    },
    {
        "candidate_id": "ade5bb38c0bcb52fc25d",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'measures are applied. Issued in Portland, Oregon, on: February 15, 1996 /s/ Randall W. Hardy Date Administrator and'."
    },
    {
        "candidate_id": "fbe2b3c6deaba6f8863f",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-2191): 2023-12-14'."
    },
    {
        "candidate_id": "684b7dd577bfb33bf528",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'of no significant impact and decision record in July 2021. Battle Mountain District Programmatic Oil and Gas Amendment2'."
    },
    {
        "candidate_id": "80b0b32da6ef0f61d488",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'signed on March 11, 2009, and amended on February 22, 2010, established the development of renewable energy as'."
    },
    {
        "candidate_id": "4d6646f8c200c839dfc0",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'of the Department of Energy was signed on October 24, 2024, by Kelly Cummins, Acting Director, Office of'."
    },
    {
        "candidate_id": "d942806c29d1876a2583",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'EPA approved the Serious Area Ozone Plan, effective June 14, 2005; \u2022 The MAG 2004 One-Hour Ozone Redesignation'."
    },
    {
        "candidate_id": "4c0c528ff9d586ed38c1",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Record of Decision Public Interest Finding Completion Date May 1, 2010 CLOSED March 31, 2011 February 28, 2012'."
    },
    {
        "candidate_id": "ae3a02ff4178fbcf5161",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1441): 2003-03-06'."
    },
    {
        "candidate_id": "b4222051dee2a634bb6f",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Assessment for the Energy Northwest WNP-1/4 Lease Renewal January 2017 6 3.1.1 Land Use Per the Final Hanford'."
    },
    {
        "candidate_id": "83000620295645479c49",
        "label": "decision",
        "notes": "Decision: agency authorization, quote 'the project would enter service in 2021. On August 27, 2015, Cheniere Corpus Christi received DOE approval to'."
    },
    {
        "candidate_id": "3c86136c31ac4606abef",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Bid Package \u2010 Area 1 (P\u2010A) 2 mons 12/10/13 2/1/14 33 Prep Bid Package \u2010 Area 2'."
    },
    {
        "candidate_id": "fb5b1ef5abc4ab24bade",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Discipline Signature Lands X Minerals /s/ Paul Misiaszek 04/04/2013 X Range /s/ Michael M. Blanton 04/03/2013 /s/'."
    },
    {
        "candidate_id": "87903033cc7ce685dbcd",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Bill (AB) 751: Approved by Governor Brown on October 3, 2011, AB 751 repeals provisions allowing Caltrans to'."
    },
    {
        "candidate_id": "2492b1e2e2eca123e03d",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1652): 2009-07-30'."
    },
    {
        "candidate_id": "6ee8a77b0fceaacf4880",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Significant Impact (FONSI) Tennessee Historical Commission (THC) 8/25/2022 8/26/2022 8/30/2022 12/15/2022 Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "12e4ffd43bfa6cafe4b0",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'management plans, policies, and programs. /s/ Amanda Hoffman 02/08/2019 Amanda Hoffman Date Morley Nelson Snake River Birds'."
    },
    {
        "candidate_id": "2cce7c8e13fbb7dad4f9",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'in-situ chemical oxidation pilot test work plan dated November 14, 2011. Second GPR survey was scheduled for November'."
    },
    {
        "candidate_id": "dc0d8c8f4d589d1c0ca3",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-2297): 2025-07-30'."
    },
    {
        "candidate_id": "0f0cd2d2102fa8c45778",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'a, Ntale Digitally signed by Kajumba, Ntale Date: 2023.11.07 21:51:03 -05'00''."
    },
    {
        "candidate_id": "b565fa85d93092bea44b",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2018-02-20'."
    },
    {
        "candidate_id": "68fa0d4730343f685cbc",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'by CARB in 2009 and became effective on April 15, 2010. The regulation establishes annual performance standards for'."
    },
    {
        "candidate_id": "d02963c42444c5889188",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'EA process is as follows: Comment scoping period December 29, 2017 thru January 29, 2018 Draft EA available'."
    },
    {
        "candidate_id": "6b8fabf7e0899112ddf5",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-02-15'."
    },
    {
        "candidate_id": "2532195635e2c6db0383",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2023-04-20'."
    },
    {
        "candidate_id": "ea475e16597e26cb6b71",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Sciences, Inc. ES-22 Westbrook Draft EIS USACE #200500938 May 2013 Resource Topic/Impact Proposed Action (PA) No Action (NA)'."
    },
    {
        "candidate_id": "0f61d7bb2513f29a9980",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'ready for review upon approval of project management. November 4, 2011: Mr. Dongoske called Beverly Heffernan, Reclamation, with'."
    },
    {
        "candidate_id": "2750608f3523a893fe61",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote 'Rockwood Lithium, Inc. December 2012 Silver Peak Area Geothermal Exploration Project Page 39'."
    },
    {
        "candidate_id": "799d93583d6aacb5746c",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote '(DOE Memorandum, Office of NEPA Policy and Compliance, December 1, 2006). Therefore, DOE is including a discussion of'."
    },
    {
        "candidate_id": "e102acdefae438264518",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'with the Shoshone-Paiute Tribes on January 19, 2017, February 16, 2017, April 20, 2017 and May 18, 2017.'."
    },
    {
        "candidate_id": "d2447dd617a0945b9079",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote '12, 2005 through March 1, 2007 (revised from October 31, 2006). The purpose of extending the moratorium is'."
    },
    {
        "candidate_id": "3ef46166e986bb43b3f3",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-2055): 2016-12-22'."
    },
    {
        "candidate_id": "8de3d14e4b7f10ba82d0",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'species. Comment noted. 5 Department of Arkansas On May 5, 2010, we found that this undertaking would have'."
    },
    {
        "candidate_id": "e705fb7aba153632fda2",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-0826): 2005-04-22'."
    },
    {
        "candidate_id": "aebf63c60d8d45dca5df",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'Memorandum of Understanding signed by agency leadership in August 1994 and 1997 (Simpson 2018). FICMNEW represents a formal'."
    },
    {
        "candidate_id": "43d25cbc62a9ee7d4b93",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'at the Idaho National Laboratory (DOE/EIS\u20130453\u2013F) issued on September 23, 2016. The NNPP will recapitalize the infrastructure supporting'."
    },
    {
        "candidate_id": "71478aea0e0fc22d7ab7",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Park Service (NPS) were adopted by DOE in February 2011. In the FONSI, the NPS determined that an'."
    },
    {
        "candidate_id": "eba5ad11c904b54640d2",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Management Plan (RAMP)/CDCA Plan Amendment Record of Decision, June 2013 The Imperial Sand Dunes RAMP and Record of'."
    },
    {
        "candidate_id": "c0ec6140020f8680d0fe",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'review of substantive comments received on the draft April 2020 FR/EA, and continued refinement of the study, USACE'."
    },
    {
        "candidate_id": "b5b4c1c79eb517f95aa3",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'Management and Response Plan filed by PG&E in November 2013. Under 10(a), California Fish and Wildlife also recommends,'."
    },
    {
        "candidate_id": "126e184bf8b303f9b337",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'Creek Uranium In-Situ Recovery Project, Sweetwater County, Wyoming October 2012 BLM High Desert District \u2013 Rawlins Field Office,'."
    },
    {
        "candidate_id": "953bab932082f490216d",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'opinion on implementing the final 4(d) rule dated January 5, 2016, signed by Lynn Lewis (US Fish and'."
    },
    {
        "candidate_id": "94652d3c2df2b4b33200",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'no longer in-priority and was called out on June 20, 1977. On July 6, 1977 Reclamation notified Springs'."
    },
    {
        "candidate_id": "85897427b66828a7151f",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'and was fully executed by all parties on September 24, 2012. Decisions regarding the identification of historic properties,'."
    },
    {
        "candidate_id": "0b19da21b34e246d6a1d",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (rod): 2024-12-31'."
    },
    {
        "candidate_id": "3c8137f9cf0acd2b1967",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Water Resources Control Board adopted Decision 1641 on December 29, 1999. The Decision, intended to provide for operations'."
    },
    {
        "candidate_id": "abd687d862e9975f7481",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'copy of an e-mail message we received on June 23, 2008 from Mr. Jerry R. Gould, who says'."
    },
    {
        "candidate_id": "76712ee10db547a63eee",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'a maintenance tract for the Buckskin Mine on August 31, 2000. The tract is referred to as the'."
    },
    {
        "candidate_id": "129873ab6facdd5d2e1f",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2017-11-06'."
    },
    {
        "candidate_id": "b445c7e02495510d5e76",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote 'SJVAPCD in April 2007. Approved by ARB on June 14, 2007. Carbon monoxide (CO) 2004 Revision to the'."
    },
    {
        "candidate_id": "c0a1141d09d60f2aaa40",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote '(MW) by 2015. 3. Secretarial Order 3285A1, dated March 11, 2009 and amended on February 22, 2010, which'."
    },
    {
        "candidate_id": "44178126048246f2d5ff",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'KRYSTLE MCCLAIN Digitally signed by KRYSTLE MCCLAIN Date: 2024.02.16 09:45:53 -06'00''."
    },
    {
        "candidate_id": "e0a49e04241da05a295f",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-2192): 2022-12-01'."
    },
    {
        "candidate_id": "e10f5f16c1fa1215a1b8",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Action in the Environmental Assessment (EA) prepared in February 1998, one adult collection facility and one acclimation facility'."
    },
    {
        "candidate_id": "645d5aa01c0b5f162427",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-06-02'."
    },
    {
        "candidate_id": "c79e2ebe62b75c3f4b01",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'of the alluvial aquifer and subpile soils. In November 2004, DOE issued the draft EA for public comment.'."
    },
    {
        "candidate_id": "6fcba9b64263db57ca41",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'an oil and gas lease parcel at the June 2023 Competitive Oil and Gas Lease Sale. The selected'."
    },
    {
        "candidate_id": "b55c823a9efb55445c0d",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'by the Plan Teams, SSC and Council in December 2016. The most recent revisions to DMR estimation were'."
    },
    {
        "candidate_id": "cc556c1aef4b5f292f82",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'in conformance with the plan of operations dated April 10, 2000 and the modification dated December 10, 2002'."
    },
    {
        "candidate_id": "126c81f79ad774611f82",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'the Baltimore County Council was in place by December 31, 2024. TTT supported a public process led by'."
    },
    {
        "candidate_id": "1ca619e1f00a4c062978",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'Facility to the WSRHD. The MOA, executed in December 2021, included three measures to mitigate adverse effects to'."
    },
    {
        "candidate_id": "c53bd09facd45be388e0",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote '524-7522 or myself at (208) 524-7555. /s/Jeremy Casterson 8/31/2018 Upper Snake Field Manager Date'."
    },
    {
        "candidate_id": "d3a317a50e52359e1c76",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'was published once a week for four weeks (August 31, 2016\u2013September 21, 2016) in newspapers of general circulation'."
    },
    {
        "candidate_id": "37af29f8108a40bb1bdb",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Notice for U.S. Army Corps of Engineers Permit August 27, 2020 Section 4(f) Determination January 19, 2021 Executed'."
    },
    {
        "candidate_id": "d278dd1aaefdcdf30c8d",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote '12. 2008 Summer Improvements (Decision Memo signed in July 2008). Approved Polar Plunge gladed skiing, Buckboard Connector ski'."
    },
    {
        "candidate_id": "65adcdaa0d7a51517fcb",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'public meeting on these revisions was held on June 10, 2015. A letter of final determination was expected'."
    },
    {
        "candidate_id": "e7d33c5cce120fef5ac3",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Chapter 3\u2013Comments and Responses CRD-264 Final SNL/NM SWEIS DOE/EIS-0281\u2014October 1999 29 1 MS. LEVINGS: Well, they prepared the'."
    },
    {
        "candidate_id": "92ad8ee00f44fc8c2a8c",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'health and safety.\u201d Approved and Forest Order Issued May 31, 2016 Dispersed camping in the Gothic Corridor within'."
    },
    {
        "candidate_id": "291d718725b07a8d71b9",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'environmental protection laws. /s/ Chel Ethun Acting For 5/5/2015 Lenore Heppler Manager, Eastern Interior Field Office Date'."
    },
    {
        "candidate_id": "dd8c716572bf3f317cb4",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'Agency Scoping Meeting in Chicago February 22, 2012 HDR Engineering, Inc. 8404 Indian Hills Drive Omaha,'."
    },
    {
        "candidate_id": "1f43eb18712d9cda2f94",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2023-02-22'."
    },
    {
        "candidate_id": "01d6097c7e121c173fa4",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'LRTP for MUMPO. \uf0b7 Amendment 1 is dated July 20, 2011, with a F...'."
    },
    {
        "candidate_id": "830a11e8cc7cc3e3d192",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'for Permit to drill, and ROW FF092931. BLM. December 2008 new lakes. 52.45 MG water EA: DOI-BLM-LLAK01000-2012\u00ad 0001.'."
    },
    {
        "candidate_id": "7ecdd32c3c80423c0a13",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2013-08-02'."
    },
    {
        "candidate_id": "6b3e0e75e89436775edd",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'DOI-BLM-AKF01000-2017-0041EA Page 20 of 53 December 8, 2017 2.1.8 Contingency Plans Within the National Petroleum Reserve'."
    },
    {
        "candidate_id": "f99785ba26e50553ecdb",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Monitoring Guidelines for the South Pacific Division, dated January 12, 2015, and Regulatory Guidance Letter, dated October 10,'."
    },
    {
        "candidate_id": "ab0c453ec6c914b3714a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Schedule Activity Date Occurred/Tentative Release of Scoping Report April 2018 Notice of Availability of Draft Design Report/Draft Environmental'."
    },
    {
        "candidate_id": "e99639cb7daa9f097523",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote '2014). This document was approved by FEMA on July 14, 2014. The Suffolk Coun...'."
    },
    {
        "candidate_id": "17c23c4ac9cbeb5504e3",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'of Land Management U.S. Army Corps of Engineers July 2020'."
    },
    {
        "candidate_id": "c7475a7e6e2692dddd97",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (rod): 2015-02-13'."
    },
    {
        "candidate_id": "5187e9b92378ee70bb3a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'in April 2011, and coordination letters were delivered October 13, 2011. The draft EA was available for public'."
    },
    {
        "candidate_id": "7d9cb820172686ab025c",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Impact Statement/Environmental Impact Report. USACE, Sacramento District. \u2022 June 27, 1996, Chief\u2019s Report on FSEIS, signed by Acting'."
    },
    {
        "candidate_id": "7cacd443a7f4671060d8",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2014-05-02'."
    },
    {
        "candidate_id": "987f993c95366fae3c90",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Fighter Navy/Marine Corps Variant Concept Demonstration Phase Flight, July 2000 (Finding of No Significant Impact [FONSI] signed August'."
    },
    {
        "candidate_id": "90639db73bd49e36a6f8",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '4, 2011 for an agency scoping meeting on October 11, 2011. Attached to the meeting invitations was a'."
    },
    {
        "candidate_id": "ca3f79634f4343178676",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'pre-filing review process. We approved this request on April 14, 2015, and pre-filing Docket No. PF15-14-000 was established'."
    },
    {
        "candidate_id": "206453de713acd8e74a2",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (rod, DOE/EIS-0146): 1989-12-14'."
    },
    {
        "candidate_id": "29cba0f5171d2ded80f5",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-05-19'."
    },
    {
        "candidate_id": "c301550db6d88c5e3cbf",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '1/4/24 Reminder on EA and draft FONSI comments U.S.'."
    },
    {
        "candidate_id": "cda7216d936b75edca4a",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'herein were approved on an Interim basis effective October 1, 1982. These rate schedules and provisions were approved'."
    },
    {
        "candidate_id": "e6362347fe9f281f7809",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'local landowners on March 2010, February 2011, and February 2012 in Emmett, Idaho. Scoping documents were sent via'."
    },
    {
        "candidate_id": "dd7479d166f7ccbf4ee4",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'location for the South Extension waste pile. Approved October 1987 Not Available N-66896 Not Available Approved February 1990'."
    },
    {
        "candidate_id": "b17155ca8d6e08bd6747",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Decision Husky 1 North Dry Ridge Phosphate Mine November 2022 12 3 PUBLIC INVOLVEMENT AND ISSUES 3.1 Public'."
    },
    {
        "candidate_id": "bf280b0848b891f14756",
        "label": "decision",
        "notes": "Decision: ROD date, quote 'Borderlands Wind Project Record of Decision August 2020 Page 4 The Proponent submitted a final Plan'."
    },
    {
        "candidate_id": "8f04432a8cda433f88ae",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'action was to provide disposal capability, beginning in October 2017, to replace the existing RWMC disposal capability, and'."
    },
    {
        "candidate_id": "06b6a2dec90495b96455",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-10-15'."
    },
    {
        "candidate_id": "e2c08445e75feb178598",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2017-06-09'."
    },
    {
        "candidate_id": "3dbfec57392c13b0ebf6",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'application submitted to the OFO on or about June 15, 2013, which included applications for a fence to'."
    },
    {
        "candidate_id": "ce8f1754b2c4f90413bf",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'November 2010. Pending publication of EIR late July/early August 2011.ROW grant authorized on September 27, 2011. *This project'."
    },
    {
        "candidate_id": "b408b246c087868621f0",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Conditions Since 2005 Record of Decision DSEIS \u2502 AUGUST 2019 4-1 4. AFFECTED ENVIRONMENT AND SOCIAL, ECONOMIC AND'."
    },
    {
        "candidate_id": "3ed729593310506b9685",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote 'the construction site, Guidance on Hazing California Condors (September 2014), may be used, as necessary. \u2022 The Project'."
    },
    {
        "candidate_id": "481b784677b72f269b92",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Submitted By: Mary Perlea (916-557-7185). Submitted On: Jan 28 2014 Revised Jan 28 2014. 1-0 Evaluation For Information'."
    },
    {
        "candidate_id": "7a69e3575a0311228df8",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'about $25 million; allow a reactor startup of February 1, 1985; and require an initial (averaged) reactor power'."
    },
    {
        "candidate_id": "231c5cac8ca405925827",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote 'Amendment (RMPA) approved March 14, 2019. Date Approved: August 2015 Decision Number and Page: GRZ-MA-01, GRZ-MA-03, and GRZ-MA-13;'."
    },
    {
        "candidate_id": "32001919c92f91c1cd7f",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '2011 - Draft EA Available for Public Review January 6, 2012 - Draft EA Public Comment Period Ends'."
    },
    {
        "candidate_id": "293651ff63f7a1890bee",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1394): 2003-02-24'."
    },
    {
        "candidate_id": "bc6d2c5cf3285b68ca0a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Protest Report approved by the BLM Director on May 27, 2016. On April 23, 2015, the USFWS withdrew'."
    },
    {
        "candidate_id": "134222072c338954db0a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'H-92 Truckee Canal XM Final Environmental Impact Statement September 2020 Row Last Name First Name Organization Name Letter'."
    },
    {
        "candidate_id": "244c02121994599f287f",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'at library, City Hall, published in Federal Register April 10, 2014 Public Meeting Public Meeting in Green River'."
    },
    {
        "candidate_id": "97625090d4a3d8ad20c3",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Department of Agriculture, Natural Resources Conservation Service 10/4/23 12/12/23 1/4/24 Notice of Intent to Prepare an Environmental'."
    },
    {
        "candidate_id": "88b96cfc9e8fbf151158",
        "label": "decision",
        "notes": "Decision: agency authorization, quote 'in Section 4.1 of the attached EA. APPROVAL: 7/26/2021 Stephanie Carman, Acting District Manager Date Attachment: 1.'."
    },
    {
        "candidate_id": "08e3ed2c1184f790857d",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'The Notice of Intent to prepare an EIS (May 20, 2019) \u2022 The Agency Scoping meeting (August 2019)'."
    },
    {
        "candidate_id": "198cd686b20b83e2ead2",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'FONSI - DOE/EA-2231 2 January 2024 Together, the KCNSC NMO support more than 350'."
    },
    {
        "candidate_id": "10e797b80f931743c9d7",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'project approved by the Board of Supervisors in April 2009. Construction began in 2009. 1,007 Wind energy development'."
    },
    {
        "candidate_id": "9644cc5749b84d9b391e",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'alternatives was approved by the Forest Supervisor on 7/30/2010. This project could have a number and combination'."
    },
    {
        "candidate_id": "3bd43916a7181bfa60fb",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'gas) to FTA countries over 20 years; a June 1, 2015 application to export 1...'."
    },
    {
        "candidate_id": "fb827278393c67d48ad5",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'concurred on the Purpose and Need statement. In February 2020, the Cooperating Agencies concurred on the identification of'."
    },
    {
        "candidate_id": "679b97bab3fa92ce13c5",
        "label": "decision",
        "notes": "Decision: agency authorization, quote 'Digitally signed by MARK A MC CLARDY Date: 2021.05.13 15:37:26 -07'00''."
    },
    {
        "candidate_id": "1de02e10b1421bf803f9",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Species, Vegetation, Noxious Weeds and Non-native Invasive Species 01/13/2023 John Sullivan Archaeologist/Tribal Liaison Cultural Resources, Native American'."
    },
    {
        "candidate_id": "2294c0711b275f10aaf2",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'was developed by the NJDCA and approved on April 29, 2013. The Action Plan proposes a range of'."
    },
    {
        "candidate_id": "8f0d886673644a2f37a4",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote 'BLM NPR- A Subsistence Advisory Panel (SAP) on February 25, 2015 in Anchorage, Alaska, on September 3, 2015'."
    },
    {
        "candidate_id": "25c667d6c50aedde1311",
        "label": "initiation",
        "notes": "Initiation: scoping started, quote 'scoping period was held by the Corps from June 30, 2017 through July 30, 2017 and 77 comments'."
    },
    {
        "candidate_id": "61be381bc4bc5d40a509",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'the Colorado State Historic Preservation Office (SHPO) in March 2022. The SHPO did not provide any objections to'."
    },
    {
        "candidate_id": "8c70fc9e66f7e6fef26c",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-07-08'."
    },
    {
        "candidate_id": "e44a4c06f5db8f6a6ca9",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2020-02-07'."
    },
    {
        "candidate_id": "e570c4026db8fceed5b1",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'the Draft Amended IAP/EIS from June 9 through August 24, 2004. More than 214,000 comments were received and'."
    },
    {
        "candidate_id": "f369653919204cc25105",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Date Received Individual, Agency or Organization Email cont. 1/4/2007 Dani Sullivan, DSULLI1@state.wy.us 1/4/2007 Lisa Eadens, PLIntern@nwf.org 1/4/2007'."
    },
    {
        "candidate_id": "17c8e5eb0b1564ce334d",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-11-16'."
    },
    {
        "candidate_id": "ce3046239fc384c674d2",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'by the National Research Council\u2019s Klamath Basin Report (November 28, 2007). The goal would be to develop a'."
    },
    {
        "candidate_id": "82c5ea9c775cd171c3b3",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'Divert Infrastructure Improvements APPENDIX D: BIOLOGICAL RESOURCES CONSULTATIONS May 2019 | D-63'."
    },
    {
        "candidate_id": "659946f45e54649b1328",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2017-02-01'."
    },
    {
        "candidate_id": "31aaec6792706fccc7dd",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Public Scoping Meeting (July 2019) and Public Meeting (February 2020 and March 2022) and the respective Summary Reports,'."
    },
    {
        "candidate_id": "c569e59dd9ce16ec4196",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'Western News. A notice was also mailed, on December 21, 2010, to individuals, agencies, organizations and tribal governments'."
    },
    {
        "candidate_id": "4b0adbfde06ac49e201c",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Introduction DOE/EA-1816 3 February 2011 or local agency to consider the results of'."
    },
    {
        "candidate_id": "75c619958ccd5be72612",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Register / Vol. 67, No. 130 / Monday, July 8, 2002 / Notices \u2022 Use of Existing Borrow'."
    },
    {
        "candidate_id": "19395878b1614f3a4d4a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'announcing the availability of the Preliminary EA on March 20, 2009. An assessment of impacts to floodplains and'."
    },
    {
        "candidate_id": "7d11894c87c798e7f92c",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote '2020 East Bell Road Phoenix, Arizona 85022 623-580-5500 August 22, 2023'."
    },
    {
        "candidate_id": "3c9a4bebc9d379cabe3c",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'Prehistoric artifact scatter Not individually eligible (SHPO concurrence 2/10/2021); assumed eligible as a contributing element to the'."
    },
    {
        "candidate_id": "13ee09dd83cb5967776c",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Subject: Author: Barbara Morgan Keywords: Comments: Creation Date: 7/31/2002 4:24 PM Change Number: 1 Last Saved On:'."
    },
    {
        "candidate_id": "97d11dcda03624c409ab",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Final ROC CIS EIS 9 February 2017 Response: Concur with comment. Suggested sentence referring to'."
    },
    {
        "candidate_id": "d095f0d7a413e881c0c4",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'Record of Decision 4/30/13 4 FTA FD Approval 7/1/13 5 FTA FFGA Approval 12/1/14 6 Right\u2010of\u2010Way Acquisition'."
    },
    {
        "candidate_id": "ef8d7cfbffb7bf6b71f5",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'the more important changes . . . On January 5, 1994 PGE amended its application to EFSC for'."
    },
    {
        "candidate_id": "11a7fd2721c6c677d2eb",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Accession No. ML072400511 (NRC 2007). Adopted as DOE/EIS-0555 July 2023'."
    },
    {
        "candidate_id": "bd56d822f25629766f49",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote '161 7 Cole Boulevard Golden. Colorado 80401 -3393 March 24, 1998 DISTRZBUTION LIST SUBJECT: PREDECISIONAL D M T'."
    },
    {
        "candidate_id": "071618d62533103785a4",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'JENNIFER FOX Digitally signed by JENNIFER FOX Date: 2021.08.18 16:55:36 -06'00' BRENDA TODD Digitally signed by BRENDA'."
    },
    {
        "candidate_id": "2f2bf8c8dc5acb599841",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Impact DOE/EA-1714 Toda America, Incorporated, Battle Creek, MI March 2010 12 Visual Resources: The site is bordered by'."
    },
    {
        "candidate_id": "ee87d16ad6af27891560",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'access to the project area. By letter dated November 7, 2012, the Kansas State Historic Preservation Officer concurred'."
    },
    {
        "candidate_id": "f1eb6082589b0ae050c8",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Alamos, Oak Ridge, and Sandia National Laboratories. On September 14, 1993, the DOE issued a Finding of No'."
    },
    {
        "candidate_id": "89fa054f18330bd788ed",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'was called out on June 20, 1977. On July 6, 1977 Reclamation notified Springs Utilities by telephone that'."
    },
    {
        "candidate_id": "6e8ec22c4f0f5ccc000d",
        "label": "neither",
        "notes": "Neither: reviewer/specialist date, quote '(Gant) Massey Assistant District Manager, Resources Managerial Review 6/28/2019'."
    },
    {
        "candidate_id": "a14124bb472cdbde6486",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'DOE NEPA Register decision date (fonsi, DOE/EA-1260): 1998-08-27'."
    },
    {
        "candidate_id": "afb68b3280d07992129b",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'ordinance for regulating and permitting wind turbines. In January 2010, Kittitas County issued a Draft County Development Code,'."
    },
    {
        "candidate_id": "6cb0e96efda9926386d3",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'off on City of Cranston\u2019s completed form SF-424. February 2008: Meeting with officials from NRCS, GZA, Town of'."
    },
    {
        "candidate_id": "876c892c5b837969c9d6",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2019-06-19'."
    },
    {
        "candidate_id": "a7dc462df8a44f3c21ec",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'A. The Kennedy Creek project was signed in May 2010. The project took place in compartments 69, 70,'."
    },
    {
        "candidate_id": "ea132679fe0b3b585d20",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'were released for 30- day public comment in May 2018. Alaska Department of Fish and Game, and seven'."
    },
    {
        "candidate_id": "5f8a1f84600a0779ac50",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Leist Geologist Groundwater, Geology and Minerals, Induced Seismicity 06/14/2023 Tracy Mullins Environmental Protection Specialist Air Quality and'."
    },
    {
        "candidate_id": "92b6c816cacbb481ae72",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2022-10-24'."
    },
    {
        "candidate_id": "944b080eb78e9ed64054",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'landfill Paducah Gaseous Diffusion Plant, Paducah, Kentucky, DOE/EA-1414, August 2002. DOE G 435.1-1, Implementation Guide for use with'."
    },
    {
        "candidate_id": "0c835afbff0e967bd4d8",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'Maps and Data Accuracy, Fire Management, Proposed Action 10/13/2023 Nicole Morris Wildlife Biologist (Contractor) Special Status Species;'."
    },
    {
        "candidate_id": "1f2f310194804e921c22",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'v. BLM; CV 16-21-GF-BMM; 3/26/2018 and 7/31/2018). In September 2015, the BLM approved the Record of Decision for'."
    },
    {
        "candidate_id": "05109d967a066d216170",
        "label": "neither",
        "notes": "Neither: permit term/expiration date, quote 'the Castine site until June 6, 2013. In August 2013, UMaine proposed to extend the turbine deployment until'."
    },
    {
        "candidate_id": "3ddfe262bc105906d7f7",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote '2014. A second supplemental EIS was released in February 2015 and in March 2015, BOEM issued a Record'."
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
