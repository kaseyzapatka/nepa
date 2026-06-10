import pandas as pd


LABELS = [
    {
        "candidate_id": "a52d525dd143c5dea723",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'in its Order Amending Authorization Under Section 3 of the Natural Gas Act issued on October'."
    },
    {
        "candidate_id": "03bd69389bdac789073a",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'BUREAU OF LAND MANAGEMENT Decision Record for Categorical Exclusion Range Telephone Coop Inc'."
    },
    {
        "candidate_id": "4b1e155759a509677853",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Linn Caleb M. Hiner Field Manager 5/11/2017 Date 3 Administrative Review or Appeal'."
    },
    {
        "candidate_id": "d3358c2b8e947e751745",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '03 11 10 32 02-04100 NEPA Compliance Officer John Ganz Digitally signed by John Ganz DN'."
    },
    {
        "candidate_id": "4a0d9e1d017b3cc71385",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'OFFICIAL: Bruce Sillitoe, Field Manager DATE SIGNED: 4/23/18 Attachment 1 DOI-BLM-CO-N010'."
    },
    {
        "candidate_id": "a518a190f26487844b99",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Signature] Brian Little NEPA Compliance Officer Rocky Mountain Customer Service Region Western Area'."
    },
    {
        "candidate_id": "a05ed40811cabfa866f4",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Bybee Date Bristlecone Field Manager Contact Person Leslie Riley- Assistant Field Manager'."
    },
    {
        "candidate_id": "003c9ca41f4b28cd7439",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'review. Approved By DOE NEPA Compliance Officer Stephen Reese NEPA Review Summary Created By: Dubuc'."
    },
    {
        "candidate_id": "455df9331b474a6082b5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the proposed activity. Authorized Officer: ZACHARY ORMSBY Digitally signed by ZACHARY ORMSBY'."
    },
    {
        "candidate_id": "bc568d2a64b9bae97a87",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Jane Summerson Comments: Date Determined: Nov 6,'."
    },
    {
        "candidate_id": "6ad56b3c52dc4c685e99",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Carrie Stewart Date Determined: 12/20/2017'."
    },
    {
        "candidate_id": "f544740faee78b2a1ab4",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2020.01.06 09:11:21'."
    },
    {
        "candidate_id": "b3b5385e5dce0d9b511b",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date: Kimberly D. Dow Field Manager For additional information regarding this decision'."
    },
    {
        "candidate_id": "a3efb0d5ffc1f81e983e",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '15 14:29:25 -04'00' Date Determined: 09/14/2017 Comments: OBU-A-2017-0116, Rev. 0'."
    },
    {
        "candidate_id": "3cfe09aeb58e25fd4bc9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "2675c65f42366b77bf1f",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 01/17/2012 Comments: DOE-F-2012-0001, Rev. 0'."
    },
    {
        "candidate_id": "aec0e5e5c4c648d79832",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 's/ Ty Allen Carlsbad Field Office Manager Date: 10/07/2019'."
    },
    {
        "candidate_id": "cadbe8b8114e7e465b74",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "1109cb550492a040cacc",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '02 21:49:33 -05'00' Date Determined: 02/18/2021 Comments: TC-A-2020-0113, Rev. 0'."
    },
    {
        "candidate_id": "b48002c615a7889e4b68",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Attachment: Environmental Checklist Date: January'."
    },
    {
        "candidate_id": "9859ab534267af0777b5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Alan Bittner Anchorage Field Manager Date 09/04/2013 Attachments 1. Permit Stipulations'."
    },
    {
        "candidate_id": "606cb49713ded1815d5e",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2011 NNSA-B-11-0292 Date Determined: 08/03/2011'."
    },
    {
        "candidate_id": "7ace54a5c2d8e9c8ca99",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'applies. D: Signature Authorizing Official: Christina Price, Acting Field Manager Date: 8/29'."
    },
    {
        "candidate_id": "9009b696be4bdb3efa0c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Signed via email Date Determined May 11 2021 This form will be locked for editing'."
    },
    {
        "candidate_id": "5f2b6c011cb9eb3fc9ba",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Practices 46 NPR-A IAP Record of Decision best management practices if approved by the authorized'."
    },
    {
        "candidate_id": "4acf8cc1e582f19144df",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Approved via email Date Determined: 06/02/2022 (This form will be locked for editing upon signature'."
    },
    {
        "candidate_id": "05890c59fdba0f2286d5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Keith E. Berger, Field Manager 6 Bureau of Land Management Royal Gorge Field Office'."
    },
    {
        "candidate_id": "6323acb2d8175494445a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '28 10:07:52 -05'00' Date Determined: 02/28/2019 Comments: OBU-H-2018-0352, Rev. 1'."
    },
    {
        "candidate_id": "9c04b91c7d6bcfdb3b13",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 12/18/2012'."
    },
    {
        "candidate_id": "65c96cdf7df2477a1ed3",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '18 14:39:47 -04'00' Date Determined: 10/04/2021 Comments: PBU-N-2021-0067, Rev. 0'."
    },
    {
        "candidate_id": "a89f0a8e54666503cca2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 15 10 02 04 04 00 Date Determined 03/15/2023 Comments EEC No TC-A-2023-00013 Rev No'."
    },
    {
        "candidate_id": "69775148795682d00451",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'A\u201d \u2013 Map. Signature Authorizing Official: /s/ John Elliott Date: 10/8/2021 John R. Elliott'."
    },
    {
        "candidate_id": "14d826a363d1feb995b3",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'assessment, with the Finding of No Significant Impact (FONSI) and Decision Record (DR) signed July 12'."
    },
    {
        "candidate_id": "141754e06278bcb8af30",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Ralph W. Russell, DOE NEPA Compliance Officer June 20, 2011 Date 3'."
    },
    {
        "candidate_id": "754ef4d8070caed50f95",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Amelia Taylor Assistant Field Manager 1/6/2021 \u00a746.215(b) Have significant effects on'."
    },
    {
        "candidate_id": "bb20beb9897e2f67ba53",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "69bbaa4d016afd2df3c6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 16:28:42 -05'00' Date Determined: 02/12/2021'."
    },
    {
        "candidate_id": "ed49ce3114cb1cc5b8ca",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'in August of 2005 and Finding of No Significant Impact and EA Decision Record were signed'."
    },
    {
        "candidate_id": "c777bb94a3d217427758",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: /s/ Date Determined: 11/19/2012 2'."
    },
    {
        "candidate_id": "14b6f73387aa5791724e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'David A. Pacioretty, Field Manager Field Manager Date: January 27, 2014 Authenticated'."
    },
    {
        "candidate_id": "94129156b9a6396d2118",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'review Tracy A Ribeiro NEPA Compliance Officer Tracy A Ribeiro 2017 04 17 16 54 12'."
    },
    {
        "candidate_id": "23def1558b911630ac5b",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Berger Keith E. Berger, Field Manager DATE SIGNED: 2/9/17'."
    },
    {
        "candidate_id": "03681ad6fd2eefb2772f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for Keith E. Berger, Field Manager 3 Bureau of Land Management Royal Gorge Field Office'."
    },
    {
        "candidate_id": "4d8a1730b5929f5b65a1",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Digitally signed by Andrew R.'."
    },
    {
        "candidate_id": "0ca4df8c36bc8d443a0f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Environmental Coordinator Authorized Officer Signature KERI NELSON Digitally signed by KERI NELSON'."
    },
    {
        "candidate_id": "63b1c7d96c1478b2c9c6",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'may be required. NA-LA NEPA Compliance Officer: Date: Signature: SNCO Page 4 of 4 2/1/18'."
    },
    {
        "candidate_id": "0c9a4ced06196c3a46ce",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Stephen A. Danker Digitally signed by Stephen A.'."
    },
    {
        "candidate_id": "54c9df380919b1c3b2cd",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Babcock Lindsey Babcock Field Manager 11/3/2017 Date E. Contact Person & Reviewers For'."
    },
    {
        "candidate_id": "f6806736db22ca74e157",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '18 11:15:32 -04'00' Date Determined: 08/03/2022 Comments: TC-A-2022-0054, Rev. 0'."
    },
    {
        "candidate_id": "0b4546ef7dfc97d50769",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date: 01/22/2013 Field Manager: /s/ Ruben A. S\u00e1nchez Date: 10/15/2012'."
    },
    {
        "candidate_id": "c21b85214ce9e2be9c18",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2020.02.13 17:09:25-05'."
    },
    {
        "candidate_id": "6e4dec17458246757311",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Berger Keith E. Berger, Field Manager DATE SIGNED: 5/19/16 Gallup Chas Mine 7 Chase'."
    },
    {
        "candidate_id": "d386dfa475aa8fd0f131",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 15:18:05 -04'00' Date Determined: 03/23/2022 Comments: PBU-F-2022-0006, Rev. 0'."
    },
    {
        "candidate_id": "fb75f0c292c3e753d18f",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Attachment: Environmental Checklist Date: August'."
    },
    {
        "candidate_id": "e26af1e1b3060295c229",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Othalene J. Laurence Date Determined: May 11, 2010 Comments: Webmaster: Lawrence Wiggins Digitally signed'."
    },
    {
        "candidate_id": "9435afd3093ecf220f8d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 20 07 19 01 04 00 Date Determined 03/20/2023 Comments EEC No TC-W-2017-0042 Rev No'."
    },
    {
        "candidate_id": "8fc4de422b3494f80201",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "4f8130fdebc13b50cbe6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'attached Form 1842-1. APPROVING OFFICIAL: /s/ Jason West 1/24/2018 Jason West Date Field Manager'."
    },
    {
        "candidate_id": "9c047c8e887d2da49b48",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'March 8 2013 and the Record of Decision was signed on March 25 2016 Two appeals'."
    },
    {
        "candidate_id": "f71e745f9de986978f65",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Colleen Dingman, Field Manager Attachment: Form 1842-1'."
    },
    {
        "candidate_id": "d14847479e9551ff8fe1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 09/11/2012'."
    },
    {
        "candidate_id": "5552a72598d9d6b63cbc",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2018 month day year NEPA Compliance Officer: PIERINA FAYISH Digitally signed by PIERINA FAYISH'."
    },
    {
        "candidate_id": "7d739b2775a61c839970",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 01/06/2017 Comments: OBU-K-2016-0186, Rev. 0'."
    },
    {
        "candidate_id": "45450f821b5485003278",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'A. Smith, PhD Acting NEPA Compliance Officer Office of Electricity Date: 5/10/19'."
    },
    {
        "candidate_id": "462d040f7caf21828fd0",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: [Signature] Date Determined: 11/29/2011'."
    },
    {
        "candidate_id": "ffe94a66d09cd0141cb5",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2023 month day year NEPA Compliance Officer: PIERINA FAYISH Digitally signed by PIERINA FAYISH'."
    },
    {
        "candidate_id": "190d985f7303a6e63e45",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Gabriel R. Garcia Field Manager, Bakersfield Field Office G. Contact Person Brian'."
    },
    {
        "candidate_id": "ff73f08482203217e250",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'o DOE-SR ou EQMD Date Determined Mar 23 2011 Andrew R Grainger Andreagrainger Date 2011 05'."
    },
    {
        "candidate_id": "2f590d1280915b986fc1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '07 16:33:17 -05:00 **Date Determined**: 03/07/2023 **Comments**: EEC No: OBU-H-2023-00022 Rev No: 0'."
    },
    {
        "candidate_id": "7f64dfa37200665057a0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2015 month day year NEPA Compliance Officer MARK LUSK Digitally signed by MARK LUSK DN cn'."
    },
    {
        "candidate_id": "cf92925440abcafa3cf5",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 03/06/2017 Comments: TC-A-2017-0007, Rev. 0'."
    },
    {
        "candidate_id": "42b2dd5c258d35a42698",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Berger Keith E. Berger, Field Manager DATE: 6/27/2018 6'."
    },
    {
        "candidate_id": "f8c982f6071a24d4962f",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'review \u2022 GRR approval \u2022 Record of Decision signed Public Scoping Jan. 2016 Draft SEIS Jun 2018'."
    },
    {
        "candidate_id": "60ea808e102e51f3c34f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'further NEPA analysis. Authorized Officer: Richard Roy, Three Rivers Resource Area Field Manager'."
    },
    {
        "candidate_id": "d3a29038b54a3f781a1b",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "207250161d562f66bbd1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2023-00010 Rev No: 1 Date Determined: 01/24/2024'."
    },
    {
        "candidate_id": "8b44cbbfe5cee2c25f4e",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 05/08/2017 Comments: CBU-H-2016-0044, Rev. 1'."
    },
    {
        "candidate_id": "17b07ad2ffc26fc227e1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '05 17:12:29 -05'00' Date Determined: 12/05/2017 Comments: OBU-K-2017-0120, Rev. 1'."
    },
    {
        "candidate_id": "a2030adcf6dccd479acd",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Alabama, March 2002. ROD issued May 2002 Action was to seek extension of'."
    },
    {
        "candidate_id": "c3487280b20d2a8b4489",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Katherine S. Pierce NEPA Compliance Officer Date: 03/02/2015 Attachment: Environmental Checklist'."
    },
    {
        "candidate_id": "c7e0d99078499a5fb063",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Geoffrey Goode Digitally signed by Geoffrey Goode'."
    },
    {
        "candidate_id": "2c3d33a969976bdaec69",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Loren C. Wickstrom Date Field Manager North Dakota Field Office 5 Exhibit B: Pipeline Crossing'."
    },
    {
        "candidate_id": "7700828d7ee4c296fdc0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "81e6e7f41ff55b637730",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Grainger Andreagrainger Date Determined: Jan 13, 2011 Submit via Email Submit to Website Print Form for'."
    },
    {
        "candidate_id": "6114ecef0c3d9a13d3be",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Tracy L. Williams **Date Determined:** 03/19/2020 **Comments:** TC-A-2013-0107, Rev. 1'."
    },
    {
        "candidate_id": "d70ac8d91b6c75250dc9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'R. Halt John R. Holt NEPA Compliance Officer 5/3/11 Date GILA SUBSTATION BREAKER REPLACEMENT'."
    },
    {
        "candidate_id": "4f95c0e280fa0f620ab2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2024-00004 Rev No: 0 Date Determined: 02/23/2024'."
    },
    {
        "candidate_id": "c6d14a73f94642a24837",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '06.18 20:02:26 -04:00 Date Determined: 06/15/2018 Comments: OBU-K-2018-0157, Rev. 0'."
    },
    {
        "candidate_id": "79fb5302c57dc609241e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'applies. D. Signature Authorizing Official: Donald K. Hoffheins Field Manager Date: 9/1/2015'."
    },
    {
        "candidate_id": "5bd9086eb68b7903922a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '17 16:16:39 -04'00' Date Determined: 05/09/2022 Comments: OBU-H-2022-0034, Rev. 1'."
    },
    {
        "candidate_id": "357a126c32b7380dfbf4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: James Barrows Date Determined: September 25, 2013'."
    },
    {
        "candidate_id": "08431b4690e0a94ddc86",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "e6eacc59b6070f929f7f",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 16:25:04 -04'00' Date Determined: 10/20/2021 Comments: EP-B-2021-0014, Rev. 0'."
    },
    {
        "candidate_id": "2a790217d68044bbd4c1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Carrie Stewart Date Determined: July 11, 2018'."
    },
    {
        "candidate_id": "2f211493bdd1f12b3ed3",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '08.07 08:19:49-04:00 Date Determined: 07/19/2019 Comments: CBU-M-2019-0042, Rev. 0'."
    },
    {
        "candidate_id": "535b94b76a7bb11c9551",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "4a775f96c120b72151fd",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Date: 2021.04.01 10:35:57 -04'."
    },
    {
        "candidate_id": "f7a1487e730be9d881e6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2023-00002 Rev No: 0 Date Determined: 07/31/2023'."
    },
    {
        "candidate_id": "2e725081c959cf60c9ac",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Codie Martin Shoshone Field Manager Enclosures: 1- Signed ROW Grant, IDI 027046/IDID106064422'."
    },
    {
        "candidate_id": "48ba3d22ecc02dcf80cb",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "e103cf1ca6a499187c43",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'P060-2014-0135-EIS and Record of Decision approved December 23, 2020. 1 ## Page 3 Impact'."
    },
    {
        "candidate_id": "48e62c04f0a3f409c730",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2020.03.20 15:44:38'."
    },
    {
        "candidate_id": "c7062faa537453055cea",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '13. Ryan Chatterton Field Manager Digitally signed by MICHAEL CHATTERTON Date: 2020'."
    },
    {
        "candidate_id": "a967452a7931f22defb9",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Christina Stark Assistant Field Manager (Planning and Environmental Coordinator) SIGNATURE'."
    },
    {
        "candidate_id": "e29fea9eb7e6ebe986e6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'TC-A-2011-00020 Rev 0 Date Determined Mar 28 2011 Submit via Email Submit to Website Print Form'."
    },
    {
        "candidate_id": "0ed8c3fb2229a4b31d51",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for Brent Ralston Date Field Manager Four Rivers Field Office'."
    },
    {
        "candidate_id": "272ee224bb79876be3da",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2019 month day year NEPA Compliance Officer: MARK LUSK Digitally signed by MARK LUSK Date: 2019'."
    },
    {
        "candidate_id": "b7a729190b9654e06c8b",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "cebb9269e70274562f5b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'GormanPL yso doe gov c US Date Determined Jan 5 2010 Date 2010 01 05 10'."
    },
    {
        "candidate_id": "cec74afc2167ef87d2e0",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '12 01 12 27 06 05 00 Date Determined 12/01/2022 Comments EEC No OBU-K-2020-0183 Rev No'."
    },
    {
        "candidate_id": "c086aef8300d4a3db524",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Stephen Leslie, Assistant Field Manager Las Vegas Field Office, Division of Resources Digitally'."
    },
    {
        "candidate_id": "2b954e98ee53396ad229",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '27 12:37:58 -04'00' Date Determined: 03/11/2019 Comments: CBU-H-2019-0017, Rev. 0'."
    },
    {
        "candidate_id": "79db214ea17eeab46a09",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2018.03.27 12:01:47-04'."
    },
    {
        "candidate_id": "a9c6cec87a7912327125",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for William J. Mills, Field Manager DOI-BLM-CO-N050-2023-0015-CX 8 Appendix A. Figures'."
    },
    {
        "candidate_id": "18afc24a8bff640cb531",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "d407e2328b386c620022",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '04 11:51:43 -07'00' Authorized Officer: Date: Jacob Palma, Field Manager Contact Person For'."
    },
    {
        "candidate_id": "938e6a8b042a48b91d41",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '09 09:19:39 -0400mson Date Determined: 07/05/2022 Comments: OBU-H-2015-0053, Rev. 7'."
    },
    {
        "candidate_id": "7a03428b9940630b1075",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Management Framework Plan and Record of Decision (MFP/ROD). Date Approved or Amended: 3/30/1983]'."
    },
    {
        "candidate_id": "15b752636c24cf4631c7",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy Williams Date Determined: 12/07/2021 Comments: DOE-G-2021-0015, Rev. 1'."
    },
    {
        "candidate_id": "93901f8c7dc581f8c4df",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '17 12:02:32 -06'00' Authorized Officer/Date 7| Page'."
    },
    {
        "candidate_id": "6c8afc2da812b1316aec",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Grange Katey Grange NEPA Compliance Officer Date: May 12, 2023 Attachment(s): Environmental'."
    },
    {
        "candidate_id": "6ea21d2b1fa4e872a095",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '07 19:57:35 -04'00' Date Determined: 07/20/2022 Comments: TC-A-2022-0058, Rev. 0'."
    },
    {
        "candidate_id": "3a340bf459b54e982b19",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'significant impacts. Signature Authorizing Official: /s/ R. Cade Powell Richard Cade Powell Cody Field'."
    },
    {
        "candidate_id": "30da1699af662cdc7e2c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '08 17:53:56 -04'00' Date Determined: 09/26/2018 Comments: LWO-H-2018-0022, Rev. 0'."
    },
    {
        "candidate_id": "0cac2e4b4675288aa9d8",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'editing upon signature) Date Determined: 9/5/2012'."
    },
    {
        "candidate_id": "cee24343f342e433e253",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Attachment(s): Environmental Checklist Date: May'."
    },
    {
        "candidate_id": "fe2539f8a5619f1fb0f5",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote '28 1995 and the Record of Decision was issued on June 1 1995 An amended Record'."
    },
    {
        "candidate_id": "0e07371ba58159119650",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Gary S. Hartman Date Determined: 6/17/2014 2'."
    },
    {
        "candidate_id": "3273a06852c9a7c16bd0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2012 month day year NEPA Compliance Officer: john ganz Digitally signed by john ganz DN: cn-john'."
    },
    {
        "candidate_id": "fb9b80f3f2fee66b7c3f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '95825. /s/Jennifer Mata Field Manager 3/20/18 Date DOI-BLM-CAN060-2018-0016-CX Page'."
    },
    {
        "candidate_id": "be469ba76ff618aa28b1",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Comments: Digitally signed by Andrew R. Grainger'."
    },
    {
        "candidate_id": "bd410900b10a197e6427",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Workflow Approved by SPRPMO NEPA Compliance Officer 12/22/15 Determination Date'."
    },
    {
        "candidate_id": "1df2226cafef45d71491",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 's/ Terry A Humphrey Authorized Officer: Terry A. Humphrey Four Rivers Field Manager 6/1'."
    },
    {
        "candidate_id": "2907bed9f5b742e18fb6",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "e3c22c4f0ce62ccaa35c",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Field Office under the Record of Decision signed in December 2007. According to the PROPOSED'."
    },
    {
        "candidate_id": "8199dca1bdb71935a547",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '15 10:20:12 -06'00' Authorized Officer/Date CONTACT: For additional information concerning'."
    },
    {
        "candidate_id": "2d357fe314e4910af522",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 08/27/2013 Comments: TC-A-2013-0109, Rev. 0'."
    },
    {
        "candidate_id": "9073d8f25d0f7ddd9889",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'OEIS (July 2010 Record of Decision, DoN 2010a), and would complete construction of four'."
    },
    {
        "candidate_id": "a33d199125cc604ca3d6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '24 14:20:52 -05'00' Date Determined: 11/05/2020 Comments: PBU-N-2020-0011, Rev. 0'."
    },
    {
        "candidate_id": "ea4f1aeba37cff5771af",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'dated June 2009. The record of decision (ROD) was signed in July 2019 initiating the next'."
    },
    {
        "candidate_id": "41c101e6ffcaec0aff7a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 7/22/2010 Comments: Webmaster: Record ID: 24'."
    },
    {
        "candidate_id": "12fdf7af130845272b6c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2018.02.21 07:14:37'."
    },
    {
        "candidate_id": "cba77d91265ddc0d7d67",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'TC-W-2010-219 Rev 0 Date Determined Dec 17 2010 Submit via Email Submit to Website Print Form'."
    },
    {
        "candidate_id": "07ed44af66f19e035c4d",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Blythe, California (BLM ROD 2019a). In response to this application, BLM prepared'."
    },
    {
        "candidate_id": "05cbc200a1e4a64159c0",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer:** **Comments:** **Date Determined:** Sep 3, 2010 Digitally signed by Andrew R. Grainger DN: cn=Andrew'."
    },
    {
        "candidate_id": "87d6428e3ad9c56a431c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '24 13:38:56 -05'00' Date Determined: 11/04/2020 OBU-A-2020-0255, Rev. 0 Comments:'."
    },
    {
        "candidate_id": "b05486508c353d242f83",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 18 15 42 49 05 00 Date Determined 01/18/2023 Comments EEC No OBU-N-2022-0022 Rev No'."
    },
    {
        "candidate_id": "252fdb335cd440a1d464",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '07 25 10 33 20 04 00 Date Determined 07/25/2023 Comments EEC No EP-R-2023-00001 Rev No'."
    },
    {
        "candidate_id": "3121510f6d23106c3aca",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 03/23/2017 Comments: TC-A-2017-0018, Rev. 0'."
    },
    {
        "candidate_id": "fe62008f06b74bf36d77",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2023-00001 Rev No: 0 Date Determined: 02/09/2023'."
    },
    {
        "candidate_id": "3314968d5e05fdecdafe",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Attachment(s): Environmental Checklist Date: October'."
    },
    {
        "candidate_id": "4ac68db62d4b29b4e7eb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Field Office, Assistant Field Manager Resources Date'."
    },
    {
        "candidate_id": "2513f2fe435e6dedbda2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '04 10:29:48 -04'00' Date Determined: 03/31/2022 Comments: OBU-E-2022-0029, Rev. 0'."
    },
    {
        "candidate_id": "b679efec62009df0a1de",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'editing upon signature) Date Determined: 03/16/2016'."
    },
    {
        "candidate_id": "ffc083249174ace30d9b",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'written approval from the authorized officer. E. Preparer/s Prepared By: DANA BORUCH Digitally'."
    },
    {
        "candidate_id": "4d86fd2047abf881c868",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer Signature and Determination Date Digitally signed'."
    },
    {
        "candidate_id": "acba313dfb0c16533464",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "fb225b411000ea5f7b91",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '20 11:48:50 -07'00' Authorizing Official: Monte Senor Assistant Field Manager Date: Contact'."
    },
    {
        "candidate_id": "a37bad0acaed4e9baf60",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '08.25 18:11:29-04'00' Date Determined: 08/12/2016 Comments: OBU-F-2016-0101, Rev. 0'."
    },
    {
        "candidate_id": "ab1134acb2402cb7440c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'required. MITCHELL OWENS Approving Official: Mitchell Owens Digitally signed by MITCHELL OWENS'."
    },
    {
        "candidate_id": "e18c10af82dde0f4c57f",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'closure methods. The ROD was issued in July 2016. Colbert Fossil Plant'."
    },
    {
        "candidate_id": "bb7738b45a1645ee9d52",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Fields May 17, 2018 Date Field Manager, Farmington Field Office'."
    },
    {
        "candidate_id": "27e19204a9d70b3d82e7",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Signature] Gene Iley, Jr. NEPA Compliance Officer Rocky Mountain Customer Service Region Western Area'."
    },
    {
        "candidate_id": "9f96e62e6229799ccfb6",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'on public land The right-of-way grant was approved by the Bureau of Land Management on September'."
    },
    {
        "candidate_id": "52130ec25a50f0db9fb4",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "fbfa4be02da055c01afb",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Billahue J. Lawrence] Date Determined: Mar 25, 2011 Comments: Webmaster: THINK BEFORE YOU PRINT'."
    },
    {
        "candidate_id": "ad21e97c82996250992b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 06 11 21 21 05 00 Date Determined 03/06/2023 Comments EEC No TC-A-2023-00008 Rev No'."
    },
    {
        "candidate_id": "5e6ae3e918b12bf6ae29",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Attached Stipulations. APPROVING OFFICIAL: /s/ Melissa Warren TITLE: Field Manager DATE: 2'."
    },
    {
        "candidate_id": "c6342304136f947721a1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Pamel L. Soma Date Determined: 2/13/2012'."
    },
    {
        "candidate_id": "d940f4dbe32024cd9b48",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'February 2022 9-9 Record of Decision (ROD) \u2014 A concise public document that records a Federal'."
    },
    {
        "candidate_id": "861afa21fea6ffb6f0a8",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "a9e7715f0a951818895f",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 16:51:13 -04'00' Date Determined: 07/13/2022 Comments: OBU-A-2019-0264, Rev. 2'."
    },
    {
        "candidate_id": "b2ca27fa3d837a373072",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Garcia Gabriel Garcia Field Manager 8/7/19 Date 7 US Department of the Interior Bureau'."
    },
    {
        "candidate_id": "79be89912a4c149ca4f4",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '06'00' Eric Lepisto Field Manager Miles City Field Office Date 4|Page Exhibit A ARTALENT'."
    },
    {
        "candidate_id": "d09a1860ee464e0c48b2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Berger Keith E. Berger, Field Manager DATE SIGNED: 5/14/19'."
    },
    {
        "candidate_id": "27eb4e5a29a33c69ff57",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Attachment(s): Environmental Checklist Date: February'."
    },
    {
        "candidate_id": "ba70190cad1af1512487",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 11/20/2019 Comments: EP-M-2019-0039, Rev. 0'."
    },
    {
        "candidate_id": "41dd8d2e237a717318c9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Attachment(s): Environmental Checklist Date: July'."
    },
    {
        "candidate_id": "f581e2feba8fde074ae9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Date: October 27, 2014 Attachment(s): Environmental'."
    },
    {
        "candidate_id": "f36150305b9d746f0141",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Andrew R. Grainger Digitally signed by Andrew R.'."
    },
    {
        "candidate_id": "df16f3d5ec0fd4ab06ad",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'ed December 1992. FONSI issued April 1994. FONS1 issued August 1994, Final'."
    },
    {
        "candidate_id": "a8842ba11e0b0b9ec4ca",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '07 11 11 58 07 04 00 Date Determined 07/11/2023 Comments EEC No OBU-H-2023-00056 Rev No'."
    },
    {
        "candidate_id": "b55113773fdb8bae8462",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '2018 Date Carlsbad Field Office Manager'."
    },
    {
        "candidate_id": "362b3dae229ae515e965",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '] Aron C. King Date Field Manager F. Contact Person and Reviewers For additional information'."
    },
    {
        "candidate_id": "ccfaa394357461444e0f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Keith F. Berger, Field Manager 3 Bureau of Land Management Royal Gorge Field Office'."
    },
    {
        "candidate_id": "834ff47ad34960b3cb40",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Manager: Katherine Johnesc Date Determined: 5/4/15 The above description accurately describes the proposed'."
    },
    {
        "candidate_id": "82960a524db6517244d1",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '09 11 10 00 54-04-00 NEPA Compliance Officer John Ganz Digitally signed by John Ganz DN'."
    },
    {
        "candidate_id": "ba7452b1842f0953ef78",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Management Coos Bay District Decision Record for Categorical Exclusion DOI-BLM-ORWA-C000-2018-0001'."
    },
    {
        "candidate_id": "14af797fd2c9eb0e3425",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 08/25/2020 Digitally signed by Tracy L. Williams Date: 2020.09'."
    },
    {
        "candidate_id": "bde1b7f4524594c26419",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 08 14 58 47 05 00 Date Determined 03/08/2023 Comments EEC No OBU-F-2023-00005 Rev No'."
    },
    {
        "candidate_id": "2b68a569b768b0935db7",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 02/22/2016'."
    },
    {
        "candidate_id": "dcdea4edccfaf74d271f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '06'00' Todd D Yeager Field Manager Miles City Field Office Date Page 6 of 10 S'."
    },
    {
        "candidate_id": "d1b7bf1bdf21dbdae2cc",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Bierbra by Kafferminz Date Determined: 08/14/2012'."
    },
    {
        "candidate_id": "314b6a598726160024e5",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '25 13:39:44 -04'00' Date Determined: 05/24/2022 OBU-B-2022-0134, Rev. 0 Comments:'."
    },
    {
        "candidate_id": "0997ca3de56431d2d0e1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 7/1/2010 Comments: Webmaster: Record ID: 1099'."
    },
    {
        "candidate_id": "872adb31c1338ed7e2e1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '30 15:26:36 -05'00' Date Determined: 11/03/2021 Comments: TC-A-2013-0113, Rev. 3'."
    },
    {
        "candidate_id": "d0da331062e8929c6b93",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'ROBERTS Amanda S. Roberts, Field Manager, Central Oregon Field Office Digitally signed by AMANDA'."
    },
    {
        "candidate_id": "35e0da010e6a65fd91f0",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Amelia Taylor Assistant Field Manager 07/13/2021 [ ] [ ] \u00a746.215(b) Have significant'."
    },
    {
        "candidate_id": "50726ceb511ce3da039c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '06 17:44:33 -04'00' Date Determined: 06/30/2020 Comments: DOE-G-2019-0003, Rev. 1'."
    },
    {
        "candidate_id": "44a7859ba392113c02ae",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: [Signature] Date Determined: Dec 2, 2009 ORE NEPA'."
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
