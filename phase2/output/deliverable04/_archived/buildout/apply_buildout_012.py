import pandas as pd


LABELS = [
    {
        "candidate_id": "dd7caedcd5604acbd16e",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'EA and the unsigned FONSI were available for a 15-day public review and comment'."
    },
    {
        "candidate_id": "5b80c293ca5668fdabd6",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Date: 2022.05.17 16:18:36 -04'."
    },
    {
        "candidate_id": "6da343b54c4c6ca08869",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "001832756bcceebda586",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: GEOFFREY GOODE Digitally signed by GEOFFREY GOODE'."
    },
    {
        "candidate_id": "ea813ea7c08885fa23de",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Project \u2013 BPA signed a Record of Decision (ROD) in October 2002. Minor construction will'."
    },
    {
        "candidate_id": "dac354ac6c75fd53734a",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'W000-2017-0001- EA, Finding of No Significant Impact (November 2018), and Decision Record (November 2018'."
    },
    {
        "candidate_id": "0c85e06ccef3efbe280a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Gary S. Hartman Date Determined: 9/15/2011 Comments: Webmaster: Record ID: 28'."
    },
    {
        "candidate_id": "8f0d874faa1a3d6b59d9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "a11d16d49285c65e1559",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: G. Scott Watson ORO NEPA Compliance Officer Date'."
    },
    {
        "candidate_id": "97ac3c233a45974c0a71",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2022-01002 Rev No: 0 Date Determined: 09/23/2022'."
    },
    {
        "candidate_id": "760769ed0a748350415a",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'review. Approved By DOE NEPA Compliance Officer Stephen Reese NEPA Review Summary Created By: Auger'."
    },
    {
        "candidate_id": "d49a71e4117caf103222",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'by BLM\u2019s October 2010 ROD. The initial project proponent and applicant for the'."
    },
    {
        "candidate_id": "f886872a56782244691e",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'file Approved by SPRPMO NEPA Compliance Officer 11/16/11 Determination Date'."
    },
    {
        "candidate_id": "aac9f12653a62e9b3486",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: OBU-N-2011-0051, Rev'."
    },
    {
        "candidate_id": "c357d9ba0dc07e8f9304",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '14 09:30:27 -05:00 **Date Determined**: 12/14/2023 **Comments**: EEC No: LWO-Z-2023-00010 Rev No: 0'."
    },
    {
        "candidate_id": "332aff29d11a3adb2caa",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '09.10 17:41:53 -04:00 Date Determined: 09/04/2018 Comments: EP-A-2018-0006, Rev. 0'."
    },
    {
        "candidate_id": "e572f3ff608f6f295ffe",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '21 16 35 12 04'00' Date Determined Aug 19 2010 Submit via Email Submit to Website'."
    },
    {
        "candidate_id": "a337256d82ef92a79a42",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "8d7dd0ae70635d4e0345",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for) Melissa Warren Field Manager 10/30/2020 Date Appeals Information: Appeals information'."
    },
    {
        "candidate_id": "1ff3872de35568af33dd",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '14 12:20:30 -05'00' Date Determined: 01/14/2010 Comments: Webmaster: THINK BEFORE YOU PRINT 25A3191'."
    },
    {
        "candidate_id": "510df8180075d6c4926e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'stay. /s/ Will Runnoe Field Manager 9/3/13 Date Attachments: Categorical Exclusion'."
    },
    {
        "candidate_id": "7021c0e6e53d9a721258",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'signed by BRYAN FOLEY Date Determined Date 2022 12 01 07 18 13-05'00' TJSO NEPA Coordinator'."
    },
    {
        "candidate_id": "4e956419f4e3101933a2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer James L. Elmore Date Determined: 5/17/2010 Comments: Webmaster: Record ID: 1093'."
    },
    {
        "candidate_id": "d51e59c75092085d4293",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'to the EA and a new FONSI was signed. After considering the protest reasons'."
    },
    {
        "candidate_id": "ce5b2c4d44ae7b3374a6",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Comments: TC-A-2021-0005, Rev. 0 Digitally'."
    },
    {
        "candidate_id": "b2e82ba1e58a7cb417a5",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '27 13:23:06 -04'00' Date Determined: 06/02/2022 Comments: LWO-Z-2022-0029, Rev. 0'."
    },
    {
        "candidate_id": "e10db1e9f96717495048",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "287b379a815bf23b716d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'For Linda Marianito Date Determined: 6/9/14 CONTINUATION SHEET for Erosion repair work at Structure'."
    },
    {
        "candidate_id": "588d01991dd65f358724",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2019.02.20 12:23:40-05'."
    },
    {
        "candidate_id": "9709cd908b01b2fbf72f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Environmental Coordinator Authorizing Official: Deferrema Digitally signed by rema DOUGLAS HERREMA'."
    },
    {
        "candidate_id": "9fd5b35826fecfb24803",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2018.03.04 09:52:41'."
    },
    {
        "candidate_id": "bd61e70c404468186ef2",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "3ff6d749e9848025033d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. **NEPA Compliance Officer:** **Comments:** Digitally signed by Andrew R. Grainger'."
    },
    {
        "candidate_id": "1265dff796a79461293d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2023 month day year NEPA Compliance Officer: Pierina Fayish Digitally signed by Pierina Fayish'."
    },
    {
        "candidate_id": "6bb7e19e5e99b9af493e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for: Keith E. Berger, Field Manager COC-61619 Martin Road ROW Assignment Project Map 27S'."
    },
    {
        "candidate_id": "fdec575b66c0ac4c98b2",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2023 month day year NEPA Compliance Officer: Pierina Fayish Digitally signed by Pierina Fayish'."
    },
    {
        "candidate_id": "2e865da54678b0a3a9fb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Nichelle W. Jacobson Field Manager, Central Yukon Field Office Contact Person Robin Walthour'."
    },
    {
        "candidate_id": "605c690b386319ebfb1d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "283b00de5dc880d40b78",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'EA also tiers to the Record of Decision (ROD) for the Final Vegetation Treatments Using Aminopyralid'."
    },
    {
        "candidate_id": "e924b04b9d7f68e4a59c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03.07 11:17:46-05'00' Date Determined: 03/07/2013 Comments: CBU-M-2013-0009, Rev. 0'."
    },
    {
        "candidate_id": "7d6d29edee4fe76b31c6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '04.11 15:11:16-04:00 Date Determined: 04/11/2024 Comments: EEC No: CBU-G-2024-00003 Rev No: 0'."
    },
    {
        "candidate_id": "8772b08e04ea98470dd3",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Field Office under the Record of Decision signed in December 2007. According to the PROPOSED'."
    },
    {
        "candidate_id": "24fac9ba215cc27a116e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Lindsey Babcock, Field Manager 9/19/23 Date Contact Person For additional information'."
    },
    {
        "candidate_id": "af6e50023b2533847e8f",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '06 23 14 54 30-04 00 NEPA Compliance Officer John Ganz Digitally signed by John Ganz'."
    },
    {
        "candidate_id": "53acb39368e23398d43a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '25 16:08:31 -06'00' Field Manager Date Grand Junction Field Office ATTACHMENTS: Exhibit'."
    },
    {
        "candidate_id": "b55630b72c2a617246c6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Carrie Stewart Date Determined: September 23, 2019'."
    },
    {
        "candidate_id": "fc23b8e63a7f4661576c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Comments: Andrew R. Grainger Digitally signed by'."
    },
    {
        "candidate_id": "57dd01968909a9e8ab19",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2018 month day year NEPA Compliance Officer: Jesse Garcia Digitally signed by Jesse Garcia Date'."
    },
    {
        "candidate_id": "e47535cf3327acaab628",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Stephen A. Danker Digitally signed by Stephen A.'."
    },
    {
        "candidate_id": "463144393763fc95b1ad",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Timothy D. Gilloon, Field Manager Date'."
    },
    {
        "candidate_id": "6611ade0889009f92ff1",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Exhibit B). MELISSA WARREN Field Manager: Date: Digitally signed by MELISSA WARREN Date: ['."
    },
    {
        "candidate_id": "2352194e0c79d4444d89",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2020 month day year NEPA Compliance Officer: FRED POZZUTO Digitally signed by FRED POZZUTO Date'."
    },
    {
        "candidate_id": "8af8c3e391b827e655fb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '9/30/16 Assistant Field Manager Minerals & Lands Authorizing Official: NA Date Field'."
    },
    {
        "candidate_id": "37ebd281d2b5d3c0e8ff",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Ribeiro Tracy Ribeiro 2013.05.28 08:03'."
    },
    {
        "candidate_id": "aeb3f588ef96af194940",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Signature) Carlsbad Field Office Manager Date 09/08/2021'."
    },
    {
        "candidate_id": "d4892523b1da1e76da25",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Date: June 21, 2016 Attachment(s): Environmental'."
    },
    {
        "candidate_id": "a797ddaae0010aa0ca15",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environment. D. SIGNATURE Authorizing Official: /s/Dennis J. Carpenter October 3, 2017 Field Manager'."
    },
    {
        "candidate_id": "da84f4279dcf33cbf80d",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Development Project EIS ROD was approved on September 22, 2016. The 4 CD-C'."
    },
    {
        "candidate_id": "9e5161a15728cbb594cd",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '26 11 57 04 04'00' Date Determined Apr 28 2010 ARRA S 2009 089 Rev 0'."
    },
    {
        "candidate_id": "b1873b0cac54a86e64e3",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2024-00002 Rev No: 0 Date Determined: 02/23/2024'."
    },
    {
        "candidate_id": "eb5650f45ecd072d7597",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Bittner Date Anchorage Field Manager Attachments 1. Triumvirate LLC Commercial Heli-skiing'."
    },
    {
        "candidate_id": "ccc43bcc7ef6625aa5e3",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Lorraine Christian, Field Manager, ASFO Date and signature /s/ S. Dao 10/22/2018 /s'."
    },
    {
        "candidate_id": "70bbf459c7cba6d88492",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '11 01 08 42 37 04 00 Date Determined 11/01/2023 Comments EEC No OBU-H-2021-0130 Rev No'."
    },
    {
        "candidate_id": "932347598153d36c9557",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'names of the following Right-of-Ways ROWs from Mid-Rivers to Verizon communications sites MTM-30443 is an authorization'."
    },
    {
        "candidate_id": "f811d2b51380888485cc",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'written approval from the authorized officer. E. Preparer/s DANA BORUCH Digitally signed by DANA'."
    },
    {
        "candidate_id": "9f04fc33d44c05da05db",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'PHILIP D'AMO Philip D'Amo Field Manager (Detail) Mother Lode Field Office Digitally signed'."
    },
    {
        "candidate_id": "16e928c009019cf89af2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '25 10:33:38 -04'00' Date Determined: 04/04/2022 Comments: TC-A-2022-0025, Rev. 0'."
    },
    {
        "candidate_id": "7064784ef949bc8dcc08",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'of Water Resources. AUTHORIZING OFFICIAL: [Signature] NAME: Edward J. Kender TITLE: Field Manager'."
    },
    {
        "candidate_id": "f47dd3a76049e0dfdec2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'present. THOMAS DARRINGTON Authorized Officer/Date Digitally signed by THOMAS DARRINGTON Date:'."
    },
    {
        "candidate_id": "a62b1a5ddf54baef6cb7",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "70794a00ba6027ea18eb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '24 09:29:10 -06'00' Authorized Officer/Date 4|Page'."
    },
    {
        "candidate_id": "3f731ba0f3548483070c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Jamie Moeini, Assistant Field Manager Las Vegas Field Office, Division of Lands Categorical'."
    },
    {
        "candidate_id": "0ee76f2e3cb33c433282",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer Signature and Determination Date J.Chy Digitally signed'."
    },
    {
        "candidate_id": "adae38bc32c55ca2f646",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Date: October 27, 2016 Attachment: Environmental'."
    },
    {
        "candidate_id": "048d86c8135a1f11795e",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Statement (July 2005). The Record of Decision, approved by both agencies in September 2006,'."
    },
    {
        "candidate_id": "a580c1e7c632f325f071",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Date Determined: 08/14/2023 Digitally Signed By WILLIAMS'."
    },
    {
        "candidate_id": "9ee6d99f1a827542e4e7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'R.Collum Eagle Lake Field Manager 1/12/17 Date Decision Record for DOI-BLM-CA-N050'."
    },
    {
        "candidate_id": "759fc38eaeeaffe6cdb0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2012 month day year NEPA Compliance Officer: john ganz Digitally signed by john ganz DN: cn-john'."
    },
    {
        "candidate_id": "fe6618fc92b3a3e27bfb",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '30 17:28:39 -05'00' Date Determined: 11/19/2020'."
    },
    {
        "candidate_id": "626bfd040d10801cfc22",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Adam Carr Adam Carr Field Manager Eastern Interior Field Office 3/20/18 Date Contact'."
    },
    {
        "candidate_id": "288caa1ee5ab098e7ba7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Dana Wilson Acting Field Manager DATE SIGNED: 09/26/16 Attachment: EXHIBIT A \u2013'."
    },
    {
        "candidate_id": "1704193b17953e5aa616",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environmental analysis. Authorized Officer: Richard Roy, Three Rivers Field Manager Signature'."
    },
    {
        "candidate_id": "4ed97ab68eeb29b2c805",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2016 month day year NEPA Compliance Officer: Pierina Fayish Digitally signed by Pierina Fayish'."
    },
    {
        "candidate_id": "fa48b4d6d468427981cc",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '31 13:19:06 -04'00' Date Determined: 05/19/2022 Comments: OBU-H-2017-0146, Rev. 4'."
    },
    {
        "candidate_id": "3d581085dde4417e4a10",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Berger Keith E. Berger, Field Manager DATE: Oct 23, 2018'."
    },
    {
        "candidate_id": "14cc28972be5dc765f78",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2018.10.31 16:58:19-04'."
    },
    {
        "candidate_id": "bfb027ce37af9426e9db",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 01/18/2012 Comments: TC-A-2011-0132, Rev. 0'."
    },
    {
        "candidate_id": "9c7de673720f6850afa8",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 11/23/2015 Comments: FSSBU-L-2015-0002, Rev. 0'."
    },
    {
        "candidate_id": "09464f5a929694c9a1cb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'mitigation is proposed. AUTHORIZING OFFICIAL: NAME: Scott Cooke Thomas Schnell TITLE: Field Manager'."
    },
    {
        "candidate_id": "3f04ab61ecf77f7d2a14",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 01/26/2023 Comments: EEC No: TC-W-2013-0154 Rev No: 4'."
    },
    {
        "candidate_id": "5e3b55978bb903752e43",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '26 -06'00' Dave Pals, Field Manager'."
    },
    {
        "candidate_id": "fc51ff784f63a64436e5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'required. EDWARD KENDER AUTHORIZING OFFICIAL: Digitally signed by EDWARD KENDER Date: 2022.02'."
    },
    {
        "candidate_id": "90ae5801ed342be3dc47",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'For Scott C. Cooke Field Manager Date: 11/21/17 S:WEPA Projects\\2018\\2018-0002'."
    },
    {
        "candidate_id": "c3c405f6ce3094c80448",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2023-00029 Rev No: 0 Date Determined: 04/11/2023'."
    },
    {
        "candidate_id": "e9d50e6d40bf5eb1892c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Babcock Lindsey Babcock Field Manager 3/29/2016 Date E. Contact Person & Reviewers For'."
    },
    {
        "candidate_id": "5e9adb3257218f13c985",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'o DOE-SR ou EQMD Date Determined Jun 23 2011 Andrew R Grainger Andreagrainger Date 2011 07'."
    },
    {
        "candidate_id": "1724bb9e4ef3214d9583",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 09/07/2016 Comments: OBU-H-2015-0007, Rev. 0'."
    },
    {
        "candidate_id": "4f8c85bc35e3f7a8a560",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2020 month day year NEPA Compliance Officer: PIERINA FAYISH Digitally signed by PIERINA FAYISH'."
    },
    {
        "candidate_id": "62011e24ba9d72d25541",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Berger Keith E. Berger, Field Manager DATE SIGNED: 7/15/16'."
    },
    {
        "candidate_id": "81c21dcfc5a714c571e7",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '02 09:21:31 -05'00' Date Determined: 01/03/2022 Comments: TC-A-2017-0035, Rev. 1'."
    },
    {
        "candidate_id": "cf91d5f69b2a5dab565f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Lorraine Christian, Field Manager, ASFO Date and signature /s/ R. Cox 6/26/2019'."
    },
    {
        "candidate_id": "08c5dee60af9d5b750a3",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 12/29/2011 Comments: TC-W-2011-0083, Rev. 0'."
    },
    {
        "candidate_id": "2c44bf02cce6a23c4232",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Attachment(s): Environmental Checklist Date: May'."
    },
    {
        "candidate_id": "f2ac85e657f478daf491",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: [signature] Date Determined: 03/08/2016 (This form will be locked for editing upon signature'."
    },
    {
        "candidate_id": "fc1e32c8d9d6addfad99",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Tyler Cox 1/29/2019 Authorized Officer Date Acting Assistant Field Manager Lands & Mineral'."
    },
    {
        "candidate_id": "3d9bd4c5bb1fd73ded1d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2018.10.31 17:15:47-04'."
    },
    {
        "candidate_id": "61f81164da346f1ebd54",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2014 month day year NEPA Compliance Officer's Comment: The original CX was signed on 4/25/2012'."
    },
    {
        "candidate_id": "35f2daaf5891bbbc8b79",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'MSEP Unit 2 Project ROW grant issued August 2014 Pending construction Aurora Solar Kern California 44'."
    },
    {
        "candidate_id": "05cd752743316abb812c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'mitigation measures. APPROVING OFFICIAL: /s/Jayme M. Lopez TITLE: Tucson Field Manager DATE'."
    },
    {
        "candidate_id": "a0a6bebe86c0b3ab49ff",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "8d37067d5bfa57544111",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Carrie Stewart Date Determined: May 10, 2022'."
    },
    {
        "candidate_id": "4c70255b77d6fabc0802",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'JENNIFER \u039c\u0391\u03a4\u0391 Jennifer Mata Field Manager Digitally signed by JENNIFER MATA Date: 2020.08'."
    },
    {
        "candidate_id": "b18ad0962c5e16d49e35",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Digitally signed by TY ALLEN Authorizing Official: Date: 2021.06.04 Date: Carlsbad Field Offic5KPATAGPO'."
    },
    {
        "candidate_id": "ad83c698acf92996bed0",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'the MSA via the 2007 reauthorization The range of alternatives for analysis was approved by the'."
    },
    {
        "candidate_id": "bd03aa58000efe5a96c0",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Stephen A. Danker Date Determined: 09/10/2015 Comments: OBU-N-2015-0093, Rev. 0'."
    },
    {
        "candidate_id": "7569c031e94093858e71",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'and a draft unsigned Finding of No Significant Impact FONSI to the e-Planning website for a'."
    },
    {
        "candidate_id": "bccbfe12a02ec226c07b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 06/10/2014 Comments: TC-A-2014-0041, Rev. 0'."
    },
    {
        "candidate_id": "7d6006d6bf6d812ee7c2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer Date Determined Jul 27 2010 Comments Date 2010 08 20 11 59 23 04'00''."
    },
    {
        "candidate_id": "d132c4eeab9f11a20ff0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Katherine S. Pierce NEPA Compliance Officer Date: April 20, 2015 Attachment: Environmental'."
    },
    {
        "candidate_id": "e372d9c69f69e0e4bfc9",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'L030- 2019-0003-EA) & Finding of No Significant Impact (FONSI) was signed April 5, 2019. The Northeast'."
    },
    {
        "candidate_id": "518913442acea1eb1ab8",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "c663bb106401e0d55423",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "0f42bef3d3bcc8e222c4",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Development Project ROD was approved on September 22, 2016. The CD-C FEIS'."
    },
    {
        "candidate_id": "ed6b945565cc9cf4cd0c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 06/10/2014 Comments: OBU-A-2014-0060, Rev. 0'."
    },
    {
        "candidate_id": "6ce3ce6dce6771f3c2c0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer Signature and Determination Date JOYCE CHAVEZ Digitally'."
    },
    {
        "candidate_id": "f3dae8b6530a61c38e65",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'o DOE-SR ou EQMD Date Determined Jun 14 2011 Date 2011 07 08 15 59 24'."
    },
    {
        "candidate_id": "4a13fcd5df1c1db8335b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '24 09 51 22 05'00' Date Determined 03/24/2014 This form will be locked for editing upon'."
    },
    {
        "candidate_id": "f234c32c6760957aff07",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '31 13:28:11 -04'00' Date Determined: 08/22/2022 Comments: LWO-S-2022-0050, Rev. 0'."
    },
    {
        "candidate_id": "dec86cc8284be0344dbd",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "b13d2d1b643940dff47d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2020.08.24 15:51:06'."
    },
    {
        "candidate_id": "ff97e44c785c38013a2e",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2020.01.02 15:13:41'."
    },
    {
        "candidate_id": "4e0c39838b27d5f1c3f4",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Uncompahgre Field Office, Field Manager DATE SIGNED: 10.23.2015 EXHIBIT A STIPULATIONS'."
    },
    {
        "candidate_id": "07efd4632c0b8548ffef",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environmental review. Authorizing Official: GAVIN LOVELL Digitally signed by GAVIN LOVELL Date'."
    },
    {
        "candidate_id": "22c3bada9484425337f6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '05 09:56:19 -05'00' Date Determined: 01/07/2021 Comments: EP-M-2020-0022, Rev. 0'."
    },
    {
        "candidate_id": "ce6551e2f07121793457",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 4/13/2010 Comments: Webmaster: Record ID: 920'."
    },
    {
        "candidate_id": "01cb8ad659f1208360f0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Comments: OBU - G - 2010 - 028, Rev.0 Digitally signed'."
    },
    {
        "candidate_id": "0189ddc3cefa64e6e17f",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '04 14 11 29 10 04 00 Date Determined 04/14/2023 Comments EEC No LWO-Z-2022-01004 Rev No'."
    },
    {
        "candidate_id": "d2d1853db361688caf9d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '22 16:12:52 -05'00' Date Determined: 11/22/2016 Comments: TC-A-2013-0001, Rev. 1'."
    },
    {
        "candidate_id": "1795eccc347f200dd285",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "cf5cab2c41399a06046a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Official: fr Carlsbad Field Office Manager Date 05/25/2021'."
    },
    {
        "candidate_id": "39bd76169cdd1fccf6bb",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'P060-2014-0135-EIS and Record of Decision approved December 23, 2020. 1 **Page 3** Spruce'."
    },
    {
        "candidate_id": "5fddfe6a29fa01c859bb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '] [x] [ ] Assistant Field Manager \u00a746.215(b) Have significant effects on such unique'."
    },
    {
        "candidate_id": "b96e36ab15ed8a2dfd9b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03.04 10:02:26 -0500 Date Determined: 03/01/2019 Comments: CBU-F-2019-0008, Rev. 0'."
    },
    {
        "candidate_id": "c7ae00b3c938073ae820",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Project Description The right-of-way granted by the Bureau of Land Management (BLM) to SunZia Transmission'."
    },
    {
        "candidate_id": "107bd4545b30a2c5e40f",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'BUREAU OF LAND MANAGEMENT RIGHT-OF-WAY GRANT SERIAL NUMBER WYY-188558 Issuing Office Buffalo Field Office 1 A'."
    },
    {
        "candidate_id": "1ab607b7e0348e4b8f0c",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'with the Ely District Record of Decision and Approved Resource Management Plan (BLM 2008b,'."
    },
    {
        "candidate_id": "fd03330d91bedf1e7cc9",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Number WYW-186925 RIGHT-OF-WAY GRANT/TEMPORARY USE PERMIT 1 A right-of-way permit is hereby granted pursuant to x'."
    },
    {
        "candidate_id": "17f18e9cf50b25bab982",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'discovery until written authorization to proceed is issued by the Authorized Officer In addition the area'."
    },
    {
        "candidate_id": "70e11259bda9461ee2ce",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'proposal is similar to a right-of-way issued by BLM to Hancock Forest Management in September'."
    },
    {
        "candidate_id": "c93176d16c8a0d929728",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Office September 2017 DECISION RECORD BP America Production Company Champlin #452L 11-20H'."
    },
    {
        "candidate_id": "427e5f8f8ce6ab444ade",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Rawlins Field Office RIGHT-OF-WAY GRANT/TEMPORARY USE PERMIT Serial Number WYW-104376 1 A right-of-way permit is hereby'."
    },
    {
        "candidate_id": "cb9e1cf09b6f792c676e",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'issued early in 1999. A ROD (DOE 1999c) was issued in September 1999, and a Mitigation'."
    },
    {
        "candidate_id": "096b10ca7061ac6b0f0a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'S\u00e1nchez Ruben A. S\u00e1nchez Field Manager, Kingman Field Office Date: 07/01/2013 Exhibit: Original'."
    },
    {
        "candidate_id": "10f1f71a2040a7fa79a1",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Office April 2017 DECISION RECORD Power Company of Wyoming Chokecherry and Sierra Madre'."
    },
    {
        "candidate_id": "be9b234bc3f61ca1e5e1",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Number WYW-185371 1 A right-of-way permit is hereby granted pursuant to a Title V of the'."
    },
    {
        "candidate_id": "0a7558e9b420fc39dfac",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'discovery until written authorization to proceed is issued by the Authorized Officer An evaluation of the'."
    },
    {
        "candidate_id": "45a9d815a4f1fecdd7e1",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Office May 2016 DECISION RECORD Kortes Dam Access Road access road. ROW: WYW-185388'."
    },
    {
        "candidate_id": "e772190bda7007229141",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'in May 1990. The Record of Decision approving the airport and conveying the land was issued'."
    },
    {
        "candidate_id": "344faf90f67fccf77a0b",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Office November 2018 DECISION RECORD BP America Production Company Chain Lakes 27-130d'."
    },
    {
        "candidate_id": "9365707360302387ff4c",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Forest Service issued a ROD, and on January 14, 2021, the BLM issued a ROD granting'."
    },
    {
        "candidate_id": "1c3ddf833c0e974ac48a",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Health and issue the grazing permit to the applicant The term of the permits will run'."
    },
    {
        "candidate_id": "a6096360a8614cb34898",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Expenditure Plan for the Reauthorization of the Local Sales Tax for Transportation approved June 17 2003'."
    },
    {
        "candidate_id": "0905061d339be304f5ca",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote '1 3 2 History of BLM Right-of-Way Grants The BLM granted the CVWD a right of'."
    },
    {
        "candidate_id": "b8b0e8da256b961eebb2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'alternative selected by the Forest Supervisor for the Clear Creek Integrated Restoration Project'."
    },
    {
        "candidate_id": "51274c2be4ec7cea2377",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'requested renewal of right-of-way AZA 21339 for a buried water line This right-of-way was originally issued'."
    },
    {
        "candidate_id": "00d7350e4fa26b0197c6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environmental review. Authorizing Official: (Signature) Date: 9/24/2015 Name: Joanna Nara'."
    },
    {
        "candidate_id": "e9a16d3bf82504b3c889",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'August 4, 1995, Right-of-Way CACA-035520 was granted jointly to Bruce L. McKenzie and James W. McKenzie'."
    },
    {
        "candidate_id": "c788d14881cd700e5b91",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Jonathan W. Hartwell Authorizing Official Title Acting Field Manager 05/16/2016 Date Page'."
    },
    {
        "candidate_id": "afadc3f7294e9844627a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Heath Cline for Rawlins Field Manager December 28 2017 Date 5 Appendix 1 Maps Map 1'."
    },
    {
        "candidate_id": "462a12861943686b5494",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'commend that a right-of-way be issued to Cellular Inc. Network Corp d/b/a Verizon Wireless'."
    },
    {
        "candidate_id": "6ff66760312423c7d9ef",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'upon signing by the authorized officer and shall remain in effect pending an appeal (43 CFR'."
    },
    {
        "candidate_id": "afadaf82567e233a16e0",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'SIGNATURE: Jason West Lander Field Office Manager 3/22/2019 Date'."
    },
    {
        "candidate_id": "de9e098ea9eadb336a25",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for Richard A. Fields Field Manager, Farmington Field Office 9-12-18 Date 9/13/18'."
    },
    {
        "candidate_id": "6d7edbb361ae3672329d",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Decision dated 5/1/1980 Right-of-Way CACA 005461 was originally granted to Beacon Oil Company for a period'."
    },
    {
        "candidate_id": "4a9787e732e618e3d4ac",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Management Plan and Record of Decision as amended by the Record of Decision and Approved'."
    },
    {
        "candidate_id": "50b95cdab0d4fe27bd56",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Don McClure, Assistant Field Manager Supervisor Project Description: The BLM-KFO will authorize'."
    },
    {
        "candidate_id": "90b90a65026eb745992e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'rotest filed with the authorized officer shall contain a written statement of reasons for protesting'."
    },
    {
        "candidate_id": "85db322c991d922df2e0",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '523-1256. Signature Authorizing Official: Sariwond (Signature) Name: Lorid woud Date: 4/3'."
    },
    {
        "candidate_id": "095777479e61a834d326",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'structures within this right-of-way in strict conformity with the plan of development which was approved and'."
    },
    {
        "candidate_id": "2913af83f5ddcca7913e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the discretion of the authorized officers. SIGNATURE OF PREPARER: Marnie Medina DATE: 6/21'."
    },
    {
        "candidate_id": "d6e8bf4005a4b118939f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'J. Carpenter Rawlins Field Manager September 15, 2017 Date 5 SWEETWATER CO., WY Appendix'."
    },
    {
        "candidate_id": "5c2e45fe96bb49d89a5e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'J. Phillips Worland Field Office Manager 5/27/16 Date Contact Person For additional information'."
    },
    {
        "candidate_id": "fc26d69322bb96785547",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Powell R. Cade Powell, Field Manager, Cody Administrative Review or Appeal Opportunities'."
    },
    {
        "candidate_id": "3436b979fe0834594c7a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'if applicable): None. Authorized Officer: [Signature] Date: 4/25/2016'."
    },
    {
        "candidate_id": "60638c767ab664178b18",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'letter dated 09/01/2011 Right-of-Way grant was issued to San Pablo Bay Pipeline Co LLC for a'."
    },
    {
        "candidate_id": "cfd9ac871bd4abd78097",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Scott C. Cooke, SFO Field Manager NEPA Coord. Assigned Critical Elements and Other Issues'."
    },
    {
        "candidate_id": "8b770fec78cd18ee0247",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'it is accepted by the Authorized Officer. Authorized Officer: James M. Sparks Date: 10/30'."
    },
    {
        "candidate_id": "16295a4ad5acde0b2878",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the following pages. Authorized Officer: [Signature] Date: 2/7/2017 3928000 3930000 3932000'."
    },
    {
        "candidate_id": "10cd71a7248a4a8ddf54",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'J. Carpenter Rawlins Field Manager January 18, 2018 Date 5 SWEETWATER CO., WY Map 1:'."
    },
    {
        "candidate_id": "fc9f439df04476d9ffa5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Jason West Assistant Field Manager-Land & Resources Date: 4/1/15 Project Description'."
    },
    {
        "candidate_id": "6c114769d0e6f5125835",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'transfer of this FTA authorization from Annova LNG LLC to Annova the applicant in this proceeding'."
    },
    {
        "candidate_id": "a0918a6c7fbaf03fb4d9",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Elizabeth Meyer-Shields Field Manager Mother Lode Field Office 8/14/19 Date NEPA Compliance'."
    },
    {
        "candidate_id": "871211bada23277409ce",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'stay should be granted. Field Manager USDI, Bureau of Land Management Rawlins Field Office'."
    },
    {
        "candidate_id": "01cbf5c0deb8c6f3c570",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Linda R. Price Salmon Field Manager 10/22/2013 Date'."
    },
    {
        "candidate_id": "2302e025d0aaba4b94ea",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Jerry Kenczka, Assistant Field Manager Chapter Chapter 1 Categorical Exclusion D. Conditions'."
    },
    {
        "candidate_id": "b344c8a3b7e62ae61ee9",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date David J. Lefevre Field Manager V. Contact Person For additional information concerning'."
    },
    {
        "candidate_id": "12e38007b1a747d15e26",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'requested renewal of Right-of-Way UTU-57093 Under the authority that it was granted the right-of-way may be'."
    },
    {
        "candidate_id": "ab3cef64a75ecb042f1f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Ruben Sanchez Assistant Field Manager Division of Lands & Minerals Roswell Field Office'."
    },
    {
        "candidate_id": "6ed4a03eddb2a582a017",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environment. D. SIGNATURE Authorizing Official: /s/Dennis J. Carpenter September 18, 2017 Field'."
    },
    {
        "candidate_id": "b94caa32903b40cac7b9",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'EIS and prepared the ROD, which is attached as an appendix to the Order. The'."
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
