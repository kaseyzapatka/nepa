import pandas as pd


LABELS = [
    {
        "candidate_id": "ec8c90fd13937edd93ed",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '24 13:41:09 -05'00' Date Determined: 11/04/2020 Comments: TC-A-2020-0081, Rev. 0'."
    },
    {
        "candidate_id": "16d8cb9a87029a773e35",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Digitally signed by Neil Kirschner DN: cn-Neil Kirschner'."
    },
    {
        "candidate_id": "f8ad65c9dbfbcbfc5fcf",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'OFFICIAL: Smith B. Mcle a Field Manager DATE SIGNED: 9/3/15 DOI-BLM-CO-N010-2015-0039'."
    },
    {
        "candidate_id": "5abeb3fd5193315e8e44",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Jenny Tennant DATE: 12 / 10 / 2009 month day year'."
    },
    {
        "candidate_id": "f12149037f05613fb560",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '02 13:43:27 -05'00' Date Determined: 01/31/2022 Comments: OBU-A-2022-0012, Rev. 0'."
    },
    {
        "candidate_id": "5522c338576fdf75a086",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '08 01 10 35 01-04 00 NEPA Compliance Officer John Ganz Digitally signed by John Ganz'."
    },
    {
        "candidate_id": "5b2029857578f8cdc751",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Steven Markovich DATE: 01 / 15 / 2010 month day year'."
    },
    {
        "candidate_id": "9bc44e604159634e9243",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '29 18:20:32 -04'00' Date Determined: 06/28/2017 Comments: TC-A-2017-0045, Rev. 0'."
    },
    {
        "candidate_id": "4914abcbf40db202e5ae",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Robin W. Ames Date: 04/17/2015 month day year NEPA'."
    },
    {
        "candidate_id": "499455e25c73231b2983",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Manager Doug Herrema, Field Manager Janet Cheek, acting for Digitally signed by Janet'."
    },
    {
        "candidate_id": "e7035b5f26826189d039",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "32043382714ed7f7cec4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'signed by MATTHEW MARSH Date Determined: 04/03/2018 Date: 2018.04.03 15:45:28-06'00''."
    },
    {
        "candidate_id": "87e04522fe1ca6fa2b8e",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '06.24 14:42:05-04'00' Date Determined: 06/17/2016 Comments: OBU-N-2016-0018, Rev. 1'."
    },
    {
        "candidate_id": "d3905762c665a1e49f89",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '30 15:07:15 -05'00' Date Determined: 11/01/2021 Comments: OBU-G-2020-0152, Rev. 1'."
    },
    {
        "candidate_id": "bbe1459cedbcb71d1eb8",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2024-00004 Rev No: 0 Date Determined: 01/12/2024'."
    },
    {
        "candidate_id": "deb370ef2cef4fd1bfe4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 03/11/2013'."
    },
    {
        "candidate_id": "34097e863052979a461f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Jamie Moeini, Assistant Field Manager Las Vegas Field Office, Division of Lands Categorical'."
    },
    {
        "candidate_id": "50c6c9be4162e4965b43",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '30 17:03:07 -05'00' Date Determined: 11/19/2020 LWO-Z-2020-0034, Rev. 0 Comments:'."
    },
    {
        "candidate_id": "51ac5a45c9d6ea477006",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'RECORD OF THIS DECISION. NEPA Compliance Officer Signature: Kristin Kerwin Date: 9/22/2010 NEPA'."
    },
    {
        "candidate_id": "f1854ea06c9c54ff9e76",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the Agency Official: Field Manager, El Centro Field Office Date 7/10/2014 Programmatic'."
    },
    {
        "candidate_id": "8fa6217619c5566c81ab",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'review) Public Hearing March 22, 2018 Publication of Final EIS September 14, 2018'."
    },
    {
        "candidate_id": "8e0188ee99421352f063",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JOSE FIGUEROA Date: 02 / 08 / 2018 month day year'."
    },
    {
        "candidate_id": "2a429ce223942b2c4951",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 08/26/2014 Comments: TC-A-2014-0061, Rev. 0'."
    },
    {
        "candidate_id": "892715fb8f2c1e2e59b0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2013 month day year NEPA Compliance Officer: John Ganz Digitally signed by John Ganz DN: cn-John'."
    },
    {
        "candidate_id": "61b99ed340bb8b5090ad",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '28 13:19:29 -07'00' Field Manager Date E. Contact For more information, contact Lindsey'."
    },
    {
        "candidate_id": "99c0d71c63d208bb8d16",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Date: June 7, 2017 Attachment(s): Environmental'."
    },
    {
        "candidate_id": "9405520e068889278f97",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 4/30/2010 Comments: Webmaster: Record ID: 94'."
    },
    {
        "candidate_id": "2183731131e4743aa74a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Knapp o DOE ou OCRWM Date Determined Apr 22 2010 email-kathryn knapp ymp gov c US'."
    },
    {
        "candidate_id": "1265a77c342fac7836b5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Erin Russell-Story DATE: 06 / 21 / 2011 month day'."
    },
    {
        "candidate_id": "7961fcef1e2fdc9fb1c3",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Reviewer: Assistant Field Manager M&L Date Ben Se Authorizing Officer 7-5-23 Date Contact'."
    },
    {
        "candidate_id": "aad1da154449e236c51b",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the document with the authorized Officer. JENNIFER MATA Field Manager Digitally signed by JENNIFER'."
    },
    {
        "candidate_id": "4ee33d9f8e6baee8f98f",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 18 14 25 33 05 00 Date Determined 01/18/2023 Comments EEC No TC-W-2023-00001 Rev No'."
    },
    {
        "candidate_id": "0f6b04badfdbbf44c8c6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environmental review. Authorizing Official: KIMBERLEE FOSTER Digitally signed by KIMBERLEE FOSTER'."
    },
    {
        "candidate_id": "ca1909f058370919d264",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: CHARLES PRUSS Date: 03/07/2024 month day year NEPA'."
    },
    {
        "candidate_id": "48dade54e926cf530e8b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 10/1/2010 Comments: Webmaster: Record ID: 5'."
    },
    {
        "candidate_id": "c8b1bd8324f276d2fe37",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'resource values and uses.\" ROD Management Objective 2: \"Provide opportunities for'."
    },
    {
        "candidate_id": "1e5a6571e228b4338261",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 07/24/2012 Comments: FSSBU-H-2012-0002, Rev. 0'."
    },
    {
        "candidate_id": "8916dcebfff7feb58a2f",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '26 16 23 21 04'00' Date Determined May 24 2010 Submit via Email Submit to Website'."
    },
    {
        "candidate_id": "099ff65b511eb128119c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '14 month day year NEPA Compliance Officer John Ganz Digitally signed by John Ganz DN cn'."
    },
    {
        "candidate_id": "d228c54cf895c9f2a88f",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'effects of the proposal. NEPA Compliance Officer Signature and Determination Date Joyce E. Chavez Digitally'."
    },
    {
        "candidate_id": "47db80c5f107ad141220",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'CLEMENTSON Connie Clementson Field Manager Digitally signed by CONNIE CLEMENTSON Date: 2022'."
    },
    {
        "candidate_id": "d570d0a93ccbd4ad5a32",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Geoffrey Goode Digitally signed by Geoffrey Goode'."
    },
    {
        "candidate_id": "170717806d25686c08ec",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 07/16/2020 Comments: OBU-F-2020-0175, Rev. 0'."
    },
    {
        "candidate_id": "54f0951e7a0155cc03be",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'MEASURES/OTHER REMARKS: APPROVING OFFICIAL: /s/ Jayme M. Lopez TITLE: FIELD MANAGER DATE: 12'."
    },
    {
        "candidate_id": "9df20d065cfe7aea330d",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'public comment period \u2022 August 2023 \u2013 Final EIS publication \u2022 October 2023 \u2013 Record of Decision Next'."
    },
    {
        "candidate_id": "f2480a6b137cf369a66b",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: OBU-A-2011-00013 - Rev'."
    },
    {
        "candidate_id": "ebec270428b4354ba404",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer James Elmore Date Determined: 4/13/2010 Comments: Webmaster: Record ID: 1050'."
    },
    {
        "candidate_id": "cdffba26bf32843eaa13",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 03/20/2017 Comments: TC-A-2013-0049, Rev. 1'."
    },
    {
        "candidate_id": "dfb9e6282f1346fe3d99",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Kristen Kief DATE: 08 /02 / 2010 month day year NEPA'."
    },
    {
        "candidate_id": "551f3c8160db1d1038e4",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote '1997). DOE issued a FONSI on January 12, 1987 for the 1987 EA and November'."
    },
    {
        "candidate_id": "8ba1effda292542fbc83",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy A. Ribeiro Date Determined: 2017.07.03 12:27:15-06'00''."
    },
    {
        "candidate_id": "4caffd8be565745535a4",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "3b975017b65ddfc98ac1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Carrie Stewart Date Determined: January 11, 2018'."
    },
    {
        "candidate_id": "4f75e3d240a659be47c3",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Steven Richardson DATE: 04/14/2010 month day year'."
    },
    {
        "candidate_id": "e780cca8ef01a81ec953",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'significant impacts. Signature Authorizing Official:/s/Michael J. Phillips Michael J. Phillips, Worland'."
    },
    {
        "candidate_id": "2b947510864e0936e6af",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'active ingredients. The Record of Decision, signed December 22, 2016, approved the new herbicide'."
    },
    {
        "candidate_id": "fcd9b414f6e99bd412ed",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Teresa Jones DATE: 5 / 17 / 2011 NEPA Compliance'."
    },
    {
        "candidate_id": "ffd0a89dc80b8b865fe4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 12/6/2011'."
    },
    {
        "candidate_id": "f9d42374519e9bd351fd",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2022-01011 Rev No: 0 Date Determined: 12/01/2022'."
    },
    {
        "candidate_id": "85705247cb3f237d8dd7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Reviewer: Assistant Field Manager M&L Date 6-23-2022 Authorizing Official/Field Manager'."
    },
    {
        "candidate_id": "fbf8472b6fe60117df16",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Cheryl Adcock Siuslaw Field Manager Date: Feb 23, 2018 -3-'."
    },
    {
        "candidate_id": "ce83405d55ad00304bcb",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2023-00010 Rev No: 0 Date Determined: 04/11/2023'."
    },
    {
        "candidate_id": "921332dd26363c7e9eb7",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '26 18:29:26 -04'00' Date Determined: 06/26/2013 Comments: TC-A-2013-0071, Rev. 0'."
    },
    {
        "candidate_id": "8c6ff02701823acdd93e",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 12/02/2016 Comments: OBU-F-2016-0141, Rev. 0'."
    },
    {
        "candidate_id": "7109e162735f57a4ccc5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date Timothy D. Gilloon Field Manager'."
    },
    {
        "candidate_id": "190f22a79e6ddf7a129e",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03.21 17:38:10 -04:00 Date Determined: 03/20/2018 Comments: OBU-G-2018-0057, Rev. 0'."
    },
    {
        "candidate_id": "f27db51993891dc0d3b0",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: KAROL SCHREMS Date: 5 / 29 /2019 NEPA Compliance'."
    },
    {
        "candidate_id": "0b88d8843a83ef16d823",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Roak Parker Digitally signed by Roak Parker Date'."
    },
    {
        "candidate_id": "3548ce480b5258da36ae",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'development project; the FONSI and Decision Record were signed June 2, 2014.'."
    },
    {
        "candidate_id": "32b46a7240905a45abe9",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Action NMFS issued a Finding of No Significant Impact FONSI on the Proposed Action in January'."
    },
    {
        "candidate_id": "f1ebc3ba2873a683fa48",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Comments: Digitally signed by Andrew R. Grainger'."
    },
    {
        "candidate_id": "8d93d3dddb6f81ab8ff3",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: SEAN BERRY Digitally signed by SEAN BERRY Date:'."
    },
    {
        "candidate_id": "112789607ef7ede18744",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'grainger srs gov c US Date Determined May 24 2010 NEPA Compliance Officer Comments Date 2010'."
    },
    {
        "candidate_id": "cf16c24ab8e04a0b3764",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Timothy J. La Marr Field Manager, Central Yukon Field Office Contact Person Al Burton'."
    },
    {
        "candidate_id": "503178100076edd2e832",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'determined by the BLM Authorized Officer. 10. For the purpose of determining joint maintenance'."
    },
    {
        "candidate_id": "a5b69f347efe2c4b5707",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'ALEXANDER PRINGLE Assistant Field Manager Digitally signed by ALEXANDER PRINGLE Date: 2022.06'."
    },
    {
        "candidate_id": "0daa7a7bd8d165013281",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '06'00' Date Assistant Field Manager Date WILLIAM MIER Digitally signed by WILLIAM MIER'."
    },
    {
        "candidate_id": "64adcc92050df10b3bcb",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2017 month day year NEPA Compliance Officer: FRED POZZUTO Digitally signed by FRED POZZUTO Date'."
    },
    {
        "candidate_id": "a3806c7b2adb00f7b5ea",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer Signature and Determination Date Joyce E. Chavez Date'."
    },
    {
        "candidate_id": "e51bfdeb4e60beb8ac05",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'PAWELEK Robert Pawelek Field Manager Oklahoma Field Office Digitally signed by ROBERT PAWELEK'."
    },
    {
        "candidate_id": "812b65439631c2df13db",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Stephen A. Danker Date Determined: 05/21/2015 Comments: CBU-H-2015-0021, Rev. 0'."
    },
    {
        "candidate_id": "5ff27c878f20f7c4480a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '06 18:13:48 -04'00' Date Determined: 06/30/2022 Comments: OBU-K-2022-0159, Rev. 0'."
    },
    {
        "candidate_id": "ce01f71ba6b0210a3cfc",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2018.01.22 12:51:41'."
    },
    {
        "candidate_id": "4bcfd303ca29e7981509",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '580-5649. Rem Hawes Field Manager /S/ 4-11-14 Enclosures HFO:hconner:db:x649:4/3/14'."
    },
    {
        "candidate_id": "03ea3da67d9edf1d664b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 2/15/2011 Comments: Webmaster: Record ID: 203'."
    },
    {
        "candidate_id": "ecb0308e7cef0baf7b0a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Compliance Officer: Date Determined: 11/03/2016 (This form will be locked for editing upon signature'."
    },
    {
        "candidate_id": "be5b7dbbe5646bfb428d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Fields Susan Fields Date Determined: 5/18/18 Based on my review of the proposed action, as NEPA'."
    },
    {
        "candidate_id": "ca5a30e0c638c00e104d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '09.02 10:05:25-04:00 Date Determined: 08/25/2020 Comments: TC-W-2020-0074, Rev. 0'."
    },
    {
        "candidate_id": "0ae37144a4da7089c061",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '10(b)). SIGNATURE OF AUTHORIZED OFFICER SUZANNE COPPING Digitally signed by SUZANNE COPPING'."
    },
    {
        "candidate_id": "5a48b9e350d6c5b0fdaf",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'P060-2014-0135-EIS and Record of Decision approved December 23, 2020. 1 WOLD PATTERSON 3874'."
    },
    {
        "candidate_id": "23e160ad7de4dbd20b4c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Loren C. Wickstrom Field Manager, North Dakota Field Office 2 DEPARTMENT OF THE INTERIOR'."
    },
    {
        "candidate_id": "2e94194c558b6c8b3fd9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2012 month day year NEPA Compliance Officer: john ganz Digitally signed by john ganz DN: cn-john'."
    },
    {
        "candidate_id": "202fb183f49b21c9d2a6",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2021 month day year NEPA Compliance Officer: Mark Lusk Digitally signed by Mark Lusk Date: 2021'."
    },
    {
        "candidate_id": "ab645e7c38c079ce7796",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Elizabeth Meyer-Shields Field Manager Mother Lode Field Office Date H. Contact For more'."
    },
    {
        "candidate_id": "292f5534b85c7c4a5ae1",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Initials Date Wells Field Office Manager B. Mulligan None /s/ BAM 6/18/2015 Tuscarora Field'."
    },
    {
        "candidate_id": "15edb5c3c33ce405c768",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: George S. Darakos Digitally signed by George S. Darakos'."
    },
    {
        "candidate_id": "dbcb3c4264c82319d9a6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'D: Signature AMANDA Authorizing Official: DODSON Digitally signed by AMANDA DODSON Date: ['."
    },
    {
        "candidate_id": "aae95fe5028f275e1fb9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Signature Brian Mills NEPA Compliance Officer Office of Electricity Delivery and Energy Reliability'."
    },
    {
        "candidate_id": "4a4aa64a41329d072a32",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'P060-2014-0135-EIS and Record of Decision approved December 23, 2020. Table 1. Short and'."
    },
    {
        "candidate_id": "fcf9d8a6704208916a7b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: DATE: month day year NEPA Compliance Officer: Cliff'."
    },
    {
        "candidate_id": "977ef75b4bc855a95c11",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Appeal is filed with the Authorized Officer. Signature of Authorized Official William J Mill MILLS'."
    },
    {
        "candidate_id": "6e0072b9391b31a13525",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '2013 for Leon Thomas Field Manager Sierra Front Field Office (date) SFFO June 2012'."
    },
    {
        "candidate_id": "e5dbe1ad239752c1b4ce",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'Significant Impact issued on January 5 2018 which is incorporated by reference This DR also incorporates'."
    },
    {
        "candidate_id": "0d9151fb649790eceb7b",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Monmouth IL and received a Record of Decision 2003 ROD on August 18 2003 Since the'."
    },
    {
        "candidate_id": "9f58c34557877beddd35",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '06/11/18 Assistant Field Manager E. Approval I have reviewed the Proposed Action for'."
    },
    {
        "candidate_id": "2f0810fa2bb8e38fbdf0",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Richard C. Baker Digitally signed by Richard C. Baker'."
    },
    {
        "candidate_id": "417cbae17b074f0a23c9",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 03/24/2014 Comments: TC-A-2013-0117, Rev. 1'."
    },
    {
        "candidate_id": "36d217357e382e49f535",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '05 11 37 10-04'00' Date Determined Apr 16 2010 TC-A-2010 020 Rev 0 Submit via Email'."
    },
    {
        "candidate_id": "543c015675d25850e0f9",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Othalene Lawrence Date Determined: Apr 20, 2010 Comments: Webmaster: Billie Newland THINK BEFORE YOU'."
    },
    {
        "candidate_id": "142716cdb2db712b2220",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '19 14:36:12 -04'00' Date Determined: 05/11/2021 OBU-G-2021-0113, Rev. 0 Comments:'."
    },
    {
        "candidate_id": "7a2d7d76a6b870a73857",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '1, February 4, 2010 Date Determined: 2/9/15'."
    },
    {
        "candidate_id": "81625e0886b91673c008",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'action. \u0414. hlan Assistant District Manager Date: July 14, 2017'."
    },
    {
        "candidate_id": "284c8963020ce513659b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01.22 15:38:08 -05:00 Date Determined: 01/22/2019 Comments: OBU-L-2017-0110, Rev. 3'."
    },
    {
        "candidate_id": "79a02935de6e3aa1e6a7",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '22 18:22:54 -04'00' Date Determined: 08/14/2012 Comments: OBU-C-2012-0040, Rev. 1'."
    },
    {
        "candidate_id": "fa4ede81009d1bd7ef84",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Gary Covatch Digitally signed by Gary Cavatch DATE'."
    },
    {
        "candidate_id": "f3fdadf2d8a39008bda5",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Travel Plan Revision Record of Decision and Final Supplemental Environmental Impact Statement'."
    },
    {
        "candidate_id": "0b1de8116a6329e65482",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2022 month day year NEPA Compliance Officer: JILL TRIULZI Digitally signed by JILL TRIULZI Date'."
    },
    {
        "candidate_id": "12972cd464c991122580",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: BRIAN MOLLOHAN Date: 05 / 09 / 2014 month day year'."
    },
    {
        "candidate_id": "41ce99cbef888d0c9645",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '47 -08'00' TAPS ROW Authorized Officer (Acting) Deputy State Director, Cadastral (Acting'."
    },
    {
        "candidate_id": "90e9fc6057ebd7616629",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 07/22/2016 Comments: FSSBU-A-2016-0002, Rev. 0'."
    },
    {
        "candidate_id": "bc8e44b40aee3530c85a",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'assessment (DOE 1994b); a Finding of No Significant Impact was issued in August 1994, Additional storage'."
    },
    {
        "candidate_id": "73f9978af7257e79b3dc",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '12 11 15 30 57 05 00 Date Determined 12/11/2023 Comments EEC No TC-A-2022-01012 Rev No'."
    },
    {
        "candidate_id": "b6bbbf2137534e69d0c6",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Complex, December 1998 (ROD signed May 17, 1999). This EIS assessed the potential'."
    },
    {
        "candidate_id": "b94fb7481cf746a1cba5",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '05 08:42:02 -05'00' Date Determined: 12/16/2021 Comments: LWO-Z-2021-0078, Rev. 0'."
    },
    {
        "candidate_id": "6a6adf2951e4941b414a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 13:50:11 -04'00' Date Determined: 07/12/2022 Comments: LWO-S-2022-0040, Rev. 0'."
    },
    {
        "candidate_id": "b4d21888694509e3f573",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Joseph B. Renk III DATE: 05 / 04 / 2010 month day'."
    },
    {
        "candidate_id": "986194b3ed095a067cbb",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'of Draft EIS March 2 April 16 2018 45 day review Public Hearing March 22 2018'."
    },
    {
        "candidate_id": "4ed7d3bb61534d47e0c1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '15 15:16:39 -04'00' Date Determined: 05/12/2020 Comments: OBU-H-2020-0117, Rev. 0'."
    },
    {
        "candidate_id": "22101a6ed1f704d5564b",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'RECORD OF THIS DECISION. NEPA Compliance Officer Signature: NEPA Compliance Officer Date: 7/16/10'."
    },
    {
        "candidate_id": "97ca89894d48866cb26e",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Andrew R. Grainger Digitally signed by Andrew R.'."
    },
    {
        "candidate_id": "3bf1297f9d1197e6a8ef",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Attachment(s): Environmental Checklist Date: April'."
    },
    {
        "candidate_id": "abb00eb30fa3adb00ecc",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'analysis is required. Approving Official: WILLIAM MACK Title: Field Manager, Yuma Field Office'."
    },
    {
        "candidate_id": "88e14e46ec82d3852e87",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Date Determined: 02/02/2023 Digitally Signed By WILLIAMS'."
    },
    {
        "candidate_id": "c7e3bc7ec6d2e863766e",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Andrea McNemar DATE: 05 /04 / 2010 month day year'."
    },
    {
        "candidate_id": "d6dc1916d1600ad7a036",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Gant) Massey Assistant District Manager, Resources NEPA Compliance JGM 12/15/2023 Brian Kennedy'."
    },
    {
        "candidate_id": "5a8de266a1bfb594e88d",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'analysis is warranted A Finding of No Significant Impact was signed on February 13 2009 pertaining'."
    },
    {
        "candidate_id": "6daef7ce91696306b1ee",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Comments: TC-A - 2010 - 059, Rev.0 DN: cn=Andrew'."
    },
    {
        "candidate_id": "15bb42dad6932aafe177",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: TC-A - 2010 - 083, Rev'."
    },
    {
        "candidate_id": "959e4a4a7a6678a6969a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '10/23 Lisa Wilkolak Authorized Officer: Dave Pals, Field Manager Date: 10/11/23 Dave Pals'."
    },
    {
        "candidate_id": "03343084e8d2f589b717",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 12/08/2015 Comments: DOE-G-2014-0001, Rev. 0'."
    },
    {
        "candidate_id": "0189b2130fac46b3c7b3",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "b6504bbad96116e612e2",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: ANDREW JONES Date: 02 / 21 / 2020 Digitally signed'."
    },
    {
        "candidate_id": "4db37e7076981a0ae94e",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'by a 30-day scoping/public comment period An amended ROD was issued on October 12 2011 DOE'."
    },
    {
        "candidate_id": "eebb86f6adf2c4588bef",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Joseph P. Kanosky Digitally signed by Joseph P. Kanosky'."
    },
    {
        "candidate_id": "c5ac2e8d1b01900954bb",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'which BLM will issue a Record of Decision that publicly states its decision. Developments on'."
    },
    {
        "candidate_id": "33e413fc6e6639bab32f",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'January 19 2010 through February 19 2010 the Air Force conducted a total of 20 public'."
    },
    {
        "candidate_id": "39ed6f7303c68b5f3fd2",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Rondle E Harp DATE: 09 /19 / 2011 month day year'."
    },
    {
        "candidate_id": "7269897a02b88926a6ae",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 09/04/2018 Comments: PBU-K-2018-0006, Rev. 0'."
    },
    {
        "candidate_id": "f3de4c0e02902b0f61ee",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Official Catrina Williams, Field Manager Red Rock/Sloan Field Office Date: 2023.11.09 15'."
    },
    {
        "candidate_id": "88f9299e1d1cd24c6262",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Valley Field Office Record of Decision and Approved Resource Management Plan (ROD/RMP). Date'."
    },
    {
        "candidate_id": "04603de3a0a0c4adfef7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for July 14, 2017 Field Manager Date 7 Appendix 1: Maps Map 1: Luman 9-10H & 9-20H'."
    },
    {
        "candidate_id": "19d23136d34c8e5fb9a7",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "fde39b75409435d96025",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2020 month day year NEPA Compliance Officer: Mark Lusk Digitally signed by Mark Lusk Date: 2020'."
    },
    {
        "candidate_id": "62d1d77da577da1e0b63",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: [Signature field] NEPA Compliance Officer: Cliff'."
    },
    {
        "candidate_id": "c4031b4fccd117ee21c9",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '20 15:04:30 -05'00' Date Determined: 02/14/2019 Comments: OBU-N-2019-0047, Rev. 0'."
    },
    {
        "candidate_id": "d17715a6b3552e46fb61",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: Digitally signed by'."
    },
    {
        "candidate_id": "d84852e5ea1853fd1e86",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2014 month day year NEPA Compliance Officer John Ganz Date 2014 05 27 15 54 30-04'."
    },
    {
        "candidate_id": "c8a1103719a059e16577",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Gary Covatch Digitally signed by Gary Cavatch NEPA'."
    },
    {
        "candidate_id": "fe0c0918563c491ef316",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Gary Covatch NEPA Compliance Officer: john ganz NCO'."
    },
    {
        "candidate_id": "4f768882ffe3fc47410f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Sullivan, Upper Willamette Field Office Manager Date: 3/10/17 Contact Person For additional information'."
    },
    {
        "candidate_id": "7dad9c5715a22ee894f1",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 12/30/2019 Comments: OBU-H-2019-0176, Rev. 0'."
    },
    {
        "candidate_id": "a9c09124b1be0c800eac",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'EA-04-087 16 CBNG 05/05/2004 Federal W-67912 15-15 aka USA 15-15 WY-3109/82-439-P 1 Oil 03/03/1982 Powder River'."
    },
    {
        "candidate_id": "7e6c1222c14c5eb3f53a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'accordance with the Field Manager's Final Decision dated February 20, 2008, for'."
    },
    {
        "candidate_id": "2d599b17d83086215557",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 08/14/2012 Comments: TC-W-2012-0089, Rev. 0'."
    },
    {
        "candidate_id": "469580a26c648265a928",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '10.08 18:09:06-04:00 Date Determined: 09/26/2018 Comments: TC-A-2012-0078, Rev. 2'."
    },
    {
        "candidate_id": "bd67f58453552d78839b",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'P060-2014-0135-EIS and Record of Decision approved December 23, 2020. 1 Devon's SSU MLT'."
    },
    {
        "candidate_id": "b4ef3cbefec7c2868059",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'apply. D: Signature Authorizing Official: Beth Ransel, Field Manager Date: 9/17/15 Contact'."
    },
    {
        "candidate_id": "0548b720099876cc7135",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 30 08 32 43 05 00 Date Determined 01/30/2024 Comments EEC No DOE-K-2024-00001 Rev No'."
    },
    {
        "candidate_id": "9af0c153f88c74dde76c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2024 month day year NEPA Compliance Officer: STEPHEN WITMER Digitally signed by STEPHEN WITMER'."
    },
    {
        "candidate_id": "56ce6261a02298a0e6ec",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'written approval from the authorized officer. E. Preparer/s DENISE BOUDREAULT Project Lead Digitally'."
    },
    {
        "candidate_id": "e1bc7da63f2daf86541c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer James L. Elmore Date Determined: 9/21/2011 Comments: Record ID: 280 Webmaster:'."
    },
    {
        "candidate_id": "8711add36c55e8652c1c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Attachment C). E. Signature Authorizing Official: /s/ Codie Martin Name: Codie Martin Title: Field'."
    },
    {
        "candidate_id": "2c9685a0006b67e5e9b5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: C. Elaine Everitt Date: 9 /17/2013 month day'."
    },
    {
        "candidate_id": "0d27e3e22a79ccd8a36c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '10 19 07 59 08 04 00 Date Determined 10/19/2022 Comments EEC No OBU-H-2022-01010 Rev No'."
    },
    {
        "candidate_id": "73f90f90fcf45cde2f80",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Assessment (EA) and Finding of No Significant Impact (FONSI) in the Federal Register in December 2016'."
    },
    {
        "candidate_id": "eaefcea70a13f8fc5aba",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Loren C. Wickstrom Field Manager, NDFO 5 Attachment 1 \u2013 Surface Conditions of Approval'."
    },
    {
        "candidate_id": "9b2f5faf3bad91f8eb40",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 01/03/2018 Comments: TC-A-2017-0104, Rev. 0'."
    },
    {
        "candidate_id": "c8e029cfa67056d7d1bc",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'documentation. Steve Tuggle NEPA Compliance Officer Date: 2/16/10 Approved CC: Requestor: Susan Sinclair'."
    },
    {
        "candidate_id": "cb363ce6ed560737440b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '30 14:32:45 -05'00' Date Determined: 11/12/2020 Comments: OBU-H-2020-0283, Rev. 0'."
    },
    {
        "candidate_id": "6a03e972b20459268211",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 09/04/2018 Comments: TC-A-2015-0068, Rev. 2'."
    },
    {
        "candidate_id": "38ce7d23e11c7b750114",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '15 17:13:24 -04'00' Date Determined: 05/08/2018 Comments: LWO-H-2017-0006, Rev. 0'."
    },
    {
        "candidate_id": "1aee4d88c556c15a706f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Todd D. Yeager Buffalo Field Office Manager Page 2 Bureau of Land Management Buffalo Field Office'."
    },
    {
        "candidate_id": "5ea744f56125f2cfbfba",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'ENVIRONMENTAL IMPACT STATEMENT March 2024 Record of Decision 13-38 13 9 5 1 Outreach to EJ'."
    },
    {
        "candidate_id": "9a845fab08fa384fc7b8",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: [Signature] Date Determined: 08/12/2015'."
    },
    {
        "candidate_id": "1540803e925c652b686b",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'significant impact (FONSI \u2013 signed September 26, 2012). My decision to authorize'."
    },
    {
        "candidate_id": "2e4524873df1331b3b67",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '13 Will Runnoe Date Field Manager Contact Person For additional information concerning'."
    },
    {
        "candidate_id": "c5354a82195e370296aa",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BRIAN O'PALKO Date: 07/12/2018 month day year NEPA'."
    },
    {
        "candidate_id": "30527d482b151e58ba03",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Date: 2019.04.10 15:36:59'."
    },
    {
        "candidate_id": "ec31654e29987ce267cf",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Coordinator SIGNATURE OF AUTHORIZED OFFICER SUZANNE COPPING Digitally signed by SUZANNE COPPING'."
    },
    {
        "candidate_id": "da29416b84b3cfb1f50d",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Recommending Official Authorizing Official Jared Bybee Robbie McAboy Field Manager, Bristlecone'."
    },
    {
        "candidate_id": "3927894f2d496c30eb82",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 04/17/2020 Comments: OBU-A-2020-0092, Rev. 0'."
    },
    {
        "candidate_id": "a024e38fa51ae1d5c17d",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the document with the Authorized Officer and/or IBLA. /s/ Matt Preston Matt Preston Salt Lake'."
    },
    {
        "candidate_id": "97d33b4f75fc4f01b579",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Andrew R. Grainger Digitally signed by Andrew R.'."
    },
    {
        "candidate_id": "fd67f2124ca2a89ea495",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '12:10:32 -07'00' Name Authorized Officer: DAVID PALS Digitally signed by DAVID PALS Date: 2024'."
    },
    {
        "candidate_id": "94f1a896b0725c0d0424",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Signature of the Authorized Officer Josh Cocke Acting Field Manager Date'."
    },
    {
        "candidate_id": "d46c3dcd7a9eb4e6e187",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Bryant Smith Carson City District Manager 4/14/14 (date)'."
    },
    {
        "candidate_id": "137b2e0c99a97a4135ca",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'DM 2 apply. Signature Authorizing Official: /s/ John R. Elliott John R. Elliott, Lander Field'."
    },
    {
        "candidate_id": "f9cbe84febf3e13caea3",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Gregory L. Williamson Date: 2022.08.24 15:19'."
    },
    {
        "candidate_id": "097e19b50e5c6865fbca",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "44b7986f6c883a721aab",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Loren C. Wickstrom Field Manager North Dakota Field Office 2 Attachment 1 \u2013 Surface'."
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
