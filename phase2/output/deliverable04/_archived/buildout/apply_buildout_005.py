import pandas as pd


LABELS = [
    {
        "candidate_id": "eabe12ba63e61fa70ba1",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'Federal Register notice NOI to prepare Supplemental EIS published in Federal Register 7 calendar days June'."
    },
    {
        "candidate_id": "7f2e76b299e9f5e9ca91",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'the 60-day review and comment period on the Draft EIS started on March 28 2003 and'."
    },
    {
        "candidate_id": "c485f89d22d0aa52ae5d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "41010e94ec386fe4d6c9",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "6556f5e8bffb9ff2d40e",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "afdbeb7215ebd39d51f4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "b7454a1602083e043d51",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'new grazing permit On April 13 2022 the Lewistown BLM Field Office received completed transfer of'."
    },
    {
        "candidate_id": "e1c51fd365724d256b72",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Robert Gross Digitally signed by Robert Gross department'."
    },
    {
        "candidate_id": "36be33cd6ee493766d87",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "a70ea3caf84b44dc4bf3",
        "label": "initiation",
        "notes": "Initiation: application/ROW filed or received, quote 'Health System UNHS filed an application for an assignment and renewal of communication site lease UTU-88558'."
    },
    {
        "candidate_id": "1f84283b9066b2aa0e29",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "1f132e273fc9ef0917b8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "31bb36ef48b80ac12543",
        "label": "neither",
        "notes": "Neither: map/figure/print date, quote '2 1 5 Miles 1 50 000 Date Created 11/16/2016 Created By mpereira NAD 1983 UTM'."
    },
    {
        "candidate_id": "e0e1a839fdfe1fd6cf80",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'comment period April 30 through July 01 2013 and conducted a public meeting in Corona de'."
    },
    {
        "candidate_id": "4213b14160cb4df95f1a",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Steven Richardson DATE: 08 /04 / 2011 month day'."
    },
    {
        "candidate_id": "8b54b2c6c6b4b1403349",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "0ed0cd4481503bdc3d63",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'alternatives and methods for public comment The comment period closed on September 30 2014 The final'."
    },
    {
        "candidate_id": "f533928943b15dc30c49",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "28827a2967ff2c9751f9",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Accession No ML17047A679 April 20 2017 Letter from NRC to Joe Bunch United Keetoowah Band of'."
    },
    {
        "candidate_id": "161b0be0174e84fa111a",
        "label": "initiation",
        "notes": "Initiation: NOI published/issued, quote 'EIS preparation On November 4 2013 Ecology issued a SEPA Determination of Significance The Notice of'."
    },
    {
        "candidate_id": "35ba0764ecbf08d41945",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "f39541878228e12027c3",
        "label": "neither",
        "notes": "Neither: comment-period close/deadline, quote 'October 16 2015 The DEIS comment period began on October 23 2015 and ended December 7'."
    },
    {
        "candidate_id": "08a60c134635a1afad33",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote 'of the notice or by November 20 2014 or 15 days after the last public meeting'."
    },
    {
        "candidate_id": "5c0b2e930144f6cd0c13",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date'."
    },
    {
        "candidate_id": "358fb2610e5ed254e3a7",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote 'scoping meetings in October 1993 at each site In February 1994 BPA released an Implementation Plan'."
    },
    {
        "candidate_id": "bea03480d4cb67ad13fc",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Timothy Hammond, EIFO Field Manager Date Authorized Officer TIMOTHY HAMMOND Digitally'."
    },
    {
        "candidate_id": "387c166bf6630c97b8c6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Elizabeth Meyer-Shields Field Manager Mother Lode Field Office Digitally signed by ELIZABETH'."
    },
    {
        "candidate_id": "eb83f6063d304c18af8a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the document with the authorized officer and/or IBLA. /s/ Tate Fischer September 2, 2014'."
    },
    {
        "candidate_id": "f6f5b35d9d1f76bfe59d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 09/21/2017 Comments: OBU-A-2017-0127, Rev. 0'."
    },
    {
        "candidate_id": "42a276bba3a85e91662b",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Elizabeth Meyer Shields Field Manager, Mother Lode Date **E. Contact** For more information'."
    },
    {
        "candidate_id": "51bdca34cb3733489001",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Bruce W. Lani DATE: 12 /09 / 2010 month day year'."
    },
    {
        "candidate_id": "c238149b5251f3e9f680",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'analysis is required. D: Authorizing Official Name: Umanda m. Dodson Title: Field Manager Date:'."
    },
    {
        "candidate_id": "986814af70581a5b7bf5",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2019 month day year NEPA Compliance Officer: JESSE GARCIA Digitally signed by JESSE GARCIA Date'."
    },
    {
        "candidate_id": "fbf2fa48880caf17747c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "4178c0dad0dcfbcb792f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Official: Cody by Carlsbad Field Office Manager Date 06/14/2021'."
    },
    {
        "candidate_id": "6e96e4bd0d26f383db1e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Craig Drake Applegate Field Office Manager Digitally signed by CRAIG DRAKE Date: 2021.03.12'."
    },
    {
        "candidate_id": "2200f9edebaea0c4e7fb",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: William J. Buchko Date: 7 / 23 / 2014 month day'."
    },
    {
        "candidate_id": "6dcae8c1f9c7bf02a43c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2012 month day year NEPA Compliance Officer: john ganz Digitally signed by john ganz DN: cn-john'."
    },
    {
        "candidate_id": "c9ad46caf931fb72d6cc",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "2c84cd464276f6010859",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Stephen A. Danker Date Determined: 07/16/2015 Comments: CBU-G-2015-0032, Rev. 0'."
    },
    {
        "candidate_id": "ea42b80cf878194be19c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for Keith E. Berger, Field Manager DATE SIGNED: 8/8/18 COC-75417 Commnet COW - 31-Mile'."
    },
    {
        "candidate_id": "88a225f22a6e3c410b33",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "ddc07c18dd4e60ee0339",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: NEPA ORO MEPA Compliance Officer Date Determined'."
    },
    {
        "candidate_id": "07babb37f8dba7f9ae0c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer James L. Elmore Date Determined: 10/5/2010 Comments: Record ID: 550 Webmaster:'."
    },
    {
        "candidate_id": "60625c753eb4e29cfdb8",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Date: 2022.10.03 09:59:22 -04'00''."
    },
    {
        "candidate_id": "dbbb8a9caeacbb86bd3a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 6/1/2011 Comments: Webmaster: Record ID: 215'."
    },
    {
        "candidate_id": "d5da9839d100c283505f",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: Digitally signed by'."
    },
    {
        "candidate_id": "e4b6335eb5bd734d465a",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Jr.] Gene Iley, Jr. NEPA Compliance Officer Rocky Mountain Customer Service Region Western Area'."
    },
    {
        "candidate_id": "e4083679443aa04b0d52",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Linda Coates-Markle Field Manager 4/28/16 Date Attached: Categorical Exclusion Documentation'."
    },
    {
        "candidate_id": "8768cb78390275b9733e",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '19 14:59:24 -04'00' Date Determined: 05/05/2021 Comments: OBU-H-2021-0116, Rev. 0'."
    },
    {
        "candidate_id": "267613998c55e5bf019e",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "56f7487b9f707eb3c64f",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: G. Billie Newland Date Determined: Dec 2, 2009 Comments: Webmaster: Billie Newland THINK BEFORE YOU'."
    },
    {
        "candidate_id": "690186d87749881817ef",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Management Plan and Record of Decision (GRRMP/ROD), as amended. Date Approved/Amended: ['."
    },
    {
        "candidate_id": "1033b303bbf1caf54b69",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Andrew R. Grainger Date Determined: 01/14/2015 Comments: DOE-A-2014-0003, Rev. 1'."
    },
    {
        "candidate_id": "aacde663185cb596fa7b",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '18 14:12:45 -05:00 **Date Determined**: 01/18/2023 **Comments**: EEC No: OBU-G-2023-00001 Rev No: 0'."
    },
    {
        "candidate_id": "55dd51f4635cdb7ca974",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2024 month day year NEPA Compliance Officer: Pierina Fayish Digitally signed by Pierina Fayish'."
    },
    {
        "candidate_id": "3e9d38690d449850233c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2022 month day year NEPA Compliance Officer: PIERINA FAYISH Digitally signed by PIERINA FAYISH'."
    },
    {
        "candidate_id": "98b005ce3f1b3f4468c6",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'submitted to SHPO in early December 2003. SHPO concurrence on the report determination was received'."
    },
    {
        "candidate_id": "566d51b928af525feb81",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Gary S. Hartman Date Determined: 5/27/2010 Comments: Webmaster: Record ID: 7'."
    },
    {
        "candidate_id": "166529beec82c9657401",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'editing upon signature) Date Determined: 06/18/2018'."
    },
    {
        "candidate_id": "e6ef220a5b25f9f64f14",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '02 09:27:10 -05'00' Date Determined: 11/22/2021'."
    },
    {
        "candidate_id": "67b0d1b5066cfdba6dc0",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Richard White Date Field Manager'."
    },
    {
        "candidate_id": "0b0a029aee327ecfc6a8",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Date: March 18, 2015 Attachment: Environmental'."
    },
    {
        "candidate_id": "f8bf79f26259a21785d5",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Signature Brian Little NEPA Compliance Officer Rocky Mountain Customer Service Region Western Area'."
    },
    {
        "candidate_id": "f27e735bd8b5d4fce153",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '17 14:11:29 -07'00' Field Manager, Butte Field Office 11/17/22 Date For additional information'."
    },
    {
        "candidate_id": "5fc5f6782ca9c127ba21",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'attached Form 1842-1. APPROVING OFFICIAL: JAMES BRYAN Digitally signed by JAMES BRYAN Date'."
    },
    {
        "candidate_id": "33282a4366cb0dd99cc9",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'MEASURES/OTHER REMARKS: APPROVING OFFICIAL: ARON KING Digitally signed by ARON KING Date: 2020'."
    },
    {
        "candidate_id": "d2392c597cef3addc45f",
        "label": "neither",
        "notes": "Neither: prior authorization/history date, quote 'Renewal of WYW109293 an existing 3 diameter surface natural gas pipeline serving the Rim Rock 2'."
    },
    {
        "candidate_id": "45b9cb1bdd0bb0a4acba",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 13:30:53 -04'00' Date Determined: 04/27/2021 Comments: NNSA-H-2021-0011, Rev. 0'."
    },
    {
        "candidate_id": "1a3699de5843e1c7b4ba",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '04 16:25:32 -05'00' Date Determined: 01/07/2021 Comments: OBU-H-2020-0276, Rev. 0'."
    },
    {
        "candidate_id": "dfbc401c969d8d0890ae",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '2014-0037 Rev No: 3 Date Determined: 10/12/2022'."
    },
    {
        "candidate_id": "863b142dc6312eb15f45",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '12 01 09 41 23 05 00 Date Determined 12/01/2022 Comments EEC No TC-A-2022-01017 Rev No'."
    },
    {
        "candidate_id": "89ba5204b62a4b92a082",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Washington to Crane NSWC ROD issued by FHWA: January 2010 Construction complete'."
    },
    {
        "candidate_id": "0f35b0a189830f3c8c64",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Comments: TC-A-2019-0067, Rev. 3 Digitally'."
    },
    {
        "candidate_id": "3ba45486e825c8e07373",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Chloe L. Hutchison Digitally signed by Chloe L. Hutchison'."
    },
    {
        "candidate_id": "5b3340fb44cad9e505cc",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '30 11:35:09 -05'00' Date Determined: 12/04/2020 Comments: EP-B-2020-0019, Rev. 0'."
    },
    {
        "candidate_id": "0ca40fb577a5229b65d7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Jessica Wade Date Field Manager Attachment 1. Appeal Form 1842-1 4'."
    },
    {
        "candidate_id": "604a2b633d94db81eb41",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'o=DOE-SR, ou=EQMD, Date Determined: May 19, 2011 TC-A-2011-0054, Rev.0 Andreagrainger Date: 2011.05'."
    },
    {
        "candidate_id": "6583cc003e6eb03d7235",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Gant) Massey Assistant District Manager, Resources NEPA Compliance 08/17/2023 Brian Kennedy'."
    },
    {
        "candidate_id": "5ffa34beda7d8edd337f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date) (Signature of Authorized Officer) astong Area Manager (Title) 10/28/96 (Effective Date'."
    },
    {
        "candidate_id": "5d1f03f8cdc7f4ba26cf",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "c133b713f9a16c97726e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'David Mankiewicz Acting Field Manager, Farmington Field Office 8/6/2021 Date Attachments'."
    },
    {
        "candidate_id": "127c3c31985bd7fc7f31",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Tracy L. Williams Date Determined: 03/21/2016 Comments: TC-W-2016-0020, Rev. 0'."
    },
    {
        "candidate_id": "1cebd01b92501b71c65f",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'project. \u2022 2009: Sucker rod fencing was installed around portions of five stock'."
    },
    {
        "candidate_id": "04b7bee2796b9d6973ac",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "2f6c5ed4fbdeebd56478",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '] Kimber Liebhauser Field Manager, Lake Havasu Field Office Exhibits: A. Site Map B'."
    },
    {
        "candidate_id": "12f021d4542d062be50d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. **NEPA Compliance Officer:** Tracy L. Williams Digitally signed by Tracy L.'."
    },
    {
        "candidate_id": "f7fa1152becf177387b4",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: TC-A-2010-00228, Rev'."
    },
    {
        "candidate_id": "cde26204f78ec62b774c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'L. Wallace Assistant District Manager'."
    },
    {
        "candidate_id": "9606d92f0cdb3c164487",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '04 15:08:20 -05'00' Date Determined: 01/04/2021 Comments: OBU-K-2019-0210, Rev. 4'."
    },
    {
        "candidate_id": "b7697a99b2566d21fcf2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the document with the Authorized Officer and/or IBLA. Authorizing Official KEITH RIGTRUP Digitally'."
    },
    {
        "candidate_id": "d6952bde81df594b60bc",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'BLM California Desert District Manager on September 15, 2003 and approved by the County'."
    },
    {
        "candidate_id": "27690d33fdbc3ecafa1b",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'urther NEPA review. DOE INITIATOR SIGNATURE: Gordon R. Holcomb Digitally signed by Gordon R. Holcomb'."
    },
    {
        "candidate_id": "5224e697faefda4e0197",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Forest Service signed its Record of Decision on March 20, 2006. The Proposed Action and alternatives'."
    },
    {
        "candidate_id": "17ac21cc9294d1cbbd8d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '28 13:04:15 -05'00' Date Determined: 01/06/2021 Comments: CBU-H-2020-0043, Rev. 0'."
    },
    {
        "candidate_id": "ac7d3aa9da128b5529c3",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'o DOE-SR ou EQMD Date Determined Mar 23 2011 Andrew R Grainger Andreagrainger Date 2011 05'."
    },
    {
        "candidate_id": "c3b91e0ba00cc168109b",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2012 month day year NEPA Compliance Officer's Comment: CX form updated to revise period of performance'."
    },
    {
        "candidate_id": "38250bf0038d4e9f7121",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Authorized Official El Centro Field Office Manager Approval MICHAEL CHATTERTON 2021.02.10 10:03:18'."
    },
    {
        "candidate_id": "27715bddc1837eadf7b6",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Keith A. Dodrill Digitally signed by Keith A. Dodrill'."
    },
    {
        "candidate_id": "555e4e7d8569532ddcbd",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Attachment(s): Environmental Checklist Date: May'."
    },
    {
        "candidate_id": "8fa84820ca7ee2bd48f8",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '09:56:11-06'00' Date: Authorizing Official: Contact Person For additional information concerning'."
    },
    {
        "candidate_id": "8ba49b7dd1ba5c18d58b",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Development Project EIS ROD was approved on September 22, 2016. The 4 CD-C'."
    },
    {
        "candidate_id": "ceb72470945396b36da5",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Specialist s/ A Sparks 4/3/2023 Gloria Bulletts Benson Tribal Liaison s/ G Bulletts Benson 4/28/2023 Justin'."
    },
    {
        "candidate_id": "1ad72d2e35f9b3095feb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'applies. D: Signature Authorizing Official: [Signature] Date: 11.27.18 Contact Person For'."
    },
    {
        "candidate_id": "a512259a3bf09b324047",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "5d0304a43ba8d93f1af2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 09:51:52 -04'00' Date Determined: 08/22/2022 Comments: OBU-H-2022-0190, Rev. 0'."
    },
    {
        "candidate_id": "8d957c6b8c0b60aad5f6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Reviewed By: Assistant Field Manager Minerals & Lands Date Authorizing Official: /s/Doug'."
    },
    {
        "candidate_id": "250f49913c52c46c98aa",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '12 15 13 23 07 05 00 Date Determined 12/15/2023 Comments EEC No TC-A-2021-0108 Rev No'."
    },
    {
        "candidate_id": "b463c8dfbbb2a836e77c",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "794f2f6b5a2636f6ffa8",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'R. Halt John R. Holt NEPA Compliance Officer 3/11/10 Date Wescom GIS Desert Southwest Region'."
    },
    {
        "candidate_id": "1e8db3d6475839774e60",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Berger Keith E. Berger, Field Manager DATE SIGNED: 3/1/17'."
    },
    {
        "candidate_id": "5c3496322f701a6ec2a4",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Timothy D. Gilloon Field Manager 8/24/21 Date ATTACHMENTS Exhibit A - Map Exhibit B'."
    },
    {
        "candidate_id": "c8497582e21e57c6e623",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '2010 Buddy Green Date Field Office Manager, Owyhee Field Office'."
    },
    {
        "candidate_id": "ad106b695a5ee35d9fe5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Appeal is filed with the Authorized Officer. Signature of Authorized Official District Manager'."
    },
    {
        "candidate_id": "2d79b7bcfd153e6df8f6",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '= EQMD, OU = DOE-SR Date Determined: 9/09/09 Date: 2009.11.09 09:41:40-05'00' ARRA - M - 2009'."
    },
    {
        "candidate_id": "03acb848b3786acc832f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'favors granting the stay. Authorized Officer: /s/Brent Ralston, 9/4/2019 Brent Ralston Field'."
    },
    {
        "candidate_id": "c6fbd95a359e3bcb1f76",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Washington was signed on August 3 2020 No other comments were received from The Cowlitz Indian'."
    },
    {
        "candidate_id": "080d908a881ef3c5764f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: K. David Lyons Date: 09 / 10 / 2013 month day'."
    },
    {
        "candidate_id": "55111d6c89329f9ac0b2",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '06 15:21:53 -04'00' Date Determined: 06/22/2022'."
    },
    {
        "candidate_id": "8de96354886a45cc8ac4",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Marnie Graham Glennallen Field Manager Date: 2021.02.05 11:00:29 -09'00' Date DOI-BLM'."
    },
    {
        "candidate_id": "19ac82337ce6b4c46fc9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "c07c202071c0f1be6b8c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' Date: (Signature) Authorizing Official: Date: JUNE 7, 2023 (Signature) Name: Perry B. Wickham'."
    },
    {
        "candidate_id": "e1aac29c553de85f8cec",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote '2013, and signed a record of decision (ROD) on June 26, 2013. A Notice of Availability'."
    },
    {
        "candidate_id": "f1853225afe0c662ea78",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Vicki Duvall Digitally signed by Vicki Duvall DATE'."
    },
    {
        "candidate_id": "a0b8bedf6268e420d4fc",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'editing upon signature) Date Determined: 03/06/2020'."
    },
    {
        "candidate_id": "496cffeaee0ff2d36f97",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: Digitally signed by'."
    },
    {
        "candidate_id": "5206019ae84942bb503a",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "2709c4007721675b91eb",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'archeological site The SHPO concurred with the APE in a letter dated March 16 2015 The'."
    },
    {
        "candidate_id": "c8e8036de48e9619d9e5",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Jose R. Benitez NEPA Compliance Officer: john ganz'."
    },
    {
        "candidate_id": "04c148c6dbfd457a15f5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '00' William J. Mills, Field Manager DOI-BLM-CO-N050-2024-0013-CX 10 Appendix A. Figures'."
    },
    {
        "candidate_id": "38c66b891e9c75e332da",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2021 month day year NEPA Compliance Officer: PIERINA FAYISH Digitally signed by PIERINA FAYISH'."
    },
    {
        "candidate_id": "ca41b338850d805218ab",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'determined by the BLM Authorized Officer. 31. All equipment in the facility should be clearly'."
    },
    {
        "candidate_id": "ac9cff31a9a8a727fcfe",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote '1994 the FAA issued a Record of Decision (ROD) on the EIS. Land acquisition and construction'."
    },
    {
        "candidate_id": "f803fdedc92081edf9ee",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review 2chy NEPA Compliance Officer Joyce E Chavez 2017 07 31 15 00 28-06'00''."
    },
    {
        "candidate_id": "6ccf365c32098c3ad49d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "e05d02002037292c2657",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '07.22 17:49:29-04'00' Date Determined: 07/21/2014 Comments: EC-G-2014-0002, Rev. 0'."
    },
    {
        "candidate_id": "81a10458230406bc98b4",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: [Signature] Date Determined: 12/7/2015 LLNL NEPA Categorical Exclusion Form: Revision 1, February 4, 2010'."
    },
    {
        "candidate_id": "b61cde7621a65fe2b74d",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '12 09:10:31 -07'00' Authorized Officer: Date: Gabriel Garcia Field Manager, Bakersfield Field'."
    },
    {
        "candidate_id": "52308db02bbe25e64fc0",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'the issuance of the ROD, were also considered. 1.2.3 Tier 1 ROD FHWA issued'."
    },
    {
        "candidate_id": "e7b7c88030af4ac4ad35",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 5/2/2011 Comments: Webmaster: Record ID: 1000'."
    },
    {
        "candidate_id": "ed467744279264285c9e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '15 11:27:27 -07'00' Approving Official: William Mack Title: District Manager Field Manager'."
    },
    {
        "candidate_id": "ea28ab3c47ab6bc14fcd",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'for Loren C. Wickstrom Field Manager North Dakota Field Office 2'."
    },
    {
        "candidate_id": "d4d36527d56d37fd7b3e",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'on April 20, 2007. A ROD was signed by District Ranger Sibbernsen on September'."
    },
    {
        "candidate_id": "ed7f98d50b92edd504a9",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '15 08:28:05 -07'00' Authorized Officer: Jacob Palma, Field Manager F. CONTACT PERSON For'."
    },
    {
        "candidate_id": "1f64e2f110417c4ab6a9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "8cfc36bffa8e78a6dfea",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'needed. DAVID MUSHOVIC Authorized Officer Digitally signed by DAVID MUSHOVIC Date: 2022.09.26'."
    },
    {
        "candidate_id": "67ba72cee945139cac70",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2013 month day year NEPA Compliance Officer: Fred E. Pozzuto Date: 07/30/2013 month day year'."
    },
    {
        "candidate_id": "08f2ed9fe6300c514f06",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2021 month day year NEPA Compliance Officer: JESSE GARCIA Digitally signed by JESSE GARCIA Date'."
    },
    {
        "candidate_id": "6da33609eacb51913a71",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'the document with the authorized Officer. JENNIFER MATA Jennifer Mata, Field Manager Digitally'."
    },
    {
        "candidate_id": "c259d6d894567fc2988d",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Christopher Johnson NEPA Compliance Officer: john'."
    },
    {
        "candidate_id": "5107f9473246dab6e469",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '01 25 17 54 09-05'00' Date Determined Jan 12 2011 Submit via Email Submit to Website'."
    },
    {
        "candidate_id": "73b8e320e63480c26aac",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Jane Summerson Date Determined: Jan 4, 2010 Comments: This determination is limited to the performance'."
    },
    {
        "candidate_id": "87063e99d680dee53d0a",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '04.10 15:35:08 -04:00 Date Determined: 04/05/2019 Comments: FSSBU-L-2019-0004, Rev. 0'."
    },
    {
        "candidate_id": "4ef3e1f598295b780a7f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: BRIGGS WHITE Date: 08 / 12 / 2015 month day year'."
    },
    {
        "candidate_id": "63f5d123647db2b5f68d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer Signature: Danny Johnson January 3, 2019 Date'."
    },
    {
        "candidate_id": "e0f00824bae38930d563",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Date: July 12, 2017 Attachment: Environmental'."
    },
    {
        "candidate_id": "c679db852f559553aca8",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Stipulations Signature: Authorizing Official: Digitally signed by TY ALLEN Date: 2022.01.06]'."
    },
    {
        "candidate_id": "502fca9d63b409991b0f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'renewal. Ryan Chatterton Field Manager Digitally signed by MICHAEL CHATTERTON Date: 2020'."
    },
    {
        "candidate_id": "b59b11adfba4eaa84ba4",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2014 month day year NEPA Compliance Officer: John Ganz Digitally signed by John Ganz DN: cn-John'."
    },
    {
        "candidate_id": "141a3eec13f435aa8df6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'stay should be granted. Field Manager USDI, Bureau of Land Management Rawlins Field Office'."
    },
    {
        "candidate_id": "b35d97b63615b2953c3a",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'P060-2014-0135-EIS, Record of Decision was approved December 23, 2020. The proposed activity'."
    },
    {
        "candidate_id": "305d45b4f2a2e9a470b3",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '02 11:50:58 -05'00' Date Determined: 11/30/2021 Comments: TC-W-2021-0094, Rev. 0'."
    },
    {
        "candidate_id": "48c41b7f9ff15ad00ac5",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '05 16:39:28 -05'00' Date Determined: 10/23/2020 OBU-H-2020-0236, Rev. 0 Comments:'."
    },
    {
        "candidate_id": "4af9aa280c7c56c5503c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '03 15 12 48 28 04 00 Date Determined 03/15/2023 Comments EEC No OBU-H-2017-0146 Rev No'."
    },
    {
        "candidate_id": "e670ce047b19c9eec01b",
        "label": "neither",
        "notes": "Neither: EA/EIS document or availability date, quote '2004 and Vernal Utah October 21 2004 All written and oral comments received during the comment'."
    },
    {
        "candidate_id": "b79803170ae0776ef967",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Date /s/Jon D. Sherve Authorizing Official Signature of Authorizing Official 4-8-2019 Date Contact'."
    },
    {
        "candidate_id": "f227a1cf10bec3c97a06",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE INITIATOR SIGNATURE: Donald Krastman NEPA Compliance Officer: john ganz'."
    },
    {
        "candidate_id": "c9204af5c8e4670278b1",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Babcock Lindsey Babcock Field Manager 6/15/2016 Date F. Contact Person and Reviewers'."
    },
    {
        "candidate_id": "fe87e667e15324732994",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Alicia R. Dalton-Tingler Date: 05/15/2013 month'."
    },
    {
        "candidate_id": "8c3f4bbd6ae6d6a83696",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: MARY DAILEY Date: 02 / 05 / 2024 month day year NEPA'."
    },
    {
        "candidate_id": "45ac5a43b03547f99e26",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Grainger Comments: Date Determined: Dec 7, 2010 OBU-K-2010-200, Rev.0 Submit via Email Submit to Website'."
    },
    {
        "candidate_id": "603fdcbfaa9e90c63e7f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '11 13:42:40-08'00' Field Manager Date E. Contact For more information, contact Chad'."
    },
    {
        "candidate_id": "0ef71cb4c305b429f438",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "5e5bac404da00162ebf4",
        "label": "decision",
        "notes": "Decision: decision/FONSI/ROD signed or issued, quote 'Management Plan (GRRMP) and Record of Decision (ROD), as amended. Date Approved: August 8, 1997'."
    },
    {
        "candidate_id": "2f961499af4bded22c6a",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy L. Williams Digitally signed by Tracy L. Williams'."
    },
    {
        "candidate_id": "c00537eee908040405a0",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'from further review. NEPA Compliance Officer: Andrew R. Grainger Comments: TC - A - 2010 - 001'."
    },
    {
        "candidate_id": "208038fc384a4b95b610",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'activities.\" D: Signature Authorizing Official: Douglas J. Herrema Field Manager Date: 12/20/2018'."
    },
    {
        "candidate_id": "fecbb38cafc5e4fe1f2d",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Andrew R. Grainger Digitally signed by Andrew R.'."
    },
    {
        "candidate_id": "f4decb250d22c9c8b068",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '12.04 16:27:07 -05:00 Date Determined: 11/21/2019 Comments: OBU-F-2019-0420, Rev. 0'."
    },
    {
        "candidate_id": "321a4c59c5ce2c1d4945",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2011 month day year NEPA Compliance Officer: Jesse Garcia Digitally signed by Jesse Garcia DN'."
    },
    {
        "candidate_id": "b799331ebd55ead1bf96",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Mason Stacy L. Mason NEPA Compliance Officer Attachment(s): Environmental Checklist Date: May'."
    },
    {
        "candidate_id": "3919687fbf7000cffb59",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Attachment(s): Environmental Checklist Date: February'."
    },
    {
        "candidate_id": "167819c04c114db3ed55",
        "label": "neither",
        "notes": "Neither: meeting or site-visit date, quote '2 Virtual Public Scoping Meeting held on September 29 2020 from 4 30 to 6 00'."
    },
    {
        "candidate_id": "1d64a95b0c6c8fc5c007",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: Othalene Lawrence Date Determined: Apr 20, 2010 Comments: Webmaster: Billie Newland'."
    },
    {
        "candidate_id": "58a7082f2cd5129ff9b5",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'Commission s June 27 2024 authorization of the Project will apply if approved and are therefore'."
    },
    {
        "candidate_id": "ae8d868e46aa723d90c8",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: GEOFFREY GOODE Digitally signed by GEOFFREY GOODE'."
    },
    {
        "candidate_id": "3b3e0fc6b305ecd1cb25",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Tracy Williams Digitally signed by Tracy Williams'."
    },
    {
        "candidate_id": "717454d4163600bfc584",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote '09 19 14 31 51 04 00 Date Determined 09/19/2023 Comments EEC No OBU-G-2022-01005 Rev No'."
    },
    {
        "candidate_id": "f6feefc7dff404c3e03d",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer: [Signature] Date Determined: 12/31/2012'."
    },
    {
        "candidate_id": "64bc26142bc02524c4ed",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 9/3/2010 Comments: Webmaster: Record ID: 556'."
    },
    {
        "candidate_id": "33f8ef1838a55f1d2792",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'further NEPA review. NEPA Compliance Officer: Gregory L. Williamson Date: 2022.01.09 17:08'."
    },
    {
        "candidate_id": "f3e1588059024d19698c",
        "label": "decision",
        "notes": "Decision: operative Date Determined, quote 'Officer Gary S. Hartman Date Determined: 9/15/2011 Comments: Webmaster: Record ID: 222'."
    },
    {
        "candidate_id": "358f0a0b3ec9c1868499",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'by A. Rose Assistant Field Manager Date: 06/08/18 E. Approval I have reviewed the Proposed'."
    },
    {
        "candidate_id": "753836da8c3780dcfc62",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Lana R Dawson, Acting Field Manager, Uncompahgre Field Office DATE SIGNED: 3/11/16]'."
    },
    {
        "candidate_id": "9c894d34048ffa29d6eb",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '06'00' SIGNATURE OF AUTHORIZED OFFICER SUZANNE COPPING Suzanne Copping Uncompahgre Field'."
    },
    {
        "candidate_id": "1235216e26ed2d65f86f",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: JOSEPH HANNA Date: 06 / 09 / 2022 month day year'."
    },
    {
        "candidate_id": "fa581ff4d99a69716cb9",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2021 month day year NEPA Compliance Officer: JESSE GARCIA Digitally signed by JESSE GARCIA Date'."
    },
    {
        "candidate_id": "0ce918f3d64e31ab09c6",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'further NEPA review. DOE Initiator Signature: Andrea M. Dunn Digtally signed by Andrea M. Du ent'."
    },
    {
        "candidate_id": "ff786333c5e00881d12e",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: STEVEN MARKOVICH NEPA Compliance Officer: Pierina'."
    },
    {
        "candidate_id": "69e7640ab67ec412b1a4",
        "label": "initiation",
        "notes": "Initiation: DOE Initiator signature, quote 'NEPA DETERMINATION. DOE Initiator Signature: Carl P. Laird Digitally signed by Carl P. Laird Date'."
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
