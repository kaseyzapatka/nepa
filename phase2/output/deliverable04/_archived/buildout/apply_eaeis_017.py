import pandas as pd


LABELS = [
    {
        "candidate_id": "508393d9f9bd269b82d8",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote '6, 2009 and authorization issued February 22, 2008, December 17, 2008, and March 10, 2009. \u2022 USACE, Vicksburg'."
    },
    {
        "candidate_id": "de550528251bd14a423f",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'RICHARD WHITE Digitally signed by RICHARD WHITE Date: 2023.02.22 15:34:28 -08'00''."
    },
    {
        "candidate_id": "e71cf2640c9f04e489eb",
        "label": "decision",
        "notes": "Decision: permit or ROW authorization, quote 'ANDREW ARCHULETA Digitally signed by ANDREW ARCHULETA Date: 2020.07.31 12:55:30 -07'00' 7/31/20'."
    },
    {
        "candidate_id": "49be7ae3fb2736e47610",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'TIMOTHY HAMMOND Digitally signed by TIMOTHY HAMMOND Date: 2021.04.20 08:08:06 -08'00''."
    },
    {
        "candidate_id": "fd5709ba8944aa878992",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'the trial widening work will be completed by October 30, 2016 so that crews can repair the thrust'."
    },
    {
        "candidate_id": "6818170558fad78999f8",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'ANTHONY BOBO Digitally signed by ANTHONY BOBO Date: 2023.09.25 11:18:24 -04'00''."
    },
    {
        "candidate_id": "cf9f4a3c3409c923fcb0",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'issued a Finding of No Significant Impact on March 26, 2012. Decision and Rationale After careful consideration of'."
    },
    {
        "candidate_id": "ec7c10a8ab09f4d9a7d9",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'issued under the WMRNP which is scheduled for October 2019. 1.3 PURPOSE AND NEED The need for this'."
    },
    {
        "candidate_id": "aed87b311224386e0e5b",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '2007a) that was released to the public on June 29, 2007. The Record of Decision was signed September'."
    },
    {
        "candidate_id": "12c7dff312406918e55d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'sites. A Record of Decision was issued in April 1997 in which DOE selected the Proposed Action Alternative'."
    },
    {
        "candidate_id": "5652f165cc48ba6cadba",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'are attached to the authorization. /s/ Adam Carr_________________ 10/26/2018 Adam Carr Date Field Manager, Eastern Interior Field'."
    },
    {
        "candidate_id": "7cc30ef63da251f4376e",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'an EA, and a FONSI was signed on June 1, 2001 (DOE 2001). Oak Ridge Science and Technology'."
    },
    {
        "candidate_id": "3af063b7b86b81246287",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'STEVEN NELSON Digitally signed by STEVEN NELSON Date: 2021.01.15 10:53:47 -08'00''."
    },
    {
        "candidate_id": "1ff19358b6886cf56122",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'public review and comment period (November 6 \u2013 December 6, 2019). The BLM also mailed or emailed a'."
    },
    {
        "candidate_id": "d62a3d2d9cd2d9b73eba",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'not issued, an EIS may be developed. 2 April 13, 1995'."
    },
    {
        "candidate_id": "db6be93ba8cdbf2cddb8",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'RICHARD WHITE Digitally signed by RICHARD WHITE Date: 2024.02.29 16:01:03 -08'00''."
    },
    {
        "candidate_id": "7a668cc24eebe4fa295c",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Field Office Digitally signed by BRENT RALSTON Date: 2022.06.10 07:32:05 -06'00''."
    },
    {
        "candidate_id": "701ceddbc7c30a5e8c0d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Activity Plan/Environmental Impact Statement Record of Decision signed February 2013. This EA is tiered to this document and'."
    },
    {
        "candidate_id": "7900dddbf354ecaf3872",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'August 22, 2023 and ended at 11:59pm on September 13, 2023. There was a total of 132 public'."
    },
    {
        "candidate_id": "bee3a16d32fb67eac57b",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Of No Significant Impact (FONSI) was completed on December 17, 2020, in which I determined that Alternative C'."
    },
    {
        "candidate_id": "544ecac31f2d79671dd8",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'March 26, 2019. The comment period ended on April 25, 2019. The results of coordinating the proposal are'."
    },
    {
        "candidate_id": "762c8010521a1968897f",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Digitally signed by TIMOTHY GILLOON TIMOTHY GILLOON Date: 2021.12.20 12/20/21 19:04:30 -08'00' Timothy D. Gilloon Date Field'."
    },
    {
        "candidate_id": "af00b4c5aeb7950d7f2f",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'and socioeconomic resources. The PEIS was published in June 2005, and in December 2005 the Record of Decision'."
    },
    {
        "candidate_id": "e42a87573c8898999338",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Production Act of 1976 (NPRPA, P.L. 94-258). APPROVED: December 11, 2015 /s/Stacie McIntosh Date Arctic Field Office Manager'."
    },
    {
        "candidate_id": "f68466cc06bac5bc8f59",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'of no significant impact (FONSI) was signed on August 16, 2019, and concluded that the final decision to'."
    },
    {
        "candidate_id": "fc3c45f5722835969848",
        "label": "decision",
        "notes": "Decision: NEPA determination, quote 'Evaluation of Approved Siting Notification for NEPA Ramifications, March 12, 1999.'."
    },
    {
        "candidate_id": "b1a5efb3cfd23e8ef105",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'issued a Finding of No Significant Impact in July 1999. 2.4.3 Demolition of Vacated Buildings The demolition of'."
    },
    {
        "candidate_id": "dedf343cbaec6bf3c221",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'by Avista biologists by vehicle and foot in May 2016, February 2017, and April 2017. The presence of'."
    },
    {
        "candidate_id": "c29acc87bed54aa50c09",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'MELISSA WARREN Digitally signed by MELISSA WARREN Date: 2020.12.17 10:41:00 -07'00''."
    },
    {
        "candidate_id": "60655cc1dcdd90ce1e87",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'and socioeconomic literature; \u2022 Public response to the March 14, 2014 NOI to prepare the EA; \u2022 Public'."
    },
    {
        "candidate_id": "8de27df15068ae24b3e9",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'SHAYNE BANKS Digitally signed by SHAYNE BANKS Date: 2021.08.18 10:02:07 -05'00''."
    },
    {
        "candidate_id": "6a7eb0f0d2f229da1942",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'is not required. /s/ Michelle Ethun, acting for May 4, 2012 _________________________________________ __________________ Lenore Heppler Date Field Manager,'."
    },
    {
        "candidate_id": "a83388f55cf3d45000d2",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Activity Plan/Environmental Impact Statement Record of Decision signed February 2013 and the 2014 Supplemental Environmental Impact Statement for'."
    },
    {
        "candidate_id": "0d1318c75274a496413b",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'of the substation is anticipated to begin in July 2011. 2.4.2 Other Projects in the Project Vicinity Lower'."
    },
    {
        "candidate_id": "84b3d0169df3df3562e5",
        "label": "neither",
        "notes": "Neither: consultation date, quote '12/13/2022 Tribes, Individuals, Organizations, or Agencies Consulted On August 30, 2022, a consultation response was received from the'."
    },
    {
        "candidate_id": "9e1cd863145237d2b359",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'and Southwestern Montana (USDI-BLM 2015). /s/ Jeremy Casterson 12/06/2017_______ Jeremy Casterson Date Field Manager Upper Snake Field'."
    },
    {
        "candidate_id": "bacf642579ea34755394",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'I o n January 27, 1993, for the August 1992 Final Environmental Impact Statement and Environmental Impact Report'."
    },
    {
        "candidate_id": "9cda460b95c07852ebf4",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'issued a Finding of No Significant Impact on March 25, 2013. Decision The decision has been made to'."
    },
    {
        "candidate_id": "9fedab0ec5ad5b43dc58",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'CA 92262 Digitally signed by DOUGLAS HERREMA Date: 2019.11.01 14:30:41 -07'00''."
    },
    {
        "candidate_id": "a515717ff6c6f06c3af3",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Eastern Mojave Management Plan, Record of Decision approved December 20, 2002 (BLM, 2002). It is also in conformance'."
    },
    {
        "candidate_id": "cdb7432815fc19da2e28",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote '23 (DOE/EA-1592-S1) and supported by a FONSI, issued August 28, 2019. Those documents should be consulted for additional'."
    },
    {
        "candidate_id": "9b055e527e2a4a2a272d",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'A. Pacioretty, Field Manager Pocatello Field Office Date: 01/22/2014'."
    },
    {
        "candidate_id": "8a5f58615992f5a2ee69",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Lorraine Bodi Vice President Environment, Fish and Wildlife 12/29/2016 Date'."
    },
    {
        "candidate_id": "6723d3ae93ce1159cb3b",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'required. Signature Acting Eastern Interior Field ManagerDate signed May 30, 2012 Date'."
    },
    {
        "candidate_id": "87c42a2b813361f2e28b",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'from Authorized Official /s/ Timothy J. La Marr August 8, 2017 Timothy J. La Marr Date Field Manager,'."
    },
    {
        "candidate_id": "31f2e7b309cda466a8fc",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'consultation response was received from the USFWS on February 2, 2021 concerning the presence of the northern long-eared'."
    },
    {
        "candidate_id": "c542c83ceb6bfb4bb21b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Spring Chinook Salmon Supplementation Program EA (DOE/EA-1173 completed April 3, 1998 with a Finding of No Significant Impact'."
    },
    {
        "candidate_id": "24d951b1b11ce36e1caa",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'for the draft WMP occurred February 15 to March 17, 2017. In February and early March, three scoping'."
    },
    {
        "candidate_id": "1ef5e52a98e0089688fd",
        "label": "decision",
        "notes": "Decision: permit or ROW authorization, quote 'and local laws (see Chapter 4) Signature /s/ 6/12/17 _______________________________ _________________________ Kurt Pavlat Date Field Manager Coeur'."
    },
    {
        "candidate_id": "d8cfbb22e035b5200a13",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'KEVIN COODEY Digitally signed by KEVIN COODEY Date: 2020.09.10 11:03:33 -07'00''."
    },
    {
        "candidate_id": "725616a01c98ecbf3340",
        "label": "decision",
        "notes": "Decision: permit or ROW authorization, quote 'Officer) ________________________________ ___Field Manager__________________ (Print Name) (Title) ________________________________ ____3/19/2013_____________________ (Title) (Effective Date of Grant) ________________________________ (Date)'."
    },
    {
        "candidate_id": "5808bcfe989016de535f",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'properties at the Y-12 Complex was signed on August 25, 2003. A site-wide programmatic agreement among the DOE'."
    },
    {
        "candidate_id": "3131b43362f23c579acc",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'made available for a 30-day public comment period, October 11, 2022 to November 11, 2022. Several public comments'."
    },
    {
        "candidate_id": "3e672ecb6c2199b6cb76",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'FONSI U.S. Fish and Wildlife Service (USFWS) 8/10/2022 8/12/2022 12/15/2022 Notice of Intent to Prepare an EA'."
    },
    {
        "candidate_id": "902d9f9efdafc9cf4957",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Kennedy Assistant District Manager, Minerals NEPA Compliance BK 12/13/2023 Tribes, Individuals, Organizations, or Agencies Consulted On November'."
    },
    {
        "candidate_id": "b980620e0d934b5d42c9",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Notice of Final Decision (NOFD) was signed on December 9, 2019, which included a section that addressed the'."
    },
    {
        "candidate_id": "8721f6ce44531bfc376a",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'DOE issued a FONSI for these activities in March 2007. Since that time, DOE has researched historical documents'."
    },
    {
        "candidate_id": "fa4c96370adddc3853d9",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Activity Plan/Environmental Impact Statement Record of Decision signed February 2013 and the 2014 Supplemental Environmental Impact Statement for'."
    },
    {
        "candidate_id": "58b2b6b47448873ba528",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'an EA, and a FONSI was signed on June 1, 2001 (DOE 2001b). Oak Ridge Science and Technology'."
    },
    {
        "candidate_id": "ec87f3a4ab0cdcfebb9b",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'draft FONSI U.S. Fish and Wildlife Service (USFWS) 8/10/2022 8/12/2022 12/15/2022 Notice of Intent to Prepare an'."
    },
    {
        "candidate_id": "da26341510618a029b3c",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Act of 1976 (NPRPA, P.L. 94-258). APPROVED: /s/ February 26, 2016 Stacie McIntosh Date Arctic Field Office Manager'."
    },
    {
        "candidate_id": "82c803a25ef9d12da662",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'RICHARD WHITE Digitally signed by RICHARD WHITE Date: 2023.02.22 15:34:28 -08'00''."
    },
    {
        "candidate_id": "1aedd10da514b5e1d7e1",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Field Office Digitally signed by BRENT RALSTON Date: 2023.12.15 10:21:57 -07'00''."
    },
    {
        "candidate_id": "c9da9c1c092ed2dda11d",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Impact Statement (Final EIS) for this Project on April 26, 2013 and a Record of Decision (ROD) on'."
    },
    {
        "candidate_id": "119478ea9428f2596f67",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'anticipated, as stated in the FONSI signed on September 20, 2017, thus an Environmental Impact Statement is not'."
    },
    {
        "candidate_id": "2cefe8e82b5276d666ac",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Aden L. Seidlitz Aden L. Seidlitz District Manager September 1, 2010 Date 4'."
    },
    {
        "candidate_id": "50d650074df7b69d0688",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'May 14, 2014 update. \uf0b7 National Seed Strategy (August 2015). \uf0b7 Oregon/Washington State Protocol Agreement between Oregon/Washington State'."
    },
    {
        "candidate_id": "52e04bf4c1cf80f515fc",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote '14, 2021 and shared with USFWS during the March 2021 meeting. National Historic Preservation Act Section 106 Consultation'."
    },
    {
        "candidate_id": "5826551d349928b300ad",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Field Office Digitally signed by CARRIE SAHAGUN Date: 2020.04.09 12:18:42 -07'00' Digitally signed by MICHAEL CHATTERTON Date:'."
    },
    {
        "candidate_id": "ee87d44484608e5e82d9",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote '2019. Field Manager signature: /s/ Bart Zwetzig Date: May 3, 2019'."
    },
    {
        "candidate_id": "e5115ac2b4bb7e64d7f4",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'this decision has been signed. /s/ William Haigh June 1, 2016 ____________________________________ __________________ William S. Haigh Date Field'."
    },
    {
        "candidate_id": "9f17f865d793f2c4944c",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'consultation response was received from the USFWS on November 17, 2020 concerning the presence of the NLEB within'."
    },
    {
        "candidate_id": "5f84444ca8dc11e53afd",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'statement is not required. /s/ Leonard A. Marceau 09/30/2013 Assistant Field Manager, Non-Renewable Resources, Kingman Date DECISION'."
    },
    {
        "candidate_id": "e8df5c88deb37c0fb4b1",
        "label": "decision",
        "notes": "Decision: ROD or Decision Record, quote 'Fire Record of Decision and Approved Management Plan, March 2008; and the Management Plan for Public Use and'."
    },
    {
        "candidate_id": "ac82c213d5b170e98370",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'with the determination and concluded informal consultation on May 2021 with a letter of concurrence. Tribal Consultation The'."
    },
    {
        "candidate_id": "8c1106565e7760028b87",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'anticipated, as stated in the FONSI signed on November 18, 2014 thus an Environmental Impact Statement is not'."
    },
    {
        "candidate_id": "59a465d7b67c2bc763c8",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'likely begin in April 2014 and continue through December 2014. Details of the Proposed Action are presented in'."
    },
    {
        "candidate_id": "b804d7ce3cb99735ba86",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'issued for a 30-day public review period, beginning 25 April 2016 and ending 25 May 2016. The draft'."
    },
    {
        "candidate_id": "af4b791682f5c4a1d5e0",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'environment (Section 1.4). /s/ Michele McDaniel Acting For October 17, 2013 _________________ ________________ Loretta V. Chandler Date Field'."
    },
    {
        "candidate_id": "86a3a50a28f946b4c324",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'NO SIGNIFICANT IMPACT for Rattlesnake Road Right-of-Way DOI-BLM-ID-C020-2023-0003-EA 2/29/24 INTRODUCTION: The Bureau of Land Management (BLM) received'."
    },
    {
        "candidate_id": "7e70c505739e31ae5105",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'public review and comment period (May 26 \u2013 June 24, 2020). The BLM also mailed or emailed a'."
    },
    {
        "candidate_id": "7ffd5d3e7283c51233c2",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'part of a Final Revised FONSI issued on August 30, 2016, for the changes to the transfer action.'."
    },
    {
        "candidate_id": "fbf799b0d676f77b61aa",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote '_/s/ A. C. King, authenticated by A. Rose_ __01/22/2020_______ Aron C. King Date Field Manager Yuma Field'."
    },
    {
        "candidate_id": "0e1435819b9d38517d10",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'Digitally signed by ERIC MORGAN ERIC MORGAN Date: 2022.08.22 08:38:35 -07'00' Eric Morgan Date (Acting) Central Coast'."
    },
    {
        "candidate_id": "e4ca5a65ab9de5e51d98",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'signed the associated Decision Record and FONSI on May 7, 2019. In that decision, BLM affirmed the nine'."
    },
    {
        "candidate_id": "c410191467b1d7e6599c",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'final SEIS is anticipated to be issued by July 2011. Although construction of the CMRR NF has not'."
    },
    {
        "candidate_id": "e24fa7a70ca6f4395e3a",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'SUZANNE COPPING Digitally signed by SUZANNE COPPING Date: 2022.12.19 15:38:16 -07'00''."
    },
    {
        "candidate_id": "c9ad583bda24f16d89ee",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'reasons. The NOFD was received by WWP on December 11, 2019. WWP filed an appeal on January 10,'."
    },
    {
        "candidate_id": "ff8f99beaa5b1247fa68",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'WAPA hosted a web-based virtual scoping meeting on January 11, 2021, in which interested parties were provided an'."
    },
    {
        "candidate_id": "48d85005d0b8df7ba4dc",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Field Office Digitally signed by LORRAINE CHRISTIAN Date: 2019.11.21 12:25:56 -07'00''."
    },
    {
        "candidate_id": "b1d81df4288d4b079027",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Rivers Field Office Schedule of Proposed Actions in December 2009 and January 2010. BLM staff met with the'."
    },
    {
        "candidate_id": "6509d9d478126f292892",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2021-09-17'."
    },
    {
        "candidate_id": "4757ff92c145ee9659ce",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-10-01'."
    },
    {
        "candidate_id": "5beda020cc7fb690de55",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Commission\u2019s Regulations, to amend the authorizations granted on July 30, 2014 in Docket Nos. CP12-509-000 and CP12- 29-000'."
    },
    {
        "candidate_id": "047885f22638bed8162a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'period was 21 days long and ended on July 19, 2000. No comments have been received on the'."
    },
    {
        "candidate_id": "dc0e366bfa9d2373fb34",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'period that opened June 19, 2017 and ended July 19, 2017. (2) Commentors and issued raised. Only one'."
    },
    {
        "candidate_id": "952f667b91f246f555b5",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'management was published in the Federal Register on July 2, 1992. A news release announcing the intent to'."
    },
    {
        "candidate_id": "df156178d4e1183000e6",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2023-12-19'."
    },
    {
        "candidate_id": "7f5d5be33b3361e00ccd",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'participate in an on-site public meeting held on August 18, 2015. The meeting gathered initial information on trail'."
    },
    {
        "candidate_id": "57bcf00947d76cd02937",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2022-03-17'."
    },
    {
        "candidate_id": "ff4572aeaad56328e783",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2017-05-05'."
    },
    {
        "candidate_id": "e992e9c1f66579395155",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'NMFS provided a request for additional information on June 7, 2012. A response was prepared and submitted on'."
    },
    {
        "candidate_id": "18759eba3e4320096982",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'application to the Arizona State Land Department in February 2009 for the portions of the gen-tie line that'."
    },
    {
        "candidate_id": "25148e016488979933c0",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Bureau of Land Management (BLM) public land, on March 13, 2013. The ROW was denied on December 18,'."
    },
    {
        "candidate_id": "53316b36c7e630ec06f5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2017-04-19'."
    },
    {
        "candidate_id": "09e149c5f59d37378fe5",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Yakima Herald-Republic. The public comment period ended on July 23, 2020 and the EA was revised based on'."
    },
    {
        "candidate_id": "afaabbd4e6ca48ca045a",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'process used for this Project was initiated on June 21, 2022, with the publication of a description of'."
    },
    {
        "candidate_id": "0dc8883e66bc2138fd8b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-10-23'."
    },
    {
        "candidate_id": "5275b854cd747efa6b7b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-07-06'."
    },
    {
        "candidate_id": "9d493ad4079ef7ef4f78",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-09-29'."
    },
    {
        "candidate_id": "17a375a58ea4035b78c5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2020-09-03'."
    },
    {
        "candidate_id": "37ac5667026958629784",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-05-11'."
    },
    {
        "candidate_id": "4f313bb21f4a873f290d",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'of no historic properties in the APE. On April 24, 2017, a follow-up e-mail was sent to the'."
    },
    {
        "candidate_id": "07743c7a4ad6f646ac8d",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2013-08-21'."
    },
    {
        "candidate_id": "f176cc583512ac81fbc9",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'https://www.wapa.gov/transmission/EnvironmentalReviewNEPA/Pages/AZ-Energy-Storage- Project.aspx. The public scoping period began on September 25, 2019, and WAPA accepted comments on the Project'."
    },
    {
        "candidate_id": "46b69cda227052cbae28",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Well R-28 Discharge Plan and Permit Application On December 20, 2011, the Laboratory submitted a discharge plan and'."
    },
    {
        "candidate_id": "7585ab4a73611ab3bfd8",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2022-02-18'."
    },
    {
        "candidate_id": "2fd85e8e3dc98e4700f5",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'Sabine Pass LNG Terminal to Free Trade Nations (Sept. 7, 2010), amended by DOE/FE Order No. 2833-A (Oct.'."
    },
    {
        "candidate_id": "f96844e67df5a269c645",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'of Issues Internal BLM scoping was initiated on November 15, 2021. A 14 day public scoping period was'."
    },
    {
        "candidate_id": "42f77ef236019bf30799",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-09-27'."
    },
    {
        "candidate_id": "f27574a57dcc22624a2b",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'noise elements affecting the properties historic features. On July 19, 2010, DOE submitted its no effect determination and'."
    },
    {
        "candidate_id": "00c9a5f99fde63638c3b",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'renewal permit was issued for public comment in December 2002 and is expected to be issued in 2003.'."
    },
    {
        "candidate_id": "90755423eef7c283389a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'participation in monitoring [during cultural resources survey]. On July 29, 2015, Western forwarded the tribe\u2019s request on to'."
    },
    {
        "candidate_id": "e47c8d4cc95027267ee6",
        "label": "neither",
        "notes": "Neither: non-NEPA activity or historical reference, quote 'Facility Issued December 16, 2016 Previous permit issued July 20, 1992, and administratively continued until the current permit'."
    },
    {
        "candidate_id": "48187785e47c4e9892aa",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2021-10-13'."
    },
    {
        "candidate_id": "d4c9821b3b6b78d5cde6",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'action was made public through ePlanning posting on 17 January 2023. There are no comments received to date.'."
    },
    {
        "candidate_id": "656b695493e8a1ac2d05",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2022-03-08'."
    },
    {
        "candidate_id": "29052b606e87db38e29a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-12-12'."
    },
    {
        "candidate_id": "64770c283af853d55c99",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'in a public meeting that was held on August 18, 2015 at the WFCA, to discuss the proposed'."
    },
    {
        "candidate_id": "1ef130bfb7ecf153d5f0",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Tribes) in letters dated September 15, 2015 and October 17, 2017. To date, none of the tribes have'."
    },
    {
        "candidate_id": "88e9427d91837d9f3b5b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2020-07-09'."
    },
    {
        "candidate_id": "2fb6022b04103a340aa5",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2022-01-04'."
    },
    {
        "candidate_id": "3ff89dbac4689ee6c4fb",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'S:\\COMM\\NEPA\\TODO\\EA1383\\EA1.0-R.DOC 4/1/02 1-2 OMGC filed an Application for Certification (AFC)'."
    },
    {
        "candidate_id": "735781f5a6e2c2048f49",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2017-10-19'."
    },
    {
        "candidate_id": "48b6b0f97313725e85c9",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'the Proposed Action on BLM\u2019s ePlanning website on August 16, 2017. The Durango Herald published an article about'."
    },
    {
        "candidate_id": "5bc6a6102ea242776aab",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'objectives. Develop a Proposed Action and evaluate alternatives October 2017 \u2013 November 2018 Open House Meetings to solicit'."
    },
    {
        "candidate_id": "7ce80f240d65765324b3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-04-04'."
    },
    {
        "candidate_id": "f1b4759b63521b29d3db",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'Title 56. 5.0 PUBLIC REVIEW AND COMMENTS On February 23, 2018, SPLNG filed a request to utilize our'."
    },
    {
        "candidate_id": "e863a966f894b04fbc84",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'on the NEPA website (\u201cePlanning\u201d); and \uf0b7 On June 6, 2012 the BLM held a public meeting with'."
    },
    {
        "candidate_id": "74e15f383d4412d93381",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'to intervene, notices of intervention, and comments by December 14, 2020.18 DOE received two timely-filed comments in response'."
    },
    {
        "candidate_id": "cd37ecc309516d73573f",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '30-day public review and comment period, which ended September 26, 2016. The BLM sent out over 240 notifications'."
    },
    {
        "candidate_id": "da389d82a0969f8420c0",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'people attended a similar meeting in Sunriver on August 12, 2010. A public meeting was also held in'."
    },
    {
        "candidate_id": "19f0c94c0c1931f5f9df",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'on this proposal from August 3, 2016 until September 12, 2016.The BLM received only two comment submissions, which'."
    },
    {
        "candidate_id": "4a0910b6a044027b331f",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-10-16'."
    },
    {
        "candidate_id": "ba0b0c54eed812523864",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-03-14'."
    },
    {
        "candidate_id": "268d26f988291acce3f6",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'for revision of that application was received on July 21, 2012, along with comments following review of the'."
    },
    {
        "candidate_id": "4502d4f8bb6599819911",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-10-31'."
    },
    {
        "candidate_id": "0d5b35573ba2fe3f49d9",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'federal levels. The official comment period ended on March 20, 2020. Responses were received from four parties: the'."
    },
    {
        "candidate_id": "c6eac049b146482f9586",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-09-18'."
    },
    {
        "candidate_id": "0801768030fe83cc3568",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'C). Letters were sent to various tribes all November 17, 2014 notifying them of the proposed action and'."
    },
    {
        "candidate_id": "a28e09f99dd29a90ac46",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'comments on the project from the public until February 21, 2020. All project documents and comments received are'."
    },
    {
        "candidate_id": "10b12e530989eb86a53b",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2022-04-01'."
    },
    {
        "candidate_id": "ec985269794ed7593914",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-03-06'."
    },
    {
        "candidate_id": "55bfeb07db3b7a174e15",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2021-11-24'."
    },
    {
        "candidate_id": "5ace1563a29851dcdbc3",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'Steigerwald Floodplain Restoration Project EA was initiated on December 28, 2015 and closed on January 27, 2016. BPA'."
    },
    {
        "candidate_id": "41a524ace316ceeab284",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-12-14'."
    },
    {
        "candidate_id": "1672969b4b144ceaa33d",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'bull trout, marbled murrelet, and golden paintbrush on April 24, 2012; FWS concurred with staff\u2019s findings on June'."
    },
    {
        "candidate_id": "1730f16f4e674350ae04",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2016-08-25'."
    },
    {
        "candidate_id": "b314f0568d8be1b4ae20",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'and comment, and inviting the Tribes to the September 20, 2022 Field Visit and the September 21, 2022'."
    },
    {
        "candidate_id": "7ffe2506c47247ac45cc",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '2019 through January 6, 2020 inclusive of the December 5 2019 public hearing) in which the USACE was'."
    },
    {
        "candidate_id": "42cf4fd70f5f7483f02a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'public comment; the comment period ran through to March 3, 2014. The EA describes the project, its potential'."
    },
    {
        "candidate_id": "2c6ffc083830b18e4eed",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-07-18'."
    },
    {
        "candidate_id": "d2c141cfb25d994c8332",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2021-01-04'."
    },
    {
        "candidate_id": "60e33eec283292ae8f36",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-10-09'."
    },
    {
        "candidate_id": "b1b7ee4401054f00ad96",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'to Fremont County for a recreational park on August 20, 1973. By 1981, approximately 10 to 15 acres'."
    },
    {
        "candidate_id": "ed21842c85a3cbe7b4d5",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'the CDCA Plan in the Federal Register on October 4, 2012. The NOI provided for a 30-day public'."
    },
    {
        "candidate_id": "4fc46b7527c1bf80725c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-04-16'."
    },
    {
        "candidate_id": "164d13fc293aa76de1ea",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '9, July 31, August 11, November 30, and December 15, 2006. Chapter 5 Consultation and Coordination Public Scoping'."
    },
    {
        "candidate_id": "3afe193c7f400e1182e9",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '30-day public review and comment period which ended September 26, 2016. The BLM made over 240 notifications pertaining'."
    },
    {
        "candidate_id": "d0386f30081ec095ff8a",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Consultants, Inc., to Louisiana Division of Historic Preservation, February 6, 2009, and SHPO Concurrence, March 6, 2009. o'."
    },
    {
        "candidate_id": "0fff4765905e75c7feb4",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'of April 22, 2015, in Leavenworth, Washington, and April 23, 2015, in Winthrop, Washington. Four people attended the'."
    },
    {
        "candidate_id": "c204fa3e0f3c513f6455",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2024-06-26'."
    },
    {
        "candidate_id": "c6dd68b2ef102a4514a8",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'the EA. The public comment period closed on August 2, 2010. DOE did not receive any comments from'."
    },
    {
        "candidate_id": "91abd4ffef0c1c45fd2b",
        "label": "neither",
        "notes": "Neither: consultation date, quote 'Third Berth layout subsequent to the March and July 2018 concurrence from Louisiana SHPO; therefore, SPLNG submitted a'."
    },
    {
        "candidate_id": "fefcc87152e1957eb661",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'to 4. Make factual period be extended from June 6th, 2011 to June 30th, 2011. June 23, 2011.'."
    },
    {
        "candidate_id": "bb0bfcbcc421e3598222",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'carry out the terms of the MOA. On February 14, 2022, the Commission issued a draft MOA for'."
    },
    {
        "candidate_id": "f6b43346e7c5059b79e8",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'this EA. The public comment period closed on July 30, 2010; DOE did not receive any comments. Appendix'."
    },
    {
        "candidate_id": "bcd681401fc55a836e4c",
        "label": "initiation",
        "notes": "Initiation: NOI published, quote 'a public meeting regarding the requested withdrawal. On January 13, 2017, the Forest Service published a notice of'."
    },
    {
        "candidate_id": "b52f7a4b97a11a9a7ca3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-05-14'."
    },
    {
        "candidate_id": "e5b86886e88967b7540c",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-10-22'."
    },
    {
        "candidate_id": "5120ed6faf718129cb3b",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'letters phone and or E mail initiated on July 2020 was never received by us. Because of this'."
    },
    {
        "candidate_id": "027ab9539d25c1875ce7",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-09-09'."
    },
    {
        "candidate_id": "85d490f0b3a7f0e9f7fc",
        "label": "decision",
        "notes": "Decision: permit or ROW authorization, quote 'export to non-FTA countries, with comment close on March 11, 2022; Order 3662-B amending authorization to export to'."
    },
    {
        "candidate_id": "1a170320ac3d1fe721d6",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'the BLM published an Environmental Assessment (EA) on November 26, 2014. The BLM requested public comment for 30'."
    },
    {
        "candidate_id": "06e4a6866f14eaf385ba",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2017-02-23'."
    },
    {
        "candidate_id": "1d495725228e46ead57d",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Tribes were consulted during formal Government-to-Government consultation, on March 21st, 2019. The Shoshone-Bannock Tribes were consulted during formal'."
    },
    {
        "candidate_id": "faa7e1beee35b352ac2e",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'C). Letters were sent to various tribes all November 17, 2014 notifying them of the proposed action and'."
    },
    {
        "candidate_id": "c2f5fb20a98f1b02bf06",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2020-07-07'."
    },
    {
        "candidate_id": "b8529f128a3b92d72db3",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2023-06-09'."
    },
    {
        "candidate_id": "034a40888b85713e77a4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2023-01-23'."
    },
    {
        "candidate_id": "f5d2a1004ab656ce8379",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2017-01-31'."
    },
    {
        "candidate_id": "af99f2a389c845958d6a",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'external scoping was conducted from November 1,2016 through December 1, 2016. No comments were submitted during extemal scoping.'."
    },
    {
        "candidate_id": "23050344bb9144b96af4",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-08-27'."
    },
    {
        "candidate_id": "dec7ec9cfbac5e54447a",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2018-09-21'."
    },
    {
        "candidate_id": "2e25b74dc32ea7d41890",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'the terms and conditions identified below. Background On September 13, 2013, you submitted an application for a crossing'."
    },
    {
        "candidate_id": "2c4292088ea805044863",
        "label": "initiation",
        "notes": "Initiation: NEPA Register project start, quote 'BLM NEPA Register project start date: 2019-12-04'."
    },
    {
        "candidate_id": "113ab084d9534bbb83c3",
        "label": "initiation",
        "notes": "Initiation: application filed or received, quote 'was submitted more than 23 months after the December 14, 2020 deadline for the submission of motions to'."
    },
    {
        "candidate_id": "b3cac1bd04840557588c",
        "label": "initiation",
        "notes": "Initiation: scoping started or notice sent, quote 'renewal of this grazing permit was received on January 27, 2012. The Initial Allotment Review and Rangeland Health'."
    },
    {
        "candidate_id": "4364166f09d6fd4a6049",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'a hearing scheduled at the County Commission on September 15, 2020 to resolve this issue. I am concerned'."
    },
    {
        "candidate_id": "d22380da91ea010f6fb8",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'a complete Shoreline Substantial Development Permit application. \uf0b7 July 6, 2010 \u2013 The City of Port Angeles issued'."
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
