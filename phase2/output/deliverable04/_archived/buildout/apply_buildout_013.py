import pandas as pd


LABELS = [
    {
        "candidate_id": "ae86581238b9b963645b",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Appeal is filed with the Authorized Officer. DOI-BLM-N010-2018-0010-CX_Decision Record 2 Signature'."
    },
    {
        "candidate_id": "4f925c9a97b60e6716a7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environment. D. Signature Authorizing Official: /s/ Ruben A. S\u00e1nchez (Signature) Name: Ruben Sanchez'."
    },
    {
        "candidate_id": "3474bd0333c3b74fed5a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Project Map D: Signature Authorizing Official: Paul N. Briggs Field Manager riggs Date: 2/8/2018'."
    },
    {
        "candidate_id": "c3d2b3e8c1cc93365642",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Vanessa L. Hice Assistant Field Manager Division of Lands 2/28/17 Date Case Number: N'."
    },
    {
        "candidate_id": "922debbc816ee072fe04",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Finnicum Acting-Assistant Field Manager-Resources 7/26/21 Date 3 U.S. Department of the'."
    },
    {
        "candidate_id": "5ab083f3964522bb0174",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote '2016 month day year NEPA Compliance Officer: Fred E. Pozzuto Date: 04/14/2016 month day year'."
    },
    {
        "candidate_id": "97ec945aaec6636ca9dd",
        "label": "decision",
        "notes": "Decision: NEPA Compliance Officer signature, quote 'Biegel Sarah T. Biegel NEPA Compliance Officer Attachment(s): Environmental Checklist cc: (w/ enclosures'."
    },
    {
        "candidate_id": "8cc10e7cbc933fca6dc1",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '25/2016 Assistant Field Manager Authorizing Official: Kathleen A. Allen Date 2/26'."
    },
    {
        "candidate_id": "3bdd196e7fd6330316b8",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '215 applies. Acting Field Manager signature /s/ Peggy S. Redick Date 5/27/2015'."
    },
    {
        "candidate_id": "586050b59814ebd786d6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Dennis J. Carpenter Field Manager July 28, 2017 Date 8 Appendix 1: Operator Submitted'."
    },
    {
        "candidate_id": "5de02bba9363eca639e8",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'applies. D: Signature Authorizing Official: [Signature: Christie Price] Date: 5-3-2017 Contact'."
    },
    {
        "candidate_id": "3f96d35e21fe5d7194e2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environment. **D. SIGNATURE** Authorizing Official: /s/Dennis J. Carpenter Field Manager, Rawlins Field'."
    },
    {
        "candidate_id": "a327d00ad99d95ca9ec8",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Monticello Field Office Record of Decision and Approved Resource Management Plan, as Amended'."
    },
    {
        "candidate_id": "365eb1ba69c0597b5ee2",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'analysis. Todd D. Yeager Field Manager Cony Rais 11/15/19 Date Contact Person: Dan Sellers'."
    },
    {
        "candidate_id": "ee6c161339107c2ea0b0",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'pursuant to existing ROW grants UTU-94872 and UTU-94872-01 issued on June 11 2020 The ROW would'."
    },
    {
        "candidate_id": "21bfa4c25d142b513995",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'situation. Remarks: Authorizing Official: Arnold Pike Date: 2/10/11 Name: Arnold L. Pike'."
    },
    {
        "candidate_id": "3b443f0fc22b44e3f6d7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'analysis. Todd D. Yeager/ Field Manager 2/15/2019 Date Contact Person: Team Lead; Wade'."
    },
    {
        "candidate_id": "8eefad9e48774064ccc6",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'interest. Signature Authorizing Official: Scott Haight Date: 9/12/2019 Scott Haight, Field'."
    },
    {
        "candidate_id": "0c46cf7a8de6d1f49476",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Jeff Kitchens Deschutes Field Manager 12/12/15 Date Administrative Review or Appeal'."
    },
    {
        "candidate_id": "0550b0a8a18ab13ebf39",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'written approval of the authorized officer AO 2 The grantee shall not allow any use of'."
    },
    {
        "candidate_id": "866159f36050ffbb098e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'ocuments with the Authorized Officer and/or IBLA (43 CFR 4.413): Office of the Solicitor'."
    },
    {
        "candidate_id": "45cd2602258b7326e705",
        "label": "decision",
        "notes": "Decision: Decision Record/ROD, quote 'Creek Salvage CX and Decision Record Land Use Plan Name: Northwestern and Coast Oregon'."
    },
    {
        "candidate_id": "53a883719f18d8f33c7c",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Gibson Lenore Heppler Field Manager Eastern Interior Field Office 6/16/2014 Date Contact'."
    },
    {
        "candidate_id": "a38497f7e8ed30d733b0",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Enter text here Comment: Authorized Officer Name: Signature Office, Title and Contact Information'."
    },
    {
        "candidate_id": "74b9fe66e50fa3fef17f",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Loren C. Wickstrom Field Manager North Dakota Field Office Date: 10/26/2016'."
    },
    {
        "candidate_id": "7ea8e72daf32ba513c74",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'herbicides on or near the right-of-way shall be in accordance with the BLM approved plan Said'."
    },
    {
        "candidate_id": "1ac46e88505dc1b2620a",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Larry W. Sandoval, Jr., Field Manager 12/21/2021 Colorado River Valley Field Office'."
    },
    {
        "candidate_id": "161493d7349b3e32965c",
        "label": "decision",
        "notes": "Decision: permit/ROW issued or approved, quote 'structures within this right-of-way in strict conformity with the plan of development which was approved and'."
    },
    {
        "candidate_id": "3f72ae6400d92f0bd8b8",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'apply. D. Signature Authorizing Official: Michelle Campeau Date: 11/2/17 Acting Paul N'."
    },
    {
        "candidate_id": "ba9d35012ae2582fda66",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '523-1256. Signature Authorizing Official: JOSEPH W. ASAGON Date: 10/4/2022 (Signature)'."
    },
    {
        "candidate_id": "20c5e9c16ec60ce1f22e",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'Appeal is filed with the Authorized Officer. Signature of Authorized Official [Signature] Kathy'."
    },
    {
        "candidate_id": "79884f9facfe541de7d5",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '10/19/18 Assistant Field Manager: A.J.D Recommended Date: 10/19/18 V. DECISION I have'."
    },
    {
        "candidate_id": "813cba98727087463ee7",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'July 24 2018 Rawlins Field Manager Date 5 Appendix 1 Maps Map 1 Location of the'."
    },
    {
        "candidate_id": "3d0cda85a31514f51707",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote '43 CFR 46.215 apply. Authorizing Official: //Signed// Kenneth J. Crane Date: 3/21/2022 Name'."
    },
    {
        "candidate_id": "836e5e54a24f4328d843",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'For Cornelia Hudson, Field Manager 12/17/18 Date'."
    },
    {
        "candidate_id": "5b738b312c7787ec0eac",
        "label": "decision",
        "notes": "Decision: authorizing official signature, quote 'environment. D. SIGNATURE Authorizing Official: /s/Dennis J. Carpenter Field Manager, Rawlins Field'."
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
