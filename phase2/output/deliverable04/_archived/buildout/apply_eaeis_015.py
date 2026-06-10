import pandas as pd


LABELS = [
    {
        "candidate_id": "4c465b759b965abfbd41",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'final approval of the County\u2019s updated SMP on February 22, 2016 making the County\u2019s comprehensive SMP update effective'."
    },
    {
        "candidate_id": "e678097acf2d05d13f89",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'law. Noxious Weeds Rules (Idaho Administrative Procedures Act, 02.06.22) designate weeds as noxious statewide. Idaho\u2019s noxious weeds'."
    },
    {
        "candidate_id": "865f5174ad9186d90d9a",
        "label": "neither",
        "notes": "Neither: non-NEPA milestone or historical reference, quote 'by CARB in 2009 and became effective on April 15, 2010. The regulation establishes annual performance standards for'."
    },
    {
        "candidate_id": "ce0a40b62ae19fece9e2",
        "label": "neither",
        "notes": "Neither: meeting/hearing date, quote 'England, NRICP, NRCS, GZA, and RI Congressional Offices. February 2013: Public Meeting at the Cranston Senior Center.'."
    },
    {
        "candidate_id": "01faec96a4fdf24fe7ac",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote '8, 2013. The public hearing was held on January 9, 2014 and the public comment period ended on'."
    },
    {
        "candidate_id": "55af6e3d8476cb2cdd83",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2021-04-05'."
    },
    {
        "candidate_id": "ae3b418b9dbf9c218a22",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'DeWitt Interim Associate District Manager, Resources NEPA Compliance 02/06/2023 Brian Kennedy Associate District Manager, Minerals NEPA Compliance'."
    },
    {
        "candidate_id": "a86dd6b8419c8e3971e3",
        "label": "decision",
        "notes": "Decision: authorizing-official signature, quote 'Issued in Portland, Oregon. /s/ Stephen J. Wright_______ January 10, 2003 Stephen J. Wright Date Administrator and Chief'."
    },
    {
        "candidate_id": "cf09636c5d36cff1a213",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'the visual analysis; and the current project schedule. July 16, 2015 \u2013 Cooperating Agencies The Corps met with'."
    },
    {
        "candidate_id": "6364a7b3ec9a9629657f",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Final EIS Chapter 2, Proposed Action and Alternatives October 2024 2-14 and black widows, under an approved Pesticide'."
    },
    {
        "candidate_id": "e182701c91da2a4a26d0",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'Compliance Tribes, Individuals, Organizations, or Agencies Consulted On June 1, 2023, consultation letters describing the Proposed Action, as'."
    },
    {
        "candidate_id": "7d1cf0286d6337c1b9d7",
        "label": "neither",
        "notes": "Neither: consultation/coordination date, quote 'to be a Cooperating Agency with Kurt Dongoske. August/September 2012: MOU signed by joint-leads. April 17, 2014: Bruce'."
    },
    {
        "candidate_id": "38879e9763fd37242531",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'SCOTT ARMENTROUT Digitally signed by SCOTT ARMENTROUT Date: 2021.11.22 12:33:28 -08'00''."
    },
    {
        "candidate_id": "2c7c7d2cbeeeaf3c2a3c",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'SCOTT ARMENTROUT Digitally signed by SCOTT ARMENTROUT Date: 2021.10.15 11:17:34 -07'00''."
    },
    {
        "candidate_id": "ba621d2650186f2ab275",
        "label": "neither",
        "notes": "Neither: comment/review or publication date, quote 'approval by the USACE which is expected in August 2010. EIS and Public Involvement Process Proposed Action. The'."
    },
    {
        "candidate_id": "9494cc4cfc8276776abc",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2015-02-27'."
    },
    {
        "candidate_id": "9fa081e99248034c5bdf",
        "label": "decision",
        "notes": "Decision: authoritative register decision date, quote 'BLM NEPA Register decision date (fonsi): 2016-12-13'."
    },
    {
        "candidate_id": "665ec8ff5f7bd9501514",
        "label": "decision",
        "notes": "Decision: FONSI determination, quote 'to implement the proposed action. J Date signed September 25, 2002 Date ~!eld Manager Northern Field Office Si'."
    },
    {
        "candidate_id": "f1388c3bc400833a4287",
        "label": "neither",
        "notes": "Neither: EA/EIS document date, quote 'Uranium Final Environmental Impact Statement was issued in June 1996. DOE prepared this EIS because of the need'."
    },
    {
        "candidate_id": "ef4f84833b66bb9e1d94",
        "label": "decision",
        "notes": "Decision: agency authorization, quote 'Species Act of 1973 (16 U.S.C. \u00a7 1531-1544). January 13, 2021 Paul Nissenbaum Date of Approval Associate Administrator'."
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
