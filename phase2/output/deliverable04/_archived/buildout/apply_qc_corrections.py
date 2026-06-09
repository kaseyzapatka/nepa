import pandas as pd


CORRECTIONS = [
    {
        "candidate_id": "4cf216385c08f2dbf2a5",
        "label": "neither",
        "notes": "Neither: comment deadline, quote 'request for comments to be made by December 20, 2009'.",
    },
    {
        "candidate_id": "c0201d8a761503469e0a",
        "label": "neither",
        "notes": "Neither: public-hearing date, quote 'Public Hearings Poulsbo, WA: March 4, 2015 Chimacum, WA: March 3, 2015'.",
    },
    {
        "candidate_id": "89ba5204b62a4b92a082",
        "label": "neither",
        "notes": "Neither: construction completion, quote 'Construction complete: November 2012'.",
    },
    {
        "candidate_id": "c5ac2e8d1b01900954bb",
        "label": "neither",
        "notes": "Neither: historical ROD reference, quote 'subject to BLM's October 1998 Record of Decision'.",
    },
    {
        "candidate_id": "afb0b229dbaf2c1a800f",
        "label": "neither",
        "notes": "Neither: Final EIS cover month, quote 'FALLON RANGE TRAINING COMPLEX FINAL EIS DECEMBER 2015'.",
    },
    {
        "candidate_id": "2903afe69a5382bbe6a6",
        "label": "neither",
        "notes": "Neither: Draft EIS cover month, quote 'Draft EIS - Lead Agency Review April 2013'.",
    },
    {
        "candidate_id": "94f7f766b31e25203a5b",
        "label": "neither",
        "notes": "Neither: projected permit milestone, quote 'Wetland Permit Issued (if needed) May 19, 2021'.",
    },
    {
        "candidate_id": "fe2539f8a5619f1fb0f5",
        "label": "neither",
        "notes": "Neither: Final EIS issuance, quote 'final Programmatic SNF and INEL EIS was issued April 28, 1995'.",
    },
    {
        "candidate_id": "dd7caedcd5604acbd16e",
        "label": "neither",
        "notes": "Neither: comment-period end, quote 'comment period beginning June 15, 2020 and ending June 30, 2020'.",
    },
    {
        "candidate_id": "7569c031e94093858e71",
        "label": "neither",
        "notes": "Neither: comment-period end, quote 'public review and comment period (May 26 - June 24, 2020)'.",
    },
    {
        "candidate_id": "cb9e1cf09b6f792c676e",
        "label": "neither",
        "notes": "Neither: mitigation-plan issuance, quote 'Mitigation Action Plan was issued in October 1999'.",
    },
    {
        "candidate_id": "e772190bda7007229141",
        "label": "neither",
        "notes": "Neither: Final EIS issuance, quote 'final EIS (hereinafter FEIS) was issued in May 1990'.",
    },
]


def main():
    path = "phase2/output/deliverable04/labeling_sample.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    corr = pd.DataFrame(CORRECTIONS)
    merged = df.merge(corr, on="candidate_id", how="left", suffixes=("", "_fix"))
    fix = merged["label_fix"].notna() & (
        merged["split"].astype(str).str.strip() != "test"
    )
    merged.loc[fix, "label"] = merged.loc[fix, "label_fix"]
    merged.loc[fix, "notes"] = merged.loc[fix, "notes_fix"]
    merged[df.columns].to_csv(path, index=False)
    print(f"Corrected {int(fix.sum())} rows")


if __name__ == "__main__":
    main()
