import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"
OUT = ROOT / "phase2/output/deliverable04/apply_eaeis_016.py"

INITIATION = {
    "8a854a5df7dab8663d27",
    "ca68e6c2f53ab92a9e78",
    "0d6b1e394e13984c78b7",
    "543df146f57d9462a974",
    "ef1761ab26bbbaebbdf8",
    "e0e880e87fa379c683d7",
    "90ff8992aca450179b2a",
    "37a4df9ede6cbab0a4dd",
    "ad4e7b055c7f14e887ad",
    "2af85b1698f0abbca5ae",
    "6824408dd48bfa4202c5",
    "0562454afe98fa2d0a55",
    "01f35551c19c23def588",
    "8a9eec7ce2289c1ecdc0",
    "b289c84a8689080dc6ee",
    "6bb8426b7a798dffade9",
    "48c9bc97d6c9361eebe9",
    "a4e977f02f204623d86b",
    "4404904c946773227381",
    "12763f73349d7a74864d",
    "cb68ce2f8bdf9a14ba0f",
    "08fd443a9f80bf84a67d",
    "f2f7369a19a0a50fbad3",
    "906cbc33ca5850006e03",
    "695c8f98caeb1a6f1d32",
    "3f8006b3aafec3ad0e22",
    "41fdc12d594aa72c6586",
    "ea3bc210cf6b8dd75ece",
    "c52b868d54781daa425f",
    "811e7f3c05d1eaea7c98",
    "30288992f1b593503dd4",
    "c61509bf93dcd36b7a8e",
    "e97e85508dedec1b0f66",
    "cfcfb64af12a2316beab",
    "a2d229b4dc71f490f868",
    "cdf8817a39fac982d2a9",
    "3cea633aa9e7f4b59406",
    "aae99a2213daf70fc74a",
    "86ba558602fba63be7fe",
    "65696cf0a7ef9f58b425",
    "fad726551d4aa1674c54",
    "61ac2fe1dcb34925c8c4",
    "f410ab7fbf4f0a86caf7",
    "f68f53ac3a429ec84787",
    "cf8bcdb119a050cc4b28",
    "42a8dafe16ed13935bc8",
}


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def quote_around_mark(text, limit=18):
    text = normalize(text)
    match = re.search(r"\[\[(.*?)\]\]", text)
    if not match:
        tokens = list(re.finditer(r"\S+", text))
        return text[: tokens[min(limit, len(tokens)) - 1].end()] if tokens else ""

    marked = match.group(1)
    unmarked = text[: match.start()] + marked + text[match.end() :]
    mark_start = match.start()
    mark_end = mark_start + len(marked)
    tokens = list(re.finditer(r"\S+", unmarked))
    marked_indexes = [
        i
        for i, token in enumerate(tokens)
        if token.end() > mark_start and token.start() < mark_end
    ]
    first = max(0, marked_indexes[0] - 8)
    last = min(len(tokens), marked_indexes[-1] + 9, first + limit)
    return unmarked[tokens[first].start() : tokens[last - 1].end()]


def rule_for(label, context):
    low = context.lower()
    if label == "initiation":
        if "register project start date" in low:
            return "NEPA Register project start"
        if "noi date" in low or "notice of intent" in low:
            return "NOI published"
        if "scoping" in low:
            return "scoping started or notice sent"
        if "news release" in low or "published an article" in low:
            return "external scoping notice"
        return "application filed or received"

    if "comment" in low or "public review" in low:
        return "comment/review date"
    if "meeting" in low or "hearing" in low or "conference" in low:
        return "meeting/hearing date"
    if any(term in low for term in ("consult", "shpo", "usfws", "tribe", "biological assessment")):
        return "consultation date"
    if any(term in low for term in ("draft ea", "final ea", "environmental assessment")):
        return "EA document or mid-process date"
    if any(term in low for term in ("prior", "previous", "historical", "order", "grant", "lease")):
        return "historical authorization or reference"
    return "non-NEPA activity or milestone"


def write_apply(labels):
    text = f"""import pandas as pd


LABELS = {json.dumps(labels, indent=4)}


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
    print(f"Applied {{int(apply.sum())}} labels")


if __name__ == "__main__":
    main()
"""
    OUT.write_text(text)


def main():
    df = pd.read_csv(LAB, dtype=str, keep_default_na=False)
    rows = df[
        df["label"].str.strip().eq("")
        & df["split"].str.strip().ne("test")
        & df["process_type"].eq("EA")
        & df["candidate_role"].eq("clear_initiation")
    ].copy()
    assert len(rows) == 148, len(rows)
    missing = INITIATION - set(rows["candidate_id"])
    assert not missing, sorted(missing)

    labels = []
    for _, row in rows.iterrows():
        label = "initiation" if row["candidate_id"] in INITIATION else "neither"
        context = normalize(row["model_context"])
        quote = quote_around_mark(context)
        labels.append(
            {
                "candidate_id": row["candidate_id"],
                "label": label,
                "notes": f"{label.capitalize()}: {rule_for(label, context)}, quote '{quote}'.",
            }
        )

    write_apply(labels)
    print(pd.Series([item["label"] for item in labels], name="label").value_counts())
    print(f"Wrote {len(labels)} labels to {OUT}")


if __name__ == "__main__":
    main()
