import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"

EA_DECISION = {
    "d0223ec8eb3568395ad5",
    "dd385e875e428fcbbcc4",
    "fcade8dcf4ff0e0c89ac",
    "8635ffad903c4c220891",
    "5e2f17d2ade439afc6d4",
    "ddc2b8690f7d14dfd754",
    "c94467c434bfda1a83e7",
    "d1165cba70b1329e20f8",
    "031f1d5370aa941546f0",
    "fbe2b3c6deaba6f8863f",
    "ae3a02ff4178fbcf5161",
    "2492b1e2e2eca123e03d",
    "12e4ffd43bfa6cafe4b0",
    "dc0d8c8f4d589d1c0ca3",
    "b565fa85d93092bea44b",
    "6b8fabf7e0899112ddf5",
    "2532195635e2c6db0383",
    "3ef46166e986bb43b3f3",
    "e705fb7aba153632fda2",
    "71478aea0e0fc22d7ab7",
    "129873ab6facdd5d2e1f",
    "e0a49e04241da05a295f",
    "645d5aa01c0b5f162427",
    "c53bd09facd45be388e0",
    "291d718725b07a8d71b9",
    "1f43eb18712d9cda2f94",
    "7ecdd32c3c80423c0a13",
    "7cacd443a7f4671060d8",
    "29cba0f5171d2ded80f5",
    "06b6a2dec90495b96455",
    "e2c08445e75feb178598",
    "293651ff63f7a1890bee",
    "88b96cfc9e8fbf151158",
    "198cd686b20b83e2ead2",
    "8c70fc9e66f7e6fef26c",
    "e44a4c06f5db8f6a6ca9",
    "17c8e5eb0b1564ce334d",
    "659946f45e54649b1328",
    "2f2bf8c8dc5acb599841",
    "f1eb6082589b0ae050c8",
    "a14124bb472cdbde6486",
    "876c892c5b837969c9d6",
    "92b6c816cacbb481ae72",
    "55af6e3d8476cb2cdd83",
    "38879e9763fd37242531",
    "2c7c7d2cbeeeaf3c2a3c",
    "9494cc4cfc8276776abc",
    "9fa081e99248034c5bdf",
}

EA_INITIATION = {
    "d02963c42444c5889188",
    "25c667d6c50aedde1311",
}

EIS_DECISION = {
    "b0811c4cb91545cfd9e1",
    "ba2044e54a48dc48334c",
    "b5a0ffb216b78b38e1e3",
    "6e0023b0592116a9f028",
    "99d46b22d5a965b2d92c",
    "36e69334764bf9398f36",
    "1e1ddc526a66192ab873",
    "ade5bb38c0bcb52fc25d",
    "684b7dd577bfb33bf528",
    "83000620295645479c49",
    "126e184bf8b303f9b337",
    "0b19da21b34e246d6a1d",
    "17c23c4ac9cbeb5504e3",
    "c7475a7e6e2692dddd97",
    "206453de713acd8e74a2",
    "bf280b0848b891f14756",
    "679b97bab3fa92ce13c5",
    "a86dd6b8419c8e3971e3",
    "665ec8ff5f7bd9501514",
    "ef4f84833b66bb9e1d94",
}

EIS_INITIATION = {
    "5d2dd519fd1f65bdcdf2",
    "303ae636772089af5110",
    "76712ee10db547a63eee",
    "ca3f79634f4343178676",
    "3dbfec57392c13b0ebf6",
    "08e3ed2c1184f790857d",
    "3bd43916a7181bfa60fb",
    "c569e59dd9ce16ec4196",
    "ef8d7cfbffb7bf6b71f5",
    "d3a317a50e52359e1c76",
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
    if label == "decision":
        if "register decision date" in low:
            return "authoritative register decision date"
        if "finding of no significant impact" in low or "fonsi" in low:
            return "FONSI determination"
        if "record of decision" in low or re.search(r"\brod\b", low):
            return "ROD date"
        if "approval" in low or "approved" in low or "authorization" in low:
            return "agency authorization"
        return "authorizing-official signature"
    if label == "initiation":
        if "notice of intent" in low or re.search(r"\bnoi\b", low):
            return "NOI published"
        if "application" in low or "proposal" in low:
            return "application filed or received"
        return "scoping started"
    if "comment" in low or "public review" in low or "notice of availability" in low:
        return "comment/review or publication date"
    if "meeting" in low or "hearing" in low or "open house" in low:
        return "meeting/hearing date"
    if any(term in low for term in ("consult", "shpo", "tribe", "programmatic agreement", "moa")):
        return "consultation/coordination date"
    if any(term in low for term in ("draft eis", "final eis", "feis", "environmental assessment")):
        return "EA/EIS document date"
    if any(term in low for term in ("specialist", "review", "preparer", "archaeologist", "biologist")):
        return "reviewer/specialist date"
    if any(term in low for term in ("expires", "valid from", "term")):
        return "permit term/expiration date"
    return "non-NEPA milestone or historical reference"


def write_apply(labels, chunk):
    out = ROOT / f"phase2/output/deliverable04/apply_eaeis_{chunk:03d}.py"
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
    out.write_text(text)
    print(f"Wrote {len(labels)} labels to {out}")


def main():
    df = pd.read_csv(LAB, dtype=str, keep_default_na=False)
    selected = df[
        df["label"].str.strip().eq("")
        & df["process_type"].isin(["EA", "EIS"])
        & df["candidate_role"].eq("clear_decision")
    ].copy()
    assert len(selected) == 220, len(selected)

    labels = []
    for _, row in selected.iterrows():
        cid = row["candidate_id"]
        if cid in EA_DECISION or cid in EIS_DECISION:
            label = "decision"
        elif cid in EA_INITIATION or cid in EIS_INITIATION:
            label = "initiation"
        else:
            label = "neither"
        context = normalize(row["model_context"])
        rule = rule_for(label, context)
        quote = quote_around_mark(context)
        labels.append(
            {
                "candidate_id": cid,
                "label": label,
                "notes": f"{label.capitalize()}: {rule}, quote '{quote}'.",
            }
        )

    expected = EA_DECISION | EA_INITIATION | EIS_DECISION | EIS_INITIATION
    missing = expected - set(selected["candidate_id"])
    assert not missing, missing
    print(pd.DataFrame(labels)["label"].value_counts().to_string())
    write_apply(labels[:200], 14)
    write_apply(labels[200:], 15)


if __name__ == "__main__":
    main()
