import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"
OUT = ROOT / "phase2/output/deliverable04/apply_eaeis_019.py"

EA_DECISION = {
    "558e2e3e1b0a14583ba3",
    "7359ec4ead16e4b8f943",
    "29c9e52b38c741de0ce5",
    "cfc3f0275bff0a50d48c",
    "7caa0d4f73aae7f12c69",
    "b2c6692e790e9e0d41e2",
    "b2d61c2b42ebceb2b315",
    "ccbf62a5feeef09e292d",
    "977c99309705f70293dc",
    "868c45b640517a7141ae",
    "a4591fcfe6b6bf827b78",
    "65a2b77a13d271a43c8b",
    "34803914477dc040d675",
    "4ff49e6774704d864093",
    "27fcd466385a1750e378",
    "0b95e7666d228b232fe8",
    "d6038d005d624304925e",
    "51a527f8725d36099b24",
    "fee9e2c1d20e54b36a95",
    "441aa93742d9358df366",
    "c0b9892c25f3813ec971",
    "bc4725959c23d8601e64",
    "b2788dcb6902fdaf5e92",
    "b5b986e0e080b32b67ca",
    "edfce5e37b36808dbc31",
    "52b98426f2cdabce9ead",
    "40c0e0b33510f7fdcbf2",
    "c156181dd4dde542de8a",
    "df1ae932d4723c8167e5",
    "8ab0e62d5deebbec79e8",
    "90ba73ed43af57c5bc56",
    "2503bbce8edd0f02c96b",
    "7300c3ad5fcddd9c480b",
    "e6128752cea14ac06ade",
    "516a3f18a941eb56fd6d",
    "9399e63b572654d6e5ea",
    "06b9dded8bfd5ed04e14",
    "0b473b746bdccc113f55",
    "922e55b7f440767a5938",
    "0481352716efd484fb18",
    "6d1c17280ac4c62ed682",
    "1edb167ad0a681418a89",
    "f1b1df79ee245dc69ef5",
    "a3dfa4649a2ada66e9e1",
    "69a592180e3cf548c199",
    "920f09138c0f5aeb97f5",
    "a8ffc2d241bb879dd58c",
    "bc1afcf64557544124d9",
    "b2aaf0ce1fdb28f1f6ce",
    "59a2a73f280fcfa883d1",
    "0b000dad0fa43ab130ee",
    "2159e0b55c2dd0501488",
    "231a8ef9414ac2b315ae",
    "6be91556a02ea4d987ca",
    "34181d7f004a275b24f0",
    "6bf73ae8638b3a12517a",
    "141c66c11deeb967fdc9",
    "36f02974d7288eacc5c1",
    "abc83ab922e814c40d47",
    "ab1aae11bc12c1a5a785",
    "19eaed0f58337935ac5c",
    "a07e88f52f78da12c197",
    "185391fc4e31952e4494",
    "c09ff7d8eb0047862fe6",
    "90bfa74c92b356652703",
    "4ebe6de02c6e1ebf0261",
    "618eb4a3c83a20b4a47d",
    "86ba54f12f69634e94dc",
    "9ad6ae0439aa2cc8fe43",
}

EIS_DECISION = {
    "66dc0c7e2c9295ed9e7b",
    "1c695f0ee13d20eed990",
    "aab9887191756e18a041",
    "f78449b3272687cfdcfb",
    "945dc3cc1c4311c2ffb3",
    "12fb025d0d2d5996052f",
    "43566684dd4fb1e3c15c",
    "3d4e2036fa0f3b246868",
    "7352fa69f2d96ee7f4ac",
    "28ff4cdf9b03683ffd10",
    "121bdda4e1c044c91951",
    "7bd5b10a965a0a6e8ee0",
    "91f59444f90065e124e6",
    "7a0d73de1349531d1952",
    "bc03215e46578d976e2a",
    "b4d8991f1d0065e5bc6e",
    "bcd8932100baa30ed119",
    "ca750857d02475c776f0",
    "70d49c7fdfcd2f3f4cbc",
    "61e2e8cf594c8bec0d43",
    "a948c97d33be731a29b8",
    "5d27a921fa4099bcb622",
    "b8e53bd8d4558ca6758c",
    "b83666d15bae97b92257",
    "018fb4cf5205350c1476",
    "ac92d74bff9e17602a3c",
    "dffd726b0040ecc3b518",
    "41d01cb8e07008debe60",
    "5bf3fe688af8299b0563",
    "230e3f076961adf90f8b",
    "01271f52ffd2d9bcfc2c",
    "833f541e203334583043",
    "a387e20826ff284ea446",
    "a03f29dfe862f4722dc3",
    "5f41d9aea3e5a1ef3cc4",
    "703aa17f174e8c0a04fe",
    "c775374b4958ada32031",
    "62d38a033f1d6475b9ab",
    "612e78a9fcd3f0d55e78",
    "7d4f61ebe2938563ca3d",
    "d6f983d176434ab60f9a",
    "8943f4b84de5b425e1c8",
    "1e4ad073bf951746807b",
    "a21a8e6284bc4cc95e3b",
    "7afbe7db37dd90b28ebb",
    "867124b28e0dfd2b1506",
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


def rule_for(context):
    low = context.lower()
    if "register decision date" in low:
        return "authoritative register decision date"
    if "finding of no significant impact" in low or "fonsi" in low:
        return "FONSI determination"
    if "record of decision" in low or re.search(r"\brod\b", low):
        return "ROD or Decision Record"
    return "authorizing-official signature"


def main():
    df = pd.read_csv(LAB, dtype=str, keep_default_na=False)
    ids = EA_DECISION | EIS_DECISION
    rows = df[df["candidate_id"].isin(ids)].copy()
    assert len(EA_DECISION) == 69, len(EA_DECISION)
    assert len(EIS_DECISION) == 46, len(EIS_DECISION)
    assert len(rows) == len(ids), (len(rows), len(ids))
    assert rows["label"].str.strip().eq("").all()
    assert rows["split"].str.strip().ne("test").all()

    labels = []
    for _, row in rows.iterrows():
        context = normalize(row["model_context"])
        labels.append(
            {
                "candidate_id": row["candidate_id"],
                "label": "decision",
                "notes": f"Decision: {rule_for(context)}, quote '{quote_around_mark(context)}'.",
            }
        )

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
    print(f"Wrote {len(labels)} labels to {OUT}")


if __name__ == "__main__":
    main()
