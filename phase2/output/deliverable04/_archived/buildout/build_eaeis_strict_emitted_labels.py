import json
import re
import runpy
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"
TOOLS = ROOT / "phase2/output/deliverable04/buildout_label_tools.py"


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
            return "ROD or Decision Record"
        if "compliance officer" in low or "date determined" in low:
            return "NEPA determination"
        if "permit" in low or "right-of-way" in low:
            return "permit or ROW authorization"
        return "authorizing-official signature"
    if label == "initiation":
        if "register project start date" in low:
            return "NEPA Register project start"
        if "notice of intent" in low or re.search(r"\bnoi\b", low):
            return "NOI published"
        if "scoping" in low:
            return "scoping started or notice sent"
        if "initiator signature" in low:
            return "DOE Initiator signature"
        return "application filed or received"
    if "comment" in low or "public review" in low or "notice of availability" in low:
        return "comment/review or publication date"
    if "meeting" in low or "hearing" in low or "conference" in low:
        return "meeting/hearing date"
    if any(term in low for term in ("consult", "shpo", "usfws", "tribe", "biological assessment")):
        return "consultation date"
    if any(term in low for term in ("draft ea", "final ea", "draft eis", "final eis")):
        return "EA/EIS document date"
    if "doi-blm-" in low:
        return "NEPA case-number year"
    return "non-NEPA activity or historical reference"


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
    suggest = runpy.run_path(TOOLS, run_name="strict_builder")["suggest"]
    df = pd.read_csv(LAB, dtype=str, keep_default_na=False)
    target = (
        df["label"].str.strip().eq("")
        & df["split"].str.strip().ne("test")
        & df["stratum"].eq("buildout_eaeis_2026_06")
        & (
            (
                df["process_type"].eq("EA")
                & df["candidate_role"].isin(["clear_decision", "clear_initiation"])
            )
            | (
                df["process_type"].eq("EIS")
                & df["candidate_role"].eq("clear_decision")
            )
        )
    )

    labels = []
    for _, row in df[target].iterrows():
        result = suggest(row)
        if result is None:
            continue
        label, _ = result
        context = normalize(row["model_context"])
        labels.append(
            {
                "candidate_id": row["candidate_id"],
                "label": label,
                "notes": f"{label.capitalize()}: {rule_for(label, context)}, quote '{quote_around_mark(context)}'.",
            }
        )

    assert len(labels) == 384, len(labels)
    print(pd.Series([item["label"] for item in labels], name="label").value_counts())
    write_apply(labels[:200], 17)
    write_apply(labels[200:], 18)


if __name__ == "__main__":
    main()
