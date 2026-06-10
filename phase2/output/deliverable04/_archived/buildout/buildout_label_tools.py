import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"

VALID_LABELS = {"initiation", "decision", "neither"}
ROLE_ORDER = {
    "clear_initiation": 0,
    "clear_decision": 1,
    "proxy_initiation": 2,
    "proxy_decision": 3,
    "unknown": 4,
    "body_text": 5,
}


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def marked_window(ctx, before=260, after=320):
    m = re.search(r"\[\[.*?\]\]", ctx)
    if not m:
        return ctx[: before + after]
    return ctx[max(0, m.start() - before) : min(len(ctx), m.end() + after)]


def words(text):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)


def quote_from_match(match_text, limit=16):
    cleaned = norm(match_text).replace("[[", "").replace("]]", "")
    toks = words(cleaned)
    if len(toks) <= limit:
        return cleaned
    return " ".join(toks[:limit])


def quote_around(pattern, text, flags=re.I, limit=16):
    m = re.search(pattern, text, flags)
    if not m:
        return quote_from_match(text[:120], limit=limit)
    start = max(0, m.start() - 20)
    end = min(len(text), m.end() + 50)
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end : end + 1].isalnum():
        end += 1
    return quote_from_match(text[start:end], limit=limit)


def note(label, rule, quote):
    label_title = label.capitalize()
    return f"{label_title}: {rule}, quote '{quote}'."


def date_after_mark(ctx):
    mark = re.search(r"\[\[.*?\]\]", ctx)
    if not mark:
        return False
    tail = ctx[mark.end() : mark.end() + 260]
    date_pat = (
        r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b|"
        r"\b\d{4}\.\d{2}\.\d{2}\b|"
        r"\b\d{4}-\d{2}-\d{2}\b|"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+\d{1,2},?\s+\d{4}\b"
    )
    return bool(re.search(date_pat, tail, re.I))


def has(pattern, text, flags=re.I):
    return bool(re.search(pattern, text, flags))


def suggest(row):
    ctx = norm(row.get("model_context", ""))
    raw = norm(row.get("raw_date_text", ""))
    win = marked_window(ctx)
    low = win.lower()

    if not ctx:
        return None

    if raw.startswith("DOI-BLM") or has(r"\[\[DOI-BLM-[A-Z0-9-]+\]\]", win):
        return (
            "neither",
            note("neither", "NEPA case number", quote_around(r"DOI-BLM-[A-Z0-9-]+", win)),
        )

    if ctx.startswith("BLM NEPA Register project start date:"):
        return (
            "initiation",
            note(
                "initiation",
                "NEPA Register project start",
                "BLM NEPA Register project start date",
            ),
        )

    mark = r"\[\[.*?\]\]"

    strict_comment_deadline = [
        rf"(comment period|public comment|scoping comment|comments).{{0,140}}"
        rf"(closed|ended|expired|deadline|due|must be received|no later than|until|through|extended to)"
        rf".{{0,60}}{mark}",
        rf"(closed|ended|expired|deadline|due|must be received|no later than|until|through|extended to)"
        rf".{{0,80}}{mark}.{{0,140}}(comment period|public comment|scoping comment|comments)",
        rf"(between|from).{{0,100}}and\s+{mark}.{{0,140}}"
        rf"(comment period|public comment|scoping comment|comments)",
    ]
    for pat in strict_comment_deadline:
        if has(pat, win):
            return (
                "neither",
                note("neither", "comment-period close/deadline", quote_around(pat, win)),
            )

    # Positive rules with distinctive form/signature cues.
    if has(r"Date Determined.{0,80}\[\[", win):
        if has(r"\]\].{0,180}NEPA Compliance Officer", win) and date_after_mark(ctx):
            return (
                "initiation",
                note(
                    "initiation",
                    "Date Determined with later NCO signature",
                    quote_around(r"Date Determined.{0,80}\[\[.*?\]\]", win),
                ),
            )
        return (
            "decision",
            note(
                "decision",
                "operative Date Determined",
                quote_around(r"Date Determined.{0,80}\[\[.*?\]\]", win),
            ),
        )

    if has(r"DOE\s+Initiator\s+Signature.{0,220}\[\[", win) or has(
        r"DOE\s+INITIATOR\s+SIGNATURE.{0,220}\[\[", win
    ):
        return (
            "initiation",
            note(
                "initiation",
                "DOE Initiator signature",
                quote_around(r"DOE\s+Initiator\s+Signature|DOE\s+INITIATOR\s+SIGNATURE", win),
            ),
        )

    if has(r"(NEPA Compliance Officer|\bNCO\b|Compliance Officer).{0,180}\[\[", win):
        return (
            "decision",
            note(
                "decision",
                "NEPA Compliance Officer signature",
                quote_around(r"(NEPA Compliance Officer|\bNCO\b|Compliance Officer)", win),
            ),
        )

    strict_initiation_rules = [
        (
            rf"{mark}.{{0,160}}(issued|published|filed).{{0,130}}(Notice of Intent|\bNOI\b)",
            "NOI published/issued",
        ),
        (
            rf"(Notice of Intent|\bNOI\b).{{0,130}}(published|issued|filed).{{0,80}}(on|dated)?\s*{mark}",
            "NOI published/issued",
        ),
        (
            rf"(published|issued).{{0,80}}(Federal Register|FR).{{0,120}}"
            rf"(Notice of Intent|\bNOI\b|scoping).{{0,80}}{mark}",
            "NOI published/issued",
        ),
        (
            rf"(Notice of Intent|\bNOI\b|scoping).{{0,120}}"
            rf"(published|issued).{{0,80}}(Federal Register|FR).{{0,80}}{mark}",
            "NOI published/issued",
        ),
        (
            rf"(scoping period|scoping process|public scoping|formal scoping).{{0,140}}"
            rf"(began|opened|initiated|commenced|started).{{0,60}}{mark}",
            "scoping started",
        ),
        (
            rf"{mark}.{{0,120}}(began|opened|initiated|commenced|started).{{0,120}}"
            rf"(scoping period|scoping process|public scoping|formal scoping)",
            "scoping started",
        ),
        (
            rf"(posted|published|provided|available).{{0,140}}"
            rf"(ePlanning|eplanning|NEPA Register).{{0,80}}{mark}",
            "posted to ePlanning/NEPA Register",
        ),
        (
            rf"{mark}.{{0,120}}(posted|published|provided|available).{{0,140}}"
            rf"(ePlanning|eplanning|NEPA Register)",
            "posted to ePlanning/NEPA Register",
        ),
        (
            rf"{mark}.{{0,180}}(received|filed|submitted|accepted).{{0,140}}"
            rf"(right-of-way application|ROW application|SF-299|permit application|"
            rf"license application|Presidential Permit application|Special Recreation Permit|"
            rf"application from|application for|application\s*\(|application requesting)",
            "application/ROW filed or received",
        ),
        (
            rf"(received|filed|submitted|accepted).{{0,170}}"
            rf"(right-of-way application|ROW application|SF-299|permit application|"
            rf"license application|Presidential Permit application|Special Recreation Permit|"
            rf"application from|application for|application\s*\(|application requesting).{{0,90}}{mark}",
            "application/ROW filed or received",
        ),
        (
            rf"(application|permit application|license application|right-of-way application|ROW application)"
            rf".{{0,100}}(dated|filed on|submitted on|received on).{{0,40}}{mark}",
            "application/ROW filed or received",
        ),
        (
            rf"{mark}.{{0,120}}(approved|accepted).{{0,140}}(pre-filing|prefiling)",
            "FERC pre-filing approved",
        ),
        (
            rf"(approved|accepted).{{0,140}}(pre-filing|prefiling).{{0,80}}{mark}",
            "FERC pre-filing approved",
        ),
        (
            rf"{mark}.{{0,120}}(Notice of Application|application ready for environmental analysis)",
            "FERC/application notice",
        ),
    ]
    bad_application = has(
        r"\b(payment application|voucher|invoice|Request for Additional Information|"
        r"Data Requests?|eRAI|RAI|response to data request|comments filed|Date Filed)\b",
        win,
    )
    for pat, rule in strict_initiation_rules:
        if has(pat, win) and not (rule == "application/ROW filed or received" and bad_application):
            return ("initiation", note("initiation", rule, quote_around(pat, win)))

    if has(
        rf"(Decision Record|Record of Decision|\bROD\b|FONSI|Finding of No Significant Impact)"
        rf".{{0,140}}(signed|issued|approved|completed).{{0,120}}{mark}",
        win,
    ) or has(
        rf"(signed|issued|approved|completed).{{0,140}}"
        rf"(Decision Record|Record of Decision|\bROD\b|FONSI|Finding of No Significant Impact)"
        rf".{{0,120}}{mark}",
        win,
    ):
        return (
            "decision",
            note(
                "decision",
                "decision/FONSI/ROD signed or issued",
                quote_around(
                    r"Decision Record|Record of Decision|\bROD\b|FONSI|Finding of No Significant Impact",
                    win,
                ),
            ),
        )

    official = (
        r"Authorizing Official|Authorized Officer|Approving Official|Field Manager|"
        r"Field Office Manager|District Manager|Forest Supervisor"
    )
    generic_official_reference = has(
        r"(authorized officer after consulting|upon approval by the authorized officer|"
        r"appeal procedures|document with the Authorized Officer|Field Manager attended a meeting)",
        win,
    )
    if (
        not generic_official_reference
        and (
            has(rf"({official}).{{0,140}}(Date Signed|DATE SIGNED|Date:|DATE:|Signature|signed by|Title:).{{0,120}}{mark}", win)
            or has(rf"({official}).{{0,90}}{mark}.{{0,60}}(Date|DATE|Signature|signed)", win)
            or has(rf"(Date Signed|DATE SIGNED|Date:|DATE:).{{0,80}}{mark}.{{0,120}}({official})", win)
            or has(rf"{mark}.{{0,80}}(Date|DATE).{{0,100}}({official})", win)
            or has(rf"(s/|/s/).{{0,120}}({official}).{{0,120}}{mark}", win)
        )
    ):
        return (
            "decision",
            note(
                "decision",
                "authorizing official signature",
                quote_around(official, win),
            ),
        )

    if has(
        rf"(Record of Decision|Decision Record|\bROD\b).{{0,160}}"
        rf"(signed|issued|approved|decision).{{0,80}}{mark}",
        win,
    ) or has(
        rf"{mark}.{{0,120}}(Record of Decision|Decision Record|\bROD\b).{{0,160}}"
        rf"(signed|issued|approved|decision)",
        win,
    ):
        return (
            "decision",
            note(
                "decision",
                "Decision Record/ROD",
                quote_around(r"Record of Decision|Decision Record|\bROD\b", win),
            ),
        )

    if has(
        r"(right-of-way|ROW grant|grazing permit|special recreation permit|permit decision|"
        r"authorization).{0,120}(was|were|is|are|has been|hereby)?\s*(issued|granted|approved)",
        win,
    ) and not has(r"\bmay be issued\b", win):
        return (
            "decision",
            note(
                "decision",
                "permit/ROW issued or approved",
                quote_around(
                    r"(right-of-way|ROW grant|grazing permit|special recreation permit|"
                    r"permit decision|authorization).{0,120}(was|were|is|are|has been|hereby)?\s*"
                    r"(issued|granted|approved)",
                    win,
                ),
            ),
        )

    strict_neither_rules = [
        (
            rf"(public meeting|scoping meeting|open house|workshop|hearing|webinar|site visit)"
            rf".{{0,120}}(held|scheduled|conducted|hosted|occurred).{{0,80}}{mark}",
            "meeting or site-visit date",
        ),
        (
            rf"{mark}.{{0,100}}(public meeting|scoping meeting|open house|workshop|hearing|webinar|site visit)",
            "meeting or site-visit date",
        ),
        (
            rf"\b(consultation|SHPO|USFWS|tribal|Tribe|Section 106|ESA|Biological Assessment)\b"
            rf".{{0,150}}(concurrence|response|submitted|initiated|letter|request|meeting).{{0,90}}{mark}",
            "consultation date",
        ),
        (
            rf"{mark}.{{0,120}}\b(SHPO|USFWS|tribal|Tribe|Section 106|ESA|consultation)\b",
            "consultation date",
        ),
        (
            rf"\b(map|figure|drawing|Date Created|Date Prepared|Last Printed|User:|Sources: Esri)\b"
            rf".{{0,100}}{mark}",
            "map/figure/print date",
        ),
        (
            rf"(Date Filed|Commenting Entity Date Filed|comments filed|filed comments|"
            rf"reply comments|response to comments).{{0,100}}{mark}",
            "comment filing date",
        ),
        (
            rf"(survey|inspection|field visit|sampling|monitoring).{{0,120}}"
            rf"(conducted|performed|completed|occurred).{{0,80}}{mark}",
            "survey/inspection/activity date",
        ),
        (
            rf"\b(expiration|expires|term|effective until|valid through)\b.{{0,100}}{mark}",
            "permit term/expiration date",
        ),
        (
            rf"{mark}.{{0,120}}(Draft Environmental Impact Statement|Final Environmental Impact Statement|"
            rf"Draft EIS|Final EIS|DEIS|FEIS|PEIS|Programmatic Environmental Impact Statement)",
            "EA/EIS document or availability date",
        ),
        (
            rf"(previous|prior|historic|historical|existing).{{0,130}}"
            rf"(permit|grant|lease|ROD|Record of Decision|authorization|right-of-way|ROW).{{0,80}}{mark}",
            "prior authorization/history date",
        ),
        (
            rf"(construction|operation|activity).{{0,130}}"
            rf"(start|end|began|completed|operational|through).{{0,80}}{mark}",
            "construction/activity period date",
        ),
    ]
    for pat, rule in strict_neither_rules:
        if has(pat, win):
            return ("neither", note("neither", rule, quote_around(pat, win)))

    return None

    initiation_rules = [
        (
            r"(Notice of Intent|\bNOI\b).{0,160}(published|issued|filed|opened|initiated)",
            "NOI published/issued",
        ),
        (
            r"(published|issued|filed).{0,160}(Notice of Intent|\bNOI\b)",
            "NOI published/issued",
        ),
        (
            r"(scoping period|scoping process|public scoping|formal scoping).{0,140}"
            r"(began|opened|initiated|commenced|started|was conducted)",
            "scoping started",
        ),
        (
            r"(began|opened|initiated|commenced|started).{0,140}"
            r"(scoping period|scoping process|public scoping|formal scoping)",
            "scoping started",
        ),
        (
            r"(ePlanning|eplanning|NEPA Register).{0,140}(posted|published|provided|available|beginning)",
            "posted to ePlanning/NEPA Register",
        ),
        (
            r"(posted|published|provided|available).{0,140}(ePlanning|eplanning|NEPA Register)",
            "posted to ePlanning/NEPA Register",
        ),
        (
            r"(received|filed|submitted|accepted).{0,140}"
            r"(right-of-way application|ROW application|SF-299|permit application|"
            r"license application|Presidential Permit application|Special Recreation Permit application|"
            r"application \(|application from|application for)",
            "application/ROW filed or received",
        ),
        (
            r"(right-of-way application|ROW application|SF-299|permit application|"
            r"license application|Presidential Permit application|Special Recreation Permit application|"
            r"application \(|application from|application for)"
            r".{0,140}(received|filed|submitted|accepted)",
            "application/ROW filed or received",
        ),
        (
            r"(approved|accepted).{0,140}(pre-filing|prefiling)",
            "FERC pre-filing approved",
        ),
        (
            r"(Notice of Application|application ready for environmental analysis|document receipt of .* application)",
            "FERC/application notice",
        ),
    ]
    for pat, rule in initiation_rules:
        if has(pat, win):
            return ("initiation", note("initiation", rule, quote_around(pat, win)))

    return None


def load_blanks():
    df = pd.read_csv(LAB, dtype=str, keep_default_na=False)
    blank = df["label"].astype(str).str.strip().eq("")
    rows = df[blank].copy()
    rows["_role_order"] = rows["candidate_role"].map(ROLE_ORDER).fillna(99).astype(int)
    rows["_orig_order"] = range(len(rows))
    return df, rows.sort_values(["_role_order", "_orig_order"])


def build_suggestions(rows):
    labels = []
    skipped = 0
    for _, row in rows.iterrows():
        got = suggest(row)
        if got is None:
            skipped += 1
            continue
        label, notes = got
        if label not in VALID_LABELS:
            raise ValueError(label)
        labels.append(
            {
                "candidate_id": row["candidate_id"],
                "label": label,
                "notes": notes,
                "candidate_role": row["candidate_role"],
                "process_type": row["process_type"],
            }
        )
    return labels, skipped


def write_apply(labels, chunk):
    out = ROOT / f"phase2/output/deliverable04/apply_buildout_{chunk:03d}.py"
    payload = [
        {k: item[k] for k in ("candidate_id", "label", "notes")}
        for item in labels
    ]
    text = f"""import pandas as pd


LABELS = {json.dumps(payload, indent=4)}


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
    print(f"Applied {{int(apply.sum())}} labels to labeling_sample.csv")


if __name__ == "__main__":
    main()
"""
    out.write_text(text)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--only-label", choices=sorted(VALID_LABELS))
    args = parser.parse_args()

    _df, rows = load_blanks()
    labels, skipped = build_suggestions(rows)
    if args.only_label:
        labels = [item for item in labels if item["label"] == args.only_label]
    print(f"blank rows: {len(rows)}")
    print(f"suggested labels: {len(labels)}")
    print(f"skipped ambiguous: {skipped}")
    if labels:
        summary = pd.DataFrame(labels)
        print("suggested by label:")
        print(summary["label"].value_counts().to_string())
        print("suggested by candidate_role x label:")
        print(
            summary.pivot_table(
                index="candidate_role",
                columns="label",
                values="candidate_id",
                aggfunc="count",
                fill_value=0,
            ).to_string()
        )
        print("examples:")
        for item in labels[: args.examples]:
            print(f"{item['candidate_id']} | {item['candidate_role']} | {item['label']} | {item['notes']}")

    if args.write:
        if args.chunk is None:
            raise SystemExit("--chunk required with --write")
        chosen = labels[: args.limit]
        out = write_apply(chosen, args.chunk)
        print(f"wrote {len(chosen)} labels to {out}")


if __name__ == "__main__":
    main()
