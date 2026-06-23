"""D6 — extract the CEQ government-wide Categorical Exclusion (CE) catalog.

Parses the CEQ "List of Federal Agency Categorical Exclusions (CE List)"
spreadsheet (`phase2/notes/deliverable06/CE_catalog.xlsx`, one sheet per agency)
into a structured, Claude-readable JSON + Markdown so D6 can cross-reference
candidate categories against *already-existing* CEs (and avoid surfacing them).

Source (cite this): CEQ List of Federal Agency Categorical Exclusions —
https://ceq.doe.gov/nepa-practice/categorical-exclusions.html

NOTE: the spreadsheet is free text and each agency formats CEs differently
(e.g. DOE `A1`/`B1.1`; BLM lettered sections with `(n)` items), interleaved with
extraordinary-circumstance and application-procedure boilerplate. This is a
faithful, best-effort capture: every substantive row is preserved with a heuristic
`kind` tag; the per-agency CFR/source URL remains the authoritative reference.

Output (phase2/notes/deliverable06/):
  - ce_catalog_extracted.md     human/Claude-readable, grouped by agency

The committed `.md` is the durable artifact future collaborators use; this script
is kept for provenance (how the `.md` was produced from the xlsx). It depends on
`openpyxl`, which is intentionally NOT in requirements.txt (one-off ingest) — to
re-run, `pip install openpyxl` first.
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NOTES = HERE.parents[1] / "notes" / "deliverable06"
SRC = NOTES / "CE_catalog.xlsx"
MD_OUT = NOTES / "ce_catalog_extracted.md"

CATALOG_URL = "https://ceq.doe.gov/nepa-practice/categorical-exclusions.html"
CATALOG_SOURCE = ("CEQ List of Federal Agency Categorical Exclusions (CE List), "
                  "Council on Environmental Quality, Executive Office of the President")

CE_CODE = re.compile(r"^([A-Z]{1,4}\d+(?:\.\d+)*)\b")           # DOE A1, B1.1; EPA-style codes
LETTER_HEAD = re.compile(r"^([A-Z]\.|[IVXLC]{1,4}\.)\s")        # "A. Fish and Wildlife."
NUM_ITEM = re.compile(r"^(\(?\d+\)|\([a-z]\)|\([ivxlc]+\)|\d+\.)\s")  # (1), (a), (iii), 1.
URL_RX = re.compile(r"^https?://")
CITE_RX = re.compile(r"\bCFR\b|\bU\.?S\.?C\.?\b|Departmental Manual|\bDM\b|Subpart|Appendix|Part \d", re.I)
EC_RX = re.compile(r"extraordinary circumstance|resource conditions|the mere presence", re.I)


def classify(text: str) -> str:
    if CE_CODE.match(text):
        return "ce_code"
    if LETTER_HEAD.match(text):
        return "section_heading"
    if NUM_ITEM.match(text):
        return "numbered_item"
    return "text"


def extract_code(text: str) -> str:
    m = CE_CODE.match(text)
    return m.group(1) if m else ""


def main() -> None:
    xl = pd.ExcelFile(SRC)
    agency_sheets = [s for s in xl.sheet_names if s.lower() != "about"]

    agencies, total_entries, total_ce_like = [], 0, 0
    for sheet in agency_sheets:
        df = xl.parse(sheet)
        col = df.columns[0]
        department = str(col).split("\n")[0].strip()
        vals = [str(v).strip() for v in df[col].dropna().tolist() if str(v).strip()]
        if not vals:
            continue
        agency_name = vals[0]
        source_urls = [v for v in vals if URL_RX.match(v)]
        citations = [v for v in vals[1:8] if CITE_RX.search(v) and not URL_RX.match(v)][:3]

        entries = []
        for seq, v in enumerate(vals[1:], start=1):
            if URL_RX.match(v):
                continue
            kind = classify(v)
            is_ec = bool(EC_RX.search(v))
            ce_like = kind in ("ce_code", "numbered_item") and not is_ec
            entries.append({
                "seq": seq, "kind": kind, "code": extract_code(v),
                "is_extraordinary_circumstance_context": is_ec,
                "is_ce_like": ce_like, "text": v,
            })
        total_entries += len(entries)
        total_ce_like += sum(e["is_ce_like"] for e in entries)
        agencies.append({
            "sheet": sheet, "department": department, "agency": agency_name,
            "source_urls": source_urls, "citations": citations,
            "n_entries": len(entries), "n_ce_like": sum(e["is_ce_like"] for e in entries),
            "entries": entries,
        })

    # --- Markdown (grouped by agency) — the durable, Claude-readable artifact ---
    lines = [
        "# CEQ Government-Wide Categorical Exclusion (CE) Catalog — extracted",
        "",
        f"**Source:** {CATALOG_SOURCE}",
        f"**Source URL:** {CATALOG_URL}",
        f"**Source file:** `{SRC.name}`",
        "",
        f"Agencies: {len(agencies)} · entries captured: {total_entries:,} · "
        f"CE-like entries: {total_ce_like:,}",
        "",
        "> Faithful best-effort capture of the CEQ free-text spreadsheet. Each entry keeps a "
        "heuristic tag; section headings and some extraordinary-circumstance / application rows "
        "are included. The per-agency CFR/source URL is authoritative. Use this list to "
        "cross-reference D6 candidate categories so the deliverable does not re-surface an "
        "action already covered by an existing CE.",
        "",
        "---",
        "",
    ]
    for a in agencies:
        lines.append(f"## {a['agency']}  ·  sheet `{a['sheet']}`")
        if a["department"] and a["department"] != a["agency"]:
            lines.append(f"*Department:* {a['department']}")
        if a["citations"]:
            lines.append(f"*Citation:* {' | '.join(a['citations'])}")
        for u in a["source_urls"]:
            lines.append(f"*Source:* {u}")
        lines.append(f"*CE-like entries: {a['n_ce_like']} of {a['n_entries']} captured rows*")
        lines.append("")
        for e in a["entries"]:
            tag = e["code"] or ("EC" if e["is_extraordinary_circumstance_context"]
                                else ("•" if e["is_ce_like"] else "·"))
            lines.append(f"- **[{tag}]** {e['text']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ce_catalog] agencies={len(agencies)} entries={total_entries:,} "
          f"ce_like={total_ce_like:,}")
    print(f"[ce_catalog] wrote {MD_OUT}")


if __name__ == "__main__":
    main()
