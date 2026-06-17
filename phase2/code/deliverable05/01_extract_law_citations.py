#!/usr/bin/env python
"""
D5 / 01 — Extract legislation citations from review documents  [SELF-CONTAINED]

Scans document page text for explicit citations to the three major infrastructure laws
(ARRA 2009, BIL 2021, IRA 2022) plus supporting DOE funding-program signals. A citation to a
law can only appear *after* the law passed, so the rise of law-citing actions is direct,
coverage-robust evidence that a spike is *associated with* the legislation (Q2 of D5).

Scans all three corpora (CE / EA / EIS) so the by-review-type contrast is real data; CE is the
focus. To keep the scan cheap, DuckDB pre-filters to pages containing any law keyword, then
Python does precise detection + acronym disambiguation on those candidate pages.

Output: phase2/data/analysis/deliverable05/law_citations.parquet
Grain : one row per (project_id, law_name)
Cols  : project_id, process_type, law_name, citation_count, n_docs_matched, first_match_type,
        first_context, first_document_id, first_page_number, law_citations_extraction_run_at

Usage:
  python phase2/code/deliverable05/01_extract_law_citations.py                 # all sources
  python phase2/code/deliverable05/01_extract_law_citations.py --source ce --sample 200
"""

import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa'.")

import re
import argparse
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import pandas as pd

TIMELINE_PATH = Path("phase2/data/analysis/timeline/timeline_project_dates.parquet")
OUTPUT_PATH = Path("phase2/data/analysis/deliverable05/law_citations.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SOURCES = {"ce": "CE", "ea": "EA", "eis": "EIS"}

# --- Disambiguation context (checked within +/-WINDOW chars of a match) ----------------------
IRA_CONTEXT = re.compile(
    r"clean energy|renewable|solar|wind|climate|greenhouse gas|emission|battery|"
    r"transmission|electric vehicle|inflation reduction|grid|decarboniz", re.IGNORECASE)
BIL_CONTEXT = re.compile(r"infrastructure|jobs act|iija|bipartisan", re.IGNORECASE)
# "Recovery Act" alone collides with the Resource Conservation and Recovery Act (RCRA, 1976) and
# other "...Recovery Act"s, so the bare short-name must be affirmed as ARRA and not be RCRA.
ARRA_AFFIRM = re.compile(
    r"reinvestment|stimulus|\b2009\b|111-?5|\bARRA\b|recovery and reinvestment", re.IGNORECASE)
ARRA_FORBID = re.compile(r"conservation|resource\s+conservation", re.IGNORECASE)

# --- Detection patterns ---------------------------------------------------------------------
# Each entry: (pattern, match_type, require_ctx, forbid_ctx). require/forbid are checked within
# +/-WINDOW chars; None = unconditional.
CITATION_PATTERNS = {
    "ARRA": [
        (r"American Recovery and Reinvestment Act", "full_name", None, None),
        (r"Recovery and Reinvestment Act", "full_name", None, None),
        (r"\bARRA\b", "acronym", None, None),
        (r"Section 1603", "program", None, None),
        (r"Section 1705", "program", None, None),
        (r"Recovery Act", "short_name", ARRA_AFFIRM, ARRA_FORBID),
    ],
    "BIL": [
        (r"Bipartisan Infrastructure Law", "full_name", None, None),
        (r"Infrastructure Investment and Jobs Act", "full_name_alt", None, None),
        (r"\bIIJA\b", "acronym", None, None),
        (r"\bBIL\b", "acronym", BIL_CONTEXT, None),
    ],
    "IRA": [
        (r"Inflation Reduction Act", "full_name", None, None),
        (r"\bIRA\b", "acronym", IRA_CONTEXT, None),
    ],
    "DOE_funding": [
        (r"Loan Programs Office", "program", None, None),
        (r"Title XVII", "program", None, None),
        (r"Section 1703", "program", None, None),
    ],
}
COMPILED = {law: [(re.compile(p, re.IGNORECASE), t, req, forb) for p, t, req, forb in pats]
            for law, pats in CITATION_PATTERNS.items()}

# DuckDB pre-filter (RE2). Lowercase full-name signals OR case-sensitive acronyms.
PREFILTER_LOWER = ("recovery act|reinvestment|inflation reduction|infrastructure investment|"
                   "bipartisan infrastructure|loan programs office|title xvii|"
                   "section 1603|section 1705|section 1703")
PREFILTER_ACRONYM = r"\b(ARRA|IIJA|IRA|BIL)\b"

WINDOW = 200
CONTEXT_PAD = 100


def detect_in_page(text: str):
    """Yield (law, match_type, match_text, context) for every kept citation on a page."""
    for law, pats in COMPILED.items():
        for rx, mtype, require, forbid in pats:
            for m in rx.finditer(text):
                if require is not None or forbid is not None:
                    window = text[max(0, m.start() - WINDOW): m.end() + WINDOW]
                    if require is not None and not require.search(window):
                        continue
                    if forbid is not None and forbid.search(window):
                        continue
                ctx = text[max(0, m.start() - CONTEXT_PAD): m.end() + CONTEXT_PAD]
                yield law, mtype, m.group(), " ".join(ctx.split())


def process_source(con, src_lc, sample_ids):
    src_uc = SOURCES[src_lc]
    docs = f"phase2/data/processed/{src_lc}/documents.parquet"
    pages = f"phase2/data/processed/{src_lc}/pages.parquet"

    # doc -> project map, restricted to projects that have a timeline row (so all are datable)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE docmap AS
        SELECT d.document_id, d.project_id.value AS project_id
        FROM read_parquet('{docs}') d
        WHERE d.project_id.value IN (
            SELECT project_id FROM read_parquet('{TIMELINE_PATH.as_posix()}')
            WHERE process_type = '{src_uc}'
        )
    """)
    if sample_ids is not None:
        ids = ",".join(f"'{i}'" for i in sample_ids)
        con.execute(f"DELETE FROM docmap WHERE project_id NOT IN ({ids})")

    cand = con.execute(f"""
        SELECT m.project_id, p.document_id, p.page_number, p.page_text
        FROM read_parquet('{pages}') p
        JOIN docmap m ON p.document_id = m.document_id
        WHERE length(p.page_text) > 40
          AND (regexp_matches(lower(p.page_text), '{PREFILTER_LOWER}')
               OR regexp_matches(p.page_text, '{PREFILTER_ACRONYM}'))
    """).fetchdf()
    print(f"  [{src_uc}] {len(cand):,} candidate pages to scan")

    # (project, law) -> aggregate
    agg = {}
    for row in cand.itertuples(index=False):
        for law, mtype, mtext, ctx in detect_in_page(row.page_text or ""):
            key = (row.project_id, law)
            rec = agg.get(key)
            if rec is None:
                agg[key] = {
                    "project_id": row.project_id, "process_type": src_uc, "law_name": law,
                    "citation_count": 1, "docs": {row.document_id},
                    "first_match_type": mtype, "first_context": ctx[:500],
                    "first_document_id": row.document_id, "first_page_number": row.page_number,
                }
            else:
                rec["citation_count"] += 1
                rec["docs"].add(row.document_id)

    out = []
    for rec in agg.values():
        rec["n_docs_matched"] = len(rec.pop("docs"))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ce", "ea", "eis", "all"], default="all")
    ap.add_argument("--sample", type=int, default=None,
                    help="limit to N random timeline projects (per chosen source) for smoke tests")
    args = ap.parse_args()

    run_at = datetime.now(timezone.utc).isoformat()
    con = duckdb.connect()
    srcs = list(SOURCES) if args.source == "all" else [args.source]

    all_rows = []
    for src in srcs:
        sample_ids = None
        if args.sample:
            sample_ids = con.execute(f"""
                SELECT project_id FROM read_parquet('{TIMELINE_PATH.as_posix()}')
                WHERE process_type = '{SOURCES[src]}' USING SAMPLE {args.sample} ROWS (reservoir, 42)
            """).fetchdf()["project_id"].tolist()
        print(f"Scanning {SOURCES[src]}...")
        all_rows.extend(process_source(con, src, sample_ids))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No citations detected.")
        return
    df["law_citations_extraction_run_at"] = run_at
    df = df[["project_id", "process_type", "law_name", "citation_count", "n_docs_matched",
             "first_match_type", "first_context", "first_document_id", "first_page_number",
             "law_citations_extraction_run_at"]]
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df):,} (project, law) rows to {OUTPUT_PATH}")

    print("\nProjects citing each law, by process:")
    print(df.groupby(["law_name", "process_type"])["project_id"].nunique()
          .unstack(fill_value=0).to_string())
    print("\n5 sample contexts per law (eyeball IRA/BIL disambiguation):")
    for law in df["law_name"].unique():
        print(f"\n--- {law} ---")
        for c in df[df.law_name == law]["first_context"].head(5):
            print("  •", c[:160])


if __name__ == "__main__":
    main()
