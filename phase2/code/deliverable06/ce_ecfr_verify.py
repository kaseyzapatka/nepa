"""D6 A1/#37 — eCFR verification scaffold for CE adopt/expand matches (top-5 coverage adjudication).

The report calls this "the one remaining verification step": every adopt/expand verdict rests on a
TEXT-SIMILARITY match to an existing CE and is pending confirmation against the CURRENT eCFR text.
This script builds the adjudication apparatus:

  1. For the 24 adopt/expand cells, pull the top-5 retrieved CEs (candidate_ce_comparison, ranks 1-5)
     with their canonical_source_url + parsed numeric bounds.
  2. Fetch the canonical eCFR text per CE (deterministic, $0) via the eCFR renderer API, cached to
     data/raw/deliverable06/ecfr/. URLs that are NOT eCFR (agency NEPA-procedure PDFs) or legacy
     cgi-bin nodes are flagged source_type != 'ecfr_current' and left for manual fetch — itself a
     finding (that CE is not in the eCFR).
  3. Write candidate_ce_coverage.parquet with an EMPTY coverage_verdict per (cell, rank) for a reviewer
     (or the optional --llm pass) to fill: covers / partially_covers / does_not_cover / unclear.
  4. Render a human worksheet (phase2/notes/deliverable06/ce_ecfr_verification.md).

07_classify_and_rank.py gates on this file when present (best adjudicated-covering CE among ranks 1-5)
and otherwise falls back to top-1 — so this is an existence-guarded pre-07 step.

The eCFR fetch is $0. The optional LLM adjudication (--llm) is BILLABLE and USER-LAUNCHED; --llm --dry-run
prints the cost and stops. --dry-run / default never touches the Keychain.

USAGE
  python ce_ecfr_verify.py                 # fetch eCFR text + build coverage parquet + worksheet ($0)
  python ce_ecfr_verify.py --llm --dry-run # + projected cost of the LLM adjudication, no key
"""
from __future__ import annotations

import argparse
import os
import re
import time
from urllib.parse import urlparse

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import duckdb
import pandas as pd
import requests

from common import (
    D6_ANALYSIS_DIR, D6_RAW_DIR, ensure_d6_dirs, sha256_text, utc_now, write_parquet,
)
import enrich_lib

VERDICTS = D6_ANALYSIS_DIR / "candidate_verdicts.parquet"
COMPARISON = D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet"
COVERAGE_OUT = D6_ANALYSIS_DIR / "candidate_ce_coverage.parquet"
ECFR_CACHE = D6_RAW_DIR / "ecfr"
WORKSHEET = D6_ANALYSIS_DIR.parents[2] / "notes" / "deliverable06" / "ce_ecfr_verification.md"

API = "https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-{title}"
COVERAGE_VALUES = ["covers", "partially_covers", "does_not_cover", "unclear"]


def classify_url(url: str) -> str:
    if not url:
        return "none"
    host = urlparse(url).netloc.lower()
    if "ecfr.gov" not in host:
        return "agency_doc"                       # CE lives in an agency NEPA-procedures doc, not the eCFR
    if "/current/title-" in url:
        return "ecfr_current"
    return "ecfr_legacy"                          # cgi-bin/text-idx node — manual fetch


def parse_ecfr_params(url: str) -> dict | None:
    """/current/title-18/chapter-I/subchapter-W/part-380/section-380.4 -> API query params."""
    m = re.search(r"/current/title-(\d+[A-Za-z]?)", url)
    if not m:
        return None
    params = {"title": m.group(1)}
    for key in ("chapter", "subchapter", "part", "subpart", "section", "appendix"):
        mm = re.search(rf"/{key}-([^/?#]+)", url)
        if mm:
            params[key] = mm.group(1)
    return params


def fetch_ecfr_text(url: str, session: requests.Session) -> tuple[str, str]:
    """Return (text, cache_sha). Cached under D6_RAW_DIR/ecfr/. Best-effort; '' on failure."""
    ECFR_CACHE.mkdir(parents=True, exist_ok=True)
    key = sha256_text(url)
    cached = ECFR_CACHE / f"{key}.txt"
    if cached.exists():
        return cached.read_text(), key
    params = parse_ecfr_params(url)
    if not params:
        return "", key
    title = params.pop("title")
    try:
        r = session.get(API.format(title=title), params=params, timeout=25)
        if r.status_code != 200:
            return "", key
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception:
        return "", key
    cached.write_text(text)
    time.sleep(0.4)                               # be polite to the eCFR API
    return text, key


def main() -> None:
    ap = argparse.ArgumentParser(description="D6 A1/#37 eCFR verification scaffold + fetch.")
    ap.add_argument("--llm", action="store_true", help="(with --dry-run) project the LLM adjudication cost")
    ap.add_argument("--dry-run", action="store_true", help="no LLM call, no key")
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()
    ensure_d6_dirs()

    con = duckdb.connect()
    cells = [r[0] for r in con.execute(
        f"SELECT candidate_category FROM '{VERDICTS}' WHERE verdict IN ('adopt','expand')").fetchall()]
    verdict_of = {r[0]: r[1] for r in con.execute(
        f"SELECT candidate_category, verdict FROM '{VERDICTS}'").fetchall()}
    inlist = ",".join("'" + c + "'" for c in cells)
    cmp = con.execute(f"""
        SELECT candidate_category, retrieval_rank, structured_id, agency_name, agency_unit,
               ce_description, canonical_source_url, retrieval_score,
               bound_acres, bound_miles, bound_mw, bound_kv, bound_wells
        FROM '{COMPARISON}'
        WHERE candidate_category IN ({inlist}) AND retrieval_rank <= 5
        ORDER BY candidate_category, retrieval_rank
    """).df()
    print(f"[ecfr] {len(cells)} adopt/expand cells, {len(cmp)} top-5 CE rows")

    run_at = utc_now()
    session = requests.Session()
    session.headers.update({"User-Agent": "nepa-d6-ce-verify/1.0"})
    rows = []
    fetched = fetch_ok = 0
    for r in cmp.itertuples():
        url = str(r.canonical_source_url or "")
        stype = classify_url(url)
        text, sha = "", ""
        if stype == "ecfr_current":
            fetched += 1
            text, sha = fetch_ecfr_text(url, session)
            if text:
                fetch_ok += 1
        bounds = {m: getattr(r, f"bound_{m}") for m in ("acres", "miles", "mw", "kv", "wells")}
        bound_str = "; ".join(f"{k}={v}" for k, v in bounds.items() if pd.notna(v)) or ""
        rows.append({
            "candidate_category": r.candidate_category,
            "verdict": verdict_of.get(r.candidate_category, ""),
            "retrieval_rank": int(r.retrieval_rank),
            "structured_id": r.structured_id,
            "agency_name": r.agency_name,
            "agency_unit": r.agency_unit,
            "retrieval_score": round(float(r.retrieval_score or 0), 4),
            "ce_description": str(r.ce_description or "")[:600],
            "canonical_source_url": url,
            "source_type": stype,
            "ecfr_text_chars": len(text),
            "ecfr_text_sha256": sha,
            "parsed_bounds": bound_str,
            "coverage_verdict": "",            # reviewer/LLM fills: covers/partially_covers/does_not_cover/unclear
            "bound_confirmed": "",             # reviewer: yes/no/na
            "reviewer_notes": "",
            "ce_coverage_extraction_run_at": run_at,
            "ce_coverage_llm_run_at": "",
        })
    cov = pd.DataFrame(rows)
    write_parquet(cov, COVERAGE_OUT)
    print(f"[ecfr] fetched {fetch_ok}/{fetched} eCFR-current CE texts "
          f"(source types: {cov['source_type'].value_counts().to_dict()})")
    print(f"[ecfr] coverage scaffold -> {COVERAGE_OUT}")

    # --- human worksheet ---
    lines = ["---", 'title: "Deliverable 6 — eCFR verification of CE adopt/expand matches"', "---", "",
             "Every adopt/expand verdict rests on a *text-similarity* match to an existing CE, pending "
             "confirmation against the **current eCFR text**. Fill `coverage_verdict` "
             f"({' / '.join(COVERAGE_VALUES)}) for the best-covering CE per cell.", ""]
    for cat, g in cov.groupby("candidate_category"):
        lines.append(f"## {cat} — {verdict_of.get(cat,'')}")
        best = g.sort_values("retrieval_rank")
        for rr in best.itertuples():
            src = {"ecfr_current": "eCFR", "ecfr_legacy": "eCFR (legacy URL — fetch manually)",
                   "agency_doc": "AGENCY DOC — not in eCFR", "none": "no URL"}[rr.source_type]
            lines.append(f"- **rank {rr.retrieval_rank}** `{rr.structured_id}` ({rr.agency_name}) "
                         f"score {rr.retrieval_score} — {src}")
            if rr.parsed_bounds:
                lines.append(f"  - bounds to confirm: {rr.parsed_bounds}")
            lines.append(f"  - [source]({rr.canonical_source_url})  ·  fetched eCFR text: "
                         f"{rr.ecfr_text_chars} chars")
        lines.append("")
    WORKSHEET.parent.mkdir(parents=True, exist_ok=True)
    WORKSHEET.write_text("\n".join(lines) + "\n")
    print(f"[ecfr] worksheet -> {WORKSHEET}")

    if args.llm and args.dry_run:
        n = int((cov["source_type"] == "ecfr_current").sum())      # adjudicable rows with fetched text
        in_rate, out_rate = enrich_lib.pricing_for(args.model)
        in_tok = sum(1400 + len(str(t)) // 4 for t in cov.loc[cov["source_type"] == "ecfr_current", "ce_description"])
        out_tok = 120 * n
        cost = in_tok / 1e6 * in_rate + out_tok / 1e6 * out_rate
        print(f"\n[ecfr] --llm --dry-run: adjudicating {n} eCFR-current CE rows with {args.model} "
              f"(~{in_tok:,} in / {out_tok:,} out tok) -> EXACT PROJECTED COST ${cost:.2f} (BILLABLE, user-launched)")
    elif not args.llm:
        print("\n[ecfr] Human path ($0, finishes today): fill coverage_verdict in the worksheet / parquet.")
        print("[ecfr] LLM path (billable): `python ce_ecfr_verify.py --llm --dry-run` for the cost.")


if __name__ == "__main__":
    main()
