"""
Fetch individual DOE NEPA project pages from energy.gov for doc numbers that
did not get dates from the listing-page scrape in 10b.

URL pattern:
    EA:  https://www.energy.gov/nepa/listings/ea-{NNNN}-documents-available-download
    EIS: https://www.energy.gov/nepa/listings/eis-{NNNN}-documents-available-download

Page structure (text after stripping scripts/nav):
    {date}
    EA-{NNNN}: {doc_type_title}
    ...
    {date}
    EIS-{NNNN}: Record of Decision
    ...

Extracts: fonsi_date, rod_date, noi_date per doc number.

Reads:
    phase2/data/analysis/doe_register/doe_case_evidence.parquet
    phase2/data/analysis/doe_register/doe_register_records.parquet  (existing lookup)

Writes:
    phase2/data/analysis/doe_register/doe_project_page_cache.json   (raw cache)
    phase2/data/analysis/doe_register/doe_project_page_records.parquet
    phase2/data/analysis/doe_register/doe_register_records.parquet  (merged/updated)

Usage:
    python 10b2_fetch_project_pages.py
    python 10b2_fetch_project_pages.py --refetch
    python 10b2_fetch_project_pages.py --dry-run
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
DOE_DIR = ANALYSIS_DIR / "doe_register"
EVIDENCE_PATH = DOE_DIR / "doe_case_evidence.parquet"
EXISTING_RECORDS_PATH = DOE_DIR / "doe_register_records.parquet"
CACHE_PATH = DOE_DIR / "doe_project_page_cache.json"
PAGE_RECORDS_PATH = DOE_DIR / "doe_project_page_records.parquet"

BASE_URL = "https://www.energy.gov/nepa/listings"
RATE_LIMIT = 1.2

MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
          "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
          "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,"sep":9,
          "oct":10,"nov":11,"dec":12}


def _parse_date(raw: str) -> str | None:
    raw = raw.strip()
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s*(\d{4})", raw)
    if m:
        mon = MONTHS.get(m.group(1).lower())
        if mon:
            return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


# Title keywords → date field
FONSI_RE = re.compile(r"finding.{0,30}no significant|fonsi", re.IGNORECASE)
ROD_RE = re.compile(r"record of decision|amended record of decision", re.IGNORECASE)
NOI_RE = re.compile(r"notice of intent|notice of preparation|scoping", re.IGNORECASE)

# Project-page doc title: "EA-1658:", "EIS-0464:", "DOE/EIS-0269:", "DOE/EA-1870:"
DOC_TITLE_RE = re.compile(
    r"^(?:DOE/)?(EA|EIS)-(\d{4})(?:[-‐][\w\d-]*)?\s*(?:and\s+(?:DOE/)?(?:EA|EIS)-\d{4}(?:[-‐][\w\d-]*)?)?\s*:",
    re.IGNORECASE,
)


def _fetch_project_page(doc_number: str, session: requests.Session) -> dict:
    """
    Fetch energy.gov project page for a doc number.
    Returns dict with fonsi_date, rod_date, noi_date (all may be None).
    """
    m = re.match(r"DOE/(EA|EIS)-(\d{4})", doc_number, re.IGNORECASE)
    if not m:
        return {"fetch_status": "bad_format"}
    doc_type, num = m.group(1).upper(), m.group(2)
    url = f"{BASE_URL}/{doc_type.lower()}-{num}-documents-available-download"

    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 404:
            return {"fetch_status": "not_found", "url": url}
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"fetch_status": f"error: {str(e)[:60]}", "url": url}

    soup = BeautifulSoup(resp.text, "html.parser")
    for s in soup(["script", "style", "nav", "header", "footer"]): s.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    dates: dict[str, str] = {}

    for i, line in enumerate(lines):
        # Is this line a doc-type title (e.g. "EA-1658: Finding of No Significant Impact")?
        if not DOC_TITLE_RE.match(line):
            continue

        # Get the date from the preceding line
        date_str = None
        for j in range(i - 1, max(i - 4, -1), -1):
            candidate = lines[j].strip()
            parsed = _parse_date(candidate)
            if parsed:
                date_str = parsed
                break

        if not date_str:
            continue

        # Classify the doc type
        rest = line[DOC_TITLE_RE.match(line).end():].strip()
        if FONSI_RE.search(rest) and "fonsi_date" not in dates:
            dates["fonsi_date"] = date_str
        elif ROD_RE.search(rest) and "rod_date" not in dates:
            dates["rod_date"] = date_str
        elif NOI_RE.search(rest) and "noi_date" not in dates:
            dates["noi_date"] = date_str

    return {
        "fetch_status": "ok",
        "url": url,
        "fonsi_date": dates.get("fonsi_date"),
        "rod_date": dates.get("rod_date"),
        "noi_date": dates.get("noi_date"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true",
                        help="Re-fetch already-cached doc numbers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load evidence and find doc numbers without dates in existing records
    evidence = pd.read_parquet(EVIDENCE_PATH)
    accepted = evidence[evidence["acceptance"] == "accept"].copy()
    accepted_nums = set(accepted["doc_number"].str.upper())

    existing = pd.read_parquet(EXISTING_RECORDS_PATH) if EXISTING_RECORDS_PATH.exists() else pd.DataFrame()
    if not existing.empty:
        has_date = (
            existing["rod_date"].notna()
            | existing["fonsi_date"].notna()
            | (existing["noi_date"].notna() if "noi_date" in existing.columns else False)
        )
        dated_nums = set(existing[has_date]["doc_number"].str.upper())
    else:
        dated_nums = set()

    # Fetch unmatched doc numbers (accepted but no date in existing records)
    to_fetch = sorted(accepted_nums - dated_nums)
    print(f"Accepted doc numbers: {len(accepted_nums)}")
    print(f"Already have dates for: {len(dated_nums)}")
    print(f"To fetch via project pages: {len(to_fetch)}")

    # Load or init cache
    cache: dict = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    if args.refetch:
        for num in to_fetch:
            cache.pop(num, None)
        print(f"Cleared {len(to_fetch)} entries for re-fetch.")

    pending = [n for n in to_fetch if n not in cache]
    print(f"  {len(to_fetch) - len(pending)} already cached, {len(pending)} to fetch")

    if args.dry_run:
        print("DRY RUN — sample:")
        for n in pending[:10]: print(f"  {n}")
        return

    if pending:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; NEPA-academic-research/1.0)"
        })

        for i, doc_num in enumerate(pending, 1):
            result = _fetch_project_page(doc_num, session)
            result["doc_number"] = doc_num
            cache[doc_num] = result

            status = result.get("fetch_status", "?")
            dates_found = [f for f in ["fonsi_date", "rod_date", "noi_date"]
                           if result.get(f)]
            print(f"  [{i}/{len(pending)}] {doc_num} ... {status}"
                  + (f" | {', '.join(f+'='+result[f] for f in dates_found)}" if dates_found else ""))

            if i % 25 == 0:
                CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))

            time.sleep(RATE_LIMIT)

        CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))

    # Build records from all cached entries
    records = [v for k, v in cache.items() if k in accepted_nums]
    if not records:
        print("No records to write.")
        return

    page_df = pd.DataFrame(records)
    page_df = page_df[page_df.get("fetch_status", pd.Series(["ok"] * len(page_df))) == "ok"].copy()
    for col in ["fonsi_date", "rod_date", "noi_date"]:
        if col not in page_df.columns:
            page_df[col] = None

    page_df.to_parquet(PAGE_RECORDS_PATH, index=False)

    with_any_date = page_df[
        page_df["fonsi_date"].notna() | page_df["rod_date"].notna() | page_df["noi_date"].notna()
    ]
    print(f"\nPage records written: {len(page_df)} fetched ok, "
          f"{len(with_any_date)} with at least one date")
    print(f"  fonsi_date: {page_df['fonsi_date'].notna().sum()}")
    print(f"  rod_date:   {page_df['rod_date'].notna().sum()}")
    print(f"  noi_date:   {page_df['noi_date'].notna().sum()}")

    # Merge into existing doe_register_records
    if existing.empty:
        merged = page_df
    else:
        # Combine: page records supplement the listing records
        # For doc numbers already in existing (with dates), keep existing
        # For new ones, add page records
        new_nums = set(page_df["doc_number"]) - set(existing["doc_number"])
        new_rows = page_df[page_df["doc_number"].isin(new_nums)]
        # Also patch in dates for existing rows that had no date
        no_date_existing = existing[
            existing["rod_date"].isna()
            & existing["fonsi_date"].isna()
            & (existing["noi_date"].isna() if "noi_date" in existing.columns else True)
        ]["doc_number"]
        patch_rows = page_df[page_df["doc_number"].isin(no_date_existing)]
        for _, pr in patch_rows.iterrows():
            mask = existing["doc_number"] == pr["doc_number"]
            for col in ["fonsi_date", "rod_date", "noi_date"]:
                if col in pr and pd.notna(pr[col]):
                    existing.loc[mask, col] = pr[col]

        # Ensure noi_date column exists in existing
        if "noi_date" not in existing.columns:
            existing["noi_date"] = None

        merged = pd.concat([existing, new_rows[
            [c for c in new_rows.columns if c in existing.columns or c == "doc_number"]
        ]], ignore_index=True)

    merged.to_parquet(EXISTING_RECORDS_PATH, index=False)
    print(f"\nUpdated doe_register_records.parquet: {len(merged)} total rows")
    with_any = merged[
        merged["rod_date"].notna()
        | merged["fonsi_date"].notna()
        | (merged["noi_date"].notna() if "noi_date" in merged.columns else False)
    ]
    print(f"  With any date: {len(with_any)} / {len(merged)}")


if __name__ == "__main__":
    main()
