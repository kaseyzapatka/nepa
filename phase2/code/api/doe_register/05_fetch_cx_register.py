"""
Scrape energy.gov CX (Categorical Exclusion) listing pages and build a
cx_number → date / office / location lookup table.

Source:
    https://www.energy.gov/nepa/listings/categorical-exclusion-cx-determinations-date
    ~3,558 pages, 10 records each, ~35,580 CX determinations total.

Each listing record contains:
    cx_number   — integer extracted from article URL (cx-NNNNNN)
    cx_date     — ISO-8601 determination date (always present)
    cx_date_raw — raw display date string ("December 2, 2014")
    office      — DOE office / operations office (when present)
    location    — state / region (when present)
    cx_codes    — CE codes applied, e.g. "B3.6, A9" (when present)
    cx_title    — project description from listing summary

Writes:
    phase2/data/analysis/doe_register/doe_cx_register.parquet

Usage:
    python 05_fetch_cx_register.py               # full crawl (~60 min)
    python 05_fetch_cx_register.py --sample 50   # first 50 pages (~8 min)
    python 05_fetch_cx_register.py --dry-run     # discover page count only
    python 05_fetch_cx_register.py --refetch     # overwrite existing output

Rate: 1 request/second (3,558 total requests). Re-running without --refetch
skips the crawl if output already exists.
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
DOE_DIR = PHASE2 / "data" / "analysis" / "doe_register"
OUTPUT_PATH = DOE_DIR / "doe_cx_register.parquet"

LISTING_URL = (
    "https://www.energy.gov/nepa/listings/"
    "categorical-exclusion-cx-determinations-date"
)
RATE_LIMIT = 1.0          # seconds between page fetches
CHECKPOINT_EVERY = 500    # pages between partial saves

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_display_date(raw: str) -> str | None:
    """Parse 'December 2, 2014' or '12/02/2014' → '2014-12-02'."""
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


def _parse_article(article) -> dict | None:
    """Extract one CX record from a <article typeof='schema:Article'> element."""
    about = article.get("about", "")
    cx_m = re.search(r"/cx-(\d+)", about)
    if not cx_m:
        # Fallback: pull from title text "CX-NNNNNN ..."
        title_el = article.find(class_="listing-item__title")
        title_txt = title_el.get_text(strip=True) if title_el else ""
        cx_m = re.match(r"CX-(\d+)", title_txt, re.IGNORECASE)
    if not cx_m:
        return None

    cx_number = int(cx_m.group(1))

    # Date — always in .listing-item__date
    date_el = article.find(class_="listing-item__date")
    cx_date_raw = date_el.get_text(strip=True) if date_el else ""
    cx_date = _parse_display_date(cx_date_raw) if cx_date_raw else None

    # Project description — .listing-item__summary
    summary_el = article.find(class_="listing-item__summary")
    cx_title = summary_el.get_text(strip=True) if summary_el else ""

    # Structured fields embedded in raw article text
    raw_text = article.get_text(separator=" ", strip=True)

    office = None
    location = None
    cx_codes = None

    loc_m = re.search(
        r"Location\(s\):\s*(.+?)(?:\s*Offices?\(s\):|$)", raw_text
    )
    if loc_m:
        location = loc_m.group(1).strip() or None

    off_m = re.search(r"Offices?\(s\):\s*(.+?)(?:\s{3,}|$)", raw_text)
    if off_m:
        office = off_m.group(1).strip() or None

    codes_m = re.search(
        r"CX\(s\)\s+Applied:\s*([A-Z0-9.,\s]+?)(?:Date:|Location|$)", raw_text
    )
    if codes_m:
        cx_codes = re.sub(r"\s+", " ", codes_m.group(1)).strip() or None

    return {
        "cx_number": cx_number,
        "cx_date": cx_date,
        "cx_date_raw": cx_date_raw,
        "office": office,
        "location": location,
        "cx_codes": cx_codes,
        "cx_title": cx_title[:400] if cx_title else None,
    }


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; NEPA-academic-research/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    })
    return session


def _discover_total_pages(session: requests.Session) -> int:
    resp = session.get(f"{LISTING_URL}?page=0", timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    page_nums = [
        int(re.search(r"page=(\d+)", a["href"]).group(1))
        for a in soup.select("a[href*='?page=']")
        if re.search(r"page=(\d+)", a.get("href", ""))
    ]
    return max(page_nums) + 1 if page_nums else 1


def scrape_all_pages(
    session: requests.Session,
    total_pages: int,
    sample: int | None,
) -> list[dict]:
    pages_to_fetch = min(total_pages, sample) if sample else total_pages
    all_records: list[dict] = []
    checkpoint_records: list[dict] = []

    fetched_at = datetime.now(timezone.utc).isoformat()

    for page in range(pages_to_fetch):
        url = f"{LISTING_URL}?page={page}"
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            articles = soup.find_all(attrs={"typeof": "schema:Article"})
            page_records = []
            for a in articles:
                rec = _parse_article(a)
                if rec:
                    rec["fetched_at"] = fetched_at
                    page_records.append(rec)
            all_records.extend(page_records)
            checkpoint_records.extend(page_records)

            if page % 100 == 0 or page == pages_to_fetch - 1:
                print(
                    f"  page {page:>4}/{pages_to_fetch - 1}: "
                    f"{len(page_records)} records  "
                    f"(total: {len(all_records):,})"
                )

            # Periodic checkpoint save
            if len(checkpoint_records) >= CHECKPOINT_EVERY * 10:
                _save_checkpoint(all_records)
                checkpoint_records.clear()

        except requests.exceptions.RequestException as e:
            print(f"  page {page}: ERROR {e} — skipping")

        time.sleep(RATE_LIMIT)

    return all_records


def _save_checkpoint(records: list[dict]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  [checkpoint] saved {len(df):,} rows → {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Fetch only first N pages (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover page count and exit without fetching",
    )
    parser.add_argument(
        "--refetch", action="store_true",
        help="Re-crawl even if output already exists",
    )
    args = parser.parse_args()

    DOE_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not args.refetch and not args.dry_run:
        existing = pd.read_parquet(OUTPUT_PATH)
        print(
            f"Output already exists: {len(existing):,} rows in {OUTPUT_PATH}\n"
            f"Use --refetch to re-crawl."
        )
        return

    session = _get_session()

    print("Discovering total pages ...")
    total_pages = _discover_total_pages(session)
    time.sleep(RATE_LIMIT)
    print(f"  Total pages: {total_pages:,}  (~{total_pages * 10:,} records)")

    if args.dry_run:
        print("DRY RUN — exiting without fetch.")
        return

    pages_label = f"{min(total_pages, args.sample)}" if args.sample else f"{total_pages}"
    print(f"\nScraping {pages_label} pages at {RATE_LIMIT}s/request ...")
    est_min = int(min(total_pages, args.sample or total_pages) * RATE_LIMIT / 60)
    print(f"Estimated time: ~{est_min} minutes\n")

    records = scrape_all_pages(session, total_pages, args.sample)

    if not records:
        print("No records collected — check network or page structure.")
        return

    df = pd.DataFrame(records)

    # Deduplicate: keep earliest date if a CX number appears more than once
    before = len(df)
    df = df.sort_values("cx_date", na_position="last")
    df = df.drop_duplicates(subset=["cx_number"], keep="first")
    df = df.sort_values("cx_number").reset_index(drop=True)
    if before != len(df):
        print(f"  Deduplication: {before:,} → {len(df):,} rows")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows → {OUTPUT_PATH}")

    # Coverage summary
    print("\nField coverage:")
    for col in ["cx_date", "office", "location", "cx_codes", "cx_title"]:
        n = df[col].notna().sum()
        print(f"  {col:12s}: {n:>6,} / {len(df):,}  ({100*n/len(df):.1f}%)")

    print("\nDate range:")
    dated = df["cx_date"].dropna().sort_values()
    print(f"  Earliest: {dated.iloc[0]}")
    print(f"  Latest:   {dated.iloc[-1]}")

    print("\nTop 10 offices:")
    print(df["office"].value_counts().head(10).to_string())

    print("\nSample records:")
    sample_cols = ["cx_number", "cx_date", "office", "location", "cx_title"]
    print(df[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
