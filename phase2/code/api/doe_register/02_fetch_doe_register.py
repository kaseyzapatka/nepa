"""
Build DOE NEPA date lookup tables from energy.gov listing pages.

Three sources:
  1. energy.gov/nepa/listings/records-decision-rod      → EIS ROD dates
  2. energy.gov/nepa/listings/findings-no-significant-impact-fonsi → EA FONSI dates
  3. EPA EIS database (cdxapps.epa.gov)                 → EIS NOI/initiation dates

All sources are scraped once and cached as parquet lookup tables.
Re-run to refresh (no per-record cache needed — full scrape is fast).

Reads:
    phase2/data/analysis/doe_register/doe_case_evidence.parquet  (for doc numbers to look up)

Writes:
    phase2/data/analysis/doe_register/doe_rod_lookup.parquet     (EIS → ROD date)
    phase2/data/analysis/doe_register/doe_fonsi_lookup.parquet   (EA → FONSI date)
    phase2/data/analysis/doe_register/epa_eis_noi_dates.parquet  (EIS → NOI date)
    phase2/data/analysis/doe_register/doe_register_records.parquet  (combined per doc_number)

Usage:
    python 10b_fetch_doe_register.py
    python 10b_fetch_doe_register.py --sources rod fonsi
    python 10b_fetch_doe_register.py --dry-run
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
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
DOE_DIR = ANALYSIS_DIR / "doe_register"
EVIDENCE_PATH = DOE_DIR / "doe_case_evidence.parquet"

BASE_URL = "https://www.energy.gov/nepa/listings"
EPA_EIS_URL = "https://cdxapps.epa.gov/cdx-enepa-II/public/action/eis/search"

RATE_LIMIT = 1.0  # seconds between page fetches

DOE_DOC_RE = re.compile(r"DOE/(EIS-\d{4}(?:[-‐-―](?:S\d+|SA[-\s]?\d+))?|EA-\d{4})", re.IGNORECASE)
DATE_RE = re.compile(r"\b(\w+ \d{1,2},\s*\d{4}|\d{1,2}/\d{1,2}/\d{4})\b")
MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
          "july":7,"august":8,"september":9,"october":10,"november":11,"december":12}


def _parse_date(raw: str) -> str | None:
    raw = raw.strip()
    # "January 23, 2025" or "Jan 23, 2025"
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s*(\d{4})", raw)
    if m:
        mon = MONTHS.get(m.group(1).lower())
        if mon:
            return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"
    # MM/DD/YYYY
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def _normalize_doc_number(raw: str) -> str:
    n = raw.upper()
    n = re.sub(r"[‐-―]", "-", n)
    n = re.sub(r"(SA-)\s+(\d)", r"\1\2", n)
    return n


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; NEPA-academic-research/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    })
    return session


def _scrape_listing_page(
    url: str,
    session: requests.Session,
    doc_type_filter: str,   # "EA" or "EIS"
    date_field: str,        # "fonsi_date" or "rod_date"
) -> list[dict]:
    """
    Scrape one page of a DOE NEPA listing. Returns list of {doc_number, date_field, title}.

    Page structure (line-delimited text after stripping scripts):
      {date_str}
      {DOE/EIS-NNNN or DOE/EA-NNNN}: {title}
      {description ...}
      {next date_str}
      ...
    The date line immediately PRECEDES the doc-number title line.
    """
    resp = session.get(url, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for s in soup(["script", "style", "nav", "header", "footer"]): s.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    records = []
    for i, line in enumerate(lines):
        # Find lines that contain a DOE document number
        nums = DOE_DOC_RE.findall(line)
        if not nums:
            continue
        # Filter to the right doc type
        nums = [n for n in nums if n.upper().startswith(doc_type_filter)]
        if not nums:
            continue

        # The date is on the line immediately before the doc number line
        date_str = None
        for j in range(i - 1, max(i - 4, -1), -1):
            candidate = lines[j].strip()
            if DATE_RE.search(candidate) and len(candidate) < 50:
                date_str = candidate
                break

        parsed_date = _parse_date(date_str) if date_str else None

        for num in nums:
            normalized = _normalize_doc_number(f"DOE/{num}")
            records.append({
                "doc_number": normalized,
                date_field: parsed_date,
                "title": line[:200],
                "date_raw": date_str,
            })

    return records


def scrape_listing(
    path_suffix: str,
    doc_type_filter: str,
    date_field: str,
    session: requests.Session,
    dry_run: bool = False,
    max_pages: int = 200,
) -> pd.DataFrame:
    """Scrape all pages of an energy.gov listing."""
    base = f"{BASE_URL}/{path_suffix}"
    all_records = []

    # Discover total pages from page 0
    resp = session.get(f"{base}?page=0", timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    pager = soup.select("a[href*='?page=']")
    page_nums = [int(re.search(r"page=(\d+)", a["href"]).group(1))
                 for a in pager if re.search(r"page=(\d+)", a.get("href", ""))]
    total_pages = max(page_nums) + 1 if page_nums else 1
    total_pages = min(total_pages, max_pages)

    print(f"  Scraping {path_suffix} ({total_pages} pages) ...")

    if dry_run:
        print(f"  DRY RUN — would scrape {total_pages} pages of {path_suffix}")
        return pd.DataFrame()

    for page in range(total_pages):
        url = f"{base}?page={page}"
        try:
            recs = _scrape_listing_page(url, session, doc_type_filter, date_field)
            all_records.extend(recs)
            if page % 10 == 0:
                print(f"    page {page}/{total_pages-1}: {len(recs)} records "
                      f"(total so far: {len(all_records)})")
        except requests.exceptions.RequestException as e:
            print(f"    page {page} error: {e} — skipping")
        time.sleep(RATE_LIMIT)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Deduplicate: if a doc number appears multiple times, take earliest date
    df = df.dropna(subset=[date_field])
    df = df.sort_values(date_field)
    df = df.drop_duplicates(subset=["doc_number"], keep="first")
    return df.reset_index(drop=True)


def scrape_epa_eis_noi(session: requests.Session, dry_run: bool = False) -> pd.DataFrame:
    """
    Scrape DOE EIS NOI dates from the EPA EIS database.
    POST to search form with Agency=DOE, paginate by date range.
    Returns df with doc_number, noi_date, project_title.
    """
    print("  Scraping EPA EIS database for DOE NOI dates ...")
    if dry_run:
        print("  DRY RUN — would scrape EPA EIS database")
        return pd.DataFrame()

    all_records = []
    # EPA EIS search: chunk by decade to stay under 500-record cap
    year_ranges = [
        ("01/01/1990", "12/31/2000"),
        ("01/01/2001", "12/31/2010"),
        ("01/01/2011", "12/31/2018"),
        ("01/01/2019", "12/31/2026"),
    ]

    for start_date, end_date in year_ranges:
        payload = {
            "action": "search",
            "agency": "DOE",
            "startDate": start_date,
            "endDate": end_date,
            "eisType": "",
            "title": "",
            "state": "",
            "county": "",
        }
        try:
            resp = session.post(EPA_EIS_URL, data=payload, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # EPA EIS search results table
            table = soup.find("table", id="searchResultsTable") or soup.find("table")
            if not table:
                print(f"    {start_date}–{end_date}: no table found")
                continue

            rows = table.find_all("tr")[1:]  # skip header
            count = 0
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 5:
                    continue
                # Typical columns: EIS Number, Title, Agency, FR Date, etc.
                eis_num = cells[0].get_text(strip=True)
                title = cells[1].get_text(strip=True)
                fr_date_raw = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                # EIS number might be "0391" or "DOE/EIS-0391"
                if re.match(r"^\d{4}$", eis_num):
                    doc_number = f"DOE/EIS-{eis_num}"
                elif re.match(r"DOE/EIS-", eis_num, re.IGNORECASE):
                    doc_number = eis_num.upper()
                else:
                    continue

                parsed_date = _parse_date(fr_date_raw)
                if parsed_date:
                    all_records.append({
                        "doc_number": doc_number,
                        "noi_date": parsed_date,
                        "title": title[:200],
                        "date_raw": fr_date_raw,
                    })
                    count += 1

            print(f"    {start_date}–{end_date}: {count} records")
        except Exception as e:
            print(f"    {start_date}–{end_date}: error — {e}")
        time.sleep(RATE_LIMIT)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.sort_values("noi_date")
    df = df.drop_duplicates(subset=["doc_number"], keep="first")
    return df.reset_index(drop=True)


def build_combined_records(
    rod_df: pd.DataFrame,
    fonsi_df: pd.DataFrame,
    noi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all lookup tables into one doc_number-keyed records table."""
    # Collect all known doc numbers
    all_nums = set()
    for df in [rod_df, fonsi_df, noi_df]:
        if not df.empty and "doc_number" in df.columns:
            all_nums.update(df["doc_number"].tolist())

    if not all_nums:
        return pd.DataFrame()

    base = pd.DataFrame({"doc_number": sorted(all_nums)})

    if not rod_df.empty and "rod_date" in rod_df.columns:
        base = base.merge(rod_df[["doc_number", "rod_date"]].drop_duplicates("doc_number"),
                         on="doc_number", how="left")
    else:
        base["rod_date"] = None

    if not fonsi_df.empty and "fonsi_date" in fonsi_df.columns:
        base = base.merge(fonsi_df[["doc_number", "fonsi_date"]].drop_duplicates("doc_number"),
                         on="doc_number", how="left")
    else:
        base["fonsi_date"] = None

    if not noi_df.empty and "noi_date" in noi_df.columns:
        base = base.merge(noi_df[["doc_number", "noi_date"]].drop_duplicates("doc_number"),
                         on="doc_number", how="left")
    else:
        base["noi_date"] = None

    base["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+",
                        choices=["rod", "fonsi", "noi"],
                        default=["rod", "fonsi", "noi"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    DOE_DIR.mkdir(parents=True, exist_ok=True)
    session = _get_session()
    fetched_at = datetime.now(timezone.utc).isoformat()

    rod_df = pd.DataFrame()
    fonsi_df = pd.DataFrame()
    noi_df = pd.DataFrame()

    if "rod" in args.sources:
        print("\n=== Scraping ROD listing (EIS → rod_date) ===")
        rod_df = scrape_listing(
            "records-decision-rod", "EIS", "rod_date", session, args.dry_run
        )
        if not rod_df.empty and not args.dry_run:
            out = DOE_DIR / "doe_rod_lookup.parquet"
            rod_df.to_parquet(out, index=False)
            print(f"  Wrote {len(rod_df)} ROD records → {out}")
            print(f"  Sample: {rod_df[['doc_number','rod_date']].head(5).to_string(index=False)}")

    if "fonsi" in args.sources:
        print("\n=== Scraping FONSI listing (EA → fonsi_date) ===")
        fonsi_df = scrape_listing(
            "findings-no-significant-impact-fonsi", "EA", "fonsi_date", session, args.dry_run
        )
        if not fonsi_df.empty and not args.dry_run:
            out = DOE_DIR / "doe_fonsi_lookup.parquet"
            fonsi_df.to_parquet(out, index=False)
            print(f"  Wrote {len(fonsi_df)} FONSI records → {out}")
            print(f"  Sample: {fonsi_df[['doc_number','fonsi_date']].head(5).to_string(index=False)}")

    if "noi" in args.sources:
        print("\n=== Scraping EPA EIS database (EIS → noi_date) ===")
        noi_df = scrape_epa_eis_noi(session, args.dry_run)
        if not noi_df.empty and not args.dry_run:
            out = DOE_DIR / "epa_eis_noi_dates.parquet"
            noi_df.to_parquet(out, index=False)
            print(f"  Wrote {len(noi_df)} NOI records → {out}")

    if args.dry_run:
        return

    # Build combined records table
    combined = build_combined_records(rod_df, fonsi_df, noi_df)
    if not combined.empty:
        out = DOE_DIR / "doe_register_records.parquet"
        combined.to_parquet(out, index=False)
        print(f"\nWrote {len(combined)} combined records → {out}")
        print("\nCoverage:")
        for col in ["rod_date", "fonsi_date", "noi_date"]:
            if col in combined.columns:
                n = combined[col].notna().sum()
                print(f"  {col}: {n} / {len(combined)}")


if __name__ == "__main__":
    main()
