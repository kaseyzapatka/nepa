"""
Fetch BLM NEPA Register records for the case numbers found by 09a.

Two-step process (no general scraping — only the case numbers we actually have):
  Step A: POST /searchresults/?search_bar={case_number}  → JSON → D365 project ID
  Step B: GET  /Project-Home/?id={projectid}             → HTML → Start/FONSI/ROD dates

CSRF token is fetched once per session from /_layout/tokenhtml.
Responses are disk-cached — reruns skip already-fetched case numbers.

Reads:
    phase2/data/analysis/blm_register/nepatec_case_evidence.parquet

Writes:
    phase2/data/analysis/blm_register/blm_register_cache.json   (raw fetch cache)
    phase2/data/analysis/blm_register/blm_register_records.parquet

Usage:
    python 09b_fetch_blm_register.py
    python 09b_fetch_blm_register.py --acceptance accept review
    python 09b_fetch_blm_register.py --dry-run
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
BLM_DIR = ANALYSIS_DIR / "blm_register"
EVIDENCE_PATH = BLM_DIR / "nepatec_case_evidence.parquet"
CACHE_PATH = BLM_DIR / "blm_register_cache.json"
OUTPUT_PATH = BLM_DIR / "blm_register_records.parquet"

BASE_URL = "https://eplanning.blm.gov"
TOKEN_URL = f"{BASE_URL}/_layout/tokenhtml"
SEARCH_URL = f"{BASE_URL}/searchresults/"
PROJECT_URL = f"{BASE_URL}/Project-Home/"

RATE_LIMIT_SECS = 1.5  # between case number fetches

DATE_FIELD_RE = re.compile(
    r"(start\s+date|fonsi\s+date|rod\s+date|decision\s+date|noi\s+date|"
    r"notice\s+of\s+intent\s+date)\s*\n\s*(\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    if ISO_RE.match(raw):
        return raw
    return None


def _get_token(session: requests.Session) -> str:
    """Fetch a fresh CSRF token."""
    resp = session.get(TOKEN_URL, timeout=15)
    resp.raise_for_status()
    m = re.search(r'value="([^"]{20,})"', resp.text)
    if not m:
        raise RuntimeError("Could not extract CSRF token from tokenhtml endpoint.")
    return m.group(1)


def _search_case_number(
    case_number: str,
    session: requests.Session,
    token: str,
) -> dict | None:
    """POST to /searchresults/ and return the first matching record or None."""
    payload = {
        "search_bar": case_number,
        "download": "false",
        "get_total_count": "false",
        "filter_total_count": "0",
    }
    resp = session.post(
        SEARCH_URL,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "__RequestVerificationToken": token,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    records = data.get("data", [])
    # Match the record whose nepanumber equals our case number (exact match)
    for rec in records:
        if rec.get("nepanumber", "").upper() == case_number.upper():
            return rec
    # If no exact match but only one result, accept it
    if len(records) == 1:
        return records[0]
    return None


_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


def _fetch_project_page(
    project_id: str,
    session: requests.Session,
) -> dict:
    """GET /Project-Home/?id={project_id} and extract date fields."""
    url = f"{PROJECT_URL}?id={project_id}"
    resp = session.get(url, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for s in soup(["script", "style", "head"]):
        s.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    dates = {}

    # Pattern 1 — direct label → date (EA / CE format):
    #   "FONSI Date"        → date
    #   "Start Date"        → date
    #   "End Date"          → date
    #   "NOI Publication Date" → date
    direct_labels = {
        "start date": "start_date",
        "end date": "end_date",
        "fonsi date": "fonsi_date",
        "rod date": "rod_date",
        "decision date": "decision_date",
        "noi date": "noi_date",
        "noi publication date": "noi_date",
        "notice of intent date": "noi_date",
        "categorical exclusion date": "decision_date",
        "determination date": "decision_date",
        "date signed": "decision_date",
        "date completed": "end_date",
    }
    for i, line in enumerate(lines):
        label = line.lower()
        if label in direct_labels and i + 1 < len(lines):
            candidate = lines[i + 1].strip()
            if _DATE_RE.match(candidate):
                field = direct_labels[label]
                if field not in dates:          # first occurrence wins
                    dates[field] = candidate

    # Pattern 2 — milestone header → "Actual Date" → date (EIS format):
    #   "Record of Decision Publication"  →  "Actual Date"  →  date
    #   "FONSI Publication"               →  "Actual Date"  →  date
    #   "Notice of Intent Publication"    →  "NOI Publication Date" → date  (handled above)
    milestone_labels = {
        "record of decision publication": "rod_date",
        "fonsi publication": "fonsi_date",
        "decision publication": "decision_date",
    }
    for i, line in enumerate(lines):
        label = line.lower()
        if label in milestone_labels:
            # Look ahead up to 3 lines for "Actual Date" then a date
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].lower() == "actual date" and j + 1 < len(lines):
                    candidate = lines[j + 1].strip()
                    if _DATE_RE.match(candidate):
                        field = milestone_labels[label]
                        if field not in dates:
                            dates[field] = candidate
                    break
                # Sometimes the date appears directly after the milestone label
                if _DATE_RE.match(lines[j].strip()):
                    field = milestone_labels[label]
                    if field not in dates:
                        dates[field] = lines[j].strip()
                    break

    # Project name
    project_name = None
    for i, line in enumerate(lines):
        if line.lower() == "project name" and i + 1 < len(lines):
            project_name = lines[i + 1]
            break

    return {"project_name": project_name, "dates": dates, "project_url": url}


def fetch_case_numbers(
    case_numbers: list[str],
    dry_run: bool = False,
) -> pd.DataFrame:
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    to_fetch = [cn for cn in case_numbers if cn not in cache]
    cached_count = len(case_numbers) - len(to_fetch)
    print(f"  {cached_count} already cached, {len(to_fetch)} to fetch")

    if dry_run:
        print("DRY RUN — case numbers that would be fetched:")
        for cn in to_fetch[:50]:
            print(f"  {cn}")
        if len(to_fetch) > 50:
            print(f"  ... and {len(to_fetch) - 50} more")
        return pd.DataFrame()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; NEPA-academic-research/1.0)",
    })
    # Prime the session cookie
    session.get(f"{BASE_URL}/search/", timeout=15)
    token = _get_token(session)
    token_refresh_interval = 50  # refresh token every N requests

    for i, cn in enumerate(to_fetch, 1):
        # Refresh token periodically
        if i % token_refresh_interval == 0:
            try:
                token = _get_token(session)
            except Exception:
                pass  # keep using old token

        record = {
            "case_number": cn,
            "fetch_status": "error",
            "project_id_blm": None,
            "project_name": None,
            "nepastatus": None,
            "lead_office": None,
            "type_blm": None,
            "start_date": None,
            "fonsi_date": None,
            "rod_date": None,
            "decision_date": None,
            "noi_date": None,
            "project_url": None,
            "register_fetch_at": datetime.now(timezone.utc).isoformat(),
        }

        print(f"  [{i}/{len(to_fetch)}] {cn} ...", end="", flush=True)
        try:
            search_rec = _search_case_number(cn, session, token)
            if search_rec is None:
                record["fetch_status"] = "not_found"
                print(" not_found")
            else:
                record["project_id_blm"] = search_rec.get("projectid")
                record["project_name"] = search_rec.get("projectname")
                record["nepastatus"] = search_rec.get("nepastatus")
                record["lead_office"] = search_rec.get("leadoffice")
                record["type_blm"] = search_rec.get("type")

                # Step B: project page for dates
                if record["project_id_blm"]:
                    time.sleep(0.5)  # brief pause before second request
                    page_data = _fetch_project_page(record["project_id_blm"], session)
                    dates = page_data["dates"]
                    record["start_date"] = dates.get("start_date")
                    record["fonsi_date"] = dates.get("fonsi_date")
                    record["rod_date"] = dates.get("rod_date")
                    record["decision_date"] = dates.get("decision_date")
                    record["noi_date"] = dates.get("noi_date")
                    record["project_url"] = page_data["project_url"]
                    if not record["project_name"] and page_data["project_name"]:
                        record["project_name"] = page_data["project_name"]

                record["fetch_status"] = "ok"
                decision = record["fonsi_date"] or record["rod_date"] or record["decision_date"] or "no date"
                print(f" ok | {decision}")

        except requests.exceptions.Timeout:
            record["fetch_status"] = "timeout"
            print(" timeout")
        except requests.exceptions.HTTPError as e:
            record["fetch_status"] = f"http_{e.response.status_code}"
            print(f" http_{e.response.status_code}")
        except Exception as exc:
            record["fetch_status"] = f"error: {str(exc)[:60]}"
            print(f" error: {str(exc)[:60]}")

        cache[cn] = record

        if i % 25 == 0:
            CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))

        time.sleep(RATE_LIMIT_SECS)

    CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))

    records = [cache[cn] for cn in case_numbers if cn in cache]
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acceptance", nargs="+", choices=["accept", "review", "skip"],
        default=["accept"],
        help="Which rows from 09a to fetch (default: accept only)",
    )
    parser.add_argument(
        "--case-types", nargs="+",
        help="Restrict to specific case-type suffixes, e.g. --case-types EIS EA CX",
    )
    parser.add_argument(
        "--refetch", action="store_true",
        help="Re-fetch case numbers already in the cache (useful after parser fixes)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not EVIDENCE_PATH.exists():
        raise SystemExit(f"Run 09a first — {EVIDENCE_PATH} not found.")

    BLM_DIR.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_parquet(EVIDENCE_PATH)
    subset = evidence[evidence["acceptance"].isin(args.acceptance)]

    if args.case_types:
        subset = subset[subset["case_type"].isin([ct.upper() for ct in args.case_types])]

    case_numbers = sorted(subset["case_number"].unique().tolist())

    # --refetch: remove matching entries from cache so they get re-fetched
    if args.refetch and not args.dry_run:
        cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        removed = sum(1 for cn in case_numbers if cn in cache)
        for cn in case_numbers:
            cache.pop(cn, None)
        CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))
        print(f"Cleared {removed} entries from cache for re-fetch.")
    print(f"Loaded {len(case_numbers)} unique case numbers "
          f"(acceptance: {args.acceptance}) from 09a evidence.")

    if not case_numbers:
        print("No case numbers to fetch.")
        return

    print(f"\nFetching from BLM National NEPA Register ({len(case_numbers)} case numbers) ...")
    records_df = fetch_case_numbers(case_numbers, dry_run=args.dry_run)

    if args.dry_run or records_df.empty:
        return

    records_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(records_df)} records → {OUTPUT_PATH}")
    print("\nFetch status breakdown:")
    print(records_df["fetch_status"].value_counts().to_string())
    with_decision = records_df[
        records_df["fonsi_date"].notna()
        | records_df["rod_date"].notna()
        | records_df["decision_date"].notna()
    ]
    print(f"\nRecords with any decision date: {len(with_decision)} / {len(records_df)}")
    print(f"Records with start date: {records_df['start_date'].notna().sum()} / {len(records_df)}")


if __name__ == "__main__":
    main()
