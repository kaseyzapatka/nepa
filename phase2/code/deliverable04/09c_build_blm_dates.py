"""
Match BLM register records to NEPATEC project_ids and produce the final
blm_eplanning_dates.parquet for downstream D4 consumption.

Reads:
    phase2/data/analysis/blm_register/nepatec_case_evidence.parquet  (from 09a)
    phase2/data/analysis/blm_register/blm_register_records.parquet   (from 09b)

Writes:
    phase2/data/analysis/blm_register/blm_eplanning_dates.parquet
    phase2/data/analysis/blm_register/blm_manual_review.csv

Usage:
    python 09c_build_blm_dates.py
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
BLM_DIR = ANALYSIS_DIR / "blm_register"
EVIDENCE_PATH = BLM_DIR / "nepatec_case_evidence.parquet"
REGISTER_PATH = BLM_DIR / "blm_register_records.parquet"
OUTPUT_PATH = BLM_DIR / "blm_eplanning_dates.parquet"
REVIEW_PATH = BLM_DIR / "blm_manual_review.csv"
PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"


def _normalize_date(raw: str | None) -> str | None:
    if not raw or pd.isna(raw):
        return None
    raw = str(raw).strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw
    return None


def _pick_decision(row: pd.Series) -> tuple[str | None, str | None]:
    """Return (date_iso, date_type) picking the best decision date available."""
    for field, dtype in [
        ("fonsi_date", "fonsi"),
        ("rod_date", "rod"),
        ("decision_date", "decision"),
        ("end_date", "end_date_proxy"),   # EIS end date as last-resort proxy
    ]:
        val = _normalize_date(row.get(field))
        if val:
            return val, dtype
    return None, None


def main() -> None:
    BLM_DIR.mkdir(parents=True, exist_ok=True)

    if not EVIDENCE_PATH.exists():
        raise SystemExit(f"Run 09a first: {EVIDENCE_PATH} not found.")
    if not REGISTER_PATH.exists():
        raise SystemExit(f"Run 09b first: {REGISTER_PATH} not found.")

    evidence = pd.read_parquet(EVIDENCE_PATH)
    register = pd.read_parquet(REGISTER_PATH)
    projects = pd.read_parquet(PROJECTS_PATH)[
        ["project_id", "process_type", "lead_agency_harmonized"]
    ]

    print(f"Evidence rows: {len(evidence)} ({evidence['project_id'].nunique()} projects)")
    print(f"Register records: {len(register)} ({register['case_number'].nunique()} case numbers)")

    # Join evidence to register on case_number
    joined = evidence.merge(register, on="case_number", how="left")

    def _match_status(row) -> str:
        fs = str(row.get("fetch_status", "")) if not pd.isna(row.get("fetch_status", None)) else ""
        if not fs:
            return "unmatched"
        if fs == "ok" and row.get("acceptance") == "accept":
            return "accepted"
        if fs in ("ok", "no_match") and row.get("acceptance") == "review":
            return "review"
        if fs == "not_found":
            return "not_in_register"
        if fs.startswith(("http_", "timeout", "error")):
            return "fetch_error"
        return "unmatched"

    joined["blm_match_status"] = joined.apply(_match_status, axis=1)

    # Derive decision and initiation dates
    decision_info = joined.apply(_pick_decision, axis=1)
    joined["blm_decision_date"] = [d for d, _ in decision_info]
    joined["blm_decision_date_type"] = [t for _, t in decision_info]
    joined["blm_initiation_date"] = joined.apply(
        lambda r: _normalize_date(r.get("start_date") or r.get("noi_date") or r.get("noi_publication_date")), axis=1
    )

    # Suppress dates for non-accepted rows
    mask_not_accepted = joined["blm_match_status"] != "accepted"
    joined.loc[mask_not_accepted, "blm_decision_date"] = None
    joined.loc[mask_not_accepted, "blm_initiation_date"] = None
    joined.loc[mask_not_accepted, "blm_decision_date_type"] = None

    # Best accepted match per project: prefer decision_date present, then main_document
    accepted = joined[joined["blm_match_status"] == "accepted"].copy()
    accepted = accepted.sort_values(
        ["project_id", "blm_decision_date", "evidence_rank"],
        ascending=[True, True, True],
        na_position="last",
    )
    best_accepted = accepted.drop_duplicates(subset=["project_id"], keep="first")

    # All BLM projects
    blm_projects = projects[
        projects["lead_agency_harmonized"].str.lower().str.contains("bureau of land", na=False)
    ].copy()

    output = blm_projects.merge(
        best_accepted[[
            "project_id", "case_number", "blm_match_status",
            "blm_decision_date", "blm_decision_date_type", "blm_initiation_date",
            "project_name", "lead_office", "evidence_rank", "document_id", "page_number",
            "register_fetch_at", "project_url",
        ]].rename(columns={
            "case_number": "blm_case_number",
            "project_name": "blm_project_name",
            "lead_office": "blm_lead_office",
            "document_id": "blm_evidence_document_id",
            "page_number": "blm_evidence_page_number",
            "register_fetch_at": "blm_register_fetch_at",
            "project_url": "blm_project_url",
        }),
        on="project_id",
        how="left",
    )

    # Fill unmatched
    has_evidence_any = output["project_id"].isin(joined["project_id"])
    no_evidence = ~has_evidence_any
    output.loc[no_evidence, "blm_match_status"] = "unmatched"

    has_evidence_no_accept = (
        output["project_id"].isin(
            joined[joined["blm_match_status"] != "accepted"]["project_id"]
        ) & output["blm_match_status"].isna()
    )
    output.loc[has_evidence_no_accept, "blm_match_status"] = "no_accepted_match"
    output["blm_match_status"] = output["blm_match_status"].fillna("unmatched")

    output["build_run_at"] = datetime.now(timezone.utc).isoformat()
    output = output.drop(columns=["lead_agency_harmonized", "evidence_rank"], errors="ignore")

    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(output)} rows → {OUTPUT_PATH}")
    print("\nMatch status breakdown:")
    print(output["blm_match_status"].value_counts().to_string())
    accepted_out = output[output["blm_match_status"] == "accepted"]
    print(f"\nProcess type breakdown (accepted only):")
    print(accepted_out.groupby("process_type").size().to_string())
    print(f"\nAccepted with decision_date: "
          f"{accepted_out['blm_decision_date'].notna().sum()} / {len(accepted_out)}")
    print(f"Accepted with initiation_date: "
          f"{accepted_out['blm_initiation_date'].notna().sum()} / {len(accepted_out)}")

    review_cases = joined[joined["blm_match_status"] == "review"]
    if not review_cases.empty:
        review_cases.to_csv(REVIEW_PATH, index=False)
        print(f"\nWrote {len(review_cases)} review cases → {REVIEW_PATH}")


if __name__ == "__main__":
    main()
