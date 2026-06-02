"""
Join DOE case evidence to register lookup tables and build project-level date output.

Reads:
    phase2/data/analysis/doe_register/doe_case_evidence.parquet
    phase2/data/analysis/doe_register/doe_register_records.parquet

Writes:
    phase2/data/analysis/doe_register/doe_eplanning_dates.parquet
      One row per DOE project; decision and initiation dates ready for D4 pipeline.
    phase2/data/analysis/doe_register/doe_manual_review.csv
      Projects with acceptance=review for human inspection.

Usage:
    python 10c_build_doe_dates.py
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
DOE_DIR = ANALYSIS_DIR / "doe_register"
EVIDENCE_PATH = DOE_DIR / "doe_case_evidence.parquet"
RECORDS_PATH = DOE_DIR / "doe_register_records.parquet"
OUTPUT_PATH = DOE_DIR / "doe_eplanning_dates.parquet"
REVIEW_PATH = DOE_DIR / "doe_manual_review.csv"

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip()
    if ISO_RE.match(raw):
        return raw
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def _pick_decision(row: dict) -> tuple[str | None, str | None]:
    """Pick the best decision date in priority order."""
    for field, dtype in [
        ("fonsi_date", "fonsi"),
        ("rod_date", "rod"),
    ]:
        val = _normalize_date(row.get(field))
        if val:
            return val, dtype
    return None, None


def _pick_initiation(row: dict) -> tuple[str | None, str | None]:
    """Pick the best initiation date."""
    for field, dtype in [
        ("noi_date", "noi"),
    ]:
        val = _normalize_date(row.get(field))
        if val:
            return val, dtype
    return None, None


def main() -> None:
    if not EVIDENCE_PATH.exists():
        raise SystemExit(f"Run 10a first — {EVIDENCE_PATH} not found.")
    if not RECORDS_PATH.exists():
        raise SystemExit(f"Run 10b first — {RECORDS_PATH} not found.")

    evidence = pd.read_parquet(EVIDENCE_PATH)
    records = pd.read_parquet(RECORDS_PATH)

    print(f"Evidence rows: {len(evidence)} ({evidence['acceptance'].value_counts().to_dict()})")
    print(f"Register records: {len(records)}")

    # Work with accepted evidence only
    accepted = evidence[evidence["acceptance"] == "accept"].copy()
    print(f"Accepted evidence rows: {len(accepted)} across "
          f"{accepted['project_id'].nunique()} projects")

    def _base_number(n: str) -> str:
        """Strip supplement suffixes: DOE/EIS-0391-SA-05 → DOE/EIS-0391."""
        return re.sub(r"[-‐-―](S\d+|SA[-\s]?\d+)$", "", str(n).upper().strip())

    # Normalize doc numbers for join
    accepted["doc_number_norm"] = accepted["doc_number"].str.upper().str.strip()
    records["doc_number_norm"] = records["doc_number"].str.upper().str.strip()

    # Also build base-number variants for fallback matching
    accepted["doc_number_base"] = accepted["doc_number_norm"].apply(_base_number)
    records["doc_number_base"] = records["doc_number_norm"].apply(_base_number)

    # Join 1: exact match
    merged_exact = accepted.merge(records, on="doc_number_norm", how="left",
                                  suffixes=("_ev", "_reg"))

    # Join 2: base-number fallback for rows that got no dates from exact match
    no_date_mask = (
        merged_exact["rod_date"].isna()
        & merged_exact["fonsi_date"].isna()
        & merged_exact.get("noi_date", pd.Series(dtype=object)).isna()
        if "noi_date" in merged_exact.columns
        else merged_exact["rod_date"].isna() & merged_exact["fonsi_date"].isna()
    )
    if no_date_mask.any():
        fallback_ev = merged_exact[no_date_mask][
            [c for c in merged_exact.columns if c.endswith("_ev") or c in
             ["project_id", "process_type", "doc_number_norm", "doc_number_base"]]
        ].copy()
        # Rename _ev columns back
        fallback_ev.columns = [c.replace("_ev", "") for c in fallback_ev.columns]
        fallback_merged = fallback_ev.merge(
            records.drop(columns=["doc_number_norm"]),
            on="doc_number_base", how="left",
            suffixes=("_ev", "_reg"),
        )
        # Splice back dates into merged_exact
        date_cols = [c for c in ["rod_date", "fonsi_date", "noi_date"]
                     if c in fallback_merged.columns]
        for col in date_cols:
            patch = fallback_merged.set_index("project_id")[col]
            merged_exact.loc[no_date_mask, col] = (
                merged_exact.loc[no_date_mask, "project_id"].map(patch)
            )

    merged = merged_exact

    print(f"After join: {merged['project_id'].nunique()} projects matched")

    # One row per project: pick best dates
    output_rows = []
    for project_id, grp in merged.groupby("project_id"):
        # Use the row with most complete date information
        row = grp.sort_values(
            ["fonsi_date", "rod_date", "noi_date"],
            na_position="last",
        ).iloc[0].to_dict()

        decision_date, decision_type = _pick_decision(row)
        initiation_date, initiation_type = _pick_initiation(row)

        doc_number = row.get("doc_number_ev") or row.get("doc_number")
        in_register = pd.notna(row.get("rod_date")) or pd.notna(row.get("fonsi_date")) or pd.notna(row.get("noi_date"))
        match_status = "found" if in_register else "accepted_not_found"

        output_rows.append({
            "project_id": project_id,
            "process_type": row.get("process_type"),
            "doe_doc_number": doc_number,
            "doe_match_status": match_status,
            "doe_decision_date": decision_date,
            "doe_decision_date_type": decision_type,
            "doe_initiation_date": initiation_date,
            "doe_initiation_date_type": initiation_type,
            "doe_decision_tier_a_eligible": decision_date is not None,
            "doe_initiation_tier_a_eligible": initiation_date is not None,
            "doe_fonsi_date_raw": _normalize_date(row.get("fonsi_date")),
            "doe_rod_date_raw": _normalize_date(row.get("rod_date")),
            "doe_noi_date_raw": _normalize_date(row.get("noi_date")),
            "built_at": datetime.now(timezone.utc).isoformat(),
        })

    # Add review rows (projects with evidence but acceptance=review, no decision yet)
    review_ev = evidence[evidence["acceptance"] == "review"].copy()

    output_df = pd.DataFrame(output_rows)

    # Coverage analysis
    with_decision = output_df["doe_decision_date"].notna().sum()
    with_initiation = output_df["doe_initiation_date"].notna().sum()
    total = len(output_df)

    print(f"\nProject-level output: {total} projects")
    print(f"  Decision dates: {with_decision} ({with_decision/total*100:.1f}%)")
    print(f"  Initiation dates: {with_initiation} ({with_initiation/total*100:.1f}%)")

    if not output_df.empty:
        print("\nBreakdown by process_type:")
        for pt, grp in output_df.groupby("process_type"):
            d = grp["doe_decision_date"].notna().sum()
            i = grp["doe_initiation_date"].notna().sum()
            print(f"  {pt}: {len(grp)} projects, {d} decision dates, {i} initiation dates")

    output_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(output_df)} rows → {OUTPUT_PATH}")

    # Manual review export
    if not review_ev.empty:
        review_ev.to_csv(REVIEW_PATH, index=False)
        print(f"Wrote {len(review_ev)} review rows → {REVIEW_PATH}")


if __name__ == "__main__":
    main()
