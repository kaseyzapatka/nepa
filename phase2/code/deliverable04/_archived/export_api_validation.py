"""
Export API-sourced timeline dates for manual spot-checking.

Joins the final timeline_project_dates.parquet to each register source
to add project titles, URLs, and source labels — one row per project
with any API-sourced date.

Reads:
    phase2/data/analysis/timeline/timeline_project_dates.parquet  (or --dates-path)
    phase2/data/analysis/projects_combined.parquet
    phase2/data/analysis/blm_register/blm_eplanning_dates.parquet
    phase2/data/analysis/doe_register/doe_cx_dates.parquet
    phase2/data/analysis/doe_register/doe_eplanning_dates.parquet
    phase2/data/analysis/doe_register/doe_project_page_records.parquet

Writes:
    phase2/output/deliverable04/api_date_validation.csv

Usage:
    python export_api_validation.py
    python export_api_validation.py --dates-path path/to/timeline_project_dates.parquet
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"

DEFAULT_DATES_PATH = ANALYSIS_DIR / "timeline" / "timeline_project_dates.parquet"
PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
BLM_PATH = ANALYSIS_DIR / "blm_register" / "blm_eplanning_dates.parquet"
DOE_CX_PATH = ANALYSIS_DIR / "doe_register" / "doe_cx_dates.parquet"
DOE_EP_PATH = ANALYSIS_DIR / "doe_register" / "doe_eplanning_dates.parquet"
DOE_PP_PATH = ANALYSIS_DIR / "doe_register" / "doe_project_page_records.parquet"


def _source_label(evidence_text: str | None) -> str:
    if not evidence_text:
        return "unknown"
    t = str(evidence_text).lower()
    if "blm nepa register" in t:
        return "BLM ePlanning"
    if "doe cx register" in t:
        return "DOE CX Register"
    if "doe eplanning" in t:
        return "DOE ePlanning"
    if "fr noi" in t or "noi publication" in t:
        return "Federal Register NOI"
    if "nepa case" in t or "proxy" in t:
        return "NEPA Case Number (proxy)"
    return "metadata"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates-path", default=str(DEFAULT_DATES_PATH))
    args = parser.parse_args()

    dates_path = Path(args.dates_path)
    if not dates_path.exists():
        raise SystemExit(f"Dates file not found: {dates_path}\nRun scripts 02→04 first.")

    dates = pd.read_parquet(dates_path)
    print(f"Loaded {len(dates):,} project rows from {dates_path}")

    # Filter to rows with any API-sourced date
    api_decision = dates["decision_source_type"].isin(["metadata", "noi_notice"])
    api_initiation = dates["initiation_source_type"].isin(["metadata", "noi_notice"])
    api_rows = dates[api_decision | api_initiation].copy()
    print(f"API-sourced dates: {len(api_rows):,} projects")

    # Project metadata
    proj = pd.read_parquet(PROJECTS_PATH, columns=[
        "project_id", "project_title", "lead_agency",
        "noi_url", "noi_publication_date",
    ])
    proj["project_id"] = proj["project_id"].astype(str)
    api_rows["project_id"] = api_rows["project_id"].astype(str)
    api_rows = api_rows.merge(proj, on="project_id", how="left")

    # BLM ePlanning — adds blm_project_url, blm_project_name, blm_case_number
    if BLM_PATH.exists():
        blm = pd.read_parquet(BLM_PATH, columns=[
            "project_id", "blm_project_url", "blm_project_name", "blm_case_number",
        ])
        blm["project_id"] = blm["project_id"].astype(str)
        api_rows = api_rows.merge(blm, on="project_id", how="left")
    else:
        api_rows["blm_project_url"] = None
        api_rows["blm_project_name"] = None
        api_rows["blm_case_number"] = None

    # DOE CX Register — adds cx_number, cx_title, office, cx_codes; construct URL
    if DOE_CX_PATH.exists():
        cx = pd.read_parquet(DOE_CX_PATH, columns=[
            "project_id", "cx_number", "cx_title", "office", "cx_codes",
        ])
        cx["project_id"] = cx["project_id"].astype(str)
        cx = cx.drop_duplicates("project_id")
        cx["doe_cx_url"] = cx["cx_number"].apply(
            lambda n: f"https://www.energy.gov/nepa/articles/cx-{str(int(n)).zfill(6)}-categorical-exclusion-determination"
            if pd.notna(n) else None
        )
        api_rows = api_rows.merge(cx, on="project_id", how="left")
    else:
        for col in ("cx_number", "cx_title", "office", "cx_codes", "doe_cx_url"):
            api_rows[col] = None

    # DOE ePlanning — adds doe_doc_number and project URL via page records
    if DOE_EP_PATH.exists() and DOE_PP_PATH.exists():
        doe_ep = pd.read_parquet(DOE_EP_PATH, columns=["project_id", "doe_doc_number"])
        doe_ep["project_id"] = doe_ep["project_id"].astype(str)
        doe_pp = pd.read_parquet(DOE_PP_PATH, columns=["doc_number", "url"])
        doe_pp = doe_pp.rename(columns={"url": "doe_ep_url"})
        doe_ep = doe_ep.merge(doe_pp, left_on="doe_doc_number", right_on="doc_number", how="left")
        api_rows = api_rows.merge(
            doe_ep[["project_id", "doe_doc_number", "doe_ep_url"]],
            on="project_id", how="left",
        )
    else:
        api_rows["doe_doc_number"] = None
        api_rows["doe_ep_url"] = None

    # Derive source labels from evidence text heuristic
    api_rows["decision_source_label"] = api_rows["decision_evidence_text"].apply(_source_label)
    api_rows["initiation_source_label"] = api_rows["initiation_evidence_text"].apply(_source_label)

    # Override 1: year-granularity dates are always NEPA case number proxies regardless
    # of evidence text (which shows surrounding document context, not an API label).
    year_dec = api_rows["decision_date_granularity"] == "year"
    api_rows.loc[year_dec, "decision_source_label"] = "NEPA Case Number (proxy)"
    year_init = api_rows["initiation_date_granularity"] == "year"
    api_rows.loc[year_init, "initiation_source_label"] = "NEPA Case Number (proxy)"

    # Override 2: rows included due to an API *initiation* date may have a document-text
    # *decision* date. Use the actual source_type column from the dates parquet to correct.
    if "decision_source_type" in api_rows.columns:
        doc_text_dec = (api_rows["decision_source_type"] == "document_text") & \
                       (api_rows["decision_source_label"] == "metadata")
        api_rows.loc[doc_text_dec, "decision_source_label"] = "Document Text"
        doc_text_init = (api_rows["initiation_source_type"] == "document_text") & \
                        (api_rows["initiation_source_label"] == "metadata")
        api_rows.loc[doc_text_init, "initiation_source_label"] = "Document Text"

    # Derive best verification URL per row
    def _pick_url(row):
        label = row.get("decision_source_label") or row.get("initiation_source_label") or ""
        if "BLM" in label:
            return row.get("blm_project_url")
        if "DOE CX" in label:
            return row.get("doe_cx_url")
        if "DOE ePlanning" in label:
            return row.get("doe_ep_url")
        if "Federal Register" in label:
            return row.get("noi_url")
        return None

    api_rows["verification_url"] = api_rows.apply(_pick_url, axis=1)

    # Build the export
    out = api_rows[[
        "project_id",
        "project_title",
        "process_type",
        "decision_date",
        "decision_date_granularity",
        "decision_source_type",
        "decision_source_label",
        "decision_evidence_text",
        "decision_document_id",
        "initiation_date",
        "initiation_date_granularity",
        "initiation_source_type",
        "initiation_source_label",
        "initiation_evidence_text",
        "verification_url",
        # Source-specific labels for cross-reference
        "blm_case_number",
        "blm_project_name",
        "cx_number",
        "cx_title",
        "cx_codes",
        "doe_doc_number",
    ]].copy()

    out = out.sort_values(["decision_source_label", "process_type", "project_id"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "api_date_validation.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out):,} rows → {out_path}")

    print("\nBreakdown by source label:")
    for label, grp in out.groupby("decision_source_label", dropna=False):
        n_dec = grp["decision_date"].notna().sum()
        n_init = grp["initiation_date"].notna().sum()
        print(f"  {label}: {len(grp)} projects ({n_dec} decision, {n_init} initiation)")

    print("\nSample rows:")
    sample = out[out["decision_date"].notna()].head(5)
    for _, r in sample.iterrows():
        print(f"  [{r['decision_source_label']}] {str(r['project_id'])[:8]}… "
              f"{r['decision_date']} | {str(r['decision_evidence_text'])[:60]}")


if __name__ == "__main__":
    main()
