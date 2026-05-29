"""
Build the project-document index for D4 timeline extraction.

Joins project metadata, document metadata, and FR fields into a single
timeline_document_index.parquet with document role scores, appendix flags,
scan priority, and NOI Tier A eligibility.

Usage:
    python 01_build_timeline_index.py [--sample-ids path/to/ids.txt]
    python 01_build_timeline_index.py --process CE EA EIS
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
PROCESSED_DIR = PHASE2 / "data" / "processed"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
SECTIONS_PATH = ANALYSIS_DIR / "document_sections.parquet"
PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
DOCUMENTS_PATH = ANALYSIS_DIR / "documents_combined.parquet"
BLM_DATES_PATH = ANALYSIS_DIR / "blm_register" / "blm_eplanning_dates.parquet"
DOE_DATES_PATH = ANALYSIS_DIR / "doe_register" / "doe_eplanning_dates.parquet"
DOE_CX_DATES_PATH = ANALYSIS_DIR / "doe_register" / "doe_cx_dates.parquet"
OUTPUT_PATH = TIMELINE_DIR / "timeline_document_index.parquet"

SOURCE_MAP = {"CE": "ce", "EA": "ea", "EIS": "eis"}

# ---------------------------------------------------------------------------
# Field existence check: fail loud if expected FR fields are absent
# ---------------------------------------------------------------------------
REQUIRED_PROJECT_FIELDS = [
    "project_id", "project_title", "process_type", "project_energy_type",
    "lead_agency_harmonized", "project_department", "project_state",
    "project_doc_count", "noi_publication_date", "noi_match_status",
    "noi_match_confidence", "noa_availability_date", "noa_match_status",
]
REQUIRED_DOCUMENT_FIELDS = [
    "project_id", "document_id", "document_type", "document_title",
    "file_name", "total_pages", "main_document", "dataset_source",
    "document_type_clean", "document_date_from_file_name", "document_type_category",
]


# ---------------------------------------------------------------------------
# Title / filename cue patterns
# ---------------------------------------------------------------------------
INITIATION_TITLE_RE = re.compile(
    r"\b("
    r"notice\s+of\s+intent|noi|scoping|application|permit|apdplan\s+of\s+development|pod|"
    r"right[- ]of[- ]way|row|license\s+application|request|submitted|received|"
    r"project\s+proposal|environmental\s+review\s+began"
    r")\b",
    re.IGNORECASE,
)
DECISION_TITLE_RE = re.compile(
    r"\b("
    r"record\s+of\s+decision|rod|finding\s+of\s+no\s+significant\s+impact|fonsi|"
    r"decision\s+record|decision\s+notice|decision\s+memo|categorical\s+exclusion|"
    r"determination|approval|approved|signed\s+decision|ce\s+determination"
    r")\b",
    re.IGNORECASE,
)
NEGATIVE_TITLE_RE = re.compile(
    r"\b("
    r"appendix|attachment|exhibit|technical\s+report|resource\s+report|"
    r"biological\s+assessment|cultural\s+report|survey|map|figure|"
    r"comment|response|reference|bibliography|index|distribution\s+list"
    r")\b",
    re.IGNORECASE,
)
APPENDIX_TYPE_RE = re.compile(
    r"\b("
    r"appendix|attachment|exhibit|technical\s+report|resource\s+report|"
    r"biological\s+assessment|cultural\s+report|survey|map|figure|"
    r"comment|response|reference|bibliography|"
    r"mitigation\s+plan|monitoring\s+plan|weed\s+management|dust\s+control"
    r")\b",
    re.IGNORECASE,
)

# Strong cues that override appendix penalty even in appendix-like documents
STRONG_CUES_RE = re.compile(
    r"\b(record\s+of\s+decision|fonsi|finding\s+of\s+no\s+significant\s+impact|"
    r"categorical\s+exclusion\s+determination|notice\s+of\s+intent|scoping\s+notice|"
    r"application\s+received)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Document type scoring
# ---------------------------------------------------------------------------

DECISION_DOC_SCORES: dict[str, float] = {
    # CE
    "ce determination": 5.0, "categorical exclusion determination": 5.0,
    "categorical exclusion": 4.5, "decision memo": 4.5, "approval memo": 4.5,
    "signed decision": 4.5,
    # EA
    "fonsi": 5.0, "finding of no significant impact": 5.0,
    "decision record": 5.0, "decision notice": 5.0,
    "final ea": 3.0, "final environmental assessment": 3.0,
    # EIS
    "rod": 5.0, "record of decision": 5.0,
    "joint record of decision": 5.0,
    "final eis": 2.5, "final environmental impact statement": 2.5,
    # Generic
    "decision": 3.0, "approval": 2.5,
}

INITIATION_DOC_SCORES: dict[str, float] = {
    "notice of intent": 5.0, "noi": 5.0,
    "scoping notice": 4.5, "public scoping notice": 4.5,
    "application": 4.0, "permit application": 4.0,
    "apd": 4.0, "right-of-way application": 4.0,
    "plan of development": 3.5, "pod": 3.5,
    "license application": 4.0, "project proposal": 3.0,
    "draft ea": 1.5, "draft environmental assessment": 1.5, "dea": 1.5,
    "draft eis": 1.5, "draft environmental impact statement": 1.5, "deis": 1.5,
}

MAIN_DOC_BONUS = 1.5
APPENDIX_PENALTY_SCORE = 2.5


def _match_doc_score(type_str: str | None, score_map: dict[str, float]) -> float:
    if not type_str:
        return 0.0
    t = type_str.strip().lower()
    for key, score in score_map.items():
        if key in t:
            return score
    return 0.0


def _is_appendix_like(
    document_type: str | None,
    document_type_clean: str | None,
    document_type_category: str | None,
    document_title: str | None,
    file_name: str | None,
) -> bool:
    if document_type_category and str(document_type_category).lower().strip() in {
        "appendix", "attachment", "exhibit", "reference", "comment",
    }:
        return True
    for value in [document_type, document_type_clean, document_title, file_name]:
        if value and APPENDIX_TYPE_RE.search(str(value)):
            # Check whether the same text also has a strong cue; if so, not appendix-like
            if STRONG_CUES_RE.search(str(value)):
                return False
            return True
    return False


def _noi_tier_a_eligible(
    noi_pub_date,
    noi_match_status: str | None,
    noi_match_confidence,
) -> bool:
    """Return True when NOI passes the Tier A acceptance threshold from plan §1."""
    if noi_pub_date is None or pd.isna(noi_pub_date):
        return False
    bad_statuses = {"unmatched", "rejected", "low_confidence", "missing", "", None}
    status = str(noi_match_status).strip().lower() if noi_match_status else ""
    if status in bad_statuses:
        return False
    # Confidence: categorical high/medium or numeric >= 0.75
    if noi_match_confidence is not None and not pd.isna(noi_match_confidence):
        conf_str = str(noi_match_confidence).strip().lower()
        if conf_str in {"high", "medium"}:
            return True
        try:
            conf_num = float(noi_match_confidence)
            return conf_num >= 0.75
        except (ValueError, TypeError):
            # Unrecognized scale — route to review per plan §1
            return False
    return False


def _compute_scores(row: dict) -> dict:
    decision_doc_score = max(
        _match_doc_score(row.get("document_type_clean"), DECISION_DOC_SCORES),
        _match_doc_score(row.get("document_type"), DECISION_DOC_SCORES),
        _match_doc_score(row.get("document_title"), DECISION_DOC_SCORES),
    )
    initiation_doc_score = max(
        _match_doc_score(row.get("document_type_clean"), INITIATION_DOC_SCORES),
        _match_doc_score(row.get("document_type"), INITIATION_DOC_SCORES),
        _match_doc_score(row.get("document_title"), INITIATION_DOC_SCORES),
    )
    main_doc_score = MAIN_DOC_BONUS if row.get("is_main_document") else 0.0
    appendix_penalty = APPENDIX_PENALTY_SCORE if row.get("is_appendix_like") else 0.0

    scan_priority_score = (
        max(decision_doc_score, initiation_doc_score) + main_doc_score - appendix_penalty
    )

    if scan_priority_score >= 6.0 or row.get("noi_tier_a_eligible"):
        scan_priority = "priority_1"
    elif scan_priority_score >= 3.0:
        scan_priority = "priority_2"
    elif scan_priority_score >= 1.0:
        scan_priority = "priority_3"
    elif str(row.get("process_type", "")).upper() in ("EIS", "ENVIRONMENTAL IMPACT STATEMENT"):
        # Never defer EIS documents — even unrecognized types may contain ROD language.
        # Defer was designed for low-value CE form pages, not EIS documents.
        scan_priority = "priority_3"
    else:
        scan_priority = "defer"

    return {
        "decision_doc_score": decision_doc_score,
        "initiation_doc_score": initiation_doc_score,
        "main_doc_score": main_doc_score,
        "appendix_penalty": appendix_penalty,
        "scan_priority_score": scan_priority_score,
        "scan_priority": scan_priority,
    }


def assert_fields(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected field(s) missing from {source}: {missing}\n"
            f"Available: {list(df.columns)}"
        )


def load_sections_doc_set() -> set[str]:
    if not SECTIONS_PATH.exists():
        return set()
    sec = pd.read_parquet(SECTIONS_PATH, columns=["document_id"])
    return set(sec["document_id"].dropna().unique())


def build_index(project_ids: list[str] | None, process_types: list[str]) -> pd.DataFrame:
    run_at = datetime.now(timezone.utc).isoformat()

    # Load and validate projects
    print("Loading projects_combined.parquet...")
    projects = pd.read_parquet(PROJECTS_PATH)
    assert_fields(projects, REQUIRED_PROJECT_FIELDS, "projects_combined.parquet")
    projects = projects[projects["process_type"].isin(process_types)].copy()
    if project_ids:
        projects = projects[projects["project_id"].isin(project_ids)]
    print(f"  {len(projects):,} projects after filtering")

    # Load and validate documents
    print("Loading documents_combined.parquet...")
    docs = pd.read_parquet(DOCUMENTS_PATH)
    assert_fields(docs, REQUIRED_DOCUMENT_FIELDS, "documents_combined.parquet")
    docs = docs[docs["project_id"].isin(projects["project_id"])].copy()
    print(f"  {len(docs):,} document rows after filtering")

    # Check which documents have sections
    sections_doc_set = load_sections_doc_set()
    print(f"  {len(sections_doc_set):,} document_ids in document_sections.parquet")

    # Normalize booleans
    docs["is_main_document"] = docs["main_document"].astype(str).str.strip().str.upper() == "YES"
    docs["doc_page_count"] = pd.to_numeric(docs["total_pages"], errors="coerce").fillna(0).astype(int)
    docs["document_date_from_file_name"] = pd.to_datetime(
        docs["document_date_from_file_name"], errors="coerce"
    )
    docs["has_filename_date"] = docs["document_date_from_file_name"].notna()

    # Title/filename cue flags
    def has_cue(pattern: re.Pattern, *cols: str) -> pd.Series:
        combined = docs[list(cols)].fillna("").astype(str).agg(" ".join, axis=1)
        return combined.str.contains(pattern, regex=True, na=False)

    docs["has_initiation_title_cue"] = has_cue(
        INITIATION_TITLE_RE, "document_title", "file_name", "document_type_clean"
    )
    docs["has_decision_title_cue"] = has_cue(
        DECISION_TITLE_RE, "document_title", "file_name", "document_type_clean"
    )
    docs["has_negative_title_cue"] = has_cue(
        NEGATIVE_TITLE_RE, "document_title", "file_name", "document_type_clean"
    )

    # Appendix detection
    docs["is_appendix_like"] = docs.apply(
        lambda r: _is_appendix_like(
            r.get("document_type"),
            r.get("document_type_clean"),
            r.get("document_type_category"),
            r.get("document_title"),
            r.get("file_name"),
        ),
        axis=1,
    )

    # Section availability
    docs["has_sections"] = docs["document_id"].isin(sections_doc_set)

    # Project-level burden rollup
    burden = (
        docs.groupby("project_id", as_index=False)
        .agg(
            project_doc_count_calc=("document_id", "nunique"),
            total_pages=("doc_page_count", "sum"),
            max_document_pages=("doc_page_count", "max"),
            appendix_count=("is_appendix_like", "sum"),
        )
    )

    # Project metadata columns to carry through
    proj_cols = [
        "project_id", "project_title", "process_type", "project_energy_type",
        "lead_agency_harmonized", "project_department", "project_state",
        "project_doc_count",
        "noi_publication_date", "noi_match_status", "noi_match_confidence",
        "noa_availability_date", "noa_match_status",
    ]
    proj_sub = projects[proj_cols].copy()
    proj_sub["noi_publication_date"] = pd.to_datetime(
        proj_sub["noi_publication_date"], errors="coerce"
    )
    proj_sub["noa_availability_date"] = pd.to_datetime(
        proj_sub["noa_availability_date"], errors="coerce"
    )

    # NOI Tier A eligibility (project-level)
    proj_sub["noi_tier_a_eligible"] = proj_sub.apply(
        lambda r: _noi_tier_a_eligible(
            r["noi_publication_date"],
            r["noi_match_status"],
            r["noi_match_confidence"],
        ),
        axis=1,
    )

    # BLM register Tier A — merge dates if available
    if BLM_DATES_PATH.exists():
        blm_cols = [
            "project_id", "blm_match_status",
            "blm_decision_date", "blm_decision_date_type",
            "blm_initiation_date", "blm_case_number",
        ]
        blm_df = pd.read_parquet(BLM_DATES_PATH, columns=blm_cols)
        proj_sub = proj_sub.merge(blm_df, on="project_id", how="left")
        proj_sub["blm_decision_tier_a_eligible"] = (
            (proj_sub["blm_match_status"] == "accepted")
            & proj_sub["blm_decision_date"].notna()
        )
        proj_sub["blm_initiation_tier_a_eligible"] = (
            (proj_sub["blm_match_status"] == "accepted")
            & proj_sub["blm_initiation_date"].notna()
        )
    else:
        for col in ("blm_match_status", "blm_decision_date", "blm_decision_date_type",
                    "blm_initiation_date", "blm_case_number",
                    "blm_decision_tier_a_eligible", "blm_initiation_tier_a_eligible"):
            proj_sub[col] = None

    # DOE register Tier A — merge dates if available
    if DOE_DATES_PATH.exists():
        doe_cols = [
            "project_id", "doe_match_status",
            "doe_decision_date", "doe_decision_date_type",
            "doe_initiation_date", "doe_doc_number",
        ]
        doe_df = pd.read_parquet(DOE_DATES_PATH, columns=doe_cols)
        proj_sub = proj_sub.merge(doe_df, on="project_id", how="left")
        proj_sub["doe_decision_tier_a_eligible"] = (
            (proj_sub["doe_match_status"] == "found")
            & proj_sub["doe_decision_date"].notna()
        )
        proj_sub["doe_initiation_tier_a_eligible"] = (
            (proj_sub["doe_match_status"] == "found")
            & proj_sub["doe_initiation_date"].notna()
        )
    else:
        for col in ("doe_match_status", "doe_decision_date", "doe_decision_date_type",
                    "doe_initiation_date", "doe_doc_number",
                    "doe_decision_tier_a_eligible", "doe_initiation_tier_a_eligible"):
            proj_sub[col] = None

    # DOE CX register Tier A — CE determination dates via cx-NNNNNN.pdf filename join
    if DOE_CX_DATES_PATH.exists():
        doe_cx_df = pd.read_parquet(DOE_CX_DATES_PATH, columns=[
            "project_id", "cx_number",
            "doe_cx_decision_date", "doe_cx_decision_date_type", "doe_cx_tier_a_eligible",
        ])
        proj_sub = proj_sub.merge(doe_cx_df, on="project_id", how="left")
        proj_sub["doe_cx_tier_a_eligible"] = proj_sub["doe_cx_tier_a_eligible"].fillna(False)
    else:
        for col in ("cx_number", "doe_cx_decision_date", "doe_cx_decision_date_type",
                    "doe_cx_tier_a_eligible"):
            proj_sub[col] = None

    # Merge burden onto projects, then join to documents
    proj_sub = proj_sub.merge(burden, on="project_id", how="left")
    proj_sub["project_doc_count"] = proj_sub["project_doc_count"].fillna(
        proj_sub["project_doc_count_calc"]
    )
    proj_sub["total_pages"] = proj_sub["total_pages"].fillna(0).astype(int)
    proj_sub["max_document_pages"] = proj_sub["max_document_pages"].fillna(0).astype(int)
    proj_sub["appendix_count"] = proj_sub["appendix_count"].fillna(0).astype(int)

    index_df = docs.merge(
        proj_sub.drop(columns=["project_doc_count_calc"], errors="ignore"),
        on="project_id",
        how="left",
    )

    # Score each document row
    score_records = index_df.apply(
        lambda r: _compute_scores(r.to_dict()), axis=1, result_type="expand"
    )
    index_df = pd.concat([index_df.reset_index(drop=True), score_records.reset_index(drop=True)], axis=1)

    index_df["index_run_at"] = run_at

    # Select and order final columns
    keep_cols = [
        "project_id", "document_id", "process_type", "project_energy_type",
        "lead_agency_harmonized", "project_department", "project_title",
        "project_state",
        "project_doc_count", "total_pages", "max_document_pages", "appendix_count",
        "document_type", "document_type_clean", "document_type_category",
        "document_title", "file_name", "file_id",
        "doc_page_count", "main_document", "is_main_document", "is_appendix_like",
        "document_date_from_file_name", "has_filename_date",
        "has_initiation_title_cue", "has_decision_title_cue", "has_negative_title_cue",
        "has_sections",
        "decision_doc_score", "initiation_doc_score", "main_doc_score",
        "appendix_penalty", "scan_priority_score", "scan_priority",
        "noi_publication_date", "noi_match_status", "noi_match_confidence", "noi_tier_a_eligible",
        "noa_availability_date", "noa_match_status",
        "blm_case_number", "blm_match_status",
        "blm_decision_date", "blm_decision_date_type", "blm_decision_tier_a_eligible",
        "blm_initiation_date", "blm_initiation_tier_a_eligible",
        "doe_doc_number", "doe_match_status",
        "doe_decision_date", "doe_decision_date_type", "doe_decision_tier_a_eligible",
        "doe_initiation_date", "doe_initiation_tier_a_eligible",
        "cx_number", "doe_cx_decision_date", "doe_cx_decision_date_type",
        "doe_cx_tier_a_eligible",
        "index_run_at",
    ]
    # Drop any columns not present
    keep_cols = [c for c in keep_cols if c in index_df.columns]
    index_df = index_df[keep_cols]

    return index_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build timeline document index.")
    parser.add_argument(
        "--process", nargs="+", choices=["CE", "EA", "EIS"], default=["CE", "EA", "EIS"]
    )
    parser.add_argument(
        "--sample-ids",
        help="Path to a text file with one project_id per line (for sample runs).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output even if it already exists.",
    )
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force and not args.sample_ids:
        print(f"Output already exists: {OUTPUT_PATH}")
        print("Re-run only when registers change (BLM/DOE) or NEPATEC is updated.")
        print("Pass --force to overwrite.")
        return

    project_ids = None
    if args.sample_ids:
        with open(args.sample_ids) as f:
            project_ids = [line.strip() for line in f if line.strip()]
        print(f"Filtering to {len(project_ids)} sample project IDs.")

    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    index_df = build_index(project_ids, args.process)

    print(f"\nIndex shape: {index_df.shape}")
    print("scan_priority distribution:")
    print(index_df["scan_priority"].value_counts().to_string())
    print("process_type distribution:")
    print(index_df["process_type"].value_counts().to_string())
    proj_dedup = index_df.drop_duplicates("project_id")
    print(f"noi_tier_a_eligible: {proj_dedup['noi_tier_a_eligible'].sum()} projects")
    if "blm_decision_tier_a_eligible" in proj_dedup.columns:
        print(f"blm_decision_tier_a_eligible: {proj_dedup['blm_decision_tier_a_eligible'].sum()} projects")
        print(f"blm_initiation_tier_a_eligible: {proj_dedup['blm_initiation_tier_a_eligible'].sum()} projects")
    if "doe_decision_tier_a_eligible" in proj_dedup.columns:
        print(f"doe_decision_tier_a_eligible: {proj_dedup['doe_decision_tier_a_eligible'].sum()} projects")
        print(f"doe_initiation_tier_a_eligible: {proj_dedup['doe_initiation_tier_a_eligible'].sum()} projects")
    if "doe_cx_tier_a_eligible" in proj_dedup.columns:
        print(f"doe_cx_tier_a_eligible: {proj_dedup['doe_cx_tier_a_eligible'].sum()} projects")

    index_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
