"""
Cooperating-agency signal extraction for NEPA projects.

Purpose:
- Detect projects with multi-agency coordination signals in document page text.
- Keep this extraction separate from extract_data.py so the base pipeline stays fast.
- Write a project-level output that can be joined in deliverable analyses.

Inputs (must already exist from extract_data.py):
- data/analysis/projects_combined.parquet
- data/analysis/documents_combined.parquet
- data/processed/{ea,eis,ce}/pages.parquet

Outputs:
- data/analysis/coagency_projects.parquet
- data/analysis/coagency_hits.parquet (QA/evidence rows)

Usage:
  python code/extract/extract_coagency.py --run
  python code/extract/extract_coagency.py --run --include-secondary
  python code/extract/extract_coagency.py --run --all-projects --max-pages-per-project 40
"""

from __future__ import annotations

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd


# --------------------------
# PATHS
# --------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "phase1" / "data" / "analysis"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEFAULT_PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
DEFAULT_DOCUMENTS_PATH = ANALYSIS_DIR / "documents_combined.parquet"
DEFAULT_EA_PAGES_PATH = PROCESSED_DIR / "ea" / "pages.parquet"
DEFAULT_EIS_PAGES_PATH = PROCESSED_DIR / "eis" / "pages.parquet"
DEFAULT_CE_PAGES_PATH = PROCESSED_DIR / "ce" / "pages.parquet"

DEFAULT_OUTPUT_PROJECTS = ANALYSIS_DIR / "coagency_projects.parquet"
DEFAULT_OUTPUT_HITS = ANALYSIS_DIR / "coagency_hits.parquet"


# --------------------------
# CUES
# --------------------------

@dataclass(frozen=True)
class Cue:
    name: str
    pattern: str
    tier: str  # primary or secondary


PRIMARY_CUES: List[Cue] = [
    Cue(
        name="cooperating_agency",
        pattern=r"(?i)cooperat(?:ing|e|ion|ive)?[^a-z]{0,12}agenc(?:y|ies)",
        tier="primary",
    ),
    Cue(
        name="cooperating_agencies_colon",
        pattern=r"(?i)cooperating\s+agenc(?:y|ies)\s*[:\-]",
        tier="primary",
    ),
    Cue(
        name="lead_and_cooperating_pair",
        pattern=r"(?i)lead\s+agenc(?:y|ies).{0,120}cooperating\s+agenc(?:y|ies)",
        tier="primary",
    ),
    Cue(
        name="joint_lead_agency",
        pattern=r"(?i)joint\s+lead\s+agenc(?:y|ies)",
        tier="primary",
    ),
    Cue(
        name="co_lead_agency",
        pattern=r"(?i)co\s*[- ]\s*lead\s+agenc(?:y|ies)",
        tier="primary",
    ),
]

SECONDARY_CUES: List[Cue] = [
    Cue(
        name="participating_agencies",
        pattern=r"(?i)participating\s+agenc(?:y|ies)",
        tier="secondary",
    ),
    Cue(
        name="consulting_agencies",
        pattern=r"(?i)consult(?:ing|ed)?\s+agenc(?:y|ies)",
        tier="secondary",
    ),
    Cue(
        name="in_cooperation_with",
        pattern=r"(?i)in\s+cooperation\s+with",
        tier="secondary",
    ),
]


# --------------------------
# FILTERS / RULES
# --------------------------

# Filename-based auxiliary docs to deprioritize/exclude for high-confidence flags.
AUX_FILENAME_PATTERN = r"(?i)(comment|newsletter|letter|response|appendix|attachment|exhibit)"

# "Core" document categories as defined in documents_combined by extract_data.py.
CORE_DOC_CATEGORIES = {"decision", "final", "draft"}


def _sql_escape(pattern: str) -> str:
    return pattern.replace("'", "''")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _build_scan_query(
    projects_path: Path,
    documents_path: Path,
    ea_pages_path: Path,
    eis_pages_path: Path,
    ce_pages_path: Path,
    cues: List[Cue],
    max_pages_per_project: int,
    clean_energy_only: bool,
    snippet_chars: int,
) -> str:
    cue_exprs = []
    cue_cols = []
    for cue in cues:
        col = f"cue_{cue.name}"
        cue_cols.append(col)
        cue_exprs.append(
            f"regexp_matches(coalesce(page_text, ''), '{_sql_escape(cue.pattern)}') AS {col}"
        )

    any_hit_clause = " OR ".join(cue_cols) if cue_cols else "FALSE"
    cue_expr_sql = ",\n                ".join(cue_exprs)

    projects_path_sql = projects_path.as_posix().replace("'", "''")
    documents_path_sql = documents_path.as_posix().replace("'", "''")
    ea_pages_path_sql = ea_pages_path.as_posix().replace("'", "''")
    eis_pages_path_sql = eis_pages_path.as_posix().replace("'", "''")
    ce_pages_path_sql = ce_pages_path.as_posix().replace("'", "''")
    clean_filter = "WHERE project_energy_type = 'Clean'" if clean_energy_only else ""

    return f"""
    WITH target_projects AS (
        SELECT
            project_id,
            dataset_source,
            process_type,
            project_energy_type,
            project_multi_department
        FROM read_parquet('{projects_path_sql}')
        {clean_filter}
    ),
    target_docs AS (
        SELECT
            d.project_id,
            d.dataset_source,
            d.document_id,
            d.document_type_clean,
            d.document_type_category,
            d.file_name,
            d.main_document
        FROM read_parquet('{documents_path_sql}') d
        INNER JOIN target_projects p
            ON d.project_id = p.project_id
           AND d.dataset_source = p.dataset_source
    ),
    docs_scored AS (
        SELECT
            project_id,
            dataset_source,
            document_id,
            document_type_clean,
            document_type_category,
            file_name,
            main_document,
            CASE
                WHEN upper(coalesce(main_document, '')) = 'YES' THEN TRUE
                ELSE FALSE
            END AS main_document_yes,
            CASE
                WHEN coalesce(document_type_category, '') IN ('decision', 'final', 'draft') THEN TRUE
                ELSE FALSE
            END AS document_is_core,
            regexp_matches(lower(coalesce(file_name, '')), '{_sql_escape(AUX_FILENAME_PATTERN)}') AS document_is_auxiliary,
            CASE
                WHEN upper(coalesce(main_document, '')) = 'YES' THEN 0
                WHEN coalesce(document_type_category, '') IN ('decision', 'final', 'draft') THEN 1
                WHEN coalesce(document_type_category, '') = 'other' THEN 2
                ELSE 3
            END AS doc_priority
        FROM target_docs
    ),
    all_pages AS (
        SELECT
            'EA' AS dataset_source,
            document_id,
            try_cast(page_number AS INTEGER) AS page_number,
            page_text
        FROM read_parquet('{ea_pages_path_sql}')
        UNION ALL
        SELECT
            'EIS' AS dataset_source,
            document_id,
            try_cast(page_number AS INTEGER) AS page_number,
            page_text
        FROM read_parquet('{eis_pages_path_sql}')
        UNION ALL
        SELECT
            'CE' AS dataset_source,
            document_id,
            try_cast(page_number AS INTEGER) AS page_number,
            page_text
        FROM read_parquet('{ce_pages_path_sql}')
    ),
    joined AS (
        SELECT
            d.project_id,
            d.dataset_source,
            d.document_id,
            d.document_type_clean,
            d.document_type_category,
            d.file_name,
            d.main_document_yes,
            d.document_is_core,
            d.document_is_auxiliary,
            d.doc_priority,
            p.page_number,
            p.page_text
        FROM docs_scored d
        INNER JOIN all_pages p
            ON d.dataset_source = p.dataset_source
           AND d.document_id = p.document_id
        WHERE p.page_number IS NOT NULL
    ),
    ordered AS (
        SELECT
            *,
            row_number() OVER (
                PARTITION BY project_id, dataset_source
                ORDER BY doc_priority, page_number, document_id
            ) AS project_page_rank
        FROM joined
    ),
    limited AS (
        SELECT *
        FROM ordered
        WHERE project_page_rank <= {int(max_pages_per_project)}
    ),
    cue_scan AS (
        SELECT
            project_id,
            dataset_source,
            document_id,
            document_type_clean,
            document_type_category,
            file_name,
            main_document_yes,
            document_is_core,
            document_is_auxiliary,
            page_number,
            project_page_rank,
            substr(regexp_replace(coalesce(page_text, ''), '\\s+', ' ', 'g'), 1, {int(snippet_chars)}) AS snippet,
            {cue_expr_sql}
        FROM limited
    )
    SELECT *
    FROM cue_scan
    WHERE {any_hit_clause}
    ORDER BY project_id, dataset_source, project_page_rank, page_number
    """


def _json_sorted_unique(values: List[str]) -> str:
    uniq = sorted(set(v for v in values if isinstance(v, str) and v))
    return json.dumps(uniq)


def run_extraction(args: argparse.Namespace) -> None:
    _require_file(args.projects, "projects parquet")
    _require_file(args.documents, "documents parquet")
    _require_file(args.ea_pages, "EA pages parquet")
    _require_file(args.eis_pages, "EIS pages parquet")
    _require_file(args.ce_pages, "CE pages parquet")

    cues = list(PRIMARY_CUES)
    if args.include_secondary:
        cues.extend(SECONDARY_CUES)
    cue_tier_map: Dict[str, str] = {f"cue_{c.name}": c.tier for c in cues}

    clean_energy_only = not args.all_projects
    print("=== Coagency Extraction ===")
    print(f"Projects input: {args.projects}")
    print(f"Documents input: {args.documents}")
    print(f"Page inputs: EA={args.ea_pages}, EIS={args.eis_pages}, CE={args.ce_pages}")
    print(f"Scope: {'clean-energy only' if clean_energy_only else 'all projects'}")
    print(f"Cues: {len(cues)} ({'primary+secondary' if args.include_secondary else 'primary only'})")
    print(f"Max pages per project scan: {args.max_pages_per_project}")

    scan_query = _build_scan_query(
        projects_path=args.projects,
        documents_path=args.documents,
        ea_pages_path=args.ea_pages,
        eis_pages_path=args.eis_pages,
        ce_pages_path=args.ce_pages,
        cues=cues,
        max_pages_per_project=args.max_pages_per_project,
        clean_energy_only=clean_energy_only,
        snippet_chars=args.max_snippet_chars,
    )

    con = duckdb.connect()
    try:
        hits_wide = con.execute(scan_query).df()

        base_filter = "WHERE project_energy_type = 'Clean'" if clean_energy_only else ""
        base_query = f"""
        SELECT
            project_id,
            dataset_source,
            process_type,
            project_energy_type,
            project_multi_department
        FROM read_parquet('{args.projects.as_posix().replace("'", "''")}')
        {base_filter}
        """
        base_projects = con.execute(base_query).df()
    finally:
        con.close()

    if hits_wide.empty:
        print("No coagency cues detected in scoped pages.")
        out_projects = base_projects.copy()
        out_projects["project_has_coagency_signal_any"] = False
        out_projects["project_has_coagency_signal_any_nonaux"] = False
        out_projects["project_has_coagency_signal_high_conf"] = False
        out_projects["project_has_coagency_signal_primary"] = False
        out_projects["project_has_coagency_signal_secondary"] = False
        out_projects["project_coagency_signal_source"] = np.where(
            out_projects["project_multi_department"].fillna(False),
            "lead_agency_metadata",
            "none",
        )
        out_projects["project_coagency_first_hit_page"] = pd.NA
        out_projects["project_coagency_first_hit_file_name"] = pd.NA
        out_projects["project_coagency_first_hit_document_type_clean"] = pd.NA
        out_projects["project_coagency_first_hit_document_type_category"] = pd.NA
        out_projects["project_coagency_first_hit_is_aux"] = pd.NA
        out_projects["project_coagency_first_hit_main_document"] = pd.NA
        out_projects["project_coagency_first_hit_cue"] = pd.NA
        out_projects["project_coagency_cues_all"] = "[]"
        out_projects["project_coagency_cues_primary"] = "[]"
        out_projects["project_coagency_cues_secondary"] = "[]"
        out_projects["project_multi_agency"] = out_projects["project_multi_department"].fillna(False)

        args.output_projects.parent.mkdir(parents=True, exist_ok=True)
        out_projects.to_parquet(args.output_projects, index=False)
        print(f"Saved: {args.output_projects}")

        if args.write_hits:
            empty_hits = pd.DataFrame(
                columns=[
                    "project_id",
                    "dataset_source",
                    "process_type",
                    "document_id",
                    "document_type_clean",
                    "document_type_category",
                    "file_name",
                    "main_document_yes",
                    "document_is_core",
                    "document_is_auxiliary",
                    "page_number",
                    "project_page_rank",
                    "snippet",
                    "cue_name",
                    "cue_tier",
                    "is_high_conf_match",
                ]
            )
            args.output_hits.parent.mkdir(parents=True, exist_ok=True)
            empty_hits.to_parquet(args.output_hits, index=False)
            print(f"Saved: {args.output_hits}")
        return

    # Convert wide cue booleans -> long evidence rows.
    base_cols = [
        "project_id",
        "dataset_source",
        "document_id",
        "document_type_clean",
        "document_type_category",
        "file_name",
        "main_document_yes",
        "document_is_core",
        "document_is_auxiliary",
        "page_number",
        "project_page_rank",
        "snippet",
    ]

    hit_frames = []
    for cue_col, cue_tier in cue_tier_map.items():
        if cue_col not in hits_wide.columns:
            continue
        cue_name = cue_col.replace("cue_", "")
        matched = hits_wide.loc[hits_wide[cue_col], base_cols].copy()
        if matched.empty:
            continue
        matched["cue_name"] = cue_name
        matched["cue_tier"] = cue_tier
        hit_frames.append(matched)

    if not hit_frames:
        # Should be rare if query filter is correct, but keep safe.
        hits_long = pd.DataFrame(columns=base_cols + ["cue_name", "cue_tier"])
    else:
        hits_long = pd.concat(hit_frames, ignore_index=True)

    # Add process_type for hit rows (small join with base projects).
    hits_long = hits_long.merge(
        base_projects[["project_id", "dataset_source", "process_type"]],
        on=["project_id", "dataset_source"],
        how="left",
    )

    hits_long = hits_long.sort_values(
        by=["project_id", "dataset_source", "project_page_rank", "page_number", "cue_name"],
        kind="stable",
    ).reset_index(drop=True)

    # High-confidence hit definition used for analysis-level multi-agency flag.
    hits_long["is_high_conf_match"] = (
        (hits_long["cue_tier"] == "primary")
        & (~hits_long["document_is_auxiliary"].fillna(False))
        & (
            hits_long["document_is_core"].fillna(False)
            | hits_long["main_document_yes"].fillna(False)
        )
        & (hits_long["page_number"] <= args.high_conf_max_page)
    )

    group_cols = ["project_id", "dataset_source"]

    cue_agg = (
        hits_long.groupby(group_cols)
        .agg(
            project_has_coagency_signal_any=("cue_name", lambda s: True),
            project_has_coagency_signal_any_nonaux=("document_is_auxiliary", lambda s: (~s.fillna(False)).any()),
            project_has_coagency_signal_high_conf=("is_high_conf_match", "any"),
            project_has_coagency_signal_primary=("cue_tier", lambda s: (s == "primary").any()),
            project_has_coagency_signal_secondary=("cue_tier", lambda s: (s == "secondary").any()),
            project_coagency_cues_all=("cue_name", lambda s: _json_sorted_unique(list(s))),
            project_coagency_cues_primary=(
                "cue_name",
                lambda s: _json_sorted_unique(
                    [
                        cue
                        for cue in s
                        if cue_tier_map.get(f"cue_{cue}") == "primary"
                    ]
                ),
            ),
            project_coagency_cues_secondary=(
                "cue_name",
                lambda s: _json_sorted_unique(
                    [
                        cue
                        for cue in s
                        if cue_tier_map.get(f"cue_{cue}") == "secondary"
                    ]
                ),
            ),
        )
        .reset_index()
    )

    first_any = (
        hits_long.drop_duplicates(subset=group_cols, keep="first")
        .loc[
            :,
            group_cols
            + [
                "page_number",
                "file_name",
                "document_type_clean",
                "document_type_category",
                "document_is_auxiliary",
                "main_document_yes",
                "cue_name",
            ],
        ]
        .rename(
            columns={
                "page_number": "project_coagency_first_hit_page",
                "file_name": "project_coagency_first_hit_file_name",
                "document_type_clean": "project_coagency_first_hit_document_type_clean",
                "document_type_category": "project_coagency_first_hit_document_type_category",
                "document_is_auxiliary": "project_coagency_first_hit_is_aux",
                "main_document_yes": "project_coagency_first_hit_main_document",
                "cue_name": "project_coagency_first_hit_cue",
            }
        )
    )

    out_projects = (
        base_projects.merge(cue_agg, on=group_cols, how="left")
        .merge(first_any, on=group_cols, how="left")
        .copy()
    )

    bool_cols = [
        "project_multi_department",
        "project_has_coagency_signal_any",
        "project_has_coagency_signal_any_nonaux",
        "project_has_coagency_signal_high_conf",
        "project_has_coagency_signal_primary",
        "project_has_coagency_signal_secondary",
    ]
    for col in bool_cols:
        if col in out_projects.columns:
            out_projects[col] = (
                out_projects[col]
                .astype("boolean")
                .fillna(False)
                .astype(bool)
            )

    list_cols = [
        "project_coagency_cues_all",
        "project_coagency_cues_primary",
        "project_coagency_cues_secondary",
    ]
    for col in list_cols:
        out_projects[col] = out_projects[col].fillna("[]")

    out_projects["project_coagency_signal_source"] = np.select(
        [
            out_projects["project_multi_department"] & out_projects["project_has_coagency_signal_high_conf"],
            out_projects["project_multi_department"] & (~out_projects["project_has_coagency_signal_high_conf"]),
            (~out_projects["project_multi_department"]) & out_projects["project_has_coagency_signal_high_conf"],
            (~out_projects["project_multi_department"])
            & (~out_projects["project_has_coagency_signal_high_conf"])
            & out_projects["project_has_coagency_signal_any"],
        ],
        [
            "both",
            "lead_agency_metadata",
            "coagency_text_high_conf",
            "coagency_text_low_conf",
        ],
        default="none",
    )

    out_projects["project_multi_agency"] = (
        out_projects["project_multi_department"] | out_projects["project_has_coagency_signal_high_conf"]
    )

    # Write outputs.
    args.output_projects.parent.mkdir(parents=True, exist_ok=True)
    out_projects.to_parquet(args.output_projects, index=False)

    if args.write_hits:
        args.output_hits.parent.mkdir(parents=True, exist_ok=True)
        hits_long.to_parquet(args.output_hits, index=False)

    # Console summary.
    n_projects = len(out_projects)
    n_any = int(out_projects["project_has_coagency_signal_any"].sum())
    n_high = int(out_projects["project_has_coagency_signal_high_conf"].sum())
    n_meta = int(out_projects["project_multi_department"].sum())
    n_union = int(out_projects["project_multi_agency"].sum())

    print("\n=== Summary ===")
    print(f"Projects in scope: {n_projects:,}")
    print(f"Projects with any coagency text signal: {n_any:,}")
    print(f"Projects with high-confidence coagency text signal: {n_high:,}")
    print(f"Projects with metadata multi-department flag: {n_meta:,}")
    print(f"Projects flagged as multi-agency (metadata OR high-confidence text): {n_union:,}")
    print(f"\nSaved: {args.output_projects}")
    if args.write_hits:
        print(f"Saved: {args.output_hits}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract cooperating-agency signals with DuckDB.")
    parser.add_argument("--run", action="store_true", help="Run extraction.")
    parser.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS_PATH, help="projects_combined path")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH, help="documents_combined path")
    parser.add_argument("--ea-pages", type=Path, default=DEFAULT_EA_PAGES_PATH, help="EA pages parquet path")
    parser.add_argument("--eis-pages", type=Path, default=DEFAULT_EIS_PAGES_PATH, help="EIS pages parquet path")
    parser.add_argument("--ce-pages", type=Path, default=DEFAULT_CE_PAGES_PATH, help="CE pages parquet path")
    parser.add_argument(
        "--output-projects",
        type=Path,
        default=DEFAULT_OUTPUT_PROJECTS,
        help="Project-level output parquet path",
    )
    parser.add_argument(
        "--output-hits",
        type=Path,
        default=DEFAULT_OUTPUT_HITS,
        help="Hit-level evidence output parquet path",
    )
    parser.add_argument(
        "--max-pages-per-project",
        type=int,
        default=25,
        help="Maximum ordered pages to scan per project (default: 25).",
    )
    parser.add_argument(
        "--high-conf-max-page",
        type=int,
        default=25,
        help="Max page number allowed for high-confidence flag (default: 25).",
    )
    parser.add_argument(
        "--max-snippet-chars",
        type=int,
        default=320,
        help="Snippet length for evidence rows (default: 320).",
    )
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Include lower-confidence cues (participating/consulting/in cooperation with).",
    )
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Scan all projects instead of clean-energy-only scope.",
    )
    parser.add_argument(
        "--no-write-hits",
        dest="write_hits",
        action="store_false",
        help="Skip writing the hit-level output parquet.",
    )
    parser.set_defaults(write_hits=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    if not cli_args.run:
        parser.print_help()
        raise SystemExit(0)
    run_extraction(cli_args)
