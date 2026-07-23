import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from common import (
    ANALYSIS_DIR,
    D03_REVIEWS,
    D6_ANALYSIS_DIR,
    D6_OUTPUT_DIR,
    D6_VALIDATION_DIR,
    DOCUMENTS_COMBINED,
    EA_PAGES,
    PROJECTS_COMBINED,
    TIMELINE_INDEX,
    ensure_d6_dirs,
    input_hashes,
    normalize_space,
    utc_now,
    write_parquet,
)


INVENTORY = D6_ANALYSIS_DIR / "fonsi_document_inventory.parquet"
PROJECT_INVENTORY = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
SECTION_MANIFEST = D6_ANALYSIS_DIR / "fonsi_section_manifest.parquet"
ROLE_LABELS = D6_VALIDATION_DIR / "fonsi_document_role_labels.parquet"
ROLE_REVIEW = D6_OUTPUT_DIR / "fonsi_document_role_review.csv"


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def classify_role(row: pd.Series) -> tuple[str, str]:
    text = " ".join(
        normalize_space(row.get(col, ""))
        for col in ("document_type", "document_title", "file_name", "document_type_category")
    ).lower()
    explicit = normalize_space(row.get("document_type", "")).lower() == "fonsi"
    source = "explicit_document_type" if explicit else "cleaned_or_inferred_metadata"

    if re.search(r"\b(?:appendix|appendices|attachment|exhibit|supporting document)\b", text):
        return "attachment_or_appendix", source
    if re.search(r"\b(?:draft|preliminary|working draft)\b", text):
        return "draft_fonsi", source
    if re.search(r"\b(?:decision notice|decision record|notice of decision)\b", text):
        return "fonsi_decision_notice", source
    if re.search(r"\b(?:final environmental assessment|final ea)\b", text) and re.search(
        r"\b(?:fonsi|finding of no significant impact)\b", text
    ):
        return "combined_final_ea_fonsi", source
    if re.search(r"\b(?:fonsi|finding of no significant impact)\b", text) or explicit:
        return "standalone_fonsi", source
    return "uncertain", source


def fetch_fonsi_metadata(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    timeline_join = ""
    timeline_cols = """
        CAST(NULL AS VARCHAR) AS blm_decision_date,
        CAST(NULL AS VARCHAR) AS blm_decision_date_type,
        CAST(NULL AS VARCHAR) AS doe_decision_date,
        CAST(NULL AS VARCHAR) AS doe_decision_date_type,
        FALSE AS timeline_metadata_available
    """
    if TIMELINE_INDEX.exists():
        timeline_join = f"""
        LEFT JOIN (
            SELECT
                document_id,
                any_value(blm_decision_date) AS blm_decision_date,
                any_value(blm_decision_date_type) AS blm_decision_date_type,
                any_value(doe_decision_date) AS doe_decision_date,
                any_value(doe_decision_date_type) AS doe_decision_date_type
            FROM read_parquet('{sql_path(TIMELINE_INDEX)}')
            GROUP BY document_id
        ) t USING (document_id)
        """
        timeline_cols = """
        t.blm_decision_date,
        t.blm_decision_date_type,
        t.doe_decision_date,
        t.doe_decision_date_type,
        TRUE AS timeline_metadata_available
        """

    return conn.execute(
        f"""
        WITH review AS (
            SELECT
                project_id,
                any_value(project_energy_type) AS d3_project_energy_type,
                any_value(energy_group) AS d3_energy_group,
                any_value(tech_group) AS d3_tech_group,
                any_value(lead_agency_harmonized) AS d3_lead_agency_harmonized
            FROM read_parquet('{sql_path(D03_REVIEWS)}')
            WHERE process_type = 'EA'
            GROUP BY project_id
        ),
        projects AS (
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY project_id
                        ORDER BY
                            CASE WHEN process_type = 'EA' THEN 0 ELSE 1 END,
                            CASE WHEN dataset_source = 'EA' THEN 0 ELSE 1 END
                    ) AS rn
                FROM read_parquet('{sql_path(PROJECTS_COMBINED)}')
            )
            WHERE rn = 1
        )
        SELECT
            d.*,
            p.project_title,
            p.project_type,
            p.project_description,
            p.project_sector,
            p.project_energy_type,
            p.project_state,
            p.project_county,
            p.lead_agency_harmonized AS project_lead_agency_harmonized,
            COALESCE(r.d3_energy_group, p.project_energy_type, 'Other') AS energy_group,
            COALESCE(r.d3_tech_group, 'Other / Unclassified') AS tech_group,
            COALESCE(
                r.d3_lead_agency_harmonized,
                p.lead_agency_harmonized,
                p.lead_agency,
                'Unknown'
            ) AS lead_agency_harmonized,
            {timeline_cols}
        FROM read_parquet('{sql_path(DOCUMENTS_COMBINED)}') d
        LEFT JOIN projects p USING (project_id)
        LEFT JOIN review r USING (project_id)
        {timeline_join}
        WHERE d.document_type_clean = 'FONSI'
          AND COALESCE(d.document_type_clean, '') <> 'ROD'
        """
    ).fetchdf()


def fetch_ea_text_stats(
    conn: duckdb.DuckDBPyConnection, inventory: pd.DataFrame
) -> pd.DataFrame:
    targets = inventory.loc[
        inventory["dataset_source"].eq("EA"), ["document_id"]
    ].drop_duplicates()
    conn.register("fonsi_target_documents", targets)
    return conn.execute(
        f"""
        SELECT
            p.document_id,
            count(*) AS extracted_page_count,
            sum(length(COALESCE(p.page_text, ''))) AS extracted_text_chars,
            sha256(
                regexp_replace(
                    string_agg(
                        COALESCE(p.page_text, '')
                        ORDER BY COALESCE(
                            TRY_CAST(regexp_extract(CAST(p.page_number AS VARCHAR), '(\\d+)', 1) AS INTEGER),
                            1000000000
                        )
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            ) AS normalized_text_sha256
        FROM read_parquet('{sql_path(EA_PAGES)}') p
        JOIN fonsi_target_documents t USING (document_id)
        GROUP BY p.document_id
        """
    ).fetchdf()


def annotate_inventory(inventory: pd.DataFrame, text_stats: pd.DataFrame, run_at: str) -> pd.DataFrame:
    out = inventory.merge(text_stats, how="left", on="document_id")
    roles = out.apply(classify_role, axis=1, result_type="expand")
    out["document_role"] = roles[0]
    out["document_role_source"] = roles[1]
    out["stage_a_ea_source"] = out["dataset_source"].eq("EA")
    out["has_extracted_text"] = out["extracted_text_chars"].fillna(0).gt(0)
    out["normalized_text_sha256"] = out["normalized_text_sha256"].fillna("")

    out["normalized_title"] = (
        out["document_title"].fillna("").map(normalize_space).str.lower()
    )
    valid_hash = out["normalized_text_sha256"].ne("")
    out["duplicate_of_document_id"] = ""
    for _, group in out.loc[valid_hash].groupby("normalized_text_sha256", sort=False):
        representative = sorted(group["document_id"].astype(str))[0]
        out.loc[group.index, "duplicate_of_document_id"] = representative
    out["is_exact_duplicate"] = (
        out["duplicate_of_document_id"].ne("")
        & out["duplicate_of_document_id"].ne(out["document_id"].astype(str))
    )

    near_key = (
        out["project_id"].fillna("")
        + "|"
        + out["normalized_title"]
        + "|"
        + out["total_pages"].fillna(-1).astype(str)
    )
    out["near_duplicate_review"] = near_key.duplicated(keep=False) & ~out["is_exact_duplicate"]

    role_score = {
        "standalone_fonsi": 70,
        "combined_final_ea_fonsi": 65,
        "fonsi_decision_notice": 55,
        "uncertain": 25,
        "draft_fonsi": -30,
        "attachment_or_appendix": -40,
    }
    out["canonical_score"] = out["document_role"].map(role_score).fillna(0)
    out["canonical_score"] += out["main_document"].fillna("").ne("NO").astype(int) * 10
    out["canonical_score"] += out["has_extracted_text"].astype(int) * 8
    out["canonical_score"] += out["document_role_source"].eq("explicit_document_type").astype(int) * 3
    out["canonical_score"] += out["is_exact_duplicate"].astype(int) * -2
    out["canonical_fonsi"] = False
    stage_a = out.loc[out["stage_a_ea_source"]].copy()
    if not stage_a.empty:
        ranked = stage_a.sort_values(
            ["project_id", "canonical_score", "total_pages", "document_id"],
            ascending=[True, False, False, True],
            na_position="last",
        )
        canonical_idx = ranked.groupby("project_id", sort=False).head(1).index
        out.loc[canonical_idx, "canonical_fonsi"] = True
    out["fonsi_inventory_run_at"] = run_at
    return out


def build_project_inventory(inventory: pd.DataFrame, run_at: str) -> pd.DataFrame:
    canonical = inventory.loc[inventory["canonical_fonsi"]].copy()
    canonical["fonsi_document_count"] = canonical["project_id"].map(
        inventory.loc[inventory["stage_a_ea_source"]].groupby("project_id").size()
    )
    canonical["supporting_fonsi_count"] = canonical["fonsi_document_count"] - 1
    canonical["fonsi_project_inventory_run_at"] = run_at
    return canonical.rename(columns={"document_id": "canonical_fonsi_document_id"})


def build_section_manifest(
    conn: duckdb.DuckDBPyConnection, inventory: pd.DataFrame, run_at: str
) -> pd.DataFrame:
    selected_fonsis = inventory.loc[
        inventory["stage_a_ea_source"]
        & (
            inventory["canonical_fonsi"]
            | (
                ~inventory["document_role"].isin(["draft_fonsi", "attachment_or_appendix"])
                & ~inventory["is_exact_duplicate"]
            )
        )
    ].copy()
    selected_fonsis["manifest_role"] = selected_fonsis["canonical_fonsi"].map(
        {True: "canonical_fonsi", False: "supporting_fonsi"}
    )
    projects = selected_fonsis[["project_id"]].drop_duplicates()
    conn.register("fonsi_projects", projects)
    linked = conn.execute(
        f"""
        SELECT d.*
        FROM read_parquet('{sql_path(DOCUMENTS_COMBINED)}') d
        JOIN fonsi_projects p USING (project_id)
        WHERE d.dataset_source = 'EA'
          AND d.document_type_clean = 'EA'
          AND COALESCE(d.main_document, 'YES') <> 'NO'
        """
    ).fetchdf()

    meta_cols = [
        "project_id",
        "energy_group",
        "tech_group",
        "lead_agency_harmonized",
    ]
    meta = (
        inventory[meta_cols]
        .drop_duplicates(subset=["project_id"])
        .set_index("project_id")
    )
    if not linked.empty:
        linked = linked.join(meta, on="project_id")
        linked["manifest_role"] = "linked_ea"

    cols = [
        "document_id",
        "project_id",
        "energy_group",
        "tech_group",
        "lead_agency_harmonized",
        "document_title",
        "main_document",
        "manifest_role",
    ]
    frames = [selected_fonsis[cols]]
    if not linked.empty:
        frames.append(linked[cols])
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest.drop_duplicates(subset=["document_id"], keep="first")
    manifest["process_type"] = "EA"
    manifest["manifest_run_at"] = run_at
    return manifest


def write_role_review(inventory: pd.DataFrame, n: int) -> None:
    qa = inventory.copy()
    qa["qa_stratum"] = qa["document_role"].fillna("unknown") + "|" + qa[
        "energy_group"
    ].fillna("unknown")
    per_stratum = max(1, n // max(1, qa["qa_stratum"].nunique()))
    qa = pd.concat(
        [
            group.sample(min(len(group), per_stratum), random_state=42)
            for _, group in qa.groupby("qa_stratum")
        ],
        ignore_index=True,
    )
    if len(qa) < n:
        remaining = inventory.loc[~inventory["document_id"].isin(qa["document_id"])]
        if not remaining.empty:
            qa = pd.concat(
                [qa, remaining.sample(min(n - len(qa), len(remaining)), random_state=42)],
                ignore_index=True,
            )
    review_cols = [
        "project_id",
        "document_id",
        "dataset_source",
        "energy_group",
        "lead_agency_harmonized",
        "document_role",
        "document_role_source",
        "canonical_fonsi",
        "canonical_score",
        "is_exact_duplicate",
        "near_duplicate_review",
        "document_title",
        "file_name",
        "total_pages",
        "reviewed_document_role",
        "reviewed_canonical_fonsi",
        "review_notes",
    ]
    for col in ("reviewed_document_role", "reviewed_canonical_fonsi", "review_notes"):
        qa[col] = ""
    qa[review_cols].head(n).to_csv(ROLE_REVIEW, index=False)
    if not ROLE_LABELS.exists():
        write_parquet(pd.DataFrame(columns=review_cols), ROLE_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the D6 role-aware FONSI inventory.")
    parser.add_argument("--qa-sample", type=int, default=100)
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Skip full-file hashes for faster smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_d6_dirs()
    run_at = utc_now()
    conn = duckdb.connect()
    inventory = fetch_fonsi_metadata(conn)
    text_stats = fetch_ea_text_stats(conn, inventory)
    inventory = annotate_inventory(inventory, text_stats, run_at)
    project_inventory = build_project_inventory(inventory, run_at)
    manifest = build_section_manifest(conn, inventory, run_at)

    hash_paths = [PROJECTS_COMBINED, DOCUMENTS_COMBINED, D03_REVIEWS, EA_PAGES]
    if TIMELINE_INDEX.exists():
        hash_paths.append(TIMELINE_INDEX)
    hashes = "skipped_for_smoke_test" if args.skip_input_hashes else input_hashes(hash_paths)
    optional_sources = json.dumps({"timeline_document_index": TIMELINE_INDEX.exists()})
    for frame in (inventory, project_inventory, manifest):
        frame["input_hashes"] = hashes
        frame["optional_sources"] = optional_sources

    write_parquet(inventory, INVENTORY)
    write_parquet(project_inventory, PROJECT_INVENTORY)
    write_parquet(manifest, SECTION_MANIFEST)
    write_role_review(inventory, args.qa_sample)
    print(
        f"wrote {len(inventory):,} FONSI documents, "
        f"{len(project_inventory):,} EA-source canonical projects, and "
        f"{len(manifest):,} target section documents"
    )


if __name__ == "__main__":
    main()
