"""
Build a fresh stratified 100-project sample for Phase 2 timeline design.

This script intentionally does not read prior timeline outputs, runbooks, or
extractor artifacts. It uses only the project universe and document metadata
needed to select a balanced sample and describe document burden.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PROJECTS_PATH = ROOT / "phase2/data/analysis/projects_combined.parquet"
DOCUMENTS_PATH = ROOT / "phase2/data/analysis/documents_combined.parquet"
OUTPUT_DIR = ROOT / "phase2/output/deliverable04"

SAMPLE_SEED = 20260527

# Nearly equal process balance, with energy balance inside each process.
PROCESS_ENERGY_QUOTAS = {
    ("CE", "Clean"): 12,
    ("CE", "Fossil"): 11,
    ("CE", "Other"): 11,
    ("EA", "Clean"): 11,
    ("EA", "Fossil"): 11,
    ("EA", "Other"): 11,
    ("EIS", "Clean"): 11,
    ("EIS", "Fossil"): 11,
    ("EIS", "Other"): 11,
}

INITIATION_TITLE_RE = re.compile(
    r"\b("
    r"notice of intent|noi|scoping|application|permit|submitted|received|"
    r"plan of development|pod|right[- ]of[- ]way|row|proposal|request"
    r")\b",
    re.IGNORECASE,
)
DECISION_TITLE_RE = re.compile(
    r"\b("
    r"record of decision|rod|finding of no significant impact|fonsi|"
    r"decision record|decision notice|categorical exclusion|determination|"
    r"approval|approved"
    r")\b",
    re.IGNORECASE,
)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def shorten_text(value: object, max_chars: int = 500) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def flatten_listish(value: object, max_items: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""

    items: list[str]
    if isinstance(value, (list, tuple, set)):
        items = [str(x) for x in value]
    else:
        text = str(value)
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text
        if isinstance(parsed, (list, tuple, set)):
            items = [str(x) for x in parsed]
        else:
            items = [text]

    return clean_text("; ".join(items[:max_items]))


def normalize_energy(value: object) -> str:
    text = clean_text(value)
    return text if text in {"Clean", "Fossil", "Other"} else "Other"


def doc_count_bin(process_type: str, count: object) -> str:
    n = pd.to_numeric(count, errors="coerce")
    if pd.isna(n):
        return "unknown"
    if process_type == "CE":
        if n <= 1:
            return "1_doc"
        if n == 2:
            return "2_docs"
        return "3plus_docs"
    if process_type == "EA":
        if n <= 2:
            return "1_2_docs"
        if n <= 5:
            return "3_5_docs"
        if n <= 14:
            return "6_14_docs"
        return "15plus_docs"
    if process_type == "EIS":
        if n <= 2:
            return "1_2_docs"
        if n <= 10:
            return "3_10_docs"
        if n <= 55:
            return "11_55_docs"
        return "56plus_docs"
    return "unknown"


def title_has(pattern: re.Pattern[str], *values: object) -> bool:
    text = " ".join(clean_text(value) for value in values)
    return bool(pattern.search(text))


def document_rollup() -> pd.DataFrame:
    columns = [
        "project_id",
        "document_id",
        "document_title",
        "file_name",
        "total_pages",
        "main_document",
        "document_type_clean",
        "document_type_category",
        "document_date_from_file_name",
    ]
    docs = pd.read_parquet(DOCUMENTS_PATH, columns=columns)
    docs["total_pages_num"] = pd.to_numeric(docs["total_pages"], errors="coerce").fillna(0)
    docs["is_main_document"] = docs["main_document"].astype(str).str.upper().eq("YES")
    docs["has_filename_date"] = pd.to_datetime(
        docs["document_date_from_file_name"], errors="coerce"
    ).notna()
    docs["has_initiation_title_cue"] = docs.apply(
        lambda row: title_has(INITIATION_TITLE_RE, row["document_title"], row["file_name"]),
        axis=1,
    )
    docs["has_decision_title_cue"] = docs.apply(
        lambda row: title_has(DECISION_TITLE_RE, row["document_title"], row["file_name"]),
        axis=1,
    )

    rollup = (
        docs.groupby("project_id", as_index=False)
        .agg(
            document_rows=("document_id", "size"),
            n_documents=("document_id", "nunique"),
            total_pages=("total_pages_num", "sum"),
            max_doc_pages=("total_pages_num", "max"),
            n_main_documents=("is_main_document", "sum"),
            n_decision_docs=("document_type_category", lambda s: (s == "decision").sum()),
            n_final_docs=("document_type_category", lambda s: (s == "final").sum()),
            n_draft_docs=("document_type_category", lambda s: (s == "draft").sum()),
            n_appendix_docs=("document_type_category", lambda s: (s == "appendix").sum()),
            n_docs_with_filename_date=("has_filename_date", "sum"),
            n_docs_with_initiation_title_cue=("has_initiation_title_cue", "sum"),
            n_docs_with_decision_title_cue=("has_decision_title_cue", "sum"),
        )
    )
    int_cols = [
        "document_rows",
        "n_documents",
        "total_pages",
        "max_doc_pages",
        "n_main_documents",
        "n_decision_docs",
        "n_final_docs",
        "n_draft_docs",
        "n_appendix_docs",
        "n_docs_with_filename_date",
        "n_docs_with_initiation_title_cue",
        "n_docs_with_decision_title_cue",
    ]
    rollup[int_cols] = rollup[int_cols].round().astype("Int64")
    return rollup


def weighted_sample(group: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Random sample that overexposes rare document-burden bins and agencies."""
    doc_bin_counts = group["doc_count_bin"].value_counts()
    agency_counts = group["lead_agency_harmonized_short"].value_counts()
    weights = (
        group["doc_count_bin"].map(lambda x: 1 / doc_bin_counts[x])
        * group["lead_agency_harmonized_short"].map(lambda x: 1 / math.sqrt(agency_counts[x]))
    )
    return group.sample(n=n, weights=weights, random_state=random_state, replace=False)


def build_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    project_columns = [
        "project_id",
        "project_title",
        "project_sector",
        "project_type",
        "project_sponsor",
        "project_location",
        "process_type",
        "lead_agency_harmonized",
        "project_department",
        "project_energy_type",
        "project_state",
        "project_county",
        "project_doc_count",
        "project_has_decision_doc",
        "project_has_final_doc",
        "project_has_draft_doc",
        "project_has_appendix_doc",
        "noi_publication_date",
        "noi_type",
        "noi_subtype",
        "noi_match_status",
        "noi_match_confidence",
        "noa_availability_date",
        "noa_match_status",
    ]
    projects = pd.read_parquet(PROJECTS_PATH, columns=project_columns)
    projects = projects[projects["process_type"].isin(["CE", "EA", "EIS"])].copy()
    projects["project_energy_type"] = projects["project_energy_type"].map(normalize_energy)

    for col in [
        "project_sector",
        "project_type",
        "project_sponsor",
        "project_location",
        "project_state",
        "project_county",
        "lead_agency_harmonized",
    ]:
        projects[f"{col}_short"] = projects[col].map(flatten_listish)
    projects["project_sector_short"] = projects["project_sector_short"].map(
        lambda value: shorten_text(value, max_chars=300)
    )
    projects["project_location_short"] = projects["project_location_short"].map(
        lambda value: shorten_text(value, max_chars=500)
    )

    text_cols = [
        "project_id",
        "project_title",
        "project_department",
        "noi_type",
        "noi_subtype",
        "noi_match_status",
        "noi_match_confidence",
        "noa_match_status",
    ]
    for col in text_cols:
        projects[col] = projects[col].map(clean_text)

    projects = projects.merge(document_rollup(), on="project_id", how="left")
    count_cols = [
        "document_rows",
        "n_documents",
        "total_pages",
        "max_doc_pages",
        "n_main_documents",
        "n_decision_docs",
        "n_final_docs",
        "n_draft_docs",
        "n_appendix_docs",
        "n_docs_with_filename_date",
        "n_docs_with_initiation_title_cue",
        "n_docs_with_decision_title_cue",
    ]
    projects[count_cols] = projects[count_cols].fillna(0).astype("Int64")
    projects["project_doc_count"] = pd.to_numeric(
        projects["project_doc_count"], errors="coerce"
    ).fillna(projects["n_documents"])
    projects["doc_count_bin"] = projects.apply(
        lambda row: doc_count_bin(row["process_type"], row["project_doc_count"]), axis=1
    )

    sampled_parts = []
    for index, ((process_type, energy_type), quota) in enumerate(PROCESS_ENERGY_QUOTAS.items()):
        stratum = projects[
            (projects["process_type"] == process_type)
            & (projects["project_energy_type"] == energy_type)
        ].copy()
        if len(stratum) < quota:
            raise ValueError(
                f"Not enough projects for {process_type}/{energy_type}: "
                f"need {quota}, found {len(stratum)}"
            )
        sampled_parts.append(
            weighted_sample(stratum, n=quota, random_state=SAMPLE_SEED + index)
        )

    sample = pd.concat(sampled_parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=SAMPLE_SEED).reset_index(drop=True)
    sample.insert(0, "sample_id", np.arange(1, len(sample) + 1))
    sample.insert(1, "sample_seed", SAMPLE_SEED)
    sample.insert(
        2,
        "sample_stratum",
        sample["process_type"] + "/" + sample["project_energy_type"] + "/" + sample["doc_count_bin"],
    )

    for col in ["noi_publication_date", "noa_availability_date"]:
        sample[col] = pd.to_datetime(sample[col], errors="coerce").dt.strftime("%Y-%m-%d")

    sample = sample.rename(
        columns={
            "project_type_short": "project_type_summary",
            "project_sector_short": "project_sector_summary",
            "project_sponsor_short": "project_sponsor_summary",
            "project_location_short": "project_location_summary",
            "project_state_short": "project_state_summary",
            "project_county_short": "project_county_summary",
            "lead_agency_harmonized_short": "lead_agency_summary",
            "noi_publication_date": "fr_noi_publication_date",
            "noi_type": "fr_noi_type",
            "noi_subtype": "fr_noi_subtype",
            "noi_match_status": "fr_noi_match_status",
            "noi_match_confidence": "fr_noi_match_confidence",
            "noa_availability_date": "fr_noa_availability_date",
            "noa_match_status": "fr_noa_match_status",
        }
    )
    for col in [
        "gold_initiation_date",
        "gold_initiation_granularity",
        "gold_initiation_type",
        "gold_decision_date",
        "gold_decision_granularity",
        "gold_decision_type",
        "gold_notes",
        "gold_reviewer",
    ]:
        sample[col] = ""

    keep_cols = [
        "sample_id",
        "sample_seed",
        "sample_stratum",
        "project_id",
        "project_title",
        "process_type",
        "project_energy_type",
        "project_sector_summary",
        "project_type_summary",
        "lead_agency_summary",
        "project_department",
        "project_sponsor_summary",
        "project_location_summary",
        "project_state_summary",
        "project_county_summary",
        "project_doc_count",
        "doc_count_bin",
        "n_documents",
        "total_pages",
        "max_doc_pages",
        "n_main_documents",
        "n_decision_docs",
        "n_final_docs",
        "n_draft_docs",
        "n_appendix_docs",
        "n_docs_with_filename_date",
        "n_docs_with_initiation_title_cue",
        "n_docs_with_decision_title_cue",
        "project_has_decision_doc",
        "project_has_final_doc",
        "project_has_draft_doc",
        "project_has_appendix_doc",
        "fr_noi_publication_date",
        "fr_noi_type",
        "fr_noi_subtype",
        "fr_noi_match_status",
        "fr_noi_match_confidence",
        "fr_noa_availability_date",
        "fr_noa_match_status",
        "gold_initiation_date",
        "gold_initiation_granularity",
        "gold_initiation_type",
        "gold_decision_date",
        "gold_decision_granularity",
        "gold_decision_type",
        "gold_notes",
        "gold_reviewer",
    ]
    sample = sample[keep_cols]

    sample_summary = (
        sample.groupby(["process_type", "project_energy_type", "doc_count_bin"], as_index=False)
        .agg(
            n=("project_id", "size"),
            median_docs=("project_doc_count", "median"),
            median_pages=("total_pages", "median"),
            p90_pages=("total_pages", lambda s: s.quantile(0.9)),
            n_with_fr_noi=("fr_noi_publication_date", lambda s: s.notna().sum()),
            n_with_fr_noa=("fr_noa_availability_date", lambda s: s.notna().sum()),
            n_with_initiation_title_cue=("n_docs_with_initiation_title_cue", lambda s: (s > 0).sum()),
            n_with_decision_title_cue=("n_docs_with_decision_title_cue", lambda s: (s > 0).sum()),
        )
        .sort_values(["process_type", "project_energy_type", "doc_count_bin"])
    )
    numeric_cols = [
        "median_docs",
        "median_pages",
        "p90_pages",
        "n_with_fr_noi",
        "n_with_fr_noa",
        "n_with_initiation_title_cue",
        "n_with_decision_title_cue",
    ]
    sample_summary[numeric_cols] = sample_summary[numeric_cols].round(1)

    return sample, sample_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample, sample_summary = build_sample()
    sample.to_csv(OUTPUT_DIR / "timeline_sample100.csv", index=False)
    sample_summary.to_csv(OUTPUT_DIR / "timeline_sample100_summary.csv", index=False)

    print(f"Wrote {len(sample)} sampled projects")
    print(sample.groupby(["process_type", "project_energy_type"]).size().to_string())
    print()
    print("Sample burden summary:")
    print(
        sample.groupby("process_type")
        .agg(
            n=("project_id", "size"),
            median_docs=("project_doc_count", "median"),
            median_pages=("total_pages", "median"),
            max_pages=("total_pages", "max"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
