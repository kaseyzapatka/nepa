import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import json
import re

import pandas as pd

from common import (
    D03_REVIEWS,
    D6_ANALYSIS_DIR,
    D6_OUTPUT_DIR,
    PROJECTS_COMBINED,
    ensure_d6_dirs,
    normalize_space,
    utc_now,
    write_parquet,
)


TAXONOMY_PATH = D6_ANALYSIS_DIR / "fonsi_archetype_taxonomy.parquet"
ASSIGNMENTS_PATH = D6_ANALYSIS_DIR / "project_action_archetypes.parquet"
REVIEW_PATH = D6_OUTPUT_DIR / "fonsi_archetype_review.csv"
PACKETS_PATH = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"
FONSI_PROJECTS = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
TAXONOMY_VERSION = "d6_seed_v1"


SEED_ARCHETYPES = [
    ("geothermal_exploration", "Geothermal exploration", "Geothermal exploration drilling, testing, and associated temporary access.", r"\bgeothermal\b.*\b(?:explor|test|drill|well)\w*\b"),
    ("geothermal_development", "Geothermal development", "Development or operation of geothermal generation facilities and wells.", r"\bgeothermal\b"),
    ("electricity_transmission", "Electricity transmission", "Construction, upgrade, or maintenance of electric transmission lines, substations, and interconnections.", r"\b(?:transmission|power line|electric line|substation|interconnect|right[- ]of[- ]way)\w*\b"),
    ("solar_energy", "Solar energy", "Construction, expansion, or operation of photovoltaic or solar thermal facilities.", r"\b(?:solar|photovoltaic|pv)\b"),
    ("wind_energy", "Wind energy", "Construction, expansion, or operation of wind energy facilities and turbines.", r"\b(?:wind energy|wind farm|wind turbine)\w*\b"),
    ("hydropower", "Hydropower", "Hydroelectric generation, dam, conduit, or related facility actions.", r"\b(?:hydropower|hydroelectric|dam|penstock)\w*\b"),
    ("energy_storage", "Energy storage", "Battery, pumped, or other electricity storage facilities.", r"\b(?:battery|energy storage|storage facility|pumped storage)\w*\b"),
    ("hydrogen", "Hydrogen", "Hydrogen production, storage, transport, or related facility actions.", r"\bhydrogen\b"),
    ("carbon_management", "Carbon management", "Carbon capture, transport, sequestration, or related management actions.", r"\b(?:carbon capture|carbon sequestration|co2 pipeline|ccs)\b"),
    ("oil_gas", "Oil and gas", "Oil or gas exploration, drilling, production, or related facilities.", r"\b(?:oil|gas|petroleum|natural gas|well pad)\b"),
    ("pipeline", "Pipeline", "Pipeline construction, replacement, repair, or right-of-way actions.", r"\bpipeline\b"),
    ("facility_upgrade", "Facility upgrade", "Modification, replacement, maintenance, or upgrade of existing energy facilities.", r"\b(?:upgrade|replace|repair|maintenan|modif|retrofit)\w*\b"),
    ("site_characterization", "Site characterization", "Survey, geotechnical, environmental, or site characterization activities.", r"\b(?:site characterization|geotechnical|survey|sampling|test pit)\w*\b"),
    ("other_clean_energy", "Other clean-energy action", "Clean-energy action not yet represented by a more specific seed archetype.", r"$^"),
]


def build_taxonomy(run_at: str) -> pd.DataFrame:
    records = []
    for archetype_id, label, description, pattern in SEED_ARCHETYPES:
        records.append(
            {
                "archetype_id": archetype_id,
                "archetype_label": label,
                "archetype_description": description,
                "seed_project_types": "",
                "seed_tech_groups": "",
                "keyword_patterns": pattern,
                "candidate_scope": (
                    "comparison_diagnostic"
                    if archetype_id in {"oil_gas", "other_clean_energy"}
                    else "candidate"
                ),
                "taxonomy_status": "seed",
                "taxonomy_version": TAXONOMY_VERSION,
                "taxonomy_run_at": run_at,
            }
        )
    return pd.DataFrame(records)


def project_metadata() -> pd.DataFrame:
    projects = pd.read_parquet(PROJECTS_COMBINED)
    projects = projects.sort_values(
        ["project_id", "process_type"],
        key=lambda col: col.map({"EA": "0", "CE": "1", "EIS": "2"}).fillna(col)
        if col.name == "process_type" else col,
    ).drop_duplicates("project_id")
    reviews = pd.read_parquet(D03_REVIEWS)
    keep = [
        "project_id",
        "process_type",
        "project_energy_type",
        "energy_group",
        "tech_group",
        "lead_agency_harmonized",
        "project_type",
    ]
    reviews = reviews[keep].copy()
    merged = reviews.merge(
        projects[
            [
                "project_id",
                "project_title",
                "project_description",
                "project_state",
                "project_county",
            ]
        ],
        how="left",
        on="project_id",
    )
    agency = merged["lead_agency_harmonized"].fillna("")
    clean = merged["project_energy_type"].fillna("").eq("Clean") | merged[
        "energy_group"
    ].fillna("").isin(["Clean", "Decarbonization"])
    blm_or_doe = agency.str.contains(
        r"Bureau of Land Management|Department of Energy|\bBLM\b|\bDOE\b",
        case=False,
        regex=True,
    )
    return merged.loc[clean & blm_or_doe].drop_duplicates(
        ["project_id", "process_type"]
    )


def packet_text_by_project() -> dict[str, str]:
    if not PACKETS_PATH.exists():
        return {}
    packets = pd.read_parquet(PACKETS_PATH)
    text_col = "action_text" if "action_text" in packets.columns else "analysis_text"
    return dict(zip(packets["project_id"].astype(str), packets[text_col].fillna("")))


def assign_archetypes(projects: pd.DataFrame, taxonomy: pd.DataFrame, run_at: str) -> pd.DataFrame:
    packet_text = packet_text_by_project()
    rules = [
        (row.archetype_id, re.compile(row.keyword_patterns, re.I))
        for row in taxonomy.itertuples(index=False)
        if row.archetype_id != "other_clean_energy"
    ]
    records = []
    for project in projects.itertuples(index=False):
        metadata = " ".join(
            normalize_space(getattr(project, col, ""))
            for col in ("project_title", "project_description", "project_type", "tech_group")
        )
        extracted_text = normalize_space(packet_text.get(str(project.project_id), ""))
        matched = []
        for archetype_id, pattern in rules:
            if pattern.search(metadata) or (extracted_text and pattern.search(extracted_text)):
                matched.append(archetype_id)
        if "geothermal_exploration" in matched and "geothermal_development" in matched:
            matched.remove("geothermal_development")
        if not matched:
            matched = ["other_clean_energy"]
        for rank, archetype_id in enumerate(matched):
            pattern = dict(rules).get(archetype_id)
            text_supported = bool(pattern and extracted_text and pattern.search(extracted_text))
            records.append(
                {
                    "project_id": project.project_id,
                    "process_type": project.process_type,
                    "project_energy_type": project.project_energy_type,
                    "energy_group": project.energy_group,
                    "tech_group": project.tech_group,
                    "lead_agency_harmonized": project.lead_agency_harmonized,
                    "project_title": project.project_title,
                    "project_type": project.project_type,
                    "project_state": project.project_state,
                    "archetype_id": archetype_id,
                    "primary_archetype_id": matched[0],
                    "is_primary_archetype": rank == 0,
                    "assignment_method": "text_supported" if text_supported else "metadata_only",
                    "assignment_confidence": "medium" if archetype_id == "other_clean_energy" else (
                        "high" if text_supported else "medium"
                    ),
                    "taxonomy_version": TAXONOMY_VERSION,
                    "archetype_extraction_run_at": run_at,
                }
            )
    return pd.DataFrame(records)


def add_seed_sources(taxonomy: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    out = taxonomy.copy()
    for archetype_id, group in assignments.groupby("archetype_id"):
        mask = out["archetype_id"].eq(archetype_id)
        out.loc[mask, "seed_project_types"] = json.dumps(
            sorted({normalize_space(v) for v in group["project_type"].dropna() if normalize_space(v)})[:30]
        )
        out.loc[mask, "seed_tech_groups"] = json.dumps(
            sorted({normalize_space(v) for v in group["tech_group"].dropna() if normalize_space(v)})[:30]
        )
    return out


def write_review(assignments: pd.DataFrame) -> None:
    fonsi_ids: set[str] = set()
    if FONSI_PROJECTS.exists():
        fonsi_ids = set(pd.read_parquet(FONSI_PROJECTS)["project_id"].astype(str))
    review = assignments.loc[
        assignments["project_id"].astype(str).isin(fonsi_ids)
        & assignments["is_primary_archetype"]
    ].copy()
    review = review.sort_values(["archetype_id", "project_id"]).groupby(
        "archetype_id", group_keys=False
    ).head(8)
    review["reviewed_archetype_id"] = ""
    review["review_notes"] = ""
    review.to_csv(REVIEW_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the D6 action-archetype taxonomy.")
    parser.add_argument("--taxonomy-version", default=TAXONOMY_VERSION)
    return parser.parse_args()


def main() -> None:
    parse_args()
    ensure_d6_dirs()
    run_at = utc_now()
    taxonomy = build_taxonomy(run_at)
    assignments = assign_archetypes(project_metadata(), taxonomy, run_at)
    taxonomy = add_seed_sources(taxonomy, assignments)
    write_parquet(taxonomy, TAXONOMY_PATH)
    write_parquet(assignments, ASSIGNMENTS_PATH)
    write_review(assignments)
    print(
        f"wrote {len(taxonomy):,} seed archetypes and "
        f"{len(assignments):,} project-archetype assignments"
    )


if __name__ == "__main__":
    main()
