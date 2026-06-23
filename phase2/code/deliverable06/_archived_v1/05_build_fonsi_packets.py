import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from common import (
    D6_ANALYSIS_DIR,
    D6_OUTPUT_DIR,
    EA_PAGES,
    compact_join,
    ensure_d6_dirs,
    normalize_space,
    sha256_text,
    utc_now,
    write_parquet,
)


SECTIONS_PATH = D6_ANALYSIS_DIR / "fonsi_document_sections.parquet"
PROJECTS_PATH = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
MANIFEST_PATH = D6_ANALYSIS_DIR / "fonsi_section_manifest.parquet"
PACKETS_PATH = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"
SPANS_PATH = D6_ANALYSIS_DIR / "fonsi_evidence_spans.parquet"
REVIEW_PATH = D6_OUTPUT_DIR / "fonsi_packet_review.csv"

SPAN_PATTERNS = {
    "action": re.compile(r"\b(?:proposed action|project description|description of|alternatives?|purpose and need)\b", re.I),
    "finding": re.compile(r"\b(?:finding|determination|decision|environmental consequences|impact summary|no significant impact)\b", re.I),
    "resource": re.compile(r"\b(?:resource|environment|biological|wildlife|water|air quality|cultural|visual|noise|traffic|land use|geology|soil)\b", re.I),
    "condition": re.compile(r"\b(?:mitigation|monitoring|best management|bmp|design feature|condition of approval)\b", re.I),
    "boundary": re.compile(r"\b(?:limitation|constraint|extraordinary circumstance|sensitive resource|setback|not exceed|no more than|access road)\b", re.I),
}
RESOURCE_TOPICS = {
    "visual", "environmental_justice", "cultural_resources", "biological_resources",
    "water_resources", "air_quality", "greenhouse_gas", "land_use", "recreation",
    "noise", "traffic_transportation", "socioeconomics", "geology_soils",
}


def stable_id(*parts: object) -> str:
    text = "|".join(normalize_space(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_projects(sample: int | None, project_ids: list[str] | None) -> pd.DataFrame:
    projects = pd.read_parquet(PROJECTS_PATH)
    if project_ids:
        projects = projects.loc[projects["project_id"].astype(str).isin(project_ids)]
    if sample:
        projects = projects.sample(min(sample, len(projects)), random_state=42)
    return projects


def classify_section(row: pd.Series) -> list[str]:
    heading = " ".join(
        normalize_space(row.get(col, ""))
        for col in ("heading_title", "parent_heading_title", "section_topic_guess")
    )
    labels = [label for label, pattern in SPAN_PATTERNS.items() if pattern.search(heading)]
    if row.get("section_topic_guess") in RESOURCE_TOPICS and "resource" not in labels:
        labels.append("resource")
    return labels


def section_spans(projects: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if not SECTIONS_PATH.exists():
        return pd.DataFrame()
    sections = pd.read_parquet(SECTIONS_PATH)
    sections = sections.loc[sections["project_id"].isin(projects["project_id"])].copy()
    sections = sections.merge(
        manifest[["document_id", "manifest_role"]].drop_duplicates("document_id"),
        how="left",
        on="document_id",
    )
    records = []
    for _, section in sections.iterrows():
        labels = classify_section(section)
        if not labels:
            continue
        text = normalize_space(section["section_text"])
        if len(text) < 40:
            continue
        section_id = stable_id(
            section["document_id"], section["page_start"], section["line_start"], section["heading_raw"]
        )
        for span_type in labels:
            records.append(
                {
                    "project_id": section["project_id"],
                    "document_id": section["document_id"],
                    "manifest_role": section.get("manifest_role", ""),
                    "section_id": section_id,
                    "evidence_span_id": stable_id(section_id, span_type),
                    "span_type": span_type,
                    "heading_title": section["heading_title"],
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "span_text": text[:16_000],
                    "source_span_sha256": sha256_text(text),
                    "span_extraction_method": "shared_section_layer",
                }
            )
    return pd.DataFrame(records)


def fallback_spans(projects: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    covered = set(existing["project_id"]) if not existing.empty else set()
    missing = projects.loc[~projects["project_id"].isin(covered), [
        "project_id", "canonical_fonsi_document_id"
    ]].copy()
    if missing.empty:
        return pd.DataFrame()
    missing = missing.rename(columns={"canonical_fonsi_document_id": "document_id"})
    conn = duckdb.connect()
    conn.register("fallback_documents", missing)
    pages = conn.execute(
        f"""
        SELECT
            d.project_id,
            p.document_id,
            COALESCE(
                TRY_CAST(regexp_extract(CAST(p.page_number AS VARCHAR), '(\\d+)', 1) AS INTEGER),
                1000000000
            ) AS page_num,
            p.page_text
        FROM read_parquet('{EA_PAGES}') p
        JOIN fallback_documents d USING (document_id)
        QUALIFY row_number() OVER (
            PARTITION BY p.document_id
            ORDER BY page_num
        ) <= 8
        ORDER BY d.project_id, p.document_id, page_num
        """
    ).fetchdf()
    records = []
    for (project_id, document_id), group in pages.groupby(["project_id", "document_id"]):
        text = compact_join(group["page_text"], limit=16_000)
        if not text:
            continue
        section_id = stable_id(document_id, "fallback")
        records.append(
            {
                "project_id": project_id,
                "document_id": document_id,
                "manifest_role": "canonical_fonsi",
                "section_id": section_id,
                "evidence_span_id": stable_id(section_id, "fallback"),
                "span_type": "fallback",
                "heading_title": "",
                "page_start": int(group["page_num"].min()),
                "page_end": int(group["page_num"].max()),
                "span_text": text,
                "source_span_sha256": sha256_text(text),
                "span_extraction_method": "bounded_canonical_page_fallback",
            }
        )
    return pd.DataFrame(records)


def build_packets(projects: pd.DataFrame, spans: pd.DataFrame, manifest: pd.DataFrame, run_at: str) -> pd.DataFrame:
    docs = pd.DataFrame(
        [
            {
                "project_id": project_id,
                "linked_documents": json.dumps(
                    group[["document_id", "manifest_role"]].drop_duplicates().to_dict("records"),
                    sort_keys=True,
                ),
            }
            for project_id, group in manifest.loc[
                manifest["project_id"].isin(projects["project_id"])
            ].groupby("project_id")
        ]
    )
    records = []
    for project in projects.itertuples(index=False):
        group = spans.loc[spans["project_id"].eq(project.project_id)]
        by_type = {
            span_type: compact_join(group.loc[group["span_type"].eq(span_type), "span_text"])
            for span_type in ["action", "finding", "resource", "condition", "boundary", "fallback"]
        }
        action_text = by_type["action"] or by_type["fallback"]
        finding_text = by_type["finding"] or by_type["fallback"]
        records.append(
            {
                "project_id": project.project_id,
                "canonical_fonsi_document_id": project.canonical_fonsi_document_id,
                "project_title": project.project_title,
                "project_type": project.project_type,
                "project_description": project.project_description,
                "project_energy_type": project.project_energy_type,
                "energy_group": project.energy_group,
                "tech_group": project.tech_group,
                "lead_agency_harmonized": project.lead_agency_harmonized,
                "project_state": project.project_state,
                "project_county": project.project_county,
                "action_text": action_text,
                "finding_text": finding_text,
                "resource_text": by_type["resource"],
                "condition_text": by_type["condition"],
                "boundary_text": by_type["boundary"],
                "analysis_text": compact_join(
                    [action_text, finding_text, by_type["condition"], by_type["boundary"]]
                ),
                "evidence_span_count": len(group),
                "packet_extraction_run_at": run_at,
            }
        )
    packets = pd.DataFrame(records)
    return packets.merge(docs, how="left", on="project_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bounded D6 project evidence packets.")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--project-id", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_d6_dirs()
    run_at = utc_now()
    projects = load_projects(args.sample, args.project_id)
    manifest = pd.read_parquet(MANIFEST_PATH)
    spans = section_spans(projects, manifest)
    fallback = fallback_spans(projects, spans)
    if not fallback.empty:
        spans = pd.concat([spans, fallback], ignore_index=True)
    spans["evidence_extraction_run_at"] = run_at
    packets = build_packets(projects, spans, manifest, run_at)
    write_parquet(spans, SPANS_PATH)
    write_parquet(packets, PACKETS_PATH)
    packets.sort_values("evidence_span_count").head(100).to_csv(REVIEW_PATH, index=False)
    print(
        f"wrote {len(packets):,} project packets and {len(spans):,} typed evidence spans "
        f"({len(fallback):,} bounded fallbacks)"
    )


if __name__ == "__main__":
    main()
