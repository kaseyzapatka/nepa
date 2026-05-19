# --------------------------
# DELIVERABLE 3: visual section inventory
# --------------------------
# Inventory visual/aesthetic section-heading variants in EA and EIS text for the
# deliverable 03 fossil/decarbonization project universe.
#
# Output:
#   phase2/data/analysis/deliverable03/visual_section_heading_candidates.parquet
#   phase2/data/analysis/deliverable03/visual_section_name_map.parquet
#   phase2/output/deliverable03/visual_section_heading_candidates.csv
#   phase2/output/deliverable03/visual_section_name_map.csv
#   phase2/output/deliverable03/visual_section_topic_map.csv
#   phase2/output/deliverable03/visual_section_project_coverage.csv
#   phase2/output/deliverable03/visual_section_project_coverage_by_tech.csv
#
# Usage:
#   conda run -n nepa python phase2/code/deliverable03/03_inventory_visual_sections.py
#   conda run -n nepa python phase2/code/deliverable03/03_inventory_visual_sections.py --all-documents

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
PROCESSED_DIR = DATA_DIR / "processed"
D03_DIR = ANALYSIS_DIR / "deliverable03"
OUTPUT_DIR = BASE_DIR / "output" / "deliverable03"

REVIEWS_PATH = D03_DIR / "projects_nepa_reviews.parquet"
VISUAL_PATH = D03_DIR / "projects_visual_impacts.parquet"

OUT_CANDIDATES = D03_DIR / "visual_section_heading_candidates.parquet"
OUT_NAME_MAP = D03_DIR / "visual_section_name_map.parquet"

CSV_CANDIDATES = OUTPUT_DIR / "visual_section_heading_candidates.csv"
CSV_NAME_MAP = OUTPUT_DIR / "visual_section_name_map.csv"
CSV_TOPIC_MAP = OUTPUT_DIR / "visual_section_topic_map.csv"
CSV_COVERAGE = OUTPUT_DIR / "visual_section_project_coverage.csv"
CSV_COVERAGE_TECH = OUTPUT_DIR / "visual_section_project_coverage_by_tech.csv"

PAGE_KEYWORD_TERMS = [
    "visual",
    "viewshed",
    "view shed",
    "scenic",
    "scenery",
    "aesthetic",
    "landscape character",
    "glare",
    "glint",
    "shadow flicker",
    "light pollution",
    "night sky",
    "dark sky",
    "vrm",
    "vqo",
]

KEYWORD_RE = re.compile(
    r"\b(visual|viewshed|view[ -]shed|scenic|scenery|aesthetic|"
    r"landscape character|glare|glint|shadow flicker|light pollution|"
    r"night sky|dark sky|VRM|VQO)\b",
    re.IGNORECASE,
)

VISUAL_TITLE_RE = re.compile(
    r"\b(aesthetics?(?:\s*/\s*visual\s+resources?)?|"
    r"visual\s+(?:resources?|impacts?|effects?|quality|character|contrast|"
    r"sensitivity|setting)|visual\s+resource\s+management|"
    r"scenic\s+resources?|scenery|view[ -]?sheds?|glare|glint|"
    r"shadow\s+flicker|landscape\s+character|light pollution|"
    r"night sky|dark sky|lighting|VRM|VQO)\b",
    re.IGNORECASE,
)

SECTION_NUMBER_RE = re.compile(
    r"^\s*(?:section|chapter)?\s*"
    r"(?:[A-Z]?\d+(?:[.\-]\d+){0,6}|[IVXLCM]+(?:\.[A-Z0-9]+){0,4}|[A-Z])"
    r"[.)]?\s*$",
    re.IGNORECASE,
)

HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:section|chapter)?\s*"
    r"(?:[A-Z]?\d+(?:[.\-]\d+){0,6}|[IVXLCM]+(?:\.[A-Z0-9]+){0,4}|[A-Z])"
    r"[.)]?\s+",
    re.IGNORECASE,
)

EXACT_AESTHETICS_VISUAL_RE = re.compile(
    r"\baesthetics?\s*(?:/|and|&|-)\s*visual\s+resources?\b|"
    r"\bvisual\s+resources?\s*(?:/|and|&|-)\s*aesthetics?\b",
    re.IGNORECASE,
)

TOC_RE = re.compile(r"\.{3,}|\s(?:[A-Z]?-?\d+|[IVXLCM]+)\s*$")

TOPIC_REGEX = {
    "Aesthetics / Visual Resources": (
        r"(?i)\baesthetics?\s*(?:/|and|&|-)\s*visual\s+resources?\b|"
        r"\bvisual\s+resources?\s*(?:/|and|&|-)\s*aesthetics?\b"
    ),
    "Visual Resources": r"(?i)\bvisual\s+resources?\b",
    "Aesthetics": r"(?i)\baesthetics?\b",
    "Visual Resource Management": (
        r"(?i)\bvisual\s+resource\s+management\b|\bVRM\b|"
        r"\bvisual\s+quality\s+objective\b|\bVQO\b"
    ),
    "Visual Quality / Character": (
        r"(?i)\bvisual\s+(quality|character|contrast|sensitivity|setting)\b"
    ),
    "Visual Impacts / Effects": r"(?i)\bvisual\s+(impacts?|effects?)\b",
    "Scenic Resources / Scenery": r"(?i)\b(scenic|scenery)\b",
    "Viewsheds / Views": r"(?i)\b(view[ -]?shed|viewsheds?|views?)\b",
    "Landscape Character": r"(?i)\blandscape\s+character\b",
    "Glare / Glint": r"(?i)\b(glare|glint)\b",
    "Shadow Flicker": r"(?i)\bshadow\s+flicker\b",
    "Lighting / Night Sky": r"(?i)\b(light pollution|night sky|dark sky|lighting)\b",
}


def clean_line(raw: str) -> str:
    line = re.sub(r"\s+", " ", str(raw)).strip()
    line = re.sub(r"\.{3,}\s*(?:[A-Z]?-?\d+|[IVXLCM]+)?\s*$", "", line)
    return line.strip(" .")


def strip_section_prefix(line: str) -> str:
    line = re.sub(r"^\s*(?:section|chapter)\s+", "", line, flags=re.IGNORECASE)
    line = HEADING_PREFIX_RE.sub("", line)
    return line.strip(" .")


def normalize_title(title: str) -> str:
    title = title.lower().replace("&", " and ")
    title = re.sub(r"\s*/\s*", " / ", title)
    title = re.sub(r"[^a-z0-9/ ]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def is_probable_toc(raw: str, line: str) -> bool:
    low = line.lower()
    if "table of contents" in low or "list of tables" in low or "list of figures" in low:
        return True
    if re.search(r"\.{3,}", raw):
        return True
    return bool(re.search(r"\s(?:[A-Z]?-?\d+|[IVXLCM]+)\s*$", raw.strip()))


def is_heading_candidate(raw: str, prev_raw: str) -> tuple[str, str, str, bool] | None:
    line = clean_line(raw)
    prev_line = clean_line(prev_raw)
    if not line or not KEYWORD_RE.search(line):
        return None

    if SECTION_NUMBER_RE.match(prev_line):
        line = f"{prev_line} {line}"

    low = line.lower()
    if low.startswith(("table ", "figure ", "map ", "photo ", "photograph ")):
        return None
    if any(token in low for token in ("appendix table", "list of figures", "list of tables")):
        return None

    words = line.split()
    has_section_prefix = bool(HEADING_PREFIX_RE.match(line)) or low.startswith(("section ", "chapter "))
    has_visual_title_phrase = bool(VISUAL_TITLE_RE.search(line))
    headingish = (
        has_section_prefix
        or is_probable_toc(raw, line)
        or (
            has_visual_title_phrase
            and len(words) <= 12
            and not re.search(r"[.;]", line)
        )
    )
    if not headingish:
        return None

    if len(line) > 180:
        return None
    if line.endswith(".") and len(words) > 8 and not has_section_prefix:
        return None

    title = strip_section_prefix(line)
    if len(title.split()) > 14 and not EXACT_AESTHETICS_VISUAL_RE.search(title):
        return None
    if not KEYWORD_RE.search(title):
        return None

    return line, title, normalize_title(title), is_probable_toc(raw, line)


def canonical_topic(title: str) -> str:
    checks = [
        ("Aesthetics / Visual Resources", EXACT_AESTHETICS_VISUAL_RE),
        ("Visual Resource Management", re.compile(TOPIC_REGEX["Visual Resource Management"])),
        ("Visual Resources", re.compile(TOPIC_REGEX["Visual Resources"])),
        ("Aesthetics", re.compile(TOPIC_REGEX["Aesthetics"])),
        ("Visual Quality / Character", re.compile(TOPIC_REGEX["Visual Quality / Character"])),
        ("Visual Impacts / Effects", re.compile(TOPIC_REGEX["Visual Impacts / Effects"])),
        ("Scenic Resources / Scenery", re.compile(TOPIC_REGEX["Scenic Resources / Scenery"])),
        ("Viewsheds / Views", re.compile(TOPIC_REGEX["Viewsheds / Views"])),
        ("Landscape Character", re.compile(TOPIC_REGEX["Landscape Character"])),
        ("Glare / Glint", re.compile(TOPIC_REGEX["Glare / Glint"])),
        ("Shadow Flicker", re.compile(TOPIC_REGEX["Shadow Flicker"])),
        ("Lighting / Night Sky", re.compile(TOPIC_REGEX["Lighting / Night Sky"])),
    ]
    for label, pattern in checks:
        if pattern.search(title):
            return label
    return "Other Visual / Aesthetic"


def page_has_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in PAGE_KEYWORD_TERMS)


def page_reader(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    batch_size: int,
    include_supporting: bool,
):
    pages_path = PROCESSED_DIR / source.lower() / "pages.parquet"
    docs_path = PROCESSED_DIR / source.lower() / "documents.parquet"
    main_filter = (
        ""
        if include_supporting
        else "AND coalesce(nullif(d.main_document, ''), 'YES') <> 'NO'"
    )
    query = f"""
        SELECT
            '{source}' AS source,
            r.project_id,
            r.energy_group,
            r.project_energy_type,
            r.process_type,
            r.tech_group,
            d.document_id,
            d.document_title,
            d.main_document,
            p.page_number,
            p.page_text
        FROM read_parquet('{pages_path}') p
        JOIN read_parquet('{docs_path}') d USING (document_id)
        JOIN read_parquet('{REVIEWS_PATH}') r
          ON d.project_id.value = r.project_id
         AND r.process_type = '{source}'
        WHERE r.process_type IN ('EA', 'EIS')
          AND length(p.page_text) > 100
          {main_filter}
        """
    return conn.execute(query).fetch_record_batch(rows_per_batch=batch_size)


def extract_page_candidates(
    row,
    run_at: str,
) -> list[dict]:
    rows: list[dict] = []
    raw_lines = row.page_text.splitlines()
    prev = ""
    for raw in raw_lines:
        hit = is_heading_candidate(raw, prev)
        if hit:
            heading_clean, section_title, title_norm, probable_toc = hit
            rows.append(
                {
                    "source": row.source,
                    "project_id": row.project_id,
                    "energy_group": row.energy_group,
                    "project_energy_type": row.project_energy_type,
                    "process_type": row.process_type,
                    "tech_group": row.tech_group,
                    "document_id": row.document_id,
                    "document_title": row.document_title,
                    "main_document": row.main_document,
                    "page_number": row.page_number,
                    "heading_raw": str(raw).strip(),
                    "heading_clean": heading_clean,
                    "section_title": section_title,
                    "section_title_norm": title_norm,
                    "canonical_topic": canonical_topic(section_title),
                    "exact_aesthetics_visual_resources": bool(
                        EXACT_AESTHETICS_VISUAL_RE.search(section_title)
                    ),
                    "probable_toc": probable_toc,
                    "inventory_run_at": run_at,
                }
            )
        if clean_line(raw):
            prev = raw
    return rows


def extract_heading_candidates(
    conn: duckdb.DuckDBPyConnection,
    batch_size: int = 50_000,
    include_supporting: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    run_at = datetime.now(timezone.utc).isoformat()

    for source in ["EA", "EIS"]:
        reader = page_reader(conn, source, batch_size, include_supporting)
        n_seen = 0
        n_keyword_pages = 0
        print(
            f"Scanning {source} "
            f"({'all documents' if include_supporting else 'main documents'})",
            flush=True,
        )

        for batch in reader:
            pdf = batch.to_pandas()
            n_seen += len(pdf)
            for row in pdf.itertuples(index=False):
                text = row.page_text
                if not isinstance(text, str) or len(text) <= 100:
                    continue
                if not page_has_keyword(text):
                    continue
                n_keyword_pages += 1
                rows.extend(extract_page_candidates(row=row, run_at=run_at))

            if n_seen % (batch_size * 10) == 0:
                print(
                    f"  {source}: scanned {n_seen:,} rows; "
                    f"{n_keyword_pages:,} keyword pages; {len(rows):,} candidates",
                    flush=True,
                )

        print(
            f"Finished {source}: scanned {n_seen:,} rows; "
            f"{n_keyword_pages:,} keyword pages",
            flush=True,
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "project_id",
                "energy_group",
                "project_energy_type",
                "process_type",
                "tech_group",
                "document_id",
                "document_title",
                "main_document",
                "page_number",
                "heading_raw",
                "heading_clean",
                "section_title",
                "section_title_norm",
                "canonical_topic",
                "exact_aesthetics_visual_resources",
                "probable_toc",
                "inventory_run_at",
            ]
        )

    candidates = pd.DataFrame(rows)
    return candidates.drop_duplicates(
        subset=[
            "project_id",
            "document_id",
            "page_number",
            "section_title_norm",
            "canonical_topic",
            "probable_toc",
        ]
    ).reset_index(drop=True)


def modal_value(values: pd.Series) -> str:
    modes = values.dropna().astype(str).value_counts()
    return modes.index[0] if len(modes) else ""


def distinct_project_counts(
    df: pd.DataFrame,
    group_cols: list[str],
    count_name: str,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    data = df.loc[mask] if mask is not None else df
    if data.empty:
        return pd.DataFrame(columns=group_cols + [count_name])
    return (
        data.drop_duplicates(group_cols + ["project_id"])
        .groupby(group_cols, dropna=False)
        .size()
        .rename(count_name)
        .reset_index()
    )


def build_name_map(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    group_cols = ["canonical_topic", "section_title_norm"]
    name_map = candidates.groupby(group_cols, dropna=False).agg(
        display_title=("section_title", modal_value),
        n_heading_lines=("section_title", "size"),
    ).reset_index()

    count_frames = [
        distinct_project_counts(candidates, group_cols, "n_projects"),
        distinct_project_counts(
            candidates, group_cols, "n_body_projects", ~candidates["probable_toc"]
        ),
        distinct_project_counts(
            candidates,
            group_cols,
            "n_decarb_projects",
            candidates["energy_group"].eq("Decarbonization"),
        ),
        distinct_project_counts(
            candidates,
            group_cols,
            "n_fossil_projects",
            candidates["energy_group"].eq("Fossil Fuel"),
        ),
        distinct_project_counts(
            candidates, group_cols, "n_ea_projects", candidates["process_type"].eq("EA")
        ),
        distinct_project_counts(
            candidates, group_cols, "n_eis_projects", candidates["process_type"].eq("EIS")
        ),
    ]
    for frame in count_frames:
        name_map = name_map.merge(frame, on=group_cols, how="left")

    count_cols = [
        "n_projects",
        "n_body_projects",
        "n_decarb_projects",
        "n_fossil_projects",
        "n_ea_projects",
        "n_eis_projects",
    ]
    name_map[count_cols] = name_map[count_cols].fillna(0).astype(int)
    name_map["recommended_topic_regex"] = name_map["canonical_topic"].map(TOPIC_REGEX)
    return name_map.sort_values(
        ["n_projects", "n_body_projects", "canonical_topic"], ascending=[False, False, True]
    ).reset_index(drop=True)


def build_topic_map(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    group_cols = ["canonical_topic"]
    topic_variants = candidates.groupby(group_cols)["section_title_norm"].nunique()
    topic_variants = topic_variants.rename("n_section_name_variants").reset_index()
    topic_regex = pd.Series(TOPIC_REGEX, name="recommended_topic_regex")

    topic_map = distinct_project_counts(candidates, group_cols, "n_projects")
    for frame in [
        distinct_project_counts(
            candidates, group_cols, "n_body_projects", ~candidates["probable_toc"]
        ),
        distinct_project_counts(
            candidates,
            group_cols,
            "n_decarb_projects",
            candidates["energy_group"].eq("Decarbonization"),
        ),
        distinct_project_counts(
            candidates,
            group_cols,
            "n_fossil_projects",
            candidates["energy_group"].eq("Fossil Fuel"),
        ),
        distinct_project_counts(
            candidates, group_cols, "n_ea_projects", candidates["process_type"].eq("EA")
        ),
        distinct_project_counts(
            candidates, group_cols, "n_eis_projects", candidates["process_type"].eq("EIS")
        ),
    ]:
        topic_map = topic_map.merge(frame, on=group_cols, how="left")

    count_cols = [
        "n_projects",
        "n_body_projects",
        "n_decarb_projects",
        "n_fossil_projects",
        "n_ea_projects",
        "n_eis_projects",
    ]
    topic_map[count_cols] = topic_map[count_cols].fillna(0).astype(int)
    return (
        topic_map.merge(topic_variants, on="canonical_topic", how="left")
        .merge(
            topic_regex.reset_index().rename(columns={"index": "canonical_topic"}),
            on="canonical_topic",
            how="left",
        )
        .sort_values("n_projects", ascending=False)
    )


def build_coverage(
    conn: duckdb.DuckDBPyConnection, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = conn.execute(
        f"""
        SELECT project_id, energy_group, project_energy_type, process_type, tech_group
        FROM read_parquet('{REVIEWS_PATH}')
        WHERE process_type IN ('EA', 'EIS')
        """
    ).fetchdf()

    if VISUAL_PATH.exists():
        visual = conn.execute(
            f"""
            SELECT project_id, visual_section_found, visual_mention_count,
                   visual_impacts_max_similarity
            FROM read_parquet('{VISUAL_PATH}')
            """
        ).fetchdf()
        base = base.merge(visual, on="project_id", how="left")
    else:
        base["visual_section_found"] = pd.NA
        base["visual_mention_count"] = pd.NA
        base["visual_impacts_max_similarity"] = pd.NA

    if candidates.empty:
        for col in [
            "has_visual_heading_any",
            "has_visual_heading_body",
            "has_exact_aesthetics_visual_resources",
        ]:
            base[col] = False
    else:
        any_ids = set(candidates["project_id"])
        body_ids = set(candidates.loc[~candidates["probable_toc"], "project_id"])
        exact_ids = set(
            candidates.loc[candidates["exact_aesthetics_visual_resources"], "project_id"]
        )
        base["has_visual_heading_any"] = base["project_id"].isin(any_ids)
        base["has_visual_heading_body"] = base["project_id"].isin(body_ids)
        base["has_exact_aesthetics_visual_resources"] = base["project_id"].isin(exact_ids)

    def summarize(group_cols: list[str]) -> pd.DataFrame:
        out = (
            base.groupby(group_cols, dropna=False)
            .agg(
                n_projects=("project_id", "nunique"),
                n_visual_heading_any=("has_visual_heading_any", "sum"),
                n_visual_heading_body=("has_visual_heading_body", "sum"),
                n_exact_aesthetics_visual_resources=(
                    "has_exact_aesthetics_visual_resources",
                    "sum",
                ),
                n_existing_visual_section_found=("visual_section_found", "sum"),
                median_visual_mentions=("visual_mention_count", "median"),
            )
            .reset_index()
        )
        for num in [
            "visual_heading_any",
            "visual_heading_body",
            "exact_aesthetics_visual_resources",
            "existing_visual_section_found",
        ]:
            out[f"pct_{num}"] = (100 * out[f"n_{num}"] / out["n_projects"]).round(1)
        return out.sort_values(group_cols).reset_index(drop=True)

    return summarize(["energy_group", "process_type"]), summarize(
        ["energy_group", "tech_group", "process_type"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory visual/aesthetic EA/EIS section-heading variants."
    )
    parser.add_argument(
        "--all-documents",
        action="store_true",
        help="Scan supporting documents as well as main EA/EIS documents.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    D03_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    conn.execute("PRAGMA threads=4")

    print("Extracting heading candidates...", flush=True)
    candidates = extract_heading_candidates(conn, include_supporting=args.all_documents)
    print(f"Found {len(candidates):,} visual/aesthetic heading candidates", flush=True)
    candidates.to_parquet(OUT_CANDIDATES, index=False)
    candidates.to_csv(CSV_CANDIDATES, index=False)

    name_map = build_name_map(candidates)
    topic_map = build_topic_map(candidates)
    coverage, coverage_tech = build_coverage(conn, candidates)

    name_map.to_parquet(OUT_NAME_MAP, index=False)
    name_map.to_csv(CSV_NAME_MAP, index=False)
    topic_map.to_csv(CSV_TOPIC_MAP, index=False)
    coverage.to_csv(CSV_COVERAGE, index=False)
    coverage_tech.to_csv(CSV_COVERAGE_TECH, index=False)

    print("\nCoverage by energy/process:")
    print(coverage.to_string(index=False))
    print("\nTop section-name variants:")
    top_cols = [
        "canonical_topic",
        "display_title",
        "n_projects",
        "n_body_projects",
        "n_decarb_projects",
        "n_fossil_projects",
        "n_ea_projects",
        "n_eis_projects",
    ]
    print(name_map[top_cols].head(30).to_string(index=False))
    print("\nWrote:")
    for path in [
        OUT_CANDIDATES,
        OUT_NAME_MAP,
        CSV_NAME_MAP,
        CSV_TOPIC_MAP,
        CSV_COVERAGE,
        CSV_COVERAGE_TECH,
    ]:
        print(f"  {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
