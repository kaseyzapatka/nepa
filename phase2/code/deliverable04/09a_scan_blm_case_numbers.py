"""
Scan NEPATEC page text for DOI-BLM-... case numbers in BLM projects only.

Produces:
    phase2/data/analysis/blm_register/nepatec_case_evidence.parquet
      One row per (project_id, case_number) with supporting evidence.

Usage:
    python 09a_scan_blm_case_numbers.py
    python 09a_scan_blm_case_numbers.py --process EA EIS CE
    python 09a_scan_blm_case_numbers.py --sample 500
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
PROCESSED_DIR = PHASE2 / "data" / "processed"
OUT_DIR = ANALYSIS_DIR / "blm_register"
PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
DOCUMENTS_PATH = ANALYSIS_DIR / "documents_combined.parquet"
OUTPUT_PATH = OUT_DIR / "nepatec_case_evidence.parquet"

SOURCE_MAP = {"CE": "ce", "EA": "ea", "EIS": "eis"}

BLM_CASE_RE = re.compile(
    r"\bDOI-BLM-([A-Z]{2})-([A-Z0-9]+)-(\d{4})-(\d+)-([A-Z]+)\b",
    re.IGNORECASE,
)

# For CE projects, BLM uses CX (Categorical Exclusion) as the case-type suffix
CE_CASE_TYPES = {"CE", "CX", "DNA", "DR", "SCX", "DN"}

# OCR confusion: letter O misread as digit 0 in office codes (e.g. CO60 → C060)
_OCR_O_TO_0 = re.compile(r"O(?=\d)|(?<=\d)O", re.IGNORECASE)


def _normalize_case_number(case_number: str) -> str:
    """
    Fix common OCR confusion of letter O vs digit 0 in the office code segment.
    DOI-BLM-{ST}-{OFFICE}-{YEAR}-{SEQ}-{TYPE}
    """
    parts = case_number.upper().split("-")
    if len(parts) < 7:
        return case_number
    # parts[3] is the office code
    normalized_office = _OCR_O_TO_0.sub("0", parts[3])
    if normalized_office != parts[3]:
        parts[3] = normalized_office
        return "-".join(parts)
    return case_number


def _extract_matches(row: dict, context_chars: int = 300) -> list[dict]:
    text = row["page_text"] or ""
    upper = text.upper()
    records = []
    for m in BLM_CASE_RE.finditer(upper):
        start = max(0, m.start() - context_chars)
        end = min(len(text), m.end() + context_chars)
        raw = m.group(0).upper()
        normalized = _normalize_case_number(raw)
        records.append(
            {
                "project_id": row["project_id"],
                "process_type": row["process_type"],
                "document_id": row["document_id"],
                "file_name": row["file_name"],
                "main_document": row["main_document"],
                "document_type_category": row["document_type_category"],
                "page_number": row["page_number"],
                "case_number": normalized,       # use normalized as join key
                "case_number_raw": raw,           # preserve original OCR text
                "state_code": m.group(1).upper(),
                "office_code": normalized.split("-")[3] if "-" in normalized else m.group(2).upper(),
                "year": int(m.group(3)),
                "seq": m.group(4),
                "case_type": m.group(5).upper(),
                "context_window": text[start:end].strip(),
            }
        )
    return records


def scan_process_type(
    con: duckdb.DuckDBPyConnection,
    process_type: str,
    blm_project_ids: set[str],
    sample: int | None,
) -> pd.DataFrame:
    src = SOURCE_MAP[process_type]
    pages_path = str(PROCESSED_DIR / src / "pages.parquet")
    # Use documents_combined for project_id (VARCHAR), main_document, document_type_category
    docs_combined_path = str(DOCUMENTS_PATH)

    print(f"  [{process_type}] Filtering candidate pages with LIKE '%DOI-BLM-%' ...")

    # DuckDB join: pages → documents_combined (already has clean project_id + extra fields)
    query = f"""
        SELECT
            d.project_id,
            '{process_type}' AS process_type,
            p.document_id,
            d.file_name,
            d.main_document,
            d.document_type_category,
            p.page_number,
            p.page_text
        FROM read_parquet('{pages_path}') p
        JOIN read_parquet('{docs_combined_path}') d USING (document_id)
        WHERE upper(p.page_text) LIKE '%DOI-BLM-%'
    """
    df = con.execute(query).df()

    # Filter to BLM projects
    df = df[df["project_id"].isin(blm_project_ids)].copy()

    if sample is not None:
        # Sample by project, not by page
        sampled_projects = list(df["project_id"].unique())[:sample]
        df = df[df["project_id"].isin(sampled_projects)]

    print(f"  [{process_type}] {len(df)} candidate pages across "
          f"{df['project_id'].nunique()} BLM projects")

    if df.empty:
        return pd.DataFrame()

    # Apply regex in Python
    rows = []
    for rec in df.to_dict("records"):
        rows.extend(_extract_matches(rec))

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # process_match: case_type in case number agrees with project process_type
    result["process_match"] = result["case_type"] == result["process_type"]

    # evidence_rank: 1 = main doc, 2 = everything else
    result["evidence_rank"] = result["main_document"].apply(
        lambda v: 1 if str(v).lower() in ("true", "1", "yes") else 2
    )

    return result


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (project_id, case_number).
    Prefer main_document evidence. Flag multi-case-number projects.
    """
    if df.empty:
        return df

    # Best evidence per (project_id, case_number): main doc first, then lowest page
    df = df.sort_values(["project_id", "case_number", "evidence_rank", "page_number"])
    best = df.drop_duplicates(subset=["project_id", "case_number"], keep="first").copy()

    # Count distinct case numbers per project
    counts = (
        df.groupby("project_id")["case_number"]
        .nunique()
        .rename("case_number_count")
    )
    best = best.join(counts, on="project_id")
    best["multi_case_flag"] = best["case_number_count"] > 1

    return best.reset_index(drop=True)


def acceptance_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row with accept | review | skip.
      accept: effective_process_match=True, single case number
      review: effective_process_match=True + multi-case, OR main_doc cross-reference
      skip:   no process match and not main_doc
    """
    if df.empty:
        return df

    conditions = []
    for _, row in df.iterrows():
        pt = row["process_type"]
        ct = row["case_type"]

        # CE projects use CX (Categorical Exclusion) as the BLM case-type suffix.
        # Also accept DNA/DR/SCX/DN which are other CE-class designations.
        if pt == "CE":
            effective_match = ct in CE_CASE_TYPES
        else:
            effective_match = row["process_match"]

        if effective_match and not row["multi_case_flag"]:
            conditions.append("accept")
        elif effective_match and row["multi_case_flag"]:
            conditions.append("review")
        elif not effective_match and row["evidence_rank"] == 1:
            conditions.append("review")  # cross-reference in main doc
        else:
            conditions.append("skip")

    df = df.copy()
    df["acceptance"] = conditions
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", nargs="+", choices=["CE", "EA", "EIS"],
                        default=["CE", "EA", "EIS"])
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N BLM projects per process type (for testing)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    print("Loading BLM project IDs from projects_combined ...")
    projects = con.execute(f"""
        SELECT project_id, process_type, lead_agency_harmonized
        FROM read_parquet('{PROJECTS_PATH}')
        WHERE lower(lead_agency_harmonized) LIKE '%bureau of land%'
          AND process_type IN ({','.join("'" + p + "'" for p in args.process)})
    """).df()
    blm_project_ids = set(projects["project_id"].tolist())
    print(f"  {len(blm_project_ids)} BLM projects across process types: "
          f"{projects['process_type'].value_counts().to_dict()}")

    all_parts = []
    for pt in args.process:
        part = scan_process_type(con, pt, blm_project_ids, args.sample)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        print("No DOI-BLM case numbers found.")
        return

    combined = pd.concat(all_parts, ignore_index=True)
    deduped = deduplicate(combined)
    gated = acceptance_gate(deduped)

    gated["scan_run_at"] = datetime.now(timezone.utc).isoformat()

    gated.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(gated)} rows → {OUTPUT_PATH}")
    print("\nAcceptance breakdown:")
    print(gated["acceptance"].value_counts().to_string())
    print("\nProcess type breakdown:")
    print(gated.groupby(["process_type", "acceptance"]).size().to_string())
    print(f"\nUnique case numbers (accept only): "
          f"{gated[gated['acceptance']=='accept']['case_number'].nunique()}")
    print(f"Unique projects with any match: {gated['project_id'].nunique()}")

    # Print a sample of found case numbers
    print("\nSample case numbers found:")
    sample_cn = gated[gated["acceptance"] == "accept"]["case_number"].drop_duplicates().head(20)
    for cn in sample_cn:
        print(f"  {cn}")


if __name__ == "__main__":
    main()
