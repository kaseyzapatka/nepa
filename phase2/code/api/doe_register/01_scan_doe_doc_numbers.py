"""
Scan NEPATEC page text for DOE/EA-NNNN and DOE/EIS-NNNN document numbers
in DOE EA and EIS projects.

Produces:
    phase2/data/analysis/doe_register/doe_case_evidence.parquet
      One row per (project_id, doc_number) with supporting evidence.

Usage:
    python 10a_scan_doe_doc_numbers.py
    python 10a_scan_doe_doc_numbers.py --process EA EIS
    python 10a_scan_doe_doc_numbers.py --sample 50
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

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
PROCESSED_DIR = PHASE2 / "data" / "processed"
OUT_DIR = ANALYSIS_DIR / "doe_register"
PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
DOCUMENTS_PATH = ANALYSIS_DIR / "documents_combined.parquet"
OUTPUT_PATH = OUT_DIR / "doe_case_evidence.parquet"

SOURCE_MAP = {"EA": "ea", "EIS": "eis"}

# DOE/EA-NNNN and DOE/EIS-NNNN (with optional supplement suffix)
DOE_DOC_RE = re.compile(
    r"\bDOE/(EA-\d{4}|EIS-\d{4}(?:[-‐-―](?:S\d+|SA[-\s]?\d+))?)\b",
    re.IGNORECASE,
)


def _normalize_doc_number(raw: str) -> str:
    """Normalize to uppercase, ASCII hyphens, no spaces in suffix."""
    n = raw.upper()
    # Replace en-dash/em-dash with hyphen
    n = re.sub(r"[‐-―]", "-", n)
    # Remove spaces within suffix (e.g. "SA- 10" → "SA-10")
    n = re.sub(r"(SA-)\s+(\d)", r"\1\2", n)
    return n


def _classify_doc_number(doc_number: str) -> tuple[str, str]:
    """Return (doc_type, base_number). E.g. DOE/EIS-0391-SA-05 → ('EIS-SA', 'DOE/EIS-0391')."""
    m = re.match(r"DOE/(EA|EIS)-(\d{4})(.*)", doc_number)
    if not m:
        return "unknown", doc_number
    prefix, num, suffix = m.group(1), m.group(2), m.group(3).strip("-")
    if not suffix:
        return prefix, f"DOE/{prefix}-{num}"
    elif re.match(r"S\d+$", suffix):
        return f"{prefix}-SEIS", f"DOE/{prefix}-{num}"
    elif re.match(r"SA", suffix):
        return f"{prefix}-SA", f"DOE/{prefix}-{num}"
    return prefix, f"DOE/{prefix}-{num}"


def _extract_matches(row: dict, context_chars: int = 300) -> list[dict]:
    text = row["page_text"] or ""
    upper = text.upper()
    records = []
    seen = set()
    for m in DOE_DOC_RE.finditer(upper):
        raw = m.group(0).upper()
        normalized = _normalize_doc_number(raw)
        doc_type, base_number = _classify_doc_number(normalized)
        key = (row["project_id"], normalized)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - context_chars)
        end = min(len(text), m.end() + context_chars)
        records.append({
            "project_id": row["project_id"],
            "process_type": row["process_type"],
            "document_id": row["document_id"],
            "file_name": row.get("file_name"),
            "main_document": row.get("main_document"),
            "document_type_category": row.get("document_type_category"),
            "page_number": row["page_number"],
            "doc_number": normalized,
            "doc_number_raw": raw,
            "doc_type": doc_type,
            "base_number": base_number,
            "context_window": text[start:end].strip(),
        })
    return records


def scan_process_type(
    con: duckdb.DuckDBPyConnection,
    process_type: str,
    doe_project_ids: set[str],
    sample: int | None,
) -> pd.DataFrame:
    src = SOURCE_MAP[process_type]
    pages_path = str(PROCESSED_DIR / src / "pages.parquet")
    docs_combined_path = str(DOCUMENTS_PATH)

    like_filter = "DOE/EA-" if process_type == "EA" else "DOE/EIS-"
    print(f"  [{process_type}] Filtering pages with LIKE '%{like_filter}%' ...")

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
        WHERE upper(p.page_text) LIKE '%{like_filter}%'
    """
    df = con.execute(query).df()
    df = df[df["project_id"].isin(doe_project_ids)].copy()

    if sample is not None:
        sampled_projects = list(df["project_id"].unique())[:sample]
        df = df[df["project_id"].isin(sampled_projects)]

    print(f"  [{process_type}] {len(df)} candidate pages across "
          f"{df['project_id'].nunique()} DOE projects")

    if df.empty:
        return pd.DataFrame()

    rows = []
    for rec in df.to_dict("records"):
        rows.extend(_extract_matches(rec))

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # doc_type_match: does the doc number type agree with the project process type?
    result["doc_type_match"] = result.apply(
        lambda r: r["process_type"] in r["doc_type"], axis=1
    )

    # evidence_rank: 1 = main doc, 2 = everything else
    result["evidence_rank"] = result["main_document"].apply(
        lambda v: 1 if str(v).lower() in ("true", "1", "yes") else 2
    )

    return result


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Count page appearances per (project_id, doc_number) — the project's own number
    # appears far more frequently than cross-referenced numbers.
    page_counts = (
        df.groupby(["project_id", "doc_number"])
        .size()
        .rename("page_count")
        .reset_index()
    )
    df = df.sort_values(["project_id", "doc_number", "evidence_rank", "page_number"])
    best = df.drop_duplicates(subset=["project_id", "doc_number"], keep="first").copy()
    best = best.merge(page_counts, on=["project_id", "doc_number"], how="left")

    counts = df.groupby("project_id")["doc_number"].nunique().rename("doc_number_count")
    best = best.join(counts, on="project_id")
    best["multi_doc_flag"] = best["doc_number_count"] > 1
    return best.reset_index(drop=True)


def acceptance_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    accept: doc_type_match=True, single doc number
            OR doc_type_match=True, multi-doc, BUT this is the dominant doc number
               (page_count >= 3x the second-most-common, i.e. clearly the project's own number)
    review: doc_type_match=True + multi-doc + no dominant number, OR main-doc cross-reference
    skip:   no type match and not main doc
    """
    if df.empty:
        return df

    # For multi-doc projects, find the dominant doc number per project
    # (dominant = page_count at least 2x the next highest, or appears in main doc alone)
    dominant_doc: dict[str, str] = {}  # project_id → dominant doc_number
    for pid, grp in df.groupby("project_id"):
        if grp["multi_doc_flag"].any():
            type_matched = grp[grp["doc_type_match"]]
            if type_matched.empty:
                continue
            sorted_grp = type_matched.sort_values("page_count", ascending=False)
            if len(sorted_grp) == 1:
                dominant_doc[pid] = sorted_grp.iloc[0]["doc_number"]
            else:
                top = sorted_grp.iloc[0]["page_count"]
                second = sorted_grp.iloc[1]["page_count"]
                # Accept as dominant if top is at least 2x second AND appears > 3 pages
                if top >= 2 * second and top >= 3:
                    dominant_doc[pid] = sorted_grp.iloc[0]["doc_number"]

    conditions = []
    for _, row in df.iterrows():
        pid = row["project_id"]
        if row["doc_type_match"] and not row["multi_doc_flag"]:
            conditions.append("accept")
        elif row["doc_type_match"] and row["multi_doc_flag"]:
            dom = dominant_doc.get(pid)
            if dom and row["doc_number"] == dom:
                conditions.append("accept")
            elif dom:
                conditions.append("skip")  # non-dominant match in a project we resolved
            else:
                conditions.append("review")  # ambiguous multi-doc
        elif not row["doc_type_match"] and row["evidence_rank"] == 1:
            conditions.append("review")
        else:
            conditions.append("skip")

    df = df.copy()
    df["acceptance"] = conditions
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", nargs="+", choices=["EA", "EIS"],
                        default=["EA", "EIS"])
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N DOE projects per process type (for testing)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    print("Loading DOE project IDs from projects_combined ...")
    projects = con.execute(f"""
        SELECT project_id, process_type
        FROM read_parquet('{PROJECTS_PATH}')
        WHERE lower(lead_agency_harmonized) LIKE '%department of energy%'
          AND process_type IN ({','.join("'" + p + "'" for p in args.process)})
    """).df()
    doe_project_ids = set(projects["project_id"].tolist())
    print(f"  {len(doe_project_ids)} DOE projects: "
          f"{projects['process_type'].value_counts().to_dict()}")

    all_parts = []
    for pt in args.process:
        part = scan_process_type(con, pt, doe_project_ids, args.sample)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        print("No DOE document numbers found.")
        return

    combined = pd.concat(all_parts, ignore_index=True)
    deduped = deduplicate(combined)
    gated = acceptance_gate(deduped)
    gated["scan_run_at"] = datetime.now(timezone.utc).isoformat()

    gated.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(gated)} rows → {OUTPUT_PATH}")
    print("\nAcceptance breakdown:")
    print(gated["acceptance"].value_counts().to_string())
    print("\nProcess type × acceptance:")
    print(gated.groupby(["process_type", "acceptance"]).size().to_string())
    print(f"\nUnique doc numbers (accept only): "
          f"{gated[gated['acceptance']=='accept']['doc_number'].nunique()}")
    print(f"Unique DOE projects with any match: {gated['project_id'].nunique()}")
    print("\nSample accepted doc numbers:")
    for dn in gated[gated["acceptance"] == "accept"]["doc_number"].drop_duplicates().head(15):
        print(f"  {dn}")


if __name__ == "__main__":
    main()
