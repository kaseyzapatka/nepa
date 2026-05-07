# --------------------------
# DELIVERABLE 3: NEPA Review Patterns — Data Builder
# --------------------------
# Builds all analysis datasets for D03 (fossil vs. decarbonization review patterns).
# Follows the same single-Python-script pattern as deliverable01.
#
# Modules (each has its own output parquet; default runs all):
#   --reviews     projects_nepa_reviews.parquet   base table: tech_group, process_type, triggers
#   --ce          ce_citations.parquet            one row per (project_id, CE citation)
#   --visual      projects_visual_impacts.parquet semantic search for visual impact sections
#   --geothermal  projects_geothermal_og.parquet  BLM geothermal + oil/gas subset
#
# Usage:
#   python phase2/code/deliverable03/01_build_nepa_reviews.py              # all modules
#   python phase2/code/deliverable03/01_build_nepa_reviews.py --reviews    # base only (fast)
#   python phase2/code/deliverable03/01_build_nepa_reviews.py --visual --sample 20
#
# Output: phase2/data/analysis/deliverable03/

import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate the 'nepa' conda environment before running.")

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --------------------------
# PATHS
# --------------------------

BASE_DIR  = Path(__file__).resolve().parent.parent.parent   # nepa/phase2/
DATA_DIR  = BASE_DIR / "data"
ANALYSIS  = DATA_DIR / "analysis"
PROCESSED = DATA_DIR / "processed"

# Input parquets
PROJECTS_PATH = ANALYSIS / "projects_combined.parquet"
REVIEWS_PATH  = ANALYSIS / "projects_reviews.parquet"
TRIGGERS_PATH = ANALYSIS / "nepa_trigger" / "projects_nepa_trigger.parquet"
DOCS_PATH     = ANALYSIS / "documents_combined.parquet"
EA_PAGES      = PROCESSED / "ea"  / "pages.parquet"
EA_DOCS       = PROCESSED / "ea"  / "documents.parquet"
EIS_PAGES     = PROCESSED / "eis" / "pages.parquet"
EIS_DOCS      = PROCESSED / "eis" / "documents.parquet"

# Output directory + parquets
OUT_DIR      = ANALYSIS / "deliverable03"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REVIEWS_OUT  = OUT_DIR / "projects_nepa_reviews.parquet"
CE_OUT       = OUT_DIR / "ce_citations.parquet"
VISUAL_OUT   = OUT_DIR / "projects_visual_impacts.parquet"
GEO_OG_OUT   = OUT_DIR / "projects_geothermal_og.parquet"

# --------------------------
# VISUAL IMPACT CONSTANTS
# --------------------------

VISUAL_QUERY = (
    "visual impacts of the project on viewsheds, aesthetics, "
    "visual resources, landscape character"
)
VISUAL_TOP_N  = 5
VISUAL_THRESH = 0.4
CHUNK_MAX     = 200
CHUNK_MIN     = 30
EMBED_BATCH   = 256

# Lexical prefilter: chunks must contain at least one term to reach the embedder.
# Pure embedding without a prefilter always returns something, even for projects
# with no visual content at all.
VISUAL_TERMS = re.compile(
    r"visual|viewshed|scenic|aesthetic|landscape|glare|contrast|"
    r"visual resource|VRM|visual quality objective|VQO|"
    r"shadow flicker|light pollution",
    re.IGNORECASE,
)
# Section headings that signal a dedicated visual impacts section
VISUAL_HEADINGS = re.compile(
    r"visual\s+(impacts?|resources?|quality)|aesthetics?|scenery|"
    r"landscape character|visual resource management",
    re.IGNORECASE,
)

# --------------------------
# PROGRESS LOGGER
# --------------------------

_START = time.time()


def log(msg: str, pct: Optional[float] = None) -> None:
    elapsed = (time.time() - _START) / 60
    ts = datetime.now().strftime("%H:%M:%S")
    pct_str = f"[{pct:3.0f}%]" if pct is not None else "      "
    print(f"[{ts}] {pct_str} ({elapsed:.1f}m) {msg}", flush=True)


# --------------------------
# MODULE 1: build_reviews
# --------------------------

def build_reviews(conn: duckdb.DuckDBPyConnection,
                  sample: Optional[int] = None) -> None:
    """Build the base project table with tech_group derivation and left-joined metadata."""
    log("build_reviews: starting", pct=0)
    run_at = datetime.now(timezone.utc).isoformat()

    # Optional random sample — applied to projects before joins
    sample_clause = f"USING SAMPLE {sample}" if sample else ""

    # Check which optional inputs exist; skip joins gracefully if not
    reviews_join = (
        f"LEFT JOIN read_parquet('{REVIEWS_PATH}') r ON p.project_id = r.project_id"
        if REVIEWS_PATH.exists() else ""
    )
    triggers_join = (
        f"LEFT JOIN read_parquet('{TRIGGERS_PATH}') t ON p.project_id = t.project_id"
        if TRIGGERS_PATH.exists() else ""
    )
    reviews_cols = "r.process_type, r.is_linear," if REVIEWS_PATH.exists() else "NULL AS process_type, NULL AS is_linear,"
    triggers_cols = "t.nepa_trigger_primary" if TRIGGERS_PATH.exists() else "NULL AS nepa_trigger_primary"

    if not REVIEWS_PATH.exists():
        log("build_reviews: WARNING -- projects_reviews.parquet not found; process_type/is_linear will be NULL")
    if not TRIGGERS_PATH.exists():
        log("build_reviews: WARNING -- projects_nepa_trigger.parquet not found; nepa_trigger_primary will be NULL")

    # All joins + tech_group derivation in one DuckDB query.
    # project_type is stored as a list or JSON string; cast to VARCHAR for regex matching.
    # Priority order matters: first match wins (e.g., a geothermal-wind hybrid -> Wind).
    df = conn.execute(f"""
        SELECT
            p.project_id,
            p.project_energy_type,
            p.lead_agency_harmonized,
            p.project_state,
            p.project_county,
            p.project_lat,
            p.project_lon,
            p.project_type,
            CASE
                WHEN regexp_matches(lower(p.project_type::VARCHAR), 'wind')
                    THEN 'Wind'
                WHEN regexp_matches(lower(p.project_type::VARCHAR), 'solar|photovoltaic|\\bpv\\b')
                    THEN 'Solar'
                WHEN regexp_matches(lower(p.project_type::VARCHAR), 'transmission')
                    THEN 'Transmission'
                WHEN regexp_matches(lower(p.project_type::VARCHAR), 'geothermal')
                    THEN 'Geothermal'
                WHEN regexp_matches(lower(p.project_type::VARCHAR), 'natural gas|lng|methane')
                    THEN 'Natural Gas'
                WHEN regexp_matches(lower(p.project_type::VARCHAR), 'oil|petroleum')
                    THEN 'Oil & Gas'
                WHEN p.project_energy_type = 'Clean'  THEN 'Other Clean'
                WHEN p.project_energy_type = 'Fossil' THEN 'Other Fossil'
                ELSE 'Other'
            END AS tech_group,
            CASE
                WHEN p.project_energy_type = 'Clean'  THEN 'Decarbonization'
                WHEN p.project_energy_type = 'Fossil' THEN 'Fossil Fuel'
                ELSE 'Other'
            END AS energy_group,
            {reviews_cols}
            {triggers_cols},
            '{run_at}' AS nepa_reviews_extraction_run_at
        FROM read_parquet('{PROJECTS_PATH}') p {sample_clause}
        {reviews_join}
        {triggers_join}
    """).fetchdf()

    df.to_parquet(REVIEWS_OUT, index=False)
    log(f"build_reviews: wrote {len(df):,} rows -> {REVIEWS_OUT.name}", pct=25)

    # Quick tech_group sanity check
    tg_counts = df["tech_group"].value_counts()
    log(f"build_reviews: tech_group distribution:\n{tg_counts.to_string()}")


# --------------------------
# MODULE 2: build_ce_citations
# --------------------------

# Normalization patterns applied in priority order.
# Each tuple: (label, regex, extractor_fn)
CE_CODE_PATTERNS = [
    ("blm",     r"^[A-Z]\d+(?:\.\d+)?$",        lambda m: m.group(0)),
    ("cfr_emb", r"\b([A-Z]\d+\.\d+)\b",          lambda m: m.group(1)),
    ("dm",      r"(\d+\s+DM\s+[\d\.]+)",          lambda m: m.group(1)),
    ("cfr",     r"(\d+\s+CFR\s+[\d\.]+)",         lambda m: m.group(1)),
    ("statute", r"(Section\s+\d+[^,]{0,40})",     lambda m: m.group(1)),
]


def _normalize_ce_code(raw: str) -> str:
    """Extract a short, normalized CE code from the raw ce_category string."""
    raw = raw.strip()
    for _, pat, fn in CE_CODE_PATTERNS:
        m = re.search(pat, raw)
        if m:
            return fn(m).strip()
    return raw[:60]  # fallback: first 60 chars of raw string


def build_ce_citations(conn: duckdb.DuckDBPyConnection,
                       sample: Optional[int] = None) -> None:
    """Parse ce_category from documents_combined and output one row per citation."""
    log("build_ce_citations: starting", pct=25)

    if not DOCS_PATH.exists():
        log("build_ce_citations: WARNING -- documents_combined.parquet not found; skipping")
        return

    # When sampling, scope to project_ids already in the reviews parquet
    sample_join = (
        f"INNER JOIN read_parquet('{REVIEWS_OUT}') s ON d.project_id = s.project_id"
        if sample and REVIEWS_OUT.exists() else ""
    )

    docs = conn.execute(f"""
        SELECT d.project_id, d.ce_category
        FROM read_parquet('{DOCS_PATH}') d
        {sample_join}
        WHERE d.ce_category IS NOT NULL
          AND d.ce_category NOT IN ('', 'nan', '[]')
    """).fetchdf()

    log(f"build_ce_citations: parsing {len(docs):,} documents", pct=35)

    rows = []
    for _, row in docs.iterrows():
        try:
            codes = json.loads(row["ce_category"])
        except (json.JSONDecodeError, TypeError):
            codes = [row["ce_category"]]
        if not isinstance(codes, list):
            codes = [codes]
        for raw in codes:
            raw = str(raw).strip()
            if raw and raw not in ("nan", "null", "[]"):
                rows.append({
                    "project_id":     row["project_id"],
                    "ce_raw":         raw,
                    "ce_code":        _normalize_ce_code(raw),
                    "ce_description": raw,
                })

    _CE_COLS = ["project_id", "ce_raw", "ce_code", "ce_description"]
    out = pd.DataFrame(rows, columns=_CE_COLS) if rows else pd.DataFrame(columns=_CE_COLS)
    out.to_parquet(CE_OUT, index=False)
    log(f"build_ce_citations: wrote {len(rows):,} rows -> {CE_OUT.name}", pct=50)


# --------------------------
# MODULE 3: extract_visual_impacts
# --------------------------

def _chunk_page(text: str) -> list[str]:
    """Split page text into ~200-word paragraph chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    for para in paragraphs:
        words = len(para.split())
        if words < CHUNK_MIN:
            buf.append(para)
            buf_words += words
            if buf_words >= CHUNK_MAX:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
        else:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            if words <= CHUNK_MAX:
                chunks.append(para)
            else:
                sents = para.split(". ")
                sbuf: list[str] = []
                sw = 0
                for s in sents:
                    w = len(s.split())
                    if sw + w > CHUNK_MAX and sbuf:
                        chunks.append(". ".join(sbuf))
                        sbuf, sw = [s], w
                    else:
                        sbuf.append(s)
                        sw += w
                if sbuf:
                    chunks.append(". ".join(sbuf))

    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(c.split()) >= CHUNK_MIN]


def _detect_page_schema(conn: duckdb.DuckDBPyConnection, pages_path: Path) -> str:
    """Return 'project_id' if pages have it directly, else 'document_id'."""
    cols = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{pages_path}') LIMIT 0"
    ).fetchdf()["column_name"].tolist()
    return "project_id" if "project_id" in cols else "document_id"


def _build_pages_query(pages_path: Path, docs_path: Path,
                       schema_key: str, ids_str: str) -> str:
    """Build a DuckDB query that always returns (project_id, page_num, page_text)."""
    # Column name for page number differs across datasets
    pnum = "page_number" if schema_key == "document_id" else "page_num"
    if schema_key == "project_id":
        return f"""
            SELECT project_id, {pnum} AS page_num, page_text
            FROM read_parquet('{pages_path}')
            WHERE project_id IN ({ids_str})
              AND length(page_text) > 100
            ORDER BY project_id, {pnum}
        """
    # pages only have document_id; join through documents.parquet to get project_id
    return f"""
        SELECT d.project_id, p.{pnum} AS page_num, p.page_text
        FROM read_parquet('{pages_path}') p
        JOIN read_parquet('{docs_path}') d USING (document_id)
        WHERE d.project_id IN ({ids_str})
          AND length(p.page_text) > 100
        ORDER BY d.project_id, p.{pnum}
    """


def extract_visual_impacts(conn: duckdb.DuckDBPyConnection,
                           sample: Optional[int] = None) -> None:
    """
    Hybrid visual impact extraction: lexical prefilter then embedding rerank.
    Processes EA and EIS pages only (CE forms lack substantive visual sections).
    """
    log("extract_visual_impacts: loading sentence-transformers model", pct=50)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode(VISUAL_QUERY, normalize_embeddings=True)
    results: list[dict] = []

    sources = [("EA", EA_PAGES, EA_DOCS), ("EIS", EIS_PAGES, EIS_DOCS)]
    for source, pages_path, docs_path in sources:
        if not pages_path.exists():
            log(f"extract_visual_impacts: {source} pages not found, skipping")
            continue

        schema_key = _detect_page_schema(conn, pages_path)
        log(f"extract_visual_impacts: {source} schema key = '{schema_key}'")

        # Get project IDs, joined against reviews parquet so we only process
        # projects in our analysis base
        if schema_key == "project_id":
            id_src = f"SELECT DISTINCT project_id FROM read_parquet('{pages_path}')"
        else:
            id_src = (
                f"SELECT DISTINCT d.project_id "
                f"FROM read_parquet('{pages_path}') p "
                f"JOIN read_parquet('{docs_path}') d USING (document_id)"
            )

        reviews_filter = (
            f"JOIN read_parquet('{REVIEWS_OUT}') r USING (project_id)"
            if REVIEWS_OUT.exists() else ""
        )
        project_ids = conn.execute(f"""
            SELECT DISTINCT src.project_id
            FROM ({id_src}) src
            {reviews_filter}
        """).fetchdf()["project_id"].tolist()

        if sample:
            import random
            random.seed(42)
            project_ids = random.sample(project_ids, min(sample, len(project_ids)))

        n_total   = len(project_ids)
        n_batches = max(1, (n_total + 49) // 50)
        log(f"extract_visual_impacts: {source} -- {n_total:,} projects, {n_batches} batches")

        for batch_i, start in enumerate(range(0, n_total, 50)):
            batch   = project_ids[start:start + 50]
            ids_str = ", ".join(f"'{p}'" for p in batch)

            pages = conn.execute(
                _build_pages_query(pages_path, docs_path, schema_key, ids_str)
            ).fetchdf()

            for pid, grp in pages.groupby("project_id"):
                all_chunks: list[str] = []
                section_found = False
                full_text = " ".join(grp["page_text"].tolist())
                mention_count = len(VISUAL_TERMS.findall(full_text))

                for _, r in grp.iterrows():
                    for chunk in _chunk_page(r["page_text"]):
                        if not VISUAL_TERMS.search(chunk):
                            continue  # lexical prefilter
                        if VISUAL_HEADINGS.search(chunk):
                            section_found = True
                        all_chunks.append(chunk)

                base_row: dict = {
                    "project_id":           pid,
                    "source":               source,
                    "visual_section_found": section_found,
                    "visual_mention_count": mention_count,
                    "visual_run_at":        datetime.now(timezone.utc).isoformat(),
                }

                if not all_chunks:
                    results.append({
                        **base_row,
                        "visual_impacts_max_similarity": 0.0,
                        "visual_impacts_text":           [],
                    })
                    continue

                embs = model.encode(
                    all_chunks, batch_size=EMBED_BATCH, normalize_embeddings=True
                )
                sims = np.dot(embs, query_emb)
                top  = np.argsort(sims)[-VISUAL_TOP_N:][::-1]
                results.append({
                    **base_row,
                    "visual_impacts_max_similarity": float(sims[top[0]]),
                    "visual_impacts_text":           [all_chunks[j] for j in top],
                })

            pct = 50 + 40 * (batch_i + 1) / n_batches
            log(
                f"extract_visual_impacts: {source} batch {batch_i + 1}/{n_batches} "
                f"({len(batch)} projects)",
                pct=pct,
            )

    raw = pd.DataFrame(results)

    # Deduplicate to one row per project: keep the EA/EIS source with highest similarity.
    # A project processed in both EA and EIS pages would otherwise appear twice.
    if not raw.empty:
        deduped = (
            raw.sort_values("visual_impacts_max_similarity", ascending=False)
               .drop_duplicates(subset="project_id", keep="first")
               .reset_index(drop=True)
        )
    else:
        deduped = raw

    deduped.to_parquet(VISUAL_OUT, index=False)
    log(
        f"extract_visual_impacts: wrote {len(deduped):,} rows (from {len(raw):,} "
        f"source rows) -> {VISUAL_OUT.name}",
        pct=90,
    )


# --------------------------
# MODULE 4: build_geothermal_og
# --------------------------

def build_geothermal_og(conn: duckdb.DuckDBPyConnection) -> None:
    """Subset the base reviews table to BLM geothermal + oil/gas projects."""
    log("build_geothermal_og: starting", pct=90)

    if not REVIEWS_OUT.exists():
        log("build_geothermal_og: ERROR -- projects_nepa_reviews.parquet not found; run --reviews first")
        return

    df = conn.execute(f"""
        SELECT *
        FROM read_parquet('{REVIEWS_OUT}')
        WHERE tech_group IN ('Geothermal', 'Oil & Gas', 'Natural Gas')
          AND regexp_matches(lead_agency_harmonized::VARCHAR,
                             'BLM|Bureau of Land Management')
    """).fetchdf()

    df.to_parquet(GEO_OG_OUT, index=False)
    log(f"build_geothermal_og: wrote {len(df):,} rows -> {GEO_OG_OUT.name}", pct=95)


# --------------------------
# DEFAULT: run all modules
# --------------------------

def run_all(conn: duckdb.DuckDBPyConnection, sample: Optional[int]) -> None:
    build_reviews(conn, sample)
    build_ce_citations(conn, sample)
    extract_visual_impacts(conn, sample)
    build_geothermal_og(conn)
    log("Done.", pct=100)


# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build D03 analysis datasets. "
            "Run with no module flags to execute all modules in sequence."
        )
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit to N random projects (for testing)",
    )
    parser.add_argument(
        "--reviews", action="store_true",
        help="Build projects_nepa_reviews.parquet (base table)",
    )
    parser.add_argument(
        "--ce", action="store_true",
        help="Build ce_citations.parquet",
    )
    parser.add_argument(
        "--visual", action="store_true",
        help="Build projects_visual_impacts.parquet (slow; runs sentence-transformers)",
    )
    parser.add_argument(
        "--geothermal", action="store_true",
        help="Build projects_geothermal_og.parquet (requires --reviews to have run first)",
    )
    args = parser.parse_args()

    conn     = duckdb.connect()
    any_flag = args.reviews or args.ce or args.visual or args.geothermal

    if not any_flag:
        run_all(conn, args.sample)
    else:
        if args.reviews:    build_reviews(conn, args.sample)
        if args.ce:         build_ce_citations(conn, args.sample)
        if args.visual:     extract_visual_impacts(conn, args.sample)
        if args.geothermal: build_geothermal_og(conn)
        log("Done.", pct=100)
