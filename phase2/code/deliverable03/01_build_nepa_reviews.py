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
#   --geothermal  projects_geothermal_og.parquet  clean geothermal + oil/gas subset
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
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Reuse heading detection + canonical-topic logic from the inventory module.
# 03_inventory_visual_sections.py lives in the same deliverable03 directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
_inv = importlib.import_module("03_inventory_visual_sections")
is_heading_candidate = _inv.is_heading_candidate
canonical_topic = _inv.canonical_topic
HEADING_PREFIX_RE = _inv.HEADING_PREFIX_RE
KEYWORD_RE = _inv.KEYWORD_RE
page_reader = _inv.page_reader
TOPIC_REGEX = _inv.TOPIC_REGEX

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

# Stage 1-8 outputs (new visual pipeline)
VISUAL_SECTIONS_OUT     = OUT_DIR / "visual_sections.parquet"
PROJECTS_VISUAL_TEXT    = OUT_DIR / "projects_visual_text.parquet"
VISUAL_EMBED_CACHE      = OUT_DIR / "visual_chunk_embeddings.parquet"
VISUAL_FRAMING_OUT      = OUT_DIR / "visual_framing.parquet"
VISUAL_TOPICS_OUT       = OUT_DIR / "visual_topics.parquet"
VISUAL_TOPIC_SUMMARY    = OUT_DIR / "visual_topic_summary.parquet"
VISUAL_EXAMPLES_OUT     = OUT_DIR / "visual_examples.parquet"
VISUAL_QA_SAMPLE_OUT    = OUT_DIR / "visual_qa_sample.parquet"

# HTML / CSV outputs land in phase2/output/deliverable03/
OUTPUT_DIR = BASE_DIR / "output" / "deliverable03"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_SCATTERTEXT_HTML = OUTPUT_DIR / "visual_scattertext_decarb_vs_fossil.html"
VISUAL_QA_SAMPLE_CSV    = OUTPUT_DIR / "visual_qa_sample.csv"

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

    # Optional random sample — wrapped as a subquery so JOIN clauses parse cleanly.
    # DuckDB's USING SAMPLE cannot appear between a table alias and a JOIN keyword.
    # Scope to the 31,508 analysis universe: Clean (20,725) + Fossil (10,783).
    # "Other" projects are excluded — they are not energy projects and would
    # contaminate tech_group and energy_group distributions.
    # The WHERE filter must be in an inner subquery so USING SAMPLE draws from
    # the already-filtered 31,508 rows, not the full 61k+ table.
    scoped = f"SELECT * FROM read_parquet('{PROJECTS_PATH}') WHERE project_energy_type IN ('Clean', 'Fossil')"
    projects_src = (
        f"(SELECT * FROM ({scoped}) USING SAMPLE {sample} ROWS)"
        if sample else
        f"({scoped})"
    )

    # process_type is in projects_combined.parquet directly (p.process_type).
    # projects_reviews.parquet would add is_linear — use it if present, else NULL.
    reviews_join = (
        f"LEFT JOIN read_parquet('{REVIEWS_PATH}') r ON p.project_id = r.project_id"
        if REVIEWS_PATH.exists() else ""
    )
    is_linear_col = "r.is_linear," if REVIEWS_PATH.exists() else "NULL AS is_linear,"

    triggers_join = (
        f"LEFT JOIN read_parquet('{TRIGGERS_PATH}') t ON p.project_id = t.project_id"
        if TRIGGERS_PATH.exists() else ""
    )
    triggers_cols = "t.nepa_trigger_primary" if TRIGGERS_PATH.exists() else "NULL AS nepa_trigger_primary"

    if not REVIEWS_PATH.exists():
        log("build_reviews: projects_reviews.parquet not found; is_linear will be NULL (process_type read from projects_combined)")
    if not TRIGGERS_PATH.exists():
        log("build_reviews: WARNING -- projects_nepa_trigger.parquet not found; nepa_trigger_primary will be NULL")
    else:
        n_trigger = conn.execute(f"SELECT count(DISTINCT project_id) FROM read_parquet('{TRIGGERS_PATH}')").fetchone()[0]
        log(f"build_reviews: trigger file has {n_trigger:,} projects (clean energy only; fossil will have NULL triggers)")

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
            -- tech_group maps NEPATEC project_type labels to display categories.
            -- project_type is a JSON array; casting to VARCHAR lets us substring-match
            -- the known taxonomy labels (see phase1/notes/project_types.txt).
            -- Clean and fossil labels are gated by project_energy_type so cross-tagged
            -- records do not leak across comparison groups.
            CASE
                -- Clean: Renewable Energy Production labels
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Geothermal%'
                    THEN 'Geothermal'
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Wind%'
                    THEN 'Wind'
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Solar%'
                    THEN 'Solar'
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Electricity Transmission%'
                    THEN 'Transmission'
                WHEN p.project_energy_type = 'Clean'
                    AND (
                        p.project_type::VARCHAR LIKE '%Hydropower%'
                        OR p.project_type::VARCHAR LIKE '%Hydrokinetic%'
                    )
                    THEN 'Hydropower'
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Biomass%'
                    THEN 'Biomass'
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Energy Storage%'
                    THEN 'Energy Storage'
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Carbon Capture%'
                    THEN 'CCS'
                WHEN p.project_energy_type = 'Clean'
                    AND p.project_type::VARCHAR LIKE '%Nuclear%'
                    THEN 'Nuclear'
                -- Fossil: exact NEPATEC fossil labels (land-based and offshore are separate NEPATEC categories)
                WHEN p.project_energy_type = 'Fossil'
                    AND p.project_type::VARCHAR LIKE '%Land-based Oil%'
                    THEN 'Land-based Oil & Gas'
                WHEN p.project_energy_type = 'Fossil'
                    AND p.project_type::VARCHAR LIKE '%Offshore Oil%'
                    THEN 'Offshore Oil & Gas'
                WHEN p.project_energy_type = 'Fossil'
                    AND p.project_type::VARCHAR LIKE '%Coal%'
                    THEN 'Coal'
                WHEN p.project_energy_type = 'Fossil'
                    AND p.project_type::VARCHAR LIKE '%Pipeline%'
                    THEN 'Pipeline'
                WHEN p.project_energy_type = 'Fossil'
                    AND p.project_type::VARCHAR LIKE '%Rural Energy%'
                    THEN 'Rural Energy'
                -- Fallbacks by energy type (catches multi-label projects with no
                -- primary energy label, e.g. ["Research and Development", "Utilities"])
                WHEN p.project_energy_type = 'Clean'  THEN 'Other Clean'
                ELSE 'Other Fossil'
            END AS tech_group,
            CASE
                WHEN p.project_energy_type = 'Clean'  THEN 'Decarbonization'
                WHEN p.project_energy_type = 'Fossil' THEN 'Fossil Fuel'
            END AS energy_group,
            p.process_type,
            {is_linear_col}
            {triggers_cols},
            '{run_at}' AS nepa_reviews_extraction_run_at
        FROM {projects_src} p
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
    # pages only have document_id; join through documents.parquet to get project_id.
    # documents.parquet stores project_id as STRUCT("value" VARCHAR) — unwrap with .value
    return f"""
        SELECT d.project_id.value AS project_id, p.{pnum} AS page_num, p.page_text
        FROM read_parquet('{pages_path}') p
        JOIN read_parquet('{docs_path}') d USING (document_id)
        WHERE d.project_id.value IN ({ids_str})
          AND length(p.page_text) > 100
        ORDER BY d.project_id.value, p.{pnum}
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
            # documents.parquet stores project_id as STRUCT("value" VARCHAR) — unwrap with .value
            id_src = (
                f"SELECT DISTINCT d.project_id.value AS project_id "
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
# MODULE 3B: NEW VISUAL PIPELINE — STAGE 1 (sections) through STAGE 8 (QA)
# --------------------------
# These functions implement the overhaul described in
# phase2/plans/deliverable03_visual_impact.md. They run AFTER the legacy
# extract_visual_impacts() function, which is preserved unchanged for
# compatibility/calibration. Optional dependencies (bertopic, umap-learn,
# hdbscan, scattertext) are gated by try/except so the pipeline still
# completes when they are missing.

# --- Framing lexicon (Stage 2) --------------------------------------------
SIGNIFICANCE_LOW = re.compile(
    r"\b(no significant|not significant|less than significant|"
    r"minor|negligible|de minimis|insignificant)\b",
    re.I,
)
# Apply AFTER masking LOW matches so "no significant" does not double-count.
SIGNIFICANCE_HIGH = re.compile(
    r"\b(significant(?:ly)?|major|substantial|notable|"
    r"moderate[- ]to[- ]major)\b",
    re.I,
)
ADVERSITY_NEG = re.compile(
    r"\b(adverse(?:ly)?|degrade|impair|diminish|detract)\b", re.I,
)
ADVERSITY_POS = re.compile(r"\b(beneficial|enhance|improve)\b", re.I)
ADVERSITY_NONE = re.compile(r"\b(no effect|no change)\b", re.I)
MITIGATION_STRONG = re.compile(
    r"\b(fully mitigated|avoided|design features|mitigation measures|"
    r"BMP[s]?|best management practices)\b",
    re.I,
)
MITIGATION_WEAK = re.compile(
    r"\b(unavoidable|residual|cannot be fully mitigated|"
    r"would remain|long[- ]term adverse)\b",
    re.I,
)
MITIGATION_SPECIFIC = re.compile(
    r"\b(screen(?:ing)?|paint(?:ed)?|bury|buried|vegetative buffer|"
    r"lighting cutoff|key observation point|reduce height|tubular tower|"
    r"low[- ]reflective|color treatment)\b",
    re.I,
)
VRM_CLASS = re.compile(
    r"\bVRM\b[^.]{0,40}\bClass\s+(I{1,3}|IV)\b|"
    r"\bClass\s+(I{1,3}|IV)\b[^.]{0,40}\bVRM\b",
    re.I,
)

# Sentence splitter
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# --- NEPA stopwords (Stage 3 NMF/BERTopic) --------------------------------
# Domain-specific noise terms layered onto sklearn's ENGLISH_STOP_WORDS.
# Curated from frequent boilerplate in NEPA prose.
NEPA_DOMAIN_STOPWORDS = {
    # generic NEPA framing terms
    "project", "projects", "alternative", "alternatives", "action", "actions",
    "proposed", "would", "may", "shall", "must", "could", "might",
    "area", "areas", "site", "sites", "section", "sections",
    "appendix", "appendices", "table", "tables", "figure", "figures",
    "page", "pages", "see", "also", "et", "al", "etc",
    "include", "includes", "including", "included", "use", "used", "using",
    # NEPA-document boilerplate
    "nepa", "eis", "ea", "rod", "fonsi", "draft", "final", "deis", "feis",
    "agency", "agencies", "federal", "department", "bureau", "office",
    "applicant", "applicants", "operator", "lessee", "permittee",
    # process verbs
    "considered", "consider", "analyzed", "analysis", "evaluated",
    "evaluation", "assessment", "review", "reviewed", "described",
    "description", "identified", "identifies", "determined", "determination",
    # boilerplate hedge / connective
    "potential", "potentially", "associated", "result", "results",
    "resulting", "occur", "occurs", "occurred", "expected", "anticipated",
    "approximately", "generally", "typically", "primarily", "specific",
    "various", "additional", "applicable", "appropriate",
    # cardinal-direction & spatial fillers
    "north", "south", "east", "west", "northern", "southern", "eastern",
    "western", "located", "location", "locations",
    # high-frequency NEPA categories (visual-section-irrelevant)
    "resource", "resources", "environmental", "environment",
    # numerics that sneak past min_df
    "one", "two", "three", "four", "five", "ten", "first", "second",
}


def _make_nepa_stopwords() -> frozenset:
    """Return ENGLISH_STOP_WORDS | NEPA_DOMAIN_STOPWORDS as a frozenset.

    Imported lazily because sklearn import is slow.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS) | frozenset(NEPA_DOMAIN_STOPWORDS)


# --- Heading depth parser (Stage 1) ---------------------------------------

_HEADING_NUM_RE = re.compile(r"\s*([A-Z]?\d+(?:\.\d+)*)")


def heading_depth(prefix: str) -> int:
    """Parse the numeric prefix of a heading and return its depth.

    "3.8.2" -> 3, "3.8" -> 2, "3" -> 1, "A" -> 1, unknown -> 99 (sentinel).
    Termination rule consumes this: only stop on next heading whose depth
    is <= the start heading's depth.
    """
    m = _HEADING_NUM_RE.match(prefix or "")
    if not m:
        return 99
    return m.group(1).count(".") + 1


# --- Cleaning helpers (Stage 1) -------------------------------------------

_PAGE_NUM_ONLY_RE = re.compile(r"^\s*\d{1,4}\s*$")
_HYPHEN_LINEBREAK_RE = re.compile(r"-\n")
_WS_COLLAPSE_RE = re.compile(r"\s+")


def _clean_section_text(text: str, header_footer_lines: set[str]) -> str:
    """Fix OCR hyphenation, drop page-number lines and recurring headers/footers."""
    if not text:
        return ""
    # Repair hyphenated line breaks (e.g. "envir-\nonmental" -> "environmental")
    text = _HYPHEN_LINEBREAK_RE.sub("", text)
    kept: list[str] = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if _PAGE_NUM_ONLY_RE.match(stripped):
            continue
        if stripped in header_footer_lines:
            continue
        kept.append(stripped)
    joined = " ".join(kept)
    return _WS_COLLAPSE_RE.sub(" ", joined).strip()


def _collect_header_footer_lines(pages: list[str], threshold: int = 20) -> set[str]:
    """Return the set of short lines that recur >threshold× across all pages.

    Used to strip running headers/footers per the plan's cleaning rules.
    """
    counter: Counter = Counter()
    for ptext in pages:
        if not isinstance(ptext, str):
            continue
        for ln in ptext.splitlines():
            stripped = ln.strip()
            # Only worry about short-ish lines that look like header/footer text
            if 0 < len(stripped) <= 120:
                counter[stripped] += 1
    return {ln for ln, n in counter.items() if n > threshold}


# --- Chunking + embedding cache (Stage 1) ---------------------------------

def _chunk_text_by_words(text: str, target_words: int = 200) -> list[str]:
    """Split a long string into approximately-target_words chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    for i in range(0, len(words), target_words):
        chunks.append(" ".join(words[i : i + target_words]))
    return chunks


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# Patterns for false-positive heading rejection in extract_visual_sections()
# These match structural pages (preparers, references) and author-credit lines
# that contain visual terms but are not actual section headings.
_AUTHOR_CREDIT_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\s*:",  # "John Smith:", "Jane A. Doe:"
)
_PREPARER_PAGE_RE = re.compile(
    r"list\s+of\s+(?:preparers?|authors?|contributors?)"
    r"|chapter\s+\d+[\s\-–]+(?:preparers?|references?|bibliography)"
    r"|section\s+\d+[\s\-–]+(?:preparers?|references?|bibliography)"
    r"|\blist\s+of\s+references\b"
    r"|\bprepared\s+by\b",
    re.I,
)
_LOWERCASE_START_RE = re.compile(r"^[a-z]")  # heading starting lowercase = fragment
_BULLET_START_RE = re.compile(r"^[•‣◦⁃∙\-\*]\s")  # bullet/dash list item


def _is_false_positive_heading(heading_line: str, page_text: str) -> bool:
    """Return True if heading_line should be rejected as a false positive.

    Checks:
    1. Author-credit pattern: "Firstname Lastname: Visual Resources"
    2. Fragment: heading starts with a lowercase word (mid-sentence artifact)
    3. Bullet/list item: heading starts with a bullet or dash character
    4. Structural-page context: page contains preparer list / reference chapter markers
    """
    stripped = heading_line.strip()
    if _AUTHOR_CREDIT_RE.search(stripped):
        return True
    if _LOWERCASE_START_RE.match(stripped):
        return True
    if _BULLET_START_RE.match(stripped):
        return True
    if _PREPARER_PAGE_RE.search(page_text):
        return True
    return False


# ==========================================================================
# STAGE 1 — Section text extraction
# ==========================================================================

def extract_visual_sections(conn: duckdb.DuckDBPyConnection,
                            sample: Optional[int] = None) -> None:
    """Heading-anchored + fallback section extraction for visual impact analysis.

    Writes three parquets to phase2/data/analysis/deliverable03/:
      - visual_sections.parquet         (one row per section/run)
      - projects_visual_text.parquet    (one row per project, aggregated)
      - visual_chunk_embeddings.parquet (cache keyed by chunk_sha1)
    """
    log("extract_visual_sections: starting Stage 1", pct=50)
    if not REVIEWS_OUT.exists():
        log("extract_visual_sections: ERROR -- projects_nepa_reviews.parquet not found; run --reviews first")
        return

    run_at = datetime.now(timezone.utc).isoformat()

    # Pull EA + EIS page text via the reused page_reader. We collect pages
    # by (project_id, document_id) for both heading detection and fallback
    # contiguous-run extraction.
    section_rows: list[dict] = []
    project_meta: dict[str, dict] = {}
    project_pages: dict[tuple[str, str], list[tuple[int, str]]] = {}

    for source in ("EA", "EIS"):
        log(f"extract_visual_sections: reading {source} pages")
        reader = page_reader(conn, source, batch_size=50_000, include_supporting=False)
        n_rows = 0
        for batch in reader:
            pdf = batch.to_pandas()
            n_rows += len(pdf)
            for row in pdf.itertuples(index=False):
                if not isinstance(row.page_text, str) or len(row.page_text) <= 100:
                    continue
                # page_number may be int, numeric str, or prefixed str like 'Page-1'
                try:
                    pnum = int(row.page_number)
                except (TypeError, ValueError):
                    m = re.search(r"\d+", str(row.page_number))
                    if not m:
                        continue
                    pnum = int(m.group(0))
                key = (row.project_id, row.document_id)
                project_pages.setdefault(key, []).append(
                    (pnum, row.page_text)
                )
                if row.project_id not in project_meta:
                    project_meta[row.project_id] = {
                        "process_type": row.process_type,
                        "energy_group": row.energy_group,
                        "tech_group": row.tech_group,
                    }
        log(f"extract_visual_sections: {source} scanned {n_rows:,} pages")

    # Optional sample (random subset of project_ids)
    if sample:
        import random
        random.seed(42)
        all_pids = sorted(project_meta.keys())
        keep = set(random.sample(all_pids, min(sample, len(all_pids))))
        project_pages = {k: v for k, v in project_pages.items() if k[0] in keep}
        project_meta = {k: v for k, v in project_meta.items() if k in keep}
        log(f"extract_visual_sections: sample={sample} -> {len(project_meta):,} projects")

    log(f"extract_visual_sections: processing {len(project_pages):,} (project_id, document_id) docs", pct=55)

    # Iterate documents: build heading-anchored sections, fall back to
    # contiguous keyword-page runs when no heading was found.
    for (pid, doc_id), pages in project_pages.items():
        pages.sort(key=lambda x: x[0])  # by page_number
        page_nums = [p[0] for p in pages]
        page_texts = [p[1] for p in pages]
        meta = project_meta.get(pid, {})

        # Compute header/footer noise across ALL pages once per document
        hf_lines = _collect_header_footer_lines(page_texts, threshold=20)

        # --- Heading detection pass ---
        # heading_hits :: list of (page_idx_in_doc, heading_clean, section_title,
        #                          start_depth)
        heading_hits: list[tuple[int, str, str, int]] = []
        for i, ptext in enumerate(page_texts):
            prev_line = ""
            for raw in ptext.splitlines():
                hit = is_heading_candidate(raw, prev_line)
                if hit:
                    heading_clean, section_title, _, probable_toc = hit
                    if probable_toc:
                        prev_line = raw
                        continue
                    # Reject author-credit lines, lowercase fragments, and
                    # headings that appear on preparers/references pages.
                    if _is_false_positive_heading(heading_clean, ptext):
                        prev_line = raw
                        continue
                    depth = heading_depth(heading_clean)
                    heading_hits.append((i, heading_clean, section_title, depth))
                    break  # one heading per page is enough as an anchor
                if raw.strip():
                    prev_line = raw

        # --- Heading-anchored section extraction ---
        heading_section_pages: set[int] = set()
        for hidx, (start_i, heading_clean, section_title, start_depth) in enumerate(heading_hits):
            # Walk forward until we hit a same-or-shallower heading OR 50-page cap
            end_i = min(start_i + 50, len(page_texts) - 1)
            for j in range(start_i + 1, min(start_i + 51, len(page_texts))):
                # Inspect lines on page j for a heading whose depth <= start_depth
                terminate = False
                prev_line = ""
                for raw in page_texts[j].splitlines():
                    nxt = is_heading_candidate(raw, prev_line)
                    if nxt:
                        nxt_heading = nxt[0]
                        nxt_depth = heading_depth(nxt_heading)
                        # Stop ONLY on same-or-shallower numeric depth.
                        # Do NOT terminate on non-visual headings — most
                        # visual sections include Affected Environment /
                        # Environmental Consequences / Mitigation subsections.
                        if nxt_depth <= start_depth:
                            terminate = True
                            break
                    if raw.strip():
                        prev_line = raw
                if terminate:
                    end_i = j - 1
                    break
                end_i = j

            # Mark these pages as covered by a heading-anchored section.
            for k in range(start_i, end_i + 1):
                heading_section_pages.add(k)

            sec_text_raw = "\n".join(page_texts[start_i : end_i + 1])
            sec_text = _clean_section_text(sec_text_raw, hf_lines)
            n_words = len(sec_text.split())

            # Content density gate: require ≥2 visual-term mentions per 1,000 words.
            # Sections that pass heading detection but contain mostly unrelated prose
            # (glossaries, reference lists, traffic chapters) have very low density.
            if n_words > 0:
                density = len(KEYWORD_RE.findall(sec_text)) / n_words * 1000
                if density < 2.0:
                    continue

            section_rows.append({
                "project_id": pid,
                "document_id": doc_id,
                "process_type": meta.get("process_type", ""),
                "energy_group": meta.get("energy_group", ""),
                "tech_group": meta.get("tech_group", ""),
                "extraction_method": "heading_anchored",
                "page_start": page_nums[start_i],
                "page_end": page_nums[end_i],
                "n_pages": (end_i - start_i + 1),
                "heading_line": heading_clean,
                "canonical_topic": canonical_topic(section_title),
                "section_title": section_title,
                "n_words": n_words,
                "section_text": sec_text,
                "extraction_run_at": run_at,
            })

        # --- Fallback extraction: contiguous keyword-page runs ---
        # Only consider pages NOT already covered by a heading section.
        if not heading_hits:
            uncovered_idx = list(range(len(page_texts)))
        else:
            uncovered_idx = [i for i in range(len(page_texts))
                             if i not in heading_section_pages]

        # Per the plan, fallback applies to PROJECTS with no heading-anchored
        # section. If we DID find heading sections we skip fallback to avoid
        # noisy mini-runs scattered through unrelated chapters.
        if not heading_hits and uncovered_idx:
            # Find page indices that have >=1 visual term hit
            hit_idx_list: list[int] = []
            for i in uncovered_idx:
                if KEYWORD_RE.search(page_texts[i]):
                    hit_idx_list.append(i)

            if hit_idx_list:
                # Group into runs allowing gaps <=2 pages
                runs: list[list[int]] = []
                cur: list[int] = []
                for idx in hit_idx_list:
                    if not cur or idx - cur[-1] <= 3:  # gap of <=2 pages = diff <=3
                        cur.append(idx)
                    else:
                        runs.append(cur)
                        cur = [idx]
                if cur:
                    runs.append(cur)

                for run in runs:
                    # Drop isolated single-page hits
                    if len(run) < 2:
                        continue
                    start_i = run[0]
                    end_i = run[-1]
                    run_text_raw = "\n".join(page_texts[start_i : end_i + 1])
                    mentions = len(KEYWORD_RE.findall(run_text_raw))
                    # Require >=3 visual mentions across the run
                    if mentions < 3:
                        continue
                    sec_text = _clean_section_text(run_text_raw, hf_lines)
                    n_words = len(sec_text.split())
                    section_rows.append({
                        "project_id": pid,
                        "document_id": doc_id,
                        "process_type": meta.get("process_type", ""),
                        "energy_group": meta.get("energy_group", ""),
                        "tech_group": meta.get("tech_group", ""),
                        "extraction_method": "fallback_keyword_run",
                        "page_start": page_nums[start_i],
                        "page_end": page_nums[end_i],
                        "n_pages": (end_i - start_i + 1),
                        "heading_line": "",
                        "canonical_topic": "Other Visual / Aesthetic",
                        "section_title": "",
                        "n_words": n_words,
                        "section_text": sec_text,
                        "extraction_run_at": run_at,
                    })

    sections_df = pd.DataFrame(section_rows)
    if sections_df.empty:
        log("extract_visual_sections: WARNING -- no sections found; writing empty parquets")
        sections_df.to_parquet(VISUAL_SECTIONS_OUT, index=False)
        pd.DataFrame().to_parquet(PROJECTS_VISUAL_TEXT, index=False)
        pd.DataFrame().to_parquet(VISUAL_EMBED_CACHE, index=False)
        return

    sections_df.to_parquet(VISUAL_SECTIONS_OUT, index=False)
    log(f"extract_visual_sections: wrote {len(sections_df):,} sections -> {VISUAL_SECTIONS_OUT.name}", pct=60)

    # --- Project-level aggregate ---
    # Join lead_agency_harmonized from REVIEWS_OUT for the project-level table.
    reviews_meta = conn.execute(f"""
        SELECT project_id, energy_group, tech_group, process_type, lead_agency_harmonized
        FROM read_parquet('{REVIEWS_OUT}')
    """).fetchdf()

    grouped = sections_df.groupby("project_id", as_index=False).agg(
        n_sections=("section_text", "size"),
        n_words=("n_words", "sum"),
        n_pages_covered=("n_pages", "sum"),
        canonical_topics_found=("canonical_topic", lambda s: sorted(set(s.dropna().astype(str)))),
        visual_text=("section_text", lambda s: " \n\n ".join(t for t in s if t)),
        has_heading_extraction=(
            "extraction_method", lambda s: bool((s == "heading_anchored").any())
        ),
        fallback_used=(
            "extraction_method", lambda s: bool((s == "fallback_keyword_run").any())
        ),
    )
    grouped["n_chars"] = grouped["visual_text"].str.len()
    grouped["visual_text_clean"] = grouped["visual_text"].str.lower()
    grouped = grouped.merge(reviews_meta, on="project_id", how="left")
    grouped["extraction_run_at"] = run_at

    # Re-order columns to match plan
    proj_cols = [
        "project_id", "energy_group", "tech_group", "process_type",
        "lead_agency_harmonized",
        "n_sections", "n_words", "n_chars", "n_pages_covered",
        "has_heading_extraction", "fallback_used",
        "canonical_topics_found",
        "visual_text", "visual_text_clean",
        "extraction_run_at",
    ]
    grouped = grouped[[c for c in proj_cols if c in grouped.columns]]
    grouped.to_parquet(PROJECTS_VISUAL_TEXT, index=False)
    log(f"extract_visual_sections: wrote {len(grouped):,} projects -> {PROJECTS_VISUAL_TEXT.name}", pct=63)

    # --- Embedding cache (Stage 1b) ---
    # ~200-word chunks per project; cache keyed by (project_id, chunk_idx, chunk_sha1)
    log("extract_visual_sections: encoding chunk embeddings", pct=65)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cache_rows: list[dict] = []
    all_chunks: list[str] = []
    chunk_index: list[tuple[str, int, str]] = []  # (project_id, chunk_idx, sha1)
    for _, prow in grouped.iterrows():
        text = prow["visual_text"] or ""
        for ci, chunk in enumerate(_chunk_text_by_words(text, target_words=200)):
            sha = _sha1(chunk)
            chunk_index.append((prow["project_id"], ci, sha))
            all_chunks.append(chunk)

    if all_chunks:
        embs = model.encode(
            all_chunks, batch_size=256, normalize_embeddings=True, show_progress_bar=False
        )
        for (pid, ci, sha), emb in zip(chunk_index, embs):
            cache_rows.append({
                "project_id": pid,
                "chunk_idx": ci,
                "chunk_sha1": sha,
                "embedding": emb.tolist(),
                "embedding_run_at": run_at,
            })
    cache_df = pd.DataFrame(cache_rows)
    cache_df.to_parquet(VISUAL_EMBED_CACHE, index=False)
    log(f"extract_visual_sections: wrote {len(cache_df):,} chunk embeddings -> {VISUAL_EMBED_CACHE.name}", pct=68)


# ==========================================================================
# STAGE 2 — Framing / sentiment scoring
# ==========================================================================

def _count_framing_axes(text: str) -> dict:
    """Vectorized-friendly framing counter applied per project_text string.

    Splits text into sentences; applies LOW patterns first, masks their
    matches, then counts HIGH so "no significant" doesn't double-count.
    """
    if not isinstance(text, str) or not text:
        return {
            "sig_low": 0, "sig_high": 0,
            "adv_neg": 0, "adv_pos": 0, "adv_none": 0,
            "mit_strong": 0, "mit_weak": 0,
            "mit_specific_terms": [],
            "vrm_classes": [],
        }
    low = 0
    high = 0
    adv_neg = 0
    adv_pos = 0
    adv_none = 0
    mit_strong = 0
    mit_weak = 0
    mit_specific: set[str] = set()
    vrm_classes: set[str] = set()

    sentences = SENTENCE_SPLIT_RE.split(text)
    for sent in sentences:
        if not sent:
            continue
        # LOW first
        low_matches = list(SIGNIFICANCE_LOW.finditer(sent))
        low += len(low_matches)
        # Mask LOW spans before HIGH
        if low_matches:
            masked = list(sent)
            for m in low_matches:
                for k in range(m.start(), m.end()):
                    masked[k] = " "
            sent_masked = "".join(masked)
        else:
            sent_masked = sent
        high += len(SIGNIFICANCE_HIGH.findall(sent_masked))

        adv_neg += len(ADVERSITY_NEG.findall(sent))
        adv_pos += len(ADVERSITY_POS.findall(sent))
        adv_none += len(ADVERSITY_NONE.findall(sent))

        mit_strong += len(MITIGATION_STRONG.findall(sent))
        mit_weak += len(MITIGATION_WEAK.findall(sent))

        for m in MITIGATION_SPECIFIC.finditer(sent):
            mit_specific.add(m.group(0).lower())
        for m in VRM_CLASS.finditer(sent):
            cls = m.group(1) or m.group(2)
            if cls:
                vrm_classes.add(cls.upper())

    return {
        "sig_low": low, "sig_high": high,
        "adv_neg": adv_neg, "adv_pos": adv_pos, "adv_none": adv_none,
        "mit_strong": mit_strong, "mit_weak": mit_weak,
        "mit_specific_terms": sorted(mit_specific),
        "vrm_classes": sorted(vrm_classes),
    }


def build_framing(conn: duckdb.DuckDBPyConnection) -> None:
    """Compute CEQ-aligned framing axes per project. Writes visual_framing.parquet."""
    log("build_framing: starting Stage 2", pct=70)
    if not PROJECTS_VISUAL_TEXT.exists():
        log("build_framing: ERROR -- projects_visual_text.parquet missing; run extract_visual_sections first")
        return

    run_at = datetime.now(timezone.utc).isoformat()
    df = pd.read_parquet(PROJECTS_VISUAL_TEXT, columns=["project_id", "visual_text", "n_words"])

    rows = []
    for proj_id, vtxt, n_words in zip(df["project_id"], df["visual_text"], df["n_words"]):
        counts = _count_framing_axes(vtxt)
        total_words = max(int(n_words or 0), 1)
        per1000 = 1000.0 / total_words

        sig_total = counts["sig_high"] + counts["sig_low"]
        significance_ratio = (counts["sig_high"] / sig_total) if sig_total else 0.0
        mit_total = counts["mit_strong"] + counts["mit_weak"]
        mitigation_ratio = (counts["mit_strong"] / mit_total) if mit_total else 0.0

        rows.append({
            "project_id": proj_id,
            "n_words": total_words,
            "sig_low": counts["sig_low"],
            "sig_high": counts["sig_high"],
            "sig_low_per_1k": counts["sig_low"] * per1000,
            "sig_high_per_1k": counts["sig_high"] * per1000,
            "significance_ratio": significance_ratio,
            "adv_neg": counts["adv_neg"],
            "adv_pos": counts["adv_pos"],
            "adv_none": counts["adv_none"],
            "adv_neg_per_1k": counts["adv_neg"] * per1000,
            "adv_pos_per_1k": counts["adv_pos"] * per1000,
            "mit_strong": counts["mit_strong"],
            "mit_weak": counts["mit_weak"],
            "mit_strong_per_1k": counts["mit_strong"] * per1000,
            "mit_weak_per_1k": counts["mit_weak"] * per1000,
            "mitigation_ratio": mitigation_ratio,
            "mitigation_specificity": len(counts["mit_specific_terms"]),
            "mitigation_specific_terms": counts["mit_specific_terms"],
            "vrm_class_cited": bool(counts["vrm_classes"]),
            "vrm_classes": counts["vrm_classes"],
            "framing_run_at": run_at,
        })

    out = pd.DataFrame(rows)
    out.to_parquet(VISUAL_FRAMING_OUT, index=False)
    log(f"build_framing: wrote {len(out):,} rows -> {VISUAL_FRAMING_OUT.name}", pct=75)


# ==========================================================================
# STAGE 3 — Topic modeling (NMF guaranteed; BERTopic optional)
# ==========================================================================

try:
    from bertopic import BERTopic  # noqa: F401
    from umap import UMAP  # noqa: F401
    from hdbscan import HDBSCAN  # noqa: F401
    BERTOPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    BERTOPIC_AVAILABLE = False


def build_topics(conn: duckdb.DuckDBPyConnection) -> None:
    """NMF topic modeling (guaranteed) + optional BERTopic comparison.

    Writes visual_topics.parquet (per-project) and visual_topic_summary.parquet
    (per-topic with decarb/fossil/EA/EIS distributions).
    """
    log("build_topics: starting Stage 3", pct=78)
    if not PROJECTS_VISUAL_TEXT.exists():
        log("build_topics: ERROR -- projects_visual_text.parquet missing; aborting")
        return

    run_at = datetime.now(timezone.utc).isoformat()
    proj = pd.read_parquet(PROJECTS_VISUAL_TEXT)
    if proj.empty:
        log("build_topics: empty projects_visual_text; skipping topic model")
        pd.DataFrame().to_parquet(VISUAL_TOPICS_OUT, index=False)
        pd.DataFrame().to_parquet(VISUAL_TOPIC_SUMMARY, index=False)
        return

    # Train on heading-extracted projects only; transform fallback-only ones.
    train_mask = proj["has_heading_extraction"].fillna(False).astype(bool)
    train_df = proj[train_mask].copy()
    rest_df = proj[~train_mask].copy()
    log(f"build_topics: NMF training on {len(train_df):,} heading-anchored projects; "
        f"transforming {len(rest_df):,} fallback-only projects")

    # --- NMF (primary) ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    nepa_stop = list(_make_nepa_stopwords())

    vect = TfidfVectorizer(
        stop_words=nepa_stop,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        max_features=10000,
    )

    train_texts = train_df["visual_text_clean"].fillna("").tolist()
    rest_texts = rest_df["visual_text_clean"].fillna("").tolist()

    # Guard against tiny corpora (e.g. --sample)
    n_components = min(12, max(2, len(train_texts) // 10))
    if len(train_texts) < 5:
        log("build_topics: WARNING -- too few documents for NMF; writing empty outputs")
        pd.DataFrame({"project_id": proj["project_id"]}).to_parquet(VISUAL_TOPICS_OUT, index=False)
        pd.DataFrame().to_parquet(VISUAL_TOPIC_SUMMARY, index=False)
        return

    try:
        Xtr = vect.fit_transform(train_texts)
    except ValueError as e:
        log(f"build_topics: NMF vectorizer failed ({e}); writing empty topic outputs")
        pd.DataFrame({"project_id": proj["project_id"]}).to_parquet(VISUAL_TOPICS_OUT, index=False)
        pd.DataFrame().to_parquet(VISUAL_TOPIC_SUMMARY, index=False)
        return

    nmf = NMF(n_components=n_components, random_state=42, max_iter=400, alpha_W=0.001)
    W_train = nmf.fit_transform(Xtr)
    Xrest = vect.transform(rest_texts) if rest_texts else None
    W_rest = nmf.transform(Xrest) if Xrest is not None and Xrest.shape[0] else None

    topic_train = W_train.argmax(axis=1) if W_train.size else np.array([], dtype=int)
    prob_train = W_train.max(axis=1) if W_train.size else np.array([], dtype=float)
    if W_rest is not None and W_rest.size:
        topic_rest = W_rest.argmax(axis=1)
        prob_rest = W_rest.max(axis=1)
    else:
        topic_rest = np.array([], dtype=int)
        prob_rest = np.array([], dtype=float)

    # Top terms per topic
    feature_names = np.array(vect.get_feature_names_out())
    top_terms_by_topic: list[list[str]] = []
    for t in range(nmf.components_.shape[0]):
        idx = np.argsort(nmf.components_[t])[::-1][:5]
        top_terms_by_topic.append(feature_names[idx].tolist())
    topic_labels = [" / ".join(terms[:3]) for terms in top_terms_by_topic]

    # --- Optional BERTopic ---
    bertopic_train_topics = None
    bertopic_rest_topics = None
    bertopic_summary_rows: list[dict] = []
    if BERTOPIC_AVAILABLE and VISUAL_EMBED_CACHE.exists():
        try:
            log("build_topics: BERTopic available; running optional comparison")
            from bertopic import BERTopic as _BT
            from umap import UMAP as _UMAP
            from hdbscan import HDBSCAN as _HDB

            cache = pd.read_parquet(VISUAL_EMBED_CACHE)
            # Average chunk embeddings per project for a single doc-level vector.
            if not cache.empty and "embedding" in cache.columns:
                cache["embedding"] = cache["embedding"].apply(
                    lambda x: np.array(x, dtype=np.float32) if x is not None else None
                )
                emb_by_pid = (
                    cache.dropna(subset=["embedding"]).groupby("project_id")["embedding"]
                    .apply(lambda series: np.mean(np.stack(series.values), axis=0))
                    .to_dict()
                )
                train_pids = train_df["project_id"].tolist()
                rest_pids = rest_df["project_id"].tolist()
                tr_emb = np.stack([emb_by_pid[p] for p in train_pids if p in emb_by_pid])
                tr_text = [t for p, t in zip(train_pids, train_texts) if p in emb_by_pid]

                umap = _UMAP(n_neighbors=min(30, max(2, len(tr_emb) - 1)),
                             n_components=5, random_state=42)
                hdb = _HDB(min_cluster_size=20, min_samples=5)
                bt = _BT(umap_model=umap, hdbscan_model=hdb,
                         min_topic_size=15, calculate_probabilities=True,
                         embedding_model=None)
                bt_train_topics, _ = bt.fit_transform(tr_text, embeddings=tr_emb)
                bertopic_train_topics = dict(zip(
                    [p for p in train_pids if p in emb_by_pid], bt_train_topics
                ))

                if rest_pids:
                    rs_emb = np.stack([emb_by_pid[p] for p in rest_pids if p in emb_by_pid])
                    rs_text = [t for p, t in zip(rest_pids, rest_texts) if p in emb_by_pid]
                    if len(rs_emb):
                        bt_rest_topics, _ = bt.transform(rs_text, embeddings=rs_emb)
                        bertopic_rest_topics = dict(zip(
                            [p for p in rest_pids if p in emb_by_pid], bt_rest_topics
                        ))

                # BERTopic summary rows
                topic_info = bt.get_topic_info()
                for _, trow in topic_info.iterrows():
                    tid = int(trow["Topic"])
                    if tid == -1:
                        continue
                    top_words = [w for w, _ in (bt.get_topic(tid) or [])][:5]
                    bertopic_summary_rows.append({
                        "model": "bertopic",
                        "topic_id": tid,
                        "label": " / ".join(top_words[:3]),
                        "top_terms": top_words,
                        "n_total": int(trow.get("Count", 0)),
                    })
        except Exception as e:
            log(f"build_topics: BERTopic run failed ({e}); continuing with NMF only")
    elif not BERTOPIC_AVAILABLE:
        log("build_topics: bertopic/umap/hdbscan not installed; skipping BERTopic comparison")

    # --- Assemble per-project topics table ---
    pid_topic = {}
    pid_prob = {}
    for pid, t, p in zip(train_df["project_id"], topic_train, prob_train):
        pid_topic[pid] = int(t)
        pid_prob[pid] = float(p)
    for pid, t, p in zip(rest_df["project_id"], topic_rest, prob_rest):
        pid_topic[pid] = int(t)
        pid_prob[pid] = float(p)

    topics_df = pd.DataFrame({
        "project_id": proj["project_id"],
        "topic_nmf": [pid_topic.get(p, -1) for p in proj["project_id"]],
        "topic_nmf_prob": [pid_prob.get(p, 0.0) for p in proj["project_id"]],
        "topic_nmf_label": [
            topic_labels[pid_topic[p]] if pid_topic.get(p, -1) >= 0 else ""
            for p in proj["project_id"]
        ],
        "topic_bertopic": [
            (bertopic_train_topics or {}).get(p,
                (bertopic_rest_topics or {}).get(p, np.nan))
            if (bertopic_train_topics or bertopic_rest_topics) else np.nan
            for p in proj["project_id"]
        ],
        "topic_bertopic_prob": [np.nan] * len(proj),
        # NMF is the default chosen model. The user can manually promote
        # BERTopic after inspecting visual_topic_summary.parquet labels.
        "topic_chosen": [pid_topic.get(p, -1) for p in proj["project_id"]],
        "topic_chosen_model": ["nmf"] * len(proj),
        "topics_run_at": [run_at] * len(proj),
    })
    topics_df.to_parquet(VISUAL_TOPICS_OUT, index=False)
    log(f"build_topics: wrote {len(topics_df):,} project-topic rows -> {VISUAL_TOPICS_OUT.name}", pct=82)

    # --- Topic summary table ---
    meta_for_summary = proj[["project_id", "energy_group", "process_type",
                             "tech_group"]].copy()
    joined = topics_df.merge(meta_for_summary, on="project_id", how="left")
    summary_rows: list[dict] = []
    for t in range(nmf.components_.shape[0]):
        sub = joined[joined["topic_nmf"] == t]
        if sub.empty:
            continue
        top_tg = (
            sub["tech_group"].value_counts().head(3).index.tolist()
        )
        rep_ids = sub.sort_values("topic_nmf_prob", ascending=False)["project_id"].head(3).tolist()
        summary_rows.append({
            "model": "nmf",
            "topic_id": int(t),
            "label": topic_labels[t],
            "top_terms": top_terms_by_topic[t],
            "n_total": int(len(sub)),
            "n_decarb": int((sub["energy_group"] == "Decarbonization").sum()),
            "n_fossil": int((sub["energy_group"] == "Fossil Fuel").sum()),
            "n_ea": int((sub["process_type"] == "EA").sum()),
            "n_eis": int((sub["process_type"] == "EIS").sum()),
            "top_tech_groups": top_tg,
            "representative_doc_ids": rep_ids,
            "topic_summary_run_at": run_at,
        })

    # Merge in BERTopic summary rows (already have n_total) — augment with
    # decarb/fossil/EA/EIS using the per-project assignments captured above.
    if bertopic_summary_rows:
        bt_assign = pd.DataFrame({
            "project_id": list((bertopic_train_topics or {}).keys()) +
                          list((bertopic_rest_topics or {}).keys()),
            "topic_bertopic": list((bertopic_train_topics or {}).values()) +
                              list((bertopic_rest_topics or {}).values()),
        })
        bt_joined = bt_assign.merge(meta_for_summary, on="project_id", how="left")
        for r in bertopic_summary_rows:
            sub = bt_joined[bt_joined["topic_bertopic"] == r["topic_id"]]
            r["n_decarb"] = int((sub["energy_group"] == "Decarbonization").sum())
            r["n_fossil"] = int((sub["energy_group"] == "Fossil Fuel").sum())
            r["n_ea"] = int((sub["process_type"] == "EA").sum())
            r["n_eis"] = int((sub["process_type"] == "EIS").sum())
            r["top_tech_groups"] = sub["tech_group"].value_counts().head(3).index.tolist()
            r["representative_doc_ids"] = sub["project_id"].head(3).tolist()
            r["topic_summary_run_at"] = run_at
        summary_rows.extend(bertopic_summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(VISUAL_TOPIC_SUMMARY, index=False)
    log(f"build_topics: wrote {len(summary_df):,} topic-summary rows -> {VISUAL_TOPIC_SUMMARY.name}", pct=85)


# ==========================================================================
# STAGE 4 — Illustrative examples sampler
# ==========================================================================

def _trim_to_sentence_boundary(text: str, target_words: int = 250) -> str:
    """Truncate text near target_words, ending at the nearest sentence boundary."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= target_words:
        return text
    candidate = " ".join(words[: target_words + 50])
    # Find the LAST '.', '?', '!' in the candidate window
    last_term = max(candidate.rfind(". "), candidate.rfind("? "), candidate.rfind("! "))
    if last_term > 0:
        return candidate[: last_term + 1].strip()
    return " ".join(words[:target_words]).strip()


def build_examples(conn: duckdb.DuckDBPyConnection) -> None:
    """Pick 1-2 illustrative sections per (energy_group, tech_group) cell.

    Source: visual_sections.parquet (section-level, NOT project-level).
    """
    log("build_examples: starting Stage 4", pct=86)
    if not VISUAL_SECTIONS_OUT.exists():
        log("build_examples: ERROR -- visual_sections.parquet missing; aborting")
        return

    run_at = datetime.now(timezone.utc).isoformat()
    sections = pd.read_parquet(VISUAL_SECTIONS_OUT)
    if sections.empty:
        log("build_examples: empty sections parquet; writing empty examples output")
        pd.DataFrame().to_parquet(VISUAL_EXAMPLES_OUT, index=False)
        return

    # Pull project_title + lead_agency for downstream display
    titles = conn.execute(f"""
        SELECT project_id, project_title
        FROM read_parquet('{PROJECTS_PATH}')
    """).fetchdf()
    reviews_meta = conn.execute(f"""
        SELECT project_id, lead_agency_harmonized
        FROM read_parquet('{REVIEWS_OUT}')
    """).fetchdf()

    framing = (
        pd.read_parquet(VISUAL_FRAMING_OUT)
        if VISUAL_FRAMING_OUT.exists() else
        pd.DataFrame(columns=["project_id", "significance_ratio", "mitigation_ratio"])
    )

    # Filter sections to heading-anchored, 150-500 words
    filt = sections[
        (sections["extraction_method"] == "heading_anchored")
        & (sections["n_words"].between(150, 500))
    ].copy()
    if filt.empty:
        log("build_examples: no heading-anchored sections in 150-500 word window; writing empty output")
        pd.DataFrame().to_parquet(VISUAL_EXAMPLES_OUT, index=False)
        return

    # Drop cells with < 10 distinct projects entirely
    cell_proj_counts = (
        filt.groupby(["energy_group", "tech_group"])["project_id"]
        .nunique()
        .reset_index(name="n_projects_in_cell")
    )
    keep_cells = cell_proj_counts[cell_proj_counts["n_projects_in_cell"] >= 10][
        ["energy_group", "tech_group"]
    ]
    filt = filt.merge(keep_cells, on=["energy_group", "tech_group"], how="inner")

    # Bring framing scores in (significance + mitigation ratios)
    filt = filt.merge(
        framing[["project_id", "significance_ratio", "mitigation_ratio"]],
        on="project_id", how="left",
    )
    filt["significance_ratio"] = filt["significance_ratio"].fillna(0.0)
    filt["mitigation_ratio"] = filt["mitigation_ratio"].fillna(0.0)

    # For each cell, pick up to 2 contrasting sections:
    # - one significance-heavy (max significance_ratio)
    # - one mitigation-heavy (max mitigation_ratio)
    picked: list[dict] = []
    for (eg, tg), sub in filt.groupby(["energy_group", "tech_group"], dropna=False):
        sub_sorted_sig = sub.sort_values("significance_ratio", ascending=False)
        sub_sorted_mit = sub.sort_values("mitigation_ratio", ascending=False)
        rows: list[pd.Series] = []
        if not sub_sorted_sig.empty:
            rows.append(sub_sorted_sig.iloc[0])
        if not sub_sorted_mit.empty:
            cand = sub_sorted_mit.iloc[0]
            if not rows or cand["project_id"] != rows[0]["project_id"]:
                rows.append(cand)
        for r in rows:
            framing_summary = (
                "Significance-framed" if r["significance_ratio"] >= r["mitigation_ratio"]
                else "Mitigation-framed"
            )
            picked.append({
                "project_id": r["project_id"],
                "energy_group": eg,
                "tech_group": tg,
                "process_type": r["process_type"],
                "canonical_topic_primary": r["canonical_topic"],
                "framing_summary": framing_summary,
                "significance_ratio": float(r["significance_ratio"]),
                "mitigation_ratio": float(r["mitigation_ratio"]),
                "page_start": int(r["page_start"]),
                "page_end": int(r["page_end"]),
                "n_words_section": int(r["n_words"]),
                "excerpt": _trim_to_sentence_boundary(r["section_text"], 250),
                "examples_run_at": run_at,
            })

    out = pd.DataFrame(picked)
    if not out.empty:
        out = out.merge(titles, on="project_id", how="left")
        out = out.merge(reviews_meta, on="project_id", how="left")
        out = out.rename(columns={"lead_agency_harmonized": "lead_agency"})

    out.to_parquet(VISUAL_EXAMPLES_OUT, index=False)
    log(f"build_examples: wrote {len(out):,} excerpt rows -> {VISUAL_EXAMPLES_OUT.name}", pct=88)


# ==========================================================================
# STAGE 5 — Scattertext interactive HTML (optional)
# ==========================================================================

def build_scattertext(conn: duckdb.DuckDBPyConnection) -> None:
    """Optional Decarb-vs-Fossil scattertext explorer. Gated by import."""
    log("build_scattertext: starting Stage 5", pct=89)
    try:
        import scattertext as st  # noqa: F401
    except ImportError:
        log("build_scattertext: scattertext not installed; skipped")
        return

    if not PROJECTS_VISUAL_TEXT.exists():
        log("build_scattertext: ERROR -- projects_visual_text.parquet missing; aborting")
        return

    proj = pd.read_parquet(PROJECTS_VISUAL_TEXT)
    if proj.empty:
        log("build_scattertext: empty input; skipping")
        return

    # Need project_title for hover metadata
    titles = conn.execute(f"""
        SELECT project_id, project_title
        FROM read_parquet('{PROJECTS_PATH}')
    """).fetchdf()
    df = proj.merge(titles, on="project_id", how="left")
    df = df[df["visual_text_clean"].fillna("").str.len() > 0]
    df = df[df["energy_group"].isin(["Decarbonization", "Fossil Fuel"])]
    if df.empty:
        log("build_scattertext: no rows after filtering; skipping")
        return

    nepa_stop = list(_make_nepa_stopwords())
    try:
        corpus = (
            st.CorpusFromPandas(
                df,
                category_col="energy_group",
                text_col="visual_text_clean",
                nlp=st.whitespace_nlp_with_sentences,
            )
            .build()
            .remove_terms(nepa_stop, ignore_absences=True)
            .get_unigram_corpus()
        )
        html = st.produce_scattertext_explorer(
            corpus,
            category="Decarbonization",
            not_category_name="Fossil Fuel",
            width_in_pixels=1000,
            minimum_term_frequency=10,
            metadata=df["project_title"].fillna(df["project_id"]),
        )
        VISUAL_SCATTERTEXT_HTML.write_text(html, encoding="utf-8")
        log(f"build_scattertext: wrote {VISUAL_SCATTERTEXT_HTML.name}", pct=92)
    except Exception as e:
        log(f"build_scattertext: scattertext run failed ({e}); skipping")


# ==========================================================================
# STAGE 8 — Manual QA sample
# ==========================================================================

def build_qa_sample(conn: duckdb.DuckDBPyConnection) -> None:
    """Stratified 20-row sample for manual QA review against source PDFs."""
    log("build_qa_sample: starting Stage 8", pct=93)
    if not VISUAL_SECTIONS_OUT.exists():
        log("build_qa_sample: ERROR -- visual_sections.parquet missing; aborting")
        return

    run_at = datetime.now(timezone.utc).isoformat()
    sections = pd.read_parquet(VISUAL_SECTIONS_OUT)
    if sections.empty:
        log("build_qa_sample: empty sections; skipping")
        return

    # Stratified sample across energy_group × process_type × extraction_method.
    # Within each stratum, take proportional share of 20 with a floor of 1.
    strata_cols = ["energy_group", "process_type", "extraction_method"]
    grouped = sections.groupby(strata_cols, dropna=False)
    n_target = 20
    n_strata = len(grouped)
    if n_strata == 0:
        log("build_qa_sample: no strata; skipping")
        return

    per_stratum = max(1, n_target // n_strata)
    sampled_frames = []
    for _, sub in grouped:
        take = min(per_stratum, len(sub))
        sampled_frames.append(sub.sample(n=take, random_state=42))
    sampled = pd.concat(sampled_frames, ignore_index=True)
    if len(sampled) > n_target:
        sampled = sampled.sample(n=n_target, random_state=42).reset_index(drop=True)
    elif len(sampled) < n_target and len(sections) >= n_target:
        # Top up randomly from the rest
        remaining = sections.drop(sampled.index, errors="ignore")
        top_up = remaining.sample(
            n=min(n_target - len(sampled), len(remaining)), random_state=42
        )
        sampled = pd.concat([sampled, top_up], ignore_index=True)

    sampled["excerpt_first_400_chars"] = sampled["section_text"].str.slice(0, 400)
    sampled["qa_run_at"] = run_at
    out_cols = [
        "project_id", "document_id", "process_type", "energy_group",
        "tech_group", "extraction_method",
        "page_start", "page_end", "n_words",
        "excerpt_first_400_chars", "qa_run_at",
    ]
    sampled = sampled[[c for c in out_cols if c in sampled.columns]]
    sampled.to_parquet(VISUAL_QA_SAMPLE_OUT, index=False)
    sampled.to_csv(VISUAL_QA_SAMPLE_CSV, index=False)
    log(f"build_qa_sample: wrote {len(sampled):,} rows -> {VISUAL_QA_SAMPLE_OUT.name} (+ csv)", pct=95)


# --------------------------
# MODULE 4: build_geothermal_og
# --------------------------

def build_geothermal_og(conn: duckdb.DuckDBPyConnection) -> None:
    """Subset the base reviews table to Clean geothermal + all oil/gas projects."""
    log("build_geothermal_og: starting", pct=90)

    if not REVIEWS_OUT.exists():
        log("build_geothermal_og: ERROR -- projects_nepa_reviews.parquet not found; run --reviews first")
        return

    df = conn.execute(f"""
        SELECT *
        FROM read_parquet('{REVIEWS_OUT}')
        WHERE (tech_group = 'Geothermal' AND project_energy_type = 'Clean')
           OR tech_group IN ('Land-based Oil & Gas', 'Offshore Oil & Gas')
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
    # New visual pipeline (Stages 1-5, 8) runs AFTER the legacy extractor;
    # extract_visual_impacts() is preserved for fig12-14 calibration.
    extract_visual_sections(conn, sample)
    build_framing(conn)
    build_topics(conn)
    build_examples(conn)
    build_scattertext(conn)   # internally guards optional import
    build_qa_sample(conn)
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
        if args.visual:
            # Legacy extractor (untouched) — produces projects_visual_impacts.parquet
            # for fig12-14 calibration.
            extract_visual_impacts(conn, args.sample)
            # New visual pipeline (Stages 1-5, 8)
            extract_visual_sections(conn, args.sample)
            build_framing(conn)
            build_topics(conn)
            build_examples(conn)
            build_scattertext(conn)   # internally guards optional import
            build_qa_sample(conn)
        if args.geothermal: build_geothermal_og(conn)
        log("Done.", pct=100)
