"""
Retrieve context packets for D4 timeline extraction.

Implements the five-tier retrieval strategy from plan §3:
  Tier A — structured metadata candidates (NOI date, filename dates, title dates)
  Tier B — high-priority document page slices (first/last + cue pages)
  Tier C — section-based retrieval from document_sections.parquet
  Tier D — page keyword scoring across all pages of priority documents
  Tier E — recovery retrieval for unresolved cases (expands to defer documents)

Outputs:
    phase2/data/analysis/timeline/timeline_context_packets.parquet

Usage:
    python 02_retrieve_timeline_contexts.py [--process CE EA EIS] [--sample-ids path]
    python 02_retrieve_timeline_contexts.py --process CE --sample-ids ids.txt
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
PROCESSED_DIR = PHASE2 / "data" / "processed"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
SECTIONS_PATH = ANALYSIS_DIR / "document_sections.parquet"
INDEX_PATH = TIMELINE_DIR / "timeline_document_index.parquet"
OUTPUT_PATH = TIMELINE_DIR / "timeline_context_packets.parquet"

SOURCE_MAP = {"CE": "ce", "EA": "ea", "EIS": "eis"}

# Per-process packet caps (plan §3)
PACKET_CAPS = {"CE": 25, "EA": 75, "EIS": 150}

# CE section skip threshold: skip section retrieval for CE docs with <=20 total pages
CE_SECTION_SKIP_PAGES = 20

# Page slice sizes for Tier B
TIER_B_FIRST_LAST = 3  # first/last N pages of high-priority documents

# CE expanded pass threshold
CE_EXPANDED_ALL_PAGES_THRESHOLD = 50

# ---------------------------------------------------------------------------
# Page cue keywords (plan §3 page scoring)
# ---------------------------------------------------------------------------
INITIATION_CUES = re.compile(
    r"\b("
    r"application|submitted|received|filed|request|right[-\s]of[-\s]way|row|apd|"
    r"plan\s+of\s+development|pod|license\s+application|notice\s+of\s+intent|noi|"
    r"scoping|public\s+scoping|comment\s+period|environmental\s+review\s+began|"
    r"project\s+initiation|proposed\s+action\s+received|application\s+date"
    r")\b",
    re.IGNORECASE,
)
DECISION_CUES = re.compile(
    r"\b("
    r"signed|approved|issued|authorized|determined|selected\s+alternative|"
    r"categorical\s+exclusion|determination|fonsi|finding\s+of\s+no\s+significant\s+impact|"
    r"record\s+of\s+decision|rod|decision\s+record|decision\s+notice|"
    r"approval\s+date|date\s+signed|date\s+approved|decision\s+date"
    r")\b",
    re.IGNORECASE,
)
NEGATIVE_CUES = re.compile(
    r"\b("
    r"references|bibliography|literature\s+cited|map|figure|table\s+of\s+contents|"
    r"preparers|comment\s+response|appendix|cfr|usc|\bfr\b|public\s+law|"
    r"act\s+of\s+19|act\s+of\s+20|omb|form\s+approved|revised|"
    r"isbn|doi:|accessed\s+on|retrieved\s+on"
    r")\b",
    re.IGNORECASE,
)

# Signature / approval block cues that boost CE decision page detection
SIGNATURE_CUES = re.compile(
    r"\b("
    r"signature|nepa\s+compliance\s+officer|authorizing\s+official|field\s+manager|"
    r"district\s+manager|certifying\s+official|date\s+signed|approved\s+by|"
    r"concurrence|authorized\s+officer"
    r")\b",
    re.IGNORECASE,
)

# Section heading cues for Tier C
INITIATION_SECTION_CUES = re.compile(
    r"\b(introduction|background|project\s+history|purpose\s+and\s+need|"
    r"proposed\s+action|scoping|public\s+involvement|consultation|application|"
    r"notice\s+of\s+intent|project\s+description)\b",
    re.IGNORECASE,
)
DECISION_SECTION_CUES = re.compile(
    r"\b(decision|selected\s+alternative|record\s+of\s+decision|"
    r"finding\s+of\s+no\s+significant\s+impact|fonsi|decision\s+record|"
    r"decision\s+notice|approval|ce\s+determination)\b",
    re.IGNORECASE,
)
NEGATIVE_SECTION_CUES = re.compile(
    r"\b(references|bibliography|preparers|distribution\s+list|index|"
    r"comments\s+and\s+responses|appendix|table\s+of\s+contents)\b",
    re.IGNORECASE,
)


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _packet_id(project_id: str, document_id: str | None, tier: str, page_or_section: str) -> str:
    raw = f"{project_id}|{document_id}|{tier}|{page_or_section}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _score_page(text: str) -> tuple[float, float, float]:
    """Return (initiation_score, decision_score, negative_score) for page text."""
    if not text:
        return 0.0, 0.0, 0.0
    init_score = float(len(INITIATION_CUES.findall(text)))
    dec_score = float(len(DECISION_CUES.findall(text))) + 2.0 * float(
        len(SIGNATURE_CUES.findall(text))
    )
    neg_score = float(len(NEGATIVE_CUES.findall(text)))
    return init_score, dec_score, neg_score


def _section_role(heading_title: str | None) -> str:
    if NEGATIVE_SECTION_CUES.search(str(heading_title or "")):
        return "negative"
    if DECISION_SECTION_CUES.search(str(heading_title or "")):
        return "decision_likely"
    if INITIATION_SECTION_CUES.search(str(heading_title or "")):
        return "initiation_likely"
    return "neutral"


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def build_tier_a_packets(
    project_row: pd.Series,
    run_at: str,
) -> list[dict]:
    """Tier A: structured metadata — NOI date, filename dates, title dates."""
    packets: list[dict] = []
    project_id = project_row["project_id"]

    # NOI date as Tier A initiation candidate
    if project_row.get("noi_tier_a_eligible"):
        noi_date = project_row.get("noi_publication_date")
        if pd.notna(noi_date):
            noi_str = pd.Timestamp(noi_date).strftime("%Y-%m-%d") if hasattr(noi_date, "strftime") else str(noi_date)
            context = f"FR NOI publication date: {noi_str}"
            packets.append({
                "context_packet_id": _packet_id(project_id, None, "tier_a", f"noi_{noi_str}"),
                "project_id": project_id,
                "process_type": project_row["process_type"],
                "document_id": None,
                "document_title": None,
                "document_type_clean": None,
                "document_type_category": None,
                "main_document": None,
                "section_id": None,
                "page_start": None,
                "page_end": None,
                "page_numbers": "[]",
                "retrieval_mode": "first_pass",
                "retrieval_reason": "fr_noi_tier_a",
                "source_tier": "metadata",
                "retrieval_tier": "tier_a",
                "retrieval_score": 5.0,
                "initiation_page_score": 5.0,
                "decision_page_score": 0.0,
                "negative_page_score": 0.0,
                "heading_title": None,
                "parent_heading_title": None,
                "context_text": context,
                "context_chars": len(context),
                "estimated_tokens": max(1, len(context) // 4),
                "context_hash": _text_hash(context),
                "api_eligible": True,
                "created_at": run_at,
            })

    # BLM register Tier A — decision date
    if project_row.get("blm_decision_tier_a_eligible"):
        blm_date = project_row.get("blm_decision_date")
        if pd.notna(blm_date):
            blm_str = str(blm_date)
            date_type = project_row.get("blm_decision_date_type") or "decision"
            context = f"BLM NEPA Register decision date ({date_type}): {blm_str}"
            packets.append({
                "context_packet_id": _packet_id(project_id, None, "tier_a", f"blm_decision_{blm_str}"),
                "project_id": project_id,
                "process_type": project_row["process_type"],
                "document_id": None,
                "document_title": None,
                "document_type_clean": None,
                "document_type_category": None,
                "main_document": None,
                "section_id": None,
                "page_start": None,
                "page_end": None,
                "page_numbers": "[]",
                "retrieval_mode": "first_pass",
                "retrieval_reason": "blm_register_decision",
                "source_tier": "metadata",
                "retrieval_tier": "tier_a",
                "retrieval_score": 5.0,
                "initiation_page_score": 0.0,
                "decision_page_score": 5.0,
                "negative_page_score": 0.0,
                "heading_title": None,
                "parent_heading_title": None,
                "context_text": context,
                "context_chars": len(context),
                "estimated_tokens": max(1, len(context) // 4),
                "context_hash": _text_hash(context),
                "api_eligible": True,
                "created_at": run_at,
            })

    # BLM register Tier A — initiation date
    if project_row.get("blm_initiation_tier_a_eligible"):
        blm_init = project_row.get("blm_initiation_date")
        if pd.notna(blm_init):
            blm_str = str(blm_init)
            context = f"BLM NEPA Register project start date: {blm_str}"
            packets.append({
                "context_packet_id": _packet_id(project_id, None, "tier_a", f"blm_initiation_{blm_str}"),
                "project_id": project_id,
                "process_type": project_row["process_type"],
                "document_id": None,
                "document_title": None,
                "document_type_clean": None,
                "document_type_category": None,
                "main_document": None,
                "section_id": None,
                "page_start": None,
                "page_end": None,
                "page_numbers": "[]",
                "retrieval_mode": "first_pass",
                "retrieval_reason": "blm_register_initiation",
                "source_tier": "metadata",
                "retrieval_tier": "tier_a",
                "retrieval_score": 5.0,
                "initiation_page_score": 5.0,
                "decision_page_score": 0.0,
                "negative_page_score": 0.0,
                "heading_title": None,
                "parent_heading_title": None,
                "context_text": context,
                "context_chars": len(context),
                "estimated_tokens": max(1, len(context) // 4),
                "context_hash": _text_hash(context),
                "api_eligible": True,
                "created_at": run_at,
            })

    # DOE register Tier A — decision date (FONSI or ROD)
    if project_row.get("doe_decision_tier_a_eligible"):
        doe_date = project_row.get("doe_decision_date")
        if pd.notna(doe_date):
            doe_str = str(doe_date)
            date_type = project_row.get("doe_decision_date_type") or "decision"
            doe_num = project_row.get("doe_doc_number") or ""
            context = f"DOE NEPA Register decision date ({date_type}, {doe_num}): {doe_str}"
            packets.append({
                "context_packet_id": _packet_id(project_id, None, "tier_a", f"doe_decision_{doe_str}"),
                "project_id": project_id,
                "process_type": project_row["process_type"],
                "document_id": None,
                "document_title": None,
                "document_type_clean": None,
                "document_type_category": None,
                "main_document": None,
                "section_id": None,
                "page_start": None,
                "page_end": None,
                "page_numbers": "[]",
                "retrieval_mode": "first_pass",
                "retrieval_reason": "doe_register_decision",
                "source_tier": "metadata",
                "retrieval_tier": "tier_a",
                "retrieval_score": 5.0,
                "initiation_page_score": 0.0,
                "decision_page_score": 5.0,
                "negative_page_score": 0.0,
                "heading_title": None,
                "parent_heading_title": None,
                "context_text": context,
                "context_chars": len(context),
                "estimated_tokens": max(1, len(context) // 4),
                "context_hash": _text_hash(context),
                "api_eligible": True,
                "created_at": run_at,
            })

    # DOE register Tier A — initiation date (NOI)
    if project_row.get("doe_initiation_tier_a_eligible"):
        doe_init = project_row.get("doe_initiation_date")
        if pd.notna(doe_init):
            doe_str = str(doe_init)
            doe_num = project_row.get("doe_doc_number") or ""
            context = f"DOE NEPA Register NOI date ({doe_num}): {doe_str}"
            packets.append({
                "context_packet_id": _packet_id(project_id, None, "tier_a", f"doe_initiation_{doe_str}"),
                "project_id": project_id,
                "process_type": project_row["process_type"],
                "document_id": None,
                "document_title": None,
                "document_type_clean": None,
                "document_type_category": None,
                "main_document": None,
                "section_id": None,
                "page_start": None,
                "page_end": None,
                "page_numbers": "[]",
                "retrieval_mode": "first_pass",
                "retrieval_reason": "doe_register_initiation",
                "source_tier": "metadata",
                "retrieval_tier": "tier_a",
                "retrieval_score": 5.0,
                "initiation_page_score": 5.0,
                "decision_page_score": 0.0,
                "negative_page_score": 0.0,
                "heading_title": None,
                "parent_heading_title": None,
                "context_text": context,
                "context_chars": len(context),
                "estimated_tokens": max(1, len(context) // 4),
                "context_hash": _text_hash(context),
                "api_eligible": True,
                "created_at": run_at,
            })

    return packets


def build_tier_b_packets(
    doc_rows: pd.DataFrame,
    pages_df: pd.DataFrame,
    project_id: str,
    process_type: str,
    run_at: str,
) -> list[dict]:
    """Tier B: high-priority document page slices."""
    packets: list[dict] = []

    priority_docs = doc_rows[doc_rows["scan_priority"].isin(["priority_1", "priority_2"])]

    for _, doc in priority_docs.iterrows():
        doc_id = doc["document_id"]
        doc_pages = pages_df[pages_df["document_id"] == doc_id].copy()
        if doc_pages.empty:
            continue
        doc_pages["page_num"] = pd.to_numeric(doc_pages["page_number"], errors="coerce").fillna(0).astype(int)
        doc_pages = doc_pages.sort_values("page_num")
        total = len(doc_pages)

        # Score all pages
        doc_pages[["init_score", "dec_score", "neg_score"]] = pd.DataFrame(
            doc_pages["page_text"].map(_score_page).tolist(),
            index=doc_pages.index,
        )

        # For CE small docs, scan all pages
        if process_type == "CE" and total <= CE_SECTION_SKIP_PAGES:
            selected_pages = doc_pages
            reason = "ce_small_doc_all_pages"
        elif process_type == "CE" and total <= CE_EXPANDED_ALL_PAGES_THRESHOLD:
            selected_pages = doc_pages
            reason = "ce_expanded_all_pages"
        else:
            # First N + last N pages
            first_idx = set(range(min(TIER_B_FIRST_LAST, total)))
            last_idx = set(range(max(0, total - TIER_B_FIRST_LAST), total))
            # Top cue pages
            top_init = set(doc_pages.nlargest(3, "init_score").index.tolist())
            top_dec = set(doc_pages.nlargest(3, "dec_score").index.tolist())
            pos_set = first_idx | last_idx
            # Map positional indices back to iloc
            all_positions = sorted(
                pos_set
                | {doc_pages.index[i] for i in first_idx}
                | {doc_pages.index[i] for i in last_idx}
                | top_init | top_dec
            )
            selected_pages = doc_pages.loc[list(set(doc_pages.index) & set(all_positions))]
            reason = "first_last_cue_pages"

        for _, page in selected_pages.iterrows():
            text = _clean_text(page.get("page_text"))
            if not text:
                continue
            page_num = str(page.get("page_number", ""))
            init_s, dec_s, neg_s = page.get("init_score", 0.0), page.get("dec_score", 0.0), page.get("neg_score", 0.0)
            retrieval_score = init_s + dec_s - 0.5 * neg_s
            packets.append({
                "context_packet_id": _packet_id(project_id, doc_id, "tier_b", page_num),
                "project_id": project_id,
                "process_type": process_type,
                "document_id": doc_id,
                "document_title": doc.get("document_title"),
                "document_type_clean": doc.get("document_type_clean"),
                "document_type_category": doc.get("document_type_category"),
                "main_document": doc.get("main_document"),
                "section_id": None,
                "page_start": page_num,
                "page_end": page_num,
                "page_numbers": json.dumps([page_num]),
                "retrieval_mode": "first_pass",
                "retrieval_reason": reason,
                "source_tier": "page_slice",
                "retrieval_tier": "tier_b",
                "retrieval_score": retrieval_score,
                "initiation_page_score": init_s,
                "decision_page_score": dec_s,
                "negative_page_score": neg_s,
                "heading_title": None,
                "parent_heading_title": None,
                "context_text": _truncate(text, 2000),
                "context_chars": len(text),
                "estimated_tokens": max(1, len(text) // 4),
                "context_hash": _text_hash(text),
                "api_eligible": retrieval_score > 0,
                "created_at": run_at,
            })

    return packets


def build_tier_c_packets(
    doc_rows: pd.DataFrame,
    sections_df: pd.DataFrame,
    project_id: str,
    process_type: str,
    run_at: str,
) -> list[dict]:
    """Tier C: section-based retrieval."""
    packets: list[dict] = []

    # CE section policy: skip section retrieval for short CE documents
    if process_type == "CE":
        doc_rows = doc_rows[
            doc_rows["doc_page_count"] > CE_SECTION_SKIP_PAGES
        ]
        if doc_rows.empty:
            return packets

    priority_docs = doc_rows[
        doc_rows["scan_priority"].isin(["priority_1", "priority_2"]) &
        doc_rows["has_sections"]
    ]
    if priority_docs.empty:
        return packets

    doc_ids = set(priority_docs["document_id"])
    proj_sections = sections_df[sections_df["document_id"].isin(doc_ids)].copy()
    if proj_sections.empty:
        return packets

    for _, sec in proj_sections.iterrows():
        heading = sec.get("heading_title", "")
        section_text = sec.get("section_text", "")
        role = _section_role(heading)

        if role == "negative":
            continue
        # Only retrieve timeline-relevant sections
        if role not in ("decision_likely", "initiation_likely"):
            continue

        text = _clean_text(section_text)
        if not text:
            continue

        # Score the section text as a page
        init_s, dec_s, neg_s = _score_page(text[:3000])
        retrieval_score = init_s + dec_s - 0.5 * neg_s

        sec_id = f"{sec.get('document_id')}_{sec.get('page_start')}_{sec.get('page_end')}"
        packets.append({
            "context_packet_id": _packet_id(project_id, sec.get("document_id"), "tier_c", sec_id),
            "project_id": project_id,
            "process_type": process_type,
            "document_id": sec.get("document_id"),
            "document_title": sec.get("document_title"),
            "document_type_clean": None,
            "document_type_category": None,
            "main_document": None,
            "section_id": sec_id,
            "page_start": sec.get("page_start"),
            "page_end": sec.get("page_end"),
            "page_numbers": json.dumps(
                list(range(int(sec.get("page_start", 0)), int(sec.get("page_end", 0)) + 1))
            ),
            "retrieval_mode": "first_pass",
            "retrieval_reason": f"section_{role}",
            "source_tier": "section",
            "retrieval_tier": "tier_c",
            "retrieval_score": retrieval_score,
            "initiation_page_score": init_s,
            "decision_page_score": dec_s,
            "negative_page_score": neg_s,
            "heading_title": heading,
            "parent_heading_title": sec.get("parent_heading_title"),
            "context_text": _truncate(text, 3000),
            "context_chars": len(text),
            "estimated_tokens": max(1, len(text) // 4),
            "context_hash": _text_hash(text),
            "api_eligible": retrieval_score > 0,
            "created_at": run_at,
        })

    return packets


def build_tier_d_packets(
    doc_rows: pd.DataFrame,
    pages_df: pd.DataFrame,
    project_id: str,
    process_type: str,
    run_at: str,
    top_n: int = 10,
) -> list[dict]:
    """Tier D: page keyword scoring across all pages of priority documents."""
    packets: list[dict] = []

    priority_docs = doc_rows[doc_rows["scan_priority"].isin(["priority_1", "priority_2", "priority_3"])]

    all_scored: list[dict] = []
    for _, doc in priority_docs.iterrows():
        doc_id = doc["document_id"]
        doc_pages = pages_df[pages_df["document_id"] == doc_id]
        for _, page in doc_pages.iterrows():
            text = _clean_text(page.get("page_text"))
            if not text:
                continue
            init_s, dec_s, neg_s = _score_page(text)
            retrieval_score = init_s + dec_s - 0.5 * neg_s
            if retrieval_score <= 0:
                continue
            all_scored.append({
                "doc": doc,
                "page": page,
                "text": text,
                "init_s": init_s,
                "dec_s": dec_s,
                "neg_s": neg_s,
                "retrieval_score": retrieval_score,
            })

    # Take top N by score to avoid duplicating Tier B high-value pages
    all_scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    for item in all_scored[:top_n]:
        doc = item["doc"]
        page = item["page"]
        text = item["text"]
        page_num = str(page.get("page_number", ""))
        packets.append({
            "context_packet_id": _packet_id(project_id, doc["document_id"], "tier_d", page_num),
            "project_id": project_id,
            "process_type": process_type,
            "document_id": doc["document_id"],
            "document_title": doc.get("document_title"),
            "document_type_clean": doc.get("document_type_clean"),
            "document_type_category": doc.get("document_type_category"),
            "main_document": doc.get("main_document"),
            "section_id": None,
            "page_start": page_num,
            "page_end": page_num,
            "page_numbers": json.dumps([page_num]),
            "retrieval_mode": "first_pass",
            "retrieval_reason": "page_keyword_score",
            "source_tier": "page_keyword",
            "retrieval_tier": "tier_d",
            "retrieval_score": item["retrieval_score"],
            "initiation_page_score": item["init_s"],
            "decision_page_score": item["dec_s"],
            "negative_page_score": item["neg_s"],
            "heading_title": None,
            "parent_heading_title": None,
            "context_text": _truncate(text, 2000),
            "context_chars": len(text),
            "estimated_tokens": max(1, len(text) // 4),
            "context_hash": _text_hash(text),
            "api_eligible": item["retrieval_score"] >= 2.0,
            "created_at": run_at,
        })

    return packets


def deduplicate_packets(packets: list[dict]) -> list[dict]:
    """Remove duplicate packets by context_hash, keeping the highest tier."""
    tier_order = {"tier_a": 0, "tier_b": 1, "tier_c": 2, "tier_d": 3, "tier_e": 4}
    seen: dict[str, dict] = {}
    for p in packets:
        h = p["context_hash"]
        if h not in seen:
            seen[h] = p
        else:
            # Keep the packet from the higher-priority tier
            existing_order = tier_order.get(seen[h]["retrieval_tier"], 99)
            new_order = tier_order.get(p["retrieval_tier"], 99)
            if new_order < existing_order:
                seen[h] = p
    return list(seen.values())


def process_project(
    project_row: pd.Series,
    doc_rows: pd.DataFrame,
    pages_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    run_at: str,
) -> list[dict]:
    project_id = project_row["project_id"]
    process_type = project_row["process_type"]
    cap = PACKET_CAPS.get(process_type, 75)

    packets: list[dict] = []

    # Tier A: structured metadata
    packets.extend(build_tier_a_packets(project_row, run_at))

    # Tier B: page slices from high-priority documents
    packets.extend(build_tier_b_packets(doc_rows, pages_df, project_id, process_type, run_at))

    # Tier C: sections (EA/EIS or long CE only)
    if not sections_df.empty:
        packets.extend(build_tier_c_packets(doc_rows, sections_df, project_id, process_type, run_at))

    # Tier D: page keyword scoring
    packets.extend(build_tier_d_packets(doc_rows, pages_df, project_id, process_type, run_at))

    # Deduplicate by content hash
    packets = deduplicate_packets(packets)

    # Apply per-project cap
    if len(packets) > cap:
        # Prioritize by tier then retrieval_score
        tier_order = {"tier_a": 0, "tier_b": 1, "tier_c": 2, "tier_d": 3}
        packets.sort(
            key=lambda p: (tier_order.get(p["retrieval_tier"], 9), -p["retrieval_score"])
        )
        packets = packets[:cap]

    return packets


def retrieve_for_process(
    process_type: str,
    project_ids: list[str] | None,
    index_df: pd.DataFrame,
    run_at: str,
) -> pd.DataFrame:
    src = SOURCE_MAP[process_type]
    pages_path = PROCESSED_DIR / src / "pages.parquet"
    if not pages_path.exists():
        print(f"  WARNING: pages.parquet not found for {process_type} at {pages_path}")
        return pd.DataFrame()

    proc_index = index_df[index_df["process_type"] == process_type]
    if project_ids:
        proc_index = proc_index[proc_index["project_id"].isin(project_ids)]

    projects = proc_index["project_id"].unique()
    print(f"  Processing {len(projects):,} {process_type} projects...")

    con = duckdb.connect()

    # Read pages with DuckDB (never pd.read_parquet on pages files), then pre-group
    # by document_id into a dict so each per-project lookup is O(1) not O(n).
    print(f"  Loading pages for {process_type}...")
    pages_all = con.execute(
        "SELECT document_id, page_number, page_text FROM read_parquet(?)",
        [str(pages_path)],
    ).df()
    pages_by_doc: dict[str, pd.DataFrame] = {
        doc_id: grp.reset_index(drop=True)
        for doc_id, grp in pages_all.groupby("document_id", sort=False)
    }
    del pages_all

    # Load only the columns we need from sections and push the process_type filter
    # into DuckDB so we don't pull all process types into RAM.
    print(f"  Loading sections for {process_type}...")
    sections_by_proj: dict[str, pd.DataFrame] = {}
    if SECTIONS_PATH.exists():
        try:
            section_cols = [
                "project_id", "document_id", "heading_title", "parent_heading_title",
                "document_title", "page_start", "page_end", "section_text",
            ]
            sections_proc = con.execute(
                f"SELECT {', '.join(section_cols)} FROM read_parquet(?) WHERE process_type = ?",
                [str(SECTIONS_PATH), process_type],
            ).df()
            sections_by_proj = {
                pid: grp.reset_index(drop=True)
                for pid, grp in sections_proc.groupby("project_id", sort=False)
            }
            del sections_proc
        except Exception as e:
            print(f"  WARNING: Could not load sections: {e}")

    # Pre-group index rows by project_id so the per-project slice is also O(1).
    proc_index_by_proj: dict[str, pd.DataFrame] = {
        pid: grp for pid, grp in proc_index.groupby("project_id", sort=False)
    }

    # Load project-level fields for Tier A
    proj_tier_a_cols = [
        "project_id", "process_type",
        "noi_tier_a_eligible", "noi_publication_date", "noi_match_status", "noi_match_confidence",
        "blm_decision_tier_a_eligible", "blm_decision_date", "blm_decision_date_type",
        "blm_initiation_tier_a_eligible", "blm_initiation_date",
        "doe_decision_tier_a_eligible", "doe_decision_date", "doe_decision_date_type",
        "doe_initiation_tier_a_eligible", "doe_initiation_date", "doe_doc_number",
    ]
    proj_tier_a_cols = [c for c in proj_tier_a_cols if c in proc_index.columns]
    proj_meta = (
        proc_index[proj_tier_a_cols]
        .drop_duplicates("project_id")
        .set_index("project_id")
    )

    all_packets: list[dict] = []
    for i, project_id in enumerate(projects):
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{len(projects)} done...")

        doc_rows = proc_index_by_proj.get(project_id, pd.DataFrame())
        doc_ids = doc_rows["document_id"].unique()

        doc_dfs = [pages_by_doc[d] for d in doc_ids if d in pages_by_doc]
        pages_df = pd.concat(doc_dfs, ignore_index=True) if doc_dfs else pd.DataFrame()
        sections_df = sections_by_proj.get(project_id, pd.DataFrame())

        project_row = proj_meta.loc[project_id].copy() if project_id in proj_meta.index else pd.Series()
        project_row["project_id"] = project_id

        packets = process_project(project_row, doc_rows, pages_df, sections_df, run_at)
        all_packets.extend(packets)

    if not all_packets:
        return pd.DataFrame()

    return pd.DataFrame(all_packets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve timeline context packets.")
    parser.add_argument(
        "--process", nargs="+", choices=["CE", "EA", "EIS"], default=["CE", "EA", "EIS"]
    )
    parser.add_argument("--sample-ids", help="Path to a file with one project_id per line.")
    parser.add_argument("--append", action="store_true", help="Append to existing output instead of overwriting.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output even if it already exists.")
    parser.add_argument("--run-dir", help="Override output directory (default: auto-derived from --sample-ids or the main timeline/ dir).")
    args = parser.parse_args()

    # Resolve run directory: sample runs are isolated from the main timeline/ directory
    # so they never overwrite the canonical full-corpus outputs.
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.sample_ids:
        run_dir = TIMELINE_DIR / "sample_runs" / Path(args.sample_ids).stem
    else:
        run_dir = TIMELINE_DIR
    output_path = run_dir / "timeline_context_packets.parquet"

    if output_path.exists() and not args.force and not args.sample_ids and not args.append:
        print(f"Output already exists: {output_path}")
        print("Re-run only when 01_build_timeline_index.py output changes (new registers, NEPATEC update).")
        print("Pass --force to overwrite.")
        return

    project_ids = None
    if args.sample_ids:
        with open(args.sample_ids) as f:
            project_ids = [line.strip() for line in f if line.strip()]
        print(f"Filtering to {len(project_ids)} sample project IDs.")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}\nRun 01_build_timeline_index.py first.")

    run_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).isoformat()

    print(f"Loading index: {INDEX_PATH}")
    index_df = pd.read_parquet(INDEX_PATH)
    print(f"  {len(index_df):,} index rows, {index_df['project_id'].nunique():,} projects")

    all_parts: list[pd.DataFrame] = []
    for process_type in args.process:
        print(f"\n=== {process_type} ===")
        part = retrieve_for_process(process_type, project_ids, index_df, run_at)
        if not part.empty:
            all_parts.append(part)
            print(f"  {len(part):,} packets for {process_type}")

    if not all_parts:
        print("No packets generated.")
        return

    result = pd.concat(all_parts, ignore_index=True)

    # Normalize page_start / page_end to str to avoid mixed int/None/str PyArrow error.
    # Sections produce numpy int64, pages produce strings (sometimes ranges like "1-6"), Tier A produces None.
    import numpy as np
    for col in ("page_start", "page_end"):
        result[col] = result[col].apply(
            lambda x: (
                str(int(x)) if isinstance(x, (int, np.integer))
                else str(x) if x is not None and not (isinstance(x, float) and pd.isna(x))
                else None
            )
        )

    print(f"\nTotal packets: {len(result):,}")
    print("Tier distribution:")
    print(result["retrieval_tier"].value_counts().to_string())

    if args.append and output_path.exists():
        existing = pd.read_parquet(output_path)
        result = pd.concat([existing, result], ignore_index=True)
        result = result.drop_duplicates("context_packet_id")
        print(f"After merge with existing: {len(result):,} packets")

    result.to_parquet(output_path, index=False)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
