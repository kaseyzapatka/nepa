"""
Extract date candidates from context packets for D4 timeline extraction.

Applies the full date regex suite from Phase 1/2 lessons, enriches each
candidate with context signals and role pre-labels, and writes the candidate
sidecar table.

Outputs:
    phase2/data/analysis/timeline/timeline_candidates.parquet

Usage:
    python 03_extract_timeline_candidates.py [--process CE EA EIS] [--sample-ids path]
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
PACKETS_PATH = TIMELINE_DIR / "timeline_context_packets.parquet"
INDEX_PATH = TIMELINE_DIR / "timeline_document_index.parquet"
OUTPUT_PATH = TIMELINE_DIR / "timeline_candidates.parquet"

RUN_DATE = datetime.now(timezone.utc).date()

# ---------------------------------------------------------------------------
# Date regexes — full suite from plan §4 / Phase 1 lessons
# ---------------------------------------------------------------------------
MONTHS_FULL = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
MONTHS_SHORT = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"

DATE_PATTERNS = [
    (rf"({MONTHS_FULL})\s+(\d{{1,2}}),?\s+(\d{{4}})", "MDY_full"),
    (rf"({MONTHS_SHORT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})", "MDY_short"),
    (rf"(\d{{1,2}})\s+({MONTHS_FULL})\s+(\d{{4}})", "DMY_full"),
    (r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", "numeric_slash"),
    (r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2})\b", "numeric_slash_2y"),
    (r"(\d{4})-(\d{1,2})-(\d{1,2})", "ISO"),
    (r"(\d{1,2})-(\d{1,2})-(\d{4})", "numeric_dash"),
    (r"(\d{4})\.(\d{2})\.(\d{2})", "digital_sig"),
    (r"(?<![\d.])(0?[1-9]|1[0-2])\.(\d{2})\.(\d{2,4})(?![\d.])", "numeric_dot"),
    (rf"({MONTHS_FULL})\s+(\d{{4}})", "MY_full"),
    (rf"({MONTHS_SHORT})\.?\s+(\d{{4}})", "MY_short"),
]

COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), ptype) for pattern, ptype in DATE_PATTERNS
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ---------------------------------------------------------------------------
# Exclusion patterns (legal/bibliographic/map)
# ---------------------------------------------------------------------------
EXCLUSION_KEYWORDS = [
    "act of 19", "act of 20", "act (19", "act (20",
    "policy act", "preservation act", "conservation act",
    "management act", "protection act", "improvement act", "reform act",
    "recovery act", "species act", "water act", "air act", "lands act",
    "statute", "u.s.c.", " usc ", "public law", "p.l.", "amended in",
    "accessed on", "retrieved on", "available at",
    "et al.", "et al,", "eds.", "vol.", "pp.", "journal", "doi:",
    "isbn", "issn", "proceedings", "report no.",
]

# ---------------------------------------------------------------------------
# Role cue patterns — positive
# ---------------------------------------------------------------------------

# Clear initiation
CLEAR_INITIATION_STRONG = re.compile(
    r"\b("
    r"application\s+received|application\s+submitted|submitted\s+(?:a|an|the)?\s*(?:right|application|permit|plan|request)|"
    r"blm\s+received|agency\s+received|(?:noi|notice\s+of\s+intent)\s+(?:published|issued|submitted)|"
    r"scoping\s+period\s+(?:began|started|initiated|opened)|"
    r"notice\s+of\s+intent\s+was\s+published|"
    r"(?:federal\s+register).*notice\s+of\s+intent|"
    r"environmental\s+review\s+(?:began|initiated|started)"
    r")\b",
    re.IGNORECASE,
)

CLEAR_INITIATION_MED = re.compile(
    r"\b("
    r"(?:application|request|permit|apd|plan\s+of\s+development|pod|right[-\s]of[-\s]way|"
    r"license\s+application|row)\s+(?:date|filed|submitted|received)|"
    r"(?:date\s+(?:of\s+)?)?(?:application|request|submission|filing|receipt)|"
    r"doe\s+initiator|nepa\s+initiator|action\s+initiating|nepa\s+initiation"
    r")\b",
    re.IGNORECASE,
)

CE_INITIATOR_ROLE = re.compile(
    r"\b(doe\s+initiator|nepa\s+initiator|action\s+initiating\s+office|"
    r"project\s+(?:initiator|proponent|sponsor))\b",
    re.IGNORECASE,
)

# Clear decision
CLEAR_DECISION_STRONG = re.compile(
    r"\b("
    r"fonsi\s+(?:was\s+)?(?:signed|issued|approved|dated)|"
    r"finding\s+of\s+no\s+significant\s+impact\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"record\s+of\s+decision[,\s]+(?:was\s+)?(?:signed|issued|dated)|"
    r"rod\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"(?:signed|issued)\s+(?:the\s+)?(?:rod|record\s+of\s+decision|fonsi|finding\s+of\s+no)|"
    r"decision\s+(?:record|notice)\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"categorical\s+exclusion\s+(?:determination|approved|signed)|"
    r"(?:ce|cx)\s+(?:determination|approved|signed)|"
    r"(?:date\s+)?signed\s+(?:by|on).*(?:field\s+manager|district\s+manager|authorizing\s+official)|"
    r"nepa\s+compliance\s+officer.*date|"
    r"date\s+of\s+(?:decision|approval|determination)"
    r")\b",
    re.IGNORECASE,
)

# Proxy decision: FEIS/EA publication dates serve as upper bound for ROD/FONSI
PROXY_DECISION_RE = re.compile(
    r"\b("
    r"(?:final\s+(?:environmental\s+impact\s+statement|eis)|feis)\s+(?:was\s+)?(?:published|released|issued|filed|available|signed)|"
    r"notice\s+of\s+availability\s+(?:for|of)\s+(?:the\s+)?(?:feis|final\s+eis|final\s+environmental)|"
    r"(?:published|issued|released)\s+(?:an?\s+)?(?:noa|notice\s+of\s+availability)\s+(?:for|of)\s+(?:the\s+)?(?:feis|final)|"
    r"(?:published|issued|released)\s+(?:the\s+)?(?:final\s+eis|feis|final\s+ea)|"
    r"(?:final\s+)?(?:ea|environmental\s+assessment)\s+(?:was\s+)?(?:signed|issued|approved|completed|finalized)|"
    r"(?:approved|signed)\s+(?:the\s+)?(?:final\s+)?(?:ea|environmental\s+assessment)"
    r")\b",
    re.IGNORECASE,
)

CLEAR_DECISION_MED = re.compile(
    r"\b("
    r"(?:approved|signed|authorized|determined)\s+(?:by|on|this)|"
    r"(?:field\s+manager|district\s+manager|authorizing\s+official|"
    r"nepa\s+compliance\s+officer|certifying\s+official)\s+(?:signature|date|signed)|"
    r"decision\s+date|date\s+approved|date\s+signed"
    r")\b",
    re.IGNORECASE,
)

# Review / specialist (not decision)
REVIEW_CUES = re.compile(
    r"\b("
    r"environmental\s+specialist|wildlife\s+biologist|archaeologist|"
    r"cultural\s+resource\s+specialist|shpo|section\s+106|"
    r"review\s+completed|interim\s+review|phase\s+approval|"
    r"concurrence\s+(?:received|date)|coordination\s+date"
    r")\b",
    re.IGNORECASE,
)

# Historical / legal
HISTORICAL_CUES = re.compile(
    r"\b("
    r"(?:resource\s+management|land\s+use)\s+plan|rmp|lup|programmatic\s+eis|"
    r"prior\s+rod|previous\s+(?:eis|ea)|old\s+(?:lease|plan)|"
    r"communication\s+site\s+established|lease\s+issued|historical"
    r")\b",
    re.IGNORECASE,
)

# Hard reject
REJECT_CUES = re.compile(
    r"\b("
    r"omb\s+(?:control|approval)|form\s+approved|prepared\s+by|"
    r"downloaded|accessed\s+on|retrieved\s+on|revision\s+date|"
    r"map\s+(?:date|created|printed)|figure\s+\d+|table\s+\d+"
    r")\b",
    re.IGNORECASE,
)

# Signature/bottom-of-document signal
SIGNATURE_BLOCK_RE = re.compile(
    r"\b("
    r"signature|nepa\s+compliance\s+officer|authorizing\s+official|field\s+manager|"
    r"district\s+manager|certifying\s+official|approved\s+by|concurrence"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_match(match: re.Match, ptype: str) -> tuple[datetime | None, str]:
    """Return (datetime_or_None, granularity)."""
    try:
        g = match.groups()
        if ptype in ("MDY_full", "MDY_short"):
            month = MONTH_MAP.get(g[0].lower())
            return datetime(int(g[2]), month, int(g[1])), "day"
        if ptype == "DMY_full":
            month = MONTH_MAP.get(g[1].lower())
            return datetime(int(g[2]), month, int(g[0])), "day"
        if ptype == "numeric_slash":
            return datetime(int(g[2]), int(g[0]), int(g[1])), "day"
        if ptype == "numeric_slash_2y":
            yr = int(g[2])
            yr = 2000 + yr if yr <= 30 else 1900 + yr
            return datetime(yr, int(g[0]), int(g[1])), "day"
        if ptype == "ISO":
            return datetime(int(g[0]), int(g[1]), int(g[2])), "day"
        if ptype == "numeric_dash":
            return datetime(int(g[2]), int(g[0]), int(g[1])), "day"
        if ptype == "digital_sig":
            return datetime(int(g[0]), int(g[1]), int(g[2])), "day"
        if ptype == "numeric_dot":
            yr = int(g[2])
            if yr < 100:
                yr = 2000 + yr if yr <= 30 else 1900 + yr
            if not (1970 <= yr <= 2035):
                return None, "unknown"
            return datetime(yr, int(g[0]), int(g[1])), "day"
        if ptype in ("MY_full", "MY_short"):
            month = MONTH_MAP.get(g[0].lower())
            return datetime(int(g[1]), month, 1), "month"
    except (ValueError, TypeError, KeyError):
        return None, "unknown"
    return None, "unknown"


def _should_reject_date(
    parsed_date: datetime,
    context: str,
    process_type: str,
    source_tier: str,
) -> tuple[bool, str]:
    """
    Return (reject, reason) applying the plan §4 exclusion rules.
    Does not reject month-year candidates; those are granularity=month.
    """
    ctx_lower = context.lower()

    # Future date check
    if parsed_date.date() > RUN_DATE:
        return True, "future_date"

    # Process-specific year cutoffs (plan §4)
    if process_type in ("CE", "EA") and parsed_date.year < 1970:
        return True, "pre_1970_hard_reject"
    if process_type == "EIS" and parsed_date.year < 1970:
        # Soft reject: allow only with strong evidence (handled in scoring/selection)
        return True, "pre_1970_eis_reject"

    # Legal/statutory citation exclusions
    for kw in EXCLUSION_KEYWORDS:
        if kw in ctx_lower:
            return True, f"exclusion_keyword:{kw}"

    # Metadata-only sources bypass text-based exclusions
    if source_tier == "metadata":
        return False, ""

    # Reject/historical cues
    if REJECT_CUES.search(context):
        return True, "reject_cue"

    return False, ""


def _prelabel_role(
    context: str,
    source_tier: str,
    retrieval_reason: str | None,
    ptype: str,
    document_type_category: str | None = None,
) -> tuple[str, float, list[str], list[str]]:
    """
    Return (candidate_role, role_confidence_float, positive_cue_flags, negative_cue_flags).
    Role confidence 0-5 maps to 5=high, 3=medium, 1=low.
    """
    pos_cues: list[str] = []
    neg_cues: list[str] = []

    # Agency register Tier A — authoritative government source, highest confidence
    if source_tier == "metadata" and "blm_register_decision" in (retrieval_reason or ""):
        return "clear_decision", 5.0, ["blm_register_tier_a"], []
    if source_tier == "metadata" and "blm_register_initiation" in (retrieval_reason or ""):
        return "clear_initiation", 5.0, ["blm_register_tier_a"], []
    if source_tier == "metadata" and "doe_register_decision" in (retrieval_reason or ""):
        return "clear_decision", 5.0, ["doe_register_tier_a"], []
    if source_tier == "metadata" and "doe_register_initiation" in (retrieval_reason or ""):
        return "clear_initiation", 5.0, ["doe_register_tier_a"], []

    # Metadata / FR NOI
    if source_tier == "metadata" and "noi" in (retrieval_reason or ""):
        return "clear_initiation", 5.0, ["fr_noi_metadata"], []

    if source_tier in ("file_name", "title"):
        if document_type_category in ("final", "decision"):
            return "proxy_decision", 1.5, ["filename_or_title"], []
        return "proxy_initiation", 1.5, ["filename_or_title"], []

    # Check strong cues first
    if CLEAR_DECISION_STRONG.search(context):
        pos_cues.append("decision_strong")
        return "clear_decision", 5.0, pos_cues, neg_cues

    if CLEAR_INITIATION_STRONG.search(context):
        pos_cues.append("initiation_strong")
        return "clear_initiation", 5.0, pos_cues, neg_cues

    if HISTORICAL_CUES.search(context):
        neg_cues.append("historical_cue")
        return "historical", 0.0, pos_cues, neg_cues

    if REJECT_CUES.search(context):
        neg_cues.append("reject_cue")
        return "reject", 0.0, pos_cues, neg_cues

    if CLEAR_DECISION_MED.search(context):
        pos_cues.append("decision_med")
        return "clear_decision", 3.0, pos_cues, neg_cues

    if CLEAR_INITIATION_MED.search(context):
        pos_cues.append("initiation_med")
        return "clear_initiation", 3.0, pos_cues, neg_cues

    if REVIEW_CUES.search(context):
        pos_cues.append("review_cue")
        return "review", 2.0, pos_cues, neg_cues

    # Proxy decision from FEIS/EA publication language
    if PROXY_DECISION_RE.search(context):
        pos_cues.append("feis_pub_proxy")
        return "proxy_decision", 2.5, pos_cues, neg_cues

    # Month-year dates:
    #   decision doc, short context (≤18 words) → cover/signature date → clear_decision
    #   decision doc, long context → body-text reference → proxy_initiation (may be app date)
    #   final doc, short context → FEIS pub date → proxy_decision (FEIS ≠ ROD)
    #   everything else → proxy_initiation
    if ptype in ("MY_full", "MY_short"):
        if document_type_category == "decision" and len(context.split()) <= 18:
            pos_cues.append("doc_type_decision")
            return "clear_decision", 2.0, pos_cues, neg_cues
        if document_type_category == "final" and len(context.split()) <= 18:
            return "proxy_decision", 1.5, pos_cues, neg_cues
        return "proxy_initiation", 1.0, pos_cues, neg_cues

    # numeric_dot: require signature/form context for CE, else unknown
    if ptype == "numeric_dot":
        if SIGNATURE_BLOCK_RE.search(context):
            return "clear_decision", 3.0, ["signature_block"], neg_cues
        return "unknown", 1.0, pos_cues, neg_cues

    # Dates in decision-labeled documents without explicit text cues. The document
    # itself is the decision artifact, so treat these as clear_decision at low confidence
    # rather than proxy — they are not references to an external decision.
    if document_type_category == "decision":
        pos_cues.append("doc_type_decision")
        return "clear_decision", 2.0, pos_cues, neg_cues

    return "unknown", 1.5, pos_cues, neg_cues


def extract_candidates_from_packet(packet: dict) -> list[dict]:
    """
    Extract date candidates from a single context packet row.
    Returns a list of candidate dicts.
    """
    context = packet.get("context_text") or ""
    if not context:
        return []

    context_clean = " ".join(context.split())
    source_tier = packet.get("source_tier", "page_slice")
    process_type = packet.get("process_type", "")
    retrieval_reason = packet.get("retrieval_reason", "")
    document_type_clean = packet.get("document_type_clean")
    heading = packet.get("heading_title") or ""

    # Metadata tier: the whole context text IS the date value
    if source_tier == "metadata":
        # Try to extract date from the metadata string
        for compiled, ptype in COMPILED_PATTERNS:
            for m in compiled.finditer(context_clean):
                parsed, granularity = _parse_match(m, ptype)
                if parsed is None:
                    continue
                reject, reason = _should_reject_date(
                    parsed, context_clean, process_type, source_tier
                )
                if reject:
                    continue
                role, conf, pos_cues, neg_cues = _prelabel_role(
                    context_clean, source_tier, retrieval_reason, ptype,
                    document_type_category=packet.get("document_type_category"),
                )
                candidate_id = hashlib.sha1(
                    f"{packet['project_id']}|{packet.get('document_id')}|{packet.get('page_start')}|{parsed.date()}|{context_clean[:50]}".encode()
                ).hexdigest()[:20]
                return [{
                    "candidate_id": candidate_id,
                    "project_id": packet["project_id"],
                    "process_type": process_type,
                    "document_id": packet.get("document_id"),
                    "page_number": packet.get("page_start"),
                    "section_id": packet.get("section_id"),
                    "context_packet_id": packet.get("context_packet_id"),
                    "source_tier": source_tier,
                    "retrieval_tier": packet.get("retrieval_tier"),
                    "candidate_source_type": "noi_notice" if "noi" in retrieval_reason else "metadata",
                    "raw_date_text": m.group(0),
                    "parsed_date": parsed.date().isoformat(),
                    "date_granularity": granularity,
                    "context_text": context_clean,
                    "context_cleaned": context_clean,
                    "document_title": packet.get("document_title"),
                    "file_name": None,
                    "document_type_clean": document_type_clean,
                    "document_type_category": packet.get("document_type_category"),
                    "main_document": packet.get("main_document"),
                    "heading_title": heading,
                    "parent_heading_title": packet.get("parent_heading_title"),
                    "position_pct": None,
                    "section_position_pct": None,
                    "candidate_role": role,
                    "role_confidence": ("high" if conf >= 4 else "medium" if conf >= 2.5 else "low"),
                    "role_confidence_score": conf,
                    "rule_ids": "metadata_tier_a",
                    "positive_cue_flags": "|".join(pos_cues),
                    "negative_cue_flags": "|".join(neg_cues),
                    "classifier_label": None,
                    "classifier_score": None,
                    "api_label": None,
                    "api_call_id": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }]
        return []

    # Text-based tiers: extract from sentence contexts
    candidates: list[dict] = []
    seen_keys: set[str] = set()

    # Split context into sentences
    sentences = re.split(r"(?<=[.!?])\s+|\n", context_clean)
    # Also scan the whole context to catch multi-sentence date cues
    scan_blocks = sentences + [context_clean]

    for block in scan_blocks:
        block = block.strip()
        if not block:
            continue
        for compiled, ptype in COMPILED_PATTERNS:
            for m in compiled.finditer(block):
                parsed, granularity = _parse_match(m, ptype)
                if parsed is None:
                    continue

                reject, _ = _should_reject_date(
                    parsed, block, process_type, source_tier
                )
                if reject:
                    continue

                date_str = parsed.date().isoformat()
                block_norm = " ".join(block.split())[:100]
                dedup_key = f"{packet['project_id']}|{packet.get('document_id')}|{packet.get('page_start')}|{date_str}|{block_norm}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                role, conf, pos_cues, neg_cues = _prelabel_role(
                    block, source_tier, retrieval_reason, ptype,
                    document_type_category=packet.get("document_type_category"),
                )

                # Skip clear rejects
                if role == "reject" and not heading:
                    continue

                # Estimate position in document
                pos_pct: float | None = None
                if packet.get("page_start") is not None and packet.get("page_end") is not None:
                    try:
                        page_n = int(packet["page_start"])
                        # Use packet retrieval_score as proxy for importance
                        pos_pct = min(1.0, page_n / max(1.0, page_n + 10))
                    except (ValueError, TypeError):
                        pass

                candidate_id = hashlib.sha1(
                    f"{packet['project_id']}|{packet.get('document_id')}|{packet.get('page_start')}|{date_str}|{block_norm}".encode()
                ).hexdigest()[:20]

                candidates.append({
                    "candidate_id": candidate_id,
                    "project_id": packet["project_id"],
                    "process_type": process_type,
                    "document_id": packet.get("document_id"),
                    "page_number": packet.get("page_start"),
                    "section_id": packet.get("section_id"),
                    "context_packet_id": packet.get("context_packet_id"),
                    "source_tier": source_tier,
                    "retrieval_tier": packet.get("retrieval_tier"),
                    "candidate_source_type": "document_text",
                    "raw_date_text": m.group(0),
                    "parsed_date": date_str,
                    "date_granularity": granularity,
                    "context_text": block,
                    "context_cleaned": block_norm + ("..." if len(block) > 100 else ""),
                    "document_title": packet.get("document_title"),
                    "file_name": None,
                    "document_type_clean": document_type_clean,
                    "document_type_category": packet.get("document_type_category"),
                    "main_document": packet.get("main_document"),
                    "heading_title": heading,
                    "parent_heading_title": packet.get("parent_heading_title"),
                    "position_pct": pos_pct,
                    "section_position_pct": None,
                    "candidate_role": role,
                    "role_confidence": ("high" if conf >= 4 else "medium" if conf >= 2.5 else "low"),
                    "role_confidence_score": conf,
                    "rule_ids": ptype,
                    "positive_cue_flags": "|".join(pos_cues),
                    "negative_cue_flags": "|".join(neg_cues),
                    "classifier_label": None,
                    "classifier_score": None,
                    "api_label": None,
                    "api_call_id": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

    return candidates


def add_repeated_mention_counts(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Count how many times the same parsed_date appears per project."""
    if candidates_df.empty:
        return candidates_df
    counts = (
        candidates_df.groupby(["project_id", "parsed_date"])
        .size()
        .reset_index(name="date_mention_count")
    )
    return candidates_df.merge(counts, on=["project_id", "parsed_date"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract date candidates from context packets.")
    parser.add_argument(
        "--process", nargs="+", choices=["CE", "EA", "EIS"], default=["CE", "EA", "EIS"]
    )
    parser.add_argument("--sample-ids", help="Path to a file with one project_id per line.")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output even if it already exists.")
    parser.add_argument("--run-dir", help="Override run directory (reads packets from here, writes candidates here).")
    args = parser.parse_args()

    # Resolve run directory — matches the logic in script 02.
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.sample_ids:
        run_dir = TIMELINE_DIR / "sample_runs" / Path(args.sample_ids).stem
    else:
        run_dir = TIMELINE_DIR
    packets_path = run_dir / "timeline_context_packets.parquet"
    output_path = run_dir / "timeline_candidates.parquet"

    if output_path.exists() and not args.force and not args.sample_ids and not args.append:
        print(f"Output already exists: {output_path}")
        print("Re-run only when 02_retrieve_timeline_contexts.py output changes.")
        print("Pass --force to overwrite.")
        return

    if not packets_path.exists():
        raise FileNotFoundError(f"Context packets not found: {packets_path}\nRun 02_retrieve_timeline_contexts.py first.")

    project_ids: set[str] | None = None
    if args.sample_ids:
        with open(args.sample_ids) as f:
            project_ids = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(project_ids)} sample project IDs.")

    print(f"Loading context packets: {packets_path}")
    packets_df = pd.read_parquet(packets_path)
    packets_df = packets_df[packets_df["process_type"].isin(args.process)]
    if project_ids:
        packets_df = packets_df[packets_df["project_id"].isin(project_ids)]
    print(f"  {len(packets_df):,} packets to process")

    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict] = []
    packets_records = packets_df.to_dict("records")
    for i, row in enumerate(packets_records):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i:,}/{len(packets_records):,} packets, {len(all_candidates):,} candidates so far...")
        cands = extract_candidates_from_packet(row)
        all_candidates.extend(cands)

    if not all_candidates:
        print("No candidates extracted.")
        return

    df = pd.DataFrame(all_candidates)
    df = add_repeated_mention_counts(df)

    # Initialize scoring columns to zero — populated by 04_select_timeline_dates.py
    for col in [
        "source_strength", "role_cue_strength", "document_priority",
        "section_priority", "page_priority", "position_signal",
        "classifier_signal", "chronology_signal", "repeated_mention_signal",
        "negative_penalty", "ranking_score",
    ]:
        df[col] = 0.0

    df["selected_for_initiation"] = False
    df["selected_for_decision"] = False
    df["is_proxy"] = df["candidate_role"].isin(["proxy_initiation", "proxy_decision"])

    # Deduplicate by candidate_id
    df = df.drop_duplicates("candidate_id")

    print(f"\nTotal candidates: {len(df):,}")
    print("Role distribution:")
    print(df["candidate_role"].value_counts().to_string())
    print("Granularity distribution:")
    print(df["date_granularity"].value_counts().to_string())

    run_dir.mkdir(parents=True, exist_ok=True)

    if args.append and output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates("candidate_id")
        print(f"After merge with existing: {len(df):,}")

    df.to_parquet(output_path, index=False)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
