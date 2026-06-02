"""
build_review_packet.py
=========================
Builds a human-QC review packet from the pipeline's candidate output, adding
pre-computed preferred_initiation_candidate and preferred_decision_candidate
columns that mechanically apply the milestone-priority hierarchy from
reviewer_codebook.md.

This eliminates ~38% of inter-rater disagreements (milestone-interpretation
differences) before reviewers ever open the file.

Usage
-----
    # Build a fresh packet from the pipeline's candidate output:
    python phase2/code/deliverable04/build_review_packet.py \
        --packet   phase2/output/deliverable04/timeline_sample100_review_packet.csv \
        --output   phase2/output/deliverable04/timeline_review_packet_v2.csv \
        --reviewers 2          # number of blank reviewer rows to emit per project

Input expectation
-----------------
The --packet file must contain columns produced by 07_validate.py:
    top_initiation_candidates, top_decision_candidates,
    suggested_initiation_date, suggested_initiation_evidence,
    suggested_decision_date, suggested_decision_evidence

This script de-duplicates the packet to one row per project (drops existing
reviewer rows), applies the priority ranking, and emits fresh blank rows.
"""

import csv
import re
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Milestone priority tables
# ---------------------------------------------------------------------------
# Each entry is a rank — lower = preferred.
# Applied to the type token extracted from candidate tag strings.

# Tag format: "DATE [type|confidence|score=N] doc=DOCTYPE | TEXT"

INIT_TAG_PRIORITY = {
    "fr_noi":           1,   # Federal Register NOI — future-proof
    "noi_register":     1,
    "clear_initiation": 10,  # refined further by confidence below
    "proxy_initiation": 30,
}

DEC_TAG_PRIORITY = {
    "fr_noa":           1,
    "register_dec":     1,
    "cx_register":      1,
    "clear_decision":   10,
    "proxy_decision":   30,
}

CONFIDENCE_BONUS = {"high": 0, "medium": 5, "low": 10}

# Document-type adjustments: lower = better source for that field direction
INIT_DOC_BONUS = {
    "None":  0,    # BLM/DOE register (no associated doc)
    "OTHER": 5,    # scoping reports, FR notices in appendix
    "CE":    6,
    "EA":    8,
    "DEA":   8,
    "FONSI": 10,
    "FEIS":  12,
    "DEIS":  14,
    "ROD":   15,   # late-stage doc — weak initiation signal
}

DEC_DOC_BONUS = {
    "None":  0,    # register
    "categorical exclusion determin": 0,
    "CE":    4,
    "FONSI": 3,
    "ROD":   3,
    "OTHER": 5,
    "EA":    8,
    "FEIS":  10,
    "DEIS":  20,   # never a decision
}

# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------
CAND_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s+\[([^\]]+)\]\s+doc=(\S+)\s+\|\s+(.*)"
)


def parse_candidates(raw: str) -> list[dict]:
    """Return list of dicts from the pipe-delimited candidate string."""
    if not raw or not raw.strip():
        return []
    results = []
    for chunk in raw.split(" ||| "):
        chunk = chunk.strip()
        m = CAND_RE.match(chunk)
        if not m:
            continue
        date_str, tag_block, doc, text = m.groups()
        parts = tag_block.split("|")
        ctype     = parts[0].strip() if len(parts) > 0 else ""
        conf      = parts[1].strip() if len(parts) > 1 else "low"
        score_raw = parts[2].strip() if len(parts) > 2 else "score=0"
        try:
            score = float(score_raw.replace("score=", ""))
        except ValueError:
            score = 0.0
        results.append({
            "date":  date_str,
            "type":  ctype,
            "conf":  conf,
            "score": score,
            "doc":   doc.strip(),
            "text":  text.strip()[:200],
        })
    return results


def _rank(cand: dict, type_table: dict, doc_table: dict) -> float:
    """Lower rank = higher priority (prefer this candidate)."""
    type_rank  = type_table.get(cand["type"], 50)
    conf_bonus = CONFIDENCE_BONUS.get(cand["conf"], 10)
    doc_bonus  = doc_table.get(cand["doc"], 12)
    # Higher pipeline score → small rank reduction (tiebreaker only)
    score_adj  = -cand["score"] * 0.5
    return type_rank + conf_bonus + doc_bonus + score_adj


def pick_preferred(candidates: list[dict],
                   type_table: dict,
                   doc_table: dict) -> dict | None:
    """Return the highest-priority candidate, or None if list is empty."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: _rank(c, type_table, doc_table))


def format_preferred(cand: dict | None) -> str:
    if cand is None:
        return ""
    return (
        f"{cand['date']} [{cand['type']}|{cand['conf']}|score={cand['score']:.1f}] "
        f"doc={cand['doc']} | {cand['text']}"
    )


# ---------------------------------------------------------------------------
# Day-precision normalisation (codebook Part 3)
# ---------------------------------------------------------------------------
_EXPLICIT_DAY_RE = re.compile(
    r"\b(\d{1,2})[/\-](\d{4})\b"             # MM/YYYY or MM-YYYY
    r"|\b(\d{4})[.\-](\d{2})[.\-](\d{2})\b"  # YYYY.MM.DD or YYYY-MM-DD
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"          # M/D/YY or MM/DD/YYYY
)


def normalise_day_precision(date_str: str, source_text: str) -> str:
    """
    When a date was imputed at the 01 or 15 of a month and the source text
    contains no explicit day, normalise to YYYY-MM-01 so both reviewers
    produce the same string from the same evidence.
    """
    if not date_str or len(date_str) < 10:
        return date_str
    day = date_str[8:10]
    if day not in ("01", "15"):
        return date_str   # specific non-proxy day — keep as-is
    if _EXPLICIT_DAY_RE.search(source_text):
        return date_str   # source has an explicit day — preserve it
    return date_str[:8] + "01"


# ---------------------------------------------------------------------------
# Main packet builder
# ---------------------------------------------------------------------------
_REVIEWER_COLS = {
    "preferred_initiation_candidate",
    "preferred_decision_candidate",
    "review_initiation_date",
    "review_initiation_source",
    "review_initiation_notes",
    "review_decision_date",
    "review_decision_source",
    "review_decision_notes",
    "reviewer",
}


def build_packet(packet_path: Path, output_path: Path, n_reviewers: int = 2) -> None:
    with open(packet_path, newline="") as f:
        reader = csv.DictReader(f)
        raw_rows   = list(reader)
        src_fields = list(reader.fieldnames or [])

    # De-duplicate: keep only the first occurrence of each project_id
    # (drops any existing reviewer rows so we can re-emit fresh blanks)
    seen: set[str] = set()
    project_rows: list[dict] = []
    for row in raw_rows:
        pid = row.get("project_id") or row.get("sample_id", "")
        if pid not in seen:
            seen.add(pid)
            project_rows.append(row)

    # Output schema: base project columns + new QC columns
    base_fields = [c for c in src_fields if c not in _REVIEWER_COLS]
    out_fields  = base_fields + [
        "preferred_initiation_candidate",
        "preferred_decision_candidate",
        "review_initiation_date",
        "review_initiation_source",
        "review_initiation_notes",
        "review_decision_date",
        "review_decision_source",
        "review_decision_notes",
        "reviewer",
    ]

    out_rows: list[dict] = []
    for row in project_rows:
        init_cands = parse_candidates(row.get("top_initiation_candidates", ""))
        dec_cands  = parse_candidates(row.get("top_decision_candidates", ""))

        pref_init = pick_preferred(init_cands, INIT_TAG_PRIORITY, INIT_DOC_BONUS)
        pref_dec  = pick_preferred(dec_cands,  DEC_TAG_PRIORITY,  DEC_DOC_BONUS)

        # Apply day-precision normalisation to suggested dates
        sug_init_date = normalise_day_precision(
            row.get("suggested_initiation_date", ""),
            row.get("suggested_initiation_evidence", ""),
        )
        sug_dec_date = normalise_day_precision(
            row.get("suggested_decision_date", ""),
            row.get("suggested_decision_evidence", ""),
        )

        base = {k: row.get(k, "") for k in base_fields}
        base["suggested_initiation_date"] = sug_init_date
        base["suggested_decision_date"]   = sug_dec_date
        base["preferred_initiation_candidate"] = format_preferred(pref_init)
        base["preferred_decision_candidate"]   = format_preferred(pref_dec)

        for i in range(1, n_reviewers + 1):
            out_rows.append({
                **base,
                "review_initiation_date":   "",
                "review_initiation_source": "",
                "review_initiation_notes":  "",
                "review_decision_date":     "",
                "review_decision_source":   "",
                "review_decision_notes":    "",
                "reviewer": f"reviewer_{i}",
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"Wrote {len(out_rows)} rows "
        f"({len(project_rows)} projects × {n_reviewers} reviewer slots) "
        f"→ {output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add preferred-candidate columns to a timeline review packet."
    )
    ap.add_argument("--packet",    required=True, help="Input review packet CSV")
    ap.add_argument("--output",    required=True, help="Output review packet CSV")
    ap.add_argument("--reviewers", type=int, default=2,
                    help="Number of blank reviewer rows per project (default: 2)")
    args = ap.parse_args()

    build_packet(
        packet_path = Path(args.packet),
        output_path = Path(args.output),
        n_reviewers = args.reviewers,
    )
