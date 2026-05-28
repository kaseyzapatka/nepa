"""
Select project-level timeline dates from scored candidates (D4).

Implements the plan §5 two-pass selection:
  Pass 1 — score and select best clear decision candidate.
  Pass 2 — re-score initiation candidates using selected decision as chronology anchor.

Also applies historical gap rules (plan §5), manual corrections (plan §9),
and generates the manual review queue.

Modes:
  default           — run selection, write timeline_project_dates.parquet
  --import-corrections <csv>  — import a filled review queue CSV into
                                timeline_manual_corrections.parquet

Outputs:
    phase2/data/analysis/timeline/timeline_project_dates.parquet  (updated)
    phase2/data/analysis/timeline/timeline_candidates.parquet     (scoring cols updated)
    phase2/output/deliverable04/timeline_manual_review_queue.csv

Usage:
    python 04_select_timeline_dates.py [--process CE EA EIS] [--sample-ids path]
    python 04_select_timeline_dates.py --import-corrections filled_queue.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import csv
import hashlib
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"

CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
INDEX_PATH = TIMELINE_DIR / "timeline_document_index.parquet"
CORRECTIONS_PATH = TIMELINE_DIR / "timeline_manual_corrections.parquet"
DATES_PATH = TIMELINE_DIR / "timeline_project_dates.parquet"
REVIEW_QUEUE_PATH = OUTPUT_DIR / "timeline_manual_review_queue.csv"

GAP_DAYS = 730
EIS_GAP_EXEMPT = True
SAME_DAY_DURATION_FLAG = "same_day"
MAX_DURATION_YEARS = 25

# ---------------------------------------------------------------------------
# Scoring weights / component ranges (plan §5 table)
# ---------------------------------------------------------------------------

SOURCE_STRENGTH = {
    "tier_a": 5,
    "tier_b": 3,
    "tier_c": 3,
    "tier_d": 2,
    "tier_e": 1,
    "metadata": 5,
    "file_name": 2,
    "title": 2,
    "page_slice": 3,
    "section": 3,
    "page_keyword": 2,
    "recovery": 1,
}

ROLE_CUE_STRENGTH = {
    "high": 5,
    "medium": 3,
    "low": 1,
    "missing": 0,
}

DOCUMENT_TYPE_SCORES: dict[str, float] = {
    # Decision documents
    "rod": 5.0, "record of decision": 5.0, "joint record of decision": 5.0,
    "fonsi": 5.0, "finding of no significant impact": 5.0,
    "decision record": 5.0, "decision notice": 5.0, "decision memo": 5.0,
    "categorical exclusion determination": 5.0, "ce determination": 5.0,
    "approval memo": 4.5, "signed decision": 4.5,
    "final ea": 2.0, "final environmental assessment": 2.0,
    "final eis": 2.0, "final environmental impact statement": 2.0,
    # Initiation documents
    "notice of intent": 5.0, "noi": 5.0,
    "scoping notice": 4.5, "application": 3.5,
    "apd": 3.5, "plan of development": 3.0,
    "right-of-way application": 3.5, "license application": 3.5,
    # Appendices / attachments
    "appendix": -2.5, "attachment": -2.5, "exhibit": -2.0,
    "technical report": -2.0, "resource report": -1.5,
    "comment response": -2.5, "reference": -2.5, "bibliography": -2.5,
}


def _doc_type_score(dtype_clean: str | None) -> float:
    if not dtype_clean:
        return 0.0
    t = str(dtype_clean).strip().lower()
    for key, score in DOCUMENT_TYPE_SCORES.items():
        if key in t:
            return score
    return 0.0


def _compute_candidate_score(
    row: dict,
    role: str,  # "decision" or "initiation"
    selected_decision_date: date | None,
    index_map: dict,  # project_id -> {decision_doc_score, initiation_doc_score, ...}
) -> float:
    """
    Compute the composite ranking_score for a candidate.
    """
    source_tier = row.get("retrieval_tier") or row.get("source_tier") or "page_keyword"
    source_strength = SOURCE_STRENGTH.get(source_tier, 1)

    role_conf = row.get("role_confidence", "low")
    role_cue_strength = ROLE_CUE_STRENGTH.get(role_conf, 1)

    doc_priority = _doc_type_score(row.get("document_type_clean"))

    # Section priority
    heading = str(row.get("heading_title") or "").lower()
    section_priority = 0.0
    if any(kw in heading for kw in ["decision", "record of decision", "fonsi", "approval"]):
        section_priority = 3.0
    elif any(kw in heading for kw in ["introduction", "background", "purpose and need", "scoping"]):
        section_priority = 2.0
    elif any(kw in heading for kw in ["references", "bibliography", "appendix", "preparers"]):
        section_priority = -2.0

    # Page priority (use retrieval score from context packet as proxy)
    page_priority = min(3.0, max(0.0, float(row.get("retrieval_score", 0) or 0) / 3.0))

    # Position signal: if page_number is a number and document has pages, estimate bottom-of-doc
    position_signal = 0.0
    if row.get("position_pct") is not None:
        pos = float(row["position_pct"])
        if pos > 0.85:
            position_signal = 1.5  # bottom of document boost for CE decisions
        elif pos < 0.10:
            position_signal = 0.5  # header pages modest boost

    # Classifier signal (not yet available — zero)
    classifier_signal = 0.0

    # Chronology signal (only used in pass 2 for initiation)
    chronology_signal = 0.0
    if role == "initiation" and selected_decision_date is not None:
        try:
            parsed = date.fromisoformat(row["parsed_date"])
            if parsed >= selected_decision_date:
                chronology_signal = -5.0  # strong penalty: initiation after decision
            else:
                days_before = (selected_decision_date - parsed).days
                if days_before > 0:
                    chronology_signal = min(2.0, days_before / 365.0)  # small boost for valid ordering
        except (ValueError, TypeError):
            pass

    # Repeated mention signal
    mention_count = int(row.get("date_mention_count", 1) or 1)
    repeated_mention_signal = min(1.0, (mention_count - 1) * 0.25)

    # Negative penalty
    neg_flags = str(row.get("negative_cue_flags") or "")
    negative_penalty = 0.0
    if "historical_cue" in neg_flags:
        negative_penalty += 4.0
    if "reject_cue" in neg_flags:
        negative_penalty += 6.0
    if row.get("candidate_role") == "historical":
        negative_penalty += 3.0
    if row.get("historical_gap_candidate"):
        negative_penalty += 2.0

    total = (
        source_strength + role_cue_strength + doc_priority + section_priority
        + page_priority + position_signal + classifier_signal
        + chronology_signal + repeated_mention_signal - negative_penalty
    )
    return total


def _apply_historical_gap_rule(
    candidate_dates: list[date],
    process_type: str,
) -> set[date]:
    """
    Return dates that are marked historical_gap_candidate.
    For CE/EA: flag all dates before the first gap > GAP_DAYS.
    For EIS: skip (return empty set).
    """
    if process_type == "EIS" and EIS_GAP_EXEMPT:
        return set()
    if len(candidate_dates) < 2:
        return set()

    sorted_dates = sorted(set(candidate_dates))
    gap_cutoff: date | None = None

    for i in range(1, len(sorted_dates)):
        gap = (sorted_dates[i] - sorted_dates[i - 1]).days
        if gap > GAP_DAYS:
            # First gap wins
            gap_cutoff = sorted_dates[i]
            break

    if gap_cutoff is None:
        return set()

    return {d for d in sorted_dates if d < gap_cutoff}


def select_dates_for_project(
    cands: pd.DataFrame,
    process_type: str,
    index_map: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Run two-pass selection for a single project.
    Returns (project_dates_dict, updated_candidates_df).
    """
    if cands.empty:
        return _empty_project_result(process_type), cands

    # Parse dates
    cands = cands.copy()
    cands["_parsed_date"] = pd.to_datetime(cands["parsed_date"], errors="coerce").dt.date

    # --- Historical gap flagging (before either pass) ---
    valid_dates = cands["_parsed_date"].dropna().tolist()
    historical_gap_set = _apply_historical_gap_rule(valid_dates, process_type)
    cands["historical_gap_candidate"] = cands["_parsed_date"].apply(
        lambda d: d in historical_gap_set if pd.notna(d) else False
    )

    # --- Pass 1: Score and select decision ---
    decision_cands = cands[cands["candidate_role"].isin(["clear_decision", "proxy_decision"])].copy()
    decision_cands["ranking_score"] = [
        _compute_candidate_score(r, "decision", None, index_map)
        for r in decision_cands.to_dict("records")
    ]
    cands.loc[decision_cands.index, "ranking_score"] = decision_cands["ranking_score"]

    best_decision = None
    selected_decision_id = None
    decision_date_str = None
    decision_granularity = "unknown"
    decision_source_type = None
    decision_confidence = "missing"
    decision_is_proxy = False
    decision_evidence_text = None
    decision_document_id = None
    decision_page_number = None

    # Only select from clear_decision in pass 1 (proxies only if no clear found)
    clear_dec = decision_cands[
        (decision_cands["candidate_role"] == "clear_decision") &
        (decision_cands["ranking_score"] > 0)
    ]
    if not clear_dec.empty:
        best_decision = clear_dec.loc[clear_dec["ranking_score"].idxmax()]
    else:
        proxy_dec = decision_cands[
            (decision_cands["candidate_role"] == "proxy_decision") &
            (decision_cands["ranking_score"] > -2)
        ]
        if not proxy_dec.empty:
            best_decision = proxy_dec.loc[proxy_dec["ranking_score"].idxmax()]
            decision_is_proxy = True

    if best_decision is not None:
        try:
            decision_date_obj = best_decision["_parsed_date"]
            if pd.notna(decision_date_obj):
                decision_date_str = decision_date_obj.isoformat()
                decision_granularity = best_decision.get("date_granularity", "day")
                decision_source_type = best_decision.get("candidate_source_type", "document_text")
                decision_confidence = best_decision.get("role_confidence", "medium")
                decision_is_proxy = best_decision.get("candidate_role") == "proxy_decision"
                decision_evidence_text = str(best_decision.get("context_text", ""))[:300]
                decision_document_id = best_decision.get("document_id")
                decision_page_number = best_decision.get("page_number")
                selected_decision_id = best_decision.get("candidate_id")
        except Exception:
            pass

    # --- Pass 2: Score initiation with decision as anchor ---
    selected_decision_date: date | None = None
    if decision_date_str:
        try:
            selected_decision_date = date.fromisoformat(decision_date_str)
        except ValueError:
            pass

    initiation_cands = cands[
        cands["candidate_role"].isin(["clear_initiation", "proxy_initiation"])
    ].copy()
    initiation_cands["ranking_score"] = [
        _compute_candidate_score(r, "initiation", selected_decision_date, index_map)
        for r in initiation_cands.to_dict("records")
    ]
    cands.loc[initiation_cands.index, "ranking_score"] = initiation_cands["ranking_score"]

    best_initiation = None
    selected_initiation_id = None
    initiation_date_str = None
    initiation_granularity = "unknown"
    initiation_source_type = None
    initiation_confidence = "missing"
    initiation_is_proxy = False
    initiation_evidence_text = None
    initiation_document_id = None
    initiation_page_number = None

    clear_init = initiation_cands[
        (initiation_cands["candidate_role"] == "clear_initiation") &
        (initiation_cands["ranking_score"] > 0)
    ]
    # Apply chronology filter: must be before decision
    if selected_decision_date is not None:
        clear_init = clear_init[
            clear_init["_parsed_date"].apply(
                lambda d: pd.notna(d) and d < selected_decision_date
            )
        ]

    if not clear_init.empty:
        best_initiation = clear_init.loc[clear_init["ranking_score"].idxmax()]
    else:
        # Proxy fallback (sensitivity only)
        proxy_init = initiation_cands[
            (initiation_cands["candidate_role"] == "proxy_initiation") &
            (initiation_cands["ranking_score"] > -2)
        ]
        if selected_decision_date is not None:
            proxy_init = proxy_init[
                proxy_init["_parsed_date"].apply(
                    lambda d: pd.notna(d) and d < selected_decision_date
                )
            ]
        if not proxy_init.empty:
            best_initiation = proxy_init.loc[proxy_init["ranking_score"].idxmax()]
            initiation_is_proxy = True

    if best_initiation is not None:
        try:
            init_date_obj = best_initiation["_parsed_date"]
            if pd.notna(init_date_obj):
                initiation_date_str = init_date_obj.isoformat()
                initiation_granularity = best_initiation.get("date_granularity", "day")
                initiation_source_type = best_initiation.get("candidate_source_type", "document_text")
                initiation_confidence = best_initiation.get("role_confidence", "medium")
                initiation_is_proxy = best_initiation.get("candidate_role") == "proxy_initiation"
                initiation_evidence_text = str(best_initiation.get("context_text", ""))[:300]
                initiation_document_id = best_initiation.get("document_id")
                initiation_page_number = best_initiation.get("page_number")
                selected_initiation_id = best_initiation.get("candidate_id")
        except Exception:
            pass

    # --- Mark selected candidates ---
    if selected_decision_id:
        cands.loc[cands["candidate_id"] == selected_decision_id, "selected_for_decision"] = True
    if selected_initiation_id:
        cands.loc[cands["candidate_id"] == selected_initiation_id, "selected_for_initiation"] = True

    # --- Determine timeline_status and flags ---
    has_init = initiation_date_str is not None
    has_dec = decision_date_str is not None

    flags: list[str] = []
    timeline_status = "missing_both"

    if has_init and has_dec:
        init_d = date.fromisoformat(initiation_date_str)
        dec_d = date.fromisoformat(decision_date_str)
        if init_d > dec_d:
            timeline_status = "invalid_order"
            flags.append("invalid_order")
        elif init_d == dec_d:
            flags.append(SAME_DAY_DURATION_FLAG)
            if process_type == "CE":
                flags.append("same_day_ce_review")
                timeline_status = "manual_review"
            else:
                timeline_status = "complete_clear" if not (initiation_is_proxy or decision_is_proxy) else "complete_with_proxy"
        else:
            duration_days_val = (dec_d - init_d).days
            if duration_days_val / 365.25 > MAX_DURATION_YEARS:
                flags.append("duration_gt_25y")
                timeline_status = "manual_review"
            else:
                timeline_status = (
                    "complete_clear" if not (initiation_is_proxy or decision_is_proxy)
                    else "complete_with_proxy"
                )
    elif has_dec and not has_init:
        timeline_status = "missing_initiation"
        flags.append("missing_initiation")
    elif has_init and not has_dec:
        timeline_status = "missing_decision"
        flags.append("missing_decision")
    else:
        flags.append("missing_initiation")
        flags.append("missing_decision")

    if initiation_is_proxy:
        flags.append("proxy_initiation")
    if decision_is_proxy:
        flags.append("proxy_decision")
    if initiation_is_proxy and decision_is_proxy:
        flags.append("proxy_only")

    # duration_days: only when both day-granularity
    duration_days: int | None = None
    if (
        has_init and has_dec
        and initiation_granularity == "day"
        and decision_granularity == "day"
        and timeline_status not in ("invalid_order",)
    ):
        init_d = date.fromisoformat(initiation_date_str)
        dec_d = date.fromisoformat(decision_date_str)
        if dec_d >= init_d:
            duration_days = (dec_d - init_d).days
    else:
        if has_init and has_dec and initiation_granularity != "day":
            flags.append("non_day_granularity")

    # Check for multiple high-score candidates (tie situation)
    if not clear_dec.empty and len(clear_dec[clear_dec["ranking_score"] >= (clear_dec["ranking_score"].max() - 1)]) > 1:
        flags.append("multiple_high_score_candidates")

    if decision_confidence == "low" or initiation_confidence == "low":
        flags.append("low_confidence_selection")

    return {
        "project_id": cands["project_id"].iloc[0],
        "process_type": process_type,
        "initiation_date": initiation_date_str,
        "initiation_date_granularity": initiation_granularity,
        "initiation_source_type": initiation_source_type,
        "initiation_confidence": initiation_confidence,
        "initiation_is_proxy": initiation_is_proxy,
        "initiation_evidence_text": initiation_evidence_text,
        "initiation_document_id": initiation_document_id,
        "initiation_page_number": str(initiation_page_number) if initiation_page_number is not None else None,
        "decision_date": decision_date_str,
        "decision_date_granularity": decision_granularity,
        "decision_source_type": decision_source_type,
        "decision_confidence": decision_confidence,
        "decision_is_proxy": decision_is_proxy,
        "decision_evidence_text": decision_evidence_text,
        "decision_document_id": decision_document_id,
        "decision_page_number": str(decision_page_number) if decision_page_number is not None else None,
        "duration_days": duration_days,
        "timeline_status": timeline_status,
        "timeline_flags": "|".join(flags) if flags else "",
        "timeline_run_at": datetime.now(timezone.utc).isoformat(),
    }, cands


def _empty_project_result(process_type: str) -> dict:
    return {
        "project_id": None,
        "process_type": process_type,
        "initiation_date": None,
        "initiation_date_granularity": "unknown",
        "initiation_source_type": None,
        "initiation_confidence": "missing",
        "initiation_is_proxy": False,
        "initiation_evidence_text": None,
        "initiation_document_id": None,
        "initiation_page_number": None,
        "decision_date": None,
        "decision_date_granularity": "unknown",
        "decision_source_type": None,
        "decision_confidence": "missing",
        "decision_is_proxy": False,
        "decision_evidence_text": None,
        "decision_document_id": None,
        "decision_page_number": None,
        "duration_days": None,
        "timeline_status": "missing_both",
        "timeline_flags": "missing_initiation|missing_decision",
        "timeline_run_at": datetime.now(timezone.utc).isoformat(),
    }


def apply_manual_corrections(
    dates_df: pd.DataFrame,
    corrections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply manual corrections to project dates in-place."""
    if corrections_df.empty:
        return dates_df

    active = corrections_df[corrections_df["correction_status"] == "active"].copy()
    if active.empty:
        return dates_df

    for _, corr in active.iterrows():
        pid = corr["project_id"]
        role = corr["correction_role"]
        mask = dates_df["project_id"] == pid
        if not mask.any():
            continue

        corrected_date = corr.get("corrected_date")
        corrected_granularity = corr.get("corrected_date_granularity", "day")
        corrected_source = corr.get("corrected_source_type", "manual")
        corrected_conf = corr.get("corrected_confidence", "high")
        corrected_proxy = bool(corr.get("corrected_is_proxy", False))

        if role == "initiation":
            dates_df.loc[mask, "initiation_date"] = (
                corrected_date.isoformat() if pd.notna(corrected_date) else None
            )
            dates_df.loc[mask, "initiation_date_granularity"] = corrected_granularity
            dates_df.loc[mask, "initiation_source_type"] = corrected_source
            dates_df.loc[mask, "initiation_confidence"] = corrected_conf
            dates_df.loc[mask, "initiation_is_proxy"] = corrected_proxy
        elif role == "decision":
            dates_df.loc[mask, "decision_date"] = (
                corrected_date.isoformat() if pd.notna(corrected_date) else None
            )
            dates_df.loc[mask, "decision_date_granularity"] = corrected_granularity
            dates_df.loc[mask, "decision_source_type"] = corrected_source
            dates_df.loc[mask, "decision_confidence"] = corrected_conf
            dates_df.loc[mask, "decision_is_proxy"] = corrected_proxy

        # Add manual_override flag
        existing_flags = str(dates_df.loc[mask, "timeline_flags"].iloc[0])
        if "manual_override" not in existing_flags:
            new_flags = "|".join(filter(None, [existing_flags, "manual_override"]))
            dates_df.loc[mask, "timeline_flags"] = new_flags

    return dates_df


def build_review_queue(
    dates_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the manual review queue from projects that need human review.
    """
    # Trigger conditions from plan §9
    needs_review = (
        # missing initiation with plausible candidates
        (
            (dates_df["timeline_status"] == "missing_initiation") &
            dates_df["project_id"].isin(
                candidates_df[candidates_df["candidate_role"].isin(["clear_initiation", "proxy_initiation"])]["project_id"]
            )
        ) |
        # missing decision with plausible candidates
        (
            (dates_df["timeline_status"] == "missing_decision") &
            dates_df["project_id"].isin(
                candidates_df[candidates_df["candidate_role"].isin(["clear_decision", "proxy_decision"])]["project_id"]
            )
        ) |
        # invalid order
        (dates_df["timeline_status"] == "invalid_order") |
        # manual_review status
        (dates_df["timeline_status"] == "manual_review") |
        # duration >25 years
        (dates_df["duration_days"].notna() & (dates_df["duration_days"] > MAX_DURATION_YEARS * 365)) |
        # high disagreement flags
        dates_df["timeline_flags"].str.contains("multiple_high_score_candidates", na=False) |
        dates_df["timeline_flags"].str.contains("proxy_only", na=False)
    )

    queue_projects = dates_df[needs_review].copy()
    if queue_projects.empty:
        return pd.DataFrame()

    # Add top 5 initiation and decision candidates per project
    def top_cands(project_id: str, role_filter: list[str], n: int = 5) -> str:
        sub = candidates_df[
            (candidates_df["project_id"] == project_id) &
            (candidates_df["candidate_role"].isin(role_filter))
        ].nlargest(n, "ranking_score")
        if sub.empty:
            return ""
        parts = []
        for _, row in sub.iterrows():
            parts.append(
                f"{row.get('parsed_date')} [{row.get('candidate_role')}|{row.get('role_confidence')}] "
                f"score={row.get('ranking_score', 0):.1f} | "
                f"{str(row.get('context_text', ''))[:120]}"
            )
        return " ||| ".join(parts)

    queue_projects["top_initiation_candidates"] = queue_projects["project_id"].map(
        lambda pid: top_cands(pid, ["clear_initiation", "proxy_initiation"])
    )
    queue_projects["top_decision_candidates"] = queue_projects["project_id"].map(
        lambda pid: top_cands(pid, ["clear_decision", "proxy_decision"])
    )

    # Reviewer fields
    for col in ["manual_initiation_date", "manual_decision_date", "manual_notes", "manual_status"]:
        queue_projects[col] = ""

    return queue_projects


def import_corrections_from_csv(csv_path: str) -> None:
    """
    Convert a filled review queue CSV into timeline_manual_corrections.parquet entries.
    Validates required fields and appends to the corrections table.
    """
    df = pd.read_csv(csv_path)
    required_cols = ["project_id", "process_type", "manual_notes", "manual_status"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in corrections CSV: {missing}")

    rows: list[dict] = []
    skipped = 0
    run_at = datetime.now(timezone.utc).isoformat()

    for _, row in df.iterrows():
        pid = str(row.get("project_id", "")).strip()
        if not pid:
            skipped += 1
            continue
        notes = str(row.get("manual_notes", "")).strip()
        if not notes:
            print(f"  SKIP {pid}: manual_notes is empty (required)")
            skipped += 1
            continue
        status = str(row.get("manual_status", "")).strip()
        if not status:
            skipped += 1
            continue

        for role in ["initiation", "decision"]:
            date_col = f"manual_{role}_date"
            if date_col not in df.columns:
                continue
            date_val = str(row.get(date_col, "")).strip()
            if not date_val:
                continue
            corrected_date = pd.to_datetime(date_val, errors="coerce")
            if pd.isna(corrected_date):
                print(f"  WARN {pid}: could not parse {date_col}={date_val!r}")
                continue

            correction_id = hashlib.sha1(
                f"{pid}|{role}|{corrected_date.date().isoformat()}".encode()
            ).hexdigest()[:20]

            rows.append({
                "correction_id": correction_id,
                "project_id": pid,
                "process_type": str(row.get("process_type", "")),
                "correction_role": role,
                "corrected_date": corrected_date.date().isoformat(),
                "corrected_date_granularity": "day",
                "corrected_source_type": "manual",
                "corrected_confidence": "high",
                "corrected_is_proxy": False,
                "prior_date": row.get(f"{role}_date"),
                "prior_source_type": row.get(f"{role}_source_type"),
                "prior_confidence": row.get(f"{role}_confidence"),
                "correction_reason": notes,
                "evidence_text": str(row.get("top_decision_candidates" if role == "decision" else "top_initiation_candidates", ""))[:500],
                "evidence_document_id": None,
                "evidence_page_number": None,
                "reviewer": str(row.get("gold_reviewer", "reviewer")).strip() or "reviewer",
                "reviewed_at": run_at,
                "correction_status": "active",
            })

    if not rows:
        print(f"No valid corrections found (skipped {skipped}).")
        return

    new_df = pd.DataFrame(rows)

    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    if CORRECTIONS_PATH.exists():
        existing = pd.read_parquet(CORRECTIONS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates("correction_id", keep="last")
    else:
        combined = new_df

    combined.to_parquet(CORRECTIONS_PATH, index=False)
    print(f"Wrote {len(rows)} corrections (total {len(combined)}) to {CORRECTIONS_PATH}")
    print(f"Skipped {skipped} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select timeline dates from candidates.")
    parser.add_argument(
        "--process", nargs="+", choices=["CE", "EA", "EIS"], default=["CE", "EA", "EIS"]
    )
    parser.add_argument("--sample-ids", help="Path to a file with one project_id per line.")
    parser.add_argument("--import-corrections", metavar="CSV", help="Import filled review queue CSV into corrections table.")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output even if it already exists.")
    parser.add_argument("--run-dir", help="Override run directory (reads candidates from here, writes dates here).")
    args = parser.parse_args()

    if args.import_corrections:
        import_corrections_from_csv(args.import_corrections)
        return

    # Resolve run directory — matches the logic in scripts 02 and 03.
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.sample_ids:
        run_dir = TIMELINE_DIR / "sample_runs" / Path(args.sample_ids).stem
    else:
        run_dir = TIMELINE_DIR
    candidates_path = run_dir / "timeline_candidates.parquet"
    dates_path = run_dir / "timeline_project_dates.parquet"
    # INDEX_PATH and CORRECTIONS_PATH always live in the main timeline/ dir.

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates not found: {candidates_path}\nRun 03_extract_timeline_candidates.py first.")

    project_ids: set[str] | None = None
    if args.sample_ids:
        with open(args.sample_ids) as f:
            project_ids = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(project_ids)} sample project IDs.")

    run_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidates: {candidates_path}")
    candidates_df = pd.read_parquet(candidates_path)
    candidates_df = candidates_df[candidates_df["process_type"].isin(args.process)]
    if project_ids:
        candidates_df = candidates_df[candidates_df["project_id"].isin(project_ids)]
    print(f"  {len(candidates_df):,} candidates, {candidates_df['project_id'].nunique():,} projects")

    # Load index for document type scoring
    index_map: dict = {}
    if INDEX_PATH.exists():
        idx = pd.read_parquet(INDEX_PATH, columns=["project_id", "decision_doc_score", "initiation_doc_score"])
        for pid, grp in idx.groupby("project_id"):
            index_map[pid] = {
                "decision_doc_score": grp["decision_doc_score"].max(),
                "initiation_doc_score": grp["initiation_doc_score"].max(),
            }

    # Load manual corrections if available
    corrections_df = pd.DataFrame()
    if CORRECTIONS_PATH.exists():
        corrections_df = pd.read_parquet(CORRECTIONS_PATH)
        print(f"Loaded {len(corrections_df)} manual corrections.")

    # Pre-group candidates by project_id for O(1) per-project lookup (avoids O(n×m) scan)
    candidates_by_proj: dict[str, pd.DataFrame] = {
        pid: grp.reset_index(drop=True)
        for pid, grp in candidates_df.groupby("project_id", sort=False)
    }

    # Process each project
    project_dates_rows: list[dict] = []
    updated_cands_parts: list[pd.DataFrame] = []

    projects = candidates_df["project_id"].unique()
    print(f"Processing {len(projects):,} projects...")
    for i, pid in enumerate(projects):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(projects)} done...")
        proj_cands = candidates_by_proj.get(pid, pd.DataFrame())
        pt = proj_cands["process_type"].iloc[0]
        result_dict, updated_cands = select_dates_for_project(proj_cands, pt, index_map)
        result_dict["project_id"] = pid
        project_dates_rows.append(result_dict)
        updated_cands_parts.append(updated_cands)

    if not project_dates_rows:
        print("No results.")
        return

    dates_df = pd.DataFrame(project_dates_rows)

    # Apply manual corrections
    if not corrections_df.empty:
        dates_df = apply_manual_corrections(dates_df, corrections_df)
        print(f"Applied manual corrections to {dates_df['timeline_flags'].str.contains('manual_override', na=False).sum()} projects.")

    # Save project dates
    if args.append and dates_path.exists():
        existing = pd.read_parquet(dates_path)
        dates_df = pd.concat([existing, dates_df], ignore_index=True)
        dates_df = dates_df.drop_duplicates("project_id", keep="last")
    dates_df.to_parquet(dates_path, index=False)
    print(f"Wrote: {dates_path} ({len(dates_df):,} projects)")
    print("Timeline status distribution:")
    print(dates_df["timeline_status"].value_counts().to_string())

    # Save updated candidates (with scoring columns and selected flags)
    if updated_cands_parts:
        updated_cands_df = pd.concat(updated_cands_parts, ignore_index=True)
        updated_cands_df = updated_cands_df.drop(columns=["_parsed_date"], errors="ignore")
        if args.append and candidates_path.exists():
            existing_cands = pd.read_parquet(candidates_path)
            not_updated = existing_cands[~existing_cands["candidate_id"].isin(updated_cands_df["candidate_id"])]
            updated_cands_df = pd.concat([not_updated, updated_cands_df], ignore_index=True)
        updated_cands_df.to_parquet(candidates_path, index=False)

    # Build review queue
    all_cands = pd.concat(updated_cands_parts, ignore_index=True) if updated_cands_parts else candidates_df
    queue_df = build_review_queue(dates_df, all_cands)
    if not queue_df.empty:
        queue_df.to_csv(REVIEW_QUEUE_PATH, index=False)
        print(f"Wrote review queue: {REVIEW_QUEUE_PATH} ({len(queue_df)} projects)")
    else:
        print("No projects flagged for review queue.")

    # Summary
    complete_clear = (dates_df["timeline_status"] == "complete_clear").sum()
    total = len(dates_df)
    print(f"\ncomplete_clear: {complete_clear}/{total} ({100*complete_clear/total:.1f}%)")


if __name__ == "__main__":
    main()
