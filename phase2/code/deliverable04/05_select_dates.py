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
    python 05_select_dates.py [--process CE EA EIS] [--sample-ids path]
    python 05_select_dates.py --import-corrections filled_queue.csv
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

import duckdb
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

# Toggle: consume 05b's learned ranker scores (learned_*_score) to RE-RANK among eligible
# candidates. Set D4_USE_LEARNED_RANKER=0 to fall back to the pure heuristic (for A/B baselining).
USE_LEARNED_RANKER = os.environ.get("D4_USE_LEARNED_RANKER", "1") == "1"

# --- Selection-disambiguation rules (2026-06-04) -------------------------------------------
# Earliest-wins for initiation: among initiation candidates scoring within this margin of the
# best, pick the EARLIEST date (initiation = the first qualifying start signal, e.g. the first
# of two scoping periods). Kept tight (1.0) so only genuinely near-tied candidates compete —
# a looser margin lets a low-quality earlier candidate (case-number/citation month) steal the
# slot, because today this tiebreak runs on REGEX ranking_score, not classifier confidence.
# TODO: once classifier p_initiation is wired into selection, gate this on p_init instead.
INIT_EARLIEST_SCORE_MARGIN = 1.0
# A bare month-granularity date may stand in as a DECISION only for these processes: the CX
# cover month IS the determination for a CE. For EA/EIS the real decision is a dated ROD/FONSI,
# so a month-only date is dropped from the decision pool and the project routes to 06 for a
# precise date instead of locking in a coarse one.
MONTH_DECISION_PROCESSES = {"CE"}

# --- Classifier-into-selection weights (2026-06-04) ----------------------------------------
# The two-head classifier (p_initiation / p_decision) finally drives selection, not just 06
# routing. Used as an additive ranking term on the role-appropriate head; calibrated probs
# (p_init_cal / p_dec_cal from 04b --apply) are preferred when present, else raw p_*.
CLASSIFIER_WEIGHT = 5.0            # weight on the role-appropriate classifier probability
CLASSIFIER_DISAGREE_PENALTY = 3.0  # penalty when the OTHER head is more confident (looks like the other thing)
# Granularity confidence: a precise day beats a coarse month/year.
GRANULARITY_BONUS = {"day": 1.0, "month": 0.0, "year": -1.0}
# Cross-candidate agreement: corroboration when multiple candidates resolve to the same date.
AGREEMENT_WEIGHT = 0.5
AGREEMENT_CAP = 1.5
# Process-specific plausible review durations (days). Pairs outside the band are FLAGGED for
# review/06, not discarded — a sanity check, not a hard filter.
DURATION_PLAUSIBLE_MAX_DAYS = {"CE": 5 * 365, "EA": 7 * 365, "EIS": 12 * 365}
DURATION_PLAUSIBLE_MIN_DAYS = {"CE": 0, "EA": 0, "EIS": 90}

# ---------------------------------------------------------------------------
# Scoring weights / component ranges (plan §5 table)
# ---------------------------------------------------------------------------

SOURCE_STRENGTH = {
    "tier_a": 5,
    # CE project description: source_strength=4 so descriptions with clear initiation/decision
    # cues rank above generic document body text (3) but below authoritative register dates (5)
    # and strong signed-document signals (tier_a doc_priority boosts push those to 14+).
    "ce_description": 4,
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


def _classifier_probs(row: dict) -> tuple[float, float]:
    """Role-appropriate classifier probabilities, preferring calibrated (p_*_cal, written by
    04b --apply) over raw (p_initiation / p_decision) when present. Returns (p_init, p_dec)."""
    def _num(*keys: str) -> float:
        for k in keys:
            v = row.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return 0.0
    return _num("p_init_cal", "p_initiation"), _num("p_dec_cal", "p_decision")


def candidate_score_components(
    row: dict,
    role: str,  # "decision" or "initiation"
    selected_decision_date: date | None,
) -> dict[str, float]:
    """Named additive components of a candidate's ranking_score. Returned as a dict so the same
    feature set can later feed a learned ranker (LightGBM) without re-deriving the inputs — the
    heuristic score is just sum(components.values())."""
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
    elif any(kw in heading for kw in [
        "introduction", "background", "purpose and need", "scoping",
        "proposed action", "description of proposed action",
    ]):
        # "description of proposed action" is the heading on ce_description packets —
        # gives them the same section boost as background/purpose sections so they
        # rank above document body text without a section context (section_priority=0)
        # but not above decision-section or decision-document signals.
        section_priority = 2.0
    elif any(kw in heading for kw in ["references", "bibliography", "appendix", "preparers"]):
        section_priority = -2.0

    # Page priority (use retrieval score from context packet as proxy)
    page_priority = min(3.0, max(0.0, float(row.get("retrieval_score", 0) or 0) / 3.0))

    # Position signal — ROLE-AWARE: decision dates (signatures, Decision Record) cluster at the
    # END of CE/EA forms; initiation dates (NOI / scoping / application-received) near the START.
    # A "decision" date on page 1 (or an "initiation" date on the last page) is suspect.
    position_signal = 0.0
    pos_raw = row.get("position_pct")
    if pos_raw is not None:
        try:
            pos = float(pos_raw)
            if role == "decision":
                position_signal = 1.5 if pos > 0.85 else (-0.5 if pos < 0.10 else 0.0)
            else:  # initiation
                position_signal = 1.5 if pos < 0.15 else (-0.5 if pos > 0.90 else 0.0)
        except (TypeError, ValueError):
            pass

    # Classifier signal — the two-head model drives selection. Boost the role-appropriate head;
    # penalise when the OTHER head is more confident (this date looks like the other milestone).
    p_init, p_dec = _classifier_probs(row)
    own, other = (p_dec, p_init) if role == "decision" else (p_init, p_dec)
    classifier_signal = CLASSIFIER_WEIGHT * own
    if other > own:
        classifier_signal -= CLASSIFIER_DISAGREE_PENALTY * (other - own)

    # Granularity confidence: a precise day beats a coarse month / year.
    granularity_signal = GRANULARITY_BONUS.get(str(row.get("date_granularity") or ""), 0.0)

    # Cross-candidate agreement: corroboration when multiple candidates resolve to the same date
    # (precomputed per project into _agreement_count by select_dates_for_project).
    agreement_count = int(row.get("_agreement_count", 1) or 1)
    agreement_signal = min(AGREEMENT_CAP, max(0, agreement_count - 1) * AGREEMENT_WEIGHT)

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

    # July-1 year-proxy penalty: YYYY-07-01 dates are frequently constructed from a
    # NEPA case number year (DOI-BLM-...-2015-...) rather than an explicit document
    # date. A real specific date (e.g. 08/18/2015) on the same page will beat this
    # by ≥1 point. Legitimate July-1 decisions still win if other evidence supports them.
    try:
        _d = date.fromisoformat(str(row["parsed_date"]))
        if _d.month == 7 and _d.day == 1:
            negative_penalty += 2.0
    except (ValueError, TypeError, KeyError):
        pass

    return {
        "source_strength": float(source_strength),
        "role_cue_strength": float(role_cue_strength),
        "doc_priority": float(doc_priority),
        "section_priority": float(section_priority),
        "page_priority": float(page_priority),
        "position_signal": float(position_signal),
        "classifier_signal": float(classifier_signal),
        "granularity_signal": float(granularity_signal),
        "agreement_signal": float(agreement_signal),
        "chronology_signal": float(chronology_signal),
        "repeated_mention_signal": float(repeated_mention_signal),
        "negative_penalty": -float(negative_penalty),
    }


def _compute_candidate_score(
    row: dict,
    role: str,  # "decision" or "initiation"
    selected_decision_date: date | None,
    index_map: dict,  # retained for signature compatibility (callers pass it); unused here
) -> float:
    """Composite ranking_score = sum of the named components in candidate_score_components."""
    return sum(candidate_score_components(row, role, selected_decision_date).values())


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


# --- Phase C gating (Step 4, 2026-06-08) ---------------------------------------------------------
# The Step-1 recall audit on the Codex-labeled gold showed the deterministic document-text EIS
# selection is unreliable: the rod_lang/proxy tiers FABRICATE on no-ROD projects (bucket 1), and
# the clear/body tiers MIS-PICK among in-pool candidates (bucket 2). Recall is good (the true date
# is almost always already a candidate), so resolution belongs to the classifier -> ranker -> LLM
# path (Steps 2-3), NOT to deterministic selection. Until that path is built, EIS emits ONLY
# authoritative register RODs; all other EIS decisions and the Final-EIS endpoint are left missing.
# Flip these to True once the classifier/LLM resolution is validated.
EIS_DETERMINISTIC_DOC_ROD = False   # document-text ROD tiers (clear/body/proxy/rod_lang)
EIS_FINAL_EIS_ENABLED = False       # deterministic final_eis_date population (separate field)
# Tiered EIS decision: ROD-first, FEIS-fallback. Per project, has_rod = does any ROD-eligible
# candidate exist (register ROD / ROD-typed doc / explicit ROD language). If yes, the decision pool
# is ROD-eligible candidates ONLY (ranker orders within). If no, the decision falls back to FEIS-doc
# candidates ordered by the calibrated final_eis head (p_feis_cal), flagged decision_is_feis_fallback.
# ROD outranks FEIS by CONSTRUCTION (FEIS never enters the pool when a ROD exists). Validated by the
# gold-rank check (true ROD top-5 90%, true FEIS top-5 95% after the 3-head rebuild + doc-type gate).
EIS_TIERED_DECISION = True

# Calibrated initiation eligibility for EA and EIS (recover_eis.md Phase 4, §5.5/§5.6). The LightGBM
# LambdaRank score is group-relative and CANNOT serve as an existence gate, which is why ~1,680 EIS
# and ~813 EA projects have an initiation candidate that is never selected (ranking_score <= 0).
# For these processes, eligibility is the UNION of the legacy ranker-score gate and a calibrated /
# authoritative gate (authoritative source OR p_init_cal >= threshold). The union is ADDITIVE: it
# never drops a candidate the legacy gate already accepted, and recovers calibrated-eligible
# candidates the ranker suppressed. CE keeps the original ranking_score gate untouched (CE is already
# at its target rate and is too large / unvalidatable to perturb before the deadline).
# Thresholds are provisional (Phase 4 prerequisite: refine on a frozen init label set per process;
# phase0/cohort_ranker_blocked_init.txt is a labeling target).
T_INIT_CAL = {"EA": 0.5, "EIS": 0.5}
CALIBRATED_INIT_PROCESSES = set(T_INIT_CAL)  # {"EA", "EIS"} — CE excluded by design

# An initiation can never be OMB/paperwork-reduction form boilerplate (the "Public reporting burden
# ... OMB control ... Expires <date>" stamp). Leaked a wrong init (18-year duration) in sample
# testing. Excluded regardless of which gate accepts the candidate. (recover_eis.md Phase 4b.)
_INIT_NEG_RE = re.compile(
    r"public\s+reporting\s+burden|omb\s+control|paperwork\s+reduction\s+act", re.IGNORECASE
)

# Implausible-duration guard: an initiation implausibly far before the decision is almost certainly a
# wrong/incidental date. Per-process (EA reviews are short; EIS run longer). Applied in the chronology
# filter to drop absurd-early init candidates. (recover_eis.md Phase 4b.)
MAX_INIT_LOOKBACK_DAYS = {"EA": 3650, "EIS": 5475}  # ~10y EA, ~15y EIS


def _calibrated_init_eligible(df: pd.DataFrame, process_type: str, role: str) -> pd.Series:
    """Initiation eligibility for EA/EIS: UNION of the legacy ranker-score gate and a calibrated/
    authoritative gate, minus OMB/paperwork boilerplate. `role` is "clear" (score>0) or "proxy"
    (score>-2), matching the legacy thresholds so the union strictly contains the legacy pool.
    Returns a bool Series aligned to df.index."""
    score_gate = df["ranking_score"] > (0 if role == "clear" else -2)
    p = pd.to_numeric(df.get("p_init_cal"), errors="coerce").fillna(0.0)
    authoritative = df["candidate_source_type"].astype(str).eq("metadata")
    cal_gate = authoritative | (p >= T_INIT_CAL[process_type])
    ctx = df.get("model_context")
    if ctx is None:
        ctx = df.get("context_text")
    neg = ctx.fillna("").str.contains(_INIT_NEG_RE) if ctx is not None else pd.Series(False, index=df.index)
    return (score_gate | cal_gate) & ~neg

# Routing gate for 06 (LLM adjudication). A project's decision routes to the LLM when the selected
# candidate's calibrated confidence is below this (ambiguous), or when no decision was picked but
# eligible candidates exist (the LLM may resolve one), or when several candidates tie. Above the
# threshold the deterministic pick is taken as final (no LLM). Set to 0.7 to match the bimodal
# calibrated score distribution (clear picks cluster >=0.7). Tracked per project as route_to_llm.
LLM_ROUTE_THRESHOLD = 0.7

# Explicit "Record of Decision ... signed/issued/dated/approved" language — used as ROD evidence
# for EIS candidates that sit outside a ROD-typed document (mislabeled doc types).
EIS_ROD_LANG_RE = re.compile(
    r"record\s+of\s+decision[\s,]*(?:was\s+|is\s+|has\s+been\s+)?(?:sign|issu|approv|dat)"
    r"|\brod\b[\s,]*(?:was\s+|is\s+|has\s+been\s+)?(?:sign|issu|approv|dat)"
    r"|(?:sign|issu|approv)\w*\s+(?:the\s+)?(?:rod|record\s+of\s+decision)",
    re.IGNORECASE,
)


def _eis_rod_pool(decision_cands: pd.DataFrame) -> pd.DataFrame:
    """ROD-eligible decision candidates: (1) register-typed ROD (metadata `clear_decision`),
    (2) date in a ROD-typed document, or (3) explicit "Record of Decision ... signed/issued/dated"
    language. Deliberately EXCLUDES broad `clear_decision` dates in FONSI/CE/"selected alternative"
    non-ROD documents, and excludes FEIS-doc dates (those are the FEIS fallback, not a ROD)."""
    if decision_cands.empty:
        return decision_cands
    src = decision_cands["candidate_source_type"].astype(str)
    role = decision_cands["candidate_role"]
    dtc = decision_cands["document_type_clean"].astype(str).str.upper()
    is_register_rod = src.eq("metadata") & role.eq("clear_decision")
    is_rod_doc = dtc.eq("ROD")
    has_rod_lang = decision_cands["context_text"].fillna("").map(
        lambda t: bool(EIS_ROD_LANG_RE.search(str(t)))
    )
    return decision_cands[is_register_rod | is_rod_doc | has_rod_lang]


def _eis_feis_pool(cands: pd.DataFrame) -> pd.DataFrame:
    """FEIS-doc candidates eligible as the FALLBACK decision when a project has no ROD. Ordered by
    the calibrated final_eis head (p_feis_cal, gated to FEIS docs); falls back to raw p_final_eis,
    then any FEIS-doc date. Month-granularity FEIS dates are KEPT (NOA dates are often month-level)."""
    feis = cands[
        cands["document_type_clean"].astype(str).str.upper().eq("FEIS")
        & cands["_parsed_date"].notna()
    ].copy()
    if feis.empty:
        return feis
    feis["ranking_score"] = (
        pd.to_numeric(feis.get("p_feis_cal"), errors="coerce")
        .fillna(pd.to_numeric(feis.get("p_final_eis"), errors="coerce"))
        .fillna(0.0)
    )
    return feis


def _select_eis_decision(
    decision_cands: pd.DataFrame, cands: pd.DataFrame
) -> tuple[Optional[pd.Series], Optional[str], bool, bool]:
    """Tiered EIS decision: ROD-first, FEIS-fallback.

    has_rod = the project has >=1 ROD-eligible candidate. ROD outranks FEIS BY CONSTRUCTION:
    FEIS candidates only enter the decision pool when has_rod is False. Within the chosen pool the
    ranker (`ranking_score`; FEIS ordered by p_feis_cal) picks the best. The `none` outcome (no ROD,
    no FEIS) is correct for projects with neither. Returns
    (best_row|None, flag|None, has_rod, is_feis_fallback).
    """
    if not EIS_TIERED_DECISION:
        # Reversible fallback: emit ONLY authoritative register RODs (the prior Step-4 behavior).
        is_register_rod = (
            decision_cands["candidate_source_type"].astype(str).eq("metadata")
            & decision_cands["candidate_role"].eq("clear_decision")
        )
        reg = decision_cands[is_register_rod] if not decision_cands.empty else decision_cands
        if not reg.empty:
            return _select_best_decision(reg), "eis_rod_register", True, False
        return None, None, False, False

    rod_pool = _eis_rod_pool(decision_cands)
    has_rod = not rod_pool.empty
    if has_rod:
        return _select_best_decision(rod_pool), "eis_rod", True, False
    feis = _eis_feis_pool(cands)
    if not feis.empty:
        return _select_best_decision(feis), "eis_feis_fallback", False, True
    return None, None, False, False


# Explicit Final-EIS publication / availability / filing language (EIS only). Used to promote a
# FEIS-document date from a cover-proxy to an explicit Final-EIS publication date.
EIS_FEIS_PUB_RE = re.compile(
    r"notice\s+of\s+availability"
    r"|\bnoa\b"
    r"|(?:final\s+(?:eis|environmental\s+impact\s+statement)|feis)\s+(?:was\s+)?"
    r"(?:filed|filing|publish\w*|releas\w*|issu\w*|made\s+available|available)"
    r"|(?:fil(?:ed|ing)|publish\w*|releas\w*|made\s+available)\s+(?:the\s+)?"
    r"(?:final\s+(?:eis|environmental\s+impact\s+statement)|feis)"
    r"|availability\s+of\s+the\s+final",
    re.IGNORECASE,
)

_EMPTY_FINAL_EIS = {
    "final_eis_date": None,
    "final_eis_date_granularity": "unknown",
    "final_eis_source_type": None,
    "final_eis_is_proxy": False,
    "final_eis_confidence": "missing",
    "final_eis_evidence_text": None,
    "final_eis_document_id": None,
    "final_eis_page_number": None,
    "_multiple": False,
}


def _select_eis_final_eis(cands: pd.DataFrame) -> dict:
    """EIS-only (Phase C, solution 2): pick the Final-EIS publication/availability date from
    EXISTING candidates in FEIS-typed documents (Phase A: 84.7% of FEIS-no-ROD projects already
    have one). Deterministic, no ranker. `document_type_clean == 'FEIS'` is the EIS-specific
    filter (Final EA is `document_type_clean == 'EA'`). Tier: explicit FEIS
    publication/filing/availability/NOA language (clear) > cover-date proxy. Earliest availability
    wins; `_multiple` flags materially conflicting dates within the chosen tier.

    This is written to `final_eis_date` ONLY — never to `decision_date` (08 derives the endpoint
    as ROD-else-FEIS separately) and it never changes `timeline_status`.
    """
    feis = cands[
        cands["document_type_clean"].astype(str).str.upper().eq("FEIS")
        & cands["_parsed_date"].notna()
    ].copy()
    if feis.empty:
        return dict(_EMPTY_FINAL_EIS)
    explicit = feis[feis["context_text"].fillna("").map(lambda t: bool(EIS_FEIS_PUB_RE.search(str(t))))]
    is_proxy = explicit.empty
    pool = feis if is_proxy else explicit
    day = pool[pool["date_granularity"] == "day"]
    if not day.empty:
        pool = day
    multiple = pool["_parsed_date"].nunique() > 1
    pick = pool.sort_values("_parsed_date").iloc[0]  # earliest availability
    return {
        "final_eis_date": pick["_parsed_date"].isoformat(),
        "final_eis_date_granularity": pick.get("date_granularity", "day"),
        "final_eis_source_type": pick.get("candidate_source_type", "document_text"),
        "final_eis_is_proxy": bool(is_proxy),
        "final_eis_confidence": "medium" if is_proxy else "high",
        "final_eis_evidence_text": str(pick.get("context_text", ""))[:300],
        "final_eis_document_id": pick.get("document_id"),
        "final_eis_page_number": str(pick.get("page_number")) if pick.get("page_number") is not None else None,
        "_multiple": bool(multiple),
    }


def _select_best_decision(df: pd.DataFrame) -> pd.Series:
    """Pick the best decision candidate, PREFERRING day-granularity over coarser dates (a
    signature / decision-record day beats a document cover month) and breaking ties by
    ranking_score. Falls back to the full set when no day-granularity candidate exists.
    (When the learned ranker is on, ranking_score holds the learned score for eligible rows.)"""
    pool = df
    day = df[df["date_granularity"] == "day"]
    if not day.empty:
        pool = day
    return pool.loc[pool["ranking_score"].idxmax()]


def _select_earliest_initiation(df: pd.DataFrame) -> pd.Series:
    """Pick the EARLIEST initiation candidate among those scoring within
    INIT_EARLIEST_SCORE_MARGIN of the top score (initiation = first qualifying start signal);
    tie-break on higher ranking_score."""
    top = df["ranking_score"].max()
    pool = df[df["ranking_score"] >= top - INIT_EARLIEST_SCORE_MARGIN].copy()
    pool = pool.sort_values(["_parsed_date", "ranking_score"], ascending=[True, False])
    return pool.iloc[0]


EA_FINAL_DOC_TYPES = {"EA", "FONSI", "ROD"}  # Final-EA/FONSI/ROD; excludes DEA (draft)
# Event-binding for the no-FONSI month proxy: the month must read like a Final-EA / FONSI / Decision
# issuance, NOT a citation, construction schedule, scoping note, or programmatic reference. `03`
# labels every month in a final doc as a proxy, so the selector must do the binding.
EA_MONTH_ISSUANCE_RE = re.compile(
    r"finding\s+of\s+no\s+significant\s+impact|\bfonsi\b|decision\s+(?:record|notice|memo)|"
    r"environmental\s+assessment|categorical\s+exclusion|\bdetermination\b",
    re.IGNORECASE,
)
EA_MONTH_NEG_RE = re.compile(
    r"printing\s+office|\bpress\b|construction|anticipat|operation|conducted|"
    r"scoping|review(?:ed|\s+was)|received|accessed|https?:|\d+\s*cfr|\bfr\b\s*\d|prepared\s+by|"
    r"comment\s+period|standards\s+and\s+guidelines|land\s+health|general\s+plan|"
    r"literature|report\s+to|\bet\s+al\b|\bvol\.|\bpp?\.",
    re.IGNORECASE,
)
# Hard negatives for the strong-cue document day tier (Phase C). role_confidence_score==5.0 already
# means CLEAR_DECISION_STRONG matched, but guard the known leaks: preparer dates, NOA/availability,
# comment/scoping periods, citations.
EA_STRONG_NEG_RE = re.compile(
    r"prepared\s+by|preparer|\bnoa\b|made\s+available|availability|comment\s+period|"
    r"scoping|\d+\s*cfr|\bfr\b\s*\d|accessed|https?:|\bet\s+al\b",
    re.IGNORECASE,
)


def _select_ea_decision(
    decision_cands: pd.DataFrame, cands: pd.DataFrame, has_fonsi: bool
) -> tuple[Optional[pd.Series], Optional[str]]:
    """EA decision selection (ea_audit.md Phase B).

    Tier order: existing cascade (clear>0 -> proxy>-2 -> body>-2), returned UNCHANGED so
    cascade-resolved EA projects stay byte-identical; then a register gap-fill; then, as a last
    resort, a no-FONSI Final-EA month proxy.

      - Register gap-fill: an authoritative BLM/DOE Tier A *day* register date, eligible regardless
        of the learned-score gate that currently drops it (the documented selection bug; §5.1).
      - No-FONSI month proxy: when the project has NO FONSI document and no day decision was found,
        a month-granularity Final-EA/ROD issuance date stands in as a flagged proxy (midpoint
        imputation later resolves it to the 15th; granularity stays "month" so no exact duration).
        Months are read from the FULL `cands` because the upstream month-suppression strips them
        from `decision_cands` (intended for the normal pool; this tier is a gated exception).

    Returns (best_row|None, reason|None); `reason` is set only for the register / month tiers so
    cascade-resolved rows gain no new flag.
    """
    clear = decision_cands[
        (decision_cands["candidate_role"] == "clear_decision") & (decision_cands["ranking_score"] > 0)
    ]
    if not clear.empty:
        return _select_best_decision(clear), None
    proxy = decision_cands[
        (decision_cands["candidate_role"] == "proxy_decision") & (decision_cands["ranking_score"] > -2)
    ]
    if not proxy.empty:
        return _select_best_decision(proxy), None
    body = decision_cands[
        (decision_cands["candidate_role"] == "body_text") & (decision_cands["ranking_score"] > -2)
    ]
    if not body.empty:
        return _select_best_decision(body), None
    # Tier EA-1: authoritative BLM/DOE day register date — eligible regardless of learned score.
    reg = decision_cands[
        decision_cands["source_tier"].astype(str).eq("metadata")
        & decision_cands["retrieval_tier"].astype(str).eq("tier_a")
        & decision_cands["candidate_role"].eq("clear_decision")
        & decision_cands["date_granularity"].eq("day")
        & decision_cands["_parsed_date"].notna()
    ]
    if not reg.empty:
        today = date.today()
        reg = reg[reg["_parsed_date"].map(lambda d: pd.notna(d) and d <= today)]
        if not reg.empty:
            return _select_best_decision(reg), "ea_decision_register"
    # Tier EA-2 (Phase C): strong-cue document day date — a real FONSI / Decision-Record /
    # Field-Manager signature (role_confidence_score == 5.0 means CLEAR_DECISION_STRONG matched),
    # eligible REGARDLESS of the learned-score gate. This is what makes the full-read pay off: the
    # newly-surfaced signature dates would otherwise be re-dropped by the ranker gate. The ranker
    # still ORDERS within this tier (via _select_best_decision -> ranking_score); it does not gate.
    today = date.today()
    strong = decision_cands[
        decision_cands["candidate_role"].eq("clear_decision")
        & decision_cands["date_granularity"].eq("day")
        & (pd.to_numeric(decision_cands["role_confidence_score"], errors="coerce") >= 5.0)
        & decision_cands["_parsed_date"].notna()
    ]
    if not strong.empty:
        sctx = strong["context_text"].fillna("")
        strong = strong[
            strong["_parsed_date"].map(lambda d: pd.notna(d) and d <= today)
            & ~sctx.str.contains(EA_STRONG_NEG_RE)
        ]
        if not strong.empty:
            return _select_best_decision(strong), "ea_decision_strong_text"
    # Last resort: no-FONSI Final-EA month proxy (read from full `cands`; see docstring). Event-bound:
    # the month context must read like an FEA/FONSI/Decision issuance and carry no citation/
    # construction/scoping/programmatic hard-negative cue.
    if not has_fonsi:
        today = date.today()
        ctx = cands["context_text"].fillna("")
        month = cands[
            cands["candidate_role"].isin(["clear_decision", "proxy_decision"])
            & cands["date_granularity"].eq("month")
            & cands["document_type_clean"].astype(str).str.upper().isin(EA_FINAL_DOC_TYPES)
            & cands["_parsed_date"].notna()
            & ctx.str.contains(EA_MONTH_ISSUANCE_RE)
            & ~ctx.str.contains(EA_MONTH_NEG_RE)
        ]
        if not month.empty:
            month = month[month["_parsed_date"].map(lambda d: pd.notna(d) and d <= today)]
            if not month.empty:
                return _select_best_decision(month), "ea_decision_fea_month"
    return None, None


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
    # Selection is recomputed from scratch on every run. Clear stale flags left by an earlier
    # selection pass before marking the current winners.
    cands["selected_for_initiation"] = False
    cands["selected_for_decision"] = False
    cands["_parsed_date"] = pd.to_datetime(cands["parsed_date"], errors="coerce").dt.date

    # Cross-candidate agreement: how many candidates in THIS project resolve to the same date
    # (corroboration signal consumed by candidate_score_components via _agreement_count).
    _date_counts = cands["_parsed_date"].value_counts(dropna=True)
    cands["_agreement_count"] = (
        cands["_parsed_date"].map(_date_counts).fillna(1).astype(int)
    )

    # --- Historical gap flagging (before either pass) ---
    valid_dates = cands["_parsed_date"].dropna().tolist()
    historical_gap_set = _apply_historical_gap_rule(valid_dates, process_type)
    cands["historical_gap_candidate"] = cands["_parsed_date"].apply(
        lambda d: d in historical_gap_set if pd.notna(d) else False
    )

    # --- Pass 1: Score and select decision ---
    # body_text = date in a decision doc with no role cue. Included in the pool so it can
    # serve as a last-resort decision proxy (selected only if no clear/proxy decision
    # exists). Once the classifier runs, body_text candidates carry a model score and this
    # fallback is superseded.
    decision_cands = cands[
        cands["candidate_role"].isin(["clear_decision", "proxy_decision", "body_text"])
    ].copy()
    decision_cands["ranking_score"] = [
        _compute_candidate_score(r, "decision", None, index_map)
        for r in decision_cands.to_dict("records")
    ]
    # learned ranker (05b): when on, its score REPLACES the heuristic ranking_score. The lambdarank
    # score is higher for clearer cases, so the eligibility gates below (`> 0` / `> -2`) double as a
    # confidence threshold — a project whose candidates all score low yields no decision (correct when
    # the project genuinely has none). Set D4_USE_LEARNED_RANKER=0 to fall back to the heuristic.
    if USE_LEARNED_RANKER and "learned_decision_score" in cands.columns:
        decision_cands["ranking_score"] = pd.to_numeric(
            decision_cands["learned_decision_score"], errors="coerce"
        ).fillna(decision_cands["ranking_score"])
    cands.loc[decision_cands.index, "ranking_score"] = decision_cands["ranking_score"]

    # Rule: a bare month-granularity date can be the DECISION only for CE. For EA/EIS drop it
    # from the pool so a cover month never locks in as the decision — the project routes to 06
    # to find a precise ROD/FONSI date instead.
    month_decision_suppressed = False
    if process_type not in MONTH_DECISION_PROCESSES:
        is_month = decision_cands["date_granularity"].eq("month")
        if is_month.any():
            month_decision_suppressed = True
            decision_cands = decision_cands[~is_month]

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

    eis_rod_flag: str | None = None
    has_rod = False
    decision_is_feis_fallback = False
    ea_decision_reason: str | None = None
    if process_type == "EIS":
        # Tiered: ROD-first (ranker orders ROD-eligible), FEIS-fallback when has_rod is False.
        best_decision, eis_rod_flag, has_rod, decision_is_feis_fallback = _select_eis_decision(
            decision_cands, cands
        )
        # clear_dec is referenced later for the multiple_high_score flag; keep it meaningful.
        clear_dec = decision_cands[decision_cands["candidate_role"] == "clear_decision"]
    elif process_type == "EA":
        # EA Phase B: existing cascade first (byte-identical when it resolves), then register
        # gap-fill, then a no-FONSI Final-EA month proxy (last resort). has_fonsi comes from the
        # document index (all docs), not candidates (a FONSI can exist but not be retrieved).
        pid = cands["project_id"].iloc[0] if not cands.empty else None
        has_fonsi = bool(index_map.get(pid, {}).get("has_fonsi", False))
        best_decision, ea_decision_reason = _select_ea_decision(decision_cands, cands, has_fonsi)
        clear_dec = decision_cands[
            (decision_cands["candidate_role"] == "clear_decision") &
            (decision_cands["ranking_score"] > 0)
        ]
    else:
        # CE / EA unchanged: clear_decision in pass 1 (proxies only if no clear found).
        clear_dec = decision_cands[
            (decision_cands["candidate_role"] == "clear_decision") &
            (decision_cands["ranking_score"] > 0)
        ]
        if not clear_dec.empty:
            best_decision = _select_best_decision(clear_dec)
        else:
            proxy_dec = decision_cands[
                (decision_cands["candidate_role"] == "proxy_decision") &
                (decision_cands["ranking_score"] > -2)
            ]
            if not proxy_dec.empty:
                best_decision = _select_best_decision(proxy_dec)
                decision_is_proxy = True
            else:
                # Last resort: a date in a decision doc with no role cue. Always a proxy.
                body_dec = decision_cands[
                    (decision_cands["candidate_role"] == "body_text") &
                    (decision_cands["ranking_score"] > -2)
                ]
                if not body_dec.empty:
                    best_decision = _select_best_decision(body_dec)
                    decision_is_proxy = True

    if best_decision is not None:
        try:
            decision_date_obj = best_decision["_parsed_date"]
            if pd.notna(decision_date_obj):
                decision_date_str = decision_date_obj.isoformat()
                decision_granularity = best_decision.get("date_granularity", "day")
                decision_source_type = best_decision.get("candidate_source_type", "document_text")
                decision_confidence = best_decision.get("role_confidence", "medium")
                decision_is_proxy = best_decision.get("candidate_role") in ("proxy_decision", "body_text")
                # A no-FONSI Final-EA month is inherently a coarse proxy regardless of its role cue.
                if ea_decision_reason == "ea_decision_fea_month":
                    decision_is_proxy = True
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
    # learned ranker: when on, its score replaces the heuristic ranking_score (gate doubles as a
    # confidence threshold). D4_USE_LEARNED_RANKER=0 falls back to the heuristic.
    if USE_LEARNED_RANKER and "learned_init_score" in cands.columns:
        initiation_cands["ranking_score"] = pd.to_numeric(
            initiation_cands["learned_init_score"], errors="coerce"
        ).fillna(initiation_cands["ranking_score"])
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
    initiation_earliest_used = False  # set when earliest-wins disambiguated >1 candidate

    # Eligibility pool. CE: original ranker-score gate (untouched). EA/EIS (Phase 4): calibrated-prob /
    # authoritative eligibility unioned with the ranker gate — the ranker score is not an existence
    # gate (see _calibrated_init_eligible).
    _calib = process_type in CALIBRATED_INIT_PROCESSES
    _lb_days = MAX_INIT_LOOKBACK_DAYS.get(process_type, 5475)
    _lb_years = _lb_days // 365
    _clear_mask = initiation_cands["candidate_role"] == "clear_initiation"
    if _calib:
        clear_init = initiation_cands[_clear_mask & _calibrated_init_eligible(initiation_cands, process_type, "clear")]
    else:
        clear_init = initiation_cands[_clear_mask & (initiation_cands["ranking_score"] > 0)]
    # Apply chronology filter: initiation must precede decision.
    # When the decision date is year-granularity (nepa_case_year, resolves to YYYY-07-01),
    # use a year-level comparison to avoid discarding real initiations that fall after
    # July 1 in the same year (e.g. BLM register start date 2021-07-23 vs proxy 2021-07-01).
    if selected_decision_date is not None:
        if decision_granularity == "year":
            dec_year = selected_decision_date.year
            if _calib:
                # EA/EIS: also drop init candidates implausibly far before the decision.
                clear_init = clear_init[
                    clear_init["_parsed_date"].apply(
                        lambda d: pd.notna(d) and (dec_year - _lb_years) <= d.year <= dec_year
                    )
                ]
            else:
                clear_init = clear_init[
                    clear_init["_parsed_date"].apply(
                        lambda d: pd.notna(d) and d.year <= dec_year
                    )
                ]
        else:
            if _calib:
                clear_init = clear_init[
                    clear_init["_parsed_date"].apply(
                        lambda d: pd.notna(d) and d < selected_decision_date
                        and (selected_decision_date - d).days <= _lb_days
                    )
                ]
            else:
                clear_init = clear_init[
                    clear_init["_parsed_date"].apply(
                        lambda d: pd.notna(d) and d < selected_decision_date
                    )
                ]

    if not clear_init.empty:
        best_initiation = _select_earliest_initiation(clear_init)
        initiation_earliest_used = len(clear_init) > 1
    else:
        # Proxy fallback (sensitivity only)
        _proxy_mask = initiation_cands["candidate_role"] == "proxy_initiation"
        if _calib:
            proxy_init = initiation_cands[_proxy_mask & _calibrated_init_eligible(initiation_cands, process_type, "proxy")]
        else:
            proxy_init = initiation_cands[
                _proxy_mask & (initiation_cands["ranking_score"] > -2)
            ]
        if selected_decision_date is not None:
            if decision_granularity == "year":
                dec_year = selected_decision_date.year
                if _calib:
                    proxy_init = proxy_init[
                        proxy_init["_parsed_date"].apply(
                            lambda d: pd.notna(d) and (dec_year - _lb_years) <= d.year <= dec_year
                        )
                    ]
                else:
                    proxy_init = proxy_init[
                        proxy_init["_parsed_date"].apply(
                            lambda d: pd.notna(d) and d.year <= dec_year
                        )
                    ]
            else:
                if _calib:
                    proxy_init = proxy_init[
                        proxy_init["_parsed_date"].apply(
                            lambda d: pd.notna(d) and d < selected_decision_date
                            and (selected_decision_date - d).days <= _lb_days
                        )
                    ]
                else:
                    proxy_init = proxy_init[
                        proxy_init["_parsed_date"].apply(
                            lambda d: pd.notna(d) and d < selected_decision_date
                        )
                    ]
        if not proxy_init.empty:
            best_initiation = _select_earliest_initiation(proxy_init)
            initiation_is_proxy = True
            initiation_earliest_used = len(proxy_init) > 1

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

    # --- DOE CX "Date Determined" initiation recovery (deterministic rule, CE) ---
    # A DOE CX form often has a "Date Determined: <d1>" plus a later signature "<d2>".
    # When both exist (d1 < d2) and no other initiation was found, set decision = the
    # later signature and recover the earlier Date Determined as a proxy initiation
    # (a CE processing-start bracket). Accepted deterministically (it precedes the
    # decision) -> no classifier, no LLM adjudication.
    # Per project decision (2026-06-02): this RECOVERS EVEN WHEN a register determination
    # date coincides with the Date Determined — the later signature becomes the decision.
    # Guard (counter-case): a lone Date Determined with no later signature stays the
    # decision (the block below requires a later non-Date-Determined decision date).
    date_determined_init_used = False
    if process_type == "CE" and initiation_date_str is None:
        dd_flag = cands["positive_cue_flags"].fillna("").str.contains("date_determined")
        dd_cands = cands[dd_flag & cands["_parsed_date"].notna()]
        if not dd_cands.empty:
            dd_date = dd_cands["_parsed_date"].min()  # the Date Determined (earliest if several)
            sig = cands[
                cands["candidate_role"].isin(["clear_decision", "proxy_decision"])
                & ~dd_flag & cands["_parsed_date"].notna()
                & (cands["_parsed_date"] > dd_date)
            ]
            if not sig.empty:
                s = sig.sort_values("_parsed_date").iloc[-1]   # latest signature = decision
                dd = dd_cands[dd_cands["_parsed_date"] == dd_date].iloc[0]
                decision_date_str = s["_parsed_date"].isoformat()
                decision_granularity = s.get("date_granularity", "day")
                decision_source_type = s.get("candidate_source_type", "document_text")
                decision_confidence = s.get("role_confidence", "high")
                decision_is_proxy = s.get("candidate_role") == "proxy_decision"
                decision_evidence_text = str(s.get("context_text", ""))[:300]
                decision_document_id = s.get("document_id")
                decision_page_number = s.get("page_number")
                selected_decision_id = s.get("candidate_id")
                initiation_date_str = dd["_parsed_date"].isoformat()
                initiation_granularity = dd.get("date_granularity", "day")
                initiation_source_type = dd.get("candidate_source_type", "document_text")
                initiation_confidence = "medium"
                initiation_is_proxy = True
                initiation_evidence_text = str(dd.get("context_text", ""))[:300]
                initiation_document_id = dd.get("document_id")
                initiation_page_number = dd.get("page_number")
                selected_initiation_id = dd.get("candidate_id")
                date_determined_init_used = True

    # --- CE inferred-application initiation proxy (mirrors Phase 1 bert_inferred_application_date) ---
    # Phase 1's CE initiation = application date if found, else the EARLIEST dated mention. Phase 2
    # extracts CE decisions well (~98% of Phase 1) but misses CE initiation (~45%) because it has no
    # equivalent inference. When a CE project has a decision but NO initiation, adopt the earliest
    # candidate date strictly before the decision as an inferred-application proxy. Flagged is_proxy +
    # ce_inferred_application; never treated as a clear date. (full_recover.md Fix 1; CE is a build
    # issue where less-conservative is accepted, 2026-06-15. TODO: audit a sample tomorrow.)
    ce_inferred_init_used = False
    if process_type == "CE" and initiation_date_str is None and decision_date_str is not None:
        try:
            _dec_dt = pd.Timestamp(decision_date_str).date()
        except Exception:
            _dec_dt = None
        if _dec_dt is not None:
            # CE reviews are short; an "earliest date" many years before the decision is almost
            # certainly a stray citation/reference, not an application. Cap the inferred lookback at
            # 5y so the proxy doesn't pollute headline durations. (tomorrow: tighten via cue filtering)
            _MAX_CE_INFERRED_LOOKBACK_DAYS = 1825
            # Exclude regulatory/permit/compliance reference dates — the proxy was occasionally
            # grabbing e.g. "must comply with the … Permit (issued December 2010)" or a CFR citation
            # instead of an application. Skip those so it falls to the next-earliest legitimate date.
            _CE_PROXY_NEG_RE = re.compile(
                r"permit\s+(?:was\s+)?issued|must\s+comply\s+with|hazardous\s+waste\s+permit"
                r"|\d+\s+cfr\b|\bcfr\s+\d|in\s+accordance\s+with", re.IGNORECASE)
            _neg = cands["context_text"].fillna("").str.contains(_CE_PROXY_NEG_RE)
            # Vectorized (no row-wise apply): earliest parsed date in [decision-5y, decision).
            _pd_dates = pd.to_datetime(cands["_parsed_date"], errors="coerce")
            _dec_ts = pd.Timestamp(_dec_dt)
            _mask = (_pd_dates.notna()
                     & (_pd_dates < _dec_ts)
                     & (_pd_dates >= _dec_ts - pd.Timedelta(days=_MAX_CE_INFERRED_LOOKBACK_DAYS))
                     & ~_neg)
            if _mask.any():
                e = cands.loc[_pd_dates[_mask].idxmin()]   # earliest dated mention
                initiation_date_str = e["_parsed_date"].isoformat()
                initiation_granularity = e.get("date_granularity", "day")
                initiation_source_type = e.get("candidate_source_type", "document_text")
                initiation_confidence = "low"
                initiation_is_proxy = True
                initiation_evidence_text = str(e.get("context_text", ""))[:300]
                initiation_document_id = e.get("document_id")
                initiation_page_number = e.get("page_number")
                selected_initiation_id = e.get("candidate_id")
                ce_inferred_init_used = True

    # --- Mark selected candidates ---
    if selected_decision_id:
        cands.loc[cands["candidate_id"] == selected_decision_id, "selected_for_decision"] = True
    if selected_initiation_id:
        cands.loc[cands["candidate_id"] == selected_initiation_id, "selected_for_initiation"] = True

    # --- Determine timeline_status and flags ---
    has_init = initiation_date_str is not None
    has_dec = decision_date_str is not None

    flags: list[str] = []
    if date_determined_init_used:
        flags.append("date_determined_initiation")
    if ce_inferred_init_used:
        flags.append("ce_inferred_application")
    # Selection-disambiguation rules (2026-06-04): surface when they fired so 06 / review can see it.
    if month_decision_suppressed and not has_dec:
        # A non-CE month-only decision was dropped; 06 should hunt for a precise ROD/FONSI date.
        flags.append("month_decision_suppressed_non_ce")
    if initiation_earliest_used:
        flags.append("initiation_earliest_selected")
    if eis_rod_flag:
        flags.append(eis_rod_flag)
    if ea_decision_reason:
        flags.append(ea_decision_reason)
    timeline_status = "missing_both"

    if has_init and has_dec:
        init_d = date.fromisoformat(initiation_date_str)
        dec_d = date.fromisoformat(decision_date_str)
        if init_d > dec_d:
            # Year-granularity proxy decisions (nepa_case_year → YYYY-07-01) mechanically
            # precede BLM Register initiation dates from the same year. The July-1
            # placeholder carries no real ordering information, so discard the proxy
            # and fall through to missing_decision rather than flagging invalid_order.
            if decision_granularity == "year" and init_d.year <= dec_d.year:
                has_dec = False
                decision_date_str = None
                decision_granularity = "unknown"
                decision_is_proxy = False
                timeline_status = "missing_decision"
                flags.append("nepa_case_year_proxy_discarded")
                flags.append("missing_decision")
            else:
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

    # Year-granularity flags: year-proxy dates are kept for cohort analysis but
    # must never be used for duration calculations or treated as precise dates.
    # Flag them explicitly so downstream analysis can filter or cross-check.
    if decision_granularity == "year":
        flags.append("year_proxy_decision")
    if initiation_granularity == "year":
        flags.append("year_proxy_initiation")

    # Cross-check: if we have a year-proxy decision and a real initiation date,
    # flag if they imply an implausible order (decision year < initiation year).
    if decision_granularity == "year" and has_init and initiation_date_str:
        proxy_year = int(decision_date_str[:4])
        init_year = int(initiation_date_str[:4])
        if proxy_year < init_year:
            flags.append("proxy_year_before_initiation")

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

    # Duration plausibility (day-granularity pairs only): a sanity check by process type. Out-of-band
    # durations are FLAGGED for review/06, never discarded — the selected pair may still be right.
    if duration_days is not None:
        hi = DURATION_PLAUSIBLE_MAX_DAYS.get(process_type)
        lo = DURATION_PLAUSIBLE_MIN_DAYS.get(process_type)
        if hi is not None and duration_days > hi:
            flags.append("implausible_duration_long")
        elif lo is not None and duration_days < lo:
            flags.append("implausible_duration_short")

    # Check for multiple high-score candidates (tie situation)
    if not clear_dec.empty and len(clear_dec[clear_dec["ranking_score"] >= (clear_dec["ranking_score"].max() - 1)]) > 1:
        flags.append("multiple_high_score_candidates")

    if decision_confidence == "low" or initiation_confidence == "low":
        flags.append("low_confidence_selection")

    # --- Phase C (solution 2): Final-EIS endpoint (EIS only; separate from decision_date) ---
    final_eis = dict(_EMPTY_FINAL_EIS)
    if process_type == "EIS" and EIS_FINAL_EIS_ENABLED:
        final_eis = _select_eis_final_eis(cands)
        if final_eis.get("final_eis_date"):
            if final_eis.get("_multiple"):
                flags.append("final_eis_multiple_dates")
            # C6 chronology: a ROD should follow the FEIS. Flag conflicts (granularity-aware);
            # never auto-replace the ROD (Phase A: 13/90 ROD-before-FEIS — too frequent to reject).
            if decision_date_str:
                rod_d = date.fromisoformat(decision_date_str)
                feis_d = date.fromisoformat(final_eis["final_eis_date"])
                feg = final_eis["final_eis_date_granularity"]
                if decision_granularity == "year" or feg == "year":
                    conflict = rod_d.year < feis_d.year
                elif decision_granularity == "month" or feg == "month":
                    conflict = (rod_d.year, rod_d.month) < (feis_d.year, feis_d.month)
                else:
                    conflict = rod_d < feis_d
                if conflict:
                    flags.append("rod_feis_conflict")

    # --- Routing gate for 06 (LLM adjudication): confidence of the selected decision ---
    decision_confidence_cal = 0.0
    if best_decision is not None:
        _cal_key = "p_feis_cal" if decision_is_feis_fallback else "p_dec_cal"
        try:
            decision_confidence_cal = float(best_decision.get(_cal_key) or 0.0)
        except (TypeError, ValueError):
            decision_confidence_cal = 0.0
    route_to_llm = bool(
        (has_dec and decision_confidence_cal < LLM_ROUTE_THRESHOLD)   # ambiguous deterministic pick
        or (not has_dec and len(decision_cands) > 0)                  # no pick but candidates exist
        or ("multiple_high_score_candidates" in flags)                # competing candidates
    )

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
        "has_rod": bool(has_rod),
        "decision_is_feis_fallback": bool(decision_is_feis_fallback),
        "decision_confidence_cal": round(decision_confidence_cal, 4),
        "route_to_llm": bool(route_to_llm),
        "final_eis_date": final_eis["final_eis_date"],
        "final_eis_date_granularity": final_eis["final_eis_date_granularity"],
        "final_eis_source_type": final_eis["final_eis_source_type"],
        "final_eis_is_proxy": final_eis["final_eis_is_proxy"],
        "final_eis_confidence": final_eis["final_eis_confidence"],
        "final_eis_evidence_text": final_eis["final_eis_evidence_text"],
        "final_eis_document_id": final_eis["final_eis_document_id"],
        "final_eis_page_number": final_eis["final_eis_page_number"],
        "duration_days": duration_days,
        "timeline_status": timeline_status,
        "timeline_flags": "|".join(flags) if flags else "",
        "midpoint_imputed": False,  # set to True by apply_month_midpoint_imputation after corrections
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
        "has_rod": False,
        "decision_is_feis_fallback": False,
        "decision_confidence_cal": 0.0,
        "route_to_llm": False,
        "final_eis_date": None,
        "final_eis_date_granularity": "unknown",
        "final_eis_source_type": None,
        "final_eis_is_proxy": False,
        "final_eis_confidence": "missing",
        "final_eis_evidence_text": None,
        "final_eis_document_id": None,
        "final_eis_page_number": None,
        "duration_days": None,
        "timeline_status": "missing_both",
        "timeline_flags": "missing_initiation|missing_decision",
        "midpoint_imputed": False,
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


def apply_deis_only_flags(dates_df: pd.DataFrame, index_path: Path) -> pd.DataFrame:
    """
    Flag EIS projects that have a DEIS in the index but no FEIS or ROD.

    These projects have missing_decision not because of a regex failure but because
    no decision document exists in NEPATEC. Flagging them lets the report distinguish
    "structurally unresolvable" from "solvable with better retrieval or LLM recovery".
    """
    if not index_path.exists():
        return dates_df

    con = duckdb.connect()

    # For each EIS project: does it have a final/ROD doc? A draft doc?
    doc_profile = con.execute(f"""
        SELECT project_id,
            MAX(CASE WHEN document_type_category IN ('final','decision')
                          OR LOWER(document_type_clean) IN ('feis','final eis','rod','record of decision')
                     THEN 1 ELSE 0 END) AS has_final_or_rod,
            MAX(CASE WHEN document_type_category = 'draft'
                          OR LOWER(document_type_clean) IN ('deis','draft eis')
                     THEN 1 ELSE 0 END) AS has_deis
        FROM read_parquet('{index_path}')
        WHERE process_type = 'EIS'
        GROUP BY project_id
    """).df()

    deis_only_ids = set(
        doc_profile.loc[
            (doc_profile["has_final_or_rod"] == 0) & (doc_profile["has_deis"] == 1),
            "project_id",
        ]
    )

    mask = (
        dates_df["process_type"] == "EIS"
    ) & (
        dates_df["decision_date"].isna()
    ) & (
        dates_df["project_id"].isin(deis_only_ids)
    )

    def _add_flag(flags: str, new_flag: str) -> str:
        parts = [f for f in flags.split("|") if f]
        if new_flag not in parts:
            parts.append(new_flag)
        return "|".join(parts)

    dates_df.loc[mask, "timeline_flags"] = dates_df.loc[mask, "timeline_flags"].apply(
        lambda f: _add_flag(str(f) if pd.notna(f) else "", "deis_only")
    )
    n = mask.sum()
    if n > 0:
        print(f"  deis_only flag applied to {n:,} EIS projects (DEIS in index, no FEIS or ROD).")
    return dates_df


def reconcile_eis_universe(
    dates_df: pd.DataFrame,
    index_path: Path,
    project_ids: set[str] | None,
) -> pd.DataFrame:
    """Phase B — EIS universe completeness.

    The selection loop only visits projects that have candidates, so EIS projects with no
    surviving candidates (Phase A: 664 — 223 with no packets, 441 with packets but no
    candidates) never get an output row and silently vanish. Append a `missing_both` stub
    for every EIS project in the index that is absent from `dates_df`.

    EIS-only by design: recovering CE/EA dropped rows is deferred (Phase D) because it would
    change CE/EA output. Runs BEFORE manual corrections / midpoint / deis_only flags so the
    stubs still receive them. (Phase A confirmed none of the 664 carry a register ROD/NOI
    date, so the stubs are genuinely `missing_both`; the ordering is kept as a safeguard.)
    """
    if not index_path.exists():
        return dates_df
    idx = pd.read_parquet(index_path, columns=["project_id", "process_type"])
    eis_universe = set(idx.loc[idx["process_type"] == "EIS", "project_id"].unique())
    if project_ids is not None:
        eis_universe &= project_ids
    missing = sorted(eis_universe - set(dates_df["project_id"]))
    if not missing:
        return dates_df
    stubs = []
    for pid in missing:
        row = _empty_project_result("EIS")
        row["project_id"] = pid
        stubs.append(row)
    print(f"  EIS universe reconciliation: added {len(missing):,} missing EIS projects as missing_both.")
    return pd.concat([dates_df, pd.DataFrame(stubs)], ignore_index=True)


def apply_month_midpoint_imputation(dates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace month-granularity dates with the 15th of that month and flag them.

    Called AFTER manual corrections so imputation only touches dates that have
    survived all selection and correction passes — i.e., month-year is genuinely
    the best available evidence.  Script 06 (API adjudication) can override these
    by writing a day-level date and setting midpoint_imputed = False.

    Applies to both decision_date and initiation_date when granularity == "month".
    Day-level and year-level dates are never touched.
    """
    if "midpoint_imputed" not in dates_df.columns:
        dates_df["midpoint_imputed"] = False

    imputed_mask = pd.Series(False, index=dates_df.index)

    for role in ("decision", "initiation"):
        gran_col = f"{role}_date_granularity"
        date_col = f"{role}_date"
        if gran_col not in dates_df.columns or date_col not in dates_df.columns:
            continue
        month_mask = (
            dates_df[gran_col] == "month"
        ) & dates_df[date_col].notna()
        if not month_mask.any():
            continue
        dates_df.loc[month_mask, date_col] = dates_df.loc[month_mask, date_col].apply(
            lambda d: d[:8] + "15" if isinstance(d, str) and len(d) >= 8 else d
        )
        imputed_mask |= month_mask

    dates_df.loc[imputed_mask, "midpoint_imputed"] = True
    n = imputed_mask.sum()
    if n > 0:
        print(f"  Midpoint imputation applied to {n:,} projects (month-granularity dates → day 15).")
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
        raise FileNotFoundError(f"Candidates not found: {candidates_path}\nRun 03_extract_candidates.py first.")

    project_ids: set[str] | None = None
    if args.sample_ids:
        with open(args.sample_ids) as f:
            project_ids = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(project_ids)} sample project IDs.")

    run_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidates: {candidates_path}")
    candidates_df = pd.read_parquet(candidates_path)

    # Guard: refuse to write subset data to the main TIMELINE_DIR output.
    ALL_PROCESS_TYPES = {"CE", "EA", "EIS"}
    candidates_process_types = set(candidates_df["process_type"].unique())
    if run_dir == TIMELINE_DIR and candidates_process_types != ALL_PROCESS_TYPES:
        raise SystemExit(
            f"[GUARD] Candidates file contains only {candidates_process_types}, not all process types.\n"
            f"Writing subset data to {dates_path} would overwrite the full-corpus dates.\n"
            f"Use --run-dir to isolate this run, or restore full-corpus candidates by re-running "
            f"scripts 02 and 03 without --process."
        )

    candidates_df = candidates_df[candidates_df["process_type"].isin(args.process)]
    if project_ids:
        candidates_df = candidates_df[candidates_df["project_id"].isin(project_ids)]
    print(f"  {len(candidates_df):,} candidates, {candidates_df['project_id'].nunique():,} projects")

    # Load index for document type scoring
    index_map: dict = {}
    if INDEX_PATH.exists():
        idx = pd.read_parquet(
            INDEX_PATH,
            columns=["project_id", "decision_doc_score", "initiation_doc_score", "document_type_clean"],
        )
        idx["_is_fonsi"] = idx["document_type_clean"].astype(str).str.upper().str.contains("FONSI")
        for pid, grp in idx.groupby("project_id"):
            index_map[pid] = {
                "decision_doc_score": grp["decision_doc_score"].max(),
                "initiation_doc_score": grp["initiation_doc_score"].max(),
                "has_fonsi": bool(grp["_is_fonsi"].any()),
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

    # Phase B: EIS universe completeness — add missing_both rows for EIS projects with no
    # candidates (else they vanish from the output). Before corrections/midpoint/deis flags.
    if "EIS" in args.process:
        dates_df = reconcile_eis_universe(dates_df, INDEX_PATH, project_ids)

    # Apply manual corrections
    if not corrections_df.empty:
        dates_df = apply_manual_corrections(dates_df, corrections_df)
        print(f"Applied manual corrections to {dates_df['timeline_flags'].str.contains('manual_override', na=False).sum()} projects.")

    # Apply month midpoint imputation — only after all corrections so this is truly last-resort
    print("Applying month midpoint imputation...")
    dates_df = apply_month_midpoint_imputation(dates_df)

    # Flag EIS projects that have DEIS but no FEIS/ROD — structurally unresolvable by regex
    print("Applying deis_only flags...")
    dates_df = apply_deis_only_flags(dates_df, INDEX_PATH)

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
