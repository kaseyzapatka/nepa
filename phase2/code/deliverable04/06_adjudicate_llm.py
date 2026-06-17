"""
API / LLM adjudication for unresolved timeline cases (D4).

Two allowed modes (plan §7):
  candidate_adjudication  — send compact candidate packets for projects with
                            missing or conflicting dates but existing candidates.
  document_recovery       — send top page/section text for projects where no
                            usable candidates were found.

Inputs:
    phase2/data/analysis/timeline/timeline_project_dates.parquet
    phase2/data/analysis/timeline/timeline_candidates.parquet
    phase2/data/analysis/timeline/timeline_context_packets.parquet

Outputs:
    phase2/data/analysis/timeline/timeline_api_adjudications.parquet
    phase2/data/analysis/timeline/timeline_project_dates.parquet    (updated)

Usage:
    python 06_adjudicate_llm.py --mode candidate_adjudication [--process CE EA EIS]
    python 06_adjudicate_llm.py --mode document_recovery --process EIS
    python 06_adjudicate_llm.py --dry-run --process EA --sample 10
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"

DATES_PATH = TIMELINE_DIR / "timeline_project_dates.parquet"
CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
PACKETS_PATH = TIMELINE_DIR / "timeline_context_packets.parquet"
ADJUDICATIONS_PATH = TIMELINE_DIR / "timeline_api_adjudications.parquet"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PROVIDER = "anthropic"
MAX_INPUT_TOKENS = 4096
MAX_CANDIDATES = 40
MAX_RECOVERY_PAGES = 10
RETRY_SLEEP = 2.0          # throttle between calls (raise/lower per API tier rate limit)
# Resilience: transient errors (rate-limit / overloaded / timeout) are retried with exponential
# backoff; out-of-credit / billing errors are fatal and trip a fail-fast after N in a row so a
# drained account stops cleanly (work saved) instead of erroring through the whole queue.
MAX_TRANSIENT_RETRIES = 5
BACKOFF_BASE_SEC = 2.0
SAVE_EVERY = 50            # incremental checkpoint cadence (so a kill never loses > this many)
BILLING_FAILFAST_N = 3     # consecutive credit/billing errors before stopping the run

# --- Classifier-driven routing (06 now consumes 04's confidence scores) ---------------
# A project is routed to the LLM when, on top of the regex-status triggers, its best
# candidate's classifier confidence is below this. Uses raw max(p_init, p_decision) today;
# swap to the calibrated probability once calibration lands (then the threshold is meaningful).
ROUTE_CONF_THRESHOLD = 0.70
# ...or when >= COMPETE_DECISION_MIN_N candidates both look like a decision (which signature
# wins is genuinely ambiguous -> worth an LLM look).
COMPETE_DECISION_PROB = 0.50
COMPETE_DECISION_MIN_N = 2
# Regex-authoritative candidates (role_confidence_score >= this) are treated as fully
# confident and never trigger low-confidence routing — 04 leaves them unscored by design.
AUTHORITATIVE_CONF = 5.0
# Packets sent per routed project, ranked by classifier score (then ranking_score).
# NOTE: set to 3 per request. At current init/decision F1 (~0.55) the true date can rank
# below 3, so this risks pruning it before the LLM sees it — revisit after calibration.
ROUTED_TOPK = 3


# ---------------------------------------------------------------------------
# Budget controls
# ---------------------------------------------------------------------------

def _cache_key(project_id: str, candidate_ids: list[str], model: str) -> str:
    payload = f"{project_id}|{'|'.join(sorted(candidate_ids))}|{model}"
    return hashlib.sha1(payload.encode()).hexdigest()[:24]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:20]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

CANDIDATE_ADJUDICATION_SYSTEM = """You are a NEPA timeline analyst. Your task is to identify the best
initiation date and decision date for a federal NEPA review project from a provided list of
date candidates extracted from project documents.

Rules:
- You MUST select dates only from the provided candidate list. Do NOT invent or estimate dates.
- If no suitable initiation candidate exists, return null for initiation_date.
- If no suitable decision candidate exists, return null for decision_date.
- For decision: prefer ROD/FONSI/CE determination/decision-record dates over generic final document dates.
- For initiation: prefer NOI/scoping/application-received dates over generic first-document dates.
- Prefer a full day-level date (YYYY-MM-DD) over a month-only (YYYY-MM) or year-only candidate; select a
  month-year candidate ONLY when no acceptable day-level candidate exists for that role.
- Do not select dates that appear to be from legal citations, historical references, or unrelated projects.

Return valid JSON only:
{"initiation_candidate_id": "<id or null>", "decision_candidate_id": "<id or null>", "reasoning": "<brief>"}"""

DOCUMENT_RECOVERY_SYSTEM = """You are a NEPA timeline analyst. Read the provided document text and
identify the initiation date (when the NEPA review began or the application was received) and the
decision date (when the agency issued its CE determination, FONSI, or ROD).

Rules:
- Cite the exact text passage and page number where you found each date.
- Return null if you cannot find evidence with reasonable confidence.
- Do not invent dates not present in the text.

Return valid JSON only:
{"initiation_date": "<YYYY-MM-DD or null>", "initiation_evidence": "<quote>", "initiation_page": "<page or null>",
 "decision_date": "<YYYY-MM-DD or null>", "decision_evidence": "<quote>", "decision_page": "<page or null>"}"""


def _build_candidate_prompt(
    project_id: str,
    project_title: str,
    process_type: str,
    agency: str,
    candidates: pd.DataFrame,
    current_status: str,
    current_flags: str,
) -> tuple[str, list[str]]:
    """Build the candidate adjudication prompt and return (prompt_text, candidate_ids)."""
    # Drop candidates the classifier predicted `neither`: only init/decision candidates are
    # worth the LLM's context. Fall back to the full set when nothing is predicted-positive
    # (e.g. an unscored pool, or a project whose every candidate scored neither — better to
    # send the best-available than an empty packet). Regex-authoritative candidates
    # (role_confidence_score >= AUTHORITATIVE_CONF) are kept even if 04 left them unscored.
    pool = candidates
    if "classifier_label" in candidates.columns:
        lbl = candidates["classifier_label"].fillna("").astype(str).str.strip().str.lower()
        rconf = pd.to_numeric(candidates.get("role_confidence_score"), errors="coerce").fillna(0.0)
        keep = lbl.isin(("initiation", "decision")) | (rconf >= AUTHORITATIVE_CONF)
        if keep.any():
            pool = candidates[keep]
    # Rank by classifier confidence first (falls back to ranking_score when the pool
    # is unscored or columns are absent), then keep the top ROUTED_TOPK packets.
    rank_cols = [c for c in ("classifier_score", "ranking_score") if c in pool.columns]
    cand_rows = (pool.sort_values(rank_cols, ascending=False)
                 if rank_cols else pool).head(ROUTED_TOPK)
    cand_ids = cand_rows["candidate_id"].tolist()

    cand_lines = []
    for _, row in cand_rows.iterrows():
        cand_lines.append(
            f"  id={row['candidate_id']} date={row.get('parsed_date')} "
            f"role={row.get('candidate_role')} conf={row.get('role_confidence')} "
            f"score={row.get('ranking_score', 0):.1f} "
            f"p_init={row.get('p_initiation', 0):.2f} p_dec={row.get('p_decision', 0):.2f} "
            f"doc_type={str(row.get('document_type_clean', ''))[:30]} "
            f"section={str(row.get('heading_title', ''))[:30]}\n"
            f"    context: {str(row.get('context_text', ''))[:400]}"
        )

    prompt = (
        f"Project: {project_title}\n"
        f"Process type: {process_type}\n"
        f"Agency: {agency}\n"
        f"Current pipeline status: {current_status}\n"
        f"Current pipeline flags: {current_flags}\n\n"
        f"Candidates ({len(cand_rows)}):\n"
        + "\n".join(cand_lines)
    )
    return prompt, cand_ids


def _build_recovery_prompt(
    project_id: str,
    project_title: str,
    process_type: str,
    agency: str,
    packets: pd.DataFrame,
) -> tuple[str, list[str]]:
    """Build the document-recovery prompt from high-scoring context packets."""
    top_packets = packets.nlargest(MAX_RECOVERY_PAGES, "retrieval_score")
    packet_ids = top_packets["context_packet_id"].tolist()

    page_parts = []
    total_tokens = 0
    for _, row in top_packets.iterrows():
        text = str(row.get("context_text", ""))[:800]
        tokens = _estimate_tokens(text)
        if total_tokens + tokens > MAX_INPUT_TOKENS:
            break
        page_parts.append(
            f"[Page {row.get('page_start')} | Doc: {str(row.get('document_title', ''))[:40]}]\n{text}"
        )
        total_tokens += tokens

    prompt = (
        f"Project: {project_title}\n"
        f"Process type: {process_type}\n"
        f"Agency: {agency}\n\n"
        f"Document excerpts:\n"
        + "\n\n---\n\n".join(page_parts)
    )
    return prompt, packet_ids


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_api(
    system_prompt: str,
    user_prompt: str,
    model: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Call the Anthropic Claude API.
    Returns dict with keys: response_json, raw_response_excerpt, input_tokens, output_tokens, error.
    """
    if dry_run:
        return {
            "response_json": {},
            "raw_response_excerpt": "[dry_run]",
            "input_tokens": _estimate_tokens(user_prompt),
            "output_tokens": 0,
            "error": None,
            "error_kind": None,
        }

    last_err = ""
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = msg.content[0].text if msg.content else ""
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                import re
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {}
            return {
                "response_json": parsed,
                "raw_response_excerpt": raw[:500],
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "error": None,
                "error_kind": None,
            }
        except Exception as e:
            last_err = str(e)
            el = last_err.lower()
            # Classify on the SDK's typed exceptions (robust), with a string fallback. Transient =
            # rate-limit (429) / overloaded+5xx (529/500/503/...) / timeout / connection -> retry with
            # backoff. Billing/auth (out of credit, bad key, no access) -> FATAL (caller fail-fasts).
            is_transient = is_billing = False
            try:
                import anthropic as _a
                is_transient = isinstance(e, (_a.RateLimitError, _a.InternalServerError,
                                              _a.APITimeoutError, _a.APIConnectionError))
                if isinstance(e, _a.APIStatusError) and getattr(e, "status_code", None) in (408, 429, 500, 502, 503, 504, 529):
                    is_transient = True
                is_billing = isinstance(e, (_a.AuthenticationError, _a.PermissionDeniedError)) or (
                    isinstance(e, _a.BadRequestError) and any(k in el for k in ("credit", "billing", "quota", "payment")))
            except Exception:
                pass
            # string fallbacks (cover non-SDK errors / wrapped messages)
            if any(k in el for k in ("credit balance", "billing", "insufficient", "quota", "payment required")):
                is_billing = True
            if not is_billing and any(k in el for k in ("rate limit", "429", "overloaded", "529",
                                      "timeout", "timed out", "connection", "internal server", "503", "500")):
                is_transient = True
            if is_billing:
                return {"response_json": {}, "raw_response_excerpt": "", "input_tokens": 0,
                        "output_tokens": 0, "error": last_err, "error_kind": "billing"}
            if is_transient and attempt < MAX_TRANSIENT_RETRIES:
                time.sleep(BACKOFF_BASE_SEC * (2 ** attempt))
                continue
            return {"response_json": {}, "raw_response_excerpt": "", "input_tokens": 0,
                    "output_tokens": 0, "error": last_err,
                    "error_kind": "transient" if is_transient else "other"}
    return {"response_json": {}, "raw_response_excerpt": "", "input_tokens": 0,
            "output_tokens": 0, "error": last_err or "unknown", "error_kind": "other"}


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def _validate_candidate_response(
    response: dict,
    candidate_ids: list[str],
) -> tuple[str | None, str | None, list[str]]:
    """
    Validate adjudication response.
    Returns (initiation_candidate_id, decision_candidate_id, guardrail_flags).
    """
    flags: list[str] = []
    init_id = response.get("initiation_candidate_id")
    dec_id = response.get("decision_candidate_id")

    if init_id and init_id not in candidate_ids:
        flags.append("hallucinated_initiation_id")
        init_id = None
    if dec_id and dec_id not in candidate_ids:
        flags.append("hallucinated_decision_id")
        dec_id = None

    return init_id, dec_id, flags


def _validate_recovery_response(response: dict) -> tuple[str | None, str | None, list[str]]:
    """
    Validate document-recovery response.
    Returns (initiation_date, decision_date, guardrail_flags).
    """
    from datetime import date as date_type
    flags: list[str] = []

    init_date = response.get("initiation_date")
    dec_date = response.get("decision_date")

    for label, val in [("initiation", init_date), ("decision", dec_date)]:
        if val and val != "null":
            try:
                pd.Timestamp(val)
            except Exception:
                flags.append(f"invalid_{label}_date_format")
                if label == "initiation":
                    init_date = None
                else:
                    dec_date = None

    return init_date, dec_date, flags


# ---------------------------------------------------------------------------
# Select projects needing adjudication
# ---------------------------------------------------------------------------

def _classifier_route_signal(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Per-project classifier signals for routing: best confidence, # competing-decision
    candidates, and how many candidates were actually scored. Regex-authoritative
    candidates (role_confidence_score >= AUTHORITATIVE_CONF) count as fully confident."""
    c = candidates_df.copy()
    rconf = pd.to_numeric(c.get("role_confidence_score"), errors="coerce").fillna(0.0)
    cscore = pd.to_numeric(c.get("classifier_score"), errors="coerce").fillna(0.0)
    p_dec = pd.to_numeric(c.get("p_decision"), errors="coerce").fillna(0.0)
    # Authoritative regex candidates are confident even though 04 leaves them unscored.
    c["_eff_conf"] = cscore.where(rconf < AUTHORITATIVE_CONF, 1.0)
    c["_compete_dec"] = (p_dec >= COMPETE_DECISION_PROB).astype(int)
    c["_scored"] = (cscore > 0).astype(int)
    g = c.groupby("project_id")
    return pd.DataFrame({
        "best_conf": g["_eff_conf"].max(),
        "n_compete": g["_compete_dec"].sum(),
        "n_scored": g["_scored"].sum(),
    })


def _select_adjudication_queue(
    dates_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    process_types: list[str],
) -> pd.DataFrame:
    """Select projects that need candidate adjudication (plan §7 + classifier confidence)."""
    sub = dates_df[dates_df["process_type"].isin(process_types)].copy()

    # Already complete and high-confidence: skip
    sub = sub[sub["timeline_status"] != "complete_clear"]

    # Needs candidates to exist
    has_cands = set(candidates_df["project_id"].unique())
    sub = sub[sub["project_id"].isin(has_cands)]

    # Classifier-confidence triggers (in addition to the regex-status triggers below).
    sig = _classifier_route_signal(candidates_df)
    best_conf = sub["project_id"].map(sig["best_conf"])      # NaN if no scored cands -> no conf trigger
    n_compete = sub["project_id"].map(sig["n_compete"]).fillna(0)
    n_scored = sub["project_id"].map(sig["n_scored"]).fillna(0)
    low_confidence = (n_scored > 0) & (best_conf < ROUTE_CONF_THRESHOLD)
    competing_decisions = n_compete >= COMPETE_DECISION_MIN_N

    # Specific triggers from plan §7
    needs_adj = (
        sub["timeline_status"].isin(["missing_initiation", "missing_decision", "invalid_order"]) |
        sub["timeline_flags"].str.contains("multiple_high_score_candidates", na=False) |
        ((sub["timeline_status"] == "missing_both") & sub["project_id"].isin(has_cands)) |
        low_confidence | competing_decisions
    )

    # Completable gate (per missing slot): only adjudicate projects that CAN reach a COMPLETE
    # timeline — each *missing* slot must have a candidate to fill it (an already-filled slot needs
    # none). Drops structurally-incompletable projects (e.g. CE missing init with no init candidate)
    # that would otherwise waste an LLM call returning null. This is the ~11.2k "send" set.
    role = candidates_df["candidate_role"].astype(str)
    caprow = candidates_df.assign(
        _is_init=role.isin(["clear_initiation", "proxy_initiation"]),
        _is_dec=role.isin(["clear_decision", "proxy_decision"]),
    ).groupby("project_id").agg(hic=("_is_init", "max"), hdc=("_is_dec", "max"))
    hic = sub["project_id"].map(caprow["hic"]).fillna(False).astype(bool)
    hdc = sub["project_id"].map(caprow["hdc"]).fillna(False).astype(bool)
    completable = (sub["initiation_date"].notna() | hic) & (sub["decision_date"].notna() | hdc)
    # Per the affirmed rule: send only projects MISSING >=1 slot that can still complete (each
    # missing slot has a candidate). Already-both-present projects (incl. ambiguous re-checks) are
    # NOT sent — they add no completions.
    incomplete = ~(sub["initiation_date"].notna() & sub["decision_date"].notna())
    return sub[needs_adj & completable & incomplete]


def _select_recovery_queue(
    dates_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    packets_df: pd.DataFrame,
    process_types: list[str],
) -> pd.DataFrame:
    """Select projects that need document-recovery adjudication (plan §7)."""
    sub = dates_df[dates_df["process_type"].isin(process_types)].copy()
    sub = sub[sub["timeline_status"].isin(["missing_both", "missing_decision"])]

    # Only projects where NO useful candidates exist
    has_useful_cands = set(
        candidates_df[
            candidates_df["candidate_role"].isin(["clear_initiation", "clear_decision", "proxy_decision"])
        ]["project_id"].unique()
    )
    sub = sub[~sub["project_id"].isin(has_useful_cands)]

    # But strong timeline language in packets
    has_good_packets = set(
        packets_df[packets_df["retrieval_score"] >= 2.0]["project_id"].unique()
    )
    sub = sub[sub["project_id"].isin(has_good_packets)]
    return sub


# ---------------------------------------------------------------------------
# Main adjudication loop
# ---------------------------------------------------------------------------

def run_adjudication(
    mode: str,
    process_types: list[str],
    sample: int | None,
    dry_run: bool,
    model: str,
    sample_ids: str | None = None,
    no_apply: bool = False,
    workers: int = 12,
) -> None:
    print(f"Loading data...")
    dates_df = pd.read_parquet(DATES_PATH)
    candidates_df = pd.read_parquet(CANDIDATES_PATH)
    packets_df = pd.read_parquet(PACKETS_PATH) if PACKETS_PATH.exists() else pd.DataFrame()

    if mode == "candidate_adjudication":
        queue = _select_adjudication_queue(dates_df, candidates_df, process_types)
    else:
        queue = _select_recovery_queue(dates_df, candidates_df, packets_df, process_types)

    print(f"Mode: {mode} | Queue: {len(queue)} projects | Dry run: {dry_run}")

    if sample_ids:
        with open(sample_ids) as f:
            ids = {ln.strip().split(",")[0] for ln in f
                   if ln.strip() and not ln.lower().startswith("project_id")}
        queue = queue[queue["project_id"].isin(ids)]
        print(f"Filtered to {len(queue)} projects from {len(ids)} sample-ids.")
    if sample:
        queue = queue.head(sample)
        print(f"Sampling {sample} projects.")

    # Load existing adjudications to check cache. CRITICAL: only treat SUCCESSFUL (no api_error)
    # adjudications as cached/done. Errored rows (e.g. an out-of-credit run) are NOT cached, so a
    # top-up + re-run retries exactly those instead of silently skipping them.
    existing_adj: pd.DataFrame = pd.DataFrame()
    if ADJUDICATIONS_PATH.exists():
        existing_adj = pd.read_parquet(ADJUDICATIONS_PATH)
    existing_keys: set = set()
    if not existing_adj.empty:
        err = existing_adj["api_error"] if "api_error" in existing_adj.columns else pd.Series([None] * len(existing_adj))
        succeeded = existing_adj[err.isna() | err.astype(str).str.strip().isin(("", "None", "nan"))]
        existing_keys = set(succeeded["prompt_hash"].tolist())

    # Load project-level metadata for prompt building
    from phase2.code.utils.config import US_STATES  # type: ignore
    # fallback agency from dates_df or index
    agency_map: dict[str, str] = {}
    title_map: dict[str, str] = {}
    if (TIMELINE_DIR / "timeline_document_index.parquet").exists():
        idx = pd.read_parquet(
            TIMELINE_DIR / "timeline_document_index.parquet",
            columns=["project_id", "project_title", "lead_agency_harmonized"],
        ).drop_duplicates("project_id")
        agency_map = dict(zip(idx["project_id"], idx["lead_agency_harmonized"].fillna("")))
        title_map = dict(zip(idx["project_id"], idx["project_title"].fillna("")))

    adj_records: list[dict] = []
    dates_updates: list[dict] = []
    run_at = datetime.now(timezone.utc).isoformat()
    cost_usd = 0.0
    billing_errs = 0
    # Current API pricing (USD per 1K tokens) by model — replaces the stale Haiku-3 rates so the
    # cost readout reflects the model actually used.
    PRICES = {"haiku": (0.001, 0.005), "sonnet": (0.003, 0.015), "opus": (0.015, 0.075)}
    _mk = "opus" if "opus" in model else ("sonnet" if "sonnet" in model else "haiku")
    COST_PER_1K_INPUT, COST_PER_1K_OUTPUT = PRICES[_mk]

    def _save_adj() -> None:
        """Checkpoint adjudications so far (existing + new), de-duped. Safe to call repeatedly."""
        if not adj_records:
            return
        new = pd.DataFrame(adj_records)
        combined = (pd.concat([existing_adj, new], ignore_index=True).drop_duplicates("api_call_id")
                    if not existing_adj.empty else new)
        TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(ADJUDICATIONS_PATH, index=False)

    # --- Phase 1 (serial, cheap): build the work list — prompt + cache check per project. ---
    work = []
    for _, proj_row in queue.iterrows():
        pid = proj_row["project_id"]
        process_type = proj_row["process_type"]
        project_title = title_map.get(pid, pid)
        agency = agency_map.get(pid, "")
        proj_cands = candidates_df[candidates_df["project_id"] == pid]
        proj_packets = packets_df[packets_df["project_id"] == pid] if not packets_df.empty else pd.DataFrame()
        if mode == "candidate_adjudication":
            prompt_text, used_ids = _build_candidate_prompt(
                pid, project_title, process_type, agency,
                proj_cands, proj_row.get("timeline_status", ""), proj_row.get("timeline_flags", ""))
            system_prompt = CANDIDATE_ADJUDICATION_SYSTEM
            packet_ids_used: list[str] = []
        else:
            if proj_packets.empty:
                continue
            prompt_text, packet_ids_used = _build_recovery_prompt(pid, project_title, process_type, agency, proj_packets)
            used_ids = []
            system_prompt = DOCUMENT_RECOVERY_SYSTEM
        ph = _prompt_hash(prompt_text)
        if ph in existing_keys:
            continue
        work.append({"pid": pid, "process_type": process_type, "prompt": prompt_text,
                     "system_prompt": system_prompt, "used_ids": used_ids,
                     "packet_ids": packet_ids_used, "ph": ph})
    print(f"  {len(work)} projects to adjudicate ({len(queue) - len(work)} cached/skipped); workers={workers}")

    def _build_record(item, result):
        """Build the adjudication row + dates-update from a completed API result (main thread)."""
        pid, ph = item["pid"], item["ph"]
        in_tok, out_tok = result["input_tokens"], result["output_tokens"]
        call_cost = (in_tok / 1000 * COST_PER_1K_INPUT) + (out_tok / 1000 * COST_PER_1K_OUTPUT)
        rec = {
            "api_call_id": hashlib.sha1(f"{pid}|{ph}|{run_at}".encode()).hexdigest()[:20],
            "project_id": pid, "process_type": item["process_type"], "adjudication_mode": mode,
            "model": model, "provider": DEFAULT_PROVIDER, "prompt_hash": ph,
            "context_packet_ids": json.dumps(item["packet_ids"]),
            "input_tokens": in_tok, "output_tokens": out_tok, "estimated_cost_usd": round(call_cost, 6),
            "response_json": json.dumps(result["response_json"]),
            "raw_response_excerpt": result["raw_response_excerpt"],
            "api_error": result["error"], "called_at": run_at,
        }
        update = None
        if mode == "candidate_adjudication":
            init_id, dec_id, flags = _validate_candidate_response(result["response_json"], item["used_ids"])
            rec.update({"candidate_ids": json.dumps(item["used_ids"]),
                        "selected_initiation_candidate_id": init_id,
                        "selected_decision_candidate_id": dec_id,
                        "guardrail_flags": "|".join(flags) if flags else ""})
            if init_id or dec_id:
                update = {"project_id": pid, "adj_init_id": init_id, "adj_dec_id": dec_id}
        else:
            init_date, dec_date, flags = _validate_recovery_response(result["response_json"])
            resp = result["response_json"]
            rec.update({"candidate_ids": "[]", "selected_initiation_candidate_id": None,
                        "selected_decision_candidate_id": None,
                        "guardrail_flags": "|".join(flags) if flags else ""})
            if init_date or dec_date:
                update = {"project_id": pid, "recovery_init_date": init_date, "recovery_dec_date": dec_date,
                          "recovery_init_evidence": resp.get("initiation_evidence", ""),
                          "recovery_dec_evidence": resp.get("decision_evidence", "")}
        return rec, call_cost, update

    # --- Phase 2 (concurrent): API calls run in worker threads; ALL mutation stays in the main
    # thread (as_completed yields one at a time) so no locks are needed. Throttle is the worker
    # count + the per-call backoff, not an inter-call sleep. ---
    import concurrent.futures as _cf
    n_done = 0
    stop = False
    with _cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut2item = {ex.submit(_call_api, w["system_prompt"], w["prompt"], model, dry_run): w for w in work}
        for fut in _cf.as_completed(fut2item):
            rec, call_cost, update = _build_record(fut2item[fut], fut.result())
            adj_records.append(rec)
            cost_usd += call_cost
            if update:
                dates_updates.append(update)
            # Fail-fast on a drained account: count billing errors (not "consecutive" — results arrive
            # out of order under concurrency); once we cross the threshold, cancel un-started calls and
            # stop. Errored rows aren't cached, so a top-up + re-run resumes them.
            if fut.result().get("error_kind") == "billing":
                billing_errs += 1
                if billing_errs >= BILLING_FAILFAST_N:
                    for f in fut2item:
                        f.cancel()
                    stop = True
            n_done += 1
            if n_done % SAVE_EVERY == 0:
                _save_adj()
                print(f"  {n_done}/{len(work)} adjudicated | cost_usd=${cost_usd:.4f} | checkpoint saved")
            if stop:
                print(f"  STOPPING: {billing_errs} billing/credit errors (out of API credit). "
                      f"Progress saved — top up and re-run to resume the rest.")
                break

    if not adj_records:
        print("No adjudications generated.")
        return

    new_adj = pd.DataFrame(adj_records)
    if not existing_adj.empty:
        combined_adj = pd.concat([existing_adj, new_adj], ignore_index=True)
        combined_adj = combined_adj.drop_duplicates("api_call_id")
    else:
        combined_adj = new_adj

    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    combined_adj.to_parquet(ADJUDICATIONS_PATH, index=False)
    print(f"Wrote: {ADJUDICATIONS_PATH} ({len(combined_adj):,} total adjudications)")
    print(f"Estimated cost this run: ${cost_usd:.4f}")

    # Update project dates from candidate adjudications (skip with --no-apply, e.g. A/B test runs
    # that must not mutate the canonical timeline_project_dates.parquet).
    if dates_updates and not dry_run and not no_apply:
        _apply_adjudication_results(dates_df, candidates_df, dates_updates, mode, run_at)
    elif no_apply:
        print("  --no-apply: dates NOT written (adjudications saved for inspection/comparison only).")


def _apply_adjudication_results(
    dates_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    updates: list[dict],
    mode: str,
    run_at: str,
) -> None:
    """Apply adjudication results to timeline_project_dates.parquet."""
    updated = False
    cand_idx = candidates_df.set_index("candidate_id") if not candidates_df.empty else pd.DataFrame()

    # Audit-timestamp convention: <domain>_llm_run_at set per-row ONLY where the LLM changed a date.
    if "timeline_llm_run_at" not in dates_df.columns:
        dates_df["timeline_llm_run_at"] = ""

    for upd in updates:
        pid = upd["project_id"]
        mask = dates_df["project_id"] == pid
        if not mask.any():
            continue

        if mode == "candidate_adjudication":
            for role, id_col in [("initiation", "adj_init_id"), ("decision", "adj_dec_id")]:
                cid = upd.get(id_col)
                if not cid or cand_idx.empty or cid not in cand_idx.index:
                    continue
                # Don't let LLM adjudication overwrite a date that script 04 already
                # resolved via midpoint imputation — the LLM was only queued to find
                # the OTHER role (e.g. missing initiation), not to improve this one.
                if "midpoint_imputed" in dates_df.columns:
                    already_imputed = dates_df.loc[mask, "midpoint_imputed"].iloc[0]
                    existing_date = dates_df.loc[mask, f"{role}_date"].iloc[0]
                    if already_imputed and pd.notna(existing_date):
                        continue
                cand = cand_idx.loc[cid]
                new_gran = cand.get("date_granularity", "day")
                date_val = cand.get("parsed_date")
                # Month-granularity pick (e.g. an EIS FEIS-publication "July 2024"): store the
                # mid-month 15th to round out bias, mirroring 05's apply_month_midpoint_imputation
                # (the candidate's parsed_date is YYYY-MM-01). Day-level picks are untouched.
                if new_gran == "month" and isinstance(date_val, str) and len(date_val) >= 8:
                    date_val = date_val[:8] + "15"
                dates_df.loc[mask, f"{role}_date"] = date_val
                dates_df.loc[mask, f"{role}_date_granularity"] = new_gran
                dates_df.loc[mask, f"{role}_source_type"] = "api_adjudication"
                dates_df.loc[mask, f"{role}_confidence"] = cand.get("role_confidence", "medium")
                dates_df.loc[mask, f"{role}_evidence_text"] = str(cand.get("context_text", ""))[:300]
                # If API returned a day-level date, it supersedes any midpoint imputation.
                if new_gran == "day" and "midpoint_imputed" in dates_df.columns:
                    dates_df.loc[mask, "midpoint_imputed"] = False
                existing_flags = str(dates_df.loc[mask, "timeline_flags"].iloc[0])
                new_flags = "|".join(filter(None, [existing_flags, "api_adjudicated"]))
                dates_df.loc[mask, "timeline_flags"] = new_flags
                updated = True
        else:
            for role, date_col, ev_col in [
                ("initiation", "recovery_init_date", "recovery_init_evidence"),
                ("decision", "recovery_dec_date", "recovery_dec_evidence"),
            ]:
                val = upd.get(date_col)
                if not val:
                    continue
                dates_df.loc[mask, f"{role}_date"] = val
                dates_df.loc[mask, f"{role}_source_type"] = "api_adjudication"
                dates_df.loc[mask, f"{role}_confidence"] = "medium"
                dates_df.loc[mask, f"{role}_evidence_text"] = str(upd.get(ev_col, ""))[:300]
                # Recovery dates come as YYYY-MM-DD strings; if day-level, clear imputation flag.
                if isinstance(val, str) and len(val) == 10 and "midpoint_imputed" in dates_df.columns:
                    dates_df.loc[mask, "midpoint_imputed"] = False
                existing_flags = str(dates_df.loc[mask, "timeline_flags"].iloc[0])
                new_flags = "|".join(filter(None, [existing_flags, "api_recovery"]))
                dates_df.loc[mask, "timeline_flags"] = new_flags
                updated = True

        # Recompute timeline_status from the now-present dates so a date recovered here COUNTS as
        # complete. 08_analyze.R (and the coverage CSVs) key on timeline_status; leaving it stale at
        # 'missing_*' would silently exclude the recovery from the deliverable. Also tag month picks.
        row = dates_df.loc[mask].iloc[0]
        idate, ddate = row.get("initiation_date"), row.get("decision_date")
        hi = pd.notna(idate) and str(idate) not in ("", "None")
        hd = pd.notna(ddate) and str(ddate) not in ("", "None")
        if hi and hd:
            try:
                bad_order = pd.to_datetime(ddate) < pd.to_datetime(idate)
            except Exception:
                bad_order = False
            is_proxy = bool(row.get("initiation_is_proxy", False)) or bool(row.get("decision_is_proxy", False))
            new_status = "invalid_order" if bad_order else ("complete_with_proxy" if is_proxy else "complete_clear")
        elif hi:
            new_status = "missing_decision"
        elif hd:
            new_status = "missing_initiation"
        else:
            new_status = "missing_both"
        dates_df.loc[mask, "timeline_status"] = new_status
        if hd and str(row.get("decision_date_granularity")) == "month":
            ef = str(dates_df.loc[mask, "timeline_flags"].iloc[0])
            if "month_decision" not in ef:
                dates_df.loc[mask, "timeline_flags"] = "|".join(filter(None, [ef, "month_decision"]))
        dates_df.loc[mask, "timeline_llm_run_at"] = run_at  # per-row LLM audit stamp

    if updated:
        dates_df.to_parquet(DATES_PATH, index=False)
        print(f"Updated {DATES_PATH} with adjudication results.")


def main() -> None:
    parser = argparse.ArgumentParser(description="API adjudication for unresolved timeline cases.")
    parser.add_argument(
        "--mode",
        choices=["candidate_adjudication", "document_recovery"],
        default="candidate_adjudication",
    )
    parser.add_argument("--process", nargs="+", choices=["CE", "EA", "EIS"], default=["EA", "EIS"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", type=int, help="Limit to N projects for testing.")
    parser.add_argument("--sample-ids", help="Path to a file/CSV (first column = project_id) to restrict the queue to specific projects.")
    parser.add_argument("--no-apply", action="store_true", help="Save adjudications but do NOT write timeline_project_dates.parquet (for A/B test runs).")
    parser.add_argument("--workers", type=int, default=12, help="Concurrent API calls (I/O-bound; 10-15 recommended). 1 = serial.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts but do not call API.")
    args = parser.parse_args()

    if not DATES_PATH.exists():
        raise FileNotFoundError(f"Dates not found: {DATES_PATH}\nRun 05_select_dates.py first.")
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Candidates not found: {CANDIDATES_PATH}")

    run_adjudication(
        mode=args.mode,
        process_types=args.process,
        sample=args.sample,
        dry_run=args.dry_run,
        model=args.model,
        sample_ids=args.sample_ids,
        no_apply=args.no_apply,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
