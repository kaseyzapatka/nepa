"""D2 shared extraction assembly — used by 02 (FONSI) and 04 (EIS).

Consumes a NORMALIZED candidate frame (same columns from either substrate) + a mitigation
summary (keyed by source_unit_id; empty for EIS) + project context, and emits the frozen
determination schema (plan v2.11 §4) + the determination_thresholds child table. Handles both
--dry-run (regex only, key-free) and the billable LLM adjudication path.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

import pandas as pd

import common as C
from candidate_gen import threshold_hits
from significance_taxonomy import (
    DETERMINATION_CLASSES, DETERMINATION_POLARITIES, DETERMINATION_SCOPES,
    RESOURCE_CROSSWALK, RESOURCE_PROJECT_WIDE, SHARED_RESOURCE_AREAS,
    THRESHOLD_STATUSES, THRESHOLD_TYPES,
)

PROMPT_VERSION = "d2_v3"   # v3: MULTI-output — one determination per resource area in the window
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# normalized candidate columns every substrate's generator must emit
CAND_COLS = [
    "project_id", "document_id", "section_id", "source_substrate", "source_unit_id",
    "page_start", "page_end", "span_char_start", "span_char_end", "heading_title",
    "evidence_text", "evidence_text_sha256", "source_span_sha256",
    "candidate_class_guess", "determination_polarity_guess", "matched_cue_group",
    "resource_area_guess", "resource_subarea_guess",
]


_KEY_CACHE: str | None = None


def anthropic_key() -> str | None:
    """Resolve the API key ONCE per process (env var, else macOS keychain) and cache it —
    so a --batch-run (submit -> poll -> fetch) asks for the keychain password exactly once."""
    global _KEY_CACHE
    if _KEY_CACHE:
        return _KEY_CACHE
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:  # macOS keychain (user's billable key) — only reached in real mode
            key = subprocess.check_output(
                ["security", "find-generic-password", "-s", "nepa-anthropic", "-w"],
                text=True).strip()
        except Exception:
            key = None
    _KEY_CACHE = key
    return key


def _prompt_for(window: str, resource_hint: str) -> str:
    return (
        "You are coding NEPA significance determinations from an EA/FONSI or EIS span. A single "
        "span often contains SEVERAL determinations — e.g. an Environmental Consequences chapter "
        "concludes separately on air, water, biological, cultural, ... resources. Extract EVERY "
        "significance conclusion the span makes.\n"
        'Return STRICT JSON: {"determinations": [ <one object per resource area with a conclusion '
        'in the span> ], "abstain": <bool>}.\n'
        "Each object has these keys, using ONLY the controlled values:\n"
        "- determination_class: no_significant_impact | less_than_significant | "
        "less_than_significant_with_mitigation | significant_adverse | significant_unavoidable | "
        "eis_required | not_a_determination | ambiguous\n"
        "- determination_scope: project_overall | resource_specific | alternative_specific | "
        "threshold_specific | programmatic_or_tiered | procedural\n"
        "- determination_polarity: no_adverse | adverse_not_significant | adverse_significant | "
        "mixed | unknown\n"
        "- shared_resource_area: EXACTLY one of air_quality, water, biological, cultural, visual, "
        "noise, soils_geology, socioeconomic, transportation, land_use, climate_ghg, public_health, "
        "project_wide (ONLY for a project-level / FONSI conclusion, scope=project_overall), "
        "unknown (a resource-specific finding whose resource you genuinely cannot place). "
        "NEVER invent other labels. Mapping: wetlands/floodplains/groundwater -> water; "
        "wildlife/vegetation/special-status species -> biological; historic/tribal/Section 106 -> "
        "cultural; farmland/recreation/land-use plans -> land_use; worker or public safety/EMF/"
        "hazardous materials/solid waste -> public_health; environmental justice/economy/public "
        "services -> socioeconomic; GHG/climate -> climate_ghg; traffic/roads/aviation -> "
        "transportation.\n"
        "- primary_threshold_type: NAAQS | PSD | ESA_take | ESA_jeopardy | NHPA_adverse_effect | "
        "wetland_floodplain | noise_threshold | visual_vrm | other_quantitative | none | unknown "
        "— the regulatory threshold THIS resource's conclusion leans on; 'none' when not "
        "threshold-anchored (a mere statute mention is not an anchor).\n"
        "- primary_threshold_status: exceeds | does_not_exceed | may_exceed | mitigated_below | "
        "not_evaluated | unknown.\n"
        "- rationale_text: 1-2 sentences grounded in the span, specific to THIS resource.\n"
        "Rules: (1) Emit ONE object per resource area the span concludes on — do not merge several "
        "resources into one object, and do NOT invent determinations for resources the span does "
        "not discuss. (2) If the span states a project-wide FONSI/decision conclusion, ALSO include "
        "one object with scope=project_overall and shared_resource_area=project_wide. (3) A "
        "No-Action-alternative statement concluding no/reduced impact IS a determination "
        "(scope=alternative_specific). (4) less_than_significant_with_mitigation ONLY when the "
        "conclusion DEPENDS on committed mitigation; impacts minor by inherent design are plain "
        "less_than_significant. (5) If the span contains NO significance conclusion (table of "
        "contents, acronym list, boilerplate, pure project description, affected-environment/"
        "background description, methodology text, cross-reference, comment list), return "
        '{"determinations": [], "abstain": false}. Set abstain=true only if the span is unreadable.\n'
        f"Resource hint (from a keyword screen; may be wrong): {resource_hint}.\n\n"
        f"SPAN:\n{window[:C.WINDOW_CHAR_CAP]}"
    )


def _message_params(model: str, window: str, resource_hint: str) -> dict:
    """Request params for one adjudication. max_tokens is generous because a window can yield
    ~12 determinations. Sonnet 5 / Opus 4.8 reject non-default sampling params (temperature only
    on Haiku); Sonnet 5 runs adaptive thinking by default — disable it so the token budget goes
    to the JSON, not hidden reasoning (Haiku has no thinking param)."""
    p = {"model": model, "max_tokens": 3000,
         "messages": [{"role": "user", "content": _prompt_for(window, resource_hint)}]}
    if "haiku" in model:
        p["temperature"] = 0
    else:
        p["thinking"] = {"type": "disabled"}
    return p


def _coerce_determinations(raw: str) -> dict:
    """Parse an LLM response into {'determinations': [dict,...], 'abstain': bool, '_raw': str}.
    Tolerant of: the documented {"determinations":[...]} object, a bare top-level list, or a
    single determination object (older shape) — always normalized to a list."""
    out = {"determinations": [], "abstain": False, "_raw": raw}
    try:
        start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0])
        end = max(raw.rfind("}"), raw.rfind("]")) + 1
        parsed = json.loads(raw[start:end])
    except Exception:
        out["abstain"] = True
        return out
    if isinstance(parsed, list):
        out["determinations"] = [d for d in parsed if isinstance(d, dict)]
    elif isinstance(parsed, dict) and isinstance(parsed.get("determinations"), list):
        out["determinations"] = [d for d in parsed["determinations"] if isinstance(d, dict)]
        out["abstain"] = bool(parsed.get("abstain"))
    elif isinstance(parsed, dict) and parsed.get("determination_class"):
        out["determinations"] = [parsed]           # single-object fallback
        out["abstain"] = bool(parsed.get("abstain"))
    elif isinstance(parsed, dict):
        # dict without a determinations list and without a determination_class: only a genuine
        # {"abstain": true} is a valid empty; any other unrecognized shape is a parse failure.
        out["abstain"] = bool(parsed.get("abstain", True))
    else:
        out["abstain"] = True                       # bare scalar / unrecognized JSON -> parse failure
    return out


def _norm_vocab(val, allowed, default: str) -> str:
    """Snap an LLM answer onto the controlled vocabulary (case/space tolerant); else default."""
    v = str(val or "").strip().lower().replace(" ", "_").replace("-", "_")
    for a in allowed:
        if v == a.lower():
            return a
    return default


def adjudicate_llm(client, model: str, window: str, resource_hint: str) -> dict:
    """One synchronous call; returns the coerced {'determinations':[...],'abstain':bool,'_raw'}."""
    resp = client.messages.create(**_message_params(model, window, resource_hint))
    raw = next((b.text for b in resp.content if b.type == "text"), "")
    return _coerce_determinations(raw)


# ---------------------------------------------------------------- Batch API (50% price)
def _client():
    key = anthropic_key()
    if not key:
        raise SystemExit("No Anthropic key found (keychain 'nepa-anthropic' or ANTHROPIC_API_KEY). "
                         "Use --batch-run: one keychain unlock, key held in process memory only.")
    import anthropic
    return anthropic.Anthropic(api_key=key)


# API caps: 100,000 requests / 256 MB per batch. We chunk conservatively so neither
# is ever approached (EIS full run is ~20k windows / ~100 MB — one chunk today, but
# auto-splits if the candidate set grows after the retrieval spike).
MAX_BATCH_REQS = 50_000
MAX_BATCH_BYTES = 150_000_000


def _req_bytes(text: str) -> int:
    """Rough JSON-encoded request size: window + prompt scaffold + envelope + escaping."""
    return int(len(text) * 1.2) + 1_800


def submit_batch(cand: pd.DataFrame, model: str, track: str) -> list[str]:
    """Submit candidate windows via the Message Batches API (async; 50% of standard price),
    auto-chunked to stay far below the 100k-request / 256 MB per-batch caps. Saves a
    candidate snapshot + manifest so --batch-fetch can rebuild determinations later.
    Needs the key exactly once (one keychain prompt)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client()
    cand = cand.reset_index(drop=True).copy()
    cand["batch_custom_id"] = [f"{track}-{i:06d}" for i in range(len(cand))]

    # greedy, order-preserving chunking under both caps
    chunks: list[list[int]] = [[]]
    chunk_bytes = 0
    for i, text in enumerate(cand["evidence_text"]):
        b = _req_bytes(text or "")
        if chunks[-1] and (chunk_bytes + b > MAX_BATCH_BYTES or len(chunks[-1]) >= MAX_BATCH_REQS):
            chunks.append([])
            chunk_bytes = 0
        chunks[-1].append(i)
        chunk_bytes += b

    batches = []
    for n, idxs in enumerate(chunks, 1):
        sub = cand.iloc[idxs]
        requests = [
            Request(custom_id=cid,
                    params=MessageCreateParamsNonStreaming(**_message_params(model, text, hint)))
            for cid, text, hint in zip(sub["batch_custom_id"], sub["evidence_text"],
                                       sub["resource_area_guess"])
        ]
        batch = client.messages.batches.create(requests=requests)
        batches.append({"batch_id": batch.id, "n_requests": len(requests)})
        print(f"  submitted batch {n}/{len(chunks)}: {batch.id}  ({len(requests):,} windows)")

    C.write_parquet(cand, C.D2_ANALYSIS_DIR / f"batch_candidates_{track}.parquet", "batch snapshot")
    manifest_path = C.D2_ANALYSIS_DIR / f"batch_manifest_{track}.json"
    manifest_path.write_text(json.dumps({
        "batches": batches, "model": model, "track": track,
        "n_requests_total": len(cand), "submitted_at": C.utc_now(),
        "prompt_version": PROMPT_VERSION, "schema_version": C.SCHEMA_VERSION}, indent=2))
    print(f"  manifest -> {manifest_path.relative_to(C.PHASE2)}  (model={model})")
    print("  Anthropic processes offline (usually <1 h). Fetch with --batch-fetch [--wait].")
    return [b["batch_id"] for b in batches]


def fetch_batch(track: str, wait: bool) -> tuple[pd.DataFrame, dict, str]:
    """Retrieve all submitted batches -> (candidate snapshot, {custom_id: parsed JSON}, model)."""
    manifest_path = C.D2_ANALYSIS_DIR / f"batch_manifest_{track}.json"
    if not manifest_path.exists():
        raise SystemExit(f"no batch manifest at {manifest_path} — run --batch-submit first.")
    manifest = json.loads(manifest_path.read_text())
    client = _client()
    results = {}
    for entry in manifest["batches"]:
        bid = entry["batch_id"]
        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            if not wait:
                raise SystemExit(f"batch {bid} still {b.processing_status} "
                                 f"({b.request_counts.processing:,} in flight) — re-run later, or add --wait.")
            print(f"  {bid}: {b.processing_status} … {b.request_counts.processing:,} left; sleeping 60s")
            time.sleep(60)
        for r in client.messages.batches.results(bid):
            if r.result.type == "succeeded":
                raw = next((blk.text for blk in r.result.message.content if blk.type == "text"), "")
                results[r.custom_id] = _coerce_determinations(raw)
    cand = pd.read_parquet(C.D2_ANALYSIS_DIR / f"batch_candidates_{track}.parquet")
    print(f"  all batches ended: {len(results):,}/{len(cand):,} windows succeeded")
    return cand, results, manifest["model"]


def _scope_guess(cue_group: str, resource: str) -> str:
    # only the document-outcome cue (a FONSI/decision statement) implies a project-wide scope;
    # an unclassified resource hit stays resource_specific (the review flag carries "couldn't place",
    # NOT the scope — else it would be forced to project_wide downstream).
    return "project_overall" if cue_group == "document_outcome" else "resource_specific"


def _assemble_row(r, dd, *, run_at, model, prompt_v, input_hash, response_hash,
                  batch_missing, model_abstained, dry_run) -> tuple[dict, list]:
    """Assemble ONE determination row (+ its threshold child rows) from window `r` and a single
    determination dict `dd`. `dd is None` -> regex-guess row (dry-run / batch-missing); otherwise
    `dd` is one element of the LLM's per-resource-area list. A hash of `rationale_text` enters the
    id so two DISTINCT determinations sharing (resource,class,scope,threshold) in one window
    (differing only by rationale) don't collide — while a genuine byte-identical duplicate (same
    rationale too) still collapses via the `seen` dedup."""
    is_llm = dd is not None
    if not is_llm:
        dclass = r.candidate_class_guess
        dscope = _scope_guess(r.matched_cue_group, r.resource_area_guess)
        dpol, rationale = r.determination_polarity_guess, ""
        resource_raw, pt_raw, pts_raw = r.resource_area_guess, "", ""
    else:
        # snap the LLM's controlled-vocab answers (case/space/dash tolerant) so downstream
        # branching on dclass/dscope is reliable — off-vocab answers fall back to a safe default.
        dclass = _norm_vocab(dd.get("determination_class"), DETERMINATION_CLASSES, r.candidate_class_guess)
        dscope = _norm_vocab(dd.get("determination_scope"), DETERMINATION_SCOPES,
                             _scope_guess(r.matched_cue_group, r.resource_area_guess))
        dpol = _norm_vocab(dd.get("determination_polarity"), DETERMINATION_POLARITIES,
                           r.determination_polarity_guess)
        rationale = dd.get("rationale_text", "")
        resource_raw, pt_raw, pts_raw = (dd.get("shared_resource_area", r.resource_area_guess),
                                         dd.get("primary_threshold_type"),
                                         dd.get("primary_threshold_status"))
    method = "regex+llm" if is_llm else "regex"
    conf = 0.9 if is_llm else 0.5
    llm_at = run_at if is_llm else ""

    # resource is SCOPE-AUTHORITATIVE: a project-level conclusion is project_wide (never review);
    # otherwise snap onto the 12+unknown vocab — 'unknown' means "genuinely couldn't place".
    resource_off_vocab = False
    if dscope == "project_overall":
        resource = RESOURCE_PROJECT_WIDE
    else:
        resource = _norm_vocab(resource_raw, SHARED_RESOURCE_AREAS, "unknown")
        resource_off_vocab = (is_llm and resource == "unknown"
                              and str(resource_raw or "").strip().lower().replace(" ", "_")
                              .replace("-", "_") not in ("", "unknown", "none"))

    # thresholds: LLM primary is authoritative per resource; regex cue hits are the fallback and
    # apply only to the regex path (they are window-level and can't be split across resources).
    llm_pt = _norm_vocab(pt_raw, THRESHOLD_TYPES, "") if is_llm else ""
    llm_pts = _norm_vocab(pts_raw, THRESHOLD_STATUSES, "") if is_llm else ""
    regex_thr = [] if is_llm else threshold_hits(r.evidence_text)
    if llm_pt:
        primary_t, primary_ts = llm_pt, (llm_pts or "unknown")
    else:
        primary_t = regex_thr[0] if regex_thr else "none"
        primary_ts = "unknown" if regex_thr else "none"

    # keyword subarea kept only if valid under the (authoritative) shared area, else 'unknown'
    d2_resource = r.resource_subarea_guess
    if is_llm and d2_resource not in RESOURCE_CROSSWALK.get(resource, {}):
        d2_resource = "unknown"

    resource_flagged = resource_off_vocab or (is_llm and resource == "unknown"
                                              and dscope != "project_overall")
    needs_review = (dry_run or batch_missing or model_abstained
                    or dclass in ("ambiguous", "not_a_determination") or resource_flagged)
    review_reason = ("dry_run_regex_only" if dry_run
                     else "batch_result_missing" if batch_missing
                     else "model_abstained" if model_abstained
                     else "non_determination_or_ambiguous"
                     if dclass in ("ambiguous", "not_a_determination")
                     else "resource_off_vocab" if resource_off_vocab
                     else "resource_unknown" if resource_flagged
                     else "")

    # RESOURCE-MATCHED mitigation: the window-level flag (r.mitigation_flag) attaches to THIS
    # determination only when THIS resource is among the resources the matched D6 conditions
    # actually cover (or this is a project-level conclusion). This stops a single-resource
    # commitment from marking every resource in a multi-resource window as mitigation-dependent.
    # A bare 'unknown' condition-resource does not broadcast to specific resources.
    _mit_res = {t.strip().lower() for t in
                str(r.mitigation_resource_areas or "").replace(";", ",").split(",") if t.strip()}
    resource_mitigation_match = bool(r.mitigation_flag) and (
        dscope == "project_overall" or resource in _mit_res)
    # per-resource reporting field: the precise D6 resource-match OR the LLM's per-resource class
    mitigation_dependent = (resource_mitigation_match
                            or dclass == "less_than_significant_with_mitigation")

    det_id = C.sha256_join(r.project_id, r.document_id, r.source_substrate, r.source_unit_id,
                           resource, d2_resource, dclass, dscope, primary_t, primary_ts, "",
                           C.sha256_text(rationale))
    record = {
        "determination_instance_id": det_id,
        "source_substrate": r.source_substrate, "source_unit_id": r.source_unit_id,
        "project_id": r.project_id, "document_id": r.document_id,
        "process_type": r.process_type, "document_type_clean": r.doc_type,
        "agency": r.agency, "agency_scope_status": r.agency_scope_status,
        "agency_scope_rule": r.agency_scope_rule, "decision_date": r.decision_date,
        "cohort_by_date": r.cohort_by_date, "decision_source_type": r.decision_source_type,
        "decision_confidence": r.decision_confidence, "decision_is_proxy": r.decision_is_proxy,
        "time_scope_status": r.time_scope_status, "analysis_scope": r.analysis_scope,
        "decision_period": r.decision_period, "applicability_period": r.applicability_period,
        "fra_overlay": r.fra_overlay, "regime_assignment_status": r.regime_assignment_status,
        "framework_regime": r.decision_period,   # descriptive alias, materialized once here
        "shared_resource_area": resource, "d2_resource_area": d2_resource,
        "resource_area_source": ("llm" if is_llm else "keyword"),
        "determination_class": dclass, "determination_polarity": dpol,
        "determination_scope": dscope, "alternative_name": "", "rationale_text": rationale,
        "primary_threshold_type": primary_t, "primary_threshold_status": primary_ts,
        # mitigation_flag = raw WINDOW-level D6 match (any enforceable commitment in the window).
        # It over-attributes across a multi-resource window, so use it ONLY for the DOCUMENT-level
        # "mitigated FONSI" rate (OR across the document). For RESOURCE-level reporting use
        # mitigation_dependent (per-resource: D6 resource-match OR the LLM's per-resource class).
        "mitigation_flag": bool(r.mitigation_flag),
        "mitigation_resource_matched": resource_mitigation_match,
        "mitigation_dependent": mitigation_dependent,
        "mitigation_enforceability": ("permit_condition" if resource_mitigation_match else "none"),
        "matched_condition_row_count": int(r.matched_condition_row_count),
        "condition_role_set": r.condition_role_set, "obligation_level_set": r.obligation_level_set,
        "mitigation_resource_areas": r.mitigation_resource_areas,
        "section_id": r.section_id, "evidence_span_id": r.source_unit_id,
        "evidence_text": r.evidence_text, "evidence_text_sha256": r.evidence_text_sha256,
        "source_span_sha256": r.source_span_sha256, "hash_semantics": r.source_substrate,
        "page_start": r.page_start, "page_end": r.page_end,
        "span_char_start": r.span_char_start, "span_char_end": r.span_char_end,
        "quoted_span": (r.evidence_text or "")[:300],
        "extraction_method": method, "confidence": conf,
        "needs_human_review": bool(needs_review), "review_reason": review_reason,
        "llm_provider": ("anthropic" if is_llm else ""), "llm_model": (model if is_llm else ""),
        "prompt_version": prompt_v, "input_hash": input_hash, "response_hash": response_hash,
        "schema_version": C.SCHEMA_VERSION,
        "significance_extraction_run_at": run_at, "significance_llm_run_at": llm_at,
    }
    # child threshold rows — never on non-determinations (regex once fired on acronym lists)
    thr_rows = []
    if dclass not in ("not_a_determination", "ambiguous"):
        cited = list(dict.fromkeys(
            regex_thr + ([llm_pt] if llm_pt not in ("", "none", "unknown") else [])))
        for t in cited:
            thr_rows.append({
                "determination_instance_id": det_id, "project_id": r.project_id,
                "threshold_type": t,
                "threshold_status": (primary_ts if t == primary_t else "unknown"),
                "threshold_verbatim": "", "threshold_evidence_sha256": r.evidence_text_sha256,
                "threshold_specific_flag": (dscope == "threshold_specific"),
                "schema_version": C.SCHEMA_VERSION, "significance_extraction_run_at": run_at,
            })
    return record, thr_rows


def build_determinations(cand: pd.DataFrame, mit: pd.DataFrame, ctx: pd.DataFrame,
                         dry_run: bool, model: str,
                         llm_results: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the determination table (grain: document × resource_area × determination) +
    threshold child. Each LLM call returns a LIST of determinations (one per resource area the
    window concludes on), so a window explodes into multiple rows.

    llm_results: optional {batch_custom_id: coerced dict} from fetch_batch()."""
    run_at = C.utc_now()
    if mit is None or mit.empty:
        mit = pd.DataFrame(columns=["source_unit_id", "matched_condition_row_count",
                                    "condition_role_set", "obligation_level_set",
                                    "mitigation_resource_areas", "mitigation_same_section"])
    df = cand.merge(mit, on="source_unit_id", how="left").merge(ctx, on="project_id", how="left")
    df["matched_condition_row_count"] = df["matched_condition_row_count"].fillna(0).astype(int)
    for col in ("condition_role_set", "obligation_level_set", "mitigation_resource_areas"):
        df[col] = df[col].fillna("")
    df["mitigation_flag"] = df["matched_condition_row_count"] > 0

    client = None
    if not dry_run and llm_results is None:
        client = _client()  # synchronous mode only; batch mode already has its results

    # an empty LLM result (window has no determination) -> one not_a_determination row
    EMPTY_DET = {"determination_class": "not_a_determination", "determination_scope": "procedural",
                 "shared_resource_area": "unknown", "determination_polarity": "unknown"}

    rows, thr_rows, seen = [], [], set()
    for r in df.itertuples(index=False):
        input_hash = response_hash = prompt_v = ""
        batch_missing = model_abstained = False
        if dry_run:
            unit_dets = [None]                                    # regex-guess single row
        else:
            input_hash = C.sha256_join(r.project_id, r.evidence_text_sha256,
                                       PROMPT_VERSION, C.SCHEMA_VERSION, model)
            res = (llm_results.get(getattr(r, "batch_custom_id", "")) if llm_results is not None
                   else adjudicate_llm(client, model, r.evidence_text, r.resource_area_guess))
            if res is None:                                       # batch item errored/expired
                batch_missing, prompt_v, unit_dets = True, PROMPT_VERSION, [None]
            else:
                model_abstained = bool(res.get("abstain"))
                response_hash = hashlib.sha256(res.get("_raw", "").encode()).hexdigest()
                prompt_v = PROMPT_VERSION
                dets = res.get("determinations") or []
                unit_dets = dets if dets else [EMPTY_DET]        # empty -> not_a_determination
        for dd in unit_dets:
            rec, trs = _assemble_row(
                r, dd, run_at=run_at, model=model, prompt_v=prompt_v, input_hash=input_hash,
                response_hash=response_hash, batch_missing=batch_missing,
                model_abstained=model_abstained, dry_run=dry_run)
            if rec["determination_instance_id"] in seen:
                continue     # dedup identical (resource,class,scope,threshold) across the run
            seen.add(rec["determination_instance_id"])
            rows.append(rec)
            thr_rows.extend(trs)
    return pd.DataFrame(rows), pd.DataFrame(thr_rows)


def write_manifest(artifacts: dict, dry_run: bool, model: str) -> None:
    recs = []
    for name, path in artifacts.items():
        p = C.PHASE2 / path
        try:
            b = open(p, "rb").read()
            recs.append({"artifact": name, "path": str(path),
                         "n_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
        except FileNotFoundError:
            recs.append({"artifact": name, "path": str(path), "n_bytes": 0, "sha256": ""})
    man = pd.DataFrame(recs)
    man["mode"] = "dry_run" if dry_run else "llm"
    man["model"] = "" if dry_run else model
    man["prompt_version"] = PROMPT_VERSION
    man["schema_version"] = C.SCHEMA_VERSION
    man["run_at"] = C.utc_now()
    C.write_parquet(man, C.RUN_MANIFEST, "manifest")
