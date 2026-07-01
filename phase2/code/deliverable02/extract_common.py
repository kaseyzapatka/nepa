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

import pandas as pd

import common as C
from candidate_gen import threshold_hits

PROMPT_VERSION = "d2_v1"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# normalized candidate columns every substrate's generator must emit
CAND_COLS = [
    "project_id", "document_id", "section_id", "source_substrate", "source_unit_id",
    "page_start", "page_end", "span_char_start", "span_char_end", "heading_title",
    "evidence_text", "evidence_text_sha256", "source_span_sha256",
    "candidate_class_guess", "determination_polarity_guess", "matched_cue_group",
    "resource_area_guess", "resource_subarea_guess",
]


def anthropic_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:  # macOS keychain (user's billable key) — only reached in real mode
        return subprocess.check_output(
            ["security", "find-generic-password", "-s", "nepa-anthropic", "-w"],
            text=True).strip()
    except Exception:
        return None


def adjudicate_llm(client, model: str, window: str, resource_hint: str) -> dict:
    """One schema-constrained, temp-0 adjudication. Returns parsed JSON (+ raw for hashing)."""
    prompt = (
        "You are coding a NEPA significance determination from an EA/FONSI or EIS impact span. "
        "Return STRICT JSON with keys: determination_class (one of no_significant_impact, "
        "less_than_significant, less_than_significant_with_mitigation, significant_adverse, "
        "significant_unavoidable, eis_required, not_a_determination, ambiguous), "
        "determination_scope (project_overall|resource_specific|alternative_specific|"
        "threshold_specific|programmatic_or_tiered|procedural), determination_polarity "
        "(no_adverse|adverse_not_significant|adverse_significant|mixed|unknown), "
        "shared_resource_area, primary_threshold_type, primary_threshold_status, "
        "rationale_text, abstain (bool). "
        f"Resource hint: {resource_hint}.\n\nSPAN:\n{window[:4000]}"
    )
    resp = client.messages.create(
        model=model, max_tokens=600, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text
    try:
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        parsed = {"abstain": True, "determination_class": "ambiguous"}
    parsed["_raw"] = raw
    return parsed


def _scope_guess(cue_group: str, resource: str) -> str:
    if cue_group == "document_outcome":
        return "project_overall"
    return "resource_specific" if resource != "unknown" else "project_overall"


def build_determinations(cand: pd.DataFrame, mit: pd.DataFrame, ctx: pd.DataFrame,
                         dry_run: bool, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the determination table + threshold child (substrate-agnostic)."""
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
    if not dry_run:
        key = anthropic_key()
        if not key:
            raise SystemExit("No Anthropic key (keychain 'nepa-anthropic' / ANTHROPIC_API_KEY). "
                             "Run with --dry-run, or provide the key for the billable pass.")
        import anthropic
        client = anthropic.Anthropic(api_key=key)

    rows, thr_rows = [], []
    for r in df.itertuples(index=False):
        resource = r.resource_area_guess
        llm_at = ""
        if dry_run:
            dclass, dscope = r.candidate_class_guess, _scope_guess(r.matched_cue_group, resource)
            dpol, method, conf = r.determination_polarity_guess, "regex", 0.5
            rationale = prompt_v = input_hash = response_hash = ""
        else:
            input_hash = C.sha256_join(r.project_id, resource, r.evidence_text_sha256,
                                       PROMPT_VERSION, C.SCHEMA_VERSION, model)
            res = adjudicate_llm(client, model, r.evidence_text, resource)
            dclass = res.get("determination_class", r.candidate_class_guess)
            dscope = res.get("determination_scope", _scope_guess(r.matched_cue_group, resource))
            dpol = res.get("determination_polarity", r.determination_polarity_guess)
            resource = res.get("shared_resource_area", resource) or resource
            method, conf = "regex+llm", 0.9
            rationale = res.get("rationale_text", "")
            response_hash = hashlib.sha256(res.get("_raw", "").encode()).hexdigest()
            prompt_v, llm_at = PROMPT_VERSION, run_at

        thr = threshold_hits(r.evidence_text)
        primary_t = thr[0] if thr else "none"
        primary_ts = "unknown" if thr else "none"
        d2_resource = r.resource_subarea_guess
        needs_review = dry_run or dclass in ("ambiguous", "not_a_determination") or resource == "unknown"

        det_id = C.sha256_join(r.project_id, r.document_id, r.source_substrate, r.source_unit_id,
                               resource, d2_resource, dclass, dscope, primary_t, primary_ts, "")
        rows.append({
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
            "resource_area_source": ("keyword" if dry_run else "llm"),
            "determination_class": dclass, "determination_polarity": dpol,
            "determination_scope": dscope, "alternative_name": "", "rationale_text": rationale,
            "primary_threshold_type": primary_t, "primary_threshold_status": primary_ts,
            "mitigation_flag": bool(r.mitigation_flag),
            "mitigation_enforceability": ("permit_condition" if r.mitigation_flag else "none"),
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
            "needs_human_review": bool(needs_review),
            "review_reason": ("dry_run_regex_only" if dry_run else ("abstain" if needs_review else "")),
            "llm_provider": ("" if dry_run else "anthropic"), "llm_model": ("" if dry_run else model),
            "prompt_version": prompt_v, "input_hash": input_hash, "response_hash": response_hash,
            "schema_version": C.SCHEMA_VERSION,
            "significance_extraction_run_at": run_at, "significance_llm_run_at": llm_at,
        })
        for t in thr:
            thr_rows.append({
                "determination_instance_id": det_id, "project_id": r.project_id,
                "threshold_type": t, "threshold_status": "unknown",
                "threshold_verbatim": "", "threshold_evidence_sha256": r.evidence_text_sha256,
                "threshold_specific_flag": (dscope == "threshold_specific"),
                "schema_version": C.SCHEMA_VERSION, "significance_extraction_run_at": run_at,
            })
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
