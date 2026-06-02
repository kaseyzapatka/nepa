"""
LLM gold-labeling of D4 timeline candidates.

Replaces the manual label step: an LLM reads each project's candidate dates +
context and assigns a role to every candidate, then names THE initiation and
THE decision candidate. Output matches the schema that 03_import_gold_labels.py
ingests, so the downstream gold/training tables are produced unchanged.

IMPORTANT — this is the *real* labeler. The sibling 04_codex_prelabel script is
a mechanical regex echo (it copies candidate_role into gold_candidate_role); it
does NOT call a model. Train the classifier only on labels produced here (or by
a human), never on the regex-echo output.

Pipeline position:
    labeling/01_build_gold_samples.py        -> splits
    labeling/02_prepare_gold_review_packets  -> review_packets/<split>_batchNNN_*.csv
    labeling/05_llm_label_candidates.py  (this)  -> codex_labels/<split>_*_llm_labeled.csv
    labeling/03_import_gold_labels.py --projects <...>_projects_llm_labeled.csv \
                                      --candidates <...>_candidates_llm_labeled.csv
        -> gold/timeline_gold_candidate_training.parquet
    ../04_classify_candidates.py --train

Per-row audit timestamps (project convention):
    label_run_at      — set on ALL rows for the run
    label_llm_run_at  — set per row only when the LLM call for that project succeeded

Usage:
    # dry run (build prompts, no API calls, no cost):
    python 05_llm_label_candidates.py --split diagnostic_balanced_v2 --dry-run --limit 3
    # real run:
    python 05_llm_label_candidates.py --split diagnostic_balanced_v2
    python 05_llm_label_candidates.py --split train_enriched_v1 --model claude-haiku-4-5-20251001
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
PACKET_DIR = PHASE2 / "output" / "deliverable04" / "gold" / "review_packets"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04" / "gold" / "codex_labels"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
RETRY_SLEEP = 2.0
MAX_CONTEXT_CHARS = 320  # per-candidate context sent to the model

VALID_ROLES = {
    "clear_initiation", "proxy_initiation",
    "clear_decision", "proxy_decision",
    "review", "historical", "unknown",
}
VALID_SELECTED = {"initiation", "decision", "alternate_valid", "none"}

ROLE_TO_TYPE = {
    "clear_initiation": "clear", "proxy_initiation": "proxy",
    "clear_decision": "clear", "proxy_decision": "proxy",
}
CANDIDATE_ERROR_CATEGORY = {
    "historical": "historical_project", "review": "specialist_review", "unknown": "other",
}

SYSTEM_PROMPT = (
    "You are labeling dates extracted from US federal NEPA environmental review "
    "documents (Categorical Exclusions, Environmental Assessments, Environmental "
    "Impact Statements) to build a training set.\n\n"
    "For each candidate date you are given the date, the surrounding text, and the "
    "document/section it came from. Assign exactly one role:\n"
    "  clear_initiation  - explicit NEPA process start (application received, NOI "
    "published, scoping began, review initiated)\n"
    "  proxy_initiation  - weak/indirect initiation signal (draft date, filing, "
    "month-only application reference)\n"
    "  clear_decision    - explicit NEPA decision (ROD/FONSI/Decision Record signed "
    "or issued, CE determination, authorized-officer signature)\n"
    "  proxy_decision    - weak/indirect decision signal (final EA/EIS publication "
    "as upper bound, case-number year)\n"
    "  review            - a specialist/reviewer signature or coordination date, NOT "
    "the agency decision\n"
    "  historical        - a reference to a PRIOR action (previous EIS/RMP, earlier "
    "ROW grant, sub-process consultation start)\n"
    "  unknown           - any other date (event dates, map dates, citations, "
    "boilerplate)\n\n"
    "Then pick the single best INITIATION date and single best DECISION date for the "
    "project (or null if none qualifies). Initiation must precede the decision.\n\n"
    "Respond with ONLY a JSON object:\n"
    "{\"candidates\":[{\"candidate_id\":\"..\",\"role\":\"..\"}],"
    "\"initiation_candidate_id\":\"..|null\",\"decision_candidate_id\":\"..|null\","
    "\"notes\":\"..\"}"
)


def _clean(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v == ""):
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _iso(v: object) -> str:
    t = _clean(v)
    if not t:
        return ""
    parsed = pd.to_datetime(t, errors="coerce")
    return parsed.date().isoformat() if not pd.isna(parsed) else t


def _confidence(v: object) -> str:
    t = _clean(v).lower()
    if t in {"high", "medium", "low"}:
        return t
    try:
        s = float(v)
    except (TypeError, ValueError):
        return "low"
    return "high" if s >= 4 else "medium" if s >= 2 else "low"


def _build_prompt(cands: pd.DataFrame) -> str:
    lines = []
    for _, c in cands.iterrows():
        ctx = _clean(c.get("context_text"))[:MAX_CONTEXT_CHARS]
        heading = _clean(c.get("heading_title"))
        doctype = _clean(c.get("document_type_clean"))
        lines.append(
            f"- candidate_id={c['candidate_id']} | date={_clean(c.get('parsed_date'))} "
            f"| doc={doctype} | section={heading}\n  text: {ctx}"
        )
    proc = _clean(cands.iloc[0].get("process_type"))
    return f"Process type: {proc}\nCandidates:\n" + "\n".join(lines)


def _call_api(user_prompt: str, model: str, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"json": {}, "raw": "[dry_run]", "in_tok": 0, "out_tok": 0, "error": None}
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text if msg.content else ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
        return {"json": parsed, "raw": raw[:500],
                "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens,
                "error": None}
    except Exception as e:
        return {"json": {}, "raw": "", "in_tok": 0, "out_tok": 0, "error": str(e)}


def _label_project_candidates(
    cands: pd.DataFrame, model: str, dry_run: bool
) -> tuple[dict[str, str], str | None, str | None, str, str | None]:
    """Return (role_by_id, init_id, dec_id, notes, error)."""
    prompt = _build_prompt(cands)
    res = _call_api(prompt, model, dry_run)
    if res["error"]:
        return {}, None, None, "", res["error"]
    j = res["json"]
    valid_ids = set(cands["candidate_id"].astype(str))

    role_by_id: dict[str, str] = {}
    for item in (j.get("candidates") or []):
        cid = _clean(item.get("candidate_id"))
        role = _clean(item.get("role"))
        if cid in valid_ids and role in VALID_ROLES:
            role_by_id[cid] = role

    init_id = _clean(j.get("initiation_candidate_id")) or None
    dec_id = _clean(j.get("decision_candidate_id")) or None
    if init_id not in valid_ids:
        init_id = None
    if dec_id not in valid_ids:
        dec_id = None
    return role_by_id, init_id, dec_id, _clean(j.get("notes")), None


def label_candidates_df(
    cands: pd.DataFrame, role_by_id: dict, init_id, dec_id, model: str, run_at: str, ok: bool
) -> pd.DataFrame:
    out = cands.copy()
    for col in ["gold_candidate_role", "gold_selected_for", "gold_error_category",
                "gold_candidate_notes", "reviewer", "candidate_review_status",
                "label_run_at", "label_llm_run_at"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(object)

    ids = out["candidate_id"].astype(str)
    # Fall back to the regex role only if the model omitted a candidate.
    out["gold_candidate_role"] = [
        role_by_id.get(cid, _clean(out.loc[i, "candidate_role"]))
        for i, cid in zip(out.index, ids)
    ]
    out["gold_selected_for"] = "none"
    valid_roles = ["clear_initiation", "proxy_initiation", "clear_decision", "proxy_decision"]
    out.loc[out["gold_candidate_role"].isin(valid_roles), "gold_selected_for"] = "alternate_valid"
    if init_id is not None:
        out.loc[ids == str(init_id), "gold_selected_for"] = "initiation"
    if dec_id is not None:
        out.loc[ids == str(dec_id), "gold_selected_for"] = "decision"
    out["gold_error_category"] = out["gold_candidate_role"].map(CANDIDATE_ERROR_CATEGORY).fillna("")
    out["gold_candidate_notes"] = "LLM label" if ok else "LLM call failed; regex role retained"
    out["reviewer"] = f"llm:{model}"
    out["candidate_review_status"] = "llm_labeled" if ok else "llm_failed"
    out["label_run_at"] = run_at
    out["label_llm_run_at"] = run_at if ok else ""
    return out


def label_projects_df(
    projects: pd.DataFrame, labeled_cands: pd.DataFrame, model: str, run_at: str
) -> pd.DataFrame:
    out = projects.copy()
    for col in list(out.columns):
        if col.startswith("gold_") or col in {"reviewer", "review_status"}:
            out[col] = out[col].fillna("").astype(object) if col in out else ""

    cgroups = {str(pid): g for pid, g in labeled_cands.groupby("project_id", sort=False)} \
        if not labeled_cands.empty else {}

    for idx, row in out.iterrows():
        pid = str(row["project_id"])
        cands = cgroups.get(pid, pd.DataFrame())
        for role in ["initiation", "decision"]:
            sel = cands[cands["gold_selected_for"] == role] if not cands.empty else pd.DataFrame()
            if not sel.empty:
                c = sel.iloc[0]
                out.at[idx, f"gold_{role}_date"] = _iso(c.get("parsed_date"))
                out.at[idx, f"gold_{role}_granularity"] = _clean(c.get("date_granularity")) or "unknown"
                out.at[idx, f"gold_{role}_type"] = ROLE_TO_TYPE.get(_clean(c.get("gold_candidate_role")), "proxy")
                out.at[idx, f"gold_{role}_source_type"] = _clean(c.get("candidate_source_type")) or _clean(c.get("source_tier"))
                out.at[idx, f"gold_{role}_candidate_id"] = _clean(c.get("candidate_id"))
                out.at[idx, f"gold_{role}_document_id"] = _clean(c.get("document_id"))
                out.at[idx, f"gold_{role}_page_number"] = _clean(c.get("page_number"))
                out.at[idx, f"gold_{role}_evidence_text"] = _clean(c.get("context_text"))
                out.at[idx, f"gold_{role}_confidence"] = _confidence(c.get("role_confidence"))
                out.at[idx, f"gold_{role}_missing_reason"] = ""
            else:
                out.at[idx, f"gold_{role}_date"] = ""
                out.at[idx, f"gold_{role}_type"] = "missing"
                out.at[idx, f"gold_{role}_missing_reason"] = "no_evidence"
        out.at[idx, "reviewer"] = f"llm:{model}"
        out.at[idx, "review_status"] = "llm_labeled"
        out.at[idx, "label_run_at"] = run_at
    return out


def run(split: str, model: str, dry_run: bool, limit: int | None, suffix: str) -> None:
    project_paths = sorted(PACKET_DIR.glob(f"{split}_batch*_projects.csv"))
    if not project_paths:
        raise FileNotFoundError(f"No project packets for split {split!r} in {PACKET_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_at = datetime.now(timezone.utc).isoformat()
    in_tok = out_tok = n_proj = n_ok = n_fail = 0
    all_proj, all_cand = [], []

    for ppath in project_paths:
        cpath = ppath.with_name(ppath.name.replace("_projects.csv", "_candidates.csv"))
        projects = pd.read_csv(ppath)
        candidates = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame()

        labeled_cand_parts = []
        if not candidates.empty:
            for pid, group in candidates.groupby("project_id", sort=False):
                if limit is not None and n_proj >= limit:
                    break
                n_proj += 1
                role_by_id, init_id, dec_id, notes, err = _label_project_candidates(group, model, dry_run)
                ok = err is None and not dry_run
                if err:
                    n_fail += 1
                    print(f"  [{pid}] LLM error: {err}")
                elif not dry_run:
                    n_ok += 1
                labeled_cand_parts.append(
                    label_candidates_df(group, role_by_id, init_id, dec_id, model, run_at, ok)
                )
                if not dry_run:
                    time.sleep(0.0)

        labeled_cands = pd.concat(labeled_cand_parts, ignore_index=True) if labeled_cand_parts else pd.DataFrame()
        labeled_projects = label_projects_df(projects, labeled_cands, model, run_at)

        batch_id = ppath.name.replace("_projects.csv", "")
        pout = OUTPUT_DIR / f"{batch_id}_projects{suffix}.csv"
        cout = OUTPUT_DIR / f"{batch_id}_candidates{suffix}.csv"
        labeled_projects.to_csv(pout, index=False)
        labeled_cands.to_csv(cout, index=False)
        all_proj.append(labeled_projects)
        if not labeled_cands.empty:
            all_cand.append(labeled_cands)
        print(f"Wrote: {pout}\nWrote: {cout}")
        if limit is not None and n_proj >= limit:
            break

    comb_p = OUTPUT_DIR / f"{split}_projects{suffix}.csv"
    comb_c = OUTPUT_DIR / f"{split}_candidates{suffix}.csv"
    pd.concat(all_proj, ignore_index=True).to_csv(comb_p, index=False)
    if all_cand:
        pd.concat(all_cand, ignore_index=True).to_csv(comb_c, index=False)
    print(f"\nWrote combined: {comb_p}\nWrote combined: {comb_c}")
    print(f"Projects labeled: {n_proj}  ok={n_ok}  failed={n_fail}  dry_run={dry_run}")
    print(f"Tokens in/out: {in_tok}/{out_tok}")
    print(f"\nNext: python ../labeling/03_import_gold_labels.py "
          f"--projects {comb_p.name} --candidates {comb_c.name}  (paths under {OUTPUT_DIR})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM gold-labeling of D4 timeline candidates.")
    parser.add_argument("--split", required=True,
                        choices=["diagnostic_balanced_v2", "train_enriched_v1", "test_representative_v1"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts, no API calls.")
    parser.add_argument("--limit", type=int, help="Cap number of projects (for testing).")
    parser.add_argument("--suffix", default="_llm_labeled",
                        help="Output filename suffix (default _llm_labeled).")
    args = parser.parse_args()
    run(args.split, args.model, args.dry_run, args.limit, args.suffix)


if __name__ == "__main__":
    main()
