"""D6 — 10: action-verb labeling (refactor: tech_group x action grid).

Assigns each clean FONSI's action a controlled VERB (within its tech_group) so 09 can form
`tech_group__action` grid cells. Reuses the CACHED extraction summary (no document re-read) —
mirrors `03 --stage classify`. `is_codifiable` is derived DETERMINISTICALLY from the verb.

Reads:  fonsi_enrichment.parquet
Writes: fonsi_action_labels.parquet
        (project_id, action, is_codifiable, action_confidence, actionlabel_parse_ok,
         actionlabel_cache_hit, actionlabel_error, actionlabel_prompt_version,
         actionlabel_run_at, actionlabel_llm_run_at)

Usage:
  python 10_action_label.py --dry-run        # projected cost only — no key/Keychain, no spend
  python 10_action_label.py --workers 4      # the billable pass (cached; re-runs are cheap/free)
"""
import argparse
import hashlib
import json
import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa'.")

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import D6_ANALYSIS_DIR, ensure_d6_dirs, utc_now
import enrich_lib
from prompts import ACTIONLABEL_PROMPT_VERSION, build_action_label_prompt, is_codifiable_for

LLM_MODEL_DEFAULT = "claude-sonnet-4-6"
DEFAULT_WORKERS = 4
CHECKPOINT_EVERY = 20

ENRICH = D6_ANALYSIS_DIR / "fonsi_enrichment.parquet"
OUT = D6_ANALYSIS_DIR / "fonsi_action_labels.parquet"
CACHE = D6_ANALYSIS_DIR / "_raw" / "fonsi_actionlabel_cache.json"
EST_IN, EST_OUT = 1250, 80      # dry-run estimate: verb vocab prompt + summary in; short structured out


def _fld(rec, k) -> str:
    v = rec.get(k)
    return "" if v is None else str(v)


def _prompt(rec) -> str:
    return build_action_label_prompt(_fld(rec, "tech_group"), _fld(rec, "action_summary"),
                                     _fld(rec, "key_activities"), _fld(rec, "action_label_freeform"),
                                     _fld(rec, "purpose_and_need"))


def _key(model: str, prompt_text: str) -> str:
    return hashlib.sha256(f"{ACTIONLABEL_PROMPT_VERSION}|{model}|{prompt_text}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL_DEFAULT)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--sample", type=int, default=0, help="debug: first N FONSIs")
    ap.add_argument("--dry-run", action="store_true", help="projected cost only — no key/Keychain, no spend")
    args = ap.parse_args()

    ensure_d6_dirs()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    run_at = utc_now()

    en = pd.read_parquet(ENRICH)
    en["project_id"] = en["project_id"].astype(str)

    # [refactor §5.1] corpus diagnostic (no output, $0): most FONSIs carry >1 project_type tag,
    # but tech_group is D3's single resolved primary — log the multi-tag rate and how often that
    # resolved primary is NOT literally one of the raw project_type tags (a primary-tech override).
    def _tags(v):
        try:
            x = json.loads(v) if isinstance(v, str) and v.strip().startswith("[") else v
            return [str(t).strip() for t in x if str(t).strip()] if isinstance(x, list) \
                else ([str(v).strip()] if str(v).strip() else [])
        except Exception:
            return []
    if "project_type" in en.columns:
        _n = len(en)
        _pt = en["project_type"].map(_tags)
        _multi = int((_pt.map(len) > 1).sum())
        _tg = en["tech_group"].astype(str).str.strip() if "tech_group" in en.columns \
            else pd.Series([""] * _n, index=en.index)
        def _override(tags, tg):
            tg = (tg or "").strip().lower()
            if tg in ("", "nan", "none"):
                return False
            return not any(tg in t.lower() or t.lower() in tg for t in tags)
        _ovr = int(sum(_override(t, g) for t, g in zip(_pt, _tg)))
        print(f"[10] multi-tag project_type: {_multi}/{_n} ({_multi / _n:.0%}) carry >1 tag; "
              f"primary-tech override (tech_group absent from raw project_type tags): "
              f"{_ovr}/{_n} ({_ovr / _n:.0%})")

    if args.sample:
        en = en.head(args.sample)
    recs = en.to_dict("records")
    workable = [r for r in recs if str(r.get("action_summary") or "").strip()]

    if args.dry_run:
        in_price, out_price = enrich_lib.pricing_for(args.model)
        n = len(workable)
        cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        n_cached = sum(1 for r in workable if _key(args.model, _prompt(r)) in cache)
        n_new = n - n_cached
        cost = n_new * (EST_IN / 1e6 * in_price + EST_OUT / 1e6 * out_price)
        print(f"[10] dry-run @ {args.model}: {n} labelable FONSIs ({n_cached} cached, {n_new} new) "
              f"x ~{EST_IN} in / {EST_OUT} out tok")
        print(f"[10] projected cost for the NEW rows: ${cost:.2f}  (cached rows are free)")
        return

    cache: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    status: dict = {}     # pid -> (state, parsed, cache_hit, error); state in {ok, failed, skipped}
    pending = []
    for r in recs:
        pid = r["project_id"]
        if not str(r.get("action_summary") or "").strip():
            status[pid] = ("skipped", None, False, "no_summary")
            continue
        pt = _prompt(r)
        k = _key(args.model, pt)
        if k in cache:
            status[pid] = ("ok", cache[k], True, "")
        else:
            pending.append((pid, pt, k))
    n_cached = sum(1 for s in status.values() if s[0] == "ok")
    n_skipped = sum(1 for s in status.values() if s[0] == "skipped")
    print(f"[10] action-label: {n_cached} cached, {len(pending)} new, {n_skipped} skipped "
          f"(no summary); workers={args.workers}")

    if pending:
        pre = enrich_lib.actionlabel_preflight(args.model)
        if pre.get("parsed") is None:
            raise SystemExit(f"[10] preflight FAILED ({pre.get('error')}) — aborting before spend.")
        client = enrich_lib.make_client()
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(enrich_lib.call_action_label, pt, args.model, client): (pid, k)
                    for (pid, pt, k) in pending}
            for fut in as_completed(futs):
                pid, k = futs[fut]
                res = fut.result()
                parsed = res.get("parsed")
                if parsed is not None:
                    cache[k] = parsed
                    status[pid] = ("ok", parsed, False, "")
                else:
                    status[pid] = ("failed", None, False, res.get("error") or "parse_failed")
                done += 1
                if done % CHECKPOINT_EVERY == 0 or done == len(pending):
                    enrich_lib.write_json_atomic(CACHE, cache)
                    print(f"[10] checkpoint {done}/{len(pending)}")
        enrich_lib.write_json_atomic(CACHE, cache)

    rows = []
    for r in recs:
        pid = r["project_id"]
        state, parsed, hit, err = status.get(pid, ("skipped", None, False, "no_summary"))
        if state == "ok":
            verb = parsed.get("action", "other")
            rows.append({
                "project_id": pid, "action": verb, "is_codifiable": bool(is_codifiable_for(verb)),
                "action_confidence": parsed.get("action_confidence", ""),
                "actionlabel_parse_ok": True, "actionlabel_cache_hit": hit, "actionlabel_error": "",
                "actionlabel_prompt_version": ACTIONLABEL_PROMPT_VERSION,
                "actionlabel_run_at": run_at, "actionlabel_llm_run_at": run_at,
            })
        else:   # failed / skipped -> action="other" (is_codifiable True: physical unknown, keep in frame),
                # NOT stamped with a version -> a partial run can never masquerade as complete
            rows.append({
                "project_id": pid, "action": "other", "is_codifiable": True,
                "action_confidence": "", "actionlabel_parse_ok": False, "actionlabel_cache_hit": False,
                "actionlabel_error": err, "actionlabel_prompt_version": "",
                "actionlabel_run_at": run_at, "actionlabel_llm_run_at": "",
            })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    n_ok = int(out["actionlabel_parse_ok"].sum())
    n_failed = sum(1 for s in status.values() if s[0] == "failed")
    if n_failed:
        print(f"[10] {n_failed} call(s) FAILED -> action='other', parse_ok=False (not version-stamped)")
    print(f"[10] wrote {OUT.name}: {len(out)} rows, {n_ok} labeled")
    print("[10] verb counts:", out.loc[out["actionlabel_parse_ok"], "action"].value_counts().to_dict())
    n_nc = int((~out["is_codifiable"]).sum())
    print(f"[10] non-codifiable rows: {n_nc} "
          f"({out.loc[~out['is_codifiable'], 'action'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
