"""D6 — 03: comprehensive LLM enrichment, in two cached stages.

ONE read of every clean-energy EA→FONSI (452) that extracts everything Analyses 1
and 2 need, in one structured (tool-use) call per project — so the paid pass runs
ONCE and never repeats. Prompt + 37-field schema in prompts.py; packet builder,
stratified sampler, call, and quote verification shared via enrich_lib.py.

TWO STAGES, each separately cached (`--stage`):
  - EXTRACT  — the expensive pass: reads the evidence packet, extracts all fields
               (incl. a coarse action_category). Cached on the packet + schema version.
  - CLASSIFY — a cheap pass that re-asks ONLY action_category from the already-extracted
               summary, with real definitions + an enum schema (prompts.build_classification_prompt),
               and OVERWRITES action_category. Cached on the summary + classify version.
  Default `--stage both` runs extract→classify (a from-scratch run is fully correct).
  `--stage classify` reuses the cached extraction output and only re-classifies (~$1.4 for
  451) — used to fix the classifier without re-paying for extraction. Bump
  CLASSIFICATION_PROMPT_VERSION to force a classify-only re-run; bump ENRICHMENT_*_VERSION
  to force an extraction re-run.

KEY (same as D4): $ANTHROPIC_API_KEY if set, else macOS Keychain 'nepa-anthropic'
  (one prompt). --dry-run never touches the key.

OUTPUTS (suffix _pilot / _sampleN for non-full runs)
  RAW:      data/raw/deliverable06/fonsi_enrichment_raw.parquet      (everything + raw + errors)
  ANALYSIS: data/analysis/deliverable06/fonsi_enrichment.parquet     (piped fields + evidence_cited)

USAGE
  python 03_enrich_llm.py --dry-run                 # projected cost for both stages, no key
  python 03_enrich_llm.py --stage classify          # cheap re-classify only (reuses extraction)
  python 03_enrich_llm.py                           # full: extract (cache-aware) then classify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

import enrich_lib
from common import (
    D6_ANALYSIS_DIR, D6_RAW_DIR, D6_REVIEW_DIR, ensure_d6_dirs, sha256_text, utc_now, write_parquet,
)
from prompts import (
    CLASSIFICATION_PROMPT_VERSION,
    ENRICHMENT_ANALYSIS_COLUMNS,
    ENRICHMENT_FIELDS,
    ENRICHMENT_PROMPT_VERSION,
    ENRICHMENT_SCHEMA_VERSION,
    build_classification_prompt,
)

CACHE = D6_RAW_DIR / "fonsi_enrichment_cache.json"
CLASSIFY_CACHE = D6_RAW_DIR / "fonsi_classification_cache.json"
LLM_MODEL_DEFAULT = "claude-sonnet-4-6"
EST_IN_TOK, EST_OUT_TOK = 5000, 1700   # extraction dry-run cost estimate
EST_CLF_IN, EST_CLF_OUT = 650, 90      # classification dry-run cost estimate (summary in, tiny out)
MIN_SUCCESS_STRICT = 0.9               # pilot/full: tool-use should parse ~always
MIN_SUCCESS_DEBUG = 0.5                # --sample debug mode only
DEFAULT_WORKERS = 6                    # parallel API calls (SDK backoff handles overloads)
CHECKPOINT_EVERY = 10                  # flush cache + partial parquet every N completed calls


def cache_key(model: str, text: str) -> str:
    return sha256_text(f"{ENRICHMENT_PROMPT_VERSION}|{ENRICHMENT_SCHEMA_VERSION}|{model}|{text}")


def classify_key(model: str, text: str) -> str:
    return sha256_text(f"{CLASSIFICATION_PROMPT_VERSION}|{model}|{text}")


def _audit(packet_text: str, model: str, parsed, run_at: str, res: dict, tech: str,
           tag_map: dict, cache_hit: bool) -> dict:
    return {
        "enrichment_extraction_run_at": run_at,                                   # THIS run
        "enrichment_llm_run_at": (res.get("llm_run_at", run_at) if parsed is not None else ""),  # original call
        "llm_provider": "anthropic" if parsed is not None else "",
        "llm_model": model, "prompt_version": ENRICHMENT_PROMPT_VERSION,
        "schema_version": ENRICHMENT_SCHEMA_VERSION, "input_sha256": sha256_text(packet_text),
        "tech_group": tech, "n_excerpts": len(tag_map),
        "used_fallback": "packet-fallback" in packet_text, "cache_hit": cache_hit,
        "parse_ok": parsed is not None, "stop_reason": res.get("stop_reason", ""),
        "llm_error": res.get("error", ""), "in_tokens": res.get("in_tok", 0),
        "out_tokens": res.get("out_tok", 0), "raw_response": res.get("raw", ""),
    }


def skip_row(r, packet_text: str, run_at: str, model: str) -> dict:
    rec = {"project_id": r.project_id}
    rec.update({nm: None for nm, _t, _d in ENRICHMENT_FIELDS})
    rec["evidence_cited"] = "[]"
    rec.update(_audit(packet_text, model, None, run_at,
                      {"error": "no_evidence"}, str(getattr(r, "tech_group", "")), {}, False))
    return rec


def result_row(r, packet_text: str, tag_map: dict, res: dict, cache_hit: bool,
               run_at: str, model: str, main_by_doc: dict) -> dict:
    parsed = res.get("parsed")
    rec = {"project_id": r.project_id}
    rec.update(enrich_lib.coerce(parsed) if parsed is not None
               else {nm: None for nm, _t, _d in ENRICHMENT_FIELDS})
    cited = enrich_lib.cite_quotes(parsed, tag_map, main_by_doc) if parsed else []
    rec["evidence_cited"] = json.dumps(cited, ensure_ascii=False)
    rec.update(_audit(packet_text, model, parsed, run_at, res, str(getattr(r, "tech_group", "")), tag_map, cache_hit))
    return rec


# ===========================================================================
# Stage 1 — extraction (the expensive pass; cached on packet + schema version)
# ===========================================================================
def run_extraction(pk, args, run_at, client, raw_out, clean_out, suffix, mode) -> pd.DataFrame:
    spans_by_pid, main_by_doc = enrich_lib.load_spans_and_main(pk["project_id"])
    cache: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # build packets; split skip vs work, then cache-hits vs misses
    work, skip_rows, need_fb = [], [], []
    for r in pk.itertuples(index=False):
        pt, tm = enrich_lib.build_evidence_packet(r, spans_by_pid.get(r.project_id, pd.DataFrame()))
        if tm:
            work.append((r, pt, tm))
        else:
            need_fb.append(r)                        # no D6 spans/typed text -> try section recovery
    if need_fb:                                      # recover from broad document_sections before skipping
        sec_by_pid = enrich_lib.load_sections([r.project_id for r in need_fb])
        recovered = 0
        for r in need_fb:
            pt, tm = enrich_lib.build_section_fallback_packet(r, sec_by_pid.get(r.project_id))
            if tm:
                work.append((r, pt, tm)); recovered += 1
            else:
                skip_rows.append(skip_row(r, pt, run_at, args.model))
        print(f"[03] section-fallback recovered {recovered}/{len(need_fb)} zero-span project(s)")
    results: dict = {}                               # project_id -> (res, cache_hit)
    pending = []
    for (r, pt, tm) in work:
        k = cache_key(args.model, pt)
        if k in cache:
            results[r.project_id] = (cache[k], True)
        else:
            pending.append((r, pt, tm, k))
    skipped = len(skip_rows)
    print(f"[03] extract: {len(work)} to enrich ({len(results)} cached, {len(pending)} new) "
          f"+ {skipped} skipped; workers={args.workers}")

    def checkpoint():
        enrich_lib.write_json_atomic(CACHE, cache)   # cached successes survive a crash -> cheap resume
        partial = skip_rows + [result_row(r, pt, tm, *results[r.project_id], run_at, args.model, main_by_doc)
                               for (r, pt, tm) in work if r.project_id in results]
        write_parquet(pd.DataFrame(partial), raw_out)

    # parallel API calls for the misses; checkpoint periodically
    done = 0
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(enrich_lib.call_enrichment, pt, args.model, client): (r, pt, tm, k)
                    for (r, pt, tm, k) in pending}
            for fut in as_completed(futs):
                r, pt, tm, k = futs[fut]
                res = fut.result()                   # call_enrichment never raises
                if res.get("parsed") is not None:
                    res["llm_run_at"] = run_at
                    cache[k] = res
                results[r.project_id] = (res, False)
                done += 1
                if done % CHECKPOINT_EVERY == 0 or done == len(pending):
                    checkpoint()
                    print(f"[03] checkpoint {done}/{len(pending)} new calls done -> {raw_out.name}")
    else:
        checkpoint()

    rows = skip_rows + [result_row(r, pt, tm, *results[r.project_id], run_at, args.model, main_by_doc)
                        for (r, pt, tm) in work]
    ok = sum(1 for (r, _pt, _tm) in work if results[r.project_id][0].get("parsed") is not None)

    raw_df = pd.DataFrame(rows)
    write_parquet(raw_df, raw_out)
    keep = ["project_id", *ENRICHMENT_ANALYSIS_COLUMNS, "evidence_cited",
            "enrichment_extraction_run_at", "enrichment_llm_run_at"]
    # analysis-ready clean output: metadata passthrough + computed confidence (self-contained)
    clean_df = enrich_lib.add_confidence(enrich_lib.attach_metadata(raw_df[keep], pk))
    write_parquet(clean_df, clean_out)
    # evidence-level CSV: one verbatim quote per row with provenance (the quote audit surface)
    ev_out = D6_REVIEW_DIR / f"fonsi_enrichment_evidence{suffix}.csv"
    enrich_lib.build_evidence_frame(clean_df, pk).to_csv(ev_out, index=False)

    attempted = len(work)
    n_q = sum(len(json.loads(x)) for x in raw_df["evidence_cited"])
    n_v = sum(sum(c.get("verified") is True for c in json.loads(x)) for x in raw_df["evidence_cited"])
    print(f"[03] parsed {ok}/{attempted} attempted ({skipped} skipped: no evidence); "
          f"verified quotes {n_v}/{n_q}; tokens in={int(raw_df['in_tokens'].sum())} out={int(raw_df['out_tokens'].sum())}")
    if raw_df["llm_error"].astype(bool).any():
        print("[03] errors:", raw_df.loc[raw_df['llm_error'].astype(bool), 'llm_error'].value_counts().head().to_dict())

    # evidence-coverage review (skips / fallbacks / excerpt counts) — surfaced, not silent
    cov_cols = ["project_id", "tech_group", "n_excerpts", "used_fallback", "cache_hit", "parse_ok", "llm_error"]
    cov_out = D6_REVIEW_DIR / f"fonsi_enrichment_coverage{suffix}.csv"
    raw_df[cov_cols].to_csv(cov_out, index=False)
    n_fb = int(raw_df["used_fallback"].sum())
    print(f"[03] coverage: {skipped} skipped (no evidence), {n_fb} packet-fallback, "
          f"mean excerpts/packet={raw_df.loc[~raw_df['llm_error'].eq('no_evidence'),'n_excerpts'].mean():.1f} -> {cov_out.name}")
    if skipped:
        print("[03] skipped by tech_group:",
              raw_df.loc[raw_df['llm_error'].eq('no_evidence'), 'tech_group'].value_counts().to_dict())
    print(f"[03] raw  -> {raw_out}")
    print(f"[03] data -> {clean_out}")
    print(f"[03] evidence (1 quote/row) -> {ev_out}")

    min_success = MIN_SUCCESS_DEBUG if args.sample else MIN_SUCCESS_STRICT
    if attempted and ok / attempted < min_success:
        print(f"[03] FAIL: parse rate {ok}/{attempted} below {min_success:.0%} (mode={mode}).")
        sys.exit(1)
    return clean_df


# ===========================================================================
# Stage 2 — classification (cheap; reuses cached extraction; overwrites action_category)
# ===========================================================================
def _classify_input(rec: dict) -> str:
    return build_classification_prompt(
        str(rec.get("project_title") or ""),
        str(rec.get("action_summary") or ""),
        str(rec.get("key_activities") or ""),
        str(rec.get("action_label_freeform") or ""),
        str(rec.get("purpose_and_need") or ""),
    )


def run_classification(clean_df, model, client, run_at, workers):
    """Re-classify action_category from the cached summary. Overwrites action_category,
    preserving the extraction value as action_category_pass1 and adding the model's
    confidence + rationale. Cached on (classify version | model | summary-prompt)."""
    recs = clean_df.to_dict("records")
    cache: dict = json.loads(CLASSIFY_CACHE.read_text()) if CLASSIFY_CACHE.exists() else {}
    results: dict = {}
    pending, skipped = [], []
    for rec in recs:
        pid = rec["project_id"]
        if not str(rec.get("action_summary") or "").strip():
            skipped.append(pid)                      # no-evidence row: keep its category as-is
            continue
        pt = _classify_input(rec)
        k = classify_key(model, pt)
        if k in cache:
            results[pid] = cache[k]
        else:
            pending.append((pid, pt, k))
    print(f"[03] classify: {len(results)} cached, {len(pending)} new, {len(skipped)} skipped "
          f"(no summary); workers={workers}")

    done = 0
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(enrich_lib.call_classification, pt, model, client): (pid, k)
                    for (pid, pt, k) in pending}
            for fut in as_completed(futs):
                pid, k = futs[fut]
                parsed = fut.result().get("parsed")
                if parsed is not None:
                    cache[k] = parsed
                results[pid] = parsed
                done += 1
                if done % CHECKPOINT_EVERY == 0 or done == len(pending):
                    enrich_lib.write_json_atomic(CLASSIFY_CACHE, cache)
                    print(f"[03] classify checkpoint {done}/{len(pending)}")
    enrich_lib.write_json_atomic(CLASSIFY_CACHE, cache)

    df = clean_df.copy()
    if "action_category_pass1" not in df.columns:    # preserve the EXTRACTION value once, as audit
        df["action_category_pass1"] = df["action_category"]
    pids = df["project_id"].tolist()

    def field(pid, key, default):
        r = results.get(pid)
        return r[key] if isinstance(r, dict) and r.get(key) is not None else default

    df["action_category"] = [field(p, "action_category", o) for p, o in zip(pids, df["action_category"])]
    df["classification_confidence"] = [field(p, "classification_confidence", "") for p in pids]
    df["classification_rationale"] = [field(p, "classification_rationale", "") for p in pids]
    df["classification_run_at"] = run_at
    df["classification_prompt_version"] = CLASSIFICATION_PROMPT_VERSION
    # NaN-safe change count (None != None is True in pandas; the no-evidence row is None)
    n_changed = int((df["action_category"].fillna("") != df["action_category_pass1"].fillna("")).sum())
    n_fail = sum(1 for p in pids if p not in skipped and results.get(p) is None)
    if n_fail:
        print(f"[03] classify: {n_fail} call(s) failed — those rows keep their pass-1 category")
    return df, n_changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL_DEFAULT)
    ap.add_argument("--stage", choices=["both", "extract", "classify"], default="both",
                    help="both (default; extract then classify) | extract only | classify only "
                         "(reuse the cached extraction output)")
    ap.add_argument("--pilot", action="store_true", help="stratified pilot sample (representative)")
    ap.add_argument("--sample", type=int, default=0, help="debug: first N clean FONSIs")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel API calls")
    ap.add_argument("--dry-run", action="store_true", help="projected cost only — no key/Keychain, no spend")
    args = ap.parse_args()

    ensure_d6_dirs()
    run_at = utc_now()
    pk = enrich_lib.load_clean_packets()
    if args.pilot:
        ids = set(enrich_lib.pilot_sample()["project_id"])
        pk = pk[pk["project_id"].isin(ids)].reset_index(drop=True)
        suffix = "_pilot"
    elif args.sample:
        pk = pk.head(args.sample)
        suffix = f"_sample{args.sample}"
    else:
        suffix = ""
    raw_out = D6_RAW_DIR / f"fonsi_enrichment_raw{suffix}.parquet"
    clean_out = D6_ANALYSIS_DIR / f"fonsi_enrichment{suffix}.parquet"

    do_extract = args.stage in ("both", "extract")
    do_classify = args.stage in ("both", "classify")
    n = len(pk)
    mode = "pilot" if args.pilot else (f"sample{args.sample}" if args.sample else "full")
    pin, pout = enrich_lib.pricing_for(args.model)

    # --- cost preview (per stage that will run) ---
    if do_extract:
        ext = n * (EST_IN_TOK * pin + EST_OUT_TOK * pout) / 1e6
        print(f"[03] EXTRACT {n} clean FONSIs × {args.model} [{mode}]; projected ~${ext:,.2f} "
              f"(schema={ENRICHMENT_SCHEMA_VERSION}, {len(ENRICHMENT_FIELDS)} fields)")
    if do_classify:
        clf = n * (EST_CLF_IN * pin + EST_CLF_OUT * pout) / 1e6
        print(f"[03] CLASSIFY {n} FONSIs × {args.model} [{mode}]; projected ~${clf:,.2f} "
              f"(classify_version={CLASSIFICATION_PROMPT_VERSION})")
    if args.dry_run:
        print("[03] --dry-run: no key access, nothing billed.")
        return

    key = enrich_lib.get_anthropic_key()   # env or Keychain — one prompt, cached
    if not key:
        print(f"[03] no key in $ANTHROPIC_API_KEY or Keychain '{enrich_lib.KEYCHAIN_SERVICE}'. Use --dry-run for cost only.")
        return
    client = enrich_lib.make_client(key)   # built-in backoff on 429/529 overloads

    # preflight each stage that will run — verify model id + the right tool before spending
    if do_extract:
        pf = enrich_lib.preflight(args.model, client)
        if pf.get("parsed") is None:
            print(f"[03] EXTRACT PREFLIGHT FAILED for {args.model}: {pf.get('error')}. Aborting (nothing billed).")
            raise SystemExit(2)
    if do_classify:
        pf = enrich_lib.classify_preflight(args.model, client)
        if pf.get("parsed") is None:
            print(f"[03] CLASSIFY PREFLIGHT FAILED for {args.model}: {pf.get('error')}. Aborting (nothing billed).")
            raise SystemExit(2)

    # --- stage 1: extraction (or reuse the cached extraction output) ---
    if do_extract:
        clean_df = run_extraction(pk, args, run_at, client, raw_out, clean_out, suffix, mode)
    else:
        if not clean_out.exists():
            print(f"[03] --stage classify needs an existing {clean_out.name}; run extraction first.")
            raise SystemExit(2)
        clean_df = pd.read_parquet(clean_out)
        print(f"[03] classify-only: loaded {len(clean_df)} rows from {clean_out.name} (extraction reused, $0)")

    # --- stage 2: classification (overwrites action_category) ---
    if do_classify:
        clean_df, n_changed = run_classification(clean_df, args.model, client, run_at, args.workers)
        write_parquet(clean_df, clean_out)
        dist = clean_df["action_category"].value_counts().to_dict()
        print(f"[03] classification done: {n_changed} categories changed from pass 1")
        print(f"[03] action_category dist: {dist}")
        print(f"[03] data -> {clean_out}")


if __name__ == "__main__":
    main()
