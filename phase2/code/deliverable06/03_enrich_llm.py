"""D6 — 03: single comprehensive LLM enrichment pass.  [NEW — SKETCH]

ONE read of every clean-energy EA→FONSI (452) that extracts everything Analyses 1
and 2 need, in one structured (tool-use) call per project — so the paid pass runs
ONCE and never repeats. Prompt + 37-field schema in prompts.py; packet builder,
stratified sampler, call, and quote verification shared via enrich_lib.py.

What it does well now (post-review fixes):
  - BALANCED span-based evidence packet (action/finding/condition/boundary/resource,
    per-section budgets, tagged [S#] with page/document) — the model actually sees
    the finding/mitigation/boundary text, not just truncated action text.
  - tool-use structured output (schema-valid JSON), max_tokens 4096.
  - span-ref citation: every quote cites a provided [S#] and is verified against it.
  - per-row error capture (stop_reason, error, tokens, parse_ok); loud-fail if the
    success rate is low; atomic cache; model preflight before the loop.

KEY (same as D4): $ANTHROPIC_API_KEY if set, else macOS Keychain 'nepa-anthropic'
  (one prompt). --dry-run never touches the key.

OUTPUTS (suffix _pilot / _sampleN for non-full runs)
  RAW:      data/raw/deliverable06/fonsi_enrichment_raw.parquet      (everything + raw + errors)
  ANALYSIS: data/analysis/deliverable06/fonsi_enrichment.parquet     (piped fields + evidence_cited)

Standalone pilot path (intentionally NOT in _run.py). On promotion: renumber 03–08 -> 04–09; wire 04/05/07.

USAGE
  python 03_enrich_llm.py --pilot --dry-run        # stratified pilot cost, no key
  python 03_enrich_llm.py --pilot                  # stratified pilot (one key prompt)
  python 03_enrich_llm.py                          # full 452
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

import enrich_lib
from common import (
    D6_ANALYSIS_DIR, D6_RAW_DIR, D6_REVIEW_DIR, ensure_d6_dirs, sha256_text, utc_now, write_parquet,
)
from prompts import (
    ENRICHMENT_ANALYSIS_COLUMNS,
    ENRICHMENT_FIELDS,
    ENRICHMENT_PROMPT_VERSION,
    ENRICHMENT_SCHEMA_VERSION,
)

CACHE = D6_RAW_DIR / "fonsi_enrichment_cache.json"
LLM_MODEL_DEFAULT = "claude-sonnet-4-6"
EST_IN_TOK, EST_OUT_TOK = 2800, 1200   # dry-run cost estimate only
MIN_SUCCESS_STRICT = 0.9               # pilot/full: tool-use should parse ~always
MIN_SUCCESS_DEBUG = 0.5                # --sample debug mode only


def cached_enrich(text: str, model: str, cache: dict, client, run_at: str) -> tuple[dict, bool]:
    """Return (result, cache_hit). Fresh successes are stamped with llm_run_at and cached,
    so a later full run that reuses a pilot call keeps the ORIGINAL call time."""
    key = sha256_text(f"{ENRICHMENT_PROMPT_VERSION}|{ENRICHMENT_SCHEMA_VERSION}|{model}|{text}")
    if key in cache:
        return cache[key], True
    res = enrich_lib.call_enrichment(text, model, client)
    if res.get("parsed") is not None:
        res["llm_run_at"] = run_at
        cache[key] = res
    return res, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL_DEFAULT)
    ap.add_argument("--pilot", action="store_true", help="stratified pilot sample (representative)")
    ap.add_argument("--sample", type=int, default=0, help="debug: first N clean FONSIs")
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

    n = len(pk)
    mode = "pilot" if args.pilot else (f"sample{args.sample}" if args.sample else "full")
    pin, pout = enrich_lib.pricing_for(args.model)
    cost = n * (EST_IN_TOK * pin + EST_OUT_TOK * pout) / 1e6
    print(f"[03] {n} clean FONSIs × {args.model} [{mode}]; projected ~${cost:,.2f} "
          f"(schema={ENRICHMENT_SCHEMA_VERSION}, {len(ENRICHMENT_FIELDS)} fields)")
    if args.dry_run:
        print("[03] --dry-run: no key access, nothing billed.")
        return

    key = enrich_lib.get_anthropic_key()   # env or Keychain — one prompt, cached
    if not key:
        print(f"[03] no key in $ANTHROPIC_API_KEY or Keychain '{enrich_lib.KEYCHAIN_SERVICE}'. Use --dry-run for cost only.")
        return
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    pf = enrich_lib.preflight(args.model, client)   # verify model id + tool-use before the loop
    if pf.get("parsed") is None:
        print(f"[03] PREFLIGHT FAILED for {args.model}: {pf.get('error')}. Aborting (nothing else billed).")
        raise SystemExit(2)
    print(f"[03] preflight OK ({args.model}); enriching {n} projects ...")

    spans_by_pid, main_by_doc = enrich_lib.load_spans_and_main(pk["project_id"])
    cache: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    rows, ok, skipped = [], 0, 0
    for r in pk.itertuples(index=False):
        packet_text, tag_map = enrich_lib.build_evidence_packet(r, spans_by_pid.get(r.project_id, pd.DataFrame()))
        tech = str(getattr(r, "tech_group", ""))
        if not tag_map:                              # no spans AND no typed text -> don't pay for metadata-only
            skipped += 1
            rec = {"project_id": r.project_id}
            rec.update({nm: None for nm, _t, _d in ENRICHMENT_FIELDS})
            rec["evidence_cited"] = "[]"
            rec.update({
                "enrichment_extraction_run_at": run_at, "enrichment_llm_run_at": "", "llm_provider": "",
                "llm_model": args.model, "prompt_version": ENRICHMENT_PROMPT_VERSION,
                "schema_version": ENRICHMENT_SCHEMA_VERSION, "input_sha256": sha256_text(packet_text),
                "tech_group": tech, "n_excerpts": 0, "used_fallback": False, "cache_hit": False,
                "parse_ok": False, "stop_reason": "", "llm_error": "no_evidence",
                "in_tokens": 0, "out_tokens": 0, "raw_response": "",
            })
            rows.append(rec)
            continue
        res, cache_hit = cached_enrich(packet_text, args.model, cache, client, run_at)
        parsed = res.get("parsed")
        ok += int(parsed is not None)
        rec = {"project_id": r.project_id}
        rec.update(enrich_lib.coerce(parsed) if parsed is not None
                   else {nm: None for nm, _t, _d in ENRICHMENT_FIELDS})
        cited = enrich_lib.cite_quotes(parsed, tag_map, main_by_doc) if parsed else []
        rec["evidence_cited"] = json.dumps(cited, ensure_ascii=False)
        rec.update({
            "enrichment_extraction_run_at": run_at,   # THIS run
            "enrichment_llm_run_at": (res.get("llm_run_at", run_at) if parsed is not None else ""),  # original call
            "llm_provider": "anthropic" if parsed is not None else "",
            "llm_model": args.model, "prompt_version": ENRICHMENT_PROMPT_VERSION,
            "schema_version": ENRICHMENT_SCHEMA_VERSION, "input_sha256": sha256_text(packet_text),
            "tech_group": tech, "n_excerpts": len(tag_map),
            "used_fallback": "packet-fallback" in packet_text, "cache_hit": cache_hit,
            "parse_ok": parsed is not None, "stop_reason": res.get("stop_reason", ""),
            "llm_error": res.get("error", ""), "in_tokens": res.get("in_tok", 0),
            "out_tokens": res.get("out_tok", 0), "raw_response": res.get("raw", ""),
        })
        rows.append(rec)
    enrich_lib.write_json_atomic(CACHE, cache)

    raw_df = pd.DataFrame(rows)
    write_parquet(raw_df, raw_out)
    keep = ["project_id", *ENRICHMENT_ANALYSIS_COLUMNS, "evidence_cited",
            "enrichment_extraction_run_at", "enrichment_llm_run_at"]
    write_parquet(raw_df[keep], clean_out)

    attempted = n - skipped
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

    min_success = MIN_SUCCESS_DEBUG if args.sample else MIN_SUCCESS_STRICT
    if attempted and ok / attempted < min_success:
        print(f"[03] FAIL: parse rate {ok}/{attempted} below {min_success:.0%} (mode={mode}).")
        sys.exit(1)


if __name__ == "__main__":
    main()
