"""D6 — model-comparison harness for the enrichment pass (standalone tool).

Runs the SAME production enrichment (prompts.py / enrich_lib) through multiple Claude
models on the SAME stratified pilot sample, so you can decide whether Sonnet is worth
it over Haiku before the full run. Compares Haiku vs Sonnet by default.

Decision inputs it measures:
  - cost per model (measured tokens) + projected full-452 cost
  - parse-success rate (tool-use should be ~100%)
  - VERIFIED-QUOTE RATE: share of quotes whose span_ref + text verify against the source
  - field fill rate, and Haiku-vs-Sonnet agreement per field (where they diverge)

Outputs (output/deliverable06/review/):
  - d6_enrich_benchmark_comparison.csv : per project × field, each model's value
  - d6_enrich_benchmark_summary.csv    : per model — cost, parse rate, verified-quote rate, fill rate, errors
  - d6_enrich_benchmark_agreement.csv  : per field — agreement between the first two models

Calls the paid API; NOT in _run.py. --dry-run never touches the key/Keychain.

Usage:
  CONDA_DEFAULT_ENV=nepa python benchmark_models.py --dry-run         # cost only, no key
  CONDA_DEFAULT_ENV=nepa python benchmark_models.py                   # stratified pilot, both models
  CONDA_DEFAULT_ENV=nepa python benchmark_models.py --sample 8        # debug: first 8 (not stratified)
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

import enrich_lib
from common import D6_REVIEW_DIR, ensure_d6_dirs, utc_now
from prompts import ENRICHMENT_FIELDS

COMPARISON_OUT = D6_REVIEW_DIR / "d6_enrich_benchmark_comparison.csv"
SUMMARY_OUT = D6_REVIEW_DIR / "d6_enrich_benchmark_summary.csv"
AGREEMENT_OUT = D6_REVIEW_DIR / "d6_enrich_benchmark_agreement.csv"

DEFAULT_MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6"]
N_FULL = 452
EST_IN_TOK, EST_OUT_TOK = 2800, 1200
FIELDS = [f for f, _t, _d in ENRICHMENT_FIELDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--sample", type=int, default=0, help="debug: first N (not stratified)")
    ap.add_argument("--workers", type=int, default=6, help="parallel API calls per model")
    ap.add_argument("--dry-run", action="store_true", help="cost only — no key/Keychain, no spend")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    ensure_d6_dirs()
    run_at = utc_now()
    meta = enrich_lib.load_clean_packets()
    if args.sample:
        sample = meta.head(args.sample); strata = {p: "head" for p in sample["project_id"]}
    else:
        ps = enrich_lib.pilot_sample()
        strata = dict(zip(ps["project_id"], ps["stratum"]))
        sample = meta[meta["project_id"].isin(set(ps["project_id"]))].reset_index(drop=True)
    n = len(sample)
    print(f"[bench] sample={n} ({'head' if args.sample else 'stratified pilot'}); models={models}")
    if not args.sample:
        print("[bench] strata:", pd.Series(list(strata.values())).value_counts().to_dict())

    print("[bench] projected cost per model:")
    for m in models:
        pin, pout = enrich_lib.pricing_for(m)
        per = (EST_IN_TOK * pin + EST_OUT_TOK * pout) / 1e6
        print(f"   {m:28s} ~${per:.4f}/call | sample {n}: ~${per*n:.2f} | full {N_FULL}: ~${per*N_FULL:.2f}")

    if args.dry_run:
        print("\n[bench] --dry-run: no key access, nothing billed.")
        return
    key = enrich_lib.get_anthropic_key()
    try:
        import anthropic  # noqa: F401  (presence check; client made via enrich_lib)
    except ImportError:
        anthropic = None
    if not key or anthropic is None:
        print(f"\n[bench] no key (env/Keychain '{enrich_lib.KEYCHAIN_SERVICE}') or SDK missing. Use --dry-run.")
        return
    client = enrich_lib.make_client(key)   # built-in backoff on 429/529 overloads

    spans_by_pid, main_by_doc = enrich_lib.load_spans_and_main(sample["project_id"])
    packets = {r.project_id: enrich_lib.build_evidence_packet(r, spans_by_pid.get(r.project_id, pd.DataFrame()))
               for r in sample.itertuples(index=False)}
    empty = [pid for pid, (_pt, tm) in packets.items() if not tm]
    if empty:
        print(f"[bench] skipping {len(empty)} zero-evidence project(s): {empty}")
        sample = sample[~sample["project_id"].isin(set(empty))].reset_index(drop=True)
        n = len(sample)

    coerced: dict = {}
    tok = {m: [0, 0] for m in models}
    vq = {m: [0, 0] for m in models}     # [n_quotes, n_verified]
    nok = {m: 0 for m in models}
    errs = {m: [] for m in models}
    for m in models:
        pf = enrich_lib.preflight(m, client)
        if pf.get("parsed") is None:
            print(f"[bench] PREFLIGHT FAILED for {m}: {pf.get('error')} — skipping this model.")
            continue
        print(f"[bench] running {m} on {n} projects (workers={args.workers}) ...")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(enrich_lib.call_enrichment, packets[r.project_id][0], m, client): r.project_id
                    for r in sample.itertuples(index=False)}
            for fut in as_completed(futs):                       # results processed in main thread (safe)
                pid = futs[fut]
                res = fut.result()
                tag_map = packets[pid][1]
                tok[m][0] += res["in_tok"]; tok[m][1] += res["out_tok"]
                if res.get("error"):
                    errs[m].append(res["error"])
                parsed = res.get("parsed")
                if parsed is None:
                    continue
                nok[m] += 1
                coerced[(m, pid)] = enrich_lib.coerce(parsed)
                cited = enrich_lib.cite_quotes(parsed, tag_map, main_by_doc)
                vq[m][0] += len(cited); vq[m][1] += sum(c.get("verified") is True for c in cited)

    comp_rows = []
    for r in sample.itertuples(index=False):
        for f in FIELDS:
            row = {"project_id": r.project_id, "stratum": strata.get(r.project_id, ""), "field": f}
            for m in models:
                v = coerced.get((m, r.project_id), {}).get(f, "")
                row[m] = ("" if v is None else str(v))[:300]
            comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(COMPARISON_OUT, index=False)

    if len(models) >= 2:
        a, b = models[0], models[1]
        agr = [{"field": f, "agreement_rate": round((g[a] == g[b]).mean(), 3), "n": len(g)}
               for f, g in comp.groupby("field")]
        pd.DataFrame(agr).sort_values("agreement_rate").to_csv(AGREEMENT_OUT, index=False)

    summ = []
    for m in models:
        pin, pout = enrich_lib.pricing_for(m)
        sample_cost = (tok[m][0] * pin + tok[m][1] * pout) / 1e6
        filled = sum(coerced.get((m, p), {}).get(f) not in (None, "", "[]", "null")
                     for p in sample["project_id"] for f in FIELDS)
        summ.append({
            "model": m, "parse_ok": nok[m], "n": n, "parse_rate": round(nok[m] / max(n, 1), 3),
            "avg_in_tok": round(tok[m][0] / max(n, 1)), "avg_out_tok": round(tok[m][1] / max(n, 1)),
            "sample_cost_usd": round(sample_cost, 2),
            "projected_full_452_usd": round(sample_cost * (N_FULL / max(n, 1)), 2),
            "verified_quote_rate": round(vq[m][1] / max(vq[m][0], 1), 3), "n_quotes": vq[m][0],
            "field_fill_rate": round(filled / max(n * len(FIELDS), 1), 3), "n_errors": len(errs[m]),
        })
    summary = pd.DataFrame(summ)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n[bench] SUMMARY (decision inputs):")
    print(summary[["model", "parse_rate", "projected_full_452_usd", "verified_quote_rate",
                   "field_fill_rate", "n_errors"]].to_string(index=False))
    print(f"\n[bench] comparison -> {COMPARISON_OUT}")
    print(f"[bench] agreement  -> {AGREEMENT_OUT}  (low-agreement fields = where models diverge)")
    print(f"[bench] summary    -> {SUMMARY_OUT}")
    print(f"[bench] run_at={run_at}. Decision: take the cheaper model unless its parse/verified-quote rate "
          "or key-field agreement is materially worse.")


if __name__ == "__main__":
    main()
