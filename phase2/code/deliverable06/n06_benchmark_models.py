"""D6 v2 — n06: Claude model benchmark / validation harness.

Runs the SAME production extraction prompt (from n03) through multiple Claude
models on a small sample, so you can determine the **lowest-cost model that
clears the accuracy bar** before committing to a full run.

What it produces:
  - d6_model_benchmark_comparison.csv : side-by-side model outputs + deterministic
    baseline, per project × field (for human eyeballing).
  - d6_model_benchmark_cost.csv       : measured tokens + $ per model, with the
    projected cost to run all candidate / full corpus.
  - d6_model_benchmark_scores.csv     : (only if --gold given) per-model per-field
    accuracy vs. human labels, and the lowest model meeting --threshold.

This **calls the paid API** and is intentionally NOT part of `_run.py`. Without
ANTHROPIC_API_KEY (or the anthropic SDK) it runs `--dry-run`: it prints the
sample + projected cost and writes nothing billable.

Usage:
  CONDA_DEFAULT_ENV=nepa python n06_benchmark_models.py --sample 15
  CONDA_DEFAULT_ENV=nepa python n06_benchmark_models.py --sample 20 --gold gold_labels.csv
"""

import argparse
import json
import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now
from n03_extract_candidate_facts import build_facts_prompt

PACKETS = D6_ANALYSIS_DIR / "candidate_evidence_packets.parquet"
FACTS = D6_ANALYSIS_DIR / "candidate_facts.parquet"
COMPARISON_OUT = D6_OUTPUT_DIR / "d6_model_benchmark_comparison.csv"
COST_OUT = D6_OUTPUT_DIR / "d6_model_benchmark_cost.csv"
SCORES_OUT = D6_OUTPUT_DIR / "d6_model_benchmark_scores.csv"

DEFAULT_MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
# input, output $ per 1M tokens (claude-api skill table, cached 2026-05-26 — verify)
PRICING = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-6": (3.0, 15.0), "claude-opus-4-8": (5.0, 25.0)}

NUMERIC_FIELDS = ["max_acres", "max_miles", "max_megawatts"]
BOOL_FIELDS = ["within_existing_row", "no_new_access_road", "previously_disturbed_land"]
CAT_FIELDS = ["mitigation_dependence"]
FREE_FIELDS = ["action_definition", "mitigation_summary", "extraordinary_circumstances"]
ALL_FIELDS = NUMERIC_FIELDS + BOOL_FIELDS + CAT_FIELDS + FREE_FIELDS
N_CANDIDATE, N_CORPUS = 293, 452  # for projected-cost scaling


def pricing_for(model: str) -> tuple[float, float]:
    for prefix, rate in PRICING.items():
        if model.startswith(prefix):
            return rate
    return (3.0, 15.0)  # default to Sonnet-tier if unknown


def select_sample(facts: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample favoring the CE-shaped profile projects of real candidates."""
    real = ["transmission_upgrade", "solar", "geothermal_exploration"]
    pool = facts[facts["candidate_category"].isin(real)].copy()
    prof = pool[pool["is_profile_subtype"]]
    base = prof if len(prof) >= n else pool
    per = max(1, n // base["candidate_category"].nunique())
    picks = (base.groupby("candidate_category", group_keys=False)[base.columns.tolist()]
             .apply(lambda g: g.sample(min(len(g), per), random_state=seed)))
    if len(picks) < n:  # top up
        extra = base.loc[~base.index.isin(picks.index)].sample(
            min(n - len(picks), len(base) - len(picks)), random_state=seed)
        picks = pd.concat([picks, extra])
    return picks.drop_duplicates("project_id").head(n)


def call_model(client, model: str, packet_text: str, category: str):
    """Return (parsed_dict_or_None, input_tokens, output_tokens)."""
    prompt = build_facts_prompt(packet_text, category)
    try:
        msg = client.messages.create(model=model, max_tokens=700,
                                     messages=[{"role": "user", "content": prompt}])
        u = msg.usage
        try:
            data = json.loads(msg.content[0].text)
        except (json.JSONDecodeError, IndexError):
            data = None
        return data, int(u.input_tokens), int(u.output_tokens)
    except Exception as exc:  # noqa: BLE001 — record and continue
        print(f"  [warn] {model} failed on one call: {exc}")
        return None, 0, 0


def numeric_ok(pred, gold, tol=0.10) -> bool:
    try:
        p, g = float(pred), float(gold)
    except (TypeError, ValueError):
        return pd.isna(pred) and pd.isna(gold)
    return abs(p - g) <= tol * max(abs(g), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold", default="", help="CSV of human labels (project_id + scorable fields)")
    ap.add_argument("--threshold", type=float, default=0.90, help="accuracy bar for model selection")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ensure_d6_dirs()
    run_at = utc_now()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    facts = pd.read_parquet(FACTS)
    packets = pd.read_parquet(PACKETS).set_index("project_id")
    sample = select_sample(facts, args.sample, args.seed)
    print(f"[n06] sample={len(sample)} projects across {sample['candidate_category'].nunique()} candidates; "
          f"models={models}")
    print(sample["candidate_category"].value_counts().to_string())

    # cost projection (per-call ≈ ~1,650 in + ~400 out)
    print("\n[n06] projected cost per model (per-call ~1,650 in / ~400 out):")
    for m in models:
        cin, cout = pricing_for(m)
        per = 1650 * cin / 1e6 + 400 * cout / 1e6
        print(f"  {m:22s} ~${per:.4f}/call  | sample {len(sample)}: ~${per*len(sample):.2f}  "
              f"| all {N_CANDIDATE}: ~${per*N_CANDIDATE:.2f}  | corpus {N_CORPUS}: ~${per*N_CORPUS:.2f}")

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import anthropic
        sdk_ok = True
    except ImportError:
        sdk_ok = False
    if args.dry_run or not have_key or not sdk_ok:
        print(f"\n[n06] DRY RUN (api_key={have_key}, sdk={sdk_ok}) — nothing billed; "
              f"no model calls made. Re-run with a key to benchmark.")
        return

    client = anthropic.Anthropic()
    rows, cost = [], {m: [0, 0, 0] for m in models}  # model -> [in, out, calls]
    preds = {m: {} for m in models}  # model -> {project_id: data}
    for rec in sample.itertuples(index=False):
        pid, cat = rec.project_id, rec.candidate_category
        text = packets.loc[pid, "action_text"] if pid in packets.index else ""
        det = {f: getattr(rec, f, None) for f in ALL_FIELDS}
        for m in models:
            data, tin, tout = call_model(client, m, text or "", cat)
            cost[m][0] += tin; cost[m][1] += tout; cost[m][2] += 1
            preds[m][pid] = data or {}
        for f in ALL_FIELDS:
            row = {"project_id": pid, "candidate_category": cat, "field": f, "deterministic": det.get(f)}
            for m in models:
                row[m] = (preds[m][pid] or {}).get(f)
            rows.append(row)
    pd.DataFrame(rows).to_csv(COMPARISON_OUT, index=False)
    print(f"\n[n06] side-by-side comparison -> {COMPARISON_OUT}")

    # cost table (measured)
    cost_rows = []
    for m in models:
        tin, tout, n = cost[m]
        cin, cout = pricing_for(m)
        usd = tin * cin / 1e6 + tout * cout / 1e6
        per = usd / max(n, 1)
        cost_rows.append({"model": m, "calls": n, "input_tokens": tin, "output_tokens": tout,
                          "cost_usd": round(usd, 4), "cost_per_call": round(per, 5),
                          "projected_all_candidates": round(per * N_CANDIDATE, 2),
                          "projected_full_corpus": round(per * N_CORPUS, 2)})
    pd.DataFrame(cost_rows).to_csv(COST_OUT, index=False)
    print(f"[n06] measured cost -> {COST_OUT}")
    print(pd.DataFrame(cost_rows)[["model", "cost_per_call", "projected_all_candidates"]].to_string(index=False))

    # accuracy vs gold (optional) → lowest model meeting threshold
    if args.gold and os.path.exists(args.gold):
        gold = pd.read_parquet(args.gold) if args.gold.endswith(".parquet") else pd.read_csv(args.gold)
        gold = gold.set_index("project_id")
        score_rows = []
        for m in models:
            for f in NUMERIC_FIELDS + BOOL_FIELDS + CAT_FIELDS:
                hits = tot = 0
                for pid, gd in gold.iterrows():
                    if f not in gold.columns or pid not in preds[m]:
                        continue
                    pred, g = (preds[m][pid] or {}).get(f), gd[f]
                    tot += 1
                    if f in NUMERIC_FIELDS:
                        hits += numeric_ok(pred, g)
                    else:
                        hits += (str(pred).strip().lower() == str(g).strip().lower())
                if tot:
                    score_rows.append({"model": m, "field": f, "accuracy": round(hits / tot, 3), "n": tot})
        scores = pd.DataFrame(score_rows)
        scores.to_csv(SCORES_OUT, index=False)
        overall = scores.groupby("model")["accuracy"].mean().sort_values()
        print(f"\n[n06] accuracy vs gold -> {SCORES_OUT}")
        print(overall.to_string())
        order = [m for m in models]  # haiku->sonnet->opus assumed cheap->expensive
        passing = [m for m in order if overall.get(m, 0) >= args.threshold]
        rec = passing[0] if passing else f"none meets {args.threshold} (use {overall.idxmax()})"
        print(f"[n06] LOWEST model meeting accuracy >= {args.threshold}: {rec}")
    else:
        print("\n[n06] No --gold provided: eyeball d6_model_benchmark_comparison.csv to judge quality, "
              "or label a few projects and re-run with --gold to auto-pick the lowest sufficient model.")
    print(f"[n06] run_at={run_at}")


if __name__ == "__main__":
    main()
