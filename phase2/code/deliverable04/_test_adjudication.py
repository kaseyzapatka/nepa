#!/usr/bin/env python
"""A/B test harness for 06 adjudication: run the stratified 100-sample through 2 models, compare
their picks + cost. Both runs use --no-apply (never mutate timeline_project_dates.parquet) and
--workers (parallel). The adjudication cache is model-blind, so it's cleared between models.

Usage (needs ANTHROPIC_API_KEY):
  PYTHONPATH=<repo root> conda run -n nepa python code/deliverable04/_test_adjudication.py \
      --models claude-haiku-4-5-20251001 claude-sonnet-4-6 --workers 12

Outputs (output/deliverable04/audit/): test_<model>.parquet per model + test_compare.csv.
My (the agent's) reference adjudication of the same 100 is done separately and folded into the
comparison.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
TL = HERE.parent.parent / "data/analysis/timeline"
ADJ = TL / "timeline_api_adjudications.parquet"
SAMPLE = HERE.parent.parent / "output/deliverable04/test_sample_100.csv"
OUTDIR = HERE.parent.parent / "output/deliverable04/audit"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _key(model: str) -> str:
    for k in ("haiku", "sonnet", "opus"):
        if k in model:
            return k
    return model.replace("/", "_")


def run_model(model: str, workers: int) -> Path:
    if ADJ.exists():
        ADJ.unlink()  # cache is model-blind; clear so this model isn't skipped
    cmd = [sys.executable, str(HERE / "06_adjudicate_llm.py"),
           "--mode", "candidate_adjudication", "--process", "CE", "EA", "EIS",
           "--sample-ids", str(SAMPLE), "--no-apply", "--workers", str(workers), "--model", model]
    print(f"\n=== running {model} (workers={workers}) ===", flush=True)
    subprocess.run(cmd, check=True)
    out = OUTDIR / f"test_{_key(model)}.parquet"
    shutil.copy(ADJ, out)
    ADJ.unlink()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["claude-haiku-4-5-20251001", "claude-sonnet-4-6"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    outs = {m: run_model(m, args.workers) for m in args.models}
    con = duckdb.connect()
    for m, o in outs.items():
        tot = con.execute(f"SELECT SUM(estimated_cost_usd), COUNT(*) FROM '{o}'").fetchone()
        print(f"\n{m}: 100-sample cost ${tot[0]:.4f} ({tot[1]} calls) -> full 11,207 ≈ ${tot[0]*112:.2f}")

    if len(outs) >= 2:
        (ka, oa), (kb, ob) = list(outs.items())[0], list(outs.items())[1]
        df = con.execute(f"""
          SELECT a.project_id, a.process_type,
            a.selected_initiation_candidate_id ai, b.selected_initiation_candidate_id bi,
            a.selected_decision_candidate_id ad, b.selected_decision_candidate_id bd
          FROM '{oa}' a JOIN '{ob}' b USING(project_id)""").df()
        df["init_agree"] = df.ai.fillna("") == df.bi.fillna("")
        df["dec_agree"] = df.ad.fillna("") == df.bd.fillna("")
        print(f"\n=== {_key(ka)} vs {_key(kb)} agreement (n={len(df)}) ===")
        g = df.groupby("process_type").agg(n=("project_id", "size"),
            init_agree=("init_agree", "mean"), dec_agree=("dec_agree", "mean")).round(3)
        print(g.to_string())
        print(f"overall: init {df.init_agree.mean():.3f} | decision {df.dec_agree.mean():.3f}")
        df.to_csv(OUTDIR / "test_compare.csv", index=False)
        print(f"wrote {OUTDIR / 'test_compare.csv'}")


if __name__ == "__main__":
    main()
