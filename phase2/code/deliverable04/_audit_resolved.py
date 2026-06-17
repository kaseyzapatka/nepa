#!/usr/bin/env python
"""Audit the RESOLVED projects that are NOT sent to the LLM (both dates already present).

Register (metadata) picks are authoritative — low audit priority. The risk is the NON-register
picks (document_text / feis_publication / proxy) made deterministically by 05. This script:
  1. breaks down resolved projects by register vs non-register, per process and per slot;
  2. writes a stratified review sample of NON-register resolutions with the SELECTED date and the
     competing same-role candidates (so a reviewer can judge correctness);
  3. prints timeline-length (duration) summary stats by process — the same content as
     fig_d4_duration_summary_intervals / d4_duration_summary.csv.

Usage: PYTHONPATH=<repo root> conda run -n nepa python code/deliverable04/_audit_resolved.py [--n 12]
"""
import argparse
import duckdb
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
TL = HERE.parent.parent / "data/analysis/timeline"
D = str(TL / "timeline_project_dates.parquet")
C = str(TL / "timeline_candidates.parquet")
OUT = HERE.parent.parent / "output/deliverable04/audit"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="review-sample size per process")
    args = ap.parse_args()
    con = duckdb.connect()

    # Resolved = both dates present (these are NOT sent to the LLM).
    con.execute(f"""CREATE TEMP VIEW resolved AS
      SELECT * FROM '{D}'
      WHERE process_type IN ('CE','EA','EIS')
        AND initiation_date IS NOT NULL AND decision_date IS NOT NULL""")

    print("=== Resolved projects: register (authoritative) vs non-register, per slot ===")
    print(con.execute("""
      SELECT process_type,
        COUNT(*) n_complete,
        SUM((initiation_source_type='metadata')::INT) init_register,
        SUM((initiation_source_type<>'metadata')::INT) init_NONregister,
        SUM((decision_source_type='metadata')::INT) dec_register,
        SUM((decision_source_type<>'metadata')::INT) dec_NONregister,
        SUM((initiation_source_type<>'metadata' OR decision_source_type<>'metadata')::INT) any_NONregister
      FROM resolved GROUP BY 1 ORDER BY 1""").df().to_string(index=False))

    # Stratified review sample: complete projects with >=1 NON-register slot.
    samp = con.execute(f"""
      SELECT * FROM (
        SELECT project_id, process_type,
          initiation_date, initiation_source_type, initiation_date_granularity,
          decision_date, decision_source_type, decision_date_granularity,
          ROW_NUMBER() OVER (PARTITION BY process_type ORDER BY random()) rn
        FROM resolved
        WHERE initiation_source_type<>'metadata' OR decision_source_type<>'metadata')
      WHERE rn <= {args.n} ORDER BY process_type, rn""").df()

    rows = []
    for _, r in samp.iterrows():
        pid = r["project_id"]
        for slot, role_in, src, dt in [("initiation", ("clear_initiation", "proxy_initiation"),
                                        r["initiation_source_type"], r["initiation_date"]),
                                       ("decision", ("clear_decision", "proxy_decision"),
                                        r["decision_source_type"], r["decision_date"])]:
            if src == "metadata":
                continue  # register slot is authoritative; only audit the non-register slot
            comp = con.execute(f"""
              SELECT parsed_date, candidate_role, candidate_source_type, date_granularity,
                substr(regexp_replace(coalesce(context_text,''),'\\s+',' ','g'),1,90) ctx
              FROM '{C}' WHERE project_id='{pid}' AND candidate_role IN {role_in}
              ORDER BY (parsed_date={dt!r}) DESC, TRY_CAST(ranking_score AS DOUBLE) DESC LIMIT 5""").df()
            competing = " ||| ".join(
                f"{c.parsed_date}[{c.candidate_role}/{c.candidate_source_type}] {c.ctx}" for c in comp.itertuples())
            rows.append({"project_id": pid, "process_type": r["process_type"], "slot": slot,
                         "selected_date": dt, "selected_source": src,
                         "selected_granularity": r[f"{slot}_date_granularity"],
                         "competing_candidates": competing, "verdict": ""})
    rev = pd.DataFrame(rows)
    rev.to_csv(OUT / "audit_resolved_nonregister.csv", index=False)
    print(f"\nWrote review sample: {OUT / 'audit_resolved_nonregister.csv'} "
          f"({len(rev)} slot-rows across {samp['project_id'].nunique()} projects)")

    print("\n=== Timeline length (duration_days) by process — same content as fig_d4_duration_summary ===")
    print(con.execute("""
      WITH d AS (SELECT process_type,
          date_diff('day', TRY_CAST(initiation_date AS DATE), TRY_CAST(decision_date AS DATE)) dur
        FROM resolved)
      SELECT process_type, COUNT(*) n,
        ROUND(median(dur)) median_days, ROUND(median(dur)/30.4,1) median_months,
        ROUND(quantile_cont(dur,0.10)) p10_days, ROUND(quantile_cont(dur,0.90)) p90_days
      FROM d WHERE dur IS NOT NULL AND dur>=0 GROUP BY 1 ORDER BY 1""").df().to_string(index=False))


if __name__ == "__main__":
    main()
