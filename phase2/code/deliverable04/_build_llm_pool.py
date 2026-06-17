#!/usr/bin/env python
"""A3: build the LLM-adjudication candidate pool (the 06 input).

For each project flagged route_to_llm, assemble the small, best candidate set the LLM will read:
  - INIT: top-N_INIT by learned_init_score (across ALL regex roles)
  - DECISION: top-N_DEC by learned_decision_score (across ALL regex roles)
  - RULE-INCLUSION: authoritative-but-low-scored candidates are ALWAYS included regardless of rank —
      register dates (metadata), ROD-signature dates, FEIS-publication (cover/NOA) dates, NOI/scoping
      init dates. (The recall audit showed these often rank low by learned score yet are the answer.)
  - DEDUP BY DATE keeping the best-scored instance (never drop a distinct date; free slots for more).
  - Carry date_granularity; cap context to 300 chars.

Ranking by the LEARNED ranker (not classifier prob): audit recall@5 init 93% / dec 78%, @10 dec 86%;
rule-inclusion closes the tail. Output: timeline_llm_pool.parquet (one row per candidate to send).
Run AFTER 05 (needs route_to_llm) and after 05b (needs learned_*_score on all candidates incl. cover).
"""
import argparse, re
import duckdb, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
TL = HERE.parent.parent / "data/analysis/timeline"
CANDS = TL / "timeline_candidates.parquet"
COVER = TL / "timeline_candidates_feiscover.parquet"   # folded in once integrated; optional
PROJ = TL / "timeline_project_dates.parquet"
OUT = TL / "timeline_llm_pool.parquet"

N_INIT, N_DEC, CAP = 5, 10, 300
ROD = r"record of decision.{0,25}(sign|issu|approv|dat)|\brod\b.{0,20}(sign|issu|approv|dat)|(sign|issu|approv)\w*\s+(the\s+)?(rod|record of decision)"
PUB = (r"notice\s+of\s+availability|(?:final\s+(?:eis|environmental\s+impact\s+statement)|feis)\s+(?:was\s+)?"
       r"(?:filed|filing|publish\w*|releas\w*|issu\w*|made\s+available|available)|availability\s+of\s+the\s+final")
NOI = r"notice\s+of\s+intent|\bnoi\b|scoping|applied\s+for|application\s+(was\s+)?(filed|received|submitted)|pre[-\s]?filing|eplanning|nepa\s+register"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-init", type=int, default=N_INIT); ap.add_argument("--n-dec", type=int, default=N_DEC)
    ap.add_argument("--preview", action="store_true"); args = ap.parse_args()
    con = duckdb.connect()
    for n, r in [("rod", ROD), ("pub", PUB), ("noi", NOI)]:
        con.execute(f"CREATE MACRO {n}(t) AS regexp_matches(lower(coalesce(t,'')), '{r}')")
    # candidate pool = base candidates (+ cover candidates if present/scored)
    src = f"SELECT * FROM '{CANDS}'"
    if COVER.exists():
        src += f" UNION ALL BY NAME SELECT * FROM '{COVER}'"
    con.execute(f"CREATE TEMP VIEW c AS {src}")
    # routable projects: prefer route_to_llm flag; fall back to 'not complete but has candidates'
    pcols = con.execute(f"DESCRIBE SELECT * FROM '{PROJ}'").df()['column_name'].tolist()
    if "route_to_llm" in pcols:
        con.execute(f"CREATE TEMP VIEW routable AS SELECT project_id FROM '{PROJ}' WHERE route_to_llm")
    else:
        con.execute(f"""CREATE TEMP VIEW routable AS SELECT project_id FROM '{PROJ}'
          WHERE NOT (initiation_date IS NOT NULL AND decision_date IS NOT NULL)""")

    def slot_pool(scorecol, nkeep, rule_sql):
        # dedup by (project,date) keeping best learned score + richest context; then top-N by score
        # UNION rule-included candidates (always kept). Returns candidate-level rows.
        return con.execute(f"""
        WITH cc AS (SELECT * FROM c WHERE project_id IN (SELECT project_id FROM routable) AND parsed_date IS NOT NULL),
        ded AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY project_id, parsed_date
                   ORDER BY TRY_CAST({scorecol} AS DOUBLE) DESC NULLS LAST, length(coalesce(context_text,'')) DESC) dr
                FROM cc),
        best AS (SELECT * FROM ded WHERE dr=1),                       -- one row per distinct date
        ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY project_id
                     ORDER BY TRY_CAST({scorecol} AS DOUBLE) DESC NULLS LAST) rk FROM best)
        SELECT project_id, candidate_id, parsed_date, date_granularity,
               candidate_role, candidate_source_type, document_type_clean,
               substr(regexp_replace(coalesce(context_text,''),'\\s+',' ','g'),1,{CAP}) context,
               (rk<={nkeep}) AS in_topn, ({rule_sql}) AS rule_incl
        FROM ranked WHERE rk<={nkeep} OR ({rule_sql})""").df()

    init = slot_pool("learned_init_score", args.n_init,
                     "candidate_source_type='metadata' OR noi(context_text)")
    dec = slot_pool("learned_decision_score", args.n_dec,
                    "candidate_source_type='metadata' OR rod(context_text) OR (upper(document_type_clean)='FEIS' AND pub(context_text))")
    init["slot"] = "initiation"; dec["slot"] = "decision"
    pool = pd.concat([init, dec], ignore_index=True).drop_duplicates(["project_id", "slot", "candidate_id"])
    if not args.preview:
        pool.to_parquet(OUT, index=False)
    np = pool["project_id"].nunique()
    print(f"LLM pool: {np} projects | {len(pool)} candidate-rows ({len(init)} init, {len(dec)} dec) "
          f"| avg {len(pool)/max(np,1):.1f}/project | rule-included rows: {int(pool['rule_incl'].sum())}")
    if args.preview:
        ex = pool[pool["project_id"] == pool["project_id"].iloc[0]]
        print(f"\nexample project {ex['project_id'].iloc[0]}:")
        for _, r in ex.iterrows():
            print(f"  [{r['slot'][:3]}] {r['parsed_date']} ({r['date_granularity']}) topN={r['in_topn']} rule={r['rule_incl']} | {r['context'][:70]}")
    else:
        print(f"wrote -> {OUT.name}")

if __name__ == "__main__":
    main()
