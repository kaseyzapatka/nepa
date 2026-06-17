#!/usr/bin/env python
"""Generate labeling worksheets for the night-2 retrain (D4).

Reads the CURRENT candidate pool from MAIN and the existing classifier.csv, EXCLUDES every
candidate_id already labeled (append-safety: we only ever add brand-new rows), and emits one
worksheet CSV per category into the worktree `labeling/` dir for agents to fill.

Categories: active (uncertainty), neither (hard negatives), final_eis, init (targeted EA/EIS),
and a project-level ranker worksheet. A candidate appears in at most one worksheet.
"""
import duckdb, pandas as pd, re
from pathlib import Path

MAIN_CAND = "/Users/Dora/git/consulting/nepa/phase2/data/analysis/timeline/timeline_candidates.parquet"
WT = Path("/Users/Dora/git/consulting/nepa-night/phase2")
CLS = WT / "training/deliverable04/classifier.csv"
RANK = WT / "training/deliverable04/ranker.csv"
FROZEN = WT / "training/deliverable04/frozen_eval_ids.txt"
OUT = WT / "labeling"; OUT.mkdir(exist_ok=True)

con = duckdb.connect()
existing = set(pd.read_csv(CLS)["candidate_id"].astype(str))
ranked = set(pd.read_csv(RANK)["project_id"].astype(str))
frozen = {ln.strip() for ln in FROZEN.read_text().splitlines() if ln.strip()}
print(f"already-labeled candidates: {len(existing)} | ranker projects: {len(ranked)} | frozen eval: {len(frozen)}")

COLS = """candidate_id, project_id, process_type, candidate_role, role_confidence_score,
  parsed_date, date_granularity, document_type_clean, heading_title, raw_date_text,
  model_context, p_init_cal, p_dec_cal, p_feis_cal, context_text"""

def fetch(where, limit, order="ORDER BY random()"):
    df = con.execute(f"SELECT {COLS} FROM '{MAIN_CAND}' WHERE {where} {order} LIMIT {limit*3}").df()
    df = df[~df["candidate_id"].astype(str).isin(existing)]
    return df.head(limit)

claimed = set()
def emit(name, df, category):
    df = df[~df["candidate_id"].astype(str).isin(claimed)].copy()
    claimed.update(df["candidate_id"].astype(str))
    df["stratum"] = category
    df["label"] = ""           # agent fills: initiation | decision | neither | final_eis
    df["evidence_span"] = ""   # agent MUST quote the phrase justifying the label
    df.to_csv(OUT / f"worksheet_{name}.csv", index=False)
    print(f"  worksheet_{name}.csv: {len(df)} rows")

# 1. targeted init (EA/EIS, init-ish, low p_init_cal — the bimodal-low cohort)
emit("init", fetch(
    "process_type IN ('EA','EIS') AND candidate_role IN ('clear_initiation','proxy_initiation') "
    "AND TRY_CAST(p_init_cal AS DOUBLE) < 0.3", 120), "targeted_init")

# 2. neither hard-negatives (FP families: citations, permit expiration, consultation, cover months)
NEG = (r"(expir|valid through|valid until|comment period|protest period|objection period|"
       r"consultation|concurrence|section 106|SHPO|USFWS|et al\.|\(19\d\d\)|\(20\d\d\)|"
       r"intentionally left blank|table of contents|figure |appendix )")
emit("neither", fetch(
    f"regexp_matches(lower(COALESCE(context_text,'')), '{NEG.lower()}') "
    "AND candidate_role IN ('clear_initiation','clear_decision','proxy_initiation','proxy_decision')", 80),
    "neither_hardneg")

# 3. final_eis (EIS candidates in FEIS documents)
emit("final_eis", fetch(
    "process_type='EIS' AND document_type_clean='FEIS'", 40), "final_eis")

# 4. active-learning uncertainty (model unsure: top calibrated prob in the ambiguous band)
emit("active", fetch(
    "GREATEST(COALESCE(TRY_CAST(p_init_cal AS DOUBLE),0), COALESCE(TRY_CAST(p_dec_cal AS DOUBLE),0)) "
    "BETWEEN 0.35 AND 0.65", 150), "active_learning")

# 5. ranker worksheet (project-level): EA/EIS projects not yet in ranker.csv, not frozen-eval,
#    list ALL candidates per project so the agent picks true init + decision candidate_id.
proj = con.execute(f"""
  SELECT DISTINCT project_id, process_type FROM '{MAIN_CAND}'
  WHERE process_type IN ('EA','EIS') ORDER BY random() LIMIT 400""").df()
proj = proj[~proj["project_id"].astype(str).isin(ranked | frozen)].head(60)
pids = "','".join(proj["project_id"].astype(str))
rk = con.execute(f"""SELECT {COLS} FROM '{MAIN_CAND}' WHERE project_id IN ('{pids}')
  ORDER BY project_id, parsed_date""").df()
rk["stratum"] = "ranker"
rk.to_csv(OUT / "worksheet_ranker.csv", index=False)
# template the agent fills (one row per project)
proj["true_initiation_candidate_id"] = ""   # agent fills (or 'none')
proj["true_decision_candidate_id"] = ""
proj.to_csv(OUT / "worksheet_ranker_answers.csv", index=False)
print(f"  worksheet_ranker.csv: {len(rk)} candidate rows across {len(proj)} projects")
print(f"  worksheet_ranker_answers.csv: {len(proj)} project rows to fill")
print(f"TOTAL new candidate-label slots: {len(claimed)}")
