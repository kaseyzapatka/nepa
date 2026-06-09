"""Fold Codex-VERIFIED ROD/FEIS true dates into project_gold_sample.csv as decision picks
(de-noise + add FEIS-fallback examples for the 05b ranker). No new CSV — updates the canonical
project gold in place (backed up first).

For each verified row we locate the candidate_id carrying the true date (granularity-aware; among
matches, the most decision-like = highest p_dec_cal for ROD / p_feis_cal for FEIS), then set that
project's decision_candidate_id:
  - ROD gold: yes -> decision_date, no -> gold_rod_date  (the verified ROD signing date)
  - FEIS gold, has_rod=False: yes -> final_eis_date, no -> gold_feis_date  (FEIS-as-decision fallback)
  - FEIS gold, has_rod=True: SKIP (the ROD is the decision, not the FEIS)
Projects already in the gold get their decision pick OVERWRITTEN with verified truth; new projects
are appended (process_type=EIS, initiation blank, split blank -> auto-split on next train).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sel05", HERE / "05_select_dates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
AUD = HERE.parents[2] / "phase2" / "output" / "deliverable04" / "eis_audit"
GOLD = m.OUTPUT_DIR / "project_gold_sample.csv"

cand = pd.read_parquet(m.CANDIDATES_PATH,
    columns=["candidate_id", "project_id", "process_type", "parsed_date", "candidate_role",
             "document_type_clean", "p_dec_cal", "p_feis_cal"])
eis = cand[cand.process_type == "EIS"].copy()
ds = eis.parsed_date.astype(str)
eis["_d"], eis["_ym"], eis["_y"] = ds.str[:10], ds.str[:7], ds.str[:4]
for c in ["p_dec_cal", "p_feis_cal"]:
    eis[c] = pd.to_numeric(eis[c], errors="coerce").fillna(0.0)


def _col(gran):
    g = str(gran).strip().lower()
    return "_y" if g.startswith("year") else ("_ym" if g.startswith("month") else "_d")


def find_cid(pid, date_str, gran, scorecol, prefer_doc):
    if not date_str or str(date_str).strip().lower() in ("nan", "none", "", "nat"):
        return None
    col = _col(gran); key = str(date_str)[:10][:{"_y": 4, "_ym": 7, "_d": 10}[col]]
    sub = eis[(eis.project_id == pid) & (eis[col] == key)]
    if sub.empty:
        return None
    pref = sub[sub.document_type_clean.astype(str).str.upper().eq(prefer_doc)]
    pool = pref if not pref.empty else sub
    return str(pool.loc[pool[scorecol].idxmax(), "candidate_id"])


picks = {}   # project_id -> (decision_candidate_id, note)

rod = pd.read_csv(AUD / "eis_rod_promotion_sample_labeled.csv")
for _, r in rod.iterrows():
    corr = str(r.gold_is_correct_rod).strip().lower()
    date = r.decision_date if corr == "yes" else r.gold_rod_date
    gran = r.get("decision_date_granularity") if corr == "yes" else r.get("gold_rod_granularity")
    cid = find_cid(r.project_id, date, gran, "p_dec_cal", "ROD")
    if cid:
        picks[r.project_id] = (cid, f"verified_rod_{corr}")

feis = pd.read_csv(AUD / "eis_feis_sample_labeled.csv")
n_feis_skip_hasrod = 0
for _, r in feis.iterrows():
    has_rod = str(r.get("has_rod")).strip().lower() in ("true", "1", "yes")
    if has_rod:
        n_feis_skip_hasrod += 1
        continue  # FEIS is not the decision when a ROD exists
    corr = str(r.gold_is_correct_feis).strip().lower()
    date = r.final_eis_date if corr == "yes" else r.gold_feis_date
    gran = r.get("final_eis_date_granularity") if corr == "yes" else r.get("gold_feis_granularity")
    cid = find_cid(r.project_id, date, gran, "p_feis_cal", "FEIS")
    if cid and r.project_id not in picks:   # never override a verified ROD with a FEIS
        picks[r.project_id] = (cid, f"verified_feis_fallback_{corr}")

print(f"verified decision picks located: {len(picks)} "
      f"({sum('rod' in v[1] for v in picks.values())} ROD, "
      f"{sum('feis' in v[1] for v in picks.values())} FEIS-fallback); "
      f"FEIS rows skipped (has_rod=True): {n_feis_skip_hasrod}")

g = pd.read_csv(GOLD, dtype=str, keep_default_na=False)
GOLD.with_suffix(".pre_verified_fold.csv").write_text(g.to_csv(index=False))
g = g.set_index("project_id")
updated, appended = 0, 0
for pid, (cid, note) in picks.items():
    if pid in g.index:
        g.loc[pid, "decision_candidate_id"] = cid
        g.loc[pid, "notes"] = (str(g.loc[pid, "notes"]) + f";{note}").strip(";")
        updated += 1
    else:
        g.loc[pid] = {c: "" for c in g.columns}
        g.loc[pid, ["process_type", "decision_candidate_id", "notes"]] = ["EIS", cid, note]
        appended += 1
g = g.reset_index()
g.to_csv(GOLD, index=False)
print(f"project_gold_sample.csv: overwrote {updated} existing, appended {appended} new -> {len(g)} rows")
eisg = g[g.process_type == "EIS"]
print(f"EIS decision picks now: {(eisg.decision_candidate_id.str.strip().ne('')&eisg.decision_candidate_id.str.strip().ne('none')).sum()} "
      f"(was 21 before fold)")
