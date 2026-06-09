"""Gold-rank check (read-only): with the new 3-head model + doc-type gate, do the TRUE ROD / FEIS
candidates rank at the TOP of their project's candidate list?

GRANULARITY-AWARE matching: gold dates are often month/year granularity while candidates are
day-level, so an exact YYYY-MM-DD match silently drops most matches. We match in the gold date's
granularity window (day->YYYY-MM-DD, month->YYYY-MM, year->YYYY) and take the BEST (min) rank among
candidates in that window — i.e. "does a candidate carrying the true date make the top-k shortlist".

Scores: p_dec_cal (ROD, within all project candidates), p_feis_cal (FEIS, within the project's
FEIS-typed candidates), learned_decision_score (the LightGBM ranker, for comparison). Writes nothing.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2" / "data" / "analysis" / "timeline"
AUD = ROOT / "phase2" / "output" / "deliverable04" / "eis_audit"

cand = pd.read_parquet(TL / "timeline_candidates.parquet",
    columns=["candidate_id", "project_id", "process_type", "parsed_date", "candidate_role",
             "document_type_clean", "p_dec_cal", "p_feis_cal", "learned_decision_score"])
eis = cand[cand.process_type == "EIS"].copy()
ds = eis.parsed_date.astype(str)
eis["_d"], eis["_ym"], eis["_y"] = ds.str[:10], ds.str[:7], ds.str[:4]
for c in ["p_dec_cal", "p_feis_cal", "learned_decision_score"]:
    eis[c] = pd.to_numeric(eis[c], errors="coerce")

eis["rank_dec"] = eis.groupby("project_id")["p_dec_cal"].rank(ascending=False, method="min")
eis["rank_rnk"] = eis.groupby("project_id")["learned_decision_score"].rank(ascending=False, method="min")
feis_mask = eis.document_type_clean.astype(str).str.upper().str.strip() == "FEIS"
eis["rank_feis"] = np.nan
eis.loc[feis_mask, "rank_feis"] = eis[feis_mask].groupby("project_id")["p_feis_cal"].rank(
    ascending=False, method="min")
n_by_proj = eis.groupby("project_id").size().to_dict()
nfeis_by_proj = eis[feis_mask].groupby("project_id").size().to_dict()


def _key(date_str: str, gran: str):
    g = str(gran).strip().lower()
    d = str(date_str)[:10]
    if g.startswith("year"):
        return d[:4], "_y"
    if g.startswith("month"):
        return d[:7], "_ym"
    return d[:10], "_d"        # day / unknown -> exact


def match_rows(pid, date_str, gran):
    if not date_str or str(date_str).strip().lower() in ("nan", "none", "", "nat"):
        return None
    key, col = _key(date_str, gran)
    sub = eis[(eis.project_id == pid) & (eis[col] == key)]
    return sub if len(sub) else None


def topk(s, k):
    s = s.dropna()
    return 100 * (s <= k).mean() if len(s) else float("nan")


# ---- ROD ----
rod = pd.read_csv(AUD / "eis_rod_promotion_sample_labeled.csv")
R = []
for _, r in rod.iterrows():
    corr = str(r.gold_is_correct_rod).strip().lower()
    date = r.decision_date if corr == "yes" else r.gold_rod_date
    gran = r.get("decision_date_granularity") if corr == "yes" else r.get("gold_rod_granularity")
    m = match_rows(r.project_id, date, gran)
    if m is not None:
        R.append({"p_dec_cal": m.p_dec_cal.max(),
                  "rank_dec": m.rank_dec.min(), "rank_rnk": m.rank_rnk.min(),
                  "n": n_by_proj.get(r.project_id)})
R = pd.DataFrame(R)

print("=== GOLD-RANK CHECK (3-head + gate, granularity-aware matching) ===")
print(f"\n--- ROD: true ROD candidate located in pool: {len(R)} ---")
if len(R):
    print(f"p_dec_cal of true ROD: median {R.p_dec_cal.median():.3f} | "
          f">=0.5 {100*(R.p_dec_cal>=0.5).mean():.0f}% | >=0.7 {100*(R.p_dec_cal>=0.7).mean():.0f}%")
    print(f"CLASSIFIER rank (p_dec_cal): #1 {topk(R.rank_dec,1):.0f}% | top-3 {topk(R.rank_dec,3):.0f}% | "
          f"top-5 {topk(R.rank_dec,5):.0f}% (median rank {R.rank_dec.median():.0f}, median n {R.n.median():.0f})")
    print(f"RANKER rank (learned_decision_score): #1 {topk(R.rank_rnk,1):.0f}% | "
          f"top-3 {topk(R.rank_rnk,3):.0f}% | top-5 {topk(R.rank_rnk,5):.0f}%  [stale: trained on OLD scores]")
    print("BASELINE (pre-rebuild): classifier median p_dec_cal 0.047, rank ~6")

# ---- FEIS ----
feis = pd.read_csv(AUD / "eis_feis_sample_labeled.csv")
F = []
for _, r in feis.iterrows():
    corr = str(r.gold_is_correct_feis).strip().lower()
    date = r.final_eis_date if corr == "yes" else r.gold_feis_date
    gran = r.get("final_eis_date_granularity") if corr == "yes" else r.get("gold_feis_granularity")
    m = match_rows(r.project_id, date, gran)
    if m is not None:
        mf = m[m.rank_feis.notna()]
        F.append({"p_feis_cal": m.p_feis_cal.max(),
                  "rank_feis": mf.rank_feis.min() if len(mf) else np.nan,
                  "nfeis": nfeis_by_proj.get(r.project_id, np.nan),
                  "in_feis_doc": len(mf) > 0})
F = pd.DataFrame(F)
print(f"\n--- FEIS: true FEIS candidate located in pool: {len(F)} ---")
if len(F):
    print(f"true FEIS candidate sits in an FEIS-typed doc: {int(F.in_feis_doc.sum())}/{len(F)}")
    Ff = F[F.rank_feis.notna()]
    print(f"p_feis_cal of true FEIS: median {F.p_feis_cal.median():.3f} | >=0.5 {100*(F.p_feis_cal>=0.5).mean():.0f}%")
    print(f"FEIS rank (p_feis_cal, among FEIS candidates, n={len(Ff)}): #1 {topk(Ff.rank_feis,1):.0f}% | "
          f"top-3 {topk(Ff.rank_feis,3):.0f}% | top-5 {topk(Ff.rank_feis,5):.0f}% "
          f"(median rank {Ff.rank_feis.median():.0f}, median n_feis {Ff.nfeis.median():.0f})")
