"""Focused re-label: the projects that got 'none' only because the ROD-first rule suppressed the
FEIS fallback (has_rod=True, marked none, but they HAVE FEIS candidates). These have no extractable
ROD date, so per the design the decision falls back to the Final-EIS publication date. Show their
FEIS candidates and let Codex pick the publication/NOA date (or none if none is a genuine pub date).
Picks fold into ranker.csv as verified_feis_fallback. Worksheet is transient (gitignored).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sel05", HERE / "05_select_dates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
TRAINING = m.PHASE2 / "training" / "deliverable04"
LABELED = TRAINING / "_ranker_eis_labeling_batch.csv"
OUT = TRAINING / "_ranker_feis_fallback_batch.csv"
CAP = 12
PUB_RE = m.EIS_FEIS_PUB_RE   # explicit Final-EIS publication / NOA / availability language

# 1) the recoverable projects: has_rod=True, project-level none, has a FEIS candidate
w = pd.read_csv(LABELED, dtype=str, keep_default_na=False)
w["_pick"] = w.gold_pick.str.strip().str.lower().isin({"yes", "y", "true", "1", "x"})
w["_type"] = w.gold_type.str.strip().str.lower()
w["has_rod"] = w.has_rod.str.strip().str.lower().eq("true")
w["dt"] = w.document_type_clean.str.upper().str.strip()
roll = w.groupby("project_id").agg(is_none=("_type", lambda s: s.eq("none").any()),
                                   picked=("_pick", "any"),
                                   has_rod=("has_rod", "first"),
                                   has_feis=("dt", lambda s: (s == "FEIS").any()))
targets = roll[(~roll.picked) & (roll.is_none) & (roll.has_rod) & (roll.has_feis)].index.tolist()
print(f"recoverable projects (has_rod, none, has FEIS): {len(targets)}")

# 2) pull their FEIS-doc candidates from the pool (full set, not the prior cap)
cand = pd.read_parquet(m.CANDIDATES_PATH)
eis = cand[(cand.process_type == "EIS") & cand.project_id.isin(targets)].copy()
eis["dt"] = eis.document_type_clean.astype(str).str.upper().str.strip()
eis = eis[eis.dt == "FEIS"].copy()
eis["p_feis_cal"] = pd.to_numeric(eis.get("p_feis_cal"), errors="coerce").fillna(0.0)
eis["ctx"] = (eis.get("model_context").fillna(eis.get("context_text")) if "model_context" in eis
              else eis.get("context_text")).fillna("").astype(str)
eis["feis_pub_language"] = eis.ctx.str.contains(PUB_RE)
# prioritize explicit publication language, then model score
eis = eis.sort_values(["project_id", "feis_pub_language", "p_feis_cal"], ascending=[True, False, False])
eis["_rk"] = eis.groupby("project_id").cumcount()
eis = eis[eis._rk < CAP]

ws = pd.DataFrame({
    "project_id": eis.project_id,
    "candidate_id": eis.candidate_id,
    "document_type_clean": eis.dt,
    "parsed_date": eis.parsed_date.astype(str).str[:10],
    "date_granularity": eis.date_granularity,
    "candidate_role": eis.candidate_role,
    "feis_pub_language": eis.feis_pub_language,
    "raw_date_text": eis.get("raw_date_text", "").astype(str).str[:60],
    "model_context": eis.ctx.str.replace(r"\s+", " ", regex=True).str.strip().str[:400],
    "p_feis_cal": eis.p_feis_cal.round(3),
    "gold_pick": "",   # 'yes' on the ONE Final-EIS publication/NOA date
    "gold_type": "",   # feis | none
    "gold_notes": "",
})
ws.to_csv(OUT, index=False)
print(f"wrote {OUT.name}: {len(ws)} FEIS candidate rows across {ws.project_id.nunique()} projects "
      f"(median {ws.groupby('project_id').size().median():.0f}/project)")
print(f"  rows with explicit FEIS publication language: {int(ws.feis_pub_language.astype(str).eq('True').sum())}")
