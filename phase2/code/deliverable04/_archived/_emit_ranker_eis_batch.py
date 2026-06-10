"""Emit a COMPACT worksheet for Codex to pick the true EIS decision candidate per project.
The picks fold back into ranker.csv (project-level gold) — this worksheet is a transient vehicle,
NOT a new dataset. One row per plausible decision candidate; Codex marks the one true decision per
project (or 'none'). Plausible = ROD/FEIS-typed doc OR decision-role OR p_dec_cal/p_feis_cal>=0.3,
capped per project. Targets EIS projects in ranker.csv that lack a VERIFIED pick, plus a few new ones.
"""
from __future__ import annotations
import importlib.util, re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sel05", HERE / "05_select_dates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
TRAINING = m.PHASE2 / "training" / "deliverable04"
RANKER = TRAINING / "ranker.csv"
OUT = TRAINING / "_ranker_eis_labeling_batch.csv"
SEED, TARGET_PROJECTS, CAP = 42, 120, 10
DEC_ROLES = ["clear_decision", "proxy_decision", "body_text"]
ROD_LANG = re.compile(r"record of decision|\brod\b", re.I)

cand = pd.read_parquet(m.CANDIDATES_PATH)
eis = cand[cand.process_type == "EIS"].copy()
for c in ["p_dec_cal", "p_feis_cal", "p_final_eis"]:
    eis[c] = pd.to_numeric(eis.get(c), errors="coerce").fillna(0.0)
eis["dt"] = eis.document_type_clean.astype(str).str.upper().str.strip()
eis["ctx"] = (eis.get("model_context").fillna(eis.get("context_text")) if "model_context" in eis
              else eis.get("context_text")).fillna("").astype(str)

# project-level has_rod (register ROD / ROD-typed doc / explicit ROD language)
def proj_has_rod(g):
    reg = (g.candidate_source_type.astype(str).eq("metadata") & g.candidate_role.eq("clear_decision"))
    return bool((reg | g.dt.eq("ROD") | g.ctx.str.contains(ROD_LANG)).any())
has_rod = {pid: proj_has_rod(g) for pid, g in eis.groupby("project_id")}

# which EIS projects to label: ranker rows without a verified pick, + new EIS to top up
g = pd.read_csv(RANKER, dtype=str, keep_default_na=False)
eis_rank = g[g.process_type == "EIS"]
verified = set(eis_rank[eis_rank.notes.str.contains("verified", na=False)].project_id)
need = [p for p in eis_rank.project_id if p not in verified]            # unverified already in ranker
pool_eis = [p for p in eis.project_id.unique() if p not in set(g.project_id)]
rng = np.random.default_rng(SEED)
topup = list(rng.permutation(pool_eis))[: max(0, TARGET_PROJECTS - len(need))]
targets = (need + topup)[:TARGET_PROJECTS]
print(f"label targets: {len(targets)} EIS projects ({len(need)} unverified-in-ranker + {len(topup)} new)")

# plausible decision candidates per target project, capped
elig = eis[eis.project_id.isin(targets)].copy()
elig = elig[elig.dt.isin(["ROD", "FEIS"]) | elig.candidate_role.isin(DEC_ROLES)
            | (elig.p_dec_cal >= 0.3) | (elig.p_feis_cal >= 0.3)]
elig["_dtrank"] = elig.dt.map({"ROD": 0, "FEIS": 1}).fillna(2)
elig = elig.sort_values(["project_id", "_dtrank", "p_dec_cal", "p_feis_cal"],
                        ascending=[True, True, False, False])
elig["_rk"] = elig.groupby("project_id").cumcount()
elig = elig[elig._rk < CAP]

ws = pd.DataFrame({
    "project_id": elig.project_id,
    "has_rod": elig.project_id.map(has_rod),
    "candidate_id": elig.candidate_id,
    "document_type_clean": elig.dt,
    "parsed_date": elig.parsed_date.astype(str).str[:10],
    "date_granularity": elig.date_granularity,
    "candidate_role": elig.candidate_role,
    "raw_date_text": elig.get("raw_date_text", "").astype(str).str[:60],
    "model_context": elig.ctx.str.replace(r"\s+", " ", regex=True).str.strip().str[:400],
    "p_dec_cal": elig.p_dec_cal.round(3),
    "p_feis_cal": elig.p_feis_cal.round(3),
    # blanks for Codex:
    "gold_pick": "",        # mark 'yes' on the ONE true decision candidate for the project
    "gold_type": "",        # rod | feis | none
    "gold_notes": "",
})
ws.to_csv(OUT, index=False)
print(f"wrote {OUT.name}: {len(ws)} candidate rows across {ws.project_id.nunique()} projects "
      f"(median {ws.groupby('project_id').size().median():.0f} candidates/project)")
print(f"projects WITH a ROD available: {sum(has_rod[p] for p in targets if p in has_rod)} / {len(targets)}")
print("doc-type mix in worksheet:", ws.document_type_clean.value_counts().to_dict())
