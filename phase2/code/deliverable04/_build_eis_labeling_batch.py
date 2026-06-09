"""Assemble the EIS training batch and APPEND it to labeling_sample.csv (no new data CSVs).

Two parts, both written into the single canonical labels file:
  HARVEST (label filled, split=train) — free, from data we already have:
    - eis_harvest_rod_register : document-text candidates whose date matches a register-confirmed
      ROD date  -> label `decision`  (authoritative ROD positives, with rich context)
    - eis_harvest_rod_gold     : gold ROD candidates from the validation labeling -> `decision`
    - eis_harvest_distractor   : the mis-picked wrong candidates from the gold -> `neither`
    - eis_harvest_feis_gold    : gold Final-EIS candidates -> `final_eis`  (FEIS-head positives)
  EMIT (label BLANK, split=train) — for Codex to label:
    - eis_emit_decision : ROD-document decision-eligible candidates (the true ROD is almost always
      here) -> Codex marks `decision` (the ROD signing date) vs `neither`
    - eis_emit_feis     : FEIS-document candidates with Final-EIS / Notice-of-Availability language
      -> Codex marks `final_eis` (the FEIS publication date) vs `neither` (draft NOA / scoping / etc.)

Deterministic (seeded); per-project caps for diversity; never re-adds an already-labeled candidate.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2" / "data" / "analysis" / "timeline"
OUT = ROOT / "phase2" / "output" / "deliverable04"
AUD = ROOT / "phase2" / "training" / "deliverable04" / "eis_validation"
LAB = ROOT / "phase2" / "training" / "deliverable04" / "classifier.csv"
SEED = 42
DECISION_EMIT_N, FEIS_EMIT_N = 200, 250
PER_PROJECT_CAP = 6

FEIS_LANG = re.compile(
    r"notice of availability|\bnoa\b|final (eis|environmental impact statement)|\bfeis\b|"
    r"(filed|filing|publish|releas|made available|availability)", re.I)

cand = pd.read_parquet(TL / "timeline_candidates.parquet")
eis = cand[cand.process_type == "EIS"].copy()
eis["_date"] = eis.parsed_date.astype(str).str[:10]
lab = pd.read_csv(LAB)
labeled_ids = set(lab.candidate_id.astype(str))
DEC_ROLES = ["clear_decision", "proxy_decision", "body_text"]


def pick(rows: pd.DataFrame, label: str, stratum: str) -> pd.DataFrame:
    r = rows.copy()
    r["label"], r["stratum"], r["split"], r["notes"] = label, stratum, "train", ""
    return r


harvest_parts, emit_parts = [], []
used: set[str] = set(labeled_ids)


def take(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.candidate_id.astype(str).isin(used)]
    used.update(df.candidate_id.astype(str))
    return df


# ---------- HARVEST ----------
# (a) register-confirmed ROD positives: document-text candidate whose (project,date) == a register
#     ROD candidate's (project,date).
reg = eis[(eis.candidate_source_type.astype(str) == "metadata") & (eis.candidate_role == "clear_decision")]
reg_pairs = set(zip(reg.project_id, reg._date))
doc = eis[eis.candidate_source_type.astype(str) == "document_text"]
reg_pos = take(doc[[(p, d) in reg_pairs for p, d in zip(doc.project_id, doc._date)]])
harvest_parts.append(pick(reg_pos, "decision", "eis_harvest_rod_register"))

# gold ROD / distractors / FEIS — match gold dates to candidates
rodg = pd.read_csv(AUD / "eis_rod_promotion_sample_labeled.csv")
feisg = pd.read_csv(AUD / "eis_feis_sample_labeled.csv")


def match(pid, date10):
    if not date10 or str(date10) == "nan":
        return None
    sub = eis[(eis.project_id == pid) & (eis._date == str(date10)[:10])]
    if sub.empty:
        return None
    pref = sub[sub.candidate_role.isin(DEC_ROLES)]
    return (pref if not pref.empty else sub).iloc[0]


rod_gold_ids, distractor_ids, feis_gold_ids = [], [], []
for _, r in rodg.iterrows():
    corr = str(r.gold_is_correct_rod).strip().lower()
    true_date = r.decision_date if corr == "yes" else str(r.gold_rod_date or "")
    gc = match(r.project_id, true_date)
    if gc is not None and str(gc.candidate_id) not in used:
        rod_gold_ids.append(gc.candidate_id); used.add(str(gc.candidate_id))
    if corr == "no":
        mp = match(r.project_id, r.decision_date)
        if mp is not None and str(mp.candidate_id) not in used:
            distractor_ids.append(mp.candidate_id); used.add(str(mp.candidate_id))
for _, r in feisg.iterrows():
    corr = str(r.gold_is_correct_feis).strip().lower()
    true_date = r.final_eis_date if corr == "yes" else str(r.gold_feis_date or "")
    fc = match(r.project_id, true_date)
    if fc is not None and str(fc.candidate_id) not in used:
        feis_gold_ids.append(fc.candidate_id); used.add(str(fc.candidate_id))

harvest_parts.append(pick(eis[eis.candidate_id.isin(rod_gold_ids)], "decision", "eis_harvest_rod_gold"))
harvest_parts.append(pick(eis[eis.candidate_id.isin(distractor_ids)], "neither", "eis_harvest_distractor"))
harvest_parts.append(pick(eis[eis.candidate_id.isin(feis_gold_ids)], "final_eis", "eis_harvest_feis_gold"))

# ---------- EMIT (blank) ----------
def capped_sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    pool = pool[~pool.candidate_id.astype(str).isin(used)].copy()
    pool = pool.sort_values("candidate_id")
    pool["_rk"] = pool.groupby("project_id").cumcount()
    pool = pool[pool._rk < PER_PROJECT_CAP]
    out = pool.sample(n=min(n, len(pool)), random_state=SEED)
    used.update(out.candidate_id.astype(str))
    return out.drop(columns="_rk")


dec_pool = eis[(eis.document_type_clean.astype(str).str.upper() == "ROD") & eis.candidate_role.isin(DEC_ROLES)]
emit_parts.append(pick(capped_sample(dec_pool, DECISION_EMIT_N), "", "eis_emit_decision"))
feis_pool = eis[(eis.document_type_clean.astype(str).str.upper() == "FEIS")
                & eis.context_text.fillna("").str.contains(FEIS_LANG)]
emit_parts.append(pick(capped_sample(feis_pool, FEIS_EMIT_N), "", "eis_emit_feis"))

# ---------- append to labeling_sample.csv ----------
batch = pd.concat(harvest_parts + emit_parts, ignore_index=True)
batch = batch.reindex(columns=lab.columns, fill_value="")
# re-apply label/stratum/split (reindex may have blanked them if not in lab columns order)
allrows = pd.concat(harvest_parts + emit_parts, ignore_index=True)
for c in ["label", "stratum", "split"]:
    batch[c] = allrows[c].values
combined = pd.concat([lab, batch], ignore_index=True)
combined.to_csv(LAB, index=False)

print("Appended to labeling_sample.csv:")
print(batch.groupby(["stratum", "label"]).size().to_string())
print(f"\nHARVEST (pre-labeled): {sum(len(p) for p in harvest_parts)}")
print(f"EMIT (blank, for Codex): {sum(len(p) for p in emit_parts)}")
print(f"labeling_sample.csv rows: {len(lab)} -> {len(combined)}")
