"""Step 2 (free harvest + classifier-vs-ranker diagnostic), read-only except the harvest CSV.

From the Codex-labeled gold projects, (1) locate the candidate carrying the TRUE ROD date and the
candidate we MIS-PICKED, (2) report their CURRENT classifier (p_dec_cal) and ranker
(learned_decision_score) scores + within-project ranks -> tells us whether the bottleneck is the
classifier (Step 2 retrain) or the ranker/LLM (Step 3), and (3) emit candidate-level labels
(decision / neither) ready to append to labeling_sample.csv (train split).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2" / "data" / "analysis" / "timeline"
AUD = ROOT / "phase2" / "training" / "deliverable04" / "eis_validation"

cand = pd.read_parquet(TL / "timeline_candidates.parquet")
eis = cand[cand.process_type == "EIS"].copy()
eis["_date"] = eis.parsed_date.astype(str).str[:10]
for c in ["p_dec_cal", "p_decision", "learned_decision_score"]:
    eis[c] = pd.to_numeric(eis.get(c), errors="coerce")

# within-project rank (1 = best) by classifier prob and by ranker score
eis["rank_clf"] = eis.groupby("project_id")["p_dec_cal"].rank(ascending=False, method="min")
eis["rank_rnk"] = eis.groupby("project_id")["learned_decision_score"].rank(ascending=False, method="min")
n_by_proj = eis.groupby("project_id").size().to_dict()


def find_cand(pid, date10):
    if not date10 or date10 == "nan":
        return None
    sub = eis[(eis.project_id == pid) & (eis._date == str(date10)[:10])]
    if sub.empty:
        return None
    # prefer a decision-ish role if several share the date
    pref = sub[sub.candidate_role.isin(["clear_decision", "proxy_decision", "body_text"])]
    return (pref if not pref.empty else sub).iloc[0]


rod = pd.read_csv(AUD / "eis_rod_promotion_sample_labeled.csv")
harvest = []      # candidate-level labels
diag = []         # gold-ROD candidate score/rank rows
for _, r in rod.iterrows():
    pid = r.project_id
    corr = str(r.gold_is_correct_rod).strip().lower()
    gold = str(r.gold_rod_date or "").strip()
    gold = "" if gold in ("nan", "none", "") else gold
    # the true ROD candidate (yes -> the selected date; no+gold -> the gold date)
    true_date = r.decision_date if corr == "yes" else gold
    gc = find_cand(pid, true_date) if true_date else None
    if gc is not None:
        harvest.append({"candidate_id": gc.candidate_id, "project_id": pid, "label": "decision",
                        "source": "gold_rod", "candidate_role": gc.candidate_role})
        diag.append({"project_id": pid, "kind": "gold_rod", "role": gc.candidate_role,
                     "p_dec_cal": gc.p_dec_cal, "rank_clf": gc.rank_clf,
                     "learned": gc.learned_decision_score, "rank_rnk": gc.rank_rnk,
                     "n_cands": n_by_proj.get(pid)})
    # the mis-picked distractor (wrong rows) -> hard negative
    if corr == "no":
        mp = find_cand(pid, r.decision_date)
        if mp is not None and (gc is None or mp.candidate_id != gc.candidate_id):
            harvest.append({"candidate_id": mp.candidate_id, "project_id": pid, "label": "neither",
                            "source": "mispick_distractor", "candidate_role": mp.candidate_role})

# FEIS gold candidates -> hard negatives for the ROD/decision head (FEIS-pub != ROD)
feis = pd.read_csv(AUD / "eis_feis_sample_labeled.csv")
for _, r in feis.iterrows():
    gold = str(r.gold_feis_date or "").strip(); gold = "" if gold in ("nan","none","") else gold
    fd = str(r.final_eis_date)[:10]
    fc = find_cand(r.project_id, gold or fd)
    if fc is not None:
        harvest.append({"candidate_id": fc.candidate_id, "project_id": r.project_id,
                        "label": "neither", "source": "feis_pub_not_rod",
                        "candidate_role": fc.candidate_role})

H = pd.DataFrame(harvest).drop_duplicates("candidate_id")
D = pd.DataFrame(diag)
H.to_csv(AUD / "eis_gold_candidate_labels.csv", index=False)

print("=== HARVEST (candidate-level labels for labeling_sample.csv) ===")
print("total labels:", len(H), "| by label:", H.label.value_counts().to_dict())
print("by source:", H.source.value_counts().to_dict())
print()
print("=== CLASSIFIER vs RANKER diagnostic on the TRUE ROD candidates ===")
print(f"true-ROD candidates located: {len(D)}")
if len(D):
    print("p_dec_cal: median %.3f | share >=0.5: %.0f%% | >=0.7: %.0f%%" % (
        D.p_dec_cal.median(), 100*(D.p_dec_cal >= 0.5).mean(), 100*(D.p_dec_cal >= 0.7).mean()))
    print("classifier rank-in-project: #1 %.0f%% | top-3 %.0f%% | top-5 %.0f%% (median rank %.0f, median n_cands %.0f)" % (
        100*(D.rank_clf == 1).mean(), 100*(D.rank_clf <= 3).mean(), 100*(D.rank_clf <= 5).mean(),
        D.rank_clf.median(), D.n_cands.median()))
    print("ranker rank-in-project:     #1 %.0f%% | top-3 %.0f%% | top-5 %.0f%%" % (
        100*(D.rank_rnk == 1).mean(), 100*(D.rank_rnk <= 3).mean(), 100*(D.rank_rnk <= 5).mean()))
    print("true-ROD candidate roles:", D.role.value_counts().to_dict())
D.to_csv(AUD / "eis_gold_rod_scorediag.csv", index=False)
print("\nWrote eis_gold_candidate_labels.csv + eis_gold_rod_scorediag.csv")
