"""Read-only diagnostic: WHY does the final_eis head separate poorly, and would document_type /
draft-vs-final language fix it? Scores the FROZEN TEST rows through the promoted 3-head model,
then breaks down final_eis errors by document_type_clean, candidate_role, and draft/final cues.
Picks the improvement lever (doc-type gate vs more hard negatives) from evidence. Writes nothing.
"""
from __future__ import annotations
import importlib.util, re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clf04", HERE / "04_classify_candidates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

DRAFT_RE = re.compile(r"\bdraft\b|\bdeis\b|notice of intent|\bnoi\b|scoping|comment period", re.I)
FINAL_RE = re.compile(r"final\s+(?:eis|environmental impact statement)|\bfeis\b|"
                      r"record of decision|\brod\b", re.I)

model, meta = m.load_model(m.MODEL_DIR)
print("model:", meta.get("model_version"), "| label_order:", meta.get("label_order"))

df = m._load_labeled_sample()
test = df[df["split"].eq(m.TEST_SPLIT_VALUE)].reset_index(drop=True)
texts = [m.build_input_text(r) for _, r in test.iterrows()]
prob = m._to_label_probs(model.predict_proba(texts))
p_feis = prob[:, 2] if prob.shape[1] > 2 else np.zeros(len(test))
y = test["label"].eq("final_eis").to_numpy().astype(int)
pred = (p_feis >= 0.5).astype(int)
test = test.assign(p_feis=p_feis, y=y, pred=pred,
                   dt=test["document_type_clean"].astype(str).str.upper().str.strip(),
                   role=test["candidate_role"].astype(str),
                   ctx=test.get("model_context", pd.Series([""]*len(test))).fillna("").astype(str))
test["has_draft"] = test["ctx"].str.contains(DRAFT_RE)
test["has_final"] = test["ctx"].str.contains(FINAL_RE)

print(f"\n=== TEST: {len(test)} rows, final_eis positives = {int(y.sum())} ===")
tp = int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
fn = int(((pred==0)&(y==1)).sum())
print(f"@raw>=0.5: TP={tp} FP={fp} FN={fn}  P={tp/(tp+fp) if tp+fp else 0:.3f}  R={tp/(tp+fn) if tp+fn else 0:.3f}")

# (1) Does document_type_clean ALONE separate final_eis? positive rate by doc-type
print("\n--- (1) final_eis positive RATE by document_type_clean (test) ---")
g = test.groupby("dt").agg(n=("y","size"), feis_pos=("y","sum"))
g["pos_rate"] = (g.feis_pos/g.n).round(3)
print(g.sort_values("feis_pos", ascending=False).to_string())

# (2) FALSE POSITIVES: where the head wrongly fires final_eis (raw>=0.5, y==0)
fps = test[(test.pred==1)&(test.y==0)]
print(f"\n--- (2) FALSE POSITIVES (n={len(fps)}) — what is the head confusing? ---")
print("by document_type_clean:", fps.dt.value_counts().to_dict())
print("by candidate_role:     ", fps.role.value_counts().to_dict())
print(f"FPs in FEIS-typed docs: {int((fps.dt=='FEIS').sum())} / {len(fps)}")
print(f"FPs mentioning DRAFT/scoping cue: {int(fps.has_draft.sum())} / {len(fps)}")
print(f"FPs that are non-FEIS docs:       {int((fps.dt!='FEIS').sum())} / {len(fps)}")

# (3) FALSE NEGATIVES: true final_eis the head missed
fns = test[(test.pred==0)&(test.y==1)]
print(f"\n--- (3) FALSE NEGATIVES (n={len(fns)}) — true final_eis missed ---")
print("by document_type_clean:", fns.dt.value_counts().to_dict())
print("by candidate_role:     ", fns.role.value_counts().to_dict())
print(f"FNs in FEIS-typed docs: {int((fns.dt=='FEIS').sum())} / {len(fns)}")

# (4) Counterfactual: if we GATED on document_type_clean==FEIS, what happens to precision/recall?
feis_docs = test[test.dt=="FEIS"]
print(f"\n--- (4) Counterfactual doc-type GATE (only FEIS-typed candidates eligible) ---")
print(f"FEIS-typed test rows: {len(feis_docs)} | final_eis positives among them: {int(feis_docs.y.sum())} / {int(y.sum())} total")
print(f"  -> doc-type gate RECALL ceiling: {feis_docs.y.sum()/y.sum() if y.sum() else 0:.3f} "
      f"(positives lost to non-FEIS docs: {int(y.sum()-feis_docs.y.sum())})")
if len(feis_docs):
    base = feis_docs.y.mean()
    print(f"  -> within FEIS docs, final_eis base rate: {base:.3f} "
          f"(vs {y.mean():.3f} pool-wide) — higher base rate = easier head job")
    # head precision/recall restricted to FEIS docs
    p2 = (feis_docs.p_feis>=0.5).astype(int); y2=feis_docs.y.to_numpy()
    tp2=int(((p2==1)&(y2==1)).sum()); fp2=int(((p2==1)&(y2==0)).sum()); fn2=int(((p2==0)&(y2==1)).sum())
    print(f"  -> head @0.5 WITHIN FEIS docs: P={tp2/(tp2+fp2) if tp2+fp2 else 0:.3f} "
          f"R={tp2/(tp2+fn2) if tp2+fn2 else 0:.3f} (tp={tp2} fp={fp2} fn={fn2})")

# (5) ranking view: where do true positives land by raw score rank (all test)
print("\n--- (5) raw p_feis distribution: positives vs negatives ---")
for name, mask in [("positives (y=1)", y==1), ("negatives (y=0)", y==0)]:
    s = test.loc[mask, "p_feis"]
    print(f"  {name:16s} median={s.median():.3f}  q75={s.quantile(.75):.3f}  "
          f"max={s.max():.3f}  share>=0.5={ (s>=0.5).mean():.3f}")
