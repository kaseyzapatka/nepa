"""Salvage a usable 3-head classifier from a mid-run SetFit embedding checkpoint.

The killed --train run left a converged embedding body in a checkpoint dir (the contrastive
loss had plateaued well before the kill). SetFit only fits the classification head AFTER the
embedding phase, so the checkpoint has NO head. This script:
  1. loads the checkpoint body (SentenceTransformer) as a SetFitModel (one-vs-rest),
  2. fits ONLY the sklearn head on the body's embeddings of the train split (no body retraining),
  3. validates on the FROZEN test split and prints per-head P/R/F1 (incl. the new final_eis head),
  4. saves the salvaged model + meta to a SEPARATE dir (does NOT touch production until promoted).

Reuses 04_classify_candidates' own build_input_text / _labels_from_label_col / _head_metrics so
the salvaged model is byte-for-byte consumable by 04b/05/06.
"""
from __future__ import annotations
import importlib.util, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BODY_PTR = ROOT / "phase2" / "logs" / "_salvage_body.txt"
OUT_DIR = ROOT / "phase2" / "models" / "candidate_classifier_salvage"

# import 04_classify_candidates (name isn't a valid identifier) via importlib
spec = importlib.util.spec_from_file_location("clf04", HERE / "04_classify_candidates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

body_dir = Path(BODY_PTR.read_text().strip())
if not body_dir.is_absolute():
    body_dir = ROOT / body_dir
print(f"Salvage body: {body_dir}")
if not body_dir.exists():
    raise SystemExit(f"checkpoint body not found: {body_dir}")

from setfit import SetFitModel

df = m._load_labeled_sample()
is_test = df["split"].eq(m.TEST_SPLIT_VALUE)
tr_df, te_df = df[~is_test], df[is_test]
tr_texts = [m.build_input_text(r) for _, r in tr_df.iterrows()]
tr_labels = m._labels_from_label_col(tr_df["label"])          # (N, 3)
print(f"train rows={len(tr_df)} (init={int(tr_labels[:,0].sum())}, "
      f"decision={int(tr_labels[:,1].sum())}, final_eis={int(tr_labels[:,2].sum())}, "
      f"neither={int((tr_labels.sum(1)==0).sum())}) | frozen test={len(te_df)}")

# 1) load converged body; fresh one-vs-rest head (random until we fit it)
model = SetFitModel.from_pretrained(str(body_dir), multi_target_strategy="one-vs-rest")

# 2) fit HEAD ONLY on the body's embeddings — no contrastive retraining.
# class_weight="balanced" so the rare final_eis class (104 pos / 4423) isn't drowned by `neither`.
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
BALANCED = os.environ.get("SALVAGE_BALANCED", "1") == "1"
print("encoding train texts with checkpoint body ...")
emb = model.model_body.encode(tr_texts, convert_to_numpy=True, show_progress_bar=True)
if BALANCED:
    print("fitting one-vs-rest head with class_weight='balanced' ...")
    model.model_head = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=2000))
    model.model_head.fit(emb, tr_labels)
else:
    print("fitting one-vs-rest head (default weights) ...")
    model.model_head.fit(emb, tr_labels)

# 3) validate on the frozen test split
te_texts = [m.build_input_text(r) for _, r in te_df.iterrows()]
te_true = m._labels_from_label_col(te_df["label"])
te_prob = m._to_label_probs(model.predict_proba(te_texts))
print(f"\npredict_proba output width: {te_prob.shape[1]} (expect 3)")
metrics = m._head_metrics(te_true, te_prob, "SALVAGE Frozen-test validation")

# per-process EIS slice for the heads that matter here
import pandas as pd
te_df = te_df.reset_index(drop=True)
eis_mask = (te_df["process_type"].astype(str).str.upper() == "EIS").to_numpy()
if eis_mask.any():
    f1 = m._f1_per_head(te_true[eis_mask], te_prob[eis_mask])
    npos = te_true[eis_mask].sum(0).astype(int)
    print(f"\nEIS-only frozen test (n={int(eis_mask.sum())}): "
          + "  ".join(f"{h}={f1[i]:.3f}(n={npos[i]})" for i, h in enumerate(m.LABEL_ORDER)))

# 4) save salvaged model + meta to a SEPARATE dir
out_dir = Path(str(OUT_DIR) + ("_balanced" if BALANCED else "_default"))
model.save_pretrained(str(out_dir))
meta = {
    "backend": "setfit",
    "base_model": m.DEFAULT_BASE_MODEL,
    "label_order": m.LABEL_ORDER,
    "salvaged_from_checkpoint": str(body_dir),
    "head_only_fit": True,
    "n_train": int(len(tr_df)),
    "n_test": int(len(te_df)),
    "n_final_eis_pos_train": int(tr_labels[:, 2].sum()),
    "n_final_eis_pos_test": int(te_true[:, 2].sum()),
    "test_metrics": metrics,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "model_version": "salvage_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
}
meta["class_weight_balanced"] = BALANCED
(out_dir / "classifier_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nSaved salvaged model -> {out_dir}\n(NOT promoted to production; review metrics first.)")
