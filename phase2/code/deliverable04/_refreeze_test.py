"""
D4 — ONE-TIME test-set re-freeze (Step 5 of build_out_training.md).

Re-draws the train/test split over ALL labeled rows now that a real corpus exists (~1k/head,
balanced across CE/EA/EIS). Dissolves the old premature 18/18/118 test and freezes a new,
larger, better-distributed test (`test_v2`). Run ONCE, after labeling is complete. After this,
new labels default to train and the test never grows again.

Methodology note: splitting AFTER labeling is sound — labels were assigned with no model
involved, and we draw the held-out set before the next retrain, so it stays genuinely held out.
"""

import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "phase2/training/deliverable04/classifier.csv"
TEST_FRACTION = 0.20
FLOOR_PER_CELL = 10    # min test rows per process x label cell (cells are all >=200 now, so non-binding)
SEED = 42

df = pd.read_csv(LAB)
df["label"] = df["label"].fillna("").str.strip().str.lower()
labeled = df[df["label"].isin(["initiation", "decision", "neither"])].copy()

# Fresh stratified draw over ALL labeled rows — dissolves the old split entirely.
test_idx = []
for (_proc, _lab), grp in labeled.groupby(["process_type", "label"]):
    n = max(FLOOR_PER_CELL, round(len(grp) * TEST_FRACTION))
    n = min(n, len(grp) - 1) if len(grp) > 1 else 0   # never take a whole cell
    test_idx += grp.sample(n=n, random_state=SEED).index.tolist()

df["split"] = "train"
df.loc[df.index.isin(test_idx), "split"] = "test"
df.to_csv(LAB, index=False)

t = df[df["split"] == "test"]
print("NEW frozen test (test_v2) by process x label:")
print(t.pivot_table(index="process_type", columns="label",
                    values="candidate_id", aggfunc="count", fill_value=0))
print(f"\ntest total: {len(t)} | train total: {int((df['split'] == 'train').sum())}")
print("test label totals:", t["label"].value_counts().to_dict())
