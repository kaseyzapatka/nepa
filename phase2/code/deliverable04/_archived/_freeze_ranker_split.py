"""Guardrail setup (one-off): FREEZE ranker.csv's train/test split and write the frozen-eval
registry. Prevents the train/eval leak that the ad-hoc gold-rank check exposed.

- Reserve ~30% of the VERIFIED EIS picks (notes ~ 'verified', stratified ROD vs FEIS) as the
  protected evaluation set -> training/deliverable04/frozen_eval_ids.txt. These are NEVER trained.
- Assign a FROZEN stratified 80/20 split (seeded) to all gold projects, then force the frozen-eval
  ids to `test`. Persist into ranker.csv's `split` column so 05b stops re-randomizing every run.
Idempotent-ish: re-running re-derives the same split (seeded) and the same reserved ids.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sel05", HERE / "05_select_dates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
TRAINING_DIR = m.PHASE2 / "training" / "deliverable04"
RANKER = TRAINING_DIR / "ranker.csv"
FROZEN_EVAL = TRAINING_DIR / "frozen_eval_ids.txt"
SEED, EVAL_FRAC, TEST_FRAC = 42, 0.30, 0.20

g = pd.read_csv(RANKER, dtype=str, keep_default_na=False)
notes = g["notes"].fillna("").astype(str)
verified = g[notes.str.contains("verified")].copy()
verified["_kind"] = notes[verified.index].str.contains("feis").map({True: "feis", False: "rod"})
print(f"ranker.csv: {len(g)} projects | verified EIS picks: {len(verified)} "
      f"({(verified._kind=='rod').sum()} ROD, {(verified._kind=='feis').sum()} FEIS)")

# (1) reserve ~30% of verified, stratified by ROD/FEIS -> frozen eval
eval_ids = []
for kind, grp in verified.groupby("_kind"):
    n = max(1, round(len(grp) * EVAL_FRAC))
    eval_ids += grp.sample(n=n, random_state=SEED).project_id.tolist()
eval_ids = set(eval_ids)
print(f"reserved frozen-eval projects: {len(eval_ids)} "
      f"(never trained; the protected gold-rank-check yardstick)")

# (2) frozen stratified 80/20 by process, seeded; then force eval ids -> test
g["split"] = "train"
for _proc, grp in g.groupby("process_type"):
    n_test = max(1, round(len(grp) * TEST_FRAC))
    test_idx = grp.sample(n=min(n_test, len(grp) - 1) if len(grp) > 1 else 0,
                          random_state=SEED).index
    g.loc[test_idx, "split"] = "test"
g.loc[g.project_id.isin(eval_ids), "split"] = "test"   # frozen eval is always held out

g.to_csv(RANKER, index=False)
FROZEN_EVAL.write_text("\n".join(sorted(eval_ids)) + "\n")

print(f"\nfrozen split written to ranker.csv: {g.split.value_counts().to_dict()}")
print("split x process:")
print(g.pivot_table(index="process_type", columns="split", aggfunc="size", fill_value=0).to_string())
print(f"\nwrote {FROZEN_EVAL.name} ({len(eval_ids)} ids)")
# sanity: every eval id is test, none is train
leak = g[(g.project_id.isin(eval_ids)) & (g.split != "test")]
print(f"sanity — eval ids not in test (should be 0): {len(leak)}")
