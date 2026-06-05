import os
from pathlib import Path

import pandas as pd


if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")

ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet"
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"

QUOTAS = {
    "clear_initiation": 1200,
    "clear_decision": 900,
    "proxy_decision": 400,
    "proxy_initiation": 400,
    "unknown": 300,
    "body_text": 300,
}
SEED = 42

df = pd.read_parquet(CAND)
lab = pd.read_csv(LAB)
labeled = set(lab["candidate_id"])

elig = df[~df["candidate_id"].isin(labeled)].copy()
parts = []
for role, n in QUOTAS.items():
    pool = elig[elig["candidate_role"] == role]
    if len(pool):
        parts.append(pool.sample(min(n, len(pool)), random_state=SEED))

batch = pd.concat(parts).drop_duplicates("candidate_id")

out = batch.reindex(columns=lab.columns, fill_value="")
out["label"] = ""
out["notes"] = ""
out["split"] = "train"
out["stratum"] = "buildout_2026_06"

pd.concat([lab, out], ignore_index=True).to_csv(LAB, index=False)
print(
    f"Appended {len(out)} blank rows (now {len(lab) + len(out)} total). "
    f"Blank to label: {len(out)}."
)
