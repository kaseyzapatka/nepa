import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet"
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"
QUOTAS = {
    ("EA", "clear_decision"): 350,
    ("EA", "clear_initiation"): 250,
    ("EIS", "clear_decision"): 450,
    ("EIS", "clear_initiation"): 150,
    ("EA", "proxy_decision"): 100,
    ("EIS", "proxy_decision"): 100,
}
SEED = 11


df = pd.read_parquet(CAND)
lab = pd.read_csv(LAB)
labeled = set(lab["candidate_id"])
elig = df[~df["candidate_id"].isin(labeled)]
parts = []
for (proc, role), n in QUOTAS.items():
    pool = elig[
        (elig["process_type"] == proc) & (elig["candidate_role"] == role)
    ]
    if len(pool):
        parts.append(pool.sample(min(n, len(pool)), random_state=SEED))
batch = pd.concat(parts).drop_duplicates("candidate_id")
out = batch.reindex(columns=lab.columns, fill_value="")
out["label"] = ""
out["notes"] = ""
out["split"] = "train"
out["stratum"] = "buildout_eaeis_2026_06"
pd.concat([lab, out], ignore_index=True).to_csv(LAB, index=False)
print(f"Appended {len(out)} blank EA/EIS rows.")
