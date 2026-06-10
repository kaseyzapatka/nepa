import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")

from pathlib import Path

import pandas as pd


LAB = (
    Path(__file__).resolve().parents[3]
    / "phase2/training/deliverable04/classifier.csv"
)
df = pd.read_csv(LAB)
before = len(df)
keep = df[df["label"].fillna("").str.strip() != ""].copy()
keep.to_csv(LAB, index=False)
print(f"Dropped {before - len(keep)} blank rows; {len(keep)} labeled rows remain.")
