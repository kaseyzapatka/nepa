from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "phase2/output/deliverable04/project_gold_sample.csv"
CORRECTIONS = {
    "aae0f3cb01c6c344a90bd05b4c6e1ac2": "1ede3d745d14419d4f89",
    "d336bfed9f53527e4b9a3caf40c65715": "6f9ac3ecdc5c7ff725c8",
}
INIT_NOTE = "init=none; no qualifying NOI, application, scoping, or CE start candidate"


df = pd.read_csv(SAMPLE, dtype=str, keep_default_na=False)
corrected = 0

for project_id, expected_old in CORRECTIONS.items():
    mask = df["project_id"].eq(project_id)
    if int(mask.sum()) != 1:
        raise ValueError(f"Expected one sample row for {project_id}, found {int(mask.sum())}")

    row_index = df.index[mask][0]
    if df.at[row_index, "split"].strip() == "test":
        raise ValueError(f"Refusing to modify test row {project_id}")

    current = df.at[row_index, "initiation_candidate_id"].strip()
    if current == "none":
        continue
    if current != expected_old:
        raise ValueError(
            f"Unexpected initiation candidate for {project_id}: "
            f"expected {expected_old!r}, found {current!r}"
        )

    notes = df.at[row_index, "notes"]
    decision_note = notes.split("; dec=", maxsplit=1)
    if len(decision_note) != 2:
        raise ValueError(f"Could not preserve decision note for {project_id}: {notes!r}")

    df.at[row_index, "initiation_candidate_id"] = "none"
    df.at[row_index, "notes"] = f"{INIT_NOTE}; dec={decision_note[1]}"
    corrected += 1

if corrected:
    df.to_csv(SAMPLE, index=False)

print(f"Corrected {corrected} project rows")
