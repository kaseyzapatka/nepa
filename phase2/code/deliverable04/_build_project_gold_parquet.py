import os
from pathlib import Path

import pandas as pd


if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")


ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "phase2/output/deliverable04/project_gold_sample.csv"
CANDIDATES = ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet"
OUTPUT = (
    ROOT
    / "phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet"
)


def main():
    sample = pd.read_csv(SAMPLE, dtype=str, keep_default_na=False)
    sample = sample[
        sample["initiation_candidate_id"].astype(str).str.strip().ne("")
    ].copy()
    candidates = pd.read_parquet(CANDIDATES)
    candidate_map = candidates.set_index("candidate_id")[
        ["project_id", "parsed_date", "date_granularity"]
    ].to_dict("index")

    def lookup(row, column):
        candidate_id = str(row[column]).strip()
        if candidate_id == "none":
            return None, None
        candidate = candidate_map.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Missing candidate_id: {candidate_id}")
        if candidate["project_id"] != row["project_id"]:
            raise ValueError(
                f"{candidate_id} belongs to {candidate['project_id']}, "
                f"not {row['project_id']}"
            )
        return candidate["parsed_date"], candidate["date_granularity"]

    initiation = [
        lookup(row, "initiation_candidate_id")
        for _, row in sample.iterrows()
    ]
    decision = [
        lookup(row, "decision_candidate_id")
        for _, row in sample.iterrows()
    ]
    gold = pd.DataFrame(
        {
            "project_id": sample["project_id"],
            "process_type": sample["process_type"],
            "gold_initiation_date": [date for date, _ in initiation],
            "gold_initiation_granularity": [
                granularity for _, granularity in initiation
            ],
            "gold_decision_date": [date for date, _ in decision],
            "gold_decision_granularity": [
                granularity for _, granularity in decision
            ],
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(gold)} gold projects -> {OUTPUT}")


if __name__ == "__main__":
    main()
