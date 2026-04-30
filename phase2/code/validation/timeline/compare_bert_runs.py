"""
Compare a sample BERT run against the full baseline BERT output.

Usage:
    python code/validation/timeline/compare_bert_runs.py \
        --new data/analysis/test50_bert_refactored.parquet \
        --baseline data/analysis/projects_timeline_bert.parquet
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare two BERT timeline outputs")
    parser.add_argument("--new",      required=True,  help="New / refactored BERT parquet")
    parser.add_argument("--baseline", required=True,  help="Baseline BERT parquet to compare against")
    args = parser.parse_args()

    new_path  = Path(args.new)
    base_path = Path(args.baseline)

    if not new_path.exists():
        print(f"ERROR: {new_path} not found")
        return
    if not base_path.exists():
        print(f"ERROR: {base_path} not found")
        return

    new  = pd.read_parquet(new_path)
    base = pd.read_parquet(base_path)

    print(f"New:      {len(new):,} projects   ({new_path.name})")
    print(f"Baseline: {len(base):,} projects   ({base_path.name})")

    cols = ["project_id", "bert_decision_date_final", "bert_initiation_date_final",
            "bert_decision_confidence"]
    merged = new[cols].merge(base[cols], on="project_id", suffixes=("_new", "_base"))
    print(f"\nMatched on project_id: {len(merged):,} projects\n")

    def same(a, b):
        """True when both are equal or both are null (handles NaN/None comparison)."""
        return (a == b) | (a.isna() & b.isna())

    def differ(a, b):
        return ~same(a, b)

    # --- match rates ---
    decision_match   = same(merged.bert_decision_date_final_new,   merged.bert_decision_date_final_base).mean()
    initiation_match = same(merged.bert_initiation_date_final_new, merged.bert_initiation_date_final_base).mean()
    conf_match       = same(merged.bert_decision_confidence_new,   merged.bert_decision_confidence_base).mean()

    print(f"Decision date match:    {decision_match:.1%}")
    print(f"Initiation date match:  {initiation_match:.1%}")
    print(f"Confidence score match: {conf_match:.1%}")

    # --- decision date differences ---
    decision_diffs = merged[
        differ(merged.bert_decision_date_final_new, merged.bert_decision_date_final_base)
    ][["project_id", "bert_decision_date_final_new", "bert_decision_date_final_base",
       "bert_decision_confidence_new", "bert_decision_confidence_base"]]

    print(f"\nDecision date differences ({len(decision_diffs)}):")
    if decision_diffs.empty:
        print("  None")
    else:
        print(decision_diffs.to_string(index=False))

    # --- initiation date differences ---
    init_diffs = merged[
        differ(merged.bert_initiation_date_final_new, merged.bert_initiation_date_final_base)
    ][["project_id", "bert_initiation_date_final_new", "bert_initiation_date_final_base"]]

    print(f"\nInitiation date differences ({len(init_diffs)}):")
    if init_diffs.empty:
        print("  None")
    else:
        print(init_diffs.to_string(index=False))


if __name__ == "__main__":
    main()
