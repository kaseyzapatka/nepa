import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

"""
02_apply_supervision.py

Merges completed review CSVs from data/manual_supervision/ into
data/analysis/manual_training_corrections.csv.

Only rows where 'correct_label' has been filled in are applied.
Deduplicates against existing corrections so it is safe to re-run.

After running:
    python code/extract/extract_timeline.py --bert-generate
    python code/extract/extract_timeline.py --bert-train --source CE   (or EA, EIS)
"""

import pandas as pd
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent.parent.parent
SUPERVISION_DIR = BASE_DIR / "data" / "manual_supervision"
CORRECTIONS_FILE = BASE_DIR / "data" / "analysis" / "manual_training_corrections.csv"
VALID_LABELS    = {"initiation", "decision", "review", "other"}


def main(dry_run: bool = False):
    print("\n=== Applying Manual Supervision Corrections ===\n")

    # Load existing corrections
    if CORRECTIONS_FILE.exists():
        existing = pd.read_csv(CORRECTIONS_FILE)
        existing_keys = set(zip(existing["project_id"], existing["date"]))
        print(f"Existing corrections: {len(existing):,} rows")
    else:
        existing = pd.DataFrame(columns=["project_id", "date", "correct_type", "source_file"])
        existing_keys = set()
        print("No existing corrections file — will create new one")

    # Find all review CSVs with filled correct_label
    review_files = sorted(SUPERVISION_DIR.glob("review_*.csv"))
    if not review_files:
        print(f"No review files found in {SUPERVISION_DIR}")
        return

    new_rows = []
    for path in review_files:
        df = pd.read_csv(path, dtype={"correct_label": str})
        filled = df[df["correct_label"].notna() & (df["correct_label"].str.strip() != "")].copy()
        filled["correct_label"] = filled["correct_label"].str.strip().str.lower()

        invalid = filled[~filled["correct_label"].isin(VALID_LABELS)]
        if not invalid.empty:
            print(f"  WARNING: {len(invalid)} invalid labels in {path.name}: "
                  f"{invalid['correct_label'].unique().tolist()}")
            filled = filled[filled["correct_label"].isin(VALID_LABELS)]

        if filled.empty:
            truly_new = filled.copy()
        else:
            truly_new = filled[
                ~filled.apply(lambda r: (r["project_id"], r["date"]) in existing_keys, axis=1)
            ].copy()
        truly_new["source_file"] = path.name

        dupes = len(filled) - len(truly_new)
        print(f"  {path.name}: {len(filled)} filled → {len(truly_new)} new, {dupes} already present")

        new_rows.append(truly_new[["project_id", "date", "correct_label", "source_file"]]
                        .rename(columns={"correct_label": "correct_type"}))

    if not new_rows or all(r.empty for r in new_rows):
        print("\nNo new corrections to apply.")
        return

    all_new = pd.concat(new_rows, ignore_index=True)
    print(f"\nNew corrections: {len(all_new):,}")
    print(all_new["correct_type"].value_counts().to_string())

    if dry_run:
        print("\n[DRY RUN] No files written. Remove --dry-run to apply.")
        return

    combined = pd.concat([existing, all_new], ignore_index=True)
    combined.to_csv(CORRECTIONS_FILE, index=False)
    print(f"\nSaved {len(combined):,} total corrections → {CORRECTIONS_FILE}")
    print(f"  ({len(existing):,} existing + {len(all_new):,} new)\n")
    print("Next steps:")
    print("  python code/extract/extract_timeline.py --bert-generate")
    print("  python code/extract/extract_timeline.py --bert-train --source CE")
    print("  python code/extract/extract_timeline.py --bert-train --source EA")
    print("  python code/extract/extract_timeline.py --bert-train --source EIS")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
