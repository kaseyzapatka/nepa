import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

"""
02_apply_corrections.py

Merges completed review CSVs into data/analysis/manual_training_corrections.csv,
which is consumed by --bert-generate when retraining.

Reads any review CSV that has a non-empty correct_type column (output from
01_find_ce_initiation_candidates.py or manually created).

Outputs:
    data/analysis/manual_training_corrections.csv  (created or appended to)

After running this, retrain with:
    python code/extract/extract_timeline.py --bert-generate
    python code/extract/extract_timeline.py --bert-train --source CE
"""

import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
CORRECTIONS_FILE = ANALYSIS_DIR / "manual_training_corrections.csv"

VALID_TYPES = {'initiation', 'decision', 'review', 'other'}

DEFAULT_REVIEW_FILES = [
    BASE_DIR / "data" / "manual_training" / "review_ce_initiation_candidates.csv",
]


def load_existing_corrections() -> pd.DataFrame:
    if CORRECTIONS_FILE.exists():
        df = pd.read_csv(CORRECTIONS_FILE)
        print(f"  Loaded {len(df):,} existing corrections from {CORRECTIONS_FILE.name}")
        return df
    print(f"  No existing corrections file — will create new one")
    return pd.DataFrame(columns=['project_id', 'date', 'correct_type', 'source_file'])


def main(review_files: list[Path], dry_run: bool):
    print(f"\n=== Applying Manual Corrections ===")

    existing = load_existing_corrections()
    existing_keys = set(zip(existing['project_id'], existing['date']))

    new_rows = []
    for review_path in review_files:
        if not review_path.exists():
            print(f"  SKIP (not found): {review_path}")
            continue

        review = pd.read_csv(review_path, dtype={'correct_type': str})
        filled = review[review['correct_type'].notna() & (review['correct_type'].str.strip() != '')].copy()
        filled['correct_type'] = filled['correct_type'].str.strip().str.lower()

        # Validate type values
        invalid = filled[~filled['correct_type'].isin(VALID_TYPES)]
        if not invalid.empty:
            print(f"  WARNING: {len(invalid)} rows with invalid correct_type values in {review_path.name}:")
            print(f"    {invalid['correct_type'].unique().tolist()}")
            print(f"    Valid values: {sorted(VALID_TYPES)}")
            filled = filled[filled['correct_type'].isin(VALID_TYPES)]

        # Deduplicate against existing corrections
        truly_new = filled[
            ~filled.apply(lambda r: (r['project_id'], r['date']) in existing_keys, axis=1)
        ].copy()
        truly_new['source_file'] = review_path.name

        duplicates = len(filled) - len(truly_new)
        print(f"  {review_path.name}: {len(filled)} filled → {len(truly_new)} new, {duplicates} already in file")

        new_rows.append(truly_new[['project_id', 'date', 'correct_type', 'source_file']])

    if not new_rows:
        print("\nNo new corrections to apply.")
        return

    all_new = pd.concat(new_rows, ignore_index=True)

    print(f"\nNew corrections to add: {len(all_new):,}")
    print(all_new['correct_type'].value_counts().to_string())

    if dry_run:
        print("\n[DRY RUN] No file changes made. Remove --dry-run to apply.")
        print(all_new.to_string(max_rows=20))
        return

    combined = pd.concat([existing, all_new], ignore_index=True)
    combined.to_csv(CORRECTIONS_FILE, index=False)

    print(f"\nSaved {len(combined):,} total corrections to: {CORRECTIONS_FILE}")
    print(f"  ({len(existing):,} existing + {len(all_new):,} new)")
    print()
    print("=== Next steps ===")
    print("Retrain the model with the new corrections:")
    print("  python code/extract/extract_timeline.py --bert-generate")
    print("  python code/extract/extract_timeline.py --bert-train --source CE")
    print("  python code/extract/extract_timeline.py --bert-run --sample 200 --output test_after_corrections.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge review CSVs into manual_training_corrections.csv")
    parser.add_argument(
        "--input", type=str, nargs='+', default=None,
        help="Path(s) to review CSV(s). Defaults to data/manual_training/review_ce_initiation_candidates.csv"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be added without writing any files"
    )
    args = parser.parse_args()

    if args.input:
        files = [Path(f) for f in args.input]
    else:
        files = DEFAULT_REVIEW_FILES

    main(files, args.dry_run)
