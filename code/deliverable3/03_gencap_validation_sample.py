# --------------------------
# DELIVERABLE 3: GENERATION CAPACITY VALIDATION SAMPLE
# --------------------------
# Create a stratified sample for manual validation

import argparse
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_DIR = BASE_DIR / "output" / "deliverable3"


def build_sample(sample_per_source=30, seed=42, clean_energy_only=True):
    data_path = ANALYSIS_DIR / "projects_gencap.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing: {data_path}")

    df = pd.read_parquet(data_path)

    if clean_energy_only and 'project_energy_type' in df.columns:
        df = df[df['project_energy_type'] == 'Clean']

    keep_cols = [
        'project_id', 'project_title', 'dataset_source', 'project_type',
        'project_gencap_value', 'project_gencap_unit',
        'project_gencap_energy_value', 'project_gencap_energy_unit',
        'project_gencap_source', 'project_gencap_confidence',
        'project_gencap_context', 'project_gencap_matches',
        'project_gencap_energy_matches'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    samples = []
    for source in ['CE', 'EA', 'EIS']:
        subset = df[df['dataset_source'] == source]
        n = min(sample_per_source, len(subset))
        if n == 0:
            continue
        samples.append(subset.sample(n=n, random_state=seed))

    if samples:
        out = pd.concat(samples, ignore_index=True)
    else:
        out = pd.DataFrame(columns=keep_cols)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gencap_validation_stratified_sample.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved sample: {out_path} ({len(out)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Create stratified validation sample for gencap.")
    parser.add_argument('--n', type=int, default=30, help='Sample size per source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all-projects', action='store_true', help='Include all projects, not just clean energy')

    args = parser.parse_args()

    build_sample(sample_per_source=args.n, seed=args.seed, clean_energy_only=not args.all_projects)


if __name__ == '__main__':
    main()
