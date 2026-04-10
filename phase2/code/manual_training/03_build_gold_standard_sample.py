import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

"""
03_build_gold_standard_sample.py

Samples projects per source (CE, EA, EIS) and exports a template CSV for
manual verification of true initiation and decision dates.

The gold standard is used ONLY for evaluation — it never enters training.
After filling it in, run --bert-run and compare output against it to measure
real-world accuracy.

Outputs:
    data/manual_training/gold_standard_template.csv   (template to fill in)
    data/analysis/gold_standard_timelines.csv         (final verified file — you rename/copy here)

Sampling strategy:
    - Mix of timeline_status values to get honest coverage estimate
    - Prefer projects with more dates found (more document content to verify against)
    - Stratify by bert_timeline_status so the sample isn't biased toward easy cases
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_DIR = BASE_DIR / "data" / "manual_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BERT_OUTPUT = ANALYSIS_DIR / "projects_timeline_bert.parquet"
TEMPLATE_FILE = OUTPUT_DIR / "gold_standard_template.csv"
GOLD_STANDARD_FILE = ANALYSIS_DIR / "gold_standard_timelines.csv"

# How many projects per source to sample, and how to split by timeline status
SAMPLE_CONFIG = {
    'CE':  {'n': 30, 'status_weights': {'complete': 0.4, 'missing_initiation': 0.5, 'missing_decision': 0.05, 'no_dates': 0.05}},
    'EA':  {'n': 25, 'status_weights': {'complete': 0.4, 'missing_initiation': 0.3, 'missing_decision': 0.2, 'no_dates': 0.1}},
    'EIS': {'n': 25, 'status_weights': {'complete': 0.4, 'missing_initiation': 0.3, 'missing_decision': 0.2, 'no_dates': 0.1}},
}


def sample_source(df: pd.DataFrame, source: str, config: dict, seed: int = 42) -> pd.DataFrame:
    src = df[df['dataset_source'] == source].copy()
    if src.empty:
        print(f"  No {source} projects found")
        return pd.DataFrame()

    n_total = config['n']
    weights = config['status_weights']
    rng = np.random.default_rng(seed)

    sampled_parts = []
    for status, weight in weights.items():
        n_want = max(1, round(n_total * weight))
        pool = src[src['bert_timeline_status'] == status].sort_values(
            'bert_n_dates_found', ascending=False
        )
        # Take from the top half (more document content = easier to verify)
        top_half = pool.head(max(1, len(pool) // 2))
        n_take = min(n_want, len(top_half))
        if n_take > 0:
            picked = top_half.sample(n=n_take, random_state=seed)
            sampled_parts.append(picked)

    if not sampled_parts:
        return pd.DataFrame()

    result = pd.concat(sampled_parts).drop_duplicates('project_id').head(n_total)
    print(f"  {source}: sampled {len(result)} projects")
    print(f"    Status breakdown: {result['bert_timeline_status'].value_counts().to_dict()}")
    return result


def main(input_file: Path, seed: int):
    print(f"\nLoading BERT output from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  Loaded {len(df):,} projects")

    if 'bert_timeline_status' not in df.columns:
        print("ERROR: bert_timeline_status column not found. Run --bert-run first.")
        return

    rows = []
    for source, config in SAMPLE_CONFIG.items():
        if source not in df['dataset_source'].values:
            print(f"  {source}: not found in dataset, skipping")
            continue
        sampled = sample_source(df, source, config, seed)
        if sampled.empty:
            continue

        for _, proj in sampled.iterrows():
            # Pull BERT's best guesses as a starting point for the reviewer
            bert_init = proj.get('bert_initiation_date_final', '')
            bert_dec = proj.get('bert_decision_date_final', '')

            # Show top candidates from dates_json to help the reviewer
            try:
                dates = json.loads(proj.get('bert_dates_json', '[]'))
            except (json.JSONDecodeError, TypeError):
                dates = []

            init_candidates = [
                f"{d['date']} ({d.get('bert_confidence', 0):.2f}): {str(d.get('context', ''))[:80]}"
                for d in dates if d.get('type') == 'initiation'
            ][:3]

            dec_candidates = [
                f"{d['date']} ({d.get('bert_confidence', 0):.2f}): {str(d.get('context', ''))[:80]}"
                for d in dates if d.get('type') == 'decision'
            ][:3]

            rows.append({
                'project_id': proj['project_id'],
                'source': source,
                'bert_timeline_status': proj.get('bert_timeline_status', ''),
                'bert_n_dates_found': proj.get('bert_n_dates_found', 0),
                'bert_initiation_guess': bert_init,
                'bert_decision_guess': bert_dec,
                'initiation_date_verified': '',   # REVIEWER FILLS IN
                'decision_date_verified': '',      # REVIEWER FILLS IN
                'notes': '',                       # REVIEWER FILLS IN
                'bert_initiation_candidates': ' | '.join(init_candidates),
                'bert_decision_candidates': ' | '.join(dec_candidates),
            })

    if not rows:
        print("No projects sampled.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(TEMPLATE_FILE, index=False)

    print(f"\nSaved {len(out)} projects to: {TEMPLATE_FILE}")
    print()
    print("=== How to fill in the gold standard ===")
    print("1. Open the template CSV")
    print("2. For each project, look up the raw document (use page_viewer_ce.ipynb or page_viewer_ea.ipynb)")
    print("   - Check 'bert_initiation_candidates' and 'bert_decision_candidates' as starting hints")
    print("   - Verify the date against the actual document text")
    print("3. Fill in 'initiation_date_verified' and 'decision_date_verified' (YYYY-MM-DD or leave blank)")
    print("4. Add notes about what you found (e.g., 'DOE initiator signature block page 3')")
    print(f"5. When complete, copy the filled file to: {GOLD_STANDARD_FILE}")
    print()
    print("The gold standard is used for evaluation only — it never enters training.")
    print("Run after each retrain:")
    print("  python code/extract/extract_timeline.py --bert-run --eval-gold --output test_vs_gold.parquet")
    print()
    print(f"Source breakdown:")
    print(out['source'].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build gold standard sample for manual verification")
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_BERT_OUTPUT),
        help=f"Path to BERT output parquet (default: {DEFAULT_BERT_OUTPUT.name})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    args = parser.parse_args()
    main(Path(args.input), args.seed)
