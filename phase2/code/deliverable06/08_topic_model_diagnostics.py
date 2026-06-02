import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import json

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, normalize_space, utc_now, write_parquet


PACKETS_PATH = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"
ASSIGNMENTS_PATH = D6_ANALYSIS_DIR / "fonsi_topic_assignments.parquet"
DIAGNOSTICS_PATH = D6_OUTPUT_DIR / "fonsi_topic_diagnostics.csv"

NEPA_STOPWORDS = [
    "environmental", "assessment", "impact", "impacts", "project", "proposed",
    "action", "alternative", "alternatives", "agency", "section", "would",
    "fonsi", "finding", "significant", "nepa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional project-level NMF diagnostics for D6.")
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_d6_dirs()
    run_at = utc_now()
    packets = pd.read_parquet(PACKETS_PATH)
    packets["model_text"] = packets["action_text"].fillna("").map(normalize_space)
    packets = packets.loc[packets["model_text"].str.len().ge(40)].copy()
    if len(packets) < 5:
        raise SystemExit("Need at least five non-empty action packets for NMF diagnostics.")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=min(5, max(2, len(packets) // 20)),
        max_df=0.55,
        max_features=10_000,
        stop_words=NEPA_STOPWORDS,
    )
    matrix = vectorizer.fit_transform(packets["model_text"])
    terms = vectorizer.get_feature_names_out()
    max_k = min(args.max_k, matrix.shape[0] - 1, matrix.shape[1] - 1)
    diagnostics = []
    fitted = {}
    for k in range(args.min_k, max_k + 1):
        model = NMF(n_components=k, random_state=42, max_iter=400, init="nndsvda")
        weights = model.fit_transform(matrix)
        fitted[k] = (model, weights)
        top_terms = [
            [terms[idx] for idx in component.argsort()[-12:][::-1]]
            for component in model.components_
        ]
        diagnostics.append(
            {
                "n_components": k,
                "reconstruction_error": round(float(model.reconstruction_err_), 6),
                "top_terms": json.dumps(top_terms),
                "topic_diagnostic_run_at": run_at,
            }
        )
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(DIAGNOSTICS_PATH, index=False)
    chosen_k = int(diagnostics_df.sort_values("reconstruction_error").iloc[0]["n_components"])
    model, weights = fitted[chosen_k]
    packets["topic_id"] = weights.argmax(axis=1)
    packets["topic_weight"] = weights.max(axis=1)
    packets["n_components"] = chosen_k
    packets["topic_diagnostic_run_at"] = run_at
    write_parquet(
        packets[["project_id", "topic_id", "topic_weight", "n_components", "topic_diagnostic_run_at"]],
        ASSIGNMENTS_PATH,
    )
    print(f"wrote NMF diagnostics for k={args.min_k}..{max_k}; selected k={chosen_k}")


if __name__ == "__main__":
    main()

