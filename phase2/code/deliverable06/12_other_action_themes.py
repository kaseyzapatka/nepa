"""D6 v2 — 12: within-cell refinement of the 'other' action verb (item #40).

The 10-verb action vocabulary (see 10_action_label.py) can't resolve every
FONSI; 92 candidates land in the catch-all 'other' bucket. This script
clusters just those 92 on LOCAL sentence embeddings (no LLM) to surface the
sub-action themes the vocabulary missed, and gives each cluster a short
c-TF-IDF-style label from its own text.

This is SUPPLEMENTARY analysis only — it must never alter any verdict, rank,
or cell membership from 07_classify_and_rank.py. It reads candidate_facts and
fonsi_enrichment, writes a standalone output, and asserts candidate_verdicts
is byte-identical before and after the run.

Inputs:
  data/analysis/deliverable06/candidate_facts.parquet   (project_id, candidate_category, tech_group, action)
  data/analysis/deliverable06/fonsi_enrichment.parquet  (project_id, action_label_freeform, action_summary, potential_ce_theme)

Output:
  data/analysis/deliverable06/other_action_themes.parquet
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import hashlib

import numpy as np
import pandas as pd

import embeddings
from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now, write_parquet

FACTS = D6_ANALYSIS_DIR / "candidate_facts.parquet"
ENRICHMENT = D6_ANALYSIS_DIR / "fonsi_enrichment.parquet"
VERDICTS = D6_ANALYSIS_DIR / "candidate_verdicts.parquet"
OUT = D6_ANALYSIS_DIR / "other_action_themes.parquet"
SUMMARY_MAX_CHARS = 400


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_text(row) -> str:
    parts = []
    for field in ("action_label_freeform", "potential_ce_theme"):
        value = row.get(field)
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    summary = row.get("action_summary")
    if pd.notna(summary) and str(summary).strip():
        parts.append(str(summary).strip()[:SUMMARY_MAX_CHARS])
    return " ".join(parts)


def _choose_k(emb: np.ndarray) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(emb)
    if n < 6:
        return 2
    best_k, best_score = 3, -1.0
    for k in range(3, min(8, n - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb)
        score = silhouette_score(emb, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _cluster_labels(texts: list[str], labels: np.ndarray, top_n: int = 5) -> dict[int, str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    X = vec.fit_transform(texts).toarray()
    vocab = vec.get_feature_names_out()
    out = {}
    for c in sorted(set(labels)):
        mask = labels == c
        mean_vec = X[mask].mean(axis=0)
        order = mean_vec.argsort()[::-1][:top_n]
        terms = [vocab[i] for i in order if mean_vec[i] > 0]
        out[c] = "; ".join(terms)
    return out


def main() -> None:
    if not embeddings.available():
        print("sentence-transformers not available; skipping other-action theme clustering.")
        raise SystemExit(0)

    ensure_d6_dirs()

    verdicts_sha_before = _sha256(VERDICTS)

    facts = pd.read_parquet(FACTS, columns=["project_id", "candidate_category", "tech_group", "action"])
    other = facts.loc[facts["action"] == "other"].drop_duplicates(subset="project_id").copy()

    enrichment = pd.read_parquet(
        ENRICHMENT,
        columns=["project_id", "action_label_freeform", "action_summary", "potential_ce_theme"],
    )

    merged = other.merge(enrichment, on="project_id", how="left")
    merged["cluster_text"] = merged.apply(_build_text, axis=1)

    texts = merged["cluster_text"].tolist()
    emb = np.asarray(embeddings.embed(texts))

    k = _choose_k(emb)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb)
    merged["cluster_id"] = km.labels_

    labels = _cluster_labels(texts, merged["cluster_id"].to_numpy(), top_n=5)
    merged["cluster_label"] = merged["cluster_id"].map(labels)

    run_at = utc_now()
    out = merged[["project_id", "tech_group", "cluster_id", "cluster_label"]].copy()
    out["other_action_extraction_run_at"] = run_at
    out["other_action_llm_run_at"] = ""

    write_parquet(out, OUT)

    print(f"Clustered {len(merged)} 'other'-action projects into k={k} clusters.")
    for c in sorted(labels):
        sub = merged.loc[merged["cluster_id"] == c]
        tech_mix = sub["tech_group"].value_counts().to_dict()
        print(f"  cluster {c} (n={len(sub)}): {labels[c]}")
        print(f"    tech mix: {tech_mix}")

    verdicts_sha_after = _sha256(VERDICTS)
    assert verdicts_sha_after == verdicts_sha_before, "candidate_verdicts.parquet was modified!"
    print(f"candidate_verdicts.parquet UNCHANGED (sha256 {verdicts_sha_after[:12]})")


if __name__ == "__main__":
    main()
