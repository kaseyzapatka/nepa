import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from common import (
    D03_CE_CITATIONS,
    D6_ANALYSIS_DIR,
    D6_OUTPUT_DIR,
    D6_RAW_DIR,
    ensure_d6_dirs,
    normalize_space,
    utc_now,
    write_parquet,
)


CE_EXPLORER_URL = "https://ce.permitting.innovation.gov/data/exclusions.json"
TAXONOMY_PATH = D6_ANALYSIS_DIR / "fonsi_archetype_taxonomy.parquet"
SNAPSHOT_PARQUET = D6_ANALYSIS_DIR / "ce_explorer_snapshot.parquet"
CROSSWALK_PARQUET = D6_ANALYSIS_DIR / "ce_crosswalk.parquet"
REVIEW_CSV = D6_OUTPUT_DIR / "ce_crosswalk_review.csv"


def fetch_snapshot(snapshot: Path | None) -> tuple[dict, bytes, str]:
    if snapshot:
        content = snapshot.read_bytes()
        return json.loads(content), content, CE_EXPLORER_URL
    response = requests.get(CE_EXPLORER_URL, timeout=120)
    response.raise_for_status()
    return response.json(), response.content, CE_EXPLORER_URL


def save_snapshot(content: bytes, retrieved_at: str) -> Path:
    target_dir = D6_RAW_DIR / "ce_explorer"
    target_dir.mkdir(parents=True, exist_ok=True)
    date = retrieved_at[:10]
    target = target_dir / f"exclusions_{date}.json"
    if not target.exists():
        target.write_bytes(content)
    return target


def normalized_snapshot(payload: dict, content: bytes, source_url: str, retrieved_at: str) -> pd.DataFrame:
    version = payload.get("version", {})
    content_hash = hashlib.sha256(content).hexdigest()
    records = []
    for item in payload.get("exclusions", []):
        records.append(
            {
                "ce_id": str(item.get("id", "")),
                "structured_id": normalize_space(item.get("structuredID", "")),
                "agency_unit": normalize_space(item.get("unit", "")),
                "agency_name": normalize_space(item.get("longUnit", "")),
                "origin": normalize_space(item.get("origin", "")),
                "canonical_source_url": normalize_space(item.get("originUrl", "")),
                "context": normalize_space(item.get("context", "")),
                "additional_context": normalize_space(item.get("additionalContext", "")),
                "extraordinary_circumstances": normalize_space(item.get("circumstances", "")),
                "ce_description": normalize_space(item.get("exclusion", "")),
                "source_url": source_url,
                "source_version": normalize_space(version.get("version", "")),
                "source_version_date": normalize_space(version.get("date", "")),
                "source_content_sha256": content_hash,
                "retrieved_at": retrieved_at,
            }
        )
    snapshot = pd.DataFrame(records)
    if D03_CE_CITATIONS.exists() and not snapshot.empty:
        citations = pd.read_parquet(D03_CE_CITATIONS)
        counts = (
            citations.assign(
                ce_code_norm=citations["ce_code"].fillna("").map(normalize_space).str.lower()
            )
            .groupby("ce_code_norm")
            .size()
            .rename("d3_project_citation_count")
        )
        snapshot["ce_code_norm"] = snapshot["structured_id"].str.lower()
        snapshot = snapshot.join(counts, on="ce_code_norm")
        snapshot["d3_project_citation_count"] = snapshot["d3_project_citation_count"].fillna(0).astype(int)
    else:
        snapshot["d3_project_citation_count"] = 0
    return snapshot


def tokens(text: object) -> set[str]:
    stop = {
        "and", "or", "of", "the", "a", "an", "to", "for", "in", "on", "with",
        "related", "actions", "activities", "facility", "facilities",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", normalize_space(text).lower())
        if len(token) > 2 and token not in stop
    }


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokens(query)
    text_tokens = tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def embedding_scores(queries: list[str], descriptions: list[str]) -> tuple[np.ndarray, str]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        d = model.encode(descriptions, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(q) @ np.asarray(d).T, ""
    except Exception as exc:
        return np.zeros((len(queries), len(descriptions))), f"{type(exc).__name__}: {exc}"


def match_crosswalk(
    taxonomy: pd.DataFrame, snapshot: pd.DataFrame, *, use_embeddings: bool, top_k: int
) -> pd.DataFrame:
    ce_texts = (
        snapshot["ce_description"].fillna("")
        + " "
        + snapshot["context"].fillna("")
        + " "
        + snapshot["additional_context"].fillna("")
    ).tolist()
    queries = (
        taxonomy["archetype_label"].fillna("")
        + ". "
        + taxonomy["archetype_description"].fillna("")
    ).tolist()
    embeddings, embedding_error = (
        embedding_scores(queries, ce_texts)
        if use_embeddings else (np.zeros((len(queries), len(ce_texts))), "skipped")
    )
    records = []
    for tax_idx, archetype in taxonomy.reset_index(drop=True).iterrows():
        query = queries[tax_idx]
        candidates = []
        for ce_idx, ce in snapshot.reset_index(drop=True).iterrows():
            lexical = lexical_score(query, ce_texts[ce_idx])
            cosine = float(embeddings[tax_idx, ce_idx])
            retrieval = 0.65 * lexical + 0.35 * max(cosine, 0.0)
            candidates.append((retrieval, lexical, cosine, ce))
        for rank, (retrieval, lexical, cosine, ce) in enumerate(
            sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k], start=1
        ):
            agency_unit = normalize_space(ce["agency_unit"]).upper()
            if retrieval < 0.05:
                match_type = "uncertain"
            elif agency_unit in {"BLM", "DOE"}:
                match_type = "same_agency_existing"
            else:
                match_type = "other_agency_adoption_candidate"
            records.append(
                {
                    "archetype_id": archetype["archetype_id"],
                    "archetype_label": archetype["archetype_label"],
                    "taxonomy_version": archetype["taxonomy_version"],
                    "retrieval_rank": rank,
                    "retrieval_score": round(retrieval, 6),
                    "lexical_score": round(lexical, 6),
                    "embedding_cosine": round(cosine, 6),
                    "embedding_model": "all-MiniLM-L6-v2" if use_embeddings else "",
                    "embedding_error": embedding_error,
                    # Ranking scores retrieve candidates for review; they are not legal thresholds.
                    "match_type": match_type,
                    "manual_verification_status": "pending",
                    **ce.to_dict(),
                }
            )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the D6 CE Explorer retrieval crosswalk.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Use an existing CE Explorer JSON snapshot.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Run lexical retrieval only.")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_d6_dirs()
    retrieved_at = utc_now()
    payload, content, source_url = fetch_snapshot(args.snapshot)
    snapshot_path = save_snapshot(content, retrieved_at)
    snapshot = normalized_snapshot(payload, content, source_url, retrieved_at)
    taxonomy = pd.read_parquet(TAXONOMY_PATH)
    crosswalk = match_crosswalk(
        taxonomy, snapshot, use_embeddings=not args.skip_embeddings, top_k=args.top_k
    )
    run_at = utc_now()
    snapshot["ce_snapshot_run_at"] = run_at
    crosswalk["ce_crosswalk_run_at"] = run_at
    crosswalk["ce_snapshot_path"] = str(snapshot_path)
    write_parquet(snapshot, SNAPSHOT_PARQUET)
    write_parquet(crosswalk, CROSSWALK_PARQUET)
    crosswalk.to_csv(REVIEW_CSV, index=False)
    print(
        f"stored {len(snapshot):,} CE records from {snapshot_path.name}; "
        f"wrote {len(crosswalk):,} ranked crosswalk rows"
    )


if __name__ == "__main__":
    main()
