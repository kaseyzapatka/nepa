"""D6 v2 — 06 (Track C): existing-CE landscape analysis.

A standalone analysis *of the existing CE body* (`ce.json` via `ce_source`) to
learn from the current categorical-exclusion legislation:

  - **Similarity / near-duplicates** — embed every CE's text; for each, find its
    nearest CE in a *different* agency. High cross-agency similarity = a CE many
    agencies share (so the ones lacking it are ADOPT targets) and/or a
    consolidation candidate.
  - **Per-agency richness** — CE counts per agency.
  - **Bounds distribution** — how often CEs state numeric limits (acres/mi/kV) and
    their ranges → precedent context for what a defensible EXPAND looks like.
  - **Usage** — top-cited CE codes in D3 `ce_citations` (best-effort; code formats
    differ from ce.json, so this is directional).

Outputs:
  - data/analysis/deliverable06/ce_landscape_ces.parquet     (per CE)
  - output/deliverable06/ce_landscape_summary.csv            (per agency + rollups)
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import numpy as np
import pandas as pd

import bounds
import ce_source
import embeddings
from common import D03_CE_CITATIONS, D6_ANALYSIS_DIR, D6_REVIEW_DIR, ensure_d6_dirs, normalize_space, utc_now, write_parquet

CES_OUT = D6_ANALYSIS_DIR / "ce_landscape_ces.parquet"
CLUSTERS_OUT = D6_ANALYSIS_DIR / "ce_clusters.parquet"
SUMMARY_OUT = D6_REVIEW_DIR / "ce_landscape_summary.csv"
CLUSTERS_REVIEW = D6_REVIEW_DIR / "ce_cluster_map_review.csv"
NEAR_DUP_THRESHOLD = 0.85


def _components(sim: np.ndarray, thr: float) -> list[int]:
    """Connected components (union-find) over the graph of CE pairs with
    cosine >= thr. Returns a root id per CE; a component = a family of
    near-identical CEs (a 'shared action' once it spans >1 agency)."""
    n = len(sim)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in np.argwhere(np.triu(sim >= thr, k=1)):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()
    ce = ce_source.load_ce_catalog().reset_index(drop=True)
    ce["ce_text"] = ce["ce_description"].map(normalize_space)
    n = len(ce)
    print(f"[06] loaded {n} CEs across {ce['agency_unit'].nunique()} agency units")

    # --- bounds per CE ---
    bnd = ce["ce_description"].map(lambda t: bounds.parse_bounds(normalize_space(t)))
    for m in ("acres", "miles", "kv", "mw", "wells"):
        ce[f"bound_{m}"] = [b[m] for b in bnd]
    ce["states_any_bound"] = ce[[f"bound_{m}" for m in ("acres", "miles", "kv", "mw", "wells")]].notna().any(axis=1)

    # --- similarity / cross-agency near-duplicates ---
    if embeddings.available():
        emb = np.asarray(embeddings.embed(ce["ce_text"].tolist()))
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1.0)
        units = ce["agency_unit"].to_numpy().astype(str)
        nearest_x, nearest_x_cos, nearest_x_unit = [], [], []
        for i in range(n):
            diff = units != units[i]
            if diff.any():
                row = np.where(diff, sims[i], -1.0)
                j = int(row.argmax())
                nearest_x.append(ce["structured_id"].iloc[j]); nearest_x_cos.append(round(float(row[j]), 4))
                nearest_x_unit.append(units[j])
            else:
                nearest_x.append(""); nearest_x_cos.append(None); nearest_x_unit.append("")
        ce["nearest_xagency_ce"] = nearest_x
        ce["nearest_xagency_cosine"] = nearest_x_cos
        ce["nearest_xagency_unit"] = nearest_x_unit
        ce["xagency_near_duplicate"] = [(c is not None and c >= NEAR_DUP_THRESHOLD) for c in nearest_x_cos]
        ce["cluster_root"] = _components(sims, NEAR_DUP_THRESHOLD)
        # 2D projection (PCA via SVD, dependency-free) for the relatedness scatter
        embc = emb - emb.mean(axis=0)
        _, S, Vt = np.linalg.svd(embc, full_matrices=False)
        coords = embc @ Vt[:2].T
        ce["coord_x"] = coords[:, 0]; ce["coord_y"] = coords[:, 1]
    else:
        ce["nearest_xagency_ce"] = ""; ce["nearest_xagency_cosine"] = None
        ce["nearest_xagency_unit"] = ""; ce["xagency_near_duplicate"] = False
        ce["cluster_root"] = list(range(n))
        ce["coord_x"] = np.nan; ce["coord_y"] = np.nan

    # --- usage (best-effort; D3 ce_citations code format differs from ce.json) ---
    usage_top = ""
    if D03_CE_CITATIONS.exists():
        cites = pd.read_parquet(D03_CE_CITATIONS)
        vc = cites["ce_code"].dropna().astype(str).value_counts().head(15)
        usage_top = "; ".join(f"{k}={v}" for k, v in vc.items())

    ce_out = ce.drop(columns=["ce_text"])
    ce_out["landscape_run_at"] = run_at
    write_parquet(ce_out, CES_OUT)

    # --- cluster map: families of near-identical CEs (the consolidation/adopt map) ---
    # representative = shortest description in the cluster (usually the cleanest).
    rep = (ce.assign(_len=ce["ce_text"].str.len().fillna(0))
           .sort_values("_len")
           .groupby("cluster_root")
           .agg(representative_id=("structured_id", "first"),
                representative_desc=("ce_description", "first")).reset_index())
    clusters = (ce.groupby("cluster_root")
                .agg(n_ces=("ce_id", "size"),
                     n_agencies=("agency_unit", "nunique"),
                     agencies=("agency_unit", lambda s: ", ".join(sorted(set(s.astype(str))))))
                .reset_index()
                .merge(rep, on="cluster_root"))
    clusters["representative_desc"] = clusters["representative_desc"].map(lambda t: normalize_space(t)[:160])
    clusters = clusters[clusters["n_ces"] >= 2].sort_values(
        ["n_agencies", "n_ces"], ascending=False).reset_index(drop=True)
    clusters.insert(0, "cluster_id", range(1, len(clusters) + 1))
    clusters["cluster_run_at"] = run_at
    write_parquet(clusters.drop(columns=["cluster_root"]), CLUSTERS_OUT)
    xagency = clusters[clusters["n_agencies"] >= 2].drop(columns=["cluster_root"])
    xagency.to_csv(CLUSTERS_REVIEW, index=False)

    # --- per-agency summary + rollups ---
    per_agency = (ce.groupby("agency_unit")
                  .agg(n_ces=("ce_id", "size"),
                       n_xagency_near_dup=("xagency_near_duplicate", "sum"),
                       n_with_bounds=("states_any_bound", "sum"))
                  .reset_index().sort_values("n_ces", ascending=False))
    per_agency.to_csv(SUMMARY_OUT, index=False)

    n_dup = int(ce["xagency_near_duplicate"].sum())
    print(f"[06] cross-agency near-duplicates (cosine>={NEAR_DUP_THRESHOLD}): {n_dup} CEs "
          f"-> consolidation / adoption-harmonization candidates")
    print(f"[06] CE clusters (>=2 CEs): {len(clusters)}; cross-agency (shared by >1 agency): "
          f"{len(xagency)} -> {CLUSTERS_REVIEW.name}")
    if len(xagency):
        top = xagency.head(5)[["n_agencies", "n_ces", "agencies", "representative_desc"]]
        print(top.to_string(index=False))
    print(f"[06] CEs stating a numeric bound: {int(ce['states_any_bound'].sum())} of {n}")
    for m in ("acres", "miles", "kv", "mw"):
        col = ce[f"bound_{m}"].dropna()
        if len(col):
            print(f"   bound_{m}: n={len(col)} median={round(col.median(),1)} max={round(col.max(),1)}")
    print(f"[06] top-cited CE codes (D3, directional): {usage_top[:200]}")
    print(f"[06] per-agency summary -> {SUMMARY_OUT}")
    print(f"[06] per-CE landscape -> {CES_OUT}")


if __name__ == "__main__":
    main()
