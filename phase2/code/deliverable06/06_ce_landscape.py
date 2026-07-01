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

import re
from collections import Counter

import numpy as np
import pandas as pd

import bounds
import ce_source
import embeddings
from common import (
    D03_CE_CITATIONS, D6_ANALYSIS_DIR, D6_REVIEW_DIR, ensure_d6_dirs,
    normalize_space, sha256_text, utc_now, write_parquet,
)

CES_OUT = D6_ANALYSIS_DIR / "ce_landscape_ces.parquet"
CLUSTERS_OUT = D6_ANALYSIS_DIR / "ce_clusters.parquet"
KSELECT_OUT = D6_ANALYSIS_DIR / "ce_kselection.parquet"   # inertia + silhouette by k (elbow appendix)
SUMMARY_OUT = D6_REVIEW_DIR / "ce_landscape_summary.csv"
CLUSTERS_REVIEW = D6_REVIEW_DIR / "ce_cluster_map_review.csv"
NEAR_DUP_THRESHOLD = 0.85
N_CLUSTERS = 8                     # KMeans clusters for the relatedness scatter

_CLUSTER_STOP = set((
    "the and for with would that this are all any from not its such other which been were has have also any "
    "including include includes will may must per into onto under over within without their these those when "
    "actions action activities activity project projects program programs federal agency agencies department "
    "use using used new existing facility facilities site sites area areas land lands operations operation "
    "construction maintenance routine minor where applicable required pursuant section appendix exclusion "
    "categorical environmental impact impacts management proposed related associated involving normal "
    "of or to be shall result amend amended amendment gov dot com www http https chapter pursuant "
    "thereof herein parts part subpart cfr usc public law").split())


def _cluster_terms(texts: pd.Series, labels, k: int = 3, ngram: tuple = (2, 4)) -> dict:
    """Distinctive n-gram PHRASES per cluster (c-TF-IDF style), for readable scatter/table labels.
    One pseudo-document per cluster; TF-IDF across clusters surfaces the 2-4 word phrases that
    distinguish each. De-duplicates phrases that share a word so a label isn't 'equipment
    replacement; equipment installation'."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = pd.DataFrame({"t": [str(x).lower() for x in texts], "c": [int(c) for c in labels]})
    clusters = sorted(df["c"].unique())
    docs = [df.loc[df["c"] == c, "t"].str.cat(sep=" ") for c in clusters]
    vec = TfidfVectorizer(ngram_range=ngram, stop_words=sorted(_CLUSTER_STOP),
                          token_pattern=r"[a-z][a-z]+", min_df=1, max_df=0.85, max_features=6000)
    try:
        X = vec.fit_transform(docs).toarray()
    except ValueError:
        return {c: "" for c in clusters}
    vocab = vec.get_feature_names_out()
    out = {}
    for i, c in enumerate(clusters):
        order = X[i].argsort()[::-1]
        picks, used = [], set()
        for j in order:
            if X[i][j] <= 0 or len(picks) >= k:
                break
            words = set(vocab[j].split())
            if words & used:                       # skip phrases overlapping an already-picked one
                continue
            picks.append(vocab[j]); used |= words
        out[c] = "; ".join(picks)
    return out


def _k_selection(emb, k_range=range(2, 13), seed: int = 42) -> pd.DataFrame:
    """Inertia (elbow) + silhouette across k — the appendix evidence for how many clusters to use."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(emb)
        sil = silhouette_score(emb, km.labels_) if k >= 2 else float("nan")
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


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
    ce = ce_source.load_ce_catalog()
    _sort_keys = [c for c in ("agency_unit", "structured_id", "ce_id", "ce_description") if c in ce.columns]
    ce = ce.sort_values(_sort_keys, kind="mergesort").reset_index(drop=True)   # deterministic order (KMeans++ is order-sensitive)
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
        # cache embeddings (keyed on text + model) so the clustering is reproducible across runs
        txt = ce["ce_text"].tolist()
        sig = sha256_text("\x00".join(txt) + "|" + embeddings.MODEL_NAME)
        cache_npy = D6_ANALYSIS_DIR / "ce_embeddings.npy"
        cache_sig = D6_ANALYSIS_DIR / "ce_embeddings.sig"
        if cache_npy.exists() and cache_sig.exists() and cache_sig.read_text() == sig:
            emb = np.load(cache_npy)
        else:
            emb = np.asarray(embeddings.embed(txt))
            np.save(cache_npy, emb)
            cache_sig.write_text(sig)
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
        # 2D layout (t-SNE) + thematic clusters (KMeans) for the relatedness scatter
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        ts = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                  random_state=42).fit_transform(emb)
        ce["coord_x"] = ts[:, 0]; ce["coord_y"] = ts[:, 1]
        km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(emb)
        ce["cluster_km"] = km
        ce["cluster_label"] = ce["cluster_km"].map(_cluster_terms(ce["ce_text"], km))
        ksel = _k_selection(emb)                          # elbow + silhouette evidence for N_CLUSTERS
        write_parquet(ksel, KSELECT_OUT)
        print(f"[06] k-selection (silhouette peak): {ksel.loc[ksel['silhouette'].idxmax(), 'k']} "
              f"-> {KSELECT_OUT.name}")
    else:
        print(f"[06] WARNING: embeddings unavailable ({embeddings.MODEL_NAME}); similarity, "
              "near-duplicate, and t-SNE/cluster outputs are EMPTY. The report's relatedness and "
              "adopt-precedent claims require embeddings — do not ship a client-facing run in this mode.")
        ce["nearest_xagency_ce"] = ""; ce["nearest_xagency_cosine"] = None
        ce["nearest_xagency_unit"] = ""; ce["xagency_near_duplicate"] = False
        ce["cluster_root"] = list(range(n))
        ce["coord_x"] = np.nan; ce["coord_y"] = np.nan
        ce["cluster_km"] = -1; ce["cluster_label"] = ""
    ce["embedding_available"] = bool(embeddings.available())

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
    clusters["representative_desc"] = clusters["representative_desc"].map(lambda t: normalize_space(t)[:300])
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
