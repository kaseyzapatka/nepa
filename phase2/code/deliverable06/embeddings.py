"""Shared local embedding helper for D6 (all-MiniLM-L6-v2).

Cheap, CPU-friendly, no API cost. Model loads once per process (lru_cache).
Used by n03 (action-definition sentence selection) and n04 (CE ranking).
All callers must guard with `available()` and fall back gracefully.
"""

from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]):
    """Return L2-normalized embeddings (n, d) as a numpy array."""
    model = _model()
    return model.encode(list(texts), normalize_embeddings=True,
                        show_progress_bar=False, batch_size=64)


def cosine(a, b):
    """Cosine similarity matrix for L2-normalized a (n,d) and b (m,d) -> (n,m)."""
    import numpy as np
    return np.asarray(a) @ np.asarray(b).T
