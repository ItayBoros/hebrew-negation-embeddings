"""
Frozen embedding models.  ===  PERSON B  ===

Two ways to get an embedder:
  - get_embedder("multilingual-e5")  -> real sentence-transformers model
  - get_embedder("fake")             -> deterministic hash-based vectors,
                                        zero downloads, for developing the
                                        harness "anytime" without a GPU.

The real models are loaded lazily so importing this file is cheap.
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

# Registry of the four frozen base models from the proposal.
MODELS = {
    "sambert": "MPA/sambert",
    "alephbert-sentence": "imvladikon/sentence-transformers-alephbert",
    "multilingual-e5": "intfloat/multilingual-e5-base",
    "labse":           "sentence-transformers/LaBSE",
}


class Embedder:
    """Wraps a sentence-transformers model. Frozen — we never train it."""

    def __init__(self, key: str):
        from sentence_transformers import SentenceTransformer  # lazy import
        if key not in MODELS:
            raise KeyError(f"unknown model '{key}'. options: {list(MODELS)} or 'fake'")
        self.key = key
        self._model = SentenceTransformer(MODELS[key])

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=False))


class FakeEmbedder:
    """Deterministic pseudo-embeddings. NOT semantic — for plumbing tests only.

    Same text always maps to the same vector, so the harness runs end-to-end
    offline. Do not report any numbers from this; it knows nothing about Hebrew.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.key = "fake"

    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray([self._vec(t) for t in texts])


def get_embedder(key: str):
    if key == "fake":
        return FakeEmbedder()
    return Embedder(key)
