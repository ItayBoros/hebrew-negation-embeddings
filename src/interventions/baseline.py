"""Baseline: raw cosine similarity from the frozen model. No fix applied.

Owner: shared (it's the reference point both interventions are compared to).
"""
from __future__ import annotations

import numpy as np

from .base import Intervention


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12
    return float(np.dot(u, v) / denom)


class Baseline(Intervention):
    name = "baseline"

    def score(self, a: str, b: str, embedder) -> float:
        va, vb = embedder.encode([a, b])
        return cosine(va, vb)
