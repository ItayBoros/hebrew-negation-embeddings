"""
Contract 2 — every intervention exposes the same two methods.

This file is SHARED and FROZEN. Both interventions (projection, NLI re-rank)
implement this interface, so run_eval.py can loop over them uniformly.
Change it only by mutual agreement.

    fit(train_items, embedder)   -> learn parameters on the TRAIN split only.
                                    Default: no-op (for interventions that
                                    don't learn anything, like the baseline).
    score(a, b, embedder)        -> a similarity in [-1, 1]; higher = closer.

Why "score a pair" and not "transform an embedding":
    - projection fits the transform interface (encode -> project -> cosine)
    - NLI re-ranking does NOT (it's a cross-encoder over the raw text)
    A single score(a, b) signature covers both cleanly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..schema import ProbeItem


class Intervention(ABC):
    #: short identifier used in results tables
    name: str = "base"

    def fit(self, train_items: List[ProbeItem], embedder) -> None:
        """Learn parameters on the TRAIN split. Default: nothing to learn.

        IMPORTANT: never look at the test split here. Fitting on the same
        pairs you later measure on inflates the result (see PLAN.md, M2).
        """
        return None

    @abstractmethod
    def score(self, a: str, b: str, embedder) -> float:
        """Return similarity(a, b) in [-1, 1]. Higher means more similar."""
        raise NotImplementedError
