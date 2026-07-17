"""
Negation-direction projection.  ===  PERSON A  ===

Idea: there may be a single direction in embedding space that carries
"negation-ness". If we find it and remove it (or amplify it), a negated
sentence should stop looking almost-identical to its original.

This file gives you a WORKING starting point (mean-difference direction).
Your job for M2 is to make it good and honest:
  - try both a mean-difference direction and a linear-classifier normal
  - decide remove vs amplify (and by how much: a scale alpha)
  - FIT ON TRAIN ONLY, measure on TEST (see base.py / PLAN.md)
  - report the trade-off vs plain similarity (STS must not collapse)

TODOs are marked below.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..schema import ProbeItem
from .base import Intervention
from .baseline import cosine


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


class NegationProjection(Intervention):
    name = "projection"

    def __init__(self, alpha: float = 1.0):
        # alpha = 1.0 removes the full component along the negation direction.
        # TODO(M2): sweep alpha; alpha can also be > 1 (amplify) or tuned on train.
        self.alpha = alpha
        self.direction: Optional[np.ndarray] = None

    def fit(self, train_items: List[ProbeItem], embedder) -> None:
        """Estimate the negation direction as the mean (negation - target) vector.

        Rationale: target -> negation is, by construction, a 'flip the polarity'
        edit. Averaging many such edits should cancel topic noise and leave the
        shared negation component.
        """
        diffs = []
        for it in train_items:
            vt, vn = embedder.encode([it.target, it.negation])
            diffs.append(vn - vt)
        self.direction = _unit(np.mean(np.asarray(diffs), axis=0))
        # TODO(M2): alternative — fit a logistic regression separating
        # {target, paraphrase} from {negation} and take the weight vector as
        # the direction. Compare which gives a cleaner test-split gap.

    def _project_out(self, v: np.ndarray) -> np.ndarray:
        if self.direction is None:
            return v
        comp = np.dot(v, self.direction) * self.direction
        return v - self.alpha * comp

    def score(self, a: str, b: str, embedder) -> float:
        va, vb = embedder.encode([a, b])
        return cosine(self._project_out(va), self._project_out(vb))
