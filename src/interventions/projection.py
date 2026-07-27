"""
Negation-direction rescaling.  ===  PERSON A  ===

The hypothesis
--------------
If an embedding space encodes negation at all, it probably encodes it weakly and
along a roughly consistent direction: whatever "flip the polarity" does to a
vector, it should do something similar to every sentence. If we can *find* that
direction from data, we can turn its volume up — without retraining anything.

Which way to turn it
--------------------
This is the part that is easy to get backwards, so it is worth being explicit.

Write a vector as its component along the negation direction `d` plus the rest:

    v  =  v_perp  +  (v·d) d

The obvious move is to *project the direction out* (drop the `(v·d) d` term).
That is wrong here. Removing the only dimension that distinguishes a sentence
from its negation makes the pair **more** alike, which shrinks the very gap we
are trying to open. Projection-out is the right tool for *erasing* an attribute
(debiasing); we want the opposite — to *expose* one.

So we rescale the component instead of deleting it:

    v' = v + (γ - 1) · ((v·d) - μ) · d

`γ` is the scale. `γ = 1` is the identity — the intervention reduces exactly to
the baseline, which the tests check. `γ = 0` removes the direction. `γ > 1`
amplifies it, and that is where the gap should open.

Why the `μ`
-----------
`μ` is the mean component along `d` over the training sentences. Without it,
amplification can backfire: if a target and its negation both have a large
positive component along `d`, scaling both up drags both vectors toward `+d` and
makes them look *more* similar. Centring first means the two sit on opposite
sides of zero, so amplification pushes them apart instead of together.

Two ways to estimate `d`
------------------------
``mean_diff``   average of (negation − target) over the train split. Cheap, and
                the topic noise cancels because every pair shares its topic.
``classifier``  a logistic regression separating {target, paraphrase} from
                {negation}; the weight vector is the normal to the separating
                hyperplane. Uses all three sentences per item rather than just
                the two, and can pick up a direction that mean-differencing
                averages away.

Choosing γ honestly
-------------------
`fit()` sweeps γ and keeps whichever value maximises the cosine gap **on the
train split**. The test split is never touched during `fit` — that is the
guardrail in PLAN.md, and `tests/test_projection.py` asserts it by handing `fit`
an embedder that raises if it is ever shown a test sentence.

The trade-off to watch: a γ large enough to open the negation gap also distorts
ordinary similarity, so `sts_corr` in the harness is the guard. Report both.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import ProbeItem
from .base import Intervention
from .baseline import cosine

#: γ values tried during fit. Starts at the identity so the sweep can always
#: fall back to "do nothing" if amplification does not help.
DEFAULT_SCALE_GRID: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0)

DIRECTION_METHODS = ("mean_diff", "classifier")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _rescale(v: np.ndarray, direction: np.ndarray, mu: float, gamma: float) -> np.ndarray:
    """v + (γ−1)·((v·d) − μ)·d, applied row-wise. Exactly the identity at γ = 1."""
    if gamma == 1.0:
        return v
    component = (v @ direction) - mu
    return v + (gamma - 1.0) * component[:, None] * direction[None, :]


class NegationProjection(Intervention):
    name = "projection"

    def __init__(
        self,
        direction_method: str = "mean_diff",
        scale: Optional[float] = None,
        scale_grid: Sequence[float] = DEFAULT_SCALE_GRID,
        center: bool = True,
        select: str = "cv",
        n_folds: int = 5,
        seed: int = 0,
    ):
        """`scale=None` sweeps `scale_grid` and picks γ by `select`
        (`"cv"` = cross-validation inside train, `"train"` = the whole train
        split). Passing an explicit `scale` pins it and skips the sweep."""
        if direction_method not in DIRECTION_METHODS:
            raise ValueError(f"direction_method must be one of {DIRECTION_METHODS}")
        if select not in ("cv", "train"):
            raise ValueError("select must be 'cv' or 'train'")

        self.direction_method = direction_method
        self.scale = scale
        self.scale_grid = tuple(scale_grid)
        self.center = center
        self.select = select
        self.n_folds = n_folds
        self.seed = seed
        self.selection = "pinned"

        self.direction: Optional[np.ndarray] = None
        self.mu: float = 0.0
        self.sweep: List[Tuple[float, float]] = []   # (γ, train gap) — for the report
        self.at_grid_edge: bool = False
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_key: object = None

    # -- embedding cache ---------------------------------------------------
    # score() is called once per pair and the metrics call it several times per
    # item, so without a cache a real model re-encodes the same sentence dozens
    # of times. Keyed by text; reset if the embedder changes.

    def _encode(self, texts: Sequence[str], embedder) -> np.ndarray:
        key = getattr(embedder, "key", id(embedder))
        if self._cache_key != key:
            self._cache = {}
            self._cache_key = key

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = np.asarray(embedder.encode(missing), dtype=float)
            for text, vec in zip(missing, vectors):
                self._cache[text] = vec
        return np.stack([self._cache[t] for t in texts])

    # -- direction estimation ---------------------------------------------

    def _direction_mean_diff(self, items: List[ProbeItem], embedder) -> np.ndarray:
        targets = self._encode([it.target for it in items], embedder)
        negations = self._encode([it.negation for it in items], embedder)
        return _unit(np.mean(negations - targets, axis=0))

    def _direction_classifier(self, items: List[ProbeItem], embedder) -> np.ndarray:
        """Normal to the hyperplane separating negated from non-negated text.

        Targets and paraphrases share a label — that is what stops the
        classifier from simply learning topic instead of polarity.

        That alone is not enough, though. Topic dominates the norm of a sentence
        embedding, so a classifier fitted on raw vectors spends its capacity
        separating topics that happen to correlate with the label, and the
        direction it returns is mostly noise. (Measured on the planted space in
        `tests/test_projection.py`: alignment with the true direction goes from
        ~0.50 to ~0.99 once this is fixed.)

        The fix uses the structure the probe already has: subtract each item's
        own mean vector from its three sentences first. Topic is shared inside a
        triple, so centring per item cancels it and leaves polarity — the same
        reason mean-differencing works, applied to all three sentences instead
        of two. The direction is fitted on centred vectors and then applied to
        uncentred ones, which is fine: we only ever use its orientation.
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "direction_method='classifier' needs scikit-learn "
                "(`pip install scikit-learn`, already in requirements.txt)"
            ) from exc

        targets = self._encode([it.target for it in items], embedder)
        paraphrases = self._encode([it.paraphrase for it in items], embedder)
        negations = self._encode([it.negation for it in items], embedder)

        per_item_mean = (targets + paraphrases + negations) / 3.0
        X = np.concatenate([
            targets - per_item_mean,
            paraphrases - per_item_mean,
            negations - per_item_mean,
        ])
        y = np.array([0] * (2 * len(items)) + [1] * len(items))

        clf = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=self.seed
        )
        clf.fit(X, y)
        return _unit(clf.coef_[0])

    def _fit_direction(self, items: List[ProbeItem], embedder) -> Tuple[np.ndarray, float]:
        """Estimate (direction, μ) from a set of items. Used both for the final
        fit on all of train and for each fold during γ selection."""
        if self.direction_method == "mean_diff":
            direction = self._direction_mean_diff(items, embedder)
        else:
            direction = self._direction_classifier(items, embedder)

        mu = 0.0
        if self.center:
            texts = [t for it in items for t in (it.target, it.paraphrase, it.negation)]
            mu = float(np.mean(self._encode(texts, embedder) @ direction))
        return direction, mu

    # -- fit ---------------------------------------------------------------

    def fit(self, train_items: List[ProbeItem], embedder) -> None:
        """Fit the direction on all of train, and pick γ by cross-validation
        *inside* train.

        The naive alternative — fit the direction on train, then pick the γ that
        maximises the gap on that same train split — is the mistake PLAN.md warns
        about, one level down. The direction has already seen those exact pairs,
        so the gap keeps rising with γ and the sweep runs to the top of the grid
        every time. On the 6-item mock probe that produces a *negative* test gap:
        the chosen γ is fitted to noise and does not transfer.

        So by default `select="cv"` holds out a fold, fits the direction on the
        rest, and measures the gap on the held-out fold — repeated K times and
        averaged. Only then is the direction refitted on all of train at the
        chosen γ. The test split is still never involved.

        `select="train"` keeps the naive behaviour for the ablation, since
        "CV selection vs train selection" is itself worth a line in the report.
        """
        if not train_items:
            raise ValueError("projection needs a non-empty train split to fit a direction")

        self.direction, self.mu = self._fit_direction(train_items, embedder)

        if self.scale is None:
            if self.select == "cv" and len(train_items) >= 2 * self.n_folds:
                self.sweep = self._sweep_cv(train_items, embedder)
                self.selection = f"cv{self.n_folds}"
            else:
                self.sweep = [
                    (float(g), self._gap(train_items, embedder, self.direction, self.mu, float(g)))
                    for g in self.scale_grid
                ]
                self.selection = "train"
            self.scale = max(self.sweep, key=lambda pair: pair[1])[0]
            # The gap grows monotonically in γ for a while, so the argmax landing
            # on the largest value tried means the grid, not the data, chose it.
            # Worth surfacing rather than quietly reporting a boundary value.
            self.at_grid_edge = self.scale == max(self.scale_grid)

    def _folds(self, items: List[ProbeItem]) -> List[List[int]]:
        """Deterministic K folds — same items always give the same folds, so a
        rerun cannot silently change which γ was chosen."""
        order = sorted(
            range(len(items)),
            key=lambda i: hashlib.md5(f"{self.seed}:{items[i].id}".encode()).hexdigest(),
        )
        return [order[k::self.n_folds] for k in range(self.n_folds)]

    def _sweep_cv(self, items: List[ProbeItem], embedder) -> List[Tuple[float, float]]:
        totals = {float(g): 0.0 for g in self.scale_grid}
        folds = self._folds(items)
        used = 0
        for fold in folds:
            held_out = [items[i] for i in fold]
            rest = [it for i, it in enumerate(items) if i not in set(fold)]
            if not held_out or not rest:
                continue
            direction, mu = self._fit_direction(rest, embedder)
            for g in self.scale_grid:
                totals[float(g)] += self._gap(held_out, embedder, direction, mu, float(g))
            used += 1
        return [(g, total / max(used, 1)) for g, total in totals.items()]

    def _gap(
        self,
        items: List[ProbeItem],
        embedder,
        direction: np.ndarray,
        mu: float,
        scale: float,
    ) -> float:
        """Cosine gap on `items` under a given (direction, μ, γ). Selection
        signal only — never report this number; it is what is being optimised."""
        t = _rescale(self._encode([it.target for it in items], embedder), direction, mu, scale)
        p = _rescale(self._encode([it.paraphrase for it in items], embedder), direction, mu, scale)
        n = _rescale(self._encode([it.negation for it in items], embedder), direction, mu, scale)

        sim_para = np.mean([cosine(a, b) for a, b in zip(t, p)])
        sim_neg = np.mean([cosine(a, b) for a, b in zip(t, n)])
        return float(sim_para - sim_neg)

    # -- transform + score -------------------------------------------------

    def _transform(self, v: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        if self.direction is None:
            return v
        gamma = self.scale if scale is None else scale
        return _rescale(v, self.direction, self.mu, 1.0 if gamma is None else gamma)

    def score(self, a: str, b: str, embedder) -> float:
        pair = self._transform(self._encode([a, b], embedder))
        return cosine(pair[0], pair[1])

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        """One line for the results table / report appendix."""
        if self.direction is None:
            return "projection (unfitted)"
        edge = ", AT GRID EDGE" if self.at_grid_edge else ""
        return (f"projection[{self.direction_method}, γ={self.scale:g}, "
                f"center={self.center}, select={self.selection}{edge}]")

    def sweep_table(self) -> str:
        if not self.sweep:
            return "(no sweep — γ was pinned)"
        label = "held-out gap" if self.selection.startswith("cv") else "train gap"
        lines = [f"  γ      {label}"]
        for g, gap in self.sweep:
            lines.append(f"  {g:<6g} {gap:+.4f}" + ("   <- chosen" if g == self.scale else ""))
        return "\n".join(lines)
