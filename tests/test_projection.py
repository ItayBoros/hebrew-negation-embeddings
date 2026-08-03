"""
Offline checks for the projection intervention.  ===  PERSON A  ===

    python -m tests.test_projection

`FakeEmbedder` is useless for testing whether the method *works* — it hashes
text to noise, so there is no negation direction to find. So this file uses a
`PlantedEmbedder` instead: a synthetic space built to look the way we believe a
real embedder looks.

    v(sentence) = topic_vector  +  polarity · ε · d_true  +  noise

One topic vector per item, shared by all three of its sentences, so topic
dominates the norm. Polarity is −1 for the target and paraphrase and +1 for the
negation, and ε is small — that *is* the negation blind spot, in miniature: the
only thing separating a sentence from its negation is a faint component along a
single direction.

That gives ground truth to check against, which noise cannot:

  - does `fit` recover `d_true`, or just some arbitrary direction?
  - does amplification actually widen the gap, or did we get the sign backwards?
  - is the paraphrase left alone while the negation is pushed away?

Plus the guardrails that matter regardless of whether the method works:
γ = 1 must reproduce the baseline exactly, `fit` must never see the test split,
and two runs must give the same answer.
"""
from __future__ import annotations

import itertools
import sys
from typing import List, Sequence

import numpy as np

from src.interventions.baseline import Baseline, cosine
from src.interventions.projection import NegationProjection
from src.schema import ProbeItem

DIM = 64
EPSILON = 0.45          # how faintly negation is encoded — the blind spot
NOISE = 0.10


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash(tag)) % (2**32))


class PlantedEmbedder:
    """Synthetic space with a known negation direction.

    Texts are encoded as ``"<item id>|<role>"`` so the embedder can tell which
    of the three sentences it is looking at. Real text never reaches it.
    """

    key = "planted"

    def __init__(self, dim: int = DIM, epsilon: float = EPSILON, noise: float = NOISE):
        self.dim = dim
        self.epsilon = epsilon
        self.noise = noise
        self.d_true = np.zeros(dim)
        self.d_true[0] = 1.0
        # rotate off the axis so no one can pass the test by guessing e_0
        rot = _rng("rotation").standard_normal((dim, dim))
        q, _ = np.linalg.qr(rot)
        self.d_true = q @ self.d_true
        self.d_true /= np.linalg.norm(self.d_true)
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        item_id, role = text.split("|")
        topic = _rng(f"topic:{item_id}").standard_normal(self.dim)
        polarity = 1.0 if role == "negation" else -1.0
        noise = _rng(f"noise:{text}").standard_normal(self.dim) * self.noise
        return topic + polarity * self.epsilon * self.d_true + noise

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return np.asarray([self._vec(t) for t in texts])


class ForbiddenTextEmbedder(PlantedEmbedder):
    """Raises if asked to encode a sentence it was told to never see.
    This is how the no-leakage guarantee is enforced rather than assumed."""

    key = "forbidden"

    def __init__(self, forbidden: Sequence[str], **kw):
        super().__init__(**kw)
        self.forbidden = set(forbidden)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        leaked = self.forbidden.intersection(texts)
        if leaked:
            raise AssertionError(f"fit() touched held-out text: {sorted(leaked)[:3]}")
        return super().encode(texts)


def make_items(n: int, split: str, prefix: str) -> List[ProbeItem]:
    return [
        ProbeItem(
            id=f"{prefix}{i}",
            target=f"{prefix}{i}|target",
            paraphrase=f"{prefix}{i}|paraphrase",
            negation=f"{prefix}{i}|negation",
            source="synthetic",
            split=split,
        )
        for i in range(n)
    ]


def gap(items: List[ProbeItem], score_fn) -> float:
    para = np.mean([score_fn(it.target, it.paraphrase) for it in items])
    neg = np.mean([score_fn(it.target, it.negation) for it in items])
    return float(para - neg)


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)
        print(f"[FAIL] {message}")


def main() -> int:
    failures: list = []
    train = make_items(60, "train", "tr")
    test = make_items(40, "test", "te")

    # ------------------------------------------------------------------
    print("== γ = 1 must reproduce the baseline exactly ==")
    emb = PlantedEmbedder()
    baseline = Baseline()
    pinned = NegationProjection(scale=1.0)
    pinned.fit(train, emb)
    diffs = [
        abs(pinned.score(it.target, it.negation, emb) - baseline.score(it.target, it.negation, emb))
        for it in test
    ]
    check(max(diffs) < 1e-12, f"γ=1 differs from baseline by {max(diffs):.2e}", failures)

    # ------------------------------------------------------------------
    print("== fit() must not see the test split ==")
    held_out = [t for it in test for t in (it.target, it.paraphrase, it.negation)]
    guarded = ForbiddenTextEmbedder(held_out)
    try:
        for method, select in itertools.product(("mean_diff", "classifier"), ("cv", "train")):
            NegationProjection(direction_method=method, select=select).fit(train, guarded)
        print("  no leak (both directions × both γ-selection modes)")
    except AssertionError as exc:
        check(False, str(exc), failures)

    # ------------------------------------------------------------------
    print("\n== does fit recover the planted direction? ==")
    for method in ("mean_diff", "classifier"):
        emb = PlantedEmbedder()
        proj = NegationProjection(direction_method=method)
        proj.fit(train, emb)
        alignment = abs(float(proj.direction @ emb.d_true))
        print(f"  {method:<11} |cos(d_hat, d_true)| = {alignment:.3f}   γ = {proj.scale:g}")
        check(alignment > 0.90, f"{method}: recovered direction is off ({alignment:.3f})", failures)

    # ------------------------------------------------------------------
    print("\n== does it open the gap on held-out items? ==")
    for method in ("mean_diff", "classifier"):
        emb = PlantedEmbedder()
        base_gap = gap(test, lambda a, b: baseline.score(a, b, emb))
        proj = NegationProjection(direction_method=method)
        proj.fit(train, emb)
        proj_gap = gap(test, lambda a, b: proj.score(a, b, emb))

        sim_para = np.mean([proj.score(it.target, it.paraphrase, emb) for it in test])
        sim_neg = np.mean([proj.score(it.target, it.negation, emb) for it in test])
        print(f"  {method:<11} baseline gap {base_gap:+.4f} -> {proj_gap:+.4f}"
              f"   (para {sim_para:+.3f}, neg {sim_neg:+.3f})")

        check(base_gap < 0.05, f"{method}: planted space should look nearly blind, got {base_gap:.3f}", failures)
        check(proj_gap > base_gap * 3, f"{method}: gap barely moved ({base_gap:.4f} -> {proj_gap:.4f})", failures)
        check(sim_para > sim_neg, f"{method}: paraphrase is not closer than negation", failures)

    # ------------------------------------------------------------------
    print("\n== sweep behaves ==")
    emb = PlantedEmbedder()
    proj = NegationProjection()
    proj.fit(train, emb)
    print(proj.sweep_table())
    check(len(proj.sweep) == len(proj.scale_grid), "sweep did not cover the grid", failures)
    check(
        proj.scale == max(proj.sweep, key=lambda p: p[1])[0],
        "chosen γ is not the argmax of the sweep",
        failures,
    )
    check(proj.scale > 1.0, "amplification should beat the identity on planted data", failures)

    print("\n== deterministic ==")
    again = NegationProjection()
    again.fit(train, PlantedEmbedder())
    check(again.scale == proj.scale, "γ changed between identical runs", failures)
    check(np.allclose(again.direction, proj.direction), "direction changed between identical runs", failures)

    # ------------------------------------------------------------------
    print("\n== embedding cache ==")
    emb = PlantedEmbedder()
    proj = NegationProjection(scale=3.0)
    proj.fit(train, emb)
    before = emb.calls
    for _ in range(5):
        for it in test:
            proj.score(it.target, it.negation, emb)
    print(f"  {emb.calls - before} encode calls for {5 * len(test)} scored pairs")
    check(emb.calls - before <= len(test), "cache is not preventing re-encoding", failures)

    print("\n== γ selection: cross-validated vs on the train split itself ==")
    # A deliberately hostile setting: almost no planted signal and few items, so
    # the direction is mostly fitted to noise. Selecting γ on the same pairs the
    # direction was fitted on should then report a gap that is far too good.
    thin = make_items(15, "train", "thin")
    weak = PlantedEmbedder(epsilon=0.02, noise=0.6)

    naive = NegationProjection(select="train")
    naive.fit(thin, weak)
    crossval = NegationProjection(select="cv", n_folds=5)
    crossval.fit(thin, weak)

    naive_best = max(g for _, g, _u in naive.sweep)
    cv_best = max(g for _, g, _u in crossval.sweep)
    print(f"  select=train  best reported gap {naive_best:+.4f}")
    print(f"  select=cv5    best reported gap {cv_best:+.4f}")
    check(naive_best > cv_best,
          f"train selection should look inflated next to CV ({naive_best:.4f} vs {cv_best:.4f})",
          failures)
    check(crossval.selection == "cv5", f"selection label is '{crossval.selection}'", failures)
    check(naive.selection == "train", f"selection label is '{naive.selection}'", failures)

    repeat = NegationProjection(select="cv", n_folds=5)
    repeat.fit(thin, PlantedEmbedder(epsilon=0.02, noise=0.6))
    check(repeat.sweep == crossval.sweep, "CV folds are not deterministic", failures)

    tiny = NegationProjection(select="cv", n_folds=5)
    tiny.fit(make_items(4, "train", "tiny"), PlantedEmbedder())
    check(tiny.selection == "train",
          "CV should fall back to train selection when there are too few items",
          failures)

    print("\n== centring matters ==")
    # with an off-centre planted space, uncentred amplification should do worse
    class OffCentre(PlantedEmbedder):
        key = "offcentre"
        def _vec(self, text):
            return super()._vec(text) + 4.0 * self.d_true

    emb = OffCentre()
    centred = NegationProjection(center=True)
    centred.fit(train, emb)
    uncentred = NegationProjection(center=False)
    uncentred.fit(train, emb)
    g_c = gap(test, lambda a, b: centred.score(a, b, emb))
    g_u = gap(test, lambda a, b: uncentred.score(a, b, emb))
    print(f"  centred {g_c:+.4f}   uncentred {g_u:+.4f}")
    check(g_c >= g_u, f"centring should not hurt ({g_c:.4f} vs {g_u:.4f})", failures)

    print("\n== unrel-constrained γ selection ==")
    # Unconstrained selection on the planted space amplifies until it collapses:
    # every target starts looking like every other target. Constraining unrel
    # should pick a smaller γ, and that γ's own held-out unrel should respect
    # the threshold — the same failure mode the real-model run surfaced.
    emb = PlantedEmbedder()
    unconstrained = NegationProjection(direction_method="mean_diff")
    unconstrained.fit(train, emb)
    unconstrained_unrel = {g: u for g, _, u in unconstrained.sweep}[unconstrained.scale]
    print(f"  unconstrained: γ={unconstrained.scale:g}  unrel={unconstrained_unrel:.3f}")
    check(unconstrained_unrel > 0.5,
          f"expected the unconstrained pick to collapse unrel on this planted space, got {unconstrained_unrel:.3f}",
          failures)

    emb = PlantedEmbedder()
    constrained = NegationProjection(direction_method="mean_diff", constrain_unrel=True, unrel_threshold=0.5)
    constrained.fit(train, emb)
    constrained_unrel = {g: u for g, _, u in constrained.sweep}[constrained.scale]
    print(f"  constrained:   γ={constrained.scale:g}  unrel={constrained_unrel:.3f}")
    check(constrained.scale < unconstrained.scale,
          f"constrained γ should be smaller than unconstrained ({constrained.scale:g} vs {unconstrained.scale:g})",
          failures)
    check(constrained_unrel <= 0.5 + 1e-9,
          f"constrained pick should respect the threshold, got unrel={constrained_unrel:.3f}",
          failures)
    check(not constrained.constraint_relaxed, "threshold=0.5 should be satisfiable here", failures)

    # An unreachable threshold must fall back to the safest γ (lowest unrel in
    # the grid) rather than silently violating the constraint, and must flag it.
    emb = PlantedEmbedder()
    impossible = NegationProjection(direction_method="mean_diff", constrain_unrel=True, unrel_threshold=1e-6)
    impossible.fit(train, emb)
    print(f"  impossible threshold: γ={impossible.scale:g}  relaxed={impossible.constraint_relaxed}")
    check(impossible.constraint_relaxed, "an unreachable threshold should set constraint_relaxed", failures)
    check(impossible.scale == min(impossible.sweep, key=lambda t: t[2])[0],
          "fallback should be the lowest-unrel γ in the grid", failures)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        return 1
    print("all projection checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
