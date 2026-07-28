"""
Metrics.  ===  PERSON B  ===

Three numbers, matching the proposal:
  1. cosine_gap   - mean cos(target,paraphrase) - mean cos(target,negation).
                    A negation-aware model has a LARGE gap. Headline #1.
  2. pairwise_accuracy - share of items where the paraphrase is ranked closer
                    than the negation. **Chance = 0.5.** Headline #2.
                    This replaces `nevir_rank`, which was not computable from a
                    triple; `nevir_rank_full` is the real thing and needs a
                    fourth sentence per item. See that function's docstring.
  3. sts_corr     - Pearson/Spearman on a Hebrew STS set, to prove a fix did
                    not damage ordinary similarity (the trade-off check).

All three take a `score_fn(a, b) -> float`, so they work for the baseline and
for any intervention identically.
"""
from __future__ import annotations

from statistics import mean
from typing import Callable, List, Tuple

from ..schema import ProbeItem

ScoreFn = Callable[[str, str], float]


def cosine_gap(items: List[ProbeItem], score_fn: ScoreFn) -> dict:
    para = [score_fn(it.target, it.paraphrase) for it in items]
    neg = [score_fn(it.target, it.negation) for it in items]
    return {
        "sim_paraphrase": mean(para),
        "sim_negation": mean(neg),
        "cosine_gap": mean(para) - mean(neg),
    }


def pairwise_accuracy(items: List[ProbeItem], score_fn: ScoreFn) -> float:
    """Share of items where the paraphrase is ranked closer than the negation.

        correct iff  score(target, paraphrase) > score(target, negation)

    **Chance is 0.5, not 0.25.** This is a per-item pairwise preference, not the
    NevIR metric — see the note below for why a triple cannot produce NevIR.

    Note this is the per-item sign of `cosine_gap`, so it carries information the
    gap's mean does not: a model can have a positive mean gap while getting the
    sign wrong on a third of items.

    -- why this is not nevir_rank (A -> B) ---------------------------------
    The previous version tried to build NevIR's two-query/two-document setup
    out of the triple, using the negation as its own relevant document:

        q1, q2 = target, negation
        d1, d2 = paraphrase, negation          # d2 is q2
        ok2 = score(q2, d2) > score(q2, d1)    # cos(x, x) = 1 > anything

    `ok2` is therefore always true, the metric collapsed to `ok1`, and the
    reported chance level of 0.25 was wrong. Measured on the real probe it
    returned exactly 0.497 and 0.503 for the two models whose gap is ~0 — a coin
    flip, which is what 0.5-chance looks like.

    Real NevIR needs four texts: two queries and, for each, its own relevant
    document, with the two documents forming the minimal pair. A triple gives
    only three distinct pairings, so no two independent decisions exist inside
    it. Getting a genuine 0.25-chance number needs a fourth sentence per item —
    a paraphrase of the negation — which means extending ProbeItem, i.e. a
    change to the frozen schema. `nevir_rank_full` below is ready for that day
    and takes the fourth sentence externally so no contract has to move yet.
    """
    correct = sum(
        int(score_fn(it.target, it.paraphrase) > score_fn(it.target, it.negation))
        for it in items
    )
    return correct / max(len(items), 1)


#: kept so run_eval.py and any saved results keep working. Prefer the name
#: `pairwise_accuracy` in anything reported — chance 0.5.
nevir_rank = pairwise_accuracy


def nevir_rank_full(
    items: List[ProbeItem],
    score_fn: ScoreFn,
    negation_paraphrases: dict,
) -> float:
    """True NevIR-style rank. Chance = 0.25.

    `negation_paraphrases` maps item id -> a restatement of that item's
    `negation`, in different words. With it we finally have four distinct texts
    and two genuinely independent decisions:

        q1 = target     d1 = paraphrase             (same meaning as q1)
        q2 = negation   d2 = negation_paraphrase    (same meaning as q2)

        ok1 = score(q1, d1) > score(q1, d2)
        ok2 = score(q2, d2) > score(q2, d1)

    A model counts as right on the item only if both hold — hence 0.25 by
    chance. Items with no fourth sentence supplied are skipped, so this can be
    run on a partially extended probe.
    """
    scored = 0
    correct = 0
    for it in items:
        d2 = negation_paraphrases.get(it.id)
        if not d2:
            continue
        scored += 1
        ok1 = score_fn(it.target, it.paraphrase) > score_fn(it.target, d2)
        ok2 = score_fn(it.negation, d2) > score_fn(it.negation, it.paraphrase)
        correct += int(ok1 and ok2)
    if scored == 0:
        raise ValueError(
            "no negation paraphrases supplied — nevir_rank_full needs a fourth "
            "sentence per item; use pairwise_accuracy until the probe has them"
        )
    return correct / scored


def sts_corr(sts_pairs: List[Tuple[str, str, float]], score_fn: ScoreFn) -> dict:
    """sts_pairs: list of (sentence_a, sentence_b, gold_similarity).

    TODO(B): wire a Hebrew STS set here (or a held-out slice of HebNLI mapped
    to graded similarity). Returns Pearson/Spearman of score_fn vs gold.
    """
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:  # scipy not installed in a bare env
        return {"pearson": None, "spearman": None, "n": len(sts_pairs)}
    if not sts_pairs:
        return {"pearson": None, "spearman": None, "n": 0}
    preds = [score_fn(a, b) for a, b, _ in sts_pairs]
    gold = [g for _, _, g in sts_pairs]
    return {
        "pearson": float(pearsonr(preds, gold)[0]),
        "spearman": float(spearmanr(preds, gold)[0]),
        "n": len(sts_pairs),
    }
