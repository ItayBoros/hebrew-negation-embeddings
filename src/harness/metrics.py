"""
Metrics.  ===  PERSON B  ===

Three numbers, matching the proposal:
  1. cosine_gap   - mean cos(target,paraphrase) - mean cos(target,negation).
                    A negation-aware model has a LARGE gap. Headline #1.
  2. nevir_rank   - Hebrew NevIR-style right-rank accuracy. Chance = 0.25.
                    Headline #2.
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


def nevir_rank(items: List[ProbeItem], score_fn: ScoreFn) -> float:
    """NevIR-style: build two contrasting queries and two documents per item,
    count how often ALL rankings come out right (chance = 0.25).

    Construction from a triple:
        q1 = target,   q2 = negation
        d1 (relevant to q1) = paraphrase   (means the same as target)
        d2 (relevant to q2) = negation's own restatement -> we approximate with
        the negation sentence itself as its own relevant doc.
    A model is "right" on the item iff:
        score(q1, d1) > score(q1, d2)  AND  score(q2, d2) > score(q2, d1)
    TODO(B): if you build richer docs during annotation, swap them in here.
    """
    correct = 0
    for it in items:
        q1, q2 = it.target, it.negation
        d1, d2 = it.paraphrase, it.negation
        ok1 = score_fn(q1, d1) > score_fn(q1, d2)
        ok2 = score_fn(q2, d2) > score_fn(q2, d1)
        correct += int(ok1 and ok2)
    return correct / max(len(items), 1)


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
