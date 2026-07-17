"""
NLI re-ranking.  ===  PERSON B  ===

Idea: don't trust the embedding for the final call. Run a Hebrew NLI model
(trained on HebNLI) over the pair. If it predicts CONTRADICTION, the two
sentences mean opposite things -> push their similarity DOWN.

This is a STUB. Fill it in for M2. Suggested recipe:
  1. Load a Hebrew NLI model (fine-tune a Hebrew encoder on HebNLI, or use an
     existing NLI head). Cache it; it is heavy — keep the checkpoint on Drive.
  2. For a pair (a, b), get P(entailment), P(neutral), P(contradiction).
  3. Map to a similarity, e.g.:
         score = cos(a, b) - lambda * P(contradiction)
     or a pure-NLI variant:
         score = P(entailment) - P(contradiction)
     Try both; report which trades off better against STS.

Keep the SAME train/test discipline: if you tune lambda, tune it on the
probe TRAIN split only.
"""
from __future__ import annotations

from .base import Intervention
from .baseline import cosine


class NLIReranking(Intervention):
    name = "nli_rerank"

    def __init__(self, lam: float = 1.0, model_name: str = "TODO-hebrew-nli"):
        self.lam = lam
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _load(self):
        # TODO(M2): load the Hebrew NLI model here (transformers pipeline or a
        # fine-tuned classifier). Set self._model.
        raise NotImplementedError("Load the Hebrew NLI model — see recipe above.")

    def _contradiction_prob(self, a: str, b: str) -> float:
        # TODO(M2): return P(contradiction) for the ordered pair (a, b).
        raise NotImplementedError

    def score(self, a: str, b: str, embedder) -> float:
        # Reference blend once _contradiction_prob is implemented:
        #   return cosine(*embedder.encode([a, b])) - self.lam * self._contradiction_prob(a, b)
        raise NotImplementedError("Implement NLI re-ranking for M2.")
