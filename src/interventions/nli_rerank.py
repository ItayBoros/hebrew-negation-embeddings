"""NLI-adjusted similarity scoring.  ===  PERSON B  ===

The frozen embedding model supplies cosine similarity. A Hebrew NLI classifier
supplies entailment and contradiction probabilities for the ordered pair
``(premise, hypothesis)``. The signals are combined as:

    score = (1 - lambda) * cosine
            + lambda * (P(entailment) - P(contradiction))

where ``0 <= lambda <= 1``. Both terms are in [-1, 1], so the combined score is
also in [-1, 1]. Lambda currently defaults to 1.0 (pure NLI).

The checkpoint exposes only LABEL_0/LABEL_1/LABEL_2. The mapping below comes
from the authors' HebNLI/LCHAIM training code and was confirmed by running
``python -m src.interventions.check_nli_labels`` on obvious Hebrew entailment,
contradiction, and neutral pairs.

The checkpoint files live in the ``AlephBERT`` subfolder of the Hugging Face
repository. Loading is lazy so importing the evaluation harness does not load
PyTorch or download the checkpoint. Lambda tuning is intentionally not
implemented in this class yet.
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import Intervention
from .baseline import cosine


class NLIReranking(Intervention):
    """Blend embedding cosine with an ordered Hebrew NLI prediction."""

    name = "nli_rerank"

    DEFAULT_MODEL_NAME = "oriel9p/AlephBERT-FT-HebNLI-LCHAIM"
    DEFAULT_MODEL_SUBFOLDER = "AlephBERT"

    # The released config has generic names, so do not infer these indices from
    # config.id2label. Authors' training mapping + check_nli_labels.py results:
    # contradiction -> LABEL_0, entailment -> LABEL_1, neutral -> LABEL_2.
    DEFAULT_LABEL_IDS = {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    }

    def __init__(
        self,
        lam: float = 1.0,
        model_name: str = DEFAULT_MODEL_NAME,
        model_subfolder: str = DEFAULT_MODEL_SUBFOLDER,
        device: Optional[str] = None,
        contradiction_index: int = DEFAULT_LABEL_IDS["contradiction"],
        entailment_index: int = DEFAULT_LABEL_IDS["entailment"],
        neutral_index: int = DEFAULT_LABEL_IDS["neutral"],
    ):
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be between 0 and 1")

        label_ids = {
            "contradiction": contradiction_index,
            "entailment": entailment_index,
            "neutral": neutral_index,
        }
        if sorted(label_ids.values()) != [0, 1, 2]:
            raise ValueError("NLI label indices must be a permutation of 0, 1, and 2")

        self.lam = float(lam)
        self.model_name = model_name
        self.model_subfolder = model_subfolder
        self.device = device
        self.label_ids = label_ids

        self._tokenizer = None
        self._model = None
        self._device = None
        self._raw_probability_cache: Dict[tuple[str, str], tuple[float, ...]] = {}

    def _load(self) -> None:
        """Load the tokenizer and sequence classifier on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        load_kwargs = {"subfolder": self.model_subfolder}
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        device_name = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_name)
        self._model.to(self._device)
        self._model.eval()

    def raw_label_probabilities(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, ...]:
        """Return probabilities in raw LABEL_0/LABEL_1/LABEL_2 order."""
        cache_key = (premise, hypothesis)
        cached = self._raw_probability_cache.get(cache_key)
        if cached is not None:
            return cached

        self._load()

        import torch

        # This exactly follows the pair construction used in the authors'
        # training and inference code before calling their tokenizer.
        pair = f"{premise} [SEP] {hypothesis} [SEP]"
        max_length = int(self._model.config.max_position_embeddings)
        inputs = self._tokenizer(
            pair,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}

        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]
            raw_probabilities = torch.softmax(logits, dim=-1).detach().cpu()

        probabilities = tuple(float(value) for value in raw_probabilities)
        self._raw_probability_cache[cache_key] = probabilities
        return probabilities

    def _nli_probabilities(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Map raw classifier outputs to semantic NLI probabilities."""
        raw_probabilities = self.raw_label_probabilities(premise, hypothesis)
        return {
            label: raw_probabilities[label_id]
            for label, label_id in self.label_ids.items()
        }

    def score(self, a: str, b: str, embedder) -> float:
        """Return similarity with ``a`` as premise and ``b`` as hypothesis."""
        if getattr(embedder, "key", None) == "fake":
            raise NotImplementedError(
                "NLI reranking is skipped with FakeEmbedder to keep the fake run offline."
            )

        va, vb = embedder.encode([a, b])
        cosine_score = cosine(va, vb)

        probabilities = self._nli_probabilities(a, b)
        nli_score = probabilities["entailment"] - probabilities["contradiction"]

        combined_score = (
            (1.0 - self.lam) * cosine_score
            + self.lam * nli_score
        )
        return float(max(-1.0, min(1.0, combined_score)))
