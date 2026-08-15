"""NLI-adjusted similarity scoring.  ===  PERSON B  ===

The frozen embedding model supplies cosine similarity. A Hebrew NLI classifier
supplies entailment and contradiction probabilities for the ordered pair
``(premise, hypothesis)``. The signals are combined as:

    score = (1 - lambda) * cosine
            + lambda * (P(entailment) - P(contradiction))

where ``0 <= lambda <= 1``. Both terms are in [-1, 1], so the combined score is
also in [-1, 1]. Lambda currently defaults to 1.0 (pure NLI).

Two checkpoints, two conventions
--------------------------------
This class has to serve both the released checkpoint and the one
``src/nli/train_nli.py`` produces, and they differ in two ways that silently
corrupt scores if confused.

**Label indices.** The released config exposes only LABEL_0/LABEL_1/LABEL_2, so
its mapping is hardcoded below from the authors' training code and confirmed
empirically with ``check_nli_labels``. Our own checkpoints write real names into
``config.id2label``, so the mapping is read from the model and the hardcoded
indices are ignored. ``label_names`` reports whichever applied.

**Pair encoding.** ``JOINED`` builds a single ``"premise [SEP] hypothesis [SEP]"``
string, matching the released checkpoint's own training code. ``PAIR`` passes the
two sentences to the tokenizer as separate arguments, which is what we train
with. The distinction is not cosmetic: a model reads whichever it was trained on
and degrades quietly given the other, with no error to notice.

Under ``PAIR``, ``token_type_ids`` are dropped when the model's
``type_vocab_size`` is 1 — AlephBERT was pretrained without a segment
distinction, so its embedding table has a single row while its tokenizer still
emits a 1 for the second sequence. That is very likely why the released
checkpoint's authors flattened the pair in the first place.

Loading is lazy so importing the evaluation harness does not load PyTorch or
download the checkpoint.

Lambda is selected, not guessed: ``src/harness/lambda_sweep.py`` sweeps the grid
on the development splits and locks one value per embedding model. It drives this
class through ``nli_scores``, the batched entry point, because the sweep needs
every pair's NLI term exactly once and then reuses it for all 21 lambdas.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .base import Intervention
from .baseline import cosine

#: Pair-encoding modes. See the module docstring — picking the wrong one for a
#: checkpoint costs accuracy without raising anything.
JOINED = "joined"   # "premise [SEP] hypothesis [SEP]" as one string
PAIR = "pair"       # tokenizer(premise, hypothesis) as two arguments

#: A config that never had real label names produces these placeholders. Seeing
#: them means id2label carries no information and the caller's indices stand.
_PLACEHOLDER = re.compile(r"^LABEL_\d+$")


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
        model_subfolder: Optional[str] = DEFAULT_MODEL_SUBFOLDER,
        pair_encoding: str = JOINED,
        device: Optional[str] = None,
        contradiction_index: int = DEFAULT_LABEL_IDS["contradiction"],
        entailment_index: int = DEFAULT_LABEL_IDS["entailment"],
        neutral_index: int = DEFAULT_LABEL_IDS["neutral"],
    ):
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be between 0 and 1")
        if pair_encoding not in (JOINED, PAIR):
            raise ValueError(f"pair_encoding must be one of {(JOINED, PAIR)}")

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
        self.pair_encoding = pair_encoding
        self.device = device
        #: assumed mapping; replaced at load time if the config carries real names
        self.label_ids = label_ids
        self.label_source = "assumed"

        self._tokenizer = None
        self._model = None
        self._device = None
        self._use_token_type_ids = True
        self._raw_probability_cache: Dict[tuple[str, str], tuple[float, ...]] = {}

    def _load(self) -> None:
        """Load the tokenizer and sequence classifier on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Omit the key rather than passing None: a local checkpoint has its files
        # at the top level, and `subfolder=None` is not the same as no subfolder
        # as far as from_pretrained is concerned.
        load_kwargs = {"subfolder": self.model_subfolder} if self.model_subfolder else {}
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

        # Prefer what the checkpoint says over what the caller assumed: a config
        # with real names is authoritative, and getting this backwards silently
        # swaps entailment for contradiction — the score flips sign and nothing
        # complains.
        config = self._model.config
        names = [config.id2label[i] for i in sorted(config.id2label)]
        if not any(_PLACEHOLDER.match(str(n)) for n in names):
            self.label_ids = {str(n).lower(): i for i, n in enumerate(names)}
            self.label_source = "config"

        missing = {"entailment", "contradiction"} - set(self.label_ids)
        if missing:
            raise ValueError(
                f"{self.model_name} exposes labels {names} — no {sorted(missing)} to "
                "score with. Pass explicit *_index arguments if the names differ."
            )

        # Same constraint train_nli.py hits: a base pretrained without a segment
        # distinction has a one-row embedding table, and feeding it a 1 raises
        # IndexError rather than warning.
        self._use_token_type_ids = getattr(config, "type_vocab_size", 1) > 1

    def _tokenize(self, pairs: Sequence[Tuple[str, str]]) -> dict:
        """Encode ordered (premise, hypothesis) pairs into a padded batch."""
        max_length = int(self._model.config.max_position_embeddings)
        if self.pair_encoding == JOINED:
            # Exactly the pair construction used in the released checkpoint's
            # training and inference code before calling their tokenizer.
            inputs = self._tokenizer(
                [f"{premise} [SEP] {hypothesis} [SEP]" for premise, hypothesis in pairs],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
        else:
            # What train_nli.py does: two arguments, so the tokenizer places
            # [SEP] itself and truncates the longer side rather than the tail.
            inputs = self._tokenizer(
                [premise for premise, _ in pairs],
                [hypothesis for _, hypothesis in pairs],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            if not self._use_token_type_ids:
                inputs.pop("token_type_ids", None)

        return {name: value.to(self._device) for name, value in inputs.items()}

    def raw_label_probabilities(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, ...]:
        """Return probabilities in raw LABEL_0/LABEL_1/LABEL_2 order."""
        return self.raw_label_probabilities_batch([(premise, hypothesis)])[0]

    def raw_label_probabilities_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 32,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[tuple[float, ...]]:
        """Same as `raw_label_probabilities`, one forward pass per `batch_size`.

        The lambda sweep asks for thousands of pairs at once (1,804 on the
        development splits alone), and one forward pass each is minutes of GPU
        time spent on launch overhead rather than on arithmetic.

        Only pairs missing from the cache are computed, deduplicated first, so
        calling this twice with overlapping inputs costs nothing the second
        time. Padding is masked out, so a pair's probabilities do not depend on
        what it was batched with.
        """
        pairs = [(premise, hypothesis) for premise, hypothesis in pairs]
        # dict.fromkeys and not set(): the batch order stays the input order,
        # which is what makes a progress line mean anything.
        missing = [p for p in dict.fromkeys(pairs) if p not in self._raw_probability_cache]
        if missing:
            self._load()

            import torch

            for start in range(0, len(missing), batch_size):
                chunk = missing[start:start + batch_size]
                inputs = self._tokenize(chunk)
                with torch.inference_mode():
                    logits = self._model(**inputs).logits
                    raw_probabilities = torch.softmax(logits, dim=-1).detach().cpu()
                for pair, row in zip(chunk, raw_probabilities):
                    self._raw_probability_cache[pair] = tuple(float(value) for value in row)
                if progress is not None:
                    progress(min(start + batch_size, len(missing)), len(missing))

        return [self._raw_probability_cache[pair] for pair in pairs]

    def nli_scores(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 32,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[float]:
        """`P(entailment) - P(contradiction)` per ordered pair, in [-1, 1].

        The lambda-independent half of `score`: the blend's NLI term does not
        move with lambda, so a sweep computes this once per pair and reuses it
        across the whole grid.
        """
        if not pairs:
            return []
        raw = self.raw_label_probabilities_batch(pairs, batch_size=batch_size,
                                                  progress=progress)
        # No-op after the batch call above, except when the cache was seeded
        # externally: label_ids is only authoritative once the config has been read.
        self._load()
        entailment = self.label_ids["entailment"]
        contradiction = self.label_ids["contradiction"]
        return [float(row[entailment] - row[contradiction]) for row in raw]

    def label_names(self) -> Dict[int, str]:
        """Class index -> label name, as actually in force after loading.

        Loads the model if it has not been loaded, because until then the
        mapping is only the caller's assumption.
        """
        self._load()
        return {index: label for label, index in sorted(self.label_ids.items(),
                                                        key=lambda kv: kv[1])}

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
