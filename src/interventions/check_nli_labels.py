"""Check an NLI checkpoint's label indices against obvious Hebrew pairs.

Run from the repository root:

    # the released checkpoint, whose config exposes only LABEL_0/1/2
    python -m src.interventions.check_nli_labels

    # one we trained ourselves, which carries real names in its config
    python -m src.interventions.check_nli_labels \
        --model checkpoints/alephbert-hebnli-clean --subfolder ""

The examples are deliberately obvious, but predictions remain empirical
evidence. Compare them with the mapping in the authors' training code before
changing the defaults in ``NLIReranking``.

For a checkpoint from ``src/nli/train_nli.py`` the question is the opposite one:
the mapping is already written into ``config.id2label``, so this is confirming
that the names there survived training and describe what the model actually
does — a config can say anything.
"""
from __future__ import annotations

import argparse

from .nli_rerank import JOINED, PAIR, NLIReranking


EXAMPLES = [
    (
        "entailment",
        "משה אכל ירקות בריאים.",
        "משה בריא",
    ),
    (
        "contradiction",
        "דנה קראה ספר.",
        "דנה לא קראה ספר.",
    ),
    (
        "neutral",
        "דנה קראה ספר.",
        "יוסי גר בחיפה.",
    ),
    (
        "contradiction",
        "יוסי גר בחיפה",
        "יוסי לא גר בחיפה.",
    ),
    (
        "contradiction",
        "שחר אינו אוהב לאכול תפוח",
        "שחר אוהב לאכול תפוח.",
    ),
    (
        "entailment",
        "רון טבעוני",
        "רון לא אוכל בשר",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print raw NLI probabilities for obvious Hebrew pairs."
    )
    parser.add_argument(
        "--model",
        default=NLIReranking.DEFAULT_MODEL_NAME,
        help="HF id or a local checkpoint directory (default: the released one).",
    )
    parser.add_argument(
        "--subfolder",
        default=NLIReranking.DEFAULT_MODEL_SUBFOLDER,
        help="subfolder inside the HF repo. Pass an empty string for a local "
             "checkpoint, whose files sit at the top level.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        choices=[JOINED, PAIR],
        help=f"how the pair reaches the model (default: {JOINED} for the "
             f"released checkpoint, {PAIR} for anything else — see NLIReranking).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device such as 'cuda' or 'cpu' (default: auto-detect).",
    )
    args = parser.parse_args()

    # A local checkpoint we trained is a `pair` model; the released one is not.
    # Defaulting on the model name keeps the common cases flag-free while still
    # allowing an explicit override.
    encoding = args.encoding or (
        JOINED if args.model == NLIReranking.DEFAULT_MODEL_NAME else PAIR
    )

    reranker = NLIReranking(
        model_name=args.model,
        model_subfolder=args.subfolder or None,
        pair_encoding=encoding,
        device=args.device,
    )

    print(f"model     {args.model}")
    print(f"encoding  {encoding}")
    # Names if the config carries real ones, otherwise the assumed mapping.
    names = reranker.label_names()
    print(f"labels    {names}\n")

    correct = 0
    for expected, premise, hypothesis in EXAMPLES:
        probabilities = reranker.raw_label_probabilities(premise, hypothesis)
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)

        print(f"Expected: {expected}")
        for index, probability in enumerate(probabilities):
            print(f"{names[index]:>14}: {probability:.6f}")
        predicted = names[predicted_index]
        agrees = predicted == expected
        correct += int(agrees)
        print(f"Predicted: {predicted}  [{'ok' if agrees else 'MISMATCH'}]")
        print()

    # Not a metric — six hand-picked examples. It is a smoke test of the label
    # mapping, and a low score means the indices are wrong, not that the model is.
    print(f"{correct}/{len(EXAMPLES)} agree with the assumed label mapping")


if __name__ == "__main__":
    main()
