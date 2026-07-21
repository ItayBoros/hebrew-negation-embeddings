"""Check the AlephBERT NLI checkpoint's undocumented label indices.

Run from the repository root:

    python -m src.interventions.check_nli_labels

The examples are deliberately obvious, but predictions remain empirical
evidence. Compare them with the mapping in the authors' training code before
changing the defaults in ``NLIReranking``.
"""
from __future__ import annotations

import argparse

from .nli_rerank import NLIReranking


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
        description="Print raw AlephBERT NLI probabilities for obvious pairs."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device such as 'cuda' or 'cpu' (default: auto-detect).",
    )
    args = parser.parse_args()

    reranker = NLIReranking(device=args.device)

    for expected, premise, hypothesis in EXAMPLES:
        probabilities = reranker.raw_label_probabilities(premise, hypothesis)
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)

        print(f"Expected: {expected}")
        for index, probability in enumerate(probabilities):
            print(f"LABEL_{index}: {probability:.6f}")
        print(f"Predicted index: {predicted_index}")
        print()


if __name__ == "__main__":
    main()
