"""
Evaluate a fine-tuned NLI checkpoint on the clean HebNLI test set.  ===  PERSON B  ===

Why a separate module from `train_nli.py`
------------------------------------------
`train_nli.py` already reports a validation score, but that number comes from
`hebnli_val_clean.jsonl` — the split used for model selection while training was
still running. The held-out *test* split has never touched training or model
selection, so this is where the reported number actually comes from.

This module never writes to the checkpoint directory and never calls anything
that updates the model's weights: it loads a finished checkpoint, runs a
forward pass in `eval()` / `no_grad()` mode, and writes two files under
`results/`. Nothing here trains, and nothing here mutates `$CKPT`.

Encoding must match training, not be re-derived by convention
---------------------------------------------------------------
Same rule as `train_nli.py`: the pair goes to the tokenizer as two arguments
so [SEP] lands correctly, and whether `token_type_ids` survive is read off the
model's own `type_vocab_size` rather than assumed. Because this loads the
checkpoint `train_nli.py` already saved, the two are guaranteed to agree —
there is no separate choice to get wrong here.

Label indices come from the checkpoint's own `config.id2label`, written by
`train_nli.py` at save time. `verify_label_mapping` fails loudly if that
mapping does not read `{0: entailment, 1: neutral, 2: contradiction}`, the
same drift `check_nli_labels.py` was built to catch.

    python -m src.nli.eval_nli \\
        --checkpoint /content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean \\
        --test data/raw/hebnli_test_clean.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..data import hebnli
from .train_nli import LABEL_IDS, PAIR_NO_SEGMENTS, PAIR_WITH_SEGMENTS

#: Fixed report order everywhere a class list appears — the confusion matrix,
#: the per-class breakdown, the probability columns in the predictions csv.
CLASS_ORDER = hebnli.LABELS  # ("entailment", "neutral", "contradiction")

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128


# --------------------------------------------------------------------------
# pure logic — importable and testable without torch/transformers installed,
# same invariant `NLIDataset` keeps in train_nli.py
# --------------------------------------------------------------------------

def check_test_file(path: str | Path, expected_n: int) -> List[hebnli.NLIRow]:
    """Load the clean test split and enforce the row count the report depends
    on. `expected_n=0` skips the count check, for fixtures in tests.

    Fails loudly rather than silently evaluating on the wrong set: a lower
    count usually means the contamination filters ran again with a different
    probe or promptID list, and a higher count usually means an unfiltered
    file was pointed at by mistake.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.nli.prepare_data --split test "
            f"--out {path}` first"
        )
    rows = hebnli.load_jsonl(path)
    if expected_n and len(rows) != expected_n:
        raise ValueError(
            f"{path} has {len(rows)} rows, expected exactly {expected_n} — "
            "the contamination-removal filters in src/nli/prepare_data.py may "
            "not have been re-applied to this file"
        )
    return rows


def verify_label_mapping(id2label: Dict[int, str]) -> None:
    """The checkpoint's own `config.id2label`, not an assumption.

    `train_nli.py` writes real names into the config at save time, so a
    mismatch here means the checkpoint being loaded is not the one this
    project trained — the same failure mode `check_nli_labels.py` exists to
    catch before it silently mislabels every prediction.
    """
    expected = {i: label for label, i in LABEL_IDS.items()}
    if dict(id2label) != expected:
        raise ValueError(
            f"checkpoint's id2label {id2label} does not match the expected "
            f"mapping {expected} — re-run check_nli_labels.py against this "
            "checkpoint before trusting any numbers from it"
        )


def pair_encoding_for(type_vocab_size: int) -> str:
    """Same rule `train_nli.train()` uses to decide it, applied to a loaded
    checkpoint instead of a base model about to be fine-tuned."""
    return PAIR_WITH_SEGMENTS if type_vocab_size > 1 else PAIR_NO_SEGMENTS


def softmax(logits):
    """Row-wise softmax. Shifted by the row max first — HebNLI logits are
    small, but the shift is free and removes any dependence on that."""
    import numpy as np

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_true) -> float:
    """Mean negative log-likelihood of the true class — what `Trainer` reports
    as `eval_loss` for this head, recomputed here since inference does not go
    through `Trainer` at all.

    Clipped rather than left to divide-by-zero: a softmax output can floor to
    exactly 0.0 in float32 when a class is overwhelmingly unlikely, and one
    such row should not turn a real loss value into `inf`.
    """
    import numpy as np

    picked = probs[np.arange(len(y_true)), y_true]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def build_summary(
    y_true,
    y_pred,
    checkpoint: str,
    test_path: str,
    pair_encoding: str,
    test_loss: float | None,
) -> dict:
    """Everything `results/nli_test_<key>.json` must contain, per class order
    `CLASS_ORDER` throughout so the confusion matrix and the per-class table
    read the same way.

    `zero_division=0` rather than sklearn's default warning-and-nan: a class
    the model never predicts must still produce a reportable (and low) score,
    not a crash or a silent NaN in the json.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    label_indices = list(range(len(CLASS_ORDER)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_indices, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_indices, average="macro", zero_division=0)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    matrix = confusion_matrix(y_true, y_pred, labels=label_indices)

    return {
        "checkpoint": checkpoint,
        "test_file": test_path,
        "n_examples": int(len(y_true)),
        "label_ids": dict(LABEL_IDS),
        "pair_encoding": pair_encoding,
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(CLASS_ORDER)
        },
        "confusion_matrix": {
            "labels": list(CLASS_ORDER),
            "matrix": matrix.tolist(),
        },
        "test_loss": test_loss,
    }


def build_predictions_rows(rows: Sequence[hebnli.NLIRow], y_true, y_pred, probs) -> List[dict]:
    """One dict per test example for `results/nli_test_predictions.csv`.

    No sentence text — the pair_id already keys back into
    `hebnli_test_clean.jsonl` for anyone who needs it, and the predictions
    file is meant to be small enough to skim.
    """
    out = []
    for i, row in enumerate(rows):
        out.append({
            "pair_id": row.pair_id,
            "true_label": CLASS_ORDER[y_true[i]],
            "predicted_label": CLASS_ORDER[y_pred[i]],
            "prob_entailment": float(probs[i, 0]),
            "prob_neutral": float(probs[i, 1]),
            "prob_contradiction": float(probs[i, 2]),
            "correct": bool(y_true[i] == y_pred[i]),
        })
    return out


def write_predictions_csv(prediction_rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id", "true_label", "predicted_label",
              "prob_entailment", "prob_neutral", "prob_contradiction", "correct"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)


# --------------------------------------------------------------------------
# model-touching — lazy imports, never called by tests
# --------------------------------------------------------------------------

def load_checkpoint(checkpoint: str):
    """Model and tokenizer, both from `checkpoint` — never a base-model id,
    since the released base has not seen this project's label names."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # lazy

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    return model, tokenizer


def predict_logits(
    rows: Sequence[hebnli.NLIRow],
    model,
    tokenizer,
    use_token_type_ids: bool,
    batch_size: int,
    max_length: int,
    device: str,
):
    """Forward pass only — `model.eval()` plus `no_grad()`, no optimiser, no
    `.backward()`, nothing that could change a weight. Dynamic per-batch
    padding, same as `DataCollatorWithPadding` used during training."""
    import numpy as np
    import torch

    logits_chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [r.premise_he for r in batch],
                [r.hypothesis_he for r in batch],
                truncation=True, max_length=max_length, padding=True,
                return_tensors="pt",
            )
            if not use_token_type_ids:
                encoded.pop("token_type_ids", None)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits_chunks.append(model(**encoded).logits.detach().cpu().numpy())
    return np.concatenate(logits_chunks, axis=0)


def evaluate_checkpoint(
    checkpoint: str,
    test_path: str,
    expected_n: int = 883,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str | None = None,
) -> Tuple[dict, List[dict]]:
    """Load, run, score. Returns (summary dict, predictions rows) — the exact
    contents of the two result files, before either is written."""
    import numpy as np
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = check_test_file(test_path, expected_n)

    model, tokenizer = load_checkpoint(checkpoint)
    verify_label_mapping({int(k): v for k, v in model.config.id2label.items()})
    model.to(device)

    type_vocab_size = getattr(model.config, "type_vocab_size", 1)
    use_segments = type_vocab_size > 1
    encoding = pair_encoding_for(type_vocab_size)

    logits = predict_logits(rows, model, tokenizer, use_segments, batch_size, max_length, device)
    probs = softmax(logits)
    y_true = np.array([LABEL_IDS[r.label] for r in rows])
    y_pred = probs.argmax(axis=1)
    test_loss = cross_entropy(probs, y_true)

    summary = build_summary(y_true, y_pred, checkpoint, test_path, encoding, test_loss)
    predictions = build_predictions_rows(rows, y_true, y_pred, probs)
    return summary, predictions


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_summary(summary: dict, device: str) -> None:
    print(f"checkpoint           {summary['checkpoint']}")
    print(f"test file            {summary['test_file']}")
    print(f"examples             {summary['n_examples']}")
    print(f"device               {device}")
    print(f"pair encoding        {summary['pair_encoding']}")
    print(f"\naccuracy             {summary['accuracy']:.4f}")
    print(f"macro precision      {summary['macro_precision']:.4f}")
    print(f"macro recall         {summary['macro_recall']:.4f}")
    print(f"macro F1             {summary['macro_f1']:.4f}")
    if summary["test_loss"] is not None:
        print(f"test loss            {summary['test_loss']:.4f}")

    print("\nper-class:")
    for label in CLASS_ORDER:
        m = summary["per_class"][label]
        print(f"  {label:14s} precision={m['precision']:.4f}  recall={m['recall']:.4f}"
              f"  f1={m['f1']:.4f}  support={m['support']}")

    print("\nconfusion matrix (rows=true, cols=predicted, order="
          f"{', '.join(summary['confusion_matrix']['labels'])}):")
    for label, row in zip(summary["confusion_matrix"]["labels"], summary["confusion_matrix"]["matrix"]):
        print(f"  {label:14s} {row}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint directory to evaluate — same path passed to "
                         "AutoModelForSequenceClassification.from_pretrained")
    ap.add_argument("--test", default="data/raw/hebnli_test_clean.jsonl",
                    help="output of `python -m src.nli.prepare_data --split test`")
    ap.add_argument("--expected-n", type=int, default=883,
                    help="fail if the test file has any other row count; 0 disables")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--summary-out", default=None,
                    help="default: results/nli_test_<checkpoint dir name>.json")
    ap.add_argument("--predictions-out", default="results/nli_test_predictions.csv")
    args = ap.parse_args()

    import torch  # lazy, just to resolve the device for the printed line

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    summary, predictions = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        test_path=args.test,
        expected_n=args.expected_n,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    summary_out = args.summary_out or f"results/nli_test_{Path(args.checkpoint).name}.json"
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_predictions_csv(predictions, args.predictions_out)

    _print_summary(summary, device)
    print(f"\nsummary              -> {summary_out}")
    print(f"predictions          -> {args.predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
