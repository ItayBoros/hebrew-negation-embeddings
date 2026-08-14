"""
Offline check for the NLI test-set scoring logic.  ===  PERSON B  ===

No network, no models, no GPU — everything in `src/nli/eval_nli.py` that does
not touch torch/transformers, exercised against small hand-built arrays:

    python -m tests.test_nli_eval

What is worth a test here, and why: the model-loading and forward-pass code in
`eval_nli.py` is a thin, well-trodden use of the transformers API, the same
calls `train_nli.py` already makes on a GPU. The parts that are actually new
and easy to get quietly wrong are the scoring arithmetic (does a class with
zero predictions really come back as 0.0 and not a crash or a NaN?) and the
row count guard that stands between "evaluated the real test set" and
"evaluated something else that happened to be at that path".
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.data import hebnli
from src.nli.eval_nli import (
    build_predictions_rows, build_summary, check_test_file, cross_entropy,
    pair_encoding_for, softmax, verify_label_mapping, write_predictions_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hebnli_sample.jsonl"


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)
        print(f"[FAIL] {message}")


def main() -> int:
    failures: list = []

    print("== softmax ==")
    probs = softmax(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    check(np.allclose(probs.sum(axis=1), 1.0), "rows must sum to 1", failures)
    check(probs[0, 0] > probs[0, 1], "the larger logit must win", failures)
    check(np.allclose(probs[1], 1.0 / 3), "equal logits must give a uniform row", failures)

    print("\n== cross_entropy ==")
    certain = softmax(np.array([[100.0, 0.0, 0.0]]))
    loss = cross_entropy(certain, np.array([0]))
    check(loss < 1e-6, f"a near-certain correct prediction should have ~0 loss, got {loss}", failures)
    confident_wrong = softmax(np.array([[100.0, 0.0, 0.0]]))
    loss_wrong = cross_entropy(confident_wrong, np.array([2]))
    check(np.isfinite(loss_wrong) and loss_wrong > 10,
          f"a confidently wrong prediction should be a large, finite loss, got {loss_wrong}", failures)

    print("\n== pair_encoding_for ==")
    check(pair_encoding_for(1) == "pair_without_segment_ids",
          "type_vocab_size=1 (AlephBERT) must drop segment ids", failures)
    check(pair_encoding_for(2) == "pair_with_segment_ids",
          "type_vocab_size=2 (AlephBERTGimmel) must keep segment ids", failures)

    print("\n== verify_label_mapping ==")
    try:
        verify_label_mapping({0: "entailment", 1: "neutral", 2: "contradiction"})
    except ValueError:
        check(False, "the correct mapping must not raise", failures)
    try:
        verify_label_mapping({0: "neutral", 1: "entailment", 2: "contradiction"})
        check(False, "a swapped mapping must raise", failures)
    except ValueError:
        pass

    print("\n== build_summary: a class the model never predicts ==")
    # 6 true examples, 2 per class; the model only ever says entailment or
    # contradiction, so `neutral` has zero predicted rows and would divide by
    # zero under sklearn's default settings.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 2, 2, 2])
    summary = build_summary(y_true, y_pred, "chk", "test.jsonl", "pair_without_segment_ids", 0.42)

    check(summary["n_examples"] == 6, "n_examples should count every row", failures)
    check(summary["per_class"]["neutral"]["precision"] == 0.0,
          "an unpredicted class must report precision 0.0, not crash or NaN", failures)
    check(summary["per_class"]["neutral"]["recall"] == 0.0,
          "an unpredicted class must report recall 0.0, not crash or NaN", failures)
    check(summary["per_class"]["neutral"]["support"] == 2,
          "support counts true labels, not predictions, so neutral is still 2", failures)
    check(abs(summary["accuracy"] - 4 / 6) < 1e-9,
          f"expected accuracy 4/6, got {summary['accuracy']}", failures)
    check(summary["confusion_matrix"]["labels"] == ["entailment", "neutral", "contradiction"],
          "confusion matrix label order must be entailment, neutral, contradiction", failures)
    check(summary["confusion_matrix"]["matrix"] == [[2, 0, 0], [1, 0, 1], [0, 0, 2]],
          f"unexpected confusion matrix {summary['confusion_matrix']['matrix']}", failures)
    check(summary["label_ids"] == {"entailment": 0, "neutral": 1, "contradiction": 2},
          "label_ids must match hebnli.LABELS order", failures)

    print("\n== build_predictions_rows + write_predictions_csv ==")
    rows = hebnli.load_jsonl(FIXTURE)[:3]
    probs3 = softmax(np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]))
    y_true3 = np.array([0, 0, 2])
    y_pred3 = probs3.argmax(axis=1)  # [0, 1, 2] -> second row wrong
    pred_rows = build_predictions_rows(rows, y_true3, y_pred3, probs3)

    check(len(pred_rows) == 3, "one predictions row per input row", failures)
    check(pred_rows[0]["pair_id"] == rows[0].pair_id, "pair_id must come from the source row", failures)
    check(pred_rows[0]["correct"] is True, "row 0 (true=pred=entailment) should be correct", failures)
    check(pred_rows[1]["correct"] is False, "row 1 (true=entailment, pred=neutral) should be wrong", failures)
    check(abs(pred_rows[0]["prob_entailment"] - probs3[0, 0]) < 1e-9,
          "prob_entailment must be probs[:, 0]", failures)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "predictions.csv"
        write_predictions_csv(pred_rows, csv_path)
        with csv_path.open(encoding="utf-8", newline="") as f:
            written = list(csv.DictReader(f))
        check(len(written) == 3, "csv must round-trip every row", failures)
        check(set(written[0]) == {"pair_id", "true_label", "predicted_label", "prob_entailment",
                                   "prob_neutral", "prob_contradiction", "correct"},
              f"unexpected csv columns {sorted(written[0])}", failures)
        check(written[0]["true_label"] == "entailment", "labels must be written as names, not indices", failures)

    print("\n== check_test_file: the row-count guard ==")
    all_rows = hebnli.load_jsonl(FIXTURE)
    n = len(all_rows)
    check(len(check_test_file(FIXTURE, expected_n=n)) == n,
          "the correct expected_n must pass", failures)
    check(len(check_test_file(FIXTURE, expected_n=0)) == n,
          "expected_n=0 must skip the count check entirely", failures)
    try:
        check_test_file(FIXTURE, expected_n=n + 1)
        check(False, "a wrong expected_n must raise", failures)
    except ValueError:
        pass
    try:
        check_test_file(Path(FIXTURE).parent / "does_not_exist.jsonl", expected_n=0)
        check(False, "a missing file must raise", failures)
    except FileNotFoundError:
        pass

    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    print("\nall NLI eval checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
