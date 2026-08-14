"""
Offline check for run_eval's NLI-checkpoint wiring.  ===  PERSON B  ===

No network, no models, no GPU:

    python -m tests.test_run_eval

`nli_rerank` used to be built with zero arguments, always the released
checkpoint, so there was no way to point the harness at the checkpoint
`train_nli.py` produces, and no way to run both side by side for a
comparison. Two things are worth a planted-failure test here: that
`--nli-model` actually reaches the `NLIReranking` instance rather than being
silently ignored, and that running two different NLI checkpoints against the
same `--out` accumulates both rows instead of the second erasing the first —
the exact failure mode `train_nli.append_result` already guards against on
the training side (see tests/test_nli_data.py).
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from src.harness.run_eval import append_or_replace, build_interventions
from src.interventions.nli_rerank import JOINED, PAIR, NLIReranking


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)
        print(f"[FAIL] {message}")


def main() -> int:
    failures: list = []

    print("== build_interventions: nli_kwargs actually reach NLIReranking ==")
    default_interv = build_interventions(["nli_rerank"])[0]
    check(default_interv.model_name == NLIReranking.DEFAULT_MODEL_NAME,
          "with no nli_kwargs, the released checkpoint must still be the default", failures)

    custom_kwargs = dict(lam=1.0, model_name="checkpoints/alephbert-hebnli-clean",
                          model_subfolder=None, pair_encoding=PAIR)
    custom_interv = build_interventions(["nli_rerank"], custom_kwargs)[0]
    check(custom_interv.model_name == "checkpoints/alephbert-hebnli-clean",
          "--nli-model must reach the NLIReranking instance, not be ignored", failures)
    check(custom_interv.pair_encoding == PAIR,
          "--nli-encoding must reach the NLIReranking instance", failures)

    baseline_interv = build_interventions(["baseline"], custom_kwargs)[0]
    check(not hasattr(baseline_interv, "model_name"),
          "nli_kwargs must not leak into unrelated interventions", failures)

    print("\n== append_or_replace: two NLI checkpoints coexist in one table ==")
    with tempfile.TemporaryDirectory() as tmpdir:
        table = Path(tmpdir) / "results_nli_rerank.csv"
        released_row = {
            "model": "multilingual-e5", "intervention": "nli_rerank", "n_test": 151,
            "cosine_gap": 0.05, "nevir_rank": 0.8,
            "nli_checkpoint": NLIReranking.DEFAULT_MODEL_NAME,
            "nli_subfolder": NLIReranking.DEFAULT_MODEL_SUBFOLDER,
            "nli_encoding": JOINED, "nli_lam": 1.0,
        }
        check(append_or_replace([dict(released_row)], table) == 1,
              "first run should write 1 row", failures)

        ours_row = dict(released_row, nli_checkpoint="checkpoints/alephbert-hebnli-clean",
                         nli_subfolder="", nli_encoding=PAIR, cosine_gap=0.09, nevir_rank=0.85)
        check(append_or_replace([dict(ours_row)], table) == 2,
              "a different --nli-model must append, not overwrite the released row", failures)

        with table.open(encoding="utf-8", newline="") as f:
            written = list(csv.DictReader(f))
        by_checkpoint = {r["nli_checkpoint"]: r["cosine_gap"] for r in written}
        check(by_checkpoint.get(NLIReranking.DEFAULT_MODEL_NAME) == "0.05",
              f"the released checkpoint's row was lost: {by_checkpoint}", failures)
        check(by_checkpoint.get("checkpoints/alephbert-hebnli-clean") == "0.09",
              f"our checkpoint's row was lost: {by_checkpoint}", failures)

        print("\n== append_or_replace: re-running the SAME config replaces, not duplicates ==")
        rerun_row = dict(ours_row, cosine_gap=0.11)
        check(append_or_replace([dict(rerun_row)], table) == 2,
              "an identical (model, intervention, nli_checkpoint, nli_encoding, nli_lam) "
              "must replace its row rather than adding a third", failures)
        with table.open(encoding="utf-8", newline="") as f:
            written = list(csv.DictReader(f))
        by_checkpoint = {r["nli_checkpoint"]: r["cosine_gap"] for r in written}
        check(by_checkpoint.get("checkpoints/alephbert-hebnli-clean") == "0.11",
              f"the re-run should have updated the value, got {by_checkpoint}", failures)

        print("\n== append_or_replace: baseline/projection rows (blank nli_* fields) survive too ==")
        baseline_row = {
            "model": "multilingual-e5", "intervention": "baseline", "n_test": 151,
            "cosine_gap": 0.02, "nevir_rank": 0.80,
            "nli_checkpoint": "", "nli_subfolder": "", "nli_encoding": "", "nli_lam": "",
        }
        check(append_or_replace([dict(baseline_row)], table) == 3,
              "a baseline row (blank nli_* config) must coexist with the nli_rerank rows", failures)

    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    print("\nall run_eval checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
