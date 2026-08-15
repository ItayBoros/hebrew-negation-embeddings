"""
Offline checks for directional lambda selection.  ===  PERSON B  ===

No network, no models, no GPU:

    python -m tests.test_lambda_sweep

Four things here can go wrong quietly, which is why each gets a planted failure
rather than a smoke test:

1. **Direction.** NLI is asymmetric and the whole experiment rests on
   `target` being the premise. Reversing a pair, or averaging the two
   directions, changes every number without raising anything.
2. **The selection rule.** "Highest accuracy among eligible, ties by gap, then
   the smallest lambda" has to produce exactly one lambda per model, has to
   refuse a lambda that busts the STS budget however good its accuracy, and has
   to break ties the same way every run. Ties are not hypothetical: accuracy is
   a fraction over 152 items.
3. **The cost model.** Both halves of the blend are lambda-independent and the
   NLI half is embedder-independent. If either gets recomputed per lambda or per
   model, the sweep still returns the right numbers — just tens of thousands of
   forward passes later.
4. **The dev/test wall.** Lambda selection must never read a held-out file, and
   the test stage must never write a selection.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.harness import sts as sts_data
from src.harness.lambda_sweep import (
    BASELINE, BASELINE_AND_SELECTED, LAMBDA_GRID, MAX_STS_SPEARMAN_DROP, PROBE_TEST,
    SELECTED, PairComponents, _assert_development_only, _load_stage_data,
    cosine_components,
    load_selection, probe_metrics, probe_pairs, run_dev, run_test, select_lambda,
    sweep_lambdas,
)
from src.harness.models import FakeEmbedder
from src.interventions.baseline import cosine
from src.interventions.nli_rerank import PAIR
from src.schema import ProbeItem, save_probe


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)
        print(f"[FAIL] {message}")


class FakeNLI:
    """A stand-in scorer with the two properties the sweep depends on.

    Directional (it looks only at the hypothesis for the negation cue) and
    counted, so a test can prove each pair was scored exactly once.
    """

    model_name = "fake-nli-checkpoint"
    pair_encoding = PAIR

    def __init__(self) -> None:
        self.calls = 0
        self.pairs_scored = 0

    def nli_scores(self, pairs, batch_size: int = 32, progress=None):
        self.calls += 1
        self.pairs_scored += len(pairs)
        return [self.score_pair(premise, hypothesis) for premise, hypothesis in pairs]

    @staticmethod
    def score_pair(premise: str, hypothesis: str) -> float:
        """Contradiction iff the *hypothesis* negates. Asymmetric on purpose."""
        if "NEG" in hypothesis and "NEG" not in premise:
            return -0.9
        if "NEG" in premise and "NEG" not in hypothesis:
            return -0.4          # the reverse direction is scored differently
        return 0.7


def make_probe(path: Path, n: int = 12, split: str = "train") -> None:
    items = [
        ProbeItem(id=f"t{i}", target=f"sentence {i} about topic {i % 4}",
                  paraphrase=f"sentence {i} regarding topic {i % 4}",
                  negation=f"sentence {i} NEG about topic {i % 4}",
                  source="handwritten", split=split)
        for i in range(n)
    ]
    save_probe(items, path)


def make_sts(path: Path, n: int = 20, split: str = "dev") -> None:
    """A small STS csv, BOM included — the real files have one."""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "split", "genre", "sentence1_en", "sentence2_en",
                         "sentence1_he", "sentence2_he", "gold_score"])
        for i in range(n):
            # Every third hypothesis negates, so the NLI term varies across the
            # set: a constant term makes every blend a monotone transform of
            # cosine and the eligibility rule untestable.
            right = f"pair {i} right side {i % 5}" + (" NEG" if i % 3 == 0 else "")
            writer.writerow([f"fx-{i:04d}", split, "fixture", "en a", "en b",
                             f"pair {i} left side", right,
                             round(5.0 * (i % 6) / 5.0, 2)])


def main() -> int:  # noqa: C901 - a checklist, read top to bottom
    failures: list = []

    print("== the grid ==")
    check(len(LAMBDA_GRID) == 21, f"grid must have 21 points, has {len(LAMBDA_GRID)}", failures)
    check(LAMBDA_GRID[0] == 0.0 and LAMBDA_GRID[-1] == 1.0,
          f"grid must run 0.00 .. 1.00, got {LAMBDA_GRID[0]} .. {LAMBDA_GRID[-1]}", failures)
    check(all(round(b - a, 10) == 0.05 for a, b in zip(LAMBDA_GRID, LAMBDA_GRID[1:])),
          f"grid must step by 0.05: {LAMBDA_GRID}", failures)
    check(all(lam == round(lam, 2) for lam in LAMBDA_GRID),
          f"grid values must be exact 2-decimal floats: {LAMBDA_GRID}", failures)

    print("\n== direction: target is the premise, and reversing changes the score ==")
    item = ProbeItem(id="d1", target="a claim", paraphrase="a restatement",
                     negation="a NEG claim", source="handwritten", split="train")
    para_pairs, neg_pairs = probe_pairs([item])
    check(para_pairs == [("a claim", "a restatement")],
          f"paraphrase pair must be (target, paraphrase): {para_pairs}", failures)
    check(neg_pairs == [("a claim", "a NEG claim")],
          f"negation pair must be (target, negation), not reversed: {neg_pairs}", failures)
    forward = FakeNLI.score_pair(*neg_pairs[0])
    backward = FakeNLI.score_pair(neg_pairs[0][1], neg_pairs[0][0])
    check(forward != backward,
          "the fixture scorer must be asymmetric or this file cannot detect a "
          "reversed pair at all", failures)

    print("\n== blend: lambda=0 is cosine, lambda=1 is NLI, and cosine matches baseline ==")
    embedder = FakeEmbedder()
    pairs = [("one text", "another text"), ("third text", "fourth text")]
    cosines = cosine_components(pairs, embedder)
    for (a, b), value in zip(pairs, cosines):
        va, vb = embedder.encode([a, b])
        check(abs(value - cosine(va, vb)) < 1e-12,
              f"vectorised cosine must equal interventions.baseline.cosine for ({a}, {b})",
              failures)
    components = PairComponents(cosines, np.array([-0.9, 0.7]))
    check(np.allclose(components.blend(0.0), cosines),
          "lambda=0 must be exactly the cosine baseline", failures)
    check(np.allclose(components.blend(1.0), [-0.9, 0.7]),
          "lambda=1 must be exactly the NLI term", failures)
    check(np.allclose(components.blend(0.25), 0.75 * cosines + 0.25 * np.array([-0.9, 0.7])),
          "the blend must be (1-lam)*cosine + lam*nli", failures)

    print("\n== probe_metrics: pairwise_accuracy is per item, not a comparison of means ==")
    # Item 1 ranks backwards, items 2-4 rank correctly: accuracy 0.75, and the
    # mean gap stays positive - the case where the two metrics disagree.
    para = PairComponents(np.array([0.1, 0.9, 0.9, 0.9]), np.zeros(4))
    neg = PairComponents(np.array([0.5, 0.1, 0.1, 0.1]), np.zeros(4))
    metrics = probe_metrics(para, neg, 0.0)
    check(abs(metrics["pairwise_accuracy"] - 0.75) < 1e-12,
          f"pairwise_accuracy should be 0.75, got {metrics['pairwise_accuracy']}", failures)
    check(abs(metrics["mean_gap"] - 0.5) < 1e-12,
          f"mean_gap should be 0.5, got {metrics['mean_gap']}", failures)
    check(metrics["mean_gap"] > 0 and metrics["pairwise_accuracy"] < 1.0,
          "this fixture is meant to have a positive gap with a wrong item in it", failures)
    check(metrics["probe_n"] == 4, f"probe_n should be 4, got {metrics['probe_n']}", failures)

    print("\n== the selection rule ==")

    def row(lam, acc, gap, drop):
        return {"lambda": lam, "pairwise_accuracy": acc, "mean_gap": gap,
                "sts_spearman_drop": drop,
                "eligible": drop <= MAX_STS_SPEARMAN_DROP + 1e-12}

    # The best accuracy on the grid is at 0.90, but it costs 0.05 Spearman.
    rows = [row(0.0, 0.60, 0.10, 0.0), row(0.25, 0.72, 0.30, 0.01),
            row(0.50, 0.72, 0.40, 0.02), row(0.90, 0.95, 0.90, 0.05)]
    check(select_lambda(rows) == 0.50,
          "must pick 0.50: highest accuracy among eligible, tie broken by mean_gap "
          f"— got {select_lambda(rows)}", failures)

    rows = [row(0.0, 0.60, 0.10, 0.0), row(0.30, 0.80, 0.50, 0.0),
            row(0.70, 0.80, 0.50, 0.0)]
    check(select_lambda(rows) == 0.30,
          f"an accuracy+gap tie must go to the smaller lambda — got {select_lambda(rows)}",
          failures)

    rows = [row(0.0, 0.60, 0.10, 0.0), row(0.55, 0.99, 0.90, MAX_STS_SPEARMAN_DROP + 0.001)]
    check(select_lambda(rows) == 0.0,
          "a lambda just over the STS budget must be refused however good its accuracy",
          failures)
    rows = [row(0.0, 0.60, 0.10, 0.0), row(0.55, 0.99, 0.90, MAX_STS_SPEARMAN_DROP)]
    check(select_lambda(rows) == 0.55,
          "a drop of exactly the budget is inside it, not outside", failures)

    print("\n== sweep_lambdas: one selected lambda, eligibility measured per model ==")
    rng = np.random.default_rng(0)
    sts_components = PairComponents(rng.normal(size=40), rng.normal(size=40))
    gold = np.arange(40, dtype=float)
    swept = sweep_lambdas(
        PairComponents(np.full(12, 0.8), np.full(12, 0.7)),
        PairComponents(np.full(12, 0.75), np.full(12, -0.9)),
        sts_components, gold,
    )
    check(len(swept) == 21, f"a sweep must have one row per grid point, got {len(swept)}",
          failures)
    check(sum(r["selected"] for r in swept) == 1,
          f"exactly one lambda must be selected, got {sum(r['selected'] for r in swept)}",
          failures)
    zero = next(r for r in swept if r["lambda"] == 0.0)
    check(zero["eligible"] and abs(zero["sts_spearman_drop"]) < 1e-12,
          "lambda=0 is the STS baseline: drop 0, always eligible", failures)
    chosen = next(r for r in swept if r["selected"])
    eligible = [r for r in swept if r["eligible"]]
    check(chosen["pairwise_accuracy"] == max(r["pairwise_accuracy"] for r in eligible),
          "the selected lambda must have the best accuracy of the eligible ones", failures)

    print("\n== the dev/test wall ==")
    if Path(PROBE_TEST).exists():
        try:
            _assert_development_only(PROBE_TEST)
            check(False, f"lambda selection must refuse to read {PROBE_TEST}", failures)
        except ValueError:
            print(f"[ok] refused {PROBE_TEST}")
        try:
            _assert_development_only(sts_data.TEST_PATH)
            check(False, f"lambda selection must refuse to read {sts_data.TEST_PATH}",
                  failures)
        except ValueError:
            print(f"[ok] refused {sts_data.TEST_PATH}")
    else:
        print("[skip] repo data not present; path guard not exercised")

    print("\n== end to end: dev locks a lambda, test reads it and cannot change it ==")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        probe_train, probe_test = tmp / "probe_train.jsonl", tmp / "probe_test.jsonl"
        sts_dev, sts_test = tmp / "sts_dev.csv", tmp / "sts_test.csv"
        make_probe(probe_train, 12)
        make_probe(probe_test, 9, split="test")
        make_sts(sts_dev, 20)
        make_sts(sts_test, 15, split="test")
        dev_csv, selected_json, test_csv = (tmp / "dev.csv", tmp / "selected.json",
                                            tmp / "test.csv")

        nli = FakeNLI()

        # A copied or renamed held-out file must still be rejected by its
        # contents, not only by matching one known filesystem path.
        disguised_probe_test = tmp / "looks_like_train.jsonl"
        disguised_sts_test = tmp / "looks_like_dev.csv"
        make_probe(disguised_probe_test, 3, split="test")
        make_sts(disguised_sts_test, 3, split="test")
        try:
            _load_stage_data(disguised_probe_test, sts_dev, nli, 32,
                             expected_probe_split="train", expected_sts_split="dev")
            check(False, "dev must reject test Probe items even under a renamed path",
                  failures)
        except ValueError:
            print("[ok] refused renamed Probe-test content")
        try:
            _load_stage_data(probe_train, disguised_sts_test, nli, 32,
                             expected_probe_split="train", expected_sts_split="dev")
            check(False, "dev must reject STS-test rows even under a renamed path",
                  failures)
        except ValueError:
            print("[ok] refused renamed STS-test content")

        # Two embedder keys, both served by FakeEmbedder: the point is that the
        # NLI term is computed once and shared, not once per model.
        models = ["fake", "labse"]
        dev_rows = run_dev(models=models, nli=nli, probe_path=probe_train,
                           sts_path=sts_dev, out_csv=dev_csv, out_json=selected_json,
                           embedder_factory=lambda key: FakeEmbedder())

        check(nli.calls == 3,
              "the NLI model must be run once per pair set (paraphrase, negation, sts) "
              f"and reused for every embedder and every lambda — it ran {nli.calls} times",
              failures)
        check(nli.pairs_scored == 12 + 12 + 20,
              f"each pair must be scored exactly once, got {nli.pairs_scored}", failures)
        check(len(dev_rows) == 2 * 21, f"expected 42 sweep rows, got {len(dev_rows)}",
              failures)

        # A re-sweep of one model must replace that model's 21 rows and leave the
        # other model's alone - Colab sessions do run these one model at a time.
        run_dev(models=["labse"], nli=nli, probe_path=probe_train, sts_path=sts_dev,
                out_csv=dev_csv, out_json=selected_json,
                embedder_factory=lambda key: FakeEmbedder())
        with dev_csv.open(encoding="utf-8", newline="") as f:
            all_written = list(csv.DictReader(f))
        check(len(all_written) == 42,
              f"re-sweeping one model must replace its rows, not add 21 more: "
              f"{len(all_written)} rows", failures)
        check(sorted({r["model"] for r in all_written}) == ["fake", "labse"],
              f"both models must survive a single-model re-sweep: "
              f"{sorted({r['model'] for r in all_written})}", failures)
        written = [r for r in all_written if r["model"] == "fake"]
        check(len(written) == 21, f"expected 21 rows for one model, got {len(written)}",
              failures)
        check(sum(r["selected"] == "True" for r in written) == 1,
              "exactly one row per model may be flagged selected", failures)
        check([r["nli_checkpoint"] for r in written] == ["fake-nli-checkpoint"] * 21,
              "every row must record the NLI checkpoint that produced it", failures)
        check(all(r["directional"] == "True" for r in written),
              "every row must record that scoring was directional", failures)
        check([float(r["lambda"]) for r in written] == sorted(float(r["lambda"])
                                                              for r in written),
              "the dev table should be sorted by lambda", failures)

        locked = load_selection(selected_json)
        check(sorted(locked["selected"]) == ["fake", "labse"],
              f"one locked lambda per model, got {sorted(locked['selected'])}", failures)
        locked_lambda = locked["selected"]["fake"]["lambda"]
        check(locked_lambda in LAMBDA_GRID,
              f"the locked lambda must be a grid point, got {locked_lambda}", failures)

        lock_before = selected_json.read_text(encoding="utf-8")
        try:
            run_dev(models=["fake"], nli=nli, probe_path=probe_train,
                    sts_path=sts_dev, out_csv=dev_csv, out_json=selected_json,
                    grid=(0.0, 0.5, 1.0),
                    embedder_factory=lambda key: FakeEmbedder())
            check(False, "partial dev sweeps with a different grid must not merge",
                  failures)
        except ValueError:
            print("[ok] refused to mix an incompatible grid into the lock")
        check(selected_json.read_text(encoding="utf-8") == lock_before,
              "an incompatible dev run must leave the selection lock unchanged", failures)

        test_rows = run_test(models=["fake"], nli=nli, probe_path=probe_test,
                             sts_path=sts_test, selected_json=selected_json,
                             out_csv=test_csv, embedder_factory=lambda key: FakeEmbedder())
        configurations = [r["configuration"] for r in test_rows]
        if locked_lambda == 0.0:
            check(configurations == [BASELINE_AND_SELECTED],
                  f"a selected lambda of 0 must not be run twice: {configurations}", failures)
        else:
            check(configurations == [BASELINE, SELECTED],
                  f"expected a baseline row and a selected row: {configurations}", failures)
            check([r["lambda"] for r in test_rows] == [0.0, locked_lambda],
                  "the test stage must evaluate exactly lambda=0 and the locked lambda",
                  failures)
        check(all(r["probe_n"] == 9 for r in test_rows),
              "the test stage must score the probe-test items", failures)
        check(all(r["sts_n"] == 15 for r in test_rows),
              "the test stage must score the STS-test pairs", failures)

        check(json.loads(selected_json.read_text(encoding="utf-8")) == locked,
              "the test stage must not modify the locked selection file", failures)

        print("\n== the test stage refuses a mismatched or missing selection ==")
        other = FakeNLI()
        other.model_name = "some-other-checkpoint"
        try:
            run_test(models=["fake"], nli=other, probe_path=probe_test, sts_path=sts_test,
                     selected_json=selected_json, out_csv=test_csv,
                     embedder_factory=lambda key: FakeEmbedder())
            check(False, "a lambda selected under a different NLI checkpoint must not be "
                  "silently reused", failures)
        except ValueError:
            print("[ok] refused a checkpoint the lambdas were not selected with")

        try:
            run_test(models=["fake"], nli=nli, probe_path=probe_test, sts_path=sts_test,
                     selected_json=tmp / "never_written.json", out_csv=test_csv,
                     embedder_factory=lambda key: FakeEmbedder())
            check(False, "the test stage must not run before a selection exists", failures)
        except FileNotFoundError:
            print("[ok] refused to evaluate before lambda was selected")

    print("\n== the real Hebrew STS dev file is readable and wired in ==")
    if Path(sts_data.DEV_PATH).exists():
        dev_pairs = sts_data.load_sts(sts_data.DEV_PATH)
        check(len(dev_pairs) == 1500, f"STS-dev should have 1500 pairs, got {len(dev_pairs)}",
              failures)
        check(all(a and b for a, b, _ in dev_pairs), "no STS sentence may be empty", failures)
        check(all(0.0 <= g <= 5.0 for _, _, g in dev_pairs),
              "STS gold scores must stay on the inherited 0-5 scale", failures)
        first = sts_data.load_sts_pairs(sts_data.DEV_PATH)[0]
        check(first.pair_id == "stsb-dev-000001",
              f"the BOM must be stripped from the first column: {first.pair_id!r}", failures)
    else:
        print("[skip] repo data not present")

    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    print("\nall lambda sweep checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
