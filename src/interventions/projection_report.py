"""
Projection ablation table.  ===  PERSON A  ===

`run_eval.py` (Person B's) answers "how do the interventions compare?" and
constructs each one with its defaults. This script answers the question that is
mine to answer: **which version of the projection, and why?** It varies the
projection's own knobs while holding everything else fixed, and prints the table
that goes in the results section.

Varied:
  direction   mean_diff vs classifier
  centring    on vs off
  γ selection cross-validated inside train vs on the whole train split
  constraint  plain argmax(gap) vs argmax(gap) subject to unrel <= threshold
  γ           swept inside each fit

The `select` column is worth reading carefully. `select=train` fits the direction
on the train split and then picks the γ that maximises the gap on those same
pairs, so its own reported number is inflated; comparing the two on the *test*
split shows how much of the naive gain was real.

The `constrain` column is the fix for what the unconstrained run actually found:
gap-only selection amplifies until the space collapses onto `direction` — real
models here hit `sim_unrelated` of 0.87-0.996, meaning nearly every sentence
pair looked alike. `--constrain-unrel` (on by default) picks γ by the same
train-only sweep but restricted to configurations whose *own* held-out unrel
stays under `--unrel-threshold`; both rows are still printed so the report can
show how much of the unconstrained gap was real vs collapse.

Reported per configuration, all on the **test** split:
  sim_para / sim_neg / cosine_gap, NevIR-style rank, and the chosen γ.
The baseline row is the same numbers with no intervention at all.

Metrics come from `src.harness.metrics` unchanged — this script does not
reimplement them, so B's definitions stay the single source of truth.

    python -m src.interventions.projection_report --models fake
    python -m src.interventions.projection_report \
        --models multilingual-e5 labse --probe data/probe/probe.jsonl
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import List

import numpy as np

from ..harness.metrics import cosine_gap, nevir_rank
from ..harness.models import get_embedder
from ..schema import ProbeItem, load_probe, split_items
from .baseline import Baseline
from .projection import DIRECTION_METHODS, NegationProjection

FIELDS = [
    "model", "config", "direction", "center", "select", "constrain_unrel",
    "unrel_threshold", "gamma", "at_grid_edge", "constraint_relaxed",
    "n_train", "n_test", "sim_paraphrase", "sim_negation", "cosine_gap", "nevir_rank",
    "sim_unrelated",
]


def unrelated_similarity(
    items: List[ProbeItem],
    score_fn,
    n_pairs: int = 400,
    seed: int = 0,
) -> float:
    """Mean |cos| between the targets of two *different* probe items.

    A stand-in trade-off guard until a Hebrew STS set exists. Two unrelated
    sentences should sit near zero; in a healthy space this stays low no matter
    what the intervention does to negation.

    It is what catches the failure mode `cosine_gap` cannot see. Amplifying γ
    without bound eventually projects every sentence onto `direction`, at which
    point cos between *any* two sentences goes to ±1 — the negation gap looks
    magnificent because the representation has been flattened to one axis and
    everything else is gone. A large gap next to a large `sim_unrelated` is that
    collapse, not a fix.

    Not a replacement for STS: it says nothing about whether *graded* similarity
    survives, only whether unrelated things stay unrelated.
    """
    rng = random.Random(seed)
    targets = [it.target for it in items]
    if len(targets) < 2:
        return float("nan")
    vals = []
    for _ in range(n_pairs):
        a, b = rng.sample(targets, 2)
        vals.append(abs(score_fn(a, b)))
    return float(np.mean(vals))


def evaluate_config(
    train: List[ProbeItem],
    test: List[ProbeItem],
    embedder,
    direction: str,
    center: bool,
    select: str,
    constrain_unrel: bool,
    unrel_threshold: float,
) -> dict:
    proj = NegationProjection(
        direction_method=direction, center=center, select=select,
        constrain_unrel=constrain_unrel, unrel_threshold=unrel_threshold,
    )
    proj.fit(train, embedder)
    score_fn = lambda a, b: proj.score(a, b, embedder)
    config = f"{direction}/{select}{'' if center else '/nocenter'}"
    config += f"/constrain<={unrel_threshold:g}" if constrain_unrel else "/unconstrained"
    return {
        "config": config,
        "direction": direction,
        "center": center,
        "select": proj.selection,
        "constrain_unrel": constrain_unrel,
        "unrel_threshold": unrel_threshold if constrain_unrel else "",
        "gamma": proj.scale,
        "at_grid_edge": proj.at_grid_edge,
        "constraint_relaxed": proj.constraint_relaxed,
        **cosine_gap(test, score_fn),
        "nevir_rank": nevir_rank(test, score_fn),
        "sim_unrelated": unrelated_similarity(test, score_fn),
        "_sweep": proj.sweep_table(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ablate the projection's own settings")
    ap.add_argument("--probe", default="data/probe/mock_probe.jsonl")
    ap.add_argument("--models", nargs="+", default=["fake"])
    ap.add_argument("--out", default="results/projection_ablation.csv")
    ap.add_argument("--show-sweeps", action="store_true", help="print the γ sweep per config")
    ap.add_argument(
        "--unrel-threshold", type=float, default=0.5,
        help="max allowed unrel for the constrained rows (default 0.5 — see "
             "NegationProjection docstring for why gap-only selection collapses)",
    )
    ap.add_argument(
        "--skip-unconstrained", action="store_true",
        help="only run constrain_unrel=True rows (halves the table; drops the "
             "side-by-side comparison with plain argmax(gap) selection)",
    )
    args = ap.parse_args()

    items = load_probe(args.probe)
    train, test = split_items(items, "train"), split_items(items, "test")
    if not train or not test:
        print(f"{args.probe} has {len(train)} train / {len(test)} test items — "
              "the projection needs both. Run build_probe finalize first.")
        return 1

    constrain_options = (True,) if args.skip_unconstrained else (False, True)

    rows = []
    for model_key in args.models:
        embedder = get_embedder(model_key)

        base = Baseline()
        base_fn = lambda a, b: base.score(a, b, embedder)
        rows.append({
            "model": model_key, "config": "baseline", "direction": "", "center": "",
            "select": "", "constrain_unrel": "", "unrel_threshold": "",
            "gamma": "", "at_grid_edge": "", "constraint_relaxed": "",
            "n_train": len(train), "n_test": len(test),
            **cosine_gap(test, base_fn), "nevir_rank": nevir_rank(test, base_fn),
            "sim_unrelated": unrelated_similarity(test, base_fn),
        })

        for direction, center, select, constrain_unrel in itertools.product(
            DIRECTION_METHODS, (True, False), ("cv", "train"), constrain_options
        ):
            try:
                result = evaluate_config(
                    train, test, embedder, direction, center, select,
                    constrain_unrel, args.unrel_threshold,
                )
            except ImportError as exc:
                print(f"[skip] {model_key} {direction}: {exc}")
                continue
            sweep = result.pop("_sweep")
            result.update({"model": model_key, "n_train": len(train), "n_test": len(test)})
            rows.append(result)
            if args.show_sweeps:
                print(f"\n{model_key} / {result['config']}\n{sweep}")

    header = (f"\n{'model':<16} {'config':<34} {'γ':>5} {'para':>7} {'neg':>7} "
              f"{'gap':>8} {'pair':>6} {'unrel':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        gamma = f"{r['gamma']:g}" if r["gamma"] != "" else "-"
        flag = " *" if r.get("at_grid_edge") else ""
        flag += " R" if r.get("constraint_relaxed") else ""
        print(f"{r['model']:<16} {r['config']:<34} {gamma:>5} "
              f"{r['sim_paraphrase']:>7.3f} {r['sim_negation']:>7.3f} "
              f"{r['cosine_gap']:>+8.4f} {r['nevir_rank']:>6.2f} "
              f"{r['sim_unrelated']:>7.3f}{flag}")
    print("\nunrel = mean |cos| between targets of different items. It should stay")
    print("low. A big gap next to a big unrel is the space collapsing onto one axis,")
    print("not a working fix — see unrelated_similarity().")
    if any(r.get("at_grid_edge") for r in rows):
        print("\n* γ hit the top of the sweep grid — the grid picked it, not the data. "
              "Widen DEFAULT_SCALE_GRID and check what STS does there before reporting.")
    if any(r.get("constraint_relaxed") for r in rows):
        print("\nR = no γ in the grid kept unrel under --unrel-threshold, so the "
              "constrained row fell back to the lowest-unrel γ instead of "
              "silently returning a collapsed one. Consider lowering "
              "DIRECTION_METHODS' amplification range or raising the threshold.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
