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
  γ           swept inside each fit

The `select` column is worth reading carefully. `select=train` fits the direction
on the train split and then picks the γ that maximises the gap on those same
pairs, so its own reported number is inflated; comparing the two on the *test*
split shows how much of the naive gain was real.

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
from pathlib import Path
from typing import List

from ..harness.metrics import cosine_gap, nevir_rank
from ..harness.models import get_embedder
from ..schema import ProbeItem, load_probe, split_items
from .baseline import Baseline
from .projection import DIRECTION_METHODS, NegationProjection

FIELDS = [
    "model", "config", "direction", "center", "select", "gamma", "at_grid_edge",
    "n_train", "n_test", "sim_paraphrase", "sim_negation", "cosine_gap", "nevir_rank",
]


def evaluate_config(
    train: List[ProbeItem],
    test: List[ProbeItem],
    embedder,
    direction: str,
    center: bool,
    select: str,
) -> dict:
    proj = NegationProjection(direction_method=direction, center=center, select=select)
    proj.fit(train, embedder)
    score_fn = lambda a, b: proj.score(a, b, embedder)
    return {
        "config": f"{direction}/{select}{'' if center else '/nocenter'}",
        "direction": direction,
        "center": center,
        "select": proj.selection,
        "gamma": proj.scale,
        "at_grid_edge": proj.at_grid_edge,
        **cosine_gap(test, score_fn),
        "nevir_rank": nevir_rank(test, score_fn),
        "_sweep": proj.sweep_table(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="ablate the projection's own settings")
    ap.add_argument("--probe", default="data/probe/mock_probe.jsonl")
    ap.add_argument("--models", nargs="+", default=["fake"])
    ap.add_argument("--out", default="results/projection_ablation.csv")
    ap.add_argument("--show-sweeps", action="store_true", help="print the γ sweep per config")
    args = ap.parse_args()

    items = load_probe(args.probe)
    train, test = split_items(items, "train"), split_items(items, "test")
    if not train or not test:
        print(f"{args.probe} has {len(train)} train / {len(test)} test items — "
              "the projection needs both. Run build_probe finalize first.")
        return 1

    rows = []
    for model_key in args.models:
        embedder = get_embedder(model_key)

        base = Baseline()
        base_fn = lambda a, b: base.score(a, b, embedder)
        rows.append({
            "model": model_key, "config": "baseline", "direction": "", "center": "",
            "select": "", "gamma": "", "at_grid_edge": "",
            "n_train": len(train), "n_test": len(test),
            **cosine_gap(test, base_fn), "nevir_rank": nevir_rank(test, base_fn),
        })

        for direction, center, select in itertools.product(
            DIRECTION_METHODS, (True, False), ("cv", "train")
        ):
            try:
                result = evaluate_config(train, test, embedder, direction, center, select)
            except ImportError as exc:
                print(f"[skip] {model_key} {direction}: {exc}")
                continue
            sweep = result.pop("_sweep")
            result.update({"model": model_key, "n_train": len(train), "n_test": len(test)})
            rows.append(result)
            if args.show_sweeps:
                print(f"\n{model_key} / {result['config']}\n{sweep}")

    header = f"\n{'model':<16} {'config':<28} {'γ':>5} {'para':>7} {'neg':>7} {'gap':>8} {'nevir':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        gamma = f"{r['gamma']:g}" if r["gamma"] != "" else "-"
        flag = " *" if r.get("at_grid_edge") else ""
        print(f"{r['model']:<16} {r['config']:<28} {gamma:>5} "
              f"{r['sim_paraphrase']:>7.3f} {r['sim_negation']:>7.3f} "
              f"{r['cosine_gap']:>+8.4f} {r['nevir_rank']:>6.2f}{flag}")
    if any(r.get("at_grid_edge") for r in rows):
        print("\n* γ hit the top of the sweep grid — the grid picked it, not the data. "
              "Widen DEFAULT_SCALE_GRID and check what STS does there before reporting.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
