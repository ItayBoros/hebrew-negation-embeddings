"""
End-to-end evaluation runner.  ===  PERSON B  ===

Loads the probe, and for each (embedder x intervention):
  - fit the intervention on the TRAIN split
  - measure cosine_gap + nevir_rank (+ sts_corr when wired) on the TEST split
  - write one row to results/results.csv

Run the plumbing offline, no downloads:
    python -m src.harness.run_eval --models fake --probe data/probe/mock_probe.jsonl

Real run later:
    python -m src.harness.run_eval --models multilingual-e5 labse
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from ..schema import load_probe, split_items, ProbeItem
from .models import get_embedder
from .metrics import cosine_gap, nevir_rank, sts_corr

from ..interventions.baseline import Baseline
from ..interventions.projection import NegationProjection
from ..interventions.nli_rerank import NLIReranking


def build_interventions(names: List[str]):
    table = {
        "baseline": Baseline,
        "projection": NegationProjection,
        "nli_rerank": NLIReranking,
    }
    return [table[n]() for n in names]


def evaluate(probe_path: str, model_keys: List[str], intervention_names: List[str],
             out_csv: str = "results/results.csv") -> None:
    items = load_probe(probe_path)
    train, test = split_items(items, "train"), split_items(items, "test")
    if not test:
        test = items  # mock file may be tiny; fall back so plumbing still runs

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mkey in model_keys:
        embedder = get_embedder(mkey)
        for interv in build_interventions(intervention_names):
            try:
                interv.fit(train, embedder)
                score_fn = lambda a, b, _i=interv: _i.score(a, b, embedder)
                gap = cosine_gap(test, score_fn)
                nevir = nevir_rank(test, score_fn)
                sts = sts_corr([], score_fn)  # TODO(B): pass a real Hebrew STS set
                rows.append({
                    "model": mkey, "intervention": interv.name,
                    "n_test": len(test), **gap, "nevir_rank": nevir,
                    "sts_pearson": sts["pearson"], "sts_spearman": sts["spearman"],
                })
                print(f"[ok] {mkey:16s} {interv.name:12s} "
                      f"gap={gap['cosine_gap']:+.3f} nevir={nevir:.2f}")
            except NotImplementedError as e:
                print(f"[skip] {mkey} {interv.name}: {e}")

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="data/probe/mock_probe.jsonl")
    ap.add_argument("--models", nargs="+", default=["fake"])
    ap.add_argument("--interventions", nargs="+",
                    default=["baseline", "projection", "nli_rerank"])
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()
    evaluate(args.probe, args.models, args.interventions, args.out)


if __name__ == "__main__":
    main()
