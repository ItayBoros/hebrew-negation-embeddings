"""
Per-category error analysis: where do the interventions still fail?  === JOINT ===

`projection_ablation.csv` and `nli_lambda_test.csv` report one aggregate
number per model. That hides whether a fix works everywhere a little or
works great on most items and not at all on a specific negation type — the
question M3 in PLAN.md actually asks (morphological negation, quantifiers,
double negation...).

The probe already carries the answer key: `build_probe.py._stratum` bucketed
every item into a negation category *before* the train/test split even
happened (so the split would stay balanced), and it's saved as each item's
`note` field:

    particle      לא          (plain clausal negation)
    existential   אין / אינ-  (existential/copular negation)
    quantifier    a phrase marker (אף אחד, אף פעם, ...) -- these win over a
                  bare לא if both are present, since the phrase is the more
                  specific fact about the item
    privative     ללא / בלי / בלתי / בלעדי / אי-           (prefixal/prepositional)
    other/question/neg-raising/... whatever a human annotator wrote in note
                  during review (src/data/build_probe.py review step)

This script re-fits projection per model with the exact recipe that won in
projection_ablation.csv (same direction/center/select, constrain_unrel=True
at the same threshold -- so it reproduces the same gamma deterministically,
never touches test to pick it) and, for every TEST item, records whether the
paraphrase ended up closer than the negation. Grouped by `note`.

nli_rerank needs Person B's fine-tuned checkpoint, which lives on their Drive,
not in this repo -- pass --nli-model/--nli-subfolder/--nli-encoding to include
it (e.g. from the Colab notebook that already has Drive mounted); without
them this only covers baseline vs. projection, and says so.

    python -m src.report.error_analysis
    python -m src.report.error_analysis \\
        --nli-model /content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean \\
        --nli-subfolder "" --nli-encoding pair --nli-lambda 0.05:multilingual-e5 0.35:labse ...
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..harness.models import get_embedder
from ..interventions.baseline import Baseline
from ..interventions.projection import NegationProjection
from ..schema import ProbeItem, load_probe, split_items

MODELS = ["multilingual-e5", "labse", "alephbert-sentence", "sambert"]


def winning_projection_config(proj_path: Path, model: str, threshold: str = "0.5") -> Optional[dict]:
    """The projection_ablation.csv row for the single fixed configuration
    'mean_diff/cv/constrain<=<threshold>' -- the same one
    final_comparison.py now uses (direction=mean_diff, center=True,
    select=cv: NegationProjection's own constructor defaults).

    This used to argmax cosine_gap across all 8 ablated configurations per
    model. That is test-set selection across hypotheses (every row's
    cosine_gap in projection_ablation.csv is computed on the test split),
    so the resulting per-category numbers in error_analysis_summary.csv
    were inflated the same way Table 6 was before it got fixed -- see
    final_comparison.py's module docstring. Returns the fields needed to
    refit the fixed configuration, or None if that row is missing."""
    target_config = f"mean_diff/cv/constrain<={threshold}"
    with proj_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == model and row["config"] == target_config:
                return row
    return None


def item_pass(score_fn, item: ProbeItem) -> bool:
    """True iff the paraphrase scored closer than the negation -- the same
    per-item sign `pairwise_accuracy` (metrics.py) averages over."""
    return score_fn(item.target, item.paraphrase) > score_fn(item.target, item.negation)


def run_baseline_and_projection(
    test_items: List[ProbeItem],
    train_items: List[ProbeItem],
    proj_path: Path,
    threshold: str,
) -> List[dict]:
    rows: List[dict] = []
    for model in MODELS:
        cfg = winning_projection_config(proj_path, model, threshold)
        if cfg is None:
            print(f"[skip] {model}: no constrain<={threshold} row in {proj_path}")
            continue

        embedder = get_embedder(model)
        base = Baseline()
        # cfg["select"] is proj.selection as *reported* after fit (e.g. "cv5",
        # since NegationProjection.selection = f"cv{n_folds}") -- the
        # constructor's `select` kwarg only accepts the literal "cv"/"train".
        select_kwarg = "cv" if cfg["select"].startswith("cv") else "train"
        proj = NegationProjection(
            direction_method=cfg["direction"],
            center=cfg["center"] == "True",
            select=select_kwarg,
            constrain_unrel=True,
            unrel_threshold=float(threshold),
        )
        proj.fit(train_items, embedder)
        relaxed = proj.constraint_relaxed
        print(f"{model}: refit projection -> gamma={proj.scale:g} "
              f"(ablation said {cfg['gamma']}){'  [constraint_relaxed]' if relaxed else ''}")

        base_fn = lambda a, b: base.score(a, b, embedder)
        proj_fn = lambda a, b: proj.score(a, b, embedder)

        for item in test_items:
            rows.append({
                "model": model, "category": item.note or "none", "item_id": item.id,
                "intervention": "baseline", "pass": item_pass(base_fn, item),
            })
            rows.append({
                "model": model, "category": item.note or "none", "item_id": item.id,
                "intervention": "projection", "pass": item_pass(proj_fn, item),
                "constraint_relaxed": relaxed,
            })
    return rows


def run_nli(
    test_items: List[ProbeItem],
    nli_model: str,
    nli_subfolder: str,
    nli_encoding: str,
    lambdas: Dict[str, float],
) -> List[dict]:
    from ..interventions.nli_rerank import NLIReranking

    rows: List[dict] = []
    for model, lam in lambdas.items():
        embedder = get_embedder(model)
        nli = NLIReranking(
            model_name=nli_model, model_subfolder=nli_subfolder or None,
            pair_encoding=nli_encoding, lam=lam,
        )
        score_fn = lambda a, b: nli.score(a, b, embedder)
        for item in test_items:
            rows.append({
                "model": model, "category": item.note or "none", "item_id": item.id,
                "intervention": "nli_rerank", "pass": item_pass(score_fn, item),
            })
    return rows


def aggregate(rows: List[dict]) -> List[dict]:
    """(model, intervention, category) -> n, n_pass, pass_rate."""
    buckets: Dict[Tuple[str, str, str], List[bool]] = defaultdict(list)
    for r in rows:
        buckets[(r["model"], r["intervention"], r["category"])].append(r["pass"])
    out = []
    for (model, intervention, category), passes in sorted(buckets.items()):
        n = len(passes)
        n_pass = sum(passes)
        out.append({
            "model": model, "intervention": intervention, "category": category,
            "n": n, "n_pass": n_pass, "pass_rate": n_pass / n,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="data/probe/probe.jsonl")
    ap.add_argument("--projection", default="results/projection_ablation.csv")
    ap.add_argument("--unrel-threshold", default="0.5")
    ap.add_argument("--out-items", default="results/error_analysis_items.csv")
    ap.add_argument("--out-summary", default="results/error_analysis_summary.csv")
    ap.add_argument("--nli-model", default=None,
                     help="path/HF id of Person B's checkpoint; omit to skip nli_rerank")
    ap.add_argument("--nli-subfolder", default="")
    ap.add_argument("--nli-encoding", default="pair")
    ap.add_argument("--nli-lambda", nargs="*", default=[],
                     help="model=lambda pairs, e.g. multilingual-e5=0.05 labse=0.35 ...")
    args = ap.parse_args()

    items = load_probe(args.probe)
    train, test = split_items(items, "train"), split_items(items, "test")

    print(f"test split: {len(test)} items across categories "
          f"{sorted(set(it.note for it in test))}\n")

    rows = run_baseline_and_projection(test, train, Path(args.projection), args.unrel_threshold)

    if args.nli_model:
        lambdas = dict(kv.split("=") for kv in args.nli_lambda)
        lambdas = {m: float(v) for m, v in lambdas.items()}
        missing = set(MODELS) - set(lambdas)
        if missing:
            print(f"[skip nli_rerank] no --nli-lambda given for {sorted(missing)}")
        rows += run_nli(test, args.nli_model, args.nli_subfolder, args.nli_encoding,
                         {m: l for m, l in lambdas.items() if m in MODELS})
    else:
        print("[skip] nli_rerank: no --nli-model given (checkpoint lives on Person B's "
              "Drive) -- this run only covers baseline vs. projection")

    summary = aggregate(rows)

    header = f"{'model':<20} {'intervention':<12} {'category':<12} {'n':>4} {'pass':>5} {'rate':>6}"
    print("\n" + header)
    print("-" * len(header))
    for r in summary:
        print(f"{r['model']:<20} {r['intervention']:<12} {r['category']:<12} "
              f"{r['n']:>4} {r['n_pass']:>5} {r['pass_rate']:>6.2f}")

    for path, data, fields in [
        (args.out_items, rows, ["model", "intervention", "category", "item_id", "pass", "constraint_relaxed"]),
        (args.out_summary, summary, ["model", "intervention", "category", "n", "n_pass", "pass_rate"]),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
        print(f"\nwrote {len(data)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
