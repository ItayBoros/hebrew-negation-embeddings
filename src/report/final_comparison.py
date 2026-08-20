"""
Unified results table: baseline vs. projection vs. nli_rerank, one row each,
across all four models.  ===  JOINT — not owned by one side ===

The two interventions were developed, selected, and locked independently, and
each left its numbers in its own file:

    results/nli_lambda_test.csv    baseline (lambda=0) + the selected lambda's
                                    locked test numbers, WITH real Hebrew-STS
                                    (Person B's STS-B translation) -- this file
                                    is the source of truth for the baseline row
                                    too, since it is the only one with STS
                                    wired in.
    results/projection_ablation.csv  every (direction, centring, selection,
                                    constrain_unrel) combination Person A's
                                    projection was ablated over. No STS column
                                    -- sim_unrelated is the trade-off guard
                                    used in its place (see projection.py's
                                    module docstring for why gap-only
                                    selection is unsafe without it).

This script does not compute anything new. It picks, per model, the single
projection row that would actually be reported: the module's own default
configuration (direction=mean_diff, center=True, select=cv), constrained to
sim_unrelated <= 0.5, applied identically to all four models.

IMPORTANT -- this used to argmax over the 8 (direction x centering x
cv/train) configurations, first by cosine_gap and later by pairwise
accuracy. Both versions were wrong the same way: picking, per model,
whichever of 8 configurations scored best *on the test split itself* is
test-set selection across hypotheses, not a locked evaluation of one
method -- the same category of problem as the unconstrained-gamma collapse
this project's whole guard-metric argument is about, just one level up (see
report/main.tex Section 2.4 and the Discussion). Fixed by hard-coding the
selection to one configuration -- the constructor's own defaults -- decided
without looking at any test number, applied the same way to every model.
The full 8-way ablation is still available in results/projection_ablation.csv
as a sensitivity check; it is not used to pick the headline row.

    python -m src.report.final_comparison
    python -m src.report.final_comparison --out results/final_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

MODELS = ["multilingual-e5", "labse", "alephbert-sentence", "sambert"]

FIELDS = [
    "model", "intervention", "config",
    "sim_paraphrase", "sim_negation", "cosine_gap", "pairwise_accuracy",
    "sts_pearson", "sts_spearman", "flag",
]


def _f(x: str) -> float:
    return float(x)


def load_nli_rows(path: Path) -> Dict[str, Dict[str, dict]]:
    """model -> {'baseline': row, 'selected': row}, straight from the locked
    test-set file. This is also where the baseline row's STS numbers come
    from, since results_baseline.csv predates STS being wired in."""
    out: Dict[str, Dict[str, dict]] = {m: {} for m in MODELS}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] in out:
                out[row["model"]][row["configuration"]] = row
    return out


def best_constrained_projection(path: Path, unrel_threshold: str = "0.5") -> Dict[str, dict]:
    """model -> the projection_ablation.csv row for the single fixed
    configuration 'mean_diff/cv/constrain<=<threshold>' (direction=mean_diff,
    center=True, select=cv -- NegationProjection's own constructor defaults).

    Deliberately NOT an argmax over the 8 ablated configurations by any
    metric (gap, accuracy, ...): every one of those 8 rows' numbers is
    computed on the test split (see projection_report.py), so picking
    per-model among them by a test-split number is test-set selection
    across 8 hypotheses, not a locked single-method evaluation. This
    function instead always returns the one configuration decided in
    advance, independent of what any of the 8 rows' test numbers say. If
    that fixed row's own constraint could not be satisfied for a model
    (multilingual-e5, at threshold 0.5), it is still the row returned --
    constraint_relaxed on the row itself carries that flag, unchanged from
    what projection.py already decided during fit.
    """
    target_config = f"mean_diff/cv/constrain<={unrel_threshold}"
    best: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] not in MODELS or row["config"] != target_config:
                continue
            best[row["model"]] = row
    return best


def build_rows(nli_path: Path, proj_path: Path) -> List[dict]:
    nli = load_nli_rows(nli_path)
    proj = best_constrained_projection(proj_path)

    rows: List[dict] = []
    for model in MODELS:
        base = nli[model]["baseline"]
        rows.append({
            "model": model, "intervention": "baseline", "config": "",
            "sim_paraphrase": _f(base["mean_paraphrase_score"]),
            "sim_negation": _f(base["mean_negation_score"]),
            "cosine_gap": _f(base["mean_gap"]),
            "pairwise_accuracy": _f(base["pairwise_accuracy"]),
            "sts_pearson": _f(base["sts_pearson"]),
            "sts_spearman": _f(base["sts_spearman"]),
            "flag": "",
        })

        p = proj.get(model)
        if p is None:
            rows.append({
                "model": model, "intervention": "projection", "config": "MISSING",
                "sim_paraphrase": "", "sim_negation": "", "cosine_gap": "",
                "pairwise_accuracy": "", "sts_pearson": "", "sts_spearman": "",
                "flag": "no constrain<=0.5 row found",
            })
        else:
            relaxed = p.get("constraint_relaxed") == "True"
            rows.append({
                "model": model, "intervention": "projection",
                "config": f"{p['config']} (γ={p['gamma']})",
                "sim_paraphrase": _f(p["sim_paraphrase"]),
                "sim_negation": _f(p["sim_negation"]),
                "cosine_gap": _f(p["cosine_gap"]),
                "pairwise_accuracy": _f(p["nevir_rank"]),
                "sts_pearson": "",  # projection has no STS guard wired; sim_unrelated is in projection_ablation.csv
                "sts_spearman": "",
                "flag": "constraint_relaxed: no γ kept unrel<=0.5" if relaxed else "",
            })

        sel = nli[model]["selected"]
        rows.append({
            "model": model, "intervention": "nli_rerank",
            "config": f"lambda={sel['lambda']}",
            "sim_paraphrase": _f(sel["mean_paraphrase_score"]),
            "sim_negation": _f(sel["mean_negation_score"]),
            "cosine_gap": _f(sel["mean_gap"]),
            "pairwise_accuracy": _f(sel["pairwise_accuracy"]),
            "sts_pearson": _f(sel["sts_pearson"]),
            "sts_spearman": _f(sel["sts_spearman"]),
            "flag": "",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nli", default="results/nli_lambda_test.csv")
    ap.add_argument("--projection", default="results/projection_ablation.csv")
    ap.add_argument("--unrel-threshold", default="0.5")
    ap.add_argument("--out", default="results/final_comparison.csv")
    args = ap.parse_args()

    rows = build_rows(Path(args.nli), Path(args.projection))

    header = (f"{'model':<20} {'intervention':<12} {'config':<28} "
              f"{'para':>7} {'neg':>7} {'gap':>8} {'pair_acc':>9} "
              f"{'sts_p':>7} {'sts_s':>7}  flag")
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(v: object) -> str:
            return f"{v:.3f}" if isinstance(v, float) else "-"
        print(f"{r['model']:<20} {r['intervention']:<12} {r['config']:<28} "
              f"{fmt(r['sim_paraphrase']):>7} {fmt(r['sim_negation']):>7} "
              f"{fmt(r['cosine_gap']):>8} {fmt(r['pairwise_accuracy']):>9} "
              f"{fmt(r['sts_pearson']):>7} {fmt(r['sts_spearman']):>7}  {r['flag']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
