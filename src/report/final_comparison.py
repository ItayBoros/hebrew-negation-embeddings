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
projection row that would actually be reported: among the 8 (direction x
centering x cv/train) configurations satisfying the sim_unrelated <= 0.5
constraint, the one with the highest pairwise accuracy -- "best under the
constraint", not "best full stop" -- and lines it up next to the other two
so the report's results section has one table instead of three files to
cross-reference by hand.

Note: within a single configuration, gamma itself is still chosen by
argmax-cosine-gap under the constraint (that selection already happened
upstream, in projection.py, before this script ever sees the row). This
script's own choice is a second, separate one: which of the 8 already-fit
configurations to headline for a given model, and that choice is made by
accuracy, to match how the rest of the report (baseline, nli_rerank)
reports headline numbers. See report/main.tex Section 2.4 for both rules
stated together.

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
    """model -> the projection_ablation.csv row with config ending
    '/constrain<=<threshold>' that has the highest pairwise accuracy
    (nevir_rank) for that model.

    Mirrors what `describe()` / the report would actually cite: among the 8
    (direction x centering x cv/train) configurations, each already
    gamma-selected upstream by argmax-cosine-gap under the sim_unrelated
    constraint, this picks the one with the highest pairwise accuracy --
    not the unconstrained (collapsed) one, and not the highest-gap
    configuration either, since gap and accuracy do not always agree across
    configurations (see report/main.tex Section 2.4). If every constrained
    row for a model has constraint_relaxed=True (multilingual-e5, at
    threshold 0.5), the picked row is still the best accuracy among them,
    but it is flagged.
    """
    suffix = f"/constrain<={unrel_threshold}"
    best: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] not in MODELS or not row["config"].endswith(suffix):
                continue
            acc = _f(row["nevir_rank"])
            if row["model"] not in best or acc > _f(best[row["model"]]["nevir_rank"]):
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
