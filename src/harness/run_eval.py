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

Comparing two NLI checkpoints under nli_rerank
-----------------------------------------------
`nli_rerank` defaults to the released checkpoint (`NLIReranking.DEFAULT_MODEL_NAME`).
`--nli-model`/`--nli-subfolder`/`--nli-encoding` point it at a different one instead —
typically the checkpoint `train_nli.py` produced:

    python -m src.harness.run_eval --models multilingual-e5 labse \\
        --interventions nli_rerank \\
        --nli-model /content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean \\
        --nli-subfolder "" --nli-encoding pair \\
        --out results/results_nli_rerank.csv

Run once with the default checkpoint and once with `--nli-model` overridden, both
against the same `--out`: results are appended and replaced by configuration (model,
intervention, nli_checkpoint, nli_encoding, nli_lam), not overwritten wholesale, so
the two checkpoints' rows sit side by side in one comparison table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence

from ..schema import load_probe, split_items, ProbeItem
from .models import get_embedder
from .metrics import cosine_gap, nevir_rank, sts_corr

from ..interventions.baseline import Baseline
from ..interventions.projection import NegationProjection
from ..interventions.nli_rerank import JOINED, PAIR, NLIReranking

#: Column order for the results table. The nli_* columns are blank for
#: baseline/projection rows and populated only for nli_rerank, so which
#: checkpoint produced a given row is always on the record, not implied by
#: run order.
CSV_FIELDS = [
    "model", "intervention", "n_test", "sim_paraphrase", "sim_negation",
    "cosine_gap", "nevir_rank", "sts_pearson", "sts_spearman",
    "nli_checkpoint", "nli_subfolder", "nli_encoding", "nli_lam",
]

#: What makes a row a distinct configuration. A re-run with the same model +
#: intervention + NLI checkpoint replaces its row; a different --nli-model
#: (e.g. comparing the released checkpoint against ours) appends instead of
#: clobbering the other one's numbers.
CONFIG_KEY = ["model", "intervention", "nli_checkpoint", "nli_encoding", "nli_lam"]


def build_interventions(names: List[str], nli_kwargs: Optional[dict] = None):
    table = {
        "baseline": lambda: Baseline(),
        "projection": lambda: NegationProjection(),
        "nli_rerank": lambda: NLIReranking(**(nli_kwargs or {})),
    }
    return [table[n]() for n in names]


def _signature(row: dict, keys: Sequence[str]) -> tuple:
    return tuple(str(row.get(k, "")) for k in keys)


def append_or_replace(new_rows: List[dict], path: str | Path,
                       fieldnames: Sequence[str] = CSV_FIELDS,
                       config_key: Sequence[str] = CONFIG_KEY) -> int:
    """Add `new_rows` to the table at `path`, replacing any existing row with
    a matching configuration rather than overwriting the whole file.

    Read-modify-write, same idiom as `train_nli.append_result`: without it, a
    second run with a different `--nli-model` (exactly the comparison this
    CLI exists to support) would silently erase the first run's numbers
    instead of sitting beside them.
    """
    path = Path(path)
    existing: List[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))

    new_sigs = {_signature(r, config_key) for r in new_rows}
    kept = [r for r in existing if _signature(r, config_key) not in new_sigs]
    all_rows = kept + new_rows

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fieldnames} for r in all_rows])
    return len(all_rows)


def evaluate(probe_path: str, model_keys: List[str], intervention_names: List[str],
             out_csv: str = "results/results.csv",
             nli_kwargs: Optional[dict] = None) -> None:
    items = load_probe(probe_path)
    train, test = split_items(items, "train"), split_items(items, "test")
    if not test:
        test = items  # mock file may be tiny; fall back so plumbing still runs

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mkey in model_keys:
        embedder = get_embedder(mkey)
        for interv in build_interventions(intervention_names, nli_kwargs):
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
                    "nli_checkpoint": getattr(interv, "model_name", ""),
                    "nli_subfolder": getattr(interv, "model_subfolder", "") or "",
                    "nli_encoding": getattr(interv, "pair_encoding", ""),
                    "nli_lam": getattr(interv, "lam", ""),
                })
                print(f"[ok] {mkey:16s} {interv.name:12s} "
                      f"gap={gap['cosine_gap']:+.3f} nevir={nevir:.2f}")
            except NotImplementedError as e:
                print(f"[skip] {mkey} {interv.name}: {e}")

    if rows:
        n_total = append_or_replace(rows, out_csv)
        print(f"\nwrote {len(rows)} new/updated rows -> {out_csv} ({n_total} rows total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="data/probe/mock_probe.jsonl")
    ap.add_argument("--models", nargs="+", default=["fake"])
    ap.add_argument("--interventions", nargs="+",
                    default=["baseline", "projection", "nli_rerank"])
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--nli-model", default=NLIReranking.DEFAULT_MODEL_NAME,
                    help="HF id or local checkpoint dir for the nli_rerank intervention")
    ap.add_argument("--nli-subfolder", default=NLIReranking.DEFAULT_MODEL_SUBFOLDER,
                    help="subfolder inside the HF repo; pass '' for a local checkpoint")
    ap.add_argument("--nli-encoding", default=None, choices=[JOINED, PAIR],
                    help=f"default: {JOINED} for the released checkpoint, {PAIR} otherwise")
    ap.add_argument("--nli-lam", type=float, default=1.0,
                    help="0 = pure cosine, 1 = pure NLI (default), see nli_rerank.py")
    args = ap.parse_args()

    encoding = args.nli_encoding or (JOINED if args.nli_model == NLIReranking.DEFAULT_MODEL_NAME
                                      else PAIR)
    nli_kwargs = dict(
        lam=args.nli_lam, model_name=args.nli_model,
        model_subfolder=args.nli_subfolder or None, pair_encoding=encoding,
    )
    evaluate(args.probe, args.models, args.interventions, args.out, nli_kwargs=nli_kwargs)


if __name__ == "__main__":
    main()
