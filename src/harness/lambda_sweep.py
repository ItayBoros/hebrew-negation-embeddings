"""
Directional NLI re-ranking: pick lambda on dev, then evaluate once on test.
===  PERSON B  ===

`nli_rerank` blends a frozen embedder's cosine with an ordered NLI judgement:

    nli_score(a, b)      = P(entailment | a, b) - P(contradiction | a, b)
    combined(a, b, lam)  = (1 - lam) * cosine(a, b) + lam * nli_score(a, b)

`lam` used to default to 1.0 — pure NLI, chosen by nobody, and a setting under
which the embedding model does not affect the score at all. This module replaces
that guess with a selection procedure, run independently for each of the four
frozen embedders.

Two stages, and the separation between them is the point
--------------------------------------------------------
**dev** sweeps lambda over ``0.00, 0.05, ..., 1.00`` on the development splits
only — `data/probe/splits/train.jsonl` (152 items) and `hebrew_stsb_dev.csv`
(1,500 pairs) — writes the full sweep to `results/nli_lambda_dev.csv`, and locks
one lambda per model into `results/nli_selected_lambdas.json`.

**test** reads that locked file and evaluates two configurations per model on
`data/probe/splits/test.jsonl` (151 items) and `hebrew_stsb_test.csv` (1,379
pairs): the `lambda=0` baseline and that model's selected lambda. It cannot
sweep, and it cannot change a selection — it has no code path that writes the
json. `_assert_development_only` refuses outright if a test file is handed to
the dev stage, because "we only looked at dev" is a claim about a run that
already happened and nobody can check it afterwards.

Why lambda is selected the way it is
------------------------------------
A blend that separates a sentence from its negation but wrecks ordinary
similarity is not a fix, so STS is a *constraint*, not an objective: a lambda is
eligible only while its Spearman on STS-dev stays within 0.02 (absolute) of the
same model's cosine-only Spearman. Among the eligible lambdas the winner is the
one with the highest `pairwise_accuracy`, ties broken by `mean_gap` and then by
the smallest lambda — the smallest, because a lower lambda leans less on the NLI
model and is the more conservative of two settings that measure the same. See
`LAMBDA_SELECTION.md` for the full argument. `lambda=0` is always eligible (its
drop is 0 by construction), so a selection always exists.

Direction is never averaged away
--------------------------------
NLI is asymmetric and the experiment depends on it: `target` is the premise and
`paraphrase`/`negation` the hypothesis; on STS, `sentence1` is the premise and
`sentence2` the hypothesis. No pair is reversed, and no two directions are
averaged.

One forward pass per pair, not per lambda
-----------------------------------------
Both terms of the blend are independent of lambda, so each is computed once per
pair and the 21 grid points are arithmetic on cached arrays. The NLI term does
not depend on the embedding model either, so it is computed once and reused for
all four — 1,804 ordered-pair classifications for the whole dev stage instead
of 151,536. Batching reduces the number of actual model forward calls further.

    python -m src.harness.lambda_sweep --stage dev  --nli-model <ckpt> --nli-subfolder ""
    python -m src.harness.lambda_sweep --stage test --nli-model <ckpt> --nli-subfolder ""
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import ProbeItem, load_probe
from ..interventions.nli_rerank import JOINED, PAIR, NLIReranking
from . import sts as sts_data
from .metrics import sts_corr_from_scores
from .models import MODELS, get_embedder
from .run_eval import append_or_replace

#: 0.00, 0.05, ..., 1.00. Rounded because 0.35000000000000003 as a csv cell and
#: as a json key is the kind of thing that breaks a join two weeks later.
LAMBDA_GRID: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))

#: The trade-off budget: how much absolute Spearman on STS-dev a lambda may cost
#: relative to that model's own cosine-only score before it stops being eligible.
MAX_STS_SPEARMAN_DROP = 0.02

#: Comparisons at the edge of the budget, and ties in the ranking, are decided on
#: floats that went through 1,500-element means. Anything closer than this is a
#: tie, not a difference.
TOLERANCE = 1e-12

#: All four frozen embedders. The procedure runs independently per model — the
#: selected lambda is a property of the (embedder, NLI checkpoint) pair, not a
#: single global constant.
DEFAULT_MODELS: Tuple[str, ...] = ("multilingual-e5", "labse",
                                    "alephbert-sentence", "sambert")

#: The checkpoint fine-tuned on clean HebNLI (`02_train_nli.ipynb`), the one this
#: procedure is meant to run with. It lives on Drive, so this default only
#: resolves inside Colab; `--nli-model` overrides it anywhere else.
CLEAN_CHECKPOINT = "/content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean"

PROBE_TRAIN = "data/probe/splits/train.jsonl"
PROBE_TEST = "data/probe/splits/test.jsonl"

#: Never read during lambda selection. Enforced, not just documented.
HELD_OUT_FILES = (PROBE_TEST, sts_data.TEST_PATH)

DEV_CSV = "results/nli_lambda_dev.csv"
SELECTED_JSON = "results/nli_selected_lambdas.json"
TEST_CSV = "results/nli_lambda_test.csv"

DEV_FIELDS = [
    "model", "lambda", "probe_n", "pairwise_accuracy", "mean_paraphrase_score",
    "mean_negation_score", "mean_gap", "sts_n", "sts_pearson", "sts_spearman",
    "sts_spearman_drop", "eligible", "selected",
    "nli_checkpoint", "nli_encoding", "directional",
]

TEST_FIELDS = [
    "model", "configuration", "lambda", "probe_split", "probe_n",
    "pairwise_accuracy", "mean_paraphrase_score", "mean_negation_score", "mean_gap",
    "sts_split", "sts_n", "sts_pearson", "sts_spearman",
    "nli_checkpoint", "nli_encoding", "directional",
]

#: A run emits every row a model has — all 21 lambdas, or both configurations —
#: so the unit that gets replaced is the model, not the individual row. Re-running
#: `multilingual-e5` therefore replaces exactly its own rows and leaves the other
#: three models' alone, and a stale row (a grid point that no longer exists, or a
#: `selected` flag that has moved) cannot survive underneath the new ones. A
#: different NLI checkpoint is a different configuration and sits beside it.
CONFIG_KEY = ["model", "nli_checkpoint", "nli_encoding"]

#: `configuration` values in the final table. A model whose selected lambda is 0
#: gets one row, not two identical ones.
BASELINE = "baseline"
SELECTED = "selected"
BASELINE_AND_SELECTED = "baseline+selected"


# --------------------------------------------------------------------------- #
# cached components
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairComponents:
    """The two lambda-independent halves of the blend, in pair order.

    `cosine` comes from one embedding model; `nli` from the NLI checkpoint and is
    the same array whichever embedder it is paired with.
    """

    cosine: np.ndarray
    nli: np.ndarray

    def __post_init__(self) -> None:
        if len(self.cosine) != len(self.nli):
            raise ValueError(
                f"cosine has {len(self.cosine)} entries and nli has {len(self.nli)} — "
                "these must be the same pairs in the same order"
            )

    def __len__(self) -> int:
        return len(self.cosine)

    def blend(self, lam: float) -> np.ndarray:
        """`(1 - lam) * cosine + lam * nli`, clipped exactly as `score` clips."""
        return np.clip((1.0 - lam) * self.cosine + lam * self.nli, -1.0, 1.0)


def cosine_components(pairs: Sequence[Tuple[str, str]], embedder) -> np.ndarray:
    """Cosine per ordered pair, encoding each distinct sentence exactly once.

    The probe reuses every `target` twice and STS pairs share nothing, but the
    dedup costs one dict and saves a third of the probe's forward passes.
    """
    texts = list(dict.fromkeys(text for pair in pairs for text in pair))
    vectors = np.asarray(embedder.encode(texts), dtype=np.float64)
    index = {text: i for i, text in enumerate(texts)}

    left = vectors[[index[a] for a, _ in pairs]]
    right = vectors[[index[b] for _, b in pairs]]
    # Same epsilon as interventions.baseline.cosine, so a lambda=0 row here and a
    # baseline row from run_eval are the same number and not merely close.
    denominator = (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)) + 1e-12
    return (left * right).sum(axis=1) / denominator


def probe_pairs(items: Sequence[ProbeItem]) -> Tuple[List[Tuple[str, str]],
                                                      List[Tuple[str, str]]]:
    """(target, paraphrase) and (target, negation), premise first. Never reversed."""
    return ([(it.target, it.paraphrase) for it in items],
            [(it.target, it.negation) for it in items])


# --------------------------------------------------------------------------- #
# metrics on cached components
# --------------------------------------------------------------------------- #

def probe_metrics(paraphrase: PairComponents, negation: PairComponents,
                  lam: float) -> dict:
    """`pairwise_accuracy` and the three score summaries at one lambda."""
    para = paraphrase.blend(lam)
    neg = negation.blend(lam)
    return {
        "probe_n": len(para),
        # Per item, not a comparison of the two means: a model can have a
        # positive mean gap while ranking a third of the items backwards.
        "pairwise_accuracy": float(np.mean(para > neg)),
        "mean_paraphrase_score": float(np.mean(para)),
        "mean_negation_score": float(np.mean(neg)),
        "mean_gap": float(np.mean(para - neg)),
    }


def sts_metrics(sts: PairComponents, gold: np.ndarray, lam: float) -> dict:
    """Pearson/Spearman of the blend against the inherited STS-B gold scores."""
    corr = sts_corr_from_scores(sts.blend(lam), gold)
    if corr["spearman"] is None:
        raise RuntimeError(
            "STS correlation came back empty — scipy is required for lambda "
            "selection, since the eligibility rule is defined on Spearman"
        )
    return {"sts_n": corr["n"], "sts_pearson": corr["pearson"],
            "sts_spearman": corr["spearman"]}


# --------------------------------------------------------------------------- #
# the sweep and the selection rule
# --------------------------------------------------------------------------- #

def sweep_lambdas(paraphrase: PairComponents, negation: PairComponents,
                  sts: PairComponents, gold: np.ndarray,
                  grid: Sequence[float] = LAMBDA_GRID) -> List[dict]:
    """One row per lambda, with `eligible` and `selected` already decided."""
    rows = [
        {"lambda": float(lam),
         **probe_metrics(paraphrase, negation, lam),
         **sts_metrics(sts, gold, lam)}
        for lam in grid
    ]

    baseline = _baseline_row(rows)["sts_spearman"]
    for row in rows:
        row["sts_spearman_drop"] = baseline - row["sts_spearman"]
        row["eligible"] = bool(
            row["sts_spearman"] >= baseline - MAX_STS_SPEARMAN_DROP - TOLERANCE
        )

    winner = select_lambda(rows)
    for row in rows:
        row["selected"] = bool(abs(row["lambda"] - winner) <= TOLERANCE)
    return rows


def _baseline_row(rows: Sequence[dict]) -> dict:
    """The lambda=0 row — cosine only, and every model's own reference point."""
    for row in rows:
        if abs(row["lambda"]) <= TOLERANCE:
            return row
    raise ValueError("the lambda grid must contain 0 — it is the cosine baseline")


def select_lambda(rows: Sequence[dict]) -> float:
    """Exactly one lambda from a swept grid.

    Among lambdas whose STS-dev Spearman stays within the budget: highest
    `pairwise_accuracy`, then highest `mean_gap`, then the smallest lambda. The
    last two are what make this deterministic rather than dependent on grid
    order — `pairwise_accuracy` is a fraction over 152 items, so exact ties are
    common, not hypothetical.
    """
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        # Unreachable: lambda=0's drop is 0 by construction. If it ever fires,
        # something upstream changed the baseline out from under the rule.
        raise ValueError("no eligible lambda, not even 0 — the eligibility rule is broken")
    best = min(eligible, key=lambda row: (-round(row["pairwise_accuracy"], 12),
                                          -round(row["mean_gap"], 12),
                                          row["lambda"]))
    return float(best["lambda"])


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def _assert_development_only(*paths: str | Path) -> None:
    """Refuse to run lambda selection against a held-out file.

    A comment saying "dev only" is not a guarantee; this is. Compares resolved
    paths rather than names, so a fixture that happens to be called `test.jsonl`
    is not caught by mistake.
    """
    held_out = {Path(p).resolve() for p in HELD_OUT_FILES if Path(p).exists()}
    for path in paths:
        if Path(path).resolve() in held_out:
            raise ValueError(
                f"{path} is a held-out test split and must never be read during "
                "lambda selection — run the test stage for it, once, afterwards"
            )


def _nli_terms(pairs: Sequence[Tuple[str, str]], nli, label: str,
               batch_size: int = 32) -> np.ndarray:
    """`P(entailment) - P(contradiction)` per ordered pair, computed once."""
    print(f"[nli] {label}: {len(pairs)} ordered pairs", flush=True)
    return np.asarray(nli.nli_scores(pairs, batch_size=batch_size), dtype=np.float64)


def _load_stage_data(probe_path: str | Path, sts_path: str | Path, nli,
                     batch_size: int,
                     expected_probe_split: Optional[str] = None,
                     expected_sts_split: Optional[str] = None,
                     ) -> Tuple[Dict[str, List[Tuple[str, str]]],
                                                Dict[str, np.ndarray], np.ndarray]:
    """Pairs, their NLI terms, and the STS gold scores — everything model-agnostic.

    The NLI terms are computed here, once, and then reused for every embedder:
    the classifier never sees an embedding, so re-running it per model would be
    four identical sets of forward passes.
    """
    items = load_probe(probe_path)
    if expected_probe_split is not None:
        wrong = [it.id for it in items if it.split != expected_probe_split]
        if wrong:
            preview = ", ".join(wrong[:5])
            raise ValueError(
                f"{probe_path} contains {len(wrong)} item(s) outside the "
                f"{expected_probe_split!r} split, including {preview}. Refusing "
                "to cross the dev/test boundary."
            )
    para_pairs, neg_pairs = probe_pairs(items)
    sts_triples = sts_data.load_sts(sts_path, expected_split=expected_sts_split)
    print(f"[data] probe {probe_path}: {len(items)} items | "
          f"sts {sts_path}: {len(sts_triples)} pairs")

    pair_sets = {
        "paraphrase": para_pairs,
        "negation": neg_pairs,
        "sts": [(a, b) for a, b, _ in sts_triples],
    }
    nli_terms = {name: _nli_terms(pairs, nli, name, batch_size)
                 for name, pairs in pair_sets.items()}
    gold = np.asarray([g for _, _, g in sts_triples], dtype=np.float64)
    return pair_sets, nli_terms, gold


def _metadata(nli) -> dict:
    return {
        "nli_checkpoint": getattr(nli, "model_name", ""),
        "nli_encoding": getattr(nli, "pair_encoding", ""),
        "directional": True,
    }


def model_components(model_key: str, pair_sets: Dict[str, Sequence[Tuple[str, str]]],
                     nli_terms: Dict[str, np.ndarray],
                     embedder_factory=get_embedder) -> Dict[str, PairComponents]:
    """Cosine for every pair set under one embedder, then drop the embedder.

    The embedder is local to this function on purpose: four sentence-transformers
    plus an NLI model do not comfortably share one Colab GPU, and a `del` in the
    caller's loop body would not have freed it anyway while the loop variable
    still pointed at it.
    """
    embedder = embedder_factory(model_key)
    components = {
        name: PairComponents(cosine_components(pairs, embedder), nli_terms[name])
        for name, pairs in pair_sets.items()
    }
    del embedder
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return components


def _write_table(rows: List[dict], path: str | Path, fields: Sequence[str],
                 config_key: Sequence[str], sort_key) -> int:
    """`append_or_replace`, then sort the file so the table reads in order.

    Merging matters because a Colab session may only have the memory to sweep
    one or two embedders before it has to be restarted; sorting matters because
    these files are read by humans and pasted into the report.
    """
    append_or_replace(rows, path, fields, config_key)
    with Path(path).open(encoding="utf-8", newline="") as f:
        written = list(csv.DictReader(f))
    written.sort(key=sort_key)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(written)
    return len(written)


def run_dev(models: Sequence[str] = DEFAULT_MODELS,
            nli=None,
            probe_path: str | Path = PROBE_TRAIN,
            sts_path: str | Path = sts_data.DEV_PATH,
            out_csv: str | Path = DEV_CSV,
            out_json: str | Path = SELECTED_JSON,
            grid: Sequence[float] = LAMBDA_GRID,
            embedder_factory=get_embedder,
            batch_size: int = 32) -> List[dict]:
    """Sweep lambda on the development splits and lock one value per model."""
    _assert_development_only(probe_path, sts_path)
    if nli is None:
        raise ValueError("run_dev needs an NLI scorer — build one with build_nli()")
    # A repeated model key would emit a second set of 21 rows that the config-key
    # merge cannot tell apart from the first, since both arrive in one write.
    models = list(dict.fromkeys(models))

    metadata = _metadata(nli)
    _assert_selection_merge_compatible(out_json, metadata,
                                       probe_path, sts_path, grid)

    pair_sets, nli_terms, gold = _load_stage_data(
        probe_path, sts_path, nli, batch_size,
        expected_probe_split="train", expected_sts_split="dev",
    )

    rows: List[dict] = []
    selections: Dict[str, dict] = {}

    for model_key in models:
        components = model_components(model_key, pair_sets, nli_terms, embedder_factory)

        model_rows = sweep_lambdas(components["paraphrase"], components["negation"],
                                   components["sts"], gold, grid)
        chosen = next(row for row in model_rows if row["selected"])
        baseline = _baseline_row(model_rows)

        for row in model_rows:
            row["model"] = model_key
            row.update(metadata)
        rows.extend(model_rows)

        selections[model_key] = {
            "lambda": chosen["lambda"],
            "pairwise_accuracy": chosen["pairwise_accuracy"],
            "mean_gap": chosen["mean_gap"],
            "sts_spearman": chosen["sts_spearman"],
            "sts_spearman_drop": chosen["sts_spearman_drop"],
            "baseline_pairwise_accuracy": baseline["pairwise_accuracy"],
            "baseline_mean_gap": baseline["mean_gap"],
            "baseline_sts_spearman": baseline["sts_spearman"],
            "n_eligible": sum(1 for row in model_rows if row["eligible"]),
        }
        print(f"[dev] {model_key:19s} lambda={chosen['lambda']:.2f} "
              f"acc={chosen['pairwise_accuracy']:.4f} (baseline "
              f"{baseline['pairwise_accuracy']:.4f})  gap={chosen['mean_gap']:+.4f}  "
              f"sts_rho={chosen['sts_spearman']:.4f} "
              f"(drop {chosen['sts_spearman_drop']:+.4f})", flush=True)

    total = _write_table(rows, out_csv, DEV_FIELDS, CONFIG_KEY,
                         sort_key=lambda r: (r["model"], float(r["lambda"])))
    n_locked = save_selection(selections, out_json, metadata,
                              probe_path, sts_path, grid)
    print(f"\nwrote {len(rows)} sweep rows -> {out_csv} ({total} rows total)")
    print(f"locked {n_locked} model->lambda selections -> {out_json}")
    return rows


def _selection_protocol(metadata: dict, probe_path: str | Path,
                        sts_path: str | Path, grid: Sequence[float]) -> dict:
    """Metadata that must match before partial dev sweeps may be merged."""
    return {
        "nli_checkpoint": metadata["nli_checkpoint"],
        "nli_encoding": metadata["nli_encoding"],
        "directional": True,
        "max_sts_spearman_drop": MAX_STS_SPEARMAN_DROP,
        "lambda_grid": [float(lam) for lam in grid],
        "probe_split": str(Path(probe_path).resolve()),
        "sts_split": str(Path(sts_path).resolve()),
    }


def _assert_selection_merge_compatible(path: str | Path, metadata: dict,
                                       probe_path: str | Path,
                                       sts_path: str | Path,
                                       grid: Sequence[float]) -> None:
    """Prevent one lock file from silently mixing incompatible dev runs."""
    path = Path(path)
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    expected = _selection_protocol(metadata, probe_path, sts_path, grid)
    mismatches = [
        key for key, value in expected.items()
        if previous.get(key) != value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: existing={previous.get(key)!r}, requested={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(
            f"{path} contains selections from an incompatible dev protocol "
            f"({details}). Use a different --selected-out or intentionally "
            "remove the old lock before starting a new experiment."
        )


def save_selection(selections: Dict[str, dict], path: str | Path, metadata: dict,
                   probe_path: str | Path, sts_path: str | Path,
                   grid: Sequence[float] = LAMBDA_GRID) -> int:
    """Write (or update) the locked model->lambda file.

    Per-model merge, so sweeping one embedder at a time still ends with all four
    locked. A run under a *different* NLI checkpoint starts the file over rather
    than mixing two checkpoints' selections: a lambda is only meaningful next to
    the NLI model it was chosen with.
    """
    path = Path(path)
    protocol = _selection_protocol(metadata, probe_path, sts_path, grid)
    locked = {**protocol,
              "selection_rule": ("max pairwise_accuracy, then max mean_gap, then "
                                 "smallest lambda, among lambdas whose STS-dev "
                                 f"Spearman drop is <= {MAX_STS_SPEARMAN_DROP}"),
              "selected": {}}

    if path.exists():
        _assert_selection_merge_compatible(path, metadata, probe_path, sts_path, grid)
        previous = json.loads(path.read_text(encoding="utf-8"))
        locked["selected"] = dict(previous.get("selected", {}))

    locked["selected"].update(selections)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(locked, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return len(locked["selected"])


def load_selection(path: str | Path = SELECTED_JSON) -> dict:
    """Read the locked selections, or explain what has to happen first."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run the dev stage first:\n"
            "    python -m src.harness.lambda_sweep --stage dev\n"
            "The test evaluation is only allowed to read a lambda that was "
            "already selected and saved."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_test(models: Optional[Sequence[str]] = None,
             nli=None,
             probe_path: str | Path = PROBE_TEST,
             sts_path: str | Path = sts_data.TEST_PATH,
             selected_json: str | Path = SELECTED_JSON,
             out_csv: str | Path = TEST_CSV,
             embedder_factory=get_embedder,
             batch_size: int = 32) -> List[dict]:
    """Evaluate the locked lambdas, plus the lambda=0 baseline, on the test splits.

    No grid, no selection, no write to `selected_json`. Two rows per model at
    most, and one when the selected lambda is 0 and the two configurations are
    the same run.
    """
    if nli is None:
        raise ValueError("run_test needs an NLI scorer — build one with build_nli()")

    locked = load_selection(selected_json)
    metadata = _metadata(nli)
    if (locked.get("nli_checkpoint"), locked.get("nli_encoding")) != \
            (metadata["nli_checkpoint"], metadata["nli_encoding"]):
        raise ValueError(
            f"the locked lambdas were selected with NLI checkpoint "
            f"{locked.get('nli_checkpoint')!r} ({locked.get('nli_encoding')!r}) but this "
            f"run uses {metadata['nli_checkpoint']!r} ({metadata['nli_encoding']!r}). A "
            "lambda is only valid for the NLI model it was selected with — re-run the "
            "dev stage for this checkpoint, or point --nli-model at the original one."
        )

    selected = locked.get("selected", {})
    models = list(dict.fromkeys(models if models is not None else selected))
    missing = [m for m in models if m not in selected]
    if missing:
        raise ValueError(f"no locked lambda for {missing} — sweep them on dev first")

    pair_sets, nli_terms, gold = _load_stage_data(
        probe_path, sts_path, nli, batch_size,
        expected_probe_split="test", expected_sts_split="test",
    )

    rows: List[dict] = []
    for model_key in models:
        components = model_components(model_key, pair_sets, nli_terms, embedder_factory)

        lam = float(selected[model_key]["lambda"])
        if abs(lam) <= TOLERANCE:
            # Identical run; reporting it twice would suggest two measurements.
            configurations = [(BASELINE_AND_SELECTED, 0.0)]
        else:
            configurations = [(BASELINE, 0.0), (SELECTED, lam)]

        for configuration, value in configurations:
            row = {
                "model": model_key,
                "configuration": configuration,
                "lambda": value,
                "probe_split": str(probe_path),
                **probe_metrics(components["paraphrase"], components["negation"], value),
                "sts_split": str(sts_path),
                **sts_metrics(components["sts"], gold, value),
                **metadata,
            }
            rows.append(row)
            print(f"[test] {model_key:19s} {configuration:17s} lambda={value:.2f} "
                  f"acc={row['pairwise_accuracy']:.4f} gap={row['mean_gap']:+.4f} "
                  f"sts_rho={row['sts_spearman']:.4f}", flush=True)

    order = {BASELINE: 0, BASELINE_AND_SELECTED: 0, SELECTED: 1}
    total = _write_table(rows, out_csv, TEST_FIELDS, CONFIG_KEY,
                         sort_key=lambda r: (r["model"], order.get(r["configuration"], 9)))
    print(f"\nwrote {len(rows)} final rows -> {out_csv} ({total} rows total)")
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_nli(model_name: str, subfolder: Optional[str], encoding: Optional[str],
              device: Optional[str] = None) -> NLIReranking:
    """An `NLIReranking` used only as a scorer — its own `lam` never applies here.

    `lam=0.0` rather than the class default, so that if anything ever calls
    `score` on this instance it returns plain cosine instead of silently
    contributing a second, unswept blend.
    """
    if encoding is None:
        encoding = JOINED if model_name == NLIReranking.DEFAULT_MODEL_NAME else PAIR
    return NLIReranking(lam=0.0, model_name=model_name, model_subfolder=subfolder or None,
                        pair_encoding=encoding, device=device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["dev", "test"],
                    help="dev selects lambda; test evaluates the locked selection once")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                    choices=list(MODELS) + ["fake"])
    ap.add_argument("--probe", default=None,
                    help=f"default: {PROBE_TRAIN} (dev) / {PROBE_TEST} (test)")
    ap.add_argument("--sts", default=None,
                    help=f"default: {sts_data.DEV_PATH} (dev) / {sts_data.TEST_PATH} (test)")
    ap.add_argument("--dev-out", default=DEV_CSV)
    ap.add_argument("--selected-out", default=SELECTED_JSON)
    ap.add_argument("--test-out", default=TEST_CSV)
    ap.add_argument("--nli-model", default=CLEAN_CHECKPOINT,
                    help="checkpoint fine-tuned on clean HebNLI; not retrained here")
    ap.add_argument("--nli-subfolder", default="",
                    help="subfolder inside an HF repo; '' for a local checkpoint")
    ap.add_argument("--nli-encoding", default=None, choices=[JOINED, PAIR],
                    help=f"default: {JOINED} for the released checkpoint, {PAIR} otherwise")
    ap.add_argument("--batch-size", type=int, default=32, help="NLI forward-pass batch")
    args = ap.parse_args()

    nli = build_nli(args.nli_model, args.nli_subfolder, args.nli_encoding)

    if args.stage == "dev":
        run_dev(models=args.models, nli=nli,
                probe_path=args.probe or PROBE_TRAIN,
                sts_path=args.sts or sts_data.DEV_PATH,
                out_csv=args.dev_out, out_json=args.selected_out,
                batch_size=args.batch_size)
    else:
        run_test(models=args.models, nli=nli,
                 probe_path=args.probe or PROBE_TEST,
                 sts_path=args.sts or sts_data.TEST_PATH,
                 selected_json=args.selected_out, out_csv=args.test_out,
                 batch_size=args.batch_size)


if __name__ == "__main__":
    main()
