"""
Fine-tune a Hebrew encoder for NLI on clean HebNLI.  ===  PERSON B  ===

Why we train our own
--------------------
`nli_rerank.py` defaults to `oriel9p/AlephBERT-FT-HebNLI-LCHAIM`, which was
fine-tuned on all of HebNLI — including the rows the probe was mined from. It
has seen our (target, negation) pairs labelled `contradiction`, so its scores on
the probe are partly recall, not judgement, and nothing measured with it belongs
in the report. `src/nli/prepare_data.py` removes those rows; this file trains on
what is left.

Two side benefits, both from `data/probe/README.md`: the released checkpoint was
trained partly on LCHAIM, whose premises are long paragraphs against our ~6-token
targets, and its model card is empty, which is awkward to cite.

Base checkpoint
---------------
Default is `onlplab/alephbert-base` — the same base the released checkpoint used.
Holding the base fixed makes ours-vs-theirs a one-variable comparison, and that
variable is contamination. `BASE_MODELS` makes swapping it a flag change; the key
also names the output directory so runs never overwrite one another.

Two decisions that must not drift
---------------------------------
1. **Label order.** Taken from `hebnli.LABELS` and written into `config.id2label`
   with real names. The released checkpoint exposes only LABEL_0/1/2, which is
   why `check_nli_labels.py` exists; anything we train should never need that
   guesswork again.
2. **Input encoding.** Standard two-segment `tokenizer(premise, hypothesis)`,
   which yields proper `token_type_ids`. This differs from `nli_rerank.py`, which
   joins into one `"premise [SEP] hypothesis [SEP]"` string to match the released
   checkpoint's own training code. Inference must use whatever training used —
   a mismatch degrades the model silently, with no error — so the mode is
   recorded in the manifest and `nli_rerank.py` needs a matching switch before it
   loads this checkpoint.

    # smoke first — minutes, proves the pipeline before hours of GPU time
    python -m src.nli.train_nli --train data/raw/hebnli_train_clean.jsonl \
        --val data/raw/hebnli_val_clean.jsonl --max-train 2000 --epochs 1

    # the real run
    python -m src.nli.train_nli --train data/raw/hebnli_train_clean.jsonl \
        --val data/raw/hebnli_val_clean.jsonl

Weights are written under `checkpoints/`, which `.gitignore` blocks — they go to
Drive. Only the manifest is committed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..data import hebnli

#: Base encoders we can fine-tune. Same registry shape as `harness/models.py`.
#: alephbert-base first because it matches the released checkpoint's base;
#: alephbertgimmel is the stronger model to try once the pipeline is trusted.
BASE_MODELS = {
    "alephbert":       "onlplab/alephbert-base",
    "alephbertgimmel": "dicta-il/alephbertgimmel-base",
}

#: label -> class index, fixed by `hebnli.LABELS` order:
#: 0 entailment, 1 neutral, 2 contradiction. Written into the model config, so
#: the mapping travels with the weights instead of living in a comment.
LABEL_IDS: Dict[str, int] = {label: i for i, label in enumerate(hebnli.LABELS)}

#: How the premise/hypothesis pair is fed to the tokenizer. Recorded in the
#: manifest because inference has to reproduce it exactly — see the docstring.
PAIR_ENCODING = "two_segment"

#: Column order for results/nli_train.csv — the comparison table across runs.
#: The per-run JSON manifest holds the full provenance; this holds what the
#: report needs side by side, the same way projection_ablation.csv does.
CSV_FIELDS = [
    "base", "base_model", "smoke_run", "n_train", "n_val",
    "accuracy", "macro_f1",
    "lr", "batch_size", "epochs", "max_length", "warmup_ratio", "seed",
    "max_train", "checkpoint",
]

#: What makes a run a distinct configuration. Re-running an identical
#: configuration replaces its row rather than appending a duplicate, so the file
#: stays an ablation table instead of a log of every invocation. `smoke_run` is
#: in the key so a --max-train run can never overwrite a real one.
CONFIG_KEY = (
    "base", "smoke_run", "lr", "batch_size", "epochs", "max_length", "seed", "max_train",
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class NLIDataset:
    """Tokenised view over `NLIRow`s, for `Trainer`.

    Deliberately not a `torch.utils.data.Dataset` subclass: this module must
    stay importable without torch installed (the same invariant that lets
    `run_eval --models fake` run on a bare environment). `DataLoader` only needs
    `__len__` and `__getitem__`, so nothing is lost.

    Tokenisation happens per item rather than up front so that padding is
    dynamic — HebNLI's length distribution is long-tailed, and padding every
    batch to `max_length` would waste most of the compute on padding.
    """

    def __init__(self, rows: Sequence[hebnli.NLIRow], tokenizer, max_length: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = self.tokenizer(
            row.premise_he, row.hypothesis_he,
            truncation=True, max_length=self.max_length,
        )
        encoded["labels"] = LABEL_IDS[row.label]
        return encoded


def subsample(rows: Sequence[hebnli.NLIRow], n: int, seed: int) -> List[hebnli.NLIRow]:
    """Deterministic subset for smoke runs.

    Seeded and random rather than the first n rows: HebNLI is grouped by prompt
    and ordered by genre, so a head slice is one genre with three near-identical
    rows per prompt — it would exercise the pipeline but tell us nothing about
    whether training is working.
    """
    if n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(list(rows), n)


def label_counts(rows: Sequence[hebnli.NLIRow]) -> Dict[str, int]:
    """Class balance, for the manifest. A collapsed distribution after
    filtering would quietly cap achievable accuracy."""
    return {label: sum(1 for r in rows if r.label == label) for label in hebnli.LABELS}


def _signature(row: dict) -> tuple:
    """Config identity, compared as text because a CSV read back gives strings."""
    return tuple(str(row.get(key, "")) for key in CONFIG_KEY)


def append_result(row: dict, path: str | Path) -> int:
    """Add one run to the comparison table, replacing an identical config.

    Read-modify-write rather than an append-mode handle: the existing rows have
    to be re-read anyway to find a config already present, and rewriting the
    whole file keeps the header correct if the file was missing or truncated.
    Returns the total row count.
    """
    path = Path(path)
    existing: List[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            existing = [r for r in csv.DictReader(f) if _signature(r) != _signature(row)]

    rows = existing + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in CSV_FIELDS} for r in rows])
    return len(rows)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _training_arguments(output_dir: str, args, evaluate: bool):
    """Build `TrainingArguments`, tolerating the 4.46 rename of
    `evaluation_strategy` to `eval_strategy`.

    Colab's transformers version is not ours to pin, and the two spellings are
    otherwise identical, so try the current name and fall back rather than
    demanding a specific release.
    """
    from transformers import TrainingArguments  # lazy

    common = dict(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",          # we save once at the end, ourselves
        seed=args.seed,
        report_to=[],
    )
    strategy = "epoch" if evaluate else "no"
    try:
        return TrainingArguments(eval_strategy=strategy, **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy=strategy, **common)


def compute_metrics(eval_pred) -> dict:
    """Accuracy plus macro F1 — macro because a model that never predicts
    `neutral` can still look fine on accuracy alone."""
    import numpy as np  # lazy: only needed on the training path
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def train(train_rows: Sequence[hebnli.NLIRow],
          val_rows: Sequence[hebnli.NLIRow],
          args) -> tuple[dict, str]:
    """Fine-tune and save. Returns (val metrics, output directory)."""
    from transformers import (  # lazy — importing this module must stay cheap
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer,
    )

    base = BASE_MODELS[args.base]
    tokenizer = AutoTokenizer.from_pretrained(base)

    # id2label/label2id are written into config.json and saved with the weights,
    # so a later loader reads the mapping instead of guessing it.
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=len(hebnli.LABELS),
        id2label={i: label for label, i in LABEL_IDS.items()},
        label2id=dict(LABEL_IDS),
    )

    out_dir = args.out or f"checkpoints/{args.base}-hebnli-clean"
    trainer = Trainer(
        model=model,
        args=_training_arguments(str(Path(out_dir) / "_trainer"), args, bool(val_rows)),
        train_dataset=NLIDataset(train_rows, tokenizer, args.max_length),
        eval_dataset=NLIDataset(val_rows, tokenizer, args.max_length) if val_rows else None,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics if val_rows else None,
    )
    trainer.train()

    metrics = trainer.evaluate() if val_rows else {}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return metrics, out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base", default="alephbert", choices=sorted(BASE_MODELS),
                    help="base encoder to fine-tune")
    ap.add_argument("--train", default="data/raw/hebnli_train_clean.jsonl",
                    help="output of `python -m src.nli.prepare_data --split train`")
    ap.add_argument("--val", default=None,
                    help="output of the same command with --split val; "
                         "omit to train without model selection")
    ap.add_argument("--out", default=None,
                    help="checkpoint dir (default: checkpoints/<base>-hebnli-clean)")
    ap.add_argument("--max-train", type=int, default=None,
                    help="subsample the train split — for smoke runs only")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-length", type=int, default=128,
                    help="generous against the probe's ~6-token median; HebNLI "
                         "premises are sentences, not LCHAIM's paragraphs")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--stats-out", default=None,
                    help="per-run manifest (default: results/nli_train_<base>.json). "
                         "Keyed on the base so a second model cannot overwrite it")
    ap.add_argument("--results-csv", default="results/nli_train.csv",
                    help="comparison table, one row per configuration")
    args = ap.parse_args()

    if not Path(args.train).exists():
        print(f"[problem] {args.train} not found — run `python -m src.nli.prepare_data` first")
        return 1

    train_rows = hebnli.load_jsonl(args.train)
    val_rows = hebnli.load_jsonl(args.val) if args.val else []
    if not train_rows:
        raise ValueError(f"{args.train} has no usable rows")

    n_available = len(train_rows)
    if args.max_train:
        train_rows = subsample(train_rows, args.max_train, args.seed)

    smoke = bool(args.max_train)
    print(f"base                 {BASE_MODELS[args.base]}")
    print(f"train rows           {len(train_rows)}" + (f"  (of {n_available})" if smoke else ""))
    print(f"val rows             {len(val_rows)}")
    print(f"labels               {LABEL_IDS}")
    print(f"train balance        {label_counts(train_rows)}")
    if smoke:
        print("[warn] smoke run (--max-train) — do not report numbers from it")

    metrics, out_dir = train(train_rows, val_rows, args)

    print(f"\nval accuracy         {metrics.get('eval_accuracy', float('nan')):.4f}")
    print(f"val macro F1         {metrics.get('eval_macro_f1', float('nan')):.4f}")
    print(f"checkpoint           -> {out_dir}")

    stats_out = args.stats_out or f"results/nli_train_{args.base}.json"
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_out).write_text(json.dumps({
        "base": args.base, "base_model": BASE_MODELS[args.base],
        "smoke_run": smoke,
        "train_file": args.train, "val_file": args.val,
        "n_train": len(train_rows), "n_train_available": n_available,
        "n_val": len(val_rows),
        "train_balance": label_counts(train_rows),
        "label_ids": LABEL_IDS, "pair_encoding": PAIR_ENCODING,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "checkpoint": out_dir,
        "params": {
            "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
            "max_length": args.max_length, "warmup_ratio": args.warmup_ratio,
            "seed": args.seed, "max_train": args.max_train,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest             -> {stats_out}")

    n_rows = append_result({
        "base": args.base, "base_model": BASE_MODELS[args.base], "smoke_run": smoke,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "accuracy": metrics.get("eval_accuracy", ""),
        "macro_f1": metrics.get("eval_macro_f1", ""),
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "max_length": args.max_length, "warmup_ratio": args.warmup_ratio,
        "seed": args.seed, "max_train": args.max_train or "",
        "checkpoint": out_dir,
    }, args.results_csv)
    print(f"comparison table     -> {args.results_csv}  ({n_rows} rows)")

    print("\nnext: check the label mapping survived training with")
    print(f"  python -m src.interventions.check_nli_labels  (pointed at {out_dir})")
    print("then point nli_rerank.py at it — it needs a two-segment encoding mode first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
