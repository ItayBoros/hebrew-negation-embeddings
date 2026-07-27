"""
HebNLI loading and normalisation.  ===  PERSON A  ===

HebNLI (`HebArabNlpProject/HebNLI`) is a machine translation of MultiNLI into
Hebrew. Two facts about it drive the whole probe construction:

1. **It keeps MultiNLI's `promptID`.** In MultiNLI one premise is shown to an
   annotator who writes *three* hypotheses for it — one entailment, one neutral,
   one contradiction — and all three share a `promptID` (the `pairID` is just
   `promptID` + `e`/`n`/`c`). So the dataset already hands us, for a single
   premise, both a same-meaning sentence and an opposite-meaning sentence.
   That is exactly the shape of a probe triple:

       target     = premise
       paraphrase = the entailment sibling
       negation   = the contradiction sibling

2. **Columns are inconsistent across the released files, and this breaks
   `load_dataset`.** The released `HebNLI_*.jsonl` files do not share a schema —
   one carries an extra `hebrew_label` column that another lacks. `datasets`
   tries to unify all splits into one Arrow schema and dies:

       datasets.table.CastError: Couldn't cast ... because column names don't match

   (Same root cause as the dataset viewer's `ConfigNamesError` on the hub.)

   So we do **not** go through `load_dataset`. We fetch one file with
   `hf_hub_download` and parse it ourselves, which means no schema unification
   happens at all and `ALIASES` below absorbs the naming differences.

Access note: the repo card marks the dataset private, so downloads need a token.
Pass `--hf-token` or set `HF_TOKEN` in the environment.

Nothing in this module downloads anything at import time.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LABELS = ("entailment", "neutral", "contradiction")

#: Some files carry `hebrew_label` with Hebrew strings instead of the English
#: `original_label`. Map both onto the canonical three.
LABEL_SYNONYMS = {
    "entailment": "entailment", "גרירה": "entailment", "היסק": "entailment",
    "contradiction": "contradiction", "סתירה": "contradiction",
    "neutral": "neutral", "ניטרלי": "neutral", "נייטרלי": "neutral", "נטרלי": "neutral",
}

#: normalised field -> the column names seen in the wild, best first
ALIASES: Dict[str, List[str]] = {
    "premise_he":     ["translation1", "hebrew_sentence1", "premise_he", "premise"],
    "hypothesis_he":  ["translation2", "hebrew_sentence2", "hypothesis_he", "hypothesis"],
    "premise_en":     ["sentence1", "english_sentence1"],
    "hypothesis_en":  ["sentence2", "english_sentence2"],
    "label":          ["original_label", "hebrew_label", "label", "gold_label"],
    "pair_id":        ["pairID", "pair_id", "pairid"],
    "prompt_id":      ["promptID", "prompt_id", "promptid"],
    "genre":          ["genre"],
}


@dataclass(frozen=True)
class NLIRow:
    pair_id: str
    prompt_id: str
    label: str
    premise_he: str
    hypothesis_he: str
    premise_en: str = ""
    hypothesis_en: str = ""
    genre: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ColumnMismatch(RuntimeError):
    """Raised when a file has none of the known column spellings — better to
    fail loudly than to silently mine an empty candidate set."""


def _pick(record: dict, field: str) -> Optional[str]:
    for name in ALIASES[field]:
        if name in record and record[name] is not None:
            return str(record[name])
    return None


def normalise(record: dict) -> Optional[NLIRow]:
    """Map one raw record onto NLIRow. Returns None if it is unusable
    (missing Hebrew text, or a label outside the three MultiNLI classes)."""
    premise = _pick(record, "premise_he")
    hypothesis = _pick(record, "hypothesis_he")
    raw_label = (_pick(record, "label") or "").strip().lower()
    label = LABEL_SYNONYMS.get(raw_label, "")

    if not premise or not hypothesis or label not in LABELS:
        return None

    prompt_id = _pick(record, "prompt_id")
    pair_id = _pick(record, "pair_id")
    if prompt_id is None and pair_id:
        # pairID is promptID + a label letter; recover the group key from it
        prompt_id = pair_id.rstrip("enc")
    if prompt_id is None:
        return None

    return NLIRow(
        pair_id=pair_id or f"{prompt_id}{label[0]}",
        prompt_id=str(prompt_id),
        label=label,
        premise_he=premise.strip(),
        hypothesis_he=hypothesis.strip(),
        premise_en=_pick(record, "premise_en") or "",
        hypothesis_en=_pick(record, "hypothesis_en") or "",
        genre=_pick(record, "genre") or "",
    )


def _from_records(records: Iterable[dict]) -> List[NLIRow]:
    rows, seen_any = [], False
    for rec in records:
        seen_any = True
        row = normalise(rec)
        if row is not None:
            rows.append(row)
    if seen_any and not rows:
        raise ColumnMismatch(
            "no record matched the known HebNLI column names. "
            f"expected one of {ALIASES['premise_he']} for the Hebrew premise — "
            "inspect the file and extend ALIASES in src/data/hebnli.py"
        )
    return rows


def load_jsonl(path: str | Path) -> List[NLIRow]:
    """Load a local HebNLI-shaped .jsonl (also what `--dump` writes)."""
    def _records():
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    return _from_records(_records())


#: Split name -> the file that actually holds it in the hub repo. Downloading a
#: single file sidesteps the cross-split schema clash described at the top.
SPLIT_FILES = {
    "train":      "HebNLI_train.jsonl",
    "val":        "HebNLI_val.jsonl",
    "validation": "HebNLI_val.jsonl",
    "test":       "HebNLI_test.jsonl",
    "all":        "heb_nli.jsonl",
    "sample":     "HebNLI_sampled_2000_v2.jsonl",
}


def load_hf(
    dataset: str = "HebArabNlpProject/HebNLI",
    split: str = "train",
    token: Optional[str] = None,
    filename: Optional[str] = None,
) -> List[NLIRow]:
    """Download one file from the hub and parse it here.

    Deliberately not `load_dataset`: it unifies the schemas of all splits and
    raises `CastError` on this dataset, because the released files disagree on
    which columns exist. `hf_hub_download` fetches exactly one file, so there is
    nothing to unify. Imports are lazy so this module stays importable without
    `huggingface_hub` installed.
    """
    from huggingface_hub import hf_hub_download  # lazy

    if filename is None:
        if split not in SPLIT_FILES:
            raise KeyError(f"unknown split '{split}'. options: {sorted(SPLIT_FILES)}")
        filename = SPLIT_FILES[split]

    path = hf_hub_download(
        repo_id=dataset,
        filename=filename,
        repo_type="dataset",
        token=token or os.environ.get("HF_TOKEN"),
    )
    return load_jsonl(path)


def load(
    source: str,
    split: str = "train",
    token: Optional[str] = None,
    filename: Optional[str] = None,
) -> List[NLIRow]:
    """`source` is either a path to a local .jsonl or a Hugging Face dataset id."""
    if Path(source).exists():
        return load_jsonl(source)
    return load_hf(source, split=split, token=token, filename=filename)


def group_by_prompt(rows: Iterable[NLIRow]) -> Dict[str, Dict[str, NLIRow]]:
    """`{prompt_id: {label: row}}`.

    If a prompt has duplicates for a label (it happens after translation), the
    first occurrence wins — arbitrary but deterministic, and the hand annotation
    pass sees the text anyway.
    """
    grouped: Dict[str, Dict[str, NLIRow]] = defaultdict(dict)
    for row in rows:
        grouped[row.prompt_id].setdefault(row.label, row)
    return dict(grouped)


def dump_jsonl(rows: Iterable[NLIRow], path: str | Path) -> int:
    """Cache a normalised copy so later runs need no network.
    Written under data/raw/, which .gitignore already excludes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    """Fetch HebNLI once and cache it locally:

        python -m src.data.hebnli --split train --out data/raw/hebnli_train.jsonl
    """
    import argparse

    ap = argparse.ArgumentParser(description="download and normalise HebNLI")
    ap.add_argument("--source", default="HebArabNlpProject/HebNLI")
    ap.add_argument("--split", default="train", choices=sorted(SPLIT_FILES))
    ap.add_argument("--file", default=None,
                    help="fetch this file from the repo instead of the split default")
    ap.add_argument("--out", default="data/raw/hebnli_train.jsonl")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    rows = load(args.source, split=args.split, token=args.hf_token, filename=args.file)
    grouped = group_by_prompt(rows)
    complete = sum(1 for g in grouped.values() if len(g) == 3)
    n = dump_jsonl(rows, args.out)

    print(f"rows                 {n}")
    print(f"prompts              {len(grouped)}")
    print(f"prompts with e/n/c   {complete}")
    print(f"wrote                {args.out}")


if __name__ == "__main__":
    main()
