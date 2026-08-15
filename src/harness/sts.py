"""
Hebrew STS-B loader.  ===  PERSON B  ===

`metrics.sts_corr` has always wanted a list of `(sentence_a, sentence_b, gold)`
and, until now, always got `[]` — the trade-off guard was wired to nothing and
every `sts_pearson`/`sts_spearman` cell in `results/` came out blank. This module
is the missing half: it reads the translated STS-B files in `data/probe/` and
hands back exactly that list.

Two splits, and the difference between them is the whole point (see
`data/probe/STS_README.md`):

    hebrew_stsb_dev.csv    1,500 pairs   selecting lambda
    hebrew_stsb_test.csv   1,379 pairs   the final locked evaluation, once

Direction is preserved: `sentence1` is the premise and `sentence2` the
hypothesis, so the ordered NLI term in `nli_rerank` means the same thing here as
it does on the probe. The pairs come back in file order and nothing is shuffled,
dropped, or rescored — `gold_score` is the inherited English annotation, copied
through untouched.

The files are UTF-8 with a BOM (written by the translation pipeline), so they are
read as `utf-8-sig`; with plain `utf-8` the first column name arrives as
`\\ufeffpair_id` and every lookup of `pair_id` fails.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

#: The two translated splits, and how many rows each must have. Keyed by file
#: name so a checkout under any working directory still gets the count checked;
#: a file we do not know is simply not size-checked.
DEV_PATH = "data/probe/hebrew_stsb_dev.csv"
TEST_PATH = "data/probe/hebrew_stsb_test.csv"
EXPECTED_N = {"hebrew_stsb_dev.csv": 1500, "hebrew_stsb_test.csv": 1379}
EXPECTED_SPLIT = {"hebrew_stsb_dev.csv": "dev", "hebrew_stsb_test.csv": "test"}

REQUIRED_COLUMNS = ("pair_id", "split", "sentence1_he", "sentence2_he", "gold_score")


@dataclass(frozen=True)
class STSPair:
    """One translated pair. `sentence1` is the premise, `sentence2` the hypothesis."""

    pair_id: str
    sentence1: str
    sentence2: str
    gold: float
    split: str = ""


def load_sts_pairs(path: str | Path, expected_n: Optional[int] = None,
                   expected_split: Optional[str] = None) -> List[STSPair]:
    """Read a Hebrew STS-B csv into `STSPair`s, in file order.

    `expected_n` defaults to the published count for the two known files, so
    `load_sts_pairs(DEV_PATH)` refuses to return 1,499 rows without saying so.
    """
    path = Path(path)
    if expected_n is None:
        expected_n = EXPECTED_N.get(path.name)
    if expected_split is None:
        expected_split = EXPECTED_SPLIT.get(path.name)

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; found {list(rows[0])}")

    pairs = []
    seen_ids = set()
    for row in rows:
        pair_id = (row["pair_id"] or "").strip()
        if not pair_id:
            raise ValueError(f"{path}: empty pair_id")
        if pair_id in seen_ids:
            raise ValueError(f"{path}: duplicate pair_id {pair_id}")
        seen_ids.add(pair_id)

        split = (row["split"] or "").strip()
        if expected_split is not None and split != expected_split:
            raise ValueError(
                f"{path}: pair {pair_id} has split {split!r}; expected "
                f"{expected_split!r}"
            )

        sentence1 = (row["sentence1_he"] or "").strip()
        sentence2 = (row["sentence2_he"] or "").strip()
        if not sentence1 or not sentence2:
            raise ValueError(f"{path}: empty Hebrew sentence in pair {pair_id}")
        try:
            gold = float(row["gold_score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid gold score in pair {pair_id}") from exc
        if not 0.0 <= gold <= 5.0:
            raise ValueError(
                f"{path}: gold score {gold!r} outside [0, 5] in pair {pair_id}"
            )
        pairs.append(STSPair(
            pair_id=pair_id,
            sentence1=sentence1,
            sentence2=sentence2,
            gold=gold,
            split=split,
        ))

    if expected_n is not None and len(pairs) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} pairs, read {len(pairs)}")
    return pairs


def load_sts(path: str | Path, expected_n: Optional[int] = None,
             expected_split: Optional[str] = None
             ) -> List[Tuple[str, str, float]]:
    """`(sentence_a, sentence_b, gold)` triples — the shape `sts_corr` takes."""
    return [(p.sentence1, p.sentence2, p.gold)
            for p in load_sts_pairs(path, expected_n, expected_split)]
