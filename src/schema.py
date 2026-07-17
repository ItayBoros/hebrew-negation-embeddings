"""
Contract 1 — the shape of a single probe item.

This file is SHARED and FROZEN. Both people code against it.
Change it only by mutual agreement (it breaks the other person's code otherwise).

A probe item is a *triple*:
    target      - the original Hebrew sentence
    paraphrase  - a meaning-preserving rewrite. Should stay CLOSE to target.
    negation    - a negated / reversed variant. Should move FAR from target.

The whole project hinges on one expectation:
    cos(target, paraphrase)  should be HIGH
    cos(target, negation)    should be LOW
A model that is blind to negation collapses that gap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Literal, Tuple

Split = Literal["train", "test"]
Source = Literal["hebnli", "condaqa-style", "handwritten"]
PairKind = Literal["paraphrase", "negation"]


@dataclass(frozen=True)
class ProbeItem:
    id: str
    target: str
    paraphrase: str
    negation: str
    source: str          # one of Source
    split: str           # one of Split
    note: str = ""       # optional annotator comment (e.g. "morphological negation")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "ProbeItem":
        d = json.loads(line)
        return ProbeItem(**d)


def load_probe(path: str | Path) -> List[ProbeItem]:
    path = Path(path)
    items: List[ProbeItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(ProbeItem.from_json(line))
    return items


def save_probe(items: List[ProbeItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")


def split_items(items: List[ProbeItem], which: Split) -> List[ProbeItem]:
    return [it for it in items if it.split == which]


def iter_pairs(items: List[ProbeItem], kind: PairKind) -> Iterator[Tuple[str, str]]:
    """Yield (target, variant) for the requested relationship."""
    for it in items:
        variant = it.paraphrase if kind == "paraphrase" else it.negation
        yield it.target, variant
