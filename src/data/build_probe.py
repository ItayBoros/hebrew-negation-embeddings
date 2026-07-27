"""
Probe construction pipeline.  ===  PERSON A  ===

Turns HebNLI into `data/probe/probe.jsonl` in two passes, with a human in the
middle. The split is deliberate: **no automatic filter is trusted to produce the
final probe.** The machine narrows tens of thousands of pairs to a few hundred
candidates; a person accepts, edits, or rejects each one.

    pass 1   python -m src.data.build_probe mine     ... -> data/probe/review.csv
    (human edits review.csv, filling the `keep` column)
    pass 2   python -m src.data.build_probe finalize ... -> probe.jsonl + splits/

How a candidate triple is formed
--------------------------------
MultiNLI gives three hypotheses per premise, sharing a `promptID`. That maps
directly onto our triple:

    target      = the premise
    paraphrase  = the *entailment* hypothesis   (same meaning, different words)
    negation    = the *contradiction* hypothesis (opposite meaning)

Then the filters, in order, each one answering a specific way this can go wrong:

  ``has_contradiction``  the prompt actually has a contradiction sibling
  ``length``             premise is a normal sentence, not a fragment or a wall
  ``negation_added``     the contradiction introduces a negation marker the
                         premise did not have — this is David's point that a
                         contradiction is not automatically a negation, and it
                         is where most candidates die
  ``minimal_edit``       once negation is removed the two sentences still share
                         most of their tokens, i.e. the contradiction is a
                         minimal edit rather than a rewrite about something else
  ``paraphrase_clean``   the entailment sibling exists and does *not* itself
                         negate — otherwise it is not usable as a paraphrase

Candidates failing only ``paraphrase_clean`` are still emitted, with the
paraphrase cell blank, because the negation half is the expensive part to find
and writing a paraphrase by hand is cheap. That is the CONDAQA-style edit step.

The funnel counts are printed and saved — they belong in the dataset section of
the report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..schema import ProbeItem, save_probe, load_probe
from . import hebnli
from .negation import added_negation, find_markers, overlap, tokenize

REVIEW_FIELDS = [
    "id", "keep", "target", "paraphrase", "negation",
    "neg_markers", "neg_tier", "containment", "para_containment",
    "prompt_id", "genre", "note",
]

TRUTHY = {"y", "yes", "1", "true", "כן", "v", "x"}


# --------------------------------------------------------------------------
# pass 1 — mine candidates
# --------------------------------------------------------------------------

@dataclass
class Funnel:
    """Per-stage survivor counts. Reported in the dataset section."""
    counts: Counter = field(default_factory=Counter)

    def hit(self, stage: str) -> None:
        self.counts[stage] += 1

    def report(self, stages: Sequence[str]) -> str:
        lines, prev = [], None
        for stage in stages:
            n = self.counts[stage]
            share = "" if prev in (None, 0) else f"  ({n / prev:.0%} of previous)"
            lines.append(f"  {stage:<20} {n:>7}{share}")
            prev = n
        return "\n".join(lines)


STAGES = [
    "prompts", "has_contradiction", "length",
    "negation_added", "minimal_edit", "paraphrase_clean",
]


def mine(
    rows: List[hebnli.NLIRow],
    min_containment: float = 0.6,
    min_len_ratio: float = 0.5,
    min_tokens: int = 4,
    max_tokens: int = 30,
    include_weak: bool = False,
) -> tuple[List[dict], Funnel]:
    grouped = hebnli.group_by_prompt(rows)
    funnel = Funnel()
    candidates: List[dict] = []

    for prompt_id, siblings in grouped.items():
        funnel.hit("prompts")

        contradiction = siblings.get("contradiction")
        if contradiction is None:
            continue
        funnel.hit("has_contradiction")

        target = contradiction.premise_he
        negation = contradiction.hypothesis_he
        n_target = len(tokenize(target))
        if not (min_tokens <= n_target <= max_tokens) or target == negation:
            continue
        funnel.hit("length")

        markers = added_negation(target, negation, include_weak=include_weak)
        if not markers:
            continue
        funnel.hit("negation_added")

        ov = overlap(target, negation)
        if ov["containment"] < min_containment or ov["len_ratio"] < min_len_ratio:
            continue
        funnel.hit("minimal_edit")

        entailment = siblings.get("entailment")
        paraphrase, para_containment = "", 0.0
        if entailment is not None and not added_negation(
            target, entailment.hypothesis_he, include_weak=include_weak
        ):
            paraphrase = entailment.hypothesis_he
            para_containment = overlap(target, paraphrase)["containment"]
            funnel.hit("paraphrase_clean")

        candidates.append({
            "id": f"hn{prompt_id}",
            "keep": "?",
            "target": target,
            "paraphrase": paraphrase,
            "negation": negation,
            "neg_markers": " | ".join(m.surface for m in markers),
            "neg_tier": ",".join(sorted({m.tier for m in markers})),
            "containment": f"{ov['containment']:.2f}",
            "para_containment": f"{para_containment:.2f}",
            "prompt_id": prompt_id,
            "genre": contradiction.genre,
            "note": "" if paraphrase else "WRITE PARAPHRASE",
        })

    candidates.sort(key=lambda c: (-float(c["containment"]), c["id"]))
    return candidates, funnel


def write_review(candidates: List[dict], path: str | Path) -> None:
    """UTF-8 **with BOM** — without it Excel and Sheets mangle Hebrew on open,
    and this file is meant to be edited in a spreadsheet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows(candidates)


def read_review(path: str | Path) -> List[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# pass 2 — finalise reviewed rows into a probe
# --------------------------------------------------------------------------

def _stratum(negation: str, target: str) -> str:
    """Bucket used to keep negation types balanced across train and test.

    Without this a random split can put every `אין` item on one side, and the
    projection then learns a direction it is never tested on.
    """
    markers = added_negation(target, negation) or find_markers(negation)
    if not markers:
        return "none"
    # a phrase marker is the more specific fact about the item, so it wins even
    # if a bare `לא` also appears (אף אחד **לא** אישר)
    for m in markers:
        if m.tier == "phrase":
            return "quantifier"
    lemma = markers[0].lemma
    if lemma == "לא":
        return "particle"
    if lemma.startswith("אין") or lemma.startswith("אינ") or lemma == "אין":
        return "existential"
    if lemma in {"ללא", "בלי", "בלתי", "בלעדי"} or lemma.startswith("אי"):
        return "privative"
    return "other"


def _bucket_hash(item_id: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def validate_rows(rows: List[dict], include_weak: bool = False) -> tuple[List[dict], List[str]]:
    """Keep only accepted, well-formed triples. Returns (good, problems)."""
    good, problems = [], []
    seen_ids = set()

    for row in rows:
        rid = (row.get("id") or "").strip()
        if (row.get("keep") or "").strip().lower() not in TRUTHY:
            continue

        target = (row.get("target") or "").strip()
        paraphrase = (row.get("paraphrase") or "").strip()
        negation = (row.get("negation") or "").strip()

        if not (target and paraphrase and negation):
            problems.append(f"{rid}: kept but a field is empty (paraphrase still to write?)")
            continue
        if rid in seen_ids:
            problems.append(f"{rid}: duplicate id")
            continue
        if len({target, paraphrase, negation}) < 3:
            problems.append(f"{rid}: two of the three sentences are identical")
            continue
        if not added_negation(target, negation, include_weak=include_weak):
            problems.append(f"{rid}: negation adds no negation marker vs target")
            continue
        if added_negation(target, paraphrase, include_weak=include_weak):
            problems.append(f"{rid}: paraphrase introduces a negation marker")
            continue

        seen_ids.add(rid)
        good.append(row)

    return good, problems


def to_probe_items(rows: List[dict], test_size: float, seed: int, source: str) -> List[ProbeItem]:
    """Deterministic stratified split — same input always gives the same split,
    so a rerun never silently reshuffles what the projection was fitted on."""
    by_stratum: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_stratum[_stratum(row["negation"], row["target"])].append(row)

    items: List[ProbeItem] = []
    for stratum, group in by_stratum.items():
        group = sorted(group, key=lambda r: _bucket_hash(r["id"], seed))
        n_test = round(len(group) * test_size)
        for i, row in enumerate(group):
            note = (row.get("note") or "").strip()
            items.append(ProbeItem(
                id=row["id"],
                target=row["target"].strip(),
                paraphrase=row["paraphrase"].strip(),
                negation=row["negation"].strip(),
                source=source,
                split="test" if i < n_test else "train",
                note=note if note and note != "WRITE PARAPHRASE" else stratum,
            ))
    return sorted(items, key=lambda it: it.id)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_mine(args) -> int:
    rows = hebnli.load(args.source, split=args.split, token=args.hf_token)
    candidates, funnel = mine(
        rows,
        min_containment=args.min_containment,
        min_len_ratio=args.min_len_ratio,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        include_weak=args.include_weak,
    )
    if args.limit:
        candidates = candidates[: args.limit]
    write_review(candidates, args.out)

    print(f"loaded {len(rows)} HebNLI rows from {args.source}\n")
    print(funnel.report(STAGES))
    print(f"\ncandidates written    {len(candidates)}  -> {args.out}")
    missing = sum(1 for c in candidates if not c["paraphrase"])
    print(f"needing a paraphrase  {missing}")
    print("\nnext: open the file, set `keep` to y/n on each row, fix any")
    print("paraphrase marked WRITE PARAPHRASE, then run `finalize`.")

    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps({
            "source": args.source, "split": args.split,
            "rows": len(rows), "funnel": dict(funnel.counts),
            "candidates": len(candidates), "needing_paraphrase": missing,
            "params": {
                "min_containment": args.min_containment,
                "min_len_ratio": args.min_len_ratio,
                "min_tokens": args.min_tokens, "max_tokens": args.max_tokens,
                "include_weak": args.include_weak,
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"funnel stats          -> {args.stats_out}")
    return 0


def cmd_finalize(args) -> int:
    rows = read_review(args.review)
    good, problems = validate_rows(rows, include_weak=args.include_weak)

    for p in problems:
        print(f"[reject] {p}")
    if not good:
        print("\nno usable rows — nothing written.")
        return 1

    items = to_probe_items(good, args.test_size, args.seed, args.source_tag)
    save_probe(items, args.out)

    train = [i for i in items if i.split == "train"]
    test = [i for i in items if i.split == "test"]
    save_probe(train, Path(args.out).parent / "splits" / "train.jsonl")
    save_probe(test, Path(args.out).parent / "splits" / "test.jsonl")

    print(f"\nreviewed rows   {len(rows)}")
    print(f"accepted        {len(good)}")
    print(f"rejected        {len(problems)}")
    print(f"probe           {len(items)}  ({len(train)} train / {len(test)} test)")
    print(f"                -> {args.out} and splits/")
    print("\nby negation type:")
    for stratum, c in sorted(Counter(i.note for i in items).items()):
        n_test = sum(1 for i in items if i.note == stratum and i.split == "test")
        print(f"  {stratum:<14} {c:>4}  ({n_test} test)")
    return 0


def cmd_validate(args) -> int:
    items = load_probe(args.probe)
    problems = []
    for it in items:
        if not added_negation(it.target, it.negation):
            problems.append(f"{it.id}: negation adds no marker")
        if added_negation(it.target, it.paraphrase):
            problems.append(f"{it.id}: paraphrase introduces a marker")
        if it.split not in ("train", "test"):
            problems.append(f"{it.id}: bad split '{it.split}'")
    counts = Counter(i.split for i in items)
    print(f"{len(items)} items — {counts['train']} train / {counts['test']} test")
    for p in problems:
        print(f"[problem] {p}")
    print("ok" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="HebNLI -> candidate review CSV")
    m.add_argument("--source", default="HebArabNlpProject/HebNLI",
                   help="HF dataset id, or a path to a local HebNLI jsonl")
    m.add_argument("--split", default="train")
    m.add_argument("--hf-token", default=None)
    m.add_argument("--out", default="data/probe/review.csv")
    m.add_argument("--stats-out", default="results/probe_funnel.json")
    m.add_argument("--limit", type=int, default=0, help="keep only the top N candidates")
    m.add_argument("--min-containment", type=float, default=0.6)
    m.add_argument("--min-len-ratio", type=float, default=0.5)
    m.add_argument("--min-tokens", type=int, default=4)
    m.add_argument("--max-tokens", type=int, default=30)
    m.add_argument("--include-weak", action="store_true",
                   help="also count ambiguous markers (טרם, שום, אל+imperative)")
    m.set_defaults(func=cmd_mine)

    f = sub.add_parser("finalize", help="reviewed CSV -> probe.jsonl + splits")
    f.add_argument("--review", default="data/probe/review.csv")
    f.add_argument("--out", default="data/probe/probe.jsonl")
    f.add_argument("--test-size", type=float, default=0.5)
    f.add_argument("--seed", type=int, default=17)
    f.add_argument("--source-tag", default="hebnli")
    f.add_argument("--include-weak", action="store_true")
    f.set_defaults(func=cmd_finalize)

    v = sub.add_parser("validate", help="re-check an existing probe.jsonl")
    v.add_argument("--probe", default="data/probe/probe.jsonl")
    v.set_defaults(func=cmd_validate)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
