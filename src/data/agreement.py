"""
Inter-annotator agreement on the probe.  ===  PERSON A  ===

Two commands:

    sample    draw N candidates from the raw review file into two identical
              blank copies, one per annotator
    score     merge the two filled copies and report agreement

Why this exists
---------------
The probe encodes one judgement — "is the opposition carried by a negation
marker, and by nothing else?" — applied 689 times. If that judgement is not
reproducible, the probe measures one person's taste and the whole evaluation
rests on it. The agreement number is the evidence that it is reproducible.

Two things make or break the measurement:

**Sample from the raw file.** The annotated file already carries decisions; an
annotator who sees them is agreeing, not annotating. `sample` therefore refuses
to run on a file whose `keep` column is filled.

**Report kappa, not raw agreement.** Roughly 56% of candidates are rejects, so
two annotators who both reject often will agree about half the time by luck
alone. Cohen's kappa subtracts that baseline:

    kappa = (observed - expected_by_chance) / (1 - expected_by_chance)

kappa = 1 is perfect, 0 is chance. Conventional reading: >0.8 excellent,
0.6-0.8 substantial, 0.4-0.6 moderate, <0.4 means the guideline is not crisp
enough — which is a finding about the instructions, not about the annotators.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

TRUTHY = {"y", "yes", "1", "true", "כן", "v", "x"}
FALSY = {"n", "no", "0", "false", "לא"}

#: columns an annotator needs to see — everything else is noise or a hint
SAMPLE_FIELDS = ["id", "keep", "target", "paraphrase", "negation", "neg_markers", "note"]


def _read(path: str | Path) -> List[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write(rows: List[dict], path: str | Path, fields: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _decision(value: str) -> str | None:
    v = (value or "").strip().lower()
    if v in TRUTHY:
        return "y"
    if v in FALSY:
        return "n"
    return None


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------

def cmd_sample(args) -> int:
    rows = _read(args.review)

    already = sum(1 for r in rows if _decision(r.get("keep", "")) is not None)
    if already and not args.force:
        print(f"{args.review} already has {already} decided rows.\n"
              "Sample from the *raw* review file — an annotator who sees existing\n"
              "decisions is agreeing, not annotating. Pass --force to override.")
        return 1

    if len(rows) < args.n:
        print(f"only {len(rows)} rows available, asked for {args.n}")
        return 1

    # Fixed seed, and random rather than head-of-file: the review file is sorted
    # by containment descending, so the top rows are the easy ones and sampling
    # them would inflate agreement.
    picked = random.Random(args.seed).sample(rows, args.n)
    picked.sort(key=lambda r: r["id"])
    for r in picked:
        r["keep"] = ""

    out = Path(args.out_dir)
    for name in args.annotators:
        _write(picked, out / f"sample_{name}.csv", SAMPLE_FIELDS)
        print(f"wrote {out / f'sample_{name}.csv'}")

    print(f"\n{args.n} rows, seed {args.seed}, drawn from {len(rows)} candidates")
    print(f"needing a paraphrase: {sum(1 for r in picked if not r['paraphrase'].strip())}")
    print("\nRead GUIDELINES.md first. Fill only the `keep` column, y or n.")
    print("Do not compare notes until both files are complete.")
    return 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def cohen_kappa(a: List[str], b: List[str]) -> float:
    """Kappa for two raters over the same items, computed directly so the
    script has no sklearn dependency."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    if expected == 1.0:
        return float("nan")   # both raters gave one label to everything
    return (observed - expected) / (1 - expected)


def cmd_score(args) -> int:
    files = {name: _read(path) for name, path in zip(args.annotators, args.files)}
    names = list(files)

    decisions: Dict[str, Dict[str, str]] = {}
    for name, rows in files.items():
        d = {}
        for r in rows:
            dec = _decision(r.get("keep", ""))
            if dec:
                d[r["id"]] = dec
        decisions[name] = d
        missing = len(rows) - len(d)
        if missing:
            print(f"[warn] {name}: {missing} of {len(rows)} rows not filled in")

    shared = sorted(set.intersection(*(set(d) for d in decisions.values())))
    if not shared:
        print("no rows decided by both annotators")
        return 1

    a = [decisions[names[0]][i] for i in shared]
    b = [decisions[names[1]][i] for i in shared]

    agree = sum(x == y for x, y in zip(a, b))
    raw = agree / len(shared)
    kappa = cohen_kappa(a, b)

    print(f"\nitems scored by both      {len(shared)}")
    print(f"raw agreement             {raw:.1%}  ({agree}/{len(shared)})")
    print(f"Cohen's kappa             {kappa:.3f}   {_read_kappa(kappa)}")
    print(f"\nkeep rate  {names[0]}: {a.count('y') / len(a):.0%}"
          f"   {names[1]}: {b.count('y') / len(b):.0%}")

    disagreements = [i for i, x, y in zip(shared, a, b) if x != y]
    if disagreements:
        text = {r["id"]: r for r in files[names[0]]}
        print(f"\n{len(disagreements)} disagreements — adjudicate these together:")
        for i in disagreements:
            r = text[i]
            print(f"\n  {i}   {names[0]}={decisions[names[0]][i]}  "
                  f"{names[1]}={decisions[names[1]][i]}")
            print(f"    T: {r['target']}")
            print(f"    P: {r['paraphrase'] or '<missing>'}")
            print(f"    N: {r['negation']}")

    if args.out:
        merged = []
        for r in files[names[0]]:
            i = r["id"]
            merged.append({**{k: r[k] for k in SAMPLE_FIELDS if k != "keep"},
                           f"keep_{names[0]}": decisions[names[0]].get(i, ""),
                           f"keep_{names[1]}": decisions[names[1]].get(i, ""),
                           "agreed": "" if i not in decisions[names[0]] or i not in decisions[names[1]]
                                     else ("y" if decisions[names[0]][i] == decisions[names[1]][i] else "n")})
        fields = [k for k in SAMPLE_FIELDS if k != "keep"] + \
                 [f"keep_{names[0]}", f"keep_{names[1]}", "agreed"]
        _write(merged, args.out, fields)
        print(f"\nmerged -> {args.out}")

    print("\nFor the report: n, raw agreement, kappa, and one sentence on what")
    print("the disagreements were about. That last part is usually the most")
    print("informative — it names the edge the guideline does not yet settle.")
    return 0


def _read_kappa(k: float) -> str:
    if k != k:
        return "(undefined — one rater used a single label)"
    if k > 0.8:
        return "excellent"
    if k > 0.6:
        return "substantial"
    if k > 0.4:
        return "moderate"
    return "too low — tighten GUIDELINES.md and re-run"


def main() -> None:
    ap = argparse.ArgumentParser(description="inter-annotator agreement on the probe")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="raw review file -> two blank annotator copies")
    s.add_argument("--review", default="data/probe/review.csv",
                   help="the RAW review file, before any decisions")
    s.add_argument("--out-dir", default="data/probe/agreement")
    s.add_argument("--annotators", nargs=2, default=["itay", "shachar"])
    s.add_argument("-n", type=int, default=40)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("score", help="two filled copies -> agreement + kappa")
    c.add_argument("files", nargs=2)
    c.add_argument("--annotators", nargs=2, default=["itay", "shachar"])
    c.add_argument("--out", default="data/probe/agreement/merged.csv")
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
