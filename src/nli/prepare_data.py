"""
HebNLI -> contamination-free NLI fine-tuning data.  ===  PERSON B  ===

Why this file exists
--------------------
The probe was mined from HebNLI's *train* split (see `data/probe/README.md`).
For every probe item:

    target     = a HebNLI premise
    paraphrase = that premise's entailment sibling
    negation   = that premise's contradiction sibling

So a model fine-tuned on plain HebNLI has already been shown the probe's
(target, negation) pair carrying the gold label `contradiction`. `nli_rerank`
scores `P(entailment) - P(contradiction)`, which on those items is a lookup of a
memorised label rather than a judgement about negation. Any number produced that
way measures memorisation and cannot go in the report.

This module produces the training file that does not have that problem. It is
deliberately separate from `train_nli.py`: filtering is deterministic, CPU-only
and takes seconds, so it can be checked and re-checked offline without touching
a GPU, and it leaves an inspectable artifact plus a manifest behind.

Two filters, in order
---------------------
1. **promptID exclusion** (Person A's list, `data/probe/heldout_prompt_ids.txt`).
   MultiNLI shows one premise to an annotator who writes three hypotheses, and
   all three share a `promptID`. Dropping only the contradiction row would still
   train the model on our target as a premise next to our paraphrase, so the
   whole prompt goes. 689 prompts, ~0.7% of HebNLI.

2. **text-level audit** (this file). Filter 1 is an *identifier*-level
   guarantee. HebNLI is a machine translation of MultiNLI, so the same Hebrew
   string can legitimately surface under a different promptID — a different
   English source sentence can translate to the same Hebrew. An ID filter cannot
   see that. So we also compare the probe's sentences against the surviving
   rows as text.

Both filter counts are recorded in the manifest, because "how was contamination
controlled" is a methodology claim the report has to back with numbers.

    python -m src.nli.prepare_data --split train --out data/raw/hebnli_train_clean.jsonl
    python -m src.nli.prepare_data --split val   --out data/raw/hebnli_val_clean.jsonl

`--source` takes a HF dataset id or a local jsonl, so in Colab point it at the
copy `python -m src.data.hebnli --out ...` already cached and skip the download.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from ..data import hebnli
from ..data.build_probe import Funnel
from ..schema import load_probe

#: Survivor counts per filter, in report order. Mirrors build_probe's funnel so
#: the dataset and NLI sections of the report can show the same kind of table.
STAGES = ["loaded", "prompt_id_clean", "text_clean"]

#: The probe whose sentences must not appear in training data. Read only.
DEFAULT_PROBE = "data/probe/probe.jsonl"

#: How many overlap hits to print. The console is for noticing that it happened.
PRINTED_EXAMPLES = 5

#: How many to record. Every hit, up to a sanity bound — these are the evidence
#: behind a contamination claim in the report, and a truncated list cannot be
#: audited later. The real count is single digits; the bound only exists so a
#: filter that has gone wrong cannot write a 300k-row manifest.
MANIFEST_HITS = 500

_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# text-level audit
# --------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Fold the differences that are not real differences: leading/trailing
    space and runs of internal whitespace.

    Deliberately nothing else. Stripping punctuation or niqqud would start
    matching sentences that are not the same sentence, and every extra
    normalisation step is one more thing to justify in the report. The question
    this audit answers is the narrow one — was this exact sentence trained on —
    and exact-after-whitespace is the honest way to answer it.
    """
    return _WHITESPACE.sub(" ", text).strip()


def probe_sentences(probe_path: str | Path = DEFAULT_PROBE) -> Set[str]:
    """Every sentence the probe measures on, normalised.

    All three roles, not just target and negation: the paraphrase is the
    entailment sibling, so a model that has seen it labelled `entailment`
    against the target has been handed half the probe's answer too.
    """
    sentences: Set[str] = set()
    for item in load_probe(probe_path):
        sentences.update(normalise(t) for t in (item.target, item.paraphrase, item.negation))
    sentences.discard("")
    return sentences


def find_text_overlap(
    rows: Sequence[hebnli.NLIRow],
    sentences: Set[str],
) -> Tuple[List[hebnli.NLIRow], List[dict]]:
    """Split rows into (clean, hits) by whether either side is a probe sentence.

    Both sides are checked: a probe sentence leaks whether it lands in the
    premise or the hypothesis slot, because either way the model reads it next
    to a gold NLI label.

    Returns hits rather than raising — a non-zero count is a finding to write
    up, and the caller decides what to do with it.
    """
    clean: List[hebnli.NLIRow] = []
    hits: List[dict] = []
    for row in rows:
        premise, hypothesis = normalise(row.premise_he), normalise(row.hypothesis_he)
        matched = [
            side for side, text in (("premise", premise), ("hypothesis", hypothesis))
            if text in sentences
        ]
        if matched:
            hits.append({
                "pair_id": row.pair_id, "prompt_id": row.prompt_id,
                "label": row.label, "side": "+".join(matched),
                "premise": row.premise_he, "hypothesis": row.hypothesis_he,
            })
        else:
            clean.append(row)
    return clean, hits


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def prepare(
    rows: Sequence[hebnli.NLIRow],
    heldout: Set[str],
    sentences: Set[str],
) -> Tuple[List[hebnli.NLIRow], List[dict], Funnel]:
    """Apply both filters. Returns (clean rows, text-overlap hits, funnel)."""
    funnel = Funnel()
    for _ in rows:
        funnel.hit("loaded")

    # Person A's list, applied through his own helper so there is exactly one
    # implementation of "which prompts are held out" in the project.
    kept = hebnli.drop_prompts(rows, heldout)
    for _ in kept:
        funnel.hit("prompt_id_clean")

    # Fail loudly rather than train on a silently under-filtered set: every
    # downstream number depends on this having worked.
    survivors = {r.prompt_id for r in kept} & heldout
    if survivors:
        raise ValueError(
            f"{len(survivors)} held-out promptIDs survived filtering, e.g. "
            f"{sorted(survivors)[:3]} — drop_prompts and the id file disagree, "
            "check data/probe/heldout_prompt_ids.txt"
        )

    kept, hits = find_text_overlap(kept, sentences)
    for _ in kept:
        funnel.hit("text_clean")

    return kept, hits, funnel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", default="HebArabNlpProject/HebNLI",
                    help="HF dataset id, or a path to a local HebNLI jsonl")
    ap.add_argument("--split", default="train", choices=sorted(hebnli.SPLIT_FILES))
    ap.add_argument("--probe", default=DEFAULT_PROBE,
                    help="probe whose sentences must not appear in training data")
    ap.add_argument("--out", default="data/raw/hebnli_train_clean.jsonl",
                    help="gitignored under data/raw/ — regenerate, don't commit")
    ap.add_argument("--stats-out", default=None,
                    help="manifest path (default: results/nli_data_<split>.json)")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    rows = hebnli.load(args.source, split=args.split, token=args.hf_token)
    heldout = hebnli.load_heldout_prompt_ids()
    sentences = probe_sentences(args.probe)

    kept, hits, funnel = prepare(rows, heldout, sentences)
    if not kept:
        print("[problem] every row was filtered out — check --source and --probe")
        return 1

    n = hebnli.dump_jsonl(kept, args.out)

    print(f"\nheld-out promptIDs   {len(heldout)}")
    print(f"probe sentences      {len(sentences)}")
    print(funnel.report(STAGES))
    print(f"\ntext overlap         {len(hits)} rows dropped beyond the id filter")
    for hit in hits[:PRINTED_EXAMPLES]:
        # Identify the row, don't echo it. A Windows console is cp1252 and dies
        # on Hebrew; the sentences themselves go to the manifest, which is
        # written as UTF-8 and is where you would read them anyway.
        print(f"  [warn] {hit['pair_id']} label={hit['label']} matched on {hit['side']}")
    print(f"\nwrote                {n} rows -> {args.out}")

    stats_out = args.stats_out or f"results/nli_data_{args.split}.json"
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_out).write_text(json.dumps({
        "source": args.source, "split": args.split, "probe": args.probe,
        "heldout_prompt_ids": len(heldout), "probe_sentences": len(sentences),
        "funnel": dict(funnel.counts),
        "text_overlap": {
            "rows_dropped": len(hits),
            "hits_recorded": min(len(hits), MANIFEST_HITS),
            "hits": hits[:MANIFEST_HITS],
        },
        "rows_written": n, "out": args.out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest             -> {stats_out}")

    print("\nnext: run this once per split, then train with")
    print("  python -m src.nli.train_nli --train <train out> --val <val out>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
