# The negation probe

| file | what it is |
|---|---|
| `probe.jsonl` | the probe — 303 annotated triples |
| `splits/train.jsonl`, `splits/test.jsonl` | 152 / 151, stratified by negation type |
| `review_done.csv` | every one of the 689 mined candidates with its keep/reject decision and a reason |
| `heldout_prompt_ids.txt` | HebNLI promptIDs that must not appear in NLI fine-tuning — read below |
| `review_raw.csv` | the 689 candidates as mined, before any decisions — kept so the agreement sample is reproducible |
| `agreement/` | inter-annotator agreement materials, see below |
| `mock_probe.jsonl` | 12 hand-written items, plumbing fixture only |

Each item is a triple: a `target`, a `paraphrase` that means the same thing in
different words, and a `negation` that means the opposite via a negation marker
and nothing else. The measurement is the gap between `cos(target, paraphrase)`
and `cos(target, negation)`.

## Read this before fine-tuning any NLI model

**The probe was mined from HebNLI's `train` split.** Any NLI model fine-tuned on
HebNLI has therefore already seen the probe's (target, negation) pairs *with
their gold `contradiction` label*. Scoring `nli_rerank` with such a model
measures memorisation, not negation understanding.

This applies to `oriel9p/AlephBERT-FT-HebNLI-LCHAIM`, the checkpoint
`nli_rerank.py` currently defaults to.

The fix is to fine-tune from base AlephBERT on HebNLI with these promptIDs
removed:

```python
from src.data.hebnli import load, load_heldout_prompt_ids, drop_prompts

rows = drop_prompts(load("HebArabNlpProject/HebNLI", split="train"),
                    load_heldout_prompt_ids())
```

**Exclude by `promptID`, not `pairID`.** MultiNLI gives three hypotheses per
premise, all sharing one promptID. Dropping only the contradiction pair still
leaves the model trained on our target sentence as a premise.

The list holds all 689 mined candidates rather than only the 303 that survived
review — 0.7% of HebNLI's prompts, so the cost is negligible, and it stays valid
if a later annotation pass accepts items rejected in this one.

Two side benefits of retraining rather than reusing the checkpoint: the released
one was fine-tuned partly on [LCHAIM](https://aclanthology.org/2025.findings-acl.413/),
whose premises are long paragraphs while our targets have a median of 6 tokens;
and its model card is empty, which is awkward to cite.

## Inter-annotator agreement

The probe encodes one judgement applied 689 times. The agreement number is the
evidence that the judgement is reproducible rather than one person's taste.

```bash
# already generated: 40 rows, seed 7, drawn at random from review_raw.csv
python -m src.data.agreement sample --review data/probe/review_raw.csv

# both annotators read GUIDELINES.md, fill only `keep` in their own file,
# and do not compare notes until both are done. Then:
python -m src.data.agreement score \
    data/probe/agreement/sample_itay.csv \
    data/probe/agreement/sample_shachar.csv
```

Sampled at random, not from the top of the file — `review.csv` is sorted by
containment descending, so the first rows are the easy ones and sampling them
would inflate agreement. The drawn 40 span containment 0.6 to 1.0.

Report **Cohen's kappa**, not raw agreement: 56% of candidates are rejects, so
two annotators who both reject often agree roughly half the time by luck alone.
Kappa subtracts that baseline.

`sample` refuses to run on a file that already has decisions in it — an
annotator who sees existing labels is agreeing, not annotating.

## How the probe was built

`python -m src.data.build_probe mine` narrows HebNLI to candidates, then a human
accepts, edits or rejects each one. See `src/data/build_probe.py` for the filters
and `results/probe_funnel.json` for the stage-by-stage counts.

The headline number from that funnel: of 86,352 well-formed HebNLI contradiction
pairs, **303 are usable as negation probe items** — 0.35%. 62% of contradiction
pairs introduce no negation marker at all, which is the quantified version of
"a contradiction is not automatically a negation".

Of the 689 candidates the automatic filters proposed, human review accepted 44%.
That is the argument for why the manual pass is not optional.
