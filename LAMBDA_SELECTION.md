# How lambda is chosen

**What this document is:** the argument behind `src/harness/lambda_sweep.py` —
what the number means, what it is selected against, why the rule has the shape it
has, and what could still be wrong with it. The code is the authority on *what*
happens; this is the authority on *why*.

---

## 1. What lambda controls

`nli_rerank` scores an ordered pair of Hebrew sentences by mixing two signals:

```
nli_score(a, b)      = P(entailment | a, b) - P(contradiction | a, b)
combined(a, b, lam)  = (1 - lam) * cosine(a, b) + lam * nli_score(a, b)
```

Both terms live in `[-1, 1]`, so the blend does too. Lambda is the only free
parameter, and it interpolates between two positions:

| lambda | what the score is | what it knows |
|---|---|---|
| `0.00` | pure cosine | whatever the frozen embedder encodes — **the baseline** |
| `1.00` | pure NLI | whatever the classifier decided; the embedder contributes nothing |

Before this procedure existed, `lam` defaulted to `1.0`. That default is not a
compromise between two signals, it is a decision to discard one of them: under
`lam=1` the `model` column of a results row is decorative, because the score
would be identical for all four embedders. The experiment is about *embeddings*
and negation, so a lambda that deletes the embedding is the wrong place to stop.

---

## 2. What lambda is selected against

Two measurements, and they pull in opposite directions. That opposition is the
entire reason a selection procedure is needed rather than a maximisation.

**Probe-train (152 items) — does the fix work?**
Each item is a triple `(target, paraphrase, negation)`. The headline is
`pairwise_accuracy`: the share of items where the paraphrase scores above the
negation.

```
pairwise_accuracy = mean( combined(target, paraphrase) > combined(target, negation) )
```

Chance is **0.5**, not 0.25 — it is one binary decision per item. We also record
`mean_paraphrase_score`, `mean_negation_score`, and

```
mean_gap = mean( combined(target, paraphrase) - combined(target, negation) )
```

`mean_gap` is reported alongside accuracy rather than instead of it because a
model can hold a healthy average gap while ranking a third of the individual
items backwards. Accuracy is the per-item sign; the gap is the magnitude.

**STS-dev (1,500 pairs) — did the fix break ordinary similarity?**
Hebrew STS-B (see `data/probe/STS_README.md`), scored by Pearson and Spearman
against the inherited gold scores.

The tension is direct. Pushing lambda up hands more of the score to a
three-class classifier whose output is nearly bimodal — it is built to separate
entailment from contradiction, not to place a pair on a graded 0–5 similarity
scale. A blend that is mostly NLI can therefore separate a sentence from its
negation beautifully while flattening every ordinary similarity distinction
underneath it. Optimising `pairwise_accuracy` alone would walk straight into
that, and report it as a win.

---

## 3. The rule

> **Among lambdas whose STS-dev Spearman is within 0.02 (absolute) of that same
> model's `lambda=0` Spearman, take the highest `pairwise_accuracy`. Break ties
> by the highest `mean_gap`, then by the smallest lambda.**

Grid: `0.00, 0.05, ..., 1.00` — 21 points. Run **independently for each of the
four frozen embedders**, so the output is four lambdas, not one.

### Why STS is a constraint and not a second objective

The alternative — a weighted objective like `accuracy - k * sts_drop` — needs a
`k`, and `k` is a second free parameter with no principled value and no data to
select it against. It also lets a large accuracy gain buy an unbounded amount of
semantic damage, which is exactly the failure the STS split is here to catch. A
hard budget cannot be traded away: a lambda either stays inside it or it is not
considered at all.

### Why 0.02

It is a small enough drop to be reported honestly as "no meaningful degradation"
alongside a correlation of roughly 0.7–0.8, and large enough not to reject a
lambda over sampling noise on 1,500 pairs. It is a judgement call, fixed **before
the sweep was run** and recorded in the code as `MAX_STS_SPEARMAN_DROP`, not
tuned afterwards to make a preferred lambda eligible. Changing it later means
re-running the dev stage and saying so.

### Why the budget is relative to each model's own baseline

The four embedders do not start from the same STS correlation. An absolute floor
("Spearman must exceed 0.75") would forgive a weak model for degradation it
cannot even reach and punish a strong one for a drop that leaves it still ahead.
The question is always *what did the intervention cost this model*, so every
model is measured against itself at `lambda=0`.

### Why Spearman decides eligibility and Pearson only gets reported

STS gold scores are averaged human ordinal judgements; the meaningful claim is
that the model orders pairs the way people do. Pearson additionally assumes the
relationship is linear, which a clipped blend of a cosine and a bimodal
classifier output has no reason to be. Reporting both and binding on the rank
correlation is the conservative reading. Pearson is in every result row so a
reader can check that the two do not disagree.

### Why the tie-breaks are in that order

`pairwise_accuracy` is a fraction over 152 items, so exact ties are ordinary, not
hypothetical — a whole plateau of lambdas often shares one accuracy value.

1. **`mean_gap` second.** Between two lambdas that rank the same number of items
   correctly, the one that separates them more decisively is the better
   configuration, and the more robust of the two on new items.
2. **Smallest lambda last.** If two settings are indistinguishable on both
   probe metrics, prefer the one that leans less on the NLI model. It keeps more
   of the frozen embedder in the score, is less exposed to the classifier's own
   errors and to whatever it memorised in fine-tuning, and is the more
   conservative claim to publish. It also makes the procedure deterministic:
   without a final total order, the winner would depend on grid iteration order.

`lambda=0` always has a drop of exactly 0, so it is always eligible and a
selection always exists. **A model whose selection is `0.00` is a real result,
not a failure** — it means no amount of this NLI signal bought probe accuracy
without costing more STS than the budget allows, for that embedder.

---

## 4. Direction

NLI is asymmetric: `P(entailment | a, b)` and `P(entailment | b, a)` are
different questions, and for a negation pair they are supposed to be. Every score
in this procedure is computed one way and never averaged with its reverse:

| | premise | hypothesis |
|---|---|---|
| probe, paraphrase | `target` | `paraphrase` |
| probe, negation | `target` | `negation` |
| STS | `sentence1` | `sentence2` |

Averaging the two directions would smooth over precisely the asymmetry the
experiment is trying to measure. Every result row carries `directional` so this
is on the record and not merely assumed. `tests/test_lambda_sweep.py` plants a
failure for a reversed pair, because a flipped direction changes every number
without raising anything.

---

## 5. The dev/test wall

| stage | probe | STS | may write |
|---|---|---|---|
| **dev** — selection | `splits/train.jsonl` (152) | `hebrew_stsb_dev.csv` (1,500) | `nli_lambda_dev.csv`, `nli_selected_lambdas.json` |
| **test** — final | `splits/test.jsonl` (151) | `hebrew_stsb_test.csv` (1,379) | `nli_lambda_test.csv` |

Three things enforce the ordering rather than describe it:

- `_assert_development_only` raises if a held-out file is handed to the dev
  stage. It compares resolved paths against the real files, so it cannot be
  talked around with a relative path.
- `run_test` has no code path that writes `nli_selected_lambdas.json`. It reads
  the file, and refuses to start if it does not exist or was written under a
  different NLI checkpoint than the one being run — a lambda is only meaningful
  next to the classifier it was selected with.
- The test stage takes no grid. It evaluates `lambda=0` and the one locked
  lambda per model, and nothing else. If the selected lambda *is* `0`, that is
  one run reported once, not the same numbers printed twice under two names.

The workflow in `notebooks/05_nli_lambda.ipynb` puts a commit between the two
stages on purpose: a selection committed *after* the test numbers exist proves
nothing about which came first.

**A disappointing test number is a result.** Going back to the dev stage after
seeing it converts the test split into a second dev split, silently, and every
number reported from it afterwards is an overestimate.

---

## 6. What the NLI checkpoint has to be

The checkpoint fine-tuned on **clean HebNLI** — the split that excludes the 689
promptIDs the probe was mined from (`02_train_nli.ipynb`, `src/nli/prepare_data.py`).

The released `oriel9p/AlephBERT-FT-HebNLI-LCHAIM` checkpoint was fine-tuned on
all of HebNLI, which means it has already seen our `(target, negation)` pairs
labelled `contradiction`. Selecting lambda against it would be tuning on data the
classifier had memorised, and its probe accuracy would be a measurement of that
memorisation. It stays available through `--nli-model` for comparison, and the
checkpoint that produced any given row is stamped into that row.

Not retrained here. The dev stage only runs it forward.

---

## 7. Cost

Both halves of the blend are independent of lambda, and the NLI half is
independent of the embedder as well. So each is computed exactly once per pair
and the 21 grid points are arithmetic over cached arrays:

|  | naive | as implemented |
|---|---:|---:|
| NLI ordered-pair classifications, dev stage | 151,536 | 1,804 |
| embedder encodes, dev stage | per lambda | once per model |

The 1,804 classifications are processed in batches, so the number of actual
model forward calls is much smaller and depends on `--batch-size` (about 57 at
the default batch size of 32).

`tests/test_lambda_sweep.py` counts the scorer's calls, so a regression that
reintroduces per-lambda or per-model recomputation fails the offline suite rather
than quietly costing an hour of GPU time.

---

## 8. Outputs

**`results/nli_lambda_dev.csv`** — the complete sweep, 21 rows per model:

```
model, lambda, probe_n, pairwise_accuracy, mean_paraphrase_score,
mean_negation_score, mean_gap, sts_n, sts_pearson, sts_spearman,
sts_spearman_drop, eligible, selected, nli_checkpoint, nli_encoding, directional
```

Every grid point is kept, including the ineligible ones. The rejected lambdas are
evidence: they show where the STS constraint started binding and whether the
selected point sits on a plateau or on a spike.

**`results/nli_selected_lambdas.json`** — the locked selections, one per model,
with the rule, the budget, the grid, and the two dev split paths recorded
alongside them.

**`results/nli_lambda_test.csv`** — the publication table. One row per
(model, configuration), carrying both split paths, both `n`s, the NLI checkpoint,
the encoding, and `directional`, so any row can be reproduced from what is
printed in it.

---

## 9. Known limits

- **152 items is small.** A `pairwise_accuracy` difference of one item is 0.0066.
  Differences between adjacent lambdas of that size are noise, which is part of
  why the tie-breaks end at "prefer the smaller lambda" rather than chasing the
  peak.
- **Hebrew STS-B is a translation** with inherited English gold scores, not a
  natively annotated Hebrew benchmark (`data/probe/STS_README.md`). It is a
  trade-off guard, not a headline result, and the 0.02 budget is calibrated on
  that understanding.
- **The 0.02 budget is a judgement**, not a derived quantity. It was fixed before
  the sweep and is reported as a choice.
- **A selected lambda is a property of the pair** (embedder, NLI checkpoint). It
  does not transfer to a different classifier, which is why the test stage
  refuses to run with one it was not selected against.
- **`pairwise_accuracy` is not NevIR.** Chance is 0.5; real NevIR needs a fourth
  sentence per item. See the note in `src/harness/metrics.py`.

---

## 10. Reproducing it

```bash
# dev - selects and locks lambda; never opens a test file
python -m src.harness.lambda_sweep --stage dev \
    --models multilingual-e5 labse alephbert-sentence sambert \
    --nli-model /content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean \
    --nli-subfolder "" --nli-encoding pair

# commit results/nli_lambda_dev.csv and results/nli_selected_lambdas.json here

# test - reads the locked selection, runs once, cannot change it
python -m src.harness.lambda_sweep --stage test \
    --models multilingual-e5 labse alephbert-sentence sambert \
    --nli-model /content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean \
    --nli-subfolder "" --nli-encoding pair
```

Both stages run from `notebooks/05_nli_lambda.ipynb` in Colab, which is where the
checkpoint on Drive and the GPU actually are.
