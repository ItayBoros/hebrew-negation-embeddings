# Methodology

<!-- Owner: Person B -->
<!-- Keep this section in its own file so the two of us never edit the same file. -->

<!-- DRAFT written by Itay's assistant from the repo's own code/commits/docs
     (src/interventions/nli_rerank.py, src/nli/, LAMBDA_SELECTION.md,
     src/harness/). Shachar built and ran this — please read it end to end,
     fix anything that misrepresents a decision you made, and take ownership
     of it before it goes in the submission. -->

We evaluate two independent, complementary repairs to the negation blind
spot on the same four frozen embedders (`multilingual-e5`, `LaBSE`,
`alephbert-sentence`, `sambert`) and the same probe.

## Harness

For every (embedder, intervention) pair we compute, on the held-out test
split: `cosine_gap` (defined in Problem Definition), `pairwise_accuracy` —
the share of items where `score(target, paraphrase) > score(target,
negation)`, chance = 0.5 — and, where wired, Pearson/Spearman correlation
against a Hebrew Semantic Textual Similarity set as a trade-off guard. The
guard exists because a fix that inflates `cosine_gap` by damaging the
representation generally is not a fix; see the projection ablation below for
why this is not a hypothetical concern.

## Intervention 1: `projection` (representation-space edit)

Estimates a "negation direction" `d` from the train split — either the
mean difference between negation and target embeddings, or the normal to a
logistic-regression hyperplane separating {target, paraphrase} from
{negation} — and rescales each vector's own component along `d`:

```
v' = v + (γ − 1) · ((v·d) − μ) · d
```

`γ = 1` is the identity; `γ > 1` amplifies the sentence's existing signal
along the negation direction. This is deliberately *not* the classic
debiasing move of projecting the direction out — removing it would make a
sentence and its negation *more* alike, the opposite of the goal (see
Related Work).

γ is selected by sweeping a grid and picking the value that maximizes the
train-split cosine gap under 5-fold cross-validation, never touching test.
The naive version of this selection — argmax over gap alone — turned out to
be unsafe (see Results: it collapses the representation). The fix
constrains selection to the γ values whose cross-validated `sim_unrelated`
(mean |cosine| between the targets of two unrelated probe items — a proxy
trade-off guard, since `projection` has no STS wired in) stays at or below a
fixed threshold (0.5), then argmaxes the gap among only those.

## Intervention 2: `nli_rerank` (signal-fusion)

Blends the frozen embedder's cosine with a directional Hebrew NLI
classifier's judgment:

```
nli_score(a, b)     = P(entailment | a, b) − P(contradiction | a, b)
combined(a, b, λ)   = (1 − λ) · cosine(a, b) + λ · nli_score(a, b)
```

`λ = 0` is pure cosine (the baseline); `λ = 1` discards the embedder
entirely. Both directions of an ordered pair are genuinely different
questions for an NLI model, so premise/hypothesis order is fixed
(`target` → `paraphrase`/`negation`) and never averaged with its reverse.

**Classifier.** AlephBERT fine-tuned for 3-way NLI (entailment / neutral /
contradiction) on HebNLI, `lr=2e-5`, batch size 32, 2 epochs, sequence
length 128, seed 17. Test accuracy 79.6% (macro-F1 79.4%) on the clean
HebNLI test split (883 pairs). Checkpoint: `CodingBz/alephbert-hebnli-clean`
on Hugging Face.

**Decontamination.** The probe is mined from HebNLI's own `train` split, so
any NLI model fine-tuned on ordinary HebNLI has already seen the probe's
`(target, negation)` pairs under the gold `contradiction` label — scoring
`nli_rerank` with such a model would measure memorization, not negation
understanding. All three HebNLI splits used for fine-tuning are filtered by
the 689 mined promptIDs (not just the 303 that survived review), by promptID
rather than pairID — MultiNLI gives three hypotheses per premise sharing one
promptID, so filtering by pairID alone leaves the model trained on our
target sentence as a premise for a different hypothesis. A second,
text-level audit catches the residual cases where a probe sentence resurfaces
under a *different* promptID: 9 of 300,067 train rows, 1 of 1,999 val rows,
0 of 884 test rows. Final fine-tuning set: 297,990 train / 1,989 val / 883
test pairs (from an original 300,067 / 1,999 / 884).

**Choosing λ.** λ is selected per embedder on a 21-point grid
(0.00–1.00, step 0.05), *not* by maximizing probe accuracy alone. The rule:
among λ whose Hebrew-STS-dev Spearman correlation stays within 0.02
(absolute) of that same embedder's own λ=0 Spearman, take the highest
`pairwise_accuracy`; ties broken by the largest mean gap, then by the
smallest λ. The 0.02 budget was fixed before the sweep ran. Selection uses
`splits/train.jsonl` (152 items) and the STS-B *dev* split (1,500 Hebrew
pairs, translated from English STS-B and STS-B `dev` gold scores); the
selected λ is then locked and evaluated exactly once on `splits/test.jsonl`
(151 items) and STS-B *test* (1,379 pairs) — a stage boundary enforced in
code (the test stage refuses to run without a prior, matching dev-stage
selection file).

## Comparability

Both interventions are `fit`/selected on train and Hebrew-STS-*dev* only,
and reported on test and Hebrew-STS-*test* only. Neither intervention's
selection procedure ever sees a test-split number before it is locked.
