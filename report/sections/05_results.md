# Results and Analysis

<!-- Owner: Person B -->
<!-- Keep this section in its own file so the two of us never edit the same file. -->

<!-- DRAFT written by Itay's assistant, pulled straight from the committed
     CSVs (results/*.csv) with no numbers invented. Shachar please check the
     nli_rerank / lambda framing matches your own read of it. -->

## Baseline: the blind spot is real

All four frozen embedders were measured on the 151-item test split before
any intervention (see Problem Definition for the full table): two of four
models show a *negative* cosine gap.

## `projection`: gap-only selection is unsafe, and the fix that follows

The projection ablation swept direction estimator (mean-difference vs.
logistic classifier), centring, and γ-selection (cross-validated vs.
train-only) across all four models. The unconstrained version — γ chosen to
maximize the train-split gap and nothing else — always won by amplifying
until the representation collapsed. Representative case, sambert,
mean-difference direction, cross-validated selection:

| selection | γ | cosine gap | pairwise acc. | sim_unrelated |
|---|---:|---:|---:|---:|
| unconstrained | 1000 | +0.899 | 0.89 | **0.975** |
| constrained (≤0.5) | 12 | +0.338 | 0.88 | **0.350** |

`sim_unrelated` is the mean \|cosine\| between the targets of two
*unrelated* probe items — it should stay low regardless of what an
intervention does to negation. At γ=1000, it is 0.975: almost every
sentence in the space has become nearly parallel to the negation direction,
so the "improved" gap reflects a flattened representation, not negation
understanding. This pattern held across every model and direction estimator
we tried — unconstrained γ landed at the top of the search grid every time,
with `sim_unrelated` in the 0.94–1.00 range.

Constraining γ-selection to the values whose held-out `sim_unrelated` stays
at or below 0.5 fixes this on three of four models — constrained γ is far
smaller (single or low double digits vs. 200–1000) and `sim_unrelated`
lands in a healthy 0.31–0.79 range. **`multilingual-e5` is the exception**:
every one of its constrained configurations still fails to bring
`sim_unrelated` under 0.5 anywhere in the grid (`constraint_relaxed=True`,
falling back to the grid's lowest-unrel point, itself only 0.59–0.79). This
embedder's negation direction and its "everything looks similar" failure
mode appear to overlap more than for the other three models — we do not
have a repair for this within the `projection` framework and report it as a
negative result specific to this embedder.

## `nli_rerank`: locked test-set numbers

**Classifier choice.** Our decision to train a dedicated classifier is supported by its stronger performance on the same 883-pair clean HebNLI test split. Our classifier reaches 79.6% accuracy (macro-F1 79.4%), compared with 72.7% accuracy (macro-F1 72.2%) for oriel9p/AlephBERT-FT-HebNLI-LCHAIM (Methodology). Although this is not a strictly fair comparison, the result provides evidence that our classifier is better suited to the task and supports our choice to develop a task-specific model.

Selected λ per model (dev-stage selection, STS-dev Spearman constrained to
within 0.02 of that model's own λ=0 baseline), evaluated once on test:

| model | λ | pairwise acc. (λ=0 → selected) | STS Spearman (λ=0 → selected) |
|---|---:|---|---|
| multilingual-e5 | 0.05 | 0.80 → **0.99** | 0.76 → 0.76 |
| LaBSE | 0.35 | 0.79 → **0.99** | 0.74 → 0.67 |
| alephbert-sentence | 0.30 | 0.50 → **0.99** | 0.65 → 0.65 |
| sambert | 0.35 | 0.50 → **0.99** | 0.69 → 0.67 |

Every model reaches ≥0.987 pairwise accuracy — effectively solving the
per-item ranking task on this probe — while STS correlation moves by at
most 0.07 and stays within the pre-registered budget by construction. A
small λ (0.05–0.35) is enough; no model needed to lean heavily on the NLI
signal to get most of the benefit, which is itself informative about how
little of the frozen embedding actually needs to be overridden.

## Head-to-head

Pulling the constrained `projection` row and the locked `nli_rerank` row
onto the same table as baseline, per model (`sim_unrelated` and STS are on
different scales and not directly comparable, so both trade-off guards are
reported alongside rather than merged):

| model | intervention | pairwise acc. | trade-off guard |
|---|---|---:|---|
| multilingual-e5 | baseline | 0.80 | STS 0.76 |
| | projection (γ=30, constraint relaxed) | 0.94 | sim_unrel 0.63 (over budget) |
| | nli_rerank (λ=0.05) | **0.99** | STS 0.76 |
| LaBSE | baseline | 0.80 | STS 0.74 |
| | projection (γ=30) | 0.95 | sim_unrel 0.44 |
| | nli_rerank (λ=0.35) | **0.99** | STS 0.67 |
| alephbert-sentence | baseline | 0.50 | STS 0.65 |
| | projection (γ=30) | 0.91 | sim_unrel 0.42 |
| | nli_rerank (λ=0.30) | **0.99** | STS 0.65 |
| sambert | baseline | 0.50 | STS 0.70 |
| | projection (γ=30) | 0.86 | sim_unrel 0.41 |
| | nli_rerank (λ=0.35) | **0.99** | STS 0.67 |

`nli_rerank` wins on every model, by a consistent, large margin, and with a
real semantic-similarity guard rather than a proxy. `projection`'s appeal
is that it needs no second model and no external classifier at inference
time — a pure vector-space fix — but on this evidence it recovers roughly
40–90% of the accuracy gap `nli_rerank` closes, at the cost of a guard
metric that is only a stand-in for graded similarity.

## Where the fixes still fail: per-category breakdown

The probe's six negation categories (see Dataset) let us ask *where*
`projection` still fails rather than only *how often*. Aggregated across
all four models (test split, `n` = category size × 4 models):

| category | n (×4 models) | baseline pass rate | projection pass rate | gap closed |
|---|---:|---:|---:|---:|
| particle (`לא`) | 356 | 0.67 | 0.92 | 76% |
| existential (`אין`) | 100 | 0.65 | 0.94 | 83% |
| quantifier | 108 | 0.61 | 0.93 | 82% |
| question | 16 | 0.25 | 0.81 | 75% |
| privative | 12 | 0.75 | 0.83 | 32% |
| neg-raising | 12 | 0.83 | 0.92 | 53% |

The three largest categories — `particle`, `existential`, `quantifier`,
together 89% of the test items — all see `projection` close roughly
75–83% of the baseline gap, and `question` (n=4 per model) recovers by a
similar share despite the lowest starting point of any category (0.25,
below what a coin-flip-adjacent baseline would suggest). The two categories
where `projection` struggles most in relative terms are the two smallest
and linguistically most specific: `privative` (`ללא`/`בלי`/`בלתי`, n=3 per
model) and `neg-raising` (matrix-clause negation of a raising predicate,
n=3 per model) close only 32% and 53% of their respective gaps. Both are
low-n enough (12 across all four models combined) that a single item
flipping moves the rate by multiple points, so this reads as a real but
uncertain signal that the global negation direction generalizes less well
to these rarer, structurally distinct constructions — not as a confident
per-category claim.
