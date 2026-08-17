# Discussion / Contribution

<!-- Owner: Person B -->
<!-- Keep this section in its own file so the two of us never edit the same file. -->

<!-- DRAFT written by Itay's assistant. This is the section where your own
     read of "what actually matters here" belongs most — please rewrite the
     framing in your own voice, this is meant as a scaffold with the facts
     right, not a final draft. -->

## Summary

We confirmed the negation blind spot exists in Hebrew sentence embeddings —
on our probe, two of four frozen models score a sentence's negation as
*more* similar to it than a faithful paraphrase — and tested two ways to
repair it without retraining the embedder. A signal-fusion fix
(`nli_rerank`, blending cosine with a decontaminated Hebrew NLI classifier
under a principled λ selection rule) closes the gap almost completely on
every model (≥0.987 pairwise accuracy) while a semantic-similarity guard
stays within a pre-registered 0.02 budget. A pure representation-space fix
(`projection`, amplifying each sentence's own component along an estimated
negation direction) recovers a substantial share of the gap (roughly
75–95% pairwise accuracy) without a second model at inference time, but
only once its own selection procedure is constrained against a collapse
guard — unconstrained, it "succeeds" by flattening the representation until
everything looks similar to everything else.

## What is novel here

**The methodological finding, not just the fix.** The unconstrained-γ
failure mode was not something we set out to demonstrate — it fell out of
widening the search grid on real models and noticing the headline metric
kept improving in a direction that should have been suspicious. The
contribution is not "amplifying a direction can repair negation
insensitivity" (that follows fairly directly from the representation-editing
literature); it is that *doing so safely requires a second, independent
guard metric*, that gap-only selection reliably finds the degenerate
optimum instead of the intended one, and that cross-validation — the usual
defense against overfitting a metric — does not protect against this
particular failure, because collapse improves the metric on every fold,
held out or not. `nli_rerank`'s λ-selection rule was built on the same
principle independently (a hard STS budget, not a weighted trade-off), and
the two interventions' results validate each other on this point: the one
without a real trade-off metric (`projection`, using `sim_unrelated` as a
proxy) is the one that needed the constraint added after the fact, and even
so leaves one model (`multilingual-e5`) with no available γ that satisfies
it.

**Hebrew-specific measurement.** To our knowledge this is the first
measurement of the negation blind spot specific to Hebrew sentence
embeddings, on a probe built and hand-verified for that purpose rather than
translated from an English one, stratified by the actual grammatical means
Hebrew uses to express negation.

## Limitations

- **Probe size.** 303 items (151 test) is small by NLP-dataset standards;
  the smallest per-category slices (`privative`, `neg-raising`, `question`
  — 3–4 items per model) should be read as suggestive, not conclusive (see
  Results).
- **`sim_unrelated` is a proxy**, not a real semantic-similarity benchmark;
  it catches gross representational collapse but says nothing about whether
  *graded* similarity survives an intervention, which is exactly what STS
  measures for `nli_rerank`. `projection` would benefit from the same
  Hebrew-STS guard `nli_rerank` already has; we did not have time to wire it
  in for this submission.
- **Hebrew STS-B is a translation** of the English benchmark with inherited
  English gold scores, not natively annotated in Hebrew.
- **`multilingual-e5`'s unresolved case.** No γ in our search grid keeps
  `projection`'s collapse guard satisfied for this model; whether a
  different direction-estimation method, a narrower grid, or a
  fundamentally different repair is needed is open.
- **Single-run test evaluation.** Following the dev/test discipline in
  Methodology, each configuration is evaluated once on the locked test
  split rather than averaged over seeds — appropriate for the "no peeking"
  guarantee, but it means we cannot separately quantify the variance in any
  single reported number.

## Future work

Wire a real Hebrew STS guard into `projection`'s γ-selection, matching
`nli_rerank`'s. Extend the probe with a fourth sentence per item (a
paraphrase of the negation) to enable a true NevIR-style 0.25-chance metric
rather than the pairwise 0.5-chance one used throughout. Investigate why
`multilingual-e5` resists the collapse-safe region entirely — a
model-specific property of its negation direction, or of how the model
represents multilingual content generally, worth isolating rather than
averaging away.
