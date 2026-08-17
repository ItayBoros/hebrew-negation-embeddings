# Related Work / What Exists Today

<!-- Owner: Person A -->
<!-- Keep this section in its own file so the two of us never edit the same file. -->

**Diagnosing the negation blind spot.** Negation insensitivity in neural
text representations has been documented across tasks and architectures.
NevIR (Weller et al., 2024) builds a retrieval benchmark of document pairs
differing only by negation and finds that bi-encoder and sparse retrievers —
the same family our frozen embedders belong to — perform at or below chance,
while cross-encoders do somewhat better. CONDAQA (Ravichander et al., 2022)
shows the analogous gap in reading comprehension via contrastive edits to a
passage (paraphrase, scope change, polarity reversal) — the same three-way
contrastive design (paraphrase vs. negation around a shared target) that our
probe schema is built on. Concurrent 2025 work diagnoses negation blindness
directly inside universal sentence embedding spaces and proposes an adapter
module trained to fix it, which confirms the phenomenon is representational
rather than task-specific, though it requires additional training rather
than a post-hoc intervention on a frozen model.

**Fixing it by editing the representation.** The idea of finding a linear
"concept direction" inside an embedding space and manipulating a vector's
component along it is well established, but the two established use cases
point in opposite directions from what we need. Hard-debiasing (Bolukbasi et
al., 2016) finds a direction correlated with an unwanted attribute (e.g.
gender) and *removes* it, so that the attribute stops influencing
similarity — the "projection-out" move. Activation steering and
representation-engineering methods (e.g. Turner et al.'s Activation
Addition, 2023; Zou et al.'s Representation Engineering, 2023) instead
*add* a scaled concept vector to steer a model's behavior toward a target
concept at inference time. Our `projection` intervention needs neither: we
are not trying to erase negation's influence (that would make a sentence and
its negation *more* alike, the opposite of the goal) nor add an unrelated
steering concept. Instead we amplify the sentence's own existing component
along an estimated negation direction — closer in spirit to activation
steering's amplification mechanism than to debiasing's removal, but applied
per-sentence to a signal the sentence already carries rather than injected
from outside.

**Fixing it by fusing an external signal.** An orthogonal approach to
editing the embedding is to leave it alone and combine its score with a
second, negation-aware signal at scoring time — the strategy behind
cross-encoder re-ranking in NevIR and behind our `nli_rerank` intervention,
which blends frozen cosine similarity with a Hebrew NLI model's
entailment/contradiction judgment. This trades the appeal of a frozen,
reusable embedding for the cost of a second forward pass per pair, and its
own risk of information leakage if the NLI model was trained on data
overlapping the evaluation probe (addressed in Methodology).

**Hebrew resources.** We build on AlephBERT (Seker et al., 2022), the
Hebrew pre-trained language model underlying both the `alephbert-sentence`
embedder and the NLI classifier fine-tuned for `nli_rerank`, and on the
HebNLI dataset (HebArabNlpProject) as the source corpus our negation probe
is mined from.

**What is missing that this project addresses.** To our knowledge, the
negation blind spot has not previously been measured or repaired for Hebrew
sentence embeddings specifically, and prior post-hoc representation edits in
this space are typically reported against a single headline metric without a
guard against the intervention degrading the representation elsewhere. We
found that guard to be necessary rather than optional: gap-only selection of
the amplification strength collapses the space into looking uniformly
similar (see Results), a failure mode invisible to the headline metric it
was optimized against. Constraining selection against a second,
independent similarity signal is the methodological contribution this
project adds to the representation-editing literature above.
