# Problem Definition

<!-- Owner: Person A -->
<!-- Keep this section in its own file so the two of us never edit the same file. -->

Sentence embedding models are trained to place sentences with similar meaning
close together in vector space, and are used downstream as a drop-in
similarity function for retrieval, clustering, and semantic search. That use
carries an implicit assumption: that the embedding space is *sensitive to
truth-conditional meaning*, not just to topic. Negation is the sharpest test
of that assumption. A sentence and its negation share almost every surface
word, cover the identical topic, and yet mean the opposite thing. If an
embedding model is really encoding meaning, cosine similarity should treat a
paraphrase as close and a negation as far. If the model is largely encoding
lexical overlap, it treats a sentence and its negation as near-duplicates —
the *negation blind spot*.

This is not a hypothetical failure mode. NevIR (Weller et al., 2024) shows
that neural retrieval models, including strong bi-encoders, rank documents
differing only by negation close to chance. CONDAQA (Ravichander et al.,
2022) shows the same gap in reading comprehension: state-of-the-art QA models
answer questions about a passage and its negated counterpart inconsistently
far more often than humans do. Concurrent work diagnoses the same problem
directly in sentence embedding spaces (e.g. "Semantic Adapter for Universal
Text Embeddings", 2025), confirming it is a property of the *representation*,
not just of a particular downstream task.

**This project asks whether the same blind spot exists in Hebrew sentence
embeddings, and whether it can be repaired without retraining the underlying
model.** Hebrew is a useful and under-studied test case: negation is carried
by a small, closed set of markers (`לא`, `אין`/`אינו`, `ללא`/`בלי`/`בלתי`,
quantifier phrases like `אף אחד` and `אף פעם`) rather than by a single
universal word as in English, and no prior work we are aware of has measured
or addressed this specifically for Hebrew embeddings.

### Measuring the blind spot

We define the **cosine gap** for a `(target, paraphrase, negation)` triple as

```
gap = cos(target, paraphrase) − cos(target, negation)
```

A model that is sensitive to negation should have a gap well above zero: the
paraphrase, which preserves meaning, should sit closer to the target than the
negation, which reverses it. We measured this on four frozen, publicly
available embedding models usable for Hebrew — `multilingual-e5-base`,
`LaBSE`, `sentence-transformers-alephbert`, and `sambert` — on our held-out
test split (151 triples, described in the Dataset section). The result:

| model | sim(target, paraphrase) | sim(target, negation) | cosine gap |
|---|---|---|---|
| multilingual-e5 | 0.963 | 0.941 | **+0.022** |
| LaBSE | 0.890 | 0.824 | **+0.066** |
| alephbert-sentence | 0.871 | 0.890 | **−0.020** |
| sambert | 0.879 | 0.896 | **−0.017** |

Two of the four models place the *negation* marginally closer to the target
than the paraphrase — the gap is negative. The other two show a gap under
0.07, on a similarity scale where both paraphrase and negation already sit
above 0.82. By any of these models' own metric, a sentence and its opposite
are nearly indistinguishable from a sentence and a faithful restatement of
it. This is the blind spot, quantified, in Hebrew, on all four models
available to us.

### Research questions

1. **Can the blind spot be repaired post-hoc**, without fine-tuning the
   embedding model itself, by intervening on the vector space directly?
2. **How does a representation-space fix compare to a signal-fusion fix** —
   blending the frozen embedding's cosine with an external Hebrew NLI
   classifier's judgment — on the same probe and the same models?
3. **Does a fix that looks successful on its own headline metric actually
   work**, or can it "succeed" by degenerate means (e.g. collapsing the
   representation so that everything looks equally similar to everything
   else, which trivially opens any gap)? This turned out not to be a
   hypothetical concern — see Results.

We treat (1) as the `projection` intervention, (2) as the `nli_rerank`
intervention, and (3) as a methodological finding that ended up shaping both.
