# Dataset Construction

<!-- Owner: Person A -->
<!-- Keep this section in its own file so the two of us never edit the same file. -->

No existing Hebrew negation probe fit this project's needs, so we built one
from HebNLI's `train` split (HebArabNlpProject), the Hebrew counterpart of
MultiNLI. Each usable item is a triple:

- **target** — an original Hebrew sentence.
- **paraphrase** — a meaning-preserving rewrite in different words.
- **negation** — a variant that reverses meaning through a negation marker
  and nothing else.

The design constraint that drove everything else: `negation` may differ from
`target` **only in polarity**. HebNLI's own `contradiction` label is not a
proxy for this — a contradiction can equally arise from an antonym, a
changed number, or a world-knowledge fact, none of which involve negation at
all. Distinguishing "negated" from merely "contradictory" is exactly the
CONDAQA-style discipline the probe schema enforces, and it is the reason the
probe could not simply be *sampled* from HebNLI's contradiction pairs — it
had to be *filtered and hand-verified* down to the subset where negation is
the actual mechanism.

### Mining funnel

An automatic filter (`src/data/build_probe.py mine`) narrows HebNLI's train
split to candidate triples before any human looks at them:

| stage | pairs remaining | share of previous stage |
|---|---:|---:|
| HebNLI prompts | 100,390 | — |
| has a contradiction pair | 96,929 | 97% |
| target/negation well-formed length | 86,352 | 89% |
| an actual negation marker was added | 32,733 | 38% |
| minimal single-marker edit | 689 | 2% |

The single most informative number in this table is the 38% at the
"negation marker added" stage: **62% of HebNLI's contradiction pairs contain
no negation marker whatsoever.** They are contradictions built from antonyms,
swapped numbers, or entity substitutions — real contradictions, but not
negations, and exactly the items a probe that equated "contradiction" with
"negation" would have contaminated itself with.

### Manual review

The 689 surviving candidates (0.7% of HebNLI's prompts) were each reviewed
by hand against a fixed question: *is the opposition between target and
negation carried by a negation marker, and by nothing else?* (full criteria
in `data/probe/agreement/GUIDELINES.md`). Common rejects were content
silently dropped alongside the negation marker, or a paraphrase that was
really a near-copy. **303 of 689 candidates (44%) were accepted** — the
number that argues the manual pass was not optional: an automatic filter
alone would have shipped a probe that was more than half wrong by the
project's own definition of a usable item.

**Inter-annotator agreement.** To check the accept/reject judgement is
reproducible rather than one annotator's taste, both authors independently
label the same random 40-candidate sample (`data/probe/agreement/`, drawn
uniformly rather than from the top of the sorted file, so as not to inflate
agreement by sampling only the easy cases). We report Cohen's kappa rather
than raw agreement, since roughly 56% of all candidates are rejects and two
annotators who both lean toward rejecting would otherwise look artificially
consistent. *(Scoring is implemented in `src/data/agreement.py score`; the
40-item double-annotation itself is the one piece of this section still
outstanding at time of writing and should be completed and the resulting
kappa inserted here before submission.)*

### Composition and split

The final probe has 303 items, split 152 (train) / 151 (test). The split is
deterministic and **stratified by negation type**, so a re-run cannot
silently reshuffle what an intervention was fit on, and no negation type
ends up entirely on one side of the split. Six types emerged from the data,
the first four defined automatically from the negation marker itself and the
last two added by annotators for constructions the automatic rule does not
capture cleanly:

| type | example marker(s) | count | share |
|---|---|---:|---:|
| particle | `לא` | 175 | 58% |
| quantifier | `אף אחד`, `אף פעם`, `שום` (phrase-level) | 54 | 18% |
| existential | `אין`, `אינו`/`אינה`/`אינם` | 49 | 16% |
| question | negation inside an interrogative | 13 | 4% |
| privative | `ללא`, `בלי`, `בלתי`, `אי־` prefix | 7 | 2% |
| neg-raising | matrix-clause negation of a raising predicate (`אני חושב ש...` → `אני לא חושב ש...`) | 5 | 2% |

`particle` dominates, which mirrors how negation is actually expressed in
Hebrew text — `לא` is the default, general-purpose negator, and the other
five types are progressively more specialized constructions. The two
smallest categories, `privative` and `neg-raising`, are also linguistically
the most subtle: `neg-raising` sentences negate the matrix verb (*"I think X"
→ "I don't think X"*) to convey what is semantically a negation of the
embedded clause, a mismatch between surface and logical form that a
lexical-overlap-sensitive embedding is particularly likely to miss. Their
small counts (n=3–5 per category on the test split) mean per-category
results for these two should be read as suggestive, not conclusive — see
Results.

### Reuse safeguard for NLI fine-tuning

Because the probe is mined from HebNLI's own train split, any NLI model
later fine-tuned on HebNLI for the `nli_rerank` intervention risks having
already seen the probe's (target, negation) pairs with their gold
`contradiction` label — at which point scoring `nli_rerank` with that model
would measure memorization, not negation understanding. All 689 mined
promptIDs (not just the 303 that passed review, in case a later annotation
pass revisits a reject) are recorded in `data/probe/heldout_prompt_ids.txt`
and excluded from NLI fine-tuning data by promptID — see Methodology for why
promptID rather than pairID is the correct exclusion key.
