# Hebrew Negation Embeddings

Text embedding models are largely **blind to negation**: a Hebrew sentence and
its opposite ("החתול ישן על הספה" vs "החתול לא ישן על הספה") land almost on top
of each other in vector space. This project **measures** how badly Hebrew and
multilingual embedders fail on negation, then tests **lightweight fixes** that
need no retraining of the base model.

Two headline metrics:
- **Cosine gap** — `mean cos(target, paraphrase) − mean cos(target, negation)`.
  A negation-aware model has a large gap.
- **Hebrew NevIR-style rank** — right-rank accuracy on contrasting query/doc
  pairs (chance = 0.25).

STS correlation is tracked as a **trade-off guard**: a fix that repairs negation
but wrecks ordinary similarity is not useful.

## Quickstart (offline, no downloads)

```bash
pip install -r requirements.txt        # or just numpy for the fake run
python -m src.harness.run_eval --models fake
```
This uses a deterministic `FakeEmbedder` so the pipeline runs end-to-end with no
model downloads. The numbers are meaningless — it only proves the plumbing.

## Real run

```bash
python -m src.harness.run_eval \
    --models multilingual-e5 labse \
    --interventions baseline projection \
    --probe data/probe/probe.jsonl
```
Results are written to `results/results.csv`.

## Building the probe

HebNLI is a Hebrew translation of MultiNLI, so one premise comes with three
hypotheses sharing a `promptID` — entailment, neutral, contradiction. That is
already a triple: premise as `target`, the entailment sibling as `paraphrase`,
the contradiction sibling as `negation`.

The catch is that **a contradiction is not automatically a negation** — antonyms,
number swaps and entity swaps are all labelled `contradiction`. So mining is a
filter, not a conversion, and a human reviews everything it proposes.

```bash
# 1. mine candidates -> a CSV to review in a spreadsheet
python -m src.data.build_probe mine --out data/probe/review.csv
#    (HebNLI may need a token: --hf-token ... or set HF_TOKEN)

# 2. open review.csv, set `keep` to y/n per row, fix rows marked WRITE PARAPHRASE

# 3. reviewed CSV -> probe.jsonl + a deterministic, stratified train/test split
python -m src.data.build_probe finalize --review data/probe/review.csv

# re-check an existing probe at any time
python -m src.data.build_probe validate
```

`mine` prints a funnel (how many pairs died at each filter) and saves it to
`results/probe_funnel.json` — those numbers go in the dataset section of the
report.

## Tests

```bash
python -m tests.test_data_pipeline     # mine -> finalize -> validate, offline
python -m src.data.negation --selftest # negation lexicon only
```

## How it fits together
- `src/schema.py` — the probe item format (shared contract).
- `src/interventions/base.py` — the intervention interface (shared contract).
- `src/interventions/` — `baseline`, `projection` (A), `nli_rerank` (B).
- `src/harness/` — models, metrics, and the runner (B).
- `src/data/` — HebNLI loading, Hebrew negation detection, probe mining (A).
- `data/probe/` — the negation probe (A); `mock_probe.jsonl` is a stand-in.

See **PLAN.md** for the full work split, Git workflow, and milestones.
