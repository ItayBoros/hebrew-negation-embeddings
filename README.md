# Hebrew Negation Embeddings

Text embedding models are largely **blind to negation**: a Hebrew sentence and
its opposite ("החתול ישן על הספה" vs "החתול לא ישן על הספה") land almost on top
of each other in vector space. This project **measures** how badly Hebrew and
multilingual embedders fail on negation, then tests **lightweight fixes** that
need no retraining of the base model.

Two headline metrics:
- **Cosine gap** — `mean cos(target, paraphrase) − mean cos(target, negation)`.
  A negation-aware model has a large gap.
- **Pairwise accuracy** — share of items where the paraphrase is ranked closer
  than the negation (chance = 0.5).

  This was originally specified as a NevIR-style rank with chance 0.25. It is not
  computable from a triple: NevIR needs two queries each with its own relevant
  document, and three sentences give only three distinct pairings. Getting the
  real thing needs a fourth sentence per item — a paraphrase of the negation.
  `metrics.nevir_rank_full` implements it and takes that sentence externally, so
  the frozen schema does not have to move until we decide to build them.

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

## The projection intervention

Rescales the component along a learned "negation direction" rather than
projecting it out — removing that direction would make a sentence and its
negation *more* alike, which is the opposite of the goal.

```bash
# ablate the projection's own settings: direction × centring × γ selection
python -m src.interventions.projection_report \
    --models multilingual-e5 --probe data/probe/probe.jsonl --show-sweeps
```

γ is chosen by cross-validation inside the train split, never on test. See the
module docstring in `src/interventions/projection.py` for the reasoning.

## The NLI re-ranking intervention

Blends the embedder's cosine with a Hebrew NLI model's
`P(entailment) − P(contradiction)`.

**The probe was mined from HebNLI's train split**, so any NLI model fine-tuned on
HebNLI has already seen our (target, negation) pairs carrying the gold
`contradiction` label — including `oriel9p/AlephBERT-FT-HebNLI-LCHAIM`. Scored
with such a model, this intervention measures memorisation, not negation
understanding. So we train our own on HebNLI with those rows removed.

```bash
# 1. filter, once per split. Two passes: Itay's 689 held-out promptIDs, then a
#    text-level audit for probe sentences reachable under a *different* promptID
python -m src.nli.prepare_data --split train --out data/raw/hebnli_train_clean.jsonl
python -m src.nli.prepare_data --split val   --out data/raw/hebnli_val_clean.jsonl
python -m src.nli.prepare_data --split test  --out data/raw/hebnli_test_clean.jsonl

# 2. smoke run first — minutes, on throwaway storage
python -m src.nli.train_nli --train data/raw/hebnli_train_clean.jsonl \
    --val data/raw/hebnli_val_clean.jsonl --max-train 2000 --epochs 1

# 3. the real run. --out MUST be on Drive under Colab, and --save-epochs makes a
#    disconnect recoverable; re-run the same command to resume from the newest
python -m src.nli.train_nli --train data/raw/hebnli_train_clean.jsonl \
    --val data/raw/hebnli_val_clean.jsonl --save-epochs \
    --out /content/drive/MyDrive/hebrew-negation/checkpoints/alephbert-hebnli-clean

# 4. confirm the label mapping describes what the model actually does
python -m src.interventions.check_nli_labels \
    --model checkpoints/alephbert-hebnli-clean --subfolder ""
```

`--base alephbert` (default) or `--base alephbertgimmel`; the key names the
checkpoint directory and the manifest, so runs never overwrite each other.
Each run adds a row to `results/nli_train.csv`, and the filter counts land in
`results/nli_data_<split>.json` — both go in the report.

Two things that must match between training and inference, because getting
either wrong degrades the model silently rather than raising: the **label
mapping** (ours is written into `config.id2label` with real names; the released
checkpoint exposes only LABEL_0/1/2) and the **pair encoding** (`joined` for the
released checkpoint, `pair` for ours — and `alephbert-base` cannot take segment
ids at all, since it was pretrained with `type_vocab_size=1`).

`notebooks/02_train_nli.ipynb` runs the whole path on Colab.

## Tests

```bash
python -m tests.test_data_pipeline      # mine -> finalize -> validate, offline
python -m tests.test_projection         # projection, on a planted direction
python -m tests.test_nli_data           # NLI contamination filters, offline
python -m src.data.negation --selftest  # negation lexicon only
```

`tests/test_projection.py` builds a synthetic space with a *known* negation
direction, so it can check that the method recovers it and widens the gap —
something `FakeEmbedder` cannot tell you, since it is only noise.

## How it fits together
- `src/schema.py` — the probe item format (shared contract).
- `src/interventions/base.py` — the intervention interface (shared contract).
- `src/interventions/` — `baseline`, `projection` (A), `nli_rerank` (B).
- `src/harness/` — models, metrics, and the runner (B).
- `src/nli/` — HebNLI decontamination and NLI fine-tuning (B).
- `src/data/` — HebNLI loading, Hebrew negation detection, probe mining (A).
- `data/probe/` — the negation probe (A); `mock_probe.jsonl` is a stand-in.
- `notebooks/` — Colab wrappers: `01` builds the probe (A), `02` trains NLI (B).

See **PLAN.md** for the full work split, Git workflow, and milestones.
