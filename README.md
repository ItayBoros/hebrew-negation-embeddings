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

## How it fits together
- `src/schema.py` — the probe item format (shared contract).
- `src/interventions/base.py` — the intervention interface (shared contract).
- `src/interventions/` — `baseline`, `projection` (A), `nli_rerank` (B).
- `src/harness/` — models, metrics, and the runner (B).
- `data/probe/` — the negation probe (A); `mock_probe.jsonl` is a stand-in.

See **PLAN.md** for the full work split, Git workflow, and milestones.
