# Implementation Plan

**Project:** Detecting and Repairing the Negation Blind Spot in Hebrew Text Embeddings
**Team:** Person A (Data & Projection) · Person B (Harness & NLI)

The whole plan is built around one goal: **both people can work at any time, on
their own part, and the parts merge cleanly.** That only works because two
interfaces are frozen up front — everything else hangs off them.

---

## The two contracts (freeze on Day 1)

Almost all merge pain comes from two people editing the same thing. We avoid it
by agreeing two small "contracts" first, then never editing them casually.

1. **Probe item schema** — `src/schema.py`. A probe item is a triple
   `(target, paraphrase, negation)` plus `source` and `split`. Person B codes
   the harness against this while Person A is still building the real data,
   using `data/probe/mock_probe.jsonl` as a stand-in.
2. **Intervention interface** — `src/interventions/base.py`. Every fix exposes
   `fit(train_items, embedder)` and `score(a, b, embedder) -> float`. Both
   `projection` and `nli_rerank` implement it, so the runner treats them
   identically.

If either contract must change, it's a 2-minute conversation, not a silent
commit.

---

## Who owns what

| Area | Person A — Data & Projection | Person B — Harness & NLI |
|---|---|---|
| Core | Build the probe: pull from HebNLI, filter to real negation, CONDAQA-style edits, hand-annotate, train/test split | Eval harness: load frozen models, cosine gap, NevIR-style rank, STS |
| Intervention | **projection** (`src/interventions/projection.py`) — main | **nli_rerank** (`src/interventions/nli_rerank.py`) — second |
| Report | `problem.md`, `dataset.md`, `related_work.md` | `methodology.md`, `results.md`, `contribution.md` |
| Files touched | `data/probe/**`, `interventions/projection.py` | `harness/**`, `interventions/nli_rerank.py` |

Each person's core depends on a *different* contract, and each intervention is
glued to that person's own core. Nobody waits on the other to make progress.

Shared files (edit only by agreement): `schema.py`, `interventions/base.py`.

---

## Repo map

```
hebrew-negation-embeddings/
  src/
    schema.py                  # CONTRACT 1 (shared, frozen)
    interventions/
      base.py                  # CONTRACT 2 (shared, frozen)
      baseline.py              # shared reference
      projection.py            # A
      nli_rerank.py            # B
    harness/
      models.py                # B — real embedders + FakeEmbedder (offline)
      metrics.py               # B — cosine_gap, nevir_rank, sts_corr
      run_eval.py              # B — end-to-end runner -> results/results.csv
  data/probe/
    mock_probe.jsonl           # tiny stand-in so the harness runs on day 1
    probe.jsonl                # A — the real annotated set (grows over time)
    splits/                    # A — train.jsonl / test.jsonl
  report/sections/             # one file per section = no write conflicts
  notebooks/                   # thin Colab notebooks that import from src/
  results/                     # results.csv + plots
  requirements.txt  .gitignore  README.md  PLAN.md
```

---

## Git workflow (keeps parallel work conflict-free)

- `main` stays green. Each person works on their own branch (`person-a`,
  `person-b`) or short feature branches, and opens a PR into `main`.
- The file ownership above means you almost never touch the same file. The only
  shared files are the two contracts — treat a change there as a shared decision.
- The report is split one-file-per-section, so writing never collides.
- **Do not commit heavy artifacts.** Model checkpoints and large data go on
  Google Drive; `.gitignore` already blocks `*.safetensors`, `checkpoints/`,
  etc. Keep `results.csv` (small) in git so both see the latest numbers.

### Colab + Drive
Keep logic in `src/` and use notebooks as thin wrappers:
```python
!git clone https://github.com/<you>/hebrew-negation-embeddings.git
%cd hebrew-negation-embeddings
!pip install -q -r requirements.txt
from src.harness.run_eval import evaluate
evaluate("data/probe/mock_probe.jsonl", ["multilingual-e5"], ["baseline", "projection"])
```
Mount Drive for checkpoints; never re-download a model you already cached.

---

## Milestones

### M0 — Foundation (together, ~half a day)
Set up the repo, freeze both contracts, and run the plumbing.
- **Done when:** `python -m src.harness.run_eval --models fake` writes
  `results/results.csv`, and both people have pushed a commit.

### M1 — First real signal (parallel)
- **A:** first ~100 probe triples, filtered so the opposition is really carried
  by negation (contradiction in HebNLI is *not* automatically negation).
  Double-annotate a sample with B; record the agreement rate.
- **B:** harness runs on real models (multilingual-e5, LaBSE) with the baseline;
  cosine gap + NevIR-style score + a first Hebrew STS wiring.
- **Sync point:** run B's harness on A's real probe → **first baseline
  measurement.** This is the check-in David asked for. Email him the numbers.
- **Done when:** we can state, with real models on ~100 pairs, how small the
  paraphrase-vs-negation gap is.

### M2 — Interventions (parallel)
- **A:** finish the projection intervention — try mean-difference *and* a
  classifier direction, sweep `alpha`, **fit on train / measure on test**. Grow
  the probe toward ~300.
- **B:** implement NLI re-ranking — Hebrew NLI on HebNLI, map contradiction
  probability to a similarity, tune the blend on train only.
- **Done when:** both interventions produce test-split numbers, each with its
  STS trade-off reported.

### M3 — Analysis & deliverables (together)
Full run across all four models, error analysis (where does it still break —
morphological negation, double negation, quantifiers), then the 8-page report,
slides, and the 5-minute video.
- **Done when:** report + slides + video are submitted.

---

## Guardrails (don't skip)
- **Train/test on the probe.** The projection learns a direction from data;
  learning and measuring on the same pairs inflates the result. Split first.
- **Probe quality over size.** A few hundred *clean* pairs beat thousands of
  noisy ones. Paraphrase must preserve meaning; negation must flip it; nothing
  else in the sentence changes.
- **Keep two headline metrics visible:** the cosine gap and the Hebrew
  pairwise accuracy (chance = 0.5). STS is the trade-off guard, not a headline.
  **Correction to the original plan:** the second headline was specified as a
  NevIR-style rank with chance 0.25. That is not computable from a triple — see
  `metrics.pairwise_accuracy`. Either add a fourth sentence per item (a
  paraphrase of the negation, which touches the frozen schema) or state in the
  report that the pairwise version is what we measure. A/B decision.
- **Lock scope:** two interventions run to the end beat four half-finished ones.
  Contrastive tuning is a stretch, only if time remains.
