# Notebooks

Keep logic in `src/`. Notebooks are thin wrappers that clone the repo,
install deps, mount Drive for checkpoints, and call into `src/`.

```python
!git clone https://github.com/<you>/hebrew-negation-embeddings.git
%cd hebrew-negation-embeddings
!pip install -q -r requirements.txt
from src.harness.run_eval import evaluate
evaluate("data/probe/mock_probe.jsonl", ["fake"], ["baseline", "projection"])
```

## `01_build_probe_and_evaluate.ipynb` (A)

The full Person A path: mine candidates from HebNLI → download for manual
review → upload the reviewed file → finalize into `probe.jsonl` + splits →
baseline measurement on the real models → projection ablation.

Colab is not just a convenience here. HebNLI's dataset card is marked private so
`load_dataset` needs a token, and the four frozen models are several GB — this
is the runtime where both are actually reachable. Store the token as a Colab
secret named `HF_TOKEN`.

The notebook stops in the middle on purpose (section 3): the miner proposes
candidates, a human decides.

## `03_eval_nli.ipynb` (B)

Scores whatever `02_train_nli.ipynb` produced on `hebnli_test_clean.jsonl` — the
883-row held-out split neither training nor model selection ever touched. Read-only:
it mounts Drive to load the checkpoint, runs a forward pass, and writes
`results/nli_test_*` — nothing here can change the weights.

Must never be pointed at `data/probe/review_raw.csv` (689 mined candidates) or
`data/probe/splits/test.jsonl` (151 negation-probe items) — see
`data/probe/README.md` for why those are held out of NLI fine-tuning in the first
place; scoring the test set on them measures something else entirely.

## `04_compare_nli_checkpoints.ipynb` (B)

Two comparisons against the released checkpoint (`oriel9p/AlephBERT-FT-HebNLI-LCHAIM`,
`nli_rerank.py`'s old default): HebNLI test-set accuracy (labelled with a caveat —
the released checkpoint was fine-tuned on all of HebNLI, so this test set was very
likely part of its own training data), and the actual `nli_rerank` negation-probe
score, both checkpoints side by side in `results/results_nli_rerank.csv` via
`run_eval.py`'s `--nli-model`/`--nli-subfolder`/`--nli-encoding` flags. Read-only —
mounts Drive for our checkpoint, never writes to it.

Both of its `nli_rerank` rows use `lam=1.0`, the old default — pure NLI, where the
embedder contributes nothing to the score. `05` is what replaces that with a
selected weight.

## `05_nli_lambda.ipynb` (B)

Selects `nli_rerank`'s lambda, then runs the locked final evaluation — the two
halves separated by a commit, in that order, and not reversible afterwards.

Section 4 sweeps `0.00 … 1.00` on **dev only** (`splits/train.jsonl`,
`hebrew_stsb_dev.csv`), independently for each of the four frozen embedders, and
locks one lambda per model into `results/nli_selected_lambdas.json` beside the
full sweep in `results/nli_lambda_dev.csv`. Section 6 reads that lock and
evaluates two configurations per model — `lambda=0` and the selected lambda — on
`splits/test.jsonl` and `hebrew_stsb_test.csv`, writing
`results/nli_lambda_test.csv`. It has no code path that can write a selection,
and `lambda_sweep` raises outright if a test file reaches the dev stage.

Read-only with respect to the NLI checkpoint: it mounts Drive to load the clean
HebNLI model and only runs it forward. Colab because that checkpoint lives on
Drive and the four embedders are several GB.

Why the rule is what it is: `LAMBDA_SELECTION.md` in the repo root.
