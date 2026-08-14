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
