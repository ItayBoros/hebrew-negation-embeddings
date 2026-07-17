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
