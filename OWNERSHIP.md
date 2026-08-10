# Ownership

Two-person split. Each person owns their files end-to-end; pushes to `main` are direct (no PR required).

## Person A — Itay · Data & Projection

- Negation probe: pull from HebNLI, filter to real negation, CONDAQA-style edits, annotation, train/test split
- Probe construction pipeline (`src/data/`) and its offline check (`tests/`)
- Projection intervention (`src/interventions/projection.py`)
- Data files under `data/probe/`
- Report sections: `report/sections/01_problem.md`, `02_related_work.md`, `03_dataset.md`

## Person B — Partner · Harness & NLI

- Eval harness (`src/harness/`): frozen model loading, cosine gap, NevIR-style rank, STS guard
- NLI re-ranking intervention (`src/interventions/nli_rerank.py`)
- NLI fine-tuning (`src/nli/`) and its offline check (`tests/test_nli_data.py`)
- Report sections: `report/sections/04_methodology.md`, `05_results.md`, `06_contribution.md`

## Shared / FROZEN 🔒

Change **only by mutual agreement** (both approve the PR):

- `src/schema.py` — probe item schema (target / paraphrase / negation triple)
- `src/interventions/base.py` — Intervention interface (`fit` + `score`)

These are the contracts both halves depend on. See CONTRIBUTING.md for the process.

## Jointly owned (either may edit, notify the other)

- `README.md`, `PLAN.md`, `requirements.txt`, `.gitignore`
- `src/interventions/baseline.py`
- `data/probe/mock_probe.jsonl` (plumbing fixture)
