# Contributing

Two-person project. See OWNERSHIP.md for who owns what.

## Branch workflow

- Each person works on their own branch (names per PLAN.md):
  - Person A: `person-a`
  - Person B: `person-b`
- Direct pushes to `main` are allowed — no PR required. PRs are optional; open one when you want the other person's eyes on a change.
- Keep `main` green: before pushing to `main`, run both offline checks and make sure they pass:
  - `python -m src.harness.run_eval --models fake`
  - `python -m tests.test_data_pipeline`

  Neither needs the network, a GPU, or a model download.
- Pull before you push (`git pull --rebase origin main`) to avoid drift.

## Frozen contracts 🔒

`src/schema.py` and `src/interventions/base.py` are shared contracts.

**Never edit them alone.** Process:

1. Open an issue describing the change and why.
2. Get an explicit 👍 from the other person on the issue.
3. Make the change in a dedicated PR touching only the contract (plus required call-site updates), with the other person as reviewer.

## Commits

- Prefix messages with your area: `data:`, `projection:`, `harness:`, `nli:`, `report:`, `infra:`.
- Don't commit anything in `.gitignore` scope: model weights, checkpoints, raw corpora, embedding dumps. Heavy artifacts go to the shared Google Drive.

## Milestones

Tracked as GitHub issues using the templates in `.github/ISSUE_TEMPLATE/` (M0–M3, per PLAN.md).
