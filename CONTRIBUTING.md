# Contributing

Two-person project. See OWNERSHIP.md for who owns what.

## Branch workflow

- `main` is protected — no direct pushes. It should always run (`run_eval` on the FakeEmbedder + mock probe must not break).
- Each person works on their own long-lived branch (names per PLAN.md):
  - Person A: `person-a`
  - Person B: `person-b`
- Merge into `main` via pull request. Small, frequent PRs beat big ones.
- Review rule: the **other** person reviews. For changes strictly inside your own files, a quick approval is fine; the review is mainly to keep both of us aware of the whole codebase.
- Rebase your branch on `main` regularly (`git pull --rebase origin main`) to avoid drift.

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
