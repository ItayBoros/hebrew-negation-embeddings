---
name: "M0 — Foundation"
about: "Repo setup, freeze contracts, run the plumbing (together, ~half a day)"
title: "M0: Foundation"
labels: ["milestone:M0"]
---

## Goal

Repo on GitHub, both contracts frozen, plumbing runs end-to-end offline.

## Tasks

- [ ] Repo pushed to GitHub, both people have access
- [ ] `main` branch protection enabled (PR + 1 review required)
- [ ] Both agree contracts are frozen: `src/schema.py`, `src/interventions/base.py` 🔒
- [ ] `python -m src.harness.run_eval --models fake` writes `results/results.csv`
- [ ] Person A has pushed a commit (branch `person-a`)
- [ ] Person B has pushed a commit (branch `person-b`)

## Done when

`run_eval` on the FakeEmbedder + mock probe produces `results/results.csv`, and both people have pushed.
