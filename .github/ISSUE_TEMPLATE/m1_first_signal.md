---
name: "M1 — First real signal"
about: "~100 real triples + harness on real models → first baseline measurement (parallel)"
title: "M1: First real signal"
labels: ["milestone:M1"]
---

## Goal

First baseline measurement: real models on ~100 real probe pairs.

## Person A — Data

- [ ] ~100 probe triples from HebNLI, filtered to real negation (contradiction ≠ negation)
- [ ] Double-annotate a sample with B; record agreement rate: ____
- [ ] `data/probe/probe.jsonl` committed, schema-valid

## Person B — Harness

- [ ] Harness runs on real models (multilingual-e5, LaBSE) with baseline
- [ ] Cosine gap + NevIR-style score implemented
- [ ] First Hebrew STS wiring

## Sync point

- [ ] Run B's harness on A's real probe → first baseline numbers
- [ ] Email the numbers to David (check-in)

## Done when

We can state, with real models on ~100 pairs, how small the paraphrase-vs-negation gap is.
