---
name: "M2 — Interventions"
about: "Projection + NLI re-ranking with test-split numbers and STS trade-off (parallel)"
title: "M2: Interventions"
labels: ["milestone:M2"]
---

## Goal

Both interventions produce test-split numbers with STS trade-off reported.

## Person A — Projection

- [ ] Mean-difference direction
- [ ] Classifier direction
- [ ] Alpha sweep — fit on train, measure on test only
- [ ] Probe grown toward ~300 triples

## Person B — NLI re-ranking

- [ ] Hebrew NLI model on HebNLI
- [ ] Contradiction probability → similarity mapping
- [ ] Blend weight tuned on train only

## Done when

Both interventions report test-split cosine gap + NevIR-style rank, each with its STS trade-off.
