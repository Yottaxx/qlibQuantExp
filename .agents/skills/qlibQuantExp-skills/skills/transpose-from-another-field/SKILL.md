---
name: qx-transpose-from-another-field
description: Use when stuck on a quant research direction, when seeking a fresh angle on a familiar problem, or when the team's existing vocabulary feels exhausted. Imports mechanisms from physics, biology, computer science, ecology, control theory, or economics into the current quant problem via Gentner structure-mapping — map RELATIONS not attributes, prefer higher-order systematicity. Produces 3–5 candidate base→target analogies with explicit mapping tables and one falsifiable derived prediction per analogy. Triggers; "is there an analog of X", "we need a fresh angle", "why does Y behave like that", "we've used the same metaphor too long". For testing claims you already have, use qx-forge-a-sharp-falsifier. For raw observation without an analog, use qx-notebook-the-anomaly.
license: MIT
---

# Transpose from Another Field

## Overview

This skill imports mechanisms from outside fields into the current quant problem. The output is a small set of analogies, each with an explicit mapping table and one prediction sharp enough to fail. We do not collect resemblances; we transport relational structure.

## Posture

- Equal thought partner, not instructor.
- Prefer **relations** over object-level similarities. "Both have peaks" is not an analogy; "in both, peaks coincide with susceptibility divergence" is.
- Bias toward higher-order systematicity (Gentner): the more interconnected the mapped relations, the stronger the analogy.
- Be willing to reject your own analogies inside the same turn. Most candidates do not survive.

## Quick Reference

| Phase | What we do | What we write |
|---|---|---|
| 1. Gather target structure | Articulate the qlibQuantExp phenomenon as a system of relations, not a description | one paragraph |
| 2. Search for base candidates | Propose 5–8 candidate fields/mechanisms that share the target's relational shape | bulleted list |
| 3. Build mapping tables | For each surviving candidate, write `Base → Target` for every relation | one table per candidate |
| 4. Derive predictions | From each mapping, pull one specific testable prediction in qlibQuantExp's vocabulary | one prediction per analogy |
| 5. Rank by systematicity | Which mapping carries the most interlinked relations? Which dies on first contact? | a short verdict block |

Detailed procedure in `references/structure_mapping_protocol.md`. Worked examples in `references/seed_analogies.md`.

## Dialogue flow

**Phase 1 — Gather target structure.** Ask the user (or read from `shared/notebook/`): what is the puzzling phenomenon, stated as relations? Examples of target structures in this project: "router gating entropy drops in 2018Q3", "expert cosine similarity is near 1 in layers 3–4", "RankIC declines on high-vol days". Resist describing the phenomenon — articulate the *relations between its parts*.

**Phase 2 — Search for base candidates.** Sweep across at least 4 distant fields. Useful default pools: statistical mechanics (phase transitions, criticality, percolation), ecology (niche partitioning, predator-prey, community assembly), distributed systems (load balancing, leader election, gossip protocols), neuroscience (efficient coding, predictive coding, attractor dynamics), control theory (Kalman filters, robust control, MPC), economics (auction theory, mechanism design, market microstructure). Reject candidates that share only surface features.

**Phase 3 — Build mapping tables.** For each surviving candidate, build:

```
| Base concept           | Target (qlib) concept         | Mapped relation                    |
|------------------------|-------------------------------|------------------------------------|
| order parameter        | PC1 ratio of factor returns   | grows discontinuously near crisis  |
| susceptibility         | RankIC standard error         | diverges at the transition         |
| ...                    | ...                           | ...                                |
```

A good table has at least 3 rows where the **third column** (mapped relation) is interesting, not just "both are big numbers".

**Phase 4 — Derive predictions.** Each mapping must produce one prediction in the qlibQuantExp setting that we could actually test. Not "this is interesting" — "PC1 ratio should spike *before* RankIC degrades on a 5-day lag, not after". Specificity is the test of whether you carried the analogy or just the words.

**Phase 5 — Rank.** Apply Gentner's systematicity principle: prefer the mapping whose relations interlock. A mapping with 5 disconnected one-off correspondences is weaker than one with 3 deeply linked relations. Reject monster-barring rescues ("if it doesn't hold for big caps, that's because they're different from small caps") — they kill the analogy.

## Output

Write one file per surviving analogy to `shared/analogies/<concept-name>.md` using `templates/analogy_card.md`. If no analogy survives Phase 5 honestly, write nothing — a rejected analogy is also a research outcome, log it in `shared/notebook/` instead.

## Posture line (always cite at the start of dialogue)

> "An analogy is a mapping of knowledge from one domain (the base) into another (the target). What is mapped are *relations*, particularly higher-order relational structure." — Gentner (1983)

## See also

- `qx-forge-a-sharp-falsifier` — when an analogy's prediction is promising, convert it into a pre-registered prohibition.
- `qx-notebook-the-anomaly` — anomalies are the raw material for Phase 1.
- `qx-protect-or-relax-the-hard-core` — if an analogy would force us to redefine the project's core commitments, flag it there before writing it down.
- `shared/lineages.md` §1 Gentner — the canonical reference.
