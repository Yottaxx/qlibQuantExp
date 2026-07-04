# Structure-Mapping Protocol (Gentner)

The procedural skeleton for the `transpose-from-another-field` skill. Use this when the SKILL.md body's quick-reference is not enough.

## The systematicity principle

> "A predicate is more likely to be imported into the target if it belongs to a system of relations governing higher-order constraining relations." — Gentner (1983)

Translation: prefer mappings where the imported relations *interlock* — where R1 implies R2 implies R3 in the base, and the same chain must hold in the target. A bundle of disconnected one-off resemblances is *not* an analogy in Gentner's sense.

## Procedure

### Step 1. Articulate target structure

Write the target phenomenon as a system of **relations**, not descriptions.

| Weak (descriptive) | Strong (relational) |
|---|---|
| "Markets crash sometimes." | "An order quantity (PC1 ratio) responds to a slow control parameter, with discontinuous jumps at critical thresholds and divergent fluctuations near those thresholds." |
| "Routing collapses." | "Two parallel pathways with shared input compete for a finite output budget; the gate distribution loses entropy when one pathway's gradient signal dominates by a factor X." |

The relational restatement is where 80% of the work is. If you can't restate the target as relations, the analogy will not be productive.

### Step 2. Generate base candidates from distant fields

For each target, sweep **at least 4 distant fields**. Default pools:

- **Statistical mechanics** — phase transitions, criticality, percolation, renormalization-group flow, spin glasses.
- **Ecology** — niche partitioning, predator-prey dynamics, community assembly, succession, keystone species.
- **Distributed systems** — load balancing, leader election, gossip protocols, eventual consistency, CAP trade-offs.
- **Neuroscience** — efficient coding, predictive coding, attractor dynamics, sparse coding, divisive normalization.
- **Control theory** — Kalman filtering, robust control, MPC, observability/controllability, gain scheduling.
- **Microeconomics** — auction theory, mechanism design, signalling, repeated games, market microstructure.
- **Evolutionary biology** — fitness landscapes, drift, selection, speciation, red-queen dynamics.

Reject candidates that share only surface features. A "loss landscape" candidate from ML is suspect when explaining markets — same word, different relational shape.

### Step 3. Build the mapping table

For each surviving candidate:

```
| Base concept | Target concept | Mapped relation |
|--------------|----------------|-----------------|
| ...          | ...            | (the third column is the test) |
```

The third column is where Gentner's discipline lives. If the mapped relation is "both are large numbers" or "both decrease over time", it's not a relation, it's a surface attribute. Strong relations: "A causes B in both", "the time-derivative of A constrains the magnitude of B in both", "A and C jointly determine B in both".

A good table has **at least 3 rows** where the third column is non-trivial.

### Step 4. Derive predictions

The point of the analogy is to **export predictions** from base into target. For each mapping table, produce **one** prediction in target vocabulary that:

- Is specific (metric, window, threshold)
- Could be wrong (so it's worth testing)
- Would not have been generated without the analogy

If you cannot produce such a prediction, the analogy is decorative — discard it.

### Step 5. Rank by systematicity

Compare surviving analogies. Prefer:

1. Mappings whose relations are **mutually constraining** (R1 → R2 → R3, where breaking R1 in target also breaks R2 and R3).
2. Mappings that produce **specific, non-obvious** target predictions.
3. Mappings whose **base field has mature mathematical tooling** that can be imported (e.g., physics provides scaling exponents; ecology provides Lotka-Volterra equations).

Demote (do not erase) analogies that pass Gentner's systematicity test weakly. Even weak analogies sometimes generate useful predictions; archive them.

## Anti-patterns

- **Surface-keyword matching.** "Both have 'regime' in the name" is not structural overlap.
- **One-way mapping with no return prediction.** Analogies are valuable when they let you test base claims in the target; if no target experiment is suggested, the analogy is rhetoric.
- **Force-fitting universality.** Just because physics has universality classes does not mean every domain does. Be willing to say "Pred-3 (universality) is exploratory, not central".

## See also

- The seed example: `shared/analogies/regime-as-phase-transition.md`
- Gentner's original paper: Gentner, D. (1983). "Structure-mapping: A theoretical framework for analogy." *Cognitive Science*, 7, 155–170.
- Hofstadter & Sander, *Surfaces and Essences* (2013) — the popular treatment of analogy-as-core-of-cognition.
