---
name: qx-protect-or-relax-the-hard-core
description: Use when a proposed change would touch the project's THEORETICAL CORE rather than the protective belt around it. Distinguishes progressive moves (positive heuristic — add predictive content) from degenerating moves (ad-hoc rescue — preserve the core by carving exceptions). Maintains a living manifesto of the hard core (3–6 commitments the project refuses to sacrifice), the protective belt (auxiliary hypotheses that may be modified), and the monster-barring red-flag log (every time a result was excused by redefining the eval set). Triggers; any proposal that would shift router design, expert architecture, or the structural-routing narrative; "should we change X?", "I want to redefine Y", "the result was bad but…". Pairs with qx-forge-a-sharp-falsifier when the falsifier touches a core commitment. Append decisions to qx-research-ledger.
license: MIT
---

# Protect or Relax the Hard Core

## Overview

A research programme survives by knowing what it refuses to give up. The hard core is the set of commitments that, if surrendered, mean the programme has become a different programme. The protective belt is everything else — fair game for modification.

This skill maintains `shared/hardcore.md` as the project's manifesto. Every proposed change passes through one classification: **progressive** (modifies belt, adds predictive content), **degenerating** (rescues core via ad-hoc fix, loses predictive content), or **monster-barring** (excuses a failed result by redefining what counts as an instance) — the worst.

## Posture

- Be explicit about what is *not* up for discussion this week. Without this, every result becomes a referendum on the whole research direction.
- Recognize that touching the hard core is sometimes the right move — but it must be **conscious**, and it means we are starting a new programme, not patching the old one.
- The monster-barring red flag is the most important diagnostic. macV3's failure was exactly this pattern: many switches added to defend a hypothesis, without producing novel testable predictions.

## Quick Reference

| Proposed move | Likely classification | Recommended action |
|---|---|---|
| Modify a specific architectural detail (FiLM scale, pooling alpha) | **progressive** | proceed with falsifier |
| Add a new component that increases predictive content | **progressive** | proceed; verify falsifier names what it forbids |
| Re-segment evaluation to exclude where the model fails | **monster-barring** 🚩 | reject or flag in red-flag log |
| Redefine the success metric after seeing the result | **monster-barring** 🚩 | reject |
| Bundle 15k+ insertions defending the same hypothesis with no new predictions | **degenerating** | split into smaller progressive moves |
| Replace soft routing with hard routing | **paradigm shift** — touches hard core | requires explicit core amendment + new manifesto |

## Dialogue flow

**Phase 1 — Identify what's being changed.** Restate the proposal in concrete terms: which file, which mechanism, which decision is being touched.

**Phase 2 — Locate it: core vs belt.** Read `shared/hardcore.md`. Does the change touch a hard-core item, a protective-belt item, or something not listed? If not listed, ask the user where it should live — and amend the manifesto if needed (this is itself a recordable decision).

**Phase 3 — Classify the move.** Walk through Lakatos's distinctions:
- Does the move **add** predictive content (new predictions, new domains, new precision)? → **progressive**.
- Does the move **subtract** predictive content (carve out a regime, redefine an exclusion, narrow scope to where the theory works)? → **degenerating**.
- Does the move excuse a *specific* failed result by redefining what counts (e.g., "the bull regime is special; we should evaluate excluding it")? → **monster-barring**, log it.

**Phase 4 — Commit or revise.** Three outcomes:
1. **Progressive belt amendment.** Update `shared/hardcore.md` if the belt structure changes (rare; the belt is implicit in the code). Append a `decision` row to the ledger. Proceed with `qx-forge-a-sharp-falsifier`.
2. **Hard-core amendment.** This is a paradigm shift. Write a new section in `shared/hardcore.md` titled `## Core amendment YYYY-MM-DD` documenting (a) what is being removed from the core, (b) what is replacing it, (c) why the old core could not be saved. Append a high-importance `decision` row to the ledger.
3. **Monster-barring rejection.** Refuse the move. Write a `### Red flag YYYY-MM-DD` entry in `shared/hardcore.md` documenting the proposed rescue, *who* proposed it, and *why* it was rejected. This entry is a permanent reminder.

## Output

Mutate `shared/hardcore.md` (append-only — never delete prior amendments or red flags). Trigger a `decision` entry in `shared/ledger.md` via `qx-research-ledger`.

## Specific to this project

The hard core for qlibQuantExp (as of 2026-05-22) is in `shared/hardcore.md`. It includes commitments that came out of the night-plan dialectic (e.g., "regime-routing is soft mixture not hard switch"). Future amendments are likely; document each one.

## Posture line

> "Newton's third law … is the 'hard core' of the programme. … Newton's first three laws of dynamics and his law of gravitation … cannot be denied or modified within the programme without abandoning the programme as a whole." — Lakatos (1970)

> *On monster-barring:* "There are some who, when a counter-example threatens, simply refuse to admit that the example is an instance of the concept in question." — Lakatos, *Proofs and Refutations*

## See also

- `qx-forge-a-sharp-falsifier` — when the classification is "progressive", forge the falsifier before any code touches the branch.
- `qx-research-ledger` — every classification outcome is appended as a `decision` row.
- `qx-notebook-the-anomaly` — if a recent result triggered the proposal, the original anomaly entry should be cited.
- `shared/lineages.md` §4 Lakatos.
