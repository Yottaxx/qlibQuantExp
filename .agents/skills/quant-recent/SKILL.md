---
name: quant-recent
description: Read-only summary of the last K days of experiments_ledger.jsonl grouped by card_id, plus the current memory/project_state.md snapshot. Invoke at the start of a new session, after returning from a break, or whenever the user asks "where were we?", "what did we try recently?", "what's the current state?". No side effects, no GPU usage.
---

# /quant-recent

The "wake-up" skill. Run this first in any new session to orient.

## When to use

- Start of a new session.
- User asks "where were we?", "what's the current state?", "what did we try last week?", "summarize recent work".
- Before `/quant-hypothesis` to make sure a proposed idea isn't a recent reject.

## What to read first

1. `experiments_ledger.jsonl` (full file; the function is read-only).
2. `memory/project_state.md` (resolved via `module.utils.runlog.PROJECT_STATE_PATH`).

## Algorithm

1. Default lookback `K=14` days; user can pass any positive int.
2. Read the ledger via `runlog.iter_ledger()` (oldest-first iterator).
3. Filter rows with `ts >= now - K days`.
4. Group by `card_id`.
5. For each group, compute:
   - Hypothesis text (from the `stage="hypothesis"` row).
   - Latest `stage` reached (max in the project's stage order: `hypothesis < leakage < proxy < ablation < regime < walkfwd < cost < risk < paper`).
   - Latest `verdict` overall (last non-`info` verdict, or `info`).
   - Best `metrics.rank_ic` seen across the group's rows.
   - Total `wall_clock_s` summed across rows.
6. Sort groups by the latest timestamp in each (most recent first).
7. Print a markdown table:
   ```markdown
   | card_id | hypothesis | latest stage | verdict | best rank_ic | wall-clock |
   |---|---|---|---|---|---|
   ```
8. Append a "Current best" section by reading `runlog.load_project_state()` and printing it verbatim (or "no project_state.md yet" if missing).
9. Append a one-paragraph synthesis: how many cards opened, how many promoted, how many rejected, what's currently in flight (latest stage < `paper` and verdict ∈ {info, inconclusive, promote}).
10. Append a ledger row (`stage="recent"`, `verdict="info"`).

## Output contract

```markdown
# Recent (last 14 days)

## Cards
| card_id | hypothesis | latest stage | verdict | best rank_ic | wall-clock |
|---|---|---|---|---|---|
| h-20260503-001 | factor_gate_scale 0.5→0.7 | walkfwd | promote | 0.0801 | 14.2h |
| h-20260504-002 | switch loss to ListMLE | proxy | reject | 0.0712 | 0.4h |

## Current best (project_state.md)
<verbatim contents>

## Synthesis
3 cards opened, 1 promoted (h-20260503-001), 1 rejected (h-20260504-002), 1 in flight (h-20260505-001 at proxy stage). Active failure modes: gate collapses when router_temperature < 0.7.
```

## Failure modes / loop-back

- **Empty ledger** → say so, suggest `/quant-hypothesis` to start.
- **Stale `project_state.md`** (older than the latest `stage="walkfwd"` promote-row) → flag the staleness; the user should re-run `/quant-walk-forward` or trust the ledger over the project-state file.
- **Ledger corrupt / unparseable lines** → the iterator already skips bad lines; report how many were skipped.
