---
name: qx-research-ledger
description: Use whenever a TRACKED FACT happens; a baseline anchored, a sweep started or finished, an IC/RankIC measured, a commit blessed as a comparison point, a decision made — anything "this is what actually happened today". Also use at session entry — first action when returning to the project — to scan the last 10–20 entries and produce a one-paragraph state summary ("the frontier is X, pending sweeps are Y, last baseline was Z"). Strictly separated from speculation; hypotheses, surprises, and explanations go to qx-notebook-the-anomaly; analogies go to qx-transpose-from-another-field; classified architecture decisions go to qx-protect-or-relax-the-hard-core. The ledger is for what HAPPENED, not what it MEANS. Append-only; never edit prior rows. Triggers; "where did we leave off", "log this result", "what's the current baseline", "starting session", "sweep done", "commit this anchor", any factual record event.
license: MIT
---

# Research Ledger — Faraday's Diary + Index

## Overview

The ledger is the project's append-only factual record. It exists for one purpose: when a future-Claude or future-yotta returns to the project, they can read the last 10 rows and reconstruct the trajectory in one paragraph — without re-deriving rejected directions and without losing track of which sweep produced which IC.

The ledger has two modes: **Append** (one row per fact) and **Survey** (read the last N and summarize).

It is strictly **not** for speculation, explanation, or interpretation. Faraday himself kept his Diary (numbered experiments, dated, factual) separate from his Idea Books (raw speculation). The discipline is non-negotiable.

## Posture

- Facts only. No "I think X" or "this suggests Y". Those go to `qx-notebook-the-anomaly`.
- One row per event. Terse. ≤ 200 chars in the context field.
- Append-only. Never edit prior rows. If a prior fact was wrong, append a new row with `kind=correction` citing the original.
- Sequential. Each row gets the next integer id.

## Quick Reference — the 5 entry kinds

| `kind` | When | Required fields |
|---|---|---|
| `anchor` | A commit is blessed as a reproducible baseline | sha, branch, universe, metric values, anchored_at |
| `sweep_start` | A new sweep is dispatched | claim_id (cites a falsifier), settings, seeds, expected_done |
| `sweep_done` | A sweep finishes | claim_id, mlflow_run_ids, status (complete/partial/failed) |
| `result` | A measured metric is finalized | claim_id, metric_name, value, n_seeds, universe |
| `decision` | A judgment was made (adopt/reject/defer/classify) | what, why_short, cited_artifact (file path) |

Full schema in `references/ledger_schema.md`.

## Dialogue flow — Append mode

When a factual event happens, do this:

1. **Open** `shared/ledger.md`. Read the last row to get the next id.
2. **Classify** the event into one of the 5 kinds. If unclear, ask the user — if still unclear, the event is probably speculation; redirect to `qx-notebook-the-anomaly`.
3. **Fill the row.** Use `templates/ledger_entry.md`. Keep the context column ≤ 200 chars; longer narration belongs in the cited artifact.
4. **Append.** Add the row at the bottom of `shared/ledger.md` under today's date heading (create the heading if needed).
5. **No retro-editing.** Do not modify prior rows; if a correction is needed, append a new `kind=correction` row.

## Dialogue flow — Survey mode

At session start (or whenever user asks "where are we"):

1. **Read the last 10–20 rows** from `shared/ledger.md`.
2. **Synthesize** into one paragraph (≤ 120 words) covering:
   - The current `current` baseline (most recent unsuperseded `kind=anchor`).
   - Any open sweeps (`sweep_start` with no matching `sweep_done`).
   - The 2–3 most recent decisions and what they implied.
   - One sentence on "next reasonable action" based on the trajectory.
3. **Surface open questions.** If there's a `sweep_done` without a follow-on `result`, flag it. If there's an unanswered `kind=decision` proposal, flag it.

The survey paragraph is what the returning user reads first.

## Format of `shared/ledger.md`

A single markdown file with two sections per date:

```markdown
# Research Ledger

## 2026-05-22

| id | kind | key | context |
|----|------|-----|---------|
| 1 | anchor | sha=3675090 ic=0.0706 rank_ic=0.0786 | macV2 baseline frozen on csi300 t+5; see analysis/night_plan_master.md |
| 2 | decision | adopt research/macV2-night with Tier-1 | A1 4-arm + A3 EMA + D1-D4; see analysis/night_plan_change_set.md |
| 3 | decision | defer macV3 hierarchical state field | bundling too many switches; see analysis/macV2_vs_macV3_quant_algorithm_diff.md |

## 2026-05-23

| id | kind | key | context |
| ...
```

ids are globally sequential; date headings group rows by day.

## Cumulative trial counter

The ledger's `kind=sweep_done` entries are the project's cumulative trial count. Survey mode reports `n_trials` at the end of the summary paragraph. This is the future input to deflated-Sharpe-style multiple-testing accounting; **the ledger only counts and does not haircut** — the math is deferred.

## Posture line

> Faraday kept two strictly separate categories of notebooks: the *Diary* (sequentially numbered experiments with date and conditions) and the *Idea Books* (raw speculation). Diary entries that graduated to publication were marked with vertical lines.

> "Maintain its own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials." — Reflexion (Shinn et al., 2023)

## See also

- `qx-notebook-the-anomaly` — the Idea Book half of the system. Anything subjective or interpretive goes there.
- `qx-forge-a-sharp-falsifier` — falsifiers produce `sweep_start` and later `sweep_done` rows here.
- `qx-protect-or-relax-the-hard-core` — every classification outcome appends a `decision` row here.
- `shared/lineages.md` §3 Faraday (Diary + Index), §6 MLAgentBench / Reflexion.
