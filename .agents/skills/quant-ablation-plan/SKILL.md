---
name: quant-ablation-plan
description: Build a budget-aware multi-seed ablation sweep from a search spec. Invoke after /quant-minimal-repro returns "promote" or "inconclusive". Emits a run_matrix.yaml block plus the launch command and the aggregate-runs invocation; reuses scripts/run_diagnostic_experiments.py and scripts/run_four_day_experiment_plan.py.
---

# /quant-ablation-plan

Turns a search spec + wall-clock budget into a concrete sweep that fits the budget, with explicit promotion gates already used by this project.

## When to use

- After `/quant-minimal-repro` clears (verdict ∈ {promote, inconclusive}).
- When the user asks for "a proper ablation" or "multi-seed confirm" of a card.
- Never as the first step — always after a proxy.

## What to read first

1. `run_matrix.yaml` — schema (`seeds`, `universes`, `defaults`, `run_groups`).
2. `scripts/run_diagnostic_experiments.py:30-411` — the `SETTINGS` dict structure and the env-override invocation pattern.
3. `scripts/run_four_day_experiment_plan.py:32-144` — multi-stage selection rules (the promotion thresholds — `rankic_delta ≥ 0.6%`, `≥2/3 seeds above baseline`).
4. `scripts/aggregate_kdd_runs.py` — the cross-seed aggregator schema.
5. The latest proxy ledger row to inherit the config delta.

## Algorithm

1. **Inputs**: from the user (or default to single-axis sweep around the card's delta):
   - Search grid: `{<param>: [v1, v2, v3], ...}` (cross-product or one-at-a-time).
   - Seed count (default 3 on single GPU, 5 if budget allows).
   - Wall-clock budget (hours).
2. **Estimate per-run wall-clock** from the most recent 5 full-length runs in the ledger (`stage="ablation"`); fall back to ~2.5 h/run if no history.
3. **Trim the grid** to fit budget = `runs × seeds × per_run_hours`. Trimming priority (drop lowest first):
   1. Outer-grid points furthest from the card's central value.
   2. Highest-seed runs first (cap at 3 seeds before dropping grid points).
   3. **Never** drop the baseline (current best config) — it must be in the sweep so deltas are computed against the *same-window* baseline.
   Warn the user if trimming materially shrinks coverage (>30% of original cross-product).
4. **Emit a `run_matrix.yaml` patch** under a new `run_group` named `ablation_<card_id>`. Each entry has `name`, `setting` (mapping to a `SETTINGS` key in `run_diagnostic_experiments.py`), `overrides`, `seeds`, and `run_ids: {seed: null}`.
5. **Emit the launch command** — one of:
   - **For ablation-only (single stage)**: `python scripts/run_diagnostic_experiments.py --plan <yaml-patch> --output diagnostic_runs/ablation_<card_id>` (mirroring the existing CLI flags).
   - **For staged confirm (recommended for promotion-eligible cards)**: `python scripts/run_four_day_experiment_plan.py` with the appropriate stage selection, since it already implements the promotion-gate logic from `kdd_experiments_plan.md`.
6. **After the sweep finishes**, run `python scripts/aggregate_kdd_runs.py --experiment <name> --seeds <list>` to produce the cross-seed table.
7. **Apply promotion gate** (the project's existing rule, restated):
   - `rank_ic_daily_mean − baseline_rank_ic_daily_mean ≥ 0.006` AND
   - `≥ ceil(seeds × 0.6)` seeds above baseline AND
   - `time_ratio` not collapsed in any seed (collapse_ratio < 0.3 in all).
8. **Append a ledger row** (`stage="ablation"`, `verdict ∈ {promote, reject, inconclusive}`, `metrics` = aggregate dict, `mlflow_run` = experiment name).
9. **On promote** → call `runlog.update_project_state(...)` with the new best config and metrics, *only if* it beats the prior best.

## Trimming heuristic example

Budget 24 h, per-run 2 h, requested grid 3×3×3 = 27 configs × 5 seeds = 270 h.
- First trim: 5 → 3 seeds (162 h).
- Second trim: 3×3×3 → 3×3 (drop the least-informative axis, ask user which) → 54 h.
- Third trim: drop two off-axis points → 3×2 + baseline = 7 configs × 3 seeds = 42 h.
- Final: 3×2 + baseline = 7 × 3 = 21 h ≤ budget. ✅

Document the trims in the ledger `notes` field so the trade-off is auditable.

## Output contract

- A `run_matrix.yaml` patch printed to stdout (user diff-applies).
- The launch command.
- The aggregator command.
- After run: aggregate metrics + verdict.
- Ledger row.

## Failure modes / loop-back

- **Budget < 1 full sweep** → suggest reducing grid further, or recommend `/quant-minimal-repro` with multiple seeds instead of a full ablation.
- **Aggregate shows `time_ratio` collapse on any seed** → reject + loop to `/quant-hypothesis`.
- **Promotion gate barely missed (delta in [0.4%, 0.6%])** → mark `inconclusive`, suggest extending to 5 seeds before re-evaluating.
