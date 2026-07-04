---
name: quant-stress
description: Production-rigor double-check — runs cost-stress (commission/slippage grid sensitivity) and risk-overlay (sector-neutral, per-stock cap, optional beta-target portfolio) on a candidate model's predictions. Invoke after /quant-walk-forward. Replaces the optimistic paper Sharpe with an honest "Sharpe under realistic frictions and risk constraints" number.
---

# /quant-stress

The two checks every quant firm runs before believing a backtest. Combined into one skill because they're cheap together and almost always run as a pair.

## When to use

- After `/quant-walk-forward` returns `promote` on the candidate.
- Before `/quant-paper-trade`.
- When user asks "would this work after costs?" or "what's the Sharpe with risk constraints?"

## What to read first

1. `work_flow.py:223-242` — current strategy and cost wiring (`TopkDropoutStrategy`, `topk=30`, `n_drop=5`, the disabled-by-default cost lines).
2. The Qlib `port_analysis_config` schema and `PortAnaRecord` API.
3. The latest walk-forward ledger row (the `mlflow_run` field tells you which run's predictions to score).
4. The CSI300 SW level-1 sector mapping — if not present in the project, the skill must surface this as a setup blocker (sector-neutral overlay needs sector membership data; check Qlib's `instruments` for sector labels, or fall back to industry derived from a static mapping).

## Two procedures

### A. Cost stress

Re-run the existing portfolio backtest under a grid of commission × slippage assumptions:

| | slippage 2bp | slippage 5bp | slippage 10bp |
|---|---|---|---|
| commission 2bp | (IR, Sharpe, MaxDD, Turnover) | … | … |
| commission 5bp | … | … | … |
| commission 10bp | … | … | … |

Grid is fast — predictions are reused; only the simulator re-runs. Report:
- Sharpe surface.
- **Break-even friction**: the (commission, slippage) at which Sharpe drops below 1.0 (or below baseline; user-configurable).
- Cost drag = mean(annualized_return_without_cost − annualized_return_with_cost) in bp.

### B. Risk overlay

Replace the long-only TopkDropout selection with a constrained portfolio constructor. Three layers (each off by default; turn on individually so the marginal cost of each constraint is visible):

1. **Sector neutrality**: weight per SW level-1 sector capped to `±2%` of CSI300's sector weights.
2. **Per-stock cap**: max position `2%` (or user-configurable).
3. **Beta target** (optional): solve for weights minimizing `||w·β − 1||₂` subject to sector + cap constraints.

Implementation: a small portfolio constructor function (call it `module/utils/risk_overlay.py`, ~150 lines) that takes the model's score vector + the constraints, solves a small QP/LP via `scipy.optimize.linprog` or `cvxpy`, and returns `weights`. Plug it into Qlib's strategy interface (or convert weights into `TopkDropout`-equivalent signals).

Re-run backtest under the same cost grid as procedure A. Report:
- Sharpe surface *with risk overlay* vs *without*.
- The constraint-cost: how much Sharpe is paid for each constraint added.
- Sector-exposure time-series (proves the constraint is binding correctly).

## Algorithm

1. Locate the candidate's predictions (`pred.pkl` artifact from MLflow).
2. **Cost-stress**: 9 backtests in the cost grid; collect metrics.
3. **Risk-overlay**: build the constrained portfolio; re-run 9 backtests; collect.
4. Produce a single comparison table:
   - Row: configuration (long-only / sector-neutral / sector+cap / sector+cap+beta).
   - Column: cost cell.
   - Cell: (Sharpe, IR, MaxDD, Turnover).
5. Verdict:
   - **Promote** if Sharpe > 1.0 at (5bp, 5bp) under the risk-overlay (sector+cap), AND drawdown < 25%, AND turnover stays sane (< 6× annualized, indicative).
   - Else **reject** with which constraint broke the model.
6. Append ledger rows: one for `stage="cost"`, one for `stage="risk"`, with the full surfaces in `extra`.
7. On `promote`, update `project_state.md` with the realistic-Sharpe metrics replacing the gross-Sharpe entries.

## Output contract

- Two CSV files under `diagnostic_runs/stress_<card_id>/`: `cost_grid.csv`, `risk_grid.csv`.
- Markdown summary printed.
- Two ledger rows.

## Failure modes / loop-back

- **Sector data missing** → first run a one-off setup task to ingest SW level-1 mapping for CSI300; the skill should NOT silently skip this constraint and pretend the overlay ran.
- **Sharpe collapses under sector-neutral** (e.g. drops from 2.0 paper → 0.4 sector-neutral) → the alpha is largely sector beta, not stock-picking; reject + loop to `/quant-hypothesis` with "decompose alpha into sector + stock-picking components" as the next investigation.
- **Cost-stress shows break-even at <3bp** → strategy is impractical for retail-scale execution; document and decide whether to push for higher-frequency / lower-turnover variant.
