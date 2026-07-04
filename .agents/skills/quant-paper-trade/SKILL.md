---
name: quant-paper-trade
description: Stand up a daily next-day inference loop using the latest promoted MLflow model — pulls latest data via Qlib, computes scores, applies the risk-overlay portfolio constructor, persists daily orders.csv and a paper P&L ledger. Detects feature drift and model staleness via rolling rank_ic on realized labels. Invoke only after /quant-walk-forward and /quant-stress have promoted the candidate. Stops at orders-on-disk; does NOT connect to a broker.
---

# /quant-paper-trade

The bridge to live, minus the trust-boundary step of order routing. Everything from "data pulled today" through "orders that *would* be placed" is automated; placing them is a separate, human-gated step.

## When to use

- A model has cleared `/quant-walk-forward` AND `/quant-stress` with verdict `promote`.
- User wants to start a paper-trade record before live deployment.
- Daily, as a scheduled task once the loop is stable (cron / Windows Task Scheduler).

## What to read first

1. `module/utils/market_state.py` — for live macro-state lookup (must extend the past-only feature set with the latest day's value).
2. The MLflow URI for the promoted run (from `project_state.md` or the latest `walkfwd`/`risk` ledger row's `mlflow_run`).
3. `module/utils/risk_overlay.py` (created by `/quant-stress`) — the constrained portfolio constructor.
4. Qlib's `D.features()` API for fetching latest-day features.

## Algorithm (one daily tick)

1. **Date & calendar check**: today's CN trading date via Qlib's calendar; if not a trading day, exit silently.
2. **Feature extraction**: `D.features(instruments="csi300", fields=alpha158_fields, start=today-step_len*2, end=today)`. Apply the same `infer_processors` (RobustZScoreNorm + Fillna) used at training — load the fitted state from MLflow.
3. **Macro state**: append today's macro state via `precompute_market_state.py`-equivalent computation on the latest day; ensure past-only flags are honored (no NaN-fill from future).
4. **Inference**: load the model from MLflow (`mlflow.pytorch.load_model`); run forward; obtain a score vector over CSI300 instruments.
5. **Portfolio construction**: pass scores through the risk-overlay constructor (sector-neutral + per-stock cap, exactly as cleared in `/quant-stress`). Output: target weights.
6. **Order generation**: diff target weights vs prior day's target (or vs zero on day-1) → `orders.csv` with columns `instrument, target_weight, prior_weight, delta_weight, side(buy/sell)`.
7. **Persist**: `paper_trade/<YYYY-MM-DD>/orders.csv`, `paper_trade/<YYYY-MM-DD>/scores.csv`, `paper_trade/<YYYY-MM-DD>/state.json` (the model checksum, MLflow run id, macro state file fingerprint).
8. **P&L update**: read yesterday's `orders.csv`, fetch yesterday's intended-day close → today close, compute realized return on the held weights net of the cost grid's central cell (5bp/5bp default). Append to `paper_trade/pnl_ledger.csv`.
9. **Drift / staleness checks** (run after T0 + 20 trading days, when realized labels start landing):
   - **Realized rank_ic**: rolling 20-day cross-sectional rank IC of yesterday's score vs realized t+5 return. Threshold: alert when 20-day rolling rank_ic drops below 0.7 × `walkfwd` mean.
   - **Feature drift**: KS-statistic of each input feature's today distribution vs train distribution. Alert on top-3 features exceeding KS > 0.2.
   - **Macro drift**: `market_state` PCA components for today, distance to nearest training-day in PC1/PC2 space. Alert on regimes outside training envelope.
10. Append a ledger row (`stage="paper"`, `verdict="info"`, `metrics={"realized_rank_ic_20d": ..., "ks_top1": ..., "regime_distance": ...}`, `notes` = drift alerts if any).
11. **Hard stop conditions** — emit a STALE flag (loud, top of stdout):
    - Realized rank_ic 20d < 0 for 5 consecutive days, OR
    - Feature drift KS > 0.4 on any input feature, OR
    - Regime distance > 99th percentile of train.

## What this skill does NOT do

- Connect to a broker.
- Place real orders.
- Manage real cash, positions, or margin.
- Compute performance fees, taxes, or wash-sale rules.
- Train or fine-tune anything (it is inference-only).

These are intentionally separate — the trust boundary is at `orders.csv → broker`, and crossing it must be a human, audited step (or a separate, security-reviewed skill).

## Output contract

```
paper_trade/
  YYYY-MM-DD/
    orders.csv
    scores.csv
    state.json
  pnl_ledger.csv         # appended each day
  drift_log.csv          # appended each day with KS / rank_ic / regime distance
```

Plus a daily ledger row.

## Failure modes / loop-back

- **STALE flag** → halt the daily loop; loop back to `/quant-walk-forward` to retrain on data including the recent regime, or to `/quant-hypothesis` to investigate the regime that broke the model.
- **Macro state computation fails for today** (missing benchmark close, etc.) → emit a SETUP-ALERT, do not produce orders for the day.
- **Feature schema mismatch** between training and live (Alpha158 handler version drift) → halt and require re-pinning of the feature handler.
