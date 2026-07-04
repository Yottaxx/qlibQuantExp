---
name: quant-walk-forward
description: Run a walk-forward validation by rolling train/valid/test windows quarterly across the full data range. Reports per-window rank_ic, IR, and a decay slope. Reuses QIB_DATA_OVERRIDES_JSON to override segments without modifying work_flow.py. Invoke before any production claim — single-split test results are not enough to bet real money on.
---

# /quant-walk-forward

The missing alpha-decay rung. Single-split test results are paper-grade only; live trading lives or dies by how the model holds up across rolling windows.

## When to use

- A candidate has cleared `/quant-ablation-plan` and `/quant-regime-diagnose`.
- Before any cost-stress, risk-overlay, or paper-trade work.
- When the user asks "is this alpha decaying?" or "would this have worked in 2018, 2019, 2020, 2021, 2022?"

## What to read first

1. `work_flow.py:2568-2620` — `QIB_DATA_OVERRIDES_JSON` accepts `segments` to override train/valid/test.
2. The default split in `work_flow.py:68-150` (train 2008-01..2014-12, valid 2015-01..2016-12, test 2017-01..2020-03 in current config; check at read time).
3. `scripts/aggregate_kdd_runs.py` — the per-window aggregator.
4. The current best config (from `runlog.load_project_state()`).

## Window plan (default, configurable per card)

Roll the test window forward by 1 quarter at a time, keeping the train/valid window sizes fixed (rolling-origin), or growing the train window (expanding-origin). Default: **rolling-origin, 6 windows**.

| Window | Train | Valid | Test |
|---|---|---|---|
| W1 | 2014Q1–2018Q4 | 2019Q1–Q2 | 2019Q3–Q4 |
| W2 | 2014Q3–2019Q2 | 2019Q3–Q4 | 2020Q1–Q2 |
| W3 | 2015Q1–2019Q4 | 2020Q1–Q2 | 2020Q3–Q4 |
| W4 | 2015Q3–2020Q2 | 2020Q3–Q4 | 2021Q1–Q2 |
| W5 | 2016Q1–2020Q4 | 2021Q1–Q2 | 2021Q3–Q4 |
| W6 | 2016Q3–2021Q2 | 2021Q3–Q4 | 2022Q1–Q2 |

(Adjust based on actual data range; market_state warmup must extend before each train start.)

## Algorithm

1. Read current best config (the candidate to validate) and the baseline (the prior best, for delta computation).
2. Generate N `QIB_DATA_OVERRIDES_JSON` payloads — one per window, with the `segments` field overridden.
3. **Single-GPU optimization**: launch sequentially, allow warm-start from the previous window's checkpoint (off by default; user opt-in via `--warm-start`). Reduce per-window epochs to 25 (vs full 40) since each window is more data + warm-start.
4. For each window: launch `python work_flow.py` with the override + a window-tagged `QIB_RUN_SETTING=walkfwd_<card_id>_W<n>`.
5. After all windows complete, collect per-window metrics:
   - `rank_ic_daily_mean`, `rank_ic_daily_t_hac`, `ic_pearson_daily_mean`
   - `annualized_return_with_cost`, `information_ratio_with_cost`, `max_drawdown`
   - From `/quant-regime-diagnose`: `time_ratio_mean`, `collapse_ratio`, `pc1_tail_2x2_rank_ic_spread`
6. Compute **decay diagnostics**:
   - `decay_slope` = Spearman(`rank_ic_window`, `window_index`). Negative slope = decay.
   - `worst_window_rank_ic` and `worst_window_index` (regime where the alpha breaks).
   - `regime_consistency` = std(`pc1_tail_2x2_rank_ic_spread`) across windows.
   - Same diagnostics on baseline to compute *delta* slopes.
7. **Verdict rules**:
   - `decay_slope_candidate > -0.3` (not strongly monotone-decaying) AND
   - `mean(rank_ic) across windows > baseline mean(rank_ic) across windows + 0.004` AND
   - `worst_window_rank_ic > baseline worst_window_rank_ic − 0.005` (no catastrophic regime)
   → `promote`. Else `reject` with the failing rule named.
8. Append ledger row (`stage="walkfwd"`, full per-window metrics + decay slope in `extra`).
9. On `promote`, update `project_state.md` with the walk-forward-confirmed metrics, replacing the previous "single-split" entry.

## Decay-monitor mode (lightweight)

For a model already in production, run only **one new window** per quarter against fixed prior windows — checks if the live model's alpha is decaying without retraining the whole sweep. This is the cheapest production maintenance and should be cron-able later.

## Output contract

- Per-window CSV (`diagnostic_runs/walkfwd_<card_id>/per_window.csv`).
- Decay table (markdown, embed in ledger `notes`).
- Verdict.
- Ledger row.

## Failure modes / loop-back

- **Insufficient data range** for 6 windows → reduce to 4, document.
- **Catastrophic worst-window** (rank_ic < 0 in any window) → reject; loop back to `/quant-hypothesis` with that regime as the falsifier focus.
- **Decay slope < -0.5** → alpha is decaying fast; production lifetime is months not years; flag prominently and note in `project_state.md` failure modes.
