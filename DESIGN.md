# RST-MoE Design Overview

This document summarizes the current end-to-end design for the cross-sectional stock ranking system (CSI300 / CSI800), including data flow, model, losses, macro features, and diagnostics.

## Goals & Tasks
- Primary task: daily cross-sectional ranking/selection; labels typically `Ref($close,-5) / Ref($close,-1) - 1` (t+5) but configurable.
- Benchmarks: MASTER and later SOTA; segments aligned to Qlib official splits.
- Stability across regime shifts and horizons; explainable router (time vs factor).

## Data & Sampling
- Dataset: `TSDatasetH` with `step_len` auto-matched to model `context_len`.
- Processors:
  - `infer_processors`: feature z-score + fillna.
  - `learn_processors`: `DropnaLabel`, `CSRankNorm` on label (**train uses rank-label; eval uses raw label**).
- Data keys: Train uses `DK_L`; Valid/Test/Predict use `DK_I`.
- Samplers:
  - Train: `FixedDailyBatchSampler` (per-day sampling; down/up-sample to batch_size; deterministic seed).
  - Valid/Test/Predict: `DailyChunkBatchSampler` (full per-day coverage, chunked if too many stocks).
- Macro state lookup: optional `market_state_path`, `market_state_shift`, `market_state_strict` (strict raises on missing/NaN).

## Model (QuantMoEModel)
- Embedding: value projection + factor ID embedding; dropout.
- Optional feature selector (STG-style):
  - Params: `use_feature_selection`, `selection_temperature`, `selection_noise_std`, `selection_reg_lambda`.
- Regime encoder:
  - External macro: `use_external_macro=True`, `d_macro_input` auto-set by adapter.
  - Internal stats: market-level (batch stats) or per-sample fallback; modes `regime_internal_mode={short,long}`, `regime_internal_lag`, `regime_internal_use_batch_stats`, `regime_internal_tail_threshold`.
- Router (per layer):
  - Inputs: regime embedding (+ optional `router_use_layer_summary` from layer hidden mean/std).
  - Options: `router_noise` (logit noise, train only), `router_temperature`, `router_z_loss_coef`.
  - Experts: time expert (per-factor temporal) + factor expert (per-time cross-sectional); ALiBi optional (`use_alibi`).
- Pooling & head: AdaptivePooling (`pooling_alpha` mixes attention/mean) on last time step, then linear head → stock score.

## Losses & Metrics
- Main: ListMLE (`listmle_tau` temperature, clamped to avoid NaN).
- Aux losses (directly controlled by coefficients, not via `loss_weights`):
  - Router z-loss: `router_z_loss_coef` (default 0.01) prevents router collapse.
  - Feature selection sparsity: `selection_reg_lambda` (default 1e-5) encourages sparse selection.
- Optional losses (via `loss_weights`): RankNet top/bottom (`rank_topk`), Huber (`huber_delta`).
- Monitoring (adapter):
  - Train (batch subset): `ic_pearson_batch`, `rank_ic_batch` (aliases `ic_raw/rank_ic` during train).
  - Valid/Test (full daily cross-section): `ic_pearson_daily`, `rank_ic_daily` (aliases `ic_raw/rank_ic` during eval); early-stop uses `rank_ic_daily`.
  - Model metrics: gate entropy/time_ratio, active_feat_ratio.

## Macro Features (Precomputed Market State)
- Generator: `scripts/precompute_market_state.py`.
- Features per date:
  - Global scalars: mean_abs/std/breadth/tail_2sigma.
  - Correlation/crowding: mean abs corr, Fro norm, PC1 ratio.
  - Per-factor cross-sectional stats (mean/std/breadth) → PCA (`pca_dim`).
  - Δstate (past-only): `--state_delta_lags`.
  - Market TS (past-only if chosen): return/vol/momentum/drawdown from benchmark close (`--add_market_ts`, `--market_ts_windows`).
  - Rolling/z-score (past-only): `--roll_mean`, `--zscore_windows`.
  - Filters: robust z, trade/suspension, weighting; optional `--no_norm` to skip feature z-score.
- Warmup: `--warmup_trading_days` (default auto) extends precompute start earlier so rolling/zscore/Δstate/TS are defined at training start.
- Dates normalized (no tz); strict lookup can raise on missing/NaN.
- Adapter auto-sets `d_macro_input = n_columns` of state file.

## Visuals & Reports
- `export_visuals`: gate time_ratio series + attention maps (time/factor) stored in recorder and optional PNGs.
- Official Qlib graphs: exported via `export_qlib_official_graphs` (raises if inputs missing), embedded into `kdd_report.md`.
- Report: `generate_paper_report` summarizes train curves, IC/RankIC (HAC t-stat), backtest metrics, gate stats, attention summaries; saves `run_conf` for reproducibility.

## Known Caveats / Residual Risks
- Train sampler is sampling; internal regime or layer summaries reflect sampled subset, not full day.
- Label mismatch: train uses rank-label (DK_L), eval/report uses raw label (DK_I); must be stated in paper.
- Rolling/zscore/Δstate/TS introduce early NaNs; with `market_state_strict=True` you need sufficient warmup coverage or disable strict.
- Feature selection reg uses `z.sum(dim=-1).mean()` (≈ avg selected features per sample); tune `selection_reg_lambda` accordingly (typical `1e-5~1e-4`).
