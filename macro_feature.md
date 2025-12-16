# Macro Features (Precomputed Market Daily State)

This repo supports using a **precomputed full-market daily state** as `macro_features` to drive the model’s `RegimeContextEncoder(use_external_macro=True)` and improve long-horizon settings (e.g., t+5).

The pipeline is:
1) Precompute `market_state_{universe}.pkl` (index = trading date, columns = state features)
2) Configure training with `trainer_config.market_state_path`
3) Adapter looks up each sample’s date → feeds the state vector as `macro_features` into `QuantMoEModel.forward`

---

## What Was Updated

- `scripts/precompute_market_state.py` upgraded from 4 naive scalars to a stronger daily-state design:
  - Per-day **global distribution** scalars (`mean_abs/std/breadth/tail_2sigma`)
  - **Correlation/crowding** scalars (`corr_mean_abs/corr_fro/corr_pc1_ratio`)
  - **Per-factor cross-sectional stats** (mean/std/breadth) → **PCA compression** to `k` dims (`market_state_pca_*`)
  - Optional **Δstate** features over multiple lags (past-only; `*_d1`, `*_d5`, ...)
  - Optional **benchmark market time-series** features (return/vol/momentum/drawdown)
  - Optional **past-only rolling z-score** features for stability across horizons (`*_z20`, `*_z60`, …)
  - Optional robust filtering and weighting (`--filter_*`, `--weight_field`)
  - Optional “raw-ish” feature mode without `RobustZScoreNorm` (`--no_norm`)
- `module/model_adapter.py` already supports passing `macro_features` everywhere (train/valid/predict/visuals) via:
  - `trainer_config.market_state_path`
  - `trainer_config.market_state_shift`
  - `trainer_config.market_state_strict`
- `module/utils/market_state.py` now treats **NaN/Inf** in the state as an error when `strict=True`.

---

## Generate Market State Files

This script reuses `work_flow.py:data_conf` and builds a separate dataset with `step_len=1` to get **per-day** features, then aggregates across the whole universe each day.

### CSI300 (example)

```bash
python scripts/precompute_market_state.py \
  --out market_state_csi300.pkl \
  --pca_dim 16 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --market_ts_past_only \
  --zscore_windows 20,60 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05
```

### CSI800 (example)

```bash
python scripts/precompute_market_state.py \
  --out market_state_csi800.pkl \
  --instruments csi800 \
  --pca_dim 32 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --market_ts_past_only \
  --zscore_windows 20,60,120 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05
```

Outputs:
- `market_state_*.pkl` (main state DataFrame)
- `market_state_*.pkl.pca.npz` (PCA mean/components sidecar for reproducibility)

---

## Script Parameters (scripts/precompute_market_state.py)

### Core I/O
- `--out`: output path, must end with `.pkl` / `.parquet` / `.csv`
- `--instruments`: override universe, e.g. `csi300` / `csi800`

### Use Raw-ish Features (skip RobustZScoreNorm)
- `--no_norm`: removes `RobustZScoreNorm` from `infer_processors` inside `work_flow.py:data_conf`
  - Use this when you want market state to reflect **absolute scale** (volatility regime) rather than normalized features.

### Robust Filtering (per day)
- `--filter_robust_z`: robust z-score threshold; `0` disables filtering
- `--filter_max_bad_frac`: a stock is removed if more than this fraction of factor dimensions are outliers
  - Intuition: drop “garbage” stocks/days with many extreme factor values.

### Suspension / Non-trading Filters (recommended for CSI800)
- `--trade_field`: qlib field used to detect non-trading rows (e.g. `'$volume'` or `'$amount'`)
- `--min_trade`: keep rows with `trade_field > min_trade` (set `>0` to drop 0-volume/0-amount days)
- `--suspend_field`: optional suspension flag field (keep rows with value `== 0`)
  - If your qlib data doesn’t have an explicit suspend flag, `--trade_field '$volume' --min_trade 0` is usually sufficient.

### Weighting (per stock)
- `--weight_field`: optional qlib field, e.g. `'$amount'` or `'$volume'`
  - Used to compute weighted per-factor mean/std/breadth and weighted global scalars.
  - Requires the qlib data provider to contain the requested field.

### PCA
- `--pca_dim`: PCA dimension for the compressed regime vector (`market_state_pca_0..k-1`)
  - Typical: `8~32`

### Cross-period Standardization (past-only)
- `--zscore_windows`: comma-separated windows, e.g. `20,60,120`
  - Generates `*_z{window}` columns using rolling mean/std **shifted by 1 day** (no look-ahead).
- `--roll_mean`: optional past-only rolling mean window; generates `*_roll_mean{W}` columns

### Δstate (past-only)
- `--state_delta_lags`: comma-separated lags, e.g. `1,5,10`
  - Generates `*_d{lag}` columns as `state_t - state_{t-lag}`.

### Market time-series features (temporal signal)
- `--add_market_ts`: append benchmark market TS features computed from the benchmark close series.
- `--market_index`: override benchmark instrument code (defaults: `csi300->SH000300`, `csi800->SH000906`).
- `--market_ts_windows`: comma-separated windows used for vol/mom/dd.
- `--market_ts_past_only`: shift TS features by 1 day (recommended, avoids look-ahead).

---

## Training Configuration

Enable macro features by adding these keys to `trainer_config`:

```python
"trainer_config": {
  "market_state_path": "market_state_csi300.pkl",
  "market_state_shift": 0,
  "market_state_strict": True,
}
```

Notes:
- `market_state_shift` shifts the entire state table by `k` days (within its index).
  - Use `1` if you want to be conservative about information availability (avoid any same-day leakage).
- `market_state_strict=True` makes missing dates (or NaN state rows) a hard error.
  - This is recommended for paper-quality experiments.

### Dimensionality alignment (model init)

- The model’s `d_macro_input` is set automatically to the number of columns in your saved `market_state` DataFrame.
- Turning on `--state_delta_lags`, `--add_market_ts`, `--roll_mean`, or `--zscore_windows` increases the column count; the adapter reads it and configures the model accordingly.
- **Important**: rolling/z-score/Δstate features introduce NaNs for the initial warmup period; with `market_state_strict=True` this will raise. Either:
  - set `market_state_strict=False`, or
  - ensure your experiment segments start after enough warmup days (or include earlier history in the state file).

### Warmup coverage (recommended)

`scripts/precompute_market_state.py` can automatically extend the **precompute** start backward by enough trading days
to make rolling/z-score/Δstate/market-TS features defined at your training start.

- Use `--warmup_trading_days N` to force a specific warmup length.
- Default (`--warmup_trading_days -1`) computes it from your selected windows/lags.

---

## Practical Advice (CSI800 / Long Horizon)

- CSI800 is larger and noisier: prefer `--weight_field '$amount'` and robust filtering.
- For t+5/t+20, always include `--zscore_windows` (e.g. `20,60,120`) to stabilize regimes across periods.
- If you keep `CSRankNorm` on labels, market-state features should carry more slow-moving information (correlation/crowding + PCA regime) to help the router distinguish regimes.
