# Macro Features (Precomputed Market Daily State)

This repo supports using a **precomputed full-market daily state** as `macro_features` to drive the model’s `RegimeContextEncoder(use_external_macro=True)` and improve long-horizon settings (e.g., t+5).

The pipeline is:
1) Precompute `market_state_{universe}.pkl` (index = trading date, columns = state features)
2) Configure training with `trainer_config.market_state_path`
3) Adapter looks up each sample’s date → feeds the state vector as `macro_features` into `QuantMoEModel.forward`

---

## What Was Updated

- `scripts/precompute_market_state.py` upgraded from 4 naive scalars to a stronger daily-state design:
  - Per-day **global distribution** scalars (`mean_abs/std/breadth/tail_2sigma`) — computed on day T
  - **Correlation/crowding** scalars (`corr_mean_abs/corr_fro/corr_pc1_ratio`) — computed on day T
  - **Per-factor cross-sectional stats** (mean/std/breadth) → **PCA compression** to `k` dims (`market_state_pca_*`) — computed on day T
  - Optional **Δstate** features over multiple lags (`*_d1`, `*_d5`, ...) — past-only by construction
  - Optional **benchmark market time-series** features (return/vol/momentum/drawdown) — up to day T by default
  - Optional **rolling z-score** features for stability across horizons (`*_z20`, `*_z60`, …) — up to day T by default
  - Optional robust filtering and weighting (`--filter_*`, `--weight_field`)
  - Optional “raw-ish” feature mode without `RobustZScoreNorm` (`--no_norm`)
  - Optional macro **profile slimming** (`--macro_profile=kaiming`) to keep only core stats + PCA + raw market TS
  - Optional macro **scale normalization** (`--macro_scale={zscore,robust}`) fit on train by default (recommended with `--no_norm`)
- `module/model_adapter.py` already supports passing `macro_features` everywhere (train/valid/predict/visuals) via:
  - `trainer_config.market_state_path`
  - `trainer_config.market_state_shift`
  - `trainer_config.market_state_strict`
- `module/utils/market_state.py` now treats **NaN/Inf** in the state as an error when `strict=True`.

---

## Generate Market State Files

This script reuses `work_flow.py:data_conf` and builds a separate dataset with `step_len=1` to get **per-day** features, then aggregates across the whole universe each day.

### CSI300 (kaiming profile, recommended when --no_norm)

```bash
python scripts/precompute_market_state.py \
  --out market_state_csi300.pkl \
  --pca_dim 16 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --macro_profile kaiming \
  --macro_scale robust \
  --no_norm 

# Note: --market_ts_past_only is NOT included by default.
# Only add it if your label predicts same-day returns (T → T).
# For T+k prediction (k>=1), the default (no shift) is correct.
```

### CSI300 (full profile, richer but higher-dim)

```bash
python scripts/precompute_market_state.py \
  --out market_state_csi300.pkl \
  --pca_dim 16 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --zscore_windows 20,60 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --macro_scale robust \
  --no_norm
```

### CSI800 (kaiming profile, recommended when --no_norm)

```bash
python scripts/precompute_market_state.py \
  --out market_state_csi800.pkl \
  --instruments csi800 \
  --pca_dim 32 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --macro_profile kaiming \
  --macro_scale robust \
  --no_norm 

```

### CSI800 (full profile, richer but higher-dim)

```bash
python scripts/precompute_market_state.py \
  --out market_state_csi800.pkl \
  --instruments csi800 \
  --pca_dim 32 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --zscore_windows 20,60,120 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --macro_scale robust \
  --no_norm
```

Outputs:
- `market_state_*.pkl` (main state DataFrame)
- `market_state_*.pkl.pca.npz` (PCA mean/components sidecar for reproducibility)
- `market_state_*.pkl.pca.meta.json` (PCA fit metadata: split/range/dims; useful for leakage auditing)

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
- `--pca_fit_on`: PCA fit split (default: `train`)
  - `train` (recommended for paper/strict backtests): fit PCA on train days only, then transform valid/test (no look-ahead)
  - `all`: legacy behavior (fit on all days; not recommended for strict evaluation)

### Macro Profile & Scale
- `--macro_profile`: `full` or `kaiming`
  - `full`: keep all generated columns
  - `kaiming`: keep only core scalars + PCA + **raw** market TS (drops roll/zscore/delta for a compact, high-signal state)
- `--macro_scale`: `none` (default), `zscore`, or `robust`
  - Fit on train by default and applied to all dates; recommended when using `--no_norm`
- `--macro_scale_fit_on`: `train` or `all` (default uses `--pca_fit_on`)

### Cross-period Standardization
- `--zscore_windows`: comma-separated windows, e.g. `20,60,120`
  - Generates `*_z{window}` columns using rolling mean/std computed **up to day T** (default, for T+k prediction).
  - For same-day prediction (T→T), modify the script to set `shift_stats=True`.
- `--roll_mean`: optional rolling mean window; generates `*_roll_mean{W}` columns
  - Computed **up to day T** by default.
  - For same-day prediction, uncomment the `shift(1)` line in the script.

### Δstate (past-only)
- `--state_delta_lags`: comma-separated lags, e.g. `1,5,10`
  - Generates `*_d{lag}` columns as `state_t - state_{t-lag}`.

### Market time-series features (temporal signal)
- `--add_market_ts`: append benchmark market TS features computed from the benchmark close series.
- `--market_index`: override benchmark instrument code (defaults: `csi300->SH000300`, `csi800->SH000906`).
- `--market_ts_windows`: comma-separated windows used for vol/mom/dd.
- `--market_ts_past_only`: (optional flag) shift TS features by 1 day.
  - **Default**: disabled (features computed up to day T, for T+k prediction where k>=1).
  - **Enable only** for same-day prediction (T→T) scenarios.

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

Optional model-side regularization:

```python
"model_config": {
  "regime_macro_dropout": 0.1,
}
```

Notes:
- `market_state_shift=0` (default, recommended): Use state computed on day T to predict T+1 onwards.
  - Features are computed **up to and including day T** (T's close price is known when making T+1 predictions).
  - This configuration matches labels like `Ref($close, -5) / Ref($close, -1) - 1` (T+1 to T+5 returns).
- `market_state_shift=1` (conservative): Use state computed on day T-1 to predict T onwards.
  - **Only needed** if your label predicts same-day returns (T→T), which is rare.
  - For standard T+k prediction (k>=1), this wastes 1 day of information.
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

## Time Alignment Philosophy (Important!)

### Default Behavior (T+k Prediction, k>=1)

**Assumption**: You make trading decisions **after day T closes** and predict **T+1 onwards**.

- Features are computed **up to and including day T** (no shift).
- Label example: `Ref($close, -5) / Ref($close, -1) - 1` predicts T+1 to T+5 returns.
- This is the **correct** setup for:
  - Overnight strategy (decide after T's close, execute at T+1's open)
  - Multi-day holding periods (T+5, T+10, T+20)

**Why no shift?** On day T's close, you know T's price and can compute T's market state (volatility, correlation, etc.). Using this information to predict T+1~T+5 has **no look-ahead bias**.

### When to Enable Shift (T→T Same-Day Prediction)

**Only if** your label predicts **same-day returns** (e.g., `$close / Ref($close, 1) - 1`):

1. Add `--market_ts_past_only` when generating state
2. Set `market_state_shift=1` in training config
3. Modify script: set `shift_stats=True` for zscore and uncomment `shift(1)` for roll_mean

This scenario is **rare** in practice (most quant strategies predict at least T+1).

---

## Practical Advice (CSI800 / Long Horizon)

- CSI800 is larger and noisier: prefer `--weight_field '$amount'` and robust filtering.
- For t+5/t+20: if you use the **full profile**, add `--zscore_windows` (e.g. `20,60,120`); if you use the **kaiming profile**, keep zscore off and rely on `--macro_scale=robust`.
- If you keep `CSRankNorm` on labels, market-state features should carry more slow-moving information (correlation/crowding + PCA regime) to help the router distinguish regimes.
- **Do not** use `--market_ts_past_only` or `market_state_shift=1` unless you have a same-day prediction task.

---

## Design Analysis & Verification (2024-12-18)

### Current Configuration Summary

| Component | Setting | Value |
|-----------|---------|-------|
| **Label** | `work_flow.py` | `Ref($close, -5) / Ref($close, -1) - 1` |
| **Prediction Horizon** | T+1 to T+5 | Next-day to 5-day forward return |
| **market_state_shift** | Recommended | `0` (default) |
| **--market_ts_past_only** | Recommended | `False` (default) |
| **zscore shift_stats** | Recommended | `False` (default) |
| **roll_mean shift** | Recommended | `False` (default) |

### Label Analysis

```python
# work_flow.py line 74
"label": ["Ref($close, -5) / Ref($close, -1) - 1"]
```

**Interpretation**:
- `Ref($close, -5)` = Close price 5 days into the future (T+5)
- `Ref($close, -1)` = Close price 1 day into the future (T+1)
- This computes: **(T+5 close / T+1 close) - 1** = Return from T+1 to T+5

**Key Insight**: The label predicts returns **starting from T+1**, not from T. This is a forward-looking prediction made **after observing day T's close**.

### Time Alignment Verification

#### ✅ No Look-Ahead Bias

The current design has **no look-ahead bias** for the T+1 to T+5 prediction task:

```
Timeline:
   T-2      T-1       T        T+1      T+2      T+3      T+4      T+5
    |        |        |         |        |        |        |        |
                     [Close]   [Open]
                       ↓         ↓
              Features computed  Prediction executed
              up to here         starting here
```

1. **Features (market state)**: Computed using data up to and including day T's close
2. **Decision point**: After day T's market close
3. **Execution**: Buy at T+1's open (or close for `deal_price=close`)
4. **Label**: Return from T+1 to T+5

Since features use only {T-∞, ..., T} and predict {T+1, ..., T+5}, there is **no temporal leakage**.

#### Why `market_state_shift=0` is Correct

```python
# module/utils/market_state.py lines 71-75
if shift:
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    df = df.shift(shift)  # Uses pandas shift on index
```

- `shift=0`: For date T query, return state computed on day T
- `shift=1`: For date T query, return state computed on day T-1

For T+k prediction (k≥1), `shift=0` is correct because:
1. You make the prediction after T's close
2. T's market state is fully observable at that point
3. Using T's state to predict T+1~T+5 has no information leakage

**Warning**: Only use `shift=1` if your label predicts same-day returns (T→T), which is NOT the case here.

### Precompute Script Analysis

#### Feature Categories and Their Time Properties

| Feature Category | Computed From | Look-Ahead Safe? |
|-----------------|---------------|------------------|
| Global scalars (`mean_abs`, `std`, etc.) | Day T cross-section | ✅ Yes |
| Correlation/crowding | Day T factor correlations | ✅ Yes |
| PCA components | Day T factor matrix | ✅ Yes |
| Δstate (`*_d1`, `*_d5`, etc.) | `state_T - state_{T-lag}` | ✅ Yes (past-only) |
| Rolling z-score (`*_z20`, etc.) | Rolling window up to T | ✅ Yes (with `shift_stats=False`) |
| Roll mean (`*_roll_mean20`) | Rolling window up to T | ✅ Yes (NOT shifted by default) |
| Market TS features | Benchmark close up to T | ✅ Yes (with `past_only=False`) |

#### Verified Code Paths

```python
# precompute_market_state.py line 685
z = _rolling_zscore(state_df, w, shift_stats=False)  # ✅ Correct for T+k

# precompute_market_state.py lines 672-678
rolled = state_df.rolling(r, min_periods=r).mean()
# NOT shifted by default → uses data up to T → ✅ Correct for T+k

# precompute_market_state.py line 669
ts_feat = _market_ts_features(close, m_wins, past_only=bool(args.market_ts_past_only))
# Default past_only=False → uses data up to T → ✅ Correct for T+k
```

### Potential Issues & Recommendations

#### 1. ✅ No Bugs Found in Core Logic

The time alignment is correct for the T+1 to T+5 prediction task:
- Features computed up to day T
- Label predicts T+1 to T+5
- `market_state_shift=0` correctly maps date → same-day state

#### 2. ⚠️ Warmup Period Consideration

Rolling features (z-score, roll_mean, market_ts) introduce NaN values for the initial warmup period:

```python
# Warmup auto-calculation in precompute script
need = max(max_lag, max_window + 1)
```

**Recommendation**: Ensure `--warmup_trading_days` is sufficient or set `market_state_strict=False` for initial dates.

#### 3. ⚠️ Documentation-Code Consistency

The documentation and code are now aligned:
- `macro_feature.md` correctly states `shift=0` for T+k prediction
- `precompute_market_state.py` uses correct defaults
- `model_adapter.py` defaults to `market_state_shift=0`

### Configuration Checklist for Current Setup

For the label `Ref($close, -5) / Ref($close, -1) - 1` (T+1 to T+5):

- [x] `market_state_shift=0` in trainer_config
- [x] No `--market_ts_past_only` flag when running precompute
- [x] `shift_stats=False` in `_rolling_zscore()` (default)
- [x] `roll_mean` NOT shifted (default behavior)
- [x] Warmup period extended to cover rolling windows (if enabled)
- [x] `--macro_profile=kaiming` and `--macro_scale=robust` when using `--no_norm`

### Example Complete Configuration

```python
# work_flow.py trainer_config
"trainer_config": {
    "lr": 5e-4,
    "n_epochs": 20,
    "batch_size": 128,  # Use 64-256 for proper regime estimation
    # Macro feature configuration (T+1 to T+5 prediction)
    "market_state_path": "market_state_csi300.pkl",
    "market_state_shift": 0,        # Use day T's state for T's sample
    "market_state_strict": True,    # Raise on missing/NaN states
    ...
}
```

> [!NOTE]
> For Internal Regime mode (`use_external_macro=False`), use `batch_size >= 64` to ensure meaningful correlation/crowding statistics.

```bash
# Precompute (kaiming profile, recommended when --no_norm)
python scripts/precompute_market_state.py \
  --out market_state_csi300.pkl \
  --pca_dim 16 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --macro_profile kaiming \
  --macro_scale robust \
  --no_norm
  # Note: NO --market_ts_past_only flag (correct for T+k prediction)

# Precompute (full profile, higher-dim)
python scripts/precompute_market_state.py \
  --out market_state_csi300.pkl \
  --pca_dim 16 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --zscore_windows 20,60 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --macro_scale robust \
  --no_norm
  # Note: NO --market_ts_past_only flag (correct for T+k prediction)
```

### Conclusion

**The macro feature design is correctly aligned with the current label configuration (T+1 to T+5 prediction).** No bugs were found in the time alignment logic. The default settings (`market_state_shift=0`, no `--market_ts_past_only`) are appropriate for this prediction horizon.
