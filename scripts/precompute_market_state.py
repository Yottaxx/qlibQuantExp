"""
Precompute market daily state for `macro_features`.

This script builds a per-date market state vector from the full universe:
1) Cross-sectional (over stocks) per-factor stats (mean/std/breadth)
2) PCA-compressed regime vector (8~32 dims typical)
3) Correlation / crowding scalars (mean abs corr, Frobenius norm, PC1 ratio)
4) Optional past-only rolling z-score features for multi-horizon stability

Key options
-----------
- Raw-ish features: `--no_norm` removes RobustZScoreNorm from infer_processors.
- Robust filtering: remove abnormal stocks per day using robust z-score.
- Weighting: use qlib field weights (e.g. '$amount', '$volume') if provided.

Outputs
-------
DataFrame with index=datetime, saved as .pkl/.parquet/.csv.
Also writes sidecars:
- `*.pca.npz` for PCA params (mean/components)
- `*.pca.meta.json` for PCA fit metadata (split/range/dims), useful for leakage auditing
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _normalize_date_str(s: str) -> str:
    return str(pd.Timestamp(s).normalize().date())


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, np.inf)
    s = w.sum()
    if s <= 0:
        return np.nanmean(x, axis=0)
    return np.nansum(x * w[:, None], axis=0) / s


def _weighted_var(x: np.ndarray, w: np.ndarray, mean: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, np.inf)
    s = w.sum()
    if s <= 0:
        return np.nanvar(x, axis=0)
    xc = x - mean[None, :]
    return np.nansum((xc * xc) * w[:, None], axis=0) / s


def _robust_filter_mask(x: np.ndarray, *, z_thresh: float, max_bad_frac: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med[None, :]), axis=0)
    mad = np.where(mad > 1e-12, mad, 1.0)
    rz = (x - med[None, :]) / (1.4826 * mad[None, :])
    bad_frac = np.nanmean(np.abs(rz) > float(z_thresh), axis=1)
    return bad_frac <= float(max_bad_frac)


def _corr_summaries(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 2:
        return {
            "market_state_corr_mean_abs": np.nan,
            "market_state_corr_fro": np.nan,
            "market_state_corr_pc1_ratio": np.nan,
        }
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    xz = (x - mu[None, :]) / sd[None, :]
    xz = np.nan_to_num(xz, nan=0.0, posinf=0.0, neginf=0.0)
    b = int(xz.shape[0])
    corr = (xz.T @ xz) / max(1, b - 1)
    corr = np.clip(corr, -1.0, 1.0)
    n = int(corr.shape[0])
    off = corr.copy()
    np.fill_diagonal(off, 0.0)
    denom = max(1, n * (n - 1))
    mean_abs = float(np.sum(np.abs(off)) / denom)
    fro = float(np.sqrt(np.sum(off * off)))
    try:
        eig = np.linalg.eigvalsh(corr)
        top = float(np.max(eig))
        tr = float(np.sum(eig))
        pc1 = float(top / tr) if tr > 0 else np.nan
    except Exception:
        pc1 = np.nan
    return {
        "market_state_corr_mean_abs": mean_abs,
        "market_state_corr_fro": fro,
        "market_state_corr_pc1_ratio": pc1,
    }


def _agg_global(x: np.ndarray, w: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Compute global distribution statistics across all stocks and factors.
    
    Args:
        x: Feature matrix of shape [n_stocks, n_features]
        w: Optional per-stock weights of shape [n_stocks]
    
    Returns:
        Dict with mean_abs, std, breadth, tail_2sigma statistics
    """
    x = np.asarray(x, dtype=float)
    
    if w is None:
        # Unweighted case: simple statistics over all finite values
        arr = x.reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "market_state_mean_abs": np.nan,
                "market_state_std": np.nan,
                "market_state_breadth": np.nan,
                "market_state_tail_2sigma": np.nan,
            }
        return {
            "market_state_mean_abs": float(np.mean(np.abs(arr))),
            "market_state_std": float(np.std(arr)),
            "market_state_breadth": float(np.mean(arr > 0)),
            "market_state_tail_2sigma": float(np.mean(np.abs(arr) > 2.0)),
        }

    # Weighted case: compute per-stock statistics first, then aggregate
    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, np.inf)
    
    if w.sum() <= 0:
        # Fall back to unweighted if weights are all zero
        arr = x.reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "market_state_mean_abs": np.nan,
                "market_state_std": np.nan,
                "market_state_breadth": np.nan,
                "market_state_tail_2sigma": np.nan,
            }
        return {
            "market_state_mean_abs": float(np.mean(np.abs(arr))),
            "market_state_std": float(np.std(arr)),
            "market_state_breadth": float(np.mean(arr > 0)),
            "market_state_tail_2sigma": float(np.mean(np.abs(arr) > 2.0)),
        }
    
    # Normalize weights
    w = w / w.sum()
    
    # Compute per-stock statistics (mean across features for each stock)
    # Then aggregate with stock weights
    n_stocks, n_features = x.shape
    
    # Per-stock mean absolute value (average across features)
    stock_mean_abs = np.nanmean(np.abs(x), axis=1)  # [n_stocks]
    # Per-stock std (std across features)
    stock_std = np.nanstd(x, axis=1)  # [n_stocks]
    # Per-stock breadth (fraction of positive values)
    stock_breadth = np.nanmean(x > 0, axis=1)  # [n_stocks]
    # Per-stock tail (fraction of |x| > 2)
    stock_tail = np.nanmean(np.abs(x) > 2.0, axis=1)  # [n_stocks]
    
    # Filter out stocks with all NaN features
    valid_mask = np.isfinite(stock_mean_abs)
    if valid_mask.sum() == 0:
        return {
            "market_state_mean_abs": np.nan,
            "market_state_std": np.nan,
            "market_state_breadth": np.nan,
            "market_state_tail_2sigma": np.nan,
        }
    
    # Renormalize weights for valid stocks only
    w_valid = w[valid_mask]
    w_valid = w_valid / max(w_valid.sum(), 1e-12)
    
    # Weighted aggregation across stocks
    mean_abs = float(np.sum(stock_mean_abs[valid_mask] * w_valid))
    std = float(np.sum(stock_std[valid_mask] * w_valid))
    breadth = float(np.sum(stock_breadth[valid_mask] * w_valid))
    tail = float(np.sum(stock_tail[valid_mask] * w_valid))
    
    return {
        "market_state_mean_abs": mean_abs,
        "market_state_std": std,
        "market_state_breadth": breadth,
        "market_state_tail_2sigma": tail,
    }


def _factor_stats(x: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if w is None:
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        br = np.nanmean(x > 0, axis=0)
    else:
        w = np.asarray(w, dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        mu = _weighted_mean(x, w)
        sd = np.sqrt(np.maximum(_weighted_var(x, w, mu), 0.0))
        s = np.sum(np.clip(w, 0.0, np.inf))
        if s <= 0:
            br = np.nanmean(x > 0, axis=0)
        else:
            br = np.nansum((x > 0).astype(float) * w[:, None], axis=0) / s
    return np.concatenate([mu, sd, br], axis=0).astype(np.float32, copy=False)


def _pca_fit(mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA (via SVD) on the provided matrix.

    Notes
    -----
    - This is intended to be fit on *train only* for paper-quality experiments.
    - We center with nanmean; rows with NaNs are kept but NaNs are treated as 0 after centering.
    """
    x = np.asarray(mat, dtype=float)
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    x = np.nan_to_num(x - mu[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = int(min(max(1, k), vt.shape[0]))
    comps = vt[:k]
    return mu.astype(np.float32), comps.astype(np.float32)


def _pca_transform(mat: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """
    Apply PCA transform to a matrix using pre-fit mean and components.

    Returns
    -------
    scores: np.ndarray
        Shape [n_samples, k].
    """
    x = np.asarray(mat, dtype=float)
    mu = np.asarray(mu, dtype=float)
    comps = np.asarray(comps, dtype=float)
    x = np.nan_to_num(x - mu[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    scores = x @ comps.T
    return scores.astype(np.float32, copy=False)


def _filter_df_by_datetime(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    Defensive segment filtering: keep only rows within [start, end] on the 'datetime' index level.

    We do this even if `dataset.prepare(seg)` is supposed to do it already, because `handler.fetch()`
    may return a superset on some Qlib versions.
    """
    if start is None or end is None:
        return df
    if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
        return df
    dts = pd.to_datetime(df.index.get_level_values("datetime")).normalize()
    m = (dts >= pd.Timestamp(start).normalize()) & (dts <= pd.Timestamp(end).normalize())
    # Keep index/columns as-is (MultiIndex preserved).
    return df.loc[m]


def _rolling_zscore(df: pd.DataFrame, window: int, *, shift_stats: bool = False) -> pd.DataFrame:
    """
    Compute rolling z-score normalization.
    
    Args:
        shift_stats: If True, use past-only statistics (shift by 1).
                     Only enable for same-day prediction (T → T).
                     For T+k prediction (k>=1), set False to use all available info up to T.
    """
    w = int(window)
    mu = df.rolling(w, min_periods=w).mean()
    sd = df.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    
    if shift_stats:
        # Past-only: shift stats by 1 day (for same-day prediction scenarios)
        mu = mu.shift(1)
        sd = sd.shift(1)
    
    return (df - mu) / sd


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in (s or "").split(",") if str(x).strip()]


def _market_ts_features(close: pd.Series, windows: List[int], *, past_only: bool = False) -> pd.DataFrame:
    """
    Build compact market-level time-series features from a benchmark close series.
    
    Args:
        past_only: If True, shift all features by 1 day (for same-day prediction T → T).
                   If False (default), use features up to day T (for T+k prediction, k>=1).
                   
    Note:
        For label = Ref($close, -k) where k >= 1, past_only=False is correct because
        you make decisions after T's close and predict T+1 onwards.
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    c.index = pd.to_datetime(c.index).normalize()
    c = c.sort_index()

    r1 = c.pct_change(1)
    out: Dict[str, pd.Series] = {"market_ret_1d": r1}

    for w in windows:
        w = int(w)
        if w <= 1:
            continue
        out[f"market_vol_{w}"] = r1.rolling(w, min_periods=w).std(ddof=0)
        out[f"market_mom_{w}"] = c.pct_change(w)
        # drawdown over window: 1 - close / rolling_max(close)
        roll_max = c.rolling(w, min_periods=w).max()
        out[f"market_dd_{w}"] = 1.0 - (c / roll_max)

    df = pd.DataFrame(out)
    if past_only:
        df = df.shift(1)
    return df


def _auto_warmup_lookback_days(
    *,
    state_delta_lags: List[int],
    roll_mean: int,
    zscore_windows: List[int],
    market_ts_windows: List[int],
    add_market_ts: bool,
    market_ts_past_only: bool,
) -> int:
    """
    Estimate how many prior trading days we need so the earliest training date
    has no NaN in any generated macro features, accounting for chained windows.
    """
    lags = [int(x) for x in (state_delta_lags or []) if int(x) > 0]
    z_wins = [int(x) for x in (zscore_windows or []) if int(x) > 1]
    mt_wins = [int(x) for x in (market_ts_windows or []) if int(x) > 1]

    # Base lookback from features computed directly on daily states.
    base_lb = max(lags) if lags else 0
    if add_market_ts:
        mt_lb = max([1] + mt_wins)  # market_ret_1d needs 1 prior day
        if market_ts_past_only:
            mt_lb += 1
        base_lb = max(base_lb, mt_lb)

    need = base_lb
    r = int(roll_mean or 0)
    max_z = max(z_wins) if z_wins else 0

    # Rolling mean and zscore are applied after delta/market_ts are added, so they chain.
    if r > 1:
        need = max(need, base_lb + (r - 1))
    if max_z > 1:
        need = max(need, base_lb + (max_z - 1))
    if r > 1 and max_z > 1:
        need = max(need, base_lb + (r - 1) + (max_z - 1))

    return int(max(need, 0))


def _extend_train_start_by_trading_days(
    dc: Dict,
    *,
    warmup_days: int,
    D,
) -> Dict:
    """
    Extend the earliest segment start (train) backward by `warmup_days` trading days.
    This is for macro-feature precompute only; it does not change your training workflow.
    """
    if warmup_days <= 0:
        return dc

    segs = (dc.get("kwargs", {}) or {}).get("segments", {}) or {}
    if "train" not in segs or not segs["train"]:
        return dc

    train_start = pd.Timestamp(segs["train"][0]).normalize()
    # Approximate window for calendar query (avoid relying on provider covering very old years).
    approx_start = (train_start - pd.Timedelta(days=int(warmup_days * 3))).normalize()
    cal = D.calendar(start_time=str(approx_start.date()), end_time=str(train_start.date()), freq="day")
    cal = pd.to_datetime(cal)
    if cal.size == 0:
        return dc
    cal = pd.DatetimeIndex(cal).sort_values()

    # Find position of the last calendar date <= train_start
    pos = int(cal.searchsorted(train_start, side="right") - 1)
    pos = max(0, min(pos, len(cal) - 1))
    new_pos = max(0, pos - int(warmup_days))
    new_start = pd.Timestamp(cal[new_pos]).normalize()

    new_start_str = str(new_start.date())
    dc = copy.deepcopy(dc)
    dc["kwargs"]["handler"]["kwargs"]["start_time"] = new_start_str
    # Keep handler fit_start_time consistent (some handlers/normalizers depend on this)
    if "fit_start_time" in dc["kwargs"]["handler"]["kwargs"]:
        dc["kwargs"]["handler"]["kwargs"]["fit_start_time"] = new_start_str
    # Extend only the earliest segment (train)
    train_end = segs["train"][1]
    dc["kwargs"]["segments"]["train"] = (new_start_str, train_end)
    return dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output path (.pkl/.parquet/.csv)")
    ap.add_argument("--instruments", type=str, default=None, help="Override instruments (e.g. csi300/csi800)")
    ap.add_argument("--no_norm", action="store_true", help="Remove RobustZScoreNorm from infer_processors.")
    ap.add_argument("--filter_robust_z", type=float, default=0.0, help="Robust zscore threshold (0 disables).")
    ap.add_argument("--filter_max_bad_frac", type=float, default=0.05, help="Max bad-feature fraction per stock.")
    ap.add_argument("--weight_field", type=str, default=None, help="Optional qlib field for weights (e.g. '$amount').")
    ap.add_argument(
        "--trade_field",
        type=str,
        default=None,
        help="Optional qlib field used to filter non-trading/suspended rows (e.g. '$volume' or '$amount').",
    )
    ap.add_argument(
        "--min_trade",
        type=float,
        default=0.0,
        help="Keep rows with trade_field > min_trade (set >0 to drop 0-volume/0-amount days).",
    )
    ap.add_argument(
        "--suspend_field",
        type=str,
        default=None,
        help="Optional qlib field for suspension flag (keep rows with value == 0).",
    )
    ap.add_argument("--pca_dim", type=int, default=16, help="PCA output dim (8~32 typical).")
    ap.add_argument(
        "--pca_fit_on",
        type=str,
        default="train",
        choices=["train", "all"],
        help=(
            "PCA fit split. For paper-quality experiments, use 'train' to avoid look-ahead "
            "(fit PCA on train days only, then apply to valid/test)."
        ),
    )
    ap.add_argument("--zscore_windows", type=str, default="", help="Comma-separated windows, e.g. '20,60,120'.")
    ap.add_argument("--roll_mean", type=int, default=0, help="Optional past-only rolling mean window.")
    ap.add_argument("--state_delta_lags", type=str, default="1,5,10", help="Comma-separated lags for Δstate features.")
    ap.add_argument("--add_market_ts", action="store_true", help="Append benchmark market time-series features.")
    ap.add_argument(
        "--market_index",
        type=str,
        default=None,
        help="Benchmark instrument code for market time-series features (e.g. SH000300/SH000906).",
    )
    ap.add_argument("--market_ts_windows", type=str, default="5,20,60", help="Comma-separated market TS windows.")
    ap.add_argument(
        "--market_ts_past_only",
        action="store_true",
        help="Use past-only market TS features (shift by 1 day). Only enable for same-day prediction (T→T). For T+k prediction (k>=1), keep disabled (default).",
    )
    ap.add_argument(
        "--warmup_trading_days",
        type=int,
        default=-1,
        help=(
            "Extend the precompute start backward by N trading days for rolling/zscore/delta warmup. "
            "Default (-1) uses an automatic value based on your selected feature windows."
        ),
    )
    args = ap.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Add project root to sys.path for importing work_flow
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import work_flow
    import qlib
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data import D
    from qlib.utils import init_instance_by_config

    qlib.init(provider_uri=work_flow.provider_uri, region=work_flow.REG_CN)

    dc = copy.deepcopy(work_flow.data_conf)
    if args.instruments is not None:
        dc["kwargs"]["handler"]["kwargs"]["instruments"] = args.instruments
    dc["kwargs"]["step_len"] = 1
    if args.no_norm:
        ip = dc["kwargs"]["handler"]["kwargs"].get("infer_processors", [])
        dc["kwargs"]["handler"]["kwargs"]["infer_processors"] = [
            p for p in ip if not (isinstance(p, dict) and p.get("class") == "RobustZScoreNorm")
        ]

    # Extend the earliest segment (train) backward to provide warmup for rolling/zscore/delta features.
    delta_lags = _parse_int_list(args.state_delta_lags)
    z_wins = _parse_int_list(args.zscore_windows)
    m_wins = _parse_int_list(args.market_ts_windows)
    if int(args.warmup_trading_days) >= 0:
        warmup_days = int(args.warmup_trading_days)
    else:
        warmup_days = _auto_warmup_lookback_days(
            state_delta_lags=delta_lags,
            roll_mean=int(args.roll_mean or 0),
            zscore_windows=z_wins,
            market_ts_windows=m_wins,
            add_market_ts=bool(args.add_market_ts),
            market_ts_past_only=bool(args.market_ts_past_only),
        )
    
    # Log the original train range before extension
    orig_train_start = dc.get("kwargs", {}).get("segments", {}).get("train", (None, None))[0]
    print(f"[INFO] Warmup days: {warmup_days}, original train start: {orig_train_start}")
    
    dc = _extend_train_start_by_trading_days(dc, warmup_days=warmup_days, D=D)
    
    # Log the extended train range
    extended_train_start = dc.get("kwargs", {}).get("segments", {}).get("train", (None, None))[0]
    print(f"[INFO] Extended train start for warmup: {extended_train_start}")

    dataset = init_instance_by_config(dc)

    rows: List[Dict] = []
    factor_vecs: List[np.ndarray] = []
    idx_list: List[pd.Timestamp] = []
    min_dt: Optional[pd.Timestamp] = None
    max_dt: Optional[pd.Timestamp] = None

    for seg in ("train", "valid", "test"):
        tsds = dataset.prepare(seg, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        
        # Get segment time range for filtering
        seg_start, seg_end = None, None
        try:
            segments = dc.get("kwargs", {}).get("segments", {})
            if seg in segments:
                seg_range = segments[seg]
                if isinstance(seg_range, (tuple, list)) and len(seg_range) == 2:
                    seg_start = pd.Timestamp(seg_range[0]).normalize()
                    seg_end = pd.Timestamp(seg_range[1]).normalize()
                    print(f"[INFO] Segment '{seg}' time range: {seg_start.date()} ~ {seg_end.date()}")
        except Exception:
            pass
        
        # Try multiple ways to get the underlying DataFrame (Qlib version compatibility)
        df = None
        method_used = None
        
        # Method 1: direct .data attribute (older Qlib)
        if df is None:
            df = getattr(tsds, "data", None)
            if df is not None and isinstance(df, pd.DataFrame):
                method_used = "Method 1: tsds.data attribute"
            else:
                df = None
        
        # Method 2: via handler fetch with segment time range (newer Qlib)
        if df is None:
            try:
                handler = getattr(dataset, "handler", None)
                if handler is not None:
                    # Fetch with segment selector to get only the segment's data
                    if seg_start is not None and seg_end is not None:
                        df = handler.fetch(
                            col_set="feature",
                            data_key=DataHandlerLP.DK_I,
                            selector=slice(seg_start, seg_end),
                        )
                    else:
                        df = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        method_used = "Method 2: handler.fetch()"
                    else:
                        df = None
            except Exception as e:
                print(f"[DEBUG] Method 2 failed for '{seg}': {e}")
                df = None
        
        # Method 3: iterate and collect from TSDataSampler
        if df is None:
            try:
                print(f"[INFO] Trying Method 3 (iterate TSDataSampler) for '{seg}', this may take a while...")
                idx = tsds.get_index()
                samples = []
                total_samples = len(tsds)
                for i in tqdm(range(total_samples), desc=f"Method3 {seg}", unit="sample"):
                    s = tsds[i]
                    if isinstance(s, dict):
                        x = s.get("feature", s.get("data", s.get("x", None)))
                    elif isinstance(s, (tuple, list)):
                        x = s[0] if len(s) > 0 else None
                    else:
                        x = s
                    if x is not None:
                        # x is [T, F] for TSDatasetH; take last timestep for daily features
                        x_np = np.asarray(x, dtype=float)
                        if x_np.ndim == 2:
                            x_np = x_np[-1, :]  # last time step
                        samples.append(x_np)
                if samples:
                    df = pd.DataFrame(np.stack(samples, axis=0), index=idx)
                    method_used = "Method 3: iterate TSDataSampler"
            except Exception as e:
                print(f"[DEBUG] Method 3 failed for '{seg}': {e}")
                df = None
        
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"[WARN] Segment '{seg}' returned no valid DataFrame, skipping...")
            continue

        # Defensive filter: ensure the segment range is respected even if handler.fetch returns a superset.
        df = _filter_df_by_datetime(df, seg_start, seg_end)
        if df.empty:
            print(f"[WARN] Segment '{seg}' became empty after datetime filtering, skipping...")
            continue
        
        print(f"[INFO] Segment '{seg}': loaded DataFrame with shape {df.shape} using {method_used}")
        if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
            raise RuntimeError("Expected MultiIndex with 'datetime' level in tsds.data index.")

        dt_level = df.index.names.index("datetime")
        inst_level = 0 if dt_level == 1 else 1
        inst_name = df.index.names[inst_level]

        aux_df = None
        aux_cols = []
        if args.weight_field:
            aux_cols.append(args.weight_field)
        if args.trade_field and args.trade_field not in aux_cols:
            aux_cols.append(args.trade_field)
        if args.suspend_field and args.suspend_field not in aux_cols:
            aux_cols.append(args.suspend_field)

        if aux_cols:
            insts = df.index.get_level_values(inst_name).unique().tolist()
            start = str(pd.to_datetime(df.index.get_level_values("datetime")).min().date())
            end = str(pd.to_datetime(df.index.get_level_values("datetime")).max().date())
            adf = D.features(insts, aux_cols, start_time=start, end_time=end, freq="day")
            # Normalize column names to safe internal names
            rename = {}
            if args.weight_field:
                rename[args.weight_field] = "_w"
            if args.trade_field:
                rename[args.trade_field] = "_trade"
            if args.suspend_field:
                rename[args.suspend_field] = "_susp"
            adf = adf.rename(columns=rename)
            aux_df = adf

        # Group by date and compute daily stats
        daily_groups = list(df.groupby(level="datetime"))
        for dt, sub in tqdm(daily_groups, desc=f"Processing {seg}", unit="day"):
            dtn = pd.Timestamp(dt).normalize()
            min_dt = dtn if min_dt is None else min(min_dt, dtn)
            max_dt = dtn if max_dt is None else max(max_dt, dtn)
            x = sub.to_numpy(dtype=float)
            w = None
            trade = None
            susp = None
            if aux_df is not None:
                try:
                    aux_sub = aux_df.xs(dt, level="datetime")
                except KeyError:
                    aux_sub = pd.DataFrame(index=pd.Index([], name=inst_name), columns=aux_df.columns)
                aux_sub = aux_sub.reindex(sub.index.get_level_values(inst_name))
                if "_w" in aux_sub.columns:
                    w = aux_sub["_w"].to_numpy(dtype=float)
                if "_trade" in aux_sub.columns:
                    trade = aux_sub["_trade"].to_numpy(dtype=float)
                if "_susp" in aux_sub.columns:
                    susp = aux_sub["_susp"].to_numpy(dtype=float)

            # basic row quality
            row_nan = np.mean(~np.isfinite(x), axis=1)
            keep = row_nan < 0.1
            x = x[keep]
            if w is not None:
                w = w[keep]
            if trade is not None:
                trade = trade[keep]
            if susp is not None:
                susp = susp[keep]

            # trading/suspension filter (recommended for CSI800)
            if trade is not None:
                t = np.asarray(trade, dtype=float)
                t_keep = np.isfinite(t) & (t > float(args.min_trade))
                x = x[t_keep]
                if w is not None:
                    w = w[t_keep]
                if trade is not None:
                    trade = trade[t_keep]
                if susp is not None:
                    susp = susp[t_keep]
            if susp is not None:
                s = np.asarray(susp, dtype=float)
                s_keep = np.isfinite(s) & (s == 0)
                x = x[s_keep]
                if w is not None:
                    w = w[s_keep]

            # robust outlier filtering (optional)
            if args.filter_robust_z and args.filter_robust_z > 0:
                m = _robust_filter_mask(x, z_thresh=args.filter_robust_z, max_bad_frac=args.filter_max_bad_frac)
                x = x[m]
                if w is not None:
                    w = w[m]

            if x.shape[0] < 10:
                continue

            rec: Dict[str, float] = {}
            rec.update(_agg_global(x, w))
            rec.update(_corr_summaries(x))
            rows.append(rec)
            factor_vecs.append(_factor_stats(x, w))
            idx_list.append(pd.Timestamp(dtn))

    if not idx_list:
        raise RuntimeError("No daily states computed; check filters and data.")

    base_df = pd.DataFrame(rows, index=pd.to_datetime(idx_list)).sort_index()
    if not base_df.index.is_unique:
        # Duplicate dates typically indicate mis-filtered segments (e.g., handler.fetch returned a superset).
        # This would silently overwrite during lookup, so fail fast for paper-quality usage.
        dup = base_df.index[base_df.index.duplicated()].unique()
        raise RuntimeError(
            f"Duplicate dates found in computed market_state ({len(dup)} duplicates, e.g. {dup[:5].tolist()}). "
            "Check your segment filtering / Qlib handler.fetch behavior."
        )

    factor_mat = np.stack(factor_vecs, axis=0)

    # PCA fit policy: train-only (recommended) vs all-days (legacy)
    segs = dc.get("kwargs", {}).get("segments", {}) or {}
    train_start, train_end = None, None
    if "train" in segs and segs["train"]:
        try:
            train_start = pd.Timestamp(segs["train"][0]).normalize()
            train_end = pd.Timestamp(segs["train"][1]).normalize()
        except Exception:
            train_start, train_end = None, None

    fit_mask = np.ones((base_df.shape[0],), dtype=bool)
    fit_desc = "all"
    if args.pca_fit_on == "train":
        if train_start is None or train_end is None:
            raise RuntimeError("--pca_fit_on=train requires a valid dc['kwargs']['segments']['train'] range.")
        fit_mask = (base_df.index >= train_start) & (base_df.index <= train_end)
        fit_desc = f"train[{train_start.date()}..{train_end.date()}]"
        if int(np.sum(fit_mask)) < 5:
            raise RuntimeError(f"Too few train days for PCA fit ({int(np.sum(fit_mask))}) in {fit_desc}.")

    pca_mean, pca_comps = _pca_fit(factor_mat[fit_mask], k=args.pca_dim)
    scores = _pca_transform(factor_mat, pca_mean, pca_comps)
    pca_df = pd.DataFrame(scores, index=base_df.index, columns=[f"market_state_pca_{i}" for i in range(scores.shape[1])])
    state_df = pd.concat([base_df, pca_df], axis=1)

    # Δstate features (past-only by construction)
    if delta_lags:
        # Only apply to the base state (exclude already rolling/z columns added later).
        delta_src = state_df.copy()
        for lag in delta_lags:
            if lag <= 0:
                continue
            d = delta_src - delta_src.shift(lag)
            d.columns = [f"{c}_d{lag}" for c in d.columns]
            state_df = pd.concat([state_df, d], axis=1)

    # Optional benchmark market time-series features (temporal signal)
    if args.add_market_ts:
        inst = args.market_index
        if inst is None:
            ins = str(dc["kwargs"]["handler"]["kwargs"].get("instruments", "")).lower()
            if ins == "csi300":
                inst = "SH000300"
            elif ins == "csi800":
                inst = "SH000906"
            else:
                raise RuntimeError("--add_market_ts requires --market_index for instruments other than csi300/csi800.")

        if min_dt is None or max_dt is None:
            raise RuntimeError("Cannot infer date range for market TS features.")

        mdf = D.features([inst], ["$close"], start_time=str(min_dt.date()), end_time=str(max_dt.date()), freq="day")
        if not isinstance(mdf, pd.DataFrame) or mdf.empty:
            raise RuntimeError(f"Failed to load market index data for {inst}.")
        # D.features returns MultiIndex [instrument, datetime]
        try:
            close = mdf.xs(inst, level="instrument")["$close"]
        except Exception:
            # fallback: try to find a single close-like column
            close = mdf.iloc[:, 0]
        close.index = pd.to_datetime(close.index).normalize()

        # Note: past_only should be False for T+k prediction (k>=1), True only for same-day prediction
        ts_feat = _market_ts_features(close, m_wins, past_only=bool(args.market_ts_past_only))
        state_df = state_df.join(ts_feat, how="left")

    if args.roll_mean and args.roll_mean > 1:
        r = int(args.roll_mean)
        rolled = state_df.rolling(r, min_periods=r).mean()
        # Note: NOT shifted by default. Only shift if predicting same-day returns (T → T).
        # For T+k prediction (k>=1), we want features up to and including day T.
        # If you need past-only for same-day prediction, uncomment the next line:
        # rolled = rolled.shift(1)
        rolled.columns = [f"{c}_roll_mean{r}" for c in rolled.columns]
        state_df = pd.concat([state_df, rolled], axis=1)

    for w in z_wins:
        # shift_stats=False (default): use statistics up to day T for T+k prediction
        # Set shift_stats=True only if predicting same-day returns (T → T)
        z = _rolling_zscore(state_df, w, shift_stats=False)
        z.columns = [f"{c}_z{w}" for c in z.columns]
        state_df = pd.concat([state_df, z], axis=1)

    # Save PCA params for reproducibility (and to make leakage checks explicit)
    try:
        np.savez_compressed(out_path.with_suffix(out_path.suffix + ".pca.npz"), mean=pca_mean, components=pca_comps)
        meta = {
            "pca_dim": int(pca_comps.shape[0]),
            "input_dim": int(pca_comps.shape[1]),
            "fit_on": str(args.pca_fit_on),
            "fit_desc": str(fit_desc),
        }
        if train_start is not None and train_end is not None:
            meta["train_range"] = [str(train_start.date()), str(train_end.date())]
        out_path.with_suffix(out_path.suffix + ".pca.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    if out_path.suffix in {".pkl", ".pickle"}:
        state_df.to_pickle(out_path)
    elif out_path.suffix == ".parquet":
        state_df.to_parquet(out_path)
    elif out_path.suffix == ".csv":
        state_df.to_csv(out_path)
    else:
        raise ValueError("out must end with .pkl/.parquet/.csv")

    print(f"Saved market state: {out_path} shape={state_df.shape}")


if __name__ == "__main__":
    main()
