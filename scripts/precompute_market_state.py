"""
Build split market-state assets for RST-MoE.

Outputs
-------
- daily_market_context_<universe>.pkl
- daily_market_observation_<universe>.pkl
- daily_market_field_<universe>.pkl

Notes
-----
- Observation PCA is fit on the original train split only.
- Field scaling / shock thresholds are fit on the original train split only.
- The field asset is date-indexed and numeric, so it can be consumed directly by
  `trainer_config.market_state_path` without changing model-side lookup logic.
- `observation` can be built once for a shared reference universe and reused by
  benchmark-specific `context` / `field` builds via `--input-observation`.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


OBS_RENAME_MAP = {
    "market_state_mean_abs": "market_obs_mean_abs",
    "market_state_std": "market_obs_std",
    "market_state_breadth": "market_obs_breadth",
    "market_state_tail_2sigma": "market_obs_tail_2sigma",
    "market_state_corr_mean_abs": "market_obs_corr_mean_abs",
    "market_state_corr_fro": "market_obs_corr_fro",
    "market_state_corr_pc1_ratio": "market_obs_corr_pc1_ratio",
    "market_state_corr_effective_rank": "market_obs_corr_effective_rank",
    "market_state_corr_spectral_entropy": "market_obs_corr_spectral_entropy",
    "market_state_corr_top3_eigen_ratio": "market_obs_corr_top3_eigen_ratio",
    "market_state_corr_top5_eigen_ratio": "market_obs_corr_top5_eigen_ratio",
    "market_state_corr_pc1_pc2_gap": "market_obs_corr_pc1_pc2_gap",
}

RELATIVE_COMPLEXITY_FEATURES = {
    "corr_mean_abs": "market_obs_corr_mean_abs",
    "corr_fro": "market_obs_corr_fro",
    "corr_pc1_ratio": "market_obs_corr_pc1_ratio",
    "tail_2sigma": "market_obs_tail_2sigma",
    "std": "market_obs_std",
    "breadth": "market_obs_breadth",
    "corr_effective_rank": "market_obs_corr_effective_rank",
    "corr_spectral_entropy": "market_obs_corr_spectral_entropy",
    "corr_top3_eigen_ratio": "market_obs_corr_top3_eigen_ratio",
    "corr_top5_eigen_ratio": "market_obs_corr_top5_eigen_ratio",
    "corr_pc1_pc2_gap": "market_obs_corr_pc1_pc2_gap",
}

TRI_SCOPE_FIELD_PROFILE = "tri_scope_relative_momentum_v1"
TRI_SCOPE_SCOPES = ("all", "csi300", "csi800")
TRI_SCOPE_RELATION_PAIRS = (
    ("csi300", "all"),
    ("csi800", "all"),
    ("csi300", "csi800"),
    ("csi800", "csi300"),
)

SHOCK_WEIGHT_TABLE = {
    "low": np.asarray([0.15, 0.25, 0.60], dtype=np.float32),
    "mid": np.asarray([0.20, 0.60, 0.20], dtype=np.float32),
    "high": np.asarray([0.60, 0.30, 0.10], dtype=np.float32),
}


@dataclass
class RuntimeEnv:
    work_flow: Any
    qlib: Any
    D: Any
    DataHandlerLP: Any
    dataset: Optional[Any]
    reference_dataset: Optional[Any]
    dc: Dict[str, Any]
    reference_dc: Dict[str, Any]
    orig_dc: Dict[str, Any]
    universe: str
    reference_universe: str
    reference_feature_source: str
    market_index: str
    reference_membership_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]
    calendar_pos: Dict[pd.Timestamp, int]
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    data_config_module: str
    data_config_attr: str


@dataclass
class StreamFeaturePipeline:
    feature_exprs: List[str]
    feature_names: List[str]
    provider: str
    robust_enabled: bool
    clip_outlier: bool
    fillna_enabled: bool
    fill_value: float
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    center: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    fit_rows: int = 0
    fit_expected_rows: int = 0
    fit_fetch_coverage: float = float("nan")


def _normalize_date_str(s: str) -> str:
    return str(pd.Timestamp(s).normalize().date())


def _slugify_name(s: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unknown"


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
            "market_state_corr_effective_rank": np.nan,
            "market_state_corr_spectral_entropy": np.nan,
            "market_state_corr_top3_eigen_ratio": np.nan,
            "market_state_corr_top5_eigen_ratio": np.nan,
            "market_state_corr_pc1_pc2_gap": np.nan,
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
    pc1 = spectral_entropy = effective_rank = top3 = top5 = pc1_pc2_gap = np.nan
    try:
        eig = np.linalg.eigvalsh(corr)
        eig = np.asarray(eig, dtype=float)
        eig = np.clip(eig, 0.0, np.inf)
        eig = eig[np.isfinite(eig)]
        eig = np.sort(eig)[::-1]
        tr = float(np.sum(eig))
        if tr > 1e-12 and eig.size > 0:
            p = eig / tr
            pc1 = float(p[0])
            eps = 1e-12
            ent_raw = float(-np.sum(p * np.log(p + eps)))
            spectral_entropy = float(ent_raw / np.log(max(int(eig.size), 2)))
            effective_rank = float(np.exp(ent_raw))
            top3 = float(np.sum(p[: min(3, p.size)]))
            top5 = float(np.sum(p[: min(5, p.size)]))
            pc1_pc2_gap = float(p[0] - (p[1] if p.size > 1 else 0.0))
    except Exception:
        pass
    return {
        "market_state_corr_mean_abs": mean_abs,
        "market_state_corr_fro": fro,
        "market_state_corr_pc1_ratio": pc1,
        "market_state_corr_effective_rank": effective_rank,
        "market_state_corr_spectral_entropy": spectral_entropy,
        "market_state_corr_top3_eigen_ratio": top3,
        "market_state_corr_top5_eigen_ratio": top5,
        "market_state_corr_pc1_pc2_gap": pc1_pc2_gap,
    }


def _agg_global(x: np.ndarray, w: Optional[np.ndarray]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if w is None:
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

    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, np.inf)
    if w.sum() <= 0:
        return _agg_global(x, None)
    w = w / w.sum()
    stock_mean_abs = np.nanmean(np.abs(x), axis=1)
    stock_std = np.nanstd(x, axis=1)
    stock_breadth = np.nanmean(x > 0, axis=1)
    stock_tail = np.nanmean(np.abs(x) > 2.0, axis=1)
    valid_mask = np.isfinite(stock_mean_abs)
    if valid_mask.sum() == 0:
        return {
            "market_state_mean_abs": np.nan,
            "market_state_std": np.nan,
            "market_state_breadth": np.nan,
            "market_state_tail_2sigma": np.nan,
        }
    w_valid = w[valid_mask]
    w_valid = w_valid / max(w_valid.sum(), 1e-12)
    return {
        "market_state_mean_abs": float(np.sum(stock_mean_abs[valid_mask] * w_valid)),
        "market_state_std": float(np.sum(stock_std[valid_mask] * w_valid)),
        "market_state_breadth": float(np.sum(stock_breadth[valid_mask] * w_valid)),
        "market_state_tail_2sigma": float(np.sum(stock_tail[valid_mask] * w_valid)),
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
    x = np.asarray(mat, dtype=float)
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    x = np.nan_to_num(x - mu[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    k = int(min(max(1, k), vt.shape[0]))
    comps = vt[:k]
    return mu.astype(np.float32), comps.astype(np.float32)


def _pca_transform(mat: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    x = np.asarray(mat, dtype=float)
    mu = np.asarray(mu, dtype=float)
    comps = np.asarray(comps, dtype=float)
    x = np.nan_to_num(x - mu[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    scores = x @ comps.T
    return scores.astype(np.float32, copy=False)


def _filter_df_by_datetime(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if start is None or end is None:
        return df
    if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
        return df
    dts = pd.to_datetime(df.index.get_level_values("datetime")).normalize()
    m = (dts >= pd.Timestamp(start).normalize()) & (dts <= pd.Timestamp(end).normalize())
    return df.loc[m]


def _rolling_zscore_series(s: pd.Series, window: int, *, shift_stats: bool = False) -> pd.Series:
    df = pd.DataFrame({"x": pd.to_numeric(s, errors="coerce")})
    mu = df.rolling(window, min_periods=window).mean()
    sd = df.rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    if shift_stats:
        mu = mu.shift(1)
        sd = sd.shift(1)
    return ((df - mu) / sd)["x"]


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in (s or "").split(",") if str(x).strip()]


def _normalize_market_windows(values: Sequence[int]) -> List[int]:
    windows = sorted({int(x) for x in values if int(x) > 1})
    required = {5, 20}
    missing = sorted(required.difference(windows))
    if missing:
        raise ValueError(
            "Split-asset market context currently requires market_ts_windows to include "
            f"{sorted(required)} so downstream field/compat features stay stable; missing={missing}."
        )
    return windows


def _normalize_reference_feature_source(value: str, *, universe: str, reference_universe: str) -> str:
    source = str(value or "auto").strip().lower()
    if source not in {"auto", "processed_dataset", "raw_qlib_stream"}:
        raise ValueError("--reference_feature_source must be one of: auto, processed_dataset, raw_qlib_stream")
    ref_key = str(reference_universe or "").strip().lower()
    uni_key = str(universe or "").strip().lower()
    if source == "auto":
        return "raw_qlib_stream" if ref_key == "all" and ref_key != uni_key else "processed_dataset"
    if source == "processed_dataset" and ref_key == "all" and ref_key != uni_key:
        raise ValueError(
            "reference=all cannot use reference_feature_source=processed_dataset because it materializes "
            "TSDatasetH(all) and can exhaust memory/pagefile. Use raw_qlib_stream instead."
        )
    return source


def _fit_macro_scale(df: pd.DataFrame, method: str) -> tuple[pd.Series, pd.Series]:
    method = str(method or "none").strip().lower()
    if method == "none":
        raise ValueError("_fit_macro_scale called with method='none'")
    if method == "zscore":
        center = df.mean(axis=0, skipna=True)
        scale = df.std(axis=0, skipna=True, ddof=0)
    elif method == "robust":
        center = df.median(axis=0, skipna=True)
        mad = (df.sub(center, axis=1)).abs().median(axis=0, skipna=True)
        scale = mad * 1.4826
    else:
        raise ValueError(f"Unknown macro_scale method: {method}")
    scale = scale.where(scale > 1e-12, 1.0)
    center = center.fillna(0.0)
    return center, scale


def _auto_warmup_lookback_days(*, market_ts_windows: Sequence[int], field_half_lives: Sequence[int]) -> int:
    wins = [int(x) for x in market_ts_windows if int(x) > 1]
    halves = [int(x) for x in field_half_lives if int(x) > 1]
    return int(max(wins + halves + [0]))


def _extend_train_start_by_trading_days(
    dc: Dict[str, Any],
    *,
    warmup_days: int,
    D: Any,
) -> Dict[str, Any]:
    if warmup_days <= 0:
        return dc
    segs = (dc.get("kwargs", {}) or {}).get("segments", {}) or {}
    if "train" not in segs or not segs["train"]:
        return dc
    train_start = pd.Timestamp(segs["train"][0]).normalize()
    approx_start = (train_start - pd.Timedelta(days=int(warmup_days * 3))).normalize()
    cal = D.calendar(start_time=str(approx_start.date()), end_time=str(train_start.date()), freq="day")
    cal = pd.to_datetime(cal)
    if cal.size == 0:
        return dc
    cal = pd.DatetimeIndex(cal).sort_values()
    pos = int(cal.searchsorted(train_start, side="right") - 1)
    pos = max(0, min(pos, len(cal) - 1))
    new_pos = max(0, pos - int(warmup_days))
    new_start = pd.Timestamp(cal[new_pos]).normalize()
    new_start_str = str(new_start.date())
    dc = copy.deepcopy(dc)
    dc["kwargs"]["handler"]["kwargs"]["start_time"] = new_start_str
    if "fit_start_time" in dc["kwargs"]["handler"]["kwargs"]:
        dc["kwargs"]["handler"]["kwargs"]["fit_start_time"] = new_start_str
    train_end = segs["train"][1]
    dc["kwargs"]["segments"]["train"] = (new_start_str, train_end)
    return dc


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text.lower()}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="Deprecated alias for --out-field.")
    ap.add_argument("--out-context", type=str, default=None, help="Output path for daily market context.")
    ap.add_argument("--out-observation", type=str, default=None, help="Output path for daily market observation.")
    ap.add_argument("--out-field", type=str, default=None, help="Output path for daily market field.")
    ap.add_argument("--legacy-out", type=str, default=None, help="Optional compatibility output path.")
    ap.add_argument(
        "--input-observation",
        type=str,
        default=None,
        help=(
            "Optional precomputed daily_market_observation asset to reuse. "
            "When set, expensive reference-universe observation construction is skipped and "
            "only benchmark-specific context/field are rebuilt."
        ),
    )
    ap.add_argument("--data-config-module", type=str, default="work_flow", help="Module that provides the dataset config.")
    ap.add_argument("--data-config-attr", type=str, default="data_conf", help="Attribute name for the dataset config.")
    _add_bool_arg(ap, "emit-context", default=True, help_text="Write context asset.")
    _add_bool_arg(ap, "emit-observation", default=True, help_text="Write observation asset.")
    _add_bool_arg(ap, "emit-field", default=True, help_text="Write field asset.")
    ap.add_argument(
        "--field_profile",
        type=str,
        default="stable_v1",
        choices=["stable_v1", "relative_complexity_v1", TRI_SCOPE_FIELD_PROFILE],
        help=(
            "Field construction profile. stable_v1 preserves current behavior. "
            "relative_complexity_v1 adds ref-all vs benchmark correlation-complexity features. "
            "tri_scope_relative_momentum_v1 builds all/csi300/csi800 day summary plus benchmark-relative momentum."
        ),
    )
    _add_bool_arg(
        ap,
        "preserve-observation-index",
        default=True,
        help_text="Keep observation output on its native full date index instead of cropping it to field dates.",
    )
    ap.add_argument("--instruments", type=str, default=None, help="Override instruments (for example csi300/csi800).")
    ap.add_argument(
        "--state_reference_instruments",
        type=str,
        default=None,
        help="Optional shared reference universe for market-state assets (for example all). Defaults to training instruments.",
    )
    ap.add_argument("--market_index", type=str, default=None, help="Benchmark index instrument code.")
    ap.add_argument(
        "--state_reference_market_index",
        type=str,
        default=None,
        help="Benchmark index used by the shared reference-universe context asset.",
    )
    ap.add_argument(
        "--state_membership_mode",
        type=str,
        default="point_in_time",
        choices=["point_in_time"],
        help="Reference-universe membership protocol.",
    )
    ap.add_argument(
        "--state_validity_protocol",
        type=str,
        default="legacy",
        choices=["legacy", "strict"],
        help="Validity protocol for reference-universe rows.",
    )
    ap.add_argument(
        "--reference_feature_source",
        type=str,
        default="auto",
        choices=["auto", "processed_dataset", "raw_qlib_stream"],
        help=(
            "Reference feature source. auto uses the existing processed dataset for small universes, "
            "and exact two-pass raw Qlib streaming for reference=all to avoid materializing TSDatasetH(all)."
        ),
    )
    ap.add_argument(
        "--reference_instrument_chunk_size",
        type=int,
        default=800,
        help="Instrument chunk size for raw_qlib_stream day-level D.features calls.",
    )
    ap.add_argument(
        "--reference_fetch_scope",
        type=str,
        default="active_block",
        choices=["active_block", "ever_active"],
        help=(
            "raw_qlib_stream instrument query scope. active_block queries only the point-in-time "
            "active reference members in each date block; ever_active preserves the legacy behavior "
            "of querying all ever-active reference names before membership filtering."
        ),
    )
    ap.add_argument(
        "--reference_date_chunk_size",
        type=int,
        default=20,
        help="Trading-day block size for raw_qlib_stream D.features calls.",
    )
    ap.add_argument(
        "--reference_stream_tmp_dir",
        type=str,
        default=None,
        help="Optional temp directory for raw_qlib_stream exact robust-normalizer memmap files.",
    )
    ap.add_argument(
        "--reference_stream_fit_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Disk memmap dtype for exact raw_qlib_stream robust fit. float32 matches Qlib feature storage in practice.",
    )
    ap.add_argument(
        "--reference_min_raw_coverage_warn",
        type=float,
        default=0.8,
        help=(
            "Warn when a raw-stream day has n_raw_rows / n_reference_members below this threshold. "
            "Set <=0 to disable."
        ),
    )
    ap.add_argument(
        "--reference_min_fit_coverage_warn",
        type=float,
        default=0.8,
        help=(
            "Warn when raw-stream robust-normalizer fetched rows / expected active rows falls below "
            "this threshold. Set <=0 to disable."
        ),
    )
    ap.add_argument("--market_ts_windows", type=str, default="5,20", help="Comma-separated market windows.")
    ap.add_argument(
        "--market_ts_past_only",
        action="store_true",
        help="Shift context features by 1 day. Use only for same-day prediction.",
    )
    ap.add_argument("--no_norm", action="store_true", help="Remove RobustZScoreNorm from infer_processors.")
    ap.add_argument("--filter_robust_z", type=float, default=0.0, help="Robust z-score threshold (0 disables).")
    ap.add_argument("--filter_max_bad_frac", type=float, default=0.05, help="Max bad-feature fraction per stock.")
    ap.add_argument("--weight_field", type=str, default=None, help="Optional qlib field for weights (for example $amount).")
    ap.add_argument("--trade_field", type=str, default=None, help="Optional qlib field used to drop non-trading rows.")
    ap.add_argument("--min_trade", type=float, default=0.0, help="Keep rows with trade_field > min_trade.")
    ap.add_argument("--suspend_field", type=str, default=None, help="Optional qlib field for suspension flag (keep == 0).")
    ap.add_argument(
        "--min_listing_days",
        type=int,
        default=0,
        help="Minimum listing age in trading days for reference-universe rows.",
    )
    ap.add_argument(
        "--min_valid_names",
        type=int,
        default=10,
        help="Minimum valid cross-sectional names required for a reference-universe day.",
    )
    ap.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.0,
        help="Minimum valid / total ratio required for a reference-universe day.",
    )
    ap.add_argument(
        "--max_row_nan_frac",
        type=float,
        default=0.1,
        help="Maximum per-row missing fraction allowed before dropping a reference-universe row.",
    )
    ap.add_argument("--pca_dim", type=int, default=16, help="Observation PCA dim.")
    ap.add_argument("--pca_fit_on", type=str, default="train", choices=["train", "all"], help="Observation PCA fit split.")
    ap.add_argument("--field_half_lives", type=str, default="5,20,60", help="Comma-separated EMA half-lives.")
    ap.add_argument("--field_shift", type=int, default=0, help="Optional post-build shift on final field asset.")
    ap.add_argument(
        "--warmup_trading_days",
        type=int,
        default=-1,
        help="Extend train start backward for context windows and field warmup. Default uses auto inference.",
    )
    ap.add_argument("--zscore_windows", type=str, default="", help="Deprecated; ignored.")
    ap.add_argument("--roll_mean", type=int, default=0, help="Deprecated; ignored.")
    ap.add_argument("--state_delta_lags", type=str, default="", help="Deprecated; ignored.")
    ap.add_argument("--add_market_ts", action="store_true", help="Deprecated; ignored.")
    ap.add_argument("--macro_profile", type=str, default="full", help="Deprecated; ignored.")
    ap.add_argument("--macro_scale", type=str, default="none", help="Deprecated; ignored.")
    ap.add_argument("--macro_scale_fit_on", type=str, default=None, help="Deprecated; ignored.")
    return ap.parse_args()


def _deprecated_args_used(args: argparse.Namespace) -> List[str]:
    used: List[str] = []
    if args.zscore_windows:
        used.append("--zscore_windows")
    if int(args.roll_mean or 0) > 0:
        used.append("--roll_mean")
    if str(args.state_delta_lags or "").strip():
        used.append("--state_delta_lags")
    if bool(args.add_market_ts):
        used.append("--add_market_ts")
    if str(args.macro_profile or "full").strip().lower() != "full":
        used.append("--macro_profile")
    if str(args.macro_scale or "none").strip().lower() != "none":
        used.append("--macro_scale")
    if args.macro_scale_fit_on is not None:
        used.append("--macro_scale_fit_on")
    return used


def _infer_market_index(universe: str, override: Optional[str]) -> str:
    if override:
        return str(override).strip()
    key = str(universe or "").strip().lower()
    if key == "csi300":
        return "SH000300"
    if key == "csi800":
        return "SH000906"
    raise RuntimeError("--market_index is required for instruments other than csi300/csi800.")


def _resolve_asset_paths(args: argparse.Namespace, universe: str) -> dict[str, Path]:
    def _as_path(x: Optional[str]) -> Optional[Path]:
        if x is None:
            return None
        s = str(x).strip()
        return Path(s).expanduser() if s else None

    out_field = _as_path(args.out_field)
    out_deprecated = _as_path(args.out)
    if out_deprecated is not None:
        if out_field is not None and out_field != out_deprecated:
            raise ValueError("--out and --out-field point to different paths; use only one.")
        out_field = out_deprecated

    suffix = ".pkl"
    if out_field is not None:
        suffix = out_field.suffix or ".pkl"
    base_dir = out_field.parent if out_field is not None else Path("artifacts/market_state")
    base_dir.mkdir(parents=True, exist_ok=True)
    reference_universe = str(args.state_reference_instruments or universe).strip()
    field_profile = str(getattr(args, "field_profile", "stable_v1") or "stable_v1").strip().lower()
    if field_profile == "relative_complexity_v1":
        market_index = _infer_market_index(str(universe).strip(), getattr(args, "market_index", None))
        name_core = f"ref_{_slugify_name(reference_universe)}_{str(market_index).strip()}_relative_complexity_v1"
    elif field_profile == TRI_SCOPE_FIELD_PROFILE:
        market_index = _infer_market_index(str(universe).strip(), getattr(args, "market_index", None))
        name_core = f"ref_{_slugify_name(reference_universe)}_{str(market_index).strip()}_tri_scope_relative_momentum_v1"
    elif reference_universe == str(universe).strip():
        name_core = str(universe).strip()
    else:
        name_core = f"ref_{_slugify_name(reference_universe)}"
    name_suffix = f"_{name_core}{suffix}"

    out_context = _as_path(args.out_context) or base_dir / f"daily_market_context{name_suffix}"
    if field_profile == TRI_SCOPE_FIELD_PROFILE:
        out_observation = _as_path(args.out_observation) or base_dir / f"daily_market_observation_tri_scope_v1{suffix}"
    else:
        out_observation = _as_path(args.out_observation) or base_dir / f"daily_market_observation{name_suffix}"
    out_field = out_field or base_dir / f"daily_market_field{name_suffix}"
    legacy_out = _as_path(args.legacy_out)
    if legacy_out is None and field_profile == "relative_complexity_v1":
        market_index = _infer_market_index(str(universe).strip(), getattr(args, "market_index", None))
        legacy_out = base_dir / (
            f"daily_market_ref_{_slugify_name(reference_universe)}_"
            f"{str(market_index).strip()}_relative_complexity_v1_legacy{suffix}"
        )
    if legacy_out is None and field_profile == TRI_SCOPE_FIELD_PROFILE:
        market_index = _infer_market_index(str(universe).strip(), getattr(args, "market_index", None))
        legacy_out = base_dir / (
            f"daily_market_ref_{_slugify_name(reference_universe)}_"
            f"{str(market_index).strip()}_tri_scope_relative_momentum_v1_legacy{suffix}"
        )
    return {
        "context": out_context,
        "observation": out_observation,
        "field": out_field,
        "legacy": legacy_out,
    }


def _import_runtime(args: argparse.Namespace) -> RuntimeEnv:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import qlib
    from qlib.data import D
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    runtime_module = importlib.import_module(str(args.data_config_module))
    data_conf = getattr(runtime_module, str(args.data_config_attr), None)
    if data_conf is None:
        raise AttributeError(
            f"Module {args.data_config_module!r} does not define {args.data_config_attr!r}"
        )
    fallback_module = runtime_module
    if not hasattr(fallback_module, "provider_uri") or not hasattr(fallback_module, "REG_CN"):
        fallback_module = importlib.import_module("work_flow")

    qlib.init(provider_uri=fallback_module.provider_uri, region=fallback_module.REG_CN)

    orig_dc = copy.deepcopy(data_conf)
    market_windows = _normalize_market_windows(_parse_int_list(args.market_ts_windows))
    field_half_lives = _parse_int_list(args.field_half_lives)
    if int(args.warmup_trading_days) >= 0:
        warmup_days = int(args.warmup_trading_days)
    else:
        warmup_days = _auto_warmup_lookback_days(
            market_ts_windows=market_windows,
            field_half_lives=field_half_lives,
        )

    print(f"[INFO] Warmup days: {warmup_days}")

    def _prepare_dc(base_dc: Dict[str, Any], instruments: Optional[str]) -> Dict[str, Any]:
        dc_local = copy.deepcopy(base_dc)
        if instruments is not None:
            dc_local["kwargs"]["handler"]["kwargs"]["instruments"] = instruments
        dc_local["kwargs"]["step_len"] = 1
        if args.no_norm:
            ip = dc_local["kwargs"]["handler"]["kwargs"].get("infer_processors", [])
            dc_local["kwargs"]["handler"]["kwargs"]["infer_processors"] = [
                p for p in ip if not (isinstance(p, dict) and p.get("class") == "RobustZScoreNorm")
            ]
        return _extend_train_start_by_trading_days(dc_local, warmup_days=warmup_days, D=D)

    training_override = str(args.instruments).strip() if args.instruments is not None else None
    dc = _prepare_dc(orig_dc, training_override)
    dataset: Optional[Any] = None
    universe = str(dc["kwargs"]["handler"]["kwargs"].get("instruments", "")).strip()

    reference_override = (
        str(args.state_reference_instruments).strip()
        if args.state_reference_instruments is not None
        else universe
    )
    reference_dc = _prepare_dc(orig_dc, reference_override)
    reference_universe = str(reference_dc["kwargs"]["handler"]["kwargs"].get("instruments", "")).strip()
    reference_feature_source = _normalize_reference_feature_source(
        str(args.reference_feature_source),
        universe=universe,
        reference_universe=reference_universe,
    )
    field_profile = str(getattr(args, "field_profile", "stable_v1") or "stable_v1").strip().lower()
    needs_benchmark_dataset = field_profile in {"relative_complexity_v1", TRI_SCOPE_FIELD_PROFILE}
    if reference_feature_source == "raw_qlib_stream":
        reference_dataset = None
        print(
            "[INFO] Reference feature source: raw_qlib_stream "
            f"(reference_universe={reference_universe!r}); using exact two-pass Qlib processor emulation."
        )
        if needs_benchmark_dataset:
            dataset = init_instance_by_config(dc)
    else:
        dataset = init_instance_by_config(dc)
        reference_dataset = dataset if reference_universe == universe else init_instance_by_config(reference_dc)
    if reference_feature_source == "processed_dataset":
        print(f"[INFO] Reference feature source: processed_dataset (reference_universe={reference_universe!r})")

    ref_handler_kwargs = reference_dc["kwargs"]["handler"]["kwargs"]
    ref_start = pd.Timestamp(ref_handler_kwargs["start_time"]).normalize()
    ref_end = pd.Timestamp(ref_handler_kwargs["end_time"]).normalize()
    cal = pd.to_datetime(D.calendar(start_time=str(ref_start.date()), end_time=str(ref_end.date()), freq="day"))
    cal = pd.DatetimeIndex(cal).normalize().sort_values().unique()
    calendar_pos = {pd.Timestamp(ts).normalize(): i for i, ts in enumerate(cal)}

    reference_membership_spans: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if str(args.state_membership_mode).strip().lower() == "point_in_time":
        try:
            ref_instruments = D.instruments(reference_universe)
            raw_membership = D.list_instruments(
                instruments=ref_instruments,
                start_time=str(ref_start.date()),
                end_time=str(ref_end.date()),
                freq="day",
                as_list=False,
            )
            if isinstance(raw_membership, dict):
                for inst, spans in raw_membership.items():
                    normalized_spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
                    for span in spans or []:
                        if not isinstance(span, (list, tuple)) or len(span) != 2:
                            continue
                        start_ts = pd.Timestamp(span[0]).normalize()
                        end_ts = pd.Timestamp(span[1]).normalize()
                        if pd.isna(start_ts) or pd.isna(end_ts):
                            continue
                        normalized_spans.append((start_ts, end_ts))
                    if normalized_spans:
                        reference_membership_spans[str(inst)] = sorted(normalized_spans, key=lambda item: item[0])
        except Exception as exc:
            if str(args.state_validity_protocol).strip().lower() == "strict":
                raise RuntimeError(
                    f"Failed to resolve point-in-time membership for reference universe {reference_universe!r}: {exc}"
                ) from exc
            print(f"[WARN] Failed to resolve point-in-time membership for {reference_universe!r}: {exc}")

    if (
        str(args.state_validity_protocol).strip().lower() == "strict"
        and str(args.state_membership_mode).strip().lower() == "point_in_time"
        and not reference_membership_spans
    ):
        raise RuntimeError(
            "Strict reference-universe protocol requires point-in-time membership spans, but none were resolved."
        )

    train_start = pd.Timestamp(orig_dc["kwargs"]["segments"]["train"][0]).normalize()
    train_end = pd.Timestamp(orig_dc["kwargs"]["segments"]["train"][1]).normalize()

    return RuntimeEnv(
        work_flow=runtime_module,
        qlib=qlib,
        D=D,
        DataHandlerLP=DataHandlerLP,
        dataset=dataset,
        reference_dataset=reference_dataset,
        dc=dc,
        reference_dc=reference_dc,
        orig_dc=orig_dc,
        universe=universe,
        reference_universe=reference_universe,
        reference_feature_source=reference_feature_source,
        market_index=_infer_market_index(reference_universe, args.state_reference_market_index or args.market_index),
        reference_membership_spans=reference_membership_spans,
        calendar_pos=calendar_pos,
        train_range=(train_start, train_end),
        data_config_module=str(args.data_config_module),
        data_config_attr=str(args.data_config_attr),
    )


def _load_segment_feature_df(env: RuntimeEnv, seg: str, *, use_reference: bool = False) -> pd.DataFrame:
    dataset = env.reference_dataset if use_reference else env.dataset
    dc = env.reference_dc if use_reference else env.dc
    if dataset is None:
        raise RuntimeError(
            f"Segment '{seg}' requested processed dataset features, but reference_dataset is not initialized. "
            "Use the raw_qlib_stream iterator for this reference universe."
        )
    seg_start, seg_end = None, None
    segments = dc.get("kwargs", {}).get("segments", {})
    if seg in segments:
        seg_range = segments[seg]
        if isinstance(seg_range, (tuple, list)) and len(seg_range) == 2:
            seg_start = pd.Timestamp(seg_range[0]).normalize()
            seg_end = pd.Timestamp(seg_range[1]).normalize()

    df = None
    method_used = None

    tsds = None
    handler = getattr(dataset, "handler", None)
    if handler is not None:
        try:
            if seg_start is not None and seg_end is not None:
                df = handler.fetch(
                    col_set="feature",
                    data_key=env.DataHandlerLP.DK_I,
                    selector=slice(seg_start, seg_end),
                )
            else:
                df = handler.fetch(col_set="feature", data_key=env.DataHandlerLP.DK_I)
            if isinstance(df, pd.DataFrame) and not df.empty:
                method_used = "handler.fetch"
            else:
                df = None
        except Exception as e:
            print(f"[DEBUG] handler.fetch failed for {seg}: {e}")
            df = None

    if df is None:
        try:
            tsds = dataset.prepare(seg, col_set=["feature"], data_key=env.DataHandlerLP.DK_I)
            df = getattr(tsds, "data", None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                method_used = "tsds.data"
            else:
                df = None
        except Exception as e:
            print(f"[DEBUG] dataset.prepare failed for {seg}: {e}")
            df = None

    if df is None:
        try:
            if tsds is None:
                tsds = dataset.prepare(seg, col_set=["feature"], data_key=env.DataHandlerLP.DK_I)
            idx = tsds.get_index()
            samples: List[np.ndarray] = []
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
                    x_np = np.asarray(x, dtype=float)
                    if x_np.ndim == 2:
                        x_np = x_np[-1, :]
                    samples.append(x_np)
            if samples:
                df = pd.DataFrame(np.stack(samples, axis=0), index=idx)
                method_used = "iter-tsds"
        except Exception as e:
            print(f"[DEBUG] TSDataSampler iteration failed for {seg}: {e}")
            df = None

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Segment '{seg}' returned no valid DataFrame.")
    df = _filter_df_by_datetime(df, seg_start, seg_end)
    if df.empty:
        raise RuntimeError(f"Segment '{seg}' became empty after datetime filtering.")
    if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
        raise RuntimeError(f"Expected MultiIndex with 'datetime' level in segment '{seg}'.")
    print(f"[INFO] Segment '{seg}': loaded shape={df.shape} via {method_used}")
    return df


def _load_aux_df(env: RuntimeEnv, df: pd.DataFrame, args: argparse.Namespace) -> Optional[pd.DataFrame]:
    aux_cols: List[str] = []
    if args.weight_field:
        aux_cols.append(args.weight_field)
    if args.trade_field and args.trade_field not in aux_cols:
        aux_cols.append(args.trade_field)
    if args.suspend_field and args.suspend_field not in aux_cols:
        aux_cols.append(args.suspend_field)
    strict_protocol = str(args.state_validity_protocol).strip().lower() == "strict"
    if strict_protocol:
        for field in ("$close", "$volume", "$amount"):
            if field not in aux_cols:
                aux_cols.append(field)
    if not aux_cols:
        return None

    dt_level = df.index.names.index("datetime")
    inst_level = 0 if dt_level == 1 else 1
    inst_name = df.index.names[inst_level]
    insts = df.index.get_level_values(inst_name).unique().tolist()
    start = str(pd.to_datetime(df.index.get_level_values("datetime")).min().date())
    end = str(pd.to_datetime(df.index.get_level_values("datetime")).max().date())
    aux_df = env.D.features(insts, aux_cols, start_time=start, end_time=end, freq="day")
    rename = {}
    if args.weight_field:
        rename[args.weight_field] = "_w"
    if args.trade_field:
        rename[args.trade_field] = "_trade"
    if args.suspend_field:
        rename[args.suspend_field] = "_susp"
    if strict_protocol:
        if "$close" not in rename:
            rename["$close"] = "_close"
        if "$volume" not in rename:
            rename["$volume"] = "_volume"
        if "$amount" not in rename:
            rename["$amount"] = "_amount"
    return aux_df.rename(columns=rename)


def _membership_age_for_day(
    *,
    spans_by_inst: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    calendar_pos: Dict[pd.Timestamp, int],
    instrument: str,
    dt: pd.Timestamp,
) -> float:
    cur_dt = pd.Timestamp(dt).normalize()
    cur_pos = calendar_pos.get(cur_dt)
    if cur_pos is None:
        return np.nan
    for start_dt, end_dt in spans_by_inst.get(str(instrument), []):
        if start_dt <= cur_dt <= end_dt:
            start_norm = pd.Timestamp(start_dt).normalize()
            start_pos = calendar_pos.get(start_norm)
            if start_pos is None:
                cal = getattr(_membership_age_for_day, "_cached_calendar", None)
                cal_key = getattr(_membership_age_for_day, "_cached_calendar_key", None)
                if cal is None or cal_key is not calendar_pos:
                    cal = pd.DatetimeIndex(sorted(calendar_pos.keys()))
                    setattr(_membership_age_for_day, "_cached_calendar", cal)
                    setattr(_membership_age_for_day, "_cached_calendar_key", calendar_pos)
                if len(cal) <= 0:
                    return np.nan
                if start_norm < cal[0]:
                    # Listed before the configured reference window: it is not a young listing.
                    start_pos = -10_000_000
                else:
                    insert_at = int(cal.searchsorted(start_norm, side="left"))
                    if insert_at >= len(cal):
                        return np.nan
                    start_pos = calendar_pos.get(pd.Timestamp(cal[insert_at]).normalize())
            if start_pos is None:
                return np.nan
            return float(max(0, cur_pos - start_pos))
    return np.nan


def _apply_daily_filters(
    x: np.ndarray,
    *,
    w: Optional[np.ndarray],
    trade: Optional[np.ndarray],
    susp: Optional[np.ndarray],
    close: Optional[np.ndarray],
    volume: Optional[np.ndarray],
    amount: Optional[np.ndarray],
    membership_age: Optional[np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, float]]:
    strict_protocol = str(args.state_validity_protocol).strip().lower() == "strict"
    x = np.asarray(x, dtype=float)
    stats: dict[str, float] = {
        "n_total": int(x.shape[0]),
        "n_missing": 0,
        "n_membership_invalid": 0,
        "n_young_listing": 0,
        "n_nontrading": 0,
        "n_suspended": 0,
        "n_outlier": 0,
        "n_valid": int(x.shape[0]),
        "valid_ratio": 1.0 if x.shape[0] > 0 else 0.0,
    }

    def _apply_mask(mask: np.ndarray, *, stat_key: str) -> None:
        nonlocal x, w, trade, susp, close, volume, amount, membership_age, stats
        mask = np.asarray(mask, dtype=bool)
        removed = int(mask.shape[0] - np.count_nonzero(mask))
        stats[stat_key] += removed
        x = x[mask]
        if w is not None:
            w = np.asarray(w, dtype=float)[mask]
        if trade is not None:
            trade = np.asarray(trade, dtype=float)[mask]
        if susp is not None:
            susp = np.asarray(susp, dtype=float)[mask]
        if close is not None:
            close = np.asarray(close, dtype=float)[mask]
        if volume is not None:
            volume = np.asarray(volume, dtype=float)[mask]
        if amount is not None:
            amount = np.asarray(amount, dtype=float)[mask]
        if membership_age is not None:
            membership_age = np.asarray(membership_age, dtype=float)[mask]

    row_nan = np.mean(~np.isfinite(x), axis=1)
    _apply_mask(row_nan < float(args.max_row_nan_frac), stat_key="n_missing")

    if strict_protocol:
        if membership_age is None:
            raise RuntimeError("Strict validity protocol requires point-in-time membership age.")
        age = np.asarray(membership_age, dtype=float)
        _apply_mask(np.isfinite(age), stat_key="n_membership_invalid")
        if int(args.min_listing_days or 0) > 0:
            age = np.asarray(membership_age, dtype=float)
            _apply_mask(age >= int(args.min_listing_days), stat_key="n_young_listing")

    trade_keep: Optional[np.ndarray] = None
    if trade is not None:
        t = np.asarray(trade, dtype=float)
        trade_keep = np.isfinite(t) & (t > float(args.min_trade))
    elif strict_protocol:
        close_arr = np.asarray(close, dtype=float) if close is not None else np.full((x.shape[0],), np.nan, dtype=float)
        volume_arr = np.asarray(volume, dtype=float) if volume is not None else np.full((x.shape[0],), np.nan, dtype=float)
        amount_arr = np.asarray(amount, dtype=float) if amount is not None else np.full((x.shape[0],), np.nan, dtype=float)
        trade_keep = np.isfinite(close_arr) & (
            (np.isfinite(volume_arr) & (volume_arr > 0.0)) | (np.isfinite(amount_arr) & (amount_arr > 0.0))
        )
    if trade_keep is not None:
        _apply_mask(trade_keep, stat_key="n_nontrading")

    if susp is not None:
        s = np.asarray(susp, dtype=float)
        _apply_mask(np.isfinite(s) & (s == 0), stat_key="n_suspended")

    if args.filter_robust_z and args.filter_robust_z > 0 and x.shape[0] > 0:
        mask = _robust_filter_mask(
            x,
            z_thresh=float(args.filter_robust_z),
            max_bad_frac=float(args.filter_max_bad_frac),
        )
        _apply_mask(mask, stat_key="n_outlier")

    stats["n_valid"] = int(x.shape[0])
    stats["valid_ratio"] = float(stats["n_valid"] / stats["n_total"]) if stats["n_total"] > 0 else 0.0
    return x, w, stats


def _segment_range(dc: Dict[str, Any], seg: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    segments = (dc.get("kwargs", {}) or {}).get("segments", {}) or {}
    if seg not in segments:
        raise RuntimeError(f"Segment {seg!r} is not defined in data config.")
    rng = segments[seg]
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        raise RuntimeError(f"Segment {seg!r} must be a (start, end) tuple, got: {rng!r}")
    return pd.Timestamp(rng[0]).normalize(), pd.Timestamp(rng[1]).normalize()


def _iter_calendar(env: RuntimeEnv, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    cal = pd.to_datetime(env.D.calendar(start_time=str(start.date()), end_time=str(end.date()), freq="day"))
    return pd.DatetimeIndex(cal).normalize().sort_values().unique()


def _iter_segment_calendar(env: RuntimeEnv, seg: str) -> pd.DatetimeIndex:
    start, end = _segment_range(env.reference_dc, seg)
    return _iter_calendar(env, start, end)


def _active_reference_members_for_day(env: RuntimeEnv, dt: pd.Timestamp) -> List[str]:
    if not env.reference_membership_spans:
        raise RuntimeError(
            "raw_qlib_stream requires point-in-time reference membership spans. "
            "Use state_membership_mode=point_in_time and a resolvable reference universe."
        )
    dtn = pd.Timestamp(dt).normalize()
    cache = getattr(env, "_active_reference_members_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_active_reference_members_cache", cache)
    if dtn in cache:
        return list(cache[dtn])
    members: List[str] = []
    for inst, spans in env.reference_membership_spans.items():
        for start_dt, end_dt in spans:
            if start_dt <= dtn <= end_dt:
                members.append(str(inst))
                break
    cache[dtn] = tuple(members)
    return members


def _active_reference_members_by_day(
    env: RuntimeEnv,
    dates: Sequence[pd.Timestamp],
) -> Dict[pd.Timestamp, List[str]]:
    return {
        pd.Timestamp(dt).normalize(): _active_reference_members_for_day(env, pd.Timestamp(dt).normalize())
        for dt in dates
    }


def _raw_stream_fetch_instruments_for_block(
    env: RuntimeEnv,
    args: argparse.Namespace,
    active_by_day: Dict[pd.Timestamp, List[str]],
) -> List[str]:
    scope = str(getattr(args, "reference_fetch_scope", "active_block") or "active_block").strip().lower()
    if scope == "ever_active":
        return sorted(str(x) for x in env.reference_membership_spans.keys())
    if scope != "active_block":
        raise ValueError(f"Unsupported --reference_fetch_scope: {scope!r}")
    block_members: set[str] = set()
    for members in active_by_day.values():
        block_members.update(str(x) for x in members)
    return sorted(block_members)


def _requested_aux_cols(args: argparse.Namespace) -> List[str]:
    aux_cols: List[str] = []
    for field in (args.weight_field, args.trade_field, args.suspend_field):
        if field and field not in aux_cols:
            aux_cols.append(field)
    if str(args.state_validity_protocol).strip().lower() == "strict":
        for field in ("$close", "$volume", "$amount"):
            if field not in aux_cols:
                aux_cols.append(field)
    return aux_cols


def _feature_config_from_data_config(env: RuntimeEnv) -> tuple[List[str], List[str], str]:
    handler = (((env.orig_dc or {}).get("kwargs") or {}).get("handler") or {})
    handler_class = str(handler.get("class", "")).strip()
    handler_module = str(handler.get("module_path", "qlib.contrib.data.handler")).strip()
    if handler_class != "Alpha158":
        raise RuntimeError(
            "raw_qlib_stream currently supports Alpha158 only. "
            f"Got handler class={handler_class!r} module={handler_module!r}."
        )
    try:
        mod = importlib.import_module(handler_module)
        cls = getattr(mod, handler_class)
        fields, names = cls.get_feature_config(None)
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve Alpha158 feature config for raw_qlib_stream: {exc}") from exc
    fields = [str(x) for x in fields]
    names = [str(x) for x in names]
    if len(fields) != len(names) or not fields:
        raise RuntimeError(f"Invalid feature config for {handler_class}: fields={len(fields)}, names={len(names)}")
    return fields, names, f"{handler_module}.{handler_class}"


def _iter_infer_processors(dc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((((dc.get("kwargs", {}) or {}).get("handler", {}) or {}).get("kwargs", {}) or {}).get("infer_processors", []) or [])


def _stream_feature_pipeline(env: RuntimeEnv) -> StreamFeaturePipeline:
    feature_exprs, feature_names, provider = _feature_config_from_data_config(env)
    handler_kwargs = (((env.reference_dc or {}).get("kwargs") or {}).get("handler", {}) or {}).get("kwargs", {}) or {}
    processors = _iter_infer_processors(env.reference_dc)
    supported = {"RobustZScoreNorm", "Fillna"}
    unsupported = [
        str(p.get("class"))
        for p in processors
        if isinstance(p, dict) and str(p.get("class")) not in supported
    ]
    if unsupported:
        raise RuntimeError(
            "raw_qlib_stream cannot exactly emulate unsupported infer_processors: "
            f"{unsupported}. Use processed_dataset or add explicit streaming support."
        )

    robust_cfg = next(
        (p for p in processors if isinstance(p, dict) and str(p.get("class")) == "RobustZScoreNorm"),
        None,
    )
    fill_cfg = next(
        (p for p in processors if isinstance(p, dict) and str(p.get("class")) == "Fillna"),
        None,
    )

    robust_enabled = robust_cfg is not None
    clip_outlier = True
    if robust_cfg is not None:
        kwargs = robust_cfg.get("kwargs", {}) or {}
        if kwargs.get("fields_group", "feature") not in (None, "feature"):
            raise RuntimeError("raw_qlib_stream only supports RobustZScoreNorm on fields_group='feature'.")
        clip_outlier = bool(kwargs.get("clip_outlier", True))

    fillna_enabled = fill_cfg is not None
    fill_value = 0.0
    if fill_cfg is not None:
        kwargs = fill_cfg.get("kwargs", {}) or {}
        if kwargs.get("fields_group", "feature") not in (None, "feature"):
            raise RuntimeError("raw_qlib_stream only supports Fillna on fields_group='feature'.")
        fill_value = float(kwargs.get("fill_value", 0.0))

    fit_start = pd.Timestamp(handler_kwargs.get("fit_start_time", handler_kwargs.get("start_time"))).normalize()
    fit_end = pd.Timestamp(handler_kwargs.get("fit_end_time", env.train_range[1])).normalize()
    if pd.isna(fit_start) or pd.isna(fit_end) or fit_start > fit_end:
        raise RuntimeError(f"Invalid raw_qlib_stream fit window: {fit_start}..{fit_end}")

    return StreamFeaturePipeline(
        feature_exprs=feature_exprs,
        feature_names=feature_names,
        provider=provider,
        robust_enabled=robust_enabled,
        clip_outlier=clip_outlier,
        fillna_enabled=fillna_enabled,
        fill_value=fill_value,
        fit_start=fit_start,
        fit_end=fit_end,
    )


def _to_instrument_indexed_frame(df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names or [])
        if "datetime" in names:
            dt_level = names.index("datetime")
            dts = pd.to_datetime(out.index.get_level_values(dt_level)).normalize()
            out = out.loc[dts == pd.Timestamp(dt).normalize()]
            names = list(out.index.names or [])
        inst_level = None
        for candidate in ("instrument", "symbol"):
            if candidate in names:
                inst_level = names.index(candidate)
                break
        if inst_level is None:
            inst_level = 0 if ("datetime" not in names or names.index("datetime") != 0) else 1
        out.index = pd.Index(out.index.get_level_values(inst_level).astype(str), name="instrument")
    else:
        out.index = pd.Index(out.index.astype(str), name="instrument")
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _to_datetime_instrument_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.MultiIndex):
        raise RuntimeError("raw_qlib_stream expected D.features to return a MultiIndex DataFrame.")
    names = list(out.index.names or [])
    dt_level = names.index("datetime") if "datetime" in names else None
    inst_level = None
    for candidate in ("instrument", "symbol"):
        if candidate in names:
            inst_level = names.index(candidate)
            break
    if dt_level is None or inst_level is None:
        if len(names) < 2:
            raise RuntimeError(f"Cannot infer datetime/instrument levels from index names={names}.")
        # Qlib D.features commonly returns (instrument, datetime).
        inst_level = 0
        dt_level = 1
    dts = pd.to_datetime(out.index.get_level_values(dt_level)).normalize()
    insts = out.index.get_level_values(inst_level).astype(str)
    out.index = pd.MultiIndex.from_arrays([dts, insts], names=["datetime", "instrument"])
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _fetch_raw_qlib_day_frame(
    env: RuntimeEnv,
    *,
    instruments: Sequence[str],
    fields: Sequence[str],
    dt: pd.Timestamp,
    chunk_size: int,
) -> pd.DataFrame:
    insts = [str(x) for x in instruments]
    if not insts:
        return pd.DataFrame()
    chunk = max(1, int(chunk_size or 800))
    parts: List[pd.DataFrame] = []
    dts = str(pd.Timestamp(dt).date())
    for i in range(0, len(insts), chunk):
        sub_insts = insts[i : i + chunk]
        df = env.D.features(sub_insts, list(fields), start_time=dts, end_time=dts, freq="day")
        if isinstance(df, pd.DataFrame) and not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return _to_instrument_indexed_frame(pd.concat(parts, axis=0), pd.Timestamp(dt).normalize())


def _fetch_raw_qlib_block_frame(
    env: RuntimeEnv,
    *,
    instruments: Sequence[str],
    fields: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    insts = [str(x) for x in instruments]
    if not insts:
        return pd.DataFrame()
    df = env.D.features(
        insts,
        list(fields),
        start_time=str(pd.Timestamp(start).date()),
        end_time=str(pd.Timestamp(end).date()),
        freq="day",
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    return _to_datetime_instrument_frame(df)


def _chunked(values: Sequence[Any], chunk_size: int) -> List[Sequence[Any]]:
    chunk = max(1, int(chunk_size or 1))
    return [values[i : i + chunk] for i in range(0, len(values), chunk)]


def _active_membership_mask_and_age(env: RuntimeEnv, index: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(index, pd.MultiIndex) or "datetime" not in (index.names or []) or "instrument" not in (index.names or []):
        raise RuntimeError("Expected MultiIndex ['datetime', 'instrument'] for membership filtering.")
    dts = pd.to_datetime(index.get_level_values("datetime")).normalize()
    insts = index.get_level_values("instrument").astype(str)
    mask = np.zeros((len(index),), dtype=bool)
    ages = np.full((len(index),), np.nan, dtype=float)
    for i, (dt, inst) in enumerate(zip(dts, insts)):
        age = _membership_age_for_day(
            spans_by_inst=env.reference_membership_spans,
            calendar_pos=env.calendar_pos,
            instrument=str(inst),
            dt=pd.Timestamp(dt).normalize(),
        )
        if np.isfinite(age):
            mask[i] = True
            ages[i] = float(age)
    return mask, ages


def _feature_array_from_raw_day(raw_day: pd.DataFrame, pipeline: StreamFeaturePipeline) -> np.ndarray:
    if raw_day.empty:
        return np.empty((0, len(pipeline.feature_exprs)), dtype=float)
    if all(c in raw_day.columns for c in pipeline.feature_exprs):
        feature_df = raw_day.reindex(columns=pipeline.feature_exprs)
    elif all(c in raw_day.columns for c in pipeline.feature_names):
        feature_df = raw_day.reindex(columns=pipeline.feature_names)
    else:
        missing = [c for c in pipeline.feature_exprs[:10] if c not in raw_day.columns]
        raise RuntimeError(
            "raw_qlib_stream fetched features do not match Alpha158 expressions or names; "
            f"example missing={missing}, columns_sample={[str(c) for c in list(raw_day.columns[:10])]}"
        )
    return feature_df.to_numpy(dtype=float)


def _aux_arrays_from_frame(raw_day: pd.DataFrame, args: argparse.Namespace) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    def _col(field: Optional[str]) -> Optional[np.ndarray]:
        if field and field in raw_day.columns:
            return pd.to_numeric(raw_day[field], errors="coerce").to_numpy(dtype=float)
        return None

    w = _col(args.weight_field)
    trade = _col(args.trade_field)
    susp = _col(args.suspend_field)
    close = _col("$close")
    volume = _col("$volume")
    amount = _col("$amount")
    return w, trade, susp, close, volume, amount


def _transform_stream_features(x: np.ndarray, pipeline: StreamFeaturePipeline) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    if pipeline.robust_enabled:
        if pipeline.center is None or pipeline.scale is None:
            raise RuntimeError("raw_qlib_stream robust pipeline has not been fit.")
        out -= pipeline.center[None, :]
        out /= pipeline.scale[None, :]
        if pipeline.clip_outlier:
            out = np.clip(out, -3.0, 3.0)
    if pipeline.fillna_enabled:
        out = np.where(np.isnan(out), float(pipeline.fill_value), out)
    return out


def _fit_stream_robust_normalizer(env: RuntimeEnv, args: argparse.Namespace, pipeline: StreamFeaturePipeline) -> None:
    if not pipeline.robust_enabled:
        pipeline.center = np.zeros((len(pipeline.feature_exprs),), dtype=float)
        pipeline.scale = np.ones((len(pipeline.feature_exprs),), dtype=float)
        return

    fit_dates = _iter_calendar(env, pipeline.fit_start, pipeline.fit_end)
    total_rows = 0
    for dt in fit_dates:
        total_rows += len(_active_reference_members_for_day(env, pd.Timestamp(dt).normalize()))
    if total_rows <= 0:
        raise RuntimeError(
            f"raw_qlib_stream robust fit has no reference rows in {pipeline.fit_start.date()}..{pipeline.fit_end.date()}."
        )

    n_features = len(pipeline.feature_exprs)
    chunk_size = max(1, int(args.reference_instrument_chunk_size or 800))
    date_chunk_size = max(1, int(args.reference_date_chunk_size or 20))
    dtype = np.float32 if str(args.reference_stream_fit_dtype) == "float32" else np.float64
    tmp_parent = Path(args.reference_stream_tmp_dir).resolve() if args.reference_stream_tmp_dir else None
    if tmp_parent is not None:
        tmp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="qlib_ref_all_fit_",
        dir=str(tmp_parent) if tmp_parent else None,
        ignore_cleanup_errors=True,
    ) as tmpdir:
        mmap_path = Path(tmpdir) / "feature_fit_col_major.dat"
        fit_mat = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(n_features, total_rows))
        offset = 0
        date_blocks = _chunked(list(fit_dates), date_chunk_size)
        for date_block in tqdm(date_blocks, desc="Fit robust normalizer raw-stream", unit="block"):
            start_dt = pd.Timestamp(date_block[0]).normalize()
            end_dt = pd.Timestamp(date_block[-1]).normalize()
            block_dates = [pd.Timestamp(dt).normalize() for dt in date_block]
            active_by_day = _active_reference_members_by_day(env, block_dates)
            block_instruments = _raw_stream_fetch_instruments_for_block(env, args, active_by_day)
            for inst_block in _chunked(block_instruments, chunk_size):
                raw_block = _fetch_raw_qlib_block_frame(
                    env,
                    instruments=inst_block,
                    fields=pipeline.feature_exprs,
                    start=start_dt,
                    end=end_dt,
                )
                if raw_block.empty:
                    continue
                active_mask, _ = _active_membership_mask_and_age(env, raw_block.index)
                if not bool(np.any(active_mask)):
                    continue
                raw_block = raw_block.loc[active_mask]
                x = _feature_array_from_raw_day(raw_block, pipeline).astype(dtype, copy=False)
                n = int(x.shape[0])
                if n <= 0:
                    continue
                if offset + n > total_rows:
                    raise RuntimeError(
                        "raw_qlib_stream robust fit fetched more active rows than expected "
                        f"({offset + n}>{total_rows}). Check membership spans and query duplicates."
                    )
                fit_mat[:, offset : offset + n] = x.T
                offset += n
        fit_mat.flush()

        if offset <= 0:
            raise RuntimeError("raw_qlib_stream robust fit fetched zero feature rows.")
        pipeline.fit_expected_rows = int(total_rows)
        pipeline.fit_rows = int(offset)
        pipeline.fit_fetch_coverage = float(offset / total_rows) if total_rows > 0 else float("nan")
        warn_threshold = float(getattr(args, "reference_min_fit_coverage_warn", 0.0) or 0.0)
        if warn_threshold > 0 and np.isfinite(pipeline.fit_fetch_coverage) and pipeline.fit_fetch_coverage < warn_threshold:
            print(
                "[WARN] raw_qlib_stream robust fit low fetch coverage: "
                f"fetched={offset}, expected={total_rows}, coverage={pipeline.fit_fetch_coverage:.4f}, "
                f"threshold={warn_threshold:.4f}, fetch_scope={getattr(args, 'reference_fetch_scope', 'active_block')}"
            )
        view = fit_mat[:, :offset]
        try:
            center = np.empty((n_features,), dtype=float)
            scale = np.empty((n_features,), dtype=float)
            for j in tqdm(range(n_features), desc="Finalize robust normalizer", unit="feature"):
                col = np.asarray(view[j], dtype=float)
                med = np.nanmedian(col)
                mad = np.nanmedian(np.abs(col - med))
                center[j] = med
                scale[j] = (mad + 1e-12) * 1.4826
            pipeline.center = center
            pipeline.scale = scale
        finally:
            # Windows keeps memmap-backed files locked until all views and the
            # underlying mmap handle are released explicitly.
            del view
            fit_mat.flush()
            mmap_obj = getattr(fit_mat, "_mmap", None)
            del fit_mat
            if mmap_obj is not None:
                mmap_obj.close()


def _iter_daily_feature_matrices_raw_qlib_stream(
    env: RuntimeEnv,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], list[np.ndarray], list[pd.Timestamp], pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    pipeline = _stream_feature_pipeline(env)
    _fit_stream_robust_normalizer(env, args, pipeline)
    aux_cols = _requested_aux_cols(args)
    fetch_fields = list(dict.fromkeys([*pipeline.feature_exprs, *aux_cols]))
    rows: List[Dict[str, float]] = []
    factor_vecs: List[np.ndarray] = []
    idx_list: List[pd.Timestamp] = []
    daily_filter_rows: List[Dict[str, Any]] = []
    daily_filter_index: List[pd.Timestamp] = []
    min_dt: Optional[pd.Timestamp] = None
    max_dt: Optional[pd.Timestamp] = None
    chunk_size = max(1, int(args.reference_instrument_chunk_size or 800))
    date_chunk_size = max(1, int(args.reference_date_chunk_size or 20))
    seen_dates: set[pd.Timestamp] = set()
    low_raw_coverage_days: List[Tuple[pd.Timestamp, float, int, int]] = []

    print(
        "[INFO] raw_qlib_stream transform: "
        f"features={len(pipeline.feature_exprs)}, aux={aux_cols}, provider={pipeline.provider}, "
        f"robust_enabled={pipeline.robust_enabled}, fit_rows={pipeline.fit_rows}, "
        f"fit_expected_rows={pipeline.fit_expected_rows}, fit_fetch_coverage={pipeline.fit_fetch_coverage:.4f}, "
        f"chunk_size={chunk_size}, fetch_scope={getattr(args, 'reference_fetch_scope', 'active_block')}"
    )

    for seg in ("train", "valid", "test"):
        dates = [
            pd.Timestamp(dt).normalize()
            for dt in _iter_segment_calendar(env, seg)
            if pd.Timestamp(dt).normalize() not in seen_dates
        ]
        for date_block in tqdm(_chunked(dates, date_chunk_size), desc=f"Observation {seg} raw-stream", unit="block"):
            block_dates = [pd.Timestamp(dt).normalize() for dt in date_block]
            if not block_dates:
                continue
            for dtn in block_dates:
                seen_dates.add(dtn)
                min_dt = dtn if min_dt is None else min(min_dt, dtn)
                max_dt = dtn if max_dt is None else max(max_dt, dtn)

            active_by_day = _active_reference_members_by_day(env, block_dates)
            day_parts: Dict[pd.Timestamp, Dict[str, Any]] = {
                dtn: {
                    "x": [],
                    "w": [],
                    "trade": [],
                    "susp": [],
                    "close": [],
                    "volume": [],
                    "amount": [],
                    "age": [],
                    "raw_rows": 0,
                    "members": len(active_by_day[dtn]),
                }
                for dtn in block_dates
            }

            start_dt = block_dates[0]
            end_dt = block_dates[-1]
            block_instruments = _raw_stream_fetch_instruments_for_block(env, args, active_by_day)
            for inst_block in _chunked(block_instruments, chunk_size):
                raw_block = _fetch_raw_qlib_block_frame(
                    env,
                    instruments=inst_block,
                    fields=fetch_fields,
                    start=start_dt,
                    end=end_dt,
                )
                if raw_block.empty:
                    continue
                active_mask, membership_age = _active_membership_mask_and_age(env, raw_block.index)
                if not bool(np.any(active_mask)):
                    continue
                raw_block = raw_block.loc[active_mask]
                membership_age = membership_age[active_mask]
                x_all = _transform_stream_features(_feature_array_from_raw_day(raw_block, pipeline), pipeline)
                w_all, trade_all, susp_all, close_all, volume_all, amount_all = _aux_arrays_from_frame(raw_block, args)
                dts = pd.to_datetime(raw_block.index.get_level_values("datetime")).normalize()
                for dtn in block_dates:
                    mask = np.asarray(dts == dtn, dtype=bool)
                    if not bool(np.any(mask)):
                        continue
                    part = day_parts[dtn]
                    part["x"].append(x_all[mask])
                    part["age"].append(membership_age[mask])
                    part["raw_rows"] += int(np.count_nonzero(mask))
                    for key, arr in (
                        ("w", w_all),
                        ("trade", trade_all),
                        ("susp", susp_all),
                        ("close", close_all),
                        ("volume", volume_all),
                        ("amount", amount_all),
                    ):
                        if arr is not None:
                            part[key].append(arr[mask])

            for dtn in block_dates:
                part = day_parts[dtn]
                if part["x"]:
                    x = np.vstack(part["x"])
                    membership_age = np.concatenate(part["age"]) if part["age"] else None
                    w = np.concatenate(part["w"]) if part["w"] else None
                    trade = np.concatenate(part["trade"]) if part["trade"] else None
                    susp = np.concatenate(part["susp"]) if part["susp"] else None
                    close = np.concatenate(part["close"]) if part["close"] else None
                    volume = np.concatenate(part["volume"]) if part["volume"] else None
                    amount = np.concatenate(part["amount"]) if part["amount"] else None
                    x, w, stats = _apply_daily_filters(
                        x,
                        w=w,
                        trade=trade,
                        susp=susp,
                        close=close,
                        volume=volume,
                        amount=amount,
                        membership_age=membership_age,
                        args=args,
                    )
                else:
                    x = np.empty((0, len(pipeline.feature_exprs)), dtype=float)
                    w = None
                    stats = {
                        "n_total": 0,
                        "n_missing": 0,
                        "n_membership_invalid": 0,
                        "n_young_listing": 0,
                        "n_nontrading": 0,
                        "n_suspended": 0,
                        "n_outlier": 0,
                        "n_valid": 0,
                        "valid_ratio": 0.0,
                    }

                day_stats = {
                    "segment": seg,
                    "date": str(dtn.date()),
                    "reference_universe": env.reference_universe,
                    "training_universe": env.universe,
                    "reference_feature_source": env.reference_feature_source,
                    "reference_feature_provider": pipeline.provider,
                    "reference_feature_semantics": "qlib_dk_i_robust_zscore_fillna_stream_exact",
                    "n_reference_members": int(part["members"]),
                    "n_raw_rows": int(part["raw_rows"]),
                    "raw_coverage": (
                        float(part["raw_rows"] / part["members"])
                        if int(part["members"]) > 0
                        else 0.0
                    ),
                    **stats,
                    "min_valid_names_required": int(max(10, args.min_valid_names)),
                    "min_valid_ratio_required": float(max(0.0, args.min_valid_ratio)),
                }
                warn_threshold = float(getattr(args, "reference_min_raw_coverage_warn", 0.0) or 0.0)
                if (
                    warn_threshold > 0
                    and int(day_stats["n_reference_members"]) > 0
                    and float(day_stats["raw_coverage"]) < warn_threshold
                ):
                    low_raw_coverage_days.append(
                        (
                            dtn,
                            float(day_stats["raw_coverage"]),
                            int(day_stats["n_raw_rows"]),
                            int(day_stats["n_reference_members"]),
                        )
                    )
                day_stats["accepted"] = bool(
                    x.shape[0] >= max(10, int(args.min_valid_names))
                    and float(stats["valid_ratio"]) >= float(max(0.0, args.min_valid_ratio))
                )
                daily_filter_rows.append(day_stats)
                daily_filter_index.append(dtn)
                if not day_stats["accepted"]:
                    continue
                rows.append(_agg_global(x, w))
                rows[-1].update(_corr_summaries(x))
                factor_vecs.append(_factor_stats(x, w))
                idx_list.append(dtn)

    if not idx_list or min_dt is None or max_dt is None:
        raise RuntimeError("No daily observation states computed; check filters and data coverage.")
    daily_filter_df = pd.DataFrame(daily_filter_rows, index=pd.to_datetime(daily_filter_index)).sort_index()
    if low_raw_coverage_days:
        first = low_raw_coverage_days[0]
        print(
            "[WARN] raw_qlib_stream low raw coverage days: "
            f"count={len(low_raw_coverage_days)}, first={first[0].date()}, "
            f"coverage={first[1]:.4f}, raw_rows={first[2]}, reference_members={first[3]}, "
            f"threshold={float(getattr(args, 'reference_min_raw_coverage_warn', 0.0) or 0.0):.4f}"
        )
    daily_filter_df.attrs.update(
        {
            "reference_fetch_scope": str(getattr(args, "reference_fetch_scope", "active_block")),
            "reference_min_raw_coverage_warn": float(getattr(args, "reference_min_raw_coverage_warn", 0.0) or 0.0),
            "reference_min_fit_coverage_warn": float(getattr(args, "reference_min_fit_coverage_warn", 0.0) or 0.0),
            "stream_fit_expected_rows": int(pipeline.fit_expected_rows),
            "stream_fit_rows": int(pipeline.fit_rows),
            "stream_fit_fetch_coverage": float(pipeline.fit_fetch_coverage),
            "low_raw_coverage_day_count": int(len(low_raw_coverage_days)),
        }
    )
    return rows, factor_vecs, idx_list, min_dt, max_dt, daily_filter_df


def _iter_daily_feature_matrices(
    env: RuntimeEnv,
    args: argparse.Namespace,
    *,
    use_reference: bool = True,
) -> tuple[list[dict[str, float]], list[np.ndarray], list[pd.Timestamp], pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    if use_reference and env.reference_feature_source == "raw_qlib_stream":
        return _iter_daily_feature_matrices_raw_qlib_stream(env, args)

    rows: List[Dict[str, float]] = []
    factor_vecs: List[np.ndarray] = []
    idx_list: List[pd.Timestamp] = []
    daily_filter_rows: List[Dict[str, Any]] = []
    daily_filter_index: List[pd.Timestamp] = []
    min_dt: Optional[pd.Timestamp] = None
    max_dt: Optional[pd.Timestamp] = None
    seen_dates: set[pd.Timestamp] = set()

    for seg in ("train", "valid", "test"):
        df = _load_segment_feature_df(env, seg, use_reference=use_reference)
        aux_df = _load_aux_df(env, df, args)
        dt_level = df.index.names.index("datetime")
        inst_level = 0 if dt_level == 1 else 1
        inst_name = df.index.names[inst_level]
        for dt, sub in tqdm(df.groupby(level="datetime"), desc=f"Observation {seg}", unit="day"):
            dtn = pd.Timestamp(dt).normalize()
            if dtn in seen_dates:
                continue
            seen_dates.add(dtn)
            min_dt = dtn if min_dt is None else min(min_dt, dtn)
            max_dt = dtn if max_dt is None else max(max_dt, dtn)
            x = sub.to_numpy(dtype=float)
            insts = [str(inst) for inst in sub.index.get_level_values(inst_name)]
            membership_age = None
            if env.reference_membership_spans:
                membership_age = np.asarray(
                    [
                        _membership_age_for_day(
                            spans_by_inst=env.reference_membership_spans,
                            calendar_pos=env.calendar_pos,
                            instrument=inst,
                            dt=dtn,
                        )
                        for inst in insts
                    ],
                    dtype=float,
                )
            w = trade = susp = close = volume = amount = None
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
                if "_close" in aux_sub.columns:
                    close = aux_sub["_close"].to_numpy(dtype=float)
                if "_volume" in aux_sub.columns:
                    volume = aux_sub["_volume"].to_numpy(dtype=float)
                if "_amount" in aux_sub.columns:
                    amount = aux_sub["_amount"].to_numpy(dtype=float)

            x, w, stats = _apply_daily_filters(
                x,
                w=w,
                trade=trade,
                susp=susp,
                close=close,
                volume=volume,
                amount=amount,
                membership_age=membership_age,
                args=args,
            )
            day_stats = {
                "segment": seg,
                "date": str(dtn.date()),
                "reference_universe": env.reference_universe if use_reference else env.universe,
                "training_universe": env.universe,
                "asset_scope": "reference" if use_reference else "benchmark",
                **stats,
                "min_valid_names_required": int(max(10, args.min_valid_names)),
                "min_valid_ratio_required": float(max(0.0, args.min_valid_ratio)),
            }
            day_stats["accepted"] = bool(
                x.shape[0] >= max(10, int(args.min_valid_names))
                and float(stats["valid_ratio"]) >= float(max(0.0, args.min_valid_ratio))
            )
            daily_filter_rows.append(day_stats)
            daily_filter_index.append(dtn)
            if not day_stats["accepted"]:
                continue
            rows.append(_agg_global(x, w))
            rows[-1].update(_corr_summaries(x))
            factor_vecs.append(_factor_stats(x, w))
            idx_list.append(dtn)

    if not idx_list or min_dt is None or max_dt is None:
        raise RuntimeError("No daily observation states computed; check filters and data coverage.")
    daily_filter_df = pd.DataFrame(daily_filter_rows, index=pd.to_datetime(daily_filter_index)).sort_index()
    return rows, factor_vecs, idx_list, min_dt, max_dt, daily_filter_df


def _date_fit_mask(index: pd.Index, *, fit_on: str, train_range: Tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    fit_on = str(fit_on or "train").strip().lower()
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    if fit_on == "all":
        return np.ones((len(idx),), dtype=bool)
    if fit_on != "train":
        raise ValueError(f"Unsupported fit_on: {fit_on}")
    train_start, train_end = train_range
    mask = (idx >= train_start) & (idx <= train_end)
    if int(mask.sum()) < 5:
        raise RuntimeError(f"Too few train days in fit range {train_start.date()}..{train_end.date()}: {int(mask.sum())}")
    return mask


def build_market_observation_df(
    *,
    env: RuntimeEnv,
    args: argparse.Namespace,
    use_reference: bool = True,
    asset_role: str = "observation",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, factor_vecs, idx_list, min_dt, max_dt, daily_filter_stats = _iter_daily_feature_matrices(
        env,
        args,
        use_reference=use_reference,
    )
    base_df = pd.DataFrame(rows, index=pd.to_datetime(idx_list)).sort_index()
    if not base_df.index.is_unique:
        dup = base_df.index[base_df.index.duplicated()].unique()
        raise RuntimeError(f"Duplicate dates found in observation_df ({len(dup)} duplicates, e.g. {dup[:5].tolist()})")

    factor_mat = np.stack(factor_vecs, axis=0)
    fit_mask = _date_fit_mask(base_df.index, fit_on=args.pca_fit_on, train_range=env.train_range)
    pca_mean, pca_comps = _pca_fit(factor_mat[fit_mask], k=int(args.pca_dim))
    scores = _pca_transform(factor_mat, pca_mean, pca_comps)
    pca_df = pd.DataFrame(
        scores,
        index=base_df.index,
        columns=[f"market_obs_pca_{i}" for i in range(scores.shape[1])],
    )
    obs_df = pd.concat([base_df.rename(columns=OBS_RENAME_MAP), pca_df], axis=1)
    obs_df = obs_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if obs_df.empty:
        raise RuntimeError("Observation asset became empty after dropping non-finite rows.")

    meta = {
        "build_version": "stable_market_state_field_v1",
        "asset_type": "observation",
        "asset_role": str(asset_role),
        "universe": env.universe,
        "reference_universe": env.reference_universe if use_reference else env.universe,
        "reference_feature_source": env.reference_feature_source if use_reference else "processed_dataset",
        "reference_feature_semantics": (
            "qlib_dk_i_robust_zscore_fillna_stream_exact"
            if use_reference and env.reference_feature_source == "raw_qlib_stream"
            else "qlib_processed_dataset_dk_i"
        ),
        "feature_schema": list(obs_df.columns),
        "fit_on": str(args.pca_fit_on),
        "fit_range": [str(env.train_range[0].date()), str(env.train_range[1].date())],
        "coverage_start": str(obs_df.index.min().date()),
        "coverage_end": str(obs_df.index.max().date()),
        "n_days": int(len(obs_df)),
        "dim": int(obs_df.shape[1]),
        "pca_dim": int(pca_comps.shape[0]),
        "pca_input_dim": int(pca_comps.shape[1]),
        "min_source_date": str(min_dt.date()),
        "max_source_date": str(max_dt.date()),
        "state_membership_mode": str(args.state_membership_mode),
        "state_validity_protocol": str(args.state_validity_protocol),
        "min_listing_days": int(args.min_listing_days),
        "min_valid_names": int(args.min_valid_names),
        "min_valid_ratio": float(args.min_valid_ratio),
        "max_row_nan_frac": float(args.max_row_nan_frac),
        "trade_field": args.trade_field,
        "suspend_field": args.suspend_field,
        "min_trade": float(args.min_trade),
    }
    if use_reference and env.reference_feature_source == "raw_qlib_stream":
        meta.update(
            {
                "reference_fetch_scope": str(getattr(args, "reference_fetch_scope", "active_block")),
                "reference_min_raw_coverage_warn": float(
                    getattr(args, "reference_min_raw_coverage_warn", 0.0) or 0.0
                ),
                "reference_min_fit_coverage_warn": float(
                    getattr(args, "reference_min_fit_coverage_warn", 0.0) or 0.0
                ),
                "stream_fit_expected_rows": int(daily_filter_stats.attrs.get("stream_fit_expected_rows", 0)),
                "stream_fit_rows": int(daily_filter_stats.attrs.get("stream_fit_rows", 0)),
                "stream_fit_fetch_coverage": float(
                    daily_filter_stats.attrs.get("stream_fit_fetch_coverage", float("nan"))
                ),
                "low_raw_coverage_day_count": int(daily_filter_stats.attrs.get("low_raw_coverage_day_count", 0)),
            }
        )
    artifacts = {
        "pca_mean": pca_mean,
        "pca_comps": pca_comps,
        "daily_filter_stats": daily_filter_stats,
        "meta": meta,
    }
    return obs_df, artifacts


def _extract_single_instrument_frame(df: pd.DataFrame, inst: str) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        for level_name in ("instrument", "symbol"):
            if level_name in (df.index.names or []):
                try:
                    out = df.xs(inst, level=level_name)
                    if isinstance(out, pd.Series):
                        out = out.to_frame()
                    return out
                except Exception:
                    pass
        try:
            out = df.xs(inst)
            if isinstance(out, pd.Series):
                out = out.to_frame()
            return out
        except Exception:
            pass
    return df


def build_market_context_df(
    *,
    env: RuntimeEnv,
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    market_windows: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    windows = _normalize_market_windows(market_windows)
    cols = ["$close", "$high", "$low", "$volume", "$amount"]
    raw = env.D.features([env.market_index], cols, start_time=str(start.date()), end_time=str(end.date()), freq="day")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise RuntimeError(f"Failed to load benchmark series for {env.market_index}.")
    raw = _extract_single_instrument_frame(raw, env.market_index).copy()
    raw.index = pd.to_datetime(raw.index).normalize()
    raw = raw.sort_index()

    def _series(name: str) -> pd.Series:
        if name not in raw.columns:
            return pd.Series(index=raw.index, dtype=float)
        return pd.to_numeric(raw[name], errors="coerce").astype(float)

    close = _series("$close")
    high = _series("$high")
    low = _series("$low")
    volume = _series("$volume")
    amount = _series("$amount")

    ret_1d = close.pct_change(1)
    range_1d = (high - low) / close.replace(0.0, np.nan).abs()
    amount_z20 = _rolling_zscore_series(amount, 20, shift_stats=False)
    volume_z20 = _rolling_zscore_series(volume, 20, shift_stats=False)

    feature_map: Dict[str, pd.Series] = {
        "market_ctx_ret_1d": ret_1d,
        "market_ctx_range_1d": range_1d,
    }
    for w in windows:
        feature_map[f"market_ctx_ret_{w}d"] = close.pct_change(w)
        feature_map[f"market_ctx_vol_{w}d"] = ret_1d.rolling(w, min_periods=w).std(ddof=0)
        roll_max = close.rolling(w, min_periods=w).max()
        feature_map[f"market_ctx_dd_{w}d"] = 1.0 - (close / roll_max)
    feature_map["market_ctx_amount_z20"] = amount_z20
    feature_map["market_ctx_volume_z20"] = volume_z20

    context_df = pd.DataFrame(feature_map).sort_index()

    if args.market_ts_past_only:
        context_df = context_df.shift(1)

    context_df = context_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if context_df.empty:
        raise RuntimeError("Context asset became empty after dropping non-finite rows.")

    meta = {
        "build_version": "stable_market_state_field_v1",
        "asset_type": "context",
        "universe": env.universe,
        "market_index": env.market_index,
        "feature_schema": list(context_df.columns),
        "fit_on": "none",
        "fit_range": [str(env.train_range[0].date()), str(env.train_range[1].date())],
        "coverage_start": str(context_df.index.min().date()),
        "coverage_end": str(context_df.index.max().date()),
        "n_days": int(len(context_df)),
        "dim": int(context_df.shape[1]),
        "past_only": bool(args.market_ts_past_only),
    }
    return context_df, meta


def _scale_df_train_only(
    df: pd.DataFrame,
    *,
    fit_mask: np.ndarray,
    method: str,
) -> tuple[pd.DataFrame, dict[str, Dict[str, float]]]:
    fit_df = df.loc[fit_mask]
    center, scale = _fit_macro_scale(fit_df, method)
    scaled = (df - center) / scale
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    params = {
        "center": {str(k): float(v) for k, v in center.items()},
        "scale": {str(k): float(v) for k, v in scale.items()},
        "method": method,
    }
    return scaled, params


def _half_life_to_lambda(half_life: int) -> float:
    h = max(int(half_life), 1)
    return float(np.exp(-np.log(2.0) / h))


def _run_multiscale_filter(df: pd.DataFrame, half_lives: Sequence[int]) -> tuple[list[pd.DataFrame], list[float]]:
    branches: List[pd.DataFrame] = []
    lambdas: List[float] = []
    for half_life in half_lives:
        lam = _half_life_to_lambda(int(half_life))
        alpha = 1.0 - lam
        branch = df.ewm(alpha=alpha, adjust=False).mean()
        branches.append(branch)
        lambdas.append(lam)
    return branches, lambdas


def _numeric_col_or_zero(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(0.0, index=df.index, dtype=float)


def _compute_shock_components(context_df: pd.DataFrame, observation_df: pd.DataFrame) -> pd.DataFrame:
    pc1 = _numeric_col_or_zero(
        observation_df,
        ["market_obs_corr_pc1_ratio", "scope_all_corr_pc1_ratio", "scope_ref_all_corr_pc1_ratio"],
    )
    std = _numeric_col_or_zero(observation_df, ["market_obs_std", "scope_all_std", "scope_ref_all_std"])
    tail = _numeric_col_or_zero(
        observation_df,
        ["market_obs_tail_2sigma", "scope_all_tail_2sigma", "scope_ref_all_tail_2sigma"],
    )
    breadth = _numeric_col_or_zero(
        observation_df,
        ["market_obs_breadth", "scope_all_breadth", "scope_ref_all_breadth"],
    )
    return pd.DataFrame(
        {
            "delta_pc1_ratio": pc1.diff().abs(),
            "delta_std": std.diff().abs(),
            "delta_tail_2sigma": tail.diff().abs(),
            "delta_breadth": breadth.diff().abs(),
            "market_ctx_vol_5d": pd.to_numeric(context_df["market_ctx_vol_5d"], errors="coerce"),
        },
        index=context_df.index,
    )


def _strip_observation_prefix(col: str) -> str:
    text = str(col)
    for prefix in ("market_obs_", "market_state_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _scope_prefix_observation(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    slug = _slugify_name(scope)
    out = df.copy()
    out.columns = [f"scope_{slug}_{_strip_observation_prefix(c)}" for c in out.columns]
    return out


def _add_momentum_features(raw: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = raw.copy()
    for col in columns:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_mom_1d"] = s.diff(1)
        out[f"{col}_mom_5d"] = s.diff(5)
        ema5 = s.ewm(span=5, adjust=False, min_periods=1).mean()
        ema20 = s.ewm(span=20, adjust=False, min_periods=1).mean()
        out[f"{col}_ema5_minus_ema20"] = ema5 - ema20
        mu20 = s.rolling(20, min_periods=5).mean()
        sd20 = s.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
        out[f"{col}_z20"] = (s - mu20) / sd20
    return out


def _build_scope_relation_features(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_scope: str,
    right_scope: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    idx = left.index.intersection(right.index).sort_values()
    if idx.empty:
        raise RuntimeError(f"No overlapping dates for scope relation {left_scope}_vs_{right_scope}.")
    left = left.loc[idx]
    right = right.loc[idx]
    raw = pd.DataFrame(index=idx)
    missing: List[str] = []
    eps = 1e-6
    left_slug = _slugify_name(left_scope)
    right_slug = _slugify_name(right_scope)
    base_cols: List[str] = []
    for feature_name, obs_col in RELATIVE_COMPLEXITY_FEATURES.items():
        if obs_col not in left.columns or obs_col not in right.columns:
            missing.append(feature_name)
            continue
        left_v = pd.to_numeric(left[obs_col], errors="coerce")
        right_v = pd.to_numeric(right[obs_col], errors="coerce")
        delta_col = f"scope_{left_slug}_vs_{right_slug}_{feature_name}_delta"
        ratio_col = f"scope_{left_slug}_vs_{right_slug}_{feature_name}_ratio"
        raw[delta_col] = left_v - right_v
        raw[ratio_col] = (left_v + eps) / (right_v + eps)
        base_cols.extend([delta_col, ratio_col])
    if raw.empty:
        raise RuntimeError(
            f"Could not build relation {left_scope}_vs_{right_scope}; "
            f"missing={missing}, left_cols={list(left.columns)[:8]}, right_cols={list(right.columns)[:8]}"
        )
    raw = _add_momentum_features(raw, base_cols)
    meta = {
        "relation": f"{left_slug}_vs_{right_slug}",
        "base_columns": base_cols,
        "missing_canonical_features": missing,
        "eps": eps,
    }
    return raw, meta


def _build_tri_scope_observation_df(
    *,
    scope_observations: dict[str, pd.DataFrame],
    train_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_scopes = [s for s in TRI_SCOPE_SCOPES if s not in scope_observations]
    if missing_scopes:
        raise RuntimeError(f"Missing tri-scope observation inputs: {missing_scopes}")
    common_index: Optional[pd.Index] = None
    for df in scope_observations.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    if common_index is None or common_index.empty:
        raise RuntimeError("Tri-scope observations have no overlapping dates.")
    common_index = pd.DatetimeIndex(pd.to_datetime(common_index)).normalize().sort_values()

    parts: List[pd.DataFrame] = []
    relation_meta: List[dict[str, Any]] = []
    aligned = {scope: df.loc[common_index].copy() for scope, df in scope_observations.items()}
    for scope in TRI_SCOPE_SCOPES:
        parts.append(_scope_prefix_observation(aligned[scope], scope))
    for left_scope, right_scope in TRI_SCOPE_RELATION_PAIRS:
        rel_df, rel_meta = _build_scope_relation_features(
            left=aligned[left_scope],
            right=aligned[right_scope],
            left_scope=left_scope,
            right_scope=right_scope,
        )
        parts.append(rel_df.loc[common_index])
        relation_meta.append(rel_meta)

    raw = pd.concat(parts, axis=1)
    raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    fit_mask = _date_fit_mask(raw.index, fit_on="train", train_range=train_range)
    scaled, scale = _scale_df_train_only(raw, fit_mask=fit_mask, method="robust")
    scaled = scaled.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if scaled.empty:
        raise RuntimeError("Tri-scope observation became empty after scaling/non-finite cleanup.")
    sample = raw.head(3).to_dict(orient="index")
    sample_json = {
        str(pd.Timestamp(k).date()): {str(kk): float(vv) for kk, vv in row.items() if np.isfinite(vv)}
        for k, row in sample.items()
    }
    meta = {
        "build_version": "stable_market_state_field_v1",
        "asset_type": "observation",
        "asset_role": "shared_tri_scope_observation",
        "observation_profile": "tri_scope_v1",
        "scopes": list(TRI_SCOPE_SCOPES),
        "relation_pairs": [f"{a}_vs_{b}" for a, b in TRI_SCOPE_RELATION_PAIRS],
        "relation_meta": relation_meta,
        "feature_schema": list(scaled.columns),
        "raw_feature_schema": list(raw.columns),
        "scale": scale,
        "fit_on": "train",
        "fit_range": [str(train_range[0].date()), str(train_range[1].date())],
        "coverage_start": str(scaled.index.min().date()),
        "coverage_end": str(scaled.index.max().date()),
        "n_days": int(len(scaled)),
        "dim": int(scaled.shape[1]),
        "raw_audit_sample": sample_json,
        "scope_dims": {scope: int(scope_observations[scope].shape[1]) for scope in TRI_SCOPE_SCOPES},
        "scope_n_days": {scope: int(len(scope_observations[scope])) for scope in TRI_SCOPE_SCOPES},
    }
    return scaled, {"meta": meta}


def _benchmark_scope_from_context(*, universe: str, market_index: str) -> str:
    key = str(universe or "").strip().lower()
    if key in {"csi300", "csi800"}:
        return key
    idx = str(market_index or "").strip().upper()
    if idx == "SH000300":
        return "csi300"
    if idx == "SH000906":
        return "csi800"
    raise RuntimeError(f"Cannot infer benchmark scope from universe={universe!r}, market_index={market_index!r}.")


def _build_benchmark_relative_from_tri_scope(
    *,
    tri_scope_observation_df: pd.DataFrame,
    benchmark_scope: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bench = _slugify_name(benchmark_scope)
    if bench == "csi300":
        prefixes = ("scope_csi300_vs_all_", "scope_csi300_vs_csi800_")
    elif bench == "csi800":
        prefixes = ("scope_csi800_vs_all_", "scope_csi800_vs_csi300_")
    else:
        prefixes = (f"scope_{bench}_vs_all_",)
    cols = [c for c in tri_scope_observation_df.columns if any(str(c).startswith(p) for p in prefixes)]
    if not cols:
        raise RuntimeError(f"No benchmark-relative tri-scope columns found for benchmark_scope={benchmark_scope!r}.")
    out = tri_scope_observation_df.loc[:, cols].copy()
    out.columns = [f"benchrel_{str(c).removeprefix('scope_')}" for c in out.columns]
    meta = {
        "benchmark_scope": bench,
        "benchmark_relative_prefixes": list(prefixes),
        "benchmark_relative_features": list(out.columns),
        "benchmark_relative_source_features": cols,
    }
    return out, meta


def _build_relative_complexity_df(
    *,
    reference_observation_df: pd.DataFrame,
    benchmark_observation_df: pd.DataFrame,
    fit_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    idx = reference_observation_df.index.intersection(benchmark_observation_df.index).sort_values()
    if idx.empty:
        raise RuntimeError("Reference and benchmark observations have no overlap for relative_complexity_v1.")
    ref = reference_observation_df.loc[idx]
    bench = benchmark_observation_df.loc[idx]

    raw = pd.DataFrame(index=idx)
    missing: List[str] = []
    eps = 1e-6
    for feature_name, obs_col in RELATIVE_COMPLEXITY_FEATURES.items():
        if obs_col not in ref.columns or obs_col not in bench.columns:
            missing.append(feature_name)
            continue
        ref_v = pd.to_numeric(ref[obs_col], errors="coerce")
        bench_v = pd.to_numeric(bench[obs_col], errors="coerce")
        raw[f"ref_all_{feature_name}"] = ref_v
        raw[f"bench_{feature_name}"] = bench_v
        raw[f"rel_{feature_name}_delta"] = ref_v - bench_v
        raw[f"rel_{feature_name}_ratio"] = (ref_v + eps) / (bench_v + eps)

    if raw.empty:
        raise RuntimeError(
            "relative_complexity_v1 could not build any feature; "
            f"missing={missing}, ref_cols={list(ref.columns)[:10]}, bench_cols={list(bench.columns)[:10]}"
        )

    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.ffill().fillna(0.0)
    raw = raw.dropna(how="any")
    if raw.empty:
        raise RuntimeError("relative_complexity_v1 raw features became empty after finite-value cleanup.")

    aligned_mask = np.asarray(fit_mask, dtype=bool)[reference_observation_df.index.get_indexer(raw.index)]
    scaled, scale = _scale_df_train_only(raw, fit_mask=aligned_mask, method="robust")
    scaled = scaled.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if scaled.empty:
        raise RuntimeError("relative_complexity_v1 scaled features became empty.")

    sample = raw.head(3).to_dict(orient="index")
    sample_json = {
        str(pd.Timestamp(k).date()): {str(kk): float(vv) for kk, vv in row.items() if np.isfinite(vv)}
        for k, row in sample.items()
    }
    meta = {
        "relative_complexity_features": list(scaled.columns),
        "relative_complexity_raw_features": list(raw.columns),
        "relative_complexity_missing_canonical_features": missing,
        "relative_complexity_scale": scale,
        "relative_complexity_eps": eps,
        "relative_complexity_audit_sample_raw": sample_json,
    }
    return scaled, meta


def _assign_shock_bucket(score: pd.Series, *, q1: float, q2: float) -> pd.Series:
    out = pd.Series(index=score.index, dtype=object)
    out[score <= q1] = "low"
    out[(score > q1) & (score <= q2)] = "mid"
    out[score > q2] = "high"
    return out


def build_market_field_df(
    *,
    context_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    fit_mask: np.ndarray,
    half_lives: Sequence[int],
    field_shift: int,
    field_profile: str = "stable_v1",
    benchmark_observation_df: Optional[pd.DataFrame] = None,
    benchmark_scope: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not context_df.index.equals(observation_df.index):
        raise ValueError("context_df and observation_df must be aligned before field build.")

    scaled_context, context_scale = _scale_df_train_only(context_df, fit_mask=fit_mask, method="robust")
    scaled_obs, obs_scale = _scale_df_train_only(observation_df, fit_mask=fit_mask, method="robust")
    relative_scaled = None
    relative_meta: dict[str, Any] = {}
    field_profile = str(field_profile or "stable_v1").strip().lower()
    if field_profile == "relative_complexity_v1":
        if benchmark_observation_df is None:
            raise ValueError("relative_complexity_v1 requires benchmark_observation_df.")
        if not observation_df.index.equals(benchmark_observation_df.index):
            raise ValueError("observation_df and benchmark_observation_df must be aligned for relative_complexity_v1.")
        relative_scaled, relative_meta = _build_relative_complexity_df(
            reference_observation_df=observation_df,
            benchmark_observation_df=benchmark_observation_df,
            fit_mask=fit_mask,
        )
    elif field_profile == TRI_SCOPE_FIELD_PROFILE:
        if not benchmark_scope:
            raise ValueError(f"{TRI_SCOPE_FIELD_PROFILE} requires benchmark_scope.")
        relative_scaled, relative_meta = _build_benchmark_relative_from_tri_scope(
            tri_scope_observation_df=observation_df,
            benchmark_scope=str(benchmark_scope),
        )
    elif field_profile != "stable_v1":
        raise ValueError(f"Unsupported field_profile: {field_profile}")

    fused_parts = [scaled_context, scaled_obs]
    if relative_scaled is not None and field_profile != TRI_SCOPE_FIELD_PROFILE:
        fused_parts.append(relative_scaled.reindex(observation_df.index))
    fused = pd.concat(fused_parts, axis=1)
    fused = fused.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if fused.empty:
        raise RuntimeError("Field fused input became empty after scaling.")

    aligned_mask = np.asarray(fit_mask, dtype=bool)[context_df.index.get_indexer(fused.index)]
    branches, lambdas = _run_multiscale_filter(fused, half_lives=half_lives)

    shock_components = _compute_shock_components(context_df.loc[fused.index], observation_df.loc[fused.index])
    shock_scaled, shock_scale = _scale_df_train_only(shock_components.fillna(0.0), fit_mask=aligned_mask, method="robust")
    shock_score = shock_scaled.abs().mean(axis=1)

    fit_scores = shock_score.loc[aligned_mask]
    if int(fit_scores.notna().sum()) < 5:
        raise RuntimeError("Too few train days to fit field shock thresholds.")
    q1, q2 = fit_scores.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    bucket = _assign_shock_bucket(shock_score, q1=float(q1), q2=float(q2)).fillna("mid")

    weight_rows = np.stack([SHOCK_WEIGHT_TABLE[str(b)] for b in bucket.tolist()], axis=0)
    weights_df = pd.DataFrame(
        weight_rows,
        index=fused.index,
        columns=["market_field_w_fast", "market_field_w_mid", "market_field_w_slow"],
    )

    field_values = (
        branches[0].mul(weights_df["market_field_w_fast"], axis=0)
        + branches[1].mul(weights_df["market_field_w_mid"], axis=0)
        + branches[2].mul(weights_df["market_field_w_slow"], axis=0)
    )
    field_values.columns = [f"market_field_{i}" for i in range(field_values.shape[1])]
    field_parts = [field_values]
    if relative_scaled is not None:
        field_parts.append(relative_scaled.loc[field_values.index])
    field_parts.extend([weights_df, shock_score.rename("market_field_shock_score")])
    field_df = pd.concat(field_parts, axis=1)
    if int(field_shift or 0) > 0:
        field_df = field_df.shift(int(field_shift))
    field_df = field_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if field_df.empty:
        raise RuntimeError("Field asset became empty after shift/non-finite filtering.")

    meta = {
        "build_version": "stable_market_state_field_v1",
        "asset_type": "field",
        "field_profile": field_profile,
        "universe": None,
        "feature_schema": list(field_df.columns),
        "fit_on": "train",
        "fit_range": None,
        "coverage_start": str(field_df.index.min().date()),
        "coverage_end": str(field_df.index.max().date()),
        "n_days": int(len(field_df)),
        "dim": int(field_df.shape[1]),
        "field_shift": int(field_shift or 0),
        "half_lives": [int(x) for x in half_lives],
        "half_life_lambdas": [float(x) for x in lambdas],
        "scale_context": context_scale,
        "scale_observation": obs_scale,
        "shock_scale": shock_scale,
        "shock_thresholds": {
            "q1": float(q1),
            "q2": float(q2),
        },
        "shock_weight_table": {
            k: [float(x) for x in v.tolist()] for k, v in SHOCK_WEIGHT_TABLE.items()
        },
    }
    meta.update(relative_meta)
    return field_df, meta


def _make_legacy_compat_df(
    *,
    context_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    field_df: pd.DataFrame,
) -> pd.DataFrame:
    compat = pd.concat([context_df, observation_df, field_df], axis=1, join="inner").sort_index()
    alias_map = {
        "market_obs_mean_abs": "market_state_mean_abs",
        "market_obs_std": "market_state_std",
        "market_obs_breadth": "market_state_breadth",
        "market_obs_tail_2sigma": "market_state_tail_2sigma",
        "market_obs_corr_mean_abs": "market_state_corr_mean_abs",
        "market_obs_corr_fro": "market_state_corr_fro",
        "market_obs_corr_pc1_ratio": "market_state_corr_pc1_ratio",
        "market_ctx_ret_1d": "market_ret_1d",
        "market_ctx_vol_5d": "market_vol_5",
        "market_ctx_vol_20d": "market_vol_20",
        "market_ctx_dd_20d": "market_dd_20",
        "market_ctx_range_1d": "market_range_1d",
        "market_ctx_amount_z20": "market_amount_z20",
        "market_ctx_volume_z20": "market_volume_z20",
    }
    for src, dst in alias_map.items():
        if src in compat.columns and dst not in compat.columns:
            compat[dst] = compat[src]
    for col in list(compat.columns):
        if col.startswith("market_obs_pca_"):
            dst = col.replace("market_obs_pca_", "market_state_pca_")
            if dst not in compat.columns:
                compat[dst] = compat[col]
    compat = compat.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return compat


def _read_df(path: Path) -> pd.DataFrame:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"DataFrame asset not found: {p}")
    if p.suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(p)
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p, index_col=0)
    else:
        raise ValueError(f"Unsupported asset suffix for {p} (use .pkl/.parquet/.csv)")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Expected a non-empty DataFrame asset: {p}")
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)
    if not df.index.is_unique:
        dup = df.index[df.index.duplicated()].unique()
        raise ValueError(f"Duplicate dates found in {p}: {len(dup)} duplicate date(s), e.g. {dup[:5].tolist()}")
    df = df.apply(pd.to_numeric, errors="coerce")
    arr = df.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"Asset contains {n_bad} non-finite values: {p}")
    return df


def _load_json_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read JSON sidecar {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON sidecar must contain an object: {path}")
    return data


def _load_observation_asset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = Path(path).expanduser()
    df = _read_df(p)
    meta = _load_json_optional(p.with_suffix(p.suffix + ".meta.json"))
    meta.setdefault("build_version", "stable_market_state_field_v1")
    meta.setdefault("asset_type", "observation")
    meta.setdefault("feature_schema", list(df.columns))
    meta.setdefault("coverage_start", str(df.index.min().date()))
    meta.setdefault("coverage_end", str(df.index.max().date()))
    meta.setdefault("n_days", int(len(df)))
    meta.setdefault("dim", int(df.shape[1]))
    meta["input_observation_path"] = str(p)

    artifacts: Dict[str, Any] = {"meta": meta}
    pca_path = p.with_suffix(p.suffix + ".pca.npz")
    if pca_path.exists():
        with np.load(pca_path) as loaded:
            if "mean" in loaded and "components" in loaded:
                artifacts["pca_mean"] = loaded["mean"]
                artifacts["pca_comps"] = loaded["components"]
    daily_stats_path = p.with_suffix(p.suffix + ".daily_filter_stats.csv")
    if daily_stats_path.exists():
        artifacts["daily_filter_stats"] = pd.read_csv(daily_stats_path, index_col=0)
    return df, artifacts


def _write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
    elif path.suffix == ".parquet":
        df.to_parquet(path)
    elif path.suffix == ".csv":
        df.to_csv(path)
    else:
        raise ValueError(f"Unsupported output suffix for {path} (use .pkl/.parquet/.csv)")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_asset(path: Path, df: pd.DataFrame, meta: dict[str, Any], *, extra_objects: Optional[dict[str, Any]] = None) -> None:
    _write_df(path, df)
    _write_json(path.with_suffix(path.suffix + ".meta.json"), meta)
    extra = extra_objects or {}
    if "pca_mean" in extra and "pca_comps" in extra:
        np.savez_compressed(path.with_suffix(path.suffix + ".pca.npz"), mean=extra["pca_mean"], components=extra["pca_comps"])
        pca_meta = {
            "pca_dim": int(extra["pca_comps"].shape[0]),
            "input_dim": int(extra["pca_comps"].shape[1]),
            "fit_on": str(meta.get("fit_on", "train")),
            "fit_range": meta.get("fit_range"),
        }
        _write_json(path.with_suffix(path.suffix + ".pca.meta.json"), pca_meta)
    if "daily_filter_stats" in extra and isinstance(extra["daily_filter_stats"], pd.DataFrame):
        extra["daily_filter_stats"].to_csv(path.with_suffix(path.suffix + ".daily_filter_stats.csv"))


def _daily_filter_stats_summary(df: Any) -> Optional[dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    out: dict[str, Any] = {"n_days": int(len(df))}
    for col in ("n_total", "n_valid", "valid_ratio", "raw_coverage"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = float(s.mean()) if s.notna().any() else float("nan")
            out[f"{col}_min"] = float(s.min()) if s.notna().any() else float("nan")
            out[f"{col}_p10"] = float(s.quantile(0.10)) if s.notna().any() else float("nan")
    if "accepted" in df.columns:
            out["accepted_days"] = int(pd.Series(df["accepted"]).astype(bool).sum())
    return out


def _namespace_copy_with(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def _build_processed_scope_observation(
    *,
    base_args: argparse.Namespace,
    scope: str,
    asset_role: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scope_args = _namespace_copy_with(
        base_args,
        instruments=scope,
        state_reference_instruments=scope,
        reference_feature_source="processed_dataset",
        market_index=_infer_market_index(scope, None),
        state_reference_market_index=_infer_market_index(scope, None),
        input_observation=None,
    )
    scope_env = _import_runtime(scope_args)
    return build_market_observation_df(
        env=scope_env,
        args=scope_args,
        use_reference=True,
        asset_role=asset_role,
    )


def main() -> None:
    args = parse_args()
    deprecated = _deprecated_args_used(args)
    if deprecated:
        print(f"[WARN] Ignoring deprecated options in split-asset mode: {', '.join(deprecated)}")

    env = _import_runtime(args)
    out_paths = _resolve_asset_paths(args, env.universe)
    shared_source_meta = {
        "data_config_module": env.data_config_module,
        "data_config_attr": env.data_config_attr,
        "training_universe": env.universe,
        "reference_universe": env.reference_universe,
        "reference_feature_source": env.reference_feature_source,
        "reference_instrument_chunk_size": int(args.reference_instrument_chunk_size or 800),
        "reference_fetch_scope": str(getattr(args, "reference_fetch_scope", "active_block")),
        "reference_min_raw_coverage_warn": float(getattr(args, "reference_min_raw_coverage_warn", 0.0) or 0.0),
        "reference_min_fit_coverage_warn": float(getattr(args, "reference_min_fit_coverage_warn", 0.0) or 0.0),
        "reference_feature_semantics": (
            "qlib_dk_i_robust_zscore_fillna_stream_exact"
            if env.reference_feature_source == "raw_qlib_stream"
            else "qlib_processed_dataset_dk_i"
        ),
        "state_membership_mode": str(args.state_membership_mode),
        "state_validity_protocol": str(args.state_validity_protocol),
        "min_listing_days": int(args.min_listing_days),
        "min_valid_names": int(args.min_valid_names),
        "min_valid_ratio": float(args.min_valid_ratio),
        "max_row_nan_frac": float(args.max_row_nan_frac),
        "trade_field": args.trade_field,
        "suspend_field": args.suspend_field,
        "min_trade": float(args.min_trade),
    }
    benchmark_source_meta = {
        **shared_source_meta,
        "reference_market_index": env.market_index,
        "depends_on_market_index": True,
        "asset_scope": "benchmark_specific",
    }
    observation_source_meta = {
        **shared_source_meta,
        "depends_on_market_index": False,
        "asset_scope": "shared_reference" if env.reference_universe != env.universe else "universe_specific",
    }
    market_windows = _normalize_market_windows(_parse_int_list(args.market_ts_windows))
    half_lives = _parse_int_list(args.field_half_lives)
    if len(half_lives) != 3:
        raise ValueError("--field_half_lives must provide exactly 3 values (fast, mid, slow).")

    field_profile = str(getattr(args, "field_profile", "stable_v1") or "stable_v1").strip().lower()
    input_observation_path = Path(args.input_observation).expanduser() if args.input_observation else None
    if input_observation_path is not None:
        obs_df, obs_artifacts = _load_observation_asset(input_observation_path)
        print(f"[INFO] Loaded shared observation asset: {input_observation_path} shape={obs_df.shape}")
    else:
        obs_df, obs_artifacts = build_market_observation_df(
            env=env,
            args=args,
            use_reference=True,
            asset_role="shared_reference_observation",
        )

    benchmark_obs_df = None
    benchmark_obs_artifacts = None
    if field_profile == "relative_complexity_v1":
        benchmark_obs_df, benchmark_obs_artifacts = build_market_observation_df(
            env=env,
            args=args,
            use_reference=False,
            asset_role="benchmark_observation_for_relative_complexity",
        )
        print(f"[INFO] Built benchmark observation for relative_complexity_v1: shape={benchmark_obs_df.shape}")
    elif field_profile == TRI_SCOPE_FIELD_PROFILE:
        scope_observations = {"all": obs_df}
        scope_artifacts: dict[str, dict[str, Any]] = {"all": obs_artifacts}
        for scope in ("csi300", "csi800"):
            scope_df, scope_meta = _build_processed_scope_observation(
                base_args=args,
                scope=scope,
                asset_role=f"tri_scope_{scope}_observation",
            )
            scope_observations[scope] = scope_df
            scope_artifacts[scope] = scope_meta
            print(f"[INFO] Built tri-scope observation scope={scope}: shape={scope_df.shape}")
        obs_df, tri_scope_artifacts = _build_tri_scope_observation_df(
            scope_observations=scope_observations,
            train_range=env.train_range,
        )
        tri_scope_artifacts["scope_artifacts"] = scope_artifacts
        obs_artifacts = tri_scope_artifacts
        print(f"[INFO] Built tri-scope shared observation: shape={obs_df.shape}")

    context_df, context_meta = build_market_context_df(
        env=env,
        args=args,
        start=pd.Timestamp(obs_df.index.min()).normalize(),
        end=pd.Timestamp(obs_df.index.max()).normalize(),
        market_windows=market_windows,
    )

    common_index = context_df.index.intersection(obs_df.index).sort_values()
    if common_index.empty:
        raise RuntimeError("Context and observation assets have no overlapping dates.")
    context_for_field = context_df.loc[common_index]
    obs_for_field = obs_df.loc[common_index]
    benchmark_obs_for_field = None
    if benchmark_obs_df is not None:
        common_index = common_index.intersection(benchmark_obs_df.index).sort_values()
        if common_index.empty:
            raise RuntimeError("Context, reference observation, and benchmark observation have no overlapping dates.")
        context_for_field = context_df.loc[common_index]
        obs_for_field = obs_df.loc[common_index]
        benchmark_obs_for_field = benchmark_obs_df.loc[common_index]
    fit_mask = _date_fit_mask(common_index, fit_on="train", train_range=env.train_range)

    field_df, field_meta = build_market_field_df(
        context_df=context_for_field,
        observation_df=obs_for_field,
        fit_mask=fit_mask,
        half_lives=half_lives,
        field_shift=int(args.field_shift or 0),
        field_profile=field_profile,
        benchmark_observation_df=benchmark_obs_for_field,
        benchmark_scope=_benchmark_scope_from_context(universe=env.universe, market_index=env.market_index)
        if field_profile == TRI_SCOPE_FIELD_PROFILE
        else None,
    )

    context_out_df = context_for_field.loc[field_df.index]
    obs_field_aligned_df = obs_for_field.loc[field_df.index]
    obs_out_df = obs_df if bool(args.preserve_observation_index) else obs_field_aligned_df

    if input_observation_path is not None and args.out_observation is None and field_profile != TRI_SCOPE_FIELD_PROFILE:
        observation_path_for_meta = input_observation_path
        write_observation = False
    else:
        observation_path_for_meta = out_paths["observation"]
        write_observation = bool(args.emit_observation)

    context_meta.update(
        {
            "fit_range": [str(env.train_range[0].date()), str(env.train_range[1].date())],
            "coverage_start": str(context_out_df.index.min().date()),
            "coverage_end": str(context_out_df.index.max().date()),
            "n_days": int(len(context_out_df)),
            "market_ts_windows": [int(x) for x in market_windows],
            **benchmark_source_meta,
        }
    )
    obs_meta = dict(obs_artifacts.get("meta") or {})
    if input_observation_path is None:
        obs_meta.update(observation_source_meta)
        obs_meta["observation_source"] = "computed"
    else:
        for key, value in observation_source_meta.items():
            obs_meta.setdefault(key, value)
        obs_meta["observation_source"] = "input_observation"
        obs_meta["input_observation_path"] = str(input_observation_path)
        obs_meta["consumer_training_universe"] = env.universe
    obs_meta.update(
        {
            "coverage_start": str(obs_out_df.index.min().date()),
            "coverage_end": str(obs_out_df.index.max().date()),
            "n_days": int(len(obs_out_df)),
            "dim": int(obs_out_df.shape[1]),
            "feature_schema": list(obs_out_df.columns),
            "preserve_observation_index": bool(args.preserve_observation_index),
            "field_aligned_n_days": int(len(obs_field_aligned_df)),
            "depends_on_market_index": False,
        }
    )
    if field_profile == TRI_SCOPE_FIELD_PROFILE:
        obs_meta.update(
            {
                "observation_profile": "tri_scope_v1",
                "tri_scope_source": "input_ref_all_plus_processed_benchmarks"
                if input_observation_path is not None
                else "computed_ref_all_plus_processed_benchmarks",
                "scope_daily_filter_stats": {
                    scope: _daily_filter_stats_summary((art.get("daily_filter_stats") if isinstance(art, dict) else None))
                    for scope, art in (obs_artifacts.get("scope_artifacts") or {}).items()
                },
            }
        )
    obs_artifacts["meta"] = obs_meta

    field_meta.update(
        {
            "universe": env.universe,
            "fit_range": [str(env.train_range[0].date()), str(env.train_range[1].date())],
            "coverage_start": str(field_df.index.min().date()),
            "coverage_end": str(field_df.index.max().date()),
            "n_days": int(len(field_df)),
            "context_path": str(out_paths["context"]) if args.emit_context else None,
            "observation_path": str(observation_path_for_meta),
            "observation_source": "input_observation" if input_observation_path is not None else "computed",
            "reference_observation_daily_filter_stats": _daily_filter_stats_summary(
                obs_artifacts.get("daily_filter_stats")
            ),
            "benchmark_observation_source": "computed" if benchmark_obs_df is not None else None,
            "benchmark_observation_dim": int(benchmark_obs_df.shape[1]) if benchmark_obs_df is not None else None,
            "benchmark_observation_feature_schema": list(benchmark_obs_df.columns) if benchmark_obs_df is not None else None,
            "benchmark_observation_daily_filter_stats": (
                _daily_filter_stats_summary(benchmark_obs_artifacts.get("daily_filter_stats"))
                if benchmark_obs_artifacts is not None
                else None
            ),
            "tri_scope_observation_path": str(observation_path_for_meta)
            if field_profile == TRI_SCOPE_FIELD_PROFILE
            else None,
            "benchmark_scope": _benchmark_scope_from_context(universe=env.universe, market_index=env.market_index)
            if field_profile == TRI_SCOPE_FIELD_PROFILE
            else None,
            **benchmark_source_meta,
        }
    )

    if args.emit_context:
        _save_asset(out_paths["context"], context_out_df, context_meta)
        print(f"[INFO] Saved context asset: {out_paths['context']} shape={context_out_df.shape}")
    if write_observation:
        _save_asset(out_paths["observation"], obs_out_df, obs_artifacts["meta"], extra_objects=obs_artifacts)
        print(f"[INFO] Saved observation asset: {out_paths['observation']} shape={obs_out_df.shape}")
    elif args.emit_observation and input_observation_path is not None:
        print(
            "[INFO] Reusing input observation asset without copying; "
            f"pass --out-observation to write a benchmark-specific copy: {input_observation_path}"
        )
    if args.emit_field:
        _save_asset(out_paths["field"], field_df, field_meta)
        print(f"[INFO] Saved field asset: {out_paths['field']} shape={field_df.shape}")

    if out_paths["legacy"] is not None:
        compat_df = _make_legacy_compat_df(context_df=context_out_df, observation_df=obs_field_aligned_df, field_df=field_df)
        compat_meta = {
            "build_version": "stable_market_state_field_v1",
            "asset_type": "legacy_compat",
            "universe": env.universe,
            "source_field_path": str(out_paths["field"]),
            "coverage_start": str(compat_df.index.min().date()),
            "coverage_end": str(compat_df.index.max().date()),
            "n_days": int(len(compat_df)),
            "dim": int(compat_df.shape[1]),
            **benchmark_source_meta,
        }
        _save_asset(out_paths["legacy"], compat_df, compat_meta)
        print(f"[INFO] Saved compatibility asset: {out_paths['legacy']} shape={compat_df.shape}")


if __name__ == "__main__":
    main()
