from __future__ import annotations

import logging
from dataclasses import dataclass
import bisect
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketStateLookup:
    dim: int
    by_date: Dict[pd.Timestamp, np.ndarray]
    dates: tuple[pd.Timestamp, ...] = ()


@dataclass
class MarketStateBundle:
    field_path: Path
    field_df: pd.DataFrame
    analysis_df: pd.DataFrame
    context_path: Optional[Path] = None
    context_df: Optional[pd.DataFrame] = None
    observation_path: Optional[Path] = None
    observation_df: Optional[pd.DataFrame] = None


_LEGACY_TO_NEW = {
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
    "market_ret_1d": "market_ctx_ret_1d",
    "market_vol_5": "market_ctx_vol_5d",
    "market_vol_20": "market_ctx_vol_20d",
    "market_dd_20": "market_ctx_dd_20d",
    "market_range_1d": "market_ctx_range_1d",
    "market_amount_z20": "market_ctx_amount_z20",
    "market_volume_z20": "market_ctx_volume_z20",
}

_NEW_TO_LEGACY = {v: k for k, v in _LEGACY_TO_NEW.items()}


def _normalize_market_state_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"market_state_path must contain a non-empty DataFrame, got: {type(df)}")

    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all(axis=None):
        raise ValueError("market_state_df contains no numeric values after coercion.")
    return df


def resolve_market_state_path(
    path: str | Path,
    *,
    search_dirs: Sequence[str | Path] | None = None,
) -> Optional[Path]:
    """
    Resolve `market_state_path` across a few candidate roots.

    Notes
    -----
    - `path` may be absolute or relative.
    - `search_dirs` are tried in order (useful for recorder local_dir / script dir).
    - Returns the first existing path, else None.
    """
    if path is None:
        return None
    path_str = str(path).strip()
    if not path_str:
        return None

    p0 = Path(path_str).expanduser()
    candidates: List[Path] = [p0]
    for d in (search_dirs or []):
        try:
            candidates.append(Path(d).expanduser() / p0)
        except Exception:
            continue

    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def load_market_state_df(path: str | Path) -> pd.DataFrame:
    """
    Load precomputed market state DataFrame.

    Expected format
    ---------------
    - index: datetime-like (trading date)
    - columns: state feature names
    - values: numeric
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"market_state_path not found: {p}")

    if p.suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(p)
    elif p.suffix in {".parquet"}:
        df = pd.read_parquet(p)
    elif p.suffix in {".csv"}:
        df = pd.read_csv(p, index_col=0)
    else:
        raise ValueError(f"Unsupported market_state file type: {p.suffix} (use .pkl/.parquet/.csv)")

    return _normalize_market_state_df(df)


def _derive_sibling_asset_path(path: Path, *, target: str) -> Optional[Path]:
    name = path.name
    prefixes = {
        "field": "daily_market_field_",
        "context": "daily_market_context_",
        "observation": "daily_market_observation_",
    }
    field_prefix = prefixes["field"]
    if not name.startswith(field_prefix):
        return None
    suffix = name[len(field_prefix) :]
    return path.with_name(prefixes[target] + suffix)


def derive_market_state_sibling_path(path: str | Path, *, target: str) -> Optional[Path]:
    return _derive_sibling_asset_path(Path(path).expanduser(), target=target)


def build_market_state_analysis_view(
    *,
    field_df: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
    observation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = field_df.copy().sort_index()
    if context_df is not None:
        df = df.join(context_df.sort_index(), how="left")
    if observation_df is not None:
        df = df.join(observation_df.sort_index(), how="left")

    for legacy, new in _LEGACY_TO_NEW.items():
        if legacy in df.columns and new not in df.columns:
            df[new] = df[legacy]
        if new in df.columns and legacy not in df.columns:
            df[legacy] = df[new]

    for col in list(df.columns):
        m_old = re.fullmatch(r"market_state_pca_(\d+)", str(col))
        if m_old:
            new = f"market_obs_pca_{m_old.group(1)}"
            if new not in df.columns:
                df[new] = df[col]
            continue
        m_new = re.fullmatch(r"market_obs_pca_(\d+)", str(col))
        if m_new:
            legacy = f"market_state_pca_{m_new.group(1)}"
            if legacy not in df.columns:
                df[legacy] = df[col]

    return df


def load_market_state_bundle(
    path: str | Path,
    *,
    search_dirs: Sequence[str | Path] | None = None,
) -> MarketStateBundle:
    resolved = resolve_market_state_path(path, search_dirs=search_dirs)
    if resolved is None:
        raise FileNotFoundError(f"market_state_path not found: {path}")

    field_path = Path(resolved).expanduser()
    field_df = load_market_state_df(field_path)

    context_path = _derive_sibling_asset_path(field_path, target="context")
    observation_path = _derive_sibling_asset_path(field_path, target="observation")
    context_df = None
    observation_df = None

    if context_path is not None and context_path.exists():
        try:
            context_df = load_market_state_df(context_path)
        except Exception as e:
            logger.warning(f"[market_state] Failed to load sibling context asset {context_path}: {e}")
            context_df = None
    if observation_path is not None and observation_path.exists():
        try:
            observation_df = load_market_state_df(observation_path)
        except Exception as e:
            logger.warning(f"[market_state] Failed to load sibling observation asset {observation_path}: {e}")
            observation_df = None

    analysis_df = build_market_state_analysis_view(
        field_df=field_df,
        context_df=context_df,
        observation_df=observation_df,
    )
    return MarketStateBundle(
        field_path=field_path,
        field_df=field_df,
        analysis_df=analysis_df,
        context_path=context_path if context_df is not None else None,
        context_df=context_df,
        observation_path=observation_path if observation_df is not None else None,
        observation_df=observation_df,
    )


def load_market_state_analysis_df(
    path: str | Path,
    *,
    search_dirs: Sequence[str | Path] | None = None,
) -> pd.DataFrame:
    return load_market_state_bundle(path, search_dirs=search_dirs).analysis_df


def make_market_state_lookup(df: pd.DataFrame, *, shift: int = 0) -> MarketStateLookup:
    """
    Build a fast date->vector mapping. Optional shift uses prior trading days
    within the provided index (shift=1 means use previous available date).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty DataFrame")

    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)

    if shift:
        shift = int(shift)
        if shift < 0:
            raise ValueError("shift must be >= 0")
        df = df.shift(shift)

    dim = int(df.shape[1])
    by_date: Dict[pd.Timestamp, np.ndarray] = {}
    for dt, row in df.iterrows():
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            # keep NaNs; adapter will decide whether to nan_to_num or raise
            pass
        by_date[pd.Timestamp(dt).normalize()] = row.to_numpy(dtype=np.float32, copy=True)

    return MarketStateLookup(dim=dim, by_date=by_date, dates=tuple(sorted(by_date.keys())))


def _lookup_missing_market_state(
    lookup: MarketStateLookup,
    dt: pd.Timestamp,
    *,
    missing_policy: str,
) -> Optional[np.ndarray]:
    policy = str(missing_policy or "strict").strip().lower()
    if policy in {"strict", "raise"}:
        return None
    if policy in {"nan", "none"}:
        return np.full((lookup.dim,), np.nan, dtype=np.float32)
    if policy == "zero":
        return np.zeros((lookup.dim,), dtype=np.float32)
    if policy in {"ffill", "forward_fill", "previous"}:
        pos = bisect.bisect_right(lookup.dates, dt) - 1
        if pos < 0:
            return None
        return lookup.by_date.get(lookup.dates[pos], None)
    raise ValueError(
        f"Unsupported market_state missing_policy={missing_policy!r}. "
        "Supported: strict, nan, zero, ffill."
    )


def lookup_market_state(
    lookup: MarketStateLookup,
    dates: Iterable[pd.Timestamp],
    *,
    strict: bool = True,
    missing_policy: str = "strict",
) -> np.ndarray:
    """
    Map dates -> [B, dim] array.
    """
    out = []
    nan_inf_dates: List[pd.Timestamp] = []
    missing_dates: List[pd.Timestamp] = []
    
    for d in dates:
        dt = pd.Timestamp(d).normalize()
        if dt.tz is not None:
            dt = dt.tz_convert(None)
        v = lookup.by_date.get(dt, None)
        if v is None:
            v = _lookup_missing_market_state(lookup, dt, missing_policy=missing_policy)
            if v is None and strict:
                missing_dates.append(dt)
                continue  # will raise after collecting all
            if v is None:
                missing_dates.append(dt)
                v = np.full((lookup.dim,), np.nan, dtype=np.float32)
        elif not np.isfinite(v).all():
            nan_inf_dates.append(dt)
        out.append(v)
    
    # Log NaN/Inf dates regardless of strict mode
    if nan_inf_dates:
        logger.warning(
            f"[market_state] Found {len(nan_inf_dates)} dates with NaN/Inf values: "
            f"{[str(d) for d in nan_inf_dates]}"
        )
    
    # Log missing dates regardless of strict mode
    if missing_dates:
        logger.warning(
            f"[market_state] Found {len(missing_dates)} missing dates: "
            f"{[str(d) for d in missing_dates]}"
        )
    
    # Raise errors in strict mode
    if strict:
        if missing_dates:
            raise KeyError(
                f"market state missing for {len(missing_dates)} date(s): {missing_dates}"
            )
        if nan_inf_dates:
            raise KeyError(
                f"market state contains NaN/Inf for {len(nan_inf_dates)} date(s): {nan_inf_dates}"
            )
    
    return np.stack(out, axis=0)
