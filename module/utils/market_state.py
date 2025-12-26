from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketStateLookup:
    dim: int
    by_date: Dict[pd.Timestamp, np.ndarray]


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

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"market_state_path must contain a non-empty DataFrame, got: {type(df)}")

    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)

    # ensure numeric float32 for model input
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all(axis=None):
        raise ValueError("market_state_df contains no numeric values after coercion.")

    return df


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

    return MarketStateLookup(dim=dim, by_date=by_date)


def lookup_market_state(
    lookup: MarketStateLookup,
    dates: Iterable[pd.Timestamp],
    *,
    strict: bool = True,
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
            missing_dates.append(dt)
            if strict:
                continue  # will raise after collecting all
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
