#!/usr/bin/env python3
"""
Convert MASTER-style market information CSV into macro feature format.

Expected input (from pandas MultiIndex columns):
  - First header row: column group (e.g., "feature")
  - Second header row: formula string (e.g., "Mask($close/Ref($close,1)-1,'SH000300')")
  - An extra row for index name (e.g., "datetime") that should be dropped

Output:
  - A DataFrame with index=datetime (normalized), numeric columns, saved as
    .pkl/.parquet/.csv compatible with market_state_path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def _clean_name(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s


def _dedupe(names: Iterable[str]) -> List[str]:
    seen = {}
    out = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
            continue
        seen[name] += 1
        out.append(f"{name}__{seen[name]}")
    return out


def _flatten_columns(cols: pd.Index, *, mode: str, sep: str) -> List[str]:
    if not isinstance(cols, pd.MultiIndex):
        return _dedupe([str(c) for c in cols])

    out = []
    for lvl0, lvl1 in cols.to_list():
        if mode == "level1":
            name = _clean_name(lvl1) or _clean_name(lvl0)
        elif mode == "level0":
            name = _clean_name(lvl0) or _clean_name(lvl1)
        else:
            parts = [p for p in (_clean_name(lvl0), _clean_name(lvl1)) if p]
            name = sep.join(parts)
        if not name:
            name = "feature"
        out.append(name)

    return _dedupe(out)


def _read_master_market(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
    except Exception:
        df = pd.read_csv(path, header=0, index_col=0)
    return df


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index, errors="coerce").normalize()
    keep = idx.notna()
    if keep.sum() != len(df):
        df = df.loc[keep].copy()
        idx = idx[keep]
    df.index = idx
    df.index.name = None
    df.sort_index(inplace=True)
    return df


def _save_df(df: pd.DataFrame, out: Path) -> None:
    if out.suffix in {".pkl", ".pickle"}:
        df.to_pickle(out)
    elif out.suffix == ".parquet":
        df.to_parquet(out)
    elif out.suffix == ".csv":
        df.to_csv(out)
    else:
        raise ValueError(f"Unsupported output type: {out.suffix} (use .pkl/.parquet/.csv)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MASTER market info CSV into macro feature format."
    )
    parser.add_argument(
        "--input",
        default="master_market/csi_market_information.csv",
        help="Path to MASTER market info CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/market_state/market_state_master_market.pkl",
        help="Output path (.pkl/.parquet/.csv).",
    )
    parser.add_argument(
        "--flatten",
        choices=["level1", "level0", "join"],
        default="level1",
        help="How to flatten MultiIndex columns.",
    )
    parser.add_argument(
        "--sep",
        default="__",
        help="Separator when --flatten=join.",
    )
    parser.add_argument(
        "--prefix",
        default="master_market_",
        help="Prefix for column names (empty to disable).",
    )
    args = parser.parse_args()

    inp = Path(args.input).expanduser()
    out = Path(args.output).expanduser()

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = _read_master_market(inp)
    df.columns = _flatten_columns(df.columns, mode=args.flatten, sep=args.sep)

    if args.prefix:
        df.rename(columns=lambda c: f"{args.prefix}{c}", inplace=True)

    df = _normalize_index(df)
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    _save_df(df, out)

    print(f">>> Saved macro features: {out}")
    print(f">>> shape={df.shape}, dates={df.index.min().date()}..{df.index.max().date()}")
    nan_pct = float(df.isna().mean().mean() * 100.0)
    print(f">>> NaN pct: {nan_pct:.2f}%")


if __name__ == "__main__":
    main()
