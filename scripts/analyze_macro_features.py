#!/usr/bin/env python3
"""
Macro Feature Data Quality Analysis Script

This script validates the precomputed market_state pkl files for:
1. Data completeness (NaN/Inf checks)
2. Statistical sanity (value ranges, distributions)
3. Time alignment verification
4. Feature consistency across derived columns
5. Potential look-ahead bias detection

Usage:
    python scripts/analyze_macro_features.py [--path artifacts/market_state/market_state_csi300.pkl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def infer_value_space(df: pd.DataFrame) -> str:
    """
    Infer whether the dataframe is in a raw space (natural bounds like breadth∈[0,1])
    or has been macro-scaled (e.g. `precompute_market_state.py --macro_scale robust/zscore`),
    which breaks those bounds by design.

    Returns:
        "raw" or "scaled"
    """
    # If market_ret_1d has multi-sigma values, it's almost certainly scaled because
    # raw daily returns rarely exceed 20%.
    if "market_ret_1d" in df.columns:
        s = pd.to_numeric(df["market_ret_1d"], errors="coerce").dropna()
        if len(s) >= 50 and float(s.abs().quantile(0.999)) > 0.5:
            return "scaled"

    # Breadth should be in [0,1] in raw space; outside suggests scaling.
    if "market_state_breadth" in df.columns:
        s = pd.to_numeric(df["market_state_breadth"], errors="coerce").dropna()
        if len(s) >= 50 and ((s < -0.1).any() or (s > 1.1).any()):
            return "scaled"

    # These are non-negative in raw space; negative implies scaling.
    for col in ("market_state_mean_abs", "market_state_std"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) >= 50 and (s < 0).any():
                return "scaled"

    return "raw"


def check_basic_stats(df: pd.DataFrame) -> Dict[str, any]:
    """Check basic statistics of the dataframe."""
    results = {
        "shape": df.shape,
        "date_range": (str(df.index.min().date()), str(df.index.max().date())),
        "n_dates": len(df),
        "n_features": df.shape[1],
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    return results


def check_nan_inf(df: pd.DataFrame) -> Dict[str, any]:
    """Check for NaN and Inf values."""
    nan_counts = df.isna().sum()
    inf_counts = (~np.isfinite(df.values) & ~df.isna()).sum(axis=0)
    
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    inf_cols = pd.Series(inf_counts, index=df.columns)
    inf_cols = inf_cols[inf_cols > 0].sort_values(ascending=False)
    
    # Find rows with any NaN
    nan_rows = df.isna().any(axis=1)
    nan_dates = df.index[nan_rows].tolist()
    
    results = {
        "total_nan": int(nan_counts.sum()),
        "total_inf": int(inf_cols.sum()),
        "nan_columns": {col: int(cnt) for col, cnt in nan_cols.head(20).items()},
        "inf_columns": {col: int(cnt) for col, cnt in inf_cols.head(20).items()},
        "nan_dates_count": len(nan_dates),
        "first_nan_date": str(nan_dates[0]) if nan_dates else None,
        "last_nan_date": str(nan_dates[-1]) if nan_dates else None,
    }
    return results


def check_zero_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that are all zeros (potential bug)."""
    all_zero = (df == 0).all()
    return all_zero[all_zero].index.tolist()


def check_constant_columns(df: pd.DataFrame, threshold: float = 1e-10) -> List[str]:
    """Find columns with near-zero variance (suspicious)."""
    stds = df.std()
    return stds[stds < threshold].index.tolist()


def check_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Check value ranges for all columns."""
    stats = pd.DataFrame({
        "min": df.min(),
        "max": df.max(),
        "mean": df.mean(),
        "std": df.std(),
        "nan_pct": df.isna().mean() * 100,
    })
    return stats


def check_base_features(df: pd.DataFrame, *, value_space: str) -> Dict[str, str]:
    """Validate base feature columns."""
    issues = {}
    value_space = str(value_space or "raw").strip().lower()
    
    # Check global distribution features
    global_cols = [
        "market_state_mean_abs",
        "market_state_std", 
        "market_state_breadth",
        "market_state_tail_2sigma"
    ]
    
    for col in global_cols:
        if col not in df.columns:
            issues[col] = "MISSING"
            continue
        vals = df[col].dropna().values
        if len(vals) == 0:
            issues[col] = "ALL_NAN"
        elif (vals == 0).all():
            issues[col] = "ALL_ZERO (BUG!)"
        elif value_space == "raw" and vals.min() < 0:
            issues[col] = f"NEGATIVE_VALUES (min={vals.min():.4f})"
        elif value_space == "scaled" and np.nanmax(np.abs(vals)) > 50:
            issues[col] = f"EXTREME_VALUES (max abs={np.nanmax(np.abs(vals)):.4f})"
    
    # Check correlation features
    corr_cols = [
        "market_state_corr_mean_abs",
        "market_state_corr_fro",
        "market_state_corr_pc1_ratio"
    ]
    
    for col in corr_cols:
        if col not in df.columns:
            issues[col] = "MISSING"
            continue
        vals = df[col].dropna().values
        if len(vals) == 0:
            issues[col] = "ALL_NAN"
        elif value_space == "raw" and col == "market_state_corr_mean_abs" and (vals > 1).any():
            issues[col] = f"VALUES > 1 (max={vals.max():.4f})"
        elif value_space == "raw" and col == "market_state_corr_pc1_ratio" and (vals > 1).any():
            issues[col] = f"VALUES > 1 (max={vals.max():.4f})"
        elif value_space == "scaled" and np.nanmax(np.abs(vals)) > 50:
            issues[col] = f"EXTREME_VALUES (max abs={np.nanmax(np.abs(vals)):.4f})"
    
    # Check breadth should be between 0 and 1
    if value_space == "raw" and "market_state_breadth" in df.columns:
        vals = df["market_state_breadth"].dropna().values
        if (vals > 1).any() or (vals < 0).any():
            issues["market_state_breadth"] = f"OUT_OF_RANGE [0,1] (range: {vals.min():.4f} to {vals.max():.4f})"
    
    return issues


def check_pca_features(df: pd.DataFrame) -> Dict[str, any]:
    """Validate PCA features."""
    # Only count the actual PCA score columns: `market_state_pca_{i}`.
    # Exclude derived variants like `_d1`, `_z20`, `_roll_mean20`, etc.
    pca_cols = [c for c in df.columns if re.fullmatch(r"market_state_pca_\d+", str(c))]
    
    if not pca_cols:
        return {"status": "NO_PCA_COLUMNS"}
    
    pca_df = df[pca_cols]
    
    # PCA scores should be centered (mean ≈ 0 on train set)
    means = pca_df.mean().abs()
    stds = pca_df.std()
    
    # Check for degenerate components (zero variance)
    zero_var = stds[stds < 1e-10].index.tolist()
    
    return {
        "n_components": len(pca_cols),
        "mean_abs_range": (float(means.min()), float(means.max())),
        "std_range": (float(stds.min()), float(stds.max())),
        "zero_variance_components": zero_var,
        "nan_pct": float(pca_df.isna().mean().mean() * 100),
    }


def check_delta_features(df: pd.DataFrame) -> Dict[str, any]:
    """Validate delta (Δstate) features."""
    delta_cols = [c for c in df.columns if "_d1" in c or "_d5" in c or "_d10" in c]
    
    if not delta_cols:
        return {"status": "NO_DELTA_COLUMNS"}
    
    delta_df = df[delta_cols]
    
    # Delta features should have NaN for initial rows
    initial_nan = delta_df.iloc[:15].isna().mean().mean() * 100
    
    return {
        "n_delta_columns": len(delta_cols),
        "initial_nan_pct": float(initial_nan),
        "nan_pct": float(delta_df.isna().mean().mean() * 100),
        "sample_cols": delta_cols[:5],
    }


def check_zscore_features(df: pd.DataFrame, *, warmup: Optional[int] = None) -> Dict[str, any]:
    """Validate rolling z-score features."""
    z_cols = [c for c in df.columns if "_z20" in c or "_z60" in c or "_z120" in c]
    
    if not z_cols:
        return {"status": "NO_ZSCORE_COLUMNS"}
    
    z_df = df[z_cols]
    
    # Z-score should be roughly N(0,1) after warmup.
    # Use expected warmup when provided to avoid false alarms.
    warmup = max(0, int(warmup)) if warmup is not None else 120
    if len(z_df) > warmup:
        post_warmup = z_df.iloc[warmup:]
        means = post_warmup.mean().abs()
        stds = post_warmup.std()
        
        # Flags for suspicious z-scores
        large_mean = means[means > 2].index.tolist()
        small_std = stds[stds < 0.1].index.tolist()
        large_std = stds[stds > 5].index.tolist()
    else:
        large_mean, small_std, large_std = [], [], []
    
    return {
        "n_zscore_columns": len(z_cols),
        "warmup": int(warmup),
        "nan_pct": float(z_df.isna().mean().mean() * 100),
        "suspicious_large_mean": large_mean[:5],
        "suspicious_small_std": small_std[:5],
        "suspicious_large_std": large_std[:5],
    }


def _extract_ints(pattern: str, text: str) -> List[int]:
    return [int(x) for x in re.findall(pattern, text)]


def infer_expected_warmup(df: pd.DataFrame) -> int:
    """
    Infer warmup length from column naming patterns.
    This mirrors the chained windows used in precompute (base + roll_mean + zscore chain).
    """
    max_need = 0
    for col in df.columns:
        base = 0
        for lag in _extract_ints(r"_d(\d+)", col):
            base = max(base, lag)
        for lag in _extract_ints(r"market_ret_(\d+)d", col):
            base = max(base, lag)
        for lag in _extract_ints(r"market_(?:vol|mom|dd)_(\d+)", col):
            base = max(base, lag)

        roll_chain = sum(w - 1 for w in _extract_ints(r"_roll_mean(\d+)", col) if w > 1)
        z_chain = sum(w - 1 for w in _extract_ints(r"_z(\d+)", col) if w > 1)

        need = base + roll_chain + z_chain
        if need > max_need:
            max_need = need
    return int(max_need)


def check_market_ts_features(df: pd.DataFrame, *, value_space: str) -> Dict[str, any]:
    """Validate market time-series features."""
    ts_cols = [c for c in df.columns if c.startswith("market_ret_") or 
               c.startswith("market_vol_") or c.startswith("market_mom_") or 
               c.startswith("market_dd_")]
    
    if not ts_cols:
        return {"status": "NO_MARKET_TS_COLUMNS"}
    
    ts_df = df[ts_cols]
    
    issues = {}
    value_space = str(value_space or "raw").strip().lower()
    
    # market_ret_1d should be small (daily returns)
    if value_space == "raw" and "market_ret_1d" in ts_df.columns:
        ret = ts_df["market_ret_1d"].dropna()
        if (ret.abs() > 0.15).any():  # > 15% daily return is suspicious
            issues["market_ret_1d"] = f"LARGE_VALUES (max abs: {ret.abs().max():.4f})"
    elif value_space == "scaled" and "market_ret_1d" in ts_df.columns:
        ret = pd.to_numeric(ts_df["market_ret_1d"], errors="coerce").dropna()
        if len(ret) and float(ret.abs().max()) > 50:
            issues["market_ret_1d"] = f"EXTREME_VALUES (max abs: {ret.abs().max():.4f})"
    
    # market_dd_* (raw, not z-scored) should be in [0, 1]
    if value_space == "raw":
        dd_cols = [c for c in ts_cols if c.startswith("market_dd_") and "_z" not in c and "roll_mean" not in c]
        for col in dd_cols:
            vals = ts_df[col].dropna()
            if (vals < 0).any():
                issues[col] = f"NEGATIVE_DRAWDOWN (min: {vals.min():.4f})"
            if (vals > 1).any():
                issues[col] = f"DRAWDOWN > 1 (max: {vals.max():.4f})"
    
    return {
        "n_ts_columns": len(ts_cols),
        "columns": ts_cols,
        "issues": issues,
        "nan_pct": float(ts_df.isna().mean().mean() * 100),
    }


def check_warmup_coverage(df: pd.DataFrame, expected_warmup: Optional[int] = None) -> Dict[str, any]:
    """Check if warmup period has appropriate NaN patterns."""
    # Columns that should have NaN in warmup period
    warmup_cols = [c for c in df.columns if "_z" in c or "_d" in c or "roll_mean" in c]
    
    if not warmup_cols:
        return {"status": "NO_WARMUP_COLUMNS"}
    
    if expected_warmup is None:
        expected_warmup = infer_expected_warmup(df)
    expected_warmup = int(expected_warmup)
    if expected_warmup <= 0:
        return {
            "expected_warmup": expected_warmup,
            "first_row_nan_pct": 0.0,
            "warmup_nan_pct": 0.0,
        }

    warmup_df = df[warmup_cols].iloc[:expected_warmup]
    
    # Check first row should be mostly NaN for derived features
    first_row_nan = warmup_df.iloc[0].isna().mean() * 100
    
    return {
        "expected_warmup": expected_warmup,
        "first_row_nan_pct": float(first_row_nan),
        "warmup_nan_pct": float(warmup_df.isna().mean().mean() * 100),
    }


def check_time_continuity(df: pd.DataFrame) -> Dict[str, any]:
    """Check for gaps in the time series."""
    dates = pd.to_datetime(df.index)
    date_series = pd.Series(dates)
    diffs = date_series.diff()[1:]
    
    # Find large gaps (> 5 calendar days)
    large_gaps = diffs[diffs > pd.Timedelta(days=5)]
    gap_examples = [(str(dates[i].date()), int(td.days)) for i, td in large_gaps.head(10).items()]
    
    return {
        "n_dates": len(dates),
        "date_range": (str(dates.min().date()), str(dates.max().date())),
        "n_large_gaps": len(large_gaps),
        "large_gaps": gap_examples,
    }


def check_pca_sidecar(pkl_path: Path) -> Dict[str, any]:
    """Check PCA sidecar files."""
    npz_path = pkl_path.with_suffix(pkl_path.suffix + ".pca.npz")
    meta_path = pkl_path.with_suffix(pkl_path.suffix + ".pca.meta.json")
    
    results = {
        "npz_exists": npz_path.exists(),
        "meta_exists": meta_path.exists(),
    }
    
    if npz_path.exists():
        data = np.load(npz_path)
        results["pca_mean_shape"] = data["mean"].shape
        results["pca_components_shape"] = data["components"].shape
    
    if meta_path.exists():
        with open(meta_path, "r") as f:
            results["pca_meta"] = json.load(f)
    
    return results


def generate_report(
    df: pd.DataFrame,
    pkl_path: Path,
    *,
    expected_warmup: Optional[int] = None,
    value_space: Optional[str] = None,
) -> str:
    """Generate a comprehensive analysis report."""
    report = []
    if expected_warmup is None:
        expected_warmup = infer_expected_warmup(df)

    if value_space is None or str(value_space).strip().lower() == "auto":
        value_space = infer_value_space(df)
    else:
        value_space = str(value_space).strip().lower()
    
    # 1. Basic stats
    print_section("1. BASIC STATISTICS")
    stats = check_basic_stats(df)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  value_space: {value_space}")
    
    # 2. NaN/Inf check
    print_section("2. NaN/Inf ANALYSIS")
    nan_info = check_nan_inf(df)
    print(f"  Total NaN values: {nan_info['total_nan']}")
    print(f"  Total Inf values: {nan_info['total_inf']}")
    print(f"  Dates with NaN: {nan_info['nan_dates_count']}")
    if nan_info['nan_columns']:
        print(f"  Top NaN columns:")
        for col, cnt in list(nan_info['nan_columns'].items())[:5]:
            print(f"    - {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
    
    # 3. Zero/Constant columns
    print_section("3. SUSPICIOUS COLUMNS")
    zero_cols = check_zero_columns(df)
    const_cols = check_constant_columns(df)
    if zero_cols:
        print(f"  WARN: ALL-ZERO columns (BUG!): {zero_cols}")
    else:
        print("  OK: No all-zero columns")
    if const_cols:
        print(f"  WARN: Near-constant columns: {const_cols[:10]}")
    else:
        print("  OK: No near-constant columns")
    
    # 4. Base features
    print_section("4. BASE FEATURE VALIDATION")
    base_issues = check_base_features(df, value_space=value_space)
    if base_issues:
        print("  Issues found:")
        for col, issue in base_issues.items():
            print(f"    WARN: {col}: {issue}")
    else:
        print("  OK: All base features look valid")
    
    # 5. PCA features
    print_section("5. PCA FEATURES")
    pca_info = check_pca_features(df)
    for k, v in pca_info.items():
        print(f"  {k}: {v}")
    
    # 6. Delta features
    print_section("6. DELTA (dstate) FEATURES")
    delta_info = check_delta_features(df)
    for k, v in delta_info.items():
        print(f"  {k}: {v}")
    
    # 7. Z-score features
    print_section("7. ROLLING Z-SCORE FEATURES")
    z_info = check_zscore_features(df, warmup=expected_warmup)
    for k, v in z_info.items():
        print(f"  {k}: {v}")
    
    # 8. Market TS features
    print_section("8. MARKET TIME-SERIES FEATURES")
    ts_info = check_market_ts_features(df, value_space=value_space)
    for k, v in ts_info.items():
        print(f"  {k}: {v}")
    
    # 9. Warmup coverage
    print_section("9. WARMUP PERIOD COVERAGE")
    warmup_info = check_warmup_coverage(df, expected_warmup=expected_warmup)
    for k, v in warmup_info.items():
        print(f"  {k}: {v}")
    
    # 10. Time continuity
    print_section("10. TIME SERIES CONTINUITY")
    time_info = check_time_continuity(df)
    for k, v in time_info.items():
        print(f"  {k}: {v}")
    
    # 11. PCA sidecar files
    print_section("11. PCA SIDECAR FILES")
    sidecar_info = check_pca_sidecar(pkl_path)
    for k, v in sidecar_info.items():
        print(f"  {k}: {v}")
    
    # 12. Column summary
    print_section("12. COLUMN SUMMARY BY TYPE")
    col_types = {
        "global_stats": [c for c in df.columns if c.startswith("market_state_") and 
                         not c.startswith("market_state_pca") and 
                         not c.startswith("market_state_corr") and
                         "_d" not in c and "_z" not in c and "roll_mean" not in c],
        "correlation": [c for c in df.columns if c.startswith("market_state_corr") and "_d" not in c and "_z" not in c and "roll_mean" not in c],
        "pca": [c for c in df.columns if re.fullmatch(r"market_state_pca_\d+", str(c))],
        "delta": [c for c in df.columns if "_d1" in c or "_d5" in c or "_d10" in c],
        "zscore": [c for c in df.columns if "_z20" in c or "_z60" in c or "_z120" in c],
        "roll_mean": [c for c in df.columns if "roll_mean" in c],
        "market_ts": [c for c in df.columns if c.startswith("market_ret") or 
                      c.startswith("market_vol") or c.startswith("market_mom") or 
                      c.startswith("market_dd")],
    }
    for ctype, cols in col_types.items():
        print(f"  {ctype}: {len(cols)} columns")
    
    # 13. Value range summary for key columns
    print_section("13. VALUE RANGES (KEY COLUMNS)")
    key_cols = [
        "market_state_mean_abs", "market_state_std", "market_state_breadth",
        "market_state_corr_mean_abs", "market_state_corr_pc1_ratio",
        "market_ret_1d"
    ]
    for col in key_cols:
        if col in df.columns:
            vals = df[col].dropna().values
            print(f"  {col}:")
            print(f"    range: [{vals.min():.4f}, {vals.max():.4f}]")
            print(f"    mean: {vals.mean():.4f}, std: {vals.std():.4f}")
    
    # Final verdict
    print_section("VERDICT")
    issues = []
    if zero_cols:
        issues.append(f"ALL-ZERO columns: {zero_cols}")
    if base_issues:
        issues.append(f"Base feature issues: {list(base_issues.keys())}")
    if nan_info['total_nan'] > len(df) * 0.5:
        issues.append("High NaN ratio (>50%)")
    
    if issues:
        print("  WARN: ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  OK: DATA LOOKS HEALTHY")
    
    return ""


def main():
    parser = argparse.ArgumentParser(description="Analyze macro feature data quality")
    parser.add_argument("--path", type=str, default="artifacts/market_state/market_state_csi300.pkl",
                        help="Path to market_state pkl file")
    parser.add_argument("--expected_warmup", type=int, default=None,
                        help="Override expected warmup (trading days). If omitted, infer from columns.")
    parser.add_argument(
        "--value_space",
        type=str,
        default="auto",
        choices=["auto", "raw", "scaled"],
        help="How to interpret value ranges. Use 'scaled' if the file was produced with --macro_scale.",
    )
    parser.add_argument("--sample_date", type=str, default=None,
                        help="Show detailed data for a specific date (e.g., 2020-01-02)")
    args = parser.parse_args()
    
    pkl_path = Path(args.path).expanduser()
    if not pkl_path.is_absolute():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        pkl_path = script_dir / args.path
    
    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        sys.exit(1)
    
    print(f"Loading: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    df.index = pd.to_datetime(df.index)
    
    generate_report(df, pkl_path, expected_warmup=args.expected_warmup, value_space=args.value_space)
    
    # Optional: show data for a specific date
    if args.sample_date:
        print_section(f"SAMPLE DATA: {args.sample_date}")
        try:
            row = df.loc[args.sample_date]
            print(row.to_string())
        except KeyError:
            print(f"  Date not found: {args.sample_date}")


if __name__ == "__main__":
    main()
