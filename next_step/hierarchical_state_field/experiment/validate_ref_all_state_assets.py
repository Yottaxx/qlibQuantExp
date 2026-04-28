from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON sidecar must contain an object: {path}")
    return data


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    raise ValueError(f"Unsupported asset suffix: {path}")


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _add(report: dict[str, Any], ok: bool, name: str, detail: Any = None) -> None:
    report["checks"].append({"ok": bool(ok), "name": name, "detail": detail})
    if not ok:
        report["ok"] = False


def _check_frame(report: dict[str, Any], name: str, path: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    _add(report, path.exists(), f"{name}:exists", str(path))
    _add(report, _meta_path(path).exists(), f"{name}:meta_exists", str(_meta_path(path)))
    if not path.exists():
        return None, {}
    df = _read_df(path)
    meta = _read_json(_meta_path(path))
    idx = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    _add(report, df.index.is_unique, f"{name}:unique_index", int(df.index.duplicated().sum()))
    _add(report, idx.is_monotonic_increasing, f"{name}:monotonic_index", [str(idx.min().date()), str(idx.max().date())])
    _add(report, df.shape[0] > 0 and df.shape[1] > 0, f"{name}:non_empty", tuple(df.shape))
    arr = df.to_numpy(dtype=float)
    _add(report, bool(np.isfinite(arr).all()), f"{name}:finite_values", int((~np.isfinite(arr)).sum()))
    return df, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ref-all active-block state assets.")
    parser.add_argument("--root", default=r"C:\Users\60585\PycharmProjects\qlibQuantExp")
    parser.add_argument("--observation", default="artifacts/market_state/daily_market_observation_ref_all_pti_v1.pkl")
    parser.add_argument("--field-csi300", default="artifacts/market_state/daily_market_field_ref_all_SH000300.pkl")
    parser.add_argument("--field-csi800", default="artifacts/market_state/daily_market_field_ref_all_SH000906.pkl")
    parser.add_argument("--min-fit-coverage", type=float, default=0.80)
    parser.add_argument("--require-active-block", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    obs_path = root / args.observation
    field_300_path = root / args.field_csi300
    field_800_path = root / args.field_csi800
    report: dict[str, Any] = {"ok": True, "checks": []}

    obs_df, obs_meta = _check_frame(report, "observation", obs_path)
    field_300, field_300_meta = _check_frame(report, "field_SH000300", field_300_path)
    field_800, field_800_meta = _check_frame(report, "field_SH000906", field_800_path)

    _add(report, obs_meta.get("asset_type") == "observation", "observation:asset_type", obs_meta.get("asset_type"))
    _add(report, obs_meta.get("reference_universe") == "all", "observation:reference_universe", obs_meta.get("reference_universe"))
    _add(
        report,
        obs_meta.get("reference_feature_source") == "raw_qlib_stream",
        "observation:reference_feature_source",
        obs_meta.get("reference_feature_source"),
    )
    if args.require_active_block:
        _add(
            report,
            obs_meta.get("reference_fetch_scope") == "active_block",
            "observation:reference_fetch_scope",
            obs_meta.get("reference_fetch_scope"),
        )
    fit_cov = float(obs_meta.get("stream_fit_fetch_coverage", float("nan")))
    _add(
        report,
        bool(np.isfinite(fit_cov) and fit_cov >= float(args.min_fit_coverage)),
        "observation:stream_fit_fetch_coverage",
        fit_cov,
    )

    daily_path = obs_path.with_suffix(obs_path.suffix + ".daily_filter_stats.csv")
    _add(report, daily_path.exists(), "observation:daily_filter_stats_exists", str(daily_path))
    if daily_path.exists():
        daily = pd.read_csv(daily_path, index_col=0)
        for col in ("n_reference_members", "n_raw_rows", "raw_coverage", "n_valid", "accepted"):
            _add(report, col in daily.columns, f"daily_filter:has_{col}", list(daily.columns[:8]))
        if "raw_coverage" in daily.columns:
            raw_cov = pd.to_numeric(daily["raw_coverage"], errors="coerce")
            _add(
                report,
                bool(raw_cov.notna().any()),
                "daily_filter:raw_coverage_not_empty",
                {"min": float(raw_cov.min()), "median": float(raw_cov.median()), "max": float(raw_cov.max())},
            )
        if {"accepted", "n_valid"}.issubset(daily.columns):
            accepted = daily["accepted"].astype(str).str.lower().isin(["true", "1"])
            _add(report, bool(accepted.any()), "daily_filter:has_accepted_days", int(accepted.sum()))

    for name, meta, expected_benchmark in (
        ("field_SH000300", field_300_meta, "SH000300"),
        ("field_SH000906", field_800_meta, "SH000906"),
    ):
        _add(report, meta.get("reference_market_index") == expected_benchmark, f"{name}:reference_market_index", meta.get("reference_market_index"))
        _add(report, meta.get("asset_scope") == "benchmark_specific", f"{name}:asset_scope", meta.get("asset_scope"))
        _add(report, meta.get("observation_path") is not None, f"{name}:observation_path", meta.get("observation_path"))

    if obs_df is not None and field_300 is not None:
        obs_idx = set(pd.DatetimeIndex(pd.to_datetime(obs_df.index)).normalize())
        field_idx = set(pd.DatetimeIndex(pd.to_datetime(field_300.index)).normalize())
        _add(report, field_idx.issubset(obs_idx), "field_SH000300:index_subset_observation", {"field": len(field_idx), "obs": len(obs_idx)})
    if obs_df is not None and field_800 is not None:
        obs_idx = set(pd.DatetimeIndex(pd.to_datetime(obs_df.index)).normalize())
        field_idx = set(pd.DatetimeIndex(pd.to_datetime(field_800.index)).normalize())
        _add(report, field_idx.issubset(obs_idx), "field_SH000906:index_subset_observation", {"field": len(field_idx), "obs": len(obs_idx)})

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
