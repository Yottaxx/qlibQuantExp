from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


RUN_ORDER = (
    "base",
    "global_only",
    "local_only_residual",
    "global_local",
    "global_local_stable_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hierarchical_state_field ablation runs.")
    for label in RUN_ORDER:
        parser.add_argument(f"--{label}", type=str, required=True, help=f"Run directory or run_id for {label}.")
    parser.add_argument("--mlruns_dir", type=str, default="mlruns", help="MLflow runs root.")
    parser.add_argument(
        "--out_md",
        type=str,
        default="next_step/hierarchical_state_field/csi300_hierarchical_state_field_comparison.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="next_step/hierarchical_state_field/csi300_hierarchical_state_field_comparison.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def _resolve_run_dir(spec: str, mlruns_dir: Path) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    candidates = list(mlruns_dir.glob(f"*/*{spec}*"))
    exact = [c for c in candidates if c.is_dir() and c.name == spec]
    if exact:
        return exact[0]
    candidates = [c for c in candidates if c.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Unable to resolve run spec: {spec}")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _safe_load_pickle(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return _load_pickle(path)
    except Exception:
        return None


def _as_series(obj: Any) -> pd.Series:
    if isinstance(obj, pd.Series):
        return pd.to_numeric(obj, errors="coerce")
    if isinstance(obj, pd.DataFrame) and obj.shape[1] == 1:
        return pd.to_numeric(obj.iloc[:, 0], errors="coerce")
    if obj is None:
        return pd.Series(dtype=float)
    try:
        return pd.to_numeric(pd.Series(obj), errors="coerce")
    except Exception:
        return pd.Series(dtype=float)


def _series_stats(series: pd.Series) -> dict[str, float]:
    s = _as_series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"mean": math.nan, "std": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
    }


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    artifacts = run_dir / "artifacts"
    summary = _safe_load_pickle(artifacts / "run_summary")
    if isinstance(summary, dict):
        return dict(summary)
    return {}


def _load_variant_meta(run_dir: Path) -> dict[str, Any]:
    artifacts = run_dir / "artifacts"
    meta = _safe_load_pickle(artifacts / "variant_meta")
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _load_diag_stats(run_dir: Path) -> dict[str, dict[str, float]]:
    artifacts = run_dir / "artifacts"
    diag_names = {
        "time_ratio": "st_disentangle_gate_series",
        "gate_entropy": "st_disentangle_gate_entropy_series",
        "pooling_alpha_mean": "st_disentangle_pooling_alpha_mean_series",
        "pooling_alpha_std": "st_disentangle_pooling_alpha_std_series",
        "time_tau": "st_disentangle_time_tau_series",
        "time_half_life": "st_disentangle_time_half_life_series",
        "factor_gate_mean": "st_disentangle_factor_gate_mean_series",
        "factor_gate_std": "st_disentangle_factor_gate_std_series",
        "factor_gate_entropy": "st_disentangle_factor_gate_entropy_series",
        "factor_gate_topk_mass_5": "st_disentangle_factor_gate_topk_mass_5_series",
        "factor_gate_topk_mass_10": "st_disentangle_factor_gate_topk_mass_10_series",
        "global_state_norm": "st_disentangle_global_state_norm_series",
        "global_state_day_variance": "st_disentangle_global_state_day_variance_series",
        "local_state_norm": "st_disentangle_local_state_norm_series",
        "local_state_cross_sectional_variance": "st_disentangle_local_state_cross_sectional_variance_series",
        "time_ratio_stock_std": "st_disentangle_time_ratio_stock_std_series",
        "pooling_alpha_stock_std": "st_disentangle_pooling_alpha_stock_std_series",
        "router_global_sensitivity": "st_disentangle_router_global_sensitivity_series",
        "router_local_sensitivity": "st_disentangle_router_local_sensitivity_series",
        "film_global_sensitivity": "st_disentangle_film_global_sensitivity_series",
        "film_local_sensitivity": "st_disentangle_film_local_sensitivity_series",
        "pool_global_sensitivity": "st_disentangle_pool_global_sensitivity_series",
        "pool_local_sensitivity": "st_disentangle_pool_local_sensitivity_series",
        "time_ratio_chunk_std": "st_disentangle_time_ratio_chunk_std_series",
        "gate_entropy_chunk_std": "st_disentangle_gate_entropy_chunk_std_series",
        "pooling_alpha_chunk_std": "st_disentangle_pooling_alpha_chunk_std_series",
    }
    out: dict[str, dict[str, float]] = {}
    for label, name in diag_names.items():
        out[label] = _series_stats(_safe_load_pickle(artifacts / name))
    return out


def _load_regime_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    raw = _safe_load_pickle(run_dir / "artifacts" / "regime_bucket_eval")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, pd.DataFrame] = {}
    for key, value in raw.items():
        if isinstance(value, pd.DataFrame):
            out[key] = value.copy()
    return out


def _extract_bucket_metric(df: pd.DataFrame, *, bucket_col: str, bucket_value: str, metric: str) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty or bucket_col not in df.columns or metric not in df.columns:
        return math.nan
    sub = df.loc[df[bucket_col].astype(str) == str(bucket_value), metric]
    if sub.empty:
        return math.nan
    return float(pd.to_numeric(sub.iloc[0], errors="coerce"))


def _fmt(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "N/A"
    return f"{x:.4f}" if np.isfinite(x) else "N/A"


def _metric_row(section: str, metric: str, values: dict[str, float]) -> dict[str, Any]:
    stable = values.get("global_local_stable_summary", math.nan)
    base = values.get("base", math.nan)
    return {
        "section": section,
        "metric": metric,
        **{label: values.get(label, math.nan) for label in RUN_ORDER},
        "delta_stable_vs_base": (float(stable) - float(base)) if np.isfinite(stable) and np.isfinite(base) else math.nan,
    }


def _flatten_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def main() -> None:
    args = parse_args()
    mlruns_dir = Path(args.mlruns_dir)
    run_specs = {label: getattr(args, label) for label in RUN_ORDER}
    run_dirs = {label: _resolve_run_dir(spec, mlruns_dir) for label, spec in run_specs.items()}
    run_summaries = {label: _load_run_summary(path) for label, path in run_dirs.items()}
    run_meta = {label: _load_variant_meta(path) for label, path in run_dirs.items()}
    run_diag = {label: _load_diag_stats(path) for label, path in run_dirs.items()}
    run_regime = {label: _load_regime_tables(path) for label, path in run_dirs.items()}

    summary_metrics = [
        ("RankIC mean", "ric_mean"),
        ("RankIC IR", "ricir"),
        ("IC mean", "ic_mean"),
        ("ICIR", "icir"),
        ("Ann. Return", "ann_ret"),
        ("Info Ratio", "info_ratio"),
        ("Max Drawdown", "max_dd"),
        ("Turnover", "turnover"),
    ]
    diag_metrics = [
        "time_ratio",
        "gate_entropy",
        "pooling_alpha_mean",
        "pooling_alpha_std",
        "time_tau",
        "time_half_life",
        "factor_gate_mean",
        "factor_gate_std",
        "factor_gate_entropy",
        "factor_gate_topk_mass_5",
        "factor_gate_topk_mass_10",
        "global_state_norm",
        "global_state_day_variance",
        "local_state_norm",
        "local_state_cross_sectional_variance",
        "time_ratio_stock_std",
        "pooling_alpha_stock_std",
        "router_global_sensitivity",
        "router_local_sensitivity",
        "film_global_sensitivity",
        "film_local_sensitivity",
        "pool_global_sensitivity",
        "pool_local_sensitivity",
        "time_ratio_chunk_std",
        "gate_entropy_chunk_std",
        "pooling_alpha_chunk_std",
    ]

    rows: list[dict[str, Any]] = []
    for title, key in summary_metrics:
        values = {label: run_summaries[label].get(key, math.nan) for label in RUN_ORDER}
        rows.append(_metric_row("summary", title, values))

    for metric in diag_metrics:
        for stat in ("mean", "p10", "p50", "p90"):
            values = {label: run_diag[label].get(metric, {}).get(stat, math.nan) for label in RUN_ORDER}
            rows.append(_metric_row("diagnostic", f"{metric}.{stat}", values))

    regime_names = ["High-PC1 / High-Tail", "High-PC1 / Low-Tail", "Low-PC1 / High-Tail", "Low-PC1 / Low-Tail"]
    for regime_name in regime_names:
        for metric in ("RankIC_mean", "time_ratio_mean"):
            values = {}
            for label in RUN_ORDER:
                regime_2x2 = run_regime[label].get("regime_2x2")
                values[label] = _extract_bucket_metric(
                    regime_2x2,
                    bucket_col="Regime",
                    bucket_value=regime_name,
                    metric=metric,
                )
            rows.append(_metric_row("regime_2x2", f"{regime_name}::{metric}", values))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = _flatten_records(rows)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "metric", *RUN_ORDER, "delta_stable_vs_base"],
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    md_lines = [
        "# CSI300 Hierarchical State Field Comparison",
        "",
        "## Runs",
        "",
    ]
    for label in RUN_ORDER:
        meta = run_meta.get(label, {})
        summary = run_summaries.get(label, {})
        md_lines.extend(
            [
                f"### {label}",
                f"- Run: `{run_dirs[label]}`",
                f"- market_state_path: `{meta.get('market_state_path', summary.get('market_state_path', 'N/A'))}`",
                f"- market_day_summary_path: `{meta.get('market_day_summary_path', summary.get('market_day_summary_path', 'N/A'))}`",
                f"- use_hierarchical_state_field: `{meta.get('use_hierarchical_state_field', summary.get('use_hierarchical_state_field', 'N/A'))}`",
                f"- global_state_use_macro: `{meta.get('global_state_use_macro', summary.get('global_state_use_macro', 'N/A'))}`",
                f"- global_state_use_day_summary: `{meta.get('global_state_use_day_summary', summary.get('global_state_use_day_summary', 'N/A'))}`",
                f"- router_summary_source: `{meta.get('router_summary_source', summary.get('router_summary_source', 'N/A'))}`",
                f"- pooling_summary_source: `{meta.get('pooling_summary_source', summary.get('pooling_summary_source', 'N/A'))}`",
                "",
            ]
        )

    md_lines.extend(
        [
            "## Headline Metrics",
            "",
            "| Metric | " + " | ".join(RUN_ORDER) + " | stable-base |",
            "|" + ":--|" + "--:|" * len(RUN_ORDER) + "--:|",
        ]
    )
    for title, key in summary_metrics:
        values = {label: run_summaries[label].get(key, math.nan) for label in RUN_ORDER}
        row = _metric_row("summary", title, values)
        md_lines.append(
            "| "
            + title
            + " | "
            + " | ".join(_fmt(row[label]) for label in RUN_ORDER)
            + f" | {_fmt(row['delta_stable_vs_base'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Routing And Hyper-State Diagnostics",
            "",
            "| Metric | " + " | ".join(RUN_ORDER) + " | stable-base |",
            "|" + ":--|" + "--:|" * len(RUN_ORDER) + "--:|",
        ]
    )
    for metric in diag_metrics:
        values = {label: run_diag[label].get(metric, {}).get("mean", math.nan) for label in RUN_ORDER}
        row = _metric_row("diagnostic", metric, values)
        md_lines.append(
            "| "
            + metric
            + " | "
            + " | ".join(_fmt(row[label]) for label in RUN_ORDER)
            + f" | {_fmt(row['delta_stable_vs_base'])} |"
        )

    stable = "global_local_stable_summary"
    base = "base"
    stable_chunk = run_diag[stable].get("time_ratio_chunk_std", {}).get("p50", math.nan)
    stable_pool_chunk = run_diag[stable].get("pooling_alpha_chunk_std", {}).get("p50", math.nan)
    stable_time_stock = run_diag[stable].get("time_ratio_stock_std", {}).get("p50", math.nan)
    stable_pool_stock = run_diag[stable].get("pooling_alpha_stock_std", {}).get("p50", math.nan)
    stable_ic = run_summaries[stable].get("ic_mean", math.nan)
    base_ic = run_summaries[base].get("ic_mean", math.nan)
    stable_ric = run_summaries[stable].get("ric_mean", math.nan)
    base_ric = run_summaries[base].get("ric_mean", math.nan)
    router_local = run_diag[stable].get("router_local_sensitivity", {}).get("p50", math.nan)
    film_local = run_diag[stable].get("film_local_sensitivity", {}).get("p50", math.nan)
    pool_local = run_diag[stable].get("pool_local_sensitivity", {}).get("p50", math.nan)
    go = (
        np.isfinite(stable_chunk)
        and np.isfinite(stable_pool_chunk)
        and np.isfinite(stable_time_stock)
        and np.isfinite(stable_pool_stock)
        and np.isfinite(stable_ic)
        and np.isfinite(base_ic)
        and np.isfinite(stable_ric)
        and np.isfinite(base_ric)
        and np.isfinite(router_local)
        and np.isfinite(film_local)
        and np.isfinite(pool_local)
        and stable_chunk <= 1e-6
        and stable_pool_chunk <= 1e-6
        and stable_time_stock >= 1e-4
        and stable_pool_stock >= 1e-4
        and abs(float(stable_ic) - float(base_ic)) <= 0.002
        and abs(float(stable_ric) - float(base_ric)) <= 0.002
        and router_local > 0.0
        and film_local > 0.0
        and pool_local > 0.0
    )
    md_lines.extend(
        [
            "",
            "## Go/No-Go Gate",
            "",
            f"- `time_ratio_chunk_std.p50` (stable): `{_fmt(stable_chunk)}`",
            f"- `pooling_alpha_chunk_std.p50` (stable): `{_fmt(stable_pool_chunk)}`",
            f"- `time_ratio_stock_std.p50` (stable): `{_fmt(stable_time_stock)}`",
            f"- `pooling_alpha_stock_std.p50` (stable): `{_fmt(stable_pool_stock)}`",
            f"- `|IC_mean(stable) - IC_mean(base)|`: `{_fmt(abs(float(stable_ic) - float(base_ic)) if np.isfinite(stable_ic) and np.isfinite(base_ic) else math.nan)}`",
            f"- `|RankIC_mean(stable) - RankIC_mean(base)|`: `{_fmt(abs(float(stable_ric) - float(base_ric)) if np.isfinite(stable_ric) and np.isfinite(base_ric) else math.nan)}`",
            f"- `router_local_sensitivity.p50` (stable): `{_fmt(router_local)}`",
            f"- `film_local_sensitivity.p50` (stable): `{_fmt(film_local)}`",
            f"- `pool_local_sensitivity.p50` (stable): `{_fmt(pool_local)}`",
            f"- Decision: `{'GO' if go else 'NO-GO / INCOMPLETE'}`",
        ]
    )

    md_lines.extend(
        [
            "",
            "## Regime 2x2",
            "",
            "| Regime | Metric | " + " | ".join(RUN_ORDER) + " | stable-base |",
            "|" + ":--|:--|" + "--:|" * len(RUN_ORDER) + "--:|",
        ]
    )
    for regime_name in regime_names:
        for metric in ("RankIC_mean", "time_ratio_mean"):
            values = {}
            for label in RUN_ORDER:
                regime_2x2 = run_regime[label].get("regime_2x2")
                values[label] = _extract_bucket_metric(
                    regime_2x2,
                    bucket_col="Regime",
                    bucket_value=regime_name,
                    metric=metric,
                )
            row = _metric_row("regime_2x2", f"{regime_name}::{metric}", values)
            md_lines.append(
                "| "
                + regime_name
                + " | "
                + metric
                + " | "
                + " | ".join(_fmt(row[label]) for label in RUN_ORDER)
                + f" | {_fmt(row['delta_stable_vs_base'])} |"
            )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Wrote markdown comparison: {out_md}")
    print(f"[OK] Wrote csv comparison: {out_csv}")


if __name__ == "__main__":
    main()
