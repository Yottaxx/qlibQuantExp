from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def normalize_dt_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).normalize()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert(None)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _finite_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.dropna()


def build_fit_mask(
    index: pd.DatetimeIndex,
    ranges: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    idx = pd.to_datetime(index).normalize()
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    mask = np.zeros((len(idx),), dtype=bool)
    for s, e in ranges:
        s = pd.Timestamp(s).normalize()
        e = pd.Timestamp(e).normalize()
        mask |= (idx >= s) & (idx <= e)
    return mask


def fit_quantile_edges(
    s: pd.Series,
    *,
    quantiles: Sequence[float] = (1.0 / 3.0, 2.0 / 3.0),
) -> Optional[List[float]]:
    """
    Fit split edges on `s` using quantiles (train-only recommended).

    Returns
    -------
    - None if s has <2 unique values.
    - A sorted list of 1 or 2 unique edges.
    """
    x = _finite_series(s)
    if x.nunique() < 2:
        return None

    qs = [float(q) for q in quantiles]
    qs = [q for q in qs if 0.0 < q < 1.0]
    if not qs:
        return None

    try:
        edges = [float(v) for v in x.quantile(qs).to_list()]
    except Exception:
        return None

    edges = [e for e in edges if np.isfinite(e)]
    edges = sorted(set(edges))
    if not edges:
        return None
    if len(edges) > 2:
        edges = edges[:2]
    return edges


def assign_quantile_bucket(s: pd.Series, *, edges: List[float]) -> pd.Series:
    """
    Assign Low/Mid/High or Low/High buckets using pre-fit edges.
    """
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(index=s.index, dtype=object)
    if not edges:
        out[:] = "All"
        return out

    edges = [float(e) for e in edges if np.isfinite(e)]
    edges = sorted(edges)

    if len(edges) == 1:
        e0 = edges[0]
        out[x <= e0] = "Low"
        out[x > e0] = "High"
        return out

    e1, e2 = edges[0], edges[1]
    if not (e1 < e2):
        out[x <= e1] = "Low"
        out[x > e1] = "High"
        return out

    out[x <= e1] = "Low"
    out[(x > e1) & (x <= e2)] = "Mid"
    out[x > e2] = "High"
    return out


def assign_regime_2x2(
    pc1: pd.Series,
    tail: pd.Series,
    *,
    pc1_median: float,
    tail_median: float,
) -> pd.Series:
    pc1 = pd.to_numeric(pc1, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tail = pd.to_numeric(tail, errors="coerce").replace([np.inf, -np.inf], np.nan)
    regime = pd.Series(index=pc1.index, dtype=object)
    valid = pc1.notna() & tail.notna()
    pc1_high = pc1 >= float(pc1_median)
    tail_high = tail >= float(tail_median)
    regime.loc[valid & pc1_high & tail_high] = "High-PC1 / High-Tail"
    regime.loc[valid & pc1_high & ~tail_high] = "High-PC1 / Low-Tail"
    regime.loc[valid & ~pc1_high & tail_high] = "Low-PC1 / High-Tail"
    regime.loc[valid & ~pc1_high & ~tail_high] = "Low-PC1 / Low-Tail"
    return regime


def _series_ir(x: pd.Series) -> float:
    x = _finite_series(x)
    if len(x) < 2:
        return np.nan
    m = float(x.mean())
    sd = float(x.std())
    return (m / sd) if sd > 0 else np.nan


def _series_quantiles(x: pd.Series, qs: Sequence[float]) -> Dict[str, float]:
    x = _finite_series(x)
    if len(x) == 0:
        return {}
    try:
        qv = x.quantile(list(qs))
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for q in qs:
        k = f"p{int(round(100 * float(q))):d}"
        try:
            out[k] = float(qv.loc[q])
        except Exception:
            out[k] = np.nan
    return out


def summarize_by_bucket(
    df: pd.DataFrame,
    *,
    bucket_col: str,
    bucket_name: str,
    perf_cols: Sequence[str] = ("rank_ic", "ic"),
    router_cols: Sequence[str] = (
        "time_ratio",
        "gate_entropy",
        "time_tau",
        "time_half_life",
        "factor_gate_mean",
        "factor_gate_std",
        "factor_gate_entropy",
        "factor_gate_topk_mass_5",
        "factor_gate_topk_mass_10",
    ),
    quantiles: Sequence[float] = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or bucket_col not in df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for name, sub in df.groupby(bucket_col):
        if name is None or (isinstance(name, float) and not np.isfinite(name)):
            continue
        row: Dict[str, object] = {bucket_name: str(name), "Days": int(len(sub))}

        # performance: mean + IR
        for col in perf_cols:
            if col not in sub.columns:
                continue
            label = "RankIC" if col == "rank_ic" else ("IC" if col == "ic" else col)
            x = _finite_series(sub[col])
            row[f"{label}_mean"] = float(x.mean()) if len(x) > 0 else np.nan
            row[f"{label}_IR"] = _series_ir(sub[col])

        # router: mean + quantiles
        for col in router_cols:
            if col not in sub.columns:
                continue
            x = _finite_series(sub[col])
            row[f"{col}_mean"] = float(x.mean()) if len(x) > 0 else np.nan
            qd = _series_quantiles(sub[col], quantiles)
            for k, v in qd.items():
                row[f"{col}_{k}"] = v

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if bucket_name in out.columns:
        out = out.sort_values([bucket_name]).reset_index(drop=True)
    return out


def spearman_rho(x: pd.Series, y: pd.Series) -> Tuple[float, int]:
    df = pd.concat([x, y], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(df))
    if n < 3:
        return np.nan, n
    rx = df.iloc[:, 0].rank(method="average")
    ry = df.iloc[:, 1].rank(method="average")
    rho = rx.corr(ry)
    return float(rho) if rho is not None else np.nan, n


def spearman_corr_table(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    warned_no_scipy = False
    for feat in features:
        if feat not in df.columns:
            continue
        for metric in metrics:
            if metric not in df.columns:
                continue
            rho, n = spearman_rho(df[feat], df[metric])
            t_stat = np.nan
            if n > 2 and np.isfinite(rho) and abs(rho) < 1:
                t_stat = float(rho * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho)))

            p_val = np.nan
            if n > 2 and np.isfinite(t_stat):
                try:
                    from scipy.stats import t as tdist  # type: ignore

                    p_val = float(2.0 * tdist.sf(abs(t_stat), df=n - 2))
                except Exception:
                    if not warned_no_scipy:
                        warnings.warn(
                            "scipy is not available; skipping correlation p-value computation "
                            "(Spearman rho and t-stat are still reported).",
                            RuntimeWarning,
                        )
                        warned_no_scipy = True
                    p_val = np.nan

            rows.append(
                {
                    "Feature": str(feat),
                    "Metric": str(metric),
                    "N": int(n),
                    "SpearmanR": float(rho) if np.isfinite(rho) else np.nan,
                    "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                    "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_abs"] = out["SpearmanR"].abs()
    out = out.sort_values(["_abs", "N"], ascending=[False, False]).drop(columns=["_abs"]).reset_index(drop=True)
    return out


def save_scatter_grid(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    metrics: Sequence[str],
    out_path: str | Path,
    title: str,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    feats = [f for f in features if f in df.columns]
    mets = [m for m in metrics if m in df.columns]
    if not feats or not mets:
        return None

    out_path = Path(out_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    n_rows = len(mets)
    n_cols = len(feats)
    fig_w = max(6.0, 3.2 * n_cols)
    fig_h = max(3.0, 2.6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(title)

    for i, metric in enumerate(mets):
        for j, feat in enumerate(feats):
            ax = axes[i][j]
            x = pd.to_numeric(df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan)
            y = pd.to_numeric(df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
            m = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
            xs = x[m].to_numpy(dtype=float)
            ys = y[m].to_numpy(dtype=float)

            ax.scatter(xs, ys, s=16, alpha=0.8)
            ax.set_xlabel(feat)
            ax.set_ylabel(metric)

            rho, n = spearman_rho(x, y)
            ax.set_title(f"ρ={rho:.2f}, n={n}" if np.isfinite(rho) else f"n={n}")

            if n >= 2 and np.isfinite(xs).any() and np.isfinite(ys).any():
                try:
                    if np.nanstd(xs) > 0:
                        k, b = np.polyfit(xs, ys, 1)
                        x0 = float(np.nanmin(xs))
                        x1 = float(np.nanmax(xs))
                        ax.plot([x0, x1], [k * x0 + b, k * x1 + b], color="C1", linewidth=1.0)
                except Exception:
                    pass

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return out_path.name
    except Exception:
        return None
    finally:
        plt.close(fig)
