#!/usr/bin/env python3
"""
Generate thesis-ready tables (market-state RankICIR, resource allocation value)
and the "three lists" outputs from a single MLflow run folder.

This script is intentionally lightweight:
- Uses run artifacts (pred/label, sig_analysis, portfolio_analysis).
- Re-implements Qlib's default risk_analysis(mode="sum") to keep the computation
  explicit and auditable.
- Optionally uses qlib to (1) override benchmark return series and (2) validate
  that analyzed instruments stay within the training universe (no cross-universe).
- Builds/reuses a global A-share security master cache under `out_dir`
  (default: `security_master_ashare_em.csv`) for stable 股票名称/行业/行业代码 lookup.

Typical usage
-------------
python scripts/generate_thesis_tables.py --run_id <mlflow_run_id>

Examples (reproducible)
-----------------------

CSI300 (train/infer on CSI300; benchmark=SH000300; snapshots cover all regimes)

python scripts/generate_thesis_tables.py ^
  --run_id <RUN_ID_CSI300> ^
  --out_dir artifacts/thesis_outputs ^
  --as_of_dates regime ^
  --market_state artifacts/market_state/daily_market_field_csi300.pkl

CSI800 (train/infer on CSI800; compare benchmark migration to SH000906; snapshots cover all regimes)

python scripts/generate_thesis_tables.py ^
  --run_id <RUN_ID_CSI800> ^
  --out_dir artifacts/thesis_outputs ^
  --as_of_dates regime ^
  --market_state artifacts/market_state/daily_market_field_csi800.pkl ^
  --benchmark_override SH000906

Example (this repo)
-------------------
python scripts/generate_thesis_tables.py ^
  --run_id e395e6d51e8e4c88b58de249342c085f ^
  --market_state artifacts/market_state/daily_market_field_csi300.pkl ^
  --out_dir artifacts/thesis_outputs ^
  --as_of 2022-12-30


python scripts/generate_thesis_tables.py 
  --run_id <RUN_ID> 
  --out_dir artifacts/thesis_outputs 
  --export_sw_industry_years 2020-2025
"""

from __future__ import annotations

import argparse
import math
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.utils.market_state import load_market_state_analysis_df


TRADING_DAYS_CN = 238  # align with qlib.contrib.evaluate.risk_analysis(freq="day")


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _as_date(x: str) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def _normalize_dt_index(idx: Iterable[Any]) -> pd.DatetimeIndex:
    dti = pd.to_datetime(pd.Index(idx)).normalize()
    if getattr(dti, "tz", None) is not None:
        dti = dti.tz_convert(None)
    return dti


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _unique_pred_dates(pred: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(pred.index, pd.MultiIndex) or pred.index.nlevels != 2:
        raise ValueError("pred.pkl must be a DataFrame with MultiIndex(datetime, instrument).")
    d = _normalize_dt_index(pred.index.get_level_values(0))
    return pd.DatetimeIndex(sorted(pd.Index(d).unique()))


def _resolve_as_of_dates(requested: List[pd.Timestamp], *, available: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """
    Snap requested calendar dates to the last available trading date <= requested.
    Keeps order and removes duplicates.
    """
    out: List[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    av = pd.DatetimeIndex(available).sort_values()
    for req in requested:
        dt = pd.Timestamp(req).normalize()
        pos = int(av.searchsorted(dt, side="right") - 1)
        if pos < 0:
            continue
        hit = pd.Timestamp(av[pos]).normalize()
        if hit not in seen:
            out.append(hit)
            seen.add(hit)
    return out


def _parse_as_of_dates_arg(
    arg: str,
    *,
    artifacts_dir: Path,
    available: pd.DatetimeIndex,
) -> List[pd.Timestamp]:
    s = str(arg or "").strip()
    if not s:
        return []

    s_lower = s.lower()
    req: List[pd.Timestamp] = []

    if s_lower in {"attn", "attention"}:
        # Prefer the exact visualization snapshot dates if present.
        try:
            attn = _read_pickle(artifacts_dir / "st_disentangle_attn_maps")
            if isinstance(attn, dict) and attn:
                for k in attn.keys():
                    try:
                        req.append(pd.Timestamp(str(k)).normalize())
                    except Exception:
                        continue
        except Exception:
            req = []

    if not req:
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        for p in parts:
            try:
                req.append(pd.Timestamp(p).normalize())
            except Exception:
                continue

    return _resolve_as_of_dates(req, available=available)


def _log_progress(stage: str, current: int, total: int, *, width: int = 30) -> None:
    if total <= 0:
        total = 1
    ratio = min(max(float(current) / float(total), 0.0), 1.0)
    fill = int(ratio * width)
    bar = "#" * fill + "-" * (width - fill)
    print(f"[{stage}] |{bar}| {current}/{total}", file=sys.stderr)


def _find_run_dir(mlruns_dir: Path, run_id: str) -> Path:
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if not mlruns_dir.exists():
        raise FileNotFoundError(f"mlruns_dir not found: {mlruns_dir}")

    # Layout: mlruns/<experiment_id>/<run_id>/
    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        cand = exp_dir / run_id
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Run '{run_id}' not found under '{mlruns_dir}'.")


def _safe_copy2(src: Path, dst_dir: Path) -> Optional[Path]:
    if not isinstance(src, Path):
        return None
    if not src.exists() or not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = (dst_dir / src.name).resolve()
    try:
        if src.resolve() == dst:
            return dst
    except Exception:
        pass
    try:
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def _bundle_evidence(run_dir: Path, *, out_dir: Path, run_id: str) -> Dict[str, Any]:
    """
    Collect and copy run evidence (figures/HTML) into out_dir for thesis writing.

    Source: MLflow run folder (same level as meta.yaml, *.png, *.html).
    Output: out_dir/evidence_<run_id>/{figures,html,tables}/
    """
    root = (out_dir / f"evidence_{str(run_id).strip()}").resolve()
    figures_dir = root / "figures"
    html_dir = root / "html"
    tables_dir = root / "tables"
    for d in [figures_dir, html_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copied_figures: List[str] = []
    copied_html: List[str] = []
    copied_tables: List[str] = []
    copied_misc: List[str] = []

    # Copy key metadata / reports
    for name in ["meta.yaml", "kdd_report.md"]:
        p = _safe_copy2(run_dir / name, root)
        if p is not None:
            copied_misc.append(p.name)

    # Copy run-level CSVs (small diagnostics)
    for p_src in sorted(run_dir.glob("*.csv")):
        p = _safe_copy2(p_src, tables_dir)
        if p is not None:
            copied_tables.append(p.name)

    # Copy figures
    for p_src in sorted(run_dir.glob("*.png")):
        p = _safe_copy2(p_src, figures_dir)
        if p is not None:
            copied_figures.append(p.name)

    # Copy HTML reports (with plotly runtime if present)
    for p_src in sorted(run_dir.glob("*.html")):
        p = _safe_copy2(p_src, html_dir)
        if p is not None:
            copied_html.append(p.name)
    _safe_copy2(run_dir / "plotly.min.js", html_dir)

    return {
        "root": str(root),
        "figures_dir": str(figures_dir),
        "html_dir": str(html_dir),
        "tables_dir": str(tables_dir),
        "figures": copied_figures,
        "html": copied_html,
        "tables": copied_tables,
        "misc": copied_misc,
    }


def _risk_analysis_sum(r: pd.Series, *, n_trading_days: int = TRADING_DAYS_CN) -> Dict[str, float]:
    r = pd.to_numeric(r, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "annualized_return": float("nan"),
            "information_ratio": float("nan"),
            "max_drawdown": float("nan"),
        }

    mean = float(r.mean())
    std = float(r.std(ddof=1))
    annualized_return = mean * float(n_trading_days)
    # qlib default drawdown: on arithmetic cumulative returns
    max_drawdown = float((r.cumsum() - r.cumsum().cummax()).min())
    information_ratio = float(mean / std * math.sqrt(n_trading_days)) if std > 0 else float("nan")

    return {
        "mean": mean,
        "std": std,
        "annualized_return": annualized_return,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
    }


def _series_ir(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd) if sd > 0 else float("nan")


def _tail_streak(flags: pd.Series) -> int:
    arr = pd.Series(flags).fillna(False).astype(bool).to_numpy()
    streak = 0
    for v in arr[::-1]:
        if v:
            streak += 1
        else:
            break
    return int(streak)


@dataclass(frozen=True)
class Segments:
    train: Tuple[pd.Timestamp, pd.Timestamp]
    valid: Tuple[pd.Timestamp, pd.Timestamp]
    test: Tuple[pd.Timestamp, pd.Timestamp]


def _load_segments(run_conf_resolved: dict) -> Segments:
    segs = (
        (run_conf_resolved.get("data_conf", {}) or {})
        .get("kwargs", {})
        .get("segments", {})
        or {}
    )
    train = segs.get("train", None)
    valid = segs.get("valid", None)
    test = segs.get("test", None)
    if not (isinstance(train, (list, tuple)) and len(train) == 2):
        raise KeyError("Missing data_conf.kwargs.segments.train in run_conf_resolved")
    if not (isinstance(valid, (list, tuple)) and len(valid) == 2):
        raise KeyError("Missing data_conf.kwargs.segments.valid in run_conf_resolved")
    if not (isinstance(test, (list, tuple)) and len(test) == 2):
        raise KeyError("Missing data_conf.kwargs.segments.test in run_conf_resolved")
    return Segments(
        train=(_as_date(train[0]), _as_date(train[1])),
        valid=(_as_date(valid[0]), _as_date(valid[1])),
        test=(_as_date(test[0]), _as_date(test[1])),
    )


def _load_universe_name(run_conf_resolved: dict) -> str:
    # best-effort: pull the instruments name used by the handler
    handler = (
        (run_conf_resolved.get("data_conf", {}) or {})
        .get("kwargs", {})
        .get("handler", {})
        or {}
    )
    inst = (handler.get("kwargs", {}) or {}).get("instruments", None)
    return str(inst) if inst is not None else ""


def _load_universe_spec(run_conf_resolved: dict) -> Any:
    handler = (
        (run_conf_resolved.get("data_conf", {}) or {})
        .get("kwargs", {})
        .get("handler", {})
        or {}
    )
    return (handler.get("kwargs", {}) or {}).get("instruments", None)


def _try_get_universe_instruments(
    universe_spec: Any,
    *,
    provider_uri: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Optional[set[str]]:
    """
    Best-effort universe instrument set for strict alignment checks.

    - If universe_spec is list/tuple/set: use it directly.
    - If universe_spec is a string like "csi300"/"csi800": resolve via qlib stock pool.
      Note: qlib may return the union across history for an index pool, which is OK for
      detecting *extra* instruments.
    """
    if universe_spec is None:
        return None
    if isinstance(universe_spec, (list, tuple, set)):
        out = {str(x).strip() for x in universe_spec if str(x).strip()}
        return out if out else None
    if not isinstance(universe_spec, str):
        return None

    market = str(universe_spec).strip()
    if not market:
        return None
    try:
        import qlib
        from qlib.constant import REG_CN
        from qlib.data import D
    except Exception:
        return None

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    conf = D.instruments(market=market)
    # For stock pools, qlib warns start/end do not take effect; still OK for extra-check.
    ins = D.list_instruments(conf, start_time=str(start.date()) if start is not None else None, end_time=str(end.date()) if end is not None else None, as_list=True)
    out = {str(x).strip() for x in ins if str(x).strip()}
    return out if out else None


def _fit_tercile_edges(s: pd.Series, *, fit_range: Tuple[pd.Timestamp, pd.Timestamp]) -> Tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.loc[(s.index >= fit_range[0]) & (s.index <= fit_range[1])].dropna()
    if len(s) < 10:
        raise ValueError("Not enough samples to fit terciles.")
    q1, q2 = s.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    return float(q1), float(q2)


def _assign_vol_regime(vol_20: pd.Series, *, q1: float, q2: float) -> pd.Series:
    vol_20 = pd.to_numeric(vol_20, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(index=vol_20.index, dtype=object)
    out[vol_20 <= q1] = "Risk-on (Low Vol)"
    out[(vol_20 > q1) & (vol_20 <= q2)] = "Neutral (Mid Vol)"
    out[vol_20 > q2] = "Risk-off (High Vol)"
    return out


REGIME_ORDER: Tuple[str, str, str] = ("Risk-on (Low Vol)", "Neutral (Mid Vol)", "Risk-off (High Vol)")


def _pick_regime_snapshot_dates(
    *,
    available: pd.DatetimeIndex,
    market_state: pd.DataFrame,
    segments: Segments,
    q1: float,
    q2: float,
    vol_col: str = "market_vol_20",
) -> Dict[str, pd.Timestamp]:
    """
    Pick representative dates within the test segment for each market regime.

    Strategy: within each regime, choose the median date (by time order).
    """
    if vol_col not in market_state.columns:
        return {}
    if not (np.isfinite(q1) and np.isfinite(q2)):
        return {}
    if len(available) == 0:
        return {}

    ms = market_state.copy()
    ms.index = _normalize_dt_index(ms.index)
    ms.sort_index(inplace=True)

    dates = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in available])
    dates = dates[(dates >= segments.test[0]) & (dates <= segments.test[1])]
    if len(dates) == 0:
        return {}

    vol = pd.to_numeric(ms[vol_col], errors="coerce")
    vol = vol.reindex(dates)
    regime = _assign_vol_regime(vol, q1=q1, q2=q2)

    out: Dict[str, pd.Timestamp] = {}
    for name in REGIME_ORDER:
        idx = pd.DatetimeIndex(regime.index[regime == name])
        if len(idx) == 0:
            continue
        out[name] = pd.Timestamp(idx[len(idx) // 2]).normalize()
    return out


def _regime_rankic_table(
    *,
    ric: pd.Series,
    ic: pd.Series,
    market_state: pd.DataFrame,
    segments: Segments,
    vol_col: str = "market_vol_20",
) -> pd.DataFrame:
    if vol_col not in market_state.columns:
        raise KeyError(f"market_state is missing required column: {vol_col}")

    ms = market_state.copy()
    ms.index = _normalize_dt_index(ms.index)
    ms.sort_index(inplace=True)

    vol = ms[vol_col]
    q1, q2 = _fit_tercile_edges(vol, fit_range=segments.train)
    regime = _assign_vol_regime(vol, q1=q1, q2=q2)

    ric = ric.copy()
    ric.index = _normalize_dt_index(ric.index)
    ic = ic.copy()
    ic.index = _normalize_dt_index(ic.index)

    test_mask = (regime.index >= segments.test[0]) & (regime.index <= segments.test[1])
    regime_test = regime.loc[test_mask].rename("market_state")

    joined = (
        pd.DataFrame({"RankIC": ric, "IC": ic})
        .join(regime_test, how="inner")
        .dropna(subset=["RankIC"], how="all")
    )
    if joined.empty:
        raise ValueError("No overlapping dates between sig_analysis and market_state on test range.")

    rows: List[Dict[str, object]] = []
    for name, sub in joined.groupby("market_state"):
        x = pd.to_numeric(sub["RankIC"], errors="coerce")
        y = pd.to_numeric(sub["IC"], errors="coerce")
        rows.append(
            {
                "市场状态": str(name),
                "样本天数": int(x.dropna().shape[0]),
                "Rank IC": float(x.mean()),
                "Rank ICIR": _series_ir(x),
                "IC": float(y.mean()),
                "ICIR": _series_ir(y),
                "_q1": q1,
                "_q2": q2,
            }
        )
    out = pd.DataFrame(rows).sort_values("市场状态").reset_index(drop=True)
    return out


def _resource_allocation_table(report_df: pd.DataFrame) -> pd.DataFrame:
    df = report_df.copy()
    df.index = _normalize_dt_index(df.index)
    for c in ["return", "cost", "bench"]:
        if c not in df.columns:
            raise KeyError(f"portfolio report is missing required column: {c}")

    ret_gross = pd.to_numeric(df["return"], errors="coerce")
    cost = pd.to_numeric(df["cost"], errors="coerce")
    bench = pd.to_numeric(df["bench"], errors="coerce")
    ret_net = ret_gross - cost

    top10 = _risk_analysis_sum(ret_net)
    benchmark = _risk_analysis_sum(bench)
    excess_wo_cost = _risk_analysis_sum(ret_gross - bench)
    excess_w_cost = _risk_analysis_sum(ret_net - bench)

    rows = [
        {
            "组合": "Top 10% (Strategy, net of cost)",
            "年化收益": top10["annualized_return"],
            "信息比": top10["information_ratio"],
            "最大回撤": top10["max_drawdown"],
        },
        {
            "组合": "Benchmark",
            "年化收益": benchmark["annualized_return"],
            "信息比": benchmark["information_ratio"],
            "最大回撤": benchmark["max_drawdown"],
        },
        {
            "组合": "Excess Return (vs Benchmark, without cost)",
            "年化收益": excess_wo_cost["annualized_return"],
            "信息比": excess_wo_cost["information_ratio"],
            "最大回撤": excess_wo_cost["max_drawdown"],
        },
        {
            "组合": "Excess Return (vs Benchmark, with cost)",
            "年化收益": excess_w_cost["annualized_return"],
            "信息比": excess_w_cost["information_ratio"],
            "最大回撤": excess_w_cost["max_drawdown"],
        },
    ]
    return pd.DataFrame(rows)


def _load_benchmark_return_series(
    benchmark: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    provider_uri: str,
) -> pd.Series:
    """
    Load benchmark daily return series from Qlib data.

    Notes
    -----
    - Only used when --benchmark_override is set.
    - Uses close-to-close pct_change; first day return is filled as 0.0.
    """
    try:
        import qlib
        from qlib.constant import REG_CN
        from qlib.data import D
    except Exception as e:
        raise RuntimeError("qlib is required for --benchmark_override, but failed to import.") from e

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    close = D.features([str(benchmark)], ["$close"], start_time=str(start.date()), end_time=str(end.date()))
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise RuntimeError(f"Failed to load $close for benchmark={benchmark} from Qlib.")

    if isinstance(close.index, pd.MultiIndex) and close.index.nlevels == 2:
        close = close.droplevel(0)
    close.index = _normalize_dt_index(close.index)
    close = close.sort_index()
    r = pd.to_numeric(close["$close"], errors="coerce").pct_change().fillna(0.0)
    r.name = "bench"
    return r


def _instrument_to_code6(inst: str) -> str:
    s = str(inst).strip()
    if not s:
        return ""
    # Accept: SH600000 / SZ000001 / 600000.SH / 000001.SZ
    if "." in s:
        parts = s.split(".")
        s = parts[0] if parts[0] else parts[-1]
    if s[:2].upper() in {"SH", "SZ"}:
        s = s[2:]
    s = "".join(ch for ch in s if ch.isdigit())
    return s.zfill(6) if s else ""


def _code6_to_instrument_guess(code6: str) -> str:
    """
    Best-effort mapping from 6-digit stock code to qlib-like instrument (SH/SZ/BJ + code).

    Used for SW historical industry maps where the source only provides a 6-digit symbol.
    """
    c = str(code6).strip()
    c = "".join(ch for ch in c if ch.isdigit())
    if not c:
        return ""
    c = c.zfill(6)
    if c.startswith(("6", "9")):
        return f"SH{c}"
    if c.startswith(("0", "2", "3")):
        return f"SZ{c}"
    if c.startswith(("8", "4")):
        return f"BJ{c}"
    return c


def _parse_years_arg(arg: Optional[str]) -> List[int]:
    s = str(arg or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    years: List[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            y1 = int(str(a).strip())
            y2 = int(str(b).strip())
            if y2 < y1:
                y1, y2 = y2, y1
            years.extend(list(range(y1, y2 + 1)))
        else:
            years.append(int(p))
    years = sorted(set(int(y) for y in years))
    return [y for y in years if 1900 <= y <= 2100]


def _get_or_download_sw_industry_hist(*, out_dir: Path, refresh: bool) -> pd.DataFrame:
    """
    SW (申万) stock->industry change history (all stocks, includes delisted).
    Source: https://www.swsresearch.com/.../StockClassifyUse_stock.xls
    """
    cache = (Path(out_dir).expanduser().resolve() / "sw_industry_clf_hist_sw.csv").resolve()
    if (not refresh) and cache.exists():
        return pd.read_csv(cache, dtype=str, encoding="utf-8", keep_default_na=False)
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        raise RuntimeError("AkShare is required to download SW industry history.") from e
    df = ak.stock_industry_clf_hist_sw()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("AkShare returned empty SW industry history.")
    for c in ["symbol", "start_date", "industry_code", "update_time"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    return df


def _get_or_download_sw_category_cninfo(*, out_dir: Path, refresh: bool) -> pd.DataFrame:
    """
    SW (申万) industry code->name mapping from CNINFO (巨潮).
    """
    cache = (Path(out_dir).expanduser().resolve() / "sw_industry_category_cninfo.csv").resolve()
    if (not refresh) and cache.exists():
        return pd.read_csv(cache, dtype=str, encoding="utf-8", keep_default_na=False)
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        raise RuntimeError("AkShare is required to download CNINFO industry category mapping.") from e
    df = ak.stock_industry_category_cninfo(symbol="申银万国行业分类标准")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("AkShare returned empty CNINFO SW category mapping.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    return df


def _get_or_download_sw_index_first_info(*, out_dir: Path, refresh: bool) -> pd.DataFrame:
    """
    SW (申万) first-level industry index codes (e.g., 801010.SI) and names.
    """
    cache = (Path(out_dir).expanduser().resolve() / "sw_index_first_info.csv").resolve()
    if (not refresh) and cache.exists():
        return pd.read_csv(cache, dtype=str, encoding="utf-8", keep_default_na=False)
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        raise RuntimeError("AkShare is required to download SW index first info.") from e
    df = ak.sw_index_first_info()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("AkShare returned empty sw_index_first_info.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    return df


def _build_sw_industry_code_name_map(category_cninfo: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CNINFO SW category table into (industry_code, industry_name).

    CNINFO uses codes like "S340501"; SW history uses "340501".
    """
    if not isinstance(category_cninfo, pd.DataFrame) or category_cninfo.empty:
        return pd.DataFrame(columns=["industry_code", "industry_name"])
    df = category_cninfo.copy()
    if "类目编码" not in df.columns or "类目名称" not in df.columns:
        return pd.DataFrame(columns=["industry_code", "industry_name"])

    df["industry_code"] = df["类目编码"].fillna("").astype(str).str.strip()
    df = df[df["industry_code"].str.startswith("S")]
    df["industry_code"] = df["industry_code"].str[1:]
    df = df[df["industry_code"].str.match(r"^\d{6}$", na=False)]
    df["industry_name"] = df["类目名称"].fillna("").astype(str).str.strip()
    df = df[df["industry_name"] != ""]
    return df[["industry_code", "industry_name"]].drop_duplicates(subset=["industry_code"], keep="first").reset_index(drop=True)


def _build_sw_l1_maps(category_cninfo: pd.DataFrame, index_first_info: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build SW first-level mapping tables:
    - l1_digits -> l1_name (from CNINFO category)
    - l1_name -> index_code (801xxx.SI, from sw_index_first_info)
    """
    l1 = pd.DataFrame(columns=["l1_digits", "l1_name"])
    if isinstance(category_cninfo, pd.DataFrame) and (not category_cninfo.empty):
        df = category_cninfo.copy()
        if "类目编码" in df.columns and "类目名称" in df.columns:
            df["类目编码"] = df["类目编码"].fillna("").astype(str).str.strip()
            df["类目名称"] = df["类目名称"].fillna("").astype(str).str.strip()
            df = df[df["类目编码"].str.match(r"^S\d{2}$", na=False)]
            df["l1_digits"] = df["类目编码"].str[1:]
            df["l1_name"] = df["类目名称"]
            l1 = df[["l1_digits", "l1_name"]].drop_duplicates(subset=["l1_digits"], keep="first").reset_index(drop=True)

    idx = pd.DataFrame(columns=["l1_name", "l1_index_code"])
    if isinstance(index_first_info, pd.DataFrame) and (not index_first_info.empty):
        df = index_first_info.copy()
        if "行业名称" in df.columns and "行业代码" in df.columns:
            df["行业名称"] = df["行业名称"].fillna("").astype(str).str.strip()
            df["行业代码"] = df["行业代码"].fillna("").astype(str).str.strip()
            df = df[(df["行业名称"] != "") & (df["行业代码"] != "")]
            idx = df.rename(columns={"行业名称": "l1_name", "行业代码": "l1_index_code"})[["l1_name", "l1_index_code"]]
            idx = idx.drop_duplicates(subset=["l1_name"], keep="first").reset_index(drop=True)

    return l1, idx


def _build_sw_industry_map_asof_l1(
    *,
    instruments: List[str],
    as_of: pd.Timestamp,
    out_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """
    Build a SW first-level (申万一级) industry map snapshot for given instruments.
    Output columns align with the rest of the script: instrument, code, industry_name, industry_code, source.
    """
    if not instruments:
        return pd.DataFrame()
    out_dir = Path(out_dir).expanduser().resolve()

    # Prepare target code set.
    inst_df = pd.DataFrame({"instrument": list(map(str, instruments))})
    inst_df["instrument"] = inst_df["instrument"].astype(str).str.strip()
    inst_df["code"] = inst_df["instrument"].map(_instrument_to_code6)
    inst_df["code"] = inst_df["code"].fillna("").astype(str).str.strip().str.zfill(6)
    target_codes = sorted(set([c for c in inst_df["code"].tolist() if c and c.isdigit() and len(c) == 6]))
    if not target_codes:
        return pd.DataFrame()

    hist = _get_or_download_sw_industry_hist(out_dir=out_dir, refresh=refresh)
    cat = _get_or_download_sw_category_cninfo(out_dir=out_dir, refresh=refresh)
    try:
        idx_first = _get_or_download_sw_index_first_info(out_dir=out_dir, refresh=refresh)
    except Exception:
        idx_first = None
    l1_digits_map, l1_name_to_idx = _build_sw_l1_maps(cat, idx_first)

    df = hist.copy()
    for c in ["symbol", "start_date", "industry_code"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.zfill(6)
    df["industry_code"] = df["industry_code"].astype(str).str.strip().str.zfill(6)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df[df["start_date"].notna()]
    df = df[df["start_date"] <= pd.Timestamp(as_of).normalize()]
    df = df[df["symbol"].isin(set(target_codes))]
    if df.empty:
        out = inst_df.copy()
        out["industry_name"] = "未知行业"
        out["industry_code"] = "UNKNOWN"
        out["source"] = "akshare_sw_l1"
        return out[["instrument", "code", "industry_name", "industry_code", "source"]]

    df = df.sort_values(["symbol", "start_date"])
    latest = df.groupby("symbol", as_index=False).tail(1)
    latest = latest[["symbol", "industry_code"]].rename(columns={"symbol": "code", "industry_code": "leaf_code"})
    latest["l1_digits"] = latest["leaf_code"].astype(str).str[:2]

    latest = latest.merge(l1_digits_map, on="l1_digits", how="left")
    latest["l1_name"] = latest.get("l1_name", "").fillna("").astype(str).str.strip()
    latest["l1_name"] = latest["l1_name"].where(latest["l1_name"] != "", "未知行业")

    if not l1_name_to_idx.empty:
        latest = latest.merge(l1_name_to_idx, on="l1_name", how="left")
    if "l1_index_code" not in latest.columns:
        latest["l1_index_code"] = ""
    latest["l1_index_code"] = latest["l1_index_code"].fillna("").astype(str).str.strip()

    latest["industry_name"] = latest["l1_name"]
    latest["industry_code"] = latest["l1_index_code"].where(latest["l1_index_code"] != "", "S" + latest["l1_digits"].astype(str))
    latest.loc[latest["industry_name"].isin({"未知行业"}), "industry_code"] = "UNKNOWN"

    out = inst_df.merge(latest[["code", "industry_name", "industry_code"]], on="code", how="left")
    out["industry_name"] = out["industry_name"].fillna("").astype(str).str.strip()
    out["industry_code"] = out["industry_code"].fillna("").astype(str).str.strip()
    out["industry_name"] = out["industry_name"].where(out["industry_name"] != "", "未知行业")
    out["industry_code"] = out["industry_code"].where(out["industry_code"] != "", "UNKNOWN")
    out["source"] = "akshare_sw_l1"
    return out[["instrument", "code", "industry_name", "industry_code", "source"]]


def _build_sw_industry_map_asof(
    *,
    as_of: pd.Timestamp,
    hist: pd.DataFrame,
    code_name_map: pd.DataFrame,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a stock->industry snapshot as-of a given date using SW history.
    """
    if not isinstance(hist, pd.DataFrame) or hist.empty:
        return pd.DataFrame()
    df = hist.copy()
    for c in ["symbol", "start_date", "industry_code"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["symbol"] = df["symbol"].str.zfill(6)
    df["industry_code"] = df["industry_code"].str.zfill(6)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df[df["start_date"].notna()]
    df = df[df["start_date"] <= pd.Timestamp(as_of).normalize()]
    if symbols:
        sym_set = set(str(s).strip().zfill(6) for s in symbols if str(s).strip())
        df = df[df["symbol"].isin(sym_set)]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["symbol", "start_date"])
    latest = df.groupby("symbol", as_index=False).tail(1)

    m = code_name_map.copy() if isinstance(code_name_map, pd.DataFrame) else pd.DataFrame()
    if not m.empty:
        for c in ["industry_code", "industry_name"]:
            if c not in m.columns:
                m[c] = ""
            m[c] = m[c].fillna("").astype(str).str.strip()
        m = m.drop_duplicates(subset=["industry_code"], keep="first")

    out = latest[["symbol", "industry_code"]].copy()
    out["code"] = out["symbol"]
    out["instrument"] = out["symbol"].map(_code6_to_instrument_guess)
    out["stock_name"] = ""
    if not m.empty:
        out = out.merge(m, on="industry_code", how="left")
    if "industry_name" not in out.columns:
        out["industry_name"] = ""
    out["industry_name"] = out["industry_name"].fillna("").astype(str).str.strip()
    out["industry_name"] = out["industry_name"].where(out["industry_name"] != "", "未知行业")
    out["industry_code"] = out["industry_code"].where(out["industry_code"] != "", "UNKNOWN")
    out["as_of"] = pd.Timestamp(as_of).normalize().strftime("%Y-%m-%d")
    out["source"] = "akshare_sw_hist_cninfo"
    front = ["as_of", "instrument", "code", "stock_name", "industry_name", "industry_code", "source"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].sort_values(["instrument"]).reset_index(drop=True)


def _export_sw_industry_maps_by_years(
    *,
    out_dir: Path,
    years: List[int],
    refresh: bool,
) -> Optional[Path]:
    years = sorted(set(int(y) for y in (years or []) if 1900 <= int(y) <= 2100))
    if not years:
        return None
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = _get_or_download_sw_industry_hist(out_dir=out_dir, refresh=refresh)
    cat = _get_or_download_sw_category_cninfo(out_dir=out_dir, refresh=refresh)
    code_name = _build_sw_industry_code_name_map(cat)
    try:
        idx_first = _get_or_download_sw_index_first_info(out_dir=out_dir, refresh=refresh)
    except Exception:
        idx_first = None
    l1_digits_map, l1_name_to_idx = _build_sw_l1_maps(cat, idx_first)

    snaps: List[pd.DataFrame] = []
    for i, y in enumerate(years, start=1):
        _log_progress("SW-Map", i, len(years))
        as_of = pd.Timestamp(f"{int(y)}-12-31").normalize()
        snap = _build_sw_industry_map_asof(as_of=as_of, hist=hist, code_name_map=code_name)
        if snap.empty:
            continue
        # Add SW level-1 fields for thesis usage (avoid overly fine industries).
        try:
            s2 = snap.copy()
            s2["industry_l3_name"] = s2.get("industry_name", "").fillna("").astype(str).str.strip()
            s2["industry_l3_code"] = s2.get("industry_code", "").fillna("").astype(str).str.strip()
            s2["l1_digits"] = s2["industry_l3_code"].astype(str).str[:2]
            if not l1_digits_map.empty:
                s2 = s2.merge(l1_digits_map, on="l1_digits", how="left")
            if "l1_name" not in s2.columns:
                s2["l1_name"] = ""
            s2["industry_l1_name"] = s2["l1_name"].fillna("").astype(str).str.strip()
            s2["industry_l1_name"] = s2["industry_l1_name"].where(s2["industry_l1_name"] != "", "未知行业")
            if not l1_name_to_idx.empty:
                s2 = s2.merge(l1_name_to_idx, on="l1_name", how="left")
            if "l1_index_code" not in s2.columns:
                s2["l1_index_code"] = ""
            s2["l1_index_code"] = s2["l1_index_code"].fillna("").astype(str).str.strip()
            s2["industry_l1_code"] = s2["l1_index_code"].where(s2["l1_index_code"] != "", "S" + s2["l1_digits"].astype(str))
            s2.loc[s2["industry_l1_name"] == "未知行业", "industry_l1_code"] = "UNKNOWN"
            drop_cols = [c for c in ["l1_name", "l1_digits", "l1_index_code"] if c in s2.columns]
            if drop_cols:
                s2 = s2.drop(columns=drop_cols)
            snap = s2
        except Exception:
            pass
        p = out_dir / f"industry_map_sw_{int(y)}.csv"
        snap.to_csv(p, index=False, encoding="utf-8")
        snaps.append(snap)

    if not snaps:
        return None
    merged = pd.concat(snaps, ignore_index=True)
    p_merged = out_dir / f"industry_map_sw_{years[0]}_{years[-1]}.csv"
    merged.to_csv(p_merged, index=False, encoding="utf-8")
    return p_merged


def _fill_missing_industry_meta_sw(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    as_of: pd.Timestamp,
    refresh: bool,
) -> pd.DataFrame:
    """
    Best-effort fill missing/UNKNOWN industry meta using SW history (covers delisted stocks).

    Only fills blanks or placeholders ("未知行业"/"UNKNOWN") and never overwrites existing non-empty values.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "instrument" not in df.columns:
        return df

    out = df.copy()
    if "code" not in out.columns:
        out["code"] = out["instrument"].map(_instrument_to_code6)
    for c in ["code", "industry_name", "industry_code"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str).str.strip()

    miss_mask = (
        out["industry_name"].astype(str).str.strip().isin({"", "未知行业"})
        | out["industry_code"].astype(str).str.strip().isin({"", "UNKNOWN"})
    )
    if not bool(miss_mask.any()):
        return out
    need_fill = miss_mask.to_numpy()

    targets = (
        out.loc[miss_mask, "code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.zfill(6)
        .replace({"000000": ""})
    )
    target_codes = sorted(set([c for c in targets.tolist() if c and c.isdigit() and len(c) == 6]))
    if not target_codes:
        return out

    hist = _get_or_download_sw_industry_hist(out_dir=out_dir, refresh=refresh)
    cat = _get_or_download_sw_category_cninfo(out_dir=out_dir, refresh=refresh)
    code_name = _build_sw_industry_code_name_map(cat)
    snap = _build_sw_industry_map_asof(as_of=pd.Timestamp(as_of).normalize(), hist=hist, code_name_map=code_name, symbols=target_codes)
    if snap.empty:
        return out

    m = snap[["code", "industry_name", "industry_code", "source"]].copy()
    m["code"] = m["code"].fillna("").astype(str).str.strip().str.zfill(6)
    m = m.drop_duplicates(subset=["code"], keep="first")

    out = out.merge(m, on="code", how="left", suffixes=("", "_sw"))
    for c in ["industry_name", "industry_code"]:
        sw = f"{c}_sw"
        if sw not in out.columns:
            continue
        old = out[c].fillna("").astype(str).str.strip()
        new = out[sw].fillna("").astype(str).str.strip()
        old_is_missing = old.isin({"", "未知行业", "UNKNOWN"})
        new_is_valid = (new != "") & (~new.isin({"未知行业", "UNKNOWN"}))
        out[c] = old.where(~(old_is_missing & new_is_valid), new)

    if "source" in out.columns and "source_sw" in out.columns:
        src_new = out["source_sw"].fillna("").astype(str)
        filled = (
            need_fill
            & src_new.ne("").to_numpy()
            & (~out["industry_name"].fillna("").astype(str).str.strip().isin({"", "未知行业"}).to_numpy())
        )
        if bool(filled.any()):
            out.loc[filled, "source"] = src_new.loc[filled].to_numpy()

    drop_cols = [c for c in ["industry_name_sw", "industry_code_sw", "source_sw"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _load_industry_map(path: Path) -> pd.DataFrame:
    # keep_default_na=False prevents empty strings being parsed as NaN.
    df = pd.read_csv(path, dtype=str, encoding="utf-8", keep_default_na=False)
    if "instrument" not in df.columns:
        raise KeyError("industry_map must contain column: instrument")
    if "industry_name" not in df.columns:
        # allow alternative naming
        cand = [c for c in df.columns if c.lower() in {"industry", "industry_name", "sector", "sector_name"}]
        if cand:
            df = df.rename(columns={cand[0]: "industry_name"})
        else:
            raise KeyError("industry_map must contain column: industry_name (or industry/sector)")
    if "industry_code" not in df.columns:
        df["industry_code"] = ""
    if "stock_name" not in df.columns:
        df["stock_name"] = ""
    for c in ["instrument", "industry_name", "industry_code", "stock_name"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
        # sanitize common placeholder strings to empty
        df.loc[df[c].str.lower().isin({"nan", "none", "null", "-", "na"}), c] = ""
    return df


def _enrich_missing_meta_em(
    df: pd.DataFrame,
    *,
    max_fill: int = 200,
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """
    Fill missing stock_name/industry_name via Eastmoney individual info (best-effort).

    This improves coverage when some instruments are absent from current industry boards
    (e.g., delisted / suspended / changed codes).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return df

    out = df.copy()
    if "instrument" not in out.columns:
        return out
    if "code" not in out.columns:
        out["code"] = out["instrument"].map(_instrument_to_code6)
    for c in ["stock_name", "industry_name", "industry_code"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str).str.strip()
        out.loc[out[c].str.lower().isin({"nan", "none", "null", "-", "na"}), c] = ""

    miss = out[(out["stock_name"] == "") | (out["industry_name"] == "")].copy()
    if miss.empty:
        return out

    filled = 0
    for row in miss.head(int(max_fill)).itertuples(index=False):
        code = str(getattr(row, "code", "")).strip()
        if not code:
            continue
        try:
            info = ak.stock_individual_info_em(symbol=code)
        except Exception:
            continue
        if not isinstance(info, pd.DataFrame) or info.empty:
            continue
        if "item" not in info.columns or "value" not in info.columns:
            continue

        items = dict(zip(info["item"].astype(str).str.strip(), info["value"].astype(str).str.strip()))
        name = items.get("股票简称", "").strip()
        industry = items.get("行业", "").strip()
        if name in {"-", "nan", "None", "null"}:
            name = ""
        if industry in {"-", "nan", "None", "null"}:
            industry = ""

        mask = out["code"].astype(str) == code
        if name:
            out.loc[mask & (out["stock_name"] == ""), "stock_name"] = name
        if industry:
            out.loc[mask & (out["industry_name"] == ""), "industry_name"] = industry

        filled += 1
        if sleep_s and filled % 10 == 0:
            time.sleep(float(sleep_s))

    return out


def _fill_industry_code_by_board_name_em(
    df: pd.DataFrame,
    *,
    boards: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fill missing industry_code using Eastmoney industry board list mapping: board_name -> board_code.

    This mainly fixes cases where we can infer industry_name (e.g. from individual info),
    but the stock is not present in current industry board constituents (so industry_code is blank).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "industry_name" not in df.columns or "industry_code" not in df.columns:
        return df

    out = df.copy()
    out["industry_name"] = out["industry_name"].fillna("").astype(str).str.strip()
    out["industry_code"] = out["industry_code"].fillna("").astype(str).str.strip()
    mask = (out["industry_code"] == "") & (out["industry_name"] != "")
    if not bool(mask.any()):
        return out

    b = boards
    if b is None:
        try:
            import akshare as ak  # type: ignore
        except Exception:
            return out
        try:
            b = ak.stock_board_industry_name_em()
        except Exception:
            return out
    if not isinstance(b, pd.DataFrame) or b.empty:
        return out
    if "板块名称" not in b.columns or "板块代码" not in b.columns:
        return out

    name2code: Dict[str, str] = {}
    for name, code in b[["板块名称", "板块代码"]].itertuples(index=False, name=None):
        k = str(name).strip()
        v = str(code).strip()
        if k and v and k not in name2code:
            name2code[k] = v

    out.loc[mask, "industry_code"] = out.loc[mask, "industry_name"].map(lambda x: name2code.get(str(x).strip(), ""))
    return out


def _build_industry_map_em(
    instruments: List[str],
    *,
    max_boards: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build stock->industry mapping using Eastmoney industry boards via AkShare.

    Notes
    -----
    - This is "current classification" (not historical).
    - We only keep the first matched industry if duplicates exist.
    """
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        raise RuntimeError("AkShare is not available; install it or pass --industry_map.") from e

    target_codes = {c for c in (_instrument_to_code6(x) for x in instruments) if c}
    if not target_codes:
        raise ValueError("No valid instruments to build industry map.")

    boards = ak.stock_board_industry_name_em()
    if not isinstance(boards, pd.DataFrame) or boards.empty:
        raise RuntimeError("Failed to fetch industry board list from Eastmoney.")
    if "板块名称" not in boards.columns or "板块代码" not in boards.columns:
        raise RuntimeError(f"Unexpected board schema: {list(boards.columns)}")

    code2ind: Dict[str, Dict[str, str]] = {}
    code2name: Dict[str, str] = {}
    dup: Dict[str, List[str]] = {}
    board_iter = boards[["板块名称", "板块代码"]].itertuples(index=False, name=None)
    for i, (board_name, board_code) in enumerate(board_iter, start=1):
        if max_boards is not None and i > int(max_boards):
            break
        if len(code2ind) >= len(target_codes):
            break

        try:
            cons = ak.stock_board_industry_cons_em(symbol=str(board_code))
        except Exception:
            continue
        if not isinstance(cons, pd.DataFrame) or cons.empty or "代码" not in cons.columns:
            continue
        name_col = "名称" if "名称" in cons.columns else None
        cols = ["代码"] + ([name_col] if name_col else [])
        for row in cons[cols].itertuples(index=False, name=None):
            raw = row[0]
            raw_name = row[1] if name_col else ""
            code = str(raw).strip().zfill(6)
            if code not in target_codes:
                continue
            if code not in code2name and raw_name:
                code2name[code] = str(raw_name).strip()
            if code in code2ind:
                dup.setdefault(code, []).append(str(board_name))
                continue
            code2ind[code] = {"industry_name": str(board_name), "industry_code": str(board_code)}

    rows: List[Dict[str, str]] = []
    for inst in instruments:
        code = _instrument_to_code6(inst)
        meta = code2ind.get(code, {})
        rows.append(
            {
                "instrument": str(inst),
                "code": code,
                "stock_name": code2name.get(code, ""),
                "industry_name": meta.get("industry_name", ""),
                "industry_code": meta.get("industry_code", ""),
                "source": "akshare_em",
            }
        )
    df = pd.DataFrame(rows)
    if dup:
        # keep first assignment; record duplicates for auditing
        df["dup_industries"] = df["code"].map(lambda c: ",".join(dup.get(str(c), [])))
    # Try to fill missing names/industries for better thesis tables.
    df = _enrich_missing_meta_em(df, max_fill=200, sleep_s=0.0)
    # If we got industry_name from other sources, map it back to board_code.
    df = _fill_industry_code_by_board_name_em(df, boards=boards)
    return df


def _get_or_build_industry_map(
    instruments: List[str],
    *,
    industry_map_path: Optional[Path],
    out_dir: Path,
    refresh: bool,
    source: str,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    if industry_map_path is not None:
        p = Path(industry_map_path).expanduser().resolve()
        return _load_industry_map(p), p

    cache = out_dir / f"industry_map_{source}.csv"
    if (not refresh) and cache.exists():
        # Backward-compatible cache upgrade: old caches may not include stock_name.
        try:
            header = pd.read_csv(cache, nrows=0, encoding="utf-8")
            has_stock_name = "stock_name" in header.columns
        except Exception:
            has_stock_name = False
        cached = _load_industry_map(cache)
        if has_stock_name:
            # Normalize line endings/format for downstream tools.
            if source == "em":
                cached = _enrich_missing_meta_em(cached, max_fill=200, sleep_s=0.0)
                cached = _fill_industry_code_by_board_name_em(cached, boards=None)
            cached.to_csv(cache, index=False, encoding="utf-8")
            return cached, cache
        # Missing stock_name in file -> rebuild to enrich outputs.

    if source == "em":
        df = _build_industry_map_em(instruments)
    else:
        raise ValueError(f"Unsupported industry_source: {source} (supported: em)")

    df.to_csv(cache, index=False, encoding="utf-8")
    return df, cache


def _read_instruments_from_provider(
    *,
    provider_uri: str,
    instruments_file: str = "csiall.txt",
    exchanges: Optional[set[str]] = None,
) -> List[str]:
    """
    Read qlib instrument list file to get a stable "A-share universe" code set.
    """
    root = Path(str(provider_uri)).expanduser().resolve()
    p = root / "instruments" / str(instruments_file)
    if not p.exists():
        raise FileNotFoundError(f"qlib instruments file not found: {p}")
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        inst = s.split()[0].strip()
        if not inst:
            continue
        if exchanges is not None:
            ex = inst[:2].upper()
            if ex not in exchanges:
                continue
        out.append(inst)
    return sorted(set(out))


def _load_security_master_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8", keep_default_na=False)
    # Normalize possible Chinese aliases to the internal schema.
    rename: Dict[str, str] = {}
    if "股票名称" in df.columns and "stock_name" not in df.columns:
        rename["股票名称"] = "stock_name"
    if "行业" in df.columns and "industry_name" not in df.columns:
        rename["行业"] = "industry_name"
    if "行业代码" in df.columns and "industry_code" not in df.columns:
        rename["行业代码"] = "industry_code"
    if rename:
        df = df.rename(columns=rename)

    for c in ["instrument", "code", "stock_name", "industry_name", "industry_code", "universe", "industry_source", "source"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
        df.loc[df[c].str.lower().isin({"nan", "none", "null", "-", "na"}), c] = ""
    return df


def _build_security_master_cache_ashare(
    *,
    provider_uri: str,
    industry_source: str,
    instruments_file: str = "csiall.txt",
    exchanges: Optional[set[str]] = None,
) -> pd.DataFrame:
    instruments = _read_instruments_from_provider(
        provider_uri=provider_uri,
        instruments_file=instruments_file,
        exchanges=exchanges,
    )
    if not instruments:
        raise ValueError("No instruments found for security master cache.")

    src = str(industry_source).strip().lower()
    if src == "em":
        industry_map = _build_industry_map_em(instruments)
    else:
        raise ValueError(f"Unsupported industry_source for security master cache: {industry_source} (supported: em)")

    out = industry_map.copy()
    for c in ["instrument", "code", "stock_name", "industry_name", "industry_code", "source"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str).str.strip()

    out["universe"] = "ashare"
    out["industry_source"] = src

    # Ensure the cache is usable as a lookup table (no empty meta fields).
    out["stock_name"] = out["stock_name"].where(out["stock_name"] != "", out["instrument"])
    out["industry_name"] = out["industry_name"].where(out["industry_name"] != "", "未知行业")
    out["industry_code"] = out["industry_code"].where(out["industry_code"] != "", "UNKNOWN")

    # Keep only stable columns; extra audit columns (e.g., dup_industries) are preserved if present.
    front = ["instrument", "code", "stock_name", "industry_name", "industry_code", "universe", "industry_source", "source"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].drop_duplicates(subset=["instrument"], keep="first").sort_values("instrument").reset_index(drop=True)


def _get_or_build_security_master_cache(
    *,
    cache_path: Path,
    provider_uri: str,
    industry_source: str,
    instruments_file: str = "csiall.txt",
    exchanges: Optional[set[str]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = Path(cache_path).expanduser().resolve()
    if (not refresh) and cache_path.exists():
        df = _load_security_master_cache(cache_path)
        # Backward-compatible upgrade: old caches may have blanks -> normalize to placeholders.
        df2 = df.copy()
        df2["stock_name"] = df2["stock_name"].where(df2["stock_name"] != "", df2["instrument"])
        df2["industry_name"] = df2["industry_name"].where(df2["industry_name"] != "", "未知行业")
        df2["industry_code"] = df2["industry_code"].where(df2["industry_code"] != "", "UNKNOWN")
        if not df2.equals(df):
            df2.to_csv(cache_path, index=False, encoding="utf-8")
        return df2

    df = _build_security_master_cache_ashare(
        provider_uri=provider_uri,
        industry_source=industry_source,
        instruments_file=instruments_file,
        exchanges=exchanges,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, encoding="utf-8")
    return df


def _subset_industry_map_from_security_master(
    security_master: pd.DataFrame,
    *,
    instruments: List[str],
) -> pd.DataFrame:
    if not isinstance(security_master, pd.DataFrame) or security_master.empty:
        return pd.DataFrame()
    if not instruments:
        return pd.DataFrame()
    cols = [c for c in ["instrument", "stock_name", "industry_name", "industry_code", "source"] if c in security_master.columns]
    df = security_master[security_master["instrument"].astype(str).isin(set(map(str, instruments)))].copy()
    if cols:
        df = df[cols]
    for c in ["instrument", "stock_name", "industry_name", "industry_code"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    if "source" not in df.columns:
        df["source"] = ""
    return df.drop_duplicates(subset=["instrument"], keep="first").reset_index(drop=True)


def _prepare_pred_window(
    pred: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    window: int,
    quantile: float,
) -> pd.DataFrame:
    if not isinstance(pred.index, pd.MultiIndex) or pred.index.nlevels != 2:
        raise ValueError("pred.pkl must be a DataFrame with MultiIndex(datetime, instrument).")
    if "score" not in pred.columns:
        raise KeyError("pred.pkl must contain a 'score' column.")

    p = pred.copy().reset_index()
    p.columns = ["datetime", "instrument", *p.columns[2:]]
    p["datetime"] = _normalize_dt_index(p["datetime"])
    p = p[p["datetime"] <= as_of]
    if p.empty:
        raise ValueError(f"No prediction rows on/before as_of={as_of.date()}.")

    dates = pd.Index(sorted(p["datetime"].unique()))
    win_dates = dates[-int(window) :]
    p = p[p["datetime"].isin(win_dates)]

    p["rank_pct"] = p.groupby("datetime")["score"].rank(method="average", pct=True, ascending=True)
    p["is_top"] = p["rank_pct"] >= (1.0 - float(quantile))
    p["is_bottom"] = p["rank_pct"] <= float(quantile)
    return p


def _signal_persistence_lists(
    pred: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    window: int,
    quantile: float,
    min_streak: int,
    min_streak_crowd: Optional[int] = None,
    max_items: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = _prepare_pred_window(pred, as_of=as_of, window=window, quantile=quantile)

    latest = p[p["datetime"] == as_of].set_index("instrument")
    latest_rank = latest["rank_pct"]
    latest_score = latest["score"]

    out_rows: List[Dict[str, object]] = []
    for inst, sub in p.sort_values(["instrument", "datetime"]).groupby("instrument"):
        sub = sub.sort_values("datetime")
        is_top = sub["is_top"]
        is_bottom = sub["is_bottom"]
        out_rows.append(
            {
                "instrument": str(inst),
                "rank_pct_latest": float(latest_rank.get(inst, np.nan)),
                "score_latest": float(latest_score.get(inst, np.nan)),
                "top_count": int(is_top.sum()),
                "top_streak": _tail_streak(is_top),
                "bottom_count": int(is_bottom.sum()),
                "bottom_streak": _tail_streak(is_bottom),
            }
        )

    stat = pd.DataFrame(out_rows)

    min_streak_risk = int(min_streak)
    min_streak_crowd_v = int(min_streak_crowd) if min_streak_crowd is not None else min_streak_risk

    risk_warning = stat[stat["bottom_streak"] >= min_streak_risk].copy()
    crowding_warning = stat[stat["top_streak"] >= min_streak_crowd_v].copy()

    risk_warning = risk_warning.sort_values(
        ["bottom_streak", "bottom_count", "rank_pct_latest"],
        ascending=[False, False, True],
    ).head(int(max_items))

    crowding_warning = crowding_warning.sort_values(
        ["top_streak", "top_count", "rank_pct_latest"],
        ascending=[False, False, False],
    ).head(int(max_items))

    return risk_warning.reset_index(drop=True), crowding_warning.reset_index(drop=True)


def _attach_instrument_meta(df: pd.DataFrame, *, industry_map: Optional[pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if industry_map is None or not isinstance(industry_map, pd.DataFrame) or industry_map.empty:
        return df
    if "instrument" not in df.columns or "instrument" not in industry_map.columns:
        return df

    m = industry_map.copy()
    keep = ["instrument"]
    for c in ["stock_name", "industry_name", "industry_code"]:
        if c in m.columns:
            keep.append(c)
    m = m[keep].drop_duplicates()
    for c in keep:
        m[c] = m[c].astype(str).str.strip()

    out = df.merge(m, on="instrument", how="left")
    rename = {}
    if "stock_name" in out.columns:
        rename["stock_name"] = "股票名称"
    if "industry_name" in out.columns:
        rename["industry_name"] = "行业"
    if "industry_code" in out.columns:
        rename["industry_code"] = "行业代码"
    out = out.rename(columns=rename)

    # Ensure no NaN/"nan" leakage in final lists.
    if "股票名称" in out.columns:
        out["股票名称"] = out["股票名称"].fillna("").astype(str).str.strip()
        out.loc[out["股票名称"].str.lower().isin({"nan", "none", "null", "-", "na"}), "股票名称"] = ""
        out.loc[out["股票名称"] == "", "股票名称"] = out["instrument"].astype(str)
    if "行业" in out.columns:
        out["行业"] = out["行业"].fillna("").astype(str).str.strip()
        out.loc[out["行业"].str.lower().isin({"nan", "none", "null", "-", "na"}), "行业"] = ""
        out.loc[out["行业"] == "", "行业"] = "未知行业"
    if "行业代码" in out.columns:
        out["行业代码"] = out["行业代码"].fillna("").astype(str).str.strip()
        out.loc[out["行业代码"].str.lower().isin({"nan", "none", "null", "-", "na"}), "行业代码"] = ""
        out.loc[out["行业代码"] == "", "行业代码"] = "UNKNOWN"

    front = ["instrument"]
    for c in ["股票名称", "行业", "行业代码"]:
        if c in out.columns:
            front.append(c)
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def _build_security_master(
    instruments: List[str],
    *,
    industry_map: pd.DataFrame,
    universe_name: str,
    source: str,
) -> pd.DataFrame:
    if not isinstance(industry_map, pd.DataFrame) or industry_map.empty:
        raise ValueError("industry_map is empty; cannot build security master.")
    if "instrument" not in industry_map.columns:
        raise KeyError("industry_map must contain column: instrument")

    m = industry_map.copy()
    m["instrument"] = m["instrument"].astype(str).str.strip()
    if "code" not in m.columns:
        m["code"] = m["instrument"].map(_instrument_to_code6)
    m["code"] = m["code"].fillna("").astype(str).str.strip()

    for c in ["stock_name", "industry_name", "industry_code"]:
        if c not in m.columns:
            m[c] = ""
        m[c] = m[c].fillna("").astype(str).str.strip()
        m.loc[m[c].str.lower().isin({"nan", "none", "null", "-", "na"}), c] = ""

    m = m.drop_duplicates(subset=["instrument"], keep="first")
    if instruments:
        missing = sorted(set(map(str, instruments)) - set(m["instrument"].tolist()))
        if missing:
            m = pd.concat(
                [
                    m,
                    pd.DataFrame(
                        {
                            "instrument": missing,
                            "code": [(_instrument_to_code6(x)) for x in missing],
                            "stock_name": "",
                            "industry_name": "",
                            "industry_code": "",
                        }
                    ),
                ],
                ignore_index=True,
            )

    out = m.copy()
    out["stock_name"] = out["stock_name"].where(out["stock_name"] != "", out["instrument"])
    out["industry_name"] = out["industry_name"].where(out["industry_name"] != "", "未知行业")
    out["industry_code"] = out["industry_code"].where(out["industry_code"] != "", "UNKNOWN")

    out = out.rename(
        columns={
            "stock_name": "股票名称",
            "industry_name": "行业",
            "industry_code": "行业代码",
        }
    )
    out["universe"] = str(universe_name)
    out["industry_source"] = str(source)

    front = ["instrument", "code", "股票名称", "行业", "行业代码", "universe", "industry_source"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].sort_values("instrument").reset_index(drop=True)


def _industry_rhythm_list(
    pred_window: pd.DataFrame,
    *,
    industry_map: pd.DataFrame,
    downtrend_bottom_ratio: float = 0.25,
) -> pd.DataFrame:
    """
    Aggregate stock-level ranks into an industry "rhythm list".

    Definitions (window-averaged)
    - 景气度: industry mean of daily rank_pct (0~1, higher => better)
    - 拥挤度: (industry share in top decile) / (industry share in universe)
    """
    if not isinstance(pred_window, pd.DataFrame) or pred_window.empty:
        return pd.DataFrame()

    m = industry_map.copy()
    # Ensure a 1:1 mapping to avoid double-counting across industries.
    m = m[["instrument", "industry_name", "industry_code"]].drop_duplicates(subset=["instrument"], keep="first")
    m["instrument"] = m["instrument"].astype(str).str.strip()
    m["industry_name"] = m["industry_name"].astype(str).str.strip()
    m["industry_code"] = m["industry_code"].astype(str).str.strip()

    df = pred_window.merge(m, on="instrument", how="left")
    # Robustly sanitize missing mapping; never let NaN/"nan" leak into grouping keys.
    for c in ["industry_name", "industry_code"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
        df.loc[df[c].str.lower().isin({"nan", "none", "null", "-", "na"}), c] = ""
    df["industry_name"] = df["industry_name"].where(df["industry_name"] != "", "未知行业")
    df["industry_code"] = df["industry_code"].where(df["industry_code"] != "", "UNKNOWN")
    if df.empty:
        return pd.DataFrame()

    totals = (
        df.groupby("datetime")
        .agg(total_n=("instrument", "nunique"), top_total=("is_top", "sum"))
        .reset_index()
    )
    daily = (
        df.groupby(["datetime", "industry_name", "industry_code"])
        .agg(
            n=("instrument", "nunique"),
            prosperity=("rank_pct", "mean"),
            top_count=("is_top", "sum"),
            bottom_count=("is_bottom", "sum"),
        )
        .reset_index()
        .merge(totals, on="datetime", how="left")
    )

    daily["base_share"] = daily["n"] / daily["total_n"].replace(0, np.nan)
    daily["top_share"] = daily["top_count"] / daily["top_total"].replace(0, np.nan)
    daily["crowding"] = daily["top_share"] / daily["base_share"]
    daily["top10_ratio"] = daily["top_count"] / daily["n"].replace(0, np.nan)
    daily["bottom10_ratio"] = daily["bottom_count"] / daily["n"].replace(0, np.nan)

    # IMPORTANT: "成分股数" should reflect the as_of universe (latest date) to keep
    # the table additive (sum ~= universe size, e.g. 300 for CSI300). Using window
    # max would count union across rebalancing dates and can exceed 300.
    latest_dt = pd.Timestamp(daily["datetime"].max()).normalize()
    latest_counts = (
        daily[daily["datetime"] == latest_dt][["industry_name", "industry_code", "n"]]
        .rename(columns={"n": "成分股数"})
        .drop_duplicates(subset=["industry_name", "industry_code"], keep="first")
        .reset_index(drop=True)
    )
    if latest_counts.empty:
        return pd.DataFrame()

    # Only keep industries that exist in the as_of universe.
    daily = daily.merge(latest_counts[["industry_name", "industry_code"]], on=["industry_name", "industry_code"], how="inner")

    agg = (
        daily.groupby(["industry_name", "industry_code"])
        .agg(
            景气度=("prosperity", "mean"),
            拥挤度=("crowding", "mean"),
            Top10占比=("top10_ratio", "mean"),
            Bottom10占比=("bottom10_ratio", "mean"),
        )
        .reset_index()
        .merge(latest_counts, on=["industry_name", "industry_code"], how="left")
        .rename(columns={"industry_name": "行业", "industry_code": "行业代码"})
    )
    if agg.empty:
        return agg

    # percentiles across industries (higher => higher percentile)
    agg["景气度分位"] = agg["景气度"].rank(pct=True)
    agg["拥挤度分位"] = agg["拥挤度"].rank(pct=True)

    def _bucket(p: Any) -> str:
        try:
            v = float(p)
        except Exception:
            return "NA"
        if not np.isfinite(v):
            return "NA"
        if v <= 1.0 / 3.0:
            return "Low"
        if v <= 2.0 / 3.0:
            return "Mid"
        return "High"

    agg["景气度档位"] = agg["景气度分位"].map(_bucket)
    agg["拥挤度档位"] = agg["拥挤度分位"].map(_bucket)

    def _action(row: pd.Series) -> str:
        p = str(row.get("景气度档位", "NA"))
        c = str(row.get("拥挤度档位", "NA"))
        bottom10 = row.get("Bottom10占比", np.nan)
        bottom10 = float(bottom10) if bottom10 is not None and np.isfinite(bottom10) else np.nan
        if p == "High" and c == "Low":
            return "上行顺势"
        if p == "High" and c == "High":
            return "审慎控节奏"
        if p == "Low" and c == "Low":
            # Cold + uncrowded can be a contrarian window, but if many names persistently
            # sit in the bottom tail, treat it as a downtrend/risk regime instead.
            if np.isfinite(bottom10) and bottom10 >= float(downtrend_bottom_ratio):
                return "下行审慎"
            return "下行弱势"
        if p == "Low" and c == "High":
            return "谨慎回避"
        return "中性跟踪"

    agg["动作建议"] = agg.apply(_action, axis=1)

    action_rank = {
        "上行顺势": 0,
        "审慎控节奏": 1,
        "谨慎回避": 2,
        "下行审慎": 3,
        "下行弱势": 4,
        "中性跟踪": 5,
    }
    agg["_action_rank"] = agg["动作建议"].map(lambda x: action_rank.get(str(x), 99))
    agg = agg.sort_values(
        ["_action_rank", "景气度", "拥挤度"],
        ascending=[True, False, True],
    ).drop(columns=["_action_rank"])
    return agg.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate thesis tables & three-list outputs from one MLflow run.")
    ap.add_argument("--run_id", type=str, required=True, help="MLflow run_id (directory name under mlruns/*/).")
    ap.add_argument("--mlruns_dir", type=str, default="mlruns", help="MLflow root dir (default: mlruns).")
    ap.add_argument(
        "--market_state",
        type=str,
        default="artifacts/market_state/daily_market_field_csi300.pkl",
        help="Path to market state asset (default: auto by universe, fallback: artifacts/market_state/daily_market_field_csi300.pkl).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/thesis_outputs",
        help="Output directory (default: artifacts/thesis_outputs).",
    )
    ap.add_argument(
        "--as_of",
        type=str,
        default=None,
        help="As-of date for lists (YYYY-MM-DD). Default: last date in pred.pkl.",
    )
    ap.add_argument(
        "--as_of_dates",
        type=str,
        default=None,
        help=(
            "Comma-separated as_of dates (YYYY-MM-DD), or special values: "
            "'regime' (pick representative dates for Risk-on/Neutral/Risk-off in test), "
            "'attn' (use attention snapshot dates and supplement to cover regimes), "
            "'attn_only' (only attention snapshot dates)."
        ),
    )
    ap.add_argument("--window", type=int, default=20, help="Rolling window (trading days) for persistence (default: 20).")
    ap.add_argument("--quantile", type=float, default=0.10, help="Extreme quantile (default: 0.10 for top/bottom 10%).")
    ap.add_argument("--min_streak", type=int, default=10, help="Min consecutive days in Bottom bucket for risk warning (default: 10).")
    ap.add_argument(
        "--min_streak_crowd",
        type=int,
        default=5,
        help="Min consecutive days in Top bucket for crowding warning (default: 5).",
    )
    ap.add_argument("--max_items", type=int, default=50, help="Max rows per list (default: 50).")
    ap.add_argument("--industry_md_rows", type=int, default=30, help="Max industry rows shown in Markdown per date (default: 30).")
    ap.add_argument("--list_md_rows", type=int, default=20, help="Max risk/crowding rows shown in Markdown per date (default: 20).")
    ap.add_argument(
        "--downtrend_bottom_ratio",
        type=float,
        default=0.25,
        help="When prosperity=Low & crowding=Low but Bottom10占比 >= threshold, action becomes 下行审慎 (default: 0.25).",
    )
    ap.add_argument(
        "--industry_map",
        type=str,
        default=None,
        help="Optional CSV path for stock->industry mapping (columns: instrument, industry_name[, industry_code]).",
    )
    ap.add_argument(
        "--industry_source",
        type=str,
        default="em",
        help="Auto mapping source when --industry_map is not provided (default: em via AkShare Eastmoney).",
    )
    ap.add_argument(
        "--industry_standard",
        type=str,
        default="sw_l1",
        choices=["sw_l1", "em"],
        help="Industry taxonomy used in the three lists (default: sw_l1 = 申万一级).",
    )
    ap.add_argument(
        "--industry_map_refresh",
        action="store_true",
        help="Rebuild cached industry map under out_dir even if it exists.",
    )
    ap.add_argument(
        "--export_sw_industry_years",
        type=str,
        default=None,
        help="Export SW(申万) stock->industry snapshot maps for given years (e.g., 2020-2025 or 2020,2021).",
    )
    ap.add_argument(
        "--sw_industry_refresh",
        action="store_true",
        help="Redownload SW industry history/category caches even if present.",
    )
    ap.add_argument(
        "--security_master_cache",
        type=str,
        default=None,
        help=(
            "Global A-share security master cache CSV (default: <out_dir>/security_master_ashare_<industry_source>.csv). "
            "Built once via AkShare, then reused across runs."
        ),
    )
    ap.add_argument(
        "--security_master_cache_refresh",
        action="store_true",
        help="Rebuild the global security master cache even if it exists.",
    )
    ap.add_argument(
        "--universe_check",
        type=str,
        default="error",
        choices=["off", "warn", "error"],
        help="Whether to validate pred.pkl instruments are within the training universe (default: error).",
    )
    ap.add_argument(
        "--benchmark_override",
        type=str,
        default=None,
        help="Override benchmark code (e.g., SH000906 for CSI800) to recompute excess metrics without re-running backtest.",
    )
    ap.add_argument(
        "--bundle_evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy run HTML/PNG artifacts into out_dir for thesis use (default: true).",
    )
    ap.add_argument(
        "--provider_uri",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data provider URI used when benchmark_override is set.",
    )
    return ap.parse_args()


def main() -> None:
    _configure_stdio()
    args = parse_args()

    mlruns_dir = Path(args.mlruns_dir).expanduser().resolve()
    run_dir = _find_run_dir(mlruns_dir, args.run_id)
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"artifacts dir not found: {artifacts_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_sw_industry_years:
        years = _parse_years_arg(args.export_sw_industry_years)
        try:
            p = _export_sw_industry_maps_by_years(out_dir=out_dir, years=years, refresh=bool(args.sw_industry_refresh))
            if p is not None:
                print(f"[INFO] Exported SW industry maps: {p}")
        except Exception as e:
            print(f"[WARN] export_sw_industry_years failed: {type(e).__name__}: {e}", file=sys.stderr)

    evidence: Optional[Dict[str, Any]] = None
    if bool(args.bundle_evidence):
        try:
            evidence = _bundle_evidence(run_dir, out_dir=out_dir, run_id=args.run_id)
        except Exception as e:
            print(f"[WARN] bundle_evidence failed: {type(e).__name__}: {e}", file=sys.stderr)
            evidence = None

    # --- load run config for segments/universe ---
    run_conf_resolved = _read_pickle(artifacts_dir / "run_conf_resolved")
    if not isinstance(run_conf_resolved, dict):
        raise ValueError("run_conf_resolved artifact is not a dict.")
    segments = _load_segments(run_conf_resolved)
    universe_spec = _load_universe_spec(run_conf_resolved)
    universe = _load_universe_name(run_conf_resolved)

    # --- signal analysis series ---
    ric = pd.read_pickle(artifacts_dir / "sig_analysis" / "ric.pkl")
    ic = pd.read_pickle(artifacts_dir / "sig_analysis" / "ic.pkl")

    # --- portfolio report for resource allocation table ---
    report_df = pd.read_pickle(artifacts_dir / "portfolio_analysis" / "report_normal_1day.pkl")
    df_resource_run = _resource_allocation_table(report_df)

    df_resource_override: Optional[pd.DataFrame] = None
    benchmark_override_used: Optional[str] = None
    if args.benchmark_override:
        benchmark_override_used = str(args.benchmark_override).strip()
        report_override = report_df.copy()
        report_override.index = _normalize_dt_index(report_override.index)
        b = _load_benchmark_return_series(
            benchmark_override_used,
            start=pd.Timestamp(report_override.index.min()).normalize(),
            end=pd.Timestamp(report_override.index.max()).normalize(),
            provider_uri=str(args.provider_uri),
        )
        report_override["bench"] = b.reindex(report_override.index).fillna(0.0)
        df_resource_override = _resource_allocation_table(report_override)

    # --- market state ---
    # Avoid cross-universe: default market_state follows the training universe when possible.
    ms_path_in = str(args.market_state).strip()
    universe_key = str(universe_spec).strip() if isinstance(universe_spec, str) else ""
    if ms_path_in == "artifacts/market_state/daily_market_field_csi300.pkl" and universe_key and universe_key != "csi300":
        cand = Path(f"artifacts/market_state/daily_market_field_{universe_key}.pkl")
        if cand.exists():
            ms_path_in = str(cand)

    ms_path = Path(ms_path_in).expanduser().resolve()
    market_state = load_market_state_analysis_df(ms_path)

    # --- compute tables ---
    df_regime = _regime_rankic_table(ric=ric, ic=ic, market_state=market_state, segments=segments)

    # --- three lists (use pred.pkl) ---
    pred = pd.read_pickle(artifacts_dir / "pred.pkl")
    available_dates = _unique_pred_dates(pred)
    if len(available_dates) == 0:
        raise ValueError("pred.pkl contains no dates.")

    q1 = float(df_regime["_q1"].iloc[0]) if "_q1" in df_regime.columns and len(df_regime) else float("nan")
    q2 = float(df_regime["_q2"].iloc[0]) if "_q2" in df_regime.columns and len(df_regime) else float("nan")
    regime_dates = _pick_regime_snapshot_dates(
        available=available_dates,
        market_state=market_state,
        segments=segments,
        q1=q1,
        q2=q2,
        vol_col="market_vol_20",
    )

    if args.as_of_dates:
        mode = str(args.as_of_dates).strip().lower()
        if mode in {"regime", "regimes", "state", "states"}:
            as_of_dates = _resolve_as_of_dates(list(regime_dates.values()), available=available_dates)
        elif mode in {"attn", "attention", "attn_only"}:
            attn_dates = _parse_as_of_dates_arg("attn", artifacts_dir=artifacts_dir, available=available_dates)
            if not attn_dates and regime_dates and mode in {"attn", "attention"}:
                as_of_dates = _resolve_as_of_dates(list(regime_dates.values()), available=available_dates)
            else:
                as_of_dates = attn_dates
                if mode in {"attn", "attention"} and regime_dates:
                    ms = market_state.copy()
                    ms.index = _normalize_dt_index(ms.index)
                    vol = pd.to_numeric(ms.get("market_vol_20", pd.Series(index=ms.index, dtype=float)), errors="coerce")
                    vol = vol.reindex(pd.DatetimeIndex(as_of_dates))
                    covered = set(_assign_vol_regime(vol, q1=q1, q2=q2).dropna().unique().tolist())
                    for name in REGIME_ORDER:
                        if name not in covered and name in regime_dates:
                            as_of_dates.append(regime_dates[name])
                    as_of_dates = _resolve_as_of_dates(as_of_dates, available=available_dates)
        else:
            as_of_dates = _parse_as_of_dates_arg(
                args.as_of_dates,
                artifacts_dir=artifacts_dir,
                available=available_dates,
            )
    else:
        single = _as_date(args.as_of) if args.as_of else pd.Timestamp(available_dates.max()).normalize()
        as_of_dates = _resolve_as_of_dates([single], available=available_dates)

    # Prefer showing snapshots on test segment (paper tables typically focus on test).
    as_of_dates = [d for d in as_of_dates if segments.test[0] <= d <= segments.test[1]]
    if not as_of_dates:
        raise ValueError("No valid as_of dates resolved within the test segment. Provide --as_of_dates or --as_of.")

    instruments_all = sorted(pd.Index(pred.index.get_level_values(1)).astype(str).unique().tolist())

    # --- strict universe alignment (no cross-universe instruments) ---
    check_mode = str(args.universe_check).lower()
    universe_ins: Optional[set[str]] = None
    extra_instruments: List[str] = []
    if check_mode != "off":
        universe_ins = _try_get_universe_instruments(
            universe_spec,
            provider_uri=str(args.provider_uri),
            start=segments.test[0],
            end=segments.test[1],
        )
        if universe_ins is None and universe_spec is not None and check_mode == "error":
            raise RuntimeError(
                f"universe_check=error but failed to resolve universe instruments for universe={universe!r}. "
                f"Ensure qlib data exists at provider_uri={args.provider_uri!r} or set --universe_check off."
            )
        if universe_ins is not None:
            extra_instruments = sorted(set(instruments_all) - set(universe_ins))
            if extra_instruments:
                sample = ", ".join(extra_instruments[:10])
                msg = (
                    f"pred.pkl contains {len(extra_instruments)} instrument(s) outside universe={universe!r}; "
                    f"e.g. {sample}"
                )
                if check_mode == "error":
                    raise ValueError(msg)
                if check_mode == "warn":
                    print(f"[WARN] {msg}", file=sys.stderr)

    industry_source = str(args.industry_source).strip().lower()
    industry_map: pd.DataFrame
    industry_map_used: Optional[Path]
    if args.industry_map:
        industry_map, industry_map_used = _get_or_build_industry_map(
            instruments_all,
            industry_map_path=Path(args.industry_map),
            out_dir=out_dir,
            refresh=bool(args.industry_map_refresh),
            source=industry_source,
        )
    else:
        cache_path = (
            Path(args.security_master_cache).expanduser().resolve()
            if args.security_master_cache
            else (out_dir / f"security_master_ashare_{industry_source}.csv")
        )
        try:
            sm_cache = _get_or_build_security_master_cache(
                cache_path=cache_path,
                provider_uri=str(args.provider_uri),
                industry_source=industry_source,
                instruments_file="csiall.txt",
                exchanges={"SH", "SZ"},
                refresh=bool(args.security_master_cache_refresh),
            )
            industry_map = _subset_industry_map_from_security_master(sm_cache, instruments=instruments_all)
            industry_map_used = cache_path

            # Best-effort: fill missing/UNKNOWN fields for this run and upsert back.
            miss_mask = (
                industry_map.get("stock_name", "").astype(str).str.strip().eq("")
                | industry_map.get("industry_name", "").astype(str).str.strip().isin({"", "未知行业"})
                | industry_map.get("industry_code", "").astype(str).str.strip().isin({"", "UNKNOWN"})
            )
            if bool(miss_mask.any()):
                miss_n = int(miss_mask.sum())

                # Treat placeholders as missing for enrichment.
                if "industry_name" in industry_map.columns:
                    industry_map["industry_name"] = industry_map["industry_name"].replace({"未知行业": ""})
                if "industry_code" in industry_map.columns:
                    industry_map["industry_code"] = industry_map["industry_code"].replace({"UNKNOWN": ""})

                industry_map = _enrich_missing_meta_em(industry_map, max_fill=miss_n, sleep_s=0.0)
                industry_map = _fill_industry_code_by_board_name_em(industry_map, boards=None)
                try:
                    industry_map = _fill_missing_industry_meta_sw(
                        industry_map,
                        out_dir=out_dir,
                        as_of=max(as_of_dates),
                        refresh=bool(args.sw_industry_refresh),
                    )
                except Exception as e:
                    print(f"[WARN] SW fallback fill failed: {type(e).__name__}: {e}", file=sys.stderr)

                try:
                    # Upsert (only fill blanks) into the global cache to avoid repeated network calls next time.
                    sm = _load_security_master_cache(cache_path) if cache_path.exists() else sm_cache
                    sm_i = sm.set_index("instrument")
                    upd = industry_map.copy()
                    if "code" not in upd.columns:
                        upd["code"] = upd["instrument"].map(_instrument_to_code6)
                    upd_i = upd.set_index("instrument")
                    for c in ["code", "stock_name", "industry_name", "industry_code"]:
                        if c not in sm_i.columns or c not in upd_i.columns:
                            continue
                        old = sm_i[c].fillna("").astype(str)
                        new = upd_i[c].fillna("").astype(str)
                        old_s = old.astype(str).str.strip()
                        new_s = new.astype(str).str.strip()
                        missing_values = {""}
                        if c == "industry_name":
                            missing_values.add("未知行业")
                        if c == "industry_code":
                            missing_values.add("UNKNOWN")
                        replace_mask = old_s.isin(missing_values) & new_s.ne("")
                        sm_i.loc[upd_i.index, c] = old.where(~replace_mask, new)
                    sm = sm_i.reset_index()
                    sm.to_csv(cache_path, index=False, encoding="utf-8")
                except Exception as e:
                    print(f"[WARN] Failed to upsert security_master_cache: {type(e).__name__}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to use security_master_cache; fallback to per-run industry_map. {type(e).__name__}: {e}", file=sys.stderr)
            industry_map, industry_map_used = _get_or_build_industry_map(
                instruments_all,
                industry_map_path=None,
                out_dir=out_dir,
                refresh=bool(args.industry_map_refresh),
                source=industry_source,
            )

    # Final fallback: fill any remaining missing/UNKNOWN industry meta with SW history (covers delisted).
    try:
        industry_map = _fill_missing_industry_meta_sw(
            industry_map,
            out_dir=out_dir,
            as_of=max(as_of_dates),
            refresh=bool(args.sw_industry_refresh),
        )
    except Exception as e:
        print(f"[WARN] Final SW fallback fill failed: {type(e).__name__}: {e}", file=sys.stderr)

    industry_standard = str(args.industry_standard).strip().lower()
    stock_name_map = pd.DataFrame(columns=["instrument", "stock_name"])
    if isinstance(industry_map, pd.DataFrame) and (not industry_map.empty) and "instrument" in industry_map.columns:
        tmp = industry_map.copy()
        if "stock_name" not in tmp.columns:
            tmp["stock_name"] = ""
        stock_name_map = tmp[["instrument", "stock_name"]].copy()
        stock_name_map["instrument"] = stock_name_map["instrument"].astype(str).str.strip()
        stock_name_map["stock_name"] = stock_name_map["stock_name"].fillna("").astype(str).str.strip()
        stock_name_map = stock_name_map.drop_duplicates(subset=["instrument"], keep="first").reset_index(drop=True)

    def _industry_map_for_date(as_of: pd.Timestamp) -> pd.DataFrame:
        if industry_standard == "sw_l1":
            sw_map = _build_sw_industry_map_asof_l1(
                instruments=instruments_all,
                as_of=pd.Timestamp(as_of).normalize(),
                out_dir=out_dir,
                refresh=bool(args.sw_industry_refresh),
            )
            if not stock_name_map.empty:
                sw_map = sw_map.merge(stock_name_map, on="instrument", how="left", suffixes=("", "_base"))
                if "stock_name_base" in sw_map.columns:
                    base = sw_map["stock_name_base"].fillna("").astype(str).str.strip()
                    cur = sw_map.get("stock_name", pd.Series(index=sw_map.index, dtype=str)).fillna("").astype(str).str.strip()
                    sw_map["stock_name"] = cur.where(cur != "", base)
                    sw_map = sw_map.drop(columns=["stock_name_base"])
            if "stock_name" not in sw_map.columns:
                sw_map["stock_name"] = ""
            sw_map["stock_name"] = sw_map["stock_name"].fillna("").astype(str).str.strip()
            sw_map.loc[sw_map["stock_name"] == "", "stock_name"] = sw_map["instrument"].astype(str)
            return sw_map
        return industry_map

    # --- security master: base lookup table (prevents 股票名称/行业/行业代码 showing as NaN in lists) ---
    universe_tag = "".join(ch if ch.isalnum() else "_" for ch in str(universe)) or "universe"
    industry_map_for_master = _industry_map_for_date(max(as_of_dates))
    security_master = _build_security_master(
        instruments_all,
        industry_map=industry_map_for_master,
        universe_name=str(universe),
        source=industry_standard,
    )
    security_master_path = out_dir / f"security_master_{universe_tag}_{args.run_id}.csv"
    security_master.to_csv(security_master_path, index=False, encoding="utf-8")
    security_master_missing = security_master[
        (security_master["行业"] == "未知行业") | (security_master["行业代码"] == "UNKNOWN")
    ].copy()
    security_master_missing_path = out_dir / f"security_master_missing_meta_{universe_tag}_{args.run_id}.csv"
    security_master_missing.to_csv(security_master_missing_path, index=False, encoding="utf-8")
    unknown_industry_n = int((security_master["行业"] == "未知行业").sum())

    snapshots: List[Dict[str, Any]] = []
    for as_of in as_of_dates:
        industry_map_for_date = _industry_map_for_date(as_of)
        pred_window = _prepare_pred_window(
            pred,
            as_of=as_of,
            window=int(args.window),
            quantile=float(args.quantile),
        )
        industry_list = _industry_rhythm_list(
            pred_window,
            industry_map=industry_map_for_date,
            downtrend_bottom_ratio=float(args.downtrend_bottom_ratio),
        )
        risk_list, crowd_list = _signal_persistence_lists(
            pred,
            as_of=as_of,
            window=int(args.window),
            quantile=float(args.quantile),
            min_streak=int(args.min_streak),
            min_streak_crowd=int(args.min_streak_crowd),
            max_items=int(args.max_items),
        )
        risk_list = _attach_instrument_meta(risk_list, industry_map=industry_map_for_date)
        crowd_list = _attach_instrument_meta(crowd_list, industry_map=industry_map_for_date)

        # --- per-date outputs ---
        industry_list.to_csv(out_dir / f"industry_rhythm_list_{as_of.date()}.csv", index=False)
        risk_list.to_csv(out_dir / f"risk_warning_list_{as_of.date()}.csv", index=False)
        crowd_list.to_csv(out_dir / f"crowding_warning_list_{as_of.date()}.csv", index=False)

        snapshots.append(
            {
                "as_of": as_of,
                "industry_list": industry_list,
                "risk_list": risk_list,
                "crowd_list": crowd_list,
            }
        )

    # --- write outputs ---
    df_regime.drop(columns=["_q1", "_q2"], errors="ignore").to_csv(out_dir / "risk_regime_rankicir.csv", index=False)
    df_resource_run.to_csv(out_dir / "resource_allocation_value.csv", index=False)
    if df_resource_override is not None and benchmark_override_used:
        bench_tag = "".join(ch if ch.isalnum() else "_" for ch in benchmark_override_used)
        df_resource_override.to_csv(out_dir / f"resource_allocation_value_benchmark_{bench_tag}.csv", index=False)
    # Per-date CSVs are written in the snapshots loop above.

    def _fmt_pct(x: Any) -> str:
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return "nan"
            return f"{float(x):.2%}"
        except Exception:
            return str(x)

    # markdown-friendly copies
    df_reg_md = df_regime.copy()
    for c in ["Rank IC", "Rank ICIR", "IC", "ICIR"]:
        if c in df_reg_md.columns:
            df_reg_md[c] = df_reg_md[c].map(lambda v: f"{float(v):.4f}" if np.isfinite(v) else "nan")

    def _format_resource_table(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        for c in ["年化收益", "最大回撤"]:
            if c in df_out.columns:
                df_out[c] = df_out[c].map(_fmt_pct)
        if "信息比" in df_out.columns:
            df_out["信息比"] = df_out["信息比"].map(lambda v: f"{float(v):.2f}" if np.isfinite(v) else "nan")
        return df_out

    df_res_md_run = _format_resource_table(df_resource_run)
    df_res_md_override = _format_resource_table(df_resource_override) if df_resource_override is not None else None

    md_lines: List[str] = []
    md_lines.append(f"# Thesis Tables ({args.run_id})\n")
    md_lines.append(f"- Run dir: `{run_dir}`\n")
    if universe:
        md_lines.append(f"- Universe: `{universe}`\n")
    md_lines.append(f"- Market state: `{ms_path}`\n")
    md_lines.append(
        f"- Security master: `{security_master_path.name}` "
        f"(unknown_industry={unknown_industry_n}/{len(security_master)})\n"
        f"- Missing-meta subset: `{security_master_missing_path.name}`\n"
    )
    if evidence is not None:
        md_lines.append(
            f"- Evidence bundle: `evidence_{args.run_id}` "
            f"(figures={len(evidence.get('figures', []))}, html={len(evidence.get('html', []))})\n"
        )
    if universe_ins is not None:
        extra_n = len(extra_instruments) if isinstance(extra_instruments, list) else 0
        md_lines.append(
            f"- Universe alignment: pred_instruments={len(instruments_all)}, "
            f"universe_instruments={len(universe_ins)}, extra={extra_n} "
            f"(check={str(args.universe_check).lower()})\n"
        )
        if extra_n:
            md_lines.append(f"  - Extra sample: {', '.join(extra_instruments[:10])}\n")
    md_lines.append(
        f"- Segments: train={segments.train[0].date()}..{segments.train[1].date()}, "
        f"test={segments.test[0].date()}..{segments.test[1].date()}\n"
    )
    md_lines.append("## 0) 口径与方法（用于审计与复现）\n")
    md_lines.append(
        "- 信号源：`pred.pkl` 的 `score`（日频、截面，索引为 (datetime, instrument)）。\n"
        "- `rank_pct`：对每个交易日，将 `score` 在全样本空间做截面百分位排序（0~1）。\n"
        "- RankIC：`score` 与 `label` 的截面 Spearman 相关（来自 `sig_analysis/ric.pkl` 的日度序列）。\n"
        "- RankICIR：`mean(RankIC) / std(RankIC)`（按日度序列计算，样本越少越不稳定）。\n"
    )
    md_lines.append(
        "- 市场状态划分：使用 `market_state` 中的 `market_vol_20`，在 train 段拟合三分位阈值 q1/q2；\n"
        "  `market_vol_20<=q1` 为 Risk-on，`(q1,q2]` 为 Neutral，`>q2` 为 Risk-off。\n"
    )
    md_lines.append(
        "- 资源配置价值口径：与 Qlib 默认 `risk_analysis(mode=sum, freq=day)` 对齐，N=238：\n"
        "  年化收益=mean*238，信息比=mean/std*sqrt(238)，最大回撤=min(cumsum-cummax)。\n"
        "  策略收益使用 `report_normal_1day.pkl` 的 `(return - cost)`；基准收益使用 `bench`（可用 `--benchmark_override` 替换）。\n"
        "  超额收益（不含成本）=`return - bench`；超额收益（含成本）=`(return - cost) - bench`。\n"
    )
    md_lines.append(
        "- 行业节奏清单：\n"
        f"  行业口径由 `--industry_standard={industry_standard}` 决定（默认 sw_l1=申万一级）。\n"
        "  sw_l1：使用申万行业历史（SW 变动表）在 as_of 回溯到最新行业归属，并用 CNINFO 申万分类标准映射到申万一级；\n"
        "  行业代码优先输出申万一级指数代码（801xxx.SI），缺失时退回到 CNINFO 一级代码（Sxx）。\n"
        "  相关缓存：`sw_industry_clf_hist_sw.csv`、`sw_industry_category_cninfo.csv`、`sw_index_first_info.csv`。\n"
        "  以最近 window 天为样本：\n"
        "  - 成分股数=as_of 当日该行业成分数（避免窗口跨指数调整导致加总>300）。\n"
        "  - 景气度=行业内 `rank_pct` 的窗口均值（越高越“景气”）。\n"
        "  - Top10占比=行业内 Top10% 个股占比（top_count / n）；Bottom10占比=行业内 Bottom10% 个股占比（bottom_count / n）。\n"
        "  - 拥挤度=行业在 Top10% 中的“份额”/行业自身“基准份额”，其中：行业Top10份额=top_count/top_total，行业基准份额=n/total_n。\n"
        "  动作建议：对当期行业横截面景气度/拥挤度做三分位 Low/Mid/High 组合映射；\n"
        "  - High+Low：上行顺势（趋势向上且不拥挤，可顺势加力）\n"
        "  - High+High：审慎控节奏（景气高但拥挤，防集中度/回撤）\n"
        "  - Low+High：谨慎回避（景气弱但仍拥挤，易踩踏）\n"
        "  - Low+Low 且 `Bottom10占比 >= downtrend_bottom_ratio`：下行审慎（弱势且下行面扩散）\n"
        "  - Low+Low 且 `Bottom10占比 < downtrend_bottom_ratio`：下行弱势（弱势但未到扩散性下行）\n"
        "  - 其余：中性跟踪\n"
    )
    md_lines.append(
        "- 风险/拥挤预警清单：在最近 window 天内，若连续 >= min_streak_risk 天处于 Bottom10%（风险预警），\n"
        "  或连续 >= min_streak_crowd 天处于 Top10%（拥挤预警），则触发（输出到 CSV）。\n"
    )

    md_lines.append("## 1) 跨状态稳健性（Risk-on/Neutral/Risk-off by 20d vol）\n")
    md_lines.append(df_reg_md.drop(columns=[c for c in df_reg_md.columns if c.startswith("_")], errors="ignore").to_markdown(index=False) + "\n")
    md_lines.append(
        f"- 注：状态划分使用 `market_vol_20`，分位阈值在 train 段拟合（q1={df_regime['_q1'].iloc[0]:.4f}, q2={df_regime['_q2'].iloc[0]:.4f}）。\n"
    )

    md_lines.append("## 2) 资源配置价值（Top 10% vs Benchmark vs Excess）\n")
    md_lines.append(df_res_md_run.to_markdown(index=False) + "\n")
    md_lines.append("- 注：年化收益/信息比/最大回撤口径与 Qlib 默认 `risk_analysis(mode=sum)` 一致。\n")
    if args.benchmark_override:
        md_lines.append(f"- 注：以下附表为 `--benchmark_override={benchmark_override_used}` 的对比结果（不重跑回测，仅替换基准收益序列）。\n")
        if df_res_md_override is not None and benchmark_override_used:
            bench_tag = "".join(ch if ch.isalnum() else "_" for ch in benchmark_override_used)
            md_lines.append(f"- CSV: `resource_allocation_value_benchmark_{bench_tag}.csv`\n")
            md_lines.append(df_res_md_override.to_markdown(index=False) + "\n")

    md_lines.append("## 3) 三张清单（多日期快照）\n")
    md_lines.append(
        f"- Snapshots: {', '.join(str(pd.Timestamp(d).date()) for d in as_of_dates)}\n"
        f"- Window={int(args.window)} days, quantile={float(args.quantile):.0%}, "
        f"min_streak_risk={int(args.min_streak)}, min_streak_crowd={int(args.min_streak_crowd)}, "
        f"downtrend_bottom_ratio={float(args.downtrend_bottom_ratio):.2f}\n"
    )
    md_lines.append(f"- Industry standard: `{industry_standard}`\n")
    if industry_map_used is not None:
        md_lines.append(f"- Stock meta map (stock_name lookup): `{industry_map_used}`\n")

    # Prepare market_state vol for per-date regime label
    ms = market_state.copy()
    ms.index = _normalize_dt_index(ms.index)
    vol_col = "market_vol_20"
    vol = pd.to_numeric(ms.get(vol_col, pd.Series(index=ms.index, dtype=float)), errors="coerce")
    q1 = float(df_regime["_q1"].iloc[0]) if "_q1" in df_regime.columns and len(df_regime) else np.nan
    q2 = float(df_regime["_q2"].iloc[0]) if "_q2" in df_regime.columns and len(df_regime) else np.nan

    def _regime_label(v: Any) -> str:
        try:
            vv = float(v)
        except Exception:
            return "NA"
        if not np.isfinite(vv) or not (np.isfinite(q1) and np.isfinite(q2)):
            return "NA"
        if vv <= q1:
            return "Risk-on (Low Vol)"
        if vv <= q2:
            return "Neutral (Mid Vol)"
        return "Risk-off (High Vol)"

    for i, snap in enumerate(snapshots, start=1):
        as_of = pd.Timestamp(snap["as_of"]).normalize()
        ind_df: pd.DataFrame = snap["industry_list"]
        risk_df: pd.DataFrame = snap["risk_list"]
        crowd_df: pd.DataFrame = snap["crowd_list"]
        v = vol.reindex([as_of]).iloc[0] if as_of in vol.index else np.nan
        _log_progress("Snapshots", i, len(as_of_dates))
        md_lines.append(f"### 3.{i} as_of={as_of.date()}（market_state={_regime_label(v)}, market_vol_20={float(v):.4f}）\n")

        md_lines.append("#### 行业节奏清单\n")
        md_lines.append(f"- CSV: `industry_rhythm_list_{as_of.date()}.csv`\n")
        if not ind_df.empty:
            rows_total = int(len(ind_df))
            show_n = int(min(rows_total, int(args.industry_md_rows)))
            shown = ind_df.head(show_n).copy()
            if "成分股数" in ind_df.columns:
                total_n = int(pd.to_numeric(ind_df["成分股数"], errors="coerce").fillna(0.0).sum())
                shown_n = int(pd.to_numeric(shown["成分股数"], errors="coerce").fillna(0.0).sum())
                md_lines.append(f"- 展示 {show_n}/{rows_total} 行；成分股数合计：展示={shown_n}，全量={total_n}\n")
            else:
                md_lines.append(f"- 展示 {show_n}/{rows_total} 行（全量见 CSV）\n")
            md_lines.append(shown.to_markdown(index=False) + "\n")
        else:
            md_lines.append("- (no industry list; missing industry mapping)\n")

        md_lines.append("#### 风险预警清单（持续超跌）\n")
        md_lines.append(f"- CSV: `risk_warning_list_{as_of.date()}.csv`\n")
        if not risk_df.empty:
            md_lines.append(risk_df.head(int(args.list_md_rows)).to_markdown(index=False) + "\n")
        else:
            md_lines.append("- (no items triggered)\n")

        md_lines.append("#### 拥挤预警清单（持续超涨）\n")
        md_lines.append(f"- CSV: `crowding_warning_list_{as_of.date()}.csv`\n")
        if not crowd_df.empty:
            md_lines.append(crowd_df.head(int(args.list_md_rows)).to_markdown(index=False) + "\n")
        else:
            md_lines.append("- (no items triggered)\n")

        if evidence is not None:
            fig_dir = out_dir / f"evidence_{args.run_id}" / "figures"
            rel_fig_dir = f"evidence_{args.run_id}/figures"
            attn_figs = [
                f"st_disentangle_attn_factor_{as_of.date()}.png",
                f"st_disentangle_attn_time_{as_of.date()}.png",
                f"st_disentangle_attn_pool_factor_{as_of.date()}.png",
            ]
            hit = [fn for fn in attn_figs if (fig_dir / fn).exists()]
            if hit:
                md_lines.append("#### Attention 可视化（证据链）\n")
                for fn in hit:
                    md_lines.append(f"![]({rel_fig_dir}/{fn})\n")

    if evidence is not None:
        md_lines.append("## 4) 证据包（图表/HTML/审计附件）\n")
        md_lines.append(f"- Dir: `evidence_{args.run_id}`\n")

        fig_dir = out_dir / f"evidence_{args.run_id}" / "figures"
        rel_fig_dir = f"evidence_{args.run_id}/figures"
        key_figs = [
            "train_curves_mse_rankic.png",
            "regime_scatter_perf.png",
            "regime_scatter_router.png",
            "st_disentangle_gate_series_test.png",
        ]
        for fn in key_figs:
            if (fig_dir / fn).exists():
                md_lines.append(f"![]({rel_fig_dir}/{fn})\n")

        html_dir = out_dir / f"evidence_{args.run_id}" / "html"
        rel_html_dir = f"evidence_{args.run_id}/html"
        htmls = sorted([p.name for p in html_dir.glob("*.html")])
        if htmls:
            md_lines.append("### 4.1 HTML 报告（直接点开）\n")
            for fn in htmls:
                md_lines.append(f"- [{fn}]({rel_html_dir}/{fn})\n")

        root_dir = out_dir / f"evidence_{args.run_id}"
        misc = [fn for fn in ["kdd_report.md", "meta.yaml"] if (root_dir / fn).exists()]
        if misc:
            md_lines.append("### 4.2 运行记录\n")
            for fn in misc:
                md_lines.append(f"- [{fn}](evidence_{args.run_id}/{fn})\n")

    md_path = out_dir / f"thesis_tables_{args.run_id}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Wrote: {md_path}")
    print(f"[OK] CSVs in: {out_dir}")


if __name__ == "__main__":
    main()
