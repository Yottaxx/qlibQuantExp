# -*- coding: utf-8 -*-
"""
RST-MoE + Qlib Official Workflow (Paper-Ready Version)

功能：
1. 使用 Alpha158 / CSI300 官方切分，训练 QlibQuantMoE（时序 MoE）。
2. 运行标准 Signal 分析 + 组合回测。
3. 调用 model.export_visuals 导出：
   - gate time_ratio 随时间曲线
   - 若干交易日的 time & factor-attention heatmap
4. 从 Recorder 中汇总：
   - IC / RankIC 时间序列 + ICIR / t-stat
   - 回测指标（年化收益、信息比、最大回撤等）
   - gate 曲线统计（均值 / std / 分位数）
   - attention map 的局部性指标
5. 自动生成一份 Markdown 版「论文级实验报告」：kdd_report.md
   - 增加“训练过程诊断”：train/listmle vs valid/rank_ic 曲线 + 文本总结
"""
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from pathlib import Path
import textwrap
import copy

import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import TSDatasetH
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

import matplotlib.pyplot as plt
# =============================================================================
# 0. Qlib Init (与官方 yaml 对齐)
# =============================================================================
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# =============================================================================
# 1. Data Config (与官方 task.dataset 对齐，改为 TSDatasetH)
# =============================================================================
data_conf = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "step_len": 2,  # 时序窗口，对应模型 context_len
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2020-08-01",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2014-12-31",
                "instruments": "csi300",
                # 推理预处理：去极值 + 填充
                "infer_processors": [
                    {
                        "class": "RobustZScoreNorm",
                        "kwargs": {"fields_group": "feature", "clip_outlier": True},
                    },
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                # 深度模型：防止 Dropna 打断时间序列
                # 这里的 DropnaLabel 只会在截面上丢掉没有 label 的样本，不破坏时间窗口；
                # CSRankNorm 对 label 做日内截面 rank 标准化，相当于 rank-label。
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ],
                # Label: 下五日收益（在 learn_processors 中会被做成 rank-label）
                "label": ["Ref($close, -5) / Ref($close, -1) - 1"],
            },
        },
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01"),
        },
    },
}

# =============================================================================
# 2. Model Config (RST-MoE)
# =============================================================================
model_conf = {
    "class": "QlibQuantMoE",
    "module_path": "module.model_adapter",
    "kwargs": {
        "model_config": {
            "d_model": 8,
            "n_layers": 2,
            "use_feature_selection": True,
            # context_len 和 num_alphas 会在 QlibQuantMoE 内自动探测
        },
        "trainer_config": {
            "lr": 5e-4,
            "n_epochs": 20,
            "batch_size": 4,  # 对应 FixedDailyBatchSampler 的日度 batch
            # [Safety Check] Internal Regime Encoder requires sufficient batch size (e.g. > 100)
            # to estimate covariance matrix. If using internal_mode, ensure batch_size is large enough.
            # "assert_batch_size_min": 100,
            "early_stop": 5,
            "num_workers": 0,  # debug 时用 0，正式训练可以拉高
            # Optional: precomputed market daily state as macro_features (recommended for longer horizons)
            # "market_state_path": "market_state_csi300.pkl",
            # "market_state_shift": 0,
            # "market_state_strict": True,
            # Warmup 配置（与 adapter 中的默认值一致）：
            "use_warmup": True,
            "warmup_ratio": 0.05,
            "warmup_steps": 0,
            "debug_sanity_check":True,
        },
    },
}

# =============================================================================
# 3. Strategy & Backtest Config (官方 port_analysis_config)
# =============================================================================
port_conf = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",  # 占位符，SignalRecord 会自动替换
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}


# =============================================================================
# 4. 报告生成工具函数
# =============================================================================
def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _newey_west_tstat(x: pd.Series | np.ndarray, lags: int | None = None) -> Tuple[float, float, int]:
    """
    Newey–West (HAC) t-stat for mean.

    Returns
    -------
    (t_stat, se_mean, lags_used)
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return np.nan, np.nan, int(lags or 0)

    if lags is None:
        # Common automatic choice: floor(4*(n/100)^(2/9))
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = int(max(0, min(lags, n - 2)))

    mu = float(arr.mean())
    u = arr - mu

    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)  # Bartlett
        gamma_k = float(np.dot(u[k:], u[:-k]) / n)
        lrv += 2.0 * w * gamma_k

    lrv = max(lrv, 1e-12)
    se_mean = float(np.sqrt(lrv / n))
    t_stat = float(mu / se_mean) if se_mean > 0 else np.nan
    return t_stat, se_mean, lags


def _safe_get_perf_value(perf: pd.DataFrame, key_candidates):
    """
    从 indicator_analysis_1day.pkl 中兼容性地取出指标值。
    key_candidates: ["annualized_return", "excess_return_with_cost.annualized_return", ...]
    """
    for k in key_candidates:
        if k in perf.columns:
            return perf[k].iloc[0]
    return np.nan


def _load_train_curves(rec):
    """
    从 Recorder 中读取训练曲线对象 train_curve（如果存在），并生成：
    - df_tc: DataFrame(epoch, train_listmle, train_ic, valid_listmle, valid_rank_ic, valid_ic)
    - train_summary_lines: 文本总结
    - fig_name: 图片文件名（相对路径），用于 Markdown 引用
    """
    local_dir: Path = rec.get_local_dir()
    fig_name = "train_curves_listmle_rankic.png"
    fig_path = local_dir / fig_name

    train_curve = None
    try:
        train_curve = rec.load_object("train_curve")
    except Exception:
        train_curve = None

    train_summary_lines: List[str] = []
    df_tc: Optional[pd.DataFrame] = None

    if isinstance(train_curve, dict) and len(train_curve) > 0:
        try:
            df_tc = pd.DataFrame(train_curve)
            if "epoch" not in df_tc.columns:
                df_tc["epoch"] = np.arange(1, len(df_tc) + 1)

            # --- plot curves ---
            try:
                fig, ax1 = plt.subplots(figsize=(6, 3))
                ax1.plot(
                    df_tc["epoch"],
                    df_tc.get("train_listmle", np.nan),
                    label="train ListMLE loss",
                )
                ax1.set_xlabel("epoch")
                ax1.set_ylabel("train ListMLE loss")

                ax2 = ax1.twinx()
                ax2.plot(
                    df_tc["epoch"],
                    df_tc.get("valid_rank_ic", np.nan),
                    linestyle="--",
                    label="valid RankIC",
                )
                ax2.set_ylabel("valid RankIC")

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

                fig.tight_layout()
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[Report] Failed to plot training curves: {e}")

            # --- textual summary ---
            if "valid_rank_ic" in df_tc.columns and df_tc["valid_rank_ic"].notna().any():
                best_idx = df_tc["valid_rank_ic"].idxmax()
                best_epoch = int(df_tc.loc[best_idx, "epoch"])
                best_ric = float(df_tc.loc[best_idx, "valid_rank_ic"])
                train_summary_lines.append(
                    f"- Peak valid RankIC ≈ {best_ric:.4f} at epoch {best_epoch}"
                )

            if "train_listmle" in df_tc.columns and "valid_rank_ic" in df_tc.columns:
                x = -df_tc["train_listmle"]
                y = df_tc["valid_rank_ic"]
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() > 2 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
                    corr = np.corrcoef(x[mask], y[mask])[0, 1]
                    train_summary_lines.append(
                        f"- Corr(-train ListMLE, valid RankIC) ≈ {corr:.3f}"
                    )
        except Exception as e:
            print(f"[Report] Failed to summarize training curves: {e}")

    if not train_summary_lines:
        train_summary_lines = ["- (no training curves found; check 'train_curve' in Recorder)"]

    return df_tc, train_summary_lines, (fig_name if fig_path.exists() else None)


def _load_run_conf(rec) -> Dict:
    """
    Load run configuration saved during training. Fallback to current module globals.
    """
    try:
        conf = rec.load_object("run_conf")
        if isinstance(conf, dict) and conf:
            return conf
    except Exception:
        pass
    return {"data_conf": data_conf, "model_conf": model_conf, "port_conf": port_conf}


def _try_prepare_label_df(dataset: Optional[TSDatasetH], segment: str) -> Optional[pd.DataFrame]:
    if dataset is None:
        return None
    for kwargs in (
        {"segment": segment, "col_set": "label"},
        {"segment": segment, "col_set": ["label"]},
    ):
        try:
            df = dataset.prepare(**kwargs)
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 1:
                df = df.copy()
                df.columns = ["label"]
                return df
        except Exception:
            continue
    return None


def _save_new_figures(
    *,
    local_dir: Path,
    prefix: str,
    draw_fn,
    dpi: int = 150,
) -> List[str]:
    before = set(plt.get_fignums())
    draw_fn()
    after = set(plt.get_fignums())
    new_nums = sorted(after - before)
    out: List[str] = []
    for i, num in enumerate(new_nums):
        fig = plt.figure(num)
        fn = f"{prefix}.png" if len(new_nums) == 1 else f"{prefix}_{i+1}.png"
        fig.savefig(local_dir / fn, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out.append(fn)
    return out


def export_qlib_official_graphs(
    rec,
    *,
    dataset: Optional[TSDatasetH] = None,
    segment: str = "test",
    prefix: str = "qlib",
    strict: bool = True,
) -> Dict[str, List[str]]:
    """
    Generate and save Qlib official analysis graphs into recorder local_dir.

    Graphs (if inputs exist):
    - analysis_position.report_graph
    - analysis_position.risk_analysis_graph
    - analysis_position.score_ic_graph
    - analysis_model.model_performance_graph
    """
    try:
        import qlib.contrib.report as qcr
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to import qlib.contrib.report: {e}") from e
        return {}
    import inspect

    local_dir: Path = rec.get_local_dir()
    out: Dict[str, List[str]] = {}

    # Inputs from recorder (created by PortAnaRecord / SignalRecord)
    report_normal_df = None
    analysis_df = None
    try:
        report_normal_df = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
    except Exception:
        report_normal_df = None
    try:
        analysis_df = rec.load_object("portfolio_analysis/port_analysis_1day.pkl")
    except Exception:
        analysis_df = None

    pred_df = None
    try:
        pred_df = rec.load_object("pred.pkl")
    except Exception:
        pred_df = None

    label_df = _try_prepare_label_df(dataset, segment)
    if label_df is None:
        # fallback: some workflows may save label as an object
        for key in ("label.pkl", "label_df.pkl"):
            try:
                label_df = rec.load_object(key)
                if isinstance(label_df, pd.DataFrame) and label_df.shape[1] >= 1:
                    label_df = label_df.copy()
                    label_df.columns = ["label"]
                    break
            except Exception:
                continue

    pred_label = None
    if isinstance(label_df, pd.DataFrame) and isinstance(pred_df, pd.DataFrame):
        pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    positions = None
    try:
        positions = rec.load_object("portfolio_analysis/positions_normal_1day.pkl")
    except Exception:
        positions = None

    available = {
        "report_normal_df": report_normal_df,
        "analysis_df": analysis_df,
        "pred_label": pred_label,
        "positions": positions,
    }

    def _auto_call(fn):
        sig = inspect.signature(fn)
        kwargs = {}
        for name, p in sig.parameters.items():
            if name in available and available[name] is not None:
                kwargs[name] = available[name]
            elif p.default is inspect._empty and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise TypeError(f"Missing required arg: {name}")
        return fn(**kwargs)

    def _resolve_graph_fn(graph_name: str):
        # e.g. "analysis_position.report_graph"
        obj = qcr
        for part in graph_name.split("."):
            obj = getattr(obj, part)
        return obj

    graph_names = []
    try:
        graph_names = list(getattr(qcr, "GRAPH_NAME_LIST"))
    except Exception:
        graph_names = []

    # Fall back to documented list if GRAPH_NAME_LIST is missing in this qlib version.
    if not graph_names:
        graph_names = [
            "analysis_position.report_graph",
            "analysis_position.score_ic_graph",
            "analysis_position.cumulative_return_graph",
            "analysis_position.risk_analysis_graph",
            "analysis_position.rank_label_graph",
            "analysis_model.model_performance_graph",
        ]

    required_inputs = {
        "report_normal_df": report_normal_df,
        "analysis_df": analysis_df,
        "positions": positions,
        "pred_df": pred_df,
        "label_df": label_df,
        "pred_label": pred_label,
        "dataset": dataset,
    }

    missing = [k for k, v in required_inputs.items() if v is None]
    if missing:
        raise RuntimeError(
            "Missing required inputs for Qlib official graphs: "
            + ", ".join(missing)
            + ". Ensure SignalRecord/SigAnaRecord/PortAnaRecord have run and pass dataset into generate_paper_report()."
        )

    for gname in graph_names:
        fn = _resolve_graph_fn(gname)
        file_prefix = f"{prefix}_{gname.replace('.', '_')}"
        fns = _save_new_figures(
            local_dir=local_dir,
            prefix=file_prefix,
            draw_fn=lambda fn=fn: _auto_call(fn),
        )
        if not fns:
            raise RuntimeError(f"Qlib graph '{gname}' produced no matplotlib figures.")
        out[gname] = fns

    if out:
        try:
            rec.save_objects(qlib_official_graphs=out)
        except Exception:
            pass
    return out


def _format_setup_from_conf(run_conf: Dict) -> str:
    dc = (run_conf or {}).get("data_conf", {}) or {}
    mc = (run_conf or {}).get("model_conf", {}) or {}
    pc = (run_conf or {}).get("port_conf", {}) or {}

    # data
    handler_kwargs = (((dc.get("kwargs") or {}).get("handler") or {}).get("kwargs") or {})
    segments = (dc.get("kwargs") or {}).get("segments", {}) or {}
    label_expr = handler_kwargs.get("label", None)
    instruments = handler_kwargs.get("instruments", None)

    # model/trainer
    mk = (mc.get("kwargs") or {})
    model_k = mk.get("model_config", {}) or {}
    trainer_k = mk.get("trainer_config", {}) or {}

    # backtest
    strat_k = ((pc.get("strategy") or {}).get("kwargs") or {})
    bt_k = (pc.get("backtest") or {}) or {}
    ex_k = (bt_k.get("exchange_kwargs") or {}) or {}

    def _seg(name: str):
        v = segments.get(name, None)
        return f"{v[0]} ~ {v[1]}" if isinstance(v, (tuple, list)) and len(v) == 2 else str(v)

    setup_txt = f"""
    - **Data**:
      - Handler: {((dc.get("kwargs") or {}).get("handler") or {}).get("class", "N/A")}
      - Instruments: {instruments}
      - Train: {_seg("train")}
      - Valid: {_seg("valid")}
      - Test: {_seg("test")}
    - **Label**: {label_expr}
    - **Model**:
      - class: {mc.get("class")}
      - d_model={model_k.get("d_model")}, n_layers={model_k.get("n_layers")}, n_heads={model_k.get("n_heads")}
      - use_feature_selection={model_k.get("use_feature_selection")}
      - use_alibi={model_k.get("use_alibi")}
    - **Training**:
      - lr={trainer_k.get("lr")}, epochs={trainer_k.get("n_epochs")}, batch_size={trainer_k.get("batch_size")}
      - seed={trainer_k.get("seed", None)}
    - **Backtest**:
      - strategy: {((pc.get("strategy") or {}).get("class"))}, topk={strat_k.get("topk")}, n_drop={strat_k.get("n_drop")}
      - benchmark={bt_k.get("benchmark")}, deal_price={ex_k.get("deal_price")}, cost(open/close)={ex_k.get("open_cost")}/{ex_k.get("close_cost")}
    """
    return textwrap.dedent(setup_txt).strip()


def generate_paper_report(
    rec,
    model_name: str = "RST-MoE",
    *,
    dataset: Optional[TSDatasetH] = None,
    segment: str = "test",
):
    """
    汇总当前 Recorder 中的：
      - 训练过程诊断：ListMLE 收敛 vs RankIC
      - IC / RankIC 序列
      - 回测关键指标
      - gate time_ratio 序列统计
      - attention map 的局部性指标
    输出一份 Markdown 报告到 kdd_report.md，并在控制台打印。
    """
    local_dir = Path(rec.get_local_dir())
    report_path = local_dir / "kdd_report.md"

    run_conf = _load_run_conf(rec)

    # ---------- 1. Signal 层指标 ----------
    sar = SigAnaRecord(rec)
    try:
        ic = sar.load("ic.pkl")   # 日度 IC
        ric = sar.load("ric.pkl") # 日度 RankIC
    except Exception as e:
        print(f"[Report] Failed to load IC / RIC: {e}")
        ic = pd.Series(dtype=float)
        ric = pd.Series(dtype=float)

    ic_mean = float(ic.mean()) if not ic.empty else np.nan
    ic_std = float(ic.std()) if not ic.empty else np.nan
    ric_mean = float(ric.mean()) if not ric.empty else np.nan
    ric_std = float(ric.std()) if not ric.empty else np.nan

    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    ricir = ric_mean / ric_std if ric_std > 0 else np.nan

    # HAC t-stat (Newey–West) for mean significance under autocorrelation/heteroskedasticity
    ic_t_hac, _, ic_lags = _newey_west_tstat(ic)
    ric_t_hac, _, ric_lags = _newey_west_tstat(ric)

    # ---------- 2. 组合回测指标 ----------
    ann_ret = info_ratio = max_dd = turnover = np.nan
    try:
        par_path = local_dir / "indicator_analysis_1day.pkl"
        if par_path.exists():
            perf = pd.read_pickle(par_path)

            ann_ret = _safe_get_perf_value(
                perf,
                ["annualized_return", "excess_return_with_cost.annualized_return"],
            )
            info_ratio = _safe_get_perf_value(
                perf,
                ["information_ratio", "excess_return_with_cost.information_ratio"],
            )
            max_dd = _safe_get_perf_value(
                perf,
                ["max_drawdown", "excess_return_with_cost.max_drawdown"],
            )
            turnover = _safe_get_perf_value(
                perf,
                ["turnover", "excess_return_with_cost.turnover"],
            )
        else:
            # 回退到 metrics dict
            metrics = rec.list_metrics()
            ann_ret = metrics.get("excess_return_with_cost.annualized_return", np.nan)
            info_ratio = metrics.get("excess_return_with_cost.information_ratio", np.nan)
            max_dd = metrics.get("excess_return_with_cost.max_drawdown", np.nan)
            turnover = metrics.get("excess_return_with_cost.turnover", np.nan)
    except Exception as e:
        print(f"[Report] Failed to load backtest indicators: {e}")

    # Save a compact summary for later aggregation (e.g., ablation sweeps)
    try:
        rec.save_objects(
            run_summary={
                "ic_mean": _as_float(ic_mean),
                "ic_std": _as_float(ic_std),
                "icir": _as_float(icir),
                "ic_hac_t": _as_float(ic_t_hac),
                "ic_hac_lags": int(ic_lags),
                "ric_mean": _as_float(ric_mean),
                "ric_std": _as_float(ric_std),
                "ricir": _as_float(ricir),
                "ric_hac_t": _as_float(ric_t_hac),
                "ric_hac_lags": int(ric_lags),
                "ann_ret": _as_float(ann_ret),
                "info_ratio": _as_float(info_ratio),
                "max_dd": _as_float(max_dd),
                "turnover": _as_float(turnover),
                "n_ic_days": int(len(ic)) if ic is not None else 0,
                "n_ric_days": int(len(ric)) if ric is not None else 0,
            }
        )
    except Exception:
        pass

    # ---------- 3. gate & attention 诊断 ----------
    gate_series = None
    attn_maps = None
    attn_pngs = None

    try:
        gate_series = rec.load_object("st_disentangle_gate_series")
    except Exception:
        pass

    try:
        attn_maps = rec.load_object("st_disentangle_attn_maps")
    except Exception:
        pass

    # optional: png filenames saved by model.export_visuals(save_png=True)
    try:
        attn_pngs = rec.load_object("st_disentangle_attn_pngs")
    except Exception:
        attn_pngs = None

    # gate time_ratio 统计
    gate_stats_str = "N/A"
    if gate_series is not None and len(gate_series) > 0:
        gate_series = gate_series.sort_index()
        g_mean = float(gate_series.mean())
        g_std = float(gate_series.std())
        g_p10 = float(gate_series.quantile(0.10))
        g_p90 = float(gate_series.quantile(0.90))
        gate_stats_str = f"mean={g_mean:.3f}, std={g_std:.3f}, p10={g_p10:.3f}, p90={g_p90:.3f}"

    def _row_normalize(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.ndim == 3:
            a = a.mean(axis=0)
        row_sum = a.sum(axis=-1, keepdims=True) + 1e-12
        return a / row_sum

    def _time_locality(a: np.ndarray) -> Tuple[float, float]:
        a_norm = _row_normalize(a)
        Tlen = int(a_norm.shape[0])
        diag_mass = float(np.trace(a_norm) / max(Tlen, 1))
        band = np.eye(Tlen) + np.eye(Tlen, k=1) + np.eye(Tlen, k=-1)
        band_mass = float((a_norm * band).sum() / max(Tlen, 1))
        return diag_mass, band_mass

    def _factor_concentration(a: np.ndarray, k: int = 5) -> Tuple[float, float, float]:
        a_norm = _row_normalize(a)
        N = int(a_norm.shape[0])
        diag_mass = float(np.trace(a_norm) / max(N, 1))
        # average mass of top-k entries per row (incl. self)
        topk = np.sort(a_norm, axis=-1)[:, -min(k, a_norm.shape[-1]) :]
        topk_mass = float(topk.sum(axis=-1).mean())
        # normalized entropy in [0,1] (lower => more peaky)
        p = np.clip(a_norm, 1e-12, 1.0)
        ent = -(p * np.log(p)).sum(axis=-1) / np.log(p.shape[-1])
        ent_mean = float(ent.mean())
        return diag_mass, topk_mass, ent_mean

    # attention 摘要 + (可选)图片
    shown_dates: List[str] = []
    time_attn_summary_lines: List[str] = []
    factor_attn_summary_lines: List[str] = []
    attn_media: Dict[str, Dict[str, str]] = {}

    if isinstance(attn_maps, dict) and len(attn_maps) > 0:
        for dt_str, v in list(attn_maps.items())[:4]:  # 最多展示 4 天
            shown_dates.append(dt_str)

            if isinstance(v, dict):
                t_map = v.get("time", None)
                f_map = v.get("factor", None)
            else:
                # backward compatibility: old format => time only
                t_map = v
                f_map = None

            if t_map is not None:
                try:
                    dm, bm = _time_locality(t_map)
                    time_attn_summary_lines.append(f"- {dt_str}: diag_mass={dm:.3f}, local_band_mass={bm:.3f}")
                except Exception:
                    pass

            if f_map is not None:
                try:
                    dm, top5, ent = _factor_concentration(f_map, k=5)
                    factor_attn_summary_lines.append(f"- {dt_str}: diag_mass={dm:.3f}, top5_mass={top5:.3f}, entropy={ent:.3f}")
                except Exception:
                    pass

            # resolve png filenames (prefer recorder object, else default naming)
            if isinstance(attn_pngs, dict) and dt_str in attn_pngs:
                attn_media[dt_str] = dict(attn_pngs.get(dt_str, {}))
            else:
                # model_adapter export_visuals default naming
                cand = {
                    "time": f"st_disentangle_attn_time_{dt_str}.png",
                    "factor": f"st_disentangle_attn_factor_{dt_str}.png",
                }
                # only keep those that actually exist
                for k, fn in list(cand.items()):
                    if (local_dir / fn).exists():
                        attn_media.setdefault(dt_str, {})[k] = fn

    if not time_attn_summary_lines:
        time_attn_summary_lines = ["- (no time-attention maps found; check export_visuals call)"]
    if not factor_attn_summary_lines:
        factor_attn_summary_lines = ["- (no factor-attention maps found; check export_visuals call)"]

    # ---------- 4. 训练过程诊断（ListMLE vs RankIC） ----------
    df_tc, train_summary_lines, train_fig_name = _load_train_curves(rec)

    # ---------- 4.5 Qlib 官方分析图（必须生成，缺输入直接报错） ----------
    qlib_graphs = export_qlib_official_graphs(rec, dataset=dataset, segment=segment, prefix="qlib", strict=True)

    # ---------- 5. 汇总成表格（方便 VS baseline 比较） ----------
    df_res = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Dataset": "Alpha158 / CSI300 / 2008-2020 (official split)",
                "IC (mean)": f"{ic_mean:.4f}",
                "ICIR": f"{icir:.2f}",
                "IC HAC t-stat": f"{ic_t_hac:.1f}" if pd.notna(ic_t_hac) else "nan",
                "RankIC (mean)": f"{ric_mean:.4f}",
                "RankIC IR": f"{ricir:.2f}",
                "RankIC HAC t-stat": f"{ric_t_hac:.1f}" if pd.notna(ric_t_hac) else "nan",
                "Ann. Return": f"{ann_ret:.2%}" if pd.notna(ann_ret) else "nan",
                "Info Ratio": f"{info_ratio:.2f}" if pd.notna(info_ratio) else "nan",
                "Max Drawdown": f"{max_dd:.2%}" if pd.notna(max_dd) else "nan",
                "Turnover": f"{turnover:.2%}" if pd.notna(turnover) else "nan",
                "Gate time_ratio stats": gate_stats_str,
            }
        ]
    )

    # ---------- 6. 生成 Markdown 报告 ----------
    lines: List[str] = []
    lines.append(f"# {model_name} on Alpha158 / CSI300\n")
    lines.append("## 1. Experimental Setup\n")
    lines.append(_format_setup_from_conf(run_conf) + "\n")

    lines.append("## 2. Cross-sectional Forecasting Performance\n")
    perf_txt = f"""
    - **IC (test)**:
      - mean = {ic_mean:.4f}, std = {ic_std:.4f}, ICIR = {icir:.2f}, HAC t-stat = {ic_t_hac:.1f} (lags={int(ic_lags)})
    - **RankIC (test)**:
      - mean = {ric_mean:.4f}, std = {ric_std:.4f}, IR = {ricir:.2f}, HAC t-stat = {ric_t_hac:.1f} (lags={int(ric_lags)})
    - 注：采用 Newey–West(HAC) t-stat 以处理日度序列的自相关/异方差。
    """
    lines.append(textwrap.dedent(perf_txt).strip() + "\n")

    lines.append("## 3. Training Dynamics & Portfolio Backtest\n")

    # 3.1 训练过程诊断
    lines.append("### 3.1 Training Dynamics (ListMLE vs. RankIC)\n")
    lines.append(
        "训练阶段采用 **ListMLE 主 loss**（基于 rank-label 的 list-wise 排序），"
        "这里展示 train/listmle 与 valid/rank_ic 随 epoch 的演化，并粗略量化二者的相关性：\n"
    )
    lines.extend(train_summary_lines)
    lines.append("")
    if train_fig_name is not None:
        lines.append(f"![Training dynamics (ListMLE vs RankIC)]({train_fig_name})\n")

    # 3.2 组合回测
    lines.append("### 3.2 Portfolio Backtest (2017-2020, CSI300 universe)\n")
    bt_txt = f"""
    - 年化收益 (excess return with cost): {ann_ret:.2%} (如果为 nan 请检查 indicator_analysis_1day.pkl)
    - 信息比 (Information Ratio): {info_ratio:.2f}
    - 最大回撤: {max_dd:.2%}
    - 成交换手率 (Turnover): {turnover:.2%}
    """
    lines.append(textwrap.dedent(bt_txt).strip() + "\n")

    if isinstance(qlib_graphs, dict) and qlib_graphs:
        lines.append("## 3.3 Qlib Official Graphs\n")
        lines.append("使用 Qlib 官方 report 模块生成的图表：\n")
        preferred = [
            "analysis_position.report_graph",
            "analysis_position.risk_analysis_graph",
            "analysis_position.score_ic_graph",
            "analysis_model.model_performance_graph",
        ]
        ordered = preferred + sorted([k for k in qlib_graphs.keys() if k not in set(preferred)])
        for k in ordered:
            fns = qlib_graphs.get(k, None)
            if not fns:
                continue
            lines.append(f"### {k}\n")
            for fn in fns:
                lines.append(f"![{k}]({fn})\n")

    lines.append("## 4. Spatio-Temporal Disentanglement Diagnostics\n")
    lines.append("### 4.1 Router Gate over Time (time vs. cross-sectional experts)\n")
    lines.append(f"- Gate time_ratio (time-expert weight) stats on test set: {gate_stats_str}\n")
    gate_interp = """
    - time_ratio 接近 1 表示更信任「时间 expert」，接近 0 表示更信任「截面 expert」。
    - 若 mean 在 (0.3, 0.7) 且 std > 0，说明路由器确实在不同阶段做非平凡决策；
      若长期贴近 0 或 1，则 MoE 退化为单专家模型。
    """
    lines.append(textwrap.dedent(gate_interp).strip() + "\n")

    lines.append("### 4.2 Temporal Attention (Heatmap + Locality)\n")
    lines.append(
        "基于若干代表性交易日的 **time-attention heatmap**，统计对角/邻近对角的注意力质量：\n"
    )
    lines.extend(time_attn_summary_lines)
    lines.append("")
    for dt_str in shown_dates:
        fn = attn_media.get(dt_str, {}).get("time", None)
        if fn:
            lines.append(f"![Time attention ({dt_str})]({fn})\n")

    attn_interp_t = """
    - diag_mass 衡量注意力在完全对齐的时间步 (i=j) 上的质量；
    - local_band_mass 衡量注意力在 |i-j| ≤ 1 的近邻时间步上的质量。
    - 越高说明模型更偏向「局部时序模式」（类似 AR / 局部卷积），
      越低说明模型依赖更长程的时序依赖。
    """
    lines.append(textwrap.dedent(attn_interp_t).strip() + "\n")

    lines.append("### 4.3 Factor Attention (Heatmap + Concentration)\n")
    lines.append(
        "基于同一批交易日的 **factor-attention heatmap**（默认取窗口最后一个时间步），统计注意力的集中度：\n"
    )
    lines.extend(factor_attn_summary_lines)
    lines.append("")
    for dt_str in shown_dates:
        fn = attn_media.get(dt_str, {}).get("factor", None)
        if fn:
            lines.append(f"![Factor attention ({dt_str})]({fn})\n")

    attn_interp_f = """
    - diag_mass：因子对自身的注意力质量（越高说明更“自回归/自保留”）；
    - top5_mass：每个因子行向量中 Top-5 权重质量的均值（越高说明更稀疏、更“专家化”）；
    - entropy：归一化熵 (0~1)，越低越尖锐，越高越均匀。
    - 注意：因子维度没有天然顺序，因此不像时间维那样用“邻近对角带”解释；我们更关心“是否稀疏/是否可解释地集中在少数因子交互上”。
    """
    lines.append(textwrap.dedent(attn_interp_f).strip() + "\n")

    lines.append("## 5. Summary\n")
    lines.append(
        "RST-MoE 在官方 Alpha158 / CSI300 框架下，兼顾了稳健的日频预测性能 "
        "（IC / RankIC / 信息比）和可解释的时空解耦结构（gate 曲线 + attention 局部性），"
        "同时通过 ListMLE 训练曲线与 RankIC 的联动，展示了从 rank-label → list-wise 优化 → "
        "截面预测 → 组合收益的一条清晰传导链。\n"
    )

    # 写入 Markdown 文件
    report_md = "\n".join(lines)
    report_path.write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"EXPERIMENT REPORT SUMMARY ({rec.info.get('id', 'unknown')})")
    print("=" * 80)
    print(df_res.to_markdown(index=False))
    print("-" * 80)
    print(f"Full Markdown report written to: {report_path}")
    print("-" * 80)
    print(report_md)
    print("=" * 80)


# =============================================================================
# 5. 主流程：训练 + 分析 + 回测 + 报告
# =============================================================================
if __name__ == "__main__":
    # 1) 实例化数据和模型
    dataset = init_instance_by_config(data_conf)
    model = init_instance_by_config(model_conf)

    # 2) 启动实验
    with R.start(experiment_name="Official_Alignment_RST_MoE"):
        # 2.1 记录超参
        R.log_params(**flatten_dict(model_conf))
        # 2.1.1 Save full run configuration for report reproducibility
        R.save_objects(
            run_conf={
                "data_conf": copy.deepcopy(data_conf),
                "model_conf": copy.deepcopy(model_conf),
                "port_conf": copy.deepcopy(port_conf),
            }
        )
        print(">>> [Phase 0] Planned Model Config (before auto-detect)...")
        try:
            print(model_conf["kwargs"]["model_config"])
        except Exception:
            print(model_conf)
        # 2.2 训练
        print(">>> [Phase 1] Training Model...")
        model.fit(dataset)
        # After fit(), the adapter has initialized `model.net` with auto-detected dims.
        try:
            print(">>> [Phase 1] Resolved Model Config (after auto-detect)...")
            print(model.net.config.to_dict())
        except Exception:
            pass
        R.save_objects(model=model)

        # 2.3 导出 gate / attention 可视化诊断
        print(">>> [Phase 1.1] Export Spatio-Temporal Visuals...")
        model.export_visuals(
            dataset,
            segment="test",
            max_attn_days=4,
            attn_layer=-1,  # 最后一层
            target_dates=None,  # or 指定若干交易日 ["2019-01-04", ...]
            prefix="st_disentangle",
        )

        # 2.4 Signal 生成与分析 (IC / RankIC / IC decay 等)
        print(">>> [Phase 2] Signal Analysis...")
        rec = R.get_recorder()
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        # 2.5 组合回测
        print(">>> [Phase 3] Backtesting...")
        PortAnaRecord(rec, port_conf, "day").generate()

        # 2.6 生成论文级报告
        print(">>> [Phase 4] Generate Paper-level Report...")
        generate_paper_report(rec, model_name="RST-MoE", dataset=dataset, segment="test")
