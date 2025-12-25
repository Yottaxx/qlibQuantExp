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
   - 增加“训练过程诊断”：train/main_loss vs valid/rank_ic 曲线 + 文本总结
"""
from typing import Optional, List, Tuple, Dict, Any

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
        "step_len": 4,  # 时序窗口，对应模型 context_len
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2020-08-01",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2014-12-31",
                "instruments": "csi300",
                # 推理预处理（DK_I，用于特征预处理）：
                # - 特征：去极值 + 填充
                # 注意：DropnaLabel 不能放在 infer_processors 中（Qlib 限制）
                "infer_processors": [
                    {
                        "class": "RobustZScoreNorm",
                        "kwargs": {"fields_group": "feature", "clip_outlier": True},
                    },
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                # 训练预处理（train 使用 DK_L）：
                # - DropnaLabel: 移除 NaN 标签
                # - DropExtremeLabel: 移除截面 top/bottom 2.5% 极端值（处理涨跌停，对齐 MASTER）
                # - CSZScoreNorm: 截面 ZScore 标准化
                # 注意：valid/test 使用 DK_I，不会 drop extreme，评估在全部数据上进行
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label", "method": "robust"}},
                ],
                # Label: 下五日收益
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
            "d_model": 128,
            "n_layers": 2,
            "main_loss": "mse",
            "use_feature_selection": False,
            "use_alibi": False,  # recommended default (time embedding already provides position signal)
            "regime_macro_dropout": 0.1,
            # context_len 和 num_alphas 会在 QlibQuantMoE 内自动探测
        },
        "trainer_config": {
            "lr": 5e-4,
            "n_epochs": 20,
            "batch_size": 4,  # 对应 FixedDailyBatchSampler 的日度 batch
            # Gradient accumulation across K (shuffled) daily microbatches (K dates per optimizer step)
            "grad_accum_steps": 5,
            # [Safety Check] Internal Regime Encoder requires sufficient batch size (e.g. > 100)
            # to estimate covariance matrix. If using internal_mode, ensure batch_size is large enough.
            # "assert_batch_size_min": 100,
            "seed": 42,
            "early_stop": 5,
            "num_workers": 0,  # debug 时用 0，正式训练可以拉高
            # Optional: precomputed market daily state as macro_features (recommended for longer horizons)
            "market_state_path": "market_state_csi300.pkl",
            "market_state_shift": 0,
            "market_state_strict": True,
            # Warmup 配置（与 adapter 中的默认值一致）：
            "use_warmup": True,
            "warmup_ratio": 0.05,
            "warmup_steps": 0,
            "debug_sanity_check":True,
            # Valid 只允许 DK_I：
            "strict_valid_data_key":True
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


def _get_port_analysis_risk(analysis_df: pd.DataFrame, *, key: str, metric: str) -> float:
    """
    Read risk metrics from Qlib official PortAnaRecord output `portfolio_analysis/port_analysis_*.pkl`.

    Expected schema (Qlib 0.9.7):
    - index: MultiIndex[(key, metric)] where key in {excess_return_with_cost, excess_return_without_cost}
    - column: 'risk'
    """
    if not isinstance(analysis_df, pd.DataFrame) or analysis_df is None or analysis_df.empty:
        return np.nan
    try:
        v = analysis_df.loc[(key, metric), "risk"]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        return _as_float(v)
    except Exception:
        return np.nan


def _load_train_curves(rec, *, main_loss: Optional[str] = None):
    """
    从 Recorder 中读取训练曲线对象 train_curve（如果存在），并生成：
    - df_tc: DataFrame(epoch, train_main, train_listmle, train_mse, train_ic, valid_main, valid_rank_ic, valid_ic)
    - train_summary_lines: 文本总结
    - fig_name: 图片文件名（相对路径），用于 Markdown 引用
    """
    local_dir: Path = Path(rec.get_local_dir())
    if main_loss is not None:
        main_loss = str(main_loss).lower().strip()
        if main_loss == "mle":
            main_loss = "listmle"
    fig_suffix = main_loss or "main"
    fig_name = f"train_curves_{fig_suffix}_rankic.png"
    fig_path = local_dir / fig_name
    loss_label_map = {"listmle": "ListMLE", "mse": "MSE", "ic": "IC"}
    loss_label = loss_label_map.get(main_loss, "Main")

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
                y_key = None
                for cand in ("train_main", f"train_{main_loss}" if main_loss else None, "train_listmle"):
                    if cand and cand in df_tc.columns:
                        y_key = cand
                        break
                y_val = df_tc.get(y_key, np.nan) if y_key else np.nan

                ax1.plot(
                    df_tc["epoch"],
                    y_val,
                    label=f"train {loss_label} loss",
                )
                ax1.set_xlabel("epoch")
                ax1.set_ylabel(f"train {loss_label} loss")

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

            if "valid_rank_ic" in df_tc.columns:
                if "train_main" in df_tc.columns:
                    x = -df_tc["train_main"]
                elif main_loss and f"train_{main_loss}" in df_tc.columns:
                    x = -df_tc[f"train_{main_loss}"]
                else:
                    x = -df_tc.get("train_listmle", np.nan)
                y = df_tc["valid_rank_ic"]
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() > 2 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
                    corr = np.corrcoef(x[mask], y[mask])[0, 1]
                    train_summary_lines.append(
                        f"- Corr(-train {loss_label}, valid RankIC) ≈ {corr:.3f}"
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


def _load_label_df_from_recorder(rec) -> Optional[pd.DataFrame]:
    """
    Strictly load the raw label generated by Qlib official SignalRecord (DK_R) from recorder artifact `label.pkl`.
    """
    try:
        label_df = rec.load_object("label.pkl")
    except Exception:
        return None

    if isinstance(label_df, pd.Series):
        label_df = label_df.to_frame("label")

    if not isinstance(label_df, pd.DataFrame) or label_df.shape[0] == 0 or label_df.shape[1] < 1:
        return None

    df = label_df.iloc[:, [0]].copy()
    df.columns = ["label"]
    return df


def print_metrics_summary(rec) -> None:
    """
    Print unified metrics summary from SignalRecord, SigAnaRecord, and PortAnaRecord.
    Also prints paths to generated graphs.
    """
    print("\n" + "=" * 80)
    print(">>> [Metrics Summary] SignalRecord + PortAnaRecord")
    print("=" * 80)

    # === Signal Analysis Metrics ===
    print("\n[Signal Analysis Metrics]")
    loaded = False
    try:
        ic = rec.load_object("sig_analysis/ic.pkl")
        if isinstance(ic, pd.Series) and not ic.empty:
            m = float(ic.mean())
            sd = float(ic.std())
            ir = (m / sd) if sd > 0 else np.nan
            print(f"  IC mean={m:.6f}, std={sd:.6f}, IR={ir:.6f}")
            loaded = True
    except Exception:
        pass
    try:
        ric = rec.load_object("sig_analysis/ric.pkl")
        if isinstance(ric, pd.Series) and not ric.empty:
            m = float(ric.mean())
            sd = float(ric.std())
            ir = (m / sd) if sd > 0 else np.nan
            print(f"  RankIC mean={m:.6f}, std={sd:.6f}, IR={ir:.6f}")
            loaded = True
    except Exception:
        pass

    # Backward-compat: some legacy pipelines may save `ic.pkl`/`ric.pkl` at root or `sig_analysis.pkl`
    if not loaded:
        try:
            ic = rec.load_object("ic.pkl")
            if isinstance(ic, pd.Series) and not ic.empty:
                m = float(ic.mean())
                sd = float(ic.std())
                ir = (m / sd) if sd > 0 else np.nan
                print(f"  IC mean={m:.6f}, std={sd:.6f}, IR={ir:.6f}")
                loaded = True
        except Exception:
            pass
    if not loaded:
        try:
            ric = rec.load_object("ric.pkl")
            if isinstance(ric, pd.Series) and not ric.empty:
                m = float(ric.mean())
                sd = float(ric.std())
                ir = (m / sd) if sd > 0 else np.nan
                print(f"  RankIC mean={m:.6f}, std={sd:.6f}, IR={ir:.6f}")
                loaded = True
        except Exception:
            pass
    if not loaded:
        try:
            sig_ana = rec.load_object("sig_analysis.pkl")
            if hasattr(sig_ana, "items"):
                for k, v in sig_ana.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k:30s}: {v:.6f}")
                    elif hasattr(v, "mean"):
                        print(f"  {k:30s}: mean={float(v.mean()):.6f}, std={float(v.std()):.6f}")
                    else:
                        print(f"  {k:30s}: {v}")
            else:
                print(f"  {sig_ana}")
            loaded = True
        except Exception:
            pass

    if not loaded:
        print("  (no signal-analysis artifacts found; expected SigAnaRecord outputs under `sig_analysis/`)")

    # === Portfolio Analysis Metrics ===
    try:
        port_ana = rec.load_object("portfolio_analysis/port_analysis_1day.pkl")
        print("\n[Portfolio Analysis Metrics]")
        if isinstance(port_ana, pd.DataFrame) and not port_ana.empty:
            print(port_ana.to_string(max_rows=20))
        elif hasattr(port_ana, "items"):
            for k, v in port_ana.items():
                if isinstance(v, (int, float)):
                    print(f"  {k:30s}: {v:.6f}")
                else:
                    print(f"  {k:30s}: {v}")
        else:
            print(f"  {port_ana}")
    except Exception as e:
        print(f"  [WARN] Failed to load port_analysis: {e}")

    # === Generated Graphs ===
    print("\n[Generated Graphs]")
    try:
        graphs = rec.load_object("qlib_official_graphs")
        local_dir = rec.get_local_dir()
        if graphs:
            for gname, fns in graphs.items():
                for fn in fns:
                    print(f"  - {gname}: {local_dir}/{fn}")
        else:
            print("  (no graphs found)")
    except Exception:
        print("  (no graphs found)")

    # === Graph generation errors (best-effort mode) ===
    try:
        errs = rec.load_object("qlib_official_graphs_errors") or {}
        if isinstance(errs, dict) and errs:
            print("\n[Graph Generation Errors]")
            for k, v in list(errs.items())[:6]:
                print(f"  - {k}: {v}")
    except Exception:
        pass

    # === List all saved objects ===
    print("\n[Recorder Objects]")
    try:
        for obj_name in rec.list_objects():
            print(f"  - {obj_name}")
    except Exception:
        pass

    print("=" * 80 + "\n")


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


def _coerce_graph_output_to_figures(graph_output) -> List[object]:
    """
    Normalize qlib.contrib.report graph outputs into a list of figure-like objects.

    Qlib report graphs may return:
    - plotly / matplotlib figure objects
    - iterables / generators of figures
    - dicts mapping names -> figures
    - graph wrapper objects that expose `.figure` / `.fig`
    """

    def _is_figure_like(x) -> bool:
        return hasattr(x, "write_html") or hasattr(x, "savefig")

    def _unwrap(x):
        if x is None:
            return None
        if _is_figure_like(x):
            return x
        for attr in ("figure", "fig"):
            try:
                v = getattr(x, attr)
            except Exception:
                v = None
            if v is not None and _is_figure_like(v):
                return v
        return None

    def _collect(x) -> List[object]:
        if x is None:
            return []

        fig = _unwrap(x)
        if fig is not None:
            return [fig]

        if isinstance(x, dict):
            out: List[object] = []
            for v in x.values():
                out.extend(_collect(v))
            return out

        if isinstance(x, (list, tuple, set)):
            out: List[object] = []
            for v in x:
                out.extend(_collect(v))
            return out

        # Avoid iterating over strings/bytes (iterates characters)
        if isinstance(x, (str, bytes)):
            return []

        # Generator / iterable of unknown objects
        try:
            it = iter(x)
        except TypeError:
            return []

        out: List[object] = []
        for v in it:
            out.extend(_collect(v))
        return out

    return _collect(graph_output)


def _save_graph_figures(
    *,
    local_dir: Path,
    prefix: str,
    figures: List[object],
    dpi: int = 150,
) -> List[str]:
    """
    Save a list of figure objects into local_dir.

    Priority:
    - Plotly Figure: try `.png` via `write_image` (needs kaleido); fall back to `.html` (`include_plotlyjs="directory"`).
    - Matplotlib Figure: save `.png`.
    """
    figs = list(figures) if figures else []
    out: List[str] = []
    for i, fig in enumerate(figs):
        base = prefix if len(figs) == 1 else f"{prefix}_{i+1}"

        # Matplotlib
        if hasattr(fig, "savefig"):
            try:
                fn = f"{base}.png"
                fig.savefig(local_dir / fn, dpi=dpi, bbox_inches="tight")
                out.append(fn)
                continue
            except Exception:
                pass

        # Plotly
        if hasattr(fig, "write_html"):
            try:
                fn = f"{base}.png"
                fig.write_image(str(local_dir / fn))
                out.append(fn)
                continue
            except Exception:
                fn = f"{base}.html"
                fig.write_html(str(local_dir / fn), include_plotlyjs="directory", full_html=True)
                out.append(fn)
                continue

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
        # 显式导入子模块，否则 getattr(qcr, 'analysis_position') 会失败
        # Python namespace package 不会自动将子模块暴露为父模块属性
        import qlib.contrib.report.analysis_position  # noqa: F401
        import qlib.contrib.report.analysis_model  # noqa: F401
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to import qlib.contrib.report: {e}") from e
        return {}
    import inspect
    import warnings

    local_dir: Path = Path(rec.get_local_dir())
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    out: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}

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

    # Strict: only use raw label generated by Qlib official SignalRecord (DK_R) => label.pkl
    label_df = _load_label_df_from_recorder(rec)

    pred_label = None
    if isinstance(label_df, pd.DataFrame) and isinstance(pred_df, pd.DataFrame):
        pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)


    positions = None
    try:
        positions = rec.load_object("portfolio_analysis/positions_normal_1day.pkl")
    except Exception:
        positions = None

    # available 字典必须包含 qlib graph 函数参数名的所有变体
    # 根据 qlib 源码分析，各函数参数名如下：
    #   report_graph:            report_df
    #   score_ic_graph:          pred_label
    #   cumulative_return_graph: position, report_normal, label_data
    #   risk_analysis_graph:     analysis_df, report_normal_df (opt)
    #   rank_label_graph:        position, label_data
    #   model_performance_graph: pred_label
    available = {
        # 原始 key
        "report_normal_df": report_normal_df,
        "analysis_df": analysis_df,
        "pred_label": pred_label,
        "positions": positions,
        # === 别名映射 (qlib 函数实际参数名) ===
        "report_df": report_normal_df,      # report_graph 需要
        "position": positions,               # cumulative_return_graph, rank_label_graph 需要
        "report_normal": report_normal_df,   # cumulative_return_graph 需要
        "label_data": label_df,              # cumulative_return_graph, rank_label_graph 需要
    }

    def _describe_df(df: Any, name: str) -> str:
        if df is None:
            return f"{name}=None"
        if not isinstance(df, pd.DataFrame):
            return f"{name}={type(df).__name__}"
        msg = f"{name}.shape={df.shape}"
        try:
            msg += f", nan_frac={float(df.isna().mean().mean()):.3f}"
        except Exception:
            pass
        try:
            if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
                dts = pd.to_datetime(df.index.get_level_values("datetime"))
                if len(dts) > 0:
                    msg += f", dt=[{dts.min().date()}..{dts.max().date()}]"
        except Exception:
            pass
        return msg

    def _describe_positions(pos: Any) -> str:
        if pos is None:
            return "positions=None"
        if not isinstance(pos, dict):
            return f"positions={type(pos).__name__}"
        n_days = len(pos)
        if n_days == 0:
            return "positions=dict(days=0)"

        holding_counts: List[int] = []
        nonempty_days = 0
        for _, v in pos.items():
            try:
                if hasattr(v, "position"):
                    d = dict(v.position)
                elif isinstance(v, dict):
                    d = dict(v)
                else:
                    continue
                d.pop("cash", None)
                d.pop("now_account_value", None)
                cnt = len(d)
                holding_counts.append(cnt)
                if cnt > 0:
                    nonempty_days += 1
            except Exception:
                continue
        try:
            dt_min = min(pos.keys())
            dt_max = max(pos.keys())
        except Exception:
            dt_min, dt_max = None, None

        if holding_counts:
            avg = float(np.mean(holding_counts))
            mn = int(np.min(holding_counts))
            mx = int(np.max(holding_counts))
        else:
            avg, mn, mx = 0.0, 0, 0
        return (
            f"positions=dict(days={n_days}, nonempty_days={nonempty_days}, "
            f"holdings(avg/min/max)={avg:.1f}/{mn}/{mx}, dt=[{dt_min}..{dt_max}])"
        )

    def _date_overlap_hint() -> str:
        try:
            if not isinstance(positions, dict) or not isinstance(label_df, pd.DataFrame):
                return ""
            if label_df.empty or len(positions) == 0:
                return ""
            pos_dates = pd.to_datetime(list(positions.keys())).normalize()
            lbl_dates = pd.to_datetime(label_df.index.get_level_values("datetime")).normalize()
            overlap = len(set(pos_dates) & set(lbl_dates))
            return f"date_overlap(positions,label_df)={overlap}"
        except Exception:
            return ""

    def _auto_call(fn):
        sig = inspect.signature(fn)
        kwargs = {}
        for name, p in sig.parameters.items():
            # 跳过 **kwargs 类型参数
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if name == "show_notebook":
                kwargs[name] = False
                continue
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

    for gname in graph_names:
        fn = _resolve_graph_fn(gname)
        file_prefix = f"{prefix}_{gname.replace('.', '_')}"
        try:
            graph_output = _auto_call(fn)
        except Exception as e:
            msg = f"Failed to run Qlib graph '{gname}': {e}"
            if strict:
                raise RuntimeError(msg) from e
            errors[gname] = msg
            warnings.warn(msg)
            continue

        figures = _coerce_graph_output_to_figures(graph_output)
        if not figures:
            hints = [
                f"Qlib graph '{gname}' returned no figure objects.",
                _describe_positions(positions),
                _describe_df(report_normal_df, "report_normal_df"),
                _describe_df(analysis_df, "analysis_df"),
                _describe_df(label_df, "label_df"),
                _describe_df(pred_df, "pred_df"),
            ]
            ov = _date_overlap_hint()
            if ov:
                hints.append(ov)
            msg = "\n".join([h for h in hints if h])
            if strict:
                raise RuntimeError(msg)
            errors[gname] = msg
            warnings.warn(msg)
            continue

        try:
            fns = _save_graph_figures(local_dir=local_dir, prefix=file_prefix, figures=figures)
        except Exception as e:
            msg = f"Failed to save figures for Qlib graph '{gname}': {e}"
            if strict:
                raise RuntimeError(msg) from e
            errors[gname] = msg
            warnings.warn(msg)
            continue

        if not fns:
            msg = (
                f"Qlib graph '{gname}' produced figure objects but none could be saved. "
                f"figure_types={[type(f).__name__ for f in figures][:5]}"
            )
            if strict:
                raise RuntimeError(msg)
            errors[gname] = msg
            warnings.warn(msg)
            continue

        out[gname] = fns

    if out:
        try:
            rec.save_objects(qlib_official_graphs=out)
        except Exception:
            pass
    if errors:
        try:
            rec.save_objects(qlib_official_graphs_errors=errors)
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
      - main_loss={model_k.get("main_loss", "mse")}
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
      - 训练过程诊断：main_loss 收敛 vs RankIC
      - IC / RankIC 序列
      - 回测关键指标
      - gate time_ratio 序列统计
      - attention map 的局部性指标
    输出一份 Markdown 报告到 kdd_report.md，并在控制台打印。
    """
    local_dir = Path(rec.get_local_dir())
    report_path = local_dir / "kdd_report.md"

    run_conf = _load_run_conf(rec)
    mk = ((run_conf or {}).get("model_conf") or {}).get("kwargs") or {}
    model_k = mk.get("model_config", {}) or {}
    main_loss = str(model_k.get("main_loss", "mse")).lower()
    if main_loss == "mle":
        main_loss = "listmle"
    loss_label_map = {"listmle": "ListMLE", "mse": "MSE", "ic": "IC"}
    main_loss_label = loss_label_map.get(main_loss, "Main")

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
        analysis_df = rec.load_object("portfolio_analysis/port_analysis_1day.pkl")
        report_normal_df = rec.load_object("portfolio_analysis/report_normal_1day.pkl")

        ann_ret = _get_port_analysis_risk(
            analysis_df,
            key="excess_return_with_cost",
            metric="annualized_return",
        )
        info_ratio = _get_port_analysis_risk(
            analysis_df,
            key="excess_return_with_cost",
            metric="information_ratio",
        )
        max_dd = _get_port_analysis_risk(
            analysis_df,
            key="excess_return_with_cost",
            metric="max_drawdown",
        )

        if isinstance(report_normal_df, pd.DataFrame) and "turnover" in report_normal_df.columns:
            turnover = _as_float(pd.to_numeric(report_normal_df["turnover"], errors="coerce").mean())
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
    gate_png = None
    attn_maps = None
    attn_pngs = None
    diag_series: Dict[str, pd.Series] = {}
    diag_pngs: Dict[str, str] = {}
    tau_vs_time_ratio_png = None

    try:
        gate_series = rec.load_object("st_disentangle_gate_series")
    except Exception:
        pass

    try:
        gate_png = rec.load_object("st_disentangle_gate_png")
    except Exception:
        gate_png = None

    try:
        attn_maps = rec.load_object("st_disentangle_attn_maps")
    except Exception:
        pass

    try:
        tau_vs_time_ratio_png = rec.load_object("st_disentangle_tau_vs_time_ratio_png")
    except Exception:
        tau_vs_time_ratio_png = None

    # optional: png filenames saved by model.export_visuals(save_png=True)
    try:
        attn_pngs = rec.load_object("st_disentangle_attn_pngs")
    except Exception:
        attn_pngs = None

    # Optional: extra daily diagnostics exported by model.export_visuals()
    for k in [
        "gate_entropy",
        "time_tau",
        "time_half_life",
        "factor_gate_mean",
        "factor_gate_std",
        "factor_gate_entropy",
        "factor_gate_topk_mass_5",
        "factor_gate_topk_mass_10",
    ]:
        try:
            s = rec.load_object(f"st_disentangle_{k}_series")
            if s is not None:
                diag_series[k] = s
        except Exception:
            pass
        try:
            fn = rec.load_object(f"st_disentangle_{k}_png")
            if fn:
                diag_pngs[k] = fn
        except Exception:
            pass

    def _series_stats(s: pd.Series | None) -> str:
        if s is None or len(s) == 0:
            return "N/A"
        try:
            s = s.dropna().sort_index()
        except Exception:
            return "N/A"
        if len(s) == 0:
            return "N/A"
        mean = float(s.mean())
        std = float(s.std())
        p10 = float(s.quantile(0.10))
        p90 = float(s.quantile(0.90))
        return f"mean={mean:.3f}, std={std:.3f}, p10={p10:.3f}, p90={p90:.3f}"

    # gate time_ratio 统计
    gate_stats_str = _series_stats(gate_series)
    gate_entropy_stats_str = _series_stats(diag_series.get("gate_entropy", None))
    time_tau_stats_str = _series_stats(diag_series.get("time_tau", None))
    time_half_life_stats_str = _series_stats(diag_series.get("time_half_life", None))
    factor_gate_mean_stats_str = _series_stats(diag_series.get("factor_gate_mean", None))
    factor_gate_std_stats_str = _series_stats(diag_series.get("factor_gate_std", None))
    factor_gate_entropy_stats_str = _series_stats(diag_series.get("factor_gate_entropy", None))
    factor_gate_topk5_stats_str = _series_stats(diag_series.get("factor_gate_topk_mass_5", None))
    factor_gate_topk10_stats_str = _series_stats(diag_series.get("factor_gate_topk_mass_10", None))

    # ---------- 3.5 Regime bucket evaluation (market_state) ----------
    regime_bucket_lines: List[str] = []
    regime_bucket_saved: Dict[str, Any] = {}

    def _normalize_dt_index(s: pd.Series) -> pd.Series:
        s = s.copy()
        s.index = pd.to_datetime(s.index).normalize()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert(None)
        s = s[~s.index.duplicated(keep="last")]
        return s.sort_index()

    def _resolve_market_state_path(path_str: str) -> Optional[Path]:
        if not path_str:
            return None
        cand = []
        p0 = Path(str(path_str)).expanduser()
        cand.append(p0)
        cand.append(local_dir / str(path_str))
        try:
            cand.append(Path(__file__).resolve().parent / str(path_str))
        except Exception:
            pass
        for p in cand:
            try:
                if p.exists():
                    return p
            except Exception:
                continue
        return None

    def _bucket_tercile(s: pd.Series) -> Optional[pd.Series]:
        s = pd.to_numeric(s, errors="coerce")
        if s.dropna().nunique() < 2:
            return None
        try:
            codes = pd.qcut(s, q=3, labels=False, duplicates="drop")
        except Exception:
            return None
        if codes is None:
            return None
        # codes may have NaNs, keep them
        max_code = int(pd.to_numeric(codes, errors="coerce").max()) if codes.notna().any() else -1
        n_bins = max_code + 1
        if n_bins <= 0:
            return None
        if n_bins == 1:
            labels = ["All"]
        elif n_bins == 2:
            labels = ["Low", "High"]
        else:
            labels = ["Low", "Mid", "High"][:n_bins]

        def _map_code(v):
            if pd.isna(v):
                return np.nan
            i = int(v)
            if i < 0 or i >= len(labels):
                return np.nan
            return labels[i]

        return codes.map(_map_code)

    def _series_ir(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) < 2:
            return np.nan
        m = float(x.mean())
        sd = float(x.std())
        return (m / sd) if sd > 0 else np.nan

    # Load market_state file (if configured) and compute per-regime performance/diagnostics.
    try:
        trainer_k = mk.get("trainer_config", {}) or {}
        ms_path = trainer_k.get("market_state_path", None)
        ms_file = _resolve_market_state_path(str(ms_path)) if ms_path else None
        if ms_file is not None:
            # load & align index
            if ms_file.suffix in {".pkl", ".pickle"}:
                ms_df = pd.read_pickle(ms_file)
            elif ms_file.suffix in {".parquet"}:
                ms_df = pd.read_parquet(ms_file)
            elif ms_file.suffix in {".csv"}:
                ms_df = pd.read_csv(ms_file, index_col=0)
            else:
                raise ValueError(f"Unsupported market_state file type: {ms_file.suffix}")

            if not isinstance(ms_df, pd.DataFrame) or ms_df.empty:
                raise ValueError("market_state file must contain a non-empty DataFrame")

            ms_df = ms_df.copy()
            ms_df.index = pd.to_datetime(ms_df.index).normalize()
            if getattr(ms_df.index, "tz", None) is not None:
                ms_df.index = ms_df.index.tz_convert(None)
            ms_df.sort_index(inplace=True)

            shift = int(trainer_k.get("market_state_shift", 0) or 0)
            if shift:
                ms_df = ms_df.shift(shift)

            # Core interpretable regime dimensions (if present).
            feat_pc1 = "market_state_corr_pc1_ratio"
            feat_tail = "market_state_tail_2sigma"
            feat_corr = "market_state_corr_mean_abs"
            feat_vol = "market_vol_20"

            available_feats = [c for c in [feat_pc1, feat_tail, feat_corr, feat_vol] if c in ms_df.columns]
            if available_feats:
                # Performance series on the same dates
                perf_df = pd.DataFrame(index=pd.Index([], name="datetime"))
                if isinstance(ic, pd.Series) and len(ic) > 0:
                    perf_df["ic"] = _normalize_dt_index(ic)
                if isinstance(ric, pd.Series) and len(ric) > 0:
                    perf_df["rank_ic"] = _normalize_dt_index(ric)

                # Optional model diagnostics to correlate with regimes
                if isinstance(gate_series, pd.Series) and len(gate_series) > 0:
                    perf_df["time_ratio"] = _normalize_dt_index(gate_series)
                for k in ["gate_entropy", "time_tau", "time_half_life"]:
                    s = diag_series.get(k, None)
                    if isinstance(s, pd.Series) and len(s) > 0:
                        perf_df[k] = _normalize_dt_index(s)

                # Join and keep test dates only
                joined = perf_df.join(ms_df[available_feats], how="inner")
                joined = joined.replace([np.inf, -np.inf], np.nan)
                joined = joined.dropna(subset=[c for c in ["ic", "rank_ic"] if c in joined.columns], how="all")

                if not joined.empty and ("rank_ic" in joined.columns or "ic" in joined.columns):
                    regime_bucket_lines.append("### 2.X Regime Bucket Evaluation (test segment)\n")
                    regime_bucket_lines.append(
                        "基于 `market_state`（与训练时相同的宏观状态文件）对 test 交易日做分桶，"
                        "观察不同市场状态下的预测质量与路由行为差异。\n"
                    )

                    # 1) Joint 2x2 regime by (PC1 ratio) x (Tail intensity), using median split.
                    if feat_pc1 in joined.columns and feat_tail in joined.columns:
                        pc1 = pd.to_numeric(joined[feat_pc1], errors="coerce")
                        tail = pd.to_numeric(joined[feat_tail], errors="coerce")
                        pc1_med = float(pc1.dropna().median()) if pc1.notna().any() else np.nan
                        tail_med = float(tail.dropna().median()) if tail.notna().any() else np.nan

                        regime = pd.Series(index=joined.index, dtype=object)
                        pc1_high = pc1 >= pc1_med
                        tail_high = tail >= tail_med
                        regime.loc[pc1_high & tail_high] = "High-PC1 / High-Tail"
                        regime.loc[pc1_high & ~tail_high] = "High-PC1 / Low-Tail"
                        regime.loc[~pc1_high & tail_high] = "Low-PC1 / High-Tail"
                        regime.loc[~pc1_high & ~tail_high] = "Low-PC1 / Low-Tail"

                        g = joined.copy()
                        g["regime_2x2"] = regime
                        rows = []
                        for name, sub in g.groupby("regime_2x2"):
                            if name is None or (isinstance(name, float) and not np.isfinite(name)):
                                continue
                            row = {"Regime": str(name), "Days": int(len(sub))}
                            if "rank_ic" in sub.columns:
                                row["RankIC_mean"] = float(sub["rank_ic"].mean())
                                row["RankIC_IR"] = _series_ir(sub["rank_ic"])
                            if "ic" in sub.columns:
                                row["IC_mean"] = float(sub["ic"].mean())
                                row["IC_IR"] = _series_ir(sub["ic"])
                            if "time_ratio" in sub.columns:
                                row["time_ratio_mean"] = float(pd.to_numeric(sub["time_ratio"], errors="coerce").mean())
                            if "gate_entropy" in sub.columns:
                                row["gate_entropy_mean"] = float(pd.to_numeric(sub["gate_entropy"], errors="coerce").mean())
                            if "time_tau" in sub.columns:
                                row["time_tau_mean"] = float(pd.to_numeric(sub["time_tau"], errors="coerce").mean())
                            rows.append(row)

                        df_2x2 = pd.DataFrame(rows)
                        if not df_2x2.empty:
                            df_2x2 = df_2x2.sort_values(["Regime"]).reset_index(drop=True)
                            regime_bucket_lines.append(
                                f"- 2×2 分桶：`{feat_pc1}` median={pc1_med:.4g}, `{feat_tail}` median={tail_med:.4g}\n"
                            )
                            regime_bucket_lines.append(df_2x2.to_markdown(index=False) + "\n")
                            regime_bucket_saved["regime_2x2"] = df_2x2

                    # 2) Univariate terciles for a few key features.
                    univariate_feats = [
                        (feat_pc1, "Market-mode strength (PC1 ratio)"),
                        (feat_tail, "Tail intensity (|x|>2)"),
                        (feat_corr, "Crowding (mean abs corr)"),
                        (feat_vol, "Benchmark vol (20d)"),
                    ]
                    for feat, title in univariate_feats:
                        if feat not in joined.columns:
                            continue
                        b = _bucket_tercile(joined[feat])
                        if b is None:
                            continue
                        tmp = joined.copy()
                        tmp["bucket"] = b
                        rows = []
                        for name, sub in tmp.groupby("bucket"):
                            if name is None or (isinstance(name, float) and not np.isfinite(name)):
                                continue
                            row = {"Bucket": str(name), "Days": int(len(sub))}
                            if "rank_ic" in sub.columns:
                                row["RankIC_mean"] = float(sub["rank_ic"].mean())
                                row["RankIC_IR"] = _series_ir(sub["rank_ic"])
                            if "ic" in sub.columns:
                                row["IC_mean"] = float(sub["ic"].mean())
                                row["IC_IR"] = _series_ir(sub["ic"])
                            if "time_ratio" in sub.columns:
                                row["time_ratio_mean"] = float(pd.to_numeric(sub["time_ratio"], errors="coerce").mean())
                            rows.append(row)
                        df_u = pd.DataFrame(rows)
                        if df_u.empty:
                            continue
                        # Stable order: Low->Mid->High where possible
                        order = ["Low", "Mid", "High", "All"]
                        df_u["_ord"] = df_u["Bucket"].map(lambda x: order.index(x) if x in order else 99)
                        df_u = df_u.sort_values(["_ord", "Bucket"]).drop(columns=["_ord"]).reset_index(drop=True)
                        regime_bucket_lines.append(f"- Terciles by `{feat}` ({title})\n")
                        regime_bucket_lines.append(df_u.to_markdown(index=False) + "\n")
                        regime_bucket_saved[f"terciles__{feat}"] = df_u

                    # Save for later aggregation
                    if regime_bucket_saved:
                        try:
                            rec.save_objects(regime_bucket_eval=regime_bucket_saved)
                        except Exception:
                            pass
    except Exception as e:
        regime_bucket_lines = [
            "### 2.X Regime Bucket Evaluation (test segment)\n",
            f"- Failed to compute regime buckets: {e}\n",
        ]

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

    # ---------- 4. 训练过程诊断（main_loss vs RankIC） ----------
    df_tc, train_summary_lines, train_fig_name = _load_train_curves(rec, main_loss=main_loss)

    # ---------- 4.5 Qlib 官方分析图（尽力生成：缺输入/空数据时跳过并记录原因） ----------
    qlib_graphs = export_qlib_official_graphs(rec, dataset=dataset, segment=segment, prefix="qlib", strict=False)
    qlib_graphs_errors = {}
    try:
        qlib_graphs_errors = rec.load_object("qlib_official_graphs_errors") or {}
    except Exception:
        qlib_graphs_errors = {}

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

    if regime_bucket_lines:
        lines.extend(regime_bucket_lines)
        lines.append("")

    lines.append("## 3. Training Dynamics & Portfolio Backtest\n")

    # 3.1 训练过程诊断
    lines.append(f"### 3.1 Training Dynamics ({main_loss_label} vs. RankIC)\n")
    lines.append(
        f"训练阶段采用 **{main_loss_label} 主 loss**（基于 rank-label），"
        f"这里展示 train/{main_loss} 与 valid/rank_ic 随 epoch 的演化，并粗略量化二者的相关性：\n"
    )
    lines.extend(train_summary_lines)
    lines.append("")
    if train_fig_name is not None:
        lines.append(f"![Training dynamics ({main_loss_label} vs RankIC)]({train_fig_name})\n")

    # 3.2 组合回测
    lines.append("### 3.2 Portfolio Backtest (2017-2020, CSI300 universe)\n")
    bt_txt = f"""
    - 年化收益 (excess return with cost): {ann_ret:.2%} (如果为 nan 请检查 portfolio_analysis/port_analysis_1day.pkl)
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
                if str(fn).lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                    lines.append(f"![{k}]({fn})\n")
                else:
                    lines.append(f"- [{k}]({fn})\n")
    elif qlib_graphs_errors:
        lines.append("## 3.3 Qlib Official Graphs\n")
        lines.append("Qlib 官方图表未生成（或部分缺失），常见原因：positions/label/pred 对齐后为空（全 NaN）。\n")
        lines.append("```text\n")
        # 保持可读性：最多列出 6 条
        for i, (k, v) in enumerate(list(qlib_graphs_errors.items())[:6]):
            lines.append(f"[{k}] {v}\n")
        lines.append("```\n")

    lines.append("## 4. Spatio-Temporal Disentanglement Diagnostics\n")
    lines.append("### 4.1 Router Gate over Time (time vs. cross-sectional experts)\n")
    lines.append(f"- Gate time_ratio (time-expert weight) stats on test set: {gate_stats_str}\n")
    lines.append(f"- Gate entropy stats on test set: {gate_entropy_stats_str}\n")
    gate_interp = """
    - time_ratio 接近 1 表示更信任「时间 expert」，接近 0 表示更信任「截面 expert」。
    - 若 mean 在 (0.3, 0.7) 且 std > 0，说明路由器确实在不同阶段做非平凡决策；
      若长期贴近 0 或 1，则 MoE 退化为单专家模型。
    """
    lines.append(textwrap.dedent(gate_interp).strip() + "\n")

    if gate_png and (local_dir / str(gate_png)).exists():
        lines.append(f"![Gate time_ratio series]({gate_png})\n")
    fn = diag_pngs.get("gate_entropy", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![Gate entropy series]({fn})\n")

    lines.append("### 4.1.1 Regime-Adaptive Time Scale (tau / half-life)\n")
    lines.append(f"- time_tau stats on test set: {time_tau_stats_str}\n")
    lines.append(f"- time_half_life stats on test set: {time_half_life_stats_str}\n")
    fn = diag_pngs.get("time_tau", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![time_tau series]({fn})\n")
    fn = diag_pngs.get("time_half_life", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![time_half_life series]({fn})\n")
    time_interp = """
    - `time_tau` 来自 regime-adaptive time embedding 的时间尺度参数（越大越偏长记忆，越小越偏短记忆）。
    - `time_half_life = time_tau * ln(2)`（单位为窗口时间步，若 1 步=1 天则可视作“天数半衰期”）。
    """
    lines.append(textwrap.dedent(time_interp).strip() + "\n")

    if tau_vs_time_ratio_png and (local_dir / str(tau_vs_time_ratio_png)).exists():
        lines.append("### 4.1.1.1 tau vs time_ratio (same-day overlay)\n")
        lines.append(f"![tau vs time_ratio]({tau_vs_time_ratio_png})\n")

    lines.append("### 4.1.2 Regime-Adaptive Factor Gate (concentration)\n")
    lines.append(f"- factor_gate_mean stats on test set: {factor_gate_mean_stats_str}\n")
    lines.append(f"- factor_gate_std stats on test set: {factor_gate_std_stats_str}\n")
    lines.append(f"- factor_gate_entropy stats on test set: {factor_gate_entropy_stats_str}\n")
    lines.append(f"- factor_gate_topk_mass_5 stats on test set: {factor_gate_topk5_stats_str}\n")
    lines.append(f"- factor_gate_topk_mass_10 stats on test set: {factor_gate_topk10_stats_str}\n")
    fn = diag_pngs.get("factor_gate_mean", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![factor_gate_mean series]({fn})\n")
    fn = diag_pngs.get("factor_gate_std", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![factor_gate_std series]({fn})\n")
    fn = diag_pngs.get("factor_gate_entropy", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![factor_gate_entropy series]({fn})\n")
    fn = diag_pngs.get("factor_gate_topk_mass_5", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![factor_gate_topk_mass_5 series]({fn})\n")
    fn = diag_pngs.get("factor_gate_topk_mass_10", None)
    if fn and (local_dir / fn).exists():
        lines.append(f"![factor_gate_topk_mass_10 series]({fn})\n")
    factor_gate_interp = """
    - `factor_gate_mean` / `factor_gate_std`：regime-adaptive factor gate 权重的日均值/标准差，反映因子重加权的整体幅度和波动。
    - `factor_gate_entropy` 是把 per-sample 的 factor gate 归一化后得到的分布熵（再除以 log(N) 做归一化到 0~1）。
      越低表示 gate 越“集中”，即 regime 对因子组合的重加权更强、更具结构性。
    - `factor_gate_topk_mass_5` / `factor_gate_topk_mass_10` 表示前 5 / 10 个因子（按 gate 权重排序）的累计质量占比；越高说明越稀疏/越集中。
    """
    lines.append(textwrap.dedent(factor_gate_interp).strip() + "\n")

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
        f"同时通过 {main_loss_label} 训练曲线与 RankIC 的联动，展示了从 rank-label → "
        f"{main_loss_label} 优化 → 截面预测 → 组合收益的一条清晰传导链。\n"
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
        if hasattr(model, "log_config_summary"):
            model.log_config_summary(stage="planned")
        else:
            print(">>> [Phase 0] Planned Model Config (before auto-detect)...")
            try:
                print(model_conf["kwargs"]["model_config"])
            except Exception:
                print(model_conf)
        # 2.2 训练
        print(">>> [Phase 1] Training Model...")
        model.fit(dataset)
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
            factor_use_last_time = True,
        )

        # 2.4 Signal 生成与分析 (IC / RankIC / IC decay 等)
        print(">>> [Phase 2] Signal Analysis...")
        rec = R.get_recorder()
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        # 2.5 组合回测
        print(">>> [Phase 3] Backtesting...")
        PortAnaRecord(rec, port_conf, "day").generate()

        # 2.6 统一输出所有指标（包含 Graph 路径）
        print_metrics_summary(rec)

        # 2.7 生成论文级报告
        print(">>> [Phase 4] Generate Paper-level Report...")
        generate_paper_report(rec, model_name="RST-MoE", dataset=dataset, segment="test")
