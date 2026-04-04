# -*- coding: utf-8 -*-
"""
RST-MoE + Qlib Official Workflow (Paper-Ready Version)

pip install plotly spicy statsmodels

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
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import textwrap
import copy
import pprint
import re

import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import TSDatasetH
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

import matplotlib.pyplot as plt

from module.utils.qlib_official_graphs import ensure_qlib_official_graphs
from module.utils.market_state import load_market_state_df, resolve_market_state_path
from module.utils import regime_analysis as ra


def _configure_stdio() -> None:
    """Force UTF-8 stdout/stderr to avoid garbled logs on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="strict")
        except Exception:
            pass


_configure_stdio()
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
        "step_len": 8,  # 时序窗口，对应模型 context_len
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2022-12-31",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2020-03-31",
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
            "train": ("2008-01-01", "2020-03-31"),
            "valid": ("2020-07-01", "2022-12-31"),
            "test": ("2020-04-01", "2020-06-30"),
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
            # ---- Architecture ----
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
            "dropout": 0.1,
            "initializer_range": 0.02,
            # NOTE: context_len / num_alphas will be overwritten by data-driven values in QlibQuantMoE._init_net
            "context_len": 8,
            "num_alphas": 158,
            # ---- Value embedding ----
            "value_embedding_type": "feature_tokenizer",  # shared_linear | feature_tokenizer
            "feature_tokenizer_bias": True,
            "feature_tokenizer_add_factor_id": False,
            "feature_tokenizer_init_std": 0.02,
            # ---- Regime-adaptive time embedding ----
            "use_regime_time_embedding": True,
            "time_tau_min": 0.5,
            "time_tau_max": 50.0,
            "time_tau_init": 5.0,
            "time_emb_init_std": 0.02,
            "time_decay_normalize": True,
            # ---- Regime-adaptive factor gate (FiLM) ----
            "use_regime_factor_gate": True,
            "factor_gate_scale": 1.0,
            "factor_gate_shift_scale": 0.2,
            # ---- MoE router ----
            "router_noise": 0.01,
            "router_temperature": 1.0,
            "router_z_loss_coef": 0.01,
            "router_use_layer_summary": True,
            # ---- Positional/feature selection ----
            "use_alibi": False,  # recommended default (time embedding already provides position signal)
            "use_feature_selection": False,
            "selection_reg_lambda": 1e-5,
            "selection_temperature": 0.1,
            "selection_noise_std": 0.5,
            # ---- Loss ----
            "main_loss": "mse",
            "loss_weights": {
                "listmle": 1.0,
                "mse": 1.0,
                "ic": 1.0,
                "rank": 0.0,
                "huber": 0.0,
            },
            "mse_normalize": True,
            "rank_topk": 5,
            "huber_delta": 1.0,
            "listmle_tau": 0.8,
            # ---- Macro / regime context ----
            "use_external_macro": True,
            "d_macro_input": 0,
            "regime_macro_dropout": 0.1,
            "regime_internal_mode": "long",
            "regime_internal_lag": 5,
            "regime_internal_use_batch_stats": False,
            "regime_internal_tail_threshold": 2.0,
            # ---- Pooling ----
            "pooling_alpha": 0.7,
            "pooling_mode": "full",  # static | adaptive_alpha | conditioned_query | full
            "pooling_alpha_scale": 0.3,
            "pooling_d_ff": 128,  # None means d_model; usually set to d_model for lightweight pooling
            "pooling_use_layer_summary": True,
        },
        "trainer_config": {
            "lr": 5e-5,
            "n_epochs": 40,
            "batch_size": 300,  # 对应 FixedDailyBatchSampler 的日度 batch
            "eval_batch_size": 300,
            # Mixed precision:
            # - "amp_fp16": recommended on RTX 4070S (fastest, needs GradScaler)
            # - "amp_bf16": more stable, usually no GradScaler (requires BF16 support)
            # - "fp32": baseline
            "precision": "amp_fp16",
            # Gradient accumulation across K (shuffled) daily microbatches (K dates per optimizer step)
            "grad_accum_steps": 1,
            # [Safety Check] Internal Regime Encoder requires sufficient batch size (e.g. > 100)
            # to estimate covariance matrix. If using internal_mode, ensure batch_size is large enough.
            # "assert_batch_size_min": 100,
            "seed": 15,
            # "early_stop": 5,
            "train_stop_key": "loss_main",
            "train_stop_threshold": 1.30,
            "min_epochs": 5,
            "consecutive_k": 2,
            "num_workers": 0,  # debug 时用 0，正式训练可以拉高
            # Optional: precomputed market daily state as macro_features (recommended for longer horizons)
            "market_state_path": "artifacts/market_state/market_state_csi300.pkl",
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
            "topk": 30,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2020-04-01",
        "end_time": "2020-06-30",
        "account": 100000000,
        # "benchmark": "SH000906",
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            # "limit_threshold": 0.095,
            "deal_price": "close",
            # "open_cost": 0.0005,
            # "close_cost": 0.0015,
            # "min_cost": 5,
        },
    },
}


# =============================================================================
# 3.1 Config helpers (resolved config, experiment name, MLflow description)
# =============================================================================
MODEL_CONFIG_KEYS_FULL = [
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "initializer_range",
    "num_alphas",
    "context_len",
    "value_embedding_type",
    "feature_tokenizer_bias",
    "feature_tokenizer_add_factor_id",
    "feature_tokenizer_init_std",
    "use_regime_time_embedding",
    "time_tau_min",
    "time_tau_max",
    "time_tau_init",
    "time_emb_init_std",
    "time_decay_normalize",
    "use_regime_factor_gate",
    "factor_gate_scale",
    "factor_gate_shift_scale",
    "router_noise",
    "router_temperature",
    "router_z_loss_coef",
    "router_use_layer_summary",
    "use_alibi",
    "use_feature_selection",
    "selection_reg_lambda",
    "selection_temperature",
    "selection_noise_std",
    "main_loss",
    "loss_weights",
    "mse_normalize",
    "rank_topk",
    "huber_delta",
    "listmle_tau",
    "use_external_macro",
    "d_macro_input",
    "regime_macro_dropout",
    "regime_internal_mode",
    "regime_internal_lag",
    "regime_internal_use_batch_stats",
    "regime_internal_tail_threshold",
    "pooling_alpha",
    "pooling_mode",
    "pooling_alpha_scale",
    "pooling_d_ff",
    "pooling_use_layer_summary",
]


def _pformat(obj: Any) -> str:
    try:
        return pprint.pformat(obj, width=120, sort_dicts=False)
    except TypeError:
        return pprint.pformat(obj, width=120)


def _resolve_model_config(model, fallback: Dict[str, Any]) -> Dict[str, Any]:
    cfg = None
    try:
        cfg = getattr(getattr(model, "net", None), "config", None)
    except Exception:
        cfg = None
    if cfg is None:
        return dict(fallback or {})

    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        cfg_dict = dict(getattr(cfg, "__dict__", {}) or {})

    resolved: Dict[str, Any] = {}
    for key in MODEL_CONFIG_KEYS_FULL:
        if key in cfg_dict:
            resolved[key] = cfg_dict[key]
        elif hasattr(cfg, key):
            resolved[key] = getattr(cfg, key)
        else:
            resolved[key] = (fallback or {}).get(key, None)
    return resolved


def _resolve_trainer_config(model, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if model is not None and hasattr(model, "trainer_config"):
        try:
            return dict(model.trainer_config)
        except Exception:
            pass
    return dict(fallback or {})


def _format_full_config_md(run_conf: Dict[str, Any]) -> str:
    dc = (run_conf or {}).get("data_conf", {}) or {}
    mc = (run_conf or {}).get("model_conf", {}) or {}
    pc = (run_conf or {}).get("port_conf", {}) or {}
    mk = (mc.get("kwargs") or {}) if isinstance(mc, dict) else {}
    model_k = mk.get("model_config", {}) or {}
    trainer_k = mk.get("trainer_config", {}) or {}
    lines = [
        "```python",
        "data_conf = " + _pformat(dc),
        "",
        "model_config = " + _pformat(model_k),
        "",
        "trainer_config = " + _pformat(trainer_k),
        "",
        "port_conf = " + _pformat(pc),
        "```",
    ]
    return "\n".join(lines)


def _bool01(v: Any) -> str:
    return "1" if bool(v) else "0"


def _slugify(value: Any) -> str:
    s = str(value) if value is not None else "none"
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s)
    s = s.strip("-")
    return s or "na"


def _build_experiment_name(model_k: Dict[str, Any], trainer_k: Dict[str, Any]) -> str:
    mk = dict(model_k or {})
    tk = dict(trainer_k or {})
    ms_path = tk.get("market_state_path", None)
    ms_name = Path(ms_path).stem if ms_path else "none"
    emb_type = str(mk.get("value_embedding_type", "shared_linear"))
    emb_tag = _slugify(emb_type)
    if emb_type == "feature_tokenizer":
        emb_tag = "ft"
        if bool(mk.get("feature_tokenizer_add_factor_id", False)):
            emb_tag = emb_tag + "id"
    parts = [
        "Official_Alignment_RST_MoE",
        f"loss-{_slugify(mk.get('main_loss', 'mse'))}",
        f"d{mk.get('d_model', 'na')}",
        f"l{mk.get('n_layers', 'na')}",
        f"ff{mk.get('d_ff', 'na')}",
        f"ctx{mk.get('context_len', 'na')}",
        f"emb{emb_tag}",
        f"time{_bool01(mk.get('use_regime_time_embedding', False))}",
        f"decay{_bool01(mk.get('time_decay_normalize', False))}",
        f"film{_bool01(mk.get('use_regime_factor_gate', False))}",
        f"fgs{_slugify(mk.get('factor_gate_scale', 'na'))}",
        f"fsh{_slugify(mk.get('factor_gate_shift_scale', 'na'))}",
        f"rSum{_bool01(mk.get('router_use_layer_summary', False))}",
        f"featSel{_bool01(mk.get('use_feature_selection', False))}",
        f"mseNorm{_bool01(mk.get('mse_normalize', False))}",
        f"macro{_bool01(mk.get('use_external_macro', False))}",
        f"mDrop{_slugify(mk.get('regime_macro_dropout', 'na'))}",
        f"pool{_slugify(mk.get('pooling_alpha', 'na'))}",
        f"ms{_slugify(ms_name)}",
    ]
    return "_".join(parts)


# =============================================================================
# 4. æŠ¥å‘Šç”Ÿæˆå·¥å…·å‡½æ•°
# =============================================================================
def _to_ts(d) -> pd.Timestamp:
    """
    Parse a date-like value into normalized pandas.Timestamp.

    Raises on invalid dates (e.g., '2019-02-31') to avoid silent misalignment.
    """
    return pd.to_datetime(str(d)).normalize()


def _fmt_date(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _get_segment_range(conf: Dict[str, Any], segment: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    segs = (((conf or {}).get("kwargs") or {}).get("segments") or {})
    seg = segs.get(segment, None)
    if not (isinstance(seg, (tuple, list)) and len(seg) == 2):
        return None
    s, e = seg
    s_ts = _to_ts(s)
    e_ts = _to_ts(e)
    if s_ts > e_ts:
        raise ValueError(f"Invalid segment range: {segment} start={s} end={e}")
    return s_ts, e_ts


def _infer_pred_date_range(pred_df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not isinstance(pred_df, pd.DataFrame) or pred_df.empty:
        return None
    idx = pred_df.index
    if isinstance(idx, pd.MultiIndex) and "datetime" in (idx.names or []):
        dts = pd.to_datetime(idx.get_level_values("datetime")).normalize()
    elif isinstance(idx, pd.DatetimeIndex):
        dts = pd.to_datetime(idx).normalize()
    else:
        return None
    if len(dts) <= 0:
        return None
    return pd.Timestamp(dts.min()).normalize(), pd.Timestamp(dts.max()).normalize()


def _clip_backtest_window(
    port_conf_in: Dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reason: str,
) -> Dict[str, Any]:
    """
    Clip `port_conf['backtest']` window into [start, end] (inclusive), in-place.
    """
    bt = (port_conf_in.get("backtest") or {})
    bt_start_raw = bt.get("start_time", None)
    bt_end_raw = bt.get("end_time", None)
    bt_start = _to_ts(bt_start_raw) if bt_start_raw is not None else start
    bt_end = _to_ts(bt_end_raw) if bt_end_raw is not None else end
    if bt_start > bt_end:
        raise ValueError(f"Invalid backtest range: start_time={bt_start_raw} end_time={bt_end_raw}")

    new_start = max(bt_start, start)
    new_end = min(bt_end, end)
    if new_start > new_end:
        raise RuntimeError(
            f"Backtest window has no overlap after clipping ({reason}). "
            f"backtest=[{_fmt_date(bt_start)}..{_fmt_date(bt_end)}], "
            f"clip=[{_fmt_date(start)}..{_fmt_date(end)}]"
        )

    if new_start != bt_start or new_end != bt_end:
        print(
            f">>> [Config] Clip backtest window ({reason}): "
            f"{_fmt_date(bt_start)}~{_fmt_date(bt_end)} -> {_fmt_date(new_start)}~{_fmt_date(new_end)}"
        )
        bt["start_time"] = _fmt_date(new_start)
        bt["end_time"] = _fmt_date(new_end)
        port_conf_in["backtest"] = bt
    return port_conf_in


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
                    f"- Peak valid RankIC approx {best_ric:.4f} at epoch {best_epoch}"
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
                        f"- Corr(-train {loss_label}, valid RankIC) approx {corr:.3f}"
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
        conf = rec.load_object("run_conf_resolved")
        if isinstance(conf, dict) and conf:
            return conf
    except Exception:
        pass
    try:
        conf = rec.load_object("run_conf")
        if isinstance(conf, dict) and conf:
            return conf
    except Exception:
        pass
    return {"data_conf": data_conf, "model_conf": model_conf, "port_conf": port_conf}


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

    # === Indicator Analysis Metrics (FFR / PA / POS) ===
    try:
        ind_ana = rec.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
        print("\n[Indicator Analysis Metrics]")
        if isinstance(ind_ana, pd.DataFrame) and not ind_ana.empty:
            print(ind_ana.to_string())
        else:
            print(f"  {ind_ana}")
    except Exception as e:
        print(f"  [WARN] Failed to load indicator_analysis: {e}")

    # === Generated Graphs ===
    print("\n[Generated Graphs]")
    graphs = {}
    errs = {}
    try:
        graphs, errs = ensure_qlib_official_graphs(rec, dataset=None, segment="test", prefix="qlib", strict=False)
    except Exception:
        graphs, errs = {}, {}
    try:
        local_dir = rec.get_local_dir()
    except Exception:
        local_dir = ""
    if graphs:
        for gname, fns in graphs.items():
            for fn in fns:
                print(f"  - {gname}: {local_dir}/{fn}")
    else:
        print("  (no graphs found)")

    # === Graph generation errors (best-effort mode) ===
    if isinstance(errs, dict) and errs:
        print("\n[Graph Generation Errors]")
        for k, v in list(errs.items())[:6]:
            print(f"  - {k}: {v}")

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
        - value_embedding_type={model_k.get("value_embedding_type", "shared_linear")}
        - feature_tokenizer_add_factor_id={model_k.get("feature_tokenizer_add_factor_id")}
        - feature_tokenizer_bias={model_k.get("feature_tokenizer_bias")}
        - feature_tokenizer_init_std={model_k.get("feature_tokenizer_init_std")}
        - use_feature_selection={model_k.get("use_feature_selection")}
        - use_alibi={model_k.get("use_alibi")}
    - **Training**:
      - lr={trainer_k.get("lr")}, epochs={trainer_k.get("n_epochs")}, batch_size={trainer_k.get("batch_size")}, eval_batch_size={trainer_k.get("eval_batch_size", trainer_k.get("batch_size"))}
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
    factor_topk = None
    factor_pool_topk = None
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
        factor_topk = rec.load_object("st_disentangle_factor_topk")
    except Exception:
        factor_topk = None

    try:
        factor_pool_topk = rec.load_object("st_disentangle_factor_pool_topk")
    except Exception:
        factor_pool_topk = None

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

    # Load market_state file (if configured) and compute per-regime performance/diagnostics.
    try:
        trainer_k = mk.get("trainer_config", {}) or {}
        ms_path = trainer_k.get("market_state_path", None)
        ms_file = (
            resolve_market_state_path(
                ms_path,
                search_dirs=[
                    local_dir,
                    Path(__file__).resolve().parent,
                ],
            )
            if ms_path
            else None
        )
        if ms_file is not None:
            ms_df = load_market_state_df(ms_file)

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
                    perf_df["ic"] = ra.normalize_dt_index(ic)
                if isinstance(ric, pd.Series) and len(ric) > 0:
                    perf_df["rank_ic"] = ra.normalize_dt_index(ric)

                # Optional model diagnostics to correlate with regimes
                if isinstance(gate_series, pd.Series) and len(gate_series) > 0:
                    perf_df["time_ratio"] = ra.normalize_dt_index(gate_series)
                for k, s in (diag_series or {}).items():
                    if isinstance(s, pd.Series) and len(s) > 0:
                        perf_df[k] = ra.normalize_dt_index(s)

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
                    fit_segs = trainer_k.get("regime_bucket_fit_segments", None)
                    if isinstance(fit_segs, str):
                        fit_segs = [s.strip() for s in fit_segs.split(",") if s.strip()]
                    if not isinstance(fit_segs, (list, tuple)) or not fit_segs:
                        fit_segs = ["train"]

                    fit_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
                    fit_range_meta: List[Dict[str, str]] = []
                    for seg in fit_segs:
                        rng = _get_segment_range(run_conf.get("data_conf", data_conf), str(seg))
                        if rng is not None:
                            fit_ranges.append(rng)
                            fit_range_meta.append(
                                {
                                    "segment": str(seg),
                                    "start": _fmt_date(rng[0]),
                                    "end": _fmt_date(rng[1]),
                                }
                            )

                    fit_mask = ra.build_fit_mask(ms_df.index, fit_ranges) if fit_ranges else np.ones(len(ms_df), dtype=bool)
                    fit_df = ms_df.loc[fit_mask, available_feats].copy()
                    fit_df = fit_df.replace([np.inf, -np.inf], np.nan)

                    regime_bucket_lines.append(
                        f"- market_state: `{Path(ms_file).name}`, shift={shift}, fit_segments={list(fit_segs)} "
                        f"(fit_days={int(fit_mask.sum())}, test_days={int(len(joined))})\n"
                    )

                    regime_thresholds: Dict[str, Any] = {
                        "market_state_path": str(ms_path),
                        "market_state_file": str(Path(ms_file).name),
                        "market_state_shift": int(shift),
                        "fit_segments": list(fit_segs),
                        "fit_ranges": fit_range_meta,
                        "fit_days": int(fit_mask.sum()),
                        "test_days": int(len(joined)),
                        "features": list(available_feats),
                        "medians": {},
                        "terciles": {},
                    }

                    # 1) Joint 2x2 regime by (PC1 ratio) x (Tail intensity), using median split.
                    if feat_pc1 in joined.columns and feat_tail in joined.columns:
                        pc1_fit = pd.to_numeric(fit_df.get(feat_pc1, pd.Series(dtype=float)), errors="coerce").dropna()
                        tail_fit = pd.to_numeric(fit_df.get(feat_tail, pd.Series(dtype=float)), errors="coerce").dropna()
                        pc1_med = float(pc1_fit.median()) if len(pc1_fit) > 0 else np.nan
                        tail_med = float(tail_fit.median()) if len(tail_fit) > 0 else np.nan
                        src = "fit"
                        if not np.isfinite(pc1_med) or not np.isfinite(tail_med):
                            src = "test_fallback"
                            pc1 = pd.to_numeric(joined[feat_pc1], errors="coerce").dropna()
                            tail = pd.to_numeric(joined[feat_tail], errors="coerce").dropna()
                            pc1_med = float(pc1.median()) if len(pc1) > 0 else np.nan
                            tail_med = float(tail.median()) if len(tail) > 0 else np.nan

                        regime_thresholds["medians"] = {
                            feat_pc1: {"value": float(pc1_med) if np.isfinite(pc1_med) else np.nan, "source": src},
                            feat_tail: {"value": float(tail_med) if np.isfinite(tail_med) else np.nan, "source": src},
                        }

                        g = joined.copy()
                        g["regime_2x2"] = ra.assign_regime_2x2(
                            g[feat_pc1],
                            g[feat_tail],
                            pc1_median=pc1_med,
                            tail_median=tail_med,
                        )
                        df_2x2 = ra.summarize_by_bucket(g, bucket_col="regime_2x2", bucket_name="Regime")
                        if not df_2x2.empty:
                            order = [
                                "High-PC1 / High-Tail",
                                "High-PC1 / Low-Tail",
                                "Low-PC1 / High-Tail",
                                "Low-PC1 / Low-Tail",
                            ]
                            df_2x2["_ord"] = df_2x2["Regime"].map(lambda x: order.index(x) if x in order else 99)
                            df_2x2 = df_2x2.sort_values(["_ord", "Regime"]).drop(columns=["_ord"]).reset_index(drop=True)
                            regime_bucket_lines.append(
                                f"- 2×2 分桶（median@{src}）：`{feat_pc1}`={pc1_med:.4g}, `{feat_tail}`={tail_med:.4g}\n"
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
                        edges = ra.fit_quantile_edges(fit_df.get(feat, pd.Series(dtype=float)))
                        src = "fit"
                        if not edges:
                            edges = ra.fit_quantile_edges(joined.get(feat, pd.Series(dtype=float)))
                            src = "test_fallback"
                        if not edges:
                            continue
                        regime_thresholds["terciles"][feat] = {"edges": list(edges), "source": src}
                        tmp = joined.copy()
                        tmp["bucket"] = ra.assign_quantile_bucket(tmp[feat], edges=edges)
                        df_u = ra.summarize_by_bucket(tmp, bucket_col="bucket", bucket_name="Bucket")
                        if df_u.empty:
                            continue
                        order = ["Low", "Mid", "High", "All"]
                        df_u["_ord"] = df_u["Bucket"].map(lambda x: order.index(x) if x in order else 99)
                        df_u = df_u.sort_values(["_ord", "Bucket"]).drop(columns=["_ord"]).reset_index(drop=True)
                        edges_str = ", ".join(f"{e:.4g}" for e in edges)
                        regime_bucket_lines.append(f"- Terciles by `{feat}` ({title}), edges@{src}=[{edges_str}]\n")
                        regime_bucket_lines.append(df_u.to_markdown(index=False) + "\n")
                        regime_bucket_saved[f"terciles__{feat}"] = df_u

                    if regime_thresholds.get("medians") or regime_thresholds.get("terciles"):
                        try:
                            rec.save_objects(regime_thresholds=regime_thresholds)
                        except Exception:
                            pass

                    # 3) Correlation table + scatter plots (test segment)
                    corr_feats = list(available_feats)
                    corr_metrics = [
                        c
                        for c in [
                            "rank_ic",
                            "ic",
                            "time_ratio",
                            "gate_entropy",
                            "time_tau",
                            "time_half_life",
                            "factor_gate_entropy",
                            "factor_gate_topk_mass_10",
                        ]
                        if c in joined.columns
                    ]
                    corr_df = ra.spearman_corr_table(joined, features=corr_feats, metrics=corr_metrics)
                    if not corr_df.empty:
                        corr_csv = local_dir / "regime_corr_spearman.csv"
                        try:
                            corr_df.to_csv(corr_csv, index=False)
                        except Exception:
                            corr_csv = None
                        regime_bucket_lines.append("- Spearman correlations (market_state vs metrics), top 12 by |ρ|:\n")
                        regime_bucket_lines.append(corr_df.head(12).to_markdown(index=False) + "\n")
                        if corr_csv is not None:
                            regime_bucket_lines.append(f"- Full table: `{corr_csv.name}`\n")
                        try:
                            rec.save_objects(regime_corr_spearman=corr_df)
                        except Exception:
                            pass

                    scatter_pngs: Dict[str, str] = {}
                    perf_metrics = [m for m in ["rank_ic", "ic"] if m in joined.columns]
                    router_metrics = [
                        m
                        for m in ["time_ratio", "gate_entropy", "time_tau", "time_half_life", "factor_gate_entropy"]
                        if m in joined.columns
                    ]
                    fn_perf = ra.save_scatter_grid(
                        joined,
                        features=corr_feats,
                        metrics=perf_metrics,
                        out_path=local_dir / "regime_scatter_perf.png",
                        title="Market state vs performance (test)",
                    )
                    if fn_perf:
                        scatter_pngs["perf"] = fn_perf
                        regime_bucket_lines.append(f"![Regime scatter (performance)]({fn_perf})\n")

                    fn_router = ra.save_scatter_grid(
                        joined,
                        features=corr_feats,
                        metrics=router_metrics,
                        out_path=local_dir / "regime_scatter_router.png",
                        title="Market state vs routing diagnostics (test)",
                    )
                    if fn_router:
                        scatter_pngs["router"] = fn_router
                        regime_bucket_lines.append(f"![Regime scatter (routing)]({fn_router})\n")
                    if scatter_pngs:
                        try:
                            rec.save_objects(regime_scatter_pngs=scatter_pngs)
                        except Exception:
                            pass

                    # 4) Interpretation & diagnostics (auto-generated, paper-friendly)
                    def _fmt(x, nd: int = 4) -> str:
                        try:
                            x = float(x)
                        except Exception:
                            return "N/A"
                        return f"{x:.{nd}f}" if np.isfinite(x) else "N/A"

                    def _rng(s: pd.Series, *, nd: int = 4) -> str:
                        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                        if len(s) == 0:
                            return "N/A"
                        return f"{_fmt(float(s.min()), nd)}~{_fmt(float(s.max()), nd)}"

                    def _best_worst(df: pd.DataFrame, *, metric: str, name_col: str) -> Optional[str]:
                        if not isinstance(df, pd.DataFrame) or df.empty or metric not in df.columns or name_col not in df.columns:
                            return None
                        v = pd.to_numeric(df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
                        if not v.notna().any():
                            return None
                        i_best = int(v.idxmax())
                        i_worst = int(v.idxmin())
                        best_name = str(df.loc[i_best, name_col])
                        worst_name = str(df.loc[i_worst, name_col])
                        best_v = float(v.loc[i_best])
                        worst_v = float(v.loc[i_worst])
                        return f"{metric}: best={best_name}({_fmt(best_v)}), worst={worst_name}({_fmt(worst_v)}), spread={_fmt(best_v - worst_v)}"

                    def _bucket_delta(df: pd.DataFrame, *, metric: str) -> Optional[str]:
                        if not isinstance(df, pd.DataFrame) or df.empty or "Bucket" not in df.columns or metric not in df.columns:
                            return None
                        v = df.set_index("Bucket")[metric]
                        v = pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan)
                        if "High" not in v.index or "Low" not in v.index:
                            return None
                        hi = v.loc["High"]
                        lo = v.loc["Low"]
                        if not (np.isfinite(hi) and np.isfinite(lo)):
                            return None
                        return f"Δ(High-Low)={_fmt(float(hi - lo))}"

                    # ---- metrics glossary ----
                    regime_bucket_lines.append("#### 2.X.1 指标释义（读表指南）\n")
                    regime_bucket_lines.append(
                        "- `market_state_corr_pc1_ratio`：市场“单一主导因子/同涨同跌”强度（越高越像单一市场因子驱动）。\n"
                        "- `market_state_tail_2sigma`：尾部/冲击强度（越高表示极端波动占比越高）。\n"
                        "- `market_state_corr_mean_abs`：拥挤度/相关性水平（越高越拥挤，alpha 更难独立发挥）。\n"
                        "- `market_vol_20`：基准 20 日波动（通常已标准化，解读为相对高/低波动）。\n"
                        "- `RankIC/IC`：预测排序/线性相关质量；`IR` 为日度均值/标准差（样本少时不稳定）。\n"
                        "- `time_ratio`：路由对 time-expert 的权重（高→更偏时序专家，低→更偏截面因子专家）。\n"
                        "- `gate_entropy`：路由不确定性（接近 0.693，约等于两专家均匀；越低越“果断”）。\n"
                        "- `time_tau/time_half_life`：时间记忆尺度（越大→更长记忆/更慢衰减）。\n"
                        "- `factor_gate_entropy/topk_mass`：因子重加权是否集中（topk_mass 高/entropy 低→更集中）。\n"
                    )
                    regime_bucket_lines.append("\n")

                    # ---- observed differences ----
                    regime_bucket_lines.append("#### 2.X.2 观测到的差异（test）\n")
                    if "rank_ic" in joined.columns:
                        regime_bucket_lines.append(f"- RankIC daily range: {_rng(joined['rank_ic'])}\n")
                    if "ic" in joined.columns:
                        regime_bucket_lines.append(f"- IC daily range: {_rng(joined['ic'])}\n")
                    if "time_ratio" in joined.columns:
                        regime_bucket_lines.append(f"- time_ratio daily range: {_rng(joined['time_ratio'])}\n")
                    if "gate_entropy" in joined.columns:
                        regime_bucket_lines.append(f"- gate_entropy daily range: {_rng(joined['gate_entropy'])}\n")
                    if "time_tau" in joined.columns:
                        regime_bucket_lines.append(f"- time_tau daily range: {_rng(joined['time_tau'])}\n")

                    df_2x2 = regime_bucket_saved.get("regime_2x2", None)
                    if isinstance(df_2x2, pd.DataFrame) and not df_2x2.empty:
                        bw = _best_worst(df_2x2, metric="RankIC_mean", name_col="Regime")
                        if bw:
                            regime_bucket_lines.append(f"- 2×2 regimes: {bw}\n")
                        bw_tr = _best_worst(df_2x2, metric="time_ratio_mean", name_col="Regime")
                        if bw_tr:
                            regime_bucket_lines.append(f"- 2×2 routing: {bw_tr}\n")

                    for feat in corr_feats:
                        key = f"terciles__{feat}"
                        df_u = regime_bucket_saved.get(key, None)
                        if not isinstance(df_u, pd.DataFrame) or df_u.empty:
                            continue
                        d_perf = _bucket_delta(df_u, metric="RankIC_mean")
                        d_route = _bucket_delta(df_u, metric="time_ratio_mean")
                        parts = []
                        if d_perf:
                            parts.append(f"RankIC_mean {d_perf}")
                        if d_route:
                            parts.append(f"time_ratio_mean {d_route}")
                        if parts:
                            regime_bucket_lines.append(f"- `{feat}` terciles: " + ", ".join(parts) + "\n")

                    # routing-performance coupling (within test)
                    coupling_pairs = [
                        ("time_ratio", "rank_ic"),
                        ("gate_entropy", "rank_ic"),
                        ("time_tau", "rank_ic"),
                        ("factor_gate_entropy", "rank_ic"),
                    ]
                    coupling_lines = []
                    for x, y in coupling_pairs:
                        if x in joined.columns and y in joined.columns:
                            rho, n = ra.spearman_rho(joined[x], joined[y])
                            if n >= 3 and np.isfinite(rho):
                                coupling_lines.append(f"{x}↔{y}: ρ={rho:.2f} (n={n})")
                    if coupling_lines:
                        regime_bucket_lines.append("- Routing ↔ quality coupling (Spearman, test): " + "; ".join(coupling_lines) + "\n")

                    # correlation-driven highlights
                    if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                        perf_rows = corr_df[corr_df["Metric"].isin(["rank_ic", "ic"])]
                        route_rows = corr_df[corr_df["Metric"].isin(["time_ratio", "gate_entropy", "time_tau", "factor_gate_entropy"])]
                        if not perf_rows.empty:
                            r0 = perf_rows.iloc[0]
                            regime_bucket_lines.append(
                                f"- Strongest state↔performance: `{r0['Feature']}` vs `{r0['Metric']}` "
                                f"ρ={_fmt(r0['SpearmanR'], 2)} (n={int(r0['N'])}).\n"
                            )
                        if not route_rows.empty:
                            r0 = route_rows.iloc[0]
                            regime_bucket_lines.append(
                                f"- Strongest state↔routing: `{r0['Feature']}` vs `{r0['Metric']}` "
                                f"ρ={_fmt(r0['SpearmanR'], 2)} (n={int(r0['N'])}).\n"
                            )

                    regime_bucket_lines.append(
                        "\n- 解释建议：若“预测质量差异”显著但 `time_ratio/gate_entropy/time_tau` 基本不变，说明路由/时间尺度未随宏观状态自适应；"
                        "反之若路由显著变化但 RankIC 不变，可能是“在动但没带来收益”。\n"
                        "- 注意：test 天数通常较少，分桶后的 `IR` 与相关性更偏诊断用途；建议在更长窗口/多次 run 上复核。 \n"
                    )

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
                    "pool": f"st_disentangle_attn_pool_factor_{dt_str}.png",
                }
                # only keep those that actually exist
                for k, fn in list(cand.items()):
                    if (local_dir / fn).exists():
                        attn_media.setdefault(dt_str, {})[k] = fn

    if not time_attn_summary_lines:
        time_attn_summary_lines = ["- (no time-attention maps found; check export_visuals call)"]
    if not factor_attn_summary_lines:
        factor_attn_summary_lines = ["- (no factor-attention maps found; check export_visuals call)"]

    factor_topk_lines: List[str] = []
    if isinstance(factor_topk, dict) and len(factor_topk) > 0:
        for dt_str in shown_dates:
            entry = factor_topk.get(dt_str, None)
            if not isinstance(entry, dict):
                continue
            ids = entry.get("ids", None)
            if ids is None:
                continue
            try:
                ids_int = [int(i) for i in list(ids)]
            except Exception:
                continue
            if ids_int:
                factor_topk_lines.append(f"- {dt_str}: top{len(ids_int)} ids={ids_int}")
    if not factor_topk_lines:
        factor_topk_lines = ["- (no factor-topk found; check export_visuals call)"]

    factor_pool_topk_lines: List[str] = []
    if isinstance(factor_pool_topk, dict) and len(factor_pool_topk) > 0:
        for dt_str in shown_dates:
            entry = factor_pool_topk.get(dt_str, None)
            if not isinstance(entry, dict):
                continue
            ids = entry.get("ids", None)
            if ids is None:
                continue
            try:
                ids_int = [int(i) for i in list(ids)]
            except Exception:
                continue
            if ids_int:
                factor_pool_topk_lines.append(f"- {dt_str}: top{len(ids_int)} ids={ids_int}")
    if not factor_pool_topk_lines:
        # Fallback: compute from attn_maps["factor_pool"] when old runs didn't save factor_pool_topk.
        try:
            if isinstance(attn_maps, dict) and len(attn_maps) > 0:
                for dt_str in shown_dates:
                    v = attn_maps.get(dt_str, None)
                    if not isinstance(v, dict):
                        continue
                    w = v.get("factor_pool", None)
                    if w is None:
                        continue
                    w = np.asarray(w, dtype=float).reshape(-1)
                    if w.ndim != 1 or w.size <= 0:
                        continue
                    k = min(10, int(w.size))
                    idx = np.argsort(w)[::-1][:k]
                    ids_int = [int(i) for i in idx]
                    if ids_int:
                        factor_pool_topk_lines.append(f"- {dt_str}: top{len(ids_int)} ids={ids_int}")
        except Exception:
            pass
    if not factor_pool_topk_lines:
        factor_pool_topk_lines = ["- (no factor-pooling topk found; check export_visuals call)"]

    # ---------- 4. 训练过程诊断（main_loss vs RankIC） ----------
    df_tc, train_summary_lines, train_fig_name = _load_train_curves(rec, main_loss=main_loss)

    # ---------- 4.5 Qlib 官方分析图（尽力生成：缺输入/空数据时跳过并记录原因） ----------
    try:
        qlib_graphs, qlib_graphs_errors = ensure_qlib_official_graphs(
            rec,
            dataset=dataset,
            segment=segment,
            prefix="qlib",
            strict=False,
        )
    except Exception:
        qlib_graphs, qlib_graphs_errors = {}, {}

    # ---------- 5. 汇总成表格（方便 VS baseline 比较） ----------
    df_res = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Dataset": (
                    f"Alpha158 / "
                    f"{(((run_conf or {}).get('data_conf') or {}).get('kwargs') or {}).get('handler', {}).get('kwargs', {}).get('instruments', 'N/A')} / "
                    f"{(((run_conf or {}).get('data_conf') or {}).get('kwargs') or {}).get('handler', {}).get('kwargs', {}).get('start_time', 'N/A')}"
                    f"~{(((run_conf or {}).get('data_conf') or {}).get('kwargs') or {}).get('handler', {}).get('kwargs', {}).get('end_time', 'N/A')}"
                ),
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
    lines.append("### 1.1 Full Config (Resolved)\n")
    lines.append(_format_full_config_md(run_conf) + "\n")


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
    bt = (((run_conf or {}).get("port_conf") or {}).get("backtest") or {})
    bt_start = bt.get("start_time", "N/A")
    bt_end = bt.get("end_time", "N/A")
    inst = (
        (((run_conf or {}).get("data_conf") or {}).get("kwargs") or {})
        .get("handler", {})
        .get("kwargs", {})
        .get("instruments", "N/A")
    )
    lines.append(f"### 3.2 Portfolio Backtest ({bt_start}~{bt_end}, {inst} universe)\n")
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
    lines.append("#### 4.3.1 Factor Attention Top-K (ids)\n")
    lines.extend(factor_topk_lines)
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

    lines.append("### 4.4 Factor Pooling Attention (Heatmap + Top-K)\n")
    lines.append(
        "来自 attention pooling 的 `factor_attention_weights`（模型输出 `factor_pool_weights`），按日对 batch 做均值后导出 heatmap 与 Top-K：\n"
    )
    lines.append("#### 4.4.1 Factor Pooling Top-K (ids)\n")
    lines.extend(factor_pool_topk_lines)
    lines.append("")
    for dt_str in shown_dates:
        fn = attn_media.get(dt_str, {}).get("pool", None)
        if fn:
            lines.append(f"![Factor pooling attention ({dt_str})]({fn})\n")

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
    def _safe_print(text: str) -> None:
        try:
            print(text)
            return
        except UnicodeEncodeError:
            pass

        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="strict")
            print(text)
            return
        except Exception:
            pass

        safe = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(safe)

    _safe_print(report_md)
    print("=" * 80)


# =============================================================================
# 5. 主流程：训练 + 分析 + 回测 + 报告
# =============================================================================
if __name__ == "__main__":
    # 1) 实例化数据和模型
    dataset = init_instance_by_config(data_conf)
    model = init_instance_by_config(model_conf)
    port_conf_run = copy.deepcopy(port_conf)

    # Align backtest window to test segment to avoid silent no-trade / NaN indicators
    seg_rng = _get_segment_range(data_conf, "test")
    if seg_rng is not None:
        seg_start, seg_end = seg_rng
        port_conf_run = _clip_backtest_window(
            port_conf_run,
            start=seg_start,
            end=seg_end,
            reason="dataset.test segment",
        )

    # 2) 启动实验
    exp_name = _build_experiment_name(
        (model_conf.get('kwargs') or {}).get('model_config', {}),
        (model_conf.get('kwargs') or {}).get('trainer_config', {}),
    )
    resolved_model_conf = None
    with R.start(experiment_name=exp_name):
        # 2.1 记录超参
        R.log_params(**flatten_dict(model_conf))
        # 2.1.1 Save full run configuration for report reproducibility
        R.save_objects(
            run_conf={
                "data_conf": copy.deepcopy(data_conf),
                "model_conf": copy.deepcopy(model_conf),
                "port_conf": copy.deepcopy(port_conf_run),
            }
        )
        model.log_config_summary(stage="planned")

        # 2.2 训练
        print(">>> [Phase 1] Training Model...")
        model.fit(dataset)

        resolved_model_k = _resolve_model_config(
            model, (model_conf.get('kwargs') or {}).get('model_config', {})
        )
        resolved_trainer_k = _resolve_trainer_config(
            model, (model_conf.get('kwargs') or {}).get('trainer_config', {})
        )
        resolved_model_conf = copy.deepcopy(model_conf)
        resolved_model_conf['kwargs']['model_config'] = copy.deepcopy(resolved_model_k)
        resolved_model_conf['kwargs']['trainer_config'] = copy.deepcopy(resolved_trainer_k)

        try:
            full_desc = (
                "[Model Full]\n" + _pformat(resolved_model_k) + "\n\n"
                "[Trainer Full]\n" + _pformat(resolved_trainer_k)
            )
            R.set_tags(**{"mlflow.note.content": full_desc})
        except Exception:
            pass

        R.save_objects(
            run_conf_resolved={
                'data_conf': copy.deepcopy(data_conf),
                'model_conf': copy.deepcopy(resolved_model_conf),
                'port_conf': copy.deepcopy(port_conf_run),
            }
        )
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

        # Clip backtest window to *actual* prediction availability (pred.pkl datetime range)
        try:
            pred_df = rec.load_object("pred.pkl")
            pred_rng = _infer_pred_date_range(pred_df)
            if pred_rng is not None:
                p_start, p_end = pred_rng
                port_conf_run = _clip_backtest_window(
                    port_conf_run,
                    start=p_start,
                    end=p_end,
                    reason="pred.pkl datetime range",
                )
                conf_for_save = resolved_model_conf or model_conf
                R.save_objects(
                    run_conf_resolved={
                        "data_conf": copy.deepcopy(data_conf),
                        "model_conf": copy.deepcopy(conf_for_save),
                        "port_conf": copy.deepcopy(port_conf_run),
                    }
                )
        except Exception as e:
            print(f">>> [WARN] Failed to align backtest window to pred.pkl: {e}")

        # 2.5 组合回测
        print(">>> [Phase 3] Backtesting...")
        PortAnaRecord(rec, port_conf_run, "day").generate()

        # 2.6 Qlib 官方分析图（使 print_metrics_summary 可见）
        print(">>> [Phase 3.1] Export Qlib Official Graphs...")
        ensure_qlib_official_graphs(rec, dataset=dataset, segment="test", prefix="qlib", strict=False)

        # 2.7 统一输出所有指标（包含 Graph 路径）
        print_metrics_summary(rec)

        # 2.8 生成论文级报告
        print(">>> [Phase 4] Generate Paper-level Report...")
        generate_paper_report(rec, model_name="RST-MoE", dataset=dataset, segment="test")

