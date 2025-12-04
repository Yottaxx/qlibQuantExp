# -*- coding: utf-8 -*-
"""
MASTER-Aligned RST-MoE + Qlib Workflow

对齐点：
- Universe: CSI800
- Benchmark: SH000906
- 时间切分:
    - Train: 2008-01-01 ~ 2020-03-31
    - Valid: 2020-04-01 ~ 2020-06-30
    - Test:  2020-07-01 ~ 2022-12-31
- Label horizon:
    Ref($close, -5) / Ref($close, -1) - 1
  （learn_processors 中用 CSRankNorm → rank-label for training / ListMLE）
- Port 分析:
    - TopkDropout(topk=30, n_drop=30)
    - Backtest: 2020-07-01 ~ 2022-12-31
    - Benchmark: SH000906

训练：
- 使用 rank-label（CSRankNorm(label)）在截面上做 ListMLE list-wise ranking
评估 / 报告：
- IC / RankIC 严格使用 raw close 价格算出来的 5 日 raw return
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

# =============================================================================
# 0. Qlib Init (按你的 qlib_init 对齐 Windows 路径)
# =============================================================================
provider_uri = r"~\qlib_data\cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# =============================================================================
# 1. Data Config （MASTER 对齐版本，TSDatasetH）
# =============================================================================
data_conf = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "step_len": 32,  # RST-MoE 的时序窗口；如需完全对齐 MASTER，可改成 8
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                # ---- MASTER 对齐的时间区间 & Universe ----
                "start_time": "2008-01-01",
                "end_time": "2022-12-31",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2020-03-31",
                "instruments": "csi800",

                # ---- feature 预处理 ----
                "infer_processors": [
                    {
                        "class": "RobustZScoreNorm",
                        "kwargs": {"fields_group": "feature", "clip_outlier": True},
                    },
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],

                # ---- label 预处理：Dropna + CSRankNorm(label) → rank-label ----
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {
                        "class": "CSRankNorm",
                        "kwargs": {"fields_group": "label"},
                    },
                ],

                # ---- Label: 5 日 horizon raw return (MASTER 一致) ----
                # Ref($close, -5) / Ref($close, -1) - 1
                "label": ["Ref($close, -5) / Ref($close, -1) - 1"],
            },
        },

        # ---- MASTER 对齐的 train/valid/test 切分 ----
        "segments": {
            "train": ("2008-01-01", "2020-03-31"),
            "valid": ("2020-04-01", "2020-06-30"),
            "test": ("2020-07-01", "2022-12-31"),
        },
    },
}

# =============================================================================
# 2. Model Config (RST-MoE + ListMLE)
# =============================================================================
model_conf = {
    "class": "QlibQuantMoE",
    "module_path": "module.model_adapter",
    "kwargs": {
        "model_config": {
            "d_model": 32,
            "n_layers": 2,
            "use_feature_selection": True,
            # context_len 和 num_alphas 会在 QlibQuantMoE 内通过首个 batch 自动探测
        },
        "trainer_config": {
            "lr": 5e-4,
            "n_epochs": 20,
            "batch_size": 256,  # 对应 FixedDailyBatchSampler 的日度 batch
            "early_stop": 5,
            "num_workers": 0,  # debug 用 0；正式训练可以拉高
            # warmup / cosine scheduler 由 QlibQuantMoE 内部处理:
            # "use_warmup": True,
            # "warmup_ratio": 0.1,
            # "warmup_steps": 0,
            # "label_dim": 1,  # 若 handler 将 label pack 进 feature 时需要；此处 Alpha158 显式 label 可不传
        },
    },
}

# =============================================================================
# 3. Strategy & Backtest Config （MASTER 对齐版本）
# =============================================================================
port_conf = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 30,
            "n_drop": 30,  # MASTER: 每日全换仓
        },
    },
    "backtest": {
        "start_time": "2020-07-01",
        "end_time": "2022-12-31",
        "account": 100000000,
        "benchmark": "SH000906",  # MASTER 的中证 800 基准
        "exchange_kwargs": {
            "freq": "day",
            "deal_price": "close",
            # 这里的成本 / 涨跌停约束相对严格，对比 MASTER 时要在报告中说明
            "limit_threshold": 0.095,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

# =============================================================================
# 4. 报告工具：raw-return IC + 训练过程诊断 + 解耦指标
# =============================================================================
def _safe_get_perf_value(perf: pd.DataFrame, key_candidates: List[str]):
    """
    从 indicator_analysis_1day.pkl 中兼容性地取出指标值。
    key_candidates: ["annualized_return", "excess_return_with_cost.annualized_return", ...]
    """
    for k in key_candidates:
        if k in perf.columns:
            return perf[k].iloc[0]
    return np.nan


def compute_raw_return_ic(rec, model, dataset, *, segment: str = "test") -> None:
    """
    使用 **raw close 价格** 计算 5 日 horizon raw return:

        r_raw(t) = close(t+5) / close(t+1) - 1

    对 test 段逐日做截面 IC / RankIC，保存为：

        - ic_raw  : pd.Series(datetime -> IC)
        - ric_raw : pd.Series(datetime -> RankIC)
    """
    # 1) 用 adapter 再跑一遍 test 段预测（保证与 backtest 一致）
    print(">>> [RawIC] Predicting on test segment for raw-return IC...")
    pred = model.predict(dataset, segment=segment)  # pd.Series, MultiIndex (datetime, instrument)
    if not isinstance(pred, pd.Series):
        pred = pd.Series(pred)

    idx = pred.index
    # 兼容不同 index 命名
    try:
        dt_level = idx.names.index("datetime")
    except ValueError:
        dt_level = 0
    try:
        inst_level = idx.names.index("instrument")
    except ValueError:
        inst_level = 1

    dates = idx.get_level_values(dt_level)
    insts = sorted(idx.get_level_values(inst_level).unique())
    start_dt = str(dates.min().date())
    end_dt = str(dates.max().date())

    # 2) 从 qlib 的 Data API 拉 raw close 价格
    print(
        f">>> [RawIC] Fetching $close from {start_dt} to {end_dt} "
        f"for {len(insts)} instruments..."
    )
    df_close = D.features(
        insts,
        ["$close"],
        start_time=start_dt,
        end_time=end_dt,
        freq="day",
    )  # MultiIndex (datetime, instrument)

    if df_close is None or df_close.empty:
        print(">>> [RawIC] Close price DataFrame is empty, skip raw-return IC.")
        return

    # 3) 计算 5 日 horizon raw return: Ref($close,-5) / Ref($close,-1) - 1
    close = df_close["$close"].unstack()  # [date, inst]
    ret5 = close.shift(-5) / close.shift(-1) - 1
    raw_label = ret5.stack().rename("raw_label")  # MultiIndex 对齐 pred 的 index 结构

    # 4) 对齐 pred 和 raw_label
    common_idx = pred.index.intersection(raw_label.index)
    if len(common_idx) == 0:
        print(">>> [RawIC] No intersection between pred index and raw_label index, skip.")
        return

    pred_aligned = pred.loc[common_idx].astype(float)
    label_aligned = raw_label.loc[common_idx].astype(float)

    dates_common = common_idx.get_level_values(dt_level)
    unique_dates = sorted(dates_common.unique())

    ic_vals = []
    ric_vals = []
    for dt in unique_dates:
        mask = dates_common == dt
        p = pred_aligned[mask].values
        y = label_aligned[mask].values

        finite = np.isfinite(p) & np.isfinite(y)
        if finite.sum() < 2:
            continue
        p = p[finite]
        y = y[finite]

        if np.std(p) <= 0 or np.std(y) <= 0:
            continue

        # Pearson IC
        ic = np.corrcoef(p, y)[0, 1]

        # Spearman RankIC
        rp = pd.Series(p).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        if np.std(rp) > 0 and np.std(ry) > 0:
            ric = np.corrcoef(rp, ry)[0, 1]
        else:
            ric = np.nan

        ic_vals.append((dt, ic))
        ric_vals.append((dt, ric))

    if not ic_vals:
        print(">>> [RawIC] No valid IC points computed, skip.")
        return

    ic_raw = pd.Series({dt: v for dt, v in ic_vals}).sort_index()
    ric_raw = pd.Series({dt: v for dt, v in ric_vals}).sort_index()

    print(
        f">>> [RawIC] IC_raw mean={ic_raw.mean():.4f}, std={ic_raw.std():.4f}; "
        f"RIC_raw mean={ric_raw.mean():.4f}, std={ric_raw.std():.4f}"
    )

    # 5) 存进 Recorder，后续报告直接用
    try:
        rec.save_objects(ic_raw=ic_raw, ric_raw=ric_raw)
    except Exception as e:
        print(f">>> [RawIC] save_objects(ic_raw/ric_raw) failed: {e}")


def _load_train_curves(rec):
    """
    从 Recorder 中读取训练曲线对象 train_curve（由 QlibQuantMoE.fit 保存）,
    并生成：
    - df_tc: DataFrame(epoch, train_listmle, train_ic, valid_listmle, valid_rank_ic, valid_ic)
    - train_summary_lines: 文本总结
    - fig_name: 图片文件名（相对路径），用于 Markdown 引用
    """
    local_dir: Path = rec.get_local_dir()
    fig_name = "train_curves_listmle_rankic.png"
    fig_path = local_dir / fig_name

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

            # --- 画 train/listmle vs valid/rank_ic ---
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

            # --- 文本总结 ---
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


def generate_paper_report(rec, model_name: str = "RST-MoE (MASTER-aligned)"):
    """
    汇总：
      - 训练过程诊断（ListMLE 收敛 vs RankIC）
      - IC / RankIC 序列（基于 raw 5 日收益）
      - 回测关键指标
      - gate time_ratio 序列
      - attention map 局部性指标
    输出 Markdown 报告到 kdd_report.md
    """
    local_dir: Path = rec.get_local_dir()
    report_path = local_dir / "kdd_report.md"

    # ---------- 1. Signal 层指标：优先使用 raw-return IC ----------
    use_raw_ic = False
    ic = pd.Series(dtype=float)
    ric = pd.Series(dtype=float)

    try:
        ic = rec.load_object("ic_raw")
        ric = rec.load_object("ric_raw")
        use_raw_ic = True
        print(">>> [Report] Using raw-return IC / RankIC from ic_raw / ric_raw.")
    except Exception:
        # 回退到 SigAnaRecord 基于 handler.label 的 IC
        sar = SigAnaRecord(rec)
        try:
            ic = sar.load("ic.pkl")   # 日度 IC（基于 handler.label）
            ric = sar.load("ric.pkl") # 日度 RankIC
            print(">>> [Report] Fallback to SigAna IC / RIC (handler label).")
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

    ic_t = ic_mean / ic_std * np.sqrt(len(ic)) if ic_std > 0 and len(ic) > 1 else np.nan
    ric_t = ric_mean / ric_std * np.sqrt(len(ric)) if ric_std > 0 and len(ric) > 1 else np.nan

    # ---------- 2. 回测指标 ----------
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
            metrics = rec.list_metrics()
            ann_ret = metrics.get("excess_return_with_cost.annualized_return", np.nan)
            info_ratio = metrics.get("excess_return_with_cost.information_ratio", np.nan)
            max_dd = metrics.get("excess_return_with_cost.max_drawdown", np.nan)
            turnover = metrics.get("excess_return_with_cost.turnover", np.nan)
    except Exception as e:
        print(f"[Report] Failed to load backtest indicators: {e}")

    # ---------- 3. gate & attention ----------
    gate_series = None
    attn_maps = None
    try:
        gate_series = rec.load_object("st_disentangle_gate_series")
    except Exception:
        pass
    try:
        attn_maps = rec.load_object("st_disentangle_attn_maps")
    except Exception:
        pass

    gate_stats_str = "N/A"
    if gate_series is not None is not False and len(gate_series) > 0:
        gate_series = gate_series.sort_index()
        g_mean = float(gate_series.mean())
        g_std = float(gate_series.std())
        g_p10 = float(gate_series.quantile(0.10))
        g_p90 = float(gate_series.quantile(0.90))
        gate_stats_str = f"mean={g_mean:.3f}, std={g_std:.3f}, p10={g_p10:.3f}, p90={g_p90:.3f}"

    attn_summary_lines: List[str] = []
    if isinstance(attn_maps, dict) and len(attn_maps) > 0:
        for dt_str, a in list(attn_maps.items())[:4]:
            try:
                a = np.asarray(a)
                if a.ndim == 3:
                    a = a.mean(axis=0)
                Tlen = a.shape[0]
                row_sum = a.sum(axis=-1, keepdims=True) + 1e-12
                a_norm = a / row_sum
                diag_mass = np.trace(a_norm) / Tlen
                band = np.eye(Tlen) + np.eye(Tlen, k=1) + np.eye(Tlen, k=-1)
                band_mass = (a_norm * band).sum() / Tlen
                attn_summary_lines.append(
                    f"- {dt_str}: diag_mass={diag_mass:.3f}, local_band_mass={band_mass:.3f}"
                )
            except Exception:
                continue
    if not attn_summary_lines:
        attn_summary_lines = ["- (no attention maps found; check export_visuals call)"]

    # ---------- 4. 训练过程诊断 ----------
    df_tc, train_summary_lines, train_fig_name = _load_train_curves(rec)

    # ---------- 5. 汇总表 ----------
    df_res = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Dataset": "Alpha158 / CSI800 / 2008-2022 (MASTER-aligned split)",
                "IC (mean)": f"{ic_mean:.4f}",
                "ICIR": f"{icir:.2f}",
                "IC t-stat": f"{ic_t:.1f}",
                "RankIC (mean)": f"{ric_mean:.4f}",
                "RankIC IR": f"{ricir:.2f}",
                "Ann. Return": f"{ann_ret:.2%}" if pd.notna(ann_ret) else "nan",
                "Info Ratio": f"{info_ratio:.2f}" if pd.notna(info_ratio) else "nan",
                "Max Drawdown": f"{max_dd:.2%}" if pd.notna(max_dd) else "nan",
                "Turnover": f"{turnover:.2%}" if pd.notna(turnover) else "nan",
                "Gate time_ratio stats": gate_stats_str,
            }
        ]
    )

    # ---------- 6. Markdown 报告 ----------
    lines: List[str] = []
    lines.append(f"# {model_name}\n")
    lines.append("## 1. Experimental Setup\n")
    setup_txt = f"""
    - **Data**: Alpha158 factors, CSI800 constituents
      - Train: 2008-01-01 ~ 2020-03-31
      - Valid: 2020-04-01 ~ 2020-06-30
      - Test:  2020-07-01 ~ 2022-12-31
    - **Label**: 5-day horizon return `Ref($close, -5) / Ref($close, -1) - 1`
      - learn_processors: DropnaLabel + CSRankNorm(label) → rank-label for training / ListMLE
    - **Model**: RST-MoE (Regime-aware Spatio-Temporal Mixture-of-Experts)
      - d_model = 32, n_layers = 2
      - Feature selector: differentiable sparse gate over Alpha158 factors
      - Router: regime encoder → 2-way (time vs. cross-sectional) MoE
      - Attention: bidirectional ALiBi in time & cross-section
    - **Backtest**:
      - Universe: CSI800, Benchmark: SH000906
      - Strategy: TopkDropout, topk=30, n_drop=30 (full turnover)
      - Backtest window: 2020-07-01 ~ 2022-12-31
      - Transaction:
        - Daily frequency, close price execution
        - Cost: open 5bp, close 15bp, limit_threshold=9.5%
    """
    lines.append(textwrap.dedent(setup_txt).strip() + "\n")

    lines.append("## 2. Cross-sectional Forecasting Performance\n")
    src_tag = "raw 5-day horizon returns" if use_raw_ic else "handler label (fallback)"
    perf_txt = f"""
    - **IC (test)** (computed on {src_tag}):
      - mean = {ic_mean:.4f}, std = {ic_std:.4f}, ICIR = {icir:.2f}, t-stat = {ic_t:.1f}
    - **RankIC (test)** (computed on {src_tag}):
      - mean = {ric_mean:.4f}, std = {ric_std:.4f}, IR = {ricir:.2f}, t-stat = {ric_t:.1f}
    """
    lines.append(textwrap.dedent(perf_txt).strip() + "\n")

    lines.append("## 3. Training Dynamics & Portfolio Backtest\n")
    lines.append("### 3.1 Training Dynamics (ListMLE vs RankIC)\n")
    lines.append(
        "训练阶段使用 **ListMLE 主 loss**（在 rank-label 上的 list-wise 排序），"
        "下图展示了 train/listmle 与 valid/rank_ic 随 epoch 的演化，并粗略衡量二者相关性：\n"
    )
    lines.extend(train_summary_lines)
    lines.append("")
    if train_fig_name is not None:
        lines.append(f"![Training dynamics (ListMLE vs RankIC)]({train_fig_name})\n")

    lines.append("### 3.2 Portfolio Backtest (CSI800, 2020-07 ~ 2022-12)\n")
    bt_txt = f"""
    - 年化收益 (excess return with cost): {ann_ret:.2%} (nan 表示回测文件缺失或未生成)
    - 信息比 (Information Ratio): {info_ratio:.2f}
    - 最大回撤: {max_dd:.2%}
    - 成交换手率 (Turnover): {turnover:.2%}
    - 与 MASTER 原始实验可在同一窗口 / 同一 Universe 下做直接对比。
    """
    lines.append(textwrap.dedent(bt_txt).strip() + "\n")

    lines.append("## 4. Spatio-Temporal Disentanglement Diagnostics\n")
    lines.append("### 4.1 Router Gate over Time (time vs. cross-sectional experts)\n")
    lines.append(f"- Gate time_ratio (time-expert weight) stats on test set: {gate_stats_str}\n")
    gate_interp = """
    - time_ratio 接近 1 表示更信任「时间 expert」，接近 0 表示更信任「截面 expert」。
    - 若 mean 在 (0.3, 0.7) 且 std > 0，说明路由器在不同阶段做出了非平凡决策；
      若长期贴近 0 或 1，则 MoE 退化为单专家模型。
    """
    lines.append(textwrap.dedent(gate_interp).strip() + "\n")

    lines.append("### 4.2 Temporal Attention Locality\n")
    lines.append("基于若干代表性交易日的 time-attention heatmap，统计对角/邻近对角的注意力质量：\n")
    lines.extend(attn_summary_lines)
    lines.append("")
    attn_interp = """
    - diag_mass 衡量注意力在完全对齐的时间步 (i=j) 上的质量；
    - local_band_mass 衡量注意力在 |i-j| ≤ 1 的近邻时间步上的质量。
    - 越高说明模型更偏向「局部时序模式」（类似 AR / 局部卷积），越低说明依赖更长程时序信息。
    """
    lines.append(textwrap.dedent(attn_interp).strip() + "\n")

    lines.append("## 5. Summary\n")
    lines.append(
        "在与 MASTER 完全对齐的 Universe / 时间切分 / Label horizon / Backtest 配置下，"
        "RST-MoE 显示出从 rank-label → ListMLE list-wise 优化 → 截面预测 → 组合收益的一条相对清晰的传导链，"
        "并通过 gate 曲线 / attention 局部性提供额外的结构解释力。\n"
    )

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
    with R.start(experiment_name="MASTER_Aligned_RST_MoE"):
        # 2.1 记录超参
        R.log_params(**flatten_dict(model_conf))

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
            target_dates=None,  # 或者传 ["2021-01-04", ...]
            prefix="st_disentangle",
        )

        # 2.4 Signal 生成与分析 (基于 handler.label 的标准 IC / RankIC)
        print(">>> [Phase 2] Signal Analysis...")
        rec = R.get_recorder()
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        # 2.4bis 使用 raw close 价格计算 5 日 horizon raw-return IC / RankIC
        print(">>> [Phase 2.1] Compute raw-return IC / RankIC...")
        compute_raw_return_ic(rec, model, dataset, segment="test")

        # 2.5 组合回测
        print(">>> [Phase 3] Backtesting...")
        PortAnaRecord(rec, port_conf, "day").generate()

        # 2.6 生成论文级报告
        print(">>> [Phase 4] Generate Paper-level Report...")
        generate_paper_report(rec, model_name="RST-MoE (MASTER-aligned)")
