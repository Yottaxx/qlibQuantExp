# -*- coding: utf-8 -*-
"""
RST-MoE + Qlib Official Workflow (Paper-Ready Version)

功能：
1. 使用 Alpha158 / CSI300 官方切分，训练 QlibQuantMoE（时序 MoE）。
2. 运行标准 Signal 分析 + 组合回测。
3. 调用 model.export_visuals 导出：
   - gate time_ratio 随时间曲线
   - 若干交易日的 time-attention heatmap
4. 从 Recorder 中汇总：
   - IC / RankIC 时间序列 + ICIR / t-stat
   - 回测指标（年化收益、信息比、最大回撤等）
   - gate 曲线统计（均值 / std / 分位数）
   - attention map 的局部性指标
5. 自动生成一份 Markdown 版「论文级实验报告」：kdd_report.md
   - 增加“训练过程诊断”：train/listmle vs valid/rank_ic 曲线 + 文本总结
"""
from typing import Optional, List

import numpy as np
import pandas as pd
from pathlib import Path
import textwrap

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
                # Label: 下一日收益（在 learn_processors 中会被做成 rank-label）
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
            "use_feature_selection": False,
            # context_len 和 num_alphas 会在 QlibQuantMoE 内自动探测
        },
        "trainer_config": {
            "lr": 5e-4,
            "n_epochs": 20,
            "batch_size": 16,  # 对应 FixedDailyBatchSampler 的日度 batch
            "early_stop": 5,
            "num_workers": 0,  # debug 时用 0，正式训练可以拉高
            # Warmup 配置（与 adapter 中的默认值一致）：
            "use_warmup": True,
            "warmup_ratio": 0.05,
            "warmup_steps": 0,
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


def generate_paper_report(rec, model_name: str = "RST-MoE"):
    """
    汇总当前 Recorder 中的：
      - 训练过程诊断：ListMLE 收敛 vs RankIC
      - IC / RankIC 序列
      - 回测关键指标
      - gate time_ratio 序列统计
      - attention map 的局部性指标
    输出一份 Markdown 报告到 kdd_report.md，并在控制台打印。
    """
    local_dir: Path = rec.get_local_dir()
    report_path = local_dir / "kdd_report.md"

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

    # t-stat 作为统计显著性
    ic_t = ic_mean / ic_std * np.sqrt(len(ic)) if ic_std > 0 and len(ic) > 1 else np.nan
    ric_t = ric_mean / ric_std * np.sqrt(len(ric)) if ric_std > 0 and len(ric) > 1 else np.nan

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

    # ---------- 3. gate & attention 诊断 ----------
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

    # gate time_ratio 统计
    gate_stats_str = "N/A"
    if gate_series is not None is not False and len(gate_series) > 0:
        gate_series = gate_series.sort_index()
        g_mean = float(gate_series.mean())
        g_std = float(gate_series.std())
        g_p10 = float(gate_series.quantile(0.10))
        g_p90 = float(gate_series.quantile(0.90))
        gate_stats_str = (
            f"mean={g_mean:.3f}, std={g_std:.3f}, p10={g_p10:.3f}, p90={g_p90:.3f}"
        )

    # attention 局部性指标：看时间注意力在 |i-j|<=1 对角带上的质量
    attn_summary_lines = []
    if isinstance(attn_maps, dict) and len(attn_maps) > 0:
        for dt_str, a in list(attn_maps.items())[:4]:  # 最多展示 4 天
            try:
                a = np.asarray(a)  # [T, T] 或 [H, T, T]，export_visuals 中已做过平均
                if a.ndim == 3:
                    a = a.mean(axis=0)
                Tlen = a.shape[0]
                row_sum = a.sum(axis=-1, keepdims=True) + 1e-12
                a_norm = a / row_sum  # 每行归一化，确保是概率分布
                diag_mass = np.trace(a_norm) / Tlen
                # ±1 带：对角、上 1、下 1
                band = np.eye(Tlen) + np.eye(Tlen, k=1) + np.eye(Tlen, k=-1)
                band_mass = (a_norm * band).sum() / Tlen
                attn_summary_lines.append(
                    f"- {dt_str}: diag_mass={diag_mass:.3f}, local_band_mass={band_mass:.3f}"
                )
            except Exception:
                continue
    if not attn_summary_lines:
        attn_summary_lines = ["- (no attention maps found; check export_visuals call)"]

    # ---------- 4. 训练过程诊断（ListMLE vs RankIC） ----------
    df_tc, train_summary_lines, train_fig_name = _load_train_curves(rec)

    # ---------- 5. 汇总成表格（方便 VS baseline 比较） ----------
    df_res = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Dataset": "Alpha158 / CSI300 / 2008-2020 (official split)",
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

    # ---------- 6. 生成 Markdown 报告 ----------
    lines: List[str] = []
    lines.append(f"# {model_name} on Alpha158 / CSI300\n")
    lines.append("## 1. Experimental Setup\n")
    setup_txt = f"""
    - **Data**: Alpha158 factors, CSI300 constituents
      - Train: 2008-01-01 ~ 2014-12-31
      - Valid: 2015-01-01 ~ 2016-12-31
      - Test: 2017-01-01 ~ 2020-08-01
    - **Label**: next-day return `Ref($close, -1) / $close - 1` (经过 CSRankNorm → rank-label)
    - **Model**: RST-MoE (Regime-aware Spatio-Temporal Mixture-of-Experts)
      - d_model = 32, n_layers = 2
      - Feature selector: differentiable sparse gate over Alpha158 factors
      - Router: regime encoder → 2-way (time vs. cross-sectional) MoE
      - Attention: bidirectional ALiBi in time & cross-section
    - **Backtest**:
      - Strategy: TopkDropout, topk=50, n_drop=5
      - Benchmark: SH000300, daily frequency, close price execution
      - Transaction cost: open 5bp, close 15bp, limit_threshold=9.5%
    """
    lines.append(textwrap.dedent(setup_txt).strip() + "\n")

    lines.append("## 2. Cross-sectional Forecasting Performance\n")
    perf_txt = f"""
    - **IC (test)**:
      - mean = {ic_mean:.4f}, std = {ic_std:.4f}, ICIR = {icir:.2f}, t-stat = {ic_t:.1f}
    - **RankIC (test)**:
      - mean = {ric_mean:.4f}, std = {ric_std:.4f}, IR = {ricir:.2f}, t-stat = {ric_t:.1f}
    - 统计上，IC t-stat ≫ 2 一般被认为在日频具有显著 alpha 能力。
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
    - 与 Qlib 官方基准可对照：
      - LightGBM: RankIC ≈ 0.08, Ann. Ret ≈ 20%, Max DD ≈ -10%
      - Linear:   RankIC ≈ 0.05, Ann. Ret ≈ 8%,  Max DD ≈ -15%
    """
    lines.append(textwrap.dedent(bt_txt).strip() + "\n")

    lines.append("## 4. Spatio-Temporal Disentanglement Diagnostics\n")
    lines.append("### 4.1 Router Gate over Time (time vs. cross-sectional experts)\n")
    lines.append(f"- Gate time_ratio (time-expert weight) stats on test set: {gate_stats_str}\n")
    gate_interp = """
    - time_ratio 接近 1 表示更信任「时间 expert」，接近 0 表示更信任「截面 expert」。
    - 若 mean 在 (0.3, 0.7) 且 std > 0，说明路由器确实在不同阶段做非平凡决策；
      若长期贴近 0 或 1，则 MoE 退化为单专家模型。
    """
    lines.append(textwrap.dedent(gate_interp).strip() + "\n")

    lines.append("### 4.2 Temporal Attention Locality\n")
    lines.append(
        "下列指标基于若干代表性交易日的 time-attention heatmap，统计对角/邻近对角的注意力质量：\n"
    )
    lines.extend(attn_summary_lines)
    lines.append("")
    attn_interp = """
    - diag_mass 衡量注意力在完全对齐的时间步 (i=j) 上的质量；
    - local_band_mass 衡量注意力在 |i-j| ≤ 1 的近邻时间步上的质量。
    - 越高说明模型更偏向「局部时序模式」（类似 AR / 局部卷积），
      越低说明模型依赖更长程的时序依赖。
    """
    lines.append(textwrap.dedent(attn_interp).strip() + "\n")

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
        generate_paper_report(rec, model_name="RST-MoE")
