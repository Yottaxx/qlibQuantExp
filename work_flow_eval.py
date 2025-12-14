# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import pprint
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# Qlib report utilities (plotly figs)
from qlib.contrib.report import analysis_model, analysis_position
from qlib.contrib.evaluate import risk_analysis  # benchmark risk stats path per qlib docs

# ----------------------------
# 0) 复用训练脚本里的 data_conf / port_conf
# ----------------------------
def build_data_conf() -> dict:
    # 推荐：直接复用训练脚本里的 dict（确保一致）
    from work_flow import data_conf  # noqa
    return data_conf


def build_port_conf() -> dict:
    from work_flow import port_conf  # noqa
    return port_conf


# ----------------------------
# 1) Recorder / model 加载
# ----------------------------
def _try_load_object(rec, candidates: Iterable[str]) -> Any:
    last_err = None
    for name in candidates:
        try:
            return rec.load_object(name)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load object: candidates={list(candidates)}; last_err={last_err}")


def load_trained_model(train_exp: str, train_rid: str):
    train_rec = R.get_recorder(experiment_name=train_exp, recorder_id=train_rid)

    # 兼容你不同保存习惯：model / trained_model / *.pkl
    model = _try_load_object(train_rec, ["model", "trained_model", "model.pkl", "trained_model.pkl"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 兼容你的 adapter：有 net 就搬 device
    if hasattr(model, "device"):
        try:
            model.device = device
        except Exception:
            pass
    if getattr(model, "net", None) is not None:
        try:
            model.net.to(device)
        except Exception:
            pass

    return model


# ----------------------------
# 2) Metrics helpers：优先 list_metrics，其次 artifacts
# ----------------------------
def _to_float(x) -> float:
    try:
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
        return float(x)
    except Exception:
        return float("nan")


def pick_metric(metrics: Dict[str, Any], keys: List[str]) -> float:
    """
    keys: 按优先级尝试的 key 列表
    兼容：有些版本 metric key 前面会带 freq 前缀（如 1day.xxx）
    """
    for k in keys:
        if k in metrics:
            return _to_float(metrics[k])

    # 再做一次“智能猜测”：如果 keys 不含 1day. 前缀，补上试试
    for k in keys:
        kk = f"1day.{k}" if not k.startswith("1day.") else k
        if kk in metrics:
            return _to_float(metrics[kk])

    return float("nan")


def _risk_value_from_df(df: Any, key: str) -> float:
    """
    risk_analysis 的返回常见形态：
      index=['mean','std','annualized_return','information_ratio','max_drawdown']
      columns=['risk']
    """
    if isinstance(df, pd.DataFrame):
        if key in df.index:
            if "risk" in df.columns:
                return float(df.loc[key, "risk"])
            if df.shape[1] == 1:
                return float(df.loc[key, df.columns[0]])
    return float("nan")


# ----------------------------
# 3) Plotly figs 保存（HTML），并 log_artifact
# ----------------------------
def _ensure_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        out = []
        for x in obj:
            out.extend(_ensure_list(x))
        return out
    return [obj]


def save_plotly_figs(figs: Any, out_dir: Path, stem: str, rec) -> List[Path]:
    """
    Qlib report 返回 plotly Figure list/tuple；这里统一保存为 html。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_list = _ensure_list(figs)

    saved: List[Path] = []
    try:
        import plotly.io as pio  # noqa
    except Exception as e:
        print(f"[Graphs] plotly not available: {e}")
        return saved

    for i, fig in enumerate(fig_list):
        # 有些返回里混入非 Figure 对象，跳过
        if not hasattr(fig, "to_html"):
            continue
        path = out_dir / (f"{stem}.html" if i == 0 else f"{stem}_{i}.html")
        try:
            fig.write_html(str(path), include_plotlyjs="inline", full_html=True)
            saved.append(path)
        except Exception as e:
            print(f"[Graphs] write_html failed: {path} | {e}")

    # 统一把目录作为 artifact（mlflow 支持目录）
    try:
        rec.log_artifact(str(out_dir))
    except Exception:
        # fallback：逐文件
        for p in saved:
            try:
                rec.log_artifact(str(p))
            except Exception:
                pass
    return saved


# ----------------------------
# 4) Graphs：把你要的“分组收益/IC 曲线/风险曲线/回测报告曲线”全画出来
# ----------------------------
def generate_analysis_graphs(rec, freq: str = "1day") -> None:
    """
    使用 Qlib 官方 report 模块画图并保存：
      - 回测 report 曲线：analysis_position.report_graph
      - 风险曲线：analysis_position.risk_analysis_graph
      - IC 曲线：analysis_position.score_ic_graph
      - 模型性能（含分组收益/IC/自相关）：analysis_model.model_performance_graph
    """
    out_dir = Path(rec.get_local_dir()) / "analysis_graphs"

    # --- load objects (全部来自 artifacts，保证可复现/可验证)
    pred_df = rec.load_object("pred.pkl")
    label_df = rec.load_object("label.pkl")  # SignalRecord 已保存
    report_normal = rec.load_object(f"portfolio_analysis/report_normal_{freq}.pkl")
    port_analysis = rec.load_object(f"portfolio_analysis/port_analysis_{freq}.pkl")

    # --- normalize pred/label schema
    if isinstance(pred_df, pd.Series):
        pred_df = pred_df.to_frame("score")
    elif isinstance(pred_df, pd.DataFrame) and pred_df.shape[1] == 1:
        pred_df.columns = ["score"]
    elif isinstance(pred_df, pd.DataFrame) and "score" not in pred_df.columns:
        # 兜底：取第一列当 score
        pred_df = pred_df.iloc[:, [0]].copy()
        pred_df.columns = ["score"]

    if isinstance(label_df, pd.Series):
        label_df = label_df.to_frame("label")
    elif isinstance(label_df, pd.DataFrame) and label_df.shape[1] == 1:
        label_df.columns = ["label"]
    elif isinstance(label_df, pd.DataFrame) and "label" not in label_df.columns:
        label_df = label_df.iloc[:, [0]].copy()
        label_df.columns = ["label"]

    # pred_label: MultiIndex(instrument, datetime) with columns [label, score]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    # --- 1) report curves
    try:
        figs = analysis_position.report_graph(report_normal, show_notebook=False)
        save_plotly_figs(figs, out_dir / "position", "report_graph", rec)
    except Exception as e:
        print(f"[Graphs] report_graph failed: {e}")

    # --- 2) risk curves
    try:
        figs = analysis_position.risk_analysis_graph(port_analysis, report_normal, show_notebook=False)
        save_plotly_figs(figs, out_dir / "position", "risk_analysis_graph", rec)
    except Exception as e:
        print(f"[Graphs] risk_analysis_graph failed: {e}")

    # --- 3) IC curve
    try:
        figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
        save_plotly_figs(figs, out_dir / "position", "score_ic_graph", rec)
    except Exception as e:
        print(f"[Graphs] score_ic_graph failed: {e}")

    # --- 4) model performance (group_return / pred_ic / pred_autocorr)
    try:
        figs = analysis_model.model_performance_graph(
            pred_label,
            lag=1,
            N=5,
            rank=False,
            graph_names=["group_return", "pred_ic", "pred_autocorr"],
            show_notebook=False,
            show_nature_day=True,
        )
        save_plotly_figs(figs, out_dir / "model", "model_performance_graph", rec)
    except Exception as e:
        print(f"[Graphs] model_performance_graph failed: {e}")

    print(f"[Graphs] saved under: {out_dir}")


# ----------------------------
# 5) Report：summary 指标优先 list_metrics；benchmark 用 report_normal['bench'] risk_analysis
# ----------------------------
def generate_paper_report(rec, *, data_conf: dict, port_conf: dict, model_name: str = "RST-MoE", prefix: str = "st_disentangle") -> Path:
    local_dir = Path(rec.get_local_dir())
    report_path = local_dir / "kdd_report.md"

    # 1) segments
    seg_txt = "N/A"
    try:
        seg = (((data_conf or {}).get("kwargs") or {}).get("segments")) or None
        if isinstance(seg, dict):
            seg_txt = "\n".join([f"  - {k}: {v[0]} ~ {v[1]}" for k, v in seg.items() if isinstance(v, (list, tuple)) and len(v) == 2])
    except Exception:
        pass

    # 2) Signal (IC / RankIC) from artifacts
    ic = pd.Series(rec.load_object("sig_analysis/ic.pkl"))
    ric = pd.Series(rec.load_object("sig_analysis/ric.pkl"))

    def _series_stats(s: pd.Series):
        s = pd.Series(s).dropna()
        if len(s) == 0:
            return dict(mean=np.nan, std=np.nan, ir=np.nan, t=np.nan, n=0)
        m = float(s.mean())
        sd = float(s.std())
        ir = (m / sd) if sd > 0 else np.nan
        t = (m / sd * np.sqrt(len(s))) if (sd > 0 and len(s) > 1) else np.nan
        return dict(mean=m, std=sd, ir=ir, t=t, n=int(len(s)))

    ic_s = _series_stats(ic)
    ric_s = _series_stats(ric)

    # 3) Backtest scalars: prefer list_metrics()
    metrics = rec.list_metrics() or {}

    # excess return without/with cost
    wo = {
        "mean": pick_metric(metrics, ["excess_return_without_cost.mean", "1day.excess_return_without_cost.mean"]),
        "std": pick_metric(metrics, ["excess_return_without_cost.std", "1day.excess_return_without_cost.std"]),
        "ann_ret": pick_metric(metrics, ["excess_return_without_cost.annualized_return", "1day.excess_return_without_cost.annualized_return"]),
        "ir": pick_metric(metrics, ["excess_return_without_cost.information_ratio", "1day.excess_return_without_cost.information_ratio"]),
        "mdd": pick_metric(metrics, ["excess_return_without_cost.max_drawdown", "1day.excess_return_without_cost.max_drawdown"]),
    }
    wc = {
        "mean": pick_metric(metrics, ["excess_return_with_cost.mean", "1day.excess_return_with_cost.mean"]),
        "std": pick_metric(metrics, ["excess_return_with_cost.std", "1day.excess_return_with_cost.std"]),
        "ann_ret": pick_metric(metrics, ["excess_return_with_cost.annualized_return", "1day.excess_return_with_cost.annualized_return"]),
        "ir": pick_metric(metrics, ["excess_return_with_cost.information_ratio", "1day.excess_return_with_cost.information_ratio"]),
        "mdd": pick_metric(metrics, ["excess_return_with_cost.max_drawdown", "1day.excess_return_with_cost.max_drawdown"]),
    }

    # indicators (ffr/pa/pos) usually stored as 1day.ffr / 1day.pa / 1day.pos
    ind = {
        "ffr": pick_metric(metrics, ["1day.ffr", "ffr"]),
        "pa": pick_metric(metrics, ["1day.pa", "pa"]),
        "pos": pick_metric(metrics, ["1day.pos", "pos"]),
    }

    # benchmark risk: DO NOT rely on indicator_analysis (often missing); compute from report_normal['bench']
    bench = dict(mean=np.nan, std=np.nan, ann_ret=np.nan, ir=np.nan, mdd=np.nan)
    try:
        report_normal = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        if isinstance(report_normal, pd.DataFrame) and "bench" in report_normal.columns:
            ra = risk_analysis(report_normal["bench"], freq="day")
            bench = {
                "mean": _risk_value_from_df(ra, "mean"),
                "std": _risk_value_from_df(ra, "std"),
                "ann_ret": _risk_value_from_df(ra, "annualized_return"),
                "ir": _risk_value_from_df(ra, "information_ratio"),
                "mdd": _risk_value_from_df(ra, "max_drawdown"),
            }
    except Exception as e:
        print(f"[Report] benchmark risk_analysis failed: {e}")

    # 4) gate/attn diagnostics (artifacts)
    gate_stats_str = "N/A"
    try:
        gate_series = rec.load_object(f"{prefix}_gate_series")
        if isinstance(gate_series, pd.Series) and len(gate_series) > 0:
            gate_stats_str = f"mean={gate_series.mean():.3f}, std={gate_series.std():.3f}, p10={gate_series.quantile(0.10):.3f}, p90={gate_series.quantile(0.90):.3f}"
    except Exception:
        pass

    attn_lines = ["- (no attention maps found; check export_visuals)"]
    try:
        attn_maps = rec.load_object(f"{prefix}_attn_maps")
        if isinstance(attn_maps, dict) and len(attn_maps) > 0:
            attn_lines = []
            for dt_str, a in list(attn_maps.items())[:4]:
                a = np.asarray(a)
                if a.ndim == 3:
                    a = a.mean(axis=0)
                T = a.shape[0]
                row_sum = a.sum(axis=-1, keepdims=True) + 1e-12
                a_norm = a / row_sum
                diag_mass = float(np.trace(a_norm) / T)
                band = np.eye(T) + np.eye(T, k=1) + np.eye(T, k=-1)
                band_mass = float((a_norm * band).sum() / T)
                attn_lines.append(f"- {dt_str}: diag_mass={diag_mass:.3f}, local_band_mass={band_mass:.3f}")
    except Exception:
        pass

    # 5) write markdown
    lines: List[str] = []
    lines.append(f"# {model_name} (Eval)\n")
    lines.append("## 1. Experimental Setup\n")
    lines.append(f"- Recorder ID: {rec.id}\n")
    lines.append("- segments (hit: root.kwargs.segments)\n")
    lines.append(seg_txt + "\n")
    lines.append("- Backtest config (brief):\n")
    lines.append(f"  - strategy: {port_conf.get('strategy')}\n")
    lines.append(f"  - backtest: {port_conf.get('backtest')}\n")

    lines.append("\n## 2. Signal Quality\n")
    lines.append(
        f"- IC: mean={ic_s['mean']:.6f}, std={ic_s['std']:.6f}, ICIR={ic_s['ir']:.3f}, t={ic_s['t']:.2f}, n={ic_s['n']}  (hit: sig_analysis/ic.pkl)\n"
        f"- RankIC: mean={ric_s['mean']:.6f}, std={ric_s['std']:.6f}, IR={ric_s['ir']:.3f}, t={ric_s['t']:.2f}, n={ric_s['n']}  (hit: sig_analysis/ric.pkl)\n"
    )

    lines.append("\n## 3. Portfolio Backtest\n")
    lines.append("- scalars prefer: `rec.list_metrics()`\n")
    lines.append(
        f"\n- Benchmark return (1day): mean={bench['mean']:.6f}, std={bench['std']:.6f}, ann_ret={bench['ann_ret']:.6f}, IR={bench['ir']:.3f}, maxDD={bench['mdd']:.6f}\n"
        f"\n- Excess Return without cost (1day): mean={wo['mean']:.6f}, std={wo['std']:.6f}, ann_ret={wo['ann_ret']:.6f}, IR={wo['ir']:.3f}, maxDD={wo['mdd']:.6f}\n"
        f"\n- Excess Return with cost (1day): mean={wc['mean']:.6f}, std={wc['std']:.6f}, ann_ret={wc['ann_ret']:.6f}, IR={wc['ir']:.3f}, maxDD={wc['mdd']:.6f}\n"
        f"\n- Indicators(1day): {ind}\n"
    )

    lines.append("\n## 4. Spatio-Temporal Diagnostics\n")
    lines.append(f"- Gate time_ratio stats: {gate_stats_str} (hit: {prefix}_gate_series)\n")
    lines.append(f"\n- Attention maps hit: {prefix}_attn_maps\n")
    lines.append("\n### Temporal Attention Locality\n")
    lines.extend(attn_lines)
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        rec.log_artifact(str(report_path))
    except Exception:
        pass
    print(f"[Report] written: {report_path}")
    return report_path


# ----------------------------
# 6) main：eval-only
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider_uri", type=str, default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--region", type=str, default="cn")
    ap.add_argument("--src_exp", type=str, required=False, default="Official_Alignment_RST_MoE")
    ap.add_argument("--src_rid", type=str, required=False, default="9464b2992ce2453cbd8863379be1aa27")

    ap.add_argument("--eval_exp", type=str, default="Official_Alignment_RST_MoE_Eval")
    ap.add_argument("--segment", type=str, default="test")
    ap.add_argument("--max_attn_days", type=int, default=4)
    ap.add_argument("--attn_layer", type=int, default=-1)
    ap.add_argument("--no_graphs", action="store_true", help="disable analysis graphs")
    args = ap.parse_args()

    provider_uri = os.path.expanduser(args.provider_uri)
    region = REG_CN if args.region.lower() == "cn" else REG_CN

    qlib.init(provider_uri=provider_uri, region=region)

    data_conf = build_data_conf()
    port_conf = build_port_conf()

    dataset = init_instance_by_config(data_conf)
    model = load_trained_model(args.src_exp, args.src_rid)

    done_local_dir: Optional[str] = None
    done_rid: Optional[str] = None

    with R.start(experiment_name=args.eval_exp, recorder_name=f"eval_from_{args.src_rid}"):
        rec = R.get_recorder()
        done_local_dir = rec.get_local_dir()
        done_rid = rec.id

        rec.set_tags(src_exp=args.src_exp, src_rid=args.src_rid, segment=args.segment)
        R.log_params(**flatten_dict({"data_conf": data_conf, "port_conf": port_conf, "eval_args": vars(args)}))

        # 1) 诊断图（你自定义的 export_visuals）
        print(">>> [Eval Phase 1] Export Spatio-Temporal Visuals...")
        if hasattr(model, "export_visuals"):
            model.export_visuals(
                dataset,
                segment=args.segment,
                max_attn_days=args.max_attn_days,
                attn_layer=args.attn_layer,
                prefix="st_disentangle",
            )
        else:
            print("[Warn] model has no export_visuals(); skip.")

        # 2) Signal & SigAna
        print(">>> [Eval Phase 2] Signal Analysis...")
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        # 3) Backtest
        print(">>> [Eval Phase 3] Backtesting...")
        PortAnaRecord(rec, port_conf, "day").generate()

        # 4) Graphs (group return / ic / risk / report)
        if not args.no_graphs:
            print(">>> [Eval Phase 4] Generate Analysis Graphs...")
            generate_analysis_graphs(rec, freq="1day")

        # 5) Report
        print(">>> [Eval Phase 5] Report...")
        generate_paper_report(rec, data_conf=data_conf, port_conf=port_conf, model_name="RST-MoE", prefix="st_disentangle")

    print(f"[Done] eval recorder_id={done_rid} local_dir={done_local_dir}")


if __name__ == "__main__":
    main()


