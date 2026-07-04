#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Long-running orchestrator for the four-day diagnostic plan.

The script is intentionally self-contained. It waits for the current matrix
runner to finish, then runs:

1. the same diagnostic matrix with seed=42, 40 epochs, no early stop;
2. the full_135 capacity/head/dropout Stage 1;
3. seed=43/44 repeats for cap00 and Stage-1 top2 non-baseline configs;
4. optional Stage 3 aggressive configs when Stage 1 shows no clear overfit.

Every run restores the best valid RankIC checkpoint via trainer overrides.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MATRIX_SETTINGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "full_135": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
    },
    "base": {
        "model": {
            "use_regime_time_embedding": False,
            "use_regime_factor_gate": False,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
    },
    "time_only": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": False,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
    },
    "film_only": {
        "model": {
            "use_regime_time_embedding": False,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
    },
    "no_layer_summary": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": False,
            "router_mode": "learned",
        },
    },
    "fixed_router_05": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "fixed_05",
        },
    },
}


FULL135_FIXED_MODEL = {
    "use_regime_time_embedding": True,
    "use_regime_factor_gate": True,
    "router_use_layer_summary": True,
    "router_mode": "learned",
    "main_loss": "mse",
    "regime_macro_dropout": 0.1,
    "pooling_alpha": 0.7,
    "use_feature_selection": False,
    "use_alibi": False,
}


CAPACITY_STAGE1: List[Dict[str, Any]] = [
    {"setting": "cap00_d64_h4_l2_ff128_do010", "d_model": 64, "n_heads": 4, "d_head": 16, "n_layers": 2, "d_ff": 128, "dropout": 0.10, "params_plan": 203847},
    {"setting": "cap01_d64_h4_l2_ff256_do015", "d_model": 64, "n_heads": 4, "d_head": 16, "n_layers": 2, "d_ff": 256, "dropout": 0.15, "params_plan": 236871},
    {"setting": "cap02_d80_h4_l2_ff320_do015", "d_model": 80, "n_heads": 4, "d_head": 20, "n_layers": 2, "d_ff": 320, "dropout": 0.15, "params_plan": 355607},
    {"setting": "cap03_d96_h4_l2_ff288_do015", "d_model": 96, "n_heads": 4, "d_head": 24, "n_layers": 2, "d_ff": 288, "dropout": 0.15, "params_plan": 461095},
    {"setting": "cap04_d96_h4_l2_ff288_do020", "d_model": 96, "n_heads": 4, "d_head": 24, "n_layers": 2, "d_ff": 288, "dropout": 0.20, "params_plan": 461095},
    {"setting": "cap05_d96_h4_l2_ff384_do015", "d_model": 96, "n_heads": 4, "d_head": 24, "n_layers": 2, "d_ff": 384, "dropout": 0.15, "params_plan": 498151},
    {"setting": "cap06_d96_h4_l2_ff384_do020", "d_model": 96, "n_heads": 4, "d_head": 24, "n_layers": 2, "d_ff": 384, "dropout": 0.20, "params_plan": 498151},
    {"setting": "cap07_d96_h6_l2_ff384_do015", "d_model": 96, "n_heads": 6, "d_head": 16, "n_layers": 2, "d_ff": 384, "dropout": 0.15, "params_plan": 498151},
    {"setting": "cap08_d96_h4_l3_ff384_do020", "d_model": 96, "n_heads": 4, "d_head": 24, "n_layers": 3, "d_ff": 384, "dropout": 0.20, "params_plan": 674265},
]


CAPACITY_STAGE3: List[Dict[str, Any]] = [
    {"setting": "cap09_d128_h8_l2_ff384_do020", "d_model": 128, "n_heads": 8, "d_head": 16, "n_layers": 2, "d_ff": 384, "dropout": 0.20, "params_plan": 788871},
    {"setting": "cap10_d128_h8_l2_ff512_do025", "d_model": 128, "n_heads": 8, "d_head": 16, "n_layers": 2, "d_ff": 512, "dropout": 0.25, "params_plan": 854663},
    {"setting": "cap11_d128_h4_l2_ff512_do020", "d_model": 128, "n_heads": 4, "d_head": 32, "n_layers": 2, "d_ff": 512, "dropout": 0.20, "params_plan": 854663},
]


SUMMARY_KEYS = [
    "performance/daily_rank_ic_mean",
    "performance/rank_icir",
    "performance/daily_ic_mean",
    "performance/icir",
    "performance/rolling20_rank_ic_min",
    "performance/rolling60_rank_ic_min",
    "performance/top_bottom_spread_mean",
    "portfolio/annualized_return_with_cost",
    "portfolio/information_ratio_with_cost",
    "portfolio/max_drawdown_with_cost",
    "portfolio/turnover",
    "portfolio/cost_drag",
    "optimization/post_peak_decay",
    "optimization/loss_gap_last",
    "optimization/rank_ic_gap_last",
    "optimization/train_grad_norm_last",
    "optimization/train_grad_nonfinite_rate_last",
    "optimization/train_grad_skipped_rate_last",
    "optimization/train_score_std_last",
    "optimization/valid_score_std_last",
    "attention_pooling/factor_entropy_norm_mean",
    "attention_pooling/factor_top10_mass_mean",
    "router/entropy_norm_mean",
    "router/time_ratio_mean",
    "router/collapse_ratio",
    "regime/pc1_tail_2x2_rank_ic_spread",
]


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def prevent_system_sleep() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        es_continuous = 0x80000000
        es_system_required = 0x00000001
        es_awaymode_required = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required | es_awaymode_required
        )
        log("requested Windows execution state: keep system awake while orchestrator runs")
    except Exception as exc:
        log(f"could not request keep-awake execution state: {exc!r}")


def pid_exists(pid: int) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        cp = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            errors="ignore",
        )
        out = (cp.stdout or "") + (cp.stderr or "")
        return str(int(pid)) in out and "No tasks" not in out
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def read_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def find_latest_diagnostic(root: Path, *, started_at: float) -> str:
    mlruns = root / "mlruns"
    if not mlruns.exists():
        return ""
    best_path = ""
    best_mtime = started_at
    for path in mlruns.rglob("diagnostic_matrix.csv"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= best_mtime:
            best_mtime = mtime
            best_path = str(path)
    return best_path


def load_diag_metrics(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get('group', '')}/{row.get('metric', '')}"
                try:
                    val = float(row.get("value", "nan"))
                except Exception:
                    continue
                if math.isfinite(val):
                    out[key] = val
    except Exception:
        pass
    return out


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def load_train_curve_summary(diag_path: str) -> Dict[str, Any]:
    if not diag_path:
        return {}
    run_dir = Path(diag_path).parent
    curve_path = run_dir / "artifacts" / "train_curve"
    if not curve_path.exists():
        return {}
    try:
        with curve_path.open("rb") as f:
            curve = pickle.load(f)
    except Exception as exc:
        return {"train_curve_error": str(exc)}
    if not isinstance(curve, dict):
        return {}

    epochs = list(curve.get("epoch", []) or [])
    rankics = list(curve.get("valid_rank_ic", []) or [])
    best_i = None
    best_val = None
    for i, raw in enumerate(rankics):
        val = finite_float(raw)
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_val = val
            best_i = i
    out: Dict[str, Any] = {
        "epochs_recorded": len(epochs),
        "final_epoch": epochs[-1] if epochs else "",
    }
    if best_i is None:
        return out

    def at(key: str) -> Any:
        arr = curve.get(key, []) or []
        if best_i < len(arr):
            val = finite_float(arr[best_i])
            return "" if val is None else val
        return ""

    final_rankic = finite_float(rankics[-1]) if rankics else None
    out.update(
        {
            "best_valid_rank_ic": best_val,
            "best_valid_rank_ic_epoch": epochs[best_i] if best_i < len(epochs) else best_i + 1,
            "valid_ic_at_best": at("valid_ic"),
            "valid_main_at_best": at("valid_main"),
            "train_main_at_best": at("train_main"),
            "train_grad_nonfinite_rate_at_best": at("train_grad_nonfinite_rate"),
            "train_grad_skipped_rate_at_best": at("train_grad_skipped_rate"),
            "final_valid_rank_ic": "" if final_rankic is None else final_rankic,
            "best_minus_final_valid_rank_ic": (
                "" if final_rankic is None or best_val is None else best_val - final_rankic
            ),
        }
    )

    info_path = run_dir / "artifacts" / "best_checkpoint_info"
    if info_path.exists():
        try:
            with info_path.open("rb") as f:
                info = pickle.load(f)
            if isinstance(info, dict):
                out["checkpoint_metric"] = info.get("metric", "")
                out["checkpoint_epoch"] = info.get("epoch", "")
                out["checkpoint_score"] = info.get("score", "")
                out["checkpoint_restored"] = info.get("restored", "")
        except Exception:
            pass
    return out


def base_long_trainer(seed: int) -> Dict[str, Any]:
    return {
        "lr": 5e-5,
        "n_epochs": 40,
        "batch_size": 300,
        "grad_accum_steps": 1,
        "precision": "amp_fp16",
        "use_warmup": True,
        "warmup_ratio": 0.05,
        "warmup_steps": 0,
        "early_stop": 0,
        "min_delta": 0.001,
        "train_stop_threshold": None,
        "min_epochs": 40,
        "consecutive_k": 999999,
        "seed": int(seed),
        "use_tqdm": False,
        "strict_valid_data_key": True,
        "market_state_path": "data/market_state_csi300.pkl",
        "market_state_shift": 0,
        "market_state_strict": True,
        "checkpoint_metric": "valid_rank_ic",
        "checkpoint_mode": "max",
        "checkpoint_min_delta": 0.0,
    }


def run_single(
    *,
    root: Path,
    python_exe: str,
    workflow: Path,
    out_dir: Path,
    setting: str,
    model_overrides: Dict[str, Any],
    trainer_overrides: Dict[str, Any],
    data_overrides: Optional[Dict[str, Any]] = None,
    port_overrides: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    continue_finished: bool = True,
) -> Dict[str, Any]:
    setting_dir = out_dir / setting
    setting_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = setting_dir / "manifest.json"
    existing = read_manifest(manifest_path)
    if continue_finished and existing and existing.get("returncode") == 0 and existing.get("diagnostic_matrix"):
        diag_path = str(existing.get("diagnostic_matrix", ""))
        if Path(diag_path).exists():
            existing["metrics"] = load_diag_metrics(diag_path)
            existing["curve_summary"] = load_train_curve_summary(diag_path)
            log(f"skip completed {setting}")
            return existing

    cmd = [python_exe, "-u", str(workflow)]
    manifest: Dict[str, Any] = {
        "setting": setting,
        "cmd": cmd,
        "model_overrides": model_overrides,
        "trainer_overrides": trainer_overrides,
        "data_overrides": data_overrides or {},
        "port_overrides": port_overrides or {},
        "meta": meta or {},
        "stdout": str(setting_dir / "stdout.log"),
        "stderr": str(setting_dir / "stderr.log"),
        "returncode": None,
        "started_at": now_str(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["QIB_RUN_SETTING"] = setting
    env["QIB_MODEL_OVERRIDES_JSON"] = json.dumps(model_overrides, ensure_ascii=False)
    env["QIB_TRAINER_OVERRIDES_JSON"] = json.dumps(trainer_overrides, ensure_ascii=False)
    if data_overrides:
        env["QIB_DATA_OVERRIDES_JSON"] = json.dumps(data_overrides, ensure_ascii=False)
    if port_overrides:
        env["QIB_PORT_OVERRIDES_JSON"] = json.dumps(port_overrides, ensure_ascii=False)

    log(f"start {setting}")
    started_at = time.time()
    with open(setting_dir / "stdout.log", "wb") as out_f, open(setting_dir / "stderr.log", "wb") as err_f:
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=out_f, stderr=err_f)

    manifest["returncode"] = int(proc.returncode)
    manifest["finished_at"] = now_str()
    if proc.returncode == 0:
        manifest["diagnostic_matrix"] = find_latest_diagnostic(root, started_at=started_at)
        manifest["metrics"] = load_diag_metrics(str(manifest.get("diagnostic_matrix", "")))
        manifest["curve_summary"] = load_train_curve_summary(str(manifest.get("diagnostic_matrix", "")))
        log(f"completed {setting}")
    else:
        log(f"failed {setting}: returncode={proc.returncode}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def metric_value(manifest: Dict[str, Any], key: str) -> Optional[float]:
    metrics = manifest.get("metrics", {}) or {}
    return finite_float(metrics.get(key))


def write_suite_summary(out_dir: Path, manifests: List[Dict[str, Any]], *, title: str) -> None:
    rows: List[Dict[str, Any]] = []
    for m in manifests:
        metrics = m.get("metrics", {}) or {}
        curve = m.get("curve_summary", {}) or {}
        meta = m.get("meta", {}) or {}
        row: Dict[str, Any] = {
            "setting": m.get("setting", ""),
            "seed": (m.get("trainer_overrides", {}) or {}).get("seed", ""),
            "returncode": m.get("returncode"),
            "diagnostic_matrix": m.get("diagnostic_matrix", ""),
            "d_model": meta.get("d_model", ""),
            "n_heads": meta.get("n_heads", ""),
            "d_head": meta.get("d_head", ""),
            "n_layers": meta.get("n_layers", ""),
            "d_ff": meta.get("d_ff", ""),
            "dropout": meta.get("dropout", ""),
            "params_plan": meta.get("params_plan", ""),
            "epochs_recorded": curve.get("epochs_recorded", ""),
            "best_valid_rank_ic": curve.get("best_valid_rank_ic", ""),
            "best_valid_rank_ic_epoch": curve.get("best_valid_rank_ic_epoch", ""),
            "final_valid_rank_ic": curve.get("final_valid_rank_ic", ""),
            "best_minus_final_valid_rank_ic": curve.get("best_minus_final_valid_rank_ic", ""),
            "checkpoint_metric": curve.get("checkpoint_metric", ""),
            "checkpoint_epoch": curve.get("checkpoint_epoch", ""),
            "checkpoint_score": curve.get("checkpoint_score", ""),
            "checkpoint_restored": curve.get("checkpoint_restored", ""),
        }
        for key in SUMMARY_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)

    csv_path = out_dir / "suite_summary.csv"
    fieldnames = [
        "setting",
        "seed",
        "returncode",
        "d_model",
        "n_heads",
        "d_head",
        "n_layers",
        "d_ff",
        "dropout",
        "params_plan",
        "epochs_recorded",
        "best_valid_rank_ic",
        "best_valid_rank_ic_epoch",
        "final_valid_rank_ic",
        "best_minus_final_valid_rank_ic",
        "checkpoint_metric",
        "checkpoint_epoch",
        "checkpoint_score",
        "checkpoint_restored",
        *SUMMARY_KEYS,
        "diagnostic_matrix",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    completed = [r for r in rows if r.get("returncode") == 0]
    best = None
    for r in completed:
        val = finite_float(r.get("performance/daily_rank_ic_mean"))
        if val is not None and (best is None or val > best[0]):
            best = (val, r.get("setting", ""))

    md = [f"# {title}", ""]
    if best is not None:
        md.append(f"- Best test diagnostic RankIC: `{best[1]}` ({best[0]:.6f})")
    md.append(f"- CSV: `{csv_path.name}`")
    md.append("")
    md.append("| setting | seed | rc | best valid RankIC | best epoch | test RankIC | RankICIR | AnnRet(cost) | IR(cost) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            "| {setting} | {seed} | {rc} | {best_valid} | {best_epoch} | {rankic} | {rankicir} | {ann} | {ir} |".format(
                setting=r.get("setting", ""),
                seed=r.get("seed", ""),
                rc=r.get("returncode", ""),
                best_valid=r.get("best_valid_rank_ic", ""),
                best_epoch=r.get("best_valid_rank_ic_epoch", ""),
                rankic=r.get("performance/daily_rank_ic_mean", ""),
                rankicir=r.get("performance/rank_icir", ""),
                ann=r.get("portfolio/annualized_return_with_cost", ""),
                ir=r.get("portfolio/information_ratio_with_cost", ""),
            )
        )
    (out_dir / "suite_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "suite_manifest.json").write_text(json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8")


def elimination_reasons(manifest: Dict[str, Any]) -> List[str]:
    metrics = manifest.get("metrics", {}) or {}
    checks = [
        ("optimization/train_grad_nonfinite_rate_last", "grad_nonfinite_rate > 0.005", lambda x: x > 0.005),
        ("optimization/train_grad_skipped_rate_last", "grad_skipped_rate > 0.005", lambda x: x > 0.005),
        ("optimization/post_peak_decay", "post_peak_decay > 0.010", lambda x: x > 0.010),
        ("portfolio/max_drawdown_with_cost", "max_drawdown_with_cost < -0.12", lambda x: x < -0.12),
        ("portfolio/turnover", "turnover > 0.45", lambda x: x > 0.45),
        ("router/collapse_ratio", "router_collapse_ratio > 0.05", lambda x: x > 0.05),
    ]
    reasons = []
    for key, reason, pred in checks:
        val = finite_float(metrics.get(key))
        if val is not None and pred(val):
            reasons.append(reason)
    return reasons


def stage1_selectable(manifest_by_setting: Dict[str, Dict[str, Any]], setting: str) -> Tuple[bool, List[str]]:
    m = manifest_by_setting.get(setting, {})
    reasons = elimination_reasons(m)
    if reasons:
        return False, reasons

    if setting == "cap07_d96_h6_l2_ff384_do015":
        cap07 = manifest_by_setting.get(setting, {})
        cap05 = manifest_by_setting.get("cap05_d96_h4_l2_ff384_do015", {})
        r07 = metric_value(cap07, "performance/daily_rank_ic_mean")
        r05 = metric_value(cap05, "performance/daily_rank_ic_mean")
        ir07 = metric_value(cap07, "performance/rank_icir")
        ir05 = metric_value(cap05, "performance/rank_icir")
        if r05 is not None and ir05 is not None and r07 is not None and ir07 is not None:
            if not (r07 >= r05 + 0.005 and ir07 >= ir05):
                return False, ["H=6 did not beat cap05 H=4 by >=0.005 RankIC without RankICIR drop"]

    if setting == "cap08_d96_h4_l3_ff384_do020":
        l2_vals = []
        for k, mm in manifest_by_setting.items():
            meta = mm.get("meta", {}) or {}
            if k != setting and meta.get("n_layers") == 2:
                val = metric_value(mm, "performance/daily_rank_ic_mean")
                if val is not None:
                    l2_vals.append(val)
        best_l2 = max(l2_vals) if l2_vals else None
        r08 = metric_value(m, "performance/daily_rank_ic_mean")
        decay = metric_value(m, "optimization/post_peak_decay")
        if best_l2 is not None and r08 is not None:
            if not (r08 >= best_l2 + 0.006 and (decay is None or decay <= 0.005)):
                return False, ["L3 did not beat best L2 by >=0.006 with post_peak_decay <=0.005"]
    return True, []


def choose_top2_stage1(stage1: List[Dict[str, Any]]) -> List[str]:
    by_setting = {m.get("setting", ""): m for m in stage1}
    candidates = []
    for m in stage1:
        setting = str(m.get("setting", ""))
        if setting == "cap00_d64_h4_l2_ff128_do010" or m.get("returncode") != 0:
            continue
        ok, _ = stage1_selectable(by_setting, setting)
        rankic = metric_value(m, "performance/daily_rank_ic_mean")
        if ok and rankic is not None:
            meta = m.get("meta", {}) or {}
            candidates.append(
                {
                    "setting": setting,
                    "rankic": rankic,
                    "params": int(meta.get("params_plan", 10**12) or 10**12),
                    "heads": int(meta.get("n_heads", 10**6) or 10**6),
                    "dropout": float(meta.get("dropout", 10**6) or 10**6),
                }
            )

    selected: List[str] = []
    remaining = candidates[:]
    while remaining and len(selected) < 2:
        top_rankic = max(x["rankic"] for x in remaining)
        close = [x for x in remaining if top_rankic - x["rankic"] < 0.005]
        chosen = sorted(close, key=lambda x: (x["params"], x["heads"], x["dropout"], -x["rankic"]))[0]
        selected.append(chosen["setting"])
        remaining = [x for x in remaining if x["setting"] != chosen["setting"]]
    return selected


def write_capacity_decision(out_dir: Path, stage1: List[Dict[str, Any]], top2: List[str]) -> None:
    by_setting = {m.get("setting", ""): m for m in stage1}
    rows = []
    for m in stage1:
        setting = str(m.get("setting", ""))
        ok, reasons = stage1_selectable(by_setting, setting)
        rows.append(
            {
                "setting": setting,
                "returncode": m.get("returncode"),
                "selectable": ok,
                "selected_for_stage2": setting in top2 or setting == "cap00_d64_h4_l2_ff128_do010",
                "reasons": "; ".join(reasons),
                "rankic": metric_value(m, "performance/daily_rank_ic_mean"),
                "rankicir": metric_value(m, "performance/rank_icir"),
                "post_peak_decay": metric_value(m, "optimization/post_peak_decay"),
                "drawdown": metric_value(m, "portfolio/max_drawdown_with_cost"),
                "turnover": metric_value(m, "portfolio/turnover"),
                "grad_nonfinite_rate": metric_value(m, "optimization/train_grad_nonfinite_rate_last"),
                "grad_skipped_rate": metric_value(m, "optimization/train_grad_skipped_rate_last"),
            }
        )
    path = out_dir / "stage1_selection.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["setting"])
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Stage 1 Selection",
        "",
        f"- Stage 2 baseline: `cap00_d64_h4_l2_ff128_do010`",
        f"- Stage 2 top2 non-baseline: `{', '.join(top2) if top2 else 'none'}`",
        f"- CSV: `{path.name}`",
        "",
    ]
    (out_dir / "stage1_selection.md").write_text("\n".join(md), encoding="utf-8")


def aggregate_stage2(out_dir: Path, manifests: Iterable[Dict[str, Any]], selected_top2: List[str]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for m in manifests:
        if m.get("returncode") != 0:
            continue
        grouped.setdefault(str(m.get("setting", "")), []).append(m)

    baseline_key = "cap00_d64_h4_l2_ff128_do010"
    baseline = grouped.get(baseline_key, [])
    base_rankics = [metric_value(m, "performance/daily_rank_ic_mean") for m in baseline]
    base_irs = [metric_value(m, "performance/rank_icir") for m in baseline]
    base_grad_nf = [metric_value(m, "optimization/train_grad_nonfinite_rate_last") for m in baseline]
    base_grad_skip = [metric_value(m, "optimization/train_grad_skipped_rate_last") for m in baseline]

    def avg(vals: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in vals if x is not None]
        return sum(xs) / len(xs) if xs else None

    base_rankic_avg = avg(base_rankics)
    base_ir_avg = avg(base_irs)
    base_nf_avg = avg(base_grad_nf)
    base_skip_avg = avg(base_grad_skip)

    rows = []
    for setting, runs in grouped.items():
        rankics = [metric_value(m, "performance/daily_rank_ic_mean") for m in runs]
        irs = [metric_value(m, "performance/rank_icir") for m in runs]
        nfs = [metric_value(m, "optimization/train_grad_nonfinite_rate_last") for m in runs]
        skips = [metric_value(m, "optimization/train_grad_skipped_rate_last") for m in runs]
        rankic_avg = avg(rankics)
        ir_avg = avg(irs)
        nf_avg = avg(nfs)
        skip_avg = avg(skips)
        seeds_above_baseline = ""
        pass_stage2 = ""
        if setting != baseline_key and base_rankic_avg is not None and rankic_avg is not None:
            seeds_above_baseline = sum(1 for x in rankics if x is not None and x > base_rankic_avg)
            pass_stage2 = (
                rankic_avg >= base_rankic_avg + 0.006
                and seeds_above_baseline >= 2
                and (base_ir_avg is None or ir_avg is not None and ir_avg >= base_ir_avg - 0.03)
                and (base_nf_avg is None or nf_avg is not None and nf_avg <= base_nf_avg + 1e-12)
                and (base_skip_avg is None or skip_avg is not None and skip_avg <= base_skip_avg + 1e-12)
            )
        rows.append(
            {
                "setting": setting,
                "seeds": ",".join(str((m.get("trainer_overrides", {}) or {}).get("seed", "")) for m in runs),
                "n_runs": len(runs),
                "avg_rankic": rankic_avg,
                "avg_rankicir": ir_avg,
                "avg_grad_nonfinite_rate": nf_avg,
                "avg_grad_skipped_rate": skip_avg,
                "baseline_avg_rankic": base_rankic_avg,
                "rankic_minus_baseline": "" if base_rankic_avg is None or rankic_avg is None else rankic_avg - base_rankic_avg,
                "seeds_above_baseline_avg": seeds_above_baseline,
                "pass_stage2_rules": pass_stage2,
                "selected_top2_after_stage1": setting in selected_top2,
            }
        )

    rows = sorted(rows, key=lambda r: finite_float(r.get("avg_rankic")) or -999.0, reverse=True)
    path = out_dir / "stage2_multiseed_aggregate.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["setting"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "stage2_multiseed_aggregate.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def stage3_should_run(stage1: List[Dict[str, Any]]) -> Tuple[bool, str]:
    target = {
        "cap05_d96_h4_l2_ff384_do015",
        "cap06_d96_h4_l2_ff384_do020",
        "cap07_d96_h6_l2_ff384_do015",
        "cap08_d96_h4_l3_ff384_do020",
    }
    eligible = []
    for m in stage1:
        setting = str(m.get("setting", ""))
        if setting not in target or m.get("returncode") != 0:
            continue
        if elimination_reasons(m):
            continue
        decay = metric_value(m, "optimization/post_peak_decay")
        if decay is None or decay <= 0.005:
            eligible.append(setting)
    if eligible:
        return True, "high-capacity Stage1 configs show no clear overfit: " + ", ".join(eligible)
    return False, "no cap05-cap08 setting passed the no-clear-overfit gate"


def wait_for_current_matrix(current_out_dir: Path, poll_seconds: int) -> None:
    if not current_out_dir:
        return
    proc_info = load_json(current_out_dir / "_runner_process.json")
    pid = int(proc_info.get("pid", 0) or 0)
    if pid and pid_exists(pid):
        log(f"waiting for current matrix runner pid={pid}: {current_out_dir}")
        while pid_exists(pid):
            suite = current_out_dir / "suite_manifest.json"
            completed = []
            for manifest_path in current_out_dir.glob("*/manifest.json"):
                m = read_manifest(manifest_path)
                if m and m.get("returncode") == 0:
                    completed.append(manifest_path.parent.name)
            log(f"current matrix still running; completed={completed}")
            time.sleep(max(30, int(poll_seconds)))
    else:
        log(f"current matrix runner is not active: {current_out_dir}")

    for _ in range(20):
        if (current_out_dir / "suite_manifest.json").exists():
            break
        time.sleep(15)
    if (current_out_dir / "suite_manifest.json").exists():
        log("current matrix suite manifest is present")
    else:
        log("current matrix suite manifest is still missing; continuing to next phase")


def run_matrix_40(root: Path, python_exe: str, workflow: Path, out_dir: Path) -> List[Dict[str, Any]]:
    manifests = []
    for setting, conf in MATRIX_SETTINGS.items():
        model = json.loads(json.dumps(conf["model"]))
        trainer = base_long_trainer(seed=42)
        manifests.append(
            run_single(
                root=root,
                python_exe=python_exe,
                workflow=workflow,
                out_dir=out_dir,
                setting=setting,
                model_overrides=model,
                trainer_overrides=trainer,
            )
        )
        write_suite_summary(out_dir, manifests, title="40 Epoch Matrix Seed 42")
    return manifests


def cap_model_overrides(spec: Dict[str, Any]) -> Dict[str, Any]:
    model = json.loads(json.dumps(FULL135_FIXED_MODEL))
    for key in ["d_model", "n_heads", "n_layers", "d_ff", "dropout"]:
        model[key] = spec[key]
    return model


def cap_meta(spec: Dict[str, Any], *, seed: int, stage: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "seed": seed,
        "d_model": spec["d_model"],
        "n_heads": spec["n_heads"],
        "d_head": spec["d_head"],
        "n_layers": spec["n_layers"],
        "d_ff": spec["d_ff"],
        "dropout": spec["dropout"],
        "params_plan": spec["params_plan"],
    }


def run_capacity_plan(root: Path, python_exe: str, workflow: Path, out_dir: Path) -> None:
    stage1_dir = out_dir / "stage1_seed42"
    stage1: List[Dict[str, Any]] = []
    spec_by_setting = {s["setting"]: s for s in CAPACITY_STAGE1 + CAPACITY_STAGE3}

    for spec in CAPACITY_STAGE1:
        stage1.append(
            run_single(
                root=root,
                python_exe=python_exe,
                workflow=workflow,
                out_dir=stage1_dir,
                setting=spec["setting"],
                model_overrides=cap_model_overrides(spec),
                trainer_overrides=base_long_trainer(seed=42),
                meta=cap_meta(spec, seed=42, stage="stage1"),
            )
        )
        write_suite_summary(stage1_dir, stage1, title="Capacity Stage 1 Seed 42")

    top2 = choose_top2_stage1(stage1)
    write_capacity_decision(stage1_dir, stage1, top2)
    log(f"Stage 1 top2 selected for multi-seed: {top2}")

    stage2_dir = out_dir / "stage2_seed43_44"
    stage2: List[Dict[str, Any]] = []
    stage2_settings = ["cap00_d64_h4_l2_ff128_do010", *top2]
    for seed in [43, 44]:
        for setting in stage2_settings:
            spec = spec_by_setting[setting]
            stage2.append(
                run_single(
                    root=root,
                    python_exe=python_exe,
                    workflow=workflow,
                    out_dir=stage2_dir,
                    setting=f"{setting}_seed{seed}",
                    model_overrides=cap_model_overrides(spec),
                    trainer_overrides=base_long_trainer(seed=seed),
                    meta=cap_meta(spec, seed=seed, stage="stage2"),
                )
            )
            write_suite_summary(stage2_dir, stage2, title="Capacity Stage 2 Seed 43/44")

    combined_for_agg = []
    for m in stage1:
        if m.get("setting") in stage2_settings:
            combined_for_agg.append(m)
    for m in stage2:
        raw = str(m.get("setting", ""))
        for base_name in stage2_settings:
            if raw.startswith(base_name + "_seed"):
                mm = dict(m)
                mm["setting"] = base_name
                combined_for_agg.append(mm)
                break
    aggregate_stage2(out_dir, combined_for_agg, top2)

    run_stage3, reason = stage3_should_run(stage1)
    (out_dir / "stage3_decision.json").write_text(
        json.dumps({"run_stage3": run_stage3, "reason": reason}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Stage 3 decision: run={run_stage3}; {reason}")
    if run_stage3:
        stage3_dir = out_dir / "stage3_aggressive_seed42"
        stage3: List[Dict[str, Any]] = []
        for spec in CAPACITY_STAGE3:
            stage3.append(
                run_single(
                    root=root,
                    python_exe=python_exe,
                    workflow=workflow,
                    out_dir=stage3_dir,
                    setting=spec["setting"],
                    model_overrides=cap_model_overrides(spec),
                    trainer_overrides=base_long_trainer(seed=42),
                    meta=cap_meta(spec, seed=42, stage="stage3"),
                )
            )
            write_suite_summary(stage3_dir, stage3, title="Capacity Stage 3 Aggressive Seed 42")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the four-day RST-MoE experiment plan.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workflow", default="work_flow.py")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--current-out-dir", default="diagnostic_runs/seed42_matrix_gradfix_20260430_2105")
    ap.add_argument("--poll-seconds", type=int, default=300)
    args = ap.parse_args()

    root = Path.cwd()
    workflow = Path(args.workflow)
    if not workflow.is_absolute():
        workflow = root / workflow
    if not workflow.exists():
        raise SystemExit(f"workflow not found: {workflow}")

    prevent_system_sleep()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else root / "diagnostic_runs" / f"four_day_plan_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_manifest = {
        "started_at": now_str(),
        "python": args.python,
        "workflow": str(workflow),
        "out_dir": str(out_dir),
        "current_out_dir": args.current_out_dir,
        "protocol": {
            "n_epochs": 40,
            "early_stop": 0,
            "train_stop_threshold": None,
            "checkpoint_metric": "valid_rank_ic",
            "seed_matrix": 42,
        },
    }
    (out_dir / "orchestrator_manifest.json").write_text(
        json.dumps(plan_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        wait_for_current_matrix(Path(args.current_out_dir), args.poll_seconds)

        matrix_dir = out_dir / "matrix_40epoch_seed42"
        log("phase start: 40 epoch diagnostic matrix")
        run_matrix_40(root, args.python, workflow, matrix_dir)
        log("phase done: 40 epoch diagnostic matrix")

        capacity_dir = out_dir / "capacity_full135_40epoch"
        log("phase start: capacity/head/dropout plan")
        run_capacity_plan(root, args.python, workflow, capacity_dir)
        log("phase done: capacity/head/dropout plan")

        plan_manifest["finished_at"] = now_str()
        plan_manifest["returncode"] = 0
        (out_dir / "orchestrator_manifest.json").write_text(
            json.dumps(plan_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        plan_manifest["failed_at"] = now_str()
        plan_manifest["returncode"] = 1
        plan_manifest["error"] = repr(exc)
        (out_dir / "orchestrator_manifest.json").write_text(
            json.dumps(plan_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
