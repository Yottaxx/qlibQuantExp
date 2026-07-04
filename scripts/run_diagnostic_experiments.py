#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the seed-42 minimum diagnostic experiment matrix.

The runner keeps `work_flow.py` immutable by passing JSON overrides through
environment variables consumed by the workflow:

    QIB_RUN_SETTING
    QIB_MODEL_OVERRIDES_JSON
    QIB_TRAINER_OVERRIDES_JSON

Default settings:
    full_135, base, time_only, film_only, no_layer_summary, fixed_router_05
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


SETTINGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "full_135": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": 1.35,
        },
    },
    "base": {
        "model": {
            "use_regime_time_embedding": False,
            "use_regime_factor_gate": False,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": 1.35,
        },
    },
    "time_only": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": False,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": 1.35,
        },
    },
    "film_only": {
        "model": {
            "use_regime_time_embedding": False,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": 1.35,
        },
    },
    "no_layer_summary": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": False,
            "router_mode": "learned",
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": 1.35,
        },
    },
    "fixed_router_05": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "fixed_05",
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": 1.35,
        },
    },
    "tau_scale_01": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 0.1,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "tau_scale_10": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 1.0,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "tau_scale_20": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 2.0,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "tau_scale_50": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 5.0,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "tau_scale_02": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 0.2,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "pool_r2_pure_attn": {  # pool-forensics R2: remove the 0.3 mean-blend bypass
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "pooling_alpha": 1.0,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "pool_r1_multihead": {  # pool-forensics R1: 4-head attention pool (break single-query symmetry)
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "pool_n_heads": 4,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "tau_scale_05": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 0.5,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "tau_scale_08": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 0.8,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
    "tau_scale_15": {
        "model": {
            "use_regime_time_embedding": True,
            "use_regime_factor_gate": True,
            "router_use_layer_summary": True,
            "router_mode": "learned",
            "time_tau_mlp_out_scale": 1.5,
        },
        "trainer": {
            "seed": 42,
            "train_stop_threshold": None,
        },
    },
}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def parse_json_arg(raw: str, *, name: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise SystemExit(f"{name} must be valid JSON: {e}")
    if not isinstance(obj, dict):
        raise SystemExit(f"{name} must decode to a JSON object")
    return obj


def smoke_data_overrides() -> Dict[str, Any]:
    """Small date window for fast metric-completeness smoke checks."""
    return {
        "kwargs": {
            "handler": {
                "kwargs": {
                    "start_time": "2019-01-01",
                    "end_time": "2020-10-15",
                    "fit_start_time": "2019-01-01",
                    "fit_end_time": "2019-03-31",
                }
            },
            "segments": {
                "train": ["2019-01-01", "2019-03-31"],
                "valid": ["2020-07-01", "2020-09-30"],
                "test": ["2020-07-01", "2020-09-30"],
            },
        }
    }


def resolve_settings(raw: str) -> List[str]:
    if not raw or raw.strip().lower() == "all":
        return list(SETTINGS.keys())
    out = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in out if x not in SETTINGS]
    if unknown:
        raise SystemExit(f"Unknown setting(s): {unknown}. Available: {list(SETTINGS)}")
    return out


def find_latest_diagnostic(root: Path, *, started_at: float) -> str:
    mlruns = root / "mlruns"
    if not mlruns.exists():
        return ""
    best_path = ""
    best_mtime = started_at
    for p in mlruns.rglob("diagnostic_matrix.csv"):
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt >= best_mtime:
            best_mtime = mt
            best_path = str(p)
    return best_path


def load_diag_metrics(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('group', '')}/{row.get('metric', '')}"
            try:
                out[key] = float(row.get("value", "nan"))
            except Exception:
                continue
    return out


def write_suite_summary(out_dir: Path, manifests: List[Dict[str, Any]]) -> None:
    keys = [
        "performance/daily_rank_ic_mean",
        "performance/rank_icir",
        "performance/daily_ic_mean",
        "performance/icir",
        "portfolio/annualized_return_with_cost",
        "portfolio/information_ratio_with_cost",
        "portfolio/max_drawdown_with_cost",
        "portfolio/turnover",
        "optimization/post_peak_decay",
        "optimization/train_grad_norm_last",
        "optimization/train_grad_nonfinite_rate_last",
        "optimization/train_grad_skipped_rate_last",
        "optimization/train_optimizer_steps_last",
        "regime/pc1_tail_2x2_rank_ic_spread",
        "router/time_ratio_mean",
        "time_embedding/tau_range_utilization",
        "factor_film/film_pool_top10_overlap",
        "attention_pooling/factor_entropy_norm_mean",
    ]
    rows = []
    for m in manifests:
        metrics = load_diag_metrics(str(m.get("diagnostic_matrix", "")))
        row = {
            "setting": m.get("setting", ""),
            "returncode": m.get("returncode"),
            "diagnostic_matrix": m.get("diagnostic_matrix", ""),
        }
        for k in keys:
            row[k] = metrics.get(k, "")
        rows.append(row)

    csv_path = out_dir / "suite_diagnostic_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["setting", "returncode", "diagnostic_matrix", *keys]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    completed = [r for r in rows if r.get("returncode") == 0]
    best = None
    for r in completed:
        try:
            val = float(r.get("performance/daily_rank_ic_mean", "nan"))
        except Exception:
            continue
        if best is None or val > best[0]:
            best = (val, r.get("setting", ""))

    lines = ["# Diagnostic Suite Summary", ""]
    if best is not None:
        lines.append(f"- Best setting by daily RankIC mean: `{best[1]}` ({best[0]:.6f})")
    else:
        lines.append("- Best setting by daily RankIC mean: unavailable")
    lines.append(f"- Table: `{csv_path.name}`")
    lines.append("")
    lines.append("| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        hints = []
        try:
            if float(r.get("time_embedding/tau_range_utilization", "nan")) < 0.01:
                hints.append("tau under-adaptive")
        except Exception:
            pass
        try:
            if float(r.get("attention_pooling/factor_entropy_norm_mean", "nan")) > 0.98:
                hints.append("factor attention uniform")
        except Exception:
            pass
        try:
            if float(r.get("regime/pc1_tail_2x2_rank_ic_spread", "nan")) > 0.03:
                hints.append("regime-specific spread")
        except Exception:
            pass
        try:
            if float(r.get("optimization/post_peak_decay", "nan")) > 0.005:
                hints.append("post-peak decay")
        except Exception:
            pass
        try:
            if float(r.get("optimization/train_grad_nonfinite_rate_last", "nan")) > 0.0:
                hints.append("AMP grad overflow")
        except Exception:
            pass
        lines.append(
            "| {setting} | {returncode} | {rankic} | {rankicir} | {ann} | {ir} | {hints} |".format(
                setting=r.get("setting", ""),
                returncode=r.get("returncode"),
                rankic=r.get("performance/daily_rank_ic_mean", ""),
                rankicir=r.get("performance/rank_icir", ""),
                ann=r.get("portfolio/annualized_return_with_cost", ""),
                ir=r.get("portfolio/information_ratio_with_cost", ""),
                hints=", ".join(hints) if hints else "",
            )
        )
    (out_dir / "suite_diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run seed-42 RST-MoE diagnostic experiments.")
    ap.add_argument("--settings", default="all", help="Comma-separated setting names or 'all'.")
    ap.add_argument("--workflow", default="work_flow.py", help="Workflow script path.")
    ap.add_argument("--python", default=sys.executable, help="Python executable.")
    ap.add_argument("--out-dir", default="", help="Log directory. Default: diagnostic_runs/<timestamp>.")
    ap.add_argument("--smoke", action="store_true", help="Run a one-epoch smoke check for metric completeness.")
    ap.add_argument("--print-only", action="store_true", help="Print commands and manifests without executing.")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue remaining settings after a failure.")
    ap.add_argument("--model-json", default="", help="Extra model_config JSON merged into every setting.")
    ap.add_argument("--trainer-json", default="", help="Extra trainer_config JSON merged into every setting.")
    ap.add_argument("--data-json", default="", help="Extra data_conf JSON merged into every setting.")
    ap.add_argument("--port-json", default="", help="Extra port_conf JSON merged into every setting.")
    args = ap.parse_args()

    root = Path.cwd()
    workflow = Path(args.workflow)
    if not workflow.is_absolute():
        workflow = root / workflow
    if not workflow.exists():
        raise SystemExit(f"Workflow not found: {workflow}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else root / "diagnostic_runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_model = parse_json_arg(args.model_json, name="--model-json")
    extra_trainer = parse_json_arg(args.trainer_json, name="--trainer-json")
    extra_data = parse_json_arg(args.data_json, name="--data-json")
    extra_port = parse_json_arg(args.port_json, name="--port-json")
    settings = resolve_settings(args.settings)

    suite_manifest: List[Dict[str, Any]] = []
    for setting in settings:
        base_conf = SETTINGS[setting]
        model_overrides = json.loads(json.dumps(base_conf.get("model", {})))
        trainer_overrides = json.loads(json.dumps(base_conf.get("trainer", {})))
        data_overrides: Dict[str, Any] = {}
        port_overrides: Dict[str, Any] = {}
        deep_update(model_overrides, extra_model)
        deep_update(trainer_overrides, extra_trainer)
        deep_update(data_overrides, extra_data)
        deep_update(port_overrides, extra_port)
        # Long matrix runs should stay grep-friendly and avoid huge tqdm stderr logs.
        trainer_overrides.setdefault("use_tqdm", False)
        if args.smoke:
            deep_update(data_overrides, smoke_data_overrides())
            deep_update(
                trainer_overrides,
                {
                    "n_epochs": 1,
                    "min_epochs": 1,
                    "consecutive_k": 1,
                },
            )

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["QIB_RUN_SETTING"] = setting
        env["QIB_MODEL_OVERRIDES_JSON"] = json.dumps(model_overrides, ensure_ascii=False)
        env["QIB_TRAINER_OVERRIDES_JSON"] = json.dumps(trainer_overrides, ensure_ascii=False)
        if data_overrides:
            env["QIB_DATA_OVERRIDES_JSON"] = json.dumps(data_overrides, ensure_ascii=False)
        if port_overrides:
            env["QIB_PORT_OVERRIDES_JSON"] = json.dumps(port_overrides, ensure_ascii=False)

        cmd = [args.python, "-u", str(workflow)]
        setting_dir = out_dir / setting
        setting_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "setting": setting,
            "cmd": cmd,
            "model_overrides": model_overrides,
            "trainer_overrides": trainer_overrides,
            "data_overrides": data_overrides,
            "port_overrides": port_overrides,
            "stdout": str(setting_dir / "stdout.log"),
            "stderr": str(setting_dir / "stderr.log"),
            "returncode": None,
        }
        (setting_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n=== Running {setting} ===")
        print(" ".join(cmd))
        print(f"logs: {setting_dir}")

        if args.print_only:
            suite_manifest.append(manifest)
            continue

        started_at = time.time()
        with open(setting_dir / "stdout.log", "wb") as out_f, open(setting_dir / "stderr.log", "wb") as err_f:
            proc = subprocess.run(cmd, cwd=root, env=env, stdout=out_f, stderr=err_f)

        manifest["returncode"] = int(proc.returncode)
        if proc.returncode == 0:
            manifest["diagnostic_matrix"] = find_latest_diagnostic(root, started_at=started_at)
        (setting_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        suite_manifest.append(manifest)

        if proc.returncode != 0:
            print(f"FAILED {setting}: returncode={proc.returncode}")
            if not args.continue_on_error:
                break
        else:
            print(f"completed {setting}")

    (out_dir / "suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_suite_summary(out_dir, suite_manifest)

    failed = [m for m in suite_manifest if m.get("returncode") not in (0, None)]
    if failed:
        print(f"\nFailed settings: {[m['setting'] for m in failed]}")
        return 1
    print(f"\nSuite manifest: {out_dir / 'suite_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
