from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
VARIANT_SCRIPT = PROJECT_ROOT / "scripts" / "run_workflow_market_state_variant.py"
QUANT_ENV_PYTHON = Path(r"C:\Users\60585\miniconda3\envs\quantEnv\python.exe")
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

MARKET_STATE_PATH = "artifacts/market_state/daily_market_field_ref_all_SH000300_tri_scope_relative_momentum_v1.pkl"
MARKET_DAY_SUMMARY_PATH = "artifacts/market_state/daily_market_observation_tri_scope_v1.pkl"

STEP = {
    "name": "stocktime_simplepool_mhc",
    "label": "csi300_tri_scope_stocktime_simplepool_mhc",
    "workflow_module": (
        "next_step.hierarchical_state_field.experiment.configs."
        "ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc"
    ),
    "intent": "stock_time_factor_v1 plus mHC-lite temporal mixing and simple fixed-alpha factor pooling.",
}


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _default_python() -> str:
    return str(QUANT_ENV_PYTHON if QUANT_ENV_PYTHON.exists() else Path(sys.executable))


def _quote_arg(arg: str) -> str:
    return '"' + str(arg).replace('"', r"\"") + '"'


def _command_to_text(cmd: list[str]) -> str:
    return " ".join(_quote_arg(part) for part in cmd)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:]) + ("\n" if lines else "")


def _write_latest_tail(run_dir: Path, stdout_path: Path, stderr_path: Path) -> None:
    _write_text(
        run_dir / "latest_tail.txt",
        f"[{STEP['name']}] stdout tail\n"
        f"{'-' * 80}\n"
        f"{_tail(stdout_path)}\n"
        f"[{STEP['name']}] stderr tail\n"
        f"{'-' * 80}\n"
        f"{_tail(stderr_path)}",
    )


def _build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MLFLOW_TRACKING_URI"] = f"file:{MLRUNS_DIR.resolve()}"
    return env


def _build_command(*, python_exe: str, run_tag: str, seed: int, skip_visuals: bool) -> list[str]:
    cmd = [
        python_exe,
        "-u",
        str(VARIANT_SCRIPT),
        "--qlib_kernels",
        "1",
        "--workflow_module",
        STEP["workflow_module"],
        "--market_state_path",
        MARKET_STATE_PATH,
        "--market_day_summary_path",
        MARKET_DAY_SUMMARY_PATH,
        "--label",
        STEP["label"],
        "--experiment_suffix",
        f"{run_tag}_{STEP['name']}",
        "--seed",
        str(seed),
        "--enable_local_counterfactual_diag",
        "1",
        "--enable_expert_advantage_diag",
        "1",
    ]
    if skip_visuals:
        cmd.append("--skip_visuals")
    return cmd


def _run_step(*, cmd: list[str], run_dir: Path, status: dict[str, Any]) -> int:
    stdout_path = run_dir / f"{STEP['name']}.stdout.log"
    stderr_path = run_dir / f"{STEP['name']}.stderr.log"
    step_status = status["steps"][STEP["name"]]
    step_status["status"] = "running"
    step_status["started_at"] = _now()
    step_status["runner_pid"] = os.getpid()
    step_status["stdout_log"] = str(stdout_path)
    step_status["stderr_log"] = str(stderr_path)
    step_status["mlflow_tracking_uri"] = f"file:{MLRUNS_DIR.resolve()}"
    _write_json(run_dir / "status.json", status)

    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as stderr:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=_build_child_env(),
            )
            step_status["pid"] = proc.pid
            step_status["pid_alive"] = True
            _write_json(run_dir / "status.json", status)

            while True:
                returncode = proc.poll()
                step_status["heartbeat_at"] = _now()
                step_status["pid_alive"] = returncode is None
                step_status["stdout_bytes"] = stdout_path.stat().st_size if stdout_path.exists() else 0
                step_status["stderr_bytes"] = stderr_path.stat().st_size if stderr_path.exists() else 0
                _write_json(run_dir / "status.json", status)
                _write_latest_tail(run_dir, stdout_path, stderr_path)
                if returncode is not None:
                    break
                time.sleep(60)
        except BaseException:
            stderr.write("\n[runner_exception]\n")
            stderr.write(traceback.format_exc())
            stderr.flush()
            returncode = 1

    step_status["status"] = "completed" if returncode == 0 else "failed"
    step_status["finished_at"] = _now()
    step_status["returncode"] = returncode
    step_status["pid_alive"] = False
    _write_json(run_dir / "status.json", status)
    _write_latest_tail(run_dir, stdout_path, stderr_path)
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSI300 stock-time simple-pooling mHC-lite probe.")
    parser.add_argument("--python", dest="python_exe", default=_default_python())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-tag",
        default=dt.datetime.now().strftime("hsf_stocktime_simplepool_mhc_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-visuals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = RUNS_DIR / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(
        python_exe=args.python_exe,
        run_tag=args.run_tag,
        seed=args.seed,
        skip_visuals=not bool(args.with_visuals),
    )
    _write_text(run_dir / "commands.txt", f"[{STEP['name']}]\n{_command_to_text(cmd)}\n")

    status: dict[str, Any] = {
        "run_tag": args.run_tag,
        "created_at": _now(),
        "project_root": str(PROJECT_ROOT),
        "variant_script": str(VARIANT_SCRIPT),
        "mlflow_tracking_uri": f"file:{MLRUNS_DIR.resolve()}",
        "python_exe": args.python_exe,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "skip_visuals": not bool(args.with_visuals),
        "policy": "stocktime_simplepool_mhc_probe",
        "baseline": "ref_all_hsf_csi300_tri_scope_resgru_mean_legacy_summary_stockrel",
        "market_state_path": MARKET_STATE_PATH,
        "market_day_summary_path": MARKET_DAY_SUMMARY_PATH,
        "acceptance": {
            "best_rankic_min": 0.0615,
            "portfolio_ir_min": 0.71,
            "max_drawdown_floor": -0.135,
            "rankic_ir_min": 0.36,
        },
        "design": {
            "factor_gate_scale": 1.0,
            "factor_gate_shift_scale": 0.2,
            "router_summary_source": "batch",
            "router_summary_fusion_mode": "legacy_concat_v1",
            "inner_cross_stock_mode": "stock_time_factor_v1",
            "pooling_mode": "simple_static",
            "pooling_alpha": 0.7,
            "pooling_summary_source": "none",
            "temporal_pooling_mode": "gru_mhc_lite_v1",
            "temporal_mhc_mix_init": 0.05,
            "temporal_mhc_mix_max": 0.25,
        },
        "steps": {
            STEP["name"]: {
                "status": "planned",
                "workflow_module": STEP["workflow_module"],
                "label": STEP["label"],
                "intent": STEP["intent"],
                "command": _command_to_text(cmd),
            }
        },
    }
    _write_json(run_dir / "status.json", status)
    _write_text(
        run_dir / "README.md",
        "# Stock-Time Simple Pooling mHC-lite Probe\n\n"
        "- Baseline: `ref_all_hsf_csi300_tri_scope_resgru_mean_legacy_summary_stockrel`.\n"
        "- Deltas: `gru_mhc_lite_v1` temporal mixing and `simple_static` factor pooling.\n"
        "- Pooling path: final-normalized h -> bounded GRU/h_last mix -> fixed-alpha factor pooling over N only.\n"
        "- Scope: CSI300 only, seed 42, fixed MSE loss and fixed tri-scope macro assets.\n",
    )

    print(f"[RunDir] {run_dir}", flush=True)
    print(f"[Commands] {run_dir / 'commands.txt'}", flush=True)
    if args.dry_run:
        status["status"] = "dry_run_completed"
        status["finished_at"] = _now()
        _write_json(run_dir / "status.json", status)
        return 0

    status["status"] = "running"
    status["started_at"] = _now()
    _write_json(run_dir / "status.json", status)

    print(f"[Start] {STEP['name']}", flush=True)
    returncode = _run_step(cmd=cmd, run_dir=run_dir, status=status)
    print(f"[Finish] {STEP['name']} returncode={returncode}", flush=True)

    status["status"] = "completed" if returncode == 0 else "failed"
    status["finished_at"] = _now()
    status["exit_code"] = returncode
    _write_json(run_dir / "status.json", status)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
