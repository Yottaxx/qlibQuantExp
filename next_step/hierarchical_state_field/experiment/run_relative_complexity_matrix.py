from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
VARIANT_SCRIPT = PROJECT_ROOT / "scripts" / "run_workflow_market_state_variant.py"
PRECOMPUTE_SCRIPT = PROJECT_ROOT / "scripts" / "precompute_market_state.py"
QUANT_ENV_PYTHON = Path(r"C:\Users\60585\miniconda3\envs\quantEnv\python.exe")

OBS_PATH = "artifacts/market_state/daily_market_observation_ref_all_pti_v1.pkl"
FIELD_37D = "artifacts/market_state/daily_market_field_ref_all_SH000300.pkl"
FIELD_LEGACY100 = "artifacts/market_state/daily_market_ref_all_SH000300_legacy.pkl"
FIELD_REL = "artifacts/market_state/daily_market_field_ref_all_SH000300_relative_complexity_v1.pkl"


FIRST_ROUND = [
    {
        "name": "hsf_refall37_current",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_e395_trainproto",
        "market_state_path": FIELD_37D,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
    {
        "name": "hsf_legacy100_current",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_legacy100_state",
        "market_state_path": FIELD_LEGACY100,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
    {
        "name": "hsf_relative_complexity_v1",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_macro_plus_day_summary",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
    {
        "name": "hsf_relative_complexity_macro_only",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_macro_only",
        "market_state_path": FIELD_REL,
        "router_summary_source": "none",
        "pooling_summary_source": "none",
    },
    {
        "name": "hsf_relative_complexity_day_only",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_day_summary_only",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
]

SINGLE_RELATIVE = [
    {
        "name": "hsf_relative_complexity_v1",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_macro_plus_day_summary",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
]

SECOND_ROUND = [
    {
        "name": "relative_complexity_global_only",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_global_only",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
    {
        "name": "relative_complexity_local_router_only",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_local_router_only",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
    {
        "name": "relative_complexity_local_film_only",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_local_film_only",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
    {
        "name": "relative_complexity_local_pool_only",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_local_pool_only",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
    {
        "name": "relative_complexity_bounded_sum",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_relative_complexity_bounded_sum",
        "market_state_path": FIELD_REL,
        "router_summary_source": "day_asset",
        "pooling_summary_source": "day_asset",
    },
]


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_python() -> str:
    return str(QUANT_ENV_PYTHON if QUANT_ENV_PYTHON.exists() else Path(sys.executable))


def _quote_arg(arg: str) -> str:
    return '"' + str(arg).replace('"', r'\"') + '"'


def _command_to_text(cmd: list[str]) -> str:
    return " ".join(_quote_arg(x) for x in cmd)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:]) + ("\n" if lines else "")


def _precompute_command(python_exe: str) -> list[str]:
    return [
        python_exe,
        str(PRECOMPUTE_SCRIPT),
        "--data-config-module",
        "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300",
        "--instruments",
        "csi300",
        "--state_reference_instruments",
        "all",
        "--market_index",
        "SH000300",
        "--state_reference_market_index",
        "SH000300",
        "--state_validity_protocol",
        "strict",
        "--reference_feature_source",
        "raw_qlib_stream",
        "--reference_fetch_scope",
        "active_block",
        "--field_profile",
        "relative_complexity_v1",
        "--no-emit-observation",
        "--out-field",
        FIELD_REL,
        "--legacy-out",
        "artifacts/market_state/daily_market_ref_all_SH000300_relative_complexity_v1_legacy.pkl",
    ]


def _variant_command(step: dict[str, str], python_exe: str, run_tag: str, seed: int) -> list[str]:
    cmd = [
        python_exe,
        str(VARIANT_SCRIPT),
        "--workflow_module",
        step["workflow_module"],
        "--market_state_path",
        step["market_state_path"],
        "--market_day_summary_path",
        OBS_PATH,
        "--router_summary_source",
        step["router_summary_source"],
        "--pooling_summary_source",
        step["pooling_summary_source"],
        "--label",
        f"{step['name']}_SH000300",
        "--experiment_suffix",
        f"{run_tag}_{step['name']}",
        "--seed",
        str(seed),
        "--use_hierarchical_state_field",
        "1",
        "--global_state_use_macro",
        "1",
        "--enable_local_counterfactual_diag",
        "1",
        "--enable_expert_advantage_diag",
        "1",
    ]
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or materialize the HSF relative-complexity experiment matrix.")
    parser.add_argument("--python", dest="python_exe", default=_default_python())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default=dt.datetime.now().strftime("hsf_relative_complexity_%Y%m%d_%H%M%S"))
    parser.add_argument("--phase", choices=["single", "first", "local", "all"], default="first")
    parser.add_argument("--include-precompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "single":
        steps = SINGLE_RELATIVE
    elif args.phase == "first":
        steps = FIRST_ROUND
    elif args.phase == "local":
        steps = SECOND_ROUND
    else:
        steps = FIRST_ROUND + SECOND_ROUND
    run_dir = RUNS_DIR / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    commands: dict[str, list[str]] = {}
    if args.include_precompute:
        commands["precompute_relative_complexity_v1"] = _precompute_command(args.python_exe)
    for step in steps:
        commands[step["name"]] = _variant_command(step, args.python_exe, args.run_tag, args.seed)

    commands_text = []
    for name, cmd in commands.items():
        commands_text.extend([f"[{name}]", _command_to_text(cmd), ""])
    _write_text(run_dir / "commands.txt", "\n".join(commands_text))

    status: dict[str, Any] = {
        "run_tag": args.run_tag,
        "created_at": _now(),
        "phase": args.phase,
        "dry_run": bool(args.dry_run),
        "include_precompute": bool(args.include_precompute),
        "project_root": str(PROJECT_ROOT),
        "policy": "hsf_relative_complexity_macro_state_and_local_fusion_denoising",
        "steps": {
            name: {
                "status": "planned",
                "command": _command_to_text(cmd),
            }
            for name, cmd in commands.items()
        },
    }
    _write_json(run_dir / "status.json", status)
    _write_text(
        run_dir / "README.md",
        "# HSF Relative Complexity Matrix\n\n"
        "- First round validates ref-all vs benchmark relative macro state.\n"
        "- Second round isolates local injection and bounded residual fusion.\n"
        "- Loss is fixed to `main_loss=mse`, `mse_normalize=False` by config.\n"
        f"- Commands: `{run_dir / 'commands.txt'}`\n",
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

    for name, cmd in commands.items():
        stdout_path = run_dir / f"{name}.stdout.log"
        stderr_path = run_dir / f"{name}.stderr.log"
        status["steps"][name]["status"] = "running"
        status["steps"][name]["started_at"] = _now()
        _write_json(run_dir / "status.json", status)
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
            "w",
            encoding="utf-8",
            errors="replace",
        ) as stderr:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True)
            status["steps"][name]["pid"] = proc.pid
            _write_json(run_dir / "status.json", status)
            returncode = proc.wait()
        status["steps"][name].update(
            {
                "status": "completed" if returncode == 0 else "failed",
                "finished_at": _now(),
                "returncode": int(returncode),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        )
        _write_json(run_dir / "status.json", status)
        _write_text(
            run_dir / "latest_tail.txt",
            f"[{name}] stdout tail\n{'-' * 80}\n{_tail(stdout_path)}\n"
            f"[{name}] stderr tail\n{'-' * 80}\n{_tail(stderr_path)}",
        )
        if returncode != 0:
            status["status"] = "failed"
            status["failed_step"] = name
            status["finished_at"] = _now()
            _write_json(run_dir / "status.json", status)
            return int(returncode)

    status["status"] = "completed"
    status["finished_at"] = _now()
    _write_json(run_dir / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
