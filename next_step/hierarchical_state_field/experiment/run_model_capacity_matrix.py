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
QUANT_ENV_PYTHON = Path(r"C:\Users\60585\miniconda3\envs\quantEnv\python.exe")

OBS_PATH = "artifacts/market_state/daily_market_observation_ref_all_pti_v1.pkl"
FIELD_37D = "artifacts/market_state/daily_market_field_ref_all_SH000300.pkl"
FIELD_100D_LEGACY = "artifacts/market_state/daily_market_ref_all_SH000300_legacy.pkl"

STEPS = [
    {
        "name": "hsf_factorproj_sum_norm",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_factorproj",
        "market_state_path": FIELD_37D,
        "label": "hsf_factorproj_sum_norm_ref_all_SH000300",
        "local_state_input_mode": "factor_projection_v1",
        "state_fusion_mode": "sum_norm",
    },
    {
        "name": "hsf_factorproj_branchmlp",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_factorproj_branchmlp",
        "market_state_path": FIELD_37D,
        "label": "hsf_factorproj_branchmlp_ref_all_SH000300",
        "local_state_input_mode": "factor_projection_v1",
        "state_fusion_mode": "branch_mlp_v1",
    },
    {
        "name": "hsf_legacy100_state",
        "workflow_module": "next_step.hierarchical_state_field.experiment.configs.ref_all_hsf_csi300_legacy100_state",
        "market_state_path": FIELD_100D_LEGACY,
        "label": "hsf_legacy100_state_ref_all_SH000300",
        "local_state_input_mode": "last_mean_std_trend_vol",
        "state_fusion_mode": "sum_norm",
    },
]


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_python() -> str:
    return str(QUANT_ENV_PYTHON if QUANT_ENV_PYTHON.exists() else Path(sys.executable))


def _quote_arg(arg: str) -> str:
    escaped = str(arg).replace('"', r'\"')
    return f'"{escaped}"'


def _command_to_text(cmd: list[str]) -> str:
    return " ".join(_quote_arg(part) for part in cmd)


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


def _build_command(step: dict[str, str], python_exe: str, experiment_suffix: str, seed: int) -> list[str]:
    return [
        python_exe,
        str(VARIANT_SCRIPT),
        "--workflow_module",
        step["workflow_module"],
        "--market_state_path",
        step["market_state_path"],
        "--market_day_summary_path",
        OBS_PATH,
        "--router_summary_source",
        "day_asset",
        "--pooling_summary_source",
        "day_asset",
        "--label",
        step["label"],
        "--experiment_suffix",
        f"{experiment_suffix}_{step['name']}",
        "--seed",
        str(seed),
        "--use_hierarchical_state_field",
        "1",
        "--d_global_state",
        "64",
        "--d_local_state",
        "64",
        "--global_state_use_macro",
        "1",
        "--global_state_use_day_summary",
        "1",
        "--local_state_input_mode",
        step["local_state_input_mode"],
        "--state_fusion_mode",
        step["state_fusion_mode"],
        "--router_use_global_state",
        "1",
        "--router_use_local_state",
        "1",
        "--film_use_global_state",
        "1",
        "--film_use_local_state",
        "1",
        "--pooling_use_global_state",
        "1",
        "--pooling_use_local_state",
        "1",
        "--enable_local_counterfactual_diag",
        "1",
    ]


def _run_step(*, step: dict[str, str], cmd: list[str], run_dir: Path, status: dict[str, Any]) -> int:
    stdout_path = run_dir / f"{step['name']}.stdout.log"
    stderr_path = run_dir / f"{step['name']}.stderr.log"
    status["steps"][step["name"]]["status"] = "running"
    status["steps"][step["name"]]["started_at"] = _now()
    _write_json(run_dir / "status.json", status)

    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as stderr:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        status["steps"][step["name"]]["pid"] = proc.pid
        _write_json(run_dir / "status.json", status)
        returncode = proc.wait()

    status["steps"][step["name"]]["status"] = "completed" if returncode == 0 else "failed"
    status["steps"][step["name"]]["finished_at"] = _now()
    status["steps"][step["name"]]["returncode"] = returncode
    status["steps"][step["name"]]["stdout_log"] = str(stdout_path)
    status["steps"][step["name"]]["stderr_log"] = str(stderr_path)
    _write_json(run_dir / "status.json", status)
    _write_text(
        run_dir / "latest_tail.txt",
        f"[{step['name']}] stdout tail\n"
        f"{'-' * 80}\n"
        f"{_tail(stdout_path)}\n"
        f"[{step['name']}] stderr tail\n"
        f"{'-' * 80}\n"
        f"{_tail(stderr_path)}",
    )
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or materialize the HSF model-capacity experiment matrix.")
    parser.add_argument("--python", dest="python_exe", default=_default_python())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-tag",
        default=dt.datetime.now().strftime("hsf_model_capacity_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--only",
        choices=["all", "factorproj_sum_norm", "factorproj_branchmlp", "legacy100_state"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [
        step
        for step in STEPS
        if args.only == "all" or step["name"].endswith(args.only)
    ]
    run_dir = RUNS_DIR / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    commands = {step["name"]: _build_command(step, args.python_exe, args.run_tag, args.seed) for step in selected}
    commands_text = []
    for step_name, cmd in commands.items():
        commands_text.append(f"[{step_name}]")
        commands_text.append(_command_to_text(cmd))
        commands_text.append("")
    _write_text(run_dir / "commands.txt", "\n".join(commands_text))

    status: dict[str, Any] = {
        "run_tag": args.run_tag,
        "created_at": _now(),
        "project_root": str(PROJECT_ROOT),
        "variant_script": str(VARIANT_SCRIPT),
        "python_exe": args.python_exe,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "policy": "hsf_model_capacity_p0_p1_p2_p3_matrix",
        "shared_observation_path": OBS_PATH,
        "steps": {
            step["name"]: {
                "status": "planned",
                "workflow_module": step["workflow_module"],
                "market_state_path": step["market_state_path"],
                "market_day_summary_path": OBS_PATH,
                "label": step["label"],
                "local_state_input_mode": step["local_state_input_mode"],
                "state_fusion_mode": step["state_fusion_mode"],
                "enable_local_counterfactual_diag": True,
                "command": _command_to_text(commands[step["name"]]),
            }
            for step in selected
        },
    }
    _write_json(run_dir / "status.json", status)
    _write_text(
        run_dir / "README.md",
        "# HSF Model Capacity Matrix\n\n"
        "- Scope: P0/P1/P2/P3 model-capacity isolation.\n"
        "- Universe: CSI300.\n"
        "- Train segment: 2008-01-30 to 2020-03-31.\n"
        "- Valid/test/backtest segment: 2020-07-01 to 2022-12-31.\n"
        "- Feature embedding: feature_tokenizer.\n"
        "- Pooling mode: full.\n"
        "- Counterfactual diagnostics: enabled for all matrix runs.\n"
        "- Steps: factorproj+sum_norm, factorproj+branch_mlp_v1, legacy100 state control.\n",
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

    exit_code = 0
    for step in selected:
        print(f"[Start] {step['name']}", flush=True)
        returncode = _run_step(step=step, cmd=commands[step["name"]], run_dir=run_dir, status=status)
        print(f"[Finish] {step['name']} returncode={returncode}", flush=True)
        if returncode != 0:
            exit_code = returncode
            break

    status["status"] = "completed" if exit_code == 0 else "failed"
    status["finished_at"] = _now()
    status["exit_code"] = exit_code
    _write_json(run_dir / "status.json", status)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
