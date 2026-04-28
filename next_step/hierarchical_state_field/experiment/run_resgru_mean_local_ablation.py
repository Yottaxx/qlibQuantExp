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

MARKET_STATE_PATH = "artifacts/market_state/daily_market_field_ref_all_SH000300_tri_scope_relative_momentum_v1.pkl"
MARKET_DAY_SUMMARY_PATH = "artifacts/market_state/daily_market_observation_tri_scope_v1.pkl"

STEPS = [
    {
        "name": "resgru_mean_global_only",
        "label": "csi300_tri_scope_resgru_mean_global_only",
        "workflow_module": (
            "next_step.hierarchical_state_field.experiment.configs."
            "ref_all_hsf_csi300_tri_scope_resgru_mean_global_only"
        ),
        "intent": "Disable local injection into router and FiLM.",
    },
    {
        "name": "resgru_mean_router_only",
        "label": "csi300_tri_scope_resgru_mean_router_only",
        "workflow_module": (
            "next_step.hierarchical_state_field.experiment.configs."
            "ref_all_hsf_csi300_tri_scope_resgru_mean_router_only"
        ),
        "intent": "Allow local state only through router.",
    },
    {
        "name": "resgru_mean_film_only",
        "label": "csi300_tri_scope_resgru_mean_film_only",
        "workflow_module": (
            "next_step.hierarchical_state_field.experiment.configs."
            "ref_all_hsf_csi300_tri_scope_resgru_mean_film_only"
        ),
        "intent": "Allow local state only through FiLM.",
    },
]


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


def _build_command(step: dict[str, str], *, python_exe: str, run_tag: str, seed: int, skip_visuals: bool) -> list[str]:
    cmd = [
        python_exe,
        str(VARIANT_SCRIPT),
        "--qlib_kernels",
        "1",
        "--workflow_module",
        step["workflow_module"],
        "--market_state_path",
        MARKET_STATE_PATH,
        "--market_day_summary_path",
        MARKET_DAY_SUMMARY_PATH,
        "--router_summary_source",
        "day_asset",
        "--pooling_summary_source",
        "none",
        "--label",
        step["label"],
        "--experiment_suffix",
        f"{run_tag}_{step['name']}",
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


def _run_step(*, step: dict[str, str], cmd: list[str], run_dir: Path, status: dict[str, Any]) -> int:
    stdout_path = run_dir / f"{step['name']}.stdout.log"
    stderr_path = run_dir / f"{step['name']}.stderr.log"
    step_status = status["steps"][step["name"]]
    step_status["status"] = "running"
    step_status["started_at"] = _now()
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
        step_status["pid"] = proc.pid
        _write_json(run_dir / "status.json", status)
        returncode = proc.wait()

    step_status["status"] = "completed" if returncode == 0 else "failed"
    step_status["finished_at"] = _now()
    step_status["returncode"] = returncode
    step_status["stdout_log"] = str(stdout_path)
    step_status["stderr_log"] = str(stderr_path)
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
    parser = argparse.ArgumentParser(description="Run CSI300 resGRU mean local-channel ablation matrix.")
    parser.add_argument("--python", dest="python_exe", default=_default_python())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-tag",
        default=dt.datetime.now().strftime("hsf_resgru_mean_local_ablation_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--only",
        choices=["all", "global_only", "router_only", "film_only"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-visuals", action="store_true")
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

    commands = {
        step["name"]: _build_command(
            step,
            python_exe=args.python_exe,
            run_tag=args.run_tag,
            seed=args.seed,
            skip_visuals=not bool(args.with_visuals),
        )
        for step in selected
    }
    commands_text = []
    for step in selected:
        commands_text.append(f"[{step['name']}]")
        commands_text.append(_command_to_text(commands[step["name"]]))
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
        "skip_visuals": not bool(args.with_visuals),
        "policy": "resgru_mean_local_channel_ablation",
        "baseline": "ref_all_hsf_csi300_tri_scope_resgru_mean_bounded_inner",
        "market_state_path": MARKET_STATE_PATH,
        "market_day_summary_path": MARKET_DAY_SUMMARY_PATH,
        "acceptance": {
            "best_rankic_min": 0.0615,
            "portfolio_ir_min": 0.71,
            "max_drawdown_floor": -0.135,
            "rankic_ir_min": 0.36,
            "time_ratio_delta_time_factor_corr_target": 0.10,
        },
        "steps": {
            step["name"]: {
                "status": "planned",
                "workflow_module": step["workflow_module"],
                "label": step["label"],
                "intent": step["intent"],
                "command": _command_to_text(commands[step["name"]]),
            }
            for step in selected
        },
    }
    _write_json(run_dir / "status.json", status)
    _write_text(
        run_dir / "README.md",
        "# ResGRU Mean Local-Channel Ablation\n\n"
        "- Baseline: `ref_all_hsf_csi300_tri_scope_resgru_mean_bounded_inner`.\n"
        "- Scope: CSI300 only, seed 42, fixed MSE loss and fixed tri-scope macro assets.\n"
        "- Goal: identify whether local state should enter no channel, router only, or FiLM only.\n"
        "- Metrics: Best RankIC, RankIC IR, with-cost portfolio IR, MaxDD, local counterfactual, expert advantage alignment.\n",
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
