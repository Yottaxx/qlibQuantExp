from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PYTHON = Path(r"C:\Users\60585\miniconda3\envs\quantEnv\python.exe")
DEFAULT_BASELINE_RUN_ID = "dce0b75c75344c7b8ee0a19f61ace8b0"
DEFAULT_MLRUNS_DIR = PROJECT_ROOT / "mlruns"

BASELINE_CURRENT = {
    "label": "base_current_existing",
    "market_state_path": "artifacts/market_state/daily_market_field_csi300.pkl",
    "market_day_summary_path": "artifacts/market_state/daily_market_observation_csi300.pkl",
    "run_id": DEFAULT_BASELINE_RUN_ID,
}

SHARED_CSI800_ASSETS = {
    "market_state_path": "artifacts/market_state/daily_market_field_csi800.pkl",
    "market_day_summary_path": "artifacts/market_state/daily_market_observation_csi800.pkl",
}


@dataclass
class StepSpec:
    name: str
    label: str
    kind: str
    args: list[str]
    required: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight best-effort target experiment runner.")
    parser.add_argument("--python", type=str, default=str(DEFAULT_PYTHON))
    parser.add_argument("--repo_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--mlruns_dir", type=str, default=str(DEFAULT_MLRUNS_DIR))
    parser.add_argument("--run_tag", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--baseline_run_id", type=str, default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--skip_support_run", action="store_true")
    parser.add_argument("--skip_fallback_run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
        except Exception:
            pass


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def safe_load_pickle(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return read_pickle(path)
    except Exception:
        return None


def resolve_run_dir(spec: str, mlruns_dir: Path) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    matches = [x for x in mlruns_dir.glob(f"*/*{spec}*") if x.is_dir()]
    exact = [x for x in matches if x.name == spec]
    if exact:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Unable to resolve run spec: {spec}")


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary = safe_load_pickle(run_dir / "artifacts" / "run_summary")
    return dict(summary) if isinstance(summary, dict) else {}


def load_variant_meta(run_dir: Path) -> dict[str, Any]:
    meta = safe_load_pickle(run_dir / "artifacts" / "variant_meta")
    return dict(meta) if isinstance(meta, dict) else {}


def fmt_num(x: Any, digits: int = 4) -> str:
    try:
        value = float(x)
    except Exception:
        return "N/A"
    if value != value or value in (float("inf"), float("-inf")):
        return "N/A"
    return f"{value:.{digits}f}"


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def build_variant_args(
    *,
    repo_root: Path,
    label: str,
    experiment_suffix: str,
    market_state_path: str,
    market_day_summary_path: str,
    use_hierarchical: bool,
    seed: int,
) -> list[str]:
    args = [
        str(repo_root / "scripts" / "run_workflow_market_state_variant.py"),
        "--market_state_path",
        market_state_path,
        "--market_day_summary_path",
        market_day_summary_path,
        "--router_summary_source",
        "day_asset",
        "--pooling_summary_source",
        "day_asset",
        "--label",
        label,
        "--experiment_suffix",
        experiment_suffix,
        "--seed",
        str(seed),
        "--use_hierarchical_state_field",
        "1" if use_hierarchical else "0",
    ]
    if use_hierarchical:
        args.extend(
            [
                "--d_global_state",
                "64",
                "--d_local_state",
                "64",
                "--global_state_use_macro",
                "1",
                "--global_state_use_day_summary",
                "1",
                "--local_state_input_mode",
                "last_mean_std_trend_vol",
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
            ]
        )
    return args


def make_steps(args: argparse.Namespace, repo_root: Path) -> list[StepSpec]:
    suffix = f"overnight_target_{args.run_tag}"
    steps = [
        StepSpec(
            name="target_hsf_shared_ref_csi800",
            label="hsf_shared_ref_csi800_target",
            kind="run",
            args=build_variant_args(
                repo_root=repo_root,
                label="hsf_shared_ref_csi800_target",
                experiment_suffix=suffix,
                market_state_path=SHARED_CSI800_ASSETS["market_state_path"],
                market_day_summary_path=SHARED_CSI800_ASSETS["market_day_summary_path"],
                use_hierarchical=True,
                seed=args.seed,
            ),
            required=True,
        )
    ]
    if not args.skip_support_run:
        steps.append(
            StepSpec(
                name="support_base_shared_ref_csi800",
                label="base_shared_ref_csi800_support",
                kind="run",
                args=build_variant_args(
                    repo_root=repo_root,
                    label="base_shared_ref_csi800_support",
                    experiment_suffix=suffix,
                    market_state_path=SHARED_CSI800_ASSETS["market_state_path"],
                    market_day_summary_path=SHARED_CSI800_ASSETS["market_day_summary_path"],
                    use_hierarchical=False,
                    seed=args.seed,
                ),
                required=False,
            )
        )
    if not args.skip_fallback_run:
        steps.append(
            StepSpec(
                name="fallback_hsf_current_ref_csi300",
                label="hsf_current_ref_csi300_fallback",
                kind="fallback",
                args=build_variant_args(
                    repo_root=repo_root,
                    label="hsf_current_ref_csi300_fallback",
                    experiment_suffix=suffix,
                    market_state_path=BASELINE_CURRENT["market_state_path"],
                    market_day_summary_path=BASELINE_CURRENT["market_day_summary_path"],
                    use_hierarchical=True,
                    seed=args.seed,
                ),
                required=False,
            )
        )
    return steps


def default_status(*, args: argparse.Namespace, repo_root: Path, run_dir: Path, steps: list[StepSpec]) -> dict[str, Any]:
    return {
        "run_tag": args.run_tag,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "python": args.python,
        "baseline_current": {
            **BASELINE_CURRENT,
            "run_id": args.baseline_run_id,
        },
        "shared_reference_assets": dict(SHARED_CSI800_ASSETS),
        "steps": [
            {
                "name": step.name,
                "label": step.label,
                "kind": step.kind,
                "required": step.required,
                "status": "pending",
                "command": [args.python, *step.args],
            }
            for step in steps
        ],
        "final_verdict": "running",
        "final_notes": [],
    }


def write_commands(run_dir: Path, status: dict[str, Any]) -> None:
    command_lines = []
    for step in status["steps"]:
        joined = " ".join(json.dumps(token, ensure_ascii=False) for token in step["command"])
        command_lines.append(f"[{step['name']}]")
        command_lines.append(joined)
        command_lines.append("")
    write_text(run_dir / "commands.txt", "\n".join(command_lines))


def update_status(run_dir: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = now_iso()
    write_json(run_dir / "status.json", status)
    write_json(run_dir / "manifest.json", status)
    write_text(run_dir / "summary.md", render_summary(status, Path(status["repo_root"])))


def render_metrics_table(rows: Iterable[dict[str, Any]]) -> list[str]:
    lines = [
        "| Label | Run | RankIC mean | IC mean | Info Ratio | Ann. Return | Max DD |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + row["label"]
            + " | "
            + row.get("run_ref", "N/A")
            + " | "
            + fmt_num(row.get("ric_mean"))
            + " | "
            + fmt_num(row.get("ic_mean"))
            + " | "
            + fmt_num(row.get("info_ratio"))
            + " | "
            + fmt_num(row.get("ann_ret"))
            + " | "
            + fmt_num(row.get("max_dd"))
            + " |"
        )
    return lines


def render_summary(status: dict[str, Any], repo_root: Path) -> str:
    lines = [
        "# Overnight Target Experiment Status",
        "",
        f"- Run tag: `{status['run_tag']}`",
        f"- Started: `{status['started_at']}`",
        f"- Updated: `{status['updated_at']}`",
        f"- Final verdict: `{status.get('final_verdict', 'running')}`",
        "",
        "## Primary Intent",
        "",
        "- Target run: `hsf_shared_ref_csi800_target`",
        "- Shared broader reference assets: `daily_market_field_csi800.pkl` + `daily_market_observation_csi800.pkl`",
        "- Baseline comparison target: existing `base_current` run",
        "",
        "## Step Status",
        "",
        "| Step | Kind | Status | Recorder ID | Log | Last phase | Last valid line |",
        "|:--|:--|:--|:--|:--|:--|:--|",
    ]
    for step in status["steps"]:
        log_path = step.get("log_path")
        lines.append(
            "| "
            + step["name"]
            + " | "
            + step["kind"]
            + " | "
            + step["status"]
            + " | "
            + (step.get("recorder_id") or "N/A")
            + " | "
            + (relpath(Path(log_path), repo_root) if log_path else "N/A")
            + " | "
            + (step.get("last_phase") or "N/A")
            + " | "
            + (step.get("last_valid_line") or "N/A").replace("|", "\\|")
            + " |"
        )

    comparison_rows = []
    baseline = status.get("baseline_metrics")
    if isinstance(baseline, dict):
        comparison_rows.append({"label": "base_current_existing", **baseline})
    for step in status["steps"]:
        metrics = step.get("metrics")
        if isinstance(metrics, dict):
            comparison_rows.append({"label": step["label"], **metrics})
    if comparison_rows:
        lines.extend(["", "## Metrics Snapshot", ""])
        lines.extend(render_metrics_table(comparison_rows))

    target = next((s for s in status["steps"] if s["name"] == "target_hsf_shared_ref_csi800"), None)
    if target and isinstance(target.get("metrics"), dict) and isinstance(baseline, dict):
        delta_ric = target["metrics"].get("ric_mean")
        delta_ic = target["metrics"].get("ic_mean")
        if delta_ric is not None and baseline.get("ric_mean") is not None:
            delta_ric = float(delta_ric) - float(baseline["ric_mean"])
        else:
            delta_ric = None
        if delta_ic is not None and baseline.get("ic_mean") is not None:
            delta_ic = float(delta_ic) - float(baseline["ic_mean"])
        else:
            delta_ic = None
        lines.extend(
            [
                "",
                "## Target vs Existing Base",
                "",
                f"- Delta RankIC mean: `{fmt_num(delta_ric)}`",
                f"- Delta IC mean: `{fmt_num(delta_ic)}`",
            ]
        )

    if status.get("final_notes"):
        lines.extend(["", "## Notes", ""])
        for note in status["final_notes"]:
            lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Morning Checklist",
            "",
            f"1. Read [`summary.md`]({run_dir_link(status['run_dir'])}) via this run directory.",
            "2. If the target step succeeded, use its `recorder_id` to inspect full MLflow artifacts.",
            "3. If the target step failed, read the target log first, then check whether the fallback step produced a run.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_dir_link(path: str) -> str:
    return path


def find_step(status: dict[str, Any], name: str) -> dict[str, Any]:
    for step in status["steps"]:
        if step["name"] == name:
            return step
    raise KeyError(name)


def load_baseline_metrics(status: dict[str, Any], mlruns_dir: Path) -> None:
    run_dir = resolve_run_dir(status["baseline_current"]["run_id"], mlruns_dir)
    summary = load_run_summary(run_dir)
    meta = load_variant_meta(run_dir)
    status["baseline_current"]["resolved_run_dir"] = str(run_dir)
    status["baseline_metrics"] = {
        "run_ref": relpath(run_dir, Path(status["repo_root"])),
        "ric_mean": summary.get("ric_mean"),
        "ic_mean": summary.get("ic_mean"),
        "info_ratio": summary.get("info_ratio"),
        "ann_ret": summary.get("ann_ret"),
        "max_dd": summary.get("max_dd"),
        "market_state_path": meta.get("market_state_path", summary.get("market_state_path")),
        "market_day_summary_path": meta.get("market_day_summary_path", summary.get("market_day_summary_path")),
    }


def collect_run_metrics(mlruns_dir: Path, recorder_id: str, repo_root: Path) -> dict[str, Any]:
    run_dir = resolve_run_dir(recorder_id, mlruns_dir)
    summary = load_run_summary(run_dir)
    meta = load_variant_meta(run_dir)
    return {
        "run_ref": relpath(run_dir, repo_root),
        "recorder_id": recorder_id,
        "ric_mean": summary.get("ric_mean"),
        "ic_mean": summary.get("ic_mean"),
        "info_ratio": summary.get("info_ratio"),
        "ann_ret": summary.get("ann_ret"),
        "max_dd": summary.get("max_dd"),
        "turnover": summary.get("turnover"),
        "market_state_path": meta.get("market_state_path", summary.get("market_state_path")),
        "market_day_summary_path": meta.get("market_day_summary_path", summary.get("market_day_summary_path")),
    }


def run_step(
    *,
    run_dir: Path,
    status: dict[str, Any],
    step: StepSpec,
    python_exe: str,
    mlruns_dir: Path,
    repo_root: Path,
) -> bool:
    step_state = find_step(status, step.name)
    if step_state["status"] == "completed":
        return True

    log_path = run_dir / f"{step.name}.log"
    step_state["status"] = "running"
    step_state["started_at"] = now_iso()
    step_state["log_path"] = str(log_path)
    update_status(run_dir, status)

    proc = subprocess.Popen(
        [python_exe, *step.args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(repo_root),
    )

    recorder_pat = re.compile(r">>> \[Variant\] recorder_id=(.+)$")
    phase_pat = re.compile(r"^>>> \[(Phase[^\]]+)\]")
    valid_pat = re.compile(r"^\| Valid \d+ \|")
    train_pat = re.compile(r"^\| Train \d+ \|")
    tail = deque(maxlen=80)
    last_write = time.time()

    with log_path.open("w", encoding="utf-8") as log_file:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            print(line, flush=True)
            log_file.write(raw_line)
            log_file.flush()
            tail.append(line)
            m = recorder_pat.search(line)
            if m:
                step_state["recorder_id"] = m.group(1).strip()
            p = phase_pat.search(line)
            if p:
                step_state["last_phase"] = p.group(1)
            if train_pat.search(line):
                step_state["last_train_line"] = line
            if valid_pat.search(line):
                step_state["last_valid_line"] = line
            step_state["last_output_line"] = line
            now_t = time.time()
            if now_t - last_write >= 10:
                write_text(run_dir / "latest_tail.txt", "\n".join(tail) + "\n")
                update_status(run_dir, status)
                last_write = now_t

    exit_code = proc.wait()
    step_state["completed_at"] = now_iso()
    step_state["exit_code"] = int(exit_code)
    write_text(run_dir / "latest_tail.txt", "\n".join(tail) + "\n")
    if exit_code == 0:
        step_state["status"] = "completed"
        recorder_id = step_state.get("recorder_id")
        if recorder_id:
            step_state["metrics"] = collect_run_metrics(mlruns_dir, recorder_id, repo_root)
    else:
        step_state["status"] = "failed"
    update_status(run_dir, status)
    return exit_code == 0


def main() -> None:
    configure_stdio()
    args = parse_args()
    repo_root = Path(args.repo_root)
    python_exe = str(Path(args.python))
    mlruns_dir = Path(args.mlruns_dir)
    run_dir = repo_root / "next_step" / "hierarchical_state_field" / "experiment" / "runs" / f"overnight_target_{args.run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    steps = make_steps(args, repo_root)
    status = default_status(args=args, repo_root=repo_root, run_dir=run_dir, steps=steps)
    write_commands(run_dir, status)
    load_baseline_metrics(status, mlruns_dir)
    status["final_notes"].append("Primary target uses existing csi800 split assets as the broader shared reference.")
    status["final_notes"].append("This avoids blocking on the current ref_all strict build, which is still memory-constrained on this machine.")
    update_status(run_dir, status)

    target_ok = run_step(
        run_dir=run_dir,
        status=status,
        step=steps[0],
        python_exe=python_exe,
        mlruns_dir=mlruns_dir,
        repo_root=repo_root,
    )

    if target_ok and len(steps) >= 2 and steps[1].kind == "run":
        run_step(
            run_dir=run_dir,
            status=status,
            step=steps[1],
            python_exe=python_exe,
            mlruns_dir=mlruns_dir,
            repo_root=repo_root,
        )
    elif not target_ok:
        fallback = next((s for s in steps if s.kind == "fallback"), None)
        if fallback is not None:
            status["final_notes"].append("Primary target failed; fallback hierarchical run on current csi300 assets was started.")
            update_status(run_dir, status)
            run_step(
                run_dir=run_dir,
                status=status,
                step=fallback,
                python_exe=python_exe,
                mlruns_dir=mlruns_dir,
                repo_root=repo_root,
            )

    target_step = find_step(status, "target_hsf_shared_ref_csi800")
    if target_step["status"] == "completed":
        status["final_verdict"] = "target_completed"
        status["final_notes"].append("Morning priority: inspect target recorder artifacts first.")
    elif target_step["status"] == "failed":
        fallback = next((s for s in status["steps"] if s["kind"] == "fallback" and s["status"] == "completed"), None)
        if fallback is not None:
            status["final_verdict"] = "fallback_completed"
            status["final_notes"].append("Primary target failed, but fallback produced a completed hierarchical run.")
        else:
            status["final_verdict"] = "failed"
            status["final_notes"].append("No completed overnight run was produced.")
    else:
        status["final_verdict"] = "partial"

    update_status(run_dir, status)
    print(f"Run directory: {run_dir}")
    print(f"Status file: {run_dir / 'status.json'}")
    print(f"Summary file: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        raise
