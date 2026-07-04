from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


MODEL_OVERRIDES = {
    "use_regime_time_embedding": True,
    "use_regime_factor_gate": True,
    "router_use_layer_summary": True,
    "router_mode": "learned",
}

PORT_OVERRIDES = {"strategy": {"kwargs": {"n_drop": 5}}}
REPORT_STEM = "full_daily_sampler_paired_promotion_bestvalid_20260525"

DECISION_KEYS = {
    "rankic": "performance/daily_rank_ic_mean",
    "rankicir": "performance/rank_icir",
    "ir_cost": "portfolio/information_ratio_with_cost",
    "maxdd_cost": "portfolio/max_drawdown_with_cost",
    "post_peak_decay": "optimization/post_peak_decay",
    "rankic_gap": "optimization/rank_ic_gap_last",
    "coverage_gap": "sampler/coverage_gap_vs_full_last",
    "duplicate_rate": "sampler/duplicate_rate_last",
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def load_diag_metrics(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            group = str(row.get("group", "")).strip()
            metric = str(row.get("metric", "")).strip()
            val = finite_float(row.get("value"))
            if group and metric and val is not None:
                out[f"{group}/{metric}"] = val
    return out


def find_latest_diagnostic(root: Path, *, started_at: float) -> str:
    best_path = ""
    best_mtime = started_at
    mlruns = root / "mlruns"
    if not mlruns.exists():
        return ""
    for path in mlruns.rglob("diagnostic_matrix.csv"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= best_mtime:
            best_mtime = mtime
            best_path = str(path)
    return best_path


def train_epochs_recorded(diag_path: str, stdout_path: Path) -> Optional[int]:
    if diag_path:
        run_dir = Path(diag_path).parent
        for rel in ("artifacts/train_curve", "train_curve"):
            p = run_dir / rel
            if not p.exists():
                continue
            try:
                with p.open("rb") as f:
                    obj = pickle.load(f)
                epochs = obj.get("epoch") if isinstance(obj, dict) else None
                if isinstance(epochs, list):
                    return len(epochs)
            except Exception:
                pass
    if stdout_path.exists():
        try:
            count = 0
            with stdout_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("| Train "):
                        count += 1
            return count or None
        except Exception:
            return None
    return None


def load_checkpoint_info(diag_path: str) -> Dict[str, Any]:
    if not diag_path:
        return {}
    run_dir = Path(diag_path).parent
    for rel in ("artifacts/best_checkpoint_info", "best_checkpoint_info"):
        p = run_dir / rel
        if not p.exists():
            continue
        try:
            with p.open("rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def load_train_curve_summary(diag_path: str) -> Dict[str, Any]:
    if not diag_path:
        return {}
    run_dir = Path(diag_path).parent
    curve = None
    for rel in ("artifacts/train_curve", "train_curve"):
        p = run_dir / rel
        if not p.exists():
            continue
        try:
            with p.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                curve = obj
                break
        except Exception:
            return {}
    if not isinstance(curve, dict):
        return {}

    def _list(name: str) -> List[Any]:
        v = curve.get(name, [])
        return v if isinstance(v, list) else []

    def _at(name: str, idx: int) -> Optional[float]:
        vals = _list(name)
        if idx < 0 or idx >= len(vals):
            return None
        return finite_float(vals[idx])

    valid = _list("valid_rank_ic")
    epochs = _list("epoch") or list(range(1, len(valid) + 1))
    best_idx = None
    best_val = None
    for i, v in enumerate(valid):
        fv = finite_float(v)
        if fv is None:
            continue
        if best_val is None or fv > best_val:
            best_val = fv
            best_idx = i

    ckpt = load_checkpoint_info(diag_path)
    ckpt_epoch = ckpt.get("epoch")
    ckpt_idx = None
    try:
        if ckpt_epoch not in (None, ""):
            ckpt_idx = int(ckpt_epoch) - 1
    except Exception:
        ckpt_idx = None
    if ckpt_idx is None:
        ckpt_idx = best_idx

    train_at = _at("train_rank_ic", ckpt_idx) if ckpt_idx is not None else None
    valid_at = _at("valid_rank_ic", ckpt_idx) if ckpt_idx is not None else None
    final_valid = finite_float(valid[-1]) if valid else None
    return {
        "best_valid_rank_ic": best_val,
        "best_valid_rank_ic_epoch": (
            epochs[best_idx] if best_idx is not None and best_idx < len(epochs) else None
        ),
        "final_valid_rank_ic": final_valid,
        "best_minus_final_valid_rank_ic": (
            None if best_val is None or final_valid is None else best_val - final_valid
        ),
        "checkpoint_epoch": ckpt_epoch,
        "train_rank_ic_at_checkpoint": train_at,
        "valid_rank_ic_at_checkpoint": valid_at,
        "rank_ic_gap_at_checkpoint": (
            None if train_at is None or valid_at is None else train_at - valid_at
        ),
    }


def run_arm(
    *,
    root: Path,
    out_dir: Path,
    python_exe: str,
    seed: int,
    sampler_mode: str,
    continue_finished: bool,
) -> Dict[str, Any]:
    setting = f"full135_{sampler_mode}_seed{seed}_ndrop5_40e"
    arm_dir = out_dir / setting
    arm_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arm_dir / "manifest.json"
    stdout_path = arm_dir / "stdout.log"
    stderr_path = arm_dir / "stderr.log"

    if continue_finished and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except Exception:
            existing = {}
        diag = str(existing.get("diagnostic_matrix", ""))
        if existing.get("returncode") == 0 and diag and Path(diag).exists():
            existing["metrics"] = load_diag_metrics(diag)
            existing["epochs_recorded"] = train_epochs_recorded(diag, stdout_path)
            existing["checkpoint_info"] = load_checkpoint_info(diag)
            existing["curve_summary"] = load_train_curve_summary(diag)
            print(f"[{now_str()}] skip completed {setting}", flush=True)
            return existing

    trainer_overrides = {
        "seed": int(seed),
        "device": "auto",
        "deterministic_mode": "warn",
        "seed_workers": True,
        "train_sampler_mode": sampler_mode,
        "sampler_diag": True,
        "use_tqdm": False,
        "n_epochs": 40,
        "early_stop": 0,
        "train_stop_threshold": None,
        "min_epochs": 40,
        "consecutive_k": 999999,
        "checkpoint_metric": "valid_rank_ic",
        "checkpoint_mode": "max",
        "checkpoint_min_delta": 0.0,
    }
    manifest: Dict[str, Any] = {
        "setting": setting,
        "seed": int(seed),
        "sampler_mode": sampler_mode,
        "model_overrides": MODEL_OVERRIDES,
        "trainer_overrides": trainer_overrides,
        "port_overrides": PORT_OVERRIDES,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": None,
        "started_at": now_str(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["QIB_RUN_SETTING"] = setting
    env["QIB_MODEL_OVERRIDES_JSON"] = json.dumps(MODEL_OVERRIDES, ensure_ascii=False)
    env["QIB_TRAINER_OVERRIDES_JSON"] = json.dumps(trainer_overrides, ensure_ascii=False)
    env["QIB_PORT_OVERRIDES_JSON"] = json.dumps(PORT_OVERRIDES, ensure_ascii=False)

    print(f"[{now_str()}] start {setting}", flush=True)
    started_at = time.time()
    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        proc = subprocess.run(
            [python_exe, "-u", "work_flow.py"],
            cwd=root,
            env=env,
            stdout=out_f,
            stderr=err_f,
        )

    manifest["returncode"] = int(proc.returncode)
    manifest["finished_at"] = now_str()
    manifest["diagnostic_matrix"] = find_latest_diagnostic(root, started_at=started_at)
    manifest["metrics"] = load_diag_metrics(str(manifest.get("diagnostic_matrix", "")))
    manifest["epochs_recorded"] = train_epochs_recorded(
        str(manifest.get("diagnostic_matrix", "")),
        stdout_path,
    )
    manifest["checkpoint_info"] = load_checkpoint_info(str(manifest.get("diagnostic_matrix", "")))
    manifest["curve_summary"] = load_train_curve_summary(str(manifest.get("diagnostic_matrix", "")))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{now_str()}] finish {setting} rc={proc.returncode}", flush=True)
    return manifest


def mget(manifest: Dict[str, Any], key: str) -> Optional[float]:
    return finite_float((manifest.get("metrics", {}) or {}).get(key))


def mean_of(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def fmt(value: Optional[float], nd: int = 6) -> str:
    return "NA" if value is None else f"{value:.{nd}f}"


def judge(manifests: List[Dict[str, Any]], out_dir: Path, analysis_dir: Path) -> Dict[str, Any]:
    by_pair = {
        (int(m.get("seed")), str(m.get("sampler_mode"))): m
        for m in manifests
        if m.get("returncode") == 0
    }
    rows: List[Dict[str, Any]] = []
    for seed in (42, 43, 44):
        sampled = by_pair.get((seed, "sampled_daily"))
        full = by_pair.get((seed, "full_daily"))
        row: Dict[str, Any] = {"seed": seed}
        for label, manifest in (("sampled", sampled), ("full", full)):
            row[f"{label}_setting"] = "" if manifest is None else str(manifest.get("setting", ""))
            row[f"{label}_diagnostic_matrix"] = "" if manifest is None else str(manifest.get("diagnostic_matrix", ""))
            row[f"{label}_epochs_recorded"] = "" if manifest is None else manifest.get("epochs_recorded", "")
            ckpt = {} if manifest is None else (manifest.get("checkpoint_info", {}) or {})
            row[f"{label}_checkpoint_metric"] = ckpt.get("metric", "")
            row[f"{label}_checkpoint_epoch"] = ckpt.get("epoch", "")
            row[f"{label}_checkpoint_score"] = ckpt.get("score", "")
            row[f"{label}_checkpoint_restored"] = ckpt.get("restored", "")
            for name, key in DECISION_KEYS.items():
                if name == "rankic_gap" and manifest is not None:
                    curve = manifest.get("curve_summary", {}) or {}
                    gap = finite_float(curve.get("rank_ic_gap_at_checkpoint"))
                    row[f"{label}_{name}"] = gap if gap is not None else mget(manifest, key)
                    row[f"{label}_rankic_gap_source"] = (
                        "checkpoint" if gap is not None else "last_epoch"
                    )
                else:
                    row[f"{label}_{name}"] = None if manifest is None else mget(manifest, key)
        if sampled is not None and full is not None:
            row["gap_improvement"] = (
                row["sampled_rankic_gap"] - row["full_rankic_gap"]
                if row["sampled_rankic_gap"] is not None and row["full_rankic_gap"] is not None
                else None
            )
            row["rankic_delta"] = (
                row["full_rankic"] - row["sampled_rankic"]
                if row["full_rankic"] is not None and row["sampled_rankic"] is not None
                else None
            )
            row["rankicir_delta"] = (
                row["full_rankicir"] - row["sampled_rankicir"]
                if row["full_rankicir"] is not None and row["sampled_rankicir"] is not None
                else None
            )
            row["ir_cost_delta"] = (
                row["full_ir_cost"] - row["sampled_ir_cost"]
                if row["full_ir_cost"] is not None and row["sampled_ir_cost"] is not None
                else None
            )
            row["maxdd_cost_delta"] = (
                row["full_maxdd_cost"] - row["sampled_maxdd_cost"]
                if row["full_maxdd_cost"] is not None and row["sampled_maxdd_cost"] is not None
                else None
            )
            row["decay_delta"] = (
                row["full_post_peak_decay"] - row["sampled_post_peak_decay"]
                if row["full_post_peak_decay"] is not None and row["sampled_post_peak_decay"] is not None
                else None
            )
        rows.append(row)

    gap_mean = mean_of(r.get("gap_improvement") for r in rows)
    rankic_delta_mean = mean_of(r.get("rankic_delta") for r in rows)
    rankicir_delta_mean = mean_of(r.get("rankicir_delta") for r in rows)
    ir_delta_mean = mean_of(r.get("ir_cost_delta") for r in rows)
    maxdd_delta_mean = mean_of(r.get("maxdd_cost_delta") for r in rows)
    decay_delta_mean = mean_of(r.get("decay_delta") for r in rows)

    mechanics_ok = all(
        (r.get("full_coverage_gap") is not None and r["full_coverage_gap"] <= 0.005)
        and (r.get("full_duplicate_rate") is not None and r["full_duplicate_rate"] <= 0.005)
        for r in rows
    )
    all_pairs_ok = all(
        by_pair.get((seed, "sampled_daily")) is not None and by_pair.get((seed, "full_daily")) is not None
        for seed in (42, 43, 44)
    )
    no_cat_rankic = all(
        r.get("rankic_delta") is not None and r["rankic_delta"] >= -0.004
        for r in rows
    )
    hc6_non_degrading = {
        "rankic": rankic_delta_mean is not None and rankic_delta_mean >= 0.0,
        "ir_with_cost": ir_delta_mean is not None and ir_delta_mean >= 0.0,
        "maxdd_with_cost": maxdd_delta_mean is not None and maxdd_delta_mean >= 0.0,
        "post_peak_decay": decay_delta_mean is not None and decay_delta_mean <= 0.0,
    }
    hc6_pass_count = sum(1 for ok in hc6_non_degrading.values() if ok)
    promoted = bool(
        all_pairs_ok
        and mechanics_ok
        and gap_mean is not None and gap_mean >= 0.015
        and rankic_delta_mean is not None and rankic_delta_mean >= 0.0
        and rankicir_delta_mean is not None and rankicir_delta_mean >= -0.010
        and hc6_pass_count >= 3
        and no_cat_rankic
    )
    verdict = "promote_full_daily" if promoted else "do_not_promote_full_daily"
    if not all_pairs_ok:
        verdict = "incomplete_or_failed_runs"

    summary = {
        "created_at": now_str(),
        "verdict": verdict,
        "promoted": promoted,
        "all_pairs_ok": all_pairs_ok,
        "mechanics_ok": mechanics_ok,
        "no_catastrophic_rankic_seed": no_cat_rankic,
        "hc6_non_degrading": hc6_non_degrading,
        "hc6_pass_count": hc6_pass_count,
        "means": {
            "gap_improvement": gap_mean,
            "rankic_delta": rankic_delta_mean,
            "rankicir_delta": rankicir_delta_mean,
            "ir_cost_delta": ir_delta_mean,
            "maxdd_cost_delta": maxdd_delta_mean,
            "post_peak_decay_delta": decay_delta_mean,
        },
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_promotion_judgment.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (analysis_dir / f"{REPORT_STEM}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out_dir / "paired_summary.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Full-Daily Sampler Paired Promotion Judgment",
        "",
        f"- Created: {summary['created_at']}",
        f"- Verdict: `{verdict}`",
        f"- Mechanics ok: `{mechanics_ok}`",
        f"- HC-6 pass count: `{hc6_pass_count}/4`",
        "",
        "## Mean Deltas",
        "",
        "| Metric | Mean delta | Rule |",
        "|---|---:|---|",
        f"| gap improvement (`sampled - full`) | {fmt(gap_mean)} | >= 0.015 |",
        f"| RankIC (`full - sampled`) | {fmt(rankic_delta_mean)} | >= 0 |",
        f"| RankICIR (`full - sampled`) | {fmt(rankicir_delta_mean)} | >= -0.010 |",
        f"| IR with cost (`full - sampled`) | {fmt(ir_delta_mean)} | non-degrading for HC-6 |",
        f"| MaxDD with cost (`full - sampled`) | {fmt(maxdd_delta_mean)} | non-degrading for HC-6 |",
        f"| post-peak decay (`full - sampled`) | {fmt(decay_delta_mean)} | <= 0 for HC-6 |",
        "",
        "## Per-Seed Pairs",
        "",
        "| Seed | sampled ckpt | full ckpt | sampled RankIC | full RankIC | RankIC delta | sampled gap | full gap | gap improvement | full coverage gap | full duplicate rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            "| {seed} | {sckpt} | {fckpt} | {sr} | {fr} | {rd} | {sg} | {fg} | {gi} | {cg} | {dup} |".format(
                seed=r["seed"],
                sckpt=r.get("sampled_checkpoint_epoch", ""),
                fckpt=r.get("full_checkpoint_epoch", ""),
                sr=fmt(r.get("sampled_rankic")),
                fr=fmt(r.get("full_rankic")),
                rd=fmt(r.get("rankic_delta")),
                sg=fmt(r.get("sampled_rankic_gap")),
                fg=fmt(r.get("full_rankic_gap")),
                gi=fmt(r.get("gap_improvement")),
                cg=fmt(r.get("full_coverage_gap")),
                dup=fmt(r.get("full_duplicate_rate")),
            )
        )
    md_lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Promote only if mean gap improvement >= 0.015.",
            "- Mean RankIC must be non-degrading.",
            "- Mean RankICIR must not fall by more than 0.010.",
            "- At least 3 of 4 HC-6 tuple elements must be non-degrading.",
            "- No seed may lose more than 0.004 RankIC.",
            "- Full-daily mechanics must show coverage gap <= 0.005 and duplicate rate <= 0.005.",
        ]
    )
    md = "\n".join(md_lines) + "\n"
    (out_dir / "paired_promotion_judgment.md").write_text(md, encoding="utf-8")
    (analysis_dir / f"{REPORT_STEM}.md").write_text(md, encoding="utf-8")
    return summary


def run_hook(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    analysis_dir = (root / "analysis").resolve()
    python_exe = args.python_exe or sys.executable

    manifests: List[Dict[str, Any]] = []
    # Full seed 42 first, then complete paired controls/candidates.
    arms = [
        (42, "full_daily"),
        (42, "sampled_daily"),
        (43, "sampled_daily"),
        (43, "full_daily"),
        (44, "sampled_daily"),
        (44, "full_daily"),
    ]
    for seed, sampler_mode in arms:
        manifest = run_arm(
            root=root,
            out_dir=out_dir,
            python_exe=python_exe,
            seed=seed,
            sampler_mode=sampler_mode,
            continue_finished=not args.no_continue,
        )
        manifests.append(manifest)
        judge(manifests, out_dir, analysis_dir)
        if manifest.get("returncode") != 0 and args.stop_on_failure:
            return int(manifest.get("returncode") or 1)

    summary = judge(manifests, out_dir, analysis_dir)
    print(f"[{now_str()}] paired judgment: {summary['verdict']}", flush=True)
    return 0 if summary.get("all_pairs_ok") else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 40-epoch full_daily sampler paired falsifier and judge promotion.")
    p.add_argument("--root", default=".", help="Repository root.")
    p.add_argument("--out-dir", default="diagnostic_runs/full_daily_sampler_paired_bestvalid_20260525")
    p.add_argument("--python-exe", default="", help="Python executable for work_flow.py. Defaults to current interpreter.")
    p.add_argument("--no-continue", action="store_true", help="Do not skip completed manifests.")
    p.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed arm.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_hook(parse_args()))
