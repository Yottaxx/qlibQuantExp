#!/usr/bin/env python3
"""
Aggregate KDD sweep results from MLflow artifact folders.

Design goals
-----------
- No dependency on torch / qlib runtime objects.
- Use the run bookkeeping in `run_matrix.yaml`.
- Read per-run artifacts written by `work_flow.py -> generate_paper_report()`:
  - artifacts/run_summary      (pickle; dict of metrics)
  - optional diagnostics from `model.export_visuals()`:
      st_disentangle_gate_series
      st_disentangle_gate_entropy_series
      st_disentangle_time_tau_series
      st_disentangle_time_half_life_series
      st_disentangle_factor_gate_entropy_series
      st_disentangle_factor_gate_topk_mass_10_series

Usage
-----
python scripts/aggregate_kdd_runs.py --matrix run_matrix.yaml --out_dir analysis/kdd
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


DEFAULT_METRICS = [
    # Signal metrics
    "ric_mean",
    "ricir",
    "ric_hac_t",
    "ic_mean",
    "icir",
    "ic_hac_t",
    # Portfolio metrics
    "ann_ret",
    "info_ratio",
    "max_dd",
    "turnover",
]

DIAG_KEYS = {
    "time_ratio": "st_disentangle_gate_series",
    "gate_entropy": "st_disentangle_gate_entropy_series",
    "time_tau": "st_disentangle_time_tau_series",
    "time_half_life": "st_disentangle_time_half_life_series",
    "factor_gate_entropy": "st_disentangle_factor_gate_entropy_series",
    "factor_gate_topk_mass_10": "st_disentangle_factor_gate_topk_mass_10_series",
}


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML root: expected dict, got {type(obj)}")
    return obj


def _find_experiment_id(mlruns_dir: Path, experiment_name: str) -> str:
    # MLflow experiment folder layout:
    # mlruns/<experiment_id>/meta.yaml contains `name: <experiment_name>`
    for p in mlruns_dir.iterdir():
        if not p.is_dir():
            continue
        meta = p / "meta.yaml"
        if not meta.exists():
            continue
        try:
            d = yaml.safe_load(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and str(d.get("name", "")) == str(experiment_name):
            return p.name
    raise KeyError(
        f"Experiment '{experiment_name}' not found under '{mlruns_dir}'. "
        "Check `universes.*.experiment_name` in run_matrix.yaml or your MLflow storage."
    )


def _as_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return float("nan")


def _series_stats(s: Any) -> Dict[str, float]:
    if s is None:
        return {}
    if isinstance(s, pd.Series):
        s2 = s.dropna()
        if len(s2) == 0:
            return {}
        return {
            "mean": float(s2.mean()),
            "std": float(s2.std()),
            "p10": float(s2.quantile(0.10)),
            "p90": float(s2.quantile(0.90)),
        }
    return {}


@dataclass(frozen=True)
class RunRef:
    universe: str
    setting: str
    seed: int
    experiment_name: str
    run_id: str


def _iter_runs(cfg: dict) -> Tuple[List[int], List[RunRef]]:
    seeds = cfg.get("seeds", None)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("run_matrix.yaml must define a non-empty top-level `seeds` list.")
    seeds_int = [int(s) for s in seeds]

    universes = cfg.get("universes", None)
    if not isinstance(universes, dict) or not universes:
        raise ValueError("run_matrix.yaml must define `universes` as a non-empty dict.")

    run_groups = cfg.get("run_groups", None)
    if not isinstance(run_groups, list) or not run_groups:
        raise ValueError("run_matrix.yaml must define `run_groups` as a non-empty list.")

    out: List[RunRef] = []
    for g in run_groups:
        if not isinstance(g, dict):
            continue
        setting = str(g.get("setting", "")).strip()
        if not setting:
            raise ValueError("Each run_groups item must have a non-empty `setting`.")

        runs = g.get("runs", None)
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"run_groups[{setting}] must have a non-empty `runs` list.")

        for r in runs:
            if not isinstance(r, dict):
                continue
            universe = str(r.get("universe", "")).strip()
            if universe not in universes:
                raise KeyError(f"Unknown universe '{universe}' in run_groups[{setting}].")

            exp_name = r.get("experiment_name", None) or universes[universe].get("experiment_name", None)
            if not exp_name:
                raise ValueError(
                    f"Missing experiment_name for universe='{universe}'. "
                    "Set `universes.<name>.experiment_name` or per-run override."
                )

            run_ids = r.get("run_ids", None)
            if not isinstance(run_ids, dict):
                raise ValueError(f"run_groups[{setting}].runs[{universe}] must define `run_ids` as a dict.")

            for seed in seeds_int:
                rid = run_ids.get(seed, None)
                if rid is None:
                    rid = run_ids.get(str(seed), None)
                if rid is None or str(rid).strip() in {"", "null", "None", "TODO"}:
                    continue
                out.append(
                    RunRef(
                        universe=universe,
                        setting=setting,
                        seed=int(seed),
                        experiment_name=str(exp_name),
                        run_id=str(rid),
                    )
                )

    return seeds_int, out


def _artifact_dir_for_run(mlruns_dir: Path, experiment_name: str, run_id: str) -> Path:
    exp_id = _find_experiment_id(mlruns_dir, experiment_name)
    return mlruns_dir / exp_id / run_id / "artifacts"


def _format_mean_std(mean: float, std: float, *, digits: int, pct: bool) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std):
        std = float("nan")
    if pct:
        return f"{mean*100:.2f}±{std*100:.2f}%"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def _format_table(df_agg: pd.DataFrame) -> pd.DataFrame:
    # Expect df_agg columns: metric_mean, metric_std, n_seeds, seeds_missing
    # Produce a compact view.
    out = pd.DataFrame(index=df_agg.index)
    out["n"] = df_agg["n_seeds"]
    out["missing_seeds"] = df_agg["seeds_missing"]

    def f(metric: str, digits: int = 4, pct: bool = False) -> List[str]:
        m = df_agg.get(f"{metric}_mean", pd.Series(index=df_agg.index, dtype=float))
        s = df_agg.get(f"{metric}_std", pd.Series(index=df_agg.index, dtype=float))
        return [_format_mean_std(float(a), float(b), digits=digits, pct=pct) for a, b in zip(m, s)]

    out["RankIC"] = f("ric_mean", digits=4, pct=False)
    out["RankIC IR"] = f("ricir", digits=2, pct=False)
    out["RankIC HAC-t"] = f("ric_hac_t", digits=1, pct=False)
    out["IC"] = f("ic_mean", digits=4, pct=False)
    out["ICIR"] = f("icir", digits=2, pct=False)
    out["AnnRet"] = f("ann_ret", digits=2, pct=True)
    out["InfoRatio"] = f("info_ratio", digits=2, pct=False)
    out["MaxDD"] = f("max_dd", digits=2, pct=True)
    out["Turnover"] = f("turnover", digits=2, pct=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=str, default="run_matrix.yaml", help="Path to run_matrix.yaml")
    ap.add_argument("--out_dir", type=str, default="analysis/kdd", help="Output directory for aggregated tables")
    ap.add_argument("--strict", action="store_true", help="Fail if any referenced run is missing artifacts")
    ap.add_argument(
        "--include_diag",
        action="store_true",
        help="Also load st_disentangle_* series (if present) and export summary stats.",
    )
    args = ap.parse_args(argv)

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    cfg = _load_yaml(matrix_path)
    mlruns_dir = Path(cfg.get("mlruns_dir", "mlruns"))
    if not mlruns_dir.is_absolute():
        mlruns_dir = (matrix_path.parent / mlruns_dir).resolve()
    if not mlruns_dir.exists():
        raise FileNotFoundError(f"mlruns_dir not found: {mlruns_dir}")

    seeds_all, runs = _iter_runs(cfg)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (matrix_path.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    missing_artifacts: List[str] = []

    for rr in runs:
        art_dir = _artifact_dir_for_run(mlruns_dir, rr.experiment_name, rr.run_id)
        if not art_dir.exists():
            msg = f"[Missing] artifacts dir not found: exp='{rr.experiment_name}' run_id='{rr.run_id}' ({rr.universe}/{rr.setting}/seed={rr.seed})"
            missing_artifacts.append(msg)
            if args.strict:
                raise FileNotFoundError(msg)
            continue

        row: Dict[str, Any] = {
            "universe": rr.universe,
            "setting": rr.setting,
            "seed": rr.seed,
            "experiment_name": rr.experiment_name,
            "run_id": rr.run_id,
            "artifacts_dir": str(art_dir),
        }

        rs_path = art_dir / "run_summary"
        if rs_path.exists():
            try:
                rs = _read_pickle(rs_path)
                if isinstance(rs, dict):
                    for k, v in rs.items():
                        row[k] = _as_float(v) if not isinstance(v, (str, dict, list)) else v
                else:
                    row["run_summary_type"] = str(type(rs))
            except Exception as e:
                msg = f"[Missing] failed to load run_summary for {rr.run_id}: {type(e).__name__} {e}"
                missing_artifacts.append(msg)
                if args.strict:
                    raise
        else:
            msg = f"[Missing] run_summary not found for {rr.run_id} ({rr.universe}/{rr.setting}/seed={rr.seed})"
            missing_artifacts.append(msg)
            if args.strict:
                raise FileNotFoundError(msg)

        if args.include_diag:
            for short, key in DIAG_KEYS.items():
                p = art_dir / key
                if not p.exists():
                    continue
                try:
                    s = _read_pickle(p)
                except Exception:
                    continue
                st = _series_stats(s)
                for sk, sv in st.items():
                    row[f"diag_{short}_{sk}"] = sv

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs.csv", index=False)

    # Aggregation per (universe, setting)
    if len(df) == 0:
        (out_dir / "kdd_aggregate.md").write_text(
            f"# KDD Aggregate\n\nNo runs found. Fill `run_ids` in `{matrix_path}`.\n",
            encoding="utf-8",
        )
        if missing_artifacts:
            (out_dir / "missing.txt").write_text("\n".join(missing_artifacts) + "\n", encoding="utf-8")
        return 0

    metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    group_cols = ["universe", "setting"]

    agg = (
        df.groupby(group_cols, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .sort_index()
    )

    # Flatten MultiIndex columns: (metric, stat) -> f"{metric}_{stat}"
    agg.columns = [f"{a}_{b}" for a, b in agg.columns.to_list()]
    agg = agg.reset_index()

    # Seed coverage per group
    seeds_done = df.groupby(group_cols)["seed"].apply(lambda x: sorted({int(s) for s in x.tolist()})).reset_index()
    agg = agg.merge(seeds_done, on=group_cols, how="left")
    agg = agg.rename(columns={"seed": "seeds_done"})
    agg["n_seeds"] = agg["seeds_done"].apply(lambda xs: int(len(xs)) if isinstance(xs, list) else 0)
    agg["seeds_missing"] = agg["seeds_done"].apply(
        lambda xs: ",".join(str(s) for s in seeds_all if s not in set(xs or []))
    )

    # Export aggregate CSV
    agg.to_csv(out_dir / "aggregate.csv", index=False)

    # Write Markdown summary
    lines: List[str] = []
    lines.append("# KDD Aggregate\n")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Matrix: `{matrix_path}`")
    lines.append(f"- MLflow: `{mlruns_dir}`")
    lines.append("")

    for universe in sorted(df["universe"].dropna().unique().tolist()):
        lines.append(f"## {universe}\n")
        sub = agg[agg["universe"] == universe].copy()
        sub = sub.set_index("setting").sort_index()
        tbl = _format_table(sub)
        lines.append(tbl.to_markdown())
        lines.append("")

    if missing_artifacts:
        (out_dir / "missing.txt").write_text("\n".join(missing_artifacts) + "\n", encoding="utf-8")
        lines.append("## Missing / Skipped\n")
        lines.extend([f"- {m}" for m in missing_artifacts[:50]])
        if len(missing_artifacts) > 50:
            lines.append(f"- ... ({len(missing_artifacts) - 50} more)")
        lines.append("")

    (out_dir / "kdd_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

