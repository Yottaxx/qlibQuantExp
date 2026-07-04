#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a bottleneck report from existing qibMacV2 experiment artifacts.

The script is intentionally read-only with respect to experiment inputs.  It
uses only the Python standard library so it can run in minimal research
environments where pandas is not installed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RANKIC_COL = "performance/daily_rank_ic_mean"
RANKICIR_COL = "performance/rank_icir"
IC_COL = "performance/daily_ic_mean"
IR_COL = "portfolio/information_ratio_with_cost"
ANN_COL = "portfolio/annualized_return_with_cost"
DD_COL = "portfolio/max_drawdown_with_cost"
TURNOVER_COL = "portfolio/turnover"
SPREAD_COL = "portfolio/top_bottom_spread"
GAP_COL = "optimization/rank_ic_gap_last"
DECAY_COL = "optimization/post_peak_decay"
LOSS_GAP_COL = "optimization/loss_gap_last"
TRAIN_SCORE_STD_COL = "optimization/train_score_std_last"
VALID_SCORE_STD_COL = "optimization/valid_score_std_last"
GRAD_NF_COL = "optimization/train_grad_nonfinite_rate_last"
GRAD_SKIP_COL = "optimization/train_grad_skipped_rate_last"
TIME_RATIO_COL = "router/time_ratio_mean"
ENTROPY_COL = "router/entropy_norm_mean"
COLLAPSE_COL = "router/collapse_ratio"
PC1_SPREAD_COL = "regime/pc1_tail_2x2_rank_ic_spread"

STRUCTURAL_SETTINGS = [
    "base",
    "time_only",
    "film_only",
    "fixed_router_05",
    "full_135",
    "no_layer_summary",
]

METRIC_PREFIXES = (
    "performance/",
    "portfolio/",
    "optimization/",
    "router/",
    "time_embedding/",
    "factor_film/",
    "attention_pooling/",
    "regime/",
)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        val = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def is_nonzero_returncode(row: Dict[str, str]) -> bool:
    rc = str(row.get("returncode", "")).strip()
    if not rc or rc.lower() in {"nan", "none", "null"}:
        return False
    try:
        return int(float(rc)) != 0
    except ValueError:
        return rc != "0"


def is_success_metric_row(row: Dict[str, str]) -> bool:
    return to_float(row.get(RANKIC_COL)) is not None and not is_nonzero_returncode(row)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in values if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def sample_std(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in values if x is not None and math.isfinite(x)]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def fmt(value: Any, digits: int = 6) -> str:
    val = to_float(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    val = to_float(value)
    if val is None:
        return ""
    return f"{100.0 * val:.{digits}f}%"


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(md_escape(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(x) for x in row) + " |")
    return "\n".join(lines)


def metric_columns(rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    cols = rows[0].keys()
    return [c for c in cols if any(c.startswith(prefix) for prefix in METRIC_PREFIXES)]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def correlations(rows: Sequence[Dict[str, str]], cols: Sequence[str], target_col: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for col in cols:
        if col == target_col:
            continue
        pairs: List[Tuple[float, float]] = []
        for row in rows:
            x = to_float(row.get(col))
            y = to_float(row.get(target_col))
            if x is not None and y is not None:
                pairs.append((x, y))
        if len(pairs) < 8:
            continue
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        if r is None:
            continue
        out.append({"metric": col, "r": r, "n": len(pairs), "abs_r": abs(r)})
    out.sort(key=lambda x: x["abs_r"], reverse=True)
    return out


def group_aggregate(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("canonical_setting") or row.get("setting") or "unknown"].append(row)

    out: List[Dict[str, Any]] = []
    for setting, rs in grouped.items():
        rankics = [to_float(r.get(RANKIC_COL)) for r in rs]
        item = {
            "setting": setting,
            "n": len([x for x in rankics if x is not None]),
            "seeds": sorted({str(r.get("seed", "")).strip() for r in rs if str(r.get("seed", "")).strip()}),
            "rankic_mean": mean(rankics),
            "rankic_std": sample_std(rankics),
            "rankicir_mean": mean(to_float(r.get(RANKICIR_COL)) for r in rs),
            "ic_mean": mean(to_float(r.get(IC_COL)) for r in rs),
            "portfolio_ir_mean": mean(to_float(r.get(IR_COL)) for r in rs),
            "ann_return_mean": mean(to_float(r.get(ANN_COL)) for r in rs),
            "max_drawdown_mean": mean(to_float(r.get(DD_COL)) for r in rs),
            "turnover_mean": mean(to_float(r.get(TURNOVER_COL)) for r in rs),
            "gap_mean": mean(to_float(r.get(GAP_COL)) for r in rs),
            "decay_mean": mean(to_float(r.get(DECAY_COL)) for r in rs),
            "time_ratio_mean": mean(to_float(r.get(TIME_RATIO_COL)) for r in rs),
            "entropy_mean": mean(to_float(r.get(ENTROPY_COL)) for r in rs),
            "collapse_mean": mean(to_float(r.get(COLLAPSE_COL)) for r in rs),
            "rows": rs,
        }
        out.append(item)
    out.sort(key=lambda x: x["rankic_mean"] if x["rankic_mean"] is not None else -999.0, reverse=True)
    return out


def row_path_blob(row: Dict[str, str]) -> str:
    keys = ["path", "stdout", "stderr", "csv_path", "diagnostic_matrix"]
    return " ".join(str(row.get(k, "")) for k in keys)


def select_structural_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    candidates: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for row in rows:
        setting = row.get("canonical_setting") or row.get("setting") or ""
        if setting not in STRUCTURAL_SETTINGS:
            continue
        if to_float(row.get(RANKIC_COL)) is None or is_nonzero_returncode(row):
            continue
        blob = row_path_blob(row).replace("/", "\\")
        score = 0
        if "matrix_40epoch_seed42" in blob:
            score += 100
        if row.get("source") == "diagnostic_manifest":
            score += 20
        if str(row.get("seed", "")).strip() == "42":
            score += 10
        if "suite_csv_fill" in str(row.get("source", "")):
            score -= 5
        candidates[setting].append((score, row))
    out: Dict[str, Dict[str, str]] = {}
    for setting, items in candidates.items():
        items.sort(key=lambda x: (x[0], to_float(x[1].get(RANKIC_COL)) or -999.0), reverse=True)
        out[setting] = items[0][1]
    return out


def structural_deltas(structural: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    vals = {k: to_float(v.get(RANKIC_COL)) for k, v in structural.items()}
    rankicir = {k: to_float(v.get(RANKICIR_COL)) for k, v in structural.items()}
    base = vals.get("base")
    full = vals.get("full_135")
    out: Dict[str, Any] = {
        "rankic": vals,
        "rankicir": rankicir,
        "delta_vs_base": {},
        "film_share_of_full_delta": None,
        "time_share_of_full_delta": None,
        "full_minus_fixed_rankic": None,
        "full_minus_fixed_rankicir": None,
    }
    if base is not None:
        out["delta_vs_base"] = {k: (v - base if v is not None else None) for k, v in vals.items()}
    if base is not None and full is not None and full != base:
        full_delta = full - base
        film = vals.get("film_only")
        time_only = vals.get("time_only")
        if film is not None:
            out["film_share_of_full_delta"] = (film - base) / full_delta
        if time_only is not None:
            out["time_share_of_full_delta"] = (time_only - base) / full_delta
    fixed = vals.get("fixed_router_05")
    if full is not None and fixed is not None:
        out["full_minus_fixed_rankic"] = full - fixed
    full_ir = rankicir.get("full_135")
    fixed_ir = rankicir.get("fixed_router_05")
    if full_ir is not None and fixed_ir is not None:
        out["full_minus_fixed_rankicir"] = full_ir - fixed_ir
    return out


def cap_number(setting: str) -> int:
    if setting.startswith("cap") and len(setting) >= 5:
        try:
            return int(setting[3:5])
        except ValueError:
            return 999
    return 999


def capacity_rows(aggregates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    caps = [a for a in aggregates if str(a.get("setting", "")).startswith("cap")]
    caps.sort(key=lambda x: cap_number(str(x.get("setting", ""))))
    return caps


def failed_runs(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if is_nonzero_returncode(r)]


def source_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    return dict(Counter((r.get("source") or "unknown") for r in rows))


def top_rows(rows: Sequence[Dict[str, str]], n: int = 12) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: to_float(r.get(RANKIC_COL)) or -999.0, reverse=True)[:n]


def load_inputs(root: Path) -> Dict[str, Any]:
    analysis_dir = root / "analysis"
    return {
        "experiment_rows": read_csv_rows(analysis_dir / "experiment_inventory_unified.csv"),
        "aggregate_rows": read_csv_rows(analysis_dir / "aggregate_by_setting_unified.csv"),
        "analysis_summary": read_json(analysis_dir / "analysis_summary.json"),
        "suite_summary_paths": sorted(root.glob("diagnostic_runs/**/suite*_summary.csv")),
        "experiments_ledger_rows": read_jsonl(root / "experiments_ledger.jsonl"),
        "qx_ledger_text": (root / ".agents/skills/qlibQuantExp-skills/shared/ledger.md").read_text(
            encoding="utf-8"
        )
        if (root / ".agents/skills/qlibQuantExp-skills/shared/ledger.md").exists()
        else "",
    }


def make_bottleneck_ranking(
    aggregates: Sequence[Dict[str, Any]],
    structural_delta: Dict[str, Any],
    corr: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_setting = {a["setting"]: a for a in aggregates}
    top = aggregates[:5]
    top_gaps = [x.get("gap_mean") for x in top if x.get("gap_mean") is not None]
    avg_top_gap = mean(top_gaps)

    ranking: List[Dict[str, Any]] = []
    if avg_top_gap is not None and avg_top_gap >= 0.10:
        ranking.append(
            {
                "rank": 1,
                "bottleneck": "generalization_gap",
                "evidence": f"Top settings have mean train-valid RankIC gap around {avg_top_gap:.3f}.",
                "next_direction": "Keep full135 settings; improve deterministic runtime, sampler parity, and checkpoint discipline before changing model family.",
            }
        )

    time_share = structural_delta.get("time_share_of_full_delta")
    if time_share is not None and time_share < 0.20:
        evidence = f"time_only explains only {100.0 * time_share:.1f}% of full_135 - base RankIC delta in the seed42 structural matrix."
    else:
        time_delta = structural_delta.get("delta_vs_base", {}).get("time_only")
        evidence = f"time_only delta vs base is {time_delta:.6f} in the seed42 structural matrix." if time_delta is not None else "time_only has weak evidence."
    ranking.append(
        {
            "rank": len(ranking) + 1,
            "bottleneck": "weak_temporal_path",
            "evidence": evidence,
            "next_direction": "Instrument time expert contribution and temporal readout before adding temporal architecture.",
        }
    )

    full_minus_fixed = structural_delta.get("full_minus_fixed_rankic")
    full_minus_fixed_ir = structural_delta.get("full_minus_fixed_rankicir")
    if full_minus_fixed is not None and full_minus_fixed_ir is not None:
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "bottleneck": "router_edge_not_robust",
                "evidence": (
                    f"full_135 beats fixed_router_05 by RankIC {full_minus_fixed:.6f}, "
                    f"but RankICIR delta is {full_minus_fixed_ir:.6f}."
                ),
                "next_direction": "Add router/expert-winner diagnostics; do not make router more expressive yet.",
            }
        )

    cap06 = by_setting.get("cap06_d96_h4_l2_ff384_do020")
    if cap06 and int(cap06.get("n") or 0) == 1:
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "bottleneck": "capacity_signal_single_seed",
                "evidence": f"cap06 is strong at RankIC {cap06.get('rankic_mean'):.6f}, but n=1.",
                "next_direction": "Do not promote cap06 to default; keep full135 until deterministic baseline is locked.",
            }
        )

    if corr:
        top_corr = corr[0]
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "bottleneck": "regime_alignment_needs_measurement",
                "evidence": f"Strong monitored association: {top_corr['metric']} vs RankIC r={top_corr['r']:.3f} (n={top_corr['n']}).",
                "next_direction": "Use existing regime/router metrics to explain behavior; do not change regime state in this step.",
            }
        )

    for i, item in enumerate(ranking, start=1):
        item["rank"] = i
    return ranking


def make_report(root: Path) -> Tuple[str, Dict[str, Any]]:
    inputs = load_inputs(root)
    exp_rows = inputs["experiment_rows"]
    success_rows = [r for r in exp_rows if is_success_metric_row(r)]
    aggregates = group_aggregate(success_rows)
    top = top_rows(success_rows, 12)
    mcols = metric_columns(exp_rows)
    corr = correlations(success_rows, mcols, RANKIC_COL)
    structural = select_structural_rows(success_rows)
    sdelta = structural_deltas(structural)
    caps = capacity_rows(aggregates)
    failed = failed_runs(exp_rows)
    suite_paths = inputs["suite_summary_paths"]
    bottlenecks = make_bottleneck_ranking(aggregates, sdelta, corr)

    counts = {
        "experiment_rows": len(exp_rows),
        "valid_rankic_rows": len(success_rows),
        "aggregate_rows": len(inputs["aggregate_rows"]),
        "suite_summary_files": len(suite_paths),
        "experiments_ledger_rows": len(inputs["experiments_ledger_rows"]),
        "source_counts": source_counts(exp_rows),
    }

    top_table_rows = []
    for row in top:
        top_table_rows.append(
            [
                row.get("canonical_setting") or row.get("setting"),
                row.get("seed", ""),
                fmt(row.get(RANKIC_COL)),
                fmt(row.get(RANKICIR_COL)),
                fmt(row.get(IC_COL)),
                fmt(row.get(IR_COL)),
                fmt(row.get(ANN_COL)),
                fmt(row.get(DD_COL)),
                fmt(row.get(GAP_COL)),
                fmt(row.get(DECAY_COL)),
            ]
        )

    agg_table_rows = []
    for item in aggregates[:20]:
        agg_table_rows.append(
            [
                item["setting"],
                item["n"],
                ",".join(item["seeds"]),
                fmt(item["rankic_mean"]),
                fmt(item["rankic_std"]),
                fmt(item["rankicir_mean"]),
                fmt(item["portfolio_ir_mean"]),
                fmt(item["gap_mean"]),
                fmt(item["decay_mean"]),
                fmt(item["time_ratio_mean"]),
                fmt(item["entropy_mean"]),
            ]
        )

    structural_rows = []
    for setting in STRUCTURAL_SETTINGS:
        row = structural.get(setting, {})
        delta = sdelta.get("delta_vs_base", {}).get(setting)
        structural_rows.append(
            [
                setting,
                row.get("seed", ""),
                fmt(row.get(RANKIC_COL)),
                fmt(delta),
                fmt(row.get(RANKICIR_COL)),
                fmt(row.get(IR_COL)),
                fmt(row.get(GAP_COL)),
                fmt(row.get(DECAY_COL)),
                fmt(row.get(TIME_RATIO_COL)),
                fmt(row.get(ENTROPY_COL)),
            ]
        )

    cap_rows = []
    for item in caps:
        first = item.get("rows", [{}])[0]
        cap_rows.append(
            [
                item["setting"],
                item["n"],
                ",".join(item["seeds"]),
                first.get("d_model", ""),
                first.get("n_heads", ""),
                first.get("n_layers", ""),
                first.get("d_ff", ""),
                first.get("dropout", ""),
                fmt(item["rankic_mean"]),
                fmt(item["rankic_std"]),
                fmt(item["rankicir_mean"]),
                fmt(item["gap_mean"]),
                fmt(item["decay_mean"]),
            ]
        )

    corr_rows = [
        [c["metric"], f"{c['r']:.3f}", c["n"]]
        for c in corr[:18]
        if c["metric"] != RANKIC_COL
    ]

    failed_rows = [
        [
            r.get("canonical_setting") or r.get("setting"),
            r.get("seed", ""),
            r.get("returncode", ""),
            r.get("stderr", "") or r.get("path", ""),
        ]
        for r in failed
    ]

    confirmed: List[str] = []
    hints: List[str] = []
    insufficient: List[str] = []
    hypotheses: List[str] = []

    film_share = sdelta.get("film_share_of_full_delta")
    if film_share is not None and film_share >= 0.5:
        confirmed.append(
            f"film_only explains {100.0 * film_share:.1f}% of the seed42 full_135 - base RankIC delta."
        )
    elif film_share is not None:
        hints.append(
            f"film_only explains {100.0 * film_share:.1f}% of the seed42 full_135 - base RankIC delta."
        )

    time_delta = sdelta.get("delta_vs_base", {}).get("time_only")
    if time_delta is not None and time_delta < 0.0015:
        confirmed.append(f"time_only is weak in the seed42 matrix: delta vs base is {time_delta:.6f}.")
    elif time_delta is not None:
        hints.append(f"time_only delta vs base is {time_delta:.6f}; still materially below full_135 delta.")

    full_minus_fixed = sdelta.get("full_minus_fixed_rankic")
    full_minus_fixed_ir = sdelta.get("full_minus_fixed_rankicir")
    if full_minus_fixed is not None and full_minus_fixed_ir is not None:
        if full_minus_fixed > 0 and full_minus_fixed_ir < 0:
            confirmed.append(
                f"learned router beats fixed_router_05 in RankIC by {full_minus_fixed:.6f}, but fixed router has better RankICIR by {-full_minus_fixed_ir:.6f}."
            )
        else:
            hints.append("learned router comparison is mixed and needs tighter diagnostics.")

    top_gaps = [a.get("gap_mean") for a in aggregates[:8] if a.get("gap_mean") is not None]
    avg_gap = mean(top_gaps)
    if avg_gap is not None and avg_gap >= 0.10:
        confirmed.append(f"Top settings show large train-valid RankIC gap: mean gap {avg_gap:.3f}.")

    cap06 = next((a for a in aggregates if a["setting"] == "cap06_d96_h4_l2_ff384_do020"), None)
    if cap06:
        if cap06["n"] == 1:
            hints.append(
                f"cap06 is the best aggregate setting by RankIC ({cap06['rankic_mean']:.6f}), but it is single-seed."
            )
            insufficient.append("cap06 cannot become the new default without more evidence; keep full135 for now.")

    if failed_rows:
        confirmed.append(f"{len(failed_rows)} nonzero-returncode rows found; cap08/depth-3 remains unevaluated.")

    insufficient.append("Most capacity settings have n=1; capacity scaling is not established.")
    insufficient.append("The qx ledger is not yet a complete mirror of the May diagnostic runs.")
    hypotheses.append("Sampler train/eval mismatch may be consuming small RankIC gains; instrument duplicate rate before changing architecture.")
    hypotheses.append("Temporal path may need measurement or readout fixes, but no temporal architecture change is justified yet.")
    hypotheses.append("Router calibration should be diagnosed against expert-winner advantage before adding router capacity.")

    data = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "experiment_inventory": "analysis/experiment_inventory_unified.csv",
            "aggregate_by_setting": "analysis/aggregate_by_setting_unified.csv",
            "analysis_summary": "analysis/analysis_summary.json",
            "suite_summary_files": [str(p) for p in suite_paths],
            "experiments_ledger": "experiments_ledger.jsonl",
            "qx_ledger": ".agents/skills/qlibQuantExp-skills/shared/ledger.md",
        },
        "counts": counts,
        "top_runs": [
            {
                "setting": r.get("canonical_setting") or r.get("setting"),
                "seed": r.get("seed", ""),
                "rankic": to_float(r.get(RANKIC_COL)),
                "rankicir": to_float(r.get(RANKICIR_COL)),
                "portfolio_ir": to_float(r.get(IR_COL)),
                "gap": to_float(r.get(GAP_COL)),
                "decay": to_float(r.get(DECAY_COL)),
                "path": r.get("path", ""),
            }
            for r in top
        ],
        "aggregates": [
            {k: v for k, v in a.items() if k != "rows"}
            for a in aggregates
        ],
        "structural_deltas": sdelta,
        "correlations": corr,
        "failed_runs": failed_rows,
        "evidence_classification": {
            "confirmed_facts": confirmed,
            "single_seed_hints": hints,
            "insufficient_evidence": insufficient,
            "needs_new_experiment": hypotheses,
        },
        "bottleneck_ranking": bottlenecks,
        "recommended_direction": [
            "Keep full135 settings.",
            "Do not change regime state in the next step.",
            "First implement deterministic seed/CUDA handling and sampler parity instrumentation.",
            "Add router/expert monitoring and temporal-path diagnostics before changing loss, capacity, or macV3-style architecture.",
        ],
    }

    lines: List[str] = []
    lines.append("# Current Experiment Bottleneck Report")
    lines.append("")
    lines.append(f"Generated: {data['generated_at']}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"Parsed {counts['experiment_rows']} experiment inventory rows and {counts['valid_rankic_rows']} rows with valid RankIC. "
        "The current evidence supports keeping full135 as the working setting, not promoting cap06 or changing regime state."
    )
    lines.append("")
    for item in bottlenecks:
        lines.append(f"- {item['rank']}. `{item['bottleneck']}`: {item['evidence']} Next: {item['next_direction']}")
    lines.append("")
    lines.append("## Inputs And Coverage")
    lines.append("")
    lines.append(md_table(["Input", "Count"], [
        ["experiment inventory rows", counts["experiment_rows"]],
        ["valid RankIC rows", counts["valid_rankic_rows"]],
        ["aggregate rows", counts["aggregate_rows"]],
        ["suite summary files", counts["suite_summary_files"]],
        ["experiments ledger rows", counts["experiments_ledger_rows"]],
    ]))
    lines.append("")
    lines.append("Source distribution: `" + json.dumps(counts["source_counts"], sort_keys=True) + "`")
    lines.append("")
    lines.append("## 1. Effect Metrics")
    lines.append("")
    lines.append("Top existing runs by RankIC:")
    lines.append("")
    lines.append(md_table(
        ["setting", "seed", "RankIC", "RankICIR", "IC", "port IR", "ann ret", "max DD", "gap", "decay"],
        top_table_rows,
    ))
    lines.append("")
    lines.append("Aggregate by canonical setting:")
    lines.append("")
    lines.append(md_table(
        ["setting", "n", "seeds", "RankIC mean", "RankIC std", "RankICIR", "port IR", "gap", "decay", "time_ratio", "entropy"],
        agg_table_rows,
    ))
    lines.append("")
    lines.append("## 2. Training Process")
    lines.append("")
    lines.append(
        "The strongest settings still show large train-valid RankIC gaps. "
        "This is a bottleneck because most architectural deltas are smaller than the observed gap."
    )
    lines.append("")
    lines.append(md_table(
        ["setting", "n", "RankIC", "gap", "decay", "RankICIR"],
        [[a["setting"], a["n"], fmt(a["rankic_mean"]), fmt(a["gap_mean"]), fmt(a["decay_mean"]), fmt(a["rankicir_mean"])] for a in aggregates[:10]],
    ))
    lines.append("")
    lines.append("## 3. Routing Health")
    lines.append("")
    lines.append(
        "Router collapse is not the main failure mode, but learned routing has not clearly beaten fixed routing on stability. "
        "The monitored correlations below show which diagnostics move with RankIC in existing runs."
    )
    lines.append("")
    lines.append(md_table(["metric", "Pearson r vs RankIC", "n"], corr_rows))
    lines.append("")
    lines.append("## 4. Structural Contribution")
    lines.append("")
    lines.append("Seed42 structural matrix, preferring manifest-backed `matrix_40epoch_seed42` rows:")
    lines.append("")
    lines.append(md_table(
        ["setting", "seed", "RankIC", "delta vs base", "RankICIR", "port IR", "gap", "decay", "time_ratio", "entropy"],
        structural_rows,
    ))
    lines.append("")
    if film_share is not None:
        lines.append(f"- film_only share of full_135 - base RankIC delta: {100.0 * film_share:.1f}%.")
    if sdelta.get("time_share_of_full_delta") is not None:
        lines.append(f"- time_only share of full_135 - base RankIC delta: {100.0 * sdelta['time_share_of_full_delta']:.1f}%.")
    if full_minus_fixed is not None and full_minus_fixed_ir is not None:
        lines.append(f"- full_135 - fixed_router_05 RankIC: {full_minus_fixed:.6f}; RankICIR: {full_minus_fixed_ir:.6f}.")
    lines.append("")
    lines.append("## 5. Capacity Impact")
    lines.append("")
    lines.append("Capacity results are non-monotonic and mostly single-seed. cap06 is promising but not a new default.")
    lines.append("")
    lines.append(md_table(
        ["setting", "n", "seeds", "d_model", "heads", "layers", "d_ff", "dropout", "RankIC", "std", "RankICIR", "gap", "decay"],
        cap_rows,
    ))
    lines.append("")
    if failed_rows:
        lines.append("Failed or incomplete rows:")
        lines.append("")
        lines.append(md_table(["setting", "seed", "returncode", "stderr/path"], failed_rows))
        lines.append("")
    lines.append("## Evidence Classification")
    lines.append("")
    lines.append("### Confirmed Facts")
    lines.extend(f"- {x}" for x in confirmed)
    lines.append("")
    lines.append("### Single-Seed Hints")
    lines.extend(f"- {x}" for x in hints)
    lines.append("")
    lines.append("### Insufficient Evidence")
    lines.extend(f"- {x}" for x in insufficient)
    lines.append("")
    lines.append("### Needs New Experiment")
    lines.extend(f"- {x}" for x in hypotheses)
    lines.append("")
    lines.append("## Bottleneck Ranking And Next Direction")
    lines.append("")
    lines.append(md_table(
        ["rank", "bottleneck", "evidence", "next direction"],
        [[b["rank"], b["bottleneck"], b["evidence"], b["next_direction"]] for b in bottlenecks],
    ))
    lines.append("")
    lines.append("Recommended direction under the current constraint:")
    lines.extend(f"- {x}" for x in data["recommended_direction"])
    lines.append("")
    lines.append("Do not prioritize loss changes, capacity scaling, regime-state changes, or macV3-style bundles until these bottlenecks are closed.")
    lines.append("")

    return "\n".join(lines), data


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze existing qibMacV2 experiment bottlenecks.")
    parser.add_argument("--root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--out-md", default="analysis/current_bottleneck_report.md")
    parser.add_argument("--out-json", default="analysis/current_bottleneck_report.json")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and print summary without writing report files.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report_md, report_json = make_report(root)
    out_md = root / args.out_md
    out_json = root / args.out_json

    if args.dry_run:
        counts = report_json["counts"]
        print(
            "DRY RUN OK: "
            f"rows={counts['experiment_rows']} "
            f"valid_rankic={counts['valid_rankic_rows']} "
            f"suite_files={counts['suite_summary_files']}"
        )
        top = report_json["top_runs"][0] if report_json["top_runs"] else {}
        print(f"top_run={top.get('setting')} seed={top.get('seed')} rankic={top.get('rankic')}")
        print(f"would_write_md={out_md}")
        print(f"would_write_json={out_json}")
        return 0

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report_md, encoding="utf-8")
    out_json.write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
