#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate per-seed pool/readout forensic results into a single report.

Reads analysis/pool_readout_forensics/seed{S}/results.json for the given seeds,
aggregates KEY-SVD-GATE and PROBE-CEILING across seeds, applies the binding
decision rules, and writes aggregate.json + report.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _fmt(x, nd=5):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="analysis/pool_readout_forensics")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    root = Path(args.root)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    runs: List[Dict] = []
    for s in seeds:
        p = root / f"seed{s}" / "results.json"
        if p.exists():
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            print(f"[warn] missing {p}")
    if not runs:
        print("[error] no results found")
        return 1

    def col(path: List[str]) -> List[float]:
        out = []
        for r in runs:
            cur = r
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            out.append(float(cur) if ok and cur is not None else float("nan"))
        return out

    def m(path):
        a = np.asarray([v for v in col(path) if np.isfinite(v)], float)
        return float(a.mean()) if a.size else float("nan")

    # ---- KEY-SVD aggregation ----
    ks_keys = [
        "current_softmax_entropy_norm_mean", "key_s1_over_common_norm_mean",
        "key_sigma2_over_sigma1_mean", "key_effective_rank_mean",
        "best_achievable_entropy_norm_current_temp",
        "value_sigma2_over_sigma1_mean", "value_s1_over_common_norm_mean",
        "crossstock_attn_weight_std_mean", "current_logit_std_mean",
    ]
    key_svd_agg = {k: m(["KEY_SVD_GATE", k]) for k in ks_keys}
    temps = ["T0.5", "T1", "T2", "T3", "T5"]
    key_svd_agg["best_achievable_entropy_by_temp"] = {
        t: m(["KEY_SVD_GATE", "best_achievable_entropy_norm_by_temp", t]) for t in temps
    }
    verdicts = [r.get("KEY_SVD_GATE", {}).get("world_verdict", "?") for r in runs]
    key_svd_agg["verdicts_per_seed"] = verdicts

    # ---- PROBE aggregation ----
    gaps = col(["PROBE_CEILING", "armAprime_minus_arm0"])  # the GATE: input-conditional pooler
    arm0 = col(["PROBE_CEILING", "arm0_meanpool_rank_ic"])
    armA = col(["PROBE_CEILING", "armAprime_conditional_rank_ic"])
    decile = col(["PROBE_CEILING", "armB_slot_ic_decile_ratio"])
    coveff = col(["PROBE_CEILING", "armB_slot_signal_cov_effective_rank"])
    sanity = col(["sanity_valid_rank_ic"])
    gaps_ok = sum(1 for g in gaps if np.isfinite(g) and g >= 0.003)
    probe_agg = {
        "arm0_per_seed": arm0, "armA_per_seed": armA, "gap_per_seed": gaps,
        "gap_mean": m(["PROBE_CEILING", "armA_minus_arm0"]),
        "decile_ratio_mean": m(["PROBE_CEILING", "armB_slot_ic_decile_ratio"]),
        "cov_effrank_mean": m(["PROBE_CEILING", "armB_slot_signal_cov_effective_rank"]),
        "slot_ic_abs_max_mean": m(["PROBE_CEILING", "armB_slot_ic_abs_max"]),
        "n_seeds_gap_ge_0p003": gaps_ok,
        "n_seeds": len(runs),
        "sanity_valid_rank_ic_per_seed": sanity,
    }
    gap_majority = gaps_ok > len(runs) / 2
    disp_ok = (probe_agg["decile_ratio_mean"] >= 2.0)
    cov_ok = (probe_agg["cov_effrank_mean"] >= 2.0)
    probe_agg["sharpening_can_help"] = bool(gap_majority and disp_ok and cov_ok)
    probe_agg["rational_equilibrium_null_supported"] = bool(not (gap_majority and disp_ok))

    # ---- overall world verdict (structure-first; A vs C from PROBE) ----
    s1c = key_svd_agg["key_s1_over_common_norm_mean"]
    if np.isfinite(s1c) and s1c <= 0.05:
        world = "B_input_forced"
    elif np.isfinite(s1c) and s1c >= 0.10:
        world = "operator_state_collapse"  # A vs C decided by PROBE-CEILING below
    else:
        world = "inconclusive"

    agg = {
        "seeds": seeds, "n_runs": len(runs),
        "KEY_SVD_GATE": key_svd_agg,
        "PROBE_CEILING": probe_agg,
        "world_verdict": world,
    }
    (root / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # ---- markdown report ----
    L = []
    L.append("# Pool/readout forensics — aggregate report")
    L.append("")
    L.append(f"- seeds: {seeds}  | runs found: {len(runs)}")
    L.append(f"- sanity valid_rank_ic per seed: {[_fmt(v) for v in sanity]} (promoted anchor ~0.0768)")
    L.append("")
    L.append("## KEY-SVD-GATE (factor pool: A operator / B input / C live-projection)")
    L.append(f"- current softmax entropy (six-nines check): **{_fmt(key_svd_agg['current_softmax_entropy_norm_mean'])}**")
    L.append(f"- key s1/common-norm (across-factor key contrast size): **{_fmt(key_svd_agg['key_s1_over_common_norm_mean'],4)}**")
    L.append(f"- key sigma2/sigma1: {_fmt(key_svd_agg['key_sigma2_over_sigma1_mean'],4)}  | key effective rank: {_fmt(key_svd_agg['key_effective_rank_mean'],2)}")
    bt = key_svd_agg["best_achievable_entropy_by_temp"]
    L.append(f"- peakability entropy (unit-aligned top key dir) at target logit-std: "
             f"0.5={_fmt(bt['T0.5'],4)} 1={_fmt(bt['T1'],4)} 2={_fmt(bt['T2'],4)} 3={_fmt(bt['T3'],4)} 5={_fmt(bt['T5'],4)}")
    L.append(f"- current logit-std (operator coldness): {_fmt(key_svd_agg['current_logit_std_mean'],6)}")
    L.append(f"- value-channel s1/common-norm (world-C: value bank varies across factors?): {_fmt(key_svd_agg['value_s1_over_common_norm_mean'],4)}")
    L.append(f"- cross-stock attention-weight std (is attention even stock-varying?): {_fmt(key_svd_agg['crossstock_attn_weight_std_mean'],6)}")
    L.append(f"- per-seed verdicts: {verdicts}")
    L.append(f"- **aggregate world verdict: {world}**")
    L.append("")
    L.append("## PROBE-CEILING (does non-uniform factor reweighting help on frozen reps?)")
    L.append("| seed | arm0 (mean-pool) | armA (convex-reweight) | gap |")
    L.append("|---|---:|---:|---:|")
    for i, s in enumerate(seeds[:len(runs)]):
        L.append(f"| {s} | {_fmt(arm0[i])} | {_fmt(armA[i])} | {_fmt(gaps[i],5)} |")
    L.append(f"- gap mean: **{_fmt(probe_agg['gap_mean'],5)}**  | seeds with gap>=+0.003: {gaps_ok}/{len(runs)}")
    L.append(f"- Arm B slot-IC decile ratio (mean): {_fmt(probe_agg['decile_ratio_mean'],2)} (need >=2.0)")
    L.append(f"- Arm B slot-signal cov effective rank (mean): {_fmt(probe_agg['cov_effrank_mean'],2)} (need >=2 = non-degenerate)")
    L.append(f"- Arm B max |slot-IC| (mean): {_fmt(probe_agg['slot_ic_abs_max_mean'],4)}")
    L.append(f"- **sharpening_can_help: {probe_agg['sharpening_can_help']}** | "
             f"rational-equilibrium null supported: {probe_agg['rational_equilibrium_null_supported']}")
    L.append("")
    (root / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[done] wrote {root/'aggregate.json'} and {root/'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
