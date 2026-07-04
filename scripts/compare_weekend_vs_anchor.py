#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Weekend 001/002 matrix vs the g012 BASELINE ANCHOR (paired by seed).

Stages: h1_dropext (DropExtremeLabel), h2_stockexp (3-expert), h12_combo. Four-tuple paired deltas
+ stage-specific mechanism reads: 001 -> valid score_std (kill: collapse >30% vs anchor);
002 -> stock_ratio, router_no_stock_delta, router_stock_advantage, cosine triangle.
Kill (pre-registered, per card): paired n=3 dRankIC < +0.003 => kill; mechanism reads explain WHY.
"""
from __future__ import annotations
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# g012 anchor runs (BASELINE_ANCHOR.md)
ANCHOR = {42: "mlruns/716849652326531066/2e8e24fe541a4a31949275b387da2ef2",
          43: "mlruns/686003645691947084/500a105a10b043068d704fa2e93d60da",
          44: "mlruns/520724743245902018/79af86fa4f444f218ae3cfc4425e65da"}
KEYS = {"rank_ic": "perf/daily_rank_ic_mean", "ir": "portfolio/information_ratio_with_cost",
        "maxdd": "portfolio/max_drawdown_with_cost", "ppd": "opt/post_peak_decay"}
LOGDIR = "logs/weekend_001_002"
ANCHOR_LOGS = {s: f"logs/init_ablation_scale05/g012_d1pma_seed{s}.log" for s in (42, 43, 44)}
MECH = {
    "h1_dropext": ["score_std"],
    "h2_stockexp": ["stock_ratio", "router_no_stock_delta", "router_stock_advantage",
                    "expert_cosine_ts", "expert_cosine_fs", "stock_attn_entropy_norm",
                    "stock_attn_self_frac"],
    "h12_combo": ["score_std", "stock_ratio", "router_no_stock_delta"],
}


def metric_last(run_dir, rel):
    if not run_dir:
        return None
    p = Path(run_dir) / "metrics" / rel
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1].split()[1])
        except Exception:
            return None
    return None


def run_dir_for_setting(setting):
    hits = []
    for p in glob.glob("mlruns/*/*/params/kwargs.trainer_config.run_setting"):
        try:
            if Path(p).read_text(encoding="utf-8", errors="ignore").strip() == setting:
                rd = Path(p).parent.parent
                hits.append((rd.stat().st_mtime, str(rd)))
        except Exception:
            continue
    if not hits:
        return None
    if len(hits) > 1:
        print(f"[warn] {len(hits)} runs share run_setting={setting!r}; using newest", file=sys.stderr)
    hits.sort(reverse=True)
    return hits[0][1]


def grep_last(path, key):
    p = Path(path)
    if not p.exists():
        return None
    w = re.findall(rf"{re.escape(key)}\s*[:=]\s*([-+0-9.eE]+)", p.read_text(encoding="utf-8", errors="ignore"))
    try:
        return float(w[-1]) if w else None
    except Exception:
        return None


def diag_value(run_dir, key):
    """Read a metric from the run's pickled diagnostic_matrix artifact (router_oracle probes etc.
    live there, not in the train log)."""
    if not run_dir:
        return None
    p = Path(run_dir) / "artifacts" / "diagnostic_matrix"
    if not p.exists():
        return None
    try:
        import pickle
        with open(p, "rb") as f:
            df = pickle.load(f)
        hit = df[df["metric"].astype(str).str.endswith(key)]
        if hit.empty:
            hit = df[df["metric"].astype(str).str.contains(key, regex=False)]
        return float(hit["value"].iloc[-1]) if not hit.empty else None
    except Exception:
        return None


def stage_table(L, stage, seeds):
    diffs = []
    L.append(f"\n## {stage} vs anchor (paired by seed)")
    L.append("| seed | metric | anchor | variant | Δ |")
    L.append("|---|---|---:|---:|---:|")
    for s in seeds:
        rd = run_dir_for_setting(f"{stage}_seed{s}_scale05")
        for k, rel in KEYS.items():
            av, vv = metric_last(ANCHOR.get(s), rel), metric_last(rd, rel)
            d = (vv - av) if (av is not None and vv is not None) else None
            if k == "rank_ic" and d is not None:
                diffs.append(d)
            f = lambda v: ("%.5f" % v) if v is not None else "-"
            L.append(f"| {s} | {k} | {f(av)} | {f(vv)} | {f(d) if d is None else '%+.5f' % d} |")
        mech_bits = []
        vlog = f"{LOGDIR}/{stage}_seed{s}.log"
        for mk in MECH.get(stage, []):
            mv = grep_last(vlog, mk)
            if mv is None:
                mv = diag_value(rd, mk)  # router_oracle probes live in the pickled diag matrix
            if mk == "score_std":  # compare vs anchor's own (collapse kill-line for 001)
                av_std = grep_last(ANCHOR_LOGS.get(s, ""), mk)
                if mv is not None and av_std:
                    mech_bits.append(f"{mk}={mv:.4f} (anchor {av_std:.4f}, {mv/av_std-1.0:+.0%})")
                    continue
            if mv is not None:
                mech_bits.append(f"{mk}={mv:.4f}")
        L.append(f"| {s} | mechanism | — | {('; '.join(mech_bits)) or '-'} | |")
    if len(diffs) >= 2:
        mean, sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
        np_ = sum(1 for x in diffs if x >= 0.003)
        L.append(f"\n**{stage} paired ΔRankIC = {mean:+.5f} ± {sd:.5f} (n={len(diffs)}); seeds ≥+0.003: {np_}/{len(diffs)}** "
                 f"→ {'PASS (escalate n≥6 + HC kill-check; never promote on n=3)' if mean >= 0.003 else 'KILL line fires'}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["h1_dropext", "h2_stockexp", "h12_combo", "all"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    stages = ["h1_dropext", "h2_stockexp", "h12_combo"] if a.stage == "all" else [a.stage]

    L = [f"# Weekend matrix vs g012 anchor — {' + '.join(stages)} (25ep, scale=0.5, anchor settings)\n"]
    L.append("Anchor (n=3): rank_ic 0.08163/0.07500/0.07857 (mean 0.07840). Kill: paired ΔRankIC < +0.003. "
             "001 extra kill: valid score_std collapse >30%. 002 extra kill: stock_ratio<0.05 or attention "
             "degenerate (entropy_norm>0.999 uniform-collapse / self_frac>0.95).\n")
    summary = {}
    for st in stages:
        summary[st] = stage_table(L, st, seeds)
    if a.stage == "all" and all(len(v) >= 2 for v in summary.values()):
        L.append("\n## Interaction read (additivity)")
        m = {k: float(np.mean(v)) for k, v in summary.items()}
        L.append(f"- Δ(001)={m['h1_dropext']:+.5f}, Δ(002)={m['h2_stockexp']:+.5f}, "
                 f"Δ(001+002)={m['h12_combo']:+.5f}, additive-expectation={m['h1_dropext']+m['h2_stockexp']:+.5f} "
                 f"(combo−additive = {m['h12_combo']-m['h1_dropext']-m['h2_stockexp']:+.5f})")
    L.append("\n_Cards: h-20260610-001 / h-20260610-002. n=3 PASS only escalates to n≥6 + HC kill-check._")

    out = Path(a.out or f"analysis/weekend_001_002/RESULT_{a.stage}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
