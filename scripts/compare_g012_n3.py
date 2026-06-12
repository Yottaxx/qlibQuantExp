#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""g0.12 d1pma 3-seed kill-check vs scale=0.5 control (paired by seed).

Pre-registered: kill if paired ΔRankIC mean < +0.003 OR IR/MaxDD clearly worse.
n=3 pass would still only escalate to n>=6 + HC kill-check (house lesson), never promote directly.
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

CTRL_EXP = "mlruns/867178867749867261"
CTRL_RUN = {42: "917808d52d9a4a13b36c6e20a4155d55",
            43: "659cb7dccb784796b674257e4d13baf8",
            44: "21a05e198fd34b43bd0d3a14f63e8e2b"}
KEYS = {"rank_ic": "perf/daily_rank_ic_mean", "ir": "portfolio/information_ratio_with_cost",
        "maxdd": "portfolio/max_drawdown_with_cost", "ppd": "opt/post_peak_decay"}


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


def gate_final(seed):
    p = Path(f"logs/init_ablation_scale05/g012_d1pma_seed{seed}.log")
    if not p.exists():
        return None
    w = re.findall(r"tr_gate_g\s*[:=]\s*([-+0-9.eE]+)", p.read_text(encoding="utf-8", errors="ignore"))
    try:
        return float(w[-1]) if w else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default="analysis/readout_redesign/g012_n3_RESULT.md")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]

    L = ["# g0.12 d1pma — 3-seed kill-check vs scale=0.5 control (paired by seed)\n"]
    L.append("Pre-registered: kill if paired ΔRankIC mean < +0.003 OR IR/MaxDD clearly worse. The gate is "
             "frozen at init (mechanism), so g0.12 ≈ '12% attention mixture' — prior says parity-not-edge.\n")
    L.append("| seed | metric | control | g0.12 | Δ |")
    L.append("|---|---|---:|---:|---:|")
    diffs = {k: [] for k in KEYS}
    gates = []
    for s in seeds:
        rd = run_dir_for_setting(f"readout_full_g012_d1pma_seed{s}_scale05")
        ctrl_rd = f"{CTRL_EXP}/{CTRL_RUN[s]}" if s in CTRL_RUN else None
        for k, rel in KEYS.items():
            cv, dv = metric_last(ctrl_rd, rel), metric_last(rd, rel)
            d = (dv - cv) if (cv is not None and dv is not None) else None
            if d is not None:
                diffs[k].append(d)
            f = lambda v: ("%.5f" % v) if v is not None else "-"
            L.append(f"| {s} | {k} | {f(cv)} | {f(dv)} | {f(d) if d is None else '%+.5f' % d} |")
        g = gate_final(s)
        gates.append(g)
        L.append(f"| {s} | final_gate_g | — | {('%.3f' % g) if g is not None else '-'} | (init 0.119) |")

    L.append("\n## Verdict")
    dr = diffs["rank_ic"]
    if len(dr) >= 3:
        mean, sd = float(np.mean(dr)), float(np.std(dr, ddof=1))
        n_pass = sum(1 for x in dr if x >= 0.003)
        ir_bad = sum(1 for x in diffs["ir"] if x < -0.3)
        dd_bad = sum(1 for x in diffs["maxdd"] if x < -0.02)
        L.append(f"- paired ΔRankIC = **{mean:+.5f} ± {sd:.5f}** (n={len(dr)}); seeds ≥+0.003: {n_pass}/{len(dr)}")
        L.append(f"- portfolio: ΔIR clearly-worse seeds {ir_bad}/{len(diffs['ir'])}; ΔMaxDD clearly-worse {dd_bad}/{len(diffs['maxdd'])}")
        if mean < 0.003 or ir_bad >= 2 or dd_bad >= 2:
            L.append("- **KILL: g0.12 d1pma is parity-at-best — attention readout CLOSED for good (no init flavor "
                     "rescues it). Last-step control stands.**")
        else:
            L.append("- **PASS at n=3 — escalate to n≥6 + HC kill-check before ANY claim (n=3 reversals are the "
                     "house lesson). Do NOT promote on this.**")
    else:
        L.append(f"- insufficient data (have {len(dr)}/3 paired RankIC) — runs missing or still in flight.")
    L.append("\n_Card: time-readout-bonus-20260607 / g012-n3 kill-check._")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[done] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
