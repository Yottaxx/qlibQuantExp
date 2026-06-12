#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare from-scratch d3cid (scale=0.5) vs the existing scale=0.5 control, per seed.
RankIC + IR_with_cost + MaxDD + the tr_A temporal profile (does d3cid use earlier steps?).
Best-effort: d3cid metrics from its work_flow logs + newest diagnostic_matrix; control from MLflow."""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# control = scale=0.5 backbones (exp 867178867749867261)
CTRL_EXP = "mlruns/867178867749867261"
CTRL_RUN = {42: "917808d52d9a4a13b36c6e20a4155d55", 43: "659cb7dccb784796b674257e4d13baf8", 44: "21a05e198fd34b43bd0d3a14f63e8e2b"}
KEYS = {"rank_ic": "perf/daily_rank_ic_mean", "ir": "portfolio/information_ratio_with_cost",
        "maxdd": "portfolio/max_drawdown_with_cost", "ppd": "opt/post_peak_decay"}


def metric_last(run_dir, rel):
    p = Path(run_dir) / "metrics" / rel
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1].split()[1])
        except Exception:
            return None
    return None


def diag_metrics(csv_path):
    out = {}
    try:
        import csv
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                out[f"{row.get('group','')}/{row.get('metric','')}"] = float(row.get("value", "nan"))
    except Exception:
        pass
    return out


def grep_log(seed):
    log = Path(f"logs/d3cid_scale05/seed{seed}.log")
    info = {}
    if not log.exists():
        return info
    txt = log.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"restored best checkpoint \(valid_rank_ic=([-+0-9.eE]+), epoch=([0-9]+)\)", txt)
    if m:
        info["restored_rank_ic"] = float(m[-1][0]); info["restored_epoch"] = int(m[-1][1])
    w = re.findall(r"tr_W_last_frac:([0-9.]+)", txt)
    if w:
        info["tr_W_last_frac_final"] = float(w[-1]); info["tr_W_last_frac_traj"] = [round(float(x), 3) for x in w]
    return info


def newest_diag_for_seed(seed, after_ts=0.0):
    # find newest diagnostic_matrix.csv whose run has this seed; fall back to newest overall
    cands = []
    for p in glob.glob("mlruns/*/*/**/diagnostic_matrix.csv", recursive=True):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        run = Path(p)
        while run.name != "" and not (run / "params").exists() and run.parent != run:
            run = run.parent
        sp = run / "params" / "kwargs.trainer_config.seed"
        rseed = None
        if sp.exists():
            try:
                rseed = int(sp.read_text().strip())
            except Exception:
                pass
        cands.append((mt, p, rseed, str(run)))
    cands.sort(reverse=True)
    for mt, p, rseed, run in cands:
        if rseed == seed and mt >= after_ts:
            return p, run
    return (cands[0][1], cands[0][3]) if cands else (None, None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default="analysis/readout_redesign/d3cid_scale05_RESULT.md")
    a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(",") if s]
    L = ["# From-scratch d3cid (scale=0.5, 25ep, ckpt=valid_rank_ic) vs scale=0.5 control\n"]
    L.append("d3cid = per-channel-D linear-over-T readout BEFORE the factor pool, trained from scratch (backbone co-adapts). Control = the established scale=0.5 backbones (same protocol). tr_W_last_frac: 1.0=still last-step, <<1=uses earlier steps.\n")
    L.append("| seed | metric | control | d3cid | Δ(d3cid−ctrl) |")
    L.append("|---|---|---:|---:|---:|")
    rows = []
    for s in seeds:
        ctrl = {k: metric_last(f"{CTRL_EXP}/{CTRL_RUN[s]}", v) for k, v in KEYS.items()}
        info = grep_log(s)
        dpath, drun = newest_diag_for_seed(s)
        dm = diag_metrics(dpath) if dpath else {}
        d = {k: dm.get(v) for k, v in KEYS.items()}
        if d.get("rank_ic") is None and info.get("restored_rank_ic") is not None:
            d["rank_ic"] = info["restored_rank_ic"]
        for k in KEYS:
            cv, dv = ctrl.get(k), d.get(k)
            diff = (dv - cv) if (cv is not None and dv is not None) else None
            L.append(f"| {s} | {k} | {('%.5f'%cv) if cv is not None else '-'} | {('%.5f'%dv) if dv is not None else '-'} | {('%+.5f'%diff) if diff is not None else '-'} |")
        L.append(f"| {s} | tr_W_last_frac | 1.000 | {('%.3f'%info['tr_W_last_frac_final']) if 'tr_W_last_frac_final' in info else '-'} | (traj {info.get('tr_W_last_frac_traj','-')}) |")
        rows.append((s, ctrl, d, info))
    # paired RankIC summary
    drank = [r[2].get("rank_ic") for r in rows]; crank = [r[1].get("rank_ic") for r in rows]
    diffs = [d - c for d, c in zip(drank, crank) if d is not None and c is not None]
    L.append("")
    if diffs:
        L.append(f"- **paired ΔRankIC (d3cid−control): {np.mean(diffs):+.5f} ± {(np.std(diffs, ddof=1) if len(diffs)>1 else 0):.5f}** (n={len(diffs)}); seeds with Δ≥+0.003: {sum(1 for x in diffs if x>=0.003)}/{len(diffs)}")
    wl = [r[3].get("tr_W_last_frac_final") for r in rows if r[3].get("tr_W_last_frac_final") is not None]
    if wl:
        L.append(f"- **tr_W_last_frac (d3cid, mean): {np.mean(wl):.3f}** (<<1 ⇒ from-scratch training DID use earlier timesteps — the key signal)")
    L.append("")
    L.append("_Card: time-readout-bonus-20260607 (from-scratch d3cid residual). Promotion needs n≥6 + HC-6 + walk-forward._")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L)); print(f"\n[done] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
