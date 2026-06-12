#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare the from-scratch temporal-readout MATRIX (scale=0.5) vs the existing scale=0.5 control.

Per (design, seed): RankIC + IR_with_cost + MaxDD + post_peak_decay + the temporal profile (did the
readout move off last-step?). Each design's run is located by its QIB_RUN_SETTING param
(`readout_full_<design>_seed<seed>_scale05`); the per-design log supplies the restored-best RankIC
and the epoch-wise temporal profile. Generalizes scripts/compare_d3cid_vs_control.py.

Card: time-readout-bonus-20260607. n=1 is a SCREEN — promotion needs n>=6 + HC-6 + walk-forward.
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

CTRL_EXP = "mlruns/867178867749867261"  # scale=0.5 backbones
CTRL_RUN = {42: "917808d52d9a4a13b36c6e20a4155d55",
            43: "659cb7dccb784796b674257e4d13baf8",
            44: "21a05e198fd34b43bd0d3a14f63e8e2b"}
KEYS = {"rank_ic": "perf/daily_rank_ic_mean", "ir": "portfolio/information_ratio_with_cost",
        "maxdd": "portfolio/max_drawdown_with_cost", "ppd": "opt/post_peak_decay"}
# Temporal-profile keys per design (printed each epoch in the run log); first one is the headline.
PROFILE = {
    "d3cid": ["tr_W_last_frac"], "d3cin": ["tr_W_last_frac"],
    "d3mix": ["tr_collapse_last_frac", "tr_timemix_gain", "tr_chanmix_gain"],
    "d1pma": ["tr_attn_last_frac", "tr_gate_g"],
    "duala": ["tr_attn_last_frac", "tr_gate_g"], "dualb": ["tr_attn_last_frac", "tr_gate_g"],
}
LOGDIR = "logs/readout_matrix_scale05"


def metric_last(run_dir, rel):
    p = Path(run_dir) / "metrics" / rel
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1].split()[1])
        except Exception:
            return None
    return None


def run_dir_for_setting(setting):
    # Collect ALL runs with this run_setting (a crash+restart reuses the verbatim setting -> duplicates)
    # and return the NEWEST by mtime, warning if >1, so we never bind to a stale/failed run.
    hits = []
    for p in glob.glob("mlruns/*/*/params/kwargs.trainer_config.run_setting"):
        try:
            if Path(p).read_text(encoding="utf-8", errors="ignore").strip() == setting:
                rd = Path(p).parent.parent  # mlruns/<exp>/<run_id>
                hits.append((rd.stat().st_mtime, str(rd)))
        except Exception:
            continue
    if not hits:
        return None
    if len(hits) > 1:
        print(f"[warn] {len(hits)} runs share run_setting={setting!r}; using newest", file=sys.stderr)
    hits.sort(reverse=True)
    return hits[0][1]


def grep_log(design, seed):
    log = Path(f"{LOGDIR}/full_{design}_seed{seed}.log")
    info = {}
    if not log.exists():
        return info
    txt = log.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"restored best checkpoint \(valid_rank_ic=([-+0-9.eE]+), epoch=([0-9]+)\)", txt)
    if m:
        info["restored_rank_ic"] = float(m[-1][0]); info["restored_epoch"] = int(m[-1][1])
    for key in PROFILE.get(design, []):
        w = re.findall(rf"{re.escape(key)}\s*[:=]\s*([-+0-9.eE]+)", txt)
        if w:
            try:
                info[key] = float(w[-1])
                info[key + "_traj"] = [round(float(x), 3) for x in w[-8:]]
            except Exception:
                pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="d3cid,d3cin,d3mix,d1pma,duala,dualb")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default="analysis/readout_redesign/readout_matrix_RESULT.md")
    a = ap.parse_args()
    designs = [d for d in a.designs.split(",") if d]
    seeds = [int(s) for s in a.seeds.split(",") if s]

    L = ["# Temporal-readout matrix (from-scratch, scale=0.5, 25ep, ckpt=valid_rank_ic) vs scale=0.5 control\n"]
    L.append("Capacity ladder: control(select) -> d3cid(CI linear) -> d3cin(per-factor) -> d3mix(mixing); "
             "plus d1pma/duals (attention). Profile: tr_*_last_frac ~1.0=still last-step, <<1=uses earlier "
             "steps; tr_*_gain>0 = d3mix mixing capacity activated.\n")
    L.append("| design | seed | metric | control | design | Δ(design−ctrl) |")
    L.append("|---|---|---|---:|---:|---:|")
    paired = {}     # design -> [d_rank - c_rank]
    profiles = {}   # design -> [profile dict]
    for d in designs:
        paired[d] = []
        profiles[d] = []
        for s in seeds:
            ctrl = {k: metric_last(f"{CTRL_EXP}/{CTRL_RUN.get(s, '')}", v) for k, v in KEYS.items()}
            info = grep_log(d, s)
            rd = run_dir_for_setting(f"readout_full_{d}_seed{s}_scale05")
            des = {k: (metric_last(rd, v) if rd else None) for k, v in KEYS.items()}
            # rank_ic: perf/daily_rank_ic_mean is the TEST segment; the log's restored_rank_ic is the
            # VALIDATION segment. Show the valid value (tagged) when test is missing, but NEVER mix it
            # into the paired ΔRankIC below — that delta must be test-vs-test only (cross-segment = wrong).
            valid_fallback = info.get("restored_rank_ic") if des.get("rank_ic") is None else None
            for k in KEYS:
                cv, dv = ctrl.get(k), des.get(k)
                ccell = ("%.5f" % cv) if cv is not None else "-"
                if k == "rank_ic" and dv is None and valid_fallback is not None:
                    dcell, diffcell = f"{valid_fallback:.5f} (valid)", "-"
                else:
                    diff = (dv - cv) if (cv is not None and dv is not None) else None
                    dcell = ("%.5f" % dv) if dv is not None else "-"
                    diffcell = ("%+.5f" % diff) if diff is not None else "-"
                L.append(f"| {d} | {s} | {k} | {ccell} | {dcell} | {diffcell} |")
            prof = {k: info.get(k) for k in PROFILE.get(d, [])}
            profiles[d].append(prof)
            pstr = ", ".join(f"{k}={('%.3f' % v) if isinstance(v, float) else '-'}" for k, v in prof.items())
            traj = info.get(PROFILE[d][0] + "_traj", "-") if PROFILE.get(d) else "-"
            L.append(f"| {d} | {s} | profile | — | {pstr or '-'} | (traj {traj}) |")
            if des.get("rank_ic") is not None and ctrl.get("rank_ic") is not None:
                paired[d].append(des["rank_ic"] - ctrl["rank_ic"])

    L.append("")
    L.append("## Summary (paired ΔRankIC vs control, by design)")
    L.append("| design | n | mean ΔRankIC | seeds Δ≥+0.003 | temporal moved off last-step? |")
    L.append("|---|---:|---:|---:|---|")
    attn_designs = {"d1pma", "duala", "dualb"}
    for d in designs:
        diffs = paired[d]
        moved = "—"
        key0 = PROFILE[d][0] if PROFILE.get(d) else None
        if key0:
            vals = [p.get(key0) for p in profiles[d] if p.get(key0) is not None]
            if vals:
                lf = float(np.mean(vals))
                if d in attn_designs:
                    # The readout is z=(1-g)*last + g*attn, so the EFFECTIVE last-step weight is
                    # (1-g)+g*softmax_last, NOT the bare softmax. A genuinely-null g≈0 readout is still
                    # last-step even when softmax entropy is high — so gate the verdict on eff, not lf.
                    gv = [p.get("tr_gate_g") for p in profiles[d] if p.get("tr_gate_g") is not None]
                    g = float(np.mean(gv)) if gv else 0.5
                    eff = (1.0 - g) + g * lf
                    moved = f"eff_last={eff:.3f} (g={g:.2f}, {key0}={lf:.3f}) ({'YES' if eff < 0.7 else 'no'})"
                else:
                    moved = f"{key0}={lf:.3f} ({'YES' if lf < 0.9 else 'no'})"
        if diffs:
            sd = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
            L.append(f"| {d} | {len(diffs)} | {np.mean(diffs):+.5f} ± {sd:.5f} "
                     f"| {sum(1 for x in diffs if x >= 0.003)}/{len(diffs)} | {moved} |")
        else:
            L.append(f"| {d} | 0 | - | - | {moved} |")
    L.append("")
    L.append("_Card: time-readout-bonus-20260607. n=1 is a SCREEN — promotion needs n≥6 + HC-6 + "
             "walk-forward (n=3 reversals are the house lesson)._")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[done] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
