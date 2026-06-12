#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Init-ablation compare (task #14): does the readout INIT trap the result?

Linear A/B: uniform_mean vs onehot_last for d3cid/d3cin/d3mix (backbone identical — clean A/B).
d1pma gate sweep: g0.12 / g0.5(existing) / g0.88. Both vs the scale=0.5 control (RankIC 0.08107).
RankIC from MLflow (perf/daily_rank_ic_mean); the temporal profile (tr_W/collapse_last_frac, tr_gate_g)
from each run's log. Emits a falsification verdict.
"""
from __future__ import annotations
import argparse
import glob
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CTRL_EXP = "mlruns/867178867749867261"
CTRL_RUN = {42: "917808d52d9a4a13b36c6e20a4155d55"}
RANKIC = "perf/daily_rank_ic_mean"

# (group, arm, run_setting, log_path, profile_key). onehot d3cid/d3mix + d1pma g0.5 reuse the prior matrix.
def cells(seed):
    OLD = "logs/readout_matrix_scale05"
    NEW = "logs/init_ablation_scale05"
    s = seed
    return [
        ("d3cid", "onehot", f"readout_full_d3cid_seed{s}_scale05",          f"{OLD}/full_d3cid_seed{s}.log",        "tr_W_last_frac"),
        ("d3cid", "uniform", f"readout_full_uniform_d3cid_seed{s}_scale05", f"{NEW}/uniform_d3cid_seed{s}.log",     "tr_W_last_frac"),
        ("d3cin", "onehot", f"readout_full_onehot_d3cin_seed{s}_scale05",   f"{NEW}/onehot_d3cin_seed{s}.log",      "tr_W_last_frac"),
        ("d3cin", "uniform", f"readout_full_uniform_d3cin_seed{s}_scale05", f"{NEW}/uniform_d3cin_seed{s}.log",     "tr_W_last_frac"),
        ("d3mix", "onehot", f"readout_full_d3mix_seed{s}_scale05",          f"{OLD}/full_d3mix_seed{s}.log",        "tr_collapse_last_frac"),
        ("d3mix", "uniform", f"readout_full_uniform_d3mix_seed{s}_scale05", f"{NEW}/uniform_d3mix_seed{s}.log",     "tr_collapse_last_frac"),
        ("d1pma", "g0.5",  f"readout_full_d1pma_seed{s}_scale05",           f"{OLD}/full_d1pma_seed{s}.log",        "tr_gate_g"),
        ("d1pma", "g0.12", f"readout_full_g012_d1pma_seed{s}_scale05",      f"{NEW}/g012_d1pma_seed{s}.log",        "tr_gate_g"),
        ("d1pma", "g0.88", f"readout_full_g088_d1pma_seed{s}_scale05",      f"{NEW}/g088_d1pma_seed{s}.log",        "tr_gate_g"),
    ]


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
    hits.sort(reverse=True)
    return hits[0][1]


def profile_info(log_path, key):
    """Return (last_value, trajectory_tail, best_epoch). best_epoch>=24 of 25 => budget-censored
    (the run was still improving at the budget boundary — a 'uniform≈onehot' null is then ambiguous)."""
    p = Path(log_path)
    if not p.exists():
        return None, None, None
    txt = p.read_text(encoding="utf-8", errors="ignore")
    w = re.findall(rf"{re.escape(key)}\s*[:=]\s*([-+0-9.eE]+)", txt)
    traj = None
    last = None
    if w:
        try:
            vals = [float(v) for v in w]
            last = vals[-1]
            traj = [round(v, 3) for v in (vals[:2] + vals[-4:])]  # first 2 + final 4: direction at a glance
        except Exception:
            pass
    m = re.findall(r"restored best checkpoint \(valid_rank_ic=([-+0-9.eE]+), epoch=([0-9]+)\)", txt)
    best_epoch = int(m[-1][1]) if m else None
    return last, traj, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="analysis/readout_redesign/init_ablation_RESULT.md")
    a = ap.parse_args()
    ctrl = metric_last(f"{CTRL_EXP}/{CTRL_RUN.get(a.seed, '')}", RANKIC)

    rows = {}  # (group, arm) -> {rankic, profile, traj, best_epoch}
    for group, arm, rs, logp, pkey in cells(a.seed):
        ric = metric_last(run_dir_for_setting(rs), RANKIC)
        prof, traj, bep = profile_info(logp, pkey)
        rows[(group, arm)] = {"rankic": ric, "profile": prof, "traj": traj, "best_epoch": bep, "pkey": pkey}

    L = [f"# Init-ablation (seed{a.seed}, scale=0.5, 25ep) — does the readout INIT trap the result?\n"]
    L.append(f"Control RankIC = {('%.5f' % ctrl) if ctrl is not None else '-'}. Linears: uniform vs onehot share an "
             f"IDENTICAL backbone (separate-RNG init) ⇒ clean A/B. d1pma: gate-init sweep (g0.5 reused).\n")

    # Linear A/B
    L.append("## Linear A/B — uniform_mean vs onehot_last")
    L.append("uniform prof traj = [first 2, ..., final 4] of tr_*_last_frac: starts ~0.125; RISING toward 1 = "
             "converging back to last-step (init-invariance); flat/spread = found a different basin.")
    L.append("")
    L.append("| design | onehot RankIC | uniform RankIC | Δ(uni−onehot) | Δ(uni−ctrl) | uniform prof traj | best_ep |")
    L.append("|---|---:|---:|---:|---:|---|---:|")
    verdict_lines = []
    for d in ["d3cid", "d3cin", "d3mix"]:
        oh, un = rows.get((d, "onehot"), {}), rows.get((d, "uniform"), {})
        ohr, unr = oh.get("rankic"), un.get("rankic")
        d_uo = (unr - ohr) if (ohr is not None and unr is not None) else None
        d_uc = (unr - ctrl) if (unr is not None and ctrl is not None) else None
        bep = un.get("best_epoch")
        censored = bep is not None and bep >= 24
        f = lambda v, p="%.5f": (p % v) if v is not None else "-"
        bep_s = f"{bep}{' ⚠CENSORED' if censored else ''}" if bep is not None else "-"
        L.append(f"| {d} | {f(ohr)} | {f(unr)} | {f(d_uo, '%+.5f')} | {f(d_uc, '%+.5f')} "
                 f"| {un.get('traj') or '-'} | {bep_s} |")
        if d_uo is not None:
            cen = " [⚠ best_ep at budget boundary — uniform may still be improving; null is AMBIGUOUS, consider +epochs before closing]" if censored else ""
            if d_uo >= 0.003:
                verdict_lines.append(f"- **{d}: uniform BEATS onehot by {d_uo:+.5f} (≥+0.003)** ⇒ one-hot-last WAS trapping ⇒ escalate n≥6 + kill-check.")
            elif d_uo <= -0.003:
                verdict_lines.append(f"- {d}: uniform CLEARLY WORSE ({d_uo:+.5f}) ⇒ the diverse start actively hurts; last-step basin is the attractor.{cen}")
            else:
                verdict_lines.append(f"- {d}: uniform≈onehot (Δ={d_uo:+.5f}) ⇒ init does NOT trap; last-step robustly ~optimal.{cen}")

    # d1pma gate sweep
    L.append("\n## d1pma gate-init sweep")
    L.append("| start | RankIC | Δ vs ctrl | final tr_gate_g |")
    L.append("|---|---:|---:|---:|")
    g_rics = {}
    for arm in ["g0.12", "g0.5", "g0.88"]:
        r = rows.get(("d1pma", arm), {})
        ric, prof = r.get("rankic"), r.get("profile")
        g_rics[arm] = ric
        d_c = (ric - ctrl) if (ric is not None and ctrl is not None) else None
        f = lambda v, p="%.5f": (p % v) if v is not None else "-"
        L.append(f"| {arm} | {f(ric)} | {f(d_c, '%+.5f')} | {f(prof, '%.3f')} |")
    base = g_rics.get("g0.5")
    lean = g_rics.get("g0.12")
    if base is not None and lean is not None:
        d_gl = lean - base
        if d_gl >= 0.003:
            verdict_lines.append(f"- **d1pma: last-step-leaning gate (g0.12) recovers {d_gl:+.5f} over g0.5** ⇒ the g=0.5 start was hurting the attention ⇒ re-read with a leaning gate.")
        else:
            verdict_lines.append(f"- d1pma: g0.12≈g0.5 (Δ={d_gl:+.5f}) ⇒ the gate start did NOT trap; attention is genuinely the weaker readout.")

    L.append("\n## Falsification verdict")
    L += (verdict_lines or ["- (insufficient data — some runs missing)"])
    L.append("\n_Card: time-readout-bonus-20260607 / init-ablation. n=1 is a SCREEN; promotion needs n≥6 + HC kill-check._")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[done] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
