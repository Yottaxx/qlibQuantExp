#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Step-A time-readout screen across seeds -> RESULT_overnight.md."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def ms(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)) if a.size else (float("nan"), float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="analysis/readout_redesign/stepA")
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--out", default="analysis/readout_redesign/RESULT_overnight.md")
    args = ap.parse_args()
    root = Path(args.root); seeds = [int(s) for s in args.seeds.split(",") if s]
    runs = []
    for s in seeds:
        p = root / f"seed{s}" / "results.json"
        if p.exists():
            runs.append(json.loads(p.read_text(encoding="utf-8")))
    if not runs:
        print("[error] no results"); return 1
    designs = [d for d in runs[0]["designs"].keys()]
    L = ["# Time-axis Readout Screen (Step-A, frozen backbone, scale=0.5) — n=%d" % len(runs), ""]
    L.append("**Lower-bound SCREEN** (backbone frozen, reps NOT reshaped). Per the confound, a NULL here only escalates the best to Step-B (readout+top-block finetune); it is not a verdict. Gate = bonus ≥ +0.003 vs capacity-matched last-step baseline.")
    L.append("")
    model = ms([r.get("model_valid_rank_ic") for r in runs])
    base = ms([r.get("baseline_rank_ic") for r in runs])
    L.append(f"- model valid rank_ic (ref): {model[0]:.5f} ± {model[1]:.5f}")
    L.append(f"- baseline (trained last-step head on frozen reps): {base[0]:.5f} ± {base[1]:.5f}")
    L.append("")
    L.append("| design | valid rank_ic (mean±std) | **bonus vs baseline** | seeds bonus≥+0.003 | per-seed bonus |")
    L.append("|---|---:|---:|---:|---|")
    summary = {}
    for d in designs:
        if d == "baseline":
            continue
        vric = ms([r["designs"][d]["valid_rank_ic"] for r in runs])
        bon = [r["designs"][d]["bonus_vs_baseline"] for r in runs]
        bms = ms(bon)
        npass = sum(1 for x in bon if x is not None and np.isfinite(x) and x >= 0.003)
        per = ", ".join(f"{r['seed']}:{r['designs'][d]['bonus_vs_baseline']:+.4f}" for r in runs)
        L.append(f"| {d} | {vric[0]:.5f} ± {vric[1]:.5f} | **{bms[0]:+.5f} ± {bms[1]:.5f}** | {npass}/{len(runs)} | {per} |")
        summary[d] = {"bonus_mean": bms[0], "n_pass": npass}
    L.append("")
    # diagnostics (d1pma gate)
    gates = [r["designs"].get("d1pma", {}).get("diag", {}).get("gate_g") for r in runs]
    gms = ms(gates)
    if np.isfinite(gms[0]):
        L.append(f"- **d1pma learned gate g** (≈0 ⇒ readout falls back to last-step): {gms[0]:.3f} ± {gms[1]:.3f}")
    L.append("")
    # verdict
    winner = max(summary, key=lambda k: summary[k]["bonus_mean"]) if summary else None
    any_pass = any(summary[k]["n_pass"] > len(runs) / 2 for k in summary)
    L.append("## Screen verdict")
    if any_pass:
        passers = [k for k in summary if summary[k]["n_pass"] > len(runs) / 2]
        L.append(f"- **Designs clearing +0.003 on seed-majority (frozen lower bound): {passers}** → escalate to Step-B (readout+top-block continue-finetune, HC-6).")
    else:
        L.append(f"- **No design clears +0.003 on seed-majority (frozen lower bound).** Per the never-kill rule, escalate the best-by-point-estimate (**{winner}**, bonus {summary[winner]['bonus_mean']:+.5f}) to Step-B before any 'last-step optimal' claim. A Step-B null would then settle P2.")
    L.append(f"- Best-by-point-estimate: **{winner}** ({summary[winner]['bonus_mean']:+.5f})" if winner else "")
    L.append("")
    L.append("_Card: time-readout-bonus-20260607. Next: Step-B continue-finetune of the escalated design(s), n=6, HC-6 four-tuple._")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"summary": summary, "winner": winner, "any_pass": any_pass, "n": len(runs)},
              open(Path(args.root) / "aggregate.json", "w"), indent=2)
    print("\n".join(L))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
