#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Step-B (paired control vs d3cid finetune) -> append to RESULT_overnight.md."""
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


def load(root, arm, s):
    p = Path(root) / f"{arm}_seed{s}" / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="analysis/readout_redesign/stepB")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default="analysis/readout_redesign/RESULT_overnight.md")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    rows = []
    for s in seeds:
        c = load(args.root, "control", s); d = load(args.root, "d3cid", s)
        if c and d:
            rows.append((s, c, d))
    if not rows:
        print("[error] no step-B results"); return 1
    L = ["", "---", "", "## Step-B — readout + top-MoE-block continue-finetune (UN-confounded, n=%d, scale=0.5)" % len(rows), ""]
    L.append("Paired arms per seed: **control** = last-step readout + top-block finetuned; **d3cid** = per-channel-D linear-over-T readout + top-block finetuned. The readout's true contribution = **d3cid − control** (isolates the readout from the finetuning-the-top effect). baseline = loaded model step0.")
    L.append("")
    L.append("| seed | model(step0) | control best | d3cid best | **d3cid − control** | d3cid W_last_frac |")
    L.append("|---|---:|---:|---:|---:|---:|")
    diffs = []; cgain = []; dgain = []
    for s, c, d in rows:
        diff = d["best_valid_rank_ic"] - c["best_valid_rank_ic"]; diffs.append(diff)
        cgain.append(c["bonus_vs_step0"]); dgain.append(d["bonus_vs_step0"])
        wlf = d.get("W_last_frac")
        L.append(f"| {s} | {d['baseline_step0_rank_ic']:.5f} | {c['best_valid_rank_ic']:.5f} | {d['best_valid_rank_ic']:.5f} | **{diff:+.5f}** | {('%.3f'%wlf) if wlf is not None else '-'} |")
    dm = ms(diffs)
    L.append("")
    L.append(f"- **d3cid − control: {dm[0]:+.5f} ± {dm[1]:.5f}** (n={dm[2]}); seeds with diff≥+0.003: {sum(1 for x in diffs if x>=0.003)}/{len(diffs)}")
    L.append(f"- control gain vs step0 (top-block finetune alone): {ms(cgain)[0]:+.5f}; d3cid gain vs step0: {ms(dgain)[0]:+.5f}")
    wlf = ms([d.get("W_last_frac") for _, _, d in rows])
    if np.isfinite(wlf[0]):
        L.append(f"- d3cid learned W last-step mass-fraction: {wlf[0]:.3f} (≈1 ⇒ stays at last-step = no temporal use; <<1 ⇒ uses earlier steps)")
    L.append("")
    helps = (sum(1 for x in diffs if x >= 0.003) > len(diffs) / 2)
    L.append("### Verdict (Step-B, un-confounded)")
    if helps:
        L.append(f"- **d3cid temporal readout beats the finetuned last-step control by ≥+0.003 on seed-majority → the time readout IS a RankIC lever.** Promote: extend to n=6 + HC-6 (IR/MaxDD) + /quant-walk-forward + csi800 before any production claim. (n={len(rows)}: confirm at n≥6 per the τ kill-check lesson.)")
    else:
        L.append(f"- **d3cid does NOT beat the finetuned last-step control by +0.003 (diff {dm[0]:+.5f}, n={len(rows)}).** Even with the top block reshaping under an all-T readout gradient, the temporal readout is ~at parity → **last-step readout is ~optimal** (P2 of time-readout-bonus-20260607); headroom is upstream (blocks/data/regime), not the temporal readout. CAVEAT n=3.")
    open(args.out, "a", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"diff_mean": dm[0], "n": len(rows), "helps": helps}, open(Path(args.root) / "aggregate.json", "w"), indent=2)
    print("\n".join(L)); print(f"\n[done] appended {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
