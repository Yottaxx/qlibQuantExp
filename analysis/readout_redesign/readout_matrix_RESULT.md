# Temporal-readout matrix (from-scratch, scale=0.5, 25ep, ckpt=valid_rank_ic) vs scale=0.5 control

Capacity ladder: control(select) -> d3cid(CI linear) -> d3cin(per-factor) -> d3mix(mixing); plus d1pma/duals (attention). Profile: tr_*_last_frac ~1.0=still last-step, <<1=uses earlier steps; tr_*_gain>0 = d3mix mixing capacity activated.

| design | seed | metric | control | design | Δ(design−ctrl) |
|---|---|---|---:|---:|---:|
| d3cid | 42 | rank_ic | 0.08107 | 0.08075 | -0.00032 |
| d3cid | 42 | ir | 2.24625 | 1.79019 | -0.45606 |
| d3cid | 42 | maxdd | -0.06003 | -0.07219 | -0.01216 |
| d3cid | 42 | ppd | 0.00475 | 0.00381 | -0.00094 |
| d3cid | 42 | profile | — | tr_W_last_frac=0.906 | (traj [0.906, 0.906, 0.906, 0.906, 0.906, 0.906, 0.906, 0.906]) |
| d3cin | 42 | rank_ic | 0.08107 | - | - |
| d3cin | 42 | ir | 2.24625 | - | - |
| d3cin | 42 | maxdd | -0.06003 | - | - |
| d3cin | 42 | ppd | 0.00475 | - | - |
| d3cin | 42 | profile | — | tr_W_last_frac=0.937 | (traj [0.956, 0.954, 0.951, 0.948, 0.945, 0.942, 0.939, 0.937]) |
| d3mix | 42 | rank_ic | 0.08107 | 0.08128 | +0.00021 |
| d3mix | 42 | ir | 2.24625 | 2.04762 | -0.19863 |
| d3mix | 42 | maxdd | -0.06003 | -0.07710 | -0.01707 |
| d3mix | 42 | ppd | 0.00475 | 0.00409 | -0.00066 |
| d3mix | 42 | profile | — | tr_collapse_last_frac=0.878, tr_timemix_gain=0.128, tr_chanmix_gain=0.187 | (traj [0.879, 0.879, 0.878, 0.878, 0.878, 0.878, 0.878, 0.878]) |
| d1pma | 42 | rank_ic | 0.08107 | 0.07765 | -0.00341 |
| d1pma | 42 | ir | 2.24625 | 1.61895 | -0.62730 |
| d1pma | 42 | maxdd | -0.06003 | -0.08644 | -0.02641 |
| d1pma | 42 | ppd | 0.00475 | 0.00329 | -0.00146 |
| d1pma | 42 | profile | — | tr_attn_last_frac=0.127, tr_gate_g=0.503 | (traj [0.127, 0.127, 0.127, 0.127, 0.127, 0.127, 0.127, 0.127]) |
| duala | 42 | rank_ic | 0.08107 | 0.07511 | -0.00595 |
| duala | 42 | ir | 2.24625 | 1.90819 | -0.33806 |
| duala | 42 | maxdd | -0.06003 | -0.06838 | -0.00835 |
| duala | 42 | ppd | 0.00475 | 0.00652 | +0.00177 |
| duala | 42 | profile | — | tr_attn_last_frac=0.124, tr_gate_g=0.506 | (traj [0.124, 0.124, 0.124, 0.124, 0.124, 0.124, 0.124, 0.124]) |
| dualb | 42 | rank_ic | 0.08107 | 0.07175 | -0.00932 |
| dualb | 42 | ir | 2.24625 | 1.72185 | -0.52440 |
| dualb | 42 | maxdd | -0.06003 | -0.07326 | -0.01324 |
| dualb | 42 | ppd | 0.00475 | 0.00409 | -0.00066 |
| dualb | 42 | profile | — | tr_attn_last_frac=0.125, tr_gate_g=0.503 | (traj [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]) |

## Summary (paired ΔRankIC vs control, by design)
| design | n | mean ΔRankIC | seeds Δ≥+0.003 | temporal moved off last-step? |
|---|---:|---:|---:|---|
| d3cid | 1 | -0.00032 ± 0.00000 | 0/1 | tr_W_last_frac=0.906 (no) |
| d3cin | 0 | - | - | tr_W_last_frac=0.937 (no) |
| d3mix | 1 | +0.00021 ± 0.00000 | 0/1 | tr_collapse_last_frac=0.878 (YES) |
| d1pma | 1 | -0.00341 ± 0.00000 | 0/1 | eff_last=0.561 (g=0.50, tr_attn_last_frac=0.127) (YES) |
| duala | 1 | -0.00595 ± 0.00000 | 0/1 | eff_last=0.557 (g=0.51, tr_attn_last_frac=0.124) (YES) |
| dualb | 1 | -0.00932 ± 0.00000 | 0/1 | eff_last=0.560 (g=0.50, tr_attn_last_frac=0.125) (YES) |

_Card: time-readout-bonus-20260607. n=1 is a SCREEN — promotion needs n≥6 + HC-6 + walk-forward (n=3 reversals are the house lesson)._

---

## Addendum 2026-06-12 — d3cin completed (OOM re-run) + init-ablation verdict

**d3cin (onehot, seed42, clean 25ep re-run** — run_setting `readout_full_onehot_d3cin_seed42_scale05`,
`mlruns/992417505232301360/c540b2d2...`): rank_ic **0.07937** (Δ −0.00170 vs control), final
`tr_W_last_frac=0.916`. Portfolio NOTABLY worse: IR_with_cost **0.543** (control 2.246; without-cost 1.061
vs control's family ~2), MaxDD_with_cost **−0.413** (control −0.060) — the largest RankIC≠IR divergence in the
matrix (finding #3 amplified: near-flat RankIC, collapsed portfolio). The 6-design matrix is now complete:
**0/6 designs reach +0.003; ALL 6 are portfolio-worse than control.**

**Init-ablation (see `init_ablation_RESULT.md`):** uniform_mean init CLEARLY worse on all 3 linears
(−0.0088…−0.0137 vs identical-backbone onehot twins) and never returns toward last-step (last_frac 0.125→0.13–0.15);
d1pma gate sweep shows the gate is FROZEN at its init (final≈init at 0.12/0.5/0.88) and only the
last-step-leaning start (g0.12, rank_ic 0.08163) reaches ≈control — attention at best matches, never beats.
**Verdict: one-hot-last does NOT trap (the uniform basin is the trap); last-step `h[:,-1]` is robustly ~optimal;
the temporal-readout axis is CLOSED.** Card settled → `settled-CLOSED`.
