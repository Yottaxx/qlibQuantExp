# Init-ablation (seed42, scale=0.5, 25ep) — does the readout INIT trap the result?

Control RankIC = 0.08107. Linears: uniform vs onehot share an IDENTICAL backbone (separate-RNG init) ⇒ clean A/B. d1pma: gate-init sweep (g0.5 reused).

## Linear A/B — uniform_mean vs onehot_last
uniform prof traj = [first 2, ..., final 4] of tr_*_last_frac: starts ~0.125; RISING toward 1 = converging back to last-step (init-invariance); flat/spread = found a different basin.

| design | onehot RankIC | uniform RankIC | Δ(uni−onehot) | Δ(uni−ctrl) | uniform prof traj | best_ep |
|---|---:|---:|---:|---:|---|---:|
| d3cid | 0.08075 | 0.06710 | -0.01365 | -0.01396 | [0.127, 0.127, 0.134, 0.134, 0.134, 0.134] | 16 |
| d3cin | 0.07937 | 0.06623 | -0.01314 | -0.01484 | [0.125, 0.126, 0.138, 0.138, 0.138, 0.138] | 16 |
| d3mix | 0.08128 | 0.07252 | -0.00876 | -0.00855 | [0.104, 0.106, 0.147, 0.147, 0.147, 0.147] | 16 |

## d1pma gate-init sweep
| start | RankIC | Δ vs ctrl | final tr_gate_g |
|---|---:|---:|---:|
| g0.12 | 0.08163 | +0.00056 | 0.124 |
| g0.5 | 0.07765 | -0.00341 | 0.503 |
| g0.88 | 0.07848 | -0.00259 | 0.874 |

## Falsification verdict
- d3cid: uniform CLEARLY WORSE (-0.01365) ⇒ the diverse start actively hurts; last-step basin is the attractor.
- d3cin: uniform CLEARLY WORSE (-0.01314) ⇒ the diverse start actively hurts; last-step basin is the attractor.
- d3mix: uniform CLEARLY WORSE (-0.00876) ⇒ the diverse start actively hurts; last-step basin is the attractor.
- **d1pma: last-step-leaning gate (g0.12) recovers +0.00398 over g0.5** ⇒ the g=0.5 start was hurting the attention ⇒ re-read with a leaning gate.

_Card: time-readout-bonus-20260607 / init-ablation. n=1 is a SCREEN; promotion needs n≥6 + HC kill-check._
