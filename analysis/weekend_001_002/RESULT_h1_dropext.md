# Weekend matrix vs g012 anchor — h1_dropext (25ep, scale=0.5, anchor settings)

Anchor (n=3): rank_ic 0.08163/0.07500/0.07857 (mean 0.07840). Kill: paired ΔRankIC < +0.003. 001 extra kill: valid score_std collapse >30%. 002 extra kill: stock_ratio<0.05 or attention degenerate (entropy_norm>0.999 uniform-collapse / self_frac>0.95).


## h1_dropext vs anchor (paired by seed)
| seed | metric | anchor | variant | Δ |
|---|---|---:|---:|---:|
| 42 | rank_ic | 0.08163 | 0.06568 | -0.01594 |
| 42 | ir | 1.53606 | 1.27846 | -0.25760 |
| 42 | maxdd | -0.06634 | -0.06058 | +0.00576 |
| 42 | ppd | 0.00210 | 0.00204 | -0.00006 |
| 42 | mechanism | — | score_std=0.1879 (anchor 0.2085, -10%) | |
| 43 | rank_ic | 0.07500 | 0.06304 | -0.01196 |
| 43 | ir | 2.08932 | 1.08661 | -1.00271 |
| 43 | maxdd | -0.08020 | -0.05094 | +0.02925 |
| 43 | ppd | 0.00194 | 0.00092 | -0.00101 |
| 43 | mechanism | — | score_std=0.1917 (anchor 0.2039, -6%) | |
| 44 | rank_ic | 0.07857 | 0.06568 | -0.01289 |
| 44 | ir | 1.62313 | 0.86827 | -0.75486 |
| 44 | maxdd | -0.08045 | -0.08221 | -0.00176 |
| 44 | ppd | 0.00348 | 0.00659 | +0.00312 |
| 44 | mechanism | — | score_std=0.2076 (anchor 0.2139, -3%) | |

**h1_dropext paired ΔRankIC = -0.01360 ± 0.00208 (n=3); seeds ≥+0.003: 0/3** → KILL line fires

_Cards: h-20260610-001 / h-20260610-002. n=3 PASS only escalates to n≥6 + HC kill-check._
