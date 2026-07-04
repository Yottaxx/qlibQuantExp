# Weekend matrix vs g012 anchor — h12_combo (25ep, scale=0.5, anchor settings)

Anchor (n=3): rank_ic 0.08163/0.07500/0.07857 (mean 0.07840). Kill: paired ΔRankIC < +0.003. 001 extra kill: valid score_std collapse >30%. 002 extra kill: stock_ratio<0.05 or attention degenerate (entropy_norm>0.999 uniform-collapse / self_frac>0.95).


## h12_combo vs anchor (paired by seed)
| seed | metric | anchor | variant | Δ |
|---|---|---:|---:|---:|
| 42 | rank_ic | 0.08163 | 0.07773 | -0.00390 |
| 42 | ir | 1.53606 | 1.75589 | +0.21983 |
| 42 | maxdd | -0.06634 | -0.08821 | -0.02187 |
| 42 | ppd | 0.00210 | 0.00163 | -0.00048 |
| 42 | mechanism | — | score_std=0.2047 (anchor 0.2085, -2%); stock_ratio=0.4532; router_no_stock_delta=0.0185 | |
| 43 | rank_ic | 0.07500 | 0.06936 | -0.00563 |
| 43 | ir | 2.08932 | 1.51444 | -0.57488 |
| 43 | maxdd | -0.08020 | -0.08772 | -0.00752 |
| 43 | ppd | 0.00194 | 0.00146 | -0.00047 |
| 43 | mechanism | — | score_std=0.1948 (anchor 0.2039, -4%); stock_ratio=0.4876; router_no_stock_delta=0.0074 | |
| 44 | rank_ic | 0.07857 | 0.06874 | -0.00983 |
| 44 | ir | 1.62313 | 1.44490 | -0.17823 |
| 44 | maxdd | -0.08045 | -0.10738 | -0.02693 |
| 44 | ppd | 0.00348 | 0.00146 | -0.00202 |
| 44 | mechanism | — | score_std=0.1911 (anchor 0.2139, -11%); stock_ratio=0.4086; router_no_stock_delta=0.0092 | |

**h12_combo paired ΔRankIC = -0.00645 ± 0.00305 (n=3); seeds ≥+0.003: 0/3** → KILL line fires

_Cards: h-20260610-001 / h-20260610-002. n=3 PASS only escalates to n≥6 + HC kill-check._
