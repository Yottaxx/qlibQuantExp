# Weekend matrix vs g012 anchor — h2_stockexp (25ep, scale=0.5, anchor settings)

Anchor (n=3): rank_ic 0.08163/0.07500/0.07857 (mean 0.07840). Kill: paired ΔRankIC < +0.003. 001 extra kill: valid score_std collapse >30%. 002 extra kill: stock_ratio<0.05 or attention degenerate (entropy_norm>0.999 uniform-collapse / self_frac>0.95).


## h2_stockexp vs anchor (paired by seed)
| seed | metric | anchor | variant | Δ |
|---|---|---:|---:|---:|
| 42 | rank_ic | 0.08163 | 0.07817 | -0.00346 |
| 42 | ir | 1.53606 | -0.03671 | -1.57277 |
| 42 | maxdd | -0.06634 | -0.33691 | -0.27058 |
| 42 | ppd | 0.00210 | 0.00425 | +0.00214 |
| 42 | mechanism | — | stock_ratio=0.3874; router_no_stock_delta=0.0110; router_stock_advantage=-0.0535; expert_cosine_ts=0.1276; expert_cosine_fs=0.0663; stock_attn_entropy_norm=1.0000; stock_attn_self_frac=0.0033 | |
| 43 | rank_ic | 0.07500 | 0.07676 | +0.00176 |
| 43 | ir | 2.08932 | 1.65416 | -0.43516 |
| 43 | maxdd | -0.08020 | -0.08085 | -0.00065 |
| 43 | ppd | 0.00194 | 0.00683 | +0.00489 |
| 43 | mechanism | — | stock_ratio=0.4033; router_no_stock_delta=0.0182; router_stock_advantage=-0.0482; expert_cosine_ts=-0.0988; expert_cosine_fs=-0.0354; stock_attn_entropy_norm=1.0000; stock_attn_self_frac=0.0033 | |
| 44 | rank_ic | 0.07857 | 0.07840 | -0.00017 |
| 44 | ir | 1.62313 | 1.46274 | -0.16039 |
| 44 | maxdd | -0.08045 | -0.12712 | -0.04667 |
| 44 | ppd | 0.00348 | 0.00922 | +0.00574 |
| 44 | mechanism | — | stock_ratio=0.3674; router_no_stock_delta=0.0133; router_stock_advantage=-0.0626; expert_cosine_ts=-0.3775; expert_cosine_fs=-0.1020; stock_attn_entropy_norm=1.0000; stock_attn_self_frac=0.0033 | |

**h2_stockexp paired ΔRankIC = -0.00062 ± 0.00264 (n=3); seeds ≥+0.003: 0/3** → KILL line fires

_Cards: h-20260610-001 / h-20260610-002. n=3 PASS only escalates to n≥6 + HC kill-check._
