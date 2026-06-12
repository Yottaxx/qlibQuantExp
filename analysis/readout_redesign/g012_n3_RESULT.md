# g0.12 d1pma — 3-seed kill-check vs scale=0.5 control (paired by seed)

Pre-registered: kill if paired ΔRankIC mean < +0.003 OR IR/MaxDD clearly worse. The gate is frozen at init (mechanism), so g0.12 ≈ '12% attention mixture' — prior says parity-not-edge.

| seed | metric | control | g0.12 | Δ |
|---|---|---:|---:|---:|
| 42 | rank_ic | 0.08107 | 0.08163 | +0.00056 |
| 42 | ir | 2.24625 | 1.53606 | -0.71019 |
| 42 | maxdd | -0.06003 | -0.06634 | -0.00631 |
| 42 | ppd | 0.00475 | 0.00210 | -0.00265 |
| 42 | final_gate_g | — | 0.124 | (init 0.119) |
| 43 | rank_ic | 0.07881 | 0.07500 | -0.00381 |
| 43 | ir | 1.77380 | 2.08932 | +0.31552 |
| 43 | maxdd | -0.08060 | -0.08020 | +0.00040 |
| 43 | ppd | 0.00413 | 0.00194 | -0.00219 |
| 43 | final_gate_g | — | 0.125 | (init 0.119) |
| 44 | rank_ic | 0.07235 | 0.07857 | +0.00622 |
| 44 | ir | 1.72082 | 1.62313 | -0.09769 |
| 44 | maxdd | -0.08257 | -0.08045 | +0.00212 |
| 44 | ppd | 0.00321 | 0.00348 | +0.00027 |
| 44 | final_gate_g | — | 0.124 | (init 0.119) |

## Verdict
- paired ΔRankIC = **+0.00099 ± 0.00503** (n=3); seeds ≥+0.003: 1/3
- portfolio: ΔIR clearly-worse seeds 1/3; ΔMaxDD clearly-worse 0/3
- **KILL: g0.12 d1pma is parity-at-best — attention readout CLOSED for good (no init flavor rescues it). Last-step control stands.**

_Card: time-readout-bonus-20260607 / g012-n3 kill-check._
