# Pool/readout forensics — aggregate report

- seeds: [42, 43, 44, 45, 46, 47]  | runs found: 6
- sanity valid_rank_ic per seed: ['0.08106', '0.07881', '0.07235', '0.07809', '0.07975', '0.07463'] (promoted anchor ~0.0768)

## KEY-SVD-GATE (factor pool: A operator / B input / C live-projection)
- current softmax entropy (six-nines check): **0.99990**
- key s1/common-norm (across-factor key contrast size): **2.9033**
- key sigma2/sigma1: 0.2996  | key effective rank: 2.75
- peakability entropy (unit-aligned top key dir) at target logit-std: 0.5=0.9757 1=0.9134 2=0.7844 3=0.7107 5=0.6362
- current logit-std (operator coldness): 0.027292
- value-channel s1/common-norm (world-C: value bank varies across factors?): 1.6085
- cross-stock attention-weight std (is attention even stock-varying?): 0.000087
- per-seed verdicts: ['operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse']
- **aggregate world verdict: operator_state_collapse**

## PROBE-CEILING (does non-uniform factor reweighting help on frozen reps?)
| seed | arm0 (mean-pool) | armA (convex-reweight) | gap |
|---|---:|---:|---:|
| 42 | 0.06959 | 0.06527 | -0.00432 |
| 43 | 0.07524 | 0.07450 | -0.00074 |
| 44 | 0.06046 | 0.04775 | -0.01271 |
| 45 | 0.07337 | 0.07337 | -0.00000 |
| 46 | 0.07342 | 0.07446 | 0.00104 |
| 47 | 0.06987 | 0.06810 | -0.00177 |
- gap mean: **-0.00700**  | seeds with gap>=+0.003: 0/6
- Arm B slot-IC decile ratio (mean): 1.40 (need >=2.0)
- Arm B slot-signal cov effective rank (mean): 6.09 (need >=2 = non-degenerate)
- Arm B max |slot-IC| (mean): 0.0740
- **sharpening_can_help: False** | rational-equilibrium null supported: True

