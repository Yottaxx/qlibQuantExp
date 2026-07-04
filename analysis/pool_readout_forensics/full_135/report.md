# Pool/readout forensics — aggregate report

- seeds: [42, 43, 44, 45, 46, 47]  | runs found: 6
- sanity valid_rank_ic per seed: ['0.08020', '0.07794', '0.07347', '0.08103', '0.07916', '0.07671'] (promoted anchor ~0.0768)

## KEY-SVD-GATE (factor pool: A operator / B input / C live-projection)
- current softmax entropy (six-nines check): **0.99995**
- key s1/common-norm (across-factor key contrast size): **2.1734**
- key sigma2/sigma1: 0.4295  | key effective rank: 4.21
- peakability entropy (unit-aligned top key dir) at target logit-std: 0.5=0.9761 1=0.9109 2=0.7539 3=0.6590 5=0.5639
- current logit-std (operator coldness): 0.017261
- value-channel s1/common-norm (world-C: value bank varies across factors?): 1.6510
- cross-stock attention-weight std (is attention even stock-varying?): 0.000055
- per-seed verdicts: ['operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse', 'operator_state_collapse']
- **aggregate world verdict: operator_state_collapse**

## PROBE-CEILING (does non-uniform factor reweighting help on frozen reps?)
| seed | arm0 (mean-pool) | armA (convex-reweight) | gap |
|---|---:|---:|---:|
| 42 | 0.07147 | 0.06942 | -0.00205 |
| 43 | 0.07792 | 0.07739 | -0.00054 |
| 44 | 0.05742 | 0.05587 | -0.00155 |
| 45 | 0.07650 | 0.07650 | 0.00000 |
| 46 | 0.07335 | 0.07348 | 0.00013 |
| 47 | 0.07288 | 0.07288 | -0.00000 |
- gap mean: **-0.00762**  | seeds with gap>=+0.003: 0/6
- Arm B slot-IC decile ratio (mean): 1.45 (need >=2.0)
- Arm B slot-signal cov effective rank (mean): 6.07 (need >=2 = non-degenerate)
- Arm B max |slot-IC| (mean): 0.0746
- **sharpening_can_help: False** | rational-equilibrium null supported: True

