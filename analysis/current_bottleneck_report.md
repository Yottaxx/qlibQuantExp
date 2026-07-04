# Current Experiment Bottleneck Report

Generated: 2026-05-23T13:58:58+08:00

## Executive Summary

Parsed 34 experiment inventory rows and 29 rows with valid RankIC. The current evidence supports keeping full135 as the working setting, not promoting cap06 or changing regime state.

- 1. `generalization_gap`: Top settings have mean train-valid RankIC gap around 0.126. Next: Keep full135 settings; improve deterministic runtime, sampler parity, and checkpoint discipline before changing model family.
- 2. `weak_temporal_path`: time_only explains only 9.3% of full_135 - base RankIC delta in the seed42 structural matrix. Next: Instrument time expert contribution and temporal readout before adding temporal architecture.
- 3. `router_edge_not_robust`: full_135 beats fixed_router_05 by RankIC 0.002064, but RankICIR delta is -0.026478. Next: Add router/expert-winner diagnostics; do not make router more expressive yet.
- 4. `capacity_signal_single_seed`: cap06 is strong at RankIC 0.079374, but n=1. Next: Do not promote cap06 to default; keep full135 until deterministic baseline is locked.
- 5. `regime_alignment_needs_measurement`: Strong monitored association: performance/rolling20_rank_ic_mean vs RankIC r=0.992 (n=23). Next: Use existing regime/router metrics to explain behavior; do not change regime state in this step.

## Inputs And Coverage

| Input | Count |
| --- | --- |
| experiment inventory rows | 34 |
| valid RankIC rows | 29 |
| aggregate rows | 17 |
| suite summary files | 6 |
| experiments ledger rows | 2 |

Source distribution: `{"diagnostic_manifest": 24, "repro_stdout": 1, "suite_csv": 9}`

## 1. Effect Metrics

Top existing runs by RankIC:

| setting | seed | RankIC | RankICIR | IC | port IR | ann ret | max DD | gap | decay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap00_d64_h4_l2_ff128_do010 | 43 | 0.079639 | 0.583296 | 0.062595 | 1.939469 | 0.190911 | -0.111137 | 0.121091 | 0.005859 |
| cap06_d96_h4_l2_ff384_do020 | 42 | 0.079374 | 0.541603 | 0.068932 | 2.086089 | 0.220231 | -0.080610 | 0.120013 | 0.004990 |
| full_135 | 42 | 0.079135 | 0.549612 | 0.070520 | 1.564012 | 0.177875 | -0.098476 | 0.120813 | 0.009061 |
| cap02_d80_h4_l2_ff320_do015 | 44 | 0.079006 | 0.545007 | 0.066127 | 1.951954 | 0.200778 | -0.069323 | 0.116169 | 0.004782 |
| cap01_d64_h4_l2_ff256_do015 | 44 | 0.078119 | 0.549941 | 0.062510 | 2.253668 | 0.233283 | -0.073408 | 0.109773 | 0.002381 |
| cap11_d128_h4_l2_ff512_do020 | 42 | 0.077561 | 0.527095 | 0.064818 | 1.375004 | 0.151711 | -0.096802 | 0.133664 | 0.006845 |
| cap07_d96_h6_l2_ff384_do015 | 42 | 0.077492 | 0.530411 | 0.065184 | 1.584894 | 0.166840 | -0.086644 | 0.127671 | 0.005703 |
| cap05_d96_h4_l2_ff384_do015 | 42 | 0.077287 | 0.543495 | 0.064032 | 1.754429 | 0.183458 | -0.077819 | 0.128798 | 0.005057 |
| full_135 |  | 0.077104 | 0.551329 | 0.068568 | 1.910464 | 0.210183 | -0.068433 |  | 0.000000 |
| cap02_d80_h4_l2_ff320_do015 | 43 | 0.077080 | 0.553943 | 0.067582 | 1.870706 | 0.206997 | -0.077310 | 0.116641 | 0.003816 |
| fixed_router_05 | 42 | 0.077071 | 0.576090 | 0.067545 | 1.534843 | 0.154921 | -0.083565 | 0.110751 | 0.007323 |
| film_only | 42 | 0.076786 | 0.584437 | 0.064158 | 1.575226 | 0.179788 | -0.074843 | 0.143250 | 0.019227 |

Aggregate by canonical setting:

| setting | n | seeds | RankIC mean | RankIC std | RankICIR | port IR | gap | decay | time_ratio | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap06_d96_h4_l2_ff384_do020 | 1 | 42 | 0.079374 | 0.000000 | 0.541603 | 2.086089 | 0.120013 | 0.004990 | 0.703779 | 0.835957 |
| full_135 | 2 | 42 | 0.078120 | 0.001436 | 0.550470 | 1.737238 | 0.120813 | 0.004530 | 0.663034 | 0.897017 |
| cap11_d128_h4_l2_ff512_do020 | 1 | 42 | 0.077561 | 0.000000 | 0.527095 | 1.375004 | 0.133664 | 0.006845 | 0.570092 | 0.947984 |
| cap07_d96_h6_l2_ff384_do015 | 1 | 42 | 0.077492 | 0.000000 | 0.530411 | 1.584894 | 0.127671 | 0.005703 | 0.678328 | 0.878851 |
| cap05_d96_h4_l2_ff384_do015 | 1 | 42 | 0.077287 | 0.000000 | 0.543495 | 1.754429 | 0.128798 | 0.005057 | 0.632786 | 0.915339 |
| cap02_d80_h4_l2_ff320_do015 | 3 | 42,43,44 | 0.077077 | 0.001931 | 0.541638 | 2.029287 | 0.120569 | 0.005720 | 0.657334 | 0.889872 |
| cap10_d128_h8_l2_ff512_do025 | 1 | 42 | 0.076664 | 0.000000 | 0.520999 | 1.519312 | 0.131417 | 0.007855 | 0.487361 | 0.936345 |
| cap00_d64_h4_l2_ff128_do010 | 3 | 42,43,44 | 0.076259 | 0.003657 | 0.559786 | 1.796348 | 0.122095 | 0.007050 | 0.698650 | 0.850628 |
| cap09_d128_h8_l2_ff384_do020 | 1 | 42 | 0.075703 | 0.000000 | 0.494299 | 1.868388 | 0.142902 | 0.013711 | 0.574535 | 0.947593 |
| cap01_d64_h4_l2_ff256_do015 | 3 | 42,43,44 | 0.075677 | 0.002150 | 0.537294 | 1.904755 | 0.112264 | 0.003295 | 0.688249 | 0.875904 |
| cap03_d96_h4_l2_ff288_do015 | 1 | 42 | 0.075351 | 0.000000 | 0.510369 | 1.612065 | 0.124596 | 0.005212 | 0.613785 | 0.937776 |
| cap04_d96_h4_l2_ff288_do020 | 1 | 42 | 0.074528 | 0.000000 | 0.479963 | 1.442828 | 0.127606 | 0.009005 | 0.638969 | 0.905658 |
| film_only | 2 | 42 | 0.074355 | 0.003439 | 0.558970 | 1.510175 | 0.143250 | 0.009613 | 0.734169 | 0.807584 |
| fixed_router_05 | 2 | 42 | 0.074105 | 0.004195 | 0.549758 | 1.513951 | 0.110751 | 0.006270 | 0.500000 | 1.000000 |
| time_only | 2 | 42 | 0.071863 | 0.000152 | 0.537193 | 1.499979 | 0.099784 | 0.004164 | 0.538997 | 0.897593 |
| base | 2 | 42 | 0.069560 | 0.002370 | 0.512162 | 1.193561 | 0.106377 | 0.003507 | 0.691962 | 0.840082 |
| no_layer_summary | 2 | 42 | 0.066126 | 0.004176 | 0.494826 | 1.477007 | 0.112985 | 0.005696 | 0.646656 | 0.893887 |

## 2. Training Process

The strongest settings still show large train-valid RankIC gaps. This is a bottleneck because most architectural deltas are smaller than the observed gap.

| setting | n | RankIC | gap | decay | RankICIR |
| --- | --- | --- | --- | --- | --- |
| cap06_d96_h4_l2_ff384_do020 | 1 | 0.079374 | 0.120013 | 0.004990 | 0.541603 |
| full_135 | 2 | 0.078120 | 0.120813 | 0.004530 | 0.550470 |
| cap11_d128_h4_l2_ff512_do020 | 1 | 0.077561 | 0.133664 | 0.006845 | 0.527095 |
| cap07_d96_h6_l2_ff384_do015 | 1 | 0.077492 | 0.127671 | 0.005703 | 0.530411 |
| cap05_d96_h4_l2_ff384_do015 | 1 | 0.077287 | 0.128798 | 0.005057 | 0.543495 |
| cap02_d80_h4_l2_ff320_do015 | 3 | 0.077077 | 0.120569 | 0.005720 | 0.541638 |
| cap10_d128_h8_l2_ff512_do025 | 1 | 0.076664 | 0.131417 | 0.007855 | 0.520999 |
| cap00_d64_h4_l2_ff128_do010 | 3 | 0.076259 | 0.122095 | 0.007050 | 0.559786 |
| cap09_d128_h8_l2_ff384_do020 | 1 | 0.075703 | 0.142902 | 0.013711 | 0.494299 |
| cap01_d64_h4_l2_ff256_do015 | 3 | 0.075677 | 0.112264 | 0.003295 | 0.537294 |

## 3. Routing Health

Router collapse is not the main failure mode, but learned routing has not clearly beaten fixed routing on stability. The monitored correlations below show which diagnostics move with RankIC in existing runs.

| metric | Pearson r vs RankIC | n |
| --- | --- | --- |
| performance/rolling20_rank_ic_mean | 0.992 | 23 |
| performance/rolling60_rank_ic_mean | 0.975 | 23 |
| performance/rank_icir | 0.623 | 29 |
| portfolio/annualized_return_with_cost | 0.619 | 29 |
| portfolio/information_ratio_with_cost | 0.618 | 29 |
| performance/daily_ic_mean | 0.610 | 29 |
| optimization/train_score_std_last | 0.574 | 23 |
| router/time_ratio_vs_pc1_spearman | -0.569 | 22 |
| regime/pc1_tail_2x2_rank_ic_spread | 0.560 | 29 |
| router/time_ratio_vs_corr_mean_abs_spearman | -0.558 | 22 |
| time_embedding/tau_vs_rank_ic_spearman | -0.494 | 21 |
| attention_pooling/pooling_top10_vs_film_top10_overlap | -0.494 | 21 |
| optimization/loss_gap_last | 0.469 | 23 |
| router/layer_1_collapse_ratio_last | -0.464 | 23 |
| time_embedding/tau_vs_tail_spearman | -0.460 | 21 |
| optimization/train_lr_last | -0.444 | 23 |
| performance/top_bottom_spread_mean | 0.444 | 23 |
| portfolio/top_bottom_spread | 0.444 | 23 |

## 4. Structural Contribution

Seed42 structural matrix, preferring manifest-backed `matrix_40epoch_seed42` rows:

| setting | seed | RankIC | delta vs base | RankICIR | port IR | gap | decay | time_ratio | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 42 | 0.071236 | 0.000000 | 0.539073 | 1.369382 | 0.106377 | 0.003621 | 0.688188 | 0.840082 |
| time_only | 42 | 0.071970 | 0.000734 | 0.548347 | 1.575366 | 0.099784 | 0.005970 | 0.542891 | 0.897593 |
| film_only | 42 | 0.076786 | 0.005550 | 0.584437 | 1.575226 | 0.143250 | 0.019227 | 0.734945 | 0.807584 |
| fixed_router_05 | 42 | 0.077071 | 0.005835 | 0.576090 | 1.534843 | 0.110751 | 0.007323 | 0.500000 | 1.000000 |
| full_135 | 42 | 0.079135 | 0.007899 | 0.549612 | 1.564012 | 0.120813 | 0.009061 | 0.670812 | 0.897017 |
| no_layer_summary | 42 | 0.069079 | -0.002158 | 0.514995 | 1.843191 | 0.112985 | 0.005027 | 0.647826 | 0.893887 |

- film_only share of full_135 - base RankIC delta: 70.3%.
- time_only share of full_135 - base RankIC delta: 9.3%.
- full_135 - fixed_router_05 RankIC: 0.002064; RankICIR: -0.026478.

## 5. Capacity Impact

Capacity results are non-monotonic and mostly single-seed. cap06 is promising but not a new default.

| setting | n | seeds | d_model | heads | layers | d_ff | dropout | RankIC | std | RankICIR | gap | decay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap00_d64_h4_l2_ff128_do010 | 3 | 42,43,44 | 64 | 4 | 2 | 128 | 0.1 | 0.076259 | 0.003657 | 0.559786 | 0.122095 | 0.007050 |
| cap01_d64_h4_l2_ff256_do015 | 3 | 42,43,44 | 64 | 4 | 2 | 256 | 0.15 | 0.075677 | 0.002150 | 0.537294 | 0.112264 | 0.003295 |
| cap02_d80_h4_l2_ff320_do015 | 3 | 42,43,44 | 80 | 4 | 2 | 320 | 0.15 | 0.077077 | 0.001931 | 0.541638 | 0.120569 | 0.005720 |
| cap03_d96_h4_l2_ff288_do015 | 1 | 42 | 96 | 4 | 2 | 288 | 0.15 | 0.075351 | 0.000000 | 0.510369 | 0.124596 | 0.005212 |
| cap04_d96_h4_l2_ff288_do020 | 1 | 42 | 96 | 4 | 2 | 288 | 0.2 | 0.074528 | 0.000000 | 0.479963 | 0.127606 | 0.009005 |
| cap05_d96_h4_l2_ff384_do015 | 1 | 42 | 96 | 4 | 2 | 384 | 0.15 | 0.077287 | 0.000000 | 0.543495 | 0.128798 | 0.005057 |
| cap06_d96_h4_l2_ff384_do020 | 1 | 42 | 96 | 4 | 2 | 384 | 0.2 | 0.079374 | 0.000000 | 0.541603 | 0.120013 | 0.004990 |
| cap07_d96_h6_l2_ff384_do015 | 1 | 42 | 96 | 6 | 2 | 384 | 0.15 | 0.077492 | 0.000000 | 0.530411 | 0.127671 | 0.005703 |
| cap09_d128_h8_l2_ff384_do020 | 1 | 42 | 128 | 8 | 2 | 384 | 0.2 | 0.075703 | 0.000000 | 0.494299 | 0.142902 | 0.013711 |
| cap10_d128_h8_l2_ff512_do025 | 1 | 42 | 128 | 8 | 2 | 512 | 0.25 | 0.076664 | 0.000000 | 0.520999 | 0.131417 | 0.007855 |
| cap11_d128_h4_l2_ff512_do020 | 1 | 42 | 128 | 4 | 2 | 512 | 0.2 | 0.077561 | 0.000000 | 0.527095 | 0.133664 | 0.006845 |

Failed or incomplete rows:

| setting | seed | returncode | stderr/path |
| --- | --- | --- | --- |
| cap08_d96_h4_l3_ff384_do020 | 42 | 1 | diagnostic_runs\four_day_plan_20260501_125906\capacity_full135_40epoch\stage1_seed42\cap08_d96_h4_l3_ff384_do020\stderr.log |
| cap08_d96_h4_l3_ff384_do020 |  | 1 |  |
| cap08_d96_h4_l3_ff384_do020 | 42 | 1 |  |

## Evidence Classification

### Confirmed Facts
- film_only explains 70.3% of the seed42 full_135 - base RankIC delta.
- time_only is weak in the seed42 matrix: delta vs base is 0.000734.
- learned router beats fixed_router_05 in RankIC by 0.002064, but fixed router has better RankICIR by 0.026478.
- Top settings show large train-valid RankIC gap: mean gap 0.126.
- 3 nonzero-returncode rows found; cap08/depth-3 remains unevaluated.

### Single-Seed Hints
- cap06 is the best aggregate setting by RankIC (0.079374), but it is single-seed.

### Insufficient Evidence
- cap06 cannot become the new default without more evidence; keep full135 for now.
- Most capacity settings have n=1; capacity scaling is not established.
- The qx ledger is not yet a complete mirror of the May diagnostic runs.

### Needs New Experiment
- Sampler train/eval mismatch may be consuming small RankIC gains; instrument duplicate rate before changing architecture.
- Temporal path may need measurement or readout fixes, but no temporal architecture change is justified yet.
- Router calibration should be diagnosed against expert-winner advantage before adding router capacity.

## Bottleneck Ranking And Next Direction

| rank | bottleneck | evidence | next direction |
| --- | --- | --- | --- |
| 1 | generalization_gap | Top settings have mean train-valid RankIC gap around 0.126. | Keep full135 settings; improve deterministic runtime, sampler parity, and checkpoint discipline before changing model family. |
| 2 | weak_temporal_path | time_only explains only 9.3% of full_135 - base RankIC delta in the seed42 structural matrix. | Instrument time expert contribution and temporal readout before adding temporal architecture. |
| 3 | router_edge_not_robust | full_135 beats fixed_router_05 by RankIC 0.002064, but RankICIR delta is -0.026478. | Add router/expert-winner diagnostics; do not make router more expressive yet. |
| 4 | capacity_signal_single_seed | cap06 is strong at RankIC 0.079374, but n=1. | Do not promote cap06 to default; keep full135 until deterministic baseline is locked. |
| 5 | regime_alignment_needs_measurement | Strong monitored association: performance/rolling20_rank_ic_mean vs RankIC r=0.992 (n=23). | Use existing regime/router metrics to explain behavior; do not change regime state in this step. |

Recommended direction under the current constraint:
- Keep full135 settings.
- Do not change regime state in the next step.
- First implement deterministic seed/CUDA handling and sampler parity instrumentation.
- Add router/expert monitoring and temporal-path diagnostics before changing loss, capacity, or macV3-style architecture.

Do not prioritize loss changes, capacity scaling, regime-state changes, or macV3-style bundles until these bottlenecks are closed.
