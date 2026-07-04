# full135 Deterministic Seed42 n_drop=30 Diagnostic Readout

Run:

- MLflow run: `mlruns/239161070249761302/689cd5c9181a4f329bd281d4a5116e49`
- Diagnostic matrix: `mlruns/239161070249761302/689cd5c9181a4f329bd281d4a5116e49/diagnostic_matrix.csv`
- Setting: full135, seed 42, deterministic warn mode, sampled daily training sampler, `TopkDropoutStrategy.n_drop=30`
- Train stop: epoch 17, restored best `loss_main=1.347290`

## Effect Metrics

| metric | value |
| --- | ---: |
| daily RankIC mean | 0.078232 |
| RankICIR | 0.572067 |
| daily IC mean | 0.067585 |
| rolling20 RankIC mean | 0.079193 |
| rolling60 RankIC mean | 0.078716 |
| top-bottom spread | 0.011163 |
| annualized return with cost | -0.037488 |
| annualized return without cost | 0.356466 |
| cost drag | 0.393954 |
| IR with cost | -0.314254 |
| max drawdown with cost | -0.152931 |
| turnover | 0.827630 |

Signal quality is in the expected full135 band. Portfolio quality with cost is
poor because turnover and cost drag dominate the positive no-cost return.

## Runtime

| metric | value |
| --- | ---: |
| seed | 42 |
| cuda_available | 1 |
| cuda_device_count | 1 |
| deterministic_algorithms | 1 |
| deterministic_warn_only | 1 |
| cudnn_deterministic | 1 |
| cudnn_benchmark | 0 |
| seed_workers | 1 |
| sampler_diag | 1 |

The run used CUDA with deterministic algorithms in warn-only mode. PyTorch
warned that memory-efficient attention uses a nondeterministic CUDA path; this
matches the requested warn-only policy.

## sampler/*

| metric | value |
| --- | ---: |
| coverage_ratio_last | 0.661333 |
| coverage_gap_vs_full_last | 0.338667 |
| duplicate_rate_last | 0.366382 |
| unique_samples_last | 566264 |
| total_source_samples_last | 856246 |
| total_draws_last | 893700 |
| upsample_day_ratio_last | 0.965760 |
| upsample_days_last | 2877 |
| exact_days_last | 102 |
| downsample_day_ratio_last | 0 |

This confirms a large train/eval sampler mismatch. The fixed sampled daily
sampler covers about two thirds of source samples per epoch and creates about
one third duplicate draws.

## router_oracle/*

| metric | value |
| --- | ---: |
| default_rank_ic_mean | 0.078229 |
| time_rank_ic_mean | 0.063163 |
| factor_rank_ic_mean | -0.020108 |
| uniform_rank_ic_mean | 0.056389 |
| oracle_expert_rank_ic_mean | 0.097621 |
| router_oracle_gap | 0.019391 |
| router_time_advantage | -0.015067 |
| router_factor_advantage | -0.098338 |
| router_uniform_advantage | -0.021841 |
| default_beats_oracle_expert_day_ratio | 0.451718 |
| time_beats_factor_day_ratio | 0.671031 |
| n_days | 611 |

Default learned routing beats all static forced choices, especially forced
factor. However, the per-day oracle gap is large. This points to router
calibration or day-conditional blending, not to forcing one expert globally.

## expert/*

| metric | layer 0 | layer 1 |
| --- | ---: | ---: |
| time_expert_norm_daily_mean | 15.844334 | 28.174532 |
| factor_expert_norm_daily_mean | 4.967723 | 11.451424 |
| time_contrib_norm_daily_mean | 11.979184 | 20.572036 |
| factor_contrib_norm_daily_mean | 1.211294 | 3.105014 |
| contrib_norm_ratio_daily_mean | 11.074085 | 7.168853 |
| expert_cosine_daily_mean | 0.024632 | 0.003037 |
| time_winner_ratio_daily_mean | 1.000000 | 0.997223 |

The experts are not collinear, but the contribution path is strongly
time-dominated. Factor-only is weak, yet default routing still beats forced
time, so the factor path is acting as a small but useful blend rather than as a
standalone expert.

## temporal/*

| metric | value |
| --- | ---: |
| forced_time_rank_ic_contribution | -0.015067 |
| embedding_norm_mean | 0.125787 |
| embedding_to_value_norm_ratio_mean | 0.669873 |
| tau_range_utilization_daily_mean | 0 |
| tau_range_utilization | 0.000234 |
| attention_diag_mass_mean | 0.130564 |
| attention_local_mass_mean | 0.355408 |
| attention_long_range_mass_mean | 0.644592 |

The temporal path is active, but forced-time underperforms the default blend.
Tau is still nearly collapsed. The result supports diagnosing temporal
calibration and readout, not making the model more time-only.

## Bottleneck Ranking From This Run

1. Portfolio/execution layer under `n_drop=30`: signal is good, but turnover
   `0.828` and cost drag `0.394` make cost-adjusted IR negative.
2. Router calibration: `router_oracle_gap=0.0194` is far larger than the
   architectural deltas currently being debated.
3. Sampler mismatch: `coverage_gap=0.3387` and `duplicate_rate=0.3664` are large
   enough to justify an opt-in `full_daily` falsifier.
4. Time dominance without forced-time superiority: time contribution wins almost
   every day, but forced-time RankIC is lower than default by `0.0151`.
5. Tau/pooling under-adaptation: tau range utilization remains near zero, factor
   attention and pooling are almost uniform, and FiLM/pooling overlap is low
   (`0.0454`).

## Next Direction

Keep full135 and regime state unchanged. The highest-yield next tests are:

1. Router calibration falsifier: preserve soft mixture, but test whether a
   calibrated router can recover part of the `0.0194` oracle gap without hurting
   RankICIR.
2. Sampler falsifier: one opt-in `full_daily` run to test whether coverage gap
   reduction improves RankIC/RankICIR or only changes training statistics.
3. Portfolio-side cost falsifier: because no-cost return is positive but
   cost-adjusted IR is negative under `n_drop=30`, test turnover control as a
   portfolio layer, separate from model architecture.
4. Tau/pooling belt-parameter tests should remain separate under HC-6. Do not
   bundle them with sampler or router changes.
