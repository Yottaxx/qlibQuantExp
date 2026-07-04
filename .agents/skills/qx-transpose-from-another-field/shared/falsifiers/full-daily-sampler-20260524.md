# Falsifier Card - Full-Daily Sampler Parity

## Metadata

- **claim_id:** `full-daily-sampler-20260524`
- **filed_at:** 2026-05-24
- **filed_by:** Codex via `qx-forge-a-sharp-falsifier`
- **status:** open / not dispatched
- **anchor_commit:** `3675090`
- **anchor_run:** `mlruns/239161070249761302/689cd5c9181a4f329bd281d4a5116e49`
- **anchor_setting:** `full135_det_seed42_sampled_daily`
- **portfolio_guard_setting:** historical `full_135` seed42, `n_drop=5`
- **anchor_train_sampler_mode:** `sampled_daily`
- **candidate_train_sampler_mode:** `full_daily`
- **n_trials_at_filing:** one qx-logged deterministic full135 diagnostic run; historical experiment inventory has 29 valid RankIC runs, but this falsifier is anchored to the deterministic seed-42 run above.

## Claim

The current fixed daily sampled training sampler is a real train/eval mismatch bottleneck. Replacing it with the opt-in `full_daily` sampler, while keeping the `full135` architecture, regime state, seed policy, loss, capacity, and portfolio `n_drop=5` unchanged, should reduce the train-valid RankIC gap and preserve or improve out-of-sample ranking quality.

The risky prediction is:

> On the deterministic seed-42 `full135` anchor, `full_daily` will drive sampler coverage gap and duplicate rate to approximately zero, and will reduce `optimization/rank_ic_gap_last` by at least 0.020 absolute without degrading test `daily_rank_ic_mean` by more than 0.0015 or `rank_icir` by more than 0.015.

If this does not happen, the observed sampler diagnostics are descriptive pathology rather than a dominant performance lever.

## Anchor Facts

Training/sampler anchor from deterministic sampled-daily run `689cd5c9181a4f329bd281d4a5116e49`.
That run was recorded with an experimental `n_drop=30` portfolio layer, so only its training, sampler, prediction, RankIC, RankICIR, router, expert, and temporal diagnostics are used here.
`n_drop` does not affect training, sampler draws, predictions, RankIC, or RankICIR; it only affects the portfolio-analysis layer.

| Metric | Anchor value |
|---|---:|
| `performance/daily_rank_ic_mean` | 0.078232 |
| `performance/rank_icir` | 0.572067 |
| `performance/daily_ic_mean` | 0.067585 |
| `optimization/rank_ic_gap_last` | 0.081608 |
| `optimization/post_peak_decay` | 0.000549 |
| `sampler/coverage_gap_vs_full` | 0.338667 |
| `sampler/duplicate_rate` | 0.366382 |
| `sampler/upsample_day_ratio` | 0.965760 |
| `router_oracle/oracle_gap` | 0.019391 |
| `temporal/tau_range_utilization` | 0.000234 |

Portfolio guard anchor from historical `full_135` seed42 with the canonical `n_drop=5` setting:

| Metric | Guard value |
|---|---:|
| `portfolio/information_ratio_with_cost` | 1.564012 |
| `portfolio/annualized_return_with_cost` | 0.177875 |
| `portfolio/max_drawdown_with_cost` | -0.098476 |
| `portfolio/turnover` | 0.326519 |
| `portfolio/cost_drag` | 0.155310 |

## What This Theory Forbids

1. **Sampler mechanics failure.** If `full_daily` still reports `sampler/coverage_gap_vs_full > 0.005`, `sampler/duplicate_rate > 0.005`, or `sampler/upsample_day_ratio > 0.005`, then the implementation did not test the claim.
2. **No gap movement.** If `full_daily` reduces `optimization/rank_ic_gap_last` by less than 0.020 absolute on seed 42, then sampler mismatch is not the dominant generalization-gap lever under this setting.
3. **Gap-only cosmetic win.** If the gap improves but test `performance/daily_rank_ic_mean` falls by more than 0.0015, or `performance/rank_icir` falls by more than 0.015, then full coverage is regularizing the wrong objective.
4. **Portfolio harm.** If any HC-6 headline guard regresses materially (`IR_with_cost < anchor - 0.20`, `max_drawdown_with_cost < anchor - 0.010`, or `post_peak_decay > anchor + 0.002`), the sampler cannot be promoted as a default even if RankIC gap improves.
5. **Compute-budget confound.** If the number of optimizer steps per epoch differs by more than 5% from the sampled-daily anchor, the result is classified as a sampler-plus-compute-budget bundle and must be followed by a step-matched control before promotion.
6. **Order confound.** The training full-daily sampler must use seed+epoch-aware day-order shuffling, matching the stochastic-order discipline of `sampled_daily`. A chronological-only full-daily run is invalid for this falsifier because it changes both coverage and curriculum/order.

## Experimental Arms

### Phase A - Single-Seed Mechanism Test

Run one new arm:

| Arm | Seed | Changed field | Purpose |
|---|---:|---|---|
| `full135_full_daily_seed42_ndrop5_40e` | 42 | `trainer_config.train_sampler_mode="full_daily"`, `n_epochs=40`, `train_stop_threshold=null` | Test whether full day coverage fixes the measured sampler mismatch and reduces train-valid gap without early-stop confounding. |

Comparison anchor is already on disk:

| Arm | Seed | Sampler | Run |
|---|---:|---|---|
| `full135_det_seed42_sampled_daily` | 42 | `sampled_daily` | Training/sampler anchor: `689cd5c9181a4f329bd281d4a5116e49`; portfolio guard anchor uses historical `full_135` seed42 n_drop=5. |

### Phase B - Paired Seed Confirmation

Run only if Phase A passes the mechanism and quality gates.

| Arm | Seeds | Purpose |
|---|---|---|
| `full135_sampled_daily_ndrop5_40e` | 42, 43, 44 | Paired deterministic sampled-daily controls. |
| `full135_full_daily_ndrop5_40e` | 42, 43, 44 | Paired full-daily candidates. |

Promotion is based on the paired 3-seed aggregate `{42,43,44}`. All paired runs must complete 40 epochs; early train-loss threshold stopping is disabled for this hook.

## Fixed Configuration

The following must remain fixed across anchor and candidate arms:

- `full135` architecture settings.
- `use_regime_time_embedding=true`.
- `use_regime_factor_gate=true`.
- `router_use_layer_summary=true`.
- `router_mode="learned"`.
- Regime state inputs and `market_state_shift`.
- Loss function, optimizer, learning-rate schedule, capacity, dropout, and epoch policy.
- Deterministic runtime policy: `seed`, CUDA deterministic warn mode, fixed dataloader generator, seeded workers.
- Portfolio `n_drop=5`.
- Training full-daily sampler uses epoch-aware deterministic shuffle: same seed/epoch gives the same date order; different epochs reshuffle date order; every sample remains included exactly once.
- Paired-promotion runs use `n_epochs=40`, `early_stop=0`, `train_stop_threshold=null`, `min_epochs=40`, and `consecutive_k=999999`.
- Test, portfolio, and diagnostic artifacts must be generated after restoring the best validation-RankIC checkpoint: `checkpoint_metric="valid_rank_ic"`, `checkpoint_mode="max"`, `checkpoint_min_delta=0.0`.

No tau-scale, capacity, dropout, loss, regime-state, router, or macV3 change is allowed in this falsifier.

## Metrics

### Primary Decision Metrics

- `rank_ic_gap_at_checkpoint` for paired promotion; `optimization/rank_ic_gap_last` is retained only as a late-training decay diagnostic.
- `performance/daily_rank_ic_mean`
- `performance/rank_icir`

### Sampler Mechanism Metrics

- `sampler/coverage_ratio`
- `sampler/coverage_gap_vs_full`
- `sampler/duplicate_rate`
- `sampler/downsample_day_ratio`
- `sampler/upsample_day_ratio`
- `sampler/num_days`
- `sampler/total_draws`
- optimizer steps per epoch, if emitted by the recorder

### HC-6 Guard Metrics

- `portfolio/information_ratio_with_cost`
- `portfolio/annualized_return_with_cost`
- `portfolio/max_drawdown_with_cost`
- `optimization/post_peak_decay`

### Diagnostic Context Metrics

These do not decide the sampler claim, but must be read to prevent false stories:

- `router_oracle/default_rank_ic`
- `router_oracle/oracle_rank_ic`
- `router_oracle/oracle_gap`
- `router_oracle/time_advantage`
- `router_oracle/factor_advantage`
- `expert/layer_0_time_contribution_ratio`
- `expert/layer_1_time_contribution_ratio`
- `expert/layer_0_expert_cosine`
- `expert/layer_1_expert_cosine`
- `temporal/tau_range_utilization`
- `temporal/forced_time_contribution`

## Phase A Decision Rule

Promote to Phase B only if all are true:

1. `sampler/coverage_gap_vs_full <= 0.005`.
2. `sampler/duplicate_rate <= 0.005`.
3. `sampler/upsample_day_ratio <= 0.005`.
4. `optimization/rank_ic_gap_last <= 0.061608` (`anchor 0.081608 - 0.020`).
5. `performance/daily_rank_ic_mean >= 0.076732` (`anchor 0.078232 - 0.0015`).
6. `performance/rank_icir >= 0.557067` (`anchor 0.572067 - 0.015`).
7. No HC-6 guard breaches the thresholds in "What This Theory Forbids".
8. No optimizer-step confound above 5%.

If 1-3 pass but 4 fails, record the verdict as: "full_daily fixes sampler mechanics but sampler mismatch is not the dominant gap lever."

If 4 passes but 5 or 6 fails, record the verdict as: "full_daily regularizes training but removes useful stochasticity."

If 4-6 pass but a portfolio guard fails, record the verdict as: "full_daily improves ranking diagnostics but is not a promoted trading default."

## Phase B Promotion Rule

On paired seeds `{42,43,44}`, promote `full_daily` to the next baseline candidate only if:

1. Mean `optimization/rank_ic_gap_last` improves by at least 0.015 absolute versus paired sampled-daily controls.
2. Mean `performance/daily_rank_ic_mean` is not lower than paired sampled-daily controls.
3. Mean `performance/rank_icir` is not lower by more than 0.010.
4. At least three of the four HC-6 tuple elements are non-degrading: `{RankIC, IR_with_cost, MaxDD_with_cost, post_peak_decay}`.
5. No single seed has a catastrophic RankIC loss worse than -0.004 versus its paired sampled-daily control.

If Phase B fails, `full_daily` remains a diagnostic arm, not a default training sampler.

## Launch Commands

PowerShell command for Phase A:

```powershell
$env:QIB_RUN_SETTING = "full135_full_daily_seed42_ndrop5_40e"
$env:QIB_MODEL_OVERRIDES_JSON = '{"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'
$env:QIB_TRAINER_OVERRIDES_JSON = '{"seed":42,"device":"auto","deterministic_mode":"warn","seed_workers":true,"train_sampler_mode":"full_daily","sampler_diag":true,"use_tqdm":false,"n_epochs":40,"early_stop":0,"train_stop_threshold":null,"min_epochs":40,"consecutive_k":999999,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","checkpoint_min_delta":0.0}'
$env:QIB_PORT_OVERRIDES_JSON = '{"strategy":{"kwargs":{"n_drop":5}}}'
C:\Users\60585\miniconda3\envs\quantEnv\python.exe -u work_flow.py
```

PowerShell command template for paired controls:

```powershell
$env:QIB_RUN_SETTING = "full135_sampled_daily_seed43_ndrop5_40e"
$env:QIB_MODEL_OVERRIDES_JSON = '{"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'
$env:QIB_TRAINER_OVERRIDES_JSON = '{"seed":43,"device":"auto","deterministic_mode":"warn","seed_workers":true,"train_sampler_mode":"sampled_daily","sampler_diag":true,"use_tqdm":false,"n_epochs":40,"early_stop":0,"train_stop_threshold":null,"min_epochs":40,"consecutive_k":999999,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","checkpoint_min_delta":0.0}'
$env:QIB_PORT_OVERRIDES_JSON = '{"strategy":{"kwargs":{"n_drop":5}}}'
C:\Users\60585\miniconda3\envs\quantEnv\python.exe -u work_flow.py
```

PowerShell command template for paired candidates:

```powershell
$env:QIB_RUN_SETTING = "full135_full_daily_seed43_ndrop5_40e"
$env:QIB_MODEL_OVERRIDES_JSON = '{"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'
$env:QIB_TRAINER_OVERRIDES_JSON = '{"seed":43,"device":"auto","deterministic_mode":"warn","seed_workers":true,"train_sampler_mode":"full_daily","sampler_diag":true,"use_tqdm":false,"n_epochs":40,"early_stop":0,"train_stop_threshold":null,"min_epochs":40,"consecutive_k":999999,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","checkpoint_min_delta":0.0}'
$env:QIB_PORT_OVERRIDES_JSON = '{"strategy":{"kwargs":{"n_drop":5}}}'
C:\Users\60585\miniconda3\envs\quantEnv\python.exe -u work_flow.py
```

Repeat the paired templates for seed 44 by replacing both the run-setting name and `"seed"`.

Automated hook:

```powershell
C:\Users\60585\miniconda3\envs\quantEnv\python.exe scripts/run_full_daily_sampler_paired_hook.py --python-exe C:\Users\60585\miniconda3\envs\quantEnv\python.exe
```

## Post-Run Readout

After each run, read the emitted `diagnostic_matrix.csv` and report:

1. Sampler mechanics: coverage gap, duplicate rate, upsample/downsample ratios.
2. Primary gap/ranking result: train-valid gap, RankIC, RankICIR.
3. HC-6 guard tuple.
4. Router/expert/temporal context, especially whether `router_oracle/oracle_gap` closes or expands under full coverage.

Append one ledger row for `sweep_start` when the first Phase A run is actually launched, one `result` row when its diagnostics are read, and one `verdict` row when the Phase A decision rule is settled.

## Anti-Rescue Clause

Do not rescue a failed Phase A by changing dropout, capacity, tau scale, loss, regime state, router mode, n_drop, or the evaluation window. Any such move must become a separate falsifier card with a new claim and its own anchor.

Do not promote `full_daily` from a single seed. A seed-42 pass only authorizes Phase B.
