# Current Experiment Records - 2026-05-31

Sources scanned:

- `.agents/skills/qlibQuantExp-skills/SKILL.md`
- `.agents/skills/qlibQuantExp-skills/MANIFEST.yaml`
- `.agents/skills/qlibQuantExp-skills/shared/ledger.md`
- `.agents/skills/qlibQuantExp-skills/shared/hardcore.md`
- `diagnostic_runs/**/manifest.json`
- `diagnostic_runs/**/suite_diagnostic_summary.*`
- `analysis/tau_router_expert_temporal_20260529.csv`
- `analysis/full_daily_sampler_paired_promotion_20260524.md`
- `experiments_ledger.jsonl`

## Session Survey

The formal current anchor in the research ledger is still `sha=3675090`, macV2 baseline on CSI300 t+5, RankIC 0.0786. The active working comparison has moved to `full_135`-family diagnostic runs, but no newer `kind=anchor current=true` row supersedes the original anchor. The live open sweep is `tau_phase2_25e_20260530`: intended 6 scales x 3 seeds x 25 epochs, but disk currently contains only `seed42/full_135`, `seed42/tau_scale_05`, and running `seed42/tau_scale_08`. The last major decisions were: abandon 15e Phase 2 as too short, relaunch 25e, keep tau scale candidates focused around 0.5 and 1.0, and stop treating `router_oracle_gap` as recoverable alpha.

## Source Health

| Source | Status | Notes |
|---|---|---|
| `shared/ledger.md` | usable but messy | Contains duplicate/reused id ranges: the full-daily sampler block reuses ids 24-35 after tau rows 36-42. Interpret by track/date, not global id alone. |
| `experiments_ledger.jsonl` | stale | Last update is 2026-05-06, before the current tau/full-daily tracks. Do not use as current truth. |
| `memory/project_state.md` | missing | `quant-recent` current-best snapshot cannot be used. |
| `tau_phase2_15e_20260530/seed42/suite_diagnostic_summary.md` | stale/invalid | Shows all returncode `None`, while manifests later show four completed partial runs. Ledger already says 15e was discarded. |
| `tau_phase2_25e_20260530` | active | No suite summary yet. Use per-run manifests and logs. |

## Experiment Inventory

| suite | manifests | done | running/pending | failed | diagnostic matrix |
|---|---:|---:|---:|---:|---:|
| `four_day_plan_20260501_125906` | 24 | 23 | 0 | 1 | 23 |
| `seed42_matrix_gradfix_20260430_2105` | 6 | 6 | 0 | 0 | 6 |
| `tau_forensics_20260525` | 4 | 4 | 0 | 0 | 4 |
| `tau_forensics_refine_20260527` | 4 | 4 | 0 | 0 | 4 |
| `tau_phase2_15e_20260530` | 7 | 4 | 3 | 0 | 4 |
| `tau_phase2_25e_20260530` | 3 | 2 | 1 | 0 | 2 |
| `full_daily_sampler_paired_20260524` | 3 | 2 | 1 | 0 | 2 |
| `full_daily_sampler_paired_bestvalid_20260525` | 1 | 0 | 1 | 0 | 0 |

Old `returncode=None` manifests outside the active 25e run mostly correspond to aborted, print-only, or superseded sweeps.

## Main Track State

### 1. Baseline and Bottlenecks

The 2026-05-23 bottleneck report surveyed 31 MLflow runs / 23 diagnostic-backed rows and supported keeping `full_135` as the working setting. The important bottlenecks were:

- train-valid RankIC gap around 0.10-0.14;
- time-only contribution was close to base;
- learned routing did not yet robustly dominate fixed 0.5 routing;
- regime-alpha alignment needed measurement before adding more regime structure.

This remains the right frame: the tau fix helps one suppression mechanism, but it has not fixed the sampler/generalization gap.

### 2. Tau Clamp Forensics

The tau track found that `tau_mlp_out_scale=0.01` suppressed the regime-adaptive time embedding by about 100x. After plumbing `time_tau_mlp_out_scale`, the 40e single-seed sweep confirmed the mechanism.

Selected 40e tau response:

| scale | RankIC | IR cost | MaxDD cost | AnnRet cost | post-peak decay | gap | tau util | tau p10-p90 | pool entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0.01 ref | 0.0782 | -0.314 | -0.1529 | -0.037 | 0.0000 | 0.0816 | 0.00023 | 4.96-4.97 | 1.000000 |
| 0.1 | 0.0820 | 1.734 | -0.0602 | 0.184 | 0.0093 | 0.1186 | 0.00539 | 4.94-5.21 | 0.999987 |
| 0.2 | 0.0803 | 1.728 | -0.0722 | 0.187 | 0.0097 | 0.1210 | 0.00539 | 4.92-5.19 | 0.999999 |
| 0.5 | 0.0813 | 1.859 | -0.0788 | 0.183 | 0.0097 | 0.1201 | 0.01769 | 4.88-5.76 | 1.000000 |
| 0.8 | 0.0803 | 1.475 | -0.0678 | 0.151 | 0.0087 | 0.1195 | 0.02231 | 4.81-5.91 | 1.000000 |
| 1.0 | 0.0805 | 1.813 | -0.0617 | 0.186 | 0.0091 | 0.1202 | 0.03092 | 4.66-6.19 | 1.000000 |
| 1.5 | 0.0797 | 1.836 | -0.0680 | 0.187 | 0.0089 | 0.1206 | 0.03020 | 4.59-6.09 | 0.999999 |
| 2.0 | 0.0802 | 1.804 | -0.0949 | 0.189 | 0.0076 | 0.1156 | 0.05955 | 3.94-6.89 | 1.000000 |
| 5.0 | 0.0823 | 1.682 | -0.0776 | 0.172 | 0.0124 | 0.1217 | 0.11053 | 3.44-8.91 | 0.999948 |

Interpretation:

- The clamp hypothesis is corroborated: tau utilization rises from effectively frozen to adaptive as scale increases.
- The best single-seed IR was scale 0.5; the most theoretically clean router-regime alignment was scale 1.0.
- Scale 0.8 had a portfolio anomaly: normal RankIC/tau behavior but weak IR and AnnRet. This is a reason for multi-seed, not a reason to reject the scale from one seed.
- Scale 2.0 and 5.0 are not promotion candidates because they start trading off drawdown/post-peak behavior despite stronger tau spread.
- Pooling attention remains collapsed/uniform at all scales. Tau does not solve pooling.

### 3. Router and Expert Interpretation

The 2026-05-29 adversarial verification changed the interpretation:

- `router_oracle_gap` is hindsight max-of-experts selection bias, not usable headroom.
- The factor expert should not be pruned just because factor-only RankIC is weak; its value is decorrelation/residual contribution.
- Gross routing mass is fairly tau-invariant. The tau effect is more about correlation structure than about shifting average gate mass.
- Tau acts on additive mean-normalized positional embedding, not the attention softmax. Claims that tau directly changes attention mass should be retired.

### 4. Full-Daily Sampler

The full-daily sampler track fixed the mechanics problem, but the available paired result is bad for promotion:

| seed | sampled RankIC | full RankIC | delta full-sampled | sampled gap | full gap | full coverage gap | full duplicate rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.073216 | 0.064551 | -0.008665 | 0.116440 | 0.133019 | 0.000000 | 0.000000 |

The best-valid rerun was paused before completion. Current status: do not promote full-daily sampler; the sampler/data-side gap remains open, but this exact intervention is not supported by the available evidence.

## Active Run: Tau Phase 2, 25 Epochs

Command lineage from the active runner:

`scripts/run_diagnostic_experiments.py --settings full_135,tau_scale_05,tau_scale_08,tau_scale_10,tau_scale_15,tau_scale_20 --out-dir diagnostic_runs/tau_phase2_25e_20260530/seed42 ... --continue-on-error`

Current per-run state:

| path | setting | seed | returncode | epoch | best valid RankIC | RankIC | IR cost | MaxDD cost | AnnRet cost | post-peak decay | gap | tau util | tau p10-p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `seed42/full_135` | full_135 | 42 | 0 | 25 | 0.0802 | 0.0802 | 1.824 | -0.0587 | 0.178 | 0.0032 | 0.0920 | 0.00008 | 4.97-4.97 |
| `seed42/tau_scale_05` | tau_scale_05 | 42 | 0 | 25 | 0.0811 | 0.0811 | 2.246 | -0.0600 | 0.228 | 0.0048 | 0.0928 | 0.01150 | 4.93-5.50 |
| `seed42/tau_scale_08` | tau_scale_08 | 42 | running | 15 | 0.0791 | pending | pending | pending | pending | pending | pending | pending | pending |

Process status at inspection:

- runner process: `python scripts/run_diagnostic_experiments.py`, PID 31216, started 2026-05-30 23:13:30;
- child training process: `python -u work_flow.py`, PID 41524, started 2026-05-31 19:45:35;
- active stdout: `diagnostic_runs/tau_phase2_25e_20260530/seed42/tau_scale_08/stdout.log`, last written 2026-05-31 22:49:23;
- last parsed validation epoch: 15/25.

Provisional seed42 read:

- `tau_scale_05` is better than `full_135` on RankIC by about +0.00085 and IR by +0.422, with near-flat MaxDD degradation of about -0.0013 and modestly higher post-peak decay.
- This is promising but not promotion evidence under HC-6 because it is one seed only.
- `tau_scale_08` should not be judged until the run writes its diagnostic matrix.

## Current Conclusions

1. The most credible active improvement track is still tau unclamping, not router-oracle calibration or full-daily sampling.
2. Scale 0.5 and scale 1.0 remain the right promotion candidates. Scale 0.8 is useful as an anomaly-resolution arm.
3. The generalization/sampler gap is still unresolved; tau changed regime adaptivity and portfolio behavior but did not erase the train-valid gap.
4. The current Phase 2 is behind the intended 18-run plan: only seed42 has started, and it is midway through the third arm.
5. No production or paper claim is justified yet. HC-6 still requires multi-seed mean/std on the four-tuple.

## Next Actions

1. Let `seed42/tau_scale_08` finish, then regenerate a 25e suite summary for seed42.
2. Continue seed42 `tau_scale_10`, `tau_scale_15`, `tau_scale_20`; then run seeds 43 and 44.
3. After each completed seed, aggregate the four-tuple by scale: RankIC, IR cost, MaxDD cost, post-peak decay.
4. Append ledger rows only when the 25e sweep reaches a clear factual milestone: partial sweep done, full sweep done, or owner decision to pause/abort.
5. Before further ledger appends, fix discipline by adding a correction row noting the duplicate id block, then continue with a fresh monotonic id range.
