# Non-Tau Model Bottlenecks - 2026-05-31

Scope: exclude the `time_tau_mlp_out_scale` intervention path. I still use the default-scale diagnostic row (`0.01ref`) where it provides non-tau diagnostics such as sampler/router/expert/pooling, because that row is the current full135-style control with richer instrumentation.

## Executive Ranking

| rank | bottleneck | confidence | evidence | implication |
|---:|---|---|---|---|
| 1 | Train/eval sampling and generalization gap | high | Top non-tau settings still show RankIC gap around 0.11-0.14; sampled_daily coverage gap is about 0.339 and duplicate rate about 0.366. | The model is learning from a materially distorted per-epoch sample distribution. |
| 2 | Final factor pooling/readout is nearly uniform | high | Pool entropy is almost exactly 1.0 and top10 mass is about 0.063, which is uniform over 158 factors. | Final score head is mostly seeing mean-pooled factor state, not a selective factor readout. |
| 3 | Time expert contribution dominates factor expert | high | Default diagnostic: contribution norm ratio is about 11.1 in layer 0 and 7.2 in layer 1; time contribution wins almost every day. | HC-3 is structurally present, but the realized mixture is time-heavy. |
| 4 | Router calibration edge is real but unstable | medium-high | full135 beats fixed_0.5 by +0.0021 RankIC in seed42, but loses -0.0265 RankICIR; forced single experts are worse than default. | Do not add router capacity blindly; calibrate or regularize the existing router. |
| 5 | FiLM helps, but FiLM-readout coordination is weak | medium | film_only explains about 70% of full135-base RankIC delta, but factor gate entropy remains high and pooling/FiLM overlap is low. | The regime factor path creates useful modulation, but the final readout may not harvest it. |
| 6 | Capacity scaling is not monotonic | medium | cap06 is strong at one seed, but 128-dim variants worsen gap/decay and the multi-seed small-cap settings are noisy. | More capacity is not the next clean bottleneck fix. |
| 7 | Portfolio/cost layer can dominate model alpha | medium | n_drop=30 diagnostic had positive no-cost return but negative cost IR due turnover/cost drag. | Treat execution/turnover as a separate downstream bottleneck, not an architecture result. |

## Evidence Base

### Structural Matrix

Seed42 40e structural matrix:

| setting | RankIC | delta vs base | RankICIR | IR cost | gap | decay |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.0712 | 0.0000 | 0.539 | 1.369 | 0.106 | 0.0036 |
| time_only | 0.0720 | +0.0007 | 0.548 | 1.575 | 0.100 | 0.0060 |
| film_only | 0.0768 | +0.0056 | 0.584 | 1.575 | 0.143 | 0.0192 |
| fixed_router_05 | 0.0771 | +0.0058 | 0.576 | 1.535 | 0.111 | 0.0073 |
| full_135 | 0.0791 | +0.0079 | 0.550 | 1.564 | 0.121 | 0.0091 |
| no_layer_summary | 0.0691 | -0.0022 | 0.515 | 1.843 | 0.113 | 0.0050 |

Interpretation:

- FiLM/regime factor gating is the largest non-tau architectural contributor.
- `time_only` adds almost nothing by itself.
- `layer_summary` is important; removing it drops RankIC badly.
- Learned routing improves mean RankIC but not stability versus fixed 0.5.

### Sampler / Gap

The sampled daily trainer uses `FixedDailyBatchSampler`, which downsamples days larger than batch size and upsamples smaller days with replacement (`module/dataloader/sampler.py:70`, `module/dataloader/sampler.py:74`). The diagnostics report `coverage_gap_vs_full` and `duplicate_rate` directly (`module/dataloader/sampler.py:103`, `module/dataloader/sampler.py:105`).

Default full135 diagnostic:

| metric | value |
|---|---:|
| coverage_gap_vs_full | 0.3387 |
| duplicate_rate | 0.3664 |
| unique samples / source samples | about 0.661 |
| upsample day ratio | 0.9658 |

The full-daily falsifier fixed mechanics but did not improve performance:

| seed | sampled RankIC | full RankIC | full-sampled delta | sampled gap | full gap |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.0732 | 0.0646 | -0.0087 | 0.1164 | 0.1330 |

Conclusion: the bottleneck is not simply "use every sample once". The real issue is that training distribution, batch composition, day curriculum, and cross-sectional gradient noise are entangled. A better path is sampled_daily de-dup / weighting / accumulation diagnostics, not immediate full_daily promotion.

### Pooling / Readout

The final stock score uses the last time step, then `AdaptivePooling`, then a linear head (`module/quant_moe_model.py:402`, `module/quant_moe_model.py:405`, `module/quant_moe_model.py:408`). `AdaptivePooling` blends attention pooling and mean pooling with `alpha` (`module/architecture/attention_pooling.py:100`, `module/architecture/attention_pooling.py:129`).

The attention component itself is a single learned query into factor tokens (`module/architecture/attention_pooling.py:33`, `module/architecture/attention_pooling.py:37`, `module/architecture/attention_pooling.py:65`).

Observed non-tau metrics:

| setting | pool entropy | pool top10 mass | factor attention entropy |
|---|---:|---:|---:|
| full_135 seed42 | 1.000000 | about 0.0633 | 0.999 |
| cap06 seed42 | 0.999996 | about 0.0640 | 1.000 |
| sampled_daily paired seed42 | 0.999961 | about 0.064 | 1.000 |

For 158 factors, uniform top10 mass is about `10/158 = 0.0633`. The final pooling is therefore nearly uniform. This means a large part of the architecture is producing rich factor states that are collapsed by a near-mean readout.

### Expert Mixture

The MoE block computes both experts, then fuses them by router weights (`module/architecture/moe_block.py:173`, `module/architecture/moe_block.py:182`, `module/architecture/moe_block.py:197`). Diagnostics measure contribution norm ratio and time winner ratio (`module/architecture/moe_block.py:193`, `module/architecture/moe_block.py:194`, `module/architecture/moe_block.py:213`, `module/architecture/moe_block.py:215`).

Default full135 richer diagnostic:

| metric | layer 0 | layer 1 |
|---|---:|---:|
| time/factor contribution norm ratio | 11.1 | 7.2 |
| time winner ratio | 1.000 | 0.997 |
| expert cosine | 0.025 | 0.003 |

The experts are not collinear, so this is not a duplicate-expert problem. It is a scale/contribution problem: the factor expert contributes a small residual that helps the blend, but it is too weak to be a balanced axis-heterogeneous expert in practice.

### Router

The router input combines regime embedding and optional layer summary (`module/architecture/moe_block.py:74`, `module/architecture/moe_block.py:86`), then softmaxes two logits (`module/architecture/moe_block.py:95`, `module/architecture/moe_block.py:103`). The old matrix showed:

- full_135 beats fixed_router_05 by +0.0021 RankIC;
- full_135 loses -0.0265 RankICIR to fixed_router_05;
- forced time, forced factor, and uniform are all worse than default in the richer diagnostic;
- `router_oracle_gap` should not be treated as recoverable headroom because the oracle is hindsight-biased.

Conclusion: the router is useful, but the current learned gate is not yet a robust stability advantage. Work here should be calibration and contribution balancing, not more router features.

### Capacity

Capacity evidence remains weak:

| setting | seeds | RankIC mean | RankIC std | IR cost | gap | decay |
|---|---|---:|---:|---:|---:|---:|
| cap00 d64/ff128/drop0.10 | 42,43,44 | 0.0763 | 0.0030 | 1.796 | 0.122 | 0.0071 |
| cap01 d64/ff256/drop0.15 | 42,43,44 | 0.0757 | 0.0018 | 1.905 | 0.112 | 0.0033 |
| cap02 d80/ff320/drop0.15 | 42,43,44 | 0.0771 | 0.0016 | 2.029 | 0.121 | 0.0057 |
| cap06 d96/ff384/drop0.20 | 42 only | 0.0794 | 0.0000 | 2.086 | 0.120 | 0.0050 |
| cap09 d128/h8/ff384/drop0.20 | 42 only | 0.0757 | 0.0000 | 1.868 | 0.143 | 0.0137 |
| cap11 d128/h4/ff512/drop0.20 | 42 only | 0.0776 | 0.0000 | 1.375 | 0.134 | 0.0068 |

cap06 is promising but single-seed. The 128-dim direction looks worse on gap/decay. Capacity is not a clean bottleneck until the current full135/tau candidate is settled.

## What Not To Chase Next

- Do not chase `router_oracle_gap` directly; it is not a causal ceiling.
- Do not promote full_daily sampler from current evidence.
- Do not run a broad capacity sweep before sampler/readout/expert diagnostics are tightened.
- Do not prune the factor expert; factor-only is weak, but the default blend beats forced time.
- Do not bundle pooling + sampler + router changes; each is a separate protective-belt parameter under HC-6.

## Recommended Non-Tau Next Tests

1. **Pooling/readout falsifier:** compare current `AdaptivePooling` against one narrow alternative that can actually become non-uniform, such as temperature-scaled pooling logits or multi-query pooling. Keep everything else fixed.
2. **Expert contribution balance diagnostic:** add a lightweight penalty or normalization candidate only if it preserves default soft mixture and raises factor contribution without lowering RankICIR.
3. **Sampler de-dup variant:** keep daily stochastic sampling, but reduce replacement duplicates or weight duplicated samples. Do not jump straight to full_daily.
4. **Router calibration:** test temperature/margin/z-loss schedule only after contribution diagnostics show it changes stability rather than just gate mass.
5. **FiLM-readout coordination:** if pooling remains uniform, test whether pooling should see FiLM importance as a prior. This must be separate from a FiLM redesign.

Current non-tau diagnosis: the model has useful alpha machinery, but it is bottlenecked by a blunt training sampler and a blunt final readout. The time/factor MoE is structurally heterogeneous, yet the realized path is strongly time-dominated. That is the main architecture-level issue after tau.
