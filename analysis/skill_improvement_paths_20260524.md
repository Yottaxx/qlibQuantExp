# Skill-Based Improvement Paths, 2026-05-24

This note applies the `qlibQuantExp-skills` routing discipline to the current
experiment set. It is not a new experiment result and does not promote any new
default. It turns the existing bottleneck report into testable next paths under
the current constraints: keep `full135`, do not change regime state, do not run a
new seed sweep, and do not bundle macV3-style changes.

## Skill Routing

- `qlibQuantExp-skills`: session entry and routing.
- `qx-research-ledger`: current frontier survey; facts only.
- `qx-transpose-from-another-field`: new mechanisms via relation mapping.
- `qx-forge-a-sharp-falsifier`: convert each path into a failure rule before
  any GPU spend.

Hard-core status: all proposed paths below are protective-belt moves. They do
not change the core commitments: soft routing, heterogeneous time/factor
experts, post-LayerNorm identity-start FiLM, and no look-ahead macro features.

## Established Facts

- Current coverage: 34 inventory rows, 29 valid RankIC rows, 17 aggregate
  setting rows.
- Best single run: `cap00_d64_h4_l2_ff128_do010` seed 43,
  RankIC `0.079639`, RankICIR `0.583296`.
- Best single-seed capacity candidate: `cap06_d96_h4_l2_ff384_do020` seed 42,
  RankIC `0.079374`, RankICIR `0.541603`. This remains single-seed evidence.
- Current working setting: `full_135` aggregate RankIC `0.078120` over two
  rows, RankICIR `0.550470`.
- Seed42 structural matrix:
  - `base`: RankIC `0.071236`.
  - `time_only`: RankIC `0.071970`, delta `+0.000734`.
  - `film_only`: RankIC `0.076786`, delta `+0.005550`.
  - `fixed_router_05`: RankIC `0.077071`, delta `+0.005835`.
  - `full_135`: RankIC `0.079135`, delta `+0.007899`.
  - `no_layer_summary`: RankIC `0.069079`, delta `-0.002158`.
- `film_only` explains about `70.3%` of the seed42 `full_135 - base` RankIC
  delta; `time_only` explains about `9.3%`.
- Learned router beats fixed `0.5` routing by RankIC `+0.002064` in seed42, but
  loses RankICIR by `-0.026478`.
- Top settings still show a large train-valid RankIC gap near `0.12`.
- Capacity scaling is non-monotonic; aggregate capacity-parameter correlations
  are weak in the current data (`d_model` r `+0.118`, `n_heads` r `-0.086`,
  `d_ff` r `+0.431`, all over only 11 capacity settings, many n=1).

## Diagnostic Signals That Matter

The current data does not support another blind capacity or loss sweep. The
stronger actionable signals are mechanism diagnostics:

- `train_score_std_last` vs RankIC: r `+0.574` over 23 rows.
- `valid_score_std_last` vs RankIC: r `+0.417` over 23 rows.
- `time_ratio_vs_pc1_spearman` vs RankIC: r `-0.569` over 22 rows.
- `time_ratio_vs_corr_mean_abs_spearman` vs RankIC: r `-0.558` over 22 rows.
- `layer_1_collapse_ratio_last` vs RankIC: r `-0.464` over 23 rows.
- `pooling_top10_vs_film_top10_overlap` vs RankIC: r `-0.494` over 21 rows.
- `tau_vs_rank_ic_spearman` vs RankIC: r `-0.494` over 21 rows.
- `time_local_mass_mean` vs RankIC: r `+0.213` over 23 rows, while
  `time_long_range_mass_mean` is r `-0.213`.

Interpretation boundary: these are associations, not causal proof. They are
useful because they specify where the next small intervention should be allowed
to fail.

## New Improvement Paths

### Path 1: Sampler Coverage As A Hidden Generalization Lever

Analogy: control systems. A controller trained on biased or repeatedly sampled
sensor states can look stable in training while failing under full-state
evaluation. The mapped relation is:

| Base relation | Target relation |
| --- | --- |
| sampled sensor states differ from deployment state coverage | fixed daily sampled training differs from full-day evaluation |
| controller overfits repeated states | model overfits duplicated or under-covered train days |
| closed-loop error appears as train/deployment gap | train-valid RankIC gap near `0.12` |

Prediction: if sampler mismatch is material, high duplicate rate or low daily
coverage should co-move with the train-valid gap and with weaker validation
RankIC after the new sampler diagnostics are recorded.

Falsifier: abandon this path if `sampler/coverage_gap` and
`sampler/duplicate_rate` do not explain any material share of gap variation, or
if an opt-in `full_daily` diagnostic run lowers gap without improving valid
RankIC/RankICIR.

Allowed next action: read the new `sampler/*` diagnostics on the next natural
`full135` run. Do not make `full_daily` the default until it passes the
falsifier.

### Path 2: Router Calibration Before Router Capacity

Analogy: distributed load balancing. A load balancer is not improved by adding
more routing features if the current balancer already misses obvious expert
winners. First measure the oracle gap.

| Base relation | Target relation |
| --- | --- |
| dispatcher routes work to servers | router routes samples to time/factor experts |
| oracle assignment bounds dispatcher quality | forced-time/forced-factor/default RankIC bounds router quality |
| load imbalance hurts throughput stability | collapse and low margin hurt RankICIR |

Prediction: if routing is a bottleneck, forced expert diagnostics should show
a meaningful `router_oracle_gap`, and the gap should be largest on days where
PC1/correlation alignment is abnormal.

Falsifier: do not add router capacity if `max(forced_time, forced_factor)` does
not beat default by at least `0.0015` RankIC on the diagnostic window, or if it
improves RankIC only by degrading RankICIR materially.

Allowed next action: use `router_override` only in diagnostic export. Candidate
future interventions, if the falsifier passes, are temperature/margin
calibration and light anti-collapse discipline, not a larger router.

### Path 3: Temporal Path As Local Evidence, Not Longer Memory

Analogy: signal processing. When a weak filter bank does not add signal,
expanding the bandwidth can amplify noise. First identify whether the useful
temporal mass is local or long-range.

| Base relation | Target relation |
| --- | --- |
| useful filter has passband concentration | useful time expert has local/long-range attention structure |
| wider filter can admit more noise | longer temporal machinery may dilute RankIC |
| passband contribution is measured before redesign | forced-time RankIC and temporal contribution are measured first |

Prediction: the temporal path is "weak but not refuted" only if forced-time
RankIC contribution, expert contribution norm, or local attention mass is
positive on stronger runs.

Falsifier: abandon temporal architecture work if forced-time diagnostics are no
better than default/factor on RankIC and the time expert contribution norm is
small or collinear with the factor expert. If local mass matters but long-range
mass is negative, do not increase context length as the first move.

Allowed next action: use the new `temporal/*` and `expert/*` diagnostics before
trying A1 readout variants. If intervention is justified, start with readout or
local pooling, not regime-state expansion.

### Path 4: FiLM-Pooling Coordination

Analogy: multi-sensor fusion. A strong modulator can improve the system while
the final readout attends to a different subset of channels. The question is
not whether FiLM works; the current matrix says it does. The question is
whether the readout is harvesting the channels FiLM makes useful.

| Base relation | Target relation |
| --- | --- |
| sensor reliability weights change by context | FiLM gamma/beta changes by regime |
| fusion layer selects a subset of sensors | pooling selects top factors |
| disagreement can be either diversity or waste | low FiLM/pooling overlap may help or hurt |

Prediction: if mismatch is waste, runs with stronger FiLM but poor
pooling/FiLM alignment should show router-oracle or score-dispersion loss.

Falsifier: do not regularize overlap unless the new diagnostics show that
FiLM-pooling mismatch predicts lost RankIC/RankICIR after controlling for
default RankIC. The existing negative correlation means forced alignment could
be harmful.

Allowed next action: diagnostic-only decomposition first. A future patch should
be a narrow pooling/readout prior, not another FiLM redesign.

### Path 5: Score Dispersion As Calibration, Not Loss Replacement

Analogy: measurement calibration. A sensor can rank correctly only if its
output dynamic range is neither collapsed nor dominated by noise. Current
correlations show score standard deviation tracks RankIC more than many
architecture knobs.

Prediction: better runs maintain a healthier train/valid score-dispersion
profile without increasing nonfinite/skipped gradients.

Falsifier: if score standardization or checkpoint selection improves dispersion
but not daily RankIC/RankICIR, this is a monitoring proxy only and should not
be treated as an objective.

Allowed next action: keep the loss unchanged. Use score dispersion in
diagnostic review and checkpoint discipline; consider rank-preserving daily
post-processing only as a separate, pre-registered test.

## Ranked Next Moves

1. Use the deterministic diagnostic patch on the next ordinary `full135` run and
   inspect `sampler/*`, `router_oracle/*`, `expert/*`, and `temporal/*`. This is
   not a seed sweep and should not change the default architecture.
2. If sampler mismatch is large, test `full_daily` as an opt-in falsifier. Keep
   `sampled_daily` as default until the falsifier passes.
3. If router oracle gap is positive, tune router calibration before capacity:
   temperature, margin, or anti-collapse constraints.
4. If temporal local evidence is positive, test a narrow readout/local-pooling
   path. Do not expand regime state or context length first.
5. If diagnostics show FiLM-pooling mismatch is waste rather than diversity,
   test a small pooling prior. Otherwise leave FiLM alone.

## Explicitly Deferred

- Do not promote `cap06` to default from a single seed.
- Do not start a new capacity sweep before deterministic diagnostics are
  recorded.
- Do not change loss as the next move.
- Do not change regime state.
- Do not revive macV3 bundle changes.
- Do not claim the temporal path is dead until forced-time and expert
  contribution diagnostics have been observed.

## Ledger Hygiene Note

`shared/ledger.md` currently contains duplicate 2026-05-23 id `8`/`9` rows after
the `Future entries` marker. Future ledger automation should append corrections
rather than edit prior rows, but the duplicate layout should be treated as a
research-record hygiene issue before using the ledger as a trial counter.
