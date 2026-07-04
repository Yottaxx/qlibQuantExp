# Full-Daily Sampler Paired Promotion Judgment

- Created: 2026-05-25 15:34:16
- Verdict: `incomplete_or_failed_runs`
- Mechanics ok: `False`
- HC-6 pass count: `0/4`

## Mean Deltas

| Metric | Mean delta | Rule |
|---|---:|---|
| gap improvement (`sampled - full`) | -0.016579 | >= 0.015 |
| RankIC (`full - sampled`) | -0.008665 | >= 0 |
| RankICIR (`full - sampled`) | -0.072344 | >= -0.010 |
| IR with cost (`full - sampled`) | -0.170329 | non-degrading for HC-6 |
| MaxDD with cost (`full - sampled`) | -0.026731 | non-degrading for HC-6 |
| post-peak decay (`full - sampled`) | 0.004123 | <= 0 for HC-6 |

## Per-Seed Pairs

| Seed | sampled RankIC | full RankIC | RankIC delta | sampled gap | full gap | gap improvement | full coverage gap | full duplicate rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.073216 | 0.064551 | -0.008665 | 0.116440 | 0.133019 | -0.016579 | 0.000000 | 0.000000 |
| 43 | NA | NA | NA | NA | NA | NA | NA | NA |
| 44 | NA | NA | NA | NA | NA | NA | NA | NA |

## Decision Rule

- Promote only if mean gap improvement >= 0.015.
- Mean RankIC must be non-degrading.
- Mean RankICIR must not fall by more than 0.010.
- At least 3 of 4 HC-6 tuple elements must be non-degrading.
- No seed may lose more than 0.004 RankIC.
- Full-daily mechanics must show coverage gap <= 0.005 and duplicate rate <= 0.005.
