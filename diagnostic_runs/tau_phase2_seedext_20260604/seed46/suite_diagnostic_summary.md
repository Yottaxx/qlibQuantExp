# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `tau_scale_05` (0.079754)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| full_135 | 0 | 0.07916117974028321 | 0.5734221091229501 | 0.17093511745411427 | 1.6709368427387605 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_05 | 0 | 0.07975449395152258 | 0.5789996521168352 | 0.14965377741208954 | 1.495401380149053 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
