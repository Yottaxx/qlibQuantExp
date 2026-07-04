# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `full_135` (0.076709)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| full_135 | 0 | 0.07670936453003971 | 0.5688606172188271 | 0.19756049203399448 | 1.9748482963295166 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_05 | 0 | 0.07463451572811979 | 0.5544340829665377 | 0.1597804944445544 | 1.570895683321533 | factor attention uniform, regime-specific spread, AMP grad overflow |
