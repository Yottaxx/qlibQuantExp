# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `full_135` (0.081026)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| full_135 | 0 | 0.08102626317338336 | 0.6060585168438859 | 0.17594968386855062 | 1.7830060389461466 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_05 | 0 | 0.07809661467879561 | 0.5790065642205476 | 0.18205911376893058 | 1.8128059823767688 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
