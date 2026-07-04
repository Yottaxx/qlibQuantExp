# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `tau_scale_05` (0.081065)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| full_135 | 0 | 0.08021350330711123 | 0.5953152891575569 | 0.17789941927368919 | 1.8242361843702386 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_05 | 0 | 0.08106502378295403 | 0.6091870389205332 | 0.22795120452094184 | 2.2462499966987135 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_08 | 0 | 0.07981417208078188 | 0.5953495127357474 | 0.17654367128864185 | 1.7707827858144969 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_10 | 0 | 0.07924910322515662 | 0.58148671518311 | 0.17954641500441978 | 1.7798383425430693 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_15 | 0 | 0.0775105969399211 | 0.5614366658089623 | 0.19330674156176506 | 1.9631830258180978 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_20 | 0 | 0.08025583553666085 | 0.6084343770483259 | 0.1898754957348826 | 1.9151896473505847 | factor attention uniform, regime-specific spread, AMP grad overflow |
