# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `full_135` (0.073464)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| full_135 | 0 | 0.07346368406189176 | 0.5360507856832254 | 0.11325407489112004 | 1.1040441034307875 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_05 | 0 | 0.07234714072122798 | 0.5243172941797324 | 0.1774198631627256 | 1.7208186410225352 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_08 | 0 | 0.07169591995714882 | 0.508247744578376 | 0.15418583102955358 | 1.4701566094833816 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_10 | 0 | 0.07291014850756469 | 0.5285056411517863 | 0.17781490648308051 | 1.649156241156117 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_15 | 0 | 0.07317801648304184 | 0.5334934086858079 | 0.1727130031911749 | 1.7047929953138004 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_20 | 0 | 0.06936605359680412 | 0.4835179607083216 | 0.13774217599783176 | 1.2999179206689393 | factor attention uniform, regime-specific spread, AMP grad overflow |
