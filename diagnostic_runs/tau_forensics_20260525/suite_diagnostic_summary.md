# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `tau_scale_50` (0.082272)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| tau_scale_01 | 0 | 0.08201935816500829 | 0.5839040209966605 | 0.18395758043307942 | 1.733792687596739 | tau under-adaptive, factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
| tau_scale_10 | 0 | 0.08051044556189502 | 0.5777895220844497 | 0.1862256626403307 | 1.8131599563227214 | factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
| tau_scale_20 | 0 | 0.08022907581620788 | 0.5752516093439577 | 0.18903181470323618 | 1.8035587307060217 | factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
| tau_scale_50 | 0 | 0.08227160468174499 | 0.5943023503562194 | 0.1722168607951209 | 1.6816888475914966 | factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
