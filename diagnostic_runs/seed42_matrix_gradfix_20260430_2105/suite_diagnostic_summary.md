# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `full_135` (0.077104)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| full_135 | 0 | 0.07710432823767467 | 0.5513286017123423 | 0.21018343195055295 | 1.9104635802046284 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| base | 0 | 0.06788440268535363 | 0.4852508559407806 | 0.1125068731819935 | 1.017740200315127 | factor attention uniform, regime-specific spread, AMP grad overflow |
| time_only | 0 | 0.07175480564496155 | 0.5260382010456862 | 0.15973566927086436 | 1.4245924606363245 | tau under-adaptive, AMP grad overflow |
| film_only | 0 | 0.07192317763806869 | 0.5335026155951409 | 0.1610171471417081 | 1.4451240101619043 | factor attention uniform, regime-specific spread, AMP grad overflow |
| no_layer_summary | 0 | 0.06317243664669321 | 0.4746565924091005 | 0.1236344432636533 | 1.1108231908144603 | tau under-adaptive, factor attention uniform, post-peak decay, AMP grad overflow |
| fixed_router_05 | 0 | 0.07113857710740026 | 0.52342630922699 | 0.15498507408021225 | 1.4930583758400318 | tau under-adaptive, factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
