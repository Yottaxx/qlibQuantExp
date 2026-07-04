# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `tau_scale_05` (0.081309)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| tau_scale_02 | 0 | 0.08029259257610506 | 0.5673359656018875 | 0.18734867581324066 | 1.727803533188905 | tau under-adaptive, factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
| tau_scale_05 | 0 | 0.08130924080698723 | 0.5931556219729567 | 0.18337550411435946 | 1.8589927802422697 | factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
| tau_scale_08 | 0 | 0.08032998105465738 | 0.5823020224737012 | 0.1506163663810217 | 1.4748824475167175 | factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
| tau_scale_15 | 0 | 0.07969267205334712 | 0.5782042366376546 | 0.18675154380026773 | 1.8364103716945184 | factor attention uniform, regime-specific spread, post-peak decay, AMP grad overflow |
