# Diagnostic Suite Summary

- Best setting by daily RankIC mean: `tau_scale_08` (0.080815)
- Table: `suite_diagnostic_summary.csv`

| setting | returncode | RankIC | RankICIR | AnnRet(cost) | IR(cost) | key bottleneck hints |
|---|---:|---:|---:|---:|---:|---|
| full_135 | 0 | 0.07793646758297612 | 0.5762482386935072 | 0.14576413666279192 | 1.3763591601474203 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_05 | 0 | 0.07881082064968697 | 0.5871777320264925 | 0.17858117376209762 | 1.7738018994155857 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_08 | 0 | 0.08081530723955313 | 0.5971086151376505 | 0.1842189846652985 | 1.799951026446422 | tau under-adaptive, factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_10 | 0 | 0.07860205969709333 | 0.5782000017286074 | 0.13157841137023582 | 1.3487447370665144 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_15 | 0 | 0.07801245149716567 | 0.5670883190648928 | 0.16335151797884684 | 1.607320144115314 | factor attention uniform, regime-specific spread, AMP grad overflow |
| tau_scale_20 | 0 | 0.07715750596811109 | 0.5788345834024268 | 0.12881245973233404 | 1.3646587756763329 | factor attention uniform, regime-specific spread, AMP grad overflow |
