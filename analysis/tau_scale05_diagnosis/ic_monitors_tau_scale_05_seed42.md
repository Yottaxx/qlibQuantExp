# IC-monitor panel — tau_scale_05_seed42

## Headline  rank_ic 0.08107 | rank_icir 0.610 | ic 0.06701 | icir 0.486  (n=611)

## Decile (Q1..Q10 mean fwd-return)
- [-0.00603, -0.00272, -0.00173, -0.00103, 0.00016, 0.00087, 0.00165, 0.00194, 0.00273, 0.00508]
- monotonicity (Spearman decile->return) **0.2201** | long-short Q10-Q1 **0.01111**

## Top-30 (aligned to trading rule)
- precision@30 **0.1140** | recall@30 0.1140 | ndcg@30 **0.5590**

## Dispersion anchor (is low IC a shrinkage artifact?)
- std(pred) 0.2142 | std(label) 0.0458 | ratio 4.674
- MSE-implied optimal std(pred)* = IC*std(y) = 0.0031 | realized IC 0.06701
- **over-dispersed**

## Stability  worst quarter 2021Q2 rank_ic 0.0421 | 1.00 of 10 quarters positive