# IC-monitor panel — diverse_ensemble

## Headline  rank_ic 0.08120 | rank_icir 0.595 | ic 0.06667 | icir 0.493  (n=611)

## Decile (Q1..Q10 mean fwd-return)
- [-0.00631, -0.00294, -0.00159, -0.00045, -0.00057, 0.00108, 0.00193, 0.0019, 0.0031, 0.00479]
- monotonicity (Spearman decile->return) **0.2153** | long-short Q10-Q1 **0.01109**

## Top-30 (aligned to trading rule)
- precision@30 **0.1166** | recall@30 0.1166 | ndcg@30 **0.5582**

## Dispersion anchor (is low IC a shrinkage artifact?) — on the per-day unit-variance target
- std(pred) 0.2728 vs MSE-optimal std(pred)*=IC=0.0667 (target std 1.0) | ratio actual/optimal 4.09
- **over-dispersed vs MSE-optimum (cardinal scores over-confident; RankIC-IRRELEVANT) => IC is a real signal ceiling, NOT a shrinkage artifact**

## Stability  worst quarter 2021Q3 rank_ic 0.0510 | 1.00 of 10 quarters positive