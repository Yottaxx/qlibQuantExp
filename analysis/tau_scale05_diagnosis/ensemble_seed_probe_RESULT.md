# Seed-ensemble IC probe (zero-GPU, MSE-preserving)

## DIVERSE cross-setting ensemble
- pool: 19 runs across ['base', 'full_135', 'readout_full_g012_d1pma', 'tau_scale_05', 'tau_scale_10']
- best single run: readout_full_g012_d1pma|42 rank_ic 0.08163
- **all-runs rank-mean 0.08120 (lift vs best single -0.00043, ICIR 0.5945)**
- **best-per-setting rank-mean 0.08301 (lift vs best single +0.00138)**
- monitors: decile-monotonicity 0.2153 / precision@30 0.1166

## tau_scale_05  (n=6, seeds [42, 43, 44, 45, 46, 47])
- per-seed rank_ic: [0.08107, 0.07881, 0.07235, 0.0781, 0.07975, 0.07463]
- **mean-single 0.07745 | best-single 0.08107**
- **ensemble score-mean 0.08069 | rank-mean 0.08048**
- **lift(rank-mean vs MEAN-single) = +0.00303 | lift vs BEST-single = -0.00058**
- ensemble rank_icir 0.5922 | score-mean ic 0.06971
- ensemble-size scaling (rank-mean, k=1..N): [0.08107, 0.08164, 0.07979, 0.0802, 0.08089, 0.08048]
- monitors: ensemble decile-monotonicity 0.2210 / precision@30 0.1173 (vs best-seed 0.2201 / 0.1140)

## readout_full_g012_d1pma  (n=3, seeds [42, 43, 44])
- per-seed rank_ic: [0.08163, 0.075, 0.07857]
- **mean-single 0.07840 | best-single 0.08163**
- **ensemble score-mean 0.08084 | rank-mean 0.08090**
- **lift(rank-mean vs MEAN-single) = +0.00250 | lift vs BEST-single = -0.00073**
- ensemble rank_icir 0.5831 | score-mean ic 0.06917
- ensemble-size scaling (rank-mean, k=1..N): [0.08163, 0.08036, 0.0809]
- monitors: ensemble decile-monotonicity 0.2251 / precision@30 0.1179 (vs best-seed 0.2251 / 0.1126)
