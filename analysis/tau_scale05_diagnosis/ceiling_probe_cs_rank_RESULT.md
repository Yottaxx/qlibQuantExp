# CS-rank Ceiling Probe — does the cross-sectional coordinate lift the classic-learner ceiling?

- SMOKE=False  F=158  model_ref=0.0786  raw_window_ceiling(L-1 B)=0.0592
- train 2008-01-01..2020-03-31 | int-eval 2019-01-01..2020-03-31 | test 2020-07-01..2022-12-31
- **R=0.05345  CSR=0.05042  R+CSR=0.06200**
- **lift(R+CSR vs R snapshot) = +0.00854 | lift(R+CSR vs raw-window 0.0592) = +0.00280**
- **VERDICT: CS_INPUT_HAS_HEADROOM__L4_reopens**

## R  ceiling=0.05345
- ridge 0.04445 (alpha=1.0, int-eval 0.04893)
- lgbm 0.05345 ± 0.00077 (n=1000, seeds ['0.05331', '0.05258', '0.05446'])

## CSR  ceiling=0.05042
- ridge 0.03806 (alpha=10000.0, int-eval 0.05309)
- lgbm 0.05042 ± 0.00079 (n=600, seeds ['0.05052', '0.04941', '0.05134'])

## R+CSR  ceiling=0.06200
- ridge 0.04632 (alpha=10000.0, int-eval 0.05777)
- lgbm 0.06200 ± 0.00042 (n=300, seeds ['0.06159', '0.06183', '0.06258'])
