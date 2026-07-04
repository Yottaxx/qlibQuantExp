# CS-rank Ceiling Probe — WINDOW arm (model-matched)

- SMOKE=False F=158 WIN=8 model_ref=0.0786 raw_window(L-1 B)=0.0592
- **RW=0.05922  RW+CSR=0.06248  windowed_lift=+0.00326**
- **VERDICT: MODAL__marginal**

## RW  ceiling=0.05922
- reused from prior run log (LGBM 0.05817/0.05913/0.06036); reproduces L-1 variant-B 0.0592 exactly

## RW+CSR  ceiling=0.06248
- ridge 0.05069 (alpha=10000.0)
- lgbm 0.06248 ± 0.00052 (n=600, seeds ['0.06186', '0.06246', '0.06314'])
