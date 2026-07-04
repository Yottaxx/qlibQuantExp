# CS-rank Ceiling Probe — WINDOWED-TRAJECTORY arm (model-matched)

- SMOKE=False F=158 WIN=8 cols(RW+CSR_TS)=2528 model_ref=0.0786 raw_window(L-1 B)=0.0592
- **RW=0.05922  RW+CSR_TS=0.06159  windowed_ts_lift=+0.00237**
- day-T RW+CSR (sibling) = 0.06248  | lift_TS_over_dayT = -0.00089
- **VERDICT: TS_CS_REDUNDANT__L4_marginal**

## RW  ceiling=0.05922
- reused from sibling run log (LGBM 0.05817/0.05913/0.06036); reproduces L-1 variant-B 0.0592 exactly

## RW+CSR_TS  ceiling=0.06159
- ridge 0.04973 (alpha=1000.0, int_eval=0.06749)
- lgbm 0.06159 ± 0.00107 (n=600, seeds ['0.06166', '0.06286', '0.06024'])
