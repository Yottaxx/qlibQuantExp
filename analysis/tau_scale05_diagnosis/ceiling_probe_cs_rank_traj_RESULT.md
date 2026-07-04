# CS-rank Ceiling Probe — TRAJECTORY arm (decisive L-4 test)

- SMOKE=False F=158 WIN=8 model_ref=0.0786 raw_window(L-1 B)=0.0592 dayT_lift_ref=0.0033
- **RW=0.05922  RW+CSRt=0.06083  traj_lift=+0.00161** (day-T lift was +0.0033)
- **VERDICT: CS_TRAJ_REDUNDANT_WITH_WINDOW__L4_no_go**

## RW  ceiling=0.05922
- reused from prior window-probe run (LGBM 0.05817/0.05913/0.06036)

## RW+CSRt  ceiling=0.06083
- ridge 0.04974 (alpha=1000.0)
- lgbm 0.06083 ± 0.00093 (n=600, seeds ['0.05976', '0.06202', '0.06071'])
