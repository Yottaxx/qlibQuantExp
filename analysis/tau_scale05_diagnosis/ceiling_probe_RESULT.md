# L-1 Ceiling Probe — RAW Alpha158 (model-agnostic)

- SMOKE=False  RUN_B=True  F=158  model_ref(RankIC)=0.0786
- train 2008-01-01..2020-03-31 | int-eval 2019-01-01..2020-03-31 | test 2020-07-01..2022-12-31
- **max ceiling = 0.05922  (Δ vs model -0.01938)  → DATA_SIGNAL_CEILING__redirect_to_portfolio_IR**

## A_snapshot  ceiling=0.05345
- ridge: test RankIC 0.04445 (alpha=1.0, int-eval 0.04893)
- lgbm: test RankIC 0.05345 ± 0.00077 (n=1000, seeds ['0.05331', '0.05258', '0.05446'])

## B_window  ceiling=0.05922
- ridge: test RankIC 0.04931 (alpha=10000.0, int-eval 0.05532)
- lgbm: test RankIC 0.05922 ± 0.00089 (n=1000, seeds ['0.05817', '0.05913', '0.06036'])

## Interpretation (decisive)
The strongest non-RST-MoE learner on the **same** raw Alpha158 features — including the model-matched
8-step window (variant B) — tops out at **0.0592** test RankIC, **0.0194 below** the model's 0.0786.
This is the OPPOSITE of "the model leaves signal on the table": the architecture **beats** a properly-tuned
GBDT on the same input. Window (B) > snapshot (A) by +0.006–0.009 for both learners, confirming the 8-step
input is justified (not input-starvation) and the A-vs-B comparison is fair.

Caveat: the model's 0.0786 is selection-inflated (~+0.006: epoch ~+0.0035 + seed ~+0.003, because valid==test);
the honest model number is ~0.072–0.075 — still clearly above the 0.0592 ceiling. The cleanest confirmation
is to run the MODEL itself on this same honest forward split (a future GPU step, Step-0b).

Both readings converge on the same action: **chasing more RankIC via architecture is low-EV** (the model
already meets/beats the classic-baseline IC ceiling and every architecture axis — readout/pool/τ/expert/
stock-backbone — is killed). Redirect to the **portfolio-IR track (L-6)**: a batch-IR/Sharpe + turnover
auxiliary on the bare backbone (readout OFF, MSE kept), which puts a gradient on the deployment metrics
(IR_wc/MaxDD/turnover) that MSE is provably blind to. (ledger 84–85)

Methodology: leakage-clean — DK_L train 2008-01..2020-03 / DK_I test 2020-07..2022-12; ridge-α and LGBM-n
tuned ONLY on the train-internal 2019-01..2020-03 RankIC; no per-test-day refit; 3 LGBM seeds; features
byte-match the model (RobustZScoreNorm+Fillna inherited via qlib append semantics).
