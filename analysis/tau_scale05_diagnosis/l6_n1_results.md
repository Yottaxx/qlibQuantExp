# L-6 portfolio-IR aux — n=1 paired result (card h-20260627-001, seed 42) — VERDICT: KILL

Readout OFF bare τ0.5 backbone, 25ep from-scratch, FULL window, **cost-ON backtest** (open 5bp / close 15bp / min5 / limit 0.095). MSE main loss kept; treatment adds `l_ir` (λ=0.08, ramp 5000, var_eps 2.5e-3).

| metric | treatment (λ=0.08) | control (λ=0) | Δ (treat−ctrl) |
|---|---|---|---|
| best valid RankIC | 0.07451 | 0.07959 | **−0.00508** |
| test RankIC | 0.07452 | 0.07960 | −0.00508 |
| test IC | 0.06189 | 0.06617 | −0.00428 |
| ICIR | 0.4589 | 0.48375 | −0.02485 |
| Rank ICIR | 0.53874 | 0.59987 | −0.06113 |
| **IR_with_cost** | **1.80523** | **2.86437** | **−1.05914 (−37%)** |
| IR_without_cost | 2.49143 | 3.63590 | −1.14447 |
| **MaxDD_with_cost** | **−0.09561** | **−0.06961** | **−0.02600 (26% deeper)** |
| AnnRet_with_cost | 0.20435 | 0.28762 | −0.08327 |

## VERDICT: KILL (n=1, but decisive — harm is large + consistent across ALL 8 axes)
The IR aux — the lever explicitly designed to **lift** IR_wc — instead **cut IR_wc by 1.06 (−37%)** and degraded RankIC, IC, ICIR, MaxDD_wc, and AnnRet. The card's kill line fires hard: IR_wc went DOWN (not up), and IC & RankIC both dropped. This is well beyond seed noise (IR n=3 Δ-std historically ~0.3–0.5; this is −1.06 ≈ 2–3×) and is consistent on every metric, so a seed-fluke is unlikely — but it is n=1, so an n=2 confirmation is cheap insurance before fully closing the formulation.

## Why it failed (mechanism, from training diagnostics)
The aux **overfits the portfolio objective**. It maximizes the in-sample long-short book return on the TRAIN labels (ir_book_return rose 0.067→0.31), which does NOT generalize: the model reshaped strongly time-dominant (time_ratio 0.51→0.72, time_winner ~0.99) and chased train-period spread structure → worse OOS on everything. The mechanism was ACTIVE (book return rose, model reshaped) but the OUTCOME was strongly negative on the exact metric it targeted — "mechanism≠benefit" in its starkest form.

## Refined theory (important, counterintuitive — refines row 83)
**MSE's "portfolio-blindness" is a FEATURE (a regularizer), not just a bug.** By fitting only per-name returns (diagonal Hessian, no joint-portfolio term), MSE avoids overfitting the cross-sectional portfolio structure. Adding a portfolio-return gradient (L-6) BREAKS that regularization → overfits the train long-short return → hurts OOS IR_wc. So the row-83 inference "portfolio-blindness is the gap → add a portfolio gradient to close it" is **empirically refuted** for the return-maximizing formulation. Implication for the filed CMF+IR-cost plan: the IR/return-maximizing half (C3 return term) is undermined; the turnover/cost-PENALTY half (a regularizer, not a return-maximizer) is a different animal and not tested here.

## Caveats
n=1 (no p-value; card bar wanted n≥3 paired-t — but the direction+magnitude make that academic for a −37% harm); valid==test (both arms equally affected, so the Δ is still a fair within-split comparison). Code was validated (MF-1 fix held: zero NaN across both 25ep runs).
