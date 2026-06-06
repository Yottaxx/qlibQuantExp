# Pool / Readout Forensics — Combined Archived Report

**Claim card:** `pool-readout-gate-20260606` (KEY-SVD-GATE + PROBE-CEILING). Backbones LOADED from MLflow artifacts (no retraining): `full_135` = tau-frozen baseline (exp 325032212672679181), `tau_scale_05` = tau-unclamp scale=0.5 (exp 867178867749867261).

## 0. Faithfulness (sanity valid rank_ic recomputed by the forensic fp32 forward)

| family | n | valid rank_ic (mean ± std) | per-seed |
|---|---:|---|---|
| full_135 | 6 | 0.0781 ± 0.0027 | 42:0.0802, 43:0.0779, 44:0.0735, 45:0.0810, 46:0.0792, 47:0.0767 |
| tau_scale_05 | 6 | 0.0775 ± 0.0033 | 42:0.0811, 43:0.0788, 44:0.0724, 45:0.0781, 46:0.0797, 47:0.0746 |

Gate [0.072, 0.082]: all faithful → the geometry below is measured on the promoted-class model.

## 1. KEY-SVD-GATE — is the six-nines factor-pool collapse input-forced (B) or operator-state (A/C)?

| metric | full_135 | tau_scale_05 | reads |
|---|---:|---:|---|
| current softmax entropy | 0.99995 ± 0.00004 | 0.99990 ± 0.00008 | ≈1.0 ⇒ six-nines reproduced |
| current logit std | 0.017261 ± 0.013560 | 0.027292 ± 0.016064 | ≈0 ⇒ operator is cold |
| **key s1/common-norm** | 2.1734 ± 0.7611 | 2.9033 ± 1.5327 | **world-B indicator: ≤0.05 ⇒ keys ~constant** |
| key sigma2/sigma1 | 0.4295 ± 0.3046 | 0.2996 ± 0.2473 | across-factor key rank structure |
| key effective rank | 4.21 ± 4.22 | 2.75 ± 3.15 | # independent across-factor contrasts |
| peakability entropy @ logit-std 1 | 0.9109 ± 0.0351 | 0.9134 ± 0.0186 | shape of top key dir |
| peakability entropy @ logit-std 3 | 0.6590 ± 0.2259 | 0.7107 ± 0.1577 | <0.9 ⇒ a sharp op could peak |
| value s1/common-norm | 1.6510 ± 0.3102 | 1.6085 ± 0.1811 | world-C: value bank varies across factors? |
| cross-stock attn-weight std | 0.000055 ± 0.000043 | 0.000087 ± 0.000052 | is attention even stock-varying? |

- **full_135 verdict:** `operator_state_collapse`
- **tau_scale_05 verdict:** `operator_state_collapse`

## 2. PROBE-CEILING — can any factor-pool operator beat equal-weight (LN-mean) pooling?

All arms: ridge/Adam heads on the FROZEN backbone reps; metric daily_rank_ic on valid; +0.003 promotion gate vs Arm0. Arm0 = LN-mean parity. ArmA = STATIC global reweight (hold-selected). **ArmA' = INPUT-CONDITIONAL query pooler (the R1/R3 function class).** ArmC = PMA k=4 multi-mode. Arm-ASP = 2nd-moment dispersion (separate lever).

| arm | full_135 | tau_scale_05 | function class |
|---|---:|---:|---|
| Arm0 mean-pool rank_ic | 0.07159 ± 0.00735 | 0.07032 ± 0.00531 | parity |
| ArmA static reweight (hold-sel) | 0.07069 ± 0.00672 | 0.07032 ± 0.00531 | global static |
| **ArmA − Arm0 gap** (static) | -0.00090 ± 0.00221 | 0.00000 ± 0.00000 | global static |
| **ArmA' conditional rank_ic** | 0.07092 ± 0.00790 | 0.06724 ± 0.01027 | **per-sample (R1/R3)** |
| **ArmA' − Arm0 gap** (the GATE) | -0.00067 ± 0.00092 | -0.00308 ± 0.00506 | **per-sample (R1/R3)** |
| ArmC PMA-k4 rank_ic | 0.06920 ± 0.00646 | 0.06942 ± 0.00652 | multi-mode (R1) |
| ArmC − Arm0 gap | -0.00239 ± 0.00476 | -0.00090 ± 0.00178 | multi-mode (R1) |
| Arm-ASP std contribution | 0.00031 ± 0.00317 | 0.00021 ± 0.00248 | 2nd-moment (separate) |
| ArmA' cross-stock weight std | 0.00036 ± 0.00035 | 0.00083 ± 0.00060 | is cond. weighting used? |
| ArmB slot-IC decile ratio | 1.45 ± 0.22 | 1.40 ± 0.08 | factor equality |
| ArmB max |slot-IC| | 0.0746 ± 0.0048 | 0.0740 ± 0.0029 | best single factor |

- **full_135** ArmA' (conditional) per-seed gap: 42:-0.0021, 43:-0.0005, 44:-0.0015, 45:0.0000, 46:0.0001, 47:-0.0000  | seeds ≥+0.003: cond 0/6, pma4 0/6, static 0/6, asp 1/6  → **R1/R2/R3 helps=False**, ASP helps=False
- **tau_scale_05** ArmA' (conditional) per-seed gap: 42:-0.0043, 43:-0.0007, 44:-0.0127, 45:-0.0000, 46:0.0010, 47:-0.0018  | seeds ≥+0.003: cond 0/6, pma4 0/6, static 0/6, asp 1/6  → **R1/R2/R3 helps=False**, ASP helps=False

## 3. Cross-family: is the factor-pool pathology τ-invariant?
- Δ(key s1/common) [full_135−tau_scale_05] = -0.7300; Δ(ArmA'-conditional gap) = 0.00242.
- Established finding (ledger 16/46): τ acts only on the additive time-embedding, architecturally disconnected from `factor_pooling`. Near-zero deltas corroborate the pool pathology is τ-invariant.

## 4. Settlement (pool-readout-gate-20260606)

- KEY-SVD verdict (all families): {'operator_state_collapse'} — collapse is **operator-state, NOT input-forced** (P1 does not fire; key s1/common >> 0.05).
- **P2 FIRES (world C, full strength):** neither static reweight, the per-sample input-conditional pooler (ArmA', = the R1/R3 function class), nor PMA-k4 (R1) beats LN-mean by +0.003 on a seed-majority in either family. Equal-weight pooling is ~IC-optimal; the 158 factors are near-equally-weak & redundant. **Do NOT re-dispatch pool-forensics R1/R2/R3 to chase RankIC** — they would un-collapse pool_entropy but leave RankIC flat. Feeds the parked HC-3 / read-from-time-hidden-state decision.
- ASP 2nd-moment side-channel: no material gain — the dispersion lever is also flat.

**Scope (caveats that travel with this settlement):**
- PROVEN: static-global AND per-sample-conditional AND PMA-k4 factor reweighting are parity-or-worse at n=6×2 (csi300, t+5).
- The static ArmA magnitude was selection-inflated (Arm0 hold-λ-selected vs ArmA unselected final-step); this report uses the HOLD-SELECTED ArmA, so the honest static ceiling is parity.
- Regime scope: single split, csi300-only, 2020-2022, n=6 same-window seeds → seed-robust, not regime-robust (csi800/other windows untested).
- n=6 per family (kill-check-grade, per the τ n=3→n=6 reversal lesson).

