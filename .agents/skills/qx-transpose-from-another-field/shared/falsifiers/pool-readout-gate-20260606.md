# Falsifier Card — Pool/Readout Gate (KEY-SVD + PROBE-CEILING)

**claim_id:** `pool-readout-gate-20260606`
**filed_at:** 2026-06-06 (pre-registered BEFORE seeing results; seed-42/43/44 backbones training at filing time)
**filed_by:** pooling/readout forensics (rank-1 + rank-2 of the SOTA-falsifier shortlist)
**parent_baseline_sha:** `3675090` (macV2, current default = full_135)
**status:** open — executing (settlement appended on completion)
**gates:** `pool-forensics-20260524.md` (R1 multihead / R2 pure-attn / R3 temperature) and its dispatched run `pool_forensics_20260606` (ledger id 67). This card is a CHEAP, mostly inference-only pre-screen that decides whether those operator-fix arms can work AT ALL — run it BEFORE spending GPU on R1/R2/R3 (the team's own "τ-lesson: cheap test first", ledger id 66).

---

## Why this card exists

`attention_pooling/pool_entropy_norm_mean = 0.99999` (six nines) has been read as "the factor-attention mechanism is dead." But that metric measures only the attention **weights** `a_i`. Code (`attention_pooling.py:71-86`) shows that even at uniform weights the pooled vector is `LayerNorm(W_o(W_v · mean_n h_n))` — a learned linear recoloring the entropy metric is structurally **blind** to. So "six-nines entropy" and "the factor pool is dead" are NON-equivalent claims that have been conflated. The true liveness of the factor axis at readout has never been measured. There are three distinct worlds:

- **A — operator-state collapse (fixable).** The post-MoE keys DO vary across the N=158 factors, but the learned query is near-zero / the 1/√D temperature is cold, so attention is uniform. A sharper/larger query (R1 multihead) or lower temperature (R3) could un-collapse it.
- **B — input-forced collapse.** The keys are near-constant across factors (no across-factor contrast exists). NO operator (R1/R2/R3, entmax, PMA-k) can build a useful contrast. pool-forensics is then pre-decided to fail.
- **C — live-projection mean-pooler.** The pool is effectively a (LayerNorm'd) mean, and that is already ~IC-optimal under the cross-sectional ranking objective: non-uniform reweighting does NOT raise rank_ic. Sharpening is possible but futile.

KEY-SVD-GATE separates **B** from **(A or C)**. PROBE-CEILING separates **A** from **C**.

## The claims — stripped of hedging

- **KEY-SVD (rank 1).** The six-nines is an OPERATOR state, not input geometry: the centered cross-factor key matrix carries real across-factor variation (`key_s1_over_common_norm ≥ 0.10`), and the current attention is uniform only because the learned query is cold (`current_logit_std ≈ 0`).
- **PROBE-CEILING (rank 2).** On the FROZEN promoted backbone, a capacity-matched convex-reweight over the N factor axis (Arm A) beats the LayerNorm'd-mean-pool parity (Arm 0) by `≥ +0.003` daily rank_ic, AND the 158 per-slot marginal ICs disperse (top/bottom-decile abs-IC `≥ 2×`) with a non-degenerate slot-signal covariance (effective rank `≥ 2`).

## The prohibitions (what these claims FORBID)

| Prohibition | What its failure means |
|---|---|
| **P1 (world B).** If `key_s1_over_common_norm ≤ 0.05` (keys ~constant across factors), then the across-factor contrast needed for ANY non-uniform attention does not exist. This FORBIDS pool-forensics R1/R2/R3 from un-collapsing usefully — they become pre-decided no-ops and should NOT be run. |
| **P2 (world C / rational equilibrium).** If PROBE Arm A − Arm 0 `< +0.003` on the majority of seeds AND the slot-IC decile ratio `< 2` (marginal slot ICs ~equal), then equal-weight (LN'd mean) pooling is ~IC-optimal here and non-uniform factor weighting CANNOT raise rank_ic out-of-sample. This FORBIDS any "sharpen the factor pool" intervention (both open cards' ON arms, entmax, PMA-k) from helping — even if KEY-SVD says sharpening is achievable. |
| **P3 (gate consistency).** If KEY-SVD says world B (P1 fires) but PROBE Arm A nonetheless beats Arm 0 by `≥ +0.003` with dispersed slot ICs, the two forensics CONTRADICT (a useful reweighting on reps whose keys carry no contrast) — flag a measurement bug, do not promote either reading. |
| **P4 (faithfulness).** If the trained backbone's valid `daily_rank_ic` is not within seed-noise of the promoted anchor (~0.076 ± 0.004), the forensic is being run on a non-faithful model; abort and reconcile against the uncommitted working-tree diff before trusting any verdict. |

## Arms / measurements

Single faithful `full_135` backbone per seed (42, 43, 44), trained to the promoted protocol (early-stop on, checkpoint=valid_rank_ic:max). `h_last` = the [B, N, D] input to `net.factor_pooling`, captured via a `forward_pre_hook` (non-invasive, fp32, no autocast).

**KEY-SVD (inference-only):** per valid sample, centered cross-factor key matrix `K_c = W_k·h_last − mean_factor`; report SVD spectrum (σ₂/σ₁, participation ratio, effective rank), `s1_over_common = (σ₁/√N)/‖mean key‖` (the world-B indicator), the model's actual attention logit-std + softmax entropy (reproduce six-nines), a unit-normalized temperature sweep on the optimally-aligned top key direction (peakability, decoupled from the cold query), the value-channel across-factor variation (world-C signal mixing), and the cross-stock attention-weight std (is attention even stock-varying?). Cross-check raw `h_last`.

**PROBE-CEILING (frozen backbone, ridge/Adam on cached features):** day-subsampled train features (DK_L) fit; all valid (DK_I) eval; metric = `daily_rank_ic_mean`.
| Arm | Description | Added DoF over Arm 0 |
|---|---|---|
| **Arm 0** | parity: ridge on `LayerNorm(mean_n h_n)` | 0 |
| **Arm A** | convex-reweight: `head · LayerNorm(Σ_n a_n h_n)`, `a = softmax(θ)`, uniform warm-start (= Arm 0 at init) | the `[N]` weight vector only |
| **Arm B** | 158 marginal slot ridges (one per factor over its D-vec) → slot-IC distribution, decile ratio, slot-signal covariance condition / effective rank | n/a (diagnostic) |

The per-sample LayerNorm in Arm 0 / Arm A mirrors the model's own pool LayerNorm so the Arm A − Arm 0 gap isolates REWEIGHTING from the LN nonlinearity (capacity-matched).

## Ex-ante metric specification (binding)

| Field | Value |
|---|---|
| `universe` | `csi300` (primary); csi800 corroboration deferred to a second pass if csi300 is ambiguous |
| `horizon` | `t+5` |
| `seeds` | `[42, 43, 44]` (KEY-SVD reported per-seed for stability; PROBE decision on the 3-seed majority) |
| `KEY-SVD decision` | world B iff `s1_over_common ≤ 0.05`; operator-state-collapse iff `≥ 0.10`; else inconclusive |
| `PROBE decision` | sharpening_can_help iff `mean(ArmA−Arm0) gap ≥ +0.003 on majority of seeds` AND `decile_ratio ≥ 2` AND `cov_effrank ≥ 2`; else rational-equilibrium null supported |
| `faithfulness gate` | backbone valid `daily_rank_ic` ∈ [0.072, 0.082] per seed |
| `n_trials_at_filing` | forensic, NOT promotion — does not consume the Deflated-Sharpe budget |
| `n=3 caveat` | per the τ n=6 reversal (ledger id 61-62), a 3-seed PROBE gap is SUGGESTIVE not promotion-grade; any decision to ship a pool change needs a fresh-seed kill-check. |

## Anti-rescue rules (binding)

1. **No metric swap.** KEY-SVD verdict rests on `s1_over_common`; PROBE on `daily_rank_ic` Arm A−Arm 0 + slot dispersion. Cannot switch to IC, to in-sample fit, or to the (blind) `pool_entropy_norm` post-result.
2. **No threshold drift.** `s1_over_common` 0.05/0.10 bands and PROBE `+0.003 / 2× / effrank≥2` are fixed.
3. **No capacity laundering.** Arm A must stay capacity-matched (only the `[N]` weight added over Arm 0). Cannot widen it to a `[N×D]` free matrix to manufacture a gap.
4. **No faithfulness waiver.** If P4 fires, results are void; cannot "use them anyway."
5. **No n=3 over-claim.** A 3-seed PROBE gap may NOT be reported as a promotion; it gates whether the operator-fix arms are worth running, nothing more.

## Settlement protocol

On completion: aggregate per-seed → `analysis/pool_readout_forensics/aggregate.json` + `report.md`; apply the decision rules above; append `kind=result` and `kind=decision` rows to `shared/ledger.md` citing this card; append a `## Settlement` section below (do NOT edit the pre-registered content); route:
- **P1 fires (world B):** stop `pool-forensics-20260524` (do not re-dispatch R1/R2/R3); the six-nines is input-forced → feeds the parked HC-3 / read-from-time-hidden-state decision.
- **P2 fires (world C):** the factor pool is a rational mean; sharpening is futile → same HC-3 routing; deprioritize the whole Stage-2 factor-attention program.
- **Neither (world A + reweighting helps):** pool-forensics R1/R2/R3 are vindicated with a concrete IC-proportional target → re-dispatch with a fresh-seed kill-check.

---

*Filed against the SOTA-falsifier shortlist (workflow `wf_bfd1a207-f10`), gating `pool-forensics-20260524.md` / ledger id 66-67. Anchors `module/architecture/attention_pooling.py:71-131`, `module/quant_moe_model.py:402-408`. Implementation `scripts/pool_readout_forensics.py`, `scripts/aggregate_pool_forensics.py`. Status open until 3-seed sweep completes.*

---

## Settlement — `settled-P2-fires` (2026-06-06)

**Executed on FAITHFUL pre-trained backbones LOADED from MLflow (no retraining), n=6 seeds 42-47 × 2 families** (`full_135` τ-frozen baseline exp 325032212672679181; `tau_scale_05` exp 867178867749867261). Faithfulness gate passed: forensic fp32 valid `rank_ic` = 0.0781±0.0027 / 0.0775±0.0033, matching saved MLflow values. Artifacts: `analysis/pool_readout_forensics/REPORT.md`, `combined.json`, per-family `report.md`, 12 `seed*/results.json`. Adversarially verified (workflow `wf_0d6432f7-af6`, 4 skeptic lenses + adjudicator, conf 0.74) — verification forced the PROBE to add the input-conditional / multi-mode / 2nd-moment arms below (the original static-only probe was over-scoped).

**KEY-SVD → P1 does NOT fire (NOT world B).** `key_s1_over_common` = 2.17 / 2.90 (>> 0.05); key effective-rank 4.2 / 2.8; peakability entropy @ logit-std 3 = 0.66 / 0.71. Keys vary richly across factors and are peakable. The six-nines is an OPERATOR-STATE artifact: learned `‖q_proj‖ ≈ 0.01`, `current_logit_std ≈ 0.02`, cross-stock attention-weight std ≈ 5e-5 (a cold, near-input-independent query), NOT dead inputs.

**PROBE → P2 FIRES (world C), at full strength on the operative function class.** All arms are ridge/Adam heads on frozen reps, hold-selected, +0.003 daily_rank_ic gate vs Arm0 (LN-mean parity):
| arm (function class) | full_135 gap | tau_05 gap | seeds ≥+0.003 |
|---|---:|---:|---|
| ArmA static-global (hold-sel) | −0.0009 | 0.0000 | 0/6, 0/6 |
| **ArmA′ per-sample conditional (= R1/R3)** | **−0.0007** | **−0.0031** | **0/6, 0/6** |
| ArmC PMA-k4 multi-mode (= R1) | −0.0024 | −0.0009 | 0/6, 0/6 |
| Arm-ASP 2nd-moment (separate lever) | +0.0003 | +0.0002 | 1/6, 1/6 |

ArmB: slot-IC decile ratio 1.45/1.40 (factors near-equally weak), max single-slot |IC| ≈ 0.074 ≈ full-model IC. Even a free, hold-selected input-conditional query converges to near-uniform attention (cross-stock weight std 0.0004/0.0008). **τ-invariant** (full_135 ≈ tau_05 on every metric — corroborates τ is architecturally disconnected from the factor pool, ledger 16/46).

**Decision.** Equal-weight (LN-mean) factor pooling is ~IC-optimal here; the factor cross-section is a bag of ≈equally-weak, redundant signals best aggregated by averaging. **Do NOT re-dispatch pool-forensics R1/R2/R3 to chase RankIC** — they would un-collapse `pool_entropy` (mechanically possible: keys are peakable) but leave RankIC flat-or-worse. This is the "factor axis is signal-less at readout" outcome → **feeds the parked HC-3 / read-from-time-hidden-state decision** (the alpha is time-dominated; Stage-2 factor-attention is a dead lever for RankIC).

**Scope / caveats (binding):** PROVEN = static-global, per-sample-conditional, AND PMA-k4 factor reweighting are parity-or-worse at n=6×2 on csi300 t+5, 2020-2022. NOT covered: 2nd-moment ASP is a *separate* lever (also flat here, 1/6 — would need its own card if pursued); regime generalization (csi800 / other windows) untested → seed-robust, not regime-robust. The static-reweight magnitude in the original PROBE was selection-inflated (Arm0 hold-λ-selected vs ArmA unselected final-step); this settlement uses the HOLD-SELECTED ArmA, so the honest static ceiling is parity.

*Settled by `scripts/pool_forensics_combined_report.py`. Status: `settled-P2-fires`. Do not edit the pre-registered content above.*
