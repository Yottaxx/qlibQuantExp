# Portfolio-Aligned Cross-Stock Readout — `CMF+IR-cost` (h-20260623, **MSE-corrected 2026-06-24**)

**Date:** 2026-06-23; **corrected 2026-06-24.** **Status:** PLAN (no source modified). Supersedes the
*RankIC-only* framing of `readout_crossstock_PLAN.md` (M0–M3) and `warmq_to_readout_qknorm_PLAN.md`
(R0–R3) by adding the levers that **positively lift the portfolio vector** (IC, ICIR, IR, AR, MDD).

> **⚠ CORRECTION (2026-06-24).** The first version of this plan asserted "the live gradient is
> `main_loss=listmle`, order-only, so a per-day common-mode is gradient-free and uniform attention is a
> loss-neutral parking spot." **That premise was FALSE.** Code-verified: `work_flow.py:167`
> `main_loss="mse"`, `work_flow.py:175` `mse_normalize=False` ⇒ the live loss is **plain
> `F.mse_loss(p, y)`** (`losses.py:76-81`, no de-mean/no scale-norm); `l_ic`/`l_listmle` are
> **monitor-only** (`quant_moe_model.py:701-703`, not in the gradient); label = **`CSZScoreNorm` robust**
> (`work_flow.py:107`, per-day mean≈0 std≈1). The whole root-cause section, the math, and one ladder
> rung are revised below. **The core scheme survives and simplifies** (the calibration rung drops; the
> portfolio-aware aux is *more* justified). Verified by workflow `wf_3a3c7582-38f` (3 re-derivations ×
> adversarial verify + synthesis). Sibling docs carrying the same stale assumption are flagged in §10.

---

## 0. One-paragraph thesis (corrected)

The live objective is **plain cross-sectional MSE of the raw score `p` against a per-day robust-z-scored
label `y`** — a **calibrated per-name point-accuracy loss**. It already (a) **penalizes** score-level
common-mode (`∂L/∂c = 2·mean(p)`, pulling `mean(p)→0`) and (b) **anchors dispersion** while **directly
maximizing Pearson IC** (`L* = 1−IC²` at the calibrated optimum). So **calibration and common-mode are
NOT the gap** — MSE handles them for free. What MSE is **structurally blind** to is the **portfolio
geometry**: it is additively separable over names (`∂²L/∂pᵢ∂pⱼ = 0 ∀ i≠j`), with **zero gradient** on
pairwise bet-correlation `ρᵢⱼ`, effective breadth `BR_eff`, turnover, or with-cost drawdown. By Grinold
`IR = IC·√BR_eff·TC`, MSE trains **IC alone**. **That portfolio-blindness is the true, durable gap** —
and it is the reason the severe with-cost IR/MDD collapse is invisible in gross IC space. The fix is to
put a **positive gradient on the risk-adjusted, cost-aware portfolio outcome** (the IR/Sharpe + turnover
aux), keep the cross-stock output **idiosyncratic / breadth-preserving** (contrast operator), and
**shrink** it per-channel so it can't dominate the book — `CMF+IR-cost`.

---

## 1. Root cause — data-confirmed, two-headed, **portfolio-blind objective**

Phase-1 forensics (real preds, anchor g012 n=3 / warm-q n=5 / stockexp n=3, all 611 days) confirmed the
asymmetry: RankIC flat (0.0784 / 0.0799 / 0.0778) while risk-adjusted quality degrades. The two failing
arms have **different wounds**, both off-axis to what MSE trains:

| arm | RankIC | wound (in MSE terms) | evidence |
|---|---|---|---|
| **h2_stockexp** | 0.0778 (flat) | **signal/calibration** — corrupts the exact quantity MSE regresses | IC_mean ↓ 0.0669→0.0613 (72% of ICIR loss); bet-corr **unchanged** 0.798→0.798; eff#names *rises* 183→187. IC_mean is what `L=1−IC²` optimizes ⇒ a real MSE-relevant regression. RankIC flat because Spearman is a **coarser monotone byproduct** than the cardinal IC MSE trains. |
| **warm-q** | 0.0801 (table-high) | **breadth/risk** — lives entirely in MSE's null space | IC_std ↑ (64% of ICIR loss); implied breadth **−10.8%** (worst); rides V-escape `‖Wv‖→88`; IC only 0.0647. A high-variance rank reshape MSE under-rewards (IC ~flat) whose breadth/turnover cost MSE cannot see. |

Two caveats now **explained** (not open questions):

1. **The final-pred common-mode proxy is a dead end** because **MSE forbids score common-mode**
   (`mean(p)→0` by construction) — so warm-q/stockexp being *less* market-loaded on the scored output is
   the *expected* signature, not an anomaly. (The earlier plan listed this as a puzzle.)
2. **The with-cost collapse (IR_wc 1.75→1.03, MDD 2.4×) is invisible in gross IC space** (gross MaxDD
   only −6.7%→−7.6%) ⇒ **cost/turnover/TopK-concentration driven** — exactly the terms `IR=IC·√BR_eff·TC`
   says MSE has zero gradient on.

**Bottom line:** the cross-stock layer's degrees of freedom drift toward configs MSE doesn't penalize but
the portfolio does — diluting the calibrated per-name signal (stockexp) or buying rank with a
low-breadth high-variance channel (warm-q). **Fixing common-mode/calibration fixes neither (MSE already
does them); fixing RankIC fixes neither (it is the coarse byproduct being gamed). Only a positive
portfolio gradient + breadth-preserving operator can lift the vector.**

---

## 2. The math — the three MSE properties, and why RankIC≠IR

Live objective `L = (1/B) Σᵢ (pᵢ − yᵢ)²`, `y` = per-day robust-z (mean≈0, std≈1).

**(1) Shift-SENSITIVE** (retracts "common-mode is gradient-free"). `∂L/∂c = 2(mean(p)+c−mean(y))`; at
`c=0`, `= 2·mean(p)` (since `mean(y)≈0`). MSE actively pulls `mean(p)→0`; score common-mode strictly
raises `L`. *Contrast* the **monitor-only** location-invariant losses the old docs wrongly cited as
"active": ListMLE (`e^c` cancels in softmax ratios), `cs_ic` (de-means), `cs_mse(normalize=True)`
(de-means) — none carry gradient here.

**(2) Dispersion-ANCHORING** (makes external `l_ic` + dispersion-floor redundant/opposing).
`L = var(p) + 1 − 2·IC·std(p) + mean(p)²`. Minimize: `mean(p)*=0`, **`std(p)* = IC·std(y) ≈ IC ≈ 0.23`**
(empirically, **not** 1), residual `L* = 1−IC²` strictly decreasing in `IC²` ⇒ **minimizing MSE maximizes
Pearson IC²** with calibrated scale. A cardinal `l_ic` term double-counts MSE's own IC gradient; a
dispersion **floor** at `std(y)≈1` would *fire* (`≈0.59`) and push `std(p)` **above** MSE's optimum —
i.e. it **fights MSE**. (Both dropped — see §3 C4.)

**(3) Portfolio-BLIND** (the survivor — *stronger* under MSE than under the assumed ListMLE).
`∇_{pₖ}L = (2/B)(pₖ−yₖ)` (own pair only); Hessian is **diagonal**, `∂²L/∂pᵢ∂pⱼ = 0 ∀ i≠j`. `L` has **no**
functional dependence on `ρᵢⱼ`, `BR_eff = (1ᵀw)²/(wᵀRw)`, turnover, or with-cost MDD. By Grinold
`IR ≈ IC·√BR_eff·TC`: `∂L/∂√BR_eff = 0`, `∂L/∂TC = 0`. (MSE is *more* portfolio-blind than ListMLE,
which at least couples names through the logsumexp ordering.)

**Reframed RankIC≠IR.** RankIC (Spearman) is a **looser monotone byproduct** than the calibrated Pearson
IC MSE trains — a per-name level/scale corruption moves Pearson IC (and `L`) while leaving the argsort
≈intact, so RankIC stays flat while the MSE-relevant signal degrades (the stockexp wound). IR depends on
`√BR_eff` and turnover/`TC` — terms with **literally zero MSE gradient** — so **nothing in the live
objective trains the multipliers that carry IR.** RankIC↑ + IR↓ is the generic signature of a layer that
reshapes rank via a high-variance channel MSE under-rewards (warm-q) while damaging breadth/turnover MSE
cannot see.

---

## 3. The scheme — `CMF+IR-cost` (corrected)

Build on the **existing** readout `CrossStockBlock` — the only post-pool `[B,D]` in-series **no-exit**
site (`module/architecture/cross_stock_block.py`, applied on `h_pooled` between
`quant_moe_model.py:542/568` and the head `:574`). Default-off / byte-identical when off.

| # | change | seam | role under **MSE** | metric |
|---|---|---|---|---|
| **C0 Placement** (keep) | `use_readout_stock_attn=True`, `qknorm=True`, **ungated**, mandatory residual, **no** score de-mean | `cross_stock_block.py:91-94,114`; **never** the dead scalar-sigmoid `gated` path `:60-61,111-112` | loss-independent; repeller justification now "MSE can't fit a dispersed CSZScore target (var≈1) with a per-day constant ⇒ uniform-replacing gives `L≥var(y)≈1`, strictly dominated" | collapse |
| **C1 Operator → CONTRAST** (keep, **re-motivated**) | `vmix = v − einsum(attn,v)` before `out_proj` | `cross_stock_block.py:105` | **NOT redundant with MSE.** MSE kills only *score* common-mode (`head=Linear(64,1)` constrains the scalar); C1 removes common-mode in the **64-dim value vector** `oᵢ`, which lives partly in the head's **null space — invisible to MSE**. Also an **attention repeller** (uniform ⇒ `Σᵢoᵢ=0` ⇒ zero idiosyncratic signal). Parameter-free in the V-path (V-escape-immune); synergistic with `mean(p)→0`. | IC_mean, breadth |
| **C2 Shrinkage → per-channel LayerScale** (keep) | `self.layerscale=nn.Parameter(ones(d_model))`; `o=o*layerscale` before residual; `out_proj` zero-init | before `cross_stock_block.py:114` | loss-independent — the durable "scalar sigmoid gates **freeze** at init" lesson; per-dim gradient shrinks residual market-β channels | ICIR, MDD |
| **C3 Risk-aligned aux** (keep — **REINFORCED, load-bearing**) | dollar-neutral L1-norm weights `w=s/(‖s‖₁+ε)`, `s=p−p.mean()`; EMA-Sharpe `l_ir=−r_t/√(EMA_var+ε)`; **+ turnover** `l_to=((w−w_prev[id])²).mean()` via id-keyed EMA; `total += λ_ir·ramp(step)·(l_ir+β_to·l_to)` | `quant_moe_model.py:688` | the **ONLY** component supplying a positive gradient on `√BR_eff` and `TC` — the exact terms MSE provably lacks (§2 prop 3). **Augment-only** (`w["mse"]=1.0` fixed, `λ_ir` ramp 0→~0.1, EMA-var); replacing MSE forfeits the IC calibration that is the model's strength. L1-norm scale-pin is *more* useful (score `std≈0.23`). | **IR, AR, MDD, turnover** |
| **~~C4 Calibration~~** (**REMOVED from gradient**) | ~~`l_ic(0.3)` + dispersion floor~~ | — | **redundant/opposing with MSE** (§2 prop 2): `l_ic` double-counts MSE's IC gradient; the floor *fights* MSE's calibrated `std(p)*≈0.23`. **The only thing MSE lacks is TAIL-WEIGHTING** (it weights all names equally, but the long/short tails carry the P&L) ⇒ the sole survivable calibration lever is an **optional tiny tail-weighted/NDCG-log ListMLE** complement *under* the IR-aux, never primary. | — |
| **C5 Diagnostics** (keep + **2 MSE canaries**) | eval-only emit: `score_std`, `score_pc1_frac`, `BR_eff` proxy, EMA-IR μ/√var **+ `mean(p)` (≈0; drift ⇒ common-mode injection) + `std(p)` vs its own ~0.23/realized-IC baseline (NOT vs std(y)≈1)** | `quant_moe_model.py:601-602` | closes the kill-gate gap; **LOO probe must measure `ΔIC_mean` AND `ΔMSE` AND `Δscore_std`, not only `Δrank_ic`** (under MSE a layer can be load-bearing on calibration while rank_ic stays flat) | — |

**Param cost:** `+d_model (=64)` for LayerScale; everything else is zero-param loss terms + non-grad
buffers. **Drive all of it via `QIB_MODEL_OVERRIDES_JSON` / `loss_weights` (`work_flow.py:362`) — do NOT
edit `work_flow.py`.**

---

## 4. Ablation ladder — **R0..R5 (6 rungs, one fewer; the calibration rung removed)**

Each rung adds **exactly one** change; **n≥6 fresh seeds (42–47), from-scratch** (NO finetune — gen-gap
~0.12 wall), paired-Δ vs the g012 anchor on the **full vector**. Cheap parameter-free rungs (R1–R3) gate
the loss-machinery rungs (R4–R5).

| rung | = prev + | attributes | predicted |
|---|---|---|---|
| **R0** | anchor g012 (readout OFF) | reference | preds on disk; extend to n=6 |
| **R1** | readout ON, qknorm, ungated, **POOLING** op (current code) | placement+sharpening only | **reproduces the failure** (RankIC flat-up, IR/ICIR down) — the control proving the *operator* change is what saves it |
| **R2** | POOLING→**CONTRAST** (C1) | idiosyncratic/breadth-preservation | RankIC held, IC_mean recovers, ICIR/MDD stop degrading |
| **R3** | + per-channel **LayerScale** (C2) | shrinkage | ICIR/IR lift via shrinking residual market-β channels; RankIC protected |
| **R4** | + **batch-IR aux** `l_ir` (pre-cost) (C3a) | positive breadth gradient | **IR_wc rises above anchor 1.750** (headline) |
| **R5** | + **turnover penalty** `l_to` (C3b) | turnover/stability/cost | with-cost gap closes; MDD shallower than −0.076; the only rung touching the cost channel |

C5 diagnostics + the 2 MSE canaries run eval-only on **every** rung, not as a rung. If R4 lifts IR but R5
doesn't improve with-cost IR/MDD → the wound was pre-cost-recoverable and turnover is not the lever
(clean sub-result). Optional tail-weighted-ListMLE rides *under* R4/R5, never its own rung.

---

## 5. Pre-registered protocol & launch discipline

- **Arms:** R0..R5 (**6 configs**). **Seeds:** 42–47 (n=6 fresh, **from-scratch — NO finetune**).
- **Epochs/LR/data:** match the anchor's exact schedule (`scripts/baseline_g012_scale05.env.sh`);
  single-variable discipline.
- **Single-GPU SERIAL:** one detached `nohup … &` job per (arm,seed), `kernels=1`, `MAX_RETRY=1`, via
  `scripts/run_diagnostic_experiments.py`, driven purely by `QIB_MODEL_OVERRIDES_JSON`
  (`loss_weights{ir,turnover}`, `use_readout_stock_attn`, operator/LayerScale flags). **NEVER inside a
  subagent.**
- **Cheap gates BEFORE the n=6 sweep:** (1) `/quant-leakage-audit` on R5 first — the IR+turnover aux
  reads `y` and an id-keyed per-stock weight EMA inside the t+5 horizon; confirm the buffer never crosses
  the train/valid boundary and the label horizon is unchanged; **blocking**. (2) `/quant-minimal-repro`
  on R5 seed42 only (≤15 min) — kill if RankIC < anchor−0.003 **OR** `score_std` shrinks >5% below ~0.23
  **OR** IR_wc < anchor on the proxy.
- **Stage:** R1–R3 (parameter-free / +64 params) first; only after R3 is non-regressing do R4–R5.
- **Train/eval asymmetry to control:** the train aux portfolio is built on the sampler's down/**UP**-
  sampled B (with-replacement duplicates, `sampler.py:72-75`) while eval IR is on the full day — report
  the upsampled-day fraction (consider unique-names-only aux).
- **After a survivor:** `/quant-walk-forward` + `/quant-stress` (commission/slippage grid — the only
  honest read on the with-cost channel) **before** any production claim.

---

## 6. Kill criteria — judge the WHOLE vector (paired n=6 Δ vs anchor, with-cost)

**KILL** if ANY: (a) RankIC < 0.0784−0.002; (b) **IR_wc does NOT rise above 1.750 with paired-t p<0.1**
(FLAT IR = dead — the positive-lift bar); (c) MDD_wc deeper than −0.076 by >10%; (d) IC < 0.0669
(calibration regressed — now the **exact MSE-trained quantity**); (e) `score_std` shrinks >5% below its
~0.23 baseline **OR** `score_pc1_frac` rises **OR** `mean(p)` drifts from 0; (f) readout `entropy_norm ≥
0.98` across the run (contrast op inert).

**PROMOTE** only if RankIC ≥ anchor **AND** IR_wc up (paired-significant) **AND** MDD not deeper **AND**
IC ≥ anchor **AND** score_std not shrunk **AND** BR_eff not fallen — sign-consistent across ≥4/6 seeds.

**LOO-probe correction (substantive):** the load-bearing leave-one-out probe must measure **`ΔIC_mean`
AND `ΔMSE` AND `Δscore_std`**, not only `Δrank_ic` — under MSE a layer can be load-bearing on
calibration while rank_ic stays flat (the stockexp wound). A uniformize-LOO leaving rank_ic unchanged
does **not** prove no-op.

**Clean negative that CLOSES the axis:** if R2–R3 (parameter-free) is null-but-safe **AND** R4–R5
(risk-aligned aux) also fails the IR_wc-up bar at n≥6 + `/quant-stress`, declare the cross-stock readout
PATH a dead lever (log to ledger like the TIME-readout closure) → redirect to sampler/EMA + the
h-20260610 cards.

---

## 7. SOTA grounding (workflow `wf_a409eb39-ea7`)

- **Risk-aligned loss (S1 top pick):** differentiable **batch-Sharpe/IR** on a soft dollar-neutral
  long-short book — Zhang, Zohren, Roberts, *Deep Learning for Portfolio Optimization* (2020,
  arXiv:2005.13665). The only family whose gradient depends on **dispersion/correlation** not rank;
  dollar-neutralization makes common-mode earn zero reward. Guardrails: L1/softmax-normalize (scale-pin),
  EMA-stabilize the variance denominator, ramp λ from 0. Secondary free complement: **tail-weighted /
  NDCG-log ListMLE** (the one thing equal-weighted MSE lacks). **AVOID** soft-RankIC as a primary lever.
- **Common-mode rejection (S2):** parameter-free **PC1/market-mode removal** and **DTML
  normalized-context differencing** — our C1 contrast operator is the in-attention realization (now
  motivated by *value-space* common-mode + repeller, since MSE already handles *score* common-mode).
  **QK-norm/cosine attention** (σReparam, arXiv:2303.06296) = freeze-immune sharpening. **StockMixer
  NoGraphMixer** as the softmax-free operator A/B control.
- **Architecture that moves RISK metrics (S3):** **MASTER** ablation — IR is a property of the cross-stock
  **MIX**, not the market-gate. **Per-channel LayerScale** on a mean-removed residual = the single change
  most associated with IR/MDD lift. Turnover loss (arXiv:2509.04541) as the cost-channel amplifier.

> **Note on MASTER (corrected):** with the verified objective, **our loss/label ≈ MASTER's exactly** —
> both are masked MSE on a CSZScore label (`MASTER_mechanism.md:123-124`). The "loss differs" delta in
> the original comparison **dissolves**, which *reinforces* the MASTER-replication thesis: cross-stock
> **placement** is the sole remaining lever.

---

## 8. Risks & alternatives

1. **Pre-cost IR doesn't control turnover** (the 1.75→1.03 is cost-driven, F1 conf 0.85) → C3b turnover
   term + mandatory `/quant-stress`; pre-cost IR lift is necessary-not-sufficient.
2. **IR-aux variance denominator noisy** on a sub-sampled (sometimes duplicated) day → EMA(var) + λ ramp
   0→0.1 (a dominant Sharpe term overfits in-sample variance and hurts the IC MSE trains).
3. **Contrast-at-uniform = `(uᵢ−ū)`** in the **V-path, not the score** → no scale-invariant score
   constraint reopens; still **monitor out_norm** as a `‖Wv‖`-inflation tripwire (warm-q V-escape cousin).
4. **Train/eval portfolio asymmetry** (sampler up-samples with replacement; eval full-day) can inflate
   train breadth → report upsample-day fraction; consider unique-names-only aux.
5. **Per-channel LayerScale can overfit a few seeds** → require the IR lift to survive the n≥6 fresh-seed
   kill-check.
6. **F1 refutes the breadth story for stockexp specifically** (72% IC_mean wound) → the contrast operator
   alone (R2) may be *inert* for stockexp; the **IR-aux (R4)** is what heals the portfolio metrics — the
   ladder is ordered to **see** this. (Under MSE, R2 heals IC_mean via value-space de-noising; R4 heals
   IR/MDD.)

**Alternatives if the integrated scheme stalls:** (A) StockMixer NoGraphMixer (softmax-free,
residualizing) as a collapse-immune operator A/B. (B) tail-weighted ListMLE *under* the IR-aux, never
primary. **Avoid** FiLM regime-gating in arm 1 (MASTER ablation: market-gate is NOT the IR source).

---

## 9. First action

Run **`/quant-leakage-audit`** on the **R5** config (read-only, zero-GPU): confirm the IR+turnover aux's
id-keyed per-stock weight EMA never crosses the train/valid boundary and the t+5 label horizon is
unchanged. If it passes → the two **parameter-free** edits to `module/architecture/cross_stock_block.py`
(CONTRAST op at the `:105` value-mix + ones-init per-channel `self.layerscale` before the `:114`
residual), then a single `/quant-minimal-repro` of R5 seed42 (≤15 min) before committing to the n=6
R0..R5 ladder. **Do NOT add `l_ic` or the dispersion floor to the gradient** (former C4 — removed); a
calibration lever, if ever wanted, is the optional tail-weighted-ListMLE complement under the IR-aux.

---

## 10. Sibling docs carrying the stale loss-neutrality assumption (correct before reuse)

All assert the live loss is order-only / location-invariant / additive-constant-invariant — **wrong** for
plain raw MSE. Each has a correction banner pointing here:

- `readout_stock_attention_analysis.md` §2c, §3b, §3d (heaviest carrier — "uniform = loss-neutral parking
  spot"; "final loss invariant to per-day affine"). Block-internal collapse still holds via the
  **loss-independent router-exit** reason, so the readout recommendation **survives**.
- `readout_crossstock_PLAN.md` §0/§1 ("uniform=repeller via degenerate max-ListMLE"; "additive per-day
  constant is rank-invariant ⇒ dead"). Repeller restated as "MSE can't fit a dispersed target with a
  constant"; GO survives.
- `warmq_to_readout_qknorm_PLAN.md` §2 ("parked-mean neutrality"). P2/P3 fixes are loss-independent and
  survive.
- `MASTER_mechanism.md:144,149` ("we default to ListMLE/IC/rank composite" + "CSRankNorm label" — both
  wrong; we default to raw MSE on CSZScoreNorm). Correction *reinforces* the MASTER thesis (§7 note).
- `readout_process_deepdive.md` §0.2/§3c — its own escape hatch ("Raw MSE is the one exception — confirm
  config doesn't use it") **fires**: `mse_normalize=False` ⇒ config DOES use raw MSE.

---

*Sources: workflows `wf_a409eb39-ea7` (forensics+SOTA+design) and `wf_3a3c7582-38f` (MSE-correction:
3 re-derivations × adversarial verify + synthesis). Code-verified loss/label facts: `work_flow.py:107,167,175`,
`quant_moe_model.py:688-703`, `losses.py:56-117`. Anchor preds: `analysis/readout_redesign/archive_g012_baseline`
+ mlruns `2e8e24fe…`.*
