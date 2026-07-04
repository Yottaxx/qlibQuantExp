# From warm-q to a Readout QK-Norm Series — Migration Plan (h-20260619)

> **⚠ LOSS-ASSUMPTION CORRECTION (2026-06-24).** The P1 fix prose "removes ... the parked-mean
> neutrality" assumes a location-invariant loss. The live loss is **plain raw MSE** on a `CSZScoreNorm`
> label (`work_flow.py:107,167,175`), which *penalizes* score common-mode — there is no "parked-mean
> neutrality" at the score; an additive branch ends up near-quiet only because the target mean≈0, not by
> invariance. The **P2 (cold-query/QK-norm) and P3 (V-escape/out-norm) fixes are loss-independent and
> survive unchanged.** Corrected math: `PORTFOLIO_ALIGNED_PLAN.md` §0-§2,§10.

**Date:** 2026-06-19. **Status:** PLAN (no source modified). rms killed (its out-norm-over-D escape
moved to the B axis; `stock_expert_norm` climbed 7→125 with entropy frozen — mechanism read void).
Reference baseline reverts to **warm-q**. This doc plans the evolution warm-q → readout QK-norm.

---

## 1. Why warm-q is the reference (what it proved, what it cost)

warm-q = block-internal stock_expert, `demean=ON`, `no_wd=ON`, fixed `1/√d_head` scale.

**Result (the bar to beat):**
| seed | test RankIC | vs anchor | portfolio IR | MaxDD |
|---|---|---|---|---|
| 42 | 0.0831 | **+0.0015** | 1.41 (<1.54 anchor) | −0.083 (worse) |
| 43 | 0.0790 | **+0.0040** | *crashed (recoverable)* | — |

So warm-q is the **best cross-stock RankIC we have** (+0.0028 mean, same sign both seeds) — but it is a
**pathological** win. Three diagnosed failure modes:

| # | warm-q pathology | root cause |
|---|---|---|
| **P1** | RankIC up but **portfolio IR/MaxDD worse** (RankIC≠IR trap) | gain rides a high-variance channel, not real ranking edge |
| **P2** | **entropy frozen at 1.0000** (attention never sharpened) | cold query: `‖Wq‖→0` under WD ⇒ logits→0 ⇒ uniform |
| **P3** | **`stock_expert_norm`→88** (V-escape) | de-mean creates a B-axis constraint the model games by inflating ‖Wv‖ instead of sharpening |

warm-q's RankIC gain is real but comes **through P3, not through cross-stock attention** (entropy=1.0
proves the attention is uniform = cross-sectional mean). It is the right *reference* and the wrong
*mechanism*. The readout QK-norm series exists to keep the cross-stock idea while removing P1–P3.

---

## 2. The thesis: readout QK-norm removes all three pathologies at once

| warm-q pathology | Fix in readout QK-norm | Why it works |
|---|---|---|
| **P3 V-escape** | **no de-mean** (readout mainpath needs none) | the escape exists *only* because de-mean creates a scale-invariant constraint to game. Mainpath/replacing makes uniform a **repeller** (flat cross-section ⇒ degenerate max-ListMLE), so de-mean is unnecessary ⇒ nothing to game ⇒ P3 structurally impossible |
| **P2 cold query** | **QK-norm + learnable temp** | logit becomes `temp·cos(q,k)` — magnitude-free, so `‖Wq‖→0` can no longer pin uniform; sharpening is driven by one well-conditioned scalar `temp` with direct gradient from the ranking loss. Also **retires the `no_wd` hack** (query magnitude no longer matters) |
| **P1 RankIC≠IR / loss-neutral** | **ungated, in-series, mainpath placement** | removes the router exit (`w_stock→0`) and the parked-mean neutrality; the layer feeds the loss directly, so any gain is load-bearing, not a high-variance side channel |

**Key point:** these are three independent levers for three independent pathologies. QK-norm alone (on
the block) cures only P2 and would leave P1+P3 — that's why we move to the readout *and* add QK-norm
together. The readout is where QK-norm's payoff is unconfounded.

**What carries over from warm-q:** the cross-stock self-attention mechanism itself; the single-day-batch
invariant; the empirical RankIC bar to beat. **What is retired:** de-mean (P3 source), `no_wd` (made
redundant by QK-norm), router-gating (P1 source), fixed `1/√d` scale (P2 source).

---

## 3. The Readout QK-Norm Series (ablation ladder — each rung attributes one change)

All readout arms insert at **post-pool, pre-head** on `h_pooled:[B,D]` (`quant_moe_model.py:542→574`),
default-off / byte-identical when off. The ladder isolates *placement* from *QK-norm* so we know which
change earns the gain.

| Arm | Placement | Gate | Scale | de-mean | Isolates | Predicted |
|---|---|---|---|---|---|---|
| **W0** warm-q (ref) | block, parallel | router | fixed 1/√d | ON | the bar | RankIC+, P1–P3 |
| **R0** readout-plain | readout, mainpath | ungated | fixed 1/√d | OFF | **placement** effect vs W0 | P3 gone; P2 may persist (cold query) |
| **R1** readout-QKnorm ★ | readout, mainpath | ungated | **QK-norm+temp** | OFF | **QK-norm** effect vs R0 | **the bet:** entropy<1, load-bearing, IR-safe |
| **R2** readout-gated (ctrl) | readout, mainpath | cold sigmoid g | QK-norm+temp | OFF | does gating kill it? | freezes near init ⇒ dead (proves P1 diagnosis) |
| **R3** mixer (ctrl) | readout, mainpath | ungated | softmax-free NoGraphMixer | OFF | attention-specific? | collapse-immune; may match R1 |

**Reading the ladder:**
- **R0 vs W0** → does moving to the readout (no demean, ungated) alone help, even with a cold query? If
  R0's entropy is still ~1.0 but R0 is at least IR-safe, placement fixed P1+P3 but not P2 → QK-norm needed.
- **R1 vs R0** → does QK-norm actually warm the attention (entropy drops) and convert it to a real,
  IR-surviving gain? This is the decisive test.
- **R2** → predicted dead; confirms the gate (not the mechanism) is the killer.
- **R3** → if R1 collapses but R3 helps, the signal is real but attention can't capture it → ship the mixer.

**Minimal first cut:** run **R1 (primary)** and **R0 (placement control)** at seed 42, full data, 25ep.
That single pair already tells us placement-vs-QKnorm. Add R2/R3 only if R1's read is ambiguous.

---

## 4. Hard prerequisites before any readout arm (from deep-dive)

1. **Predict must not split a day.** `DailyChunkBatchSampler(max_batch_size=300)` chunks days >300
   (`model_adapter.py:1283,2357`) — cross-stock attention would then attend within a fragment. REQUIRED:
   at predict, set `max_batch_size ≥ max daily count` + assert no split. **Blocker — fix first.**
2. **fp32 cross-stock softmax** over B≈300 (mirror d1pma `quant_moe_model.py:537-540`); AMP-safe.
3. **Eval leave-one-out hook** (mirror `router_override`/`no_stock`, `moe_block.py:130-137`): a
   `readout_stock_override` that uniformizes the layer at eval to measure its marginal rank_ic with no retrain.
4. NaN-label stocks flow through forward (masked only in loss); leave them as keys/values for R1 (simpler),
   revisit if it matters. No BatchNorm anywhere ⇒ no other cross-stock leakage.

---

## 5. R1 architecture (the bet — for reference; implement when approved)

```
u   = h_pooled                                   # [B,D], one trading day
x   = LN_in(u)                                   # pre-LN
q,k = l2norm(Wq x), l2norm(Wk x)                 # QK-norm (cosine, magnitude-free) -> cures P2
A   = softmax( temp * (q @ k^T) )                # temp learnable scalar; fp32 softmax
o   = (A @ (Wv x)) @ Wo
u   = u + o                                      # MANDATORY residual, no gate -> cures P1
h_pooled = u                                     # NO de-mean -> P3 cannot exist
stock_score = head(h_pooled)                     # unchanged
```
n_heads=4, bias=False, permutation-equivariant, ~16K params. `temp` init so initial logits ≈ today's
effective τ (mild), then let it learn up. No de-mean, no out-norm, no `no_wd` needed.

---

## 6. Protocol & kill criteria

**Stage 1 (proxy):** R1 + R0, seed 42, full data, 25ep, serial, `kernels=1`, detached nohup (never a
subagent), MAX_RETRY=1. Early read ~epoch 8–10: readout-entropy trajectory, `‖Wv‖‖Wo‖` O(1) (must stay,
by construction), valid_rank_ic vs warm-q + anchor.

**Promote to Stage 2 iff R1:** entropy<0.999 & stable, norm O(1), valid_rank_ic ≥ warm-q **and** ≥ anchor.

**Stage 2 (kill-check):** n≥6 fresh seeds (n=3 unreliable), paired ΔRankIC **and Δportfolio-IR** vs
anchor, walk-forward + stress, eval leave-one-out.

**KILL (any one):** (1) RankIC not above anchor at n≥6; (2) leave-one-out uniformize leaves rank_ic
unchanged ⇒ no-op; (3) entropy pinned 1.0 throughout (and if R3 mixer also flat ⇒ no cross-stock signal
exists ⇒ close the axis cleanly); (4) RankIC gain dies in IR under stress (the warm-q trap repeating);
(5) `‖Wv‖‖Wo‖` inflation with any gain ⇒ escape artifact (should be impossible in R1 — a tripwire).

---

## 7. Immediate next actions

1. **Solidify the warm-q reference:** recover warm-q **seed 43 portfolio** from its saved `pred.pkl`
   (backtest only, no retrain, ~minutes, crash-proof now with `kernels=1`) → completes the n=2 warm-q
   IR/MaxDD picture the readout series must beat.
2. **Implement prerequisite §4.1** (predict day-split guard) + the `use_readout_stock_attn` flag,
   `CrossStockBlock` (R1), and R0 toggle. Default-off byte-identical. Unit-test forward + scale-invariance
   + entropy behavior on synthetic data.
3. **Launch Stage-1** R1+R0 seed 42 once implemented.

GPU is free now. Step 1 is cheap and strengthens the baseline; steps 2–3 are the build.
