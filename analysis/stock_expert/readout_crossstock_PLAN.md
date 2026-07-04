# Readout Cross-Stock Attention — Revised Plan (h-20260617b)

> **⚠ LOSS-ASSUMPTION CORRECTION (2026-06-24).** Passages framing uniform as a "repeller via degenerate
> max-ListMLE" and "additive per-day constant is rank-invariant ⇒ dead" assume an order-only/location-
> invariant loss. The live loss is **plain raw MSE** on a `CSZScoreNorm` label (`work_flow.py:107,167,175`;
> `cs_ic`/`listmle` monitor-only). Restate the repeller as **"MSE cannot fit a dispersed CSZScore target
> (var≈1) with a per-day constant ⇒ uniform-replacing gives `L≥var(y)≈1`, strictly dominated."** The **GO
> recommendation survives**; only the mechanism prose changes. Corrected math: `PORTFOLIO_ALIGNED_PLAN.md`
> §0-§2,§10.

**Date:** 2026-06-19. **Status:** DESIGN (no source modified). Synthesizes three investigations:
`readout_stock_attention_analysis.md` (loss-neutrality gradient argument), `readout_process_deepdive.md`
(gradient-level forward map + sampler/eval facts), `sota_crossstock_methods.md` (SOTA survey).
Supersedes the GO recommendation in `readout_stock_attention_analysis.md §4` by adding an anti-collapse
mechanism and a softmax-free control.

---

## 0. One-paragraph thesis

The block-internal stock_expert collapsed to uniform attention because of **placement** (router-gated,
parallel, optional ⇒ zero-cost exit + parked-mean loss-neutrality), not because cross-stock structure
is absent. The readout is the opposite site: in-series, no router, feeds the cross-sectional ranking
loss directly. SOTA agrees — **every** working cross-stock method (MASTER, DTML, RSR, THGNN, AD-GAT,
StockMixer, HIST) places stock mixing **main-path / ungated / mandatory**; the *only* router-gated-parallel
design in the literature (TRA) collapses without an auxiliary OT loss — our exact failure class. So we
place a **mandatory, ungated cross-stock layer at the readout**, add the **single most targeted
anti-collapse lever (QK-norm + learnable temperature)**, take **no de-mean** (so the V-escape has nothing
to game), and **A/B it against a softmax-free StockMixer** that is structurally collapse-immune.

---

## 1. What changed vs the first plan

| Aspect | First plan (`..._analysis.md §4`) | Revised plan (this doc) | Why |
|---|---|---|---|
| Placement | ungated main-path, post-pool `[B,D]` | **same** | SOTA consensus, not a bet |
| Collapse defense | placement only (uniform = repeller) | placement **+ QK-norm + learnable temperature** | placement removes the *exit*; it does NOT guarantee the *query warms*. QK-norm/temp is the targeted entropy lever (σReparam, arXiv:2303.06296) |
| Residual | "residual+LN, avoid scalar gate" | **pre-LN vanilla residual, no scalar gate, no ReZero** for a single block | ReZero/LayerScale only pay off in deep stacks; cold sigmoid gate is the freeze trap |
| De-mean | not needed | **explicitly OFF** | F2 (V-escape) only exists with a scale-invariant constraint to game; ungated/replacing has none |
| Market conditioning | "optional MASTER gate" | reuse `regime:[B,D]` **multiplicatively (FiLM), never additive**; Stage-2 only | additive per-day constant is rank-invariant ⇒ dead (deep-dive §2,§3) |
| Graph prior | not discussed | **none** for csi300 | MASTER (dense, market-conditioned) beats graph-DTML +31%/+46% IC/RankIC |
| Control arms | mainpath vs gated vs off | **+ NoGraphMixer (StockMixer) arm** | softmax-free mixer matches/beats attention and cannot collapse — the honest "does attention specifically help" control |
| Eval correctness | "single-day batch guaranteed" | **must guarantee predict does not split a day** (chunk sampler) | deep-dive §4: `DailyChunkBatchSampler(max_batch_size=300)` splits days >300 |

---

## 2. Architecture (recommended: arm M1)

Insertion locus: **post-pool, pre-head** — on `h_pooled : [B, D]` (`quant_moe_model.py:542` d1pma /
`:568` control), before `self.head` (`:574`). One token per stock, sequence = B stocks of one trading day.

```
# h_pooled : [B, D]   (B = one day's cross-section)
u   = h_pooled
# --- cross-stock self-attention, pre-LN, mandatory residual ---
x   = LN_in(u)                                  # pre-LN
q,k = Wq x, Wk x                                # [B, H, d_head]
q,k = l2norm(q, dim=-1), l2norm(k, dim=-1)      # QK-norm (cosine attention)
A   = softmax( (q @ k^T) * temp )               # temp = learnable scalar (or per-head), fp32 softmax
o   = (A @ (Wv x)) Wo                            # [B, D]
u   = u + o                                      # MANDATORY residual (no gate)
# optional FFN (Stage-1b): u = u + FFN(LN_ffn(u))
h_pooled = u
stock_score = head(h_pooled)                    # unchanged
```

Design specifics (SOTA-grounded):
- **QK-norm + learnable temperature** (`temp` init so initial logits are mild, e.g. scale≈ same effective
  τ as today's 1/√d_head, then let it learn up). This decouples entropy from `‖Wq‖` magnitude: the cold-query
  collapse (`‖Wq‖→0 ⇒ uniform`) cannot happen because cosine logits are magnitude-free; sharpening is driven
  by the learnable `temp`, which has direct gradient from the ranking loss. **This is the fix for "entropy
  stuck at 1.0" that placement alone does not provide.**
- **No de-mean, no out-norm** in M1 (ungated/replacing has no scale-invariant constraint ⇒ no V-escape).
- **fp32 softmax** over B≈300 (mirror d1pma `quant_moe_model.py:537-540`); AMP-safe.
- **n_heads = 4** (match model), single layer, `bias=False` on projections, permutation-equivariant (no
  positional/order term — stocks are an unordered set).
- Param cost ≈ 16K (1 MHA) + 2 LN (256) + 1 temp; +~8K if FFN. Negligible.

---

## 3. Control / falsifier arms

| Arm | What | Predicted | Purpose |
|---|---|---|---|
| **M0 off** | baseline anchor (`use_readout_stock_attn=false`) | — | reference |
| **M1 mainpath-attn** | §2 (ungated, QK-norm+temp, pre-LN, no de-mean) | entropy<1 sharpen OR honest-but-mandatory | **the bet** |
| **M2 gated control** | M1 body but `u + sigmoid(g)·o`, cold g init | freezes near init ⇒ dead (reproduces stock_expert) | confirms gate-placement is the killer, not the mechanism |
| **M3 NoGraphMixer** | softmax-free StockMixer: `u + B→market→B` low-rank MLP mix, mandatory | collapse-immune; may match/beat M1 | isolates "does attention *specifically* help vs any cross-stock mixing" |

M2 and M3 are the scientific controls: M2 proves the placement/gating diagnosis; M3 guards against the
SOTA finding that a softmax-free mixer can match attention while being immune to both our failure modes.

---

## 4. Hard implementation requirements (from deep-dive)

1. **Predict must not split a day.** `DailyChunkBatchSampler(max_batch_size=self.batch_size=300)`
   (`model_adapter.py:1283,2357`) chunks days >300. REQUIRED: either (a) at predict, set
   `max_batch_size = max daily count` (assert no split), or (b) add a per-day no-chunk predict path when
   `use_readout_stock_attn`. Add a runtime assert that each eval batch == one full day. **Blocker if unfixed.**
2. **Training cross-section is exactly B=300** via `FixedDailyBatchSampler` (down-sample >300 without
   replacement; up-sample <300 WITH replacement ⇒ duplicate stocks). Duplicates in cross-stock attention are
   benign (a stock attends to its copies = mild self-weighting) but should be noted; the random 300-subset
   per epoch acts like stock-dropout (fine, even helpful for robustness).
3. **NaN-label stocks flow through the forward** (masked only in loss, `:597-602`). They participate as
   keys/values in cross-stock attention. Decide: mask them out of attention (cleaner) or leave (they carry
   features, only the label is NaN — leaving is acceptable and simpler). Recommend leave for M1; revisit if
   it matters.
4. **fp32 cross-stock softmax** (AMP); **no BatchNorm anywhere** so no other cross-stock leakage path.
5. **Eval leave-one-out hook:** mirror `router_override`/`no_stock` (`moe_block.py:130-137`) with a
   `readout_stock_override` that uniformizes/zeros the cross-stock layer at eval, to measure its marginal
   rank_ic contribution without retraining.

---

## 5. Config / insertion surface (described, NOT implemented)

- `model_configuration.py` (next to stock_expert flags `:81-95`): `use_readout_stock_attn: bool=False`,
  `readout_stock_attn_arm: str="mainpath"` (allow-list `{"mainpath","gated","mixer"}`),
  `readout_stock_attn_qknorm: bool=True`, `readout_stock_attn_ffn: bool=False`. Default off ⇒ byte-identical.
- New module `module/architecture/cross_stock_block.py`: `CrossStockBlock` (pre-LN MHA, QK-norm, learnable
  temp, mandatory residual, optional FFN) + `NoGraphMixer`. Permutation-equivariant, fp32 softmax, bias-free.
- `quant_moe_model.py`: build in `__init__` under the flag; apply after `h_pooled` (`:542`/`:568`) before
  head (`:574`): `h_pooled = self.readout_stock_block(h_pooled.unsqueeze(0)).squeeze(0)` (seq=B).
- `work_flow.py`: override flows through `_deep_update` already (no whitelist gate); add the 4 keys to the
  provenance list (~`:307`) for clean run_conf_resolved.
- Reuse `kernels=1` crash fix (already in). Launch via detached `nohup` orchestrator, `MAX_RETRY=1`, serial
  seeds — NEVER inside a subagent.

---

## 6. Experiment protocol & kill criteria

**Stage 1 (proxy, full data, fast read):** seed 42, 25ep, arms M0/M1/M2/M3 (M0 is the existing anchor, no
rerun needed). Early read by ~epoch 8–10: `readout` entropy_norm trajectory, `‖Wv‖‖Wo‖` O(1), valid_rank_ic
vs anchor. ~3–4h/arm serial (kernels=1).

**Promotion to Stage 2 only if M1 shows:** entropy_norm drops <0.999 and stable, norm O(1), valid_rank_ic ≥
anchor by a seed-stable margin.

**Stage 2 (kill-check):** n≥6 fresh seeds (n=3 unreliable per durable finding), paired ΔRankIC **and**
Δportfolio-IR vs anchor, `/quant-walk-forward` + `/quant-stress`, eval leave-one-out.

**KILL (any one ⇒ no promotion):**
1. valid_rank_ic not above anchor by a seed-stable margin at n≥6.
2. Leave-one-out uniformize leaves rank_ic unchanged ⇒ load-neutral no-op.
3. entropy_norm pinned at 1.0 throughout ⇒ uniform collapse (and if M3 mixer ALSO flat ⇒ no cross-stock
   signal exists in csi300/Alpha158 beyond factor+time; clean negative, close the axis).
4. RankIC gain fails to survive into IR under walk-forward/stress ⇒ RankIC≠IR trap.
5. `‖Wv‖‖Wo‖` inflation co-occurring with any gain ⇒ V-escape artifact (should be impossible in M1 by
   construction; a tripwire).

**Decisive 2×2 read (M1 vs M3):**
- M1 sharpens & helps, M3 helps → cross-stock signal real; attention captures it → promote M1.
- M1 collapses but M3 helps → signal real but attention can't capture it → ship the mixer (StockMixer-style).
- both flat → no cross-stock signal at readout → close the axis cleanly (a real negative, distinct from the
  decided TIME-readout and block-stock_expert cards).
- M1 helps, M3 flat → attention-specific structure (rare; verify hard for overfit).

---

## 7. GO / NO-GO

**GO** for Stage-1 proxy with arms M0/M1/M2/M3 once the current rms run frees the GPU. The placement is SOTA
consensus, the anti-collapse lever (QK-norm+temp) directly targets the entropy-stuck failure that placement
alone won't fix, the no-de-mean design is V-escape-immune by construction, and the M3 mixer control makes the
result interpretable whichever way it lands. Hard requirement before launch: resolve the predict day-split
(§4.1). This is a genuinely new axis, not a re-run of any decided card.
