# Cross-Stock Attention at the Readout — Architecture Analysis

> **⚠ LOSS-ASSUMPTION CORRECTION (2026-06-24).** §2c, §3b, §3d below argue "uniform cross-stock attention
> is a loss-neutral parking spot" / "the final loss is invariant to a per-day additive constant (and to
> per-day affine `s→αs+β`)". **This is FALSE for the live objective.** Code-verified: `main_loss="mse"`
> (`work_flow.py:167`), `mse_normalize=False` (`:175`) ⇒ the gradient is **plain `F.mse_loss(p,y)`** on a
> **`CSZScoreNorm` robust** label (`:107`); `cs_ic`/`listmle`/`cs_mse(normalize=True)` are **monitor-only**
> (`quant_moe_model.py:701-703`). Plain MSE is shift- and scale-SENSITIVE: it *penalizes* score common-mode
> (`∂L/∂c=2·mean(p)`) and anchors dispersion (`std(p)*≈IC≈0.23`). So the §3b loss-neutrality gradient
> argument does not hold at the score level. **The §3-§4 readout recommendation still SURVIVES** — for the
> **loss-independent** reason (the block-internal stock_expert dies via the *router exit* `w_stock→0`, not
> via loss-neutrality; the readout has no router exit). See `PORTFOLIO_ALIGNED_PLAN.md` §0-§2,§10 for the
> corrected math.

Read-only analysis. No source edited, no training run, no GPU touched. All claims are cited to
`file:line` against the working tree on branch `pool-readout-forensics`.

**Verified dimensional facts** (from `work_flow.py:130-138,193` and `module/utils/model_configuration.py:17-23`):
`d_model = 64`, `n_heads = 4` (`d_head = 16`), `n_layers = 2`, `N = num_alphas = 158`,
`T = context_len = 8` (data-driven from `step_len=8`, `work_flow.py:80`), daily cross-section
`B = batch_size = 300`. Baseline anchor flags (`scripts/baseline_g012_scale05.env.sh`):
`temporal_readout="d1pma"`, `temporal_readout_gate_init=-2.0` (so the readout gate
`g = sigmoid(-2) ≈ 0.119`), `time_tau_mlp_out_scale=0.5`, `use_stock_expert` **off** at the anchor.

Your description of the architecture is correct in all material respects. One refinement: the
readout's d1pma attention is over **T only**, not over N — the N axis is collapsed by a plain
`mean(dim=2)` *before* the temporal query is even computed. That detail turns out to matter for the
cross-stock question, so it is flagged in §1 and §3.

---

## 1. Readout, end-to-end (factual map)

Block output entering the readout is `h : [B, T, N, D]` after `self.final_norm`
(`module/quant_moe_model.py:490`). The default (no `temporal_readout`) path and the active `d1pma`
path differ only in how T is collapsed; both then run the **same factor pool + linear head**.

### 1a. Default path (control, `temporal_readout==""`)
`module/quant_moe_model.py:565-568`:
```
h_last   = h[:, -1, :, :]            # [B, N, D]  -- last time step only
h_pooled, w = self.factor_pooling(h_last)   # [B, D], [B, N]
```
Then `stock_score = self.head(h_pooled).squeeze(-1)  # [B]` (`:574`), `self.head = nn.Linear(D, 1)`
(`:114`).

### 1b. Active path (`d1pma`) — what the anchor actually runs
`module/quant_moe_model.py:534-549`:
```
h_time = h.mean(dim=2)                                       # [B,T,D]   mean over N (158 factors)
logits = (h_time @ tr_q) / sqrt(D) + tr_b_t                 # [B,T]     fp32; tr_q:[D], tr_b_t:[T]
a      = softmax(logits, dim=1)                              # [B,T]     temporal weights, per-stock
attn   = einsum("bt,btnd->bnd", a, h)                       # [B,N,D]   weighted avg over T
g      = sigmoid(tr_gate)                                    # scalar    init -2 -> g≈0.119
z_T    = (1-g)*h[:, -1, :, :] + g*attn                       # [B,N,D]   gated residual w/ last step
h_pooled, w = self.factor_pooling(z_T)                       # [B,D],[B,N]
stock_score = self.head(h_pooled).squeeze(-1)                # [B]
```
Definitions of the d1pma parameters: `tr_q = nn.Parameter(randn(D)*0.02)` (single learnable query),
`tr_b_t = nn.Parameter(zeros(T))` (additive per-step bias, "uniform bias = 0" at init),
`tr_gate = nn.Parameter(full((1,), gate_init))` (`module/quant_moe_model.py:166-168`).

**What "d1pma" computes.** It is a **d**egenerate / **1**-query **p**ooled-**m**ean **a**ttention over
the **time** axis: a single global query `tr_q` scores each of the T time steps (after averaging out
the N factor axis), softmax over T gives per-stock temporal weights `a:[B,T]`, and `attn` is the
`a`-weighted average of the per-(N,D) representations across T. The "scale/temperature" is the
standard `1/sqrt(D)=1/8` dot-product scale (`:537`). It is then **gated-residual blended** with the
last-step control via scalar `g`. At `gate_init=-2 ⇒ g≈0.12`, the readout is ≈88% last-step + ≈12%
temporal-attention pooling.

### 1c. Factor pool (shared by both paths) — `module/architecture/attention_pooling.py`
`AdaptivePooling.forward` (`:116-131`): `pooled = alpha * attn_pool(x) + (1-alpha)*mean(x, dim=1)`,
`alpha = pooling_alpha = 0.7` (`work_flow.py`/config `:69`). `AttentionPooling` (`:52-97`) is a
single learnable query `query:[1,1,D]` attending over the **N=158 factor tokens** via
`nn.MultiheadAttention` (`pool_n_heads=1`), followed by `LayerNorm` (`:86`). So the pool collapses
the **N axis**, producing `[B, D]`.

### 1d. Is there attention "already" at the readout?
Yes — **two** attention operations already live in the readout, but **neither is cross-stock**:
1. **d1pma temporal attention** — query over the **T** axis (`:537-539`).
2. **factor-pool attention** — learnable query over the **N** axis (`attention_pooling.py:71-77`).

Both attend along axes that are *internal to a single stock's representation*. **There is no
attention across the B (stock) axis anywhere in the readout.** The stock axis only ever enters the
readout as the independent batch dimension; every readout op is applied per-stock and is
permutation-equivariant over B. The final `head` is a per-stock `Linear(D,1)`. So the model's entire
readout is **cross-sectionally separable**: stock `i`'s score depends only on stock `i`'s features.

This is the structural gap the user is probing.

---

## 2. Current n-stock (stock_expert) attention (factual map)

### 2a. Mechanism and axis
`module/architecture/moe_block.py:220-231`. Enabled by `use_stock_expert`
(`model_configuration.py:81`). It is a third `ParallelAttention` peer (`moe_block.py:53-54`). The
cross-stock rearrange:
```
h_stock = rearrange(x, "b t n d -> (t n) b d")   # batch=(T*N)=1264 groups, seq=B=300, dim=D
out_stock = self.stock_expert(h_stock, None)     # MHA self-attn over the B axis
out_stock = rearrange(out_stock, "(t n) b d -> b t n d")
```
So for each of the `T*N = 8*158 = 1264` (timestep, factor) slots, it runs an MHA whose **sequence is
the B≈300 stocks**: each stock attends to all other stocks in the same day's cross-section. `bias=None`
(stocks unordered, permutation-equivariant — `moe_block.py:217-218`).

### 2b. MHA params / temperature
`ParallelAttention` wraps `nn.MultiheadAttention(embed_dim=64, num_heads=4, bias=False,
batch_first=True)` (`parallel_attention.py:29-35`). Scaled dot-product uses PyTorch's built-in
`1/sqrt(d_head) = 1/sqrt(16) = 1/4` scale. **τ = √d_head = 4** in the sense that logits are
`q·k / 4`; there is no separately learnable temperature. (The factor pool and d1pma use `1/sqrt(D)`
since their query lives in full D, not per-head — a minor inconsistency, not a bug.)

### 2c. Placement: router-gated parallel, loss-NEUTRAL
The three experts are fused as a **router-weighted sum** (`moe_block.py:289-291`):
```
fused = w_time*out_time + w_factor*out_factor + w_stock*out_stock
x = residual + fused
```
with `w_* = softmax(router(...))[:, k]` (`:113,161-163`). The stock expert is therefore **optional
and parallel**: the router can drive `w_stock → 0`, and the time/factor experts plus the residual
fully carry the model. This is the structural reason it is loss-neutral *as placed*.

**Why uniform cross-stock attention is loss-neutral inside a block.** If `softmax` over the B stocks
is uniform (`a_ij = 1/B`), then `out_stock[:, t, n, :]` is the same vector for every stock i —
namely the cross-sectional mean of `V` over the day. Adding a **per-day constant** to every stock's
representation, then later pooling and scoring with a per-stock-identical head, contributes the same
scalar to every stock's score. The training losses are all **cross-sectional and
location-invariant**: `cs_ic_loss` centers pred (`losses.py:20`), `listmle_loss` subtracts
`s.max()` and depends only on the ranking (`losses.py:101,108`), `cs_mse_loss(normalize)` centers.
A per-day additive constant cancels in all of them ⇒ **zero gradient pressure to deviate from
uniform** ⇒ uniform attention is a free parking spot. Documented: `entropy_norm = 1.0000`
(`diag["stock_attn_entropy_norm"]`, `moe_block.py:340`).

### 2d. The V-escape pathology and the out-norm fix
With `stock_expert_demean` (`moe_block.py:277-278`) the per-day mean is subtracted from the output,
which *should* kill the loss-neutral parking spot (deviating from uniform becomes the only way to
move the loss). But the warm-q experiments found the model satisfied the de-mean **without
sharpening attention**: it inflated `‖V‖` (i.e. `‖Wv‖·‖Wo‖ → 88`) so that tiny residual non-uniform
attention weights still produced a non-trivial rank-changing output — a magnitude escape, not a
real attention signal. The RankIC bump from warm-q did not survive into portfolio IR (RankIC≠IR
trap; durable finding in memory). The `h-20260617` fix is `stock_expert_out_norm`
(`moe_block.py:238-272`): a **scale-invariant** per-token norm (`rms`/`unit`/`ln`, parameter-free)
applied to `out_stock` **before** de-mean, removing magnitude as a degree of freedom so the only way
to survive de-mean is to *actually sharpen* attention (Path A) or for the router to abandon the
expert (honest death). `ln_affine` is included only to *demonstrate* that a learnable gain reopens
the escape (`:255-260`). Net documented status: stock_expert is a settled, mostly-dead axis in its
**block-internal, router-gated** placement.

---

## 3. Applicability at the readout — rigorous analysis

The crux: the stock_expert is loss-neutral **because** of where it sits (optional, parallel,
parked-mean-absorbed). The readout is the *opposite* kind of site. Does moving the mechanism there
change its loss-bearing status? The answer hinges on a gradient argument.

### 3a. Axis / shape compatibility — where it inserts and what attends to what
The natural insertion point is **after T-collapse and N-collapse**, on the per-stock vector. After
the d1pma temporal collapse we have `z_T : [B, N, D]`; after the factor pool we have
`h_pooled : [B, D]` (`module/quant_moe_model.py:542`). The cleanest cross-stock site is **on
`h_pooled : [B, D]`** — a single token per stock, B=300 tokens, attend stock↔stock, then feed
`head`. This is shape-trivial: it is exactly an MHA with sequence length B, dim D, **no `(t n)`
batching at all** (contrast §2a's 1264 parallel groups). One MHA call of size [1, B, D].

Two viable sub-positions:
- **post-pool, pre-head** (`h_pooled:[B,D]` → cross-stock MHA → `[B,D]` → `head`). Cheapest,
  cleanest. **Recommended locus.**
- **pre-pool** (`z_T:[B,N,D]`, attend over B for each of N tokens). This re-introduces N parallel
  groups (158 of them) and mixes stock-interaction with factor-pooling — more params, more compute,
  and harder to interpret. No upside over post-pool. Reject.

What attends over what at the recommended locus: each stock's pooled representation `h_pooled[i]`
is a query; keys/values are the pooled representations of **all B stocks in the same trading day**.
The output replaces (or residually augments) `h_pooled[i]` before the final linear scoring. This is
precisely the MASTER (arXiv:2312.15235) intra-day cross-stock interaction, placed at the
score-forming layer.

**Validity invariant** (must hold): the B axis must be a *single day's* cross-section, else
cross-stock attention mixes stocks across dates (leakage / nonsense). The codebase already
guarantees this — `FixedDailyBatchSampler`/`DailyChunkBatchSampler` produce single-day batches in
train and per-day predict/eval (`model_adapter.py:1250-1292`; same invariant the stock_expert relies
on, `moe_block.py:216-218`). So the invariant is **free** here.

### 3b. Loss-neutrality at the readout — the gradient argument (centerpiece)

Set up the readout cross-stock layer abstractly. Let the per-stock pre-attention vectors be
`u_i = h_pooled[i] ∈ R^D`, `i=1..B`. A cross-stock self-attention produces
`o_i = Σ_j a_ij (W_v u_j)`, `a_ij = softmax_j( (W_q u_i)·(W_k u_j)/√d )`. The score is
`s_i = w·φ(o_i) + b` for the head `w` (here `φ = identity`, possibly with a residual `u_i`+pool;
keep φ=id for the derivation). Training loss `L` is a **cross-sectional ranking loss** over the day:
the main loss is `listmle` (`work_flow`/config; loss_weights listmle=1, mse=1, ic=1, rank=0,
`model_configuration.py:179-185`, `main_loss` selected at `quant_moe_model.py:639-651`). All three
active components are **invariant to a per-day additive constant** on `s` (§2c).

**Case U (uniform attention), readout:** `a_ij = 1/B ⇒ o_i = W_v ū`, where `ū = (1/B)Σ_j u_j`. Then
`o_i` is the **same vector for all i**, so `s_i = w·W_v ū + b` is a **per-day constant** added to
(or, if it *replaces* `u_i`, *equal to*) every stock's score.

Here the two placement variants diverge sharply:

- **Main-path / replacing (ungated, MASTER-style):** if the cross-stock output *replaces* `u_i`
  (i.e. `s_i = w·o_i + b` with no residual), then under uniform attention **every `s_i` is
  identical** ⇒ the cross-section is a flat line ⇒ IC = 0/0 (degenerate), ListMLE = log(B!)/... at
  its **maximum** (no ordering information). This is the **worst** possible loss, not a neutral
  parking spot. The gradient `∂L/∂a` is large and points **away** from uniform: any sharpening that
  makes `o_i` stock-dependent immediately creates score dispersion and reduces ListMLE. **Uniform is
  a repeller, not an attractor.** This is the load-bearing regime.

- **Gated-residual (`s_i = w·(u_i + g·o_i) + b`, g small or learned):** under uniform attention the
  added term `g·W_v ū` is a per-day constant; it cancels in the ranking loss exactly as in §2c.
  So the *uniform component* is loss-neutral — but the **residual `u_i` still carries the full
  cross-sectional signal**, so the model is not degenerate; it simply ignores the uniform part. The
  gradient w.r.t. the attention logits is **non-zero only through the non-uniform (rank-changing)
  part** of `o_i`. This is the same structure as the block-internal de-mean: the constant is free,
  deviation is what's rewarded — but unlike the block-internal case, **the residual path is the
  identity of the readout itself, not an optional sibling**, so there is no router that can zero the
  branch and make the whole expert vanish. The expert cannot "honestly die" by being routed to 0;
  it can only contribute the relative component or sit at g≈small.

**Decisive contrast with the block-internal stock_expert.** Inside a block the stock expert is
loss-neutral for **two compounding reasons**: (i) the uniform output is a parked per-day mean
(absorbed by location-invariant loss), **and** (ii) it is one of three router-weighted parallel
experts, so `w_stock→0` removes it entirely while time/factor+residual carry the model. Reason (ii)
is the killer: it gives the optimizer a zero-cost exit. **At the readout, reason (ii) does not
exist.** There is no parallel sibling that subsumes the function and no router gate that can delete
the layer; the cross-stock layer is *in series on the only path to the head*. If it is placed
**ungated/main-path**, reason (i) also disappears (uniform ⇒ degenerate ⇒ max loss). So:

> **A main-path, ungated, in-series cross-stock attention at the readout is loss-BEARING, not
> loss-neutral.** This is the structural reason MASTER's placement works and ours (router-gated,
> parallel, optional) collapsed. The readout is the most "feeds-loss-directly, no-exit" site in the
> model, which is exactly what defeats the uniform-collapse attractor.

The caveat: a *gated-residual* readout cross-stock layer with a cold/`g≈0` init **re-introduces a
soft exit** (set g→0, recover the separable model) and partially re-opens the neutrality door —
gradient pressure flows only through the relative part, and a scalar gate can freeze near its init
(documented: "scalar sigmoid gates freeze at init", memory + `temporal_readout_gate_init` lesson).
That is the trap to avoid. **The design recommendation in §4 follows directly from this.**

### 3c. V-escape risk at the readout
The V-escape (§2d) is a property of *de-mean + free magnitude*, not of the block. If the readout
layer uses **de-mean + scale-free output**, the same escape exists: the model can satisfy a de-mean
constraint by inflating `‖W_v‖`. **However, the readout does not need de-mean at all** if it is
main-path/replacing: there is no parked mean to subtract because uniform is already a repeller
(§3b). De-mean is only needed when a parked mean would otherwise be loss-neutral, i.e. in the
gated-residual / parallel cases. So the cleanest design (main-path, ungated, no de-mean) is
**structurally immune** to the V-escape: there is no scale-invariant constraint to game. If a
residual/gated variant is used, then yes — reuse the existing `out_norm` (`rms`/`unit`, parameter-free,
`moe_block.py:241-249`) on the cross-stock output before the residual add, exactly as the
block-internal fix does, and **never** a learnable-affine norm (`ln_affine` reopens it, `:255-260`).

### 3d. Rank-invariance trap
The final loss is invariant to **per-day affine** transforms of the score `s → α·s + β` (α>0): IC is
correlation (scale+shift invariant), ListMLE depends only on order (shift-invariant via `s.max()`
subtraction `losses.py:108`; and monotone-invariant). So the question is whether readout cross-stock
attention adds **rank-CHANGING** information or only an affine-absorbable transform.

- **Uniform attention** adds a per-day constant β ⇒ pure shift ⇒ rank-invariant ⇒ **no alpha**
  (this is the dead-axis outcome). ✔ consistent with the stock_expert collapse.
- **A per-day global scaling** (e.g. every stock multiplied by the same scalar) ⇒ rank-invariant ⇒
  no alpha.
- **Non-uniform, stock-specific attention** ⇒ `o_i` is a *different* mixture for each stock
  depending on *which other stocks it attends to* ⇒ the map `u_i → s_i` is **non-affine and
  stock-permutation-sensitive** ⇒ it **can change the cross-sectional ranking**. This is genuinely
  new information **only if** the attention is (a) non-uniform and (b) the mixture it forms is not
  itself a monotone function of the pre-existing per-stock signal.

The trap, stated precisely: cross-stock attention earns its keep **iff** it produces score
adjustments that are *not* a per-day monotone rescaling of `u`-derived scores. Whether it does is an
**empirical** question (§5) — the architecture *permits* rank-changing information (unlike the
uniform/affine degenerate modes), but does not *guarantee* the data rewards it. The mechanism is
*capable* of carrying alpha at the readout (this is strictly more than the block-internal placement,
which is structurally biased toward the loss-neutral mode); whether csi300/Alpha158/t+5 actually
contains exploitable cross-stock structure beyond what factor+time already extract is the bet.

---

## 4. Design options

All options insert at the **post-pool, pre-head** locus (`h_pooled:[B,D]` → cross-stock → `head`),
i.e. between `module/quant_moe_model.py:542/568` and `:574`. They would be flag-guarded
(`use_readout_stock_attn` + variant), default-off, baseline-byte-identical when off. Param costs
assume one MHA(D=64, H heads, bias=False): in-proj `3·D² = 12,288` + out-proj `D² = 4,096` ≈ **16K
params** per layer (≈ the size of one existing expert; negligible vs the model).

### (a) Main-path cross-stock self-attention, ungated, in-series (MASTER-style) — RECOMMENDED
```
o = CrossStockMHA(h_pooled)         # [B,D], seq=B, attend stock<->stock
h_pooled = LayerNorm(h_pooled + o)  # mandatory residual + norm (transformer block), NOT a learnable scalar gate
stock_score = head(h_pooled)
```
- **Inserts:** post-pool, pre-head.
- **Attends:** each stock's pooled rep over all B same-day stocks.
- **Loss-neutrality:** **load-bearing.** Residual is a *fixed* `+o` (no scalar gate), so there is no
  soft exit; uniform attention contributes a parked constant that the loss ignores, but the layer
  cannot collapse the model (residual carries `u`), and any sharpening is immediately rewarded
  (§3b). Use a transformer-style residual+LN (the residual is the identity-start, giving stable
  from-scratch training) but **avoid a learnable scalar `g`** — that is the freeze-at-init trap.
- **V-escape:** if you keep it pure residual+LN (no de-mean, no scale-invariant constraint), **no
  escape exists** — there is nothing to game. The LN on `(u+o)` bounds magnitude jointly, which is
  benign.
- **Param cost:** ~16K (1 MHA) + LN (128). Optionally a 2-layer FFN to make it a full block (~+8K).

### (b) Gated-residual cross-stock (scalar gate, d1pma-style)
```
o = CrossStockMHA(h_pooled); g = sigmoid(readout_stock_gate)   # init -> g small
h_pooled = u + g*o
```
- **Loss-neutrality:** **partially re-opened.** `g→0` is a soft exit back to the separable model;
  uniform `o` is loss-neutral; gradient flows only through the relative part. A scalar sigmoid gate
  **freezes near its init** (durable finding) — with a cold init it likely never rises, reproducing
  the dead-axis outcome. This is the **same failure mode** that killed the stock_expert, transplanted.
- **V-escape:** present if combined with de-mean; needs `out_norm` (rms/unit).
- **Param cost:** ~16K + 1 gate scalar.
- **Verdict:** this is the *control to compare against*, not the bet. Including it as an A/B arm is
  useful precisely to show whether the gate stays dead (predicted) vs the ungated arm (a) moving.

### (c) Cross-attention from a learned query (pooling-style, set-summary)
```
q = learned_query.expand(B)         # or per-stock query from u_i
summary = MHA(query=q, key=h_pooled, value=h_pooled)   # [B or 1, D]
```
This forms a **market-summary token** each stock reads from. If the query is global (1 token), it
produces a per-day vector broadcast to all stocks — that is again a **per-day constant** at the head
⇒ rank-invariant ⇒ dead (same trap as a "regime" context, which the model already has via
`regime_encoder`). If the query is per-stock (`q_i = u_i`), it degenerates to option (a). So (c)
either reduces to (a) or to a known-dead per-day-constant. **Reject** as a standalone.

### Recommendation: **(a)**, ungated main-path residual cross-stock self-attention, post-pool.
Justification: it is the *only* variant whose loss-neutrality analysis (§3b) is unambiguously
load-bearing, it is structurally immune to the V-escape, it is the literal MASTER placement that the
durable findings flagged as the promising hypothesis, and it sidesteps the two mechanisms that
killed the block-internal stock_expert (router exit + scalar-gate freeze). Keep it parameter-light
(one MHA + residual LN, optionally one FFN), `n_heads=4` to match the rest of the model, single
layer to start.

---

## 5. Falsifiable read — diagnostics and kill criteria

The deep lesson from the pool/stock-expert forensics: **weight statistics are blind to usefulness**
(`moe_block.py:330` "Descriptors only — NEVER verdicts"). So the primary verdict must be a
**held-out portfolio/rank metric on fresh seeds**, with mechanism diagnostics as *supporting*
(necessary-not-sufficient) evidence. The RankIC≠IR trap (warm-q bumped RankIC but died in IR) means
**RankIC alone cannot promote.**

**Distinguishing "carries alpha" from "another dead uniform axis":**

| Signal | "Carries alpha" (GO) | "Dead uniform axis" (KILL) |
|---|---|---|
| `valid_rank_ic` (checkpoint metric, `baseline env`) | ≥ baseline + a seed-stable margin over n≥6 seeds, **and** survives into IR | within noise of baseline / regresses |
| Readout stock-attn entropy_norm (analogue of `moe_block.py:340`) | < 1.0 and *stable* (sharpened) | pinned at 1.0000 (uniform collapse) |
| Ablation: zero/uniformize the cross-stock layer at eval (leave-one-out, cf. `router_override="no_stock"` pattern `moe_block.py:130-137`) | rank_ic **drops** materially | rank_ic unchanged ⇒ layer is a no-op |
| Score dispersion vs control | cross-section reshaped (rank changes) beyond affine | only affine-shifted (rank-invariant) |
| Portfolio IR under frictions (`/quant-stress`) | preserved/improved | RankIC bump evaporates (the trap) |
| `‖W_v‖·‖W_o‖` of the cross-stock layer | stays O(1) | inflates (→ tens) ⇒ V-escape, not real attention |

**Kill criteria (any one ⇒ NO promotion):**
1. `valid_rank_ic` not above baseline by a seed-stable margin at **n≥6 fresh seeds** (n=3 is
   unreliable per durable finding; kill-check with fresh seeds essential).
2. Eval-time leave-one-out (uniformize the cross-stock attention) leaves rank_ic unchanged ⇒
   load-neutral no-op.
3. Entropy_norm pinned at 1.0 throughout training ⇒ uniform collapse (same death as stock_expert).
4. Any RankIC gain fails to survive `/quant-walk-forward` + `/quant-stress` into IR ⇒ RankIC≠IR trap.
5. `‖W_v‖·‖W_o‖` inflation co-occurring with the gain ⇒ V-escape artifact, not attention.

These are *orthogonal* to the prior dead axes: this is a **new axis (cross-stock at readout)**, not
a re-run of TIME-readout (CLOSED) or block-internal stock_expert (settled). The hypothesis is not a
near-duplicate of a decided card.

---

## GO / NO-GO

**GO — conditional, single cheap run.** The loss-neutrality gradient argument (§3b) shows the
mechanism's death inside the blocks was **placement-induced** (router exit + parked-mean
absorption), not intrinsic. At the readout, in a **main-path, ungated, in-series** placement, both
exits are structurally removed: uniform attention becomes a *repeller* (degenerate max-loss under
replacement) or at worst a loss-neutral constant *on top of a mandatory signal-carrying residual*,
and there is no router to zero the branch. This is materially different from every prior dead axis,
and it is the MASTER placement the durable findings explicitly flagged as promising. The bet is
worth one early-reject proxy run.

**Minimal flag-guarded insertion point (described, NOT implemented):**
- New config flags in `module/utils/model_configuration.py` (next to the stock_expert flags
  `:81-95`): `use_readout_stock_attn: bool=False`, `readout_stock_attn_variant: str="mainpath"`
  (allow-list `{"mainpath","gated"}`), reuse `n_heads`. Default off ⇒ baseline byte-identical.
- New module: a small `CrossStockBlock` (1× `nn.MultiheadAttention(D, n_heads, bias=False,
  batch_first=True)` + residual `LayerNorm`, optional FFN), permutation-equivariant over B, `bias=None`.
- Insertion: in `module/quant_moe_model.py` immediately **after** `h_pooled` is formed
  (after `:542` for d1pma / `:568` for control) and **before** `stock_score = self.head(...)`
  at `:574`. Apply `h_pooled = readout_stock_block(h_pooled.unsqueeze(0)).squeeze(0)` (seq=B).
  Guard the whole insertion behind `if self.use_readout_stock_attn:`.
- **Variant policy:** ship `"mainpath"` (ungated residual+LN, no de-mean, no out-norm) as the bet;
  ship `"gated"` (scalar sigmoid gate, with `rms`/`unit` out-norm if de-mean is added) only as the
  predicted-dead A/B control.
- Validity invariant (single-day batch) is already guaranteed by the daily samplers
  (`model_adapter.py:1250-1292`); no data-path change needed.
- First run: `/quant-hypothesis` card → `/quant-leakage-audit` → `/quant-minimal-repro` proxy
  (mainpath vs gated vs off, ~10–15 min). Promote only if §5 kill criteria all clear at n≥6 +
  walk-forward + stress.

**NO-GO would be correct only if** the team wants to insist on a gated-residual variant with a cold
scalar gate — that re-imports the freeze-at-init failure and is predicted to reproduce the dead
axis. The GO is specifically for the **ungated main-path** placement.
