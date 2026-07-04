# Readout Process Deep-Dive — gradient-level map for cross-stock insertion

> **⚠ LOSS-ASSUMPTION CORRECTION (2026-06-24).** This doc's own escape hatch **fires**: it warns "Raw MSE
> is the one exception — confirm the config doesn't use it." It DOES — `mse_normalize=False`
> (`work_flow.py:175`) ⇒ the live loss is **plain `F.mse_loss`** on a `CSZScoreNorm` label
> (`:107,167`); `cs_ic`/`listmle`/`cs_mse(normalize)` are monitor-only. So **retract** the §0.2/§3c
> conclusion "`Σᵢ ∂L/∂sᵢ=0`, gradient is mean-zero across stocks / every candidate main_loss is
> additive-constant invariant." Plain MSE is shift- and scale-SENSITIVE (`∂L/∂c=2·mean(p)`; `std(p)*≈IC`).
> The repeller conclusion survives via "MSE can't fit a dispersed target with a constant." Corrected math:
> `PORTFOLIO_ALIGNED_PLAN.md` §0-§2,§10.

Read-only forensic. No source edited, no GPU/training run. CPU-only shape reasoning. Branch
`pool-readout-forensics`. All claims cite `file:line` against the working tree.

This document is the **deeper companion** to `analysis/stock_expert/readout_stock_attention_analysis.md`
(read first; not repeated). That file argued *why* an ungated main-path cross-stock layer is
loss-bearing. This file pins down the **exact ops, exact shapes, exact gradients, the eval/predict
path, the variable-B reality, and the insertion mechanics** an implementer needs. Where the first
analysis was slightly imprecise, it is corrected here (see §0).

---

## 0. Corrections / refinements to the first analysis

1. **The d1pma temporal-pool query uses `1/sqrt(D)` over the full D vector — confirmed exact**
   (`quant_moe_model.py:537`). The first analysis said this; verified. But note the d1pma branch's
   `attn` is `einsum("bt,btnd->bnd", a, h)` over the **original** `h` (`:539`), NOT over `h_time`.
   So the temporal weights `a:[B,T]` (computed from the N-mean `h_time`) are applied back to the
   **per-factor** tensor `h:[B,T,N,D]`. The N-collapse is only used to *compute the weights*, not to
   form the pooled value. This matters: `z_T` retains the full N axis (`:541`), so the factor pool
   still sees 158 tokens. (First analysis got the net shapes right but conflated the two.)

2. **The first analysis's "main_loss = listmle" is config-dependent, not hardcoded.** The model
   default is `main_loss="listmle"` (`quant_moe_model.py:639`), but the **adapter** default is
   `"mse"` (`model_adapter.py:2109`), and `loss_weights`/`main_loss` come from config. The active
   composite at `:654-662` always adds `l_aux` (router z-loss) and `l_reg` (selection reg)
   regardless of `main_loss`. For the cross-stock gradient argument the only load-bearing fact is
   that **every candidate `main_loss` (`listmle`/`ic`/`mse(normalize)`) is per-day additive-constant
   invariant** — proven in §3. So the §3b argument holds for whichever is configured.

3. **Variable-B is real and bites at eval.** The first analysis asserted "single-day batch" (true)
   but did not flag that `DailyChunkBatchSampler` **splits a day into chunks of `max_batch_size`**
   when the day has more stocks than `batch_size` (`sampler.py:206-210`). At predict/eval a large
   day is delivered as *multiple* batches → a naive cross-stock layer would attend only within a
   chunk, not the full day. This is a first-class design constraint, detailed in §4 and §5.

---

## 1. Full forward chain, every op, exact shapes

Dimensional anchors: `d_model=64`, `n_heads=4` (`d_head=16`), `n_layers=2`, `N=num_alphas=158`,
`T=context_len=8`, daily cross-section `B` (≈300, **variable** — see §4). Tensor entering the
readout is `h` after the final norm.

```
h = self.final_norm(h)                      # [B,T,N,D]   quant_moe_model.py:490  (LayerNorm over D, :101)
```

`final_norm = nn.LayerNorm(d_model)` (`:101`) — normalizes over the **D axis per (b,t,n) token**;
it is NOT cross-stock (no mixing over B). So nothing before the readout couples stocks except the
optional `stock_expert` *inside* the blocks (off at the anchor) and the regime stats (§2).

### 1a. Control path (`temporal_readout == ""`) — `quant_moe_model.py:565-568`
```
h_last           = h[:, -1, :, :]                      # [B,N,D]   pick last timestep
h_pooled, w      = self.factor_pooling(h_last)         # [B,D], [B,N]
```

### 1b. Active path (`d1pma`, the anchor) — `quant_moe_model.py:534-549`
```
h_time = h.mean(dim=2)                                              # [B,T,D]    mean over N=158   :536
logits = (h_time.float() @ tr_q.float())/sqrt(D) + tr_b_t.float()  # [B,T]      fp32 forced (AMP)  :537
a      = softmax(logits, dim=1).to(h.dtype)                        # [B,T]      per-stock T-weights :538
attn   = einsum("bt,btnd->bnd", a, h)                              # [B,N,D]    a-weighted avg over T :539
g      = sigmoid(tr_gate.float()).to(h.dtype)                      # scalar     init g=sigmoid(-2)=0.119 :540
z_T    = (1-g)*h[:, -1, :, :] + g*attn                             # [B,N,D]    gated residual w/ last step :541
h_pooled, w = self.factor_pooling(z_T)                             # [B,D], [B,N]   :542
```
Params (`:166-168`): `tr_q = randn(D)*0.02` [D=64]; `tr_b_t = zeros(T)` [T=8]; `tr_gate =
full((1,), gate_init)` [scalar]. The fp32 cast at `:537-540` is the AMP-safety pattern — relevant
to §7 (a cross-stock softmax should follow it).

### 1c. Factor pool — `module/architecture/attention_pooling.py` (shared by both paths)
`AdaptivePooling.forward` (`:116-131`), input `x:[B,N,D]`:
```
attn_pooled, w = self.attention_pool(x)        # AttentionPooling, [B,D], [B,N]
mean_pooled    = x.mean(dim=1)                  # [B,D]   plain mean over N
pooled         = alpha*attn_pooled + (1-alpha)*mean_pooled   # alpha = pooling_alpha = 0.7   :129
```
`AttentionPooling.forward` (`:52-97`):
```
q = self.query.expand(B,-1,-1)                  # learnable [1,1,D] -> [B,1,D]   :65 ; query init N(0,0.02) :50
attn_out, attn_w = self.mha(q, x, x,            # nn.MultiheadAttention(D, pool_n_heads, bias=False) :37-44
                            need_weights=True, average_attn_weights=False)
pooled = attn_out.squeeze(1)                    # [B,D]   :85
pooled = self.norm(pooled)                      # LayerNorm(D)   :86   (per-stock, NOT cross-stock)
```
Pool math precisely: a single learnable query attends over the **N=158 factor tokens** of *one
stock*; scale `1/sqrt(d_head)` (PyTorch built-in; `d_head = D/pool_n_heads`). `pool_n_heads=1` →
`d_head=64`. Then 0.7·attn + 0.3·mean over N, then LayerNorm over D. **Output `[B,D]`** — N axis
collapsed. Every op here is per-stock; B is a pure batch dim.

### 1d. Head — `quant_moe_model.py:114, 570-574`
```
self.head = nn.Linear(d_model, 1)              # :114   HAS a bias (default), re-zeroed at :183
stock_score = self.head(h_pooled).squeeze(-1)  # [B]    :574
```
The head **has a bias** (`nn.Linear(64,1)` default `bias=True`), but it is a **per-day-constant
added to every stock** → loss-invariant (§3). The first analysis called the head "bias-less"; it is
not, but the bias is loss-neutral, so the conclusion is unchanged. `head.weight` is
kaiming-normal-reinit at `:181`, bias zeroed at `:183`.

**Whole-readout summary of axes collapsed:**
`[B,T,N,D] --(T-collapse: last-step or d1pma)--> [B,N,D] --(factor_pool: N-collapse)--> [B,D]
--(head)--> [B]`. **B (stock) is never an attention axis anywhere in the readout.** Every readout op
is permutation-equivariant over B and cross-sectionally separable: `s_i = f(h[:, :, :, :][i])`.

---

## 2. Regime / context path — does a per-day market vector already reach the readout?

**Producer:** `regime = self.regime_encoder(x, macro_features)  # [B,D]` (`quant_moe_model.py:309`),
`RegimeContextEncoder` (`regime_encoder.py:8-189`).

**What it computes (the load-bearing detail):** with `use_external_macro=False` (the anchor) and
`internal_use_batch_stats=True`, it extracts **4 scalars from the day's cross-section** —
crowding (mean |off-diag corr|), PC1 explained ratio, crowding-drift over lag, tail/shock ratio
(`regime_encoder.py:161-176`) — computes them **once per day on `[1,4]`** then **`expand(B,-1)`**
(`:176`). So `regime[i]` is **identical for every stock i in the day**: it is a genuine **per-day
market-context vector** broadcast across the cross-section. Encoder = MLP `4→2D→D` + Tanh + LN
(`:40-46`), forced fp32 (`:138-145`).

**What it conditions today:**
- **Router input** — `router_input = regime_embedding` (`moe_block.py:83`); with `use_layer_summary`
  it is concatenated with a per-day market mean/std summary (`moe_block.py:89-95`). So regime drives
  the time/factor(/stock) expert mixing weights.
- **Regime-adaptive time embedding** — `time_embedding(regime, T)` → `[B,T,D]` added to `h`
  (`quant_moe_model.py:333-338`), produces per-day `tau`.
- **Regime-adaptive factor gate (FiLM)** — `factor_gate(regime, factor_table)` → `(gamma,beta)`
  `[B,N,D]` applied inside blocks (`quant_moe_model.py:357-358`, `moe_block.py:166-176`).

**Does any per-day context vector already reach the READOUT?** **No, not directly.** `regime` is
consumed in the embedding/block stages; it is **not** passed into the d1pma branch, the factor pool,
or the head. Its influence on the readout is only **indirect** (through the block outputs `h`). So:

> **MASTER-style market gating ingredients already exist** (`regime:[B,D]`, a clean per-day market
> vector, computed past-only/leak-free and already broadcast across the cross-section), but they are
> **not currently wired to the readout**. A MASTER-style "market-guided gating at the head" could
> reuse `regime` directly (FiLM the `h_pooled` by `regime`, or concat) **without** any new market
> encoder. Critically, because `regime[i]` is a **per-day constant across stocks**, any *purely
> additive* use of it at the head is **rank-invariant → loss-neutral** (§3); a *multiplicative*
> (FiLM `gamma`) use is rank-changing only if `gamma` interacts with the per-stock `h_pooled`. This
> is a separate lever from cross-stock attention and shares the §3 invariance analysis.

---

## 3. Loss coupling — exact gradient path

Losses in `module/utils/losses.py`. Cross-section is a single day's `p:[B]` (scores), `y:[B]`
(labels), assembled in `quant_moe_model.py:600-662`. `valid = isfinite(labels)` masks NaN labels
(`:597`), needs ≥2 valid (`:600`); `p = stock_score[valid]`, `y = labels[valid]` (`:601-602`).

### 3a. Per-day cross-sectional structure of each component
- **`cs_ic_loss(p,y)` (`losses.py:7-26`):** centers both (`x=p-p.mean()`, `y=y-y.mean()`, `:20-21`),
  returns `-corr(p,y)`. Pearson correlation over the day's cross-section.
- **`listmle_loss(p,y,tau)` (`losses.py:84-117`):** sort `p` by descending `y` → `s` (`:101-102`);
  `s = s - s.max().detach()` (`:108`); `s = s/tau` (`:110`); `L = mean_i(logcumsumexp_{j>=i} s_j -
  s_i)` (`:113-116`). Plackett-Luce listwise NLL — depends only on the **induced ordering** of `s`.
- **`cs_mse_loss(p,y,normalize)` (`losses.py:56-81`):** if `normalize` (`mse_normalize`), z-score
  both over the day (`:77-80`) then MSE; else raw MSE.
- **Composite** (`quant_moe_model.py:654-662`): `total = w[main]*l_main + l_aux + l_reg (+ rank +
  huber)`. `l_aux`/`l_reg` do not touch `p` w.r.t. the readout, so they contribute **zero** to
  `∂L/∂s`.

### 3b. Invariances of each loss in the score vector `s ∈ R^B`
Let `s → α·s + β` (per-day, same for all stocks), `α>0`:

| Loss | additive const `β` | positive scale `α` | general monotone |
|---|---|---|---|
| `cs_ic` | **invariant** (centering `:20`) | **invariant** (corr) | NOT invariant |
| `listmle` | **invariant** (`s.max()` subtract `:108`, and order unchanged) | invariant *up to tau-rescale*: `s/tau` so scaling `s` is equivalent to changing `tau`; order preserved → loss changes only via effective temperature, ranking unchanged | order-preserving monotone → invariant |
| `cs_mse(normalize)` | **invariant** (centering `:77-78`) | **invariant** (unit-var `:79-80`) | NOT invariant |
| `cs_mse(raw)` | NOT invariant | NOT invariant | NOT invariant |

**Decisive shared property:** every loss the model can be configured to optimize as `main_loss`
(`listmle`/`ic`/`mse`) — except raw (un-normalized) MSE — is **invariant to a per-day additive
constant** on `s`. So a readout op that adds the *same vector contribution to every stock's score*
(i.e. a per-day constant after the head) has **exactly zero gradient**. This is the formal
"loss-neutral" criterion for any cross-stock layer:

> A cross-stock readout layer is **loss-bearing iff it changes the cross-sectional *ordering*** of
> `s` (for listmle) or the *centered/normalized* `s` (for ic/mse-norm). A layer whose output is a
> per-day constant (uniform attention, or an additive per-day market vector) is **loss-neutral**.

### 3c. Exact ∂L/∂s
**cs_ic** (`L = -corr`). With centered `x=p-p̄`, `y=y-ȳ`, `c=mean(xy)`, `vx=mean(x²)`,
`vy=mean(y²)`, `ic = c/(sqrt(vx·vy)+eps)`:
```
∂(-ic)/∂p_i = -(1/B) · [ y_i/(sqrt(vx·vy)) − ic · x_i/vx ] / (centered)   (+ eps terms)
```
i.e. proportional to `−(ŷ_i − ic·x̂_i)` where hats are the standardized residuals — pushes `p_i`
toward the rank position of `y_i`, with a deflation along the current `p`-direction. **Sum over i is
0** (centering) → confirms additive-constant invariance: `Σ_i ∂L/∂p_i = 0`.

**listmle.** With `s` = `p` reordered by descending `y` (let `π` be that permutation, `s_k =
p_{π(k)}`), and `P_k = softmax weight of position k in its suffix`:
```
∂L/∂s_k = (1/B) · [ Σ_{m ≤ k} ( exp(s_k/τ) / Σ_{j≥m} exp(s_j/τ) ) − 1 ]   (per the logcumsumexp suffix sums; τ from :110)
```
Concretely `∂L/∂s_k = (1/(Bτ)) · ( Σ_{m=1..k} softmax_suffix_m(k) − 1 )`, where
`softmax_suffix_m(k) = exp(s_k/τ)/Σ_{j≥m}exp(s_j/τ)`. The `−1` is the "should be ranked first in
its own prefix" target. **Σ_k ∂L/∂s_k = 0** as well (each suffix-softmax column sums to 1 across its
members) → additive-constant invariance confirmed at the gradient level. Note `s.max()` is
`.detach()`ed (`:108`) so no gradient flows through the argmax — clean.

**Consequence for a cross-stock layer.** Backprop reaches the layer as `∂L/∂s` (B-vector summing to
0), then through the head `w` to `∂L/∂h_pooled[i] = w · ∂L/∂s_i` (`[B,D]`), then into the
cross-stock op. Because `Σ_i ∂L/∂s_i = 0`, **the gradient that reaches the cross-stock layer is
already mean-zero across stocks** — the uniform/common-mode component of any cross-stock output
receives no gradient pressure. Only the **stock-relative (rank-changing) component is trained.**
This is the precise mechanism behind "uniform attention is loss-neutral" and behind why a
main-path *replacing* layer (no residual) is a repeller (uniform ⇒ flat `s` ⇒ degenerate corr/
max-listmle ⇒ huge gradient away from uniform).

---

## 4. Single-day-batch invariant & variable-B

**Train sampler** `module/dataloader/sampler.py`:
- `FixedDailyBatchSampler` (`:7-114`, `sampled_daily` mode): groups indices by `datetime`
  (`:39`), and **per day** down/up-samples to **exactly `batch_size`** (`:68-78`). So in
  `sampled_daily` training **B is constant = `batch_size`** and every batch is one day's
  cross-section (down-sampled if the day has more names, up-sampled-with-replacement if fewer —
  `:70/:74`). Up-sampling means **duplicate stocks** can appear in a batch (`duplicate_draws`,
  `:88`) — a cross-stock attention would then attend to duplicated rows (benign but not a clean
  distinct-stock set).
- `DailyChunkBatchSampler` (`:117-220`, `full_daily` mode): **no resampling**; iterates each day and
  **splits into consecutive chunks of `max_batch_size`** (`:206-210`). So a day with `n >
  batch_size` yields `ceil(n/batch_size)` batches → **B varies** and **a single day's cross-section
  is split across multiple batches**. A short day yields `B = n < batch_size`.

**Eval/predict sampler:** `predict()` always uses `DailyChunkBatchSampler(tsds,
max_batch_size=self.batch_size)` (`model_adapter.py:2357`). Same chunking → **variable B at eval**,
and large days are split.

**Selection of train sampler:** `_make_daily_loader` (`model_adapter.py:1250-1268`): `full_daily`
→ `DailyChunkBatchSampler`; else `FixedDailyBatchSampler` (`:1257-1265`). The eval-collate asserts
no mixed dates per batch (`:1204`).

**Net for cross-stock attention:**
1. **No date mixing — guaranteed.** Every batch is a subset of exactly one trading day
   (`sampler.py:39/65` and `:165/194`; assert `model_adapter.py:1204`). Cross-stock attention never
   mixes dates. ✔ free invariant.
2. **B is variable** in `full_daily` train and **always variable at predict** (`sampler.py:206-210`;
   `model_adapter.py:2357`). A cross-stock MHA must accept dynamic seq-len B — trivial for
   `nn.MultiheadAttention` (no fixed-size param on the seq axis) but it means **no shape can be
   hardcoded** to 300.
3. **Day-splitting is the real hazard.** When a day has more stocks than `batch_size`, the day is
   delivered in chunks (`sampler.py:206-210`), so cross-stock attention sees only a *partial*
   cross-section per forward. For the anchor `batch_size=300`, csi300 days fit in one chunk most
   days, but this is **not guaranteed** and must be handled (see §5/§7). Train `sampled_daily`
   side-steps it by forcing B=300, but at the cost of down/up-sampling (partial/duplicated
   cross-section anyway).
4. **NaN handling.** Features: `bx_t = torch.nan_to_num(bx, 0.0)` before the model
   (`model_adapter.py:2384`, train `:1793` analogous). Labels: NaN labels are masked in the loss
   (`quant_moe_model.py:597-602`) but the **NaN-label stock still flows through the forward** (it is
   only dropped when computing the loss). So a cross-stock layer **will attend to and from
   NaN-label stocks** (their features are zero-filled, labels excluded from loss). No per-stock
   key-padding mask exists today.

---

## 5. Eval/predict path & leave-one-out feasibility

**Predict** (`model_adapter.py:2346-2398`): `net.eval()`, no-grad, `DailyChunkBatchSampler`, per
batch `out = self.net(bx_t, f_ids, macro_features=macro_t)` (`:2388`), scores scattered back to the
dataset index via positional indices `bpos` (`:2390-2393`), with a hard check that all preds are
finite (`:2395-2397`). Scores come from `out.scores` (= `stock_score`, `quant_moe_model.py:705`).

**Existing eval hook to mirror — `router_override`.** `forward(..., router_override=...)`
(`quant_moe_model.py:272`) threads down to every block (`:427`), and `moe_block.py:115-144`
implements the modes: `time`/`factor`/`stock`/`no_stock`/`uniform`. The **`no_stock`** mode
(`moe_block.py:130-137`) is the leave-one-out probe: it **zeros the stock-expert gate and
renormalizes** the learned time/factor weights at eval — measuring the stock expert's marginal
contribution **without retraining**. The router-oracle eval loop already passes
`router_override=override` per mode (`model_adapter.py:2606-2614`).

**Feasibility of a `readout_stock_override` mirror:** **High.** The clean pattern is:
1. Add a kwarg `readout_stock_override: str|None` to `forward` (next to `router_override` at
   `quant_moe_model.py:272`).
2. In the readout block, branch on it: `"off"`/`"uniformize"` (force `a_ij = 1/B`, i.e. replace
   cross-stock output with the cross-sectional mean = the loss-neutral parking value) /
   `"zero"` (skip the layer → identity residual). No retrain — exactly the `no_stock` analogue but
   at the readout.
3. Thread it through `predict`/the oracle loop the same way `router_override` is threaded
   (`model_adapter.py:2613`). The per-day eval loop already concatenates `p`/`y` per date
   (`:2616/2624-2629`) so a uniformize-vs-on RankIC delta is one extra mode in the `modes` dict.

This gives the §5 kill-criterion of the first analysis ("uniformize at eval ⇒ RankIC unchanged ⇒
no-op") a ready implementation path that mirrors proven infra.

**Day-split caveat at eval (important for the LOO probe):** because predict splits large days
(`sampler.py:206-210`), a cross-stock layer's eval output for a split day depends on chunk
boundaries. For a faithful eval the predict sampler should deliver **whole days** when
`use_readout_stock_attn` is on (e.g. set `max_batch_size` ≥ max daily count, or add a
"whole-day" predict path). Flag this to the implementer.

---

## 6. Exact insertion candidates with shapes

`D=64`, one `nn.MultiheadAttention(D, H, bias=False, batch_first=True)` costs in-proj `3D²=12,288`
+ out-proj `D²=4,096` = **16,384 params** (+LN 128). All candidates are **flag-guarded, default-off,
byte-identical when off**. B is the attention sequence length (variable, §4).

| # | Locus (file:line) | Tensor in | cross-stock op | Tensor out | Param cost | Masking / variable-B concern |
|---|---|---|---|---|---|---|
| A | **post-pool, pre-head** — after `:542`(d1pma)/`:568`(control), before `:574` | `h_pooled:[B,D]` | MHA seq=B: `h_pooled.unsqueeze(0)→[1,B,D]`, attend stock↔stock, `→[B,D]` | `[B,D]` | ~16K + LN | cleanest; 1 token/stock; needs key-padding mask only if NaN-label stocks excluded; day-split (§4.3) |
| B | pre-pool, on `z_T` | `z_T:[B,N,D]` | for each of N tokens, MHA over B → reintroduces N=158 parallel groups | `[B,N,D]` | ~16K (shared) | 158× compute; mixes stock-interaction w/ factor-pool; harder to interpret; reject |
| C | pre-pool, on `h_last`/`h[:, -1]` (control) | `[B,N,D]` | same as B for control path | `[B,N,D]` | ~16K | same as B |
| D | post-head, on scalar `s:[B]` | `s:[B]` | attention over scalars (project to D first) | `[B]` | small | a pure additive/affine map here is rank-invariant unless it lifts to D; degenerate; reject |
| E | on `regime`-FiLM of `h_pooled` (MASTER market-gate) | `h_pooled:[B,D]`, `regime:[B,D]` | `gamma,beta = MLP(regime)`; `h_pooled = gamma⊙h_pooled+beta` | `[B,D]` | ~`2·D²`=8K | NOT cross-stock attention; `regime` is per-day const ⇒ additive `beta` rank-invariant, multiplicative `gamma` rank-changing only via interaction; a different lever (§2) |

**Recommended: A** (matches first analysis). Concretely, at `quant_moe_model.py:569` (between
`h_pooled` formation and `stock_score = self.head(...)` at `:574`):
```
if self.use_readout_stock_attn:
    o = self.readout_stock_attn(h_pooled.unsqueeze(0), key_padding_mask=...)  # [1,B,D]->[1,B,D]
    h_pooled = self.readout_stock_norm(h_pooled + o.squeeze(0))               # mainpath: residual+LN, no scalar gate
stock_score = self.head(h_pooled).squeeze(-1)
```
- **in/out:** `[B,D] → [B,D]`. seq=B, no `(t n)` batching (contrast block stock_expert's 1264
  groups, `moe_block.py:222`). One MHA call.
- **head bias** (`:114`) stays loss-neutral.
- **mainpath variant:** residual+LN (`o` added unconditionally), **no scalar sigmoid gate**
  (freeze-at-init trap). **gated variant** (control): `h_pooled = h_pooled + sigmoid(gate)*o` with
  `out_norm` if de-meaned — predicted dead.

---

## 7. Gaps / risks the first analysis missed

1. **Day-splitting at eval/full_daily train (§4.3, §5 caveat).** The single biggest implementation
   risk. `DailyChunkBatchSampler` (`sampler.py:206-210`) means a cross-stock layer can see a partial
   cross-section. The first analysis treated B as "one day" cleanly; in `full_daily`/predict it is
   "one chunk of a day." Must be handled (whole-day predict path or `max_batch_size` ≥ max daily
   count when the flag is on).

2. **Up-sampling duplicates stocks (§4, `sampler.py:74`).** In `sampled_daily` train, days with
   `<batch_size` names are filled by sampling **with replacement** → duplicate rows in the
   cross-section. A cross-stock attention then has duplicate keys/queries (a stock attends to copies
   of itself). Benign for correctness but inflates self-attention mass; worth a diagnostic
   (analogue of `stock_attn_self_frac`, `moe_block.py:477`).

3. **NaN-label stocks flow through the forward (§4.4).** They are masked only in the loss
   (`quant_moe_model.py:597-602`), not dropped from the batch, and features are zero-filled
   (`model_adapter.py:2384`). So a cross-stock layer will mix zero-filled NaN-label stocks into
   every other stock's representation. **Recommend a key-padding mask** built from
   `~isfinite(labels)` (train) — but note labels aren't available at predict, so eval cannot mask
   the same way; the safest design treats all present rows as valid keys and relies on zero-filled
   features being low-signal. Flag explicitly.

4. **Factor-pool internal LayerNorm is per-stock — no cross-stock leakage (verified).**
   `AttentionPooling.norm = LayerNorm(D)` (`attention_pooling.py:47,86`) and `final_norm`
   (`quant_moe_model.py:101`) both normalize over **D within a token**; neither computes statistics
   over B. There is **no BatchNorm anywhere** in the readout/blocks — confirmed by absence in
   `attention_pooling.py`/`quant_moe_model.py`/`moe_block.py` (only LayerNorm/RMS). So **no implicit
   cross-stock coupling exists today** except (a) the optional `stock_expert` (off at anchor) and
   (b) the regime stats computed over the batch (§2, but those are detached/clamped scalars). This
   is the clean slate the first analysis assumed; now verified.

5. **The regime path DOES couple stocks (subtle leakage-class note).** `regime` is computed from
   **batch cross-sectional statistics** (corr/PC1/tail over the B stocks, `regime_encoder.py:161-176`)
   then broadcast. So the model **already has a (weak, per-day-constant) cross-stock channel** via
   regime — but it injects only a per-day constant (rank-invariant, §3) plus FiLM modulation. PC1 is
   `.detach()`ed (`regime_encoder.py:92`); crowding is differentiable. An implementer adding
   cross-stock attention should know this channel exists so as not to double-count "market state."

6. **AMP fp16 on a B≈300 softmax.** Predict/eval run under `_autocast_ctx()`
   (`model_adapter.py:2386,2607`; autocast fp16/bf16 `:331-341`). The d1pma branch **force-casts
   logits/softmax/sigmoid to fp32** (`quant_moe_model.py:537-540`) precisely to avoid AMP
   instability. A new cross-stock softmax over B=300 should **mirror this fp32 cast** — fp16 softmax
   over 300 logits with large dot-products can overflow/underflow and lose precision in the tails.
   `nn.MultiheadAttention` under autocast keeps internal scores in the autocast dtype unless the
   inputs are fp32; safest is to run the cross-stock MHA outside autocast or cast `h_pooled` to fp32
   for the attention. Flag for implementation.

7. **`stock_expert_out_norm` reuse, not LayerNorm-affine.** If a gated/de-meaned readout variant is
   built, reuse the **parameter-free** norms (`rms`/`unit`/`ln`) from `moe_block.py:241-254`; the
   `ln_affine` mode (`:255-260`) reopens the V-escape via `gamma` and exists only as a diagnostic.
   The mainpath variant needs none of this (no de-mean ⇒ nothing to game; §3 of first analysis).

8. **Two different attention scales already coexist (minor, document for consistency).** d1pma and
   factor-pool use `1/sqrt(D)` (full-D query, `quant_moe_model.py:537`,
   `attention_pooling.py:81`); the block experts and stock_expert use PyTorch's `1/sqrt(d_head)`.
   A new cross-stock MHA via `nn.MultiheadAttention` will use `1/sqrt(d_head)=1/4` for `H=4`. Not a
   bug; just be explicit so the temperature is understood (no learnable τ).

---

## Implementation-readiness checklist (`use_readout_stock_attn`)

Exact facts an implementer needs:

**Where (insertion):**
- Locus A — `module/quant_moe_model.py` between `h_pooled` formation (after `:542` d1pma / `:568`
  control) and `stock_score = self.head(h_pooled)` at `:574`. Guard with `if
  self.use_readout_stock_attn:`. Default-off ⇒ baseline byte-identical.

**Shapes:**
- In/out `h_pooled : [B,D]` with `D=64`; attention seq = B (variable, NOT 300). MHA call:
  `[1,B,D] → [1,B,D]`. One `nn.MultiheadAttention(64, n_heads=4, bias=False, batch_first=True)`,
  ~16K params + `LayerNorm(64)`.

**Variant policy (from §3 + first analysis):**
- `"mainpath"` (the bet): `h_pooled = LayerNorm(h_pooled + MHA(h_pooled))`, **unconditional
  residual, no scalar gate**, **no de-mean, no out_norm** (nothing to game).
- `"gated"` (predicted-dead control): `h_pooled = h_pooled + sigmoid(gate)*MHA(...)`; if de-meaned,
  add parameter-free `out_norm` (`rms`/`unit`, `moe_block.py:241-249`), never `ln_affine`.

**Masking / variable-B (§4):**
- B varies; never hardcode 300. `nn.MultiheadAttention` handles dynamic seq-len natively.
- **Day-split hazard:** `DailyChunkBatchSampler` (`sampler.py:206-210`) splits big days; predict
  uses it unconditionally (`model_adapter.py:2357`). For a faithful cross-stock forward, ensure
  whole-day batches when the flag is on (raise predict `max_batch_size` ≥ max daily count, or add a
  whole-day predict path).
- **NaN-label stocks** flow through the forward (masked only in loss, `quant_moe_model.py:597-602`).
  Optionally pass a `key_padding_mask = ~isfinite(labels)` in train; note labels are absent at
  predict so eval cannot mask identically — design for "all present rows are valid keys."
- **Up-sampling duplicates** in `sampled_daily` (`sampler.py:74`) → duplicate keys; add a self-/dup
  diagnostic.

**AMP (§7.6):**
- Run the cross-stock softmax in **fp32** (mirror `quant_moe_model.py:537-540`): cast `h_pooled` to
  fp32 for the MHA or run it outside autocast. Predict/eval are under autocast
  (`model_adapter.py:2386,2607`).

**Loss / gradient facts (§3):**
- Every configurable `main_loss` (`listmle`/`ic`/`mse-norm`) is **per-day additive-const invariant**
  (`losses.py:20,77-78,108`). The gradient reaching the readout is **mean-zero across stocks**
  (`Σ_i ∂L/∂s_i = 0`) → uniform/common-mode cross-stock output is untrained; only the rank-changing
  component is. Mainpath (replacing-with-residual) makes uniform a repeller (degenerate flat `s`).
  Raw (un-normalized) MSE is the one exception (not additive-invariant) — confirm config doesn't use
  it if relying on the invariance.

**Eval hook (§5):**
- Mirror `router_override` (`quant_moe_model.py:272`, threaded to blocks `:427`,
  `moe_block.py:115-144`, modes in oracle loop `model_adapter.py:2606-2614`). Add
  `readout_stock_override ∈ {off, uniformize, zero}` to `forward`, thread through `predict`/oracle.
  `"uniformize"` (force `a_ij=1/B` ⇒ cross-sectional mean) is the leave-one-out probe; RankIC delta
  vs on = the load-bearing test. No retrain needed.

**Config (§6):**
- New flags next to stock_expert flags (`model_configuration.py:79-95`): `use_readout_stock_attn:
  bool=False`, `readout_stock_attn_variant: str="mainpath"` (allow `{"mainpath","gated"}`), reuse
  `n_heads`. Init the new MHA via the existing `nn.MultiheadAttention` branch in `_init_weights`
  (`quant_moe_model.py:250-261`) — already handled by `post_init`.

**Clean-slate facts (§7.4):**
- No BatchNorm anywhere; all norms are per-token LayerNorm/RMS (`quant_moe_model.py:101`,
  `attention_pooling.py:47,86`, `moe_block.py:56-57`). The only existing cross-stock channels are
  the off-by-default `stock_expert` (`moe_block.py:221`) and the per-day-constant `regime`
  (`regime_encoder.py:161-176`) — neither reaches the readout directly today.
