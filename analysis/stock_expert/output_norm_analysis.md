# Stock-Expert Output Normalization — Closing the "V-Amplification Escape"

**Date**: 2026-06-17
**Status**: DESIGN ONLY (no source modified, no GPU run). Default-off, reversible.
**Context**: stock expert (`h-20260610-002`) at `moe_block.py:53` (`self.stock_expert = ParallelAttention(config)`), fused at `moe_block.py:248-250`.

---

## 0. The observed pathology (restate precisely)

With `stock_expert_demean=True` and stock `Wq/Wk` freed from weight decay, over 25 epochs:

- `stock_attn_entropy_norm` ≈ 0.9999 (stayed maximal/uniform — attention NEVER sharpened)
- `stock_expert_norm` ↑ from 0.055 → ~88 (output magnitude exploded ~1600×)
- `stock_winner_ratio` → 1.0 (the de-meaned stock contribution dominates time/factor by raw norm)

**Mechanism.** The de-mean signal at `moe_block.py:237` (`out_stock = out_stock - out_stock.mean(dim=0)`) creates a gradient that rewards *cross-stock variation* in the output. With near-uniform attention `A ≈ (1/B)·𝟙𝟙ᵀ`, the attention output for token `(t,n)` is

```
out_stock[b] = Σ_j A[b,j] · (V x)[j]  ≈  (1/B) Σ_j (V x)[j]   =  v̄    (the same vector for all b)
```

That common vector `v̄` is then *killed* by de-mean (`out - mean_b out ≈ 0`). So at uniform attention the expert contributes ~0 — exactly the trap the de-mean was designed to spring. There are two escape routes to produce a **nonzero de-meaned output**:

- **Path A (intended)**: sharpen `A` away from uniform → `out_stock[b]` varies across `b` → survives de-mean. Requires warming Q/K.
- **Path B (lazy escape, taken)**: keep `A` uniform but blow up `‖Wv‖‖Wo‖`. The residual cross-stock variation that survives de-mean is the *second-order* term `(A − 1/B·𝟙𝟙ᵀ)·Vx`, which is tiny — but scaling `V,O` up by factor `s` scales it by `s`. The model drives `s → ~88` to make even an almost-uniform attention produce a de-meaned signal large enough to satisfy the gradient. Q/K stay cold; the entropy diagnostic never moves.

The two paths are **multiplicatively coupled** in the de-meaned output magnitude:

```
‖demean(out_stock)‖  ≈  s · ‖demean( A · (V̂ x) )‖
                          ↑ Path B            ↑ Path A (sharpness of A)
```

where `s = ‖Wv‖‖Wo‖` is the V/O gain and `V̂` is the unit-normalized value projection. The gradient on the *product* can be satisfied by raising **either** factor. B is cheaper (linear, no need to restructure attention logits), so SGD takes B. **Output normalization removes `s` from the product, leaving only Path A.**

---

## 1. Math: what does `LayerNorm(out_stock)` actually cap?

`out_stock` is `[B,T,N,D]`. The escape rides on **cross-stock magnitude variation** — variation along the **B (stock) axis** at fixed `(t,n)`. So the object the de-mean operates on, and the object that exploded, is the **per-(t,n) cross-stock block** `out_stock[:, t, n, :]` ∈ ℝ^{B×D}.

### 1a. `LayerNorm` over D (the naive placement) does NOT close path B

`nn.LayerNorm(D)` applied to `[B,T,N,D]` normalizes over the **last dim D, independently per (b,t,n)**:

```
LN(out)[b,t,n,:] = γ ⊙ (out[b,t,n,:] − μ_{b,t,n}) / σ_{b,t,n}  + β
```

`μ, σ` are scalars computed **within a single token vector** over its D channels. This fixes each token's *vector* to unit RMS — but it does **not** couple the B stocks together. Critically:

- The **per-stock scale is NOT removed in the cross-stock sense.** Each stock `b` gets its own `σ_{b,t,n}`. After LN every token has RMS≈1, so the *gross* 88× blowup is gone — but the de-mean signal does not ride on gross magnitude, it rides on the **relative cross-stock differences** of those vectors. LN-over-D preserves *direction* per token and rescales each to the unit sphere. Two stocks that differ only in scale (both ≈ `s·v̄`) collapse to the same unit vector → their de-meaned difference → 0. That actually looks helpful...
- ...**but** Path B does not need gross per-stock scale; it needs the second-order term `(A − ūniform)·Vx` to survive, and that term has *both* a direction and magnitude component. LN-over-D removes magnitude but the **affine γ re-introduces a global per-channel gain** (see §2), and more importantly LN-over-D's per-token normalization is computed *after* de-mean would have to be re-ordered. The cleaner statement: **LN-over-D caps the per-token vector norm to O(√D·‖γ‖), which DOES bound `stock_expert_norm`** (the explosion to 88 cannot happen because every token is RMS-fixed). So as a magnitude cap it works; the question is whether γ reopens it (§2) and whether the *right axis* is D or B (below).

### 1b. The "right object": normalize the cross-stock block `[B,D]`, not the token `[D]`

The de-mean at line 237 is over `dim=0` (the B/stock axis). The pathology is **inflation of cross-stock dispersion**. The most *targeted* normalization is one that fixes the **cross-stock RMS of the block** `out_stock[:, t, n, :]`:

```
block = out_stock[:, t, n, :]                  # [B, D]
scale = block.pow(2).mean().sqrt()             # scalar RMS over (B,D) of this (t,n) block
out_stock[:, t, n, :] = block / (scale + eps)  # fixes total block energy
```

This directly caps the quantity the de-mean amplifies — total cross-sectional energy per `(t,n)` — while leaving the **relative pattern across stocks free** (it normalizes the whole block by one scalar, so which stock is large vs small is untouched). That is the cleanest "sharpen-or-die": the expert can redistribute energy across stocks (Path A) but cannot grow total energy (Path B).

**However**, per-token LN-over-D (1a) is a *stronger* cap and the conventional, well-conditioned choice. It bounds each token to the unit sphere, so the cross-stock dispersion is automatically bounded by `2` (max distance between two unit vectors). The B-block scalar-RMS is *gentler* (allows one stock to dominate the block). For the stated goal — **force sharpen-or-die** — per-token LN-over-D is the safer first cut because it makes Path B *impossible at the token level*, whereas the B-block scalar leaves a per-stock-direction-magnitude tradeoff partially open.

**Verdict on object**: normalize **over D per token** (`LN`/`RMSNorm` on the D axis) as the primary cap. It bounds `stock_expert_norm` to O(√D) regardless of `Wv,Wo`. The B-block scalar variant is a secondary, gentler option if per-token LN over-constrains and kills useful direction info. **Do NOT** rely on the gate to do the capping (§3).

---

## 2. Variant comparison — which actually forecloses Path B?

The test: after the norm, can `s = ‖Wv‖‖Wo‖` still grow without bound and re-inflate the de-meaned output? Path B is foreclosed **iff the post-norm output magnitude is invariant (or bounded) under `Wv,Wo → s·Wv,Wo`**.

| Variant | Post-norm output | Is `s·Wv,Wo` scaling neutralized? | γ re-opens escape? | Forecloses B? |
|---|---|---|---|---|
| **(a) LN with affine** `LN(out)·γ+β` | `γ⊙(out−μ)/σ + β` | YES for the `/σ` part — `σ` scales with `s`, cancels | **YES, partially.** `γ` is a free `[D]` gain. The model can grow `‖γ‖` instead of `‖Wv‖‖Wo‖` and re-inflate the de-meaned signal (de-mean over B passes through `γ` linearly: `demean_b(γ⊙·) = γ⊙demean_b(·)`). **γ is a renamed `s`.** | **NO — reopens it.** |
| **(b) LN no-affine / RMSNorm no-scale** | `(out−μ)/σ` or `out/RMS(out)` | YES — `σ`/`RMS` both scale with `s`, exactly cancel. No free gain parameter. | No γ exists. | **YES.** Output magnitude is identically O(1) for any `s`. Gradient w.r.t. `s` of the de-meaned norm is ≈0 → no incentive to grow V/O. |
| **(c) fixed unit-norm** `out/‖out‖` (per token over D) | unit vector per token | YES — pure projection to sphere, scale-invariant by construction | none | **YES**, hardest cap. Identical scale-invariance to (b); RMSNorm is essentially (c) up to a √D constant. |
| **(d) fixed scalar cap** `out·min(1, c/‖out‖)` | clipped only when ‖out‖>c | Only when the cap binds. Below `c` the gradient is untouched. | none | **Partially.** Caps the *ceiling* (no 88), but below `c` Path B is still the cheaper local move; the model parks at `‖out‖=c` with uniform attention. Bounds the symptom, not the incentive. |

**Conclusion.** Only **scale-invariant, parameter-free** normalization forecloses Path B: **(b) RMSNorm-no-scale ≡ (c) unit-norm** (the same operator family). LN-with-affine (a) **reopens** the escape through γ (the model relearns `s` as `‖γ‖` — we would see `stock_expert_norm` re-climb, just relabeled). Scalar cap (d) only truncates the ceiling and leaves the uniform-attention parking spot intact at the cap value.

**Recommended primary variant: RMSNorm without learnable scale, over the D axis, per token** (`out / sqrt(mean_D(out²) + eps)`). It is scale-invariant (kills Path B), parameter-free (nothing to re-grow), cheap, and well-conditioned. Equivalent to unit-norm up to the √D constant; preferred over raw unit-norm because RMS is the standard, numerically-stable form and composes cleanly with the existing `w_stock` gate.

---

## 3. Placement: before or after de-mean? And why didn't the gate already cap it?

### 3a. Why the gate (w_stock ≈ 0.45) did NOT cap magnitude

The fused term is `w_stock · out_stock` (line 250). The gate scales the contribution, yes — but **the gate is a number in [0,1] and the expert grew its raw output by 1600×.** `0.45 × 88 ≈ 40`, which still dwarfs `w_time·out_time` and `w_factor·out_factor` (those experts are WD-regularized and O(1)). The gate is **soft and learned slowly**; it cannot win a race against an unregularized `‖Wv‖‖Wo‖` that the de-mean gradient is actively *pushing up* every step. The gate caps the *fraction* of the sum, not the *magnitude* of the addend — and because the addend is huge, the gate would have to collapse to `~0.01` to neutralize it, which the router has no gradient pressure to do (the big stock output is *reducing* training loss via the de-mean term, so the router is rewarded for keeping `w_stock` up). **The gate is structurally the wrong lever; magnitude must be capped at the source.**

### 3b. Norm BEFORE de-mean (recommended)

Order should be: `out_stock` → **(optional xs_center input already applied) → project → NORMALIZE → de-mean → fuse**.

```
out_stock = stock_expert(h_stock)                 # line 230
out_stock = NORMALIZE(out_stock)                  # NEW — cap magnitude here
if stock_expert_demean:                           # line 236
    out_stock = out_stock - out_stock.mean(dim=0) # de-mean the already-capped output
```

Rationale:

- **Norm-then-demean**: the de-mean now operates on a unit-RMS field. The *only* way to get a nonzero de-meaned output is to make the **unit vectors point in different directions across stocks** — i.e., make attention sharpen so different stocks attend to different neighbors. Magnitude is no longer a degree of freedom. This is exactly "sharpen-or-die." ✅
- **Demean-then-norm** would be wrong: de-meaning a uniform field gives ≈0, and then normalizing 0/‖0‖ blows up / is ill-conditioned (division by ~eps amplifies noise into a fake signal). Avoid. Always **normalize first**.

So: insert the norm **immediately after line 231 (the rearrange back to [B,T,N,D]) and BEFORE the demean block at 236**.

### 3c. Interaction with `stock_expert_xs_center` (input centering, line 223-229)

That flag centers the *input* h_stock across stocks. It is orthogonal to output norm and can be combined: input-center shapes Q/K/V to carry only stock-relative signal; output-norm caps the V/O gain. Both can be on simultaneously; the smoke (§6) tests output-norm in isolation first (xs_center off) to get a clean read on the V-escape mechanism.

---

## 4. Predicted-diagnostic table (falsifiable read)

Run = `anchor + use_stock_expert + stock_expert_demean + stock_expert_out_norm=rms`. Compare against the observed pathology run (no norm).

| Diagnostic | Pathology (no norm) | **A: signal exists → sharpens** | **B: no signal → honest death** | **C: norm reopened (LN-affine bug)** |
|---|---|---|---|---|
| `stock_expert_norm` | 0.055 → **88** | **stays O(1)** (≈ √D · const, flat) | stays O(1) | **re-climbs** (γ grows) → REJECT variant |
| `stock_attn_entropy_norm` | **0.9999** (frozen) | **drops < 0.999**, trends down over epochs | stays ≈0.9999 (never sharpens) | stays ≈0.9999 |
| `stock_winner_ratio` | → **1.0** | moderate (0.2–0.6), competes honestly | → 0 | → high again (magnitude back) |
| `stock_ratio` (gate) | high (~0.45, propped by fake signal) | stable/moderate, justified | **→ < 0.05** (router abandons) | high (fake) |
| `stock_contrib_norm` | huge | O(1), comparable to time/factor | → 0 | huge again |
| `expert_cosine_ts/fs` | ~ (near 1, redundant) | **drops** (decorrelates → new info) | n/a (contrib→0) | ~1 (redundant) |
| valid `rank_ic` vs anchor | ~flat or worse | **> anchor** (the only promotion signal) | = anchor (null, harmless) | ≤ anchor |

**Decision rule:**
- **Outcome A** (entropy drops AND norm stays O(1) AND rank_ic ≥ anchor): the cross-stock signal was real and the V-escape was masking it → promote, widen seeds.
- **Outcome B** (entropy frozen, norm O(1), gate → 0, rank_ic = anchor): there is no recoverable cross-stock signal; the expert dies honestly and harmlessly. This **kills the stock-expert axis cleanly** (the right negative result — matches the project's prior pattern of axes settling NULL).
- **Outcome C** (norm re-climbs): you used affine LN; switch to RMSNorm-no-scale (variant b).

The key falsifier the original run lacked: **with magnitude pinned, entropy is the ONLY remaining knob.** If it still doesn't move, Path A genuinely has no gradient (ties to §5).

---

## 5. Gradient view — does closing Path B redirect gradient into Q/K, or just kill the expert?

**It does NOT automatically redirect gradient into Q/K. It REMOVES Path B's gradient; whether Path A's gradient then fires depends on a covariance that output-norm does not change.**

From the earlier warming analysis: attention sharpens only if `Cov_j(g_ij, kr_j) ≠ 0`, where `g_ij` is the per-target gradient signal arriving at the attention output and `kr_j` is the key-relevance of stock `j` for target `i`. The Q/K logits get gradient

```
∂L/∂logit_ij ∝ A_ij · (g_i · (v_j − out_i))
```

For uniform `A`, this is nonzero **iff** the value vectors `v_j` covary with the incoming gradient direction `g_i` across `j` — i.e. **iff different stocks carry different information that the loss wants separated.**

**What output-norm changes:** it removes the term in `∂L/∂(Wv,Wo)` that previously rewarded growing `s` (because post-norm, ∂‖out‖/∂s ≈ 0). It does **not** touch `Cov_j(g_ij, kr_j)` — that covariance is a property of the *data and the embeddings*, not of the output scale. Output-norm is **scale-invariant**, so it leaves the *directional* gradient on Q/K **untouched** while zeroing the *magnitude* gradient on V/O.

**Therefore:**
- If `Cov_j(g_ij, kr_j) ≠ 0` (signal exists): previously SGD could *ignore* it and take cheap Path B. Now Path B is gone, so the **only** remaining loss-reducing move is to sharpen → the Q/K gradient that was always present but *dominated/short-circuited* by the cheaper V-escape now becomes the steepest descent direction → **Outcome A.** Output-norm doesn't *create* the Q/K gradient; it *removes the competing cheaper sink* so the existing Q/K gradient is finally followed.
- If `Cov_j(g_ij, kr_j) ≈ 0` (no signal): there is no Q/K gradient to redirect into. Closing Path B leaves nothing → the de-mean term can't be reduced any way → router gets no benefit from `w_stock` → gate decays → **Outcome B (honest death).**

So output-norm is a **clean disambiguator**: it converts the ambiguous pathology ("is the stock signal real or is the expert just cheating with magnitude?") into a binary read. It will **either** force genuine sharpening (if signal) **or** cleanly kill the expert (if not). It will **not** produce a fake-alive state — that was exactly the affine-γ failure mode (Outcome C) to avoid.

---

## 6. Concrete design — proposed (UN-APPLIED) diff

Flag name: **`stock_expert_out_norm`**, a mode **string** (mirrors the existing `temporal_readout` string-mode pattern, not a bare bool, so all variants from §2 are reachable for falsification without new flags).

Accepted values: `"none"` (default, baseline byte-identical), `"rms"` (RMSNorm-no-scale — recommended), `"unit"` (per-token unit-norm), `"ln"` (LayerNorm-no-affine), `"ln_affine"` (LayerNorm WITH affine — included only to *demonstrate* Outcome C, not for promotion), `"block_rms"` (the §1b B-block scalar-RMS gentle variant). `"cap:<float>"` for the scalar cap (variant d).

### 6a. `module/utils/model_configuration.py`

Add to `__init__` signature (after `stock_expert_xs_center` at line 89):

```python
            # h-20260617: normalize the stock-expert OUTPUT before de-mean/fusion to close the
            # "V-amplification escape" (uniform attention + ‖Wv‖‖Wo‖→88 instead of sharpening).
            # Modes: "none"(default,baseline) | "rms" | "unit" | "ln" | "ln_affine" | "block_rms"
            #        | "cap:<float>". Scale-invariant modes (rms/unit/ln) foreclose the escape;
            # ln_affine REOPENS it via gamma (diagnostic only). Default "none" => baseline unchanged.
            stock_expert_out_norm: str = "none",
```

Add to the assignment block (after line 220):

```python
        self.stock_expert_out_norm = str(stock_expert_out_norm or "none").strip().lower()
```

### 6b. `module/architecture/moe_block.py` — insert between line 231 and 236

```python
            out_stock = rearrange(out_stock, "(t n) b d -> b t n d", t=T, n=N)
            # h-20260617: cap the stock-expert OUTPUT magnitude BEFORE de-mean to foreclose the
            # V-amplification escape (uniform attention + ||Wv||||Wo||->88). Scale-invariant modes
            # remove magnitude as a degree of freedom => the only way to survive the de-mean is to
            # SHARPEN attention (Path A) or the router abandons the expert (honest death).
            # MUST run before de-mean: de-mean of a uniform field is ~0, normalizing 0/||0|| is
            # ill-conditioned, so always normalize first.
            _onorm = getattr(self.config, "stock_expert_out_norm", "none")
            if _onorm and _onorm != "none":
                _eps = 1e-6
                if _onorm == "rms":
                    # RMSNorm over D, per (b,t,n) token; NO learnable scale (parameter-free =>
                    # nothing to re-grow into a renamed gain). Recommended.
                    out_stock = out_stock * torch.rsqrt(
                        out_stock.pow(2).mean(dim=-1, keepdim=True) + _eps
                    )
                elif _onorm == "unit":
                    # per-token unit L2 over D (hardest scale-invariant cap).
                    out_stock = out_stock / out_stock.norm(dim=-1, keepdim=True).clamp_min(_eps)
                elif _onorm == "ln":
                    # LayerNorm over D, NO affine (mean-center + unit-var, parameter-free).
                    _mu = out_stock.mean(dim=-1, keepdim=True)
                    _var = out_stock.var(dim=-1, keepdim=True, unbiased=False)
                    out_stock = (out_stock - _mu) * torch.rsqrt(_var + _eps)
                elif _onorm == "ln_affine":
                    # LayerNorm WITH affine — INCLUDED ONLY to demonstrate Outcome C (gamma
                    # reopens the escape). Do NOT promote. Lazily build the affine on first use.
                    if not hasattr(self, "_stock_out_ln"):
                        self._stock_out_ln = nn.LayerNorm(D).to(out_stock.device)
                    out_stock = self._stock_out_ln(out_stock)
                elif _onorm == "block_rms":
                    # §1b gentle variant: one scalar RMS over the whole (B,D) cross-stock block
                    # per (t,n); allows one stock to dominate but caps TOTAL cross-stock energy.
                    _scale = out_stock.pow(2).mean(dim=(0, 3), keepdim=True).sqrt()  # [1,T,N,1]
                    out_stock = out_stock / (_scale + _eps)
                elif _onorm.startswith("cap:"):
                    # variant (d): clip ceiling only; below cap the V-escape is still local-cheapest.
                    _c = float(_onorm.split(":", 1)[1])
                    _n = out_stock.norm(dim=-1, keepdim=True).clamp_min(_eps)
                    out_stock = out_stock * (_c / _n).clamp_max(1.0)
                else:
                    raise ValueError(f"Unsupported stock_expert_out_norm={_onorm!r}")
            if getattr(self.config, "stock_expert_demean", False):
```

Note the `ln_affine` branch lazily constructs `nn.LayerNorm` so no `__init__` change is needed; the recommended `rms` path is pure-functional, allocates nothing, and is byte-identical to baseline when the flag is `"none"`.

### 6c. `work_flow.py` whitelist — add after line 309

```python
    "stock_expert_xs_center",
    "stock_expert_out_norm",
]
```

(This is the `_ALLOWED_*` provenance/whitelist list around lines 307-310 where the other stock flags live.)

---

## 7. Staged smoke command (DESCRIBE only — do NOT run; GPU busy)

Use the established `QIB_*_OVERRIDES_JSON` protocol (never edits `work_flow.py`). Single-seed, short-epoch proxy to read the diagnostics before any sweep:

```
# Stage 1 — clean V-escape read: out_norm=rms, demean ON, xs_center OFF, seed42, scale=0.5
QIB_MODEL_CONFIG_OVERRIDES_JSON='{"use_stock_expert": true,
                                  "stock_expert_demean": true,
                                  "stock_expert_xs_center": false,
                                  "stock_expert_out_norm": "rms"}'
  → 25ep (or 8ep early-read), seed 42, scale 0.5; baseline = the no-norm pathology run + anchor.
```

**Primary read** (epoch-by-epoch from diag): `stock_expert_norm` must stay O(1) (NOT climb to 88) — confirms Path B closed. Then `stock_attn_entropy_norm`: drops below 0.999 ⇒ Outcome A (sharpening); frozen at 0.9999 + `stock_ratio`→<0.05 ⇒ Outcome B (honest death). Either is a clean result.

**Stage 2 (only if Stage 1 = Outcome C / ambiguous)**: re-run with `"ln_affine"` to confirm γ reopens the escape (norm re-climbs) — this *validates the mechanism* and justifies parameter-free as the rule. Not for promotion.

**Stage 3 (only if Stage 1 = Outcome A)**: hand to `/quant-minimal-repro` then `/quant-ablation-plan` for multi-seed (kill-check with fresh seeds — n=1/n=3 promotion is unreliable per project memory).

Do not combine with `xs_center` until output-norm's effect is isolated.

---

## 8. Recommendation

**Smoke first: `stock_expert_out_norm="rms"`** (RMSNorm over D, no scale), demean ON, xs_center OFF, seed 42, scale 0.5.

- It is the only variant that *both* forecloses Path B (scale-invariant) *and* has no parameter to re-grow (unlike LN-affine, which reopens the escape via γ).
- It is parameter-free, allocation-free, byte-identical to baseline when off, and fully reversible.
- It converts the ambiguous pathology into a binary falsifiable read: genuine sharpening (Outcome A → promote) or honest death (Outcome B → kill the stock axis cleanly). It will not manufacture a fake-alive state.
- Output-norm does **not** create Q/K gradient; it removes the cheaper V/O sink so the *pre-existing* sharpening gradient (if `Cov_j(g_ij,kr_j)≠0`) becomes the steepest move. If that covariance is zero, the expert dies honestly — which is itself the correct, useful negative result.
