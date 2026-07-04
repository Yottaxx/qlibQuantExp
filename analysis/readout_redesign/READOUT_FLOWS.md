# Temporal-readout designs — per-design tensor flow (archive)

Card: `time-readout-bonus-20260607` (from-scratch temporal-readout residual). Code: `module/quant_moe_model.py`
(`forward` readout block + `__init__` params), flagged by `config.temporal_readout` (allow-list in
`module/utils/model_configuration.py`). Verifier: `scripts/verify_temporal_readout.py`.

**Verification (run on a synthetic `[B,T,N,D]` batch, backbone weights copied from the control):**

| design | check | result |
|---|---|---|
| d3cid | identity-start `max|Δ vs control|` | `0.00e+00` ✓ ; `tr_W_last_frac=1.0` |
| d3cin | identity-start `max|Δ vs control|` | `0.00e+00` ✓ ; `tr_W_last_frac=1.0` |
| d3mix | identity-start `max|Δ vs control|` | `0.00e+00` ✓ ; `tr_collapse_last_frac=1.0`, `tr_timemix_gain=0`, `tr_chanmix_gain=0` |
| d1pma | finite+shape | ✓ ; `g=0.5`, `tr_attn_last_frac=0.127≈1/8`, `tr_attn_entropy=2.079=ln8` (uniform) |
| duala | finite+shape | ✓ ; `g=0.5` |
| dualb | finite+shape | ✓ ; `g=0.5` |

**NO `+20` saturated softmax bias anywhere.** Softmax designs (d1pma/duals) use a gated-residual
(`tr_b_t=0`, learnable `tr_gate` → `g=0.5`); linear designs (d3cid/d3cin/d3mix) use one-hot-last LINEAR
weights (no softmax → no saturation, AMP-safe with no fp32 cast). fp32-cast logits before softmax/sigmoid
(AMP) only in d1pma/duals. Dims: B=stocks, T=8, N=158, D=64; `h:[B,T,N,D]` after `final_norm`.

Shared prefix (all): `x[B,T,N] → embed → MoE blocks(time⊕factor expert) → h=final_norm(h) [B,T,N,D]`

## Capacity ladder (the experimental design)
`control(select last-step) → d3cid(CI linear) → d3cin(per-factor linear) → d3mix(full mixing)` is an
ablation ladder of *temporal-readout capacity*. The read at analysis time: does each added rung beat the
previous (signal) or merely widen the train/valid gap (capacity without signal — the project's #2 finding,
gen-gap=capacity)? d1pma/duals are the attention/dual-branch alternatives.

```
control (default, last-step)
 h_last = h[:,-1,:,:]                              [B,N,D]
 h_pooled, faw = factor_pooling(h_last)            [B,D],[B,N]
 score = head(h_pooled).squeeze                    [B]        head:Linear(64,1)
```
```
d3cid  (per-D linear over T — CI baseline rung, NO mixing)    params: tr_A[T,D]=512
 z   = einsum("btnd,td->bnd", h, tr_A)             [B,N,D]    tr_A one-hot-last (LINEAR init)
 h_pooled, faw = factor_pooling(z)                 [B,D]
 score = head(h_pooled)                            [B]
 diag: tr_W_last_frac = (|tr_A|.mean(dim=D))[-1]/Σ
 identity-start: tr_A[-1,:]=1 ⇒ z=h[:,-1] ⇒ score==control (EXACT)
```
```
d3cin  (per-factor linear over T — each factor N its own filter)   params: tr_A_N[N,T]=1264
 z   = einsum("btnd,nt->bnd", h, tr_A_N[:N,:])     [B,N,D]    tr_A_N one-hot-last ([:, -1]=1), shared over D
 h_pooled, faw = factor_pooling(z)                 [B,D]
 score = head(h_pooled)                            [B]
 diag: tr_W_last_frac = (|tr_A_N|.mean(dim=N))[-1]/Σ
 identity-start: tr_A_N[:,-1]=1 ⇒ z=h[:,-1] ⇒ score==control (EXACT)
```
```
d3mix  (TSMixer time-mix + channel-mix → learned collapse)   params: tr_tw[T,T]≈72 + tr_cw1[D,2D]≈8320 + tr_cw2[2D,D]≈8256 + tr_collapse[T]8 + 2×LN ≈ 17k
 X   = h                                           [B,T,N,D]
 # time-mix (mix across T, shared over D,N), residual:
 Xt  = X + tr_tw( LN_t(X) )                         [B,T,N,D]   tr_tw:Linear(T,T) on transposed T axis; zero-init ⇒ Xt=X at start
 # channel-mix (mix across D, shared over T,N), residual:
 Xc  = Xt + tr_cw2( relu( tr_cw1( LN_c(Xt) ) ) )    [B,T,N,D]   cw1:D→2D, cw2:2D→D ; tr_cw2 zero-init ⇒ Xc=Xt at start
 # learned temporal collapse T→1 (LINEAR, no softmax):
 z   = einsum("btnd,t->bnd", Xc, tr_collapse)       [B,N,D]     tr_collapse one-hot-last
 h_pooled, faw = factor_pooling(z)                 [B,D]
 score = head(h_pooled)                            [B]
 diag: tr_collapse_last_frac=|tr_collapse[-1]|/Σ ; tr_timemix_gain=‖Xt−X‖/‖X‖ ; tr_chanmix_gain=‖Xc−Xt‖/‖Xt‖
 identity-start: zero-residual (tr_tw=0, tr_cw2=0) + tr_collapse[-1]=1 ⇒ Xc=X, z=h[:,-1] ⇒ score==control (EXACT)
```
```
d1pma  (gated attention over T)         params: tr_q[D]64 + tr_b_t[T]8 + tr_gate1 = 73
 h_time = h.mean(dim=2)                            [B,T,D]    (mean over N)
 logits = (h_time.float()@tr_q.float())/√D + tr_b_t.float()   [B,T]  fp32   ← b_t=0 (NO +20)
 a      = softmax over T (logits)                  [B,T]
 attn   = einsum("bt,btnd->bnd", a, h)             [B,N,D]    (shared temporal weights, per factor)
 g      = sigmoid(tr_gate.float())                 scalar     tr_gate=0 ⇒ g=0.5 (responsive)
 z      = (1-g)·h[:,-1] + g·attn                   [B,N,D]    ← gated-RESIDUAL nests last-step
 h_pooled, faw = factor_pooling(z)                 [B,D]
 score  = head(h_pooled)                           [B]
 diag: tr_gate_g, tr_attn_last_frac=a[:,-1].mean, tr_attn_entropy
```
```
duala  (dual: time-attn ⊕ global-mean)  params: d1pma(73) + tr_head2[2D,1]=129
 z_T  = [d1pma temporal agg]                       [B,N,D]
 zTp, faw = factor_pooling(z_T)                    [B,D]      ← time branch (pooled)
 zF   = h.mean(dim=(1,2))                          [B,D]      ← global mean over (T,N)
 score = tr_head2(cat[zTp, zF]).squeeze            [B]        tr_head2:Linear(128,1)
 diag: tr_head2_zT_norm, tr_head2_zF_norm
 identity-start: tr_head2.weight[:,D:]=0 (post_init) ⇒ zF half off at init
```
```
dualb  (dual: time-attn ⊕ attn-over-N)  params: d1pma(73) + tr_qN64 + tr_bN158 + tr_head2 129
 z_T  = [d1pma temporal agg] ; zTp, faw = factor_pooling(z_T)   [B,D]
 h_fac = h.mean(dim=1)                             [B,N,D]    (mean over T)
 lN   = (h_fac.float()@tr_qN.float())/√D + tr_bN[:N].float()   [B,N] fp32
 aN   = softmax over N (lN)                        [B,N]
 zF   = einsum("bn,bnd->bd", aN, h_fac)            [B,D]      ← factor-attention branch
 score = tr_head2(cat[zTp, zF]).squeeze            [B]
 diag: tr_head2_zT_norm, tr_head2_zF_norm ; identity-start: tr_head2.weight[:,D:]=0
```

Invariants: every design ends at `[B]` via `head` (or `tr_head2` for duals); `factor_pooling` is KEPT in
all paths ⇒ `factor_pool_weights` non-None ⇒ the diagnostics/portfolio pipeline never breaks.

## Engineering notes
- **post_init ordering (R1):** `post_init()` re-inits every `nn.Linear`/`nn.LayerNorm`. Identity-start
  overrides for `tr_tw`/`tr_cw2` (zero) + `tr_collapse` (one-hot) + `tr_head2` zF-half (zero) are applied
  AFTER it. Bare `nn.Parameter`s (`tr_A`/`tr_A_N`/`tr_q`/`tr_b_t`/`tr_gate`/`tr_qN`/`tr_bN`) are NOT modules
  → untouched by `post_init`.
- **AMP fp16 (R2):** softmax/sigmoid run in fp32 under the live `amp_fp16` autocast: softmax is on
  autocast's fp32-promotion policy list, and sigmoid's input is fp32 via `tr_gate.float()`. NOTE (review
  2026-06-07): the `.float()` on the *matmul operands* (`h_time.float() @ tr_q.float()`) is a **no-op** —
  autocast re-casts matmul inputs to fp16 regardless — so the dot product itself runs in fp16. This is
  numerically benign here (logits are 1/√D-scaled, zero bias, 64-d dot, no overflow) and does NOT affect
  results. The linears (d3cid/d3cin/d3mix) carry no softmax/sigmoid → genuinely cast-free.
- **N≤num_alphas (R4):** `tr_A_N`/`tr_bN` are sized to config `num_alphas`; sliced `[:N]` at runtime.
- **duals `factor_pool_weights` is TIME-BRANCH-ONLY (review 2026-06-07):** for duala/dualb,
  `factor_pool_weights` = the factor-pool weights over the **time branch** `z_T` only; the `zF` branch
  (global-mean for duala, attn-over-N for dualb) co-produces ~half the score via `tr_head2` but is NOT
  represented. The **stock score is correct** (both branches feed `tr_head2`); only the exported
  factor-pool heatmap/Top-K is partial. **Do not draw factor-importance conclusions for duals from this
  field.** (dualb's over-N weights `aN` could be surfaced as a separate `[B,N]` diag if needed.)

## Known issues / deferred fixes (review 2026-06-07 — apply at the next NON-LIVE window)
These edit `work_flow.py`/the model, which the running pipeline re-reads — do NOT apply while a matrix run
is live. None affects training correctness or the trained weights.
- **Provenance (medium):** `temporal_readout` (and the pre-existing `pool_n_heads`) are missing from
  `MODEL_CONFIG_KEYS_FULL` in `work_flow.py`, so `run_conf_resolved` + the `[Model Full]` MLflow note
  record all designs as `temporal_readout=''` (=control). The model is still BUILT correctly (design
  injected via `_deep_update`) and saved faithfully (pickled), and the true design is recoverable from the
  `QIB_RUN_SETTING` tag (`readout_full_<design>_seed42_scale05`) + the pickled config + the `run_conf`
  (non-resolved) blob. Fix: add `"temporal_readout"` after `pooling_alpha` (~work_flow.py:301).
- **fp16 matmul `.float()` no-op (cosmetic):** drop the misleading `.float()` on the attn-logit matmul
  operands (and add a comment) OR wrap logits+softmax in `torch.autocast(...,enabled=False)` per
  `regime_encoder.py:138-145`. Defer so all 6 designs in the current matrix share identical numerics.
- **Optional model-side diagnostics:** emit `tr_effective_last_frac=(1-g)+g·a[:,-1]` (gate-aware
  "moved-off-last-step" signal) and/or a `zF`-branch factor-attribution field for duals. The compare
  script already computes the gate-aware verdict from the logged `tr_gate_g`, so this is optional polish.

## Planned: init-ablation arm — does one-hot-last TRAP the linear readout? (design panel 2026-06-07)
**Question (owner):** `tr_W_last_frac` drifts steadily off the one-hot-last init while live d3cid valid rank_ic
(~0.066–0.072) sits *below* control (0.0811). Is one-hot-last a sticky basin that biases us toward a false
"last-step is optimal" conclusion — would a SOTA/diverse init explore more?
**Verdict (4-agent panel + judge):** the basin-escape concern is *legitimate but unfalsifiable from the one-hot
run alone*, so it warrants an init-ablation arm — but **(a) NOT now / NOT a replace** (one-hot-last IS the control;
replacing forfeits the exact-nesting `z==h[:,-1]` and blinds the `tr_W_last_frac` diagnostic), and **(b) NOT raw
Kaiming/Xavier**. The linear readout is an **aggregation over T** (`z=Σ_t w_t·h`), not a projection: Kaiming/Xavier
are zero-mean (destroy the DC/level that carries the temporal signal) and scale-uncontrolled (`Var(z)` ~1×–2.8×,
fighting the head's label-variance calibration via AdaptivePooling's un-normalized mean branch). The live read is
also *mechanism-negative* so far (moving off last-step AND underperforming control — the τ-collapse pattern).
- **Applies to ALL THREE temporal-AGGREGATION params** — `tr_A[T,D]` (d3cid), `tr_A_N[N,T]` (d3cin), AND
  `tr_collapse[T]` (d3mix). All three are one-hot-last weighted sums over T (the T→1 reduction) and carry the same
  scale/DC properties; `tr_collapse` is the direct analog of `tr_A` (shared across D). The d3mix **mixing** transforms
  (`tr_tw` zero-residual, `tr_cw1` `normal(0.02)`, `tr_cw2` zero-residual) are a SEPARATE question ("active mixing at
  init", standard TSMixer practice) and stay as-is; the attentions (d1pma/duals) already soft-start (uniform softmax,
  g=0.5) so there is no one-hot to diversify there.
- **Correct diverse init:** **sum-preserving uniform-mean** `w_t = 1/T + ε`, `ε~N(0, 0.02²)` per channel/factor
  (or `softmax(small_logits)`≈uniform / row-normalized Dirichlet) ⇒ `Σ_t w≈1` ⇒ `z`-scale = `h[:,-1]`-scale (head
  calibration honored), full-rank diversity across all 8 lags. NOT raw Kaiming/Xavier (zero-mean ⇒ destroys the
  DC/level; scale-uncontrolled on a T-aggregation).
- **Implementation (next NON-LIVE window):** add a `temporal_readout_init ∈ {onehot_last(default), uniform_mean}`
  flag in `quant_moe_model.py __init__` (new branch; default preserves current behavior; applies to whichever of
  `tr_A`/`tr_A_N`/`tr_collapse` the active design uses). Extend `run_readout_matrix_scale05.sh` with an `INITS` loop.
- **Matrix (CONDITIONAL — gate on the finished run):** run ONLY if a linear design finished *moved-with-promise*
  (paired ΔRankIC flat-or-positive, not clearly negative). Design set `{d3cid, d3cin, d3mix}`. Staged like everything
  else: first `uniform_mean × seed42` (3 new runs ≈ 15h) compared against the existing `onehot_last × seed42` cells
  (free, from this matrix) + control; if uniform shows promise, escalate to `× seeds{42,43,44}` for BOTH inits
  (onehot × seed43/44 are NOT free — the main matrix is seed42-only). Compare vs the SAME `CTRL_RUN` 42/43/44.
- **Falsification contract:** (a) uniform-init → same tr_W profile + same ΔRankIC band ⇒ one-hot does NOT trap;
  last-step robustly ~optimal ⇒ **close the init lever**. (b) uniform-init → ΔRankIC ≥ +0.003 over n=3, clean
  direction ⇒ one-hot WAS trapping ⇒ escalate to n≥6 + HC kill-check before any claim.
- **If all 6 designs land null** (like the prior readout findings): **skip the arm** — a dead readout doesn't merit
  an init study (cf. factor-pool reweighting = documented dead lever). Diversity-for-promotion is across-seed, not
  within-init.
