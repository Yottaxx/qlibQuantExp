# MASTER Mechanism — Faithful Transcription & Comparison to RST-MoE

> **⚠ LOSS/LABEL CORRECTION (2026-06-24).** The comparison row (≈line 144) says "We default to ListMLE/IC/
> rank composite" and "CSRankNorm label" — **both wrong.** Code-verified: `main_loss="mse"`,
> `mse_normalize=False` ⇒ **plain masked MSE** (`work_flow.py:167,175`); label = **`CSZScoreNorm` robust**
> (`work_flow.py:107`; the model-side comment `quant_moe_model.py:302` "已是 CSRankNorm" is also stale).
> So **our objective ≈ MASTER's exactly** (both masked MSE on a CSZScore label, line 123-124) — the
> loss/label delta **dissolves**, which *reinforces* the conclusion that cross-stock **placement** is the
> sole remaining lever. The "uniform = loss-neutral parking spot" remark (≈line 149) should read
> "router-exit + target-mean≈0" (loss-independent). See `PORTFOLIO_ALIGNED_PLAN.md` §7 note.

**Paper:** Li, Liu, Shen, Wang, Chen, Huang. *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.* AAAI 2024. arXiv:2312.15235.

**Ground-truth code:** `SJTU-DMTai/MASTER` (`master.py`, `base_model.py`) — cloned and read directly. Where paper text and code disagree on details, **the code is authoritative** and is what this note transcribes.

---

## 0. Notation & tensor flow

| symbol | meaning | value in repo |
|---|---|---|
| `N` (`M` in paper) | number of stocks in the daily cross-section | variable (a whole day's universe) |
| `T` (`τ`) | look-back window length | 8 |
| `F` (`d_feat`) | per-stock per-day feature dimension fed to the encoder | 158 |
| `F'` | market-status feature dimension (the gate input slice) | 63 |
| `D` (`d_model`) | embedding width | 256 |

Input tensor to `MASTER.forward` is **`x : [N, T, F_total]`** where `F_total = F + F' (+1 label, stripped earlier)`. So **the first axis IS the stock axis** — one forward pass processes an entire day's cross-section as the batch. This is the single most important structural fact for the comparison below.

Overall flow (from `MASTER.forward` + `self.layers = nn.Sequential(...)`):

```
x [N,T,F_total]
  ├─ src        = x[:, :, :gate_start]                      # [N,T,F]
  ├─ gate_input = x[:, -1, gate_start:gate_end]             # [N,F']  (LAST time step only)
  └─ src = src * gate(gate_input).unsqueeze(1)              # market-guided feature gate
        → Linear(F→D)                                        # [N,T,D]   feature encoder
        → PositionalEncoding (additive sinusoidal over T)    # [N,T,D]
        → TAttention   (intra-stock, attention over T axis)  # [N,T,D]
        → SAttention   (inter-stock, attention over N axis)  # [N,T,D]
        → TemporalAttention (pool over T → 1 vector/stock)    # [N,D]
        → Linear(D→1)                                         # [N,1] → squeeze → [N]
```

---

## 1. Market-Guided Gating (`Gate`)

The market-status vector is **not a separate global vector** in the code — it is simply the slice of each stock's own feature row at the **last time step** covering columns `[gate_input_start_index : gate_input_end_index]` (the 63 market-information columns). Per the paper these columns are market index price/return statistics and market index volume mean/std over the past `d'` days; they are identical across stocks on a given day, so in practice `gate_input` is the day's market state broadcast per row.

Gate module (verbatim structure):

```python
class Gate(nn.Module):                       # d_input = F' = 63, d_output = F = 158
    def __init__(self, d_input, d_output, beta=1.0):
        self.trans = nn.Linear(d_input, d_output)   # WITH bias
        self.d_output = d_output
        self.t = beta
    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)   # softmax OVER THE F (feature) DIM
        return self.d_output * output                     # multiply by F
```

**Exact gating equation** (with `m_τ ∈ ℝ^{F'}` the market-status row):

```
α(m_τ) = F · softmax_β( W_α m_τ + b_α )  ∈ ℝ^F ,   Σ_f α_f = F
```

- It is a **softmax over the F feature dimensions, scaled by F** (mean-1 reweighting), NOT a sigmoid, NOT a per-feature independent gate. The `·F` makes the no-op reference point `α ≡ 1` (uniform). `β` (temperature) sharpens/softens; smaller β ⇒ stronger selection.
- **Where applied:** multiplicatively on the **raw input features**, broadcast across all T time steps, **BEFORE the `Linear(F→D)` feature encoder**:
  `x̃_{u,t} = α(m_τ) ⊙ x_{u,t}` , i.e. `src = src * gate(gate_input).unsqueeze(1)` → shape `[N,1,F]` * `[N,T,F]`.

---

## 2. Intra-Stock / cross-time Aggregation (`TAttention`) — attention over T

Operates with the **stock axis as the batch** and attends over the **time axis** (each stock independently). Pre-LN, multi-head by feature-slicing, **NO `1/√d` scaling**:

```python
x = norm1(x)                                  # [N,T,D] LayerNorm
q,k,v = qtrans(x), ktrans(x), vtrans(x)       # Linear(D,D), bias=False ; [N,T,D]
# per head h (slice the D dim into nhead chunks):
A_h = softmax( q_h @ k_h^T , dim=-1 )         # [N,T,T]   *** NO temperature / no /sqrt(d) ***
out_h = A_h @ v_h                             # [N,T,dh]
att = concat_h(out_h)                         # [N,T,D]
xt  = x + att                                 # residual
xt  = norm2(xt)
out = xt + ffn(xt)                            # FFN: Linear(D,D)-ReLU-Drop-Linear(D,D)-Drop, residual
```

- Q/K/V: `[N,T,D]` each. Attention field `T×T` per stock. `nhead = t_nhead`.
- Positional encoding is **fixed sinusoidal**, added to `[N,T,D]` once before TAttention (paper writes `Y_u = ||_t LN(f(x̃)+p_t)`; in code the LN is `norm1` inside TAttention).
- **Scaling: NONE** — `TAttention` divides by nothing (paper's only scaled attention is the inter-stock one).

## 3. Inter-Stock / momentary Aggregation (`SAttention`) — attention over N (stocks)

**This is the load-bearing block for the comparison.** It runs **after** TAttention, on the same `[N,T,D]` tensor, and attends across the **stock axis** independently at each time step.

```python
x = norm1(x)                                  # [N,T,D]
q = qtrans(x).transpose(0,1)                  # [T,N,D]   <-- stock axis becomes the attention axis
k = ktrans(x).transpose(0,1)                  # [T,N,D]
v = vtrans(x).transpose(0,1)                  # [T,N,D]
temperature = sqrt(D / nhead)
# per head h:
A_h = softmax( q_h @ k_h^T / temperature , dim=-1 )   # [T,N,N]   *** scaled, over STOCK axis ***
out_h = A_h @ v_h                             # [T,N,dh]
att = concat_h(out_h).transpose(0,1)          # back to [N,T,D]
xt  = x + att ; xt = norm2(xt) ; out = xt + ffn(xt)   # same residual/FFN pattern
```

- **Q/K/V:** `[T,N,D]` (transpose puts stocks on the attention axis); field is **`N×N` per time step**. `nhead = s_nhead` (paper uses 2). **Scaled by `√(D/nhead)`** (this is the only block that scales).
- **GRAPH / attn_bias: NONE.** Confirmed from code — `SAttention.forward(x)` takes only `x`; there is no adjacency matrix, no industry/sector mask, no relational bias, no edge weights anywhere. The cross-stock relationship is **pure learned self-attention** discovered from data ("momentary correlation"). The paper explicitly markets this as graph-free automatic relation discovery.
- **Placement:** **in the main path, in series, after intra-stock and before temporal pooling.** Every stock's representation at every retained time step is overwritten by a stock-mixing residual. It is **not gated, not optional, not a parallel branch** — 100% of signal flows through it.

## 4. Temporal Aggregation (`TemporalAttention`) — pool over T → one vector/stock

```python
h = trans(z)                                  # Linear(D,D,bias=False) ; z is [N,T,D]
query = h[:, -1, :].unsqueeze(-1)             # [N,D,1]  use LAST step as query
lam = softmax( (h @ query).squeeze(-1), dim=1 )   # [N,T]   weights over time, NO scaling
e_u = (lam.unsqueeze(1) @ z).squeeze(1)       # [N,1,T]@[N,T,D] -> [N,D]
```

Equation: `λ_{u,t} = softmax_t( ⟨W_λ z_{u,t}, z_{u,τ}⟩ )`, `e_u = Σ_t λ_{u,t} z_{u,t}`. Output `[N,D]`.

## 5. Prediction head + loss

- Head: a single `nn.Linear(D, 1)` → `[N]` (the last element of `self.layers`).
- **Loss (`base_model.loss_fn`): plain masked MSE** `mean((pred[mask] - label[mask])^2)`. NOT IC-based, NOT ranking. (IC/RankIC are eval metrics only.)
- **Label normalization: per-day cross-sectional Z-score (`CSZscoreNorm`)** applied to the label each batch (`label = zscore(label)`), with a robust variant that trims the top/bottom 2.5% before computing mean/std. So MSE on a CS-Z-scored label ≈ optimizing cross-sectional fit (correlation-like) implicitly.

## 6. Norm / residual summary

- **Pre-LN** everywhere: each attention block does `x=norm1(x)` first, then `x + attn`, then `norm2`, then `+ ffn`. FFN is `Linear-ReLU-Drop-Linear-Drop` of width `D` (not `4D`).
- Attention dropout is applied to the softmax matrix (per-head `Dropout`).
- Residuals: one around MHA, one around FFN, in each of TAttention/SAttention.

---

## Component-mapping: MASTER → RST-MoE

| MASTER component | Our analog (`module/...`) | Key difference |
|---|---|---|
| Market gate `α=F·softmax_β(W m+b)`, mean-1 reweight on **F before encoder** | `factor_gate` (FiLM γ,β, AdaLN-style, applied **post-LN inside each MoE block**) + optional `feature_mask` (L0 selector) + `regime_embedding` (from `RegimeContextEncoder`, internal stats of x) | Ours is **additive/affine FiLM conditioned on a learned regime vector**, applied repeatedly per block and per-(B,N,D); MASTER is a **single multiplicative simplex gate over the F axis driven directly by the raw market columns**, applied once pre-encoder. We have no `F·softmax` mean-1 feature gate. |
| `Linear(F→D)` + sinusoidal PE | `feature_tokenizer` / `val_proj` + `factor_id_emb` + regime-adaptive `time_embedding` | We tokenize per-(value,factor); MASTER encodes the whole F-row per (stock,t). |
| **TAttention** (intra-stock, over T, no scaling) | `time_expert = ParallelAttention` over T, batch `(B·N)` | Close analog. Ours uses `nn.MultiheadAttention` (**WITH `1/√d` scaling**); MASTER's TAttention has **no scaling**. Ours is one of several **router-gated parallel** experts, not a mandatory series stage. |
| `Linear(F→D)`/factor axis | `factor_expert = ParallelAttention` over N (the **factor/feature** axis, batch `(B·T)`) | **No MASTER counterpart.** MASTER has no attention over a "factor token" axis — it keeps F as a flat vector. Our `factor_expert` mixes *features within a stock-day*, which is a different axis than MASTER's stock mixing. |
| **SAttention** (inter-stock, over N stocks, scaled, **main path, series**) | `stock_expert = ParallelAttention` over the **batch B axis** (`rearrange "b t n d -> (t n) b d"`), optional, **router-gated parallel** | **THE critical delta.** Our cross-stock attention is (a) **optional** (`use_stock_expert`), (b) a **soft-router-weighted parallel branch** competing with time/factor experts via a softmax gate `w_stock`, and (c) attends over the **batch dimension B** (must be a single day's cross-section to be valid). MASTER puts the identical operation **unconditionally in series in the main path**. |
| **TemporalAttention** (last-step-query pool over T) | `temporal_readout` designs + `factor_pooling (AdaptivePooling)` | We default to **last-step + factor pool**; we added learned temporal readouts (identity-start at last step). MASTER's pool is last-step-as-query softmax over T — similar in spirit; ours pools over the **N/factor** axis as the primary readout. |
| `Linear(D→1)`, masked **MSE** on **CS-Zscore** label | `self.head` + `QuantLossFunctions` (default `listmle`, plus cs_ic/cs_mse/ranknet/huber mix) + CSRankNorm label | We default to **ListMLE/IC/rank composite**; MASTER is **pure MSE on CS-Zscore**. Label norm differs (rank vs zscore). |

### Where the inter-stock attention sits — the suspected collapse cause

- **MASTER:** main-path, in series, **ungated, mandatory.** 100% of every stock's representation is rewritten by `residual + SAttention(·)`. There is no scalar gate that can starve it; gradient must flow through it, so it is forced to learn useful cross-stock mixing. Stock axis = the batch's first axis natively.
- **Ours:** the stock expert is a **third peer under a softmax router** (`gate_weights[:,2]`). A scalar sigmoid/softmax gate that starts near `1/3` and **freezes at init** (per project memory) means the branch can be driven to ~0 weight and never recovers; uniform-attention output is a loss-neutral parking spot (hence the de-mean / xs-center hacks already added). The expert competes for share rather than being a mandatory transform — this is consistent with the observed collapse-to-uniform.

---

## Architectural deltas to faithfully replicate MASTER

To make our cross-stock mechanism behave like MASTER's (in priority order):

1. **Move inter-stock attention into the main path, in series, ungated.** Replace the router-gated parallel `stock_expert` with a mandatory `residual + SAttention(x)` stage applied to *every* token, after the temporal (intra-stock) stage and before the temporal pool. Remove the scalar gate entirely — do not let a frozen router decide its share. (MASTER ordering: gate → encoder+PE → T-attn → **S-attn** → temporal-pool → head.)
2. **Replicate the gate as `F·softmax_β` over the feature axis, applied once on raw features before the encoder**, driven directly by the market columns at the last time step — instead of (or in addition to) the per-block FiLM. Keep `β` as a tunable temperature; default reference point is uniform `α≡1`.
3. **Match the attention details:** SAttention uses **`/√(D/nhead)` scaling**; TAttention uses **no scaling**; both are **graph-free** (no `attn_bias`, no adjacency) with **Pre-LN + FFN(width D, ReLU)** and double residual. Our `ParallelAttention` always scales by `1/√d` — fine for S, but note MASTER's T-attn is unscaled. Ensure the cross-stock axis is the **whole daily cross-section** (already an invariant in our daily samplers).

Secondary (for full fidelity, lower priority): switch the temporal readout to MASTER's **last-step-query softmax pool over T** (`λ=softmax_t(⟨W z_t, z_τ⟩)`), and optionally A/B the loss as **pure masked MSE on a CS-Zscore label** vs our ListMLE/IC composite.

---

## Sources (read directly)

- arXiv abstract: https://arxiv.org/abs/2312.15235
- arXiv HTML (full text, equations for gate/intra/inter/temporal/loss): https://arxiv.org/html/2312.15235v1
- Reference code (GROUND TRUTH, cloned `main`): https://github.com/SJTU-DMTai/MASTER — `master.py` (`Gate`, `TAttention`, `SAttention`, `TemporalAttention`, `MASTER`), `base_model.py` (`loss_fn` = masked MSE, `zscore` = CSZscoreNorm label).
