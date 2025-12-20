# RST-MoE: Regime-Conditioned Structural Routing for Stock Ranking

## 1. Problem: Non-stationary regimes break static reasoning
Financial markets are regime-switching systems. A single architecture trained on long histories tends to
average conflicting dynamics, producing a model that is "reasonable on average" but weak in specific regimes.
This is a core mismatch between static inductive bias and dynamic market behavior.

## 2. Where prior art falls short (positioning vs SOTA)
- MASTER (AAAI 2024): market-guided attention reweights inputs but keeps a single backbone; it does not separate
  temporal vs factor reasoning, so the inductive bias remains static.
- DoubleAdapt (KDD 2023): online meta-updates adapt weights but are reactive and compute-heavy.
- TRA (KDD 2022): routes by temporal similarity without explicit regime signals or axis-specific experts.
- StockMixer / TSMixer: efficient mixing, but the architecture is fixed and not regime-conditioned.
These methods mostly change parameters or attention weights, not the reasoning structure itself.

## 3. Key insight
Regimes change the relative value of temporal patterns versus factor interactions.
Instead of one monolithic backbone, we separate these two inductive biases and let a regime signal
control their mixture at each layer.

## 4. Method summary (what the model actually does)
RST-MoE uses two axis-specific experts and a regime-conditioned router:

- Input: factor time series x in R^{B x T x N} (B stocks, T window, N factors).
- Regime embedding r: from leak-free external macro state (recommended for t+5) or internal batch stats (fallback).
- Time expert: self-attention over time for each factor (per-factor temporal modeling).
- Factor expert: self-attention over factors for each time step (cross-factor interaction modeling).
- Router (per layer): soft mixture weights w = softmax(g(r, layer_summary)).

The per-layer output is:
f(x) = w_time(r) * f_time(x) + w_factor(r) * f_factor(x).

We also condition representations via:
- Regime-adaptive time-scale embedding (tau-gated time positions).
- Regime-adaptive factor FiLM applied after LayerNorm (so it is not canceled by Pre-LN).
- Optional differentiable feature selection (ablation only).

All macro features are computed up to day T and used to predict T+1..T+k (no look-ahead; PCA fit on train only).

## 5. Novelty (precise and testable)
1) Structural specialization with soft routing: axis-specific experts mixed by regime signals,
   not a single backbone with input gating.
2) Regime-conditioned representations: adaptive time scale and factor FiLM applied post-LN to
   change how the model processes signals, not just which inputs it weights.
3) Mechanism evidence: gate entropy/time_ratio diagnostics and regime-bucket analysis,
   not only aggregate accuracy.

## 6. Evidence standard for KDD oral
We will show:
- Strong baselines under identical protocol (data, labels, costs).
- Structural routing ablations: time-only, factor-only, fixed-50/50, no-regime router, full model.
- Regime-bucket analysis: performance and gate behavior track interpretable market states.
- Stability: five seeds, HAC t-stats, and collapse rate checks (gate entropy/time_ratio).

## 7. Scope and limits (explicit)
- Routing is a soft mixture, not discrete hard switching.
- Experts model time and factor axes; there is no explicit cross-stock graph module.
- Internal regime stats require same-day cross-section batches; external macro is preferred for t+5.
- We do not perform online retraining; the method adapts via routing and conditioning only.
