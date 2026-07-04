# SOTA Cross-Stock Interaction Methods — Survey for an Ungated Main-Path Readout Cross-Stock Attention

**Scope.** SOTA methods for CROSS-STOCK (cross-sectional, stock-to-stock) interaction in deep
quant stock-prediction models, read for what we can copy into an **ungated, main-path cross-stock
self-attention placed at the score-forming READOUT** of RST-MoE (csi300 / Alpha158 / t+5
cross-sectional RANKING; ListMLE+IC+MSE; daily batch = one day's ~300 stocks; `d_model=64`,
`n_heads=4`, `T=8`, `N=158` factors). The two failure modes we must defeat are documented in
`analysis/stock_expert/readout_stock_attention_analysis.md`:

- **(F1) Uniform collapse** — cross-stock softmax goes uniform; output = per-day cross-sectional
  mean = a per-day additive constant = rank-invariant = loss-neutral under our location-invariant
  ranking losses. `entropy_norm = 1.0000`.
- **(F2) V-amplification escape** — when a de-mean constraint is imposed, the model satisfies it by
  inflating `‖Wv‖·‖Wo‖` (observed →88) instead of sharpening attention; the resulting RankIC bump
  did not survive into portfolio IR (the RankIC≠IR trap).

This survey is companion to `MASTER_mechanism.md` (faithful MASTER transcription) and
`readout_stock_attention_analysis.md` (our placement gradient argument). It adds the comparative SOTA
landscape and distills concrete, citation-grounded implementation choices.

---

## 0. The one structural fact that organizes everything

Every cross-stock method below differs along four axes that map directly onto F1/F2:

1. **Placement** — where stock-to-stock mixing sits (encoder vs readout; series vs parallel).
2. **Gating** — mandatory main-path vs gated/optional/router-weighted. *This is the single biggest
   predictor of whether the cross-stock block lives or collapses.* Every SOTA method that works puts
   cross-stock mixing **in series on the main path, ungated**. None of them gate it behind a learned
   scalar or a router. Our collapse came from a router-gated parallel placement — an architecture
   **no SOTA method uses**.
3. **Prior** — dense learned attention vs a graph/hypergraph adjacency prior.
4. **Conditioning** — whether a market/context vector modulates the interaction (MASTER gate, DTML
   global context, StockMixer market path).

The dominant SOTA verdict (2021–2024) has moved **away from predefined graphs toward dense,
graph-free, market-conditioned attention discovered from data** (MASTER, DTML, AD-GAT's unmasked
attention, THGNN's price-derived dynamic graph). That directly supports a **dense** readout
cross-stock attention for us, and de-prioritizes building an industry graph for csi300.

---

## 1. MASTER — Market-Guided Stock Transformer (the primary template)

**Paper:** Li, Liu, Shen, Wang, Chen, Huang. *MASTER: Market-Guided Stock Transformer for Stock Price
Forecasting.* AAAI 2024. arXiv:2312.15235. Code: `github.com/SJTU-DMTai/MASTER` (read directly; see
`MASTER_mechanism.md` for the verbatim transcription — not repeated here).

**Cross-stock placement.** `SAttention` ("inter-stock / momentary"), in the **main path, in series,
ungated, mandatory**, after intra-stock `TAttention` and before temporal pooling. Operates on
`[N,T,D]` transposed to put the **stock axis on the attention axis** (`[T,N,D]`); attention field is
`N×N` per time step. 100% of every stock's representation flows through `residual + SAttention(·)`.
There is **no scalar gate, no router, no optional flag** — gradient is forced through it. This is the
exact opposite of our router-gated parallel stock_expert and is *the* reason MASTER does not collapse
(no zero-cost exit exists).

**Dense vs graph.** Pure learned self-attention, **graph-free** — no adjacency, no sector mask, no
attn_bias anywhere in the code. The paper markets this as automatic, graph-free "momentary relation"
discovery.

**Normalization / residual / init.** **Pre-LN** throughout: `x = norm1(x)` → `x + attn` → `norm2` →
`x + FFN`. Double residual (one around MHA, one around FFN). FFN width = `D` (not 4D), `Linear-ReLU-
Drop-Linear-Drop`. Standard residual init (no ReZero/LayerScale — but it doesn't need one because the
block is mandatory and pre-LN with vanilla residual is already identity-stable at init).

**Attention temperature / scale.** `SAttention` uses temperature `√(D/nhead)` (i.e. the standard
per-head `1/√d_head` scale). Notably MASTER's `TAttention` (intra-stock) uses **NO** scaling — the
only scaled block is the inter-stock one. `s_nhead = 2` heads for the inter-stock block. Attention
dropout applied to the softmax matrix.

**Market conditioning (the "market-guided" part).** A `Gate` over the **feature axis** applied to
**raw features before the encoder**: `α(m) = F · softmax_β(W_α m + b_α) ∈ ℝ^F`, a mean-1 simplex
reweighting of the F=158 features driven by the day's market-status columns (63 market index
statistics at the last time step). `β` is a temperature; the `·F` makes `α≡1` (uniform/no-op) the
reference point. Applied multiplicatively, broadcast over T, **once**, before `Linear(F→D)`. This
gate modulates **feature selection**, NOT the inter-stock attention directly — the inter-stock
attention itself is ungated.

**How it avoids collapse.** (a) Mandatory main-path placement (no exit). (b) The loss is masked MSE
on a **per-day cross-sectional Z-scored label** (`CSZscoreNorm`), which is correlation-like and
forces cross-sectional dispersion — uniform output ⇒ flat scores ⇒ bad MSE. (c) Pre-LN + vanilla
residual is stable from scratch. There is **no** explicit anti-collapse regularizer (no entropy term,
no QK-norm); the placement does the work.

**Reported gains (CSI300).** IC 0.064 vs DTML 0.049 (+31%); RankIC 0.076 vs DTML 0.052 (+46%);
IR 2.4 vs 1.7; AR 0.27 vs 0.21. CSI800: IC 0.052 vs 0.039, RankIC 0.066 vs 0.053. (From the arXiv
HTML results table.)

**Applicability to our readout: HIGH (the template).** It is the literal placement
`readout_stock_attention_analysis.md` recommends. Caveat: MASTER places inter-stock attention
*per-time-step before pooling*; our cleanest locus is *post-pool on `[B,D]`* (one MHA, seq=B). The
two are equivalent in spirit (stock↔stock, ungated, in-series); post-pool is cheaper and is the
recommended adaptation.

---

## 2. DTML — Data-axis Transformer with Multi-Level Contexts

**Paper:** Yoo, Soun, Park, Kang. *Accurate Multivariate Stock Movement Prediction via Data-Axis
Transformer with Multi-Level Contexts.* KDD 2021. dl.acm.org/doi/10.1145/3447548.3467297. Code:
`github.com/simonjisu/DTML-pytorch`, `github.com/ceteris11/DTML`.

**Cross-stock placement.** A **data-axis self-attention** (transformer encoder over the *stock*
axis) — the original "stocks-as-tokens" transformer that MASTER descends from. Pipeline: (1) per-stock
**time-axis attention** over an LSTM/GRU sequence → one context vector per stock; (2) **multi-level
context aggregation**; (3) **data-axis transformer encoder** mixing stocks. The data-axis attention
is **main-path, in-series, ungated** — same non-gated mandatory structure as MASTER.

**Multi-level context (the conditioning mechanism — corrected).** DTML forms a **global market
context** from historical market-index data (its own attention-LSTM over the index series) and
combines it with each stock's local context. The aggregation is a **multi-level context vector
combining local (per-stock) and global (market) contexts** — the public descriptions state the
local↔global combination uses a **dot-product / multiplicative interaction** between the local and
global context vectors (i.e. each stock's context is modulated by its alignment with the market
context), *not* a simple `h + β·h_market` additive blend. (Several reimplementations expose a scalar
that scales the index/global contribution; treat the exact constant as implementation-specific. The
load-bearing idea is: **inject a market summary into every stock's token before the data-axis
attention** so the cross-stock attention operates on market-relativized representations.)

**Normalization / residual / scale.** Standard post-/pre-LN transformer encoder (vanilla
`TransformerEncoderLayer` in the PyTorch reimplementations: MHA + LN + residual + FFN, standard
`1/√d_head` scaling). No special anti-collapse trick.

**Loss.** Originally binary movement (classification) + a temperature-scaled ranking/pairwise hinge
in the ranking variant. For us the relevant takeaway is structural, not the loss.

**How it avoids collapse.** Mandatory main-path placement + market-context injection that gives the
data-axis attention something stock-discriminative to attend on. No explicit regularizer.

**Reported gains.** Up to 13.8 pp higher profit vs best competitor; 44.4% annualized in simulation
across six markets. (Superseded by MASTER on CSI300, see §1.)

**Applicability to our readout: HIGH (for the market-conditioning idea).** DTML is the proof that
**injecting a market/global summary token into each stock's representation before cross-stock
attention** materially helps — and we already have a regime/context vector (`RegimeContextEncoder`).
Two concrete imports: (i) the ungated data-axis encoder = our recommended readout block; (ii) the
multi-level-context idea = concatenate/add our regime vector to `h_pooled` before the cross-stock MHA
(see §11-A). The dot-product local↔global form is a stronger variant of MASTER's feature gate.

---

## 3. THGNN — Temporal & Heterogeneous Graph Neural Network

**Paper:** Xiang, Cheng, Shang, Zhang, Liang. *Temporal and Heterogeneous Graph Neural Network for
Financial Time Series Prediction.* CIKM 2022. arXiv:2305.08740. Code:
`github.com/TongjiFinLab/THGNN`.

**Cross-stock placement / prior.** Builds a **company-relation graph per trading day from historical
prices** (dynamic, data-derived — *not* a fixed industry graph), then a **transformer encoder** for
the per-stock temporal representation, then a **heterogeneous graph attention network** (positive /
negative correlation edge types) to mix stocks. Cross-stock mixing is **graph-attention on a
price-derived dynamic adjacency**, main-path, ungated. The key modern lesson: even the graph methods
have abandoned handcrafted/NLP graphs in favor of **dynamic, daily, correlation-derived** graphs.

**Normalization / stability.** Standard GAT + transformer-encoder norms; attention is over graph
neighbors (sparse), which inherently bounds the uniform-collapse failure (a stock attends to a
*selected* neighborhood, not the whole universe) — a different anti-collapse route than dense
attention.

**Applicability to our readout: MEDIUM.** The dynamic price-correlation graph is a plausible *prior*
to bias our dense readout attention (a soft attn_bias from rolling return correlation), but
`readout_stock_attention_analysis.md` and the MASTER/DTML evidence argue dense graph-free works at
least as well on CSI-scale universes and avoids the cost/complexity of daily graph construction. Use
THGNN only if dense collapses persistently (graph prior as a fallback, §11-B).

---

## 4. AD-GAT — Attribute-Driven Graph Attention (momentum spillover)

**Paper:** Cheng, Li. *Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-
Driven Graph Attention Networks.* AAAI 2021. Code: `github.com/RuichengFIC/ADGAT`.

**Cross-stock placement / prior.** Models **momentum spillover** (firm A's move spilling to related
firm B). Crucially uses an **UNMASKED attention** that **dynamically infers general firm relations
from observed market signals**, rather than restricting to predefined relations — i.e. it is
effectively a **dense (fully-connected) graph-free attention**, same philosophy as MASTER/DTML. Its
novelty is the **attribute-mattered aggregator**: it gates the spillover by an element-wise product
with a non-linear transform of the source firm's attributes (so a price drop on tiny volume doesn't
spill). Main-path, ungated.

**Anti-collapse relevance.** The attribute-gating sharpens *which* stocks influence which — a
content-based reason for non-uniform attention. This is a direct antidote to F1: if the V/output
carries an **attribute-conditioned** signal, uniform attention is strictly suboptimal because it
washes out the attribute discrimination.

**Applicability to our readout: MEDIUM.** The "unmasked dense attention beats predefined graphs"
result reinforces dense-for-us. The attribute-mattered aggregator is more than we need at the readout
(our tokens are already pooled per-stock), but it motivates feeding **per-stock content** (not just a
generic query) into K/V so the attention has something discriminative to sharpen on.

---

## 5. RSR — Temporal Relational Stock Ranking (the graph-prior baseline)

**Paper:** Feng, He, Wang, Luo, Liu, Chua. *Temporal Relational Ranking for Stock Prediction.* TOIS
2019. arXiv:1809.09441. Code: `github.com/fulifeng/Temporal_Relational_Stock_Ranking`.

**Cross-stock placement / prior.** LSTM per stock → **Temporal Graph Convolution (TGC)** over a
**predefined relation graph** (sector + Wikidata firm relations) → FC ranking head. This is the
canonical **graph-prior, ranking-loss** method, and importantly it **optimizes a pairwise ranking
loss** (closest to our ListMLE objective of any method here). Cross-stock mixing is main-path,
ungated, but **graph-masked** (only related stocks interact).

**Anti-collapse relevance.** The graph mask *prevents* uniform-over-all-stocks by construction — a
stock can only mix with its graph neighbors, so "attend to everyone equally" is not in the
hypothesis space. This is the structural anti-collapse property of all graph methods.

**Reported gains.** Strong return ratios on NYSE/NASDAQ vs SFM/LSTM baselines (return-ratio metric,
older protocol; superseded numerically by DTML/MASTER but RSR is the ranking-loss reference).

**Applicability to our readout: LOW–MEDIUM.** The predefined Wikidata/sector graph is the thing the
2022–2024 SOTA (MASTER/DTML/THGNN) explicitly moved away from; for csi300 a static graph is unlikely
to beat dense market-conditioned attention. RSR is most useful to us as the **ranking-loss precedent
for cross-stock mixing** (confirms cross-stock + ranking loss is a coherent, well-posed combination —
exactly our setting).

---

## 6. HIST — Concept-Oriented Shared Information

**Paper:** Xu, Liu, Wang, Tian, Liu, Bian, Yin, et al. *HIST: A Graph-based Framework for Stock Trend
Forecasting via Mining Concept-Oriented Shared Information.* arXiv:2110.13716 (2021). Code:
`github.com/Wentao-Xu/HIST` (built on Qlib / Alpha360 — directly comparable data lineage to us).

**Cross-stock placement / prior.** Stocks share information through **concepts** (predefined concepts
from stock-concept membership + **hidden** learned concepts). Each stock's representation is decomposed
into a predefined-concept-shared module, a hidden-concept-shared module, and an individual module; the
"shared" modules are the cross-stock interaction (stocks linked to the same concept exchange info).
Main-path, ungated. The **hidden-concept** module is effectively a **learned soft clustering /
low-rank cross-stock mixing** — stocks attend to learned concept prototypes rather than directly to
each other.

**Anti-collapse relevance.** Routing cross-stock information through a small set of **concept
prototypes** (a bottleneck) is an alternative to dense N×N attention that *cannot* collapse to a
trivial global mean, because each concept aggregates only its members. This is the same idea as a
**learned market/CLS summary token**, generalized to K prototypes.

**Applicability to our readout: MEDIUM.** Concept/prototype bottleneck is a viable *alternative
parameterization* of cross-stock mixing (attend to K learned prototypes, K≈4–16, instead of N=300
stocks) that is cheaper and has a built-in anti-collapse structure. Candidate fallback if dense N×N
attention proves unstable (§11-B). HIST's Qlib/CSI lineage makes its numbers the most directly
comparable to ours of any graph method.

---

## 7. FactorVAE — Probabilistic Dynamic Factor Model (different paradigm)

**Paper:** Duan, Wang, Zhang, Li. *FactorVAE: A Probabilistic Dynamic Factor Model Based on
Variational Autoencoder for Predicting Cross-Sectional Stock Returns.* AAAI 2022.
ojs.aaai.org/index.php/AAAI/article/view/20369.

**Cross-stock placement.** Cross-stock interaction is **implicit, via shared latent factors**: a VAE
encodes the cross-section into a small set of latent **factors** (a factor-exposure × factor-return
decomposition), and a **prior-posterior** learning scheme uses future info to guide an optimal
posterior factor model. The "interaction" is that all stocks load on a **shared low-rank factor
space** — there is no explicit stock↔stock attention. This is the **low-rank / factor-bottleneck**
view of cross-stock structure (vs the dense-attention view).

**Anti-collapse relevance.** A factor bottleneck is, like HIST's concepts, a structural guard against
trivial collapse — factors must explain cross-sectional dispersion or the reconstruction loss
suffers. But it requires the VAE/prior-posterior machinery, which is a larger architectural change
than a readout attention.

**Applicability to our readout: LOW (paradigm mismatch).** Useful as a *conceptual* point — cross-
stock structure can be captured by a **shared low-rank latent factor space** rather than dense
attention. If we wanted a non-attention cross-stock readout, a low-rank `[B,D]→[K]→[B,D]` factor
projection (à la StockMixer's NoGraphMixer, §9) is the lighter realization of the same idea. Not the
primary recommendation, but the cheapest "is there ANY cross-stock signal" probe.

---

## 8. TRA — Temporal Routing Adaptor (NOT cross-stock — a cautionary tale)

**Paper:** Lin, Zhou, Liu, Bian. *Learning Multiple Stock Trading Patterns with Temporal Routing
Adaptor and Optimal Transport.* KDD 2021. arXiv:2106.12950. Code: `microsoft/qlib` (`examples/
benchmarks/TRA`).

**What it actually is.** TRA is **NOT a cross-stock method.** It is a set of independent **temporal
predictors** plus a **router** that dispatches each *sample* to a predictor, with **Optimal Transport
(OT)** providing the assignment target. The cross-stock axis is untouched.

**Why it matters here (the warning).** TRA is the closest published analog to *our failed
architecture*: a **router over parallel experts**. The paper's central finding is that a naively
trained router produces **trivial assignments — almost all samples routed to one predictor** (i.e.
**router collapse**), and they need an **explicit OT auxiliary loss** to prevent it. This is the same
collapse class as our router-gated stock_expert dying. **The lesson is double-edged:**
- It *confirms* that learned routers over parallel experts collapse without an explicit anti-collapse
  pressure (OT, load-balancing) — vindicating the decision to **abandon the router-gated parallel
  placement** for cross-stock.
- It tells us that *if* we ever keep a gated/parallel cross-stock variant, it would need an OT/load-
  balancing auxiliary loss to survive — which is strictly more machinery than just going ungated
  main-path. **So: prefer ungated main-path; don't try to rescue the router.**

**Applicability to our readout: LOW for the mechanism, HIGH as a documented anti-pattern.** TRA is
the citation for "router over parallel experts collapses; ungated main-path avoids the whole problem."

---

## 9. StockMixer — MLP NoGraphMixer (the lightest cross-stock primitive)

**Paper:** Fan, Shen. *StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price
Forecasting.* AAAI 2024. ojs.aaai.org/index.php/AAAI/article/view/28681. Code:
`github.com/SJTU-DMTai/StockMixer`.

**Cross-stock placement / mechanism.** Three MLP-mixing stages: **indicator mixing → time mixing →
stock mixing**. The **stock mixing** is a `NoGraphMixer`: a **parameter-light, graph-free MLP** that
explicitly models **stock-to-market and market-to-stock** influence. Concretely it (i) projects the
B stocks down to a small market representation (a learned low-rank aggregation — *stock→market*),
then (ii) redistributes that market representation back to each stock (*market→stock*), with
normalization, per time-step. **No attention, no graph, no softmax** — so it is **structurally immune
to F1 (uniform softmax collapse)** because there is no softmax to go uniform. It is a low-rank linear
mixer (cf. FactorVAE's factor space and HIST's concepts — same "bottleneck through a market summary"
idea, simplest realization).

**Anti-collapse relevance — directly addresses BOTH our failures.**
- **F1:** No softmax ⇒ no uniform-attention attractor. The stock→market→stock map is a *learned
  linear operator*, not a convex combination that can degenerate to the mean.
- **F2:** No de-mean constraint is needed (the mixer isn't trying to be mean-free), so there's no
  scale-invariant constraint to game with `‖Wv‖` inflation.

**Reported gains.** Outperforms SOTA forecasting baselines with lower memory/runtime (AAAI 2024).

**Applicability to our readout: HIGH (as the no-attention control / fallback).** StockMixer's
NoGraphMixer is the **single most important non-attention alternative** for us: it is a cheap,
collapse-immune cross-stock primitive that we can drop in at the readout `[B,D]` exactly where the
MHA would go. Strong recommendation to **A/B it against the ungated attention** — if the MLP mixer
captures the same alpha without any softmax, it sidesteps F1/F2 entirely and is the safer production
choice. (It cannot do *content-based, stock-specific* routing the way attention can, so attention
remains the higher-ceiling bet *if* it can be stabilized.)

---

## 10. CI-STHPAN & other recent (2023–2025) cross-sectional transformers

**CI-STHPAN.** Xia, Ao, Li, Liu, Liu, Ye, Chai. *CI-STHPAN: Pre-trained Attention Network for Stock
Selection with Channel-Independent Spatio-Temporal Hypergraph.* AAAI 2024.
ojs.aaai.org/index.php/AAAI/article/view/28770. Two-stage: self-supervised **pre-training**
(transformer + HGAT) then **ranking fine-tuning**. Builds a **channel-independent dynamic hypergraph
from DTW similarity** of stock series. Cross-stock mixing = hypergraph attention (a stock can belong
to multiple hyperedges = soft multi-membership clusters, cf. HIST concepts). **Applicability: LOW–
MEDIUM** — the hypergraph + DTW + pretraining stack is heavy; the transferable idea is **ranking
fine-tuning on top of a cross-stock representation** (matches our ListMLE setting) and **soft multi-
cluster membership** as an anti-collapse structure.

**General trend (FinMamba, DiffsFormer, Graph-Mamba, etc., 2024–2025).** The frontier is adding
**state-space (Mamba) temporal backbones + market-aware graph enhancement**, and **diffusion-based
factor augmentation**. None changes the core cross-stock verdict: **main-path, ungated, market-
conditioned, increasingly graph-free** stock mixing.

---

## 11. General transformer practice for set-attention over ~300 unordered tokens

Our readout cross-stock attention is a **set attention** (B≈300 tokens, **no order**, variable
count). The relevant deep-learning literature:

**Permutation equivariance.** No positional encoding on the stock axis (correct — stocks are an
unordered set). Our existing stock_expert already does this (`bias=None`). The readout block must
likewise be permutation-equivariant over B. ✔ free.

**ReZero / LayerScale / warm residual init — the cold-gate trap, resolved.**
- **ReZero** (Bachlechner et al., *ReZero is All You Need*, arXiv:2003.04887): `x + α·F(x)` with a
  **single learnable scalar α initialized to 0**. Trains to identity at init, then *learns* to open.
- **LayerScale** (Touvron et al., *Going deeper with Image Transformers / CaiT*, ICCV 2021,
  arXiv:2103.17239): `x + diag(λ)·F(x)` with **per-channel** `λ` initialized to a *small* value
  (1e-4…1e-1), not exactly 0.
- **Critical nuance vs our finding.** Our memory says **"scalar sigmoid gates freeze at init."** That
  is the `sigmoid(g)`-gate trap — `g` initialized cold (`g≈0`) has near-zero gradient through the
  sigmoid saturation and never opens. **ReZero/LayerScale are NOT that trap**: they multiply the
  branch by a *raw* scalar/vector (no sigmoid saturation), and the gradient to α/λ is
  `∂L/∂α = ⟨∂L/∂out, F(x)⟩`, which is **non-zero whenever the branch output correlates with the loss
  gradient** — so α can climb off zero. The difference is *raw-scalar warm-start (works)* vs
  *sigmoid-of-cold-init (freezes)*. **Recommendation: if any residual scaling is used, use a raw
  LayerScale `λ` (per-channel, init ~1e-1) or ReZero `α` (init 0), NEVER `sigmoid(g)` with a cold
  init.** But see the stronger recommendation below: for a *mandatory* readout block, prefer **plain
  pre-LN vanilla residual (no scaling at all)** — MASTER uses exactly that and it's identity-stable
  by construction; LayerScale/ReZero are only needed for *deep* stacks (we have 1 block).

**Pre-LN vs Post-LN.** Pre-LN (`x + F(LN(x))`) is the modern default for training stability without
warmup; MASTER and the DTML reimplementations use it. **Recommendation: Pre-LN.** Post-LN can give
marginally better final quality in some NLP settings but needs warmup/careful init — not worth it for
a 1-block readout add.

**QK-norm / cosine attention — a direct lever on our temperature problem.** QK-normalization (used in
ViT-22B, and analyzed in `Query-Key Normalization in Transformers`) L2-normalizes Q and K before the
dot product, so logits become **cosine similarities × a learned scale `g`**: `logits = g · (q̂·k̂)`.
This **decouples logit magnitude from `‖W_q‖,‖W_k‖`** and makes the **attention temperature an
explicit learnable parameter `g`** instead of an emergent function of weight norms. Relevance to F1:
**uniform collapse = logits too small/flat**; QK-norm lets the model *learn* to raise `g` to sharpen,
with a clean gradient, rather than having to inflate weight norms. Relevance to F2: because magnitude
is normalized out of the QK path, **QK-norm removes one route to the V-escape's correlate** (it
doesn't directly bound `‖Wv‖‖Wo‖`, but it stops the *attention-logit* side from being gamed by norm
inflation). **Recommendation: add QK-norm (or cosine attention) with a learnable temperature `g`
initialized to give moderately sharp attention** — this is the single most targeted fix for F1.

**Attention entropy collapse & σReparam.** Zhai et al., *Stabilizing Transformer Training by
Preventing Attention Entropy Collapse*, ICML 2023, arXiv:2303.06296. Their target failure is the
*opposite* of ours — they fight **LOW** entropy (over-sharp, training instability) and our problem is
**HIGH** entropy (uniform, dead). **But the mechanism is the same knob**: they show
**attention entropy is monotone-controlled by the spectral norm / magnitude of the QK logits**, and
**σReparam** reparameterizes each weight matrix `W = (γ/σ(W))·W` (γ learnable scalar, `σ(W)` =
spectral norm) to *bound* logit growth. For us, the same logit-magnitude knob must be pushed the
**other way** — we need to *enable* logit growth, which is exactly what an **entropy regularizer with
a target / QK-norm with a learnable up-scalable `g`** does. **Key transferable fact: entropy ↔ logit
magnitude ↔ weight spectral norm are one coupled knob; control it explicitly (QK-norm `g` or an
entropy term) rather than leaving it emergent.**

**Entropy regularization of attention.** A direct fix for F1: add `+λ_ent · H(A)` is the *wrong sign*
(encourages uniform). The correct anti-uniform term is a **NEGATIVE entropy penalty** (reward low
entropy) or, more safely, an **entropy *target/floor* penalty** `λ·(H(A) − H_target)²` keeping
attention away from the uniform maximum `H=log B`. This is a soft, explicit anti-F1 pressure. Use
sparingly (over-sharpening reintroduces the σReparam instability) — QK-norm is generally cleaner.

**Market / CLS summary token.** A learned global token that aggregates the cross-section and is read
by every stock (DTML's global context; HIST's concepts; StockMixer's market path; a transformer CLS).
**Caveat from our gradient analysis** (`readout_stock_attention_analysis.md §4c`): a **single global**
summary broadcast to all stocks is a **per-day constant ⇒ rank-invariant ⇒ dead** (same trap as our
existing regime vector). So a market token only helps if it is used as a **content-based modulation**
(DTML's *dot-product* local↔global, or MASTER's feature gate — which reweights *per-stock-different*
features) **not as an additive broadcast**. **Recommendation: condition K/Q on the regime vector
multiplicatively/via FiLM (per-stock-different effect), don't just add a market token.**

---

## A. Ranked, SOTA-grounded implementation choices for OUR ungated main-path readout cross-stock attention

Locus (from `readout_stock_attention_analysis.md`): **post-pool, pre-head**, on `h_pooled:[B,D]`,
one MHA with sequence = B stocks, single day's cross-section (sampler invariant already guaranteed).
Ranked by priority/confidence:

1. **Placement: ungated, main-path, in-series, NO router, NO sigmoid gate.** (MASTER §1, DTML §2, RSR
   §5, AD-GAT §4 — *every* working method does this; TRA §8 is the counter-example proving routers
   collapse.) This is the load-bearing decision. Residual is a **fixed `+o`** (vanilla), not a
   learned scalar gate.

2. **Normalization: Pre-LN + vanilla residual.** `o = MHA(LN(h)); h = h + o` (+ optional FFN with its
   own pre-LN residual). MASTER/DTML precedent. Identity-stable at init for a single block; no
   ReZero/LayerScale needed. **If** you insist on residual scaling, use **raw LayerScale `λ`
   (per-channel, init ≈0.1)** or **ReZero `α` (init 0)** — *never* `sigmoid(g)` cold init (our
   freeze-at-init trap; §11). LayerScale init must be a *raw* multiplier, and even 0-init ReZero opens
   because its gradient isn't sigmoid-saturated.

3. **Temperature / anti-collapse: QK-norm (cosine attention) with a learnable scalar `g`.** (§11;
   arXiv:2303.06296 shows entropy↔logit-magnitude is the controlling knob.) This makes the attention
   temperature explicit and learnable, gives a clean gradient to *sharpen away from uniform*, and
   decouples logit magnitude from weight norms (partial F2 mitigation on the QK side). **This is the
   #1 targeted defense against F1.** Initialize `g` to give moderately-sharp (sub-uniform) attention,
   not flat.

4. **Heads & width: `n_heads=4` (match the model), `d_head=16`, single layer.** MASTER uses only 2
   heads for inter-stock; 4 is fine and matches our config. One block to start (we are not building a
   deep stack; depth is where LayerScale/ReZero earn their keep, and we don't have it).

5. **Market conditioning: YES, MASTER/DTML-style, but as multiplicative/FiLM modulation, not an
   additive broadcast token.** Reuse our existing `RegimeContextEncoder` vector to FiLM the Q/K (or
   the input `h_pooled`) so the cross-stock attention is **market-relativized** (DTML's multi-level
   context; MASTER's market gate). **Do NOT add a single global market token** — that is a per-day
   constant and is rank-invariant/dead (§11, our §4c). This is a *secondary* arm (ship the plain
   ungated block first; add regime-FiLM as the next ablation).

6. **Dropout: light attention dropout (≈0.1) on the softmax matrix** (MASTER applies attention
   dropout). Standard; not load-bearing.

7. **Dense attention, NOT a graph prior — for csi300.** (§3–§6: 2022–2024 SOTA moved *away* from
   predefined graphs to dense/data-derived attention; MASTER beats graph-DTML by +31/+46% IC/RankIC.)
   For a ~300-name liquid universe, a static industry graph is unlikely to beat dense market-
   conditioned attention and adds construction cost. **Skip the graph prior initially.** Keep THGNN's
   *dynamic price-correlation* attn_bias (§3) and HIST's *concept-prototype bottleneck* (§6) as
   **fallbacks if dense collapses** (B below), not as the first build.

8. **Variable stock count per day: handle with a key-padding mask.** B varies day to day; pad to
   `B_max` and pass a `key_padding_mask` to the MHA (padded stocks excluded from softmax), and exclude
   them from the loss (we already mask). No positional encoding (set, unordered). This is mechanical;
   the daily sampler already gives single-day batches.

9. **No de-mean on the readout output (for the ungated/replacing variant).** Per our gradient
   argument (`readout_stock_attention_analysis.md §3b/§4a`), ungated main-path makes uniform a
   *repeller*, so there is **no parked mean to subtract** — and adding de-mean is what *opens* the F2
   V-escape. Only the *gated/residual* control variant needs de-mean, and then it must use the
   parameter-free `rms`/`unit` out-norm (never `ln_affine`). **Ship the ungated variant without
   de-mean.**

---

## B. What SOTA does to avoid OUR EXACT two failure modes

**F1 — Uniform collapse (cross-sectional mean, rank-invariant, loss-neutral):**

| SOTA mechanism | Source | How it defeats F1 |
|---|---|---|
| **Mandatory main-path, ungated placement** (no router/gate exit) | MASTER §1, DTML §2, RSR §5, AD-GAT §4 | Removes the zero-cost exit; gradient *must* flow through cross-stock mixing. Uniform becomes a repeller (replacing) or a free-but-harmless constant atop a signal-carrying residual. **This alone is what most methods rely on.** |
| **Graph / hypergraph mask** (attend to neighbors, not the universe) | RSR §5, THGNN §3, CI-STHPAN §10 | "Attend to everyone equally" is *not in the hypothesis space* — a stock mixes only with its (sparse) neighborhood, so the global-mean degenerate mode is structurally excluded. |
| **Concept/factor/market bottleneck** (route through K prototypes, not N stocks) | HIST §6, FactorVAE §7, StockMixer §9 | Cross-stock info passes through a small learned bottleneck that must explain cross-sectional dispersion; a trivial global mean fails the reconstruction/predictive objective. |
| **Market-conditioned tokens** (content to attend on) | MASTER gate §1, DTML multi-level §2, AD-GAT attribute aggregator §4 | Gives the attention *stock-discriminative* content; uniform attention washes out the conditioning ⇒ strictly suboptimal ⇒ gradient pushes off uniform. |
| **QK-norm / learnable temperature `g`; entropy floor** | arXiv:2303.06296, QK-norm §11 | Makes the temperature explicit & learnable with a clean gradient to *sharpen*; entropy↔logit-magnitude is the controlling knob — push it the anti-uniform way. |
| **No-softmax mixer** (linear stock→market→stock) | StockMixer NoGraphMixer §9 | No softmax ⇒ no uniform attractor at all. The collapse-immune *non-attention* baseline/fallback. |
| **Dispersion-forcing label/loss** (CS-Zscore MSE; pairwise ranking) | MASTER §1, RSR §5 | Flat scores ⇒ bad CS-MSE / max ranking loss; the loss itself penalizes the uniform output (our ListMLE+IC is already location-invariant, so this is *necessary but we also need placement*). |

**F2 — V-amplification escape (`‖Wv‖‖Wo‖` blowup to satisfy a de-mean constraint without sharpening):**

| SOTA mechanism | Source | How it defeats F2 |
|---|---|---|
| **Don't impose de-mean at all** (ungated replacing placement) | `readout_stock_attention_analysis.md §3b/§4`; MASTER §1 (no de-mean) | F2 only exists when there's a scale-invariant constraint to game. MASTER never de-means; with ungated main-path, uniform is already a repeller, so de-mean is unnecessary. **Removing the constraint removes the escape.** |
| **Pre-LN on `(h+o)`** | MASTER §1 | LN after the residual bounds the *joint* magnitude of the summed representation, limiting how much raw V-magnitude can survive to the head. |
| **QK-norm (decouple logit magnitude from weight norms)** | §11 | Removes the *attention-side* incentive to inflate norms (logits become cosine·g); the model can't buy sharper-looking logits via `‖Wq‖,‖Wk‖`. (Doesn't bound `‖Wv‖‖Wo‖` directly — pair with the next row.) |
| **σReparam / spectral-norm reparam of `Wv`,`Wo`** | arXiv:2303.06296 | If a residual/de-mean variant is kept, reparameterize V/O projections by their spectral norm so a learnable γ — *not* unconstrained weight growth — controls magnitude; caps the `‖Wv‖‖Wo‖` blowup. |
| **Parameter-free `rms`/`unit` out-norm (never `ln_affine`)** | `moe_block.py:241-260` (our existing fix) | If de-mean *is* used (gated control variant), a scale-invariant parameter-free norm removes magnitude as a DOF; a learnable affine gain (`ln_affine`) re-opens the escape and must be avoided. |
| **Linear mixer (no V at all to inflate)** | StockMixer §9 | The NoGraphMixer has no separate V/O attention projections to blow up; magnitude is a single bounded linear map + norm. |

**Net F1/F2 prescription for us:** ungated main-path (kills F1's exit) + QK-norm learnable temperature
(targeted F1 sharpening lever) + **no de-mean** in the ungated variant (kills F2 by removing the
constraint) + pre-LN joint-magnitude bound. Keep StockMixer's NoGraphMixer as the softmax-free A/B
control (immune to both), and σReparam-on-Wv/Wo + parameter-free out-norm only if a gated/de-mean
variant is retained.

---

## C. Summary table — method → placement → gated? → anti-collapse → applicability to our readout

| Method (cite) | Cross-stock placement | Gated? | Prior | Anti-collapse trick | Applicability to our readout |
|---|---|---|---|---|---|
| **MASTER** (AAAI'24, 2312.15235) | Inter-stock attn, **main-path, in-series**, pre-pool, per-time-step | **Ungated/mandatory** | Dense, graph-free | Mandatory placement + CS-Zscore-MSE loss + market feature-gate; `√(D/h)` scale | **HIGH — the template** (adapt to post-pool `[B,D]`) |
| **DTML** (KDD'21, 3447548.3467297) | Data-axis transformer over stocks, **main-path** | **Ungated** | Dense, graph-free | Mandatory + **multi-level (market-context) injection** (dot-product local↔global) | **HIGH — for market-conditioning** (reuse our regime vector) |
| **StockMixer** (AAAI'24, view/28681) | NoGraphMixer **stock→market→stock**, main-path | **Ungated** | None (MLP) | **No softmax ⇒ collapse-immune**; low-rank market bottleneck | **HIGH — softmax-free A/B control / fallback** |
| **AD-GAT** (AAAI'21, ADGAT) | Unmasked dense graph-attn, main-path | **Ungated** | Dense (graph-free, unmasked) | **Attribute-mattered aggregator** (content-gated spillover) | MEDIUM — motivates content-rich K/V |
| **THGNN** (CIKM'22, 2305.08740) | Hetero graph-attn on **daily price-corr graph**, main-path | **Ungated** | **Dynamic** price-corr graph | Sparse neighborhood (no global-mean mode) | MEDIUM — corr attn_bias *fallback* |
| **HIST** (2110.13716, Qlib lineage) | Concept-shared modules (predefined+hidden) | **Ungated** | Concept graph + hidden concepts | **Prototype bottleneck** (route via K concepts) | MEDIUM — prototype-bottleneck fallback |
| **RSR** (TOIS'19, 1809.09441) | Temporal Graph Conv on **predefined** graph, main-path | **Ungated** | **Static** sector/Wikidata graph | Graph mask excludes uniform-over-all | LOW–MED — ranking-loss precedent; static graph dated |
| **FactorVAE** (AAAI'22, view/20369) | Implicit via **shared latent factors** (VAE) | n/a (latent) | Low-rank factor space | Factor bottleneck must explain dispersion | LOW — paradigm mismatch; low-rank idea |
| **CI-STHPAN** (AAAI'24, view/28770) | Spatio-temporal **hypergraph** attn + pretrain | **Ungated** | DTW dynamic hypergraph | Soft multi-cluster (hyperedge) membership | LOW–MED — heavy stack; ranking-FT idea |
| **TRA** (KDD'21, 2106.12950) | **NOT cross-stock** — temporal-pattern router | **Router-gated** | None | **OT auxiliary loss** to stop router collapse | LOW mechanism / **HIGH as anti-pattern** (router collapse = our failure) |
| *Practice:* ReZero/LayerScale (2003.04887/2103.17239) | residual scaling | warm raw scalar/vector | — | identity-start without sigmoid-freeze | applies to residual init |
| *Practice:* QK-norm / σReparam (2303.06296) | attention temperature | learnable `g`/γ | — | **entropy↔logit-magnitude knob** (our F1/F2 lever) | **HIGH — core anti-collapse import** |

---

## D. Where SOTA DISAGREES with / refines the 'ungated main-path' recommendation

1. **SOTA strongly AGREES with ungated main-path placement** — *every* working cross-stock method
   (MASTER, DTML, RSR, THGNN, AD-GAT, StockMixer, HIST) puts cross-stock mixing in series on the main
   path, ungated. The *only* router-gated parallel design in the literature (TRA) is **not even
   cross-stock** and **explicitly collapses without an OT auxiliary loss**. So the ungated-main-path
   recommendation is the *consensus*, not a contrarian bet. No SOTA method supports the router-gated
   parallel placement that failed for us.

2. **Refinement, not disagreement — placement granularity.** MASTER/DTML place cross-stock attention
   *before* temporal pooling (per-time-step / on the per-stock context), not strictly *post-pool*. Our
   post-pool `[B,D]` locus is a *simplification* (one MHA instead of T parallel ones) and is sound by
   the gradient argument, but it discards per-time-step "momentary" correlations that MASTER markets
   as its edge. **Flag:** if post-pool underperforms, the MASTER-faithful move is to run cross-stock
   attention on `z_T:[B,N,D]` per-(N) *before* the factor pool (more compute), or on the
   pre-pool per-stock sequence — closer to MASTER but heavier.

3. **A genuine alternative the survey surfaces: maybe don't use softmax attention at all.**
   StockMixer (AAAI'24) shows a **softmax-free MLP NoGraphMixer** matches/beats attention-based
   cross-stock methods while being **structurally immune to both F1 and F2**. This is a mild
   *disagreement* with "attention": for a noisy, ~300-name daily cross-section, the SOTA evidence is
   that a low-rank linear stock↔market mixer may be the *safer and equally strong* choice. **Strong
   recommendation: A/B the ungated attention against a NoGraphMixer.** Attention retains a higher
   ceiling (content-based, stock-specific routing) *only if* QK-norm/temperature control keeps it off
   the uniform attractor; if it can't be stabilized, the mixer is the production answer.

4. **Conditioning caveat (our own gradient analysis vs naive "add a market/CLS token").** A naive CLS
   /global-market-token broadcast is **rank-invariant and dead** for our location-invariant ranking
   loss (per-day constant). SOTA market conditioning that *works* (MASTER, DTML, AD-GAT) is always
   **multiplicative / content-dependent / per-stock-different** (feature gate, dot-product local↔
   global, attribute aggregator). So: condition the attention *multiplicatively via the regime
   vector*, **do not** add a global summary token. This refines, not contradicts, the recommendation.

---

## Sources

- MASTER — Li et al., *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*, AAAI 2024. arXiv:2312.15235. https://arxiv.org/abs/2312.15235 · https://arxiv.org/html/2312.15235v1 · code https://github.com/SJTU-DMTai/MASTER
- DTML — Yoo, Soun, Park, Kang, *Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts*, KDD 2021. https://dl.acm.org/doi/10.1145/3447548.3467297 · code https://github.com/simonjisu/DTML-pytorch · https://github.com/ceteris11/DTML
- HIST — Xu et al., *HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information*, arXiv:2110.13716 (2021). https://arxiv.org/abs/2110.13716 · code https://github.com/Wentao-Xu/HIST
- THGNN — Xiang et al., *Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction*, CIKM 2022. arXiv:2305.08740. https://arxiv.org/abs/2305.08740 · code https://github.com/TongjiFinLab/THGNN
- AD-GAT — Cheng & Li, *Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks*, AAAI 2021. https://ojs.aaai.org/index.php/AAAI/article/view/16077 · code https://github.com/RuichengFIC/ADGAT
- RSR — Feng et al., *Temporal Relational Ranking for Stock Prediction*, TOIS 2019. arXiv:1809.09441. https://arxiv.org/abs/1809.09441 · code https://github.com/fulifeng/Temporal_Relational_Stock_Ranking
- FactorVAE — Duan et al., *FactorVAE: A Probabilistic Dynamic Factor Model Based on VAE for Predicting Cross-Sectional Stock Returns*, AAAI 2022. https://ojs.aaai.org/index.php/AAAI/article/view/20369
- TRA — Lin et al., *Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport*, KDD 2021. arXiv:2106.12950. https://arxiv.org/abs/2106.12950 · code https://github.com/microsoft/qlib (examples/benchmarks/TRA)
- StockMixer — Fan & Shen, *StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price Forecasting*, AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/28681 · code https://github.com/SJTU-DMTai/StockMixer
- CI-STHPAN — Xia et al., *CI-STHPAN: Pre-trained Attention Network for Stock Selection with Channel-Independent Spatio-Temporal Hypergraph*, AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/28770
- ReZero — Bachlechner et al., *ReZero is All You Need: Fast Convergence at Large Depth*, arXiv:2003.04887 (2020). https://arxiv.org/abs/2003.04887
- LayerScale / CaiT — Touvron et al., *Going deeper with Image Transformers*, ICCV 2021. arXiv:2103.17239. https://arxiv.org/abs/2103.17239
- σReparam / attention entropy collapse — Zhai et al., *Stabilizing Transformer Training by Preventing Attention Entropy Collapse*, ICML 2023. arXiv:2303.06296. https://arxiv.org/abs/2303.06296
- QK-Normalization — overview: https://www.emergentmind.com/topics/query-key-normalization-qk-norm (ViT-22B / Henry et al. *Query-Key Normalization for Transformers*, EMNLP Findings 2020)
