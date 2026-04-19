# Method Section Draft: Stable Market State Field

## 1. Motivation

The current implementation uses a per-date `market_state` table as external macro features and supplements it with batch-level summary statistics inside the model.  
In practice, this leads to a mismatch between the desired market-level state and the actual conditioning signal:

$$
\hat u_{d,b} = H(\mathcal B_{d,b})
$$

where:

- $d$ denotes the trading day,
- $b$ denotes the sampled batch or chunk on that day,
- $\mathcal B_{d,b}$ is a sampled subset of the day-level cross section.

However, the desired market state should be defined at the full-day level rather than the batch level.  
We therefore introduce a stable market state field with three explicitly separated objects:

$$
m_d = \text{daily market context}
$$

$$
o_d = \text{daily market observation}
$$

$$
z_d^g = \text{stable daily market field}
$$

This separation distinguishes external market context, same-day market observation, and temporally filtered market state.

---

## 2. Daily Market Context

The daily market context is defined as:

$$
m_d = M(\mathcal C_d)
$$

where:

- $\mathcal C_d$ is the set of external market-level inputs available by day $d$,
- $M(\cdot)$ is the context feature constructor.

Typical components of $\mathcal C_d$ include:

- benchmark index returns,
- benchmark realized volatility,
- benchmark drawdown and range,
- market volume and amount signals,
- optional style or macro proxies.

The key property of $m_d$ is that it is external to the stock-level cross-sectional encoder.  
It represents slow-varying market background rather than same-day cross-sectional geometry.

---

## 3. Daily Market Observation

The daily market observation is constructed from the full day-level stock cross section:

$$
o_d = O(\mathcal X_d)
$$

where:

- $\mathcal X_d = \{x_{i,d}\}_{i \in \mathcal M_d}$ is the full stock universe on day $d$,
- $\mathcal M_d$ is the set of tradable stocks on that day,
- $O(\cdot)$ is a full-day observation encoder.

In the current project, the first-stage observation encoder is deterministic and analytic rather than learned.  
It consists of three components.

### 3.1 Global Distribution Statistics

Given the day-level feature matrix:

$$
X_d \in \mathbb{R}^{M_d \times P}
$$

we compute robust market-wide statistics:

$$
o_d^{\text{global}}
=
\left[
\text{mean\_abs}_d,\;
\text{std}_d,\;
\text{breadth}_d,\;
\text{tail}_d
\right]
$$

These summarize the magnitude, dispersion, sign balance, and tail activity of the cross section.

### 3.2 Correlation and Crowding Statistics

Let the standardized cross-sectional matrix be:

$$
\widetilde X_d = \operatorname{zscore}(X_d)
$$

and let:

$$
C_d = \operatorname{Corr}(\widetilde X_d)
$$

Then we define crowding-related observation features such as:

$$
\text{crowding}_d
=
\frac{1}{P(P-1)}
\sum_{i \neq j} |(C_d)_{ij}|
$$

$$
\text{pc1\_ratio}_d
=
\frac{\lambda_1(C_d)}{\operatorname{tr}(C_d)}
$$

where $\lambda_1(C_d)$ is the largest eigenvalue of the correlation matrix.

### 3.3 Factor-Structure Observation

We also compute factor-wise day-level statistics:

$$
f_d = F(X_d)
$$

where $F(\cdot)$ includes per-factor mean, standard deviation, and breadth.  
To control dimensionality, we fit PCA on the train split only:

$$
\bar f, U = \operatorname{PCA\_fit}(\{f_d\}_{d \in \mathcal D_{\text{train}}})
$$

and transform each day as:

$$
o_d^{\text{pca}} = U^\top (f_d - \bar f)
$$

The complete observation vector is:

$$
o_d = \left[o_d^{\text{global}},\ o_d^{\text{corr}},\ o_d^{\text{pca}}\right]
$$

The crucial property is that $o_d$ is constructed from the full day-level cross section rather than any sampled training batch.

---

## 4. Stable Market State Field

The final market field is not set equal to the raw observation.  
Instead, it is produced by causal temporal filtering:

$$
z_d^g = \Phi(z_{d-1}^g,\ m_d,\ o_d)
$$

where:

- $z_d^g$ is the stable market field used downstream,
- $\Phi(\cdot)$ is a causal temporal filter.

### 4.1 Fused Observation

We first fuse the normalized context and observation:

$$
e_d = [\tilde m_d,\ \tilde o_d]
$$

where:

- $\tilde m_d$ denotes train-scaled market context,
- $\tilde o_d$ denotes train-scaled market observation.

### 4.2 Multi-Timescale Causal Filters

We maintain a small bank of causal filtered states:

$$
z_d^{(k)} = \lambda_k z_{d-1}^{(k)} + (1-\lambda_k)e_d
$$

where:

- $k$ indexes the temporal scale,
- $\lambda_k \in (0,1)$ is the smoothing coefficient for scale $k$.

In practice, we recommend three half-life scales:

- short: 5 trading days,
- medium: 20 trading days,
- long: 60 trading days.

The corresponding smoothing factor may be defined from half-life $h_k$ by:

$$
\lambda_k = \exp\!\left(-\frac{\ln 2}{h_k}\right)
$$

### 4.3 Shock-Aware Mixing

To maintain both stability and responsiveness, we introduce shock-aware mixing:

$$
s_d = S_{\text{shock}}(o_d, o_{d-1})
$$

where $s_d$ is a shock descriptor derived from observation jumps, such as:

- dispersion change,
- tail activity change,
- breadth change,
- crowding change,
- volatility spike.

We then compute scale weights:

$$
w_d = \operatorname{softmax}(A s_d + b)
$$

and obtain the final market field:

$$
z_d^g = \sum_{k=1}^{K} w_{d,k} z_d^{(k)}
$$

This enables the market field to remain smooth in normal periods while reacting faster during market shocks.

---

## 5. Why This Design Is Preferable

The proposed design has four advantages over directly stacking delta, rolling, and z-score features into a single `market_state` table.

### 5.1 Clear Semantic Separation

The design explicitly distinguishes:

$$
m_d \neq o_d \neq z_d^g
$$

where:

- $m_d$ is external context,
- $o_d$ is same-day observation,
- $z_d^g$ is the stable filtered state.

### 5.2 Removal of Batch Contamination

The field is defined from:

$$
\mathcal X_d
$$

the full day-level cross section, rather than sampled batches:

$$
\mathcal B_{d,b}
$$

This removes sampler-induced noise from the market-level state definition.

### 5.3 Offline Reproducibility

The market field is constructed offline as a versioned daily asset.  
It can therefore be:

- audited,
- cached,
- reused across model variants,
- aligned with train-only scaling and PCA fit.

### 5.4 Smooth Upgrade Path

The deterministic filter:

$$
z_d^g = \Phi(z_{d-1}^g,\ m_d,\ o_d)
$$

can later be upgraded to:

- learned recurrent filters,
- state-space models,
- switching state models,
- variational latent fields.

This makes the proposed design a stable first-stage implementation rather than a dead-end heuristic.

---

## 6. Asset Outputs

The full pipeline produces three versioned daily assets:

### 6.1 Daily Market Context

$$
\{m_d\}_{d}
$$

Stored as a date-indexed table of external market context features.

### 6.2 Daily Market Observation

$$
\{o_d\}_{d}
$$

Stored as a date-indexed table of full-day market observation features.

### 6.3 Daily Market Field

$$
\{z_d^g\}_{d}
$$

Stored as a date-indexed table of stable filtered market states.

Only the last object is intended to replace the current `market_state.pkl` in the model-facing inference path.

---

## 7. Summary

We propose a stable market state field that decomposes the daily market signal into context, observation, and filtered state:

$$
m_d = M(\mathcal C_d)
$$

$$
o_d = O(\mathcal X_d)
$$

$$
z_d^g = \Phi(z_{d-1}^g,\ m_d,\ o_d)
$$

with:

$$
e_d = [\tilde m_d,\ \tilde o_d]
$$

$$
z_d^{(k)} = \lambda_k z_{d-1}^{(k)} + (1-\lambda_k)e_d
$$

$$
z_d^g = \sum_{k=1}^{K} w_{d,k} z_d^{(k)}
$$

This design preserves causal validity, removes batch contamination, remains easy to engineer offline, and provides a principled market-state asset for downstream regime-aware modeling.
