# Method Section Draft: Global Kronecker Low-Rank Cross Experts

## 1. Motivation

Existing RST-MoE blocks separate temporal and factor interactions through two axis-specific experts:

$$
h_{\text{base}}
=
w_t h_t + w_f h_f
$$

where:

- $h_t$ models temporal dependencies for each factor;
- $h_f$ models factor interactions at each time step.

This decomposition is efficient and well-regularized, but it is structurally limited. In particular, it cannot directly express interactions of the following form:

$$
(t', n') \rightarrow (t, n)
$$

where a historical temporal pattern over one factor subspace changes the interpretation of another factor subspace at the current step.

To address this limitation without abandoning the existing inductive bias, we introduce a small bank of global cross experts that model joint time-factor couplings.

---

## 2. Base Representation

For stock $i$ on day $d$, let the hidden panel representation at a given block be:

$$
X_{i,d} \in \mathbb{R}^{T \times N \times D}
$$

For exposition, we first suppress the hidden channel dimension and consider a single channel slice:

$$
X \in \mathbb{R}^{T \times N}
$$

The original block uses:

$$
h_t = \mathcal A_t(X), \qquad h_f = \mathcal A_f(X)
$$

and fuses them as:

$$
h_{\text{base}} = w_t h_t + w_f h_f
$$

with $[w_t, w_f]$ determined by the existing router.

---

## 3. Global Cross Expert Bank

We define a bank of $R_g$ global cross experts:

$$
\mathcal C_r(X) = A_r X B_r^\top
$$

where:

- $A_r \in \mathbb{R}^{T \times T}$ is a temporal mixing operator,
- $B_r \in \mathbb{R}^{N \times N}$ is a factor mixing operator.

This operator jointly couples temporal and factor dimensions and therefore explicitly models time-factor cross interactions.

If we vectorize $X$, the above becomes:

$$
\mathrm{vec}(\mathcal C_r(X))
=
(B_r \otimes A_r)\,\mathrm{vec}(X)
$$

which reveals the Kronecker structure.

---

## 4. Low-Rank Parameterization

Directly learning full matrices $A_r$ and $B_r$ is unnecessary and potentially unstable. We instead parameterize them in low-rank form:

$$
A_r = U_t^{(r)} {V_t^{(r)}}^\top
$$

$$
B_r = U_n^{(r)} {V_n^{(r)}}^\top
$$

where:

$$
U_t^{(r)}, V_t^{(r)} \in \mathbb{R}^{T \times k_t}, \qquad
U_n^{(r)}, V_n^{(r)} \in \mathbb{R}^{N \times k_n}
$$

with $k_t \ll T$ and $k_n \ll N$.

Substituting this decomposition yields:

$$
\mathcal C_r(X)
=
U_t^{(r)}
\left(
{V_t^{(r)}}^\top X U_n^{(r)}
\right)
{V_n^{(r)}}^\top
$$

This form has a natural interpretation:

1. project $X$ into a small set of temporal latent modes;
2. project $X$ into a small set of factor latent modes;
3. compute their joint interaction in a low-dimensional core space;
4. expand the result back to the original $T \times N$ grid.

---

## 5. Global Controller

The expert bank defines reusable structural prototypes, but the model still needs to determine which prototypes should be active under the current market condition.

We therefore introduce a global controller:

$$
c_d^g = g_\theta(z_d^g) \in \mathbb{R}^{R_g}
$$

where $z_d^g$ is a day-level condition vector. In practice, $z_d^g$ may be formed from:

$$
z_d^g = [\bar r_d,\ u_d]
$$

where:

- $\bar r_d$ is the day-level regime summary,
- $u_d$ is the block-level market summary.

The controller outputs amplitude coefficients rather than a simplex allocation:

$$
c_d^g = s_g \cdot \tanh(\mathrm{MLP}(z_d^g))
$$

This is important because the cross branch is intended as a structured residual correction, not as a competitor that replaces the existing time/factor experts.

---

## 6. Cross Correction

The global cross correction is:

$$
h_g
=
\eta_g \sum_{r=1}^{R_g} c_{d,r}^g \, \mathcal C_r(X)
$$

where $\eta_g$ is a learnable or fixed small residual scale.

The complete block fusion becomes:

$$
h_{\text{fused}}
=
w_t h_t
+
w_f h_f
+
h_g
$$

followed by the same residual and feed-forward updates as in the original block:

$$
X' = X + h_{\text{fused}}
$$

$$
X_{\text{out}} = X' + \mathrm{FFN}(\mathrm{LN}(X'))
$$

Thus, the proposed module augments rather than replaces the original architecture.

---

## 7. Expert-Controller Decomposition

The proposed design explicitly separates two learning problems.

### 7.1 What experts learn

Each expert learns a reusable operator prototype:

$$
\mathcal C_r : X \mapsto A_r X B_r^\top
$$

This corresponds to a specific type of joint temporal-factor transformation. In this sense, the expert bank acts as a dictionary of structured cross operators.

### 7.2 What the controller learns

The controller learns a market-state-dependent mixture rule:

$$
z_d^g \mapsto c_d^g
$$

This corresponds to deciding which cross-interaction prototypes should be emphasized under the current market condition.

Hence:

$$
\text{experts learn "what structures exist"}
$$

$$
\text{controller learns "when each structure should be used"}
$$

---

## 8. Why Not Learn One Large Cross Operator Directly

A direct approach would be to learn a single large operator:

$$
\mathrm{vec}(Y_{i,d}) = W_d \, \mathrm{vec}(X_{i,d})
$$

with:

$$
W_d \in \mathbb{R}^{TN \times TN}
$$

This is less desirable for several reasons.

### 8.1 Parameter efficiency

The full operator has $O(T^2 N^2)$ degrees of freedom, which is unnecessarily large for low-signal financial prediction tasks.

In contrast, the proposed decomposition assumes:

$$
W_d \approx \sum_{r=1}^{R_g} c_{d,r}^g W_r
$$

where each $W_r$ has low-rank Kronecker structure.

### 8.2 Sample efficiency

Directly learning $W_d$ requires the model to infer a high-dimensional operator for each market condition. Our formulation instead learns:

- a reusable set of shared operator bases;
- a low-dimensional controller over these bases.

This yields a significantly better bias-variance tradeoff.

### 8.3 Generalization across regimes

The proposed architecture assumes that effective regime-specific operators lie near a low-dimensional operator manifold:

$$
W_d \in \operatorname{span}\{W_1,\dots,W_{R_g}\}
$$

This is a stronger and more plausible assumption than requiring the model to generate a new unconstrained operator for every regime.

### 8.4 Interpretability

The expert bank can be inspected as a set of structural prototypes, while the controller can be analyzed as a regime-dependent selector. This makes the mechanism substantially more interpretable than a monolithic dynamic operator.

---

## 9. Computational Perspective

The proposed cross branch is substantially cheaper than a dense full 2D operator. For $R_g$ experts with ranks $k_t$ and $k_n$, complexity scales approximately with:

$$
O(R_g \cdot B \cdot D \cdot T \cdot N \cdot (k_t + k_n))
$$

up to implementation details and optional projections, which is significantly smaller than dense $TN \times TN$ mixing.

Moreover, because the branch is global and low-rank, it remains more GPU-friendly than highly irregular sparse alternatives.

---

## 10. Summary

The proposed global Kronecker low-rank cross experts extend RST-MoE with a structured correction term that explicitly models joint temporal-factor interactions:

$$
h_{\text{fused}}
=
w_t h_t
+
w_f h_f
+
\eta_g \sum_{r=1}^{R_g} c_{d,r}^g \, A_r X B_r^\top
$$

This design preserves the strengths of the original dual-expert decomposition while overcoming its key structural limitation: the lack of explicit cross-dimensional interaction modeling.
