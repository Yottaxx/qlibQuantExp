# Method Section Draft: Deterministic Global-Local Hyper-State for RST-MoE

## 1. Motivation

The current RST-MoE architecture conditions all adaptive mechanisms on a single shared latent regime vector:

$$
r_{i,d} \rightarrow \{\tau_d,\ \gamma_{i,d},\ \beta_{i,d},\ \pi_{i,d}^{(\ell)},\ q_{i,d},\ \alpha_{i,d}\}
$$

where $r_{i,d}$ is produced by the existing regime encoder from market context and input features.  
This design is effective as a first approximation, but it conflates several distinct sources of variation:

1. market-level slow regime;
2. stock-specific local structure;
3. routing preference between temporal and factor operators;
4. final readout preference.

As a result, the single shared latent becomes overloaded, and the model struggles to represent stock-level heterogeneity within the same day.

To address this limitation, we replace the single shared latent with a deterministic hierarchical state field composed of a global market state and a stock-local state.

---

## 2. Hierarchical State Field

For trading day $d$, let $m_d$ denote the external market state and $u_d$ denote a stable day-level internal market summary.  
We define the global state as:

$$
z_d^{g} = G(m_d, u_d)
$$

where:

- $G(\cdot)$ is the global state encoder;
- $z_d^{g} \in \mathbb{R}^{D_g}$ is shared by all stocks on day $d$.

Given the global state, we define a stock-specific local state for stock $i$:

$$
z_{i,d}^{\ell} = L(x_{i,d}, z_d^{g})
$$

where:

- $x_{i,d}$ is the panel input for stock $i$ on day $d$;
- $L(\cdot)$ is the local state encoder;
- $z_{i,d}^{\ell} \in \mathbb{R}^{D_{\ell}}$ captures stock-specific residual structure under the global market condition.

This yields a deterministic hierarchical state field:

$$
(m_d, u_d) \rightarrow z_d^{g} \rightarrow z_{i,d}^{\ell}
$$

The key idea is that market-level regime and stock-local structure should not be represented by the same latent object.

---

## 3. Role Assignment Across Control Heads

The proposed design does not merely introduce extra latent variables.  
Its main contribution is to assign different subsets of the state field to different control heads.

### 3.1 Time-Scale Head

The regime-adaptive temporal scale is modeled as:

$$
\tau_d = f_{\tau}(z_d^{g})
$$

where:

- $\tau_d$ is the day-level temporal memory scale;
- $f_{\tau}(\cdot)$ is the time-scale head.

We restrict $\tau_d$ to depend only on the global state because temporal memory preference is primarily a market-level property.

### 3.2 FiLM Head

The factor modulation parameters are generated as:

$$
\left(\gamma_{i,d}, \beta_{i,d}\right)
=
f_{\mathrm{FiLM}}(z_d^{g}, z_{i,d}^{\ell})
$$

where:

- $\gamma_{i,d}$ and $\beta_{i,d}$ are FiLM modulation parameters for stock $i$ on day $d$;
- $f_{\mathrm{FiLM}}(\cdot)$ is the factor modulation head.

This allows factor interpretation to vary across stocks under the same market regime.

### 3.3 Router Head

For each block $\ell$, the temporal-factor routing weight is modeled as:

$$
\pi_{i,d}^{(\ell)}
=
f_{\pi}^{(\ell)}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

where:

- $\pi_{i,d}^{(\ell)}$ denotes the routing preference of stock $i$ on day $d$ at layer $\ell$;
- $u_d$ is the stable day-level summary;
- $f_{\pi}^{(\ell)}(\cdot)$ is the router head for layer $\ell$.

This formulation is critical because routing should depend on both the market-wide regime and the stock-specific local structure.

### 3.4 Pooling Head

The final regime-adaptive pooling rule is modeled as:

$$
\left(q_{i,d}, \alpha_{i,d}\right)
=
f_{\mathrm{pool}}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

where:

- $q_{i,d}$ is the pooling query;
- $\alpha_{i,d}$ controls the mixture between pooling modes;
- $f_{\mathrm{pool}}(\cdot)$ is the pooling head.

This allows the readout rule to depend on both market context and stock-local structure.

---

## 4. Revised Conditional Computation Graph

Under the proposed state field, the adaptive computation graph becomes:

$$
(m_d, u_d) \rightarrow z_d^{g}
$$

$$
(x_{i,d}, z_d^{g}) \rightarrow z_{i,d}^{\ell}
$$

$$
z_d^{g} \rightarrow \tau_d
$$

$$
(z_d^{g}, z_{i,d}^{\ell}) \rightarrow (\gamma_{i,d}, \beta_{i,d})
$$

$$
(z_d^{g}, z_{i,d}^{\ell}, u_d) \rightarrow \pi_{i,d}^{(\ell)}
$$

$$
(z_d^{g}, z_{i,d}^{\ell}, u_d) \rightarrow (q_{i,d}, \alpha_{i,d})
$$

Compared with the original one-latent design, this architecture explicitly separates:

- market-level adaptation,
- stock-local adaptation,
- routing control,
- readout control.

---

## 5. Why This Design Is Stronger Than a Single Shared Latent

The original architecture uses:

$$
r_d \rightarrow \{\tau,\gamma,\beta,\pi,\alpha,q\}
$$

which forces one latent vector to jointly encode all conditional factors.  
The proposed design instead uses:

$$
(z_d^{g}, z_{i,d}^{\ell})
\rightarrow
\{\tau,\gamma,\beta,\pi,\alpha,q\}
$$

with head-specific state assignment.  
This brings three advantages.

### 5.1 Better Factorization of Variation

Global market condition and stock-local structure are modeled separately:

$$
\text{global regime} \neq \text{local stock state}
$$

This reduces latent overload and improves identifiability.

### 5.2 Finer Routing Granularity

The router now receives a stock-specific local state:

$$
\pi_{i,d}^{(\ell)}
=
f_{\pi}^{(\ell)}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

which enables different stocks on the same day to favor different structural paths.

### 5.3 Cleaner Mechanism Attribution

Different heads consume different state combinations:

- time scale depends on $z_d^{g}$;
- FiLM depends on $(z_d^{g}, z_{i,d}^{\ell})$;
- routing depends on $(z_d^{g}, z_{i,d}^{\ell}, u_d)$;
- pooling depends on $(z_d^{g}, z_{i,d}^{\ell}, u_d)$.

This makes the roles of different states empirically testable.

---

## 6. Relationship to Future Extensions

The proposed deterministic hierarchical state field is intended as the optimal first-stage design, not the final endpoint.

It serves as the structural foundation for three future extensions:

### 6.1 Variational State Field

The deterministic states may later be upgraded to latent distributions:

$$
q(z_d^{g} \mid m_d, u_d), \qquad
q(z_{i,d}^{\ell} \mid x_{i,d}, z_d^{g})
$$

This would allow explicit uncertainty modeling, but only after the deterministic factorization is validated.

### 6.2 Discrete Regime Layer

A discrete regime head may later be added on top of the global state:

$$
s_d = \operatorname{SwitchHead}(z_d^{g})
$$

for mechanism interpretation and regime bucket analysis.

### 6.3 Dynamic Parameterization

Once the state field is stable, selected heads may be upgraded from conditional-input modulation to shared market-conditioned dynamics:

$$
\theta_d = \theta + \Delta \theta(z_d^{g})
$$

This is a second-stage enhancement rather than a first-stage requirement.

---

## 7. Summary

The proposed deterministic Global-Local Hyper-State replaces the original single shared latent with a hierarchical state field:

$$
z_d^{g} = G(m_d, u_d), \qquad
z_{i,d}^{\ell} = L(x_{i,d}, z_d^{g})
$$

and distributes these states across adaptive heads according to their roles:

$$
\tau_d \leftarrow z_d^{g}
$$

$$
(\gamma_{i,d}, \beta_{i,d}) \leftarrow (z_d^{g}, z_{i,d}^{\ell})
$$

$$
\pi_{i,d}^{(\ell)} \leftarrow (z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

$$
(q_{i,d}, \alpha_{i,d}) \leftarrow (z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

This design is the most suitable next step for the current RST-MoE codebase because it fixes the primary structural bottleneck without prematurely introducing the optimization burden of variational inference or dynamic weight generation.
