# Seed Analogies — 10 Historical Breakthroughs by Structure-Mapping

These are worked examples of analogies that produced real scientific advances. Each shows the mapping table format the skill should produce.

## 1. Maxwell — Mechanical vortices → electromagnetism (1861)

| Base (mechanics) | Target (electromagnetism) | Mapped relation |
|---|---|---|
| Spinning vortex tubes in a fluid | Magnetic field lines | Both have a rotational orientation that creates a side-pressure |
| Idler wheels between vortices | Electric current | Both transmit motion between adjacent rotating elements |
| Fluid pressure | Electric potential | Both drive flow when unequal |
| Elastic strain | Displacement current | Both arise when a system resists abrupt change |

**Imported prediction:** electromagnetic disturbances should propagate at the speed of light. Verified.

## 2. Bohr — Solar system → atom (1913)

| Base (solar system) | Target (hydrogen atom) | Mapped relation |
|---|---|---|
| Central mass (sun) | Nucleus | Both anchor an orbiting body via inverse-square attraction |
| Orbital radius | Quantized electron orbit | Both stable trajectories with specific energy |
| Energy of orbit | Spectral line | In both, transitions release energy |

**Imported prediction:** discrete spectral lines should match a specific formula (Rydberg). Verified.

## 3. Darwin — Malthus's population pressure → species evolution (1838)

| Base (demography) | Target (biology) | Mapped relation |
|---|---|---|
| Resource scarcity in a fixed niche | Limited ecological niches | Both impose competition |
| Birth rate exceeds death rate | Heritable trait variation | Both produce excess offspring |
| Natural selection of fittest individuals | Selection of heritable traits | Both filter on differential reproductive success |

**Imported prediction:** species accumulate inheritable differences over generations. Verified (with century of subsequent work).

## 4. Kirchhoff — Electric circuit → black-body radiation (1859)

| Base | Target | Relation |
|---|---|---|
| Current conservation at a node | Energy balance at a surface | Inflow = outflow |
| Voltage drop around a loop | Frequency-resolved energy exchange | Sum to zero |

**Imported prediction:** universal blackbody spectrum independent of material. Verified, led to quantum mechanics.

## 5. Shannon — Boolean algebra → switching circuits (1937)

| Base | Target | Relation |
|---|---|---|
| AND, OR, NOT logic | Series, parallel, complement circuits | Same algebra applies |

**Imported prediction:** any computable function can be implemented in switches. Foundation of computing.

## 6. Crick — Codes / messages → genetic translation (1958)

| Base (information theory) | Target (molecular biology) | Relation |
|---|---|---|
| Discrete symbols in a sequence | DNA nucleotides | Both ordered alphabets |
| Codewords mapping to meanings | Codons mapping to amino acids | Both are encoding tables |
| Redundancy | Wobble in the third position | Both arise from finite-alphabet constraints |

**Imported prediction:** the genetic code is a triplet code with redundancy. Verified.

## 7. Anderson — Spin glasses → neural networks (1979)

| Base (condensed matter) | Target (neural networks) | Relation |
|---|---|---|
| Magnetic moments on a lattice | Neurons | Many interacting binary units |
| Frustrated interactions | Conflicting input gradients | Multiple competing constraints |
| Energy landscape with many minima | Loss landscape with many minima | Same structure |

**Imported prediction:** replica symmetry breaking should occur. Anderson and Hopfield used this. Hopfield network = Ising model.

## 8. Lotka–Volterra — Chemical reactions → predator-prey (1925)

| Base (chemistry) | Target (ecology) | Relation |
|---|---|---|
| Reactant concentrations | Prey population | Mass-action kinetics |
| Reaction rate | Predation rate | Bilinear in both |

**Imported prediction:** populations oscillate with phase lag. Verified on lynx/hare data.

## 9. Friston — Surprise / free energy → brain function (2010)

| Base (statistical physics) | Target (neuroscience) | Relation |
|---|---|---|
| Variational free energy | Predictive coding | Both minimize a divergence between predicted and observed |
| Equilibrium | Stable belief state | Free energy minimum in both |

**Imported prediction:** cortex minimizes prediction error. Mixed empirical support.

## 10. Black-Scholes — Heat equation → option pricing (1973)

| Base (physics) | Target (finance) | Relation |
|---|---|---|
| Diffusion of heat | Diffusion of asset price (log-Brownian) | Both Gaussian random walks |
| Heat at a point | Option value at strike | Both solutions to a PDE |
| Boundary conditions | Payoff at expiry | Both terminal conditions |

**Imported prediction:** options should be priced by a specific PDE. Verified (with known caveats).

---

## Use of this file

When `transpose-from-another-field` is invoked, scan this file for the closest historical pattern before generating fresh candidates. If the target structure resembles one of these, use the worked mapping table as a template. Most quant-relevant entries: §7 (Anderson, spin glasses), §10 (Black-Scholes), §8 (Lotka-Volterra for regime cycling).
