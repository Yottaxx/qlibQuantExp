# Analogy Card — Regime Switching ↔ Statistical-Mechanical Phase Transition

**concept_id:** `regime-as-phase-transition`
**filed_at:** 2026-05-22
**lineage:** Gentner structure-mapping; physics → finance via critical-phenomena formalism
**status:** seed (demonstrates the format; testable predictions noted below)

---

## Target structure (qlibQuantExp)

The system being explained:

- A panel of N=158 daily factors over B≈300 same-day stock observations.
- Cross-sectional alpha (`rank_ic`) exhibits non-stationarity: certain regimes (e.g. 2018Q3 in CSI300) show abrupt drops in factor effectiveness, while others (e.g. 2017H1 bull) show sustained high IC.
- The model treats this as "the relative value of temporal vs factor reasoning changes per regime" (project hard core HC-1).

**Relations of interest (the things we want to map):**
- R1: A latent control parameter changes slowly with macro state.
- R2: A measurable "order" quantity (here: PC1 ratio of factor returns) responds to the control parameter.
- R3: At some critical value of the control parameter, the system reorganizes — discontinuity in the order quantity, divergence in fluctuations.
- R4: Different "phases" support different statistical regularities; what works in one phase fails in another.

## Base candidate — statistical-mechanical phase transition

The Ising-model / mean-field framework for second-order phase transitions in physics.

| Base concept (physics) | Target concept (qlib) | Mapped relation |
|---|---|---|
| Control parameter (temperature T, magnetic field h) | Macro regime signal: PC1 ratio, market crowding, volatility regime | R1: both vary slowly w.r.t. time |
| Order parameter (magnetization M) | PC1 ratio of factor returns; cross-stock correlation | R2: monotone response to control |
| Susceptibility χ = ∂M/∂h | RankIC standard error across days; gate-entropy variance | R3-precursor: spikes near the critical point |
| Correlation length ξ | Cross-stock co-movement length (sector-wide, theme-wide) | R3: diverges near critical |
| Critical temperature T_c | A specific macro-state boundary (e.g., PC1 ratio threshold) | R3: discontinuity location |
| Phase A (ordered) | High-correlation regime (low cross-sectional dispersion) | R4: factor-momentum strategies underperform |
| Phase B (disordered) | Low-correlation regime (high dispersion) | R4: cross-sectional ranking is most useful |
| Universality class | Set of regime triggers that produce similar dynamics | R4-meta: different crises share statistical signatures |

## Derived predictions (Phase 4 of structure-mapping)

The point of the analogy is **falsifiable predictions in the target domain**. Three:

**Pred-1 (Susceptibility precursor).** The cross-sectional variance of `rank_ic_daily` (call it χ_ic, computed as std-across-stocks of per-stock IC in a 20-day rolling window) should spike **2–10 trading days BEFORE** an abrupt RankIC drop, not coincidently with it. The physics analog: χ diverges at T_c, not at T < T_c. **Test:** compute leading correlation between χ_ic and ΔRankIC over the test segment; expect a 2–10 day lead, not a 0-day lag.

**Pred-2 (Correlation-length spike near boundary).** Effective cross-stock correlation length (defined as the e-fold decay scale of pairwise return co-movement across sectors) should rise as we approach a regime boundary. The physics analog: ξ → ∞ at T_c. **Test:** bucket test days by their proximity to known macro-shock dates (e.g., 2018-10-11 trade-war escalation); expect ξ_eff to be largest in the 10–20 day window before such dates.

**Pred-3 (Universality across crises).** Different crisis episodes (2018 trade war, 2020 COVID shock, hypothetical future events) should produce **similar critical-exponent fingerprints** in the (χ_ic, ξ_eff, ΔPC1) space — even though the macro causes differ. The physics analog: universality classes don't care about microscopic details. **Test:** fit power-law exponents per crisis; expect them to cluster, not scatter.

## Systematicity verdict

This analogy passes Gentner's systematicity test moderately well:

- ✅ **5 mapped relations** form an interconnected web (R1 feeds R2 feeds R3 feeds R4).
- ✅ The mapping carries a **specific prediction** (Pred-1 lead-time) that would not arise from a vague "regimes change correlations" framing.
- ⚠️ One weakness: in physics, ξ is unambiguously defined; in finance, "cross-stock correlation length" requires a choice of distance metric (sector / market-cap / time-clustered). The mapping is sensitive to this choice.
- ⚠️ Universality classes (Pred-3) have a much weaker basis in finance than in physics; the equivalent of "renormalization-group flow" is not obvious. Demote Pred-3 to *exploratory*, not central.

## Next steps if pursued

1. Operationalize χ_ic and ξ_eff in `scripts/precompute_market_state.py` as new macro features (additions to the protective belt).
2. File a falsifier on Pred-1 via `qx-forge-a-sharp-falsifier`.
3. If Pred-1 holds, consider whether the lead-time signal can be added to `regime_encoder.py` as an additional input (this would touch HC-1; classify via `qx-protect-or-relax-the-hard-core` first).

---

*This seed analogy demonstrates the format. Future analogies are saved as `shared/analogies/<concept-name>.md` using the same template.*
