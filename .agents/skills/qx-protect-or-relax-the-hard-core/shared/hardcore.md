# Hard Core, Protective Belt, and Red-Flag Log

This is the project's Lakatosian manifesto. It is **the** document everything orbits.

**Reading rules:**
- The hard core lists commitments the programme refuses to give up. Touching them = paradigm shift (a different research programme), not a tweak.
- The protective belt is everything else — auxiliary hypotheses that may be modified, even aggressively, without changing the programme's identity.
- The red-flag log records every time a proposed move was rejected for monster-barring (excusing a failed result by redefining what counts).

**Mutation rules:**
- This file is append-only. Never delete prior amendments or red flags.
- Mutations go through `qx-protect-or-relax-the-hard-core`.
- Every mutation triggers a corresponding `kind=decision` row in `shared/ledger.md`.

Initial manifesto authored 2026-05-22 from `analysis/motivation_novelty.md` §3–§4 and `analysis/night_plan_dialectic.md` §6. The hard-core items below are the **strong draft**; the project owner should review and amend them within the first research week using `qx-protect-or-relax-the-hard-core`.

---

## Hard core (initial — to be ratified by the project owner)

These commitments define the RST-MoE research programme. Surrendering any of them means starting a different programme.

### HC-1. Non-stationarity is a structural problem, not noise.

The model treats regime shifts as **changes in the computation path**, not as parameter drift to be averaged through. A regime signal controls *what reasoning happens*, not just *how strongly each input is weighted*. Replacing this with weight-averaging or input-reweighting alone is a different programme.

> Anchor: `analysis/motivation_novelty.md` §1–§3.

### HC-2. Routing is a soft mixture, not a hard switch.

The gate produces a continuous distribution over the two experts. Top-1 hard routing (Switch-Transformer style) is excluded — even when it would save FLOPs. The reason is that the *blend* between time and factor reasoning is itself an object of interest; hard routing throws this away.

> Anchor: `analysis/motivation_novelty.md` §4 "soft mixture, not discrete hard switching"; `analysis/night_plan_master.md` §4 (rejection of Soft MoE / Expert-Choice).

### HC-3. Experts are axis-heterogeneous (time vs factor), not homogeneous FFNs.

The two MoE experts are structurally distinct: one attends over time at each factor, one attends over factors at each time step. Replacing either with a same-shape FFN, or making them both attention-over-the-same-axis, is a paradigm shift.

> Anchor: `analysis/motivation_novelty.md` §4.5 "structural separation … (time-attention vs factor-attention)".

### HC-4. Conditioning is post-LayerNorm with identity-start.

FiLM modulation is applied *after* the block's LayerNorm (so it is not cancelled by Pre-LN), and its projections are zero-initialized so γ≈1, β≈0 at training start. Re-introducing pre-LN gating or non-identity initialization breaks the architecture's stability story.

> Anchor: `analysis/embedding_design.md` §3.4; `analysis/design_analysis.md` §2.

### HC-5. No look-ahead in macro features. Train-only fits.

PCA, RobustScaler, shock thresholds, and any other distributional statistic in the macro pipeline are fit on the **train** split only and then *transformed* on valid/test. `market_state_shift` is set such that day-T features predict day-(T+k) returns; no future bars enter the past-only roll/zscore/Δ features. Violating this voids any KDD-publishable claim.

> Anchor: `analysis/macro_feature.md`; `analysis/design_analysis.md` §3.

### HC-6. *(Open slot — to be added by the project owner)*

Reserved. The owner should add HC-6 if a sixth project-defining commitment exists that does not collapse into HC-1..HC-5. Candidates from the existing analysis docs include: "evaluation must include 5-seed mean±std", "we report no result without HAC t-stat", "we never run two structural changes through one data pipeline at once". Decide via `qx-protect-or-relax-the-hard-core`.

---

## Protective belt (modifiable — may evolve per cycle)

Auxiliary hypotheses and specific design choices that *implement* the hard core but are themselves subject to modification.

| Belt item | Current value | Modification authority |
|---|---|---|
| `d_model` | 128 | any sweep |
| `n_layers` | 4 | any sweep |
| `n_heads` | 4 | any sweep |
| `d_ff` | 256 | any sweep |
| `pooling_alpha` (base) | 0.7 | any sweep |
| `time_tau_init` | 5.0 | any sweep |
| `factor_gate_scale` | 0.5 | any sweep |
| `router_temperature` | 0.7–1.0 | any sweep |
| `router_z_loss_coef` | 0.01 | any sweep |
| `use_alibi` | false | any sweep |
| `value_embedding_type` | `shared_linear` | any sweep |
| `use_feature_selection` (STG) | false | any sweep |
| Universe choice | CSI300 / CSI800 | any sweep |
| Label horizon | t+5 | any sweep (but cross-horizon comparisons are protective-belt evidence, not hard-core test) |
| Time-axis readout | `h[:, -1]` (V2) | currently under Tier-1 A1 ablation |
| Weight averaging (EMA) | off (V2) | currently under Tier-1 A3 ablation |
| Layer summary scope | per-block batch summary | currently A5 (deferred) |

The belt may grow new entries; old entries may be modified through any sweep that respects the hard core. Aggressive modification of the belt is *progressive* (Lakatos).

---

## Red-flag log (monster-barring incidents)

Every time a proposed move was rejected for monster-barring (rescuing a failed result by redefining what counts), it is recorded here permanently. The log exists so the same rescue is not re-proposed by a future-Claude or future-yotta who has forgotten the prior episode.

### 🚩 RF-2026-05-22-01 — macV3 hierarchical state field bundle

**Proposed:** Defend the regime-routing hypothesis by adding global+local hierarchical state + new market-state asset triples + day-summary observers + 4 new conditional branches, all in one diff (15,067 insertions across 42 files).

**Rejected because:** This bundles too many switches under one hypothesis. Per Lakatos's distinction, this is *degenerating*: many auxiliary modifications without producing novel falsifiable predictions. The "core" hypothesis (hierarchical state) cannot be attributed to gain over the baseline because too many co-changes happened simultaneously. This violates the project hard-core principle "never run two structural changes through one data pipeline at once" (candidate HC-6).

**Outcome:** macV3 set to `status=deferred pending V2 incremental exhaustion`. macV2 remains canonical baseline. The night-plan sequence (Tier-1 A1+A3+D1-D4 first, single-axis attribution) replaces V3.

**Cited:** `analysis/macV2_vs_macV3_quant_algorithm_diff.md`; `analysis/night_plan_master.md` §8.

---

## Future amendments

Append below using `## Core amendment YYYY-MM-DD` or `### 🚩 RF-YYYY-MM-DD-NN`. Each amendment must cite (a) the trigger, (b) the prior position, (c) the new position, (d) the cited artifact path.

---

## Core amendment 2026-05-24 — Ratify HC-6 (four-tuple promotion rule)

**Trigger.** The 2026-05-24 metric-responsiveness audit (`full_135` seed-42 manifest vs `cap06` seed-42 manifest, `diagnostic_runs/four_day_plan_20260501_125906/.../cap06_d96_h4_l2_ff384_do020/manifest.json` vs the user's pasted full_135 dump, cross-checked against `seed42_matrix_gradfix_20260430_2105/suite_diagnostic_summary.md`) showed that **`daily_rank_ic_mean` alone misranks two configurations** whose live-portfolio properties diverge materially: `cap06 − full_135 ≈ {+0.0003 RankIC, +0.522 IR_with_cost (+33%), +0.018 MaxDD (180 bp better), −45% post_peak_decay}`. Promoting by RankIC alone would have tied them; promoting by the four-tuple ranks cap06 strictly above full_135.

**Prior position.** HC-6 was the *open slot* (initial manifesto, line 53). Three candidate texts were listed: 5-seed mean±std, HAC t-stat, "never run two structural changes simultaneously". The audit reveals a different, more pressing gap: the *headline metric* itself was wrong.

**New position (ratified).**

### HC-6. Promotion requires a four-tuple headline + single-axis isolation.

No structural change enters the protective belt without satisfying **all three** of the following:

1. **Pre-registered falsifier.** A `shared/falsifiers/<claim-id>.md` card filed before the run, with seeds, dataset, window, and decision rule frozen ex ante. Anti-rescue clauses (no segment surgery, no metric swap, no seed selection, no subgroup mining, no arm-set surgery) are mandatory.
2. **Four-tuple headline, 5-seed mean ± std.** The decision metric is the tuple `{performance/daily_rank_ic_mean, portfolio/information_ratio_with_cost, portfolio/max_drawdown_with_cost, optimization/post_peak_decay}`. Promotion requires no element of the tuple to degrade vs the comparison anchor at 5-seed mean±std. Single-seed peaks are noise and may not be cited as promotion evidence.
3. **One belt parameter at a time.** At most one entry in the protective-belt table differs between the candidate config and its comparison anchor. Bundled changes are auditable monster-barring (per `RF-2026-05-22-01`).

> Anchor: `~/.claude/plans/tidy-sauteeing-pascal.md` Part A (metric-responsiveness audit) and Part B (seven findings, item 1).

**How this affects existing belt items.** No belt item is reclassified. The change is to the *decision procedure*, not to which items are belt vs core. The currently-canonical baseline (`full_135`, seed-42 single-split) is **not** promoted under HC-6 because it has not been measured at 5-seed mean±std on the four-tuple. Re-anchoring the baseline is a Track-6 deliverable.

**Decision row.** `kind=decision`, id=12 in `shared/ledger.md` (2026-05-24).

---

## Core amendment 2026-06-04 — Demote `post_peak_decay` to non-blocking advisory in HC-6

**Trigger.** The τ Phase-2 multi-seed sweep (6 scales × 3 seeds × 25 epochs, `analysis/tau_phase2_25e_3seed_20260604.csv`) produced a clear winner — `time_tau_mlp_out_scale=0.5` — that is the ONLY arm with a paired-significant IR edge (ΔIR +0.479 vs baseline, 3/3 seeds, t=6.90 df=2), sweeps ΔAnnRet 3/3 (+34% rel at flat turnover), improves MaxDD, and is RankIC-neutral. It fails the strict HC-6 four-tuple gate **only** on `post_peak_decay` (0.00403 vs the baseline-mean+1σ ceiling 0.00362 — a 0.0004 gap). A 4-agent adversarial-verification workflow (`wf_afa32915-371`) adjudicated this as a *category error* in the original gate.

**Prior position.** The 2026-05-24 ratification (above) made `post_peak_decay` one of four hard-gate elements: "Promotion requires no element of the tuple `{RankIC, IR_with_cost, MaxDD_with_cost, post_peak_decay}` to degrade vs the comparison anchor."

**New position (ratified by owner 2026-06-04).**

`post_peak_decay` is reclassified from a **hard promotion-veto** to a **non-blocking advisory**, for ALL current and future arms (this is a general rule, NOT a one-off carve-out for scale=0.5).

Rationale (instrument-validity): `post_peak_decay = best_valid_rank_ic − final_valid_rank_ic` measures how much the *last training epoch* over-trains relative to the peak. But the **deployed model is the best-epoch checkpoint** (`checkpoint_metric=valid_rank_ic`, `checkpoint_mode=max`) — by construction `post_peak_decay` cannot touch what ships. Gating a *deployment* decision on a *train-budget* metric is the wrong instrument.

- **Hard gates (unchanged, deployment-relevant):** `daily_rank_ic_mean`, `information_ratio_with_cost`, `max_drawdown_with_cost` — promotion still requires no element to degrade vs baseline mean−1σ at multi-seed mean±std.
- **Advisory (new):** a `post_peak_decay` above baseline+1σ does NOT block promotion; it raises a logged train-budget **action-item** ("evaluate early-stopping / shorter epoch budget for this config"). 

**Anti-laundering guard (per the skeptic's warning in `wf_afa32915-371`).** This amendment is written generally and logged with the deployed-checkpoint rationale precisely so it cannot be re-used as a per-result waiver. Any future arm that fails a *hard* gate (RankIC/IR/MaxDD) is still blocked. If a future change makes the deployed model NOT the best-epoch checkpoint (e.g. last-epoch deploy, or EMA-of-weights), this amendment is VOID and `post_peak_decay` reverts to a hard gate — the rationale is conditional on best-epoch-checkpoint deployment.

**Consequence for scale=0.5.** Under the amended gate, `time_tau_mlp_out_scale=0.5` passes all hard gates cleanly and carries a logged ppd advisory. It is promoted to **conditional Tier-S** — NOT production-blessed and NOT set as code default — pending Phase-3 walk-forward (the kill criterion, since RankIC-flat + turnover-flat means the IR edge could be single-test-split portfolio-realization variance).

**Decision row.** `kind=decision`, id=57 in `shared/ledger.md` (2026-06-04).

> **Addendum 2026-06-06.** scale=0.5 was subsequently REVERSED out of conditional Tier-S — the n=6 kill-check (3 fresh seeds {45,46,47}) collapsed its paired ΔIR from +0.479 (t=6.90) to +0.148 (t=0.92, NS); ledger rows 60-62. **This amendment is RETAINED nonetheless** (owner decision, ledger row 64): the instrument-validity rationale — `post_peak_decay` measures last-epoch over-training while the deployed model is the best-epoch checkpoint, so ppd cannot touch shipped performance — is independent of scale=0.5 and stands on its own merit. The "Consequence for scale=0.5" paragraph above is now historical context, not a live promotion. The self-voiding clause (revert to hard gate if deployment ever stops using best-epoch checkpoint) remains in force.

*— end of manifesto —*
