# Falsifier Card — Factor-axis Pooling Attention Rank-1 Collapse

**claim_id:** `pool-forensics-20260524`
**filed_at:** 2026-05-24
**filed_by:** plan v2 Track 1.5 (`~/.claude/plans/tidy-sauteeing-pascal.md`)
**parent_baseline_sha:** `3675090` (macV2)
**status:** open (not yet executed)
**independence:** This card is **architecturally independent** from `tau-collapse-forensics-20260524.md`. Per `shared/notebook/2026-05-24.md` Entry 3, τ and pool attention share no causal link. Both cards may run in parallel without violating HC-6 (each changes a different belt parameter against its own anchor).

---

## What this card is for

The `attention_pooling/pool_entropy_norm_mean` metric reads **0.99999** (six nines) in every diagnostic run on disk — meaning the `AdaptivePooling` layer's attention component is one part in 10⁵ away from uniform across N=158 factors. With `pooling_alpha=0.7` blending 70% attention with 30% mean, the blend is essentially mean-pooling regardless of α. The factor-attention mechanism, which is supposed to let the model emphasize which factors matter for a given sample, is functionally dead.

This card identifies which of three mutually-exclusive root causes is dominant, and which single belt-parameter change recovers ≥ 1 nat of attention concentration.

## The claim — stripped of hedging

The pool's rank-1 collapse is caused by **one** of:
- **R1.** Single-head, single-query setup: `n_heads=1` + 1-query × N-key attention with `softmax(QK^T/√D)` cannot escape entropy-maximizing default because of insufficient query expressivity. Multi-head (e.g., `n_heads=4`) provides 4 independent query subspaces and breaks the symmetry.
- **R2.** Blend with mean dilutes attention gradient: `pooling_alpha=0.7` puts 30% direct mean-pool, which gives the model a "free" gradient path that bypasses attention entirely. Setting `pooling_alpha=1.0` (pure attention) removes the bypass and forces attention to earn its keep.
- **R3.** Softmax temperature too cold: `scale = 1/√D = 1/8` (for D=64) is too aggressive a normalizer for the natural scale of `q·k_n` in this setup. Logits are O(0.1) instead of O(1); softmax → near-uniform.

At least one of R1, R2, R3 in isolation will raise `attention_pooling/pool_entropy_norm_mean` **below 0.95** on a single CSI300 seed-42 single-split `full_135` run, **without** any element of the four-tuple headline `{daily_rank_ic_mean, IR_with_cost, MaxDD_with_cost, post_peak_decay}` regressing past noise.

## The prohibition (what this claim FORBIDS)

| Prohibition | What its failure means |
|---|---|
| **P1.** If none of the three arms (R1/R2/R3 in isolation) drops `pool_entropy_norm_mean` below 0.95, the collapse is not solvable by simple architectural levers on the pool alone. Most likely the post-MoE factor representations themselves are too homogeneous (insufficient cross-factor variance feeding the pool). Escalate: file `factor-representation-variance-20XXXXXX.md` and read `final_norm` output statistics directly. |
| **P2.** If an arm drops `pool_entropy_norm_mean` < 0.95 but the four-tuple degrades by more than 1 noise unit (RankIC ≥ 0.003, IR_with_cost ≥ 0.20, MaxDD ≥ 100 bp, post_peak_decay ≥ 0.003), pool sharpening IS happening but harms downstream — meaning the model was actively benefiting from the uniform-pool / mean-pool baseline. Route to `/qx-protect-or-relax-the-hard-core` because this challenges the implicit assumption that attention pooling is the right aggregation primitive at all. |
| **P3.** If R1 (multi-head) and R2 (pure attention) both succeed but R3 (lower temperature) fails, the dominant cause is *insufficient query expressivity*, not *softmax saturation*. If only R3 succeeds, the opposite — temperature, not architecture, was the bottleneck. The combined outcome ranks the three causes. |
| **P4.** If `time_embedding/tau_range_utilization` rises alongside any successful arm (e.g., goes from 5e-4 to > 5e-3), it would refute the independence claim in `notebook/2026-05-24.md` Entry 3. File `pooling-tau-coupling-20XXXXXX.md` to investigate the indirect mechanism. (Mirror of P4 in `tau-collapse-forensics-20260524.md`.) |

## The 3 arms

| Arm | What changes | Belt parameter | Implementation | Wall-time |
|---|---|---|---|---|
| `baseline_repl` | nothing | n/a | re-use existing `full_135` seed-42 manifest | 0 (on disk) |
| `R1_multihead` | `n_heads_pool: 1 → 4` for pool only | new field `pool_n_heads` | needs plumbing (see pre-flight) | ~6 GPU-hr |
| `R2_pure_attn` | `pooling_alpha: 0.7 → 1.0` | existing belt field | pure `QIB_MODEL_OVERRIDES_JSON` | ~6 GPU-hr |
| `R3_temperature` | adjust softmax scale in `AttentionPooling.forward` | new field `pool_attn_logit_scale` | needs plumbing (see pre-flight) | ~6 GPU-hr |

Total new compute: ~18 GPU-hours (3 single-seed runs). R2 is currently pure-override; R1 and R3 need plumbing similar to what was done for `time_tau_mlp_out_scale`.

## Pre-flight check (read-only)

- **R1 plumbing.** `AttentionPooling.__init__(d_model, n_heads=1, dropout)` accepts `n_heads` (`attention_pooling.py:25`), but `QuantMoEModel` builds `AdaptivePooling(d_model, n_heads=1, ...)` with hardcoded `1` at `quant_moe_model.py:107`. Adding `pool_n_heads: int = 1` to `QuantMoEConfig` and one kwarg passthrough at line 107 is the same pattern as the `tau_mlp_out_scale` plumbing. **Constraint:** `d_model % n_heads == 0`; with D=64, options are {1, 2, 4, 8, 16, 32, 64}.
- **R2 plumbing.** `pooling_alpha` is already in `QuantMoEConfig` and consumed at `quant_moe_model.py:109` → `AdaptivePooling(alpha=...)`. **No plumbing needed; R2 is pure override.**
- **R3 plumbing.** `AttentionPooling` uses `nn.MultiheadAttention` which fixes `scale = 1/√d_head` internally. To override softmax temperature, the cleanest path is to add a learnable / configurable `logit_scale` parameter applied to `q` before the MHA call, OR to replace the MHA with a manual `softmax(QK^T * temperature)`. Either requires 5–10 lines of code change in `attention_pooling.py`. **Defer R3** unless P1/P3 escalation requires it — start with R1 + R2 only.

## Ex-ante metric specification (binding)

| Field | Value |
|---|---|
| `universe` | `csi300` |
| `horizon` | `t+5` |
| `decision_metric` | `attention_pooling/pool_entropy_norm_mean` (primary); four-tuple headline (degradation gate) |
| `secondary_diagnostic` | `attention_pooling/pool_top10_mass_mean` — promotion needs this to rise from ~0.063 to ≥ 0.15 (top-10 of 158 factors capture ≥ 15% of attention mass; uniform = 10/158 = 0.063) |
| `seeds` | `[42]` (forensic, not promotion) |
| `decision_rule` | any single arm raises `pool_entropy_norm_mean` to < 0.95 AND `pool_top10_mass_mean` ≥ 0.15 → corroborate that cause |
| `headline_gate` | four-tuple no element degrades beyond P2 thresholds |
| `n_trials_at_filing` | 2 (rows 13, 15 of ledger, before this card) |
| `multiple_testing_note` | Forensic, not promotion. Does NOT count toward Deflated-Sharpe haircut budget. |

## Anti-rescue rules (binding)

1. **No metric swap.** Cannot switch primary from `pool_entropy_norm_mean` to e.g. `factor_top10_mass_mean` (which is FiLM-side) post-result.
2. **No threshold drift.** "< 0.95 entropy AND ≥ 0.15 top-10 mass" is fixed; cannot weaken to "any improvement".
3. **No silent code change.** R1 / R3 plumbing diffs must be logged as `code-change` arms in the ledger before being run.
4. **No arm-set surgery.** Both R1 and R2 must complete (R3 deferred-by-design is OK; can't be retroactively added if R1+R2 fail).
5. **No reframing.** If R2 (pure attention) succeeds, the verdict is "the 0.3 mean-blend was a free gradient bypass", not "we always believed pure attention was better".

## Settlement protocol

When arms complete:
1. Append `kind=sweep_done` and `kind=result` rows to `shared/ledger.md` per arm.
2. Apply decision rule.
3. Append `kind=decision` row citing this card.
4. If P1 triggers, file `factor-representation-variance-20XXXXXX.md`.
5. If P2 triggers, route to `/qx-protect-or-relax-the-hard-core`.
6. If P4 triggers, file `pooling-tau-coupling-20XXXXXX.md`.
7. Update this card's `status` to `settled-{corroborate|refute-P1|refute-P2|partial-P3|coupled-P4}`.

---

*Filed against `~/.claude/plans/tidy-sauteeing-pascal.md` Track 1.5, `shared/notebook/2026-05-24.md` Entry 3, `shared/ledger.md` ids 13–18. Anchors `module/architecture/attention_pooling.py:25-50, 100-131`, `module/quant_moe_model.py:105-110`. Status open until sweep complete.*
