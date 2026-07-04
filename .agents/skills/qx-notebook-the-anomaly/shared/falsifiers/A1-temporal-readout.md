# Falsifier Card — A1 Temporal Evidence Readout (4-arm)

**claim_id:** `A1-temporal-readout-4arm-20260522`
**filed_at:** 2026-05-22
**filed_by:** night-plan Tier-1
**parent_baseline_sha:** `3675090` (macV2)
**status:** open (not yet executed)

---

## The claim — stripped of hedging

Replacing the V2 last-step readout `h[:, -1, :, :]` with a **regime-conditioned attention pool over T** will raise CSI300 5-seed `rank_ic_daily` mean by at least **+0.003 absolute** over the test segment `2017-01-01 .. 2020-08-01`, on **at least 4 of 5 seeds**, while CSI800 in the same window shows the same direction of effect (mean delta ≥ 0).

## The prohibition (what this claim FORBIDS)

| Prohibition | What its failure means |
|---|---|
| **P1.** If `static_attn` beats `last_step` by ≥ +0.003 absolute on ≥ 4/5 seeds, but `regime_attn` does **not** beat `static_attn` by ≥ 1 standard deviation across seeds, we have shown only that *attention-pooling-over-T helps* — not that *regime-conditioning the query helps*. The skill A1's specific claim about regime-conditioning is then refuted; we adopt `static_attn` (Occam). |
| **P2.** If `mean_t` (uniform mean over T, zero new parameters) beats `last_step` by ≥ +0.003 absolute, but `regime_attn` does **not** beat `mean_t` by ≥ 1 std, the value-add of any attention-pooling at all is refuted; we adopt `mean_t`. |
| **P3.** If `regime_attn` does not beat `last_step` by ≥ +0.003 absolute on ≥ 4/5 seeds on CSI300, the entire A1 hypothesis is refuted regardless of CSI800. |
| **P4.** If CSI300 passes but CSI800 shows mean delta < 0, the result is regime-specific and the A1 hypothesis is **partially** refuted (it does not generalize); flag as a regime-bucketed claim, not a universal one. |

These are the *risky predictions* (Popper) — outcomes we would not expect absent the A1 hypothesis.

## The 4-arm ablation

| Arm | Description | Parameters added |
|---|---|---|
| `last_step` | V2 baseline. `h[:, -1, :, :]` | 0 |
| `mean_t` | Uniform mean over T | 0 |
| `static_attn` | Learned query, attention over T (no regime input) | ~2D² |
| `regime_attn` | Regime-conditioned query, attention over T | ~3D² |

Each runs 5 seeds × 2 universes. **Identity-start gate** keeps each non-baseline arm initialized to behave like `last_step` so an OFF→ON flip starts at parity.

## Ex-ante metric specification (binding)

| Field | Value |
|---|---|
| `universe` | `csi300` (primary), `csi800` (corroboration) |
| `horizon` | `t+5` (label expression: `Ref($close, -5) / Ref($close, -1) - 1`) |
| `metric` | `rank_ic_daily` mean across the full test segment |
| `window` | `2017-01-01 .. 2020-08-01` (no extension) |
| `seeds` | `[42, 43, 44, 45, 46]` (exactly these 5) |
| `decision_rule` | mean delta vs `last_step` ≥ +0.003 absolute, on ≥ 4 of 5 seeds, on CSI300 |
| `secondary_decision` | regime_attn vs static_attn delta ≥ 1 std → adopt regime_attn; else adopt static_attn |
| `null_hypothesis` | mean delta ≤ 0 |
| `n_trials_at_filing` | 0 (read from `shared/ledger.md`) |
| `multiple_testing_note` | This is the project's first formal falsifier filing; cumulative trial count starts here. Deflated-Sharpe haircut deferred until n_trials > 20. |

## Anti-rescue rules (binding)

1. **No segment surgery.** Cannot redefine the test window or exclude regimes after seeing results.
2. **No metric swap.** Cannot switch the decision rule to IC if RankIC fails.
3. **No seed selection.** All 5 seeds must be reported; cannot drop "outliers".
4. **No subgroup mining.** Cannot claim victory on "the bull regime subset" if the overall test failed.
5. **No reinterpretation.** Cannot retroactively say "we were testing X" when the card says "we were testing Y".
6. **No arm-set surgery.** All 4 arms must complete. Cannot drop `mean_t` or `static_attn` if those happen to beat `regime_attn`.

Any rescue requires filing a **new** falsifier card with a new `claim_id` and the rescue admitted explicitly in `motivation`.

## Settlement protocol

When sweep completes:
1. `qx-research-ledger` appends `sweep_done` rows.
2. Compute the 4 arm-level results. Apply decision rule **as written above**.
3. Append a `kind=result` row and a `kind=decision` row to `shared/ledger.md` citing this file.
4. Update this card's `status` to `settled-pass` / `settled-fail` / `settled-partial`. Do not edit the prior content.
5. If the result touches a hard-core item (e.g., refutes that regime conditioning matters), trigger `qx-protect-or-relax-the-hard-core`.

---

*Filed against `analysis/night_plan_change_set.md` §A1 + `analysis/night_plan_dialectic.md` §1 (the 4-arm hardening). Status open until sweep complete.*
