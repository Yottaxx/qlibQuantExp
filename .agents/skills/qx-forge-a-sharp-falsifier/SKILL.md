---
name: qx-forge-a-sharp-falsifier
description: Use when you have a soft claim, an architectural hypothesis, or a candidate alpha and want to convert it into a risky, pre-registered prediction whose failure would refute the idea. Asks; what does this theory FORBID? Constructs a falsifier card with ex-ante decision rule, dataset/window/metric specified before any run, anti-cheat guards against "conventionalist twist" rescues. Triggers; "does X work?", "is Y better?", "I think the model is capturing Z", any pre-experiment design step. Pairs with qx-protect-or-relax-the-hard-core when the falsifier touches a core commitment. Append the resulting decision to qx-research-ledger. The output replaces hand-wavy "let's see if IC improves" with a numbered prohibition.
license: MIT
---

# Forge a Sharp Falsifier

## Overview

A claim is not a hypothesis until it has a prohibition: a statement of what should *not* happen if the claim is true. This skill converts a soft research claim into a falsifier card — a single artifact specifying the metric, the window, the threshold, and the anti-rescue clauses — all before any data is touched.

The output is one file in `shared/falsifiers/<claim-id>.md`. The corresponding decision/sweep_start is also appended to `shared/ledger.md` via `qx-research-ledger`.

## Posture

- The harder the falsifier, the better the theory. A claim that forbids almost nothing is worthless.
- Specify everything ex-ante. If the metric is chosen after seeing the result, the result is no longer a test.
- Watch for the "conventionalist twist" — Popper's name for the rescue maneuvers that preserve a theory by abandoning its empirical content. The skill catalog lives in `references/conventionalist_twist_catalog.md`; consult it before signing the card.

## Quick Reference

| Phase | Question | Output |
|---|---|---|
| 1. State the claim | What exactly is being claimed? Strip all hedging. | a one-sentence statement |
| 2. Find the prohibition | What does this claim FORBID? What outcome would force us to abandon it? | "If X then we are wrong" |
| 3. Specify the ex-ante metric | Metric name, dataset, time window, statistical test | a row in the card |
| 4. Specify anti-rescue rules | What re-segmentations / subset selections / regime exclusions are forbidden? | a numbered list |
| 5. Commit | Sign the card. From this moment, the prediction is binding. | timestamped row in ledger |

## Dialogue flow

**Phase 1 — State the claim.** Take what the user just said, strip the hedges, write one sentence. "I think the temporal-evidence readout will improve RankIC" becomes "Replacing `h[:, -1]` readout with a regime-conditioned attention-over-time pool will raise CSI300 5-seed RankIC mean by at least 0.003 absolute over test 2017-01-01 to 2020-08-01, while CSI800 in the same window shows the same direction of effect."

**Phase 2 — Find the prohibition.** The strongest test is one where the theory makes a *risky* prediction (in Popper's sense): an outcome that we would *not* have expected absent the theory. The prohibition is the converse. Examples:
- Claim: "Attention pooling beats last-step." Prohibition: "If `static_attn` beats `last_step` by at least 0.003 absolute on ≥ 4/5 seeds, but `regime_attn` does **not** beat `static_attn` by ≥ 1 standard deviation, we have *only* shown that attention pooling helps, not that regime-conditioning helps."
- Claim: "EMA reduces seed variance." Prohibition: "If EMA does not reduce the std-across-seeds of valid RankIC by at least 10%, this skill's seed-variance claim is refuted (the mean may or may not move)."

A prohibition that everyone would agree with after the result ("the model should produce something useful") is not a prohibition at all.

**Phase 3 — Specify the ex-ante metric.** Required fields on the falsifier card:

| Field | Example |
|---|---|
| `claim_id` | `A1-regime-attn-2026-05-22` |
| `parent_baseline_sha` | `3675090` |
| `universe` | `csi300` and `csi800` |
| `horizon` | `t+5` |
| `metric` | `rank_ic_daily` mean over test segment |
| `window` | `2017-01-01 .. 2020-08-01` (no extensions) |
| `decision_rule` | mean delta vs baseline ≥ +0.003 absolute on ≥ 4/5 seeds, both universes |
| `null_hypothesis_form` | delta ≤ 0 |
| `multiple_testing_note` | n_trials_at_filing (read from ledger) |

**Phase 4 — Specify anti-rescue rules.** Before any data, list which moves are *forbidden* if the result disappoints:
1. **No segment surgery.** Cannot redefine the test window or exclude regimes after seeing results.
2. **No metric swap.** Cannot switch to IC if RankIC fails, or to a different horizon.
3. **No seed selection.** Must report all 5 seeds; cannot drop "outliers".
4. **No subgroup mining.** Cannot claim victory on "the bull regime subset" if the overall test failed.
5. **No reinterpretation.** Cannot retroactively say "we were really testing X" when we wrote "we were testing Y".

Any rescue requires writing a new falsifier card with the rescue admitted in the metadata, and a new `claim_id`.

**Phase 5 — Commit.** Write the card to `shared/falsifiers/<claim-id>.md`. Append a `sweep_start` entry to `shared/ledger.md` (call `qx-research-ledger` Append mode). The card is now binding; the experiment may proceed.

## Output

`shared/falsifiers/<claim-id>.md` using `templates/falsifier_card.md`.

## Posture line

> "Every good scientific theory is a prohibition: it forbids certain things to happen. The more a theory forbids, the better it is." — Popper (1963)

## See also

- `qx-protect-or-relax-the-hard-core` — if the claim touches a core commitment, classify the move there before forging the falsifier.
- `qx-research-ledger` — the `sweep_start` and later `sweep_done` / `result` entries are recorded there.
- `qx-notebook-the-anomaly` — if the falsifier fails, the *meaning* of failure goes to the notebook, not back into the falsifier card.
- `shared/lineages.md` §2 Popper.
