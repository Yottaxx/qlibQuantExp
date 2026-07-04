# Popperian Falsifier Grammar — the 7 Claims of Conjectures & Refutations §1

These are the seven points from Popper's introduction to *Conjectures and Refutations* (1963), translated into operational rules for filing a falsifier card.

## Popper's seven claims (paraphrased)

1. **Confirmations are easy.** It is easy to obtain confirmations, or verifications, for nearly every theory — if we look for confirmations.

2. **Confirmations should count only if they are risky.** Confirmations should count only if they are the result of risky predictions — predictions which, absent the theory, we would have expected to fail.

3. **Every good theory is a prohibition.** It forbids certain things to happen. The more a theory forbids, the better it is.

4. **A theory which is not refutable by any conceivable event is non-scientific.** Irrefutability is a vice, not a virtue.

5. **Every genuine test of a theory is an attempt to falsify it.** Testability is falsifiability — degrees thereof.

6. **Confirming evidence should not count except when it is the result of a genuine test of the theory.** In which case it can be presented as a serious but unsuccessful attempt to falsify the theory.

7. **Some genuinely testable theories, when found false, are upheld by their admirers** — for example by introducing some auxiliary hypothesis (ad hoc), or by re-interpreting the theory ad hoc in such a way that it escapes refutation. This is the **conventionalist twist** — it rescues the theory from refutation but at the price of destroying its scientific status.

## Operational rules for the skill

| Popper rule | Operational consequence in the falsifier card |
|---|---|
| (3) Theories should forbid | Required field: explicit **prohibition statement** ("if X then we are wrong") |
| (2) Predictions must be risky | The prediction must specify a metric/window/threshold under which the theory could fail. "It will improve something somewhere" is insufficient |
| (4) Irrefutability is a vice | The card must be **rejectable on its face** if no event would refute it. The reviewer (or `qx-skill-creator` validator) refuses to file an unfalsifiable card |
| (7) No conventionalist twist | Anti-rescue rules must be enumerated **ex-ante**. Any rescue requires a new card with the rescue admitted explicitly |
| (1, 5) Tests, not confirmations | A `result=positive` on a card with weak anti-rescue rules counts as weaker evidence than a `result=positive` on a card with strong anti-rescue rules |

## Common failure modes

- "I think A1 will improve the model" — **not a falsifier**. No prohibition, no metric, no window.
- "A1 should help" — same.
- "Let's see what happens" — same.
- "A1 will raise RankIC" — **weak falsifier**. No threshold, no universe, no seed count.
- "A1 will raise CSI300 5-seed `rank_ic_daily` mean by ≥ +0.003 absolute on ≥ 4/5 seeds over 2017-01-01 .. 2020-08-01" — **strong falsifier**. Specific in every dimension.

## See also

- `conventionalist_twist_catalog.md` — 12 specific rescue maneuvers to refuse.
- `templates/falsifier_card.md` — the binding format.
- Popper, K. (1963). *Conjectures and Refutations*, §1 — the canonical source.
