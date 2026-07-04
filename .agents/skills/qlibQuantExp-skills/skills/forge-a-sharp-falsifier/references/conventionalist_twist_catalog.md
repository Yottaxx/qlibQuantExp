# Conventionalist Twist Catalog — 12 Rescues to Refuse

The most common ad-hoc rescues that destroy a falsifier's scientific status. When a result disappoints, this is the list to consult *before* changing anything about the card.

## 1. Segment surgery
Redefine the test window after seeing the result. "Let's exclude 2018Q3 because of the trade war." — refuse.

## 2. Metric swap
"RankIC didn't move, but look — Information Ratio did." — refuse if RankIC was the ex-ante metric.

## 3. Seed selection
"Seed 44 is an outlier; let's drop it." — refuse. Report all seeds.

## 4. Subgroup mining
"Overall didn't pass, but bull-regime subset did." — refuse, unless bull-regime was the ex-ante claim. If overall was claimed, overall result stands.

## 5. Reinterpretation
"We were really testing whether attention pooling helps at all, not regime-conditioning specifically." — refuse if the card says regime-conditioning. File a new card.

## 6. Arm-set surgery
"Drop the `mean_t` arm; it makes our story messier." — refuse. All arms ex-ante must complete.

## 7. Threshold relaxation
"+0.003 was too aggressive; +0.001 should count too." — refuse for this filing. File a follow-on card with the relaxed threshold and a new claim_id, noting it.

## 8. Hidden hyperparameter sweep
"We tried 5 EMA decay values; 0.9995 worked." — multiple-testing violation. Pre-commit to one value, or pre-commit to a sweep and report all 5 results in the ledger.

## 9. Post-hoc auxiliary hypothesis
"It would have worked except for the data-quality issue in 2017." — refuse unless the data-quality issue was *named* in the card's anti-rescue clauses.

## 10. Negative-control elimination
"The baseline arm doesn't really represent V2 fairly — let's update it." — refuse. Baseline must be pinned in the card.

## 11. Hyperparameter coupling
"A1 works *if* we also turn on A3." — refuse for this filing. The card tested A1 alone. File a new card testing A1+A3.

## 12. Statistical-test substitution
"The t-test was insignificant, but the sign test was." — refuse. The ex-ante test stands.

## How the skill uses this

Before signing off on Phase 4 (anti-rescue rules), the skill consults this list and copies all 12 into the card. They form the binding refusals. Any rescue requires a new card.

## See also

- `popperian_falsifier_grammar.md` — Popper's claim (7) on the conventionalist twist.
- `templates/falsifier_card.md` — anti-rescue field.
