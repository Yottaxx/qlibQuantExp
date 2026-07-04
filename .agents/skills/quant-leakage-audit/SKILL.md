---
name: quant-leakage-audit
description: Static-audit the current RST-MoE training config for known leakage classes — Qlib DK_I/DK_L processor boundary, precompute_market_state past-only flags, label-feature horizon alignment, train-only PCA fit, normalization fit on test data. Invoke before any GPU spend, after /quant-hypothesis, or whenever data config or feature pipeline changes. Pass/fail report; on fail, blocks downstream skills.
---

# /quant-leakage-audit

Cheap, fast gate. Always run before paying GPU hours. The classes of leakage below are the ones this project has historically had to defend against.

## When to use

- Before launching any new training run.
- After any change to `work_flow.py` data config, `precompute_market_state.py`, or feature handlers.
- When a result looks "too good" (rank_ic > +1% over baseline with no architectural reason).

## What to read first

1. `work_flow.py:68-150` — `data_conf`, `infer_processors`, `learn_processors`.
2. `scripts/precompute_market_state.py` — `--state_delta_lags`, `--add_market_ts`, `--zscore_windows`, `--pca_fit_on`, `--macro_scale`, `--warmup_trading_days`.
3. `module/utils/market_state.py` — `market_state_shift`, `market_state_strict`.
4. `DESIGN.md` § "Known Caveats / Residual Risks".
5. `git diff <last-green-sha> HEAD -- work_flow.py module/ scripts/precompute_market_state.py` — the surface that may have introduced new leakage.

## Audit checklist

For each item, output ✅ pass / ❌ fail / ⚠️ warning + one-line evidence.

1. **Processor boundary**: `learn_processors` (DK_L) used only by train; `infer_processors` (DK_I) used by valid/test/predict. `DropExtremeLabel` and label-norm only in `learn_processors`. **Fail** if any label-touching processor is in `infer_processors`.
2. **Label horizon vs feature window**: with `step_len=8` and label `Ref($close, -k) / Ref($close, -1) - 1`, confirm `k` matches the documented horizon (t+5 → `k = -5` per Qlib convention). Off-by-one here destroys claims silently.
3. **Market-state shift**: `market_state_shift >= 1`. If 0, the regime embedding sees same-day cross-section stats, which is leakage for next-day prediction.
4. **PCA fit scope**: `precompute_market_state.py --pca_fit_on=train`. **Fail** on `all` or unset.
5. **Macro normalization scope**: `--macro_scale={zscore,robust}` fit on train only. **Fail** on test-fit.
6. **Rolling / z-score / Δstate past-only**: every rolling window in `precompute_market_state.py` must use trailing-only definitions (no centered windows). Spot-check `--roll_mean`, `--zscore_windows`, `--state_delta_lags`.
7. **Market-TS past-only**: `--add_market_ts` features (return/vol/momentum/drawdown) must be computed on benchmark close lagged ≥ 1 day for t+1; ≥ k days for t+k.
8. **Strict NaN handling**: `market_state_strict=True` is **safer** for paper claims (raises on missing/NaN); if `False`, confirm the silent-fillna path is intentional.
9. **Warmup coverage**: `--warmup_trading_days` ≥ max(rolling windows, zscore windows, state_delta_lags) + buffer; otherwise early train rows have NaN-derived features.
10. **CSRankNorm / CSZScoreNorm scope**: cross-sectional, per-day. **Fail** if any cross-day normalization is fit on labels.
11. **Survivorship / suspension**: confirm Qlib's `Alpha158` handler instrument list is a *point-in-time* index (CSI300 constituents at the time of the date), not a forward-filled "stocks that ever were in CSI300". Note in report; this is a known weak spot.
12. **Diff scan**: run `git diff <last-green-sha> HEAD --stat` over the surfaces above; flag any new file added under `module/architecture/` or `module/dataloader/` that touches data flow.

## Output contract

```
PASS|FAIL — N/12 checks passed
[per-check status + evidence]

Diff vs last green: <git diff stat>

Recommendation: <next skill or fix-list>
```

Append a ledger row (`stage="leakage"`, `verdict="pass"` or `"fail"`).

## Failure modes / loop-back

- Any **fail** → block `/quant-minimal-repro` and `/quant-ablation-plan`. Loop back to fix the data pipeline; re-audit until pass.
- Multiple **warnings** → record in card's `Risk / failure mode` field; proceed with elevated scrutiny.
