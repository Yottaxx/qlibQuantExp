---
name: quant-minimal-repro
description: Generate and launch a cheap proxy run (≤10–15 min on a single workstation GPU) to early-reject a hypothesis before committing to a full ablation. Invoke after /quant-hypothesis and a passing /quant-leakage-audit. Uses work_flow.py's QIB_*_OVERRIDES_JSON env-var protocol; never modifies work_flow.py itself.
---

# /quant-minimal-repro

The single-GPU killer feature. A proxy that reliably rejects ~80% of bad ideas in 10 min beats a full 4 h run that confirms what was already obvious.

## When to use

- Right after `/quant-hypothesis` produces a card and `/quant-leakage-audit` passes.
- Never directly from a vague idea — always read the latest hypothesis card first.

## What to read first

1. The latest hypothesis card in `experiments.md` (the one matching `card_id` from the most recent `stage="hypothesis"` ledger row).
2. `work_flow.py:2568-2620` — env-var override entry point (`QIB_RUN_SETTING`, `QIB_MODEL_OVERRIDES_JSON`, `QIB_TRAINER_OVERRIDES_JSON`, `QIB_DATA_OVERRIDES_JSON`).
3. `run_matrix.yaml` — schema for the override block.
4. `experiments_ledger.jsonl` — past `stage="proxy"` entries to estimate proxy wall-clock.

## Proxy recipe (default; adjust per card)

A proxy is a *deliberately downscaled* version of the full experiment. Defaults that target ~10 min on a workstation GPU:

| Knob | Full | Proxy |
|---|---|---|
| `n_epochs` | 40 | 4 |
| `early_stop` | 5 | 2 |
| `seeds` | 5 | 1 (seed=42) |
| Universe | csi300 | csi300 *with date-range trim* (e.g. train: 2014-01..2020-03; valid: 2020-04..2020-12) |
| `step_len` | 8 | 8 (do not change — alters semantics) |
| `batch_size` | 128 | 128 (do not reduce below 128 — internal regime-stat batching requires it; see `DESIGN.md`) |

Trim **time range**, not stocks — reducing batch_size below 128 invalidates the regime encoder per `DESIGN.md` §"Batch Requirement".

## Algorithm

1. Read the latest hypothesis card; extract its `Config delta`.
2. Merge the card's delta into the proxy recipe above to produce three JSON dicts:
   - `model_overrides` (architecture knobs the card touches)
   - `trainer_overrides` (`{"n_epochs": 4, "early_stop": 2, "seed": 42}` plus card-specific)
   - `data_overrides` (date-range trim)
3. Emit the launch command:
   ```bash
   QIB_RUN_SETTING=proxy_<card_id> \
   QIB_MODEL_OVERRIDES_JSON='<json>' \
   QIB_TRAINER_OVERRIDES_JSON='<json>' \
   QIB_DATA_OVERRIDES_JSON='<json>' \
   python work_flow.py
   ```
   (On Windows bash the same; on PowerShell, set via `$env:QIB_*`.)
4. After the run completes, parse the latest MLflow run's `run_summary` artifact for `rank_ic_daily` (eval), `time_ratio` mean, `gate_entropy` mean.
5. Decision:
   - **Reject** if proxy `rank_ic_daily` < baseline `rank_ic_daily` − 0.5% (single seed signal is noisy; require *worse* than baseline by a margin).
   - **Reject** if `time_ratio` collapses (mean outside [0.2, 0.8] or collapse_ratio > 0.4).
   - **Inconclusive** if proxy `rank_ic_daily` is within ±0.5% of baseline → still proceed to `/quant-ablation-plan` (proxy noise too high for a confident reject).
   - **Promote-to-ablation** if proxy `rank_ic_daily` ≥ baseline + 0.3% AND no collapse → strong signal, ablation likely to confirm.
6. Append a ledger row (`stage="proxy"`, `verdict ∈ {reject, inconclusive, promote}`, full proxy metrics in `metrics`, `wall_clock_s` measured).

## Proxy-vs-full calibration

The decision thresholds above are conservative initial defaults. Once 5+ proxy/full pairs exist in the ledger, run a one-off calibration: Spearman correlation of proxy_rank_ic vs full_rank_ic. If ρ < 0.5, the proxy is uninformative — widen the proxy (more epochs, more data) until ρ > 0.7. Persist the calibration as a comment in this skill's frontmatter, not as a separate file.

## Output contract

- Stdout: launch command + decision rule.
- After run: parsed metrics dict + verdict.
- Ledger row.

## Failure modes / loop-back

- **Wall-clock > 30 min** → proxy is too large; cut date range further and re-emit.
- **`time_ratio` collapsed in proxy** → loop back to `/quant-hypothesis` with the failure mode noted; the architectural change likely interacts badly with routing.
- **Inconclusive on 3+ proxies in a row for similar cards** → the proxy is too noisy for this family of changes; widen it (record this in the calibration note).
