---
name: quant-regime-diagnose
description: Read recorder artifacts from one or more MLflow runs and produce a regime-health report (gate collapse, time_ratio drift, factor-gate stability, regime-bucket conditional rank_ic). Invoke after a training run completes, or to compare two runs head-to-head. Reuses module/model_adapter.py::export_visuals outputs and module/utils/regime_analysis.py.
---

# /quant-regime-diagnose

Packages the diagnostics already computed by `model_adapter.export_visuals` and `work_flow.py:1100-1220` into a single comparison report. The architecture-level diagnostics are the project's strongest defense against "rank_ic looks good but the model is broken" — gate collapse, regime overfit, FiLM saturation.

## When to use

- After any training run, before promoting it.
- When comparing two runs (baseline vs candidate) to back up `/quant-ablation-plan`'s decision.
- On suspicion of gate collapse (`time_ratio` outside `[0.3, 0.7]` in `kdd_report.md`).

## What to read first

1. `module/model_adapter.py` lines ~2653-2850 — `export_visuals()` writes `gate_series`, `gate_entropy_series`, `time_tau_series`, `factor_gate_entropy_series`, `factor_gate_topk_mass_10_series`, `attn_maps`, `factor_topk`, `factor_pool_topk`, `factor_gate_profiles`, `pool_profiles` into the recorder.
2. `module/utils/regime_analysis.py` — regime-bucket helpers.
3. `work_flow.py:1100-1220` — the metrics computation that produces `time_ratio_mean/std`, `collapse_ratio`, `entropy_norm_mean`, `tau_*`, `film_*`, `pc1_tail_2x2_rank_ic_spread`.
4. `mlruns/<exp_id>/<run_id>/artifacts/` — the recorder pickles (`run_summary`, `train_curve`, `diagnostic_matrix.csv`).

## Algorithm

1. Resolve the target run(s) — accept either an MLflow `run_id`, an experiment name + seed, or a path under `mlruns/`.
2. Load `run_summary` (pickle) — already contains aggregated metrics.
3. Compute / collect, for each run:
   - **Routing health**: `time_ratio` mean/std/p10/p90, `collapse_ratio` (% days outside `[0.1, 0.9]`), `gate_entropy_norm_mean` (1.0 = uniform, → 0 = collapsed).
   - **Time embedding**: `tau_mean`, `tau_p10/p90`, `tau_drift` (Δ between first / last 20% of test days), `tau_range_utilization`, Spearman(`tau`, `rank_ic_daily`).
   - **FiLM**: `film_gamma_strength`, `film_beta_strength`, `factor_gate_entropy_mean`, `top10_factor_stability_jaccard` (mean Jaccard of top-10 factor-gate set across consecutive days), `film_pool_overlap`.
   - **Attention**: `factor_entropy_norm`, `factor_top10_mass_mean`, `pooling_top10_mass_daily_mean`.
   - **Regime conditional**: `pc1_tail_2x2_rank_ic_spread` (rank_ic in the 4 PC1×tail buckets — large spread = the model behaves differently across regimes, which is what we want; small spread = single behavior averaged).
4. Apply health rules — emit ✅/⚠️/❌ for each:
   - `time_ratio` mean ∈ `[0.3, 0.7]` and `collapse_ratio < 0.2` → ✅
   - `gate_entropy_norm_mean > 0.6` → ✅
   - `tau_range_utilization > 0.4` → ✅ (model uses its allowed τ range)
   - `top10_factor_stability_jaccard > 0.6` → ✅ (FiLM is consistent enough to be trusted; < 0.3 means FiLM is randomly flipping, suspect noise-fitting)
   - `pc1_tail_2x2_rank_ic_spread > 0.01` → ✅ (genuine regime-specialization)
5. **Verdict**:
   - All ✅ → `healthy`
   - Any ❌ → `unhealthy` (regime-collapse / FiLM-noise / gate-collapse)
   - Mix of ✅ / ⚠️ → `borderline`
6. **Comparison mode** (two runs): produce a side-by-side table; the "winner" must beat the other on `rank_ic_daily_mean` AND not regress on any health rule.
7. Append a ledger row (`stage="regime"`, `verdict="info"` for single-run, `verdict ∈ {promote, reject}` for comparison-mode).

## Output contract

```markdown
# Regime diagnose: <run-id>

## Summary
verdict: healthy | borderline | unhealthy
rank_ic_daily_mean: 0.0786 (mean) ± 0.0042 (std across days)

## Routing health
- time_ratio: 0.52 ± 0.11 (collapse_ratio=0.05) ✅
- gate_entropy_norm: 0.78 ✅
...

## FiLM
- top10_factor_stability_jaccard: 0.71 ✅
...

## Regime conditional rank_ic (PC1 × tail 2×2)
| | tail-low | tail-high |
| pc1-low | 0.072 | 0.081 |
| pc1-high | 0.069 | 0.085 |
spread: 0.016 ✅

## Recommendation
<promote / reject / further-investigation>
```

## Failure modes / loop-back

- **Recorder artifacts missing** → run was incomplete; ask user to re-run with `export_visuals=True` (the project's default already enables it).
- **Verdict `unhealthy` on a candidate** → reject; loop back to `/quant-hypothesis` with the specific health rule that failed in the failure-mode field.
- **`pc1_tail_2x2_rank_ic_spread` near 0** → the model is *not actually regime-specialized*; even if `rank_ic` looks good, the architectural premise of RST-MoE is not confirmed. Flag prominently.
