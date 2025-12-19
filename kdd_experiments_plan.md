# KDD Experiments Plan (RST-MoE, t+5, CSI300/CSI800, 5 seeds)

This document defines a **submission-grade** experimental plan for RST-MoE with a focus on:
1) convincing evidence for the **core claim** (regime-driven structural routing),
2) fair comparisons (strong baselines + controlled ablations),
3) reproducibility (one-click reports + strict leakage controls).

Target setting (as requested):
- Universes: **CSI300**, **CSI800**
- Horizon: **t+5** (label consistent with `Ref($close, -5) / Ref($close, -1) - 1`)
- Seeds: **5** (report mean ± std; plus significance tests)

---

## 0. Claims We Must Prove (keep it to 2)

**Claim A (Novelty):** Regime-driven **structural routing** between a *time expert* and a *cross-sectional factor expert* improves performance and robustness under market regime shifts, compared to static architectures or “soft” adaptation only.

**Claim B (Mechanism):** The gains are attributable to **regime conditioning** (router + regime-adaptive embeddings), not parameter count, training tricks, or reporting artifacts.

Everything below exists to support A/B with minimal ambiguity.

---

## 1. Fixed Experimental Protocol (do not change across runs)

### 1.1 Data & Splits
- Dataset: Qlib `Alpha158` (as in `work_flow.py` / `DESIGN.md`)
- Splits (official):
  - Train: 2008-01-01 ~ 2014-12-31
  - Valid: 2015-01-01 ~ 2016-12-31
  - Test: 2017-01-01 ~ 2020-08-01
- Task: daily cross-sectional ranking (CSI300/CSI800 universe)

### 1.2 Labels
- Use the configured label expression (t+5-style):
  - `Ref($close, -5) / Ref($close, -1) - 1`
- Training label processing:
  - Train uses `DK_L` with `CSRankNorm` (rank-label).
  - Valid/Test use `DK_I` (raw label).
- Paper text must explicitly state this (see “Threats to Validity”).

### 1.3 Macro Features (External Regime Signal)
For t+5, **default = external macro features**.
- Precompute market state for each universe:
  - `market_state_csi300.pkl`
  - `market_state_csi800.pkl`
- Leakage control:
  - PCA must be **fit on train only** (use `--pca_fit_on=train`).
  - Rolling/zscore/Δstate must be past-only by construction (current script already documents this).
- Training config:
  - `trainer_config.market_state_path=<...>`
  - `trainer_config.market_state_shift=0`
  - `trainer_config.market_state_strict=True`

### 1.4 Sampler & Batch Semantics (critical)
- Training must use same-day cross-section batches (`FixedDailyBatchSampler`).
- Internal regime stats (if used as fallback) require **large enough batch_size** to estimate crowding/correlation.
  - For paper runs: enforce `batch_size >= 128` (or use full-day via chunking only for eval).
  - Avoid the demo default `batch_size=4` for any internal-regime claim.

---

## 2. Metrics (what we report, exactly)

### 2.1 Primary Metrics (submission tables)
- **RankIC** mean and IR (daily series aggregated)
- **IC** mean and ICIR
- **HAC t-stat** (Newey–West) for IC/RankIC mean (already supported by `generate_paper_report`)

### 2.2 Portfolio Metrics (supporting)
- Annualized return (with cost)
- Information ratio
- Max drawdown
- Turnover

### 2.3 Router/Regime Diagnostics (mechanism evidence)
- `time_ratio` (time expert weight) daily series
- `gate_entropy` daily series (collapse detection)
- `time_tau` / `time_half_life` daily series (if regime time embedding enabled)
- `factor_gate_entropy`, `factor_gate_topk_mass_10` (if factor FiLM enabled)

Acceptance criteria (to prevent “MoE is decoration”):
- `time_ratio` must not be stuck near 0 or 1 across the entire test set.
- `gate_entropy` must be non-trivial (avoid collapse); include collapse rate checks.

---

## 3. Baselines (strong but feasible)

We need baselines in three buckets:

### 3.1 Classical / Lightweight
- Linear / Ridge (or equivalent)
- LightGBM / XGBoost (if available in repo; otherwise replace with the repo’s strongest tree baseline)

### 3.2 Deep Sequential / Cross-sectional
- GRU / LSTM / ALSTM (available in `examples/benchmarks`)
- Transformer (vanilla)
- TRA (KDD’22)
- DoubleAdapt (KDD’23) if available; otherwise document gap and provide closest adaptation baseline in repo

### 3.3 “Closest prior” (non-stationarity aware)
- If MASTER-style baseline is not directly runnable here, do one of:
  1) implement minimal MASTER-like “market-guided gating” baseline, or
  2) cite official numbers with exact protocol match (same data, split, label, transaction cost), and be explicit about any mismatch.

Baseline discipline (Kaiming He style):
- Use **identical data protocol** and splits.
- Tune each baseline with the **same** budget and rules (see Section 6).

---

## 4. Core Ablations (to prove structural routing)

These ablations must keep parameter scale and training protocol comparable.

### 4.1 Structural Routing Controls (hard negatives)
1) **Time-only**: force gate = `[1, 0]` (disable factor expert)
2) **Factor-only**: force gate = `[0, 1]`
3) **Fixed-50/50**: gate = `[0.5, 0.5]` (no routing)
4) **Router w/o regime**: router input excludes macro/internal stats (only constant / layer summary)
5) **Full RST-MoE**: learned router + regime signal (target model)

Expected outcome:
- If Full ≈ Fixed-50/50, the “structural routing” novelty is not supported.

### 4.2 Regime Conditioning Ablations (to prove mechanism)
- `use_regime_time_embedding`: on/off
- `use_regime_factor_gate`: on/off
- `use_feature_selection`: on/off (STG)
- `use_external_macro`: on (t+5 default) vs internal-only fallback (optional, but useful to show macro necessity)

Minimum ablation table (recommended):
- Base (no time-emb, no FiLM, no STG) + routing
- + time-emb
- + FiLM
- + time-emb + FiLM (default)
- + STG (optional; expect smaller/unstable gains)

---

## 5. Regime-Shift Analysis (the “KDD narrative figure”)

Goal: show the router changes behavior *when the market regime changes*, and that this correlates with improved performance.

### 5.1 Define regime buckets (simple, auditable)
From the precomputed `market_state_*`:
- Bucket days by quantiles of 2–3 interpretable signals (choose stable ones):
  - `corr_pc1_ratio` (market-mode strength)
  - `corr_mean_abs` / crowding proxy
  - volatility proxy (from market TS features if included)

### 5.2 Report per-bucket results
For each bucket on the test set:
- RankIC mean/IR, IC mean/ICIR
- `time_ratio` mean/std, `gate_entropy` mean
- (optional) turnover / drawdown profile

### 5.3 Required plots (1 figure is enough)
- Plot: bucket index → performance (RankIC) and bucket index → `time_ratio`
- Add a short interpretation paragraph (do not overclaim causality; frame as evidence of mechanism alignment).

---

## 6. Hyperparameter Budget & Fair Tuning Rules

To avoid reviewer objections (“you tuned your method more”):
- Use the same tuning budget per model family:
  - 1–2 core configs + small grid over learning rate and model width
  - Early stopping based on valid RankIC (already in adapter)
- Fix everything else:
  - epochs, warmup schedule, cost model, backtest settings
- Record all run configs into the recorder and include in the report.

Recommended defaults for t+5 (starting point; adjust once, then freeze):
- `d_model=128`, `n_layers=3~4`, `n_heads=4`, `dropout=0.1`
- `router_z_loss_coef=0.01`, `router_temperature=0.7~1.0`, `router_noise=0.1`
- `use_regime_time_embedding=True`, `time_tau_init≈5~6`, `time_decay_normalize=True`
- `use_regime_factor_gate=True`, `factor_gate_scale=0.5`
- `use_feature_selection=False` (only enable for a dedicated ablation sweep)
- `batch_size=64~128` (CSI800 often larger; keep within memory)

---

## 7. Seeds, Aggregation, and Significance

### 7.1 Seeds
- Run **5 seeds** for every line in:
  - Baseline table
  - Key ablations (Section 4.1 and minimal 4.2)

### 7.2 Aggregation outputs
For each (universe, model/ablation):
- mean ± std over seeds for:
  - RankIC mean/IR/HAC-t
  - AnnRet/IR/MDD/Turnover
- Additionally:
  - stability score: std(RankIC mean) across seeds

### 7.3 Statistical tests (keep simple)
- Primary: HAC t-stat for daily mean IC/RankIC (already available)
- Secondary: paired test across days between Full and a strong baseline (optional but good)

---

## 8. Deliverables (what the paper will contain)

Minimum set:
1) **Table 1**: CSI300/CSI800 t+5 performance vs baselines (mean±std over 5 seeds)
2) **Table 2**: Structural routing ablations (time-only, factor-only, fixed-50/50, no-regime router, full)
3) **Figure 1**: Regime buckets vs (RankIC, time_ratio) on test set
4) **Appendix**: Gate entropy/time_ratio series + a few attention maps (qualitative)

---

## 9. Execution Order (efficient scheduling)

1) Precompute market state for CSI300/CSI800 (strict, train-fit PCA)
2) Run Full RST-MoE (CSI300, 1 seed) → sanity + report generation
3) Run structural routing controls (CSI300, 1 seed) → confirm big deltas exist
4) Scale to 5 seeds (CSI300) for Table 1/2
5) Repeat steps 2–4 for CSI800
6) Run baselines (start from cheapest to most expensive; stop early only with a written reason)
7) Generate aggregated tables + regime bucket figure

---

## 10. Threats to Validity (write this proactively)

- Train uses rank-label (`CSRankNorm`) while test uses raw label; explain why it’s acceptable for ranking, and apply the same protocol to baselines.
- Macro features: ensure PCA fit on train only; document `market_state_shift` and `market_ts_past_only` choices.
- Internal regime stats are batch-size sensitive; for paper claims prefer external macro for t+5.
- Router collapse: report `gate_entropy` and `time_ratio` distributions; treat collapse runs as failures, not hidden.

---

## 11. Sweep Bookkeeping (practical)

- Run manifest: `run_matrix.yaml`
  - Fill `run_ids` (MLflow run_id) after each run.
- Aggregation: `scripts/aggregate_kdd_runs.py`
  - Example: `python scripts/aggregate_kdd_runs.py --matrix run_matrix.yaml --out_dir analysis/kdd --include_diag`
