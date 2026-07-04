# KDD Experiments Plan (RST-MoE, t+5, CSI300/CSI800, 5 seeds)

This document defines a submission-grade experimental plan for RST-MoE with a focus on:
1) convincing evidence for the core claim (regime-driven structural routing),
2) fair comparisons (strong baselines + controlled ablations),
3) reproducibility (one-click reports + strict leakage controls).

## 0. Executive Summary (read first)

### 0.1 Core claims (limit to two)
- Claim A (Novelty): Regime-driven structural routing between a time expert and a cross-sectional factor expert improves performance and robustness under market regime shifts, compared to static architectures or soft adaptation only.
- Claim B (Mechanism): The gains are attributable to regime conditioning (router + regime-adaptive embeddings), not parameter count, training tricks, or reporting artifacts.

### 0.2 Target setting (fixed)
- Universes: CSI300, CSI800
- Horizon: t+5 (label consistent with `Ref($close, -5) / Ref($close, -1) - 1`)
- Seeds: 5 (report mean ± std; include significance tests)

### 0.3 Success criteria (evidence bar)
- Table 1 (LaTeX): CSI300/CSI800 t+5 performance vs baselines (from papers), with protocol match notes and citations.
- Table 2 (LaTeX): structural routing ablations (time-only, factor-only, fixed-50/50, no-regime router, full) with mean ± std over 5 seeds.
- Figure 1: regime buckets vs RankIC and time_ratio on test set (one combined figure is enough).
- Router diagnostics: time_ratio not collapsed to 0 or 1; gate_entropy non-trivial; report collapse rate.

### 0.4 Deliverables (LaTeX-first)
- LaTeX tables for Table 1 and Table 2.
- Regime bucket figure and short interpretation paragraph.
- Appendix with router diagnostics and qualitative plots.

---

## 1. Protocol Contract (immutable)

### 1.1 Data & Splits
- Dataset: Qlib `Alpha158` (as in `work_flow.py` / `DESIGN.md`)
- Splits (official):
  - Train: 2008-01-01 ~ 2014-12-31
  - Valid: 2015-01-01 ~ 2016-12-31
  - Test: 2017-01-01 ~ 2020-08-01
- Task: daily cross-sectional ranking (CSI300/CSI800 universe)

### 1.2 Labels & Processing
- Label expression (t+5): `Ref($close, -5) / Ref($close, -1) - 1`
- Training label processing:
  - Train uses `DK_L` with `CSRankNorm` (rank-label).
  - Valid/Test use `DK_I` (raw label).
- Paper text must explicitly state this.

### 1.3 Macro Features (External Regime Signal, default)
- Precompute market state for each universe:
  - `market_state_csi300.pkl`
  - `market_state_csi800.pkl`
- Leakage control:
  - PCA must be fit on train only (`--pca_fit_on=train`).
  - Rolling/zscore/delta features must be past-only by construction.
- Training config:
  - `trainer_config.market_state_path=<...>`
  - `trainer_config.market_state_shift=0`
  - `trainer_config.market_state_strict=True`

### 1.4 Sampler & Batch Semantics
- Training must use same-day cross-section batches (`FixedDailyBatchSampler`).
- For main results, enforce `batch_size >= 128`.
- Internal regime estimation is Appendix-only and not used for main claims.

---

## 2. Metrics & Reporting (exact)

### 2.1 Primary Metrics (submission tables)
- RankIC mean and IR (daily series aggregated)
- IC mean and ICIR
- HAC t-stat (Newey-West) for IC/RankIC mean

### 2.2 Portfolio Metrics (supporting)
- Annualized return (with cost)
- Information ratio
- Max drawdown
- Turnover

### 2.3 Router/Regime Diagnostics (mechanism evidence)
- `time_ratio` daily series
- `gate_entropy` daily series
- `time_tau` / `time_half_life` daily series (if regime time embedding enabled)
- `factor_gate_entropy`, `factor_gate_topk_mass_10` (if factor FiLM enabled)

Acceptance criteria (avoid "MoE is decoration"):
- `time_ratio` must not be stuck near 0 or 1 across the entire test set.
- `gate_entropy` must be non-trivial; report collapse rate.

---

## 3. Evidence Block I: Baselines (paper results only)

**Policy:** baselines are taken directly from published papers and not re-run in this repo.
- Include a protocol-match column (universe, horizon, label, split, cost).
- If any mismatch exists, flag it in a footnote and mark the row as reference-only.

### 3.1 Classical / Lightweight
- Linear / Ridge
- LightGBM / XGBoost (use reported results from the closest protocol)

### 3.2 Deep Sequential / Cross-sectional
- GRU / LSTM / ALSTM
- Transformer (vanilla)
- TRA (KDD'22)
- DoubleAdapt (KDD'23)
- StockMixer / TSMixer

### 3.3 Non-stationarity Aware (closest prior)
- MASTER (AAAI'24): market-guided attention, single backbone
- Any additional regime-aware baseline with matching protocol

### 3.4 Positioning vs SOTA (narrative alignment)
- MASTER: market-guided attention but no axis-specific experts
- DoubleAdapt: reactive online weight adaptation with extra compute
- TRA: temporal routing without explicit regime state
- StockMixer/TSMixer: fixed mixing, no regime conditioning

---

## 4. Evidence Block II: Structural Routing Ablations

These ablations keep parameter scale and training protocol comparable.

### 4.1 Structural Routing Controls (hard negatives)
1) Time-only: force gate = `[1, 0]` (disable factor expert)
2) Factor-only: force gate = `[0, 1]`
3) Fixed-50/50: gate = `[0.5, 0.5]` (no routing)
4) Router w/o regime: router input excludes macro/internal stats
5) Full RST-MoE: learned router + regime signal

Expected outcome:
- If Full ≈ Fixed-50/50, the structural-routing novelty is not supported.

---

## 5. Evidence Block III: Mechanism & Regime

### 5.1 Regime Conditioning Ablations
- `use_regime_time_embedding`: on/off
- `use_regime_factor_gate`: on/off
- `use_feature_selection`: on/off (STG)
- `use_external_macro`: on (t+5 default) vs internal-only fallback (Appendix only)

Minimum ablation table:
- Base (no time-emb, no FiLM, no STG) + routing
- + time-emb
- + FiLM
- + time-emb + FiLM (default)
- + STG (optional; expect smaller/unstable gains)

### 5.2 Regime-Shift Analysis (KDD narrative figure)
- Define regime buckets using quantiles of 2-3 stable signals:
  - `corr_pc1_ratio` (market-mode strength)
  - `corr_mean_abs` / crowding proxy
  - volatility proxy (from market TS features if included)
- Report per-bucket results on test set:
  - RankIC mean/IR, IC mean/ICIR
  - `time_ratio` mean/std, `gate_entropy` mean
  - Optional: turnover / drawdown profile
- Plot: bucket index -> RankIC and bucket index -> `time_ratio`
- Add a short interpretation paragraph (no causal overclaim).

---

## 6. Hyperparameter Budget & Fair Tuning Rules (ours only)

To avoid reviewer objections ("you tuned your method more"):
- Use the same tuning budget for RST-MoE variants:
  - 1-2 core configs + small grid over learning rate and model width
  - Early stopping based on valid RankIC
- Fix everything else:
  - epochs, warmup schedule, cost model, backtest settings
- Record all run configs into the recorder and include in the report.

Recommended defaults for t+5 (start once, then freeze):
- `d_model=128`, `n_layers=3~4`, `n_heads=4`, `dropout=0.1`
- `router_z_loss_coef=0.01`, `router_temperature=0.7~1.0`, `router_noise=0.1`
- `use_regime_time_embedding=True`, `time_tau_init≈5~6`, `time_decay_normalize=True`
- `use_regime_factor_gate=True`, `factor_gate_scale=0.5`
- `use_feature_selection=False` (only enable for a dedicated ablation sweep)
- `batch_size=64~128` (CSI800 often larger; keep within memory)

---

## 7. Execution & Repro (runbook)

1) Precompute market state for CSI300/CSI800 (strict, train-fit PCA)
2) Run Full RST-MoE (CSI300, 1 seed) -> sanity + report generation
3) Run structural routing controls (CSI300, 1 seed) -> confirm deltas
4) Scale to 5 seeds (CSI300) for Table 2
5) Repeat steps 2-4 for CSI800
6) Aggregate tables + regime bucket figure
7) Fill `run_matrix.yaml` with `run_ids` and archive configs

Sweep bookkeeping:
- Run manifest: `run_matrix.yaml`
- Aggregation: `scripts/aggregate_kdd_runs.py`
  - Example: `python scripts/aggregate_kdd_runs.py --matrix run_matrix.yaml --out_dir analysis/kdd --include_diag`

---

## 8. Threats to Validity (write proactively)

- Train uses rank-label (`CSRankNorm`) while test uses raw label; explain why it is acceptable for ranking, and apply the same protocol to our runs.
- Macro features: ensure PCA fit on train only; document `market_state_shift` and past-only construction.
- Baselines from papers may not match our protocol; flag mismatches and treat as reference-only.
- Router collapse: report `gate_entropy` and `time_ratio` distributions; treat collapse runs as failures, not hidden.

---

## Appendix (non-core)

- Internal regime estimation:
  - Requires large batch sizes; avoid for main claims.
  - If shown, label as exploratory and place in Appendix.
- Additional diagnostics:
  - `gate_entropy` / `time_ratio` series
  - attention maps / routing heatmaps
- Extra ablations or hyperparameter sensitivity if needed.
