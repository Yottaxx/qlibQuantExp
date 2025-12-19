# Recommended Configs (t+1 vs t+5)

下面给出两套可落地配置（模型 + 预计算 macro state）：
- **t+1**：短期，内部 regime 为主（可选轻量 macro），router 较平滑。
- **t+5**：中期，强烈建议使用预计算 `macro_features`，router 更锋利，macro state 增强稳定性。

> 训练/eval 口径：Train 用 `DK_L` + `CSRankNorm`（rank-label），Valid/Test 用 `DK_I` raw label。早期 NaN 请确保预计算覆盖足够 warmup，或设 `market_state_strict=False`。

---

## t+1 配置（短期，内部 regime）

### Model (`model_conf["kwargs"]["model_config"]`)
- `d_model`: 64~128（依资源）
- `n_layers`: 2~4
- `n_heads`: 4
- `use_feature_selection`: False（建议先做 FiLM-only baseline；再做 STG ablation）
- `selection_reg_lambda`: 1e-5~1e-4（若开 STG；`reg_loss=z.sum(dim=-1).mean()` 值域 `[0,N]`）
- `selection_temperature`: 0.1（若开 STG）
- `selection_noise_std`: 0.5（若开 STG）
- `use_regime_time_embedding`: True  （regime-adaptive time embedding）
- `time_tau_init`: 5.0
- `use_regime_factor_gate`: True  （regime-adaptive factor gate）
- `factor_gate_scale`: 0.5
- `use_alibi`: False（推荐先关；短 T + regime time embedding 下通常足够，且可避免 MHA 的 `attn_mask` 展开开销；需要时再做 ablation 开启）
- `use_external_macro`: False  （t+1 可不依赖外部 macro）
- `regime_internal_mode`: "short"
- `regime_internal_lag`: 1
- `regime_internal_use_batch_stats`: True  （日度截面 batch）
- `regime_internal_tail_threshold`: 2.0
- `router_use_layer_summary`: True  （轻量层内摘要，提高 gate 自适应）
- `router_noise`: 0.05  （小噪声探索）
- `router_temperature`: 1.0
- `router_z_loss_coef`: 1e-3
- `pooling_alpha`: 0.7
- `listmle_tau`: 1.0
- `rank_topk`: 5
- `huber_delta`: 1.0
- `loss_weights`: 默认

### Trainer (`trainer_config`)
- `lr`: 5e-4
- `batch_size`: 64~256（取决于截面规模）
- `n_epochs`: 20~40
- `early_stop`: 5
- `seed`: 42
- `use_warmup`: True, `warmup_ratio`: 0.05
- `num_workers`: 按机器设置
- `market_state_path`: 留空（内部 regime）
- `market_state_strict`: True（无 macro）

### Macro State 预计算
- 可选：不需要；若要尝试，简化版即可：`--pca_dim 8 --zscore_windows 20`

---

## t+5 配置（中期，外部 macro 驱动）

### Model (`model_conf["kwargs"]["model_config"]`)
- `d_model`: 128
- `n_layers`: 3~4
- `n_heads`: 4
- `use_feature_selection`: False（建议先做 FiLM-only baseline；再做 STG ablation）
- `selection_reg_lambda`: 1e-5~1e-4（若开 STG；`reg_loss=z.sum(dim=-1).mean()` 值域 `[0,N]`）
- `selection_temperature`: 0.1（若开 STG）
- `selection_noise_std`: 0.5（若开 STG）
- `use_regime_time_embedding`: True
- `time_tau_init`: 5.0
- `use_regime_factor_gate`: True
- `factor_gate_scale`: 0.5
- `use_alibi`: False（同上，建议先关；需要时再做 ablation）
- `use_external_macro`: True  （由 adapter 自动设置）
- `regime_internal_mode`: "long"  （仅作 fallback）
- `regime_internal_lag`: 5
- `regime_internal_use_batch_stats`: True
- `regime_internal_tail_threshold`: 2.0
- `router_use_layer_summary`: True
- `router_noise`: 0.1  （更大探索）
- `router_temperature`: 0.7~1.0  （更锋利 gate）
- `router_z_loss_coef`: 1e-3
- `pooling_alpha`: 0.7
- `listmle_tau`: 0.8~1.0  （可稍小以增强排序尖锐度）
- `rank_topk`: 5
- `huber_delta`: 1.0
- `router_z_loss_coef`: 0.01  (increase if router collapses)

### Trainer (`trainer_config`)
- `lr`: 5e-4
- `batch_size`: 64~128（截面更大时取小一些）
- `n_epochs`: 30~50
- `early_stop`: 5
- `seed`: 42
- `use_warmup`: True, `warmup_ratio`: 0.05
- `num_workers`: 按机器设置
- `market_state_path`: `market_state_csi300.pkl` 或 `market_state_csi800.pkl`
- `market_state_shift`: 0 或 1（1 可避免同日信息泄露）
- `market_state_strict`: True（推荐论文用），若早期 NaN 太多，可临时 False

### Macro State 预计算命令（示例：CSI300）
```bash
python scripts/precompute_market_state.py \
  --out market_state_csi300.pkl \
  --instruments csi300 \
  --pca_dim 16 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --market_ts_past_only \
  --zscore_windows 20,60,120 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --warmup_trading_days -1 \
  --no_norm 
```

### Macro State 预计算命令（示例：CSI800）
```bash
python scripts/precompute_market_state.py \
  --out market_state_csi800.pkl \
  --instruments csi800 \
  --pca_dim 16 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --market_ts_past_only \
  --zscore_windows 20,60,120 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --trade_field '$amount' \
  --min_trade 1 \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --warmup_trading_days -1 \
  --no_norm 

```

> Note: `--warmup_trading_days -1` automatically extends the precompute start backward, ensuring rolling/zscore/Δstate/TS features are defined at training start. If early data fields are missing, use an explicit number or set `market_state_strict=False`.
