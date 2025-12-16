# QuantMoEConfig (`model_config`) 说明

本项目中 `model_conf["kwargs"]["model_config"]` 会被 `module/model_adapter.py:QlibQuantMoE` 透传给 `module/utils/model_configuration.py:QuantMoEConfig`，并用于初始化 `module/quant_moe_model.py:QuantMoEModel`。

## 重要约定（会被自动覆盖）

- `context_len`、`num_alphas` 会在 `QlibQuantMoE._init_net()` 中根据第一批数据 `bx.shape=[B,T,F]` 自动探测并覆盖（`T->context_len`，`F->num_alphas`）。一般不需要手动设置。
- 如果在 `trainer_config` 中启用 `market_state_path`，适配器会强制设置：
  - `use_external_macro=True`
  - `d_macro_input = market_state 向量维度`
  用于将预计算的 `macro_features` 输入到 `RegimeContextEncoder`。

## 参数分组说明

### 1) Backbone / 表示维度

- `d_model`（int，默认 64）：主隐层维度 D。
- `n_heads`（int，默认 4）：注意力头数 H，要求 `d_model % n_heads == 0`。
- `n_layers`（int，默认 4）：MoE block 层数（`RegimeAdaptiveMoEBlock` 堆叠次数）。
- `d_ff`（int，默认 128）：FFN 中间层宽度。
- `dropout`（float，默认 0.1）：Dropout 概率（注意力/FFN/embedding）。
- `num_alphas`（int，默认 64）：因子数 N（由数据自动探测覆盖）。
- `context_len`（int，默认 32）：时间窗口长度 T（由数据自动探测覆盖）。

### 2) Router（MoE 门控：time expert vs factor expert）

Router 位于每层 `RegimeAdaptiveMoEBlock`，输入为 `regime_embedding`，输出 2 维 gate（time/factor）。

- `router_noise`（float，默认 0.1）：训练时对 `router_logits` 添加的高斯噪声标准差（探索机制）。`0` 表示关闭。
- `router_temperature`（float，默认 1.0）：softmax 温度，`<1` 更尖锐、`>1` 更平滑（探索/锐化机制）。
- `router_z_loss_coef`（float，默认 1e-3）：router 的 z-loss 系数（稳定 logsumexp 的正则项）。
- `router_use_layer_summary`（bool，默认 False）：是否把“每层的市场摘要 token”拼到 router 输入里，使 gate 随深度自适应（仍保持 market-level 路由，不做个股个性化路由）。

实现位置：`module/architecture/moe_block.py`。

### 3) 注意力位置偏置（ALiBi）

- `use_alibi`（bool，默认 True）：是否在 time/factor 两个维度都使用双向 ALiBi bias（`|i-j|`）。

### 4) 可微特征选择（STG-style gate）

启用后，模型会学习一个 `z∈[0,1]^N` 对因子维进行软选择（与 `loss_weights["reg"]` 配合做稀疏化）。

- `use_feature_selection`（bool，默认 True）：是否启用 `DifferentiableFeatureSelector`。
- `selection_reg_lambda`（float，默认 1e-3）：特征选择正则系数（乘到 `reg_loss` 上）。
- `selection_temperature`（float，默认 0.1）：sigmoid 温度（越小越接近硬选择）。
- `selection_noise_std`（float，默认 0.5）：训练时 gate 的噪声标准差（探索）。

### 5) Loss / 训练目标（截面排序）

主目标是 list-wise 排序（ListMLE），并可选叠加 pairwise/top-bottom 与回归型 Huber。

- `loss_weights`（dict）：各 loss 项权重，默认：
  - `listmle`：ListMLE 主损失权重
  - `rank`：RankNet top/bottom 辅助损失权重（可选）
  - `huber`：截面 Huber 辅助损失权重（可选）
  - `aux`：MoE router z-loss 权重（乘上 `router_z_loss_coef` 之后再乘它）
  - `reg`：特征选择稀疏正则权重（乘上 `selection_reg_lambda` 之后再乘它）
  - 注：IC 目前作为监控指标，不进入 `total_loss`（即使 dict 中存在 `ic` key 也不会生效）。
- `listmle_tau`（float，默认 1.0）：ListMLE 温度（越小越“硬排序”，但更不稳定）。
- `rank_topk`（int，默认 5）：RankNet top/bottom K 的 K。
- `huber_delta`（float，默认 1.0）：Huber 的 delta。

### 6) Regime Encoder（市场状态表征）

模型的 router 输入来自 `RegimeContextEncoder`：

- 外部宏观（推荐用于 t+5 或更长）：`use_external_macro=True` 时直接用 `macro_features[B,d_macro]`。
- 内部统计（无 macro 时的 fallback）：从 batch 内/样本内的 `x[B,T,N]` 推断市场状态。

参数：
- `use_external_macro`（bool，默认 False）：是否使用外部 macro features。
- `d_macro_input`（int，默认 0）：macro 特征维度（启用 macro 时必填；适配器可自动推断）。
- `regime_internal_mode`（str，默认 `"short"`）：内部统计的尺度模式：
  - `"short"`：短尺度（更贴近 t+1）
  - `"long"`：长尺度（更贴近 t+5，需要配合 `regime_internal_lag`）
- `regime_internal_lag`（int，默认 1）：`mode="long"` 时使用的跨期 lag（会被 clamp 到 `T-1`）。
- `regime_internal_use_batch_stats`（bool，默认 True）：是否把一个 batch 视为“同日截面”并在 batch 内估计 market state（然后广播）；若你的 batch 不是日内截面，建议设为 False。
- `regime_internal_tail_threshold`（float，默认 2.0）：内部统计中“极端值”阈值（针对已做过 z-score/RobustZScoreNorm 的输入）。

实现位置：`module/architecture/regime_encoder.py`。

### 7) 因子聚合（pooling）

最后一时间步的因子表示 `h_last[B,N,D]` 需要聚合成股票表示 `h_pooled[B,D]`：

- `pooling_alpha`（float，默认 0.7）：attention pooling 与 mean pooling 的混合权重：
  - `pooled = alpha * attn_pool + (1-alpha) * mean_pool`

实现位置：`module/architecture/attention_pooling.py`。

## 已移除的无用参数

- `num_dates`：原本用于 date embedding/router，但当前实现不再使用 `date_ids`，因此该参数已从 `QuantMoEConfig` 移除。

## 建议配置（常用模板）

- t+1（更偏短期）：`regime_internal_mode="short"`，`router_temperature≈1.0`，`router_noise` 可小（0~0.1）。
- t+5（更偏跨周期）：优先启用 `market_state_path`（外部 macro），或在无 macro 时用 `regime_internal_mode="long", regime_internal_lag=5`，并适当降低 `router_temperature`（例如 0.7~1.0）让 gate 更可分。
