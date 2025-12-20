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
- `initializer_range`（float，默认 0.02）：HF-style 初始化标准差（`Linear/Embedding/MultiheadAttention(in_proj)`），用于统一输入/embedding 尺度并提升训练稳定性。
- `num_alphas`（int，默认 64）：因子数 N（由数据自动探测覆盖）。
- `context_len`（int，默认 32）：时间窗口长度 T（由数据自动探测覆盖）。

### 2) Router（MoE 门控：time expert vs factor expert）

Router 位于每层 `RegimeAdaptiveMoEBlock`，输入为 `regime_embedding`，输出 2 维 gate（time/factor）。

- `router_noise`（float，默认 0.1）：训练时对 `router_logits` 添加的高斯噪声标准差（探索机制）。`0` 表示关闭。
- `router_temperature`（float，默认 1.0）：softmax 温度，`<1` 更尖锐、`>1` 更平滑（探索/锐化机制）。
- `router_z_loss_coef`（float，默认 0.01）：router 的 z-loss 系数（防止 collapse）。
- `router_use_layer_summary`（bool，默认 True）：是否把“每层的市场摘要 token”拼到 router 输入里。

#### 🚨 Router 参数详解

| 参数 | 作用 | 调参逻辑 |
|------|------|----------|
| `router_z_loss_coef` | 惩罚 logits 绝对值过大，防止 collapse | 太低(<1e-3)→collapse; 太高(>0.1)→强制均匀 |
| `router_temperature` | 控制 softmax 尖锐度 | <1→硬选择; >1→软选择 |
| `router_noise` | 训练时增加探索 | 0.1~0.3 防止早期 collapse |

**为什么需要 z-loss？**
```python
z_loss = (logsumexp(router_logits, dim=-1) ** 2).mean()
```
当 logits 差异过大（如 `[10, -10]`），softmax 几乎只选一个专家，z-loss 会变大来阻止这种情况。

实现位置：`module/architecture/moe_block.py`。

### 3) Regime-Adaptive Embeddings（Time / Factor）

这组参数用于实现 “regime 不仅切 expert ratio，也切换时间尺度/因子组合” 的闭环：

- `use_regime_time_embedding`（bool，默认 True）：是否启用 regime-adaptive time embedding（learned lag table + `time_tau(regime)` 衰减）。
- `time_tau_min`（float，默认 0.5）：`tau` 下界（越小越偏向短记忆）。
- `time_tau_max`（float，默认 50.0）：`tau` 上界（防止数值发散）。
- `time_tau_init`（float，默认 5.0）：初始化时的 `tau`（通过 `tau_base` 对齐该值，并让 MLP 输出为小幅 `delta`，训练初期更稳）。
- `time_emb_init_std`（float，默认 0.02）：time embedding 表的初始化标准差（设为 `0` 可从 0 开始学）。
- `time_decay_normalize`（bool，默认 True）：是否把衰减权重归一化到 `mean(w)=1`（避免不同 regime 下 embedding 尺度漂移）。
  - 备注：当前实现用小型 MLP 预测 `tau_delta(r)`，并采用 `tau_base + scale * tau_delta(r)` 的初始化（`tau_base` 对齐 `time_tau_init`），保证初期稳定且可学习。

- `use_regime_factor_gate`（bool，默认 True）：是否启用 regime-adaptive factor FiLM（per-sample, per-factor 的自适应 `gamma/beta`）。
- `factor_gate_scale`（float，默认 0.5）：`gamma` 幅度，元素范围约为 `[1-scale, 1+scale]`（在每个 MoE block 的 Pre-LN 之后生效，避免被 LayerNorm 抵消）。
- `factor_gate_shift_scale`（float，默认 0.0）：`beta` 幅度，元素范围约为 `[-shift_scale, +shift_scale]`（默认 0 表示只做 scale，不做 shift）。

实现位置：
- `module/architecture/regime_adaptive_embedding.py`

### 4) 注意力位置偏置（ALiBi）

- `use_alibi`（bool，默认 False）：是否在 **time 维度** 使用双向 ALiBi bias（`|i-j|`）。
  - 因子维（N axis）默认不使用 ALiBi：因子 index 通常无自然顺序，易引入伪先验且会显著放大 mask 内存。
  - 实现提示：当前 `ParallelAttention` 使用 PyTorch `nn.MultiheadAttention`，启用 ALiBi 会构造 `[B*H, L, L]` 的 `attn_mask`；对 time expert 来说有效 batch 是 `B*N`，在大截面时会显著增加显存/时间开销。若 `T` 很短（8~32）且已启用 `use_regime_time_embedding`，通常可以先把 `use_alibi=False` 作为默认 ablation。

### 5) 可微特征选择（STG-style gate）

启用后，模型会学习一个 `z∈[0,1]^N` 对因子维进行软选择。

> [!NOTE]
> 本项目的 backbone 是 Pre-LN Transformer（每个 block 开头有 `LayerNorm`），因此 gate/mask 若放在 block 之前会被 LN 近似抵消。
> 当前实现将 `factor_gate`（FiLM）与 `feature_selection` mask 都放在每个 block 的 `LayerNorm` 之后再应用，保证有效。

- `use_feature_selection`（bool，默认 True）：是否启用 `DifferentiableFeatureSelector`。
- `selection_reg_lambda`（float，默认 1e-5）：特征选择正则系数。
- `selection_temperature`（float，默认 0.1）：sigmoid 温度（越小越接近硬选择）。
- `selection_noise_std`（float，默认 0.5）：训练时 gate 的噪声标准差。

#### 🚨 特征选择参数详解

| 参数 | 作用 | 调参逻辑 |
|------|------|----------|
| `selection_reg_lambda` | 控制稀疏惩罚强度 | 1e-5~1e-4 轻度稀疏; 1e-3+ 强稀疏 |
| `selection_temperature` | sigmoid 锐度 | 0.1 硬选择; 0.5+ 软选择 |
| `selection_noise_std` | 训练探索 | 0.3~0.5 防止早期卷缩 |

**reg_loss 计算方式**（已修复）：
```python
reg_loss = z.sum(dim=-1).mean()  # 平均每个样本选择了多少个特征
```
值域为 `[0, N]`，对特征数量 N 不敏感。

#### FiLM + STG 是否冗余（推荐消融）

`feature_selection`（STG）和 `factor_gate`（FiLM/AdaLN）控制的粒度不同，但很多任务上效果会重叠。

- 推荐先以 **FiLM=on, STG=off** 作为默认基线（更简洁、更稳），再做四组消融对比：
  - `use_regime_factor_gate={False,True}`
  - `use_feature_selection={False,True}`
- 若目标是“全局因子裁剪/稀疏解释”，再考虑打开 STG，并把 `selection_reg_lambda` 从 `1e-5` 起步，避免早期塌缩。

### 6) Loss / Training Objective (Cross-Sectional)

Main objective is configurable via `main_loss` (default **MSE** on rank-label), with optional pairwise/top-bottom and Huber regression.
训练时 loss 统一基于 **DK_L 的 rank-label** 计算（与现有训练口径保持一致）。

- `main_loss` (str, default `"mse"`): 主 loss 选择：
  - `"mse"`: cross-sectional MSE on rank-label（默认）
  - `"ic"`: IC loss（`loss = -IC`，最小化即最大化 IC）
  - `"listmle"`: list-wise ranking (ListMLE)
- `loss_weights` (dict): Weights for loss components. Default:
  - `listmle`: 1.0 - ListMLE main loss weight
  - `mse`: 1.0 - MSE main loss weight
  - `ic`: 1.0 - IC main loss weight
  - `rank`: 0.0 - RankNet top/bottom auxiliary loss (optional)
  - `huber`: 0.0 - Cross-sectional Huber auxiliary loss (optional)

> IC always computed for monitoring; it enters `total_loss` **only when** `main_loss="ic"`.

> [!IMPORTANT]
> **`aux` and `reg` keys have been removed.** Router z-loss and feature selection regularization are now controlled **directly** by their respective coefficients:
> - `router_z_loss_coef` (default: 0.01) - Added directly to total loss
> - `selection_reg_lambda` (default: 1e-5) - Added directly to total loss

- `listmle_tau` (float, default 1.0): ListMLE temperature (used when `main_loss="listmle"`; lower = harder ranking, but less stable).
- `rank_topk` (int, default 5): K for RankNet top/bottom K.
- `huber_delta` (float, default 1.0): Huber delta.

### 7) Regime Encoder（市场状态表征）

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

### 8) 因子聚合（pooling）

最后一时间步的因子表示 `h_last[B,N,D]` 需要聚合成股票表示 `h_pooled[B,D]`：

- `pooling_alpha`（float，默认 0.7）：attention pooling 与 mean pooling 的混合权重：
  - `pooled = alpha * attn_pool + (1-alpha) * mean_pool`

实现位置：`module/architecture/attention_pooling.py`。

## 已移除的无用参数

- `num_dates`：原本用于 date embedding/router，但当前实现不再使用 `date_ids`，因此该参数已从 `QuantMoEConfig` 移除。

## 建议配置（常用模板）

### 防 Collapse + 适度稀疏（推荐）
```python
"model_config": {
    "router_z_loss_coef": 0.01,       # 比旧默认高 10x，防 collapse
    "router_temperature": 1.0,
    "router_noise": 0.1,
    "selection_reg_lambda": 1e-5,     # 修复后需要降低
    "selection_temperature": 0.1,
},
```

### Loss Coefficient Summary

With the current design, loss coefficients work directly:
- **z-loss effective weight** = `router_z_loss_coef` (default: 0.01)
- **reg effective weight** = `selection_reg_lambda` (default: 1e-5)

Main loss uses `loss_weights[main_loss]` (default 1.0). No extra multiplication with `loss_weights` keys is required for z-loss/reg.

### 不同预测周期的建议
- **t+1**：`regime_internal_mode="short"`，`router_temperature≈1.0`，`router_noise` 可小（0~0.1）。
- **t+5**：优先启用 `market_state_path`，或用 `regime_internal_mode="long", regime_internal_lag=5`。
