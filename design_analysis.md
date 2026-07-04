# RST-MoE 系统架构设计文档与代码审计报告

> **Date**: 2024-12-18  
> **Reviewer**: Jeff Dean-style Technical Deep Dive  
> **Project**: RST-MoE (Regime-Separated Temporal Mixture-of-Experts) for Stock Prediction

---

## Executive Summary

RST-MoE 是一个基于 **Mixture-of-Experts (MoE)** 架构的股票预测系统，核心创新在于将市场「非稳态性」(Non-Stationarity) 从**噪声问题**重新定义为**结构性问题**，通过 **Regime-Adaptive Router** 在不同市场状态下动态切换推理逻辑（时序专家 vs 截面专家）。

**整体评价**: 架构设计合理，代码质量较高，但存在若干中等风险的逻辑问题需要关注。

---

## 1. 系统架构总览

```mermaid
flowchart TB
    subgraph Data Pipeline
        A[Qlib Data Provider] --> B[Alpha158 Handler]
        B --> C[RobustZScoreNorm + CSRankNorm]
        C --> D[TSDatasetH]
    end
    
    subgraph Macro State Pipeline
        E[precompute_market_state.py] --> F[market_state_*.pkl]
        F --> G[MarketStateLookup]
    end
    
    subgraph Model Architecture
        D --> H[QlibQuantMoE Adapter]
        G --> H
        H --> I[QuantMoEModel]
        
        subgraph I[QuantMoEModel]
            J[Value Projection + Factor Embedding]
            K[DifferentiableFeatureSelector]
            L[RegimeContextEncoder]
            M[RegimeAdaptiveMoEBlock × N]
            N[AdaptivePooling]
            O[Linear Head → Stock Score]
        end
    end
    
    subgraph Training
        P[FixedDailyBatchSampler]
        Q[Main Loss (MSE/IC/ListMLE) + IC Monitoring]
        R[Cosine LR Schedule]
    end
    
    subgraph Evaluation
        S[DailyChunkBatchSampler]
        T[IC / RankIC Daily Aggregation]
        U[Qlib Backtest + Report]
    end
```

---

## 2. 核心模块详解

### 2.1 QuantMoEModel (quant_moe_model.py)

**输入维度**:
- `x: [B, T, N]` — B=股票数(日度截面)，T=时间窗口，N=因子数

**核心组件**:

| 组件 | 功能 | 关键参数 |
|------|------|----------|
| `val_proj` | 数值投影 [1] → [D] | `d_model` |
| `factor_id_emb` | 因子ID嵌入 [N] → [N, D] | `num_alphas` |
| `feature_selector` | STG-style可微特征选择 | `selection_temperature`, `selection_reg_lambda` |
| `regime_encoder` | 市场状态编码(内部统计/外部宏观) | `use_external_macro`, `d_macro_input` |
| `layers` | N层 `RegimeAdaptiveMoEBlock` | `n_layers` |
| `factor_pooling` | Attention-based因子聚合 | `pooling_alpha` |
| `head` | 股票打分 [D] → [1] | — |

**前向流程**:
```
x [B,T,N] → val_proj+factor_emb [B,T,N,D] → feature_selection → regime_encoding → MoE_blocks × N → final_norm → factor_pooling [B,D] → head → score [B]
```

### 2.2 RegimeAdaptiveMoEBlock (moe_block.py)

**核心创新**: 同一层内并行运行两个专家，通过 Router 动态加权融合：

```python
# Router: regime_embedding → gate_weights [B, 2]
# Time Expert:   [B,T,N,D] → rearrange → (B*N) attention over T → [B,T,N,D]
# Factor Expert: [B,T,N,D] → rearrange → (B*T) attention over N → [B,T,N,D]
fused = w_time * out_time + w_factor * out_factor
```

**设计亮点**:
1. **Soft Routing**: 使用 softmax 而非 hard selection，保持可微性
2. **z-loss**: 防止 Router 坍塌到单一专家 (`logsumexp^2`)
3. **温度控制**: `router_temperature` 调节 gate sharpness
4. **噪声探索**: `router_noise` 训练时添加logit噪声

### 2.3 RegimeContextEncoder (regime_encoder.py)

**两种模式**:

| 模式 | 输入 | 特征 |
|------|------|------|
| **Internal** | 当前batch `x [B,T,N]` | crowding, PC1 ratio, drift, tail intensity |
| **External** | 预计算 `macro_features [B, d_macro]` | 全市场统计+PCA+delta+market TS |

**Internal mode 计算**:
```python
# X_end = x[:, -1, :]  # [B, N] 最后时间步的因子截面
# 1. crowding = mean abs off-diagonal correlation
# 2. pc1_ratio = top eigenvalue / trace (via eigvalsh)
# 3. drift = |crowd_end - crowd_start|
# 4. tail = fraction of |x| > threshold
```

> **Note**: Internal mode 依赖 batch 内的统计假设（假设 batch = 同日截面），对于非标准采样器可能不准确。

### 2.4 ListMLE Loss (losses.py, optional)

> 主 loss 由 `main_loss` 配置选择（`mse` / `ic` / `listmle`）；ListMLE 仍保留为可选方案。

**核心 Loss**:
```python
# ListMLE: list-wise ranking loss
# 1. 按 target 降序排序 pred
# 2. L = sum_i [ logsumexp_{j>=i}(s_j) - s_i ] / n
```

**数值稳定性处理**:
```python
s = s - s.max().detach()  # 先减最大值
s = s / tau               # 再除以温度
```

### 2.5 Model Adapter (model_adapter.py)

**职责**:
1. Qlib `DatasetH` → PyTorch `DataLoader` 转换
2. 日度截面 batch 采样 (`FixedDailyBatchSampler`)
3. Market state lookup 集成
4. Train/Valid/Test 流程统一
5. Recorder 日志记录

**关键采样器**:

| 采样器 | 用途 | 特点 |
|--------|------|------|
| `FixedDailyBatchSampler` | 训练 | 每日采样固定数量股票，up/down-sample |
| `DailyChunkBatchSampler` | 验证/测试 | 每日全覆盖，分 chunk 返回 |

---

## 3. 数据流时间对齐分析

### 3.1 Label 定义

```python
# work_flow.py line 74
"label": ["Ref($close, -5) / Ref($close, -1) - 1"]
# = (T+5 close / T+1 close) - 1
# = T+1 到 T+5 的收益
```

**预测场景**: T 日收盘后决策，预测 T+1 ~ T+5 收益

### 3.2 Macro Feature 时间对齐

```
Timeline:
   T-2      T-1       T        T+1      T+2      ...      T+5
    |        |        |         |        |                 |
                     [Close]   [Open]
                       ↓         ↓
              Features computed  Prediction executed
              up to here         starting here
                       ↓
               market_state_shift=0 (correct)
```

**验证结论**: ✅ 当前配置无 look-ahead bias

| 组件 | 配置 | 时间安全性 |
|------|------|------------|
| `market_state_shift` | `0` (default) | ✅ T日状态用于T+k预测 |
| `--market_ts_past_only` | `False` (default) | ✅ 使用T日close计算的TS特征 |
| Rolling z-score | `shift_stats=False` | ✅ 滚动窗口到T日 |
| Δstate | `state_T - state_{T-lag}` | ✅ 纯历史 |

---

## 5. 架构设计评价

### 5.1 优点

| 设计点 | 评价 |
|--------|------|
| **Regime-Adaptive MoE** | 创新性强，将非稳态视为结构问题而非噪声 |
| **Dual Expert (Time/Factor)** | 符合金融直觉（趋势 vs 截面选股） |
| **Main Loss (MSE/IC/ListMLE)** | 主损失可配置；ListMLE 适合 ranking，MSE/IC 更易对齐监控 |
| **Macro Feature Pipeline** | 完整的预计算+lookup机制，避免训练时 I/O |
| **时间对齐文档** | macro_feature.md 详细说明了 shift/past_only 的正确用法 |

