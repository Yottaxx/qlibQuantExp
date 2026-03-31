# Stable Market State Field Coding Plan

> Scope: 本文是设计稿，不修改代码。目标是把当前 [scripts/precompute_market_state.py](/Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py) 从“输出一个混合 market_state 文件”的脚本，重构成能稳定产出 `m_d`、`o_d`、`z_d^g` 三层资产的离线构建器。

---

## 1. 当前脚本的真实问题

当前脚本产出的 `state_df` 同时混合了三类对象：

1. full-day cross-sectional observation；
2. external benchmark market context；
3. temporal transforms such as deltas, rolling means, and rolling z-scores。

形式上可写成：

$$
\text{state\_df}
=
[o_d,\ m_d,\ \Delta(\cdot),\ \mathrm{roll}(\cdot),\ z(\cdot)]
$$

这会导致：

- $m_d$、$o_d$、$z_d^g$ 语义不分；
- temporal filtering 只是“堆特征”，而不是显式状态更新；
- asset 无法稳定版本化；
- 后续模型无法明确知道自己读到的是 context、observation 还是 filtered field。

---

## 2. 目标重构

脚本应重构为三个 builder：

### 2.1 `build_market_context_df`

输出：

$$
m_d
$$

### 2.2 `build_market_observation_df`

输出：

$$
o_d
$$

### 2.3 `build_market_field_df`

输出：

$$
z_d^g
$$

最终产物建议是三份分离资产：

- `daily_market_context.pkl`
- `daily_market_observation.pkl`
- `daily_market_field.pkl`

这三份资产的职责分别是：

$$
m_d = \text{external market context}
$$

$$
o_d = \text{full-day cross-sectional observation}
$$

$$
z_d^g = \text{temporally filtered stable market field}
$$

其中只有最后一份 `daily_market_field.pkl` 直接进入当前模型推理路径。

---

## 2.5 完整实施目标

本次重构的完整目标不是“改一个脚本”，而是建立一条稳定的数据资产流水线：

1. **对象层**
   明确区分 `context / observation / field`。
2. **构建层**
   脚本可稳定离线生成三层资产。
3. **协议层**
   所有 fit / scale / PCA / filter 都遵守 train-only 与 causal 原则。
4. **消费层**
   模型只消费 `daily_market_field.pkl`，不再直接依赖混合式 `market_state.pkl`。
5. **审计层**
   每层资产都可追踪 schema、fit range、coverage、shift、build version。

---

## 3. 当前函数与目标对象的映射

### 3.1 应保留到 `build_market_observation_df` 的函数

这些函数语义上属于 full-day observation：

- [_agg_global]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L108 )
- [_corr_summaries]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L72 )
- [_factor_stats]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L206 )
- train-only PCA fit / transform:
  - [_pca_fit]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L225 )
  - [_pca_transform]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L244 )

### 3.2 应保留到 `build_market_context_df` 的函数

这些函数语义上属于 external market context：

- [_market_ts_features]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L360 )

但该函数目前只覆盖 benchmark close-based features，后续需要扩充到：

- returns
- volatility
- drawdown
- range
- volume / amount
- optional style or macro side inputs

### 3.3 不应继续出现在 `m_d` 或 `o_d` 的逻辑

以下逻辑不应再直接拼进 raw state table：

- `delta_lags` [scripts/precompute_market_state.py]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L856 )
- `roll_mean` [scripts/precompute_market_state.py]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L897 )
- `zscore_windows` [scripts/precompute_market_state.py]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L907 )

这些逻辑应迁移到 `build_market_field_df`，因为它们本质上属于 temporal filtering / state update，而不是原始 context or observation。

---

## 4. `build_market_context_df` 设计

### 4.1 输入

建议输入包括：

- benchmark index OHLCV / amount
- optional external macro file
- optional style index series

### 4.2 输出

输出：

$$
m_d \in \mathbb{R}^{D_m}
$$

列建议使用 `market_ctx_*` 前缀，例如：

- `market_ctx_ret_1d`
- `market_ctx_ret_5d`
- `market_ctx_vol_5d`
- `market_ctx_vol_20d`
- `market_ctx_dd_20d`
- `market_ctx_range_1d`
- `market_ctx_amount_z`
- `market_ctx_volume_z`

### 4.2.1 推荐字段表

推荐的 `m_d` 首版字段如下。

#### 价格与收益

- `market_ctx_ret_1d`
- `market_ctx_ret_5d`
- `market_ctx_ret_20d`

#### 波动与回撤

- `market_ctx_vol_5d`
- `market_ctx_vol_20d`
- `market_ctx_dd_20d`
- `market_ctx_range_1d`

#### 成交与流动性

- `market_ctx_amount_1d`
- `market_ctx_amount_z20`
- `market_ctx_volume_1d`
- `market_ctx_volume_z20`

#### 可选外部扩展

- `market_ctx_style_*`
- `market_ctx_macro_*`

### 4.2.2 v1 最小 schema

如果先走最小可用版本，建议保留 8 到 12 维：

- `market_ctx_ret_1d`
- `market_ctx_ret_5d`
- `market_ctx_ret_20d`
- `market_ctx_vol_5d`
- `market_ctx_vol_20d`
- `market_ctx_dd_20d`
- `market_ctx_range_1d`
- `market_ctx_amount_z20`
- `market_ctx_volume_z20`

### 4.3 关键约束

- 完全 independent of sampled stock batches
- 完全 causal
- train-only scaling if normalization is applied

---

## 5. `build_market_observation_df` 设计

### 5.1 输入

输入是完整的 day-level stock feature panel：

$$
\mathcal X_d = \{x_{i,d}\}_{i \in \mathcal M_d}
$$

当前脚本已经通过 [dataset.prepare]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L627 ) 和按日分组 [scripts/precompute_market_state.py]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L747 ) 获取这一路径。

### 5.2 预处理

保留当前已有的：

- 行 NaN 过滤 [scripts/precompute_market_state.py]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L770 )
- trade / suspension 过滤 [scripts/precompute_market_state.py]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L781 )
- robust outlier 过滤 [scripts/precompute_market_state.py]( /Users/yotta/PycharmProjects/publications/qlibQuantExp/scripts/precompute_market_state.py#L799 )

### 5.3 输出结构

输出：

$$
o_d = [o_d^{\text{global}},\ o_d^{\text{corr}},\ o_d^{\text{pca}}]
$$

列建议使用 `market_obs_*` 前缀：

- `market_obs_mean_abs`
- `market_obs_std`
- `market_obs_breadth`
- `market_obs_tail_2sigma`
- `market_obs_corr_mean_abs`
- `market_obs_corr_fro`
- `market_obs_corr_pc1_ratio`
- `market_obs_pca_0 ... market_obs_pca_k`

建议额外补充：

- `market_obs_corr_effective_rank`

### 5.3.1 推荐字段表

#### 全局分布统计

- `market_obs_mean_abs`
- `market_obs_std`
- `market_obs_breadth`
- `market_obs_tail_2sigma`

#### 拥挤与共振统计

- `market_obs_corr_mean_abs`
- `market_obs_corr_fro`
- `market_obs_corr_pc1_ratio`
- `market_obs_corr_effective_rank`

其中：

$$
\text{effective-rank}(C_d)
=
\exp\!\left(
-\sum_j p_j \log p_j
\right),
\qquad
p_j = \frac{\lambda_j}{\sum_k \lambda_k}
$$

#### 因子结构压缩

- `market_obs_pca_0 ... market_obs_pca_{k-1}`

### 5.3.2 v1 最小 schema

建议首版 observation 维度控制在 16 左右：

- 4 个 global stats
- 4 个 correlation / crowding stats
- 8 个 PCA stats

### 5.4 PCA 规则

- PCA fit 必须 train-only
- PCA sidecar 继续保留：
  - `.pca.npz`
  - `.pca.meta.json`

---

## 6. `build_market_field_df` 设计

这是新引入的核心 builder。

### 6.1 输入

$$
m_d,\ o_d
$$

先做 train-only scale：

$$
\tilde m_d = \operatorname{Scale}_m(m_d)
$$

$$
\tilde o_d = \operatorname{Scale}_o(o_d)
$$

再拼接：

$$
e_d = [\tilde m_d,\ \tilde o_d]
$$

### 6.2 多时间尺度滤波

对每个时间尺度 $k$：

$$
z_d^{(k)} = \lambda_k z_{d-1}^{(k)} + (1-\lambda_k)e_d
$$

推荐首版半衰期：

- `5d`
- `20d`
- `60d`

### 6.3 Shock-aware mixing

构造 shock descriptor：

$$
s_d = S_{\text{shock}}(o_d, o_{d-1})
$$

建议使用：

- `pc1_ratio` jump
- `dispersion` jump
- `tail_2sigma` jump
- `breadth` jump
- `market_ctx_vol_5d`

然后：

$$
w_d = \operatorname{softmax}(A s_d + b)
$$

首版允许：

- rule-based weights
- 或 very small offline linear fit

最终：

$$
z_d^g = \sum_{k=1}^{K} w_{d,k} z_d^{(k)}
$$

### 6.4 输出列

建议使用：

- `market_field_0 ... market_field_{D-1}`
- `market_field_w_fast`
- `market_field_w_mid`
- `market_field_w_slow`
- `market_field_shock_score`

### 6.5 推荐 temporal filtering 实现

推荐 first-stage filter 为：

$$
e_d = [\tilde m_d,\ \tilde o_d]
$$

$$
z_d^{(k)} = \lambda_k z_{d-1}^{(k)} + (1-\lambda_k)e_d
$$

$$
z_d^g = \sum_{k=1}^{K} w_{d,k} z_d^{(k)}
$$

其中：

- $K=3$
- half-lives = `[5, 20, 60]`

半衰期到平滑系数的映射：

$$
\lambda_k = \exp\!\left(-\frac{\ln 2}{h_k}\right)
$$

### 6.6 shock score 设计

推荐 shock descriptor 使用 observation jump 与 market context volatility：

$$
s_d =
\left[
|\Delta \text{pc1\_ratio}_d|,
|\Delta \text{std}_d|,
|\Delta \text{tail}_d|,
|\Delta \text{breadth}_d|,
\text{market\_ctx\_vol\_5d}
\right]
$$

首版有两种实现：

1. `rule-based`
   - 高 shock: 增大 fast 权重
   - 中 shock: 偏向 mid
   - 低 shock: 偏向 slow
2. `small offline linear fit`
   - 用一个很小的线性层产生 mixing logits

### 6.7 field 维度建议

建议：

- `context_dim`: 8~12
- `observation_dim`: 16 左右
- `field_dim`: 16~32

如果首版使用直接拼接后滤波，则：

$$
\dim(z_d^g) = \dim(e_d)
$$

若要进一步压缩，可在滤波前加 train-only PCA 或固定线性投影。

---

## 6.8 资产元数据与 sidecar

每层资产除主表外，建议都写 sidecar metadata：

- `build_version`
- `universe`
- `feature_schema`
- `fit_on`
- `fit_range`
- `shift`
- `coverage_start`
- `coverage_end`
- `n_days`
- `dim`

对 observation / field，还应记录：

- PCA dim / fit split
- scale method
- half-lives
- shock mode

---

## 7. CLI 重构建议

当前 CLI 只有单一 `--out`，这会继续强化“混合 state 文件”思维。  
建议拆成：

- `--out-context`
- `--out-observation`
- `--out-field`

以及：

- `--external-context-path`
- `--emit-context`
- `--emit-observation`
- `--emit-field`

针对 temporal filtering：

- `--field-half-lives 5,20,60`
- `--field-use-shock-mixing`
- `--field-shift 0|1`

---

## 8. 函数级重构建议

当前 `main()` 太长，建议拆成：

### 8.1 Data access

- `load_full_day_feature_df(...)`
- `load_aux_trade_weight_df(...)`
- `load_market_context_ts(...)`

### 8.2 Builders

- `build_market_context_df(...)`
- `build_market_observation_df(...)`
- `build_market_field_df(context_df, observation_df, ...)`

### 8.3 Utilities

- `fit_observation_pca(...)`
- `apply_train_scale(...)`
- `compute_shock_score(...)`
- `run_multiscale_filter(...)`

### 8.3.1 推荐函数签名

建议签名如下：

```python
def build_market_context_df(
    *,
    market_index: str,
    start: str,
    end: str,
    market_ts_windows: list[int],
    external_context_path: str | None = None,
) -> pd.DataFrame:
    ...


def build_market_observation_df(
    *,
    dataset,
    dc: dict,
    no_norm: bool,
    weight_field: str | None,
    trade_field: str | None,
    suspend_field: str | None,
    min_trade: float,
    filter_robust_z: float,
    filter_max_bad_frac: float,
    pca_dim: int,
    pca_fit_on: str,
) -> tuple[pd.DataFrame, dict]:
    ...


def build_market_field_df(
    *,
    context_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    fit_mask: pd.Series,
    field_half_lives: list[int],
    field_scale_method: str,
    field_use_shock_mixing: bool,
    field_shift: int,
) -> tuple[pd.DataFrame, dict]:
    ...
```

### 8.3.2 关键内部子函数

推荐再拆出：

- `compute_daily_market_observation(sub_df, aux_df, ...)`
- `fit_train_only_pca(observation_factor_matrix, fit_mask, pca_dim)`
- `fit_train_only_scaler(df, fit_mask, method)`
- `compute_shock_descriptor(obs_df, ctx_df)`
- `run_ema_bank(fused_df, half_lives)`
- `mix_ema_bank(bank_df, shock_df, mode)`

### 8.4 Saving

- `save_context_artifacts(...)`
- `save_observation_artifacts(...)`
- `save_field_artifacts(...)`

---

## 9. 现有参数的迁移建议

### 保留在 observation builder

- `--no_norm`
- `--filter_robust_z`
- `--filter_max_bad_frac`
- `--weight_field`
- `--trade_field`
- `--min_trade`
- `--suspend_field`
- `--pca_dim`
- `--pca_fit_on`

### 保留在 context builder

- `--add_market_ts`
- `--market_index`
- `--market_ts_windows`
- `--market_ts_past_only`

### 迁移到 field builder

- `--state_delta_lags`
- `--roll_mean`
- `--zscore_windows`
- `--macro_scale`
- `--macro_scale_fit_on`

但语义要改：
- 不再是“给混合 state_df 堆列”
- 而是“给 field 更新器提供滤波与归一化设置”

### 9.1 建议新增参数

- `--out-context`
- `--out-observation`
- `--out-field`
- `--emit-context`
- `--emit-observation`
- `--emit-field`
- `--external-context-path`
- `--field-half-lives`
- `--field-use-shock-mixing`
- `--field-shift`
- `--field-scale-method`
- `--field-dim`（若引入额外压缩）

---

## 10. 兼容当前模型路径的方式

当前 [model_adapter.py](/Users/yotta/PycharmProjects/publications/qlibQuantExp/module/model_adapter.py#L581) 只需要一个 date-indexed numeric table。

因此：

- 短期兼容路径：让 `daily_market_field.pkl` 直接替代当前 `market_state.pkl`
- 中期扩展路径：保留 `context_df` 与 `observation_df` 作为 side assets

也就是说：

$$
\text{model-facing asset} = z_d^g
$$

而：

$$
m_d,\ o_d
$$

主要用于构建、审计与后续更深研究。

---

## 11. 实施顺序

### Phase 1

- 从当前脚本中正式拆出 `context_df`
- 正式重命名 observation 列为 `market_obs_*`
- 保持当前模型仍读取旧 `market_state.pkl`

**交付物**

- `daily_market_context.pkl`
- `daily_market_observation.pkl`
- schema / metadata sidecars

**验收标准**

- 日期覆盖与旧脚本一致
- 不出现重复日期
- PCA fit 仍然 train-only
- `context_df` 与 `observation_df` 字段语义清晰无重叠

### Phase 2

- 新增 `build_market_field_df`
- 用多时间尺度 EMA 生成 `daily_market_field.pkl`

**交付物**

- `daily_market_field.pkl`
- field metadata sidecar

**验收标准**

- field 仅依赖过去与当天信息
- `field_shift` 行为清楚
- fast / mid / slow 权重可诊断
- field 在冲击时期出现合理响应

### Phase 2.5

- 生成分析报告
- 对比 `context / observation / field`
- 检查 field 与原 `market_state.pkl` 的重叠和差异

**验收标准**

- 可视化出代表性日期的 state 变化
- 能定位高 shock 日期的 field 变化来源

### Phase 3

- 将模型配置中的 `market_state_path` 改为指向 `daily_market_field.pkl`
- 保留 `context_df` / `observation_df` 作为审计资产

**验收标准**

- 当前 `model_adapter.py` 无需修改 lookup 协议即可读取 field
- 训练/推理可正常完成
- `market_state_strict` 与 coverage 一致

### Phase 3.5

- 在分析脚本中对 `time_tau / router / pooling` 与 `market_field_*` 的关系做审计
- 确认 stable field 真正在模型诊断里有信息含量

### Phase 4

- 再考虑 learned temporal filter / switching field / variational field

**升级前提**

- deterministic field 已稳定
- current field 对 downstream 机制和指标有可观察影响
- side assets 的版本化与审计链路已跑通

---

## 11.1 完整项目计划

### Stage A. 对象清洗

目标：

- 把 `market_state` 拆成 `context / observation / field`
- 停止继续扩张混合式 `state_df`

成功标准：

$$
m_d \neq o_d \neq z_d^g
$$

### Stage B. 资产构建

目标：

- 建立稳定的 daily asset pipeline
- 固化 PCA / scale / metadata / sidecar 规则

成功标准：

- 同样输入可重复构建相同资产
- train-only fit 行为稳定

### Stage C. 模型接入

目标：

- 让 `daily_market_field.pkl` 成为模型唯一 market-state 输入
- `context / observation` 退出模型主路径，保留作审计

成功标准：

- `market_state_path` 切换后不需要额外协议修改

### Stage D. 机制验证

目标：

- 检验 `field` 是否优于旧的混合 state
- 检验 `field` 是否比 batch summary 更稳定

成功标准：

- 机制诊断更稳定
- downstream conditioning 对 market shocks 更一致

---

## 12. 最终判断

对当前项目，最符合研究方向且工程可行的 temporal filtering 实现，不是复杂的 end-to-end learned SSM，而是：

$$
m_d \rightarrow \text{context asset}
$$

$$
o_d \rightarrow \text{full-day observation asset}
$$

$$
z_d^g = \Phi(z_{d-1}^g,\ m_d,\ o_d)
$$

其中 $\Phi$ 使用多时间尺度 causal filter 与 shock-aware mixing。

这条路线的优势是：

1. 语义清楚；
2. 与当前代码兼容；
3. 可离线构建；
4. 可版本化；
5. 为未来更强的 learned state field 留出平滑升级空间。

---

## 13. 风险与控制

### 风险 1: `m_d` 与 `o_d` 重复编码

控制：

- 严格规定 `m_d` 不读取 full-day stock panel
- 严格规定 `o_d` 不读取 external macro series

### 风险 2: PCA 或 scale 泄漏

控制：

- fit mask 明确只用 train split
- meta sidecar 固定记录 fit range

### 风险 3: temporal filter 过平滑

控制：

- 保留 fast branch
- 使用 shock-aware mixing
- 记录 `w_fast / w_mid / w_slow`

### 风险 4: temporal filter 过敏感

控制：

- 首版优先 rule-based mixing
- 限制 field 维度
- 不上复杂 learned filter

### 风险 5: 与当前模型接入不兼容

控制：

- 保持 output 为 date-indexed numeric DataFrame
- 首版 field 直接兼容 `market_state_path`

---

## 14. 回滚策略

如果任一阶段出现问题，应允许三种回滚：

1. 只使用 `context_df`，不使用 `field_df`
2. 使用 `observation_df` 的 train-only scaled version 直接代替旧 `market_state.pkl`
3. 完全回退到旧脚本输出

这要求新脚本在落地时保留：

- 旧输出路径兼容模式
- 新旧 schema 对照表
- 独立 sidecar metadata
