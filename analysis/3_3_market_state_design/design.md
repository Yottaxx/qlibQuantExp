# 3.3 Design: From Batch Summary to Market State System

> Scope: 本文是 `3.3 从 batch summary，升级到 stable market state field` 的上层设计稿。它定义 `3.3` 的对象、边界、分层结构、与 `3.1/3.2` 的接口关系，以及推荐的实施顺序。它不替代更细的 [analysis/stable_market_state_field](/Users/yotta/PycharmProjects/publications/qlibQuantExp/analysis/stable_market_state_field) 文档，而是把后者定位为 `3.3` 下的一个具体实现点。

---

## 1. 3.3 的真实目标

`3.3` 的目标不是“再做一个更好的 macro feature 文件”，而是把当前 runtime 的 batch summary conditioning：

$$
\hat u_{d,b} = H(\mathcal B_{d,b})
$$

升级为一个真正独立的市场状态系统：

$$
\mathcal S_d = \text{Market State System on day } d
$$

这个系统应满足四个性质：

1. **day-level**
   它必须按交易日定义，而不是按 batch / chunk 定义。
2. **assetized**
   它必须是离线构建、版本化、可审计的数据资产，而不是模型运行时临时统计。
3. **causal**
   它必须只依赖当日及以前可见信息。
4. **reusable**
   它必须能跨模型、跨 ablation、跨实验复用。

所以 `3.3` 本质上是在把“市场状态”从模型内部的副产物，升级成一个独立的一等公民。

---

## 2. 3.3 的对象分层

`3.3` 的最佳实践不是一个对象，而是三层对象。

### 2.1 Layer A: Market Context

$$
m_d = \text{daily market context}
$$

这是市场级外部上下文。  
它描述：

- 指数收益与波动
- 流动性背景
- 风格/行业背景
- 宏观背景

它的特征是：

- external
- slow-varying
- not cross-sectional

### 2.2 Layer B: Market Observation

$$
o_d = \text{daily market observation}
$$

这是由当天完整股票截面构造的市场观测。  
它描述：

- dispersion
- breadth
- crowding
- correlation concentration
- factor-space geometry

它的特征是：

- same-day
- full-day
- cross-sectional

### 2.3 Layer C: Market Field

$$
z_d^g = \text{stable daily market field}
$$

这是最终供模型消费的稳定市场状态。  
它由：

$$
z_d^g = \Phi(z_{d-1}^g,\ m_d,\ o_d)
$$

构造而来。

它的特征是：

- filtered
- persistent
- model-facing

因此 `3.3` 的核心边界是：

$$
m_d \neq o_d \neq z_d^g
$$

而不是继续把三者混成一个大 `market_state.pkl`。

---

## 3. 3.3 在整体架构中的位置

`3.3` 不属于模型主干，它属于 **state asset layer**。

如果把长期系统拆成五层，`3.3` 位于：

1. `Protocol Layer`
2. `Data Asset Layer`
3. `Model Layer`
4. `Execution Layer`
5. `Evidence Layer`

其中 `3.3` 明确属于：

$$
\text{Data Asset Layer}
$$

更准确地说，它定义的是：

$$
\text{Market State Asset Pipeline}
$$

这也是为什么 `3.3` 必须和 `3.1` 解耦：

- `3.1` 讨论的是模型内部怎么使用状态
- `3.3` 讨论的是状态本身如何被定义、构建、保存、审计

---

## 4. 3.3 与 3.1 的接口

`3.1` 的最优方案是 deterministic Global-Local Hyper-State：

$$
z_d^{g,\text{model}} = G(m_d, u_d)
$$

$$
z_{i,d}^{\ell} = L(x_{i,d}, z_d^{g,\text{model}})
$$

其中 `3.3` 提供的是上游 market-state asset：

$$
z_d^g
$$

因此，`3.3 -> 3.1` 的接口应当是：

$$
z_d^g \rightarrow G(\cdot)
$$

也就是说：

- `3.3` 负责给出稳定的、因果的 day-level market field
- `3.1` 再决定如何把它映射成模型内部的 global state

这意味着第一阶段：

$$
\text{model-facing asset} = z_d^g
$$

而不是让模型直接消费 `m_d` 或 `o_d`。

---

## 5. 3.3 与 3.2 的接口

`3.2` 讨论的是 operator family 扩展，包括：

- time expert
- factor expert
- cross expert

这些模块如果需要市场状态，应当统一从：

$$
z_d^g
$$

读取 day-level context。

所以 `3.3` 的另一个作用是：为未来的 routing、cross expert、pooling 提供一个**统一、稳定、可版本化的 day-level conditioning source**。

如果没有 `3.3`，这些模块将继续依赖：

$$
\hat u_{d,b}
$$

即 sampler 污染的 batch summary。

---

## 6. 3.3 的最佳实践原则

如果压缩成 6 条原则，我建议 `3.3` 永远遵守下面这些：

### 6.1 对象分层先于模型接入

先定义：

$$
m_d,\ o_d,\ z_d^g
$$

再讨论模型怎么消费。

### 6.2 市场状态必须是资产，不是运行时副产物

市场状态必须能独立保存、版本化、审计、复用。

### 6.3 same-day observation 与 external context 必须分离

否则你永远无法清楚知道：

- 市场背景来自哪里
- 当日冲击来自哪里

### 6.4 temporal filtering 必须显式定义为状态更新

不能继续用：

$$
\Delta + \text{rolling} + \text{zscore}
$$

特征堆叠代替真正的状态更新方程。

### 6.5 train-only fit 是协议，不是实现细节

PCA、scale、compression 都必须明示 fit range。

### 6.6 method 和 coding 必须解耦

`method section` 写对象与方程。  
`coding plan` 写资产、schema、builder、protocol、validation、rollout。

---

## 7. 3.3 的推荐内部结构

从系统设计角度，`3.3` 推荐拆成四个子点：

### 3.3.A Object Layer

定义三层对象：

$$
m_d,\ o_d,\ z_d^g
$$

### 3.3.B Asset Builder Layer

定义三类 builder：

- `context builder`
- `observation builder`
- `field builder`

### 3.3.C Protocol Layer

定义：

- causal policy
- train-only fit
- shift policy
- metadata & versioning

### 3.3.D Model Interface Layer

定义：

$$
\text{only } z_d^g \text{ enters the first-stage model path}
$$

这一层结构非常重要，因为它表明 `stable_market_state_field` 不是整个 `3.3`，而只是：

$$
\text{3.3.C + 3.3.D 的一个具体 first-stage implementation}
$$

---

## 8. `stable_market_state_field` 在 3.3 中的定位

当前 [analysis/stable_market_state_field](/Users/yotta/PycharmProjects/publications/qlibQuantExp/analysis/stable_market_state_field) 应被定位为：

$$
\text{3.3 的 first-stage deterministic implementation}
$$

它的特点是：

- 把 `m_d / o_d / z_d^g` 落成三层资产
- 用 deterministic multi-timescale causal filter 构造 `z_d^g`
- 用 shock-aware mixing 在稳定性和响应性之间平衡

但它并不等于整个 `3.3`，因为 `3.3` 还包括：

- 对象边界
- 架构位置
- 与 `3.1/3.2` 的接口
- 版本化和协议设计

所以保留单独目录是正确的：

- 本目录写上层设计
- `stable_market_state_field` 目录写具体实施方案

---

## 9. 3.3 的推荐实施顺序

最佳实践上，`3.3` 不应一步到位，而应分阶段推进。

### Phase 1: Object Separation

目标：

- 正式把 `market_state` 拆成 `context / observation / field`

成功标准：

$$
m_d \neq o_d \neq z_d^g
$$

### Phase 2: Deterministic Field Builder

目标：

- 用 deterministic filter 构造 `z_d^g`

成功标准：

- field 因果
- field 可审计
- field 与 shock 方向一致

### Phase 3: Model Interface Switch

目标：

- 模型第一阶段只消费 `z_d^g`

成功标准：

- downstream conditioning 从旧混合 state 切到 stable field

### Phase 4: Mechanism Audit

目标：

- 验证 field 是否优于 batch summary
- 验证 field 是否能稳定驱动 router / pooling / tau

### Phase 5: Advanced Extensions

目标：

- learned filter
- switching field
- variational field

这一步必须建立在 deterministic field 已经稳定的前提上。

---

## 10. 最终结论

`3.3` 的最佳实践，不是“做一个新的 macro 文件”，而是建立一个独立的 market state system：

$$
m_d = \text{context}
$$

$$
o_d = \text{observation}
$$

$$
z_d^g = \text{field}
$$

并明确：

1. `3.3` 属于 asset layer，而不是 model layer；
2. 第一阶段模型只消费 `z_d^g`；
3. [analysis/stable_market_state_field](/Users/yotta/PycharmProjects/publications/qlibQuantExp/analysis/stable_market_state_field) 是 `3.3` 的具体实施点，而不是 `3.3` 的全部。

这就是 `3.3` 的上层设计边界。
