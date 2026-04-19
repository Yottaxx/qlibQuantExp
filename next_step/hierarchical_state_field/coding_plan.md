# Deterministic Global-Local Hyper-State Coding Plan

> Status note (2026-04-09):
> Phase 1-3 has landed in code as a flagged rollout.
> The legacy path remains available, and the default workflow has not been switched yet.
> See `implementation_status_2026-04-09.md` for the current checkpoint.

> Scope: 本文是设计稿，不修改代码。目标是在当前 RST-MoE 架构中，把单一 `regime_embedding` 升级为分层状态场，同时尽量复用现有模块骨架。

---

## 1. 目标

当前主干在 [quant_moe_model.py](/Users/yotta/PycharmProjects/publications/qlibQuantExp/module/quant_moe_model.py#L239) 中先产生一个单点 `regime_embedding`：

$$
r_{i,d} = \mathrm{RegimeEncoder}(x_{i,d}, m_d)
$$

然后同一个 $r_{i,d}$ 同时驱动：

$$
r_{i,d} \rightarrow \{\tau_d,\ \gamma_{i,d},\ \beta_{i,d},\ \pi_{i,d}^{(\ell)},\ q_{i,d},\ \alpha_{i,d}\}
$$

这会把 market-level state、stock-local residual、router rule、pooling rule 全部挤在同一个 latent 上。

目标结构改为：

$$
z_d^{g} = G(m_d, u_d)
$$

$$
z_{i,d}^{\ell} = L(x_{i,d}, z_d^{g})
$$

并按职责分配控制头：

$$
\tau_d = f_{\tau}(z_d^{g})
$$

$$
\left(\gamma_{i,d}, \beta_{i,d}\right) = f_{\mathrm{FiLM}}(z_d^{g}, z_{i,d}^{\ell})
$$

$$
\pi_{i,d}^{(\ell)} = f_{\pi}^{(\ell)}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

$$
\left(q_{i,d}, \alpha_{i,d}\right) = f_{\mathrm{pool}}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

---

## 2. 设计原则

1. 先拆状态职责，不先做 variational 或 dynamic weights。
2. `global state` 必须是 day-level 稳定对象，不依赖训练 batch 抽样噪声。
3. `local state` 必须是 stock-specific，对同一天不同股票可区分。
4. 各控制头不再共享同一个 condition vector。
5. 尽量复用现有 `time embedding / FiLM / router / pooling` 模块，只改 condition flow。

---

## 3. 目标拓扑

### 3.1 状态场

$$
m_d,\ u_d \rightarrow z_d^{g}
$$

$$
x_{i,d},\ z_d^{g} \rightarrow z_{i,d}^{\ell}
$$

这里：

- $m_d$: 外部 market state，如当前 `market_state.pkl`
- $u_d$: 稳定的 day summary
- $z_d^{g}$: 全市场共享状态
- $z_{i,d}^{\ell}$: 个股局部状态

### 3.2 控制图

$$
z_d^{g} \rightarrow \tau_d
$$

$$
\left(z_d^{g}, z_{i,d}^{\ell}\right) \rightarrow \mathrm{FiLM}
$$

$$
\left(z_d^{g}, z_{i,d}^{\ell}, u_d\right) \rightarrow \mathrm{Router}
$$

$$
\left(z_d^{g}, z_{i,d}^{\ell}, u_d\right) \rightarrow \mathrm{Pooling}
$$

---

## 4. 建议新增模块

### 4.1 `GlobalStateEncoder`

职责：

$$
G:\ (m_d, u_d) \mapsto z_d^{g}
$$

输入建议：

- 外部 `macro_features`
- 稳定 `day_summary`

输出 shape 建议：

$$
z_d^{g} \in \mathbb{R}^{D_g}
$$

实现建议：

- 小型 MLP 即可
- 输出后广播成 `[B, D_g]`
- 不要直接复用当前 sample-wise `RegimeContextEncoder` 输出

### 4.2 `LocalStateEncoder`

职责：

$$
L:\ (x_{i,d}, z_d^{g}) \mapsto z_{i,d}^{\ell}
$$

输入建议：

- stock-local panel summary
- broadcast 后的 global state

stock-local summary 推荐首版用：

$$
\bar x_{i,d} = \operatorname{Mean}_{t,n}(x_{i,d,t,n,:})
$$

或：

$$
\bar x_{i,d} =
\left[
\operatorname{Mean}_{n}(x_{i,d,T,n,:}),
\operatorname{Std}_{t,n}(x_{i,d,t,n,:})
\right]
$$

输出 shape 建议：

$$
z_{i,d}^{\ell} \in \mathbb{R}^{D_{\ell}}
$$

---

## 5. 现有模块的接口调整

### 5.1 `RegimeAdaptiveTimeEmbedding`

当前语义：

$$
\tau = f_{\tau}(r)
$$

目标语义：

$$
\tau_d = f_{\tau}(z_d^{g})
$$

改动要点：

- 输入从 `regime_embedding` 改成 `global_state`
- 仍然保持当前 `tau_base + small delta` 的稳定参数化
- 不引入 stock-specific tau

### 5.2 `RegimeAdaptiveFactorGate`

当前语义：

$$
(\gamma,\beta)=f_{\mathrm{FiLM}}(r,e_n)
$$

目标语义：

$$
(\gamma_{i,d}, \beta_{i,d})
=
f_{\mathrm{FiLM}}(z_d^{g}, z_{i,d}^{\ell}, e_n)
$$

改动要点：

- condition 维度改为 `concat(global_state, local_state)`
- 保留当前 zero-init / identity-start 逻辑
- 让 FiLM 真正变成 stock-specific modulation

### 5.3 `RegimeAdaptiveMoEBlock.router`

当前语义：

$$
[w_t,w_f] = f_{\mathrm{router}}([r,u])
$$

目标语义：

$$
[w_t,w_f]_{i,d}^{(\ell)}
=
f_{\mathrm{router}}([z_d^{g}, z_{i,d}^{\ell}, u_d])
$$

改动要点：

- block `forward` 显式接收 `global_state / local_state / day_summary`
- `router_input_dim` 需要重新定义
- 仍保留当前 `time vs factor` 两分支 softmax 语义

### 5.4 `RegimeAdaptivePooling`

当前语义：

$$
(q,\alpha)=f_{\mathrm{pool}}(r,u)
$$

目标语义：

$$
(q_{i,d}, \alpha_{i,d})
=
f_{\mathrm{pool}}([z_d^{g}, z_{i,d}^{\ell}, u_d])
$$

改动要点：

- 读出规则从 quasi day-global 变成 global+local conditioned
- 保留当前 pooling 结构，不先改成 full temporal readout

---

## 6. forward 流程

首版推荐的前向流程：

1. 从输入 batch 构造 `stable day summary`：

$$
u_d = U(m_d, \text{stable-day-side-channel})
$$

2. 生成 global state：

$$
z_d^{g} = G(m_d, u_d)
$$

3. 对每只股票生成 local state：

$$
z_{i,d}^{\ell} = L(x_{i,d}, z_d^{g})
$$

4. 生成 time scale：

$$
\tau_d = f_{\tau}(z_d^{g})
$$

5. 生成 FiLM：

$$
(\gamma_{i,d}, \beta_{i,d}) = f_{\mathrm{FiLM}}(z_d^{g}, z_{i,d}^{\ell})
$$

6. 每层 block 内用：

$$
\pi_{i,d}^{(\ell)} = f_{\pi}^{(\ell)}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

7. 最终 pooling 用：

$$
(q_{i,d}, \alpha_{i,d}) = f_{\mathrm{pool}}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

---

## 7. `u_d` 的处理原则

这是该方案成败的关键之一。

当前代码里 `layer_summary` 直接来自当前 batch 统计，这会把 sampler 噪声写进 condition flow。  
在层级状态场方案中，`u_d` 不应直接等于训练 batch summary，而应是：

$$
u_d = H(\mathcal X_d)
$$

其中 $\mathcal X_d$ 是当天完整截面或其稳定缓存。

工程建议：

1. 首版允许 block 内保留 `layer_summary` 作为附加信号。
2. 但 `GlobalStateEncoder` 不应直接依赖 sampled batch summary。
3. 最稳妥的是提供独立的 day-level side channel。

---

## 8. 配置层建议

建议新增配置项：

- `use_hierarchical_state_field`
- `d_global_state`
- `d_local_state`
- `global_state_use_macro`
- `global_state_use_day_summary`
- `local_state_input_mode`
- `router_use_global_state`
- `router_use_local_state`
- `film_use_global_state`
- `film_use_local_state`
- `pooling_use_global_state`
- `pooling_use_local_state`

这些配置不应和未来的 variational / dynamic weights 配置混在一起。

---

## 9. diagnostics 建议

当前已有：

- `time_tau`
- `gate_entropy`
- `time_ratio`
- `factor_gate_*`
- `pooling_alpha`

新增后至少应记录：

- `global_state_norm`
- `global_state_day_variance`
- `local_state_norm`
- `local_state_cross_sectional_variance`
- `router_global_sensitivity`
- `router_local_sensitivity`
- `film_global_sensitivity`
- `film_local_sensitivity`
- `pool_global_sensitivity`
- `pool_local_sensitivity`

目标是验证：

1. global/local state 是否真的分工
2. router 是否不再退化成近 day-global constant gate
3. FiLM 和 pooling 是否真正利用了 local state

---

## 10. 最小 ablation matrix

最小实验矩阵应包括：

1. `base`
   当前单一 `regime_embedding`
2. `global-only`
   只引入 $z_d^{g}$，不引入 $z_{i,d}^{\ell}$
3. `local-only residual`
   在旧 global 之上加入 $z_{i,d}^{\ell}$
4. `global + local`
   完整 `Deterministic Global-Local Hyper-State`
5. `global + local + stable day summary`
   验证 `u_d` 去 sampler 化的收益

最关键的判定问题不是整体分数，而是：

$$
\text{router / FiLM / pooling 是否出现更合理的 global-local 分工}
$$

---

## 11. 分阶段实施顺序

### Phase 1

- Status 2026-04-09: done in code behind `use_hierarchical_state_field`

- 新增 `GlobalStateEncoder`
- `time_tau` 改吃 `global_state`

### Phase 2

- Status 2026-04-09: done in code behind `use_hierarchical_state_field`

- 新增 `LocalStateEncoder`
- `router` 改吃 `global + local + day_summary`

### Phase 3

- Status 2026-04-09: done in code behind `use_hierarchical_state_field`

- `FiLM` 改吃 `global + local`
- `pooling` 改吃 `global + local + day_summary`

### Phase 4

- Status 2026-04-09: not started

- 再考虑：
  - global cross expert
  - variational global state
  - dynamic router / pooling heads

---

## 12. Implementation Checkpoint (2026-04-09)

- Phase 1-3 is implemented as a flagged rollout.
- Legacy single-latent path is still available and remains the default workflow.
- `time_tau` now reads `global_state` only in the new path.
- `FiLM / router / pooling` now read explicit `global_state + local_state`, with summary branches still optional.
- Day-summary fail-fast is part of the hierarchical path when `global_state_use_day_summary=True`.
- Run metadata and comparison tooling now record the hierarchical config surface and fixed ablation labels.
- Code-level smoke checks passed for legacy path, hierarchical path, and the basic invariance check:
  - `global_state` unchanged under stock permutation when day-level inputs are fixed
  - `tau_d` unchanged under stock permutation when day-level inputs are fixed
  - `local_state` changes with stock-level `x`
- Remaining gate before any default switch:
  - run the 5-way ablation matrix
  - verify stable-summary chunk metrics stay near zero
  - verify stock-level routing / pooling heterogeneity is restored
  - verify `global_local_stable_summary` stays within the `<= 0.002` IC / RankIC degradation budget vs `base`

---

## 13. 最终判断

对于当前 RST-MoE，`Deterministic Global-Local Hyper-State` 是最优第一步，因为它：

1. 直接解决单一 shared latent 过载；
2. 不推翻现有 `time / factor / FiLM / pooling` 骨架；
3. 比 variational 方案稳得多；
4. 比 Shared Dynamics 更适合当前阶段；
5. 为后续 `global cross expert`、`variational state field`、`dynamic heads` 保留了清晰演化路径。
