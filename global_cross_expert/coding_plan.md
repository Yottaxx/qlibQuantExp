# Global Cross Expert Coding Plan

> Scope: 本文是设计稿，不修改代码。目标是在当前 RST-MoE 框架中，引入 `global Kronecker low-rank cross experts`，并保持现有 `time/factor dual expert` 主干不被推翻。

---

## 1. 设计目标

当前 `RegimeAdaptiveMoEBlock` 的主干是：

$$
h_{\text{base}}
=
w_t h_t + w_f h_f
$$

其中：

- $h_t$ 只沿时间轴传播；
- $h_f$ 只沿因子轴传播；
- 缺少显式的 time-factor 联合耦合项。

本设计的目标是在 block 内新增一个小规模、稳定、可解释的全局 cross correction：

$$
h_{\text{fused}}
=
w_t h_t
+
w_f h_f
+
\eta_g \sum_{r=1}^{R_g} c_{d,r}^g \, \mathcal C_r(x)
$$

其中：

$$
\mathcal C_r(x) = A_r x B_r^\top
$$

并对 $A_r, B_r$ 使用 low-rank 参数化。

---

## 2. 总体结构

### 2.1 当前 block

当前 block 可概括为：

1. `norm1(x)`
2. `factor_film`
3. `time_expert`
4. `factor_expert`
5. `router -> softmax over [time, factor]`
6. `fused`
7. residual add
8. `norm2 + ffn`

### 2.2 新 block

新增后的 block 推荐为：

1. `norm1(x)`
2. `factor_film`
3. `time_expert(x) -> h_t`
4. `factor_expert(x) -> h_f`
5. `build_day_condition(regime_embedding, layer_summary) -> z_d^g`
6. `global_cross_bank(x, z_d^g) -> h_g`
7. `router(time/factor only) -> [w_t, w_f]`
8. `fused = w_t h_t + w_f h_f + h_g`
9. residual add
10. `norm2 + ffn`

注意：

- `global_cross_bank` 是 residual correction branch，不进入现有 `time/factor` softmax 竞争；
- `router` 仍然只负责 `time vs factor` 的 base routing；
- `global_cross` 通过自己的 amplitude controller 单独调幅。

---

## 3. 新增模块

### 3.1 `GlobalCrossController`

职责：

$$
z_d^g \mapsto c_d^g \in \mathbb{R}^{R_g}
$$

输入建议：

$$
z_d^g = [\bar r_d,\ u_d]
$$

其中：

- $\bar r_d = \operatorname{Mean}_i(r_{i,d})$：day-level regime summary
- $u_d$：block 内已有的 layer summary

输出形式建议：

$$
c_d^g = s_g \cdot \tanh(\mathrm{MLP}(z_d^g))
$$

而不是 softmax，原因是 `global cross` 是校正项，不是和 `time/factor` 抢主预算。

### 3.2 `GlobalKroneckerCrossExpertBank`

职责：

$$
x \mapsto h_g = \eta_g \sum_{r=1}^{R_g} c_{d,r}^g \, \mathcal C_r(x)
$$

输入：

- `x: [B, T, N, D]`
- `coef_g: [B, R_g]` 或等价的 `[1, R_g]` 广播张量

输出：

- `h_g: [B, T, N, D]`

### 3.3 可选：`GlobalCrossOutputProjection`

职责：

$$
\widetilde h_g = W_{\text{out}} h_g
$$

原因：

- low-rank cross operator 只混合 $T,N$ 维；
- 若不做 channel projection，表达力可能偏弱；
- 轻量投影可增强与主干空间的对齐。

---

## 4. 参数化方案

### 4.1 每个 expert 的算子

第 $r$ 个 global cross expert：

$$
\mathcal C_r(X) = A_r X B_r^\top
$$

其中：

$$
A_r = U_t^{(r)} {V_t^{(r)}}^\top,\qquad
B_r = U_n^{(r)} {V_n^{(r)}}^\top
$$

### 4.2 参数 shape

若：

- `R_g = num_global_cross_experts`
- `k_t = cross_t_rank`
- `k_n = cross_n_rank`

则推荐参数 shape：

- `u_t: [R_g, T, k_t]`
- `v_t: [R_g, T, k_t]`
- `u_n: [R_g, N, k_n]`
- `v_n: [R_g, N, k_n]`

可选 channel 投影：

- `cross_in_proj: Linear(D, D_cross)` 或 `Linear(D, D)`
- `cross_out_proj: Linear(D_cross, D)` 或 `Linear(D, D)`

### 4.3 identity-start

为保持当前主干稳定，推荐：

$$
h_g = \eta_g \cdot \sum_{r=1}^{R_g} c_{d,r}^g \, \mathcal C_r(x)
$$

其中：

- $\eta_g$ 初始为 0 或很小；
- `cross_out_proj` 零初始化也可；
- 两者二选一即可。

---

## 5. forward 张量流

对单个 batch：

$$
x \in \mathbb{R}^{B \times T \times N \times D}
$$

### 5.1 day condition

先构造：

$$
\bar r_d = \operatorname{Mean}_{b}(r_{b,d})
$$

若启用 `router_use_layer_summary`，则已有：

$$
u_d \in \mathbb{R}^{D}
$$

组合成：

$$
z_d^g = [\bar r_d,\ u_d]
$$

再得到：

$$
c_d^g = \mathrm{Controller}(z_d^g)
$$

### 5.2 单 expert 计算

对第 $r$ 个 expert，固定某个 channel slice 的直观形式是：

$$
Y_r = A_r X B_r^\top
$$

低秩实现可改写为：

$$
Y_r
=
U_t^{(r)}
\left(
{V_t^{(r)}}^\top X U_n^{(r)}
\right)
{V_n^{(r)}}^\top
$$

在代码实现中，需要按 `B,D` 维批量广播。

### 5.3 多 expert 聚合

$$
h_g = \eta_g \sum_{r=1}^{R_g} c_{d,r}^g Y_r
$$

### 5.4 block 融合

$$
h_{\text{fused}}
=
w_t h_t
+
w_f h_f
+
h_g
$$

随后继续：

$$
x' = x_{\text{residual}} + h_{\text{fused}}
$$

$$
x_{\text{out}} = x' + \mathrm{FFN}(\mathrm{LN}(x'))
$$

---

## 6. 新增配置项

建议在 `QuantMoEConfig` 中新增：

```python
use_global_cross_expert: bool = False
num_global_cross_experts: int = 2
cross_t_rank: int = 2
cross_n_rank: int = 4
cross_hidden_dim: int | None = None
cross_controller_hidden: int = 32
cross_global_scale: float = 0.1
cross_global_zero_init: bool = True
cross_use_output_proj: bool = True
cross_use_regime_mean: bool = True
cross_use_layer_summary: bool = True
cross_dropout: float = 0.0
```

这些配置不应复用当前 `router_*` 配置，以免语义混淆。

---

## 7. diagnostics 与 recorder 接入

建议新增以下监控量，挂到 `QuantModelOutput.metrics`：

- `global_cross_coef_mean`
- `global_cross_coef_std`
- `global_cross_abs_mean`
- `global_cross_energy_ratio`
- `global_cross_rank_t_util`
- `global_cross_rank_n_util`

其中 energy ratio 可定义为：

$$
\rho_g
=
\frac{\|h_g\|_2}{\|w_t h_t + w_f h_f\|_2 + \epsilon}
$$

并建议导出：

- 每日 `coef_g` 序列
- 每日 `energy_ratio_g` 序列
- 不同 regime bucket 下 `coef_g` 的统计

这样才能证明 `global cross` 不是装饰性小扰动。

---

## 8. 最小 ablation matrix

必须有以下 5 组：

1. `base`: 当前 `time + factor`
2. `base + global_cross`
3. `base + fake_params`
4. `base + global_cross (frozen controller)`
5. `base + global_cross (single expert)`

含义：

- `fake_params`: 控制参数量解释
- `frozen controller`: 控制 “结构库有用” vs “状态调度有用”
- `single expert`: 控制 “operator bank 是否必要”

---

## 9. 实施顺序

### Phase 1

只实现：

- `GlobalCrossController`
- `GlobalKroneckerCrossExpertBank`
- `metrics`

不碰：

- local sparse experts
- stock-local controller
- 新 loss

目标：

$$
\text{验证是否确实需要全局 cross term}
$$

### Phase 2

若 `global cross` 确认有效，再考虑：

- `local sparse cross`
- disentangled controller
- ranking-consistent objective stack

---

## 10. 风险

### 风险 1

`global cross` 退化成无效小残差：

$$
\|h_g\| \approx 0
$$

应通过 energy ratio 检查。

### 风险 2

controller 只学到常数系数：

$$
c_d^g \approx \text{const}
$$

应检查日度方差和 regime bucket 对齐。

### 风险 3

cross expert 与 `time/factor` 严重重叠，无法形成额外结构增益。  
必要时需要引入 expert decorrelation / specialization regularizer。

---

## 11. 一句话总结

这套 coding plan 的核心不是“再加一条大分支”，而是：

$$
\text{在 current dual-expert backbone 上，加入一个日级共享的低秩二维结构校正项}
$$

它应当：

- 参数小；
- identity-start；
- 与现有 router 解耦；
- 能通过 diagnostics 被证实确实在工作。
