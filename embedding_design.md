# RST-MoE Embedding 模块设计文档

本文档描述当前代码中 **embedding/conditioning 子系统**的完整设计：从输入张量 `x[B,T,N]` 到进入 `RegimeAdaptiveMoEBlock` 之前/之内的所有表征构建与调制（regime/time/factor/feature-selection），并明确每个模块的原理、计算方式与端到端流程。

> 范围说明：本文档覆盖 `QuantMoEModel` 的 **输入编码 + 条件调制**，不展开 MoE 两个专家的注意力实现细节（它们属于 backbone），但会明确 gate/FiLM/mask 在 block 内的生效位置。

---

## 1. 设计逻辑（Why this embedding design）

### 1.1 核心目标

1) **把股票样本表示成“时间 × 因子 token”的序列张量**：让模型同时具备时序建模与截面（因子）建模能力。  
2) **把市场状态（regime）作为条件变量**：在不同市场状态下，动态改变：
   - 时间尺度（有效记忆长度）
   - 因子通道的调制（FiLM/AdaLN）
3) **保证在 Pre-LN Transformer 下 gate 真正生效**：任何乘法缩放若放在 block 前，会被 `LayerNorm` 近似抵消；因此 gate/mask 必须在 `LayerNorm` 之后应用。
4) **训练稳定性（ResNet-style identity start）**：条件调制层初始为近似恒等映射，避免一开始就“改坏”主干分布。
5) **论文/严格回测无 look-ahead**：macro 的 PCA 只能在 train 上拟合，再对 valid/test 做 transform。

### 1.2 输入张量语义

- `x`: `[B, T, N]`
  - `B`: batch 内股票数（训练时通常是“同一交易日的截面样本”，由 sampler 保证）
  - `T`: 时间窗口长度（context_len）
  - `N`: 因子数（num_alphas）
- `factor_ids`: `[N]`，每个因子的稳定 ID（必须与 `x[..., n]` 的因子顺序对齐）
- `macro_features`（可选）: `[B, d_macro]`，对齐到每个样本日期的全市场状态向量（同一日内通常相同，但实现允许 per-sample）

---

## 2. 模块清单（What’s inside）

embedding/conditioning 的模块分层如下：

1) **Base Token Embedding**
   - `val_proj`: 数值投影 `R -> R^D`
   - `factor_id_emb`: 因子 ID embedding `id -> R^D`
2) **Regime Context**
   - `RegimeContextEncoder`: 从 `x` 的内部统计或 `macro_features` 得到 `regime_embedding r[B,D]`
3) **Regime-Adaptive Time Embedding**
   - `RegimeAdaptiveTimeEmbedding`: 产生 `time_emb[B,T,D]` 并广播到因子维
4) **Regime-Adaptive Factor Conditioning（FiLM/AdaLN）**
   - `RegimeAdaptiveFactorGate`: 输出 `gamma/beta[B,N,D]`，在每个 block 的 `LayerNorm` 之后对 token 激活做 FiLM
5) **Differentiable Feature Selection（STG-style）**
   - `DifferentiableFeatureSelector`: 学习全局因子 mask `z[N]`；mask 在每个 block 的 `LayerNorm` 之后应用
6) **Embedding Dropout**
   - `emb_dropout`: 在进入 backbone 之前对 `[B,T,N,D]` 做 dropout
7) **统一初始化**
   - `QuantMoEModel._init_weights` + `post_init()`: 统一 `Linear/Embedding/MHA(in_proj)` 的初始化尺度，并保留 FiLM 的 identity-start

对应代码位置：
- `module/quant_moe_model.py`
- `module/architecture/regime_encoder.py`
- `module/architecture/regime_adaptive_embedding.py`
- `module/architecture/feature_selector.py`
- `module/architecture/moe_block.py`（说明 FiLM/mask 在 block 内的生效位置）
- `module/utils/model_configuration.py`

---

## 3. 各模块原理与计算方式（How each piece works）

以下约定：
- `D = d_model`
- `x_{b,t,n}` 表示 `x[b,t,n]`

### 3.1 Base Token Embedding：`val_proj + factor_id_emb`

**原理**  
把每个因子在每个时间步的标量值映射到 `D` 维，并加入因子身份向量，使得模型可以区分不同因子通道。

**计算方式**  
1) 数值投影（逐元素线性映射）：

$$v_{b,t,n} = W_v \cdot x_{b,t,n} + b_v \in \mathbb{R}^{D}$$

2) 因子 ID embedding（查表）：

$$e_n = \mathrm{Emb}(\text{factor\_id}_n) \in \mathbb{R}^{D}$$

3) 合成 token：

$$h_{b,t,n} = v_{b,t,n} + e_n$$

**实现要点（shape）**
- `val_proj(x.unsqueeze(-1))`：`[B,T,N,1] -> [B,T,N,D]`
- `factor_id_emb(factor_ids)`：`[N] -> [N,D]`，再 broadcast 到 `[1,1,N,D]`

---

### 3.2 Regime Context：`RegimeContextEncoder`

**原理**  
用一个低维“市场状态”向量 `r_b` 来 condition 后续的 time embedding 与 factor FiLM。regime 有两种来源：

1) **External macro（推荐用于 t+5 等更长周期）**：直接使用预计算的 `macro_features[B,d_macro]`  
2) **Internal stats（fallback）**：从当前 batch 的 `x[B,T,N]` 提取市场统计（依赖 batch 为“同日截面”）

**计算方式**

#### 3.2.1 External macro
直接取：

$$u_b = \text{macro_features}_b \in \mathbb{R}^{d_{\text{macro}}}$$

#### 3.2.2 Internal stats（batch-level）
令 `X_end = x[:, -1, :]`，形状 `[B,N]`（同一日内 B 只股票的因子向量）。

提取 4 维统计特征（并广播到 batch 内每个样本）：
1) **crowding**：平均绝对相关（去掉对角线）
2) **pc1_ratio**：相关矩阵第一特征值占比（market mode 强度）
3) **drift**：跨 lag 的 crowding 变化幅度
4) **tail**：极端值比例（$|x| > \text{threshold}$）

最终得到：

$$u_b \in \mathbb{R}^4$$

#### 3.2.3 编码到 `D` 维 regime embedding
$u_b \to r_b$：

$$r_b = \mathrm{LN}(W_2 \cdot \tanh(W_1 \cdot u_b))$$

> 注意：internal stats 的“市场语义”要求 batch 是单日截面；若 batch 不是同一日，需要关闭 batch-stats 或启用外部 macro。

---

### 3.3 Regime-Adaptive Time Embedding：`RegimeAdaptiveTimeEmbedding`

**原理**  
短窗口（`T≈8~32`）下，纯 attention 往往缺少显式时间位置信号。这里学习一个“lag embedding table” `p[t]`，并用 regime 预测的时间尺度 `tau(r)` 对 lag 做指数衰减，得到 regime-dependent 的时间位置编码。

**计算方式**

#### 3.3.0 符号约定（shapes & 含义）

> 约定：本文所有 `log/ln` 均为自然对数；`exp` 为自然指数函数；`softplus(z)=\log(1+e^z)`。

维度/下标
- $B$：batch 内样本数（通常是一天内的股票截面样本数）。
- $T$：时间窗口长度（`context_len`），时间索引 $t\in\{0,\dots,T-1\}$，其中 **$t=0$ 最早、$t=T-1$ 最新**。
- $N$：因子数（`num_alphas`），因子索引 $n\in\{0,\dots,N-1\}$。
- $D$：隐层维度（`d_model`），向量分量索引 $d\in\{0,\dots,D-1\}$。

主要变量（与实现一一对应）
- $r_b\in\mathbb{R}^D$：第 $b$ 个样本的 regime embedding（`RegimeContextEncoder` 输出，形状 `[B,D]`）。
- $p[t]\in\mathbb{R}^D$：第 $t$ 个时间位置的可学习基表（实现里是参数 `pos[t]`，形状 `[T,D]`）。
- $\tau_b\in\mathbb{R}_{>0}$：第 $b$ 个样本的时间尺度（同一个样本内对所有 $t$ 共用；实现里形状 `[B,1]`）。
- $\tau_{\min},\tau_{\max}$：对 $\tau$ 的下/上界（配置项 `time_tau_min/time_tau_max`）。
- $\tau_{\text{base}}$：可学习的 base 参数（实现里 `tau_base` 存在 **softplus 之前的参数空间**；初始化使得在 $\delta=0$ 时 $\tau\approx \tau_{\text{init}}$）。
- $\delta(r_b)$：由 regime 预测的增量（实现里是两层 MLP + `tau_mlp_out_scale` 的缩放）。
- $\epsilon$：数值稳定项（实现里 `eps`）。
- $w_{b,t}$：时间衰减权重（形状 `[B,T]`），只依赖 $(b,t)$，对所有因子 $n$ 共享。

1) 预测时间尺度：

$$\tau_b = \mathrm{clamp}(\mathrm{softplus}(\tau_{\text{base}} + \delta(r_b)) + \tau_{\min}, \tau_{\max})$$

其中 $\delta(\cdot)$ 是小 MLP 输出的微小增量（带缩放，确保初期稳定）。更展开地写：

$$\delta(r_b) = s_\tau \cdot W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(r_b)) \in \mathbb{R}$$

其中 $s_\tau$ 对应实现里的 `tau_mlp_out_scale`（默认较小，使 early training 更接近恒等/平稳）。

2) 定义 lag（最后一个时间步 lag=0，越往过去 lag 越大）：

$$\text{lag}(t)=T-1-t$$

3) 指数衰减权重（可选归一化，避免不同 regime 下尺度漂移）：

$$w_{b,t} = \exp(-\text{lag}(t)/\tau_b), \quad w_{b,t} \leftarrow \frac{w_{b,t}}{\mathrm{mean}_t(w_{b,t})+\epsilon}$$

4) 时间 embedding：

$$\text{time_emb}_{b,t} = w_{b,t} \cdot p[t] \in \mathbb{R}^{D}$$

5) 加到 token 上（对因子维广播）：

$$h_{b,t,n} \leftarrow h_{b,t,n} + \text{time_emb}_{b,t}$$

#### 3.3.1 `time_tau` 的语义与更稳的默认规则

`tau` 的核心含义是“**最远端（lag=T-1）相对最近端（lag=0）的权重比例**”：

$$r_{\text{tail}}=\frac{w_{lag=T-1}}{w_{lag=0}}=\exp(-(T-1)/\tau)$$

等价地，你也可以用 half-life（权重减半所需的 lag 数）来理解：

$$\text{half_life}=\tau\ln2$$

> 注意：当前实现默认 `time_decay_normalize=True` 会做 `w/=mean(w)`，它不会改变 $r_{\text{tail}}$，但会把近期的权重放大到 >1、把远端压到 <1；因此 **过小的 `tau` 会导致近期 time embedding 幅度偏大**，训练早期更不稳。

**推荐默认（更稳、且保证有“明显但不过激”的时间衰减）**

用目标 tail ratio 来设定 `tau_init`，再用 `2T` 设定 `tau_max`：

```python
# 建议：先选一个你希望的 tail ratio（最远端相对最近端）
r_tail_init = 0.30  # t+5 + 短窗口时的稳健默认：窗口最远端仍保留 ~30% 权重

time_tau_init = (context_len - 1) / log(1 / r_tail_init)  # ≈ 0.83 * context_len
time_tau_max  = min(2.0 * context_len, 50.0)              # 避免 tau 过大导致“几乎无衰减”
```

经验上：
- `r_tail_init≈0.2`：更偏向近端（t+1 常用）
- `r_tail_init≈0.3~0.4`：更均衡（t+5 更稳）

#### 3.3.2 案例分析：`T=8`、预测 `t+5`

取 `r_tail_init=0.30`：
- `time_tau_init = 7 / ln(1/0.3) ≈ 5.81`
- `half_life ≈ 5.81 ln2 ≈ 4.03`（约 4 个 lag 权重减半）
- 未归一化的 `w`（lag=0..7）：`[1.00, 0.842, 0.709, 0.597, 0.502, 0.423, 0.356, 0.300]`
- 若启用 `time_decay_normalize`（默认），归一化后约：`[1.69, 1.42, 1.20, 1.01, 0.85, 0.72, 0.60, 0.51]`

对比两个“极端”：
- 若 `tau_max=50`：`r_tail=exp(-7/50)=0.87`，几乎不衰减（regime 对 time-scale 的调制会变钝）
- 若 `tau_init` 很小（如 `2`）：归一化后最近端权重会被放大到 `~3x`，训练早期更容易出现 embedding 尺度失衡

#### 3.3.3 尺度分析：`time_emb` 加进来会不会太大？

目标：估算初始化阶段（还未训练时）三项相加的量级，判断 `time_emb` 是否可能“压过” base token。

**Step 0：明确相加项**

对每个 token（固定 `(b,t,n)`），进入 backbone 之前：

$$h_{b,t,n} = v_{b,t,n} + e_n + \text{time_emb}_{b,t}$$

其中：
- $v_{b,t,n}=\mathrm{val\_proj}(x_{b,t,n})\in\mathbb{R}^{D}$
- $e_n=\mathrm{Emb}(\text{factor_id}_n)\in\mathbb{R}^{D}$
- $\text{time_emb}_{b,t}=w_{b,t}\,p[t]\in\mathbb{R}^{D}$

**Step 1：统一初始化的前提**

初始化时（默认）：
- `Linear/Embedding`：每个权重 $W\sim\mathcal{N}(0,\sigma^2)$，$\sigma=\text{initializer_range}=0.02$
- `pos[t]`：每个元素 $p[t,d]\sim\mathcal{N}(0,\sigma_{\text{time}}^2)$，$\sigma_{\text{time}}=\text{time_emb_init_std}=0.02$

**Step 2：估算 $v=\mathrm{val\_proj}(x)$ 的每维标准差**

`val_proj` 是 `Linear(1→D)`，对任意输出维 $d$：

$$v_d = W_d\,x + b_d$$

初始化时 $b_d=0$，$W_d\sim\mathcal{N}(0,\sigma^2)$。若输入已做 z-score（近似 $x\sim\mathcal{N}(0,1)$），则：

$$\operatorname{Var}(v_d)\approx\mathbb{E}[W_d^2]\operatorname{Var}(x)=\sigma^2\cdot 1=\sigma^2$$

所以：

$$\operatorname{Std}(v_d)\approx\sigma=0.02$$

**Step 3：估算 $e=\mathrm{factor\_id\_emb}$ 的每维标准差**

同理：

$$\operatorname{Std}(e_d)\approx\sigma=0.02$$

**Step 4：base token（$v+e$）的每维标准差**

若近似独立：

$$\operatorname{Var}(v_d+e_d)\approx\sigma^2+\sigma^2=2\sigma^2$$

$$\operatorname{Std}(v_d+e_d)\approx\sqrt{2}\sigma\approx 0.0283$$

**Step 5：time embedding（$w\cdot p[t]$）的每维标准差**

对固定的 $t$：

$$\text{time_emb}_{t,d}=w_t\,p[t,d] \Rightarrow \operatorname{Std}(\text{time_emb}_{t,d})\approx |w_t|\sigma_{\text{time}}$$

在 `T=8,t+5,r_tail=0.30` 的默认推荐下，最大 $w_0\approx 1.69$，所以：

$$\operatorname{Std}(\text{time_emb}_{0,d})\approx 1.69\times 0.02 \approx 0.0338$$

对比 base token 的 $0.0283$，两者是同量级（time 约大 $1.19\times$），不属于“爆炸”。

**Step 6：三项相加后的粗略量级**

仍用独立近似（取最不利的最大 $w_0$）：

$$\operatorname{Std}(v+e+\text{time})\approx\sqrt{(0.0283)^2+(0.0338)^2}\approx 0.044$$

同时，进入每个 block 的第一步是 `LayerNorm`（Pre-LN），会进一步降低“纯尺度偏大”带来的风险；最终更关键的是 time 信号是否提供了有效的方向信息。

**什么时候可能真的“太大”？**

当 `time_decay_normalize=True` 且 $\tau$ 被推得很小（接近 `time_tau_min`）时，`w/=mean(w)` 会把近端权重显著放大到 $O(T)$。
例如 `T=8, tau=0.5` 时，$w_0/\mathrm{mean}(w)\approx 6.9$，对应

$$\operatorname{Std}(\text{time_emb}_{0,d})\approx 6.9\times 0.02 \approx 0.14$$

这时就可能压过 base token。实践上可通过提高 `time_tau_min`（如 `2.0`）、减小 `time_emb_init_std`（如 `0.01`），或关闭 `time_decay_normalize` 来规避。

#### 3.3.4 `time_decay_normalize` 的正向作用是什么？

`time_decay_normalize=True` 的核心作用是把

$$w_t \leftarrow \frac{w_t}{\mathrm{mean}_t(w_t)+\epsilon}$$

使得不同 $\tau$（不同 regime）下的时间权重满足 **$\mathrm{mean}(w)\approx 1$**。这带来三个直接好处：

1) **避免“能量漂移”**：$\tau$ 只改变相对形状（近端 vs 远端），不会顺带把 time embedding 的整体幅度变大/变小；训练更稳、也更利于比较不同 regime 的行为。
2) **可解释性更强**：你可以把 $\tau$ 更纯粹地解释为 half-life/有效记忆长度，而不是“强度+长度”的混合参数。
3) **与其它 embedding 尺度更容易对齐**：在统一初始化下，`time_emb` 的平均量级更稳定，不容易在某些 regime 下被动变成主导项或被淹没。

代价是：当 $\tau$ 过小时，$\mathrm{mean}(w)$ 很小，近端权重会被放大（上面已给出极端例子），因此一般需要配合合理的 `time_tau_min` 或 `tau` 规则一起使用。

---

### 3.4 Regime-Adaptive Factor FiLM（AdaLN）：`RegimeAdaptiveFactorGate`

**原理**  
希望 regime 能“重配因子通道”，并且在 **Pre-LN** 结构下仍然有效。做法是输出 per-factor、per-dim 的 FiLM 参数 `gamma/beta`，并在每个 block 的 `LayerNorm` **之后**应用：

$$y = x_{\text{norm}} \odot \gamma + \beta$$

同时要求 **identity-start**：初始 $\gamma \approx 1$、$\beta \approx 0$，避免扰动主干。

**计算方式**

1) 归一化输入（稳定尺度）：

$$\tilde{r}_b = \mathrm{LN}(r_b),\quad \tilde{e}_n=\mathrm{LN}(e_n)$$

2) 双线性交互（逐维 interaction）：

$$s_{b,n,d} = \frac{(W_\gamma \tilde{r}_b)_d \cdot (\tilde{e}_n)_d}{\sqrt{D}}$$

3) `gamma`（缩放）：

$$\gamma_{b,n,d} = 1 + \text{scale} \cdot \tanh(s_{b,n,d})$$

4) `beta`（平移，可选，默认关）：

$$s'_{b,n,d} = \frac{(W_\beta \tilde{r}_b)_d \cdot (\tilde{e}_n)_d}{\sqrt{D}},\quad \beta_{b,n,d} = \text{shift\_scale} \cdot \tanh(s'_{b,n,d})$$

**生效位置（关键）**
- `gamma/beta` **不是**在 `QuantMoEModel` 里对 `h` 直接乘，而是作为 `factor_film` 传入每个 `RegimeAdaptiveMoEBlock`；
- 在 block 内执行 `x = norm1(x)` 后立即应用 FiLM（确保不被 LN 抵消）。

---

### 3.5 Differentiable Feature Selection（STG-style）：`DifferentiableFeatureSelector`

**原理**  
学习一个全局因子 mask $z \in [0,1]^N$ 来做“软选择”，并在 loss 中加入稀疏正则。为了在 Pre-LN 下有效，mask 在每个 block 的 `LayerNorm` 之后应用（同 3.4）。

**计算方式**

1) 训练时带噪声的 sigmoid gate：

$$z_n = \sigma((\mu_n + \epsilon_n)/\text{temp}),\quad \epsilon_n \sim \mathcal{N}(0,\sigma^2)$$
推理时：

$$z_n = \sigma(\mu_n)$$

2) 稀疏正则（与因子数 N 解耦）：

$$\text{reg\_loss} = \mathbb{E}\left[\sum_n z_n\right]$$

3) 生效方式（post-LN factor mask）  
在 block 内：

$$x \leftarrow x \odot z$$

其中 $z$ broadcast 到 `[1,1,N,1]`。

> 注意：由于主干是 residual，mask 影响的是该层 attention/FFN 的更新项（update），而不是“硬删除”历史信息；这是一种更稳定的 gating 方式。

---

### 3.6 统一初始化（Embedding scale & identity-start）

**原理**  
如果让 `nn.Linear/nn.Embedding` 使用各自默认 init，`val_proj(1->D)` 与 embedding 的尺度可能不一致，导致某一项主导 early training。  
当前实现通过 HF 的 `post_init()` 统一调用 `QuantMoEModel._init_weights`，把所有关键层用同一 `initializer_range` 初始化。

**计算方式（策略）**

- `Linear/Embedding/MultiheadAttention(in_proj)`：`N(0, initializer_range)`  
- `LayerNorm`：weight=1, bias=0  
- FiLM 投影层（`proj_gamma/proj_beta`）标记 `_rstmoe_zero_init=True`，保持全零初始化（identity-start）

配置项：
- `initializer_range`（默认 0.02）

---

## 4. 端到端流程（How data flows）

下面给出一次 forward 的完整流程（embedding/conditioning 部分）：

### 4.1 全局流程（QuantMoEModel.forward）

1) **输入**：`x[B,T,N]`, `factor_ids[N]`, optional `macro_features[B,d_macro]`
2) **Regime embedding**：`r = RegimeContextEncoder(x, macro_features)` → `[B,D]`
3) **Base token**：
   - `v = val_proj(x)` → `[B,T,N,D]`
   - `e = factor_id_emb(factor_ids)` → `[N,D]`
   - `h = v + e`（broadcast）→ `[B,T,N,D]`
4) **Time embedding（可选）**：
   - `(time_emb, tau) = RegimeAdaptiveTimeEmbedding(r, T)` → `[B,T,D]`
   - `h += time_emb`（broadcast to N）
5) **Factor FiLM（可选）**：
   - `(gamma, beta) = RegimeAdaptiveFactorGate(r, e)` → `[B,N,D]`
   - 保存为 `factor_film`，传入每一层 block
6) **Feature selection（可选）**：
   - `(z, reg_loss) = DifferentiableFeatureSelector.sample_mask(...)` → `z[N]`
   - 保存为 `feature_mask=z`，传入每一层 block
7) **Dropout**：`h = emb_dropout(h)`
8) **进入 backbone（n_layers 次）**：每层 `RegimeAdaptiveMoEBlock` 接收 `h, r, factor_film, feature_mask`

### 4.2 单层 block 内（关键：FiLM/mask 的生效位置）

对第 `l` 层：

1) `residual = x`
2) `x = LayerNorm(x)`  （Pre-LN）
3) **FiLM**（若开启）：
   - `x = x * gamma + beta`
4) **Feature mask**（若开启）：
   - `x = x * z`
5) Router 计算 time/factor 两个专家的融合权重
6) 两个专家并行注意力 → 融合 → residual add
7) FFN（Pre-LN）→ residual add

---

## 5. 配置参数速查（Config knobs）

| 目的 | 参数 | 说明 |
|---|---|---|
| 统一初始化尺度 | `initializer_range` | `Linear/Embedding/MHA(in_proj)` 的 std |
| 启用时间 embedding | `use_regime_time_embedding` | 是否启用 3.3 |
| τ 范围/初始化 | `time_tau_min/max/init` | regime-dependent 时间尺度 |
| 时间表初始化 | `time_emb_init_std` | `pos[t]` 的初始化 std |
| 启用因子 FiLM | `use_regime_factor_gate` | 是否启用 3.4 |
| FiLM scale | `factor_gate_scale` | `gamma` 幅度（默认 0.5） |
| FiLM shift | `factor_gate_shift_scale` | `beta` 幅度（默认 0，通常先不开） |
| 启用特征选择 | `use_feature_selection` | 是否启用 3.5 |
| STG 温度/噪声 | `selection_temperature/selection_noise_std` | gate 的硬度与探索 |
| 稀疏强度 | `selection_reg_lambda` | `reg_loss` 的系数 |
| 外部 macro | `use_external_macro`, `d_macro_input` | 是否用 precomputed market state |

---

## 6. 与 Macro PCA 的严格回测约束（防泄漏）

`macro_features` 的 PCA 由 `scripts/precompute_market_state.py` 生成：

- 默认 `--pca_fit_on=train`：**只用 train 日期拟合 PCA**，再对 valid/test 做 transform（避免未来信息）
- sidecar：
  - `*.pca.npz`：mean/components
  - `*.pca.meta.json`：fit split/range/dims（审计用）

> 如果你使用外部 macro，请确保训练/回测使用的是“train-fit PCA”的 state 文件；否则即使不看 label，也会发生 look-ahead。

---

## 7. FiLM + STG：是否只保留 FiLM？

结论先行：
- **如果你的目标是“regime 条件化 + 稳定训练”**：通常 **只保留 FiLM 就足够**（更简洁、更不容易引入训练不稳定）。
- **如果你的目标是“全局稀疏/特征裁剪/可解释的因子子集”**：保留 STG（或在 FiLM 之上叠加 STG）才有明确价值。

### 7.1 机制对比（两者不等价）

| 组件 | 控制粒度 | 作用形式 | 主要收益 | 主要风险 |
|---|---|---|---|---|
| FiLM/AdaLN（`factor_gate`） | per-sample × per-factor × per-dim | `x_norm * gamma + beta` | 真正的条件化（regime→因子表征重配） | 过强会扰动分布（已通过 identity-start 缓解） |
| STG（`feature_selector`） | 全局 per-factor | `x * z`（soft mask）+ 稀疏正则 | 全局筛因子、便于裁剪/解释 | 正则过强会“掐死”学习；与 FiLM 叠加可能过度门控 |

### 7.2 推荐消融矩阵（论文/回测必做）

用两个开关就能覆盖核心消融：
- FiLM：`use_regime_factor_gate`
- STG：`use_feature_selection`

建议跑四组（其余参数保持一致）：
1) `FiLM=off, STG=off`（最小基线）
2) `FiLM=on,  STG=off`（推荐默认：干净的 regime 条件化）
3) `FiLM=off, STG=on`（验证“全局筛因子”是否真的有用）
4) `FiLM=on,  STG=on`（验证二者是否互补；很多情况下会冗余）

### 7.3 仅保留 FiLM 的充分条件（经验法则）

当你观察到以下现象时，STG 往往可以先关掉：
- `factor_gate_entropy` 明显下降、`factor_gate_topk_mass_*` 上升：FiLM 已经在主动“集中注意”到少数因子。
- 你更关心 **regime 下的相对重配**，而不是一个跨全时期的固定因子子集。
- STG 需要调参/热身（warmup）才能不伤性能，而 FiLM 单独训练稳定。

### 7.4 如果保留 STG：最小化副作用的调参建议

- 先用很小的稀疏系数：`selection_reg_lambda=1e-5~1e-4`（你当前的 `reg_loss=z.sum(dim=-1).mean()` 值域是 `[0,N]`）
- 保持温度不要太低：`selection_temperature≈0.1~0.5`（太低会近似 hard mask，早期容易塌）
- 观察 `selected_mask` 的均值/分布（以及下游 IC/RankIC），避免“全关/全开”两种退化解。
