# RST-MoE 当前问题落盘与研究级解决方案蓝图

> Scope: 本文只依据当前实际生效代码，不依据文档设想，不修改代码。  
> Ground truth code: `work_flow.py`, `module/model_adapter.py`, `module/quant_moe_model.py`, `module/architecture/moe_block.py`, `module/architecture/regime_adaptive_embedding.py`, `module/architecture/regime_adaptive_pooling.py`, `module/architecture/regime_encoder.py`, `module/dataloader/sampler.py`.

---

## 0. 真实对象定义

当前代码实际实现出来的对象，不是一个泛化的 “SOTA stock prediction engine”，而是一个：

$$
\text{day-level regime-conditioned separable panel encoder}
$$

更具体地说，对某一天 $d$ 的股票 $i$，模型学习的是：

$$
x_{i,d} \in \mathbb{R}^{T \times N}
$$

$$
s_{i,d} = F\!\left(x_{i,d};\ m_d,\ \widehat{u}_d\right)
$$

其中：

- $m_d$ 是来自 `market_state_*.pkl` 的日级外部状态；
- $\widehat{u}_d$ 是由当日 batch 内股票表征统计得到的 batch summary；
- $F$ 由时间专家、因子专家、FiLM、regime time embedding、regime-aware pooling 共同构成。

当前代码的核心思想是合理的：

$$
F_d \approx \Phi_d \circ \left(\pi_d \mathcal{A}_t + (1-\pi_d)\mathcal{A}_f\right)
$$

但它同时存在一组结构性问题。下面先落盘问题，再给出解决方案。

---

## 1. 问题台账

### P1. 算子族过于可分离，表达上限受限

代码证据：

- `time_expert` 沿时间轴做注意力：`module/architecture/moe_block.py:141-148`
- `factor_expert` 沿因子轴做注意力：`module/architecture/moe_block.py:150-157`
- 融合仅为线性 convex mixing：`module/architecture/moe_block.py:159-161`

单层 block 的本质可以写成：

$$
\mathcal{F}^{(\ell)}(X)
=
X + \pi_d^{(\ell)} \mathcal{A}_t(X) + \left(1-\pi_d^{(\ell)}\right)\mathcal{A}_f(X) + \mathrm{FFN}(X)
$$

它近似的是：

$$
\mathcal{K}_d \approx \pi_d K_t \otimes I_N + (1-\pi_d) I_T \otimes K_f
$$

这意味着模型更像在拟合低 Kronecker-rank 算子，而不是一般的二维 panel operator。  
一般二维算子自由度近似为：

$$
O(T^2 N^2)
$$

而当前显式建模的核心自由度更接近：

$$
O(T^2 + N^2)
$$

后果：

- 它难以直接表达 “过去某个时刻的某个因子如何影响当前另一个因子” 这类非分离交互；
- 多层可以部分补偿，但上限仍然受限；
- 如果真实 alpha 依赖跨时间-跨因子的联合结构，当前模型会系统性欠拟合。

### P2. 路由粒度是 day-global，而不是 stock-specific

代码证据：

- `market_state_path` 加载后强制启用 external macro：`module/model_adapter.py:581-594`
- `RegimeContextEncoder` 在 `use_external=True` 时直接使用 `macro_features`：`module/architecture/regime_encoder.py:180-188`
- `router` 输入来自 `regime_embedding + layer_summary`：`module/architecture/moe_block.py:71-85`

在当前 setting 下，同一天所有股票共享同一个外部 market state，因此近似有：

$$
r_{i,d} \approx r_d
$$

进一步地，同一层 gate 也接近日级共享：

$$
\pi_{i,d}^{(\ell)} \approx \pi_d^{(\ell)}
$$

这使当前 hypothesis 变成：

$$
\text{市场今天更偏 temporal 还是 factor interaction}
$$

而不是：

$$
\text{同一天不同股票是否该走不同结构路径}
$$

后果：

- 对市场级 regime shift 很合理；
- 对横截面异质性表达不足；
- 如果真实最优路由是 $\pi_{i,d}^\ast$，但模型只能学 $\pi_d$，则存在不可约误差：

$$
\varepsilon_{\text{route}}
=
\mathbb{E}\!\left[
\left(\pi_{i,d}^\ast - \pi_d\right)^2 \|h_t-h_f\|^2
\right]
$$

### P3. `layer_summary` 把采样器噪声写进了模型本体

代码证据：

- router summary 来自 batch 内股票均值/方差：`module/architecture/moe_block.py:73-85`
- pooling summary 也来自 batch 内统计：`module/quant_moe_model.py:344-355`
- 训练使用 `FixedDailyBatchSampler`，当日股票会被下采样或上采样：`module/dataloader/sampler.py:45-60`
- 验证/测试使用 `DailyChunkBatchSampler`，同一天可能被切成多个 chunk：`module/dataloader/sampler.py:109-113`

设真实整日 summary 为 $u_d$，batch 估计量为 $\widehat{u}_{d,b}$。当前模型真实看到的是：

$$
s_{i,d} = F\!\left(x_{i,d}; m_d, \widehat{u}_{d,b(i)}\right)
$$

而不是理想中的：

$$
s_{i,d} = F\!\left(x_{i,d}; m_d, u_d\right)
$$

对训练下采样近似，若当日有 $M_d$ 只股票，batch 大小为 $b$，则 batch mean 的方差近似为：

$$
\mathrm{Var}(\widehat{\mu}_d) \approx \frac{1-b/M_d}{b}\Sigma_d
$$

后果：

- router 和 pooling 一部分学到的是 sampling artifact，不是 market structure；
- 同一天不同 chunk 的预测函数不完全一致；
- 当前在 `CSI300 + batch_size=300` 下问题被压小，但没有从原理上解决。

### P4. 一个 regime latent 同时驱动四种机制，存在可辨识性问题

代码证据：

- `regime` 同时进入 time embedding：`module/quant_moe_model.py:258-265`
- 同时进入 factor FiLM：`module/quant_moe_model.py:266-289`
- 同时进入 router：`module/quant_moe_model.py:320-329`
- 同时进入 pooling：`module/quant_moe_model.py:356-362`

记共享 regime latent 为 $r_d$，则总梯度为：

$$
\frac{\partial \mathcal{L}}{\partial r_d}
=
\frac{\partial \mathcal{L}_{\tau}}{\partial r_d}
+
\frac{\partial \mathcal{L}_{\gamma,\beta}}{\partial r_d}
+
\sum_{\ell}\frac{\partial \mathcal{L}_{\pi^{(\ell)}}}{\partial r_d}
+
\frac{\partial \mathcal{L}_{\alpha,q}}{\partial r_d}
$$

后果：

- 同一个 latent 被迫同时承担 “时间尺度”、“因子重标定”、“结构路由”、“读出模板” 四种职责；
- 即使模型有效，也很难回答性能到底来自哪条机制；
- 解释性会被 shared latent 的梯度耦合污染。

### P5. 读出只看最后时间步，形成信息瓶颈

代码证据：

- 最终只取 `h[:, -1, :, :]`：`module/quant_moe_model.py:340-343`
- pooling 仅对最后时刻的因子表示做聚合：`module/quant_moe_model.py:356-365`

当前最终读出等价于：

$$
z_{i,d} = P\!\left(h_{i,d,T,:,:}\right)
$$

若更合理的目标是跨时间读出：

$$
z_{i,d}^{\ast}
=
\sum_{t=1}^{T}\beta_{d,t} P_t\!\left(h_{i,d,t,:,:}\right)
$$

则当前模型要求所有有效历史信息都必须先被搬运到最后一个时间步。  
这会造成：

- 历史信息的压缩损失；
- 模型过度依赖最后一个时间点的表征质量；
- time expert 的一部分收益在读出时被丢掉。

### P6. 训练目标与研究目标不完全一致

代码证据：

- 当前 `main_loss="mse"`：`work_flow.py:157`
- `total_loss` 主体由 `l_main + z_loss + reg_loss` 构成：`module/quant_moe_model.py:444-453`
- `listmle` 和 `ic` 当前只是监控项，不是联合主目标：`module/quant_moe_model.py:395-442`

当前主目标是：

$$
\mathcal{L}
=
\mathcal{L}_{\text{MSE}}
+
\lambda_z \mathcal{L}_z
+
\lambda_s \mathcal{L}_{\text{sparsity}}
$$

而 MSE 可分解为：

$$
\mathrm{MSE}
=
(\mu_p-\mu_y)^2
+
(\sigma_p-\rho \sigma_y)^2
+
\sigma_y^2(1-\rho^2)
$$

其中 $\rho$ 是相关性。  
这说明当前优化并不专注于 ranking / IC，而是同时逼近均值、尺度、相关性。

后果：

- 结构创新的排序收益可能被尺度/均值拟合掩盖；
- 机制改动与最终 RankIC 的因果关系不干净；
- checkpoint 若按训练 loss 选，则偏差更大。

### P7. router 不一定真的学到结构切换

代码证据：

- gate 是 softmax over 2 experts：`module/architecture/moe_block.py:89-107`
- 当前只有 `z-loss`，没有 expert specialization regularizer：`module/architecture/moe_block.py:100-103`

对两专家情形，设：

$$
o = g h_t + (1-g) h_f,\qquad
g = \sigma\!\left(\frac{\Delta}{\tau_r}\right)
$$

则有：

$$
\frac{\partial \mathcal{L}}{\partial \Delta}
=
\frac{1}{\tau_r} g(1-g)
\left\langle
\nabla_o \mathcal{L},
h_t-h_f
\right\rangle
$$

若 $h_t \approx h_f$，则即使 gate 可学习，梯度也会趋弱。  
当前代码没有显式机制强迫两个专家“分工”，因此 router 可能退化为平滑加权器，而不是结构切换器。

### P8. 当前所谓 “cross-sectional modeling” 实际不是 stock-stock interaction

代码证据：

- `factor_expert` 是沿因子维 `N` 做 attention：`module/architecture/moe_block.py:150-157`
- 不存在股票-股票关系模块；`B` 维只用于 batch 统计和 loss 聚合

因此当前更准确的描述是：

$$
\text{temporal interaction} \quad \text{vs} \quad \text{factor interaction}
$$

而不是：

$$
\text{temporal interaction} \quad \text{vs} \quad \text{stock cross-sectional interaction}
$$

后果：

- 如果论文 claim 写成 “cross-sectional stock relation modeling”，代码层面不成立；
- 模型尚未显式建模行业、同主题股票、共同风险暴露等 stock-stock 结构。

### P9. day-global latent 与 sample-wise dropout 之间存在训练/推理语义差异

代码证据：

- `RegimeContextEncoder` 的 encoder 带 dropout：`module/architecture/regime_encoder.py:40-46`
- 当前 external macro 模式下，同一天所有股票输入相同 macro

训练时，因为 dropout mask 对样本逐个采样，所以同一天会出现：

$$
r_{i,d}^{\text{train}} \neq r_{j,d}^{\text{train}}
\quad \text{即使} \quad m_{i,d}=m_{j,d}=m_d
$$

而测试时 dropout 关闭：

$$
r_{i,d}^{\text{test}} = r_d
$$

后果：

- 训练时 regime latent 带入了不必要的 stock-wise noise；
- 这与 “day-level shared regime” 的算法语义不完全一致；
- train-test gap 被人为放大。

---

## 2. 设计原则

基于上述问题，一个研究级方案不应继续 “堆 condition module”，而应遵守以下原则：

1. **先定义对象，再定义网络**  
   模型要先明确是在学习 day-global operator、stock-local operator，还是两者的层级组合。

2. **state 与 sampler 解耦**  
   市场状态必须来自稳定的日级状态场，而不是 batch 抽样噪声。

3. **控制变量分层**  
   时间尺度、路由、FiLM、读出不能全部挤到一个 latent 上。

4. **算子族要足够丰富，但不能失去偏置**  
   不直接跳 full attention，而是从可分离算子扩展到低秩二维算子。

5. **训练目标要与研究目标同向**  
   如果论文讲 ranking，则主目标必须是 ranking-consistent。

6. **架构必须可证伪**  
   每条机制都要有对应的 hard negative 和 failure criterion。

---

## 3. 研究级解决方案蓝图

### 3.1 从单一 shared latent，升级到层级状态场

当前阶段最优的设计，不是直接上 variational latent、离散 regime，或 dynamic weights，  
而是先把当前单一 `regime_embedding` 拆成一个 **deterministic Global-Local Hyper-State**：

$$
z_d^{g} = G(m_d, u_d)
$$

$$
z_{i,d}^{\ell} = L(x_{i,d}, z_d^{g})
$$

其中：

- $z_d^{g}$ 是日级全局状态；
- $z_{i,d}^{\ell}$ 是个股局部状态；
- $m_d$ 是外部 market state；
- $u_d$ 是稳定的日级内部 summary，而不是当前训练 batch 的即时统计；
- 全局状态负责 slow-varying market regime；
- 局部状态负责 stock-specific residual adaptation。

关键不是 “多一个 latent”，而是 **不同控制头必须吃不同的状态组合**。  
推荐的职责分配是：

$$
\tau_d = f_{\tau}(z_d^{g})
$$

$$
\left(\gamma_{i,d}, \beta_{i,d}\right)
=
f_{\mathrm{FiLM}}(z_d^{g}, z_{i,d}^{\ell})
$$

$$
\pi_{i,d}^{(\ell)}
=
f_{\pi}^{(\ell)}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

$$
\left(q_{i,d}, \alpha_{i,d}\right)
=
f_{\mathrm{pool}}(z_d^{g}, z_{i,d}^{\ell}, u_d)
$$

这里的设计原则是：

- `time_tau` 只吃 global state，因为时间记忆尺度首先是 market-level property；
- `router` 必须吃 global + local + day summary，因为路由既受市场环境影响，也必须表达同一天不同股票的结构异质性；
- `FiLM` 应吃 global + local，因为因子解释既受 regime 影响，也受 stock-local pattern 影响；
- `pooling` 应吃 global + local + day summary，因为最终读出规则既依赖市场环境，也依赖个股局部结构。

这一步直接解决：

- P2: routing 粒度过粗
- P4: shared latent 过载
- P3: state 与 sampler 纠缠

同时它也是后续三个方向的统一底座：

- 向上可扩展到 variational hierarchical latent field；
- 向外可扩展到 discrete regime head；
- 向深可扩展到 shared global dynamics / dynamic parameterization。

### 3.2 从可分离双轴算子，升级到低秩二维算子族

当前 block 等价于两类算子的 convex mixing。  
研究级版本应改为：

$$
\mathcal{K}_{i,d}
=
\sum_{r=1}^{R_t} a_{d,r}^{t}\, T_r
+
\sum_{s=1}^{R_f} a_{d,s}^{f}\, F_s
+
\sum_{u=1}^{R_{tf}} a_{i,d,u}^{tf}\, C_u
$$

其中：

- $T_r$ 是 temporal expert bank；
- $F_s$ 是 factor expert bank；
- $C_u$ 是少量 cross time-factor low-rank experts。

进一步可把 $C_u$ 写成低秩 Kronecker 形式：

$$
C_u \approx A_u \otimes B_u
$$

这样既保留强归纳偏置，又补足当前 “只能分离、不能交叉” 的硬缺陷。  
这一步直接解决 P1。

### 3.3 从 batch summary，升级到 stable market state field

当前 `layer_summary` 使用 batch 内统计，导致模型函数依赖采样器。  
研究级方案必须把日级 summary 变成外部稳定状态：

$$
u_d = H(\mathcal{X}_d)
$$

其中 $\mathcal{X}_d$ 是当天完整市场截面，而不是一个随机 batch。  
在设计上应明确：

- `market_state.pkl` 提供全局 slow state；
- `u_d` 提供全日 representation summary；
- forward 过程中不再从当前 batch 即时重算市场统计。

进一步地，在先完成 deterministic state factorization 之后，可让模型学习 state uncertainty：

$$
q(z_d^{g} \mid m_d, u_d)
$$

从而区分 “市场状态不确定” 与 “股票信号不确定”。  
这一步解决 P3，并把市场状态从 sampler artifact 中剥离出来。

### 3.4 从 last-step readout，升级到 temporal evidence accumulation

当前只使用最后一个时间步读出。  
更合理的读出应该是：

$$
z_{i,d}
=
\sum_{t=1}^{T} \beta_{i,d,t}\, P_t(h_{i,d,t,:,:})
$$

其中：

$$
\beta_{i,d,t} = \operatorname{softmax}(b(z_d^{g}, z_{i,d}^{\ell}, t))
$$

这意味着模型可以：

- 在动量 regime 下偏向近期时点；
- 在均值回复 regime 下偏向更长时间跨度；
- 在波动冲击 regime 下自适应降低最后时点的偶然噪声权重。

这一步直接解决 P5，并提升 time expert 的有效利用率。

### 3.5 从 shared-control 到 disentangled-control

要让机制解释成立，必须把不同控制头分拆成不同子空间，而不是共用一套 latent。

推荐的结构是：

$$
z_d^{g}
\rightarrow
\left\{
z_d^{\tau},
z_d^{\gamma},
z_d^{\pi},
z_d^{\alpha}
\right\}
$$

再分别驱动：

- time scale head
- FiLM head
- router head
- pooling head

同时加入可辨识性约束，例如：

$$
\mathcal{L}_{\text{dis}}
=
\sum_{a \neq b}
\left\|
\operatorname{Cov}(z^a, z^b)
\right\|_F^2
$$

或者 expert specialization regularizer：

$$
\mathcal{L}_{\text{spec}}
=
\sum_{\ell}
\operatorname{cos}^2\!\left(\bar{h}_t^{(\ell)}, \bar{h}_f^{(\ell)}\right)
$$

目标不是漂亮，而是可证伪、可解释、可归因。

### 3.6 从 MSE 主导，升级到 ranking-consistent objective stack

研究叙事若围绕 RankIC / ranking alpha，则目标函数应改写为：

$$
\mathcal{L}
=
\lambda_{\text{rank}} \mathcal{L}_{\text{rank}}
+
\lambda_{\text{ic}} \mathcal{L}_{\text{IC}}
+
\lambda_{\text{cal}} \mathcal{L}_{\text{cal}}
+
\lambda_{\text{reg}} \mathcal{L}_{\text{reg}}
+
\lambda_{\text{spec}} \mathcal{L}_{\text{spec}}
$$

其中：

- $\mathcal{L}_{\text{rank}}$ 是 listwise / pairwise ranking objective；
- $\mathcal{L}_{\text{IC}}$ 直接推动截面相关性；
- $\mathcal{L}_{\text{cal}}$ 只保留少量校准项，而不是让 MSE 完全主导；
- $\mathcal{L}_{\text{spec}}$ 用于维持专家分工。

其核心思想是：

$$
\text{先对齐研究目标，再允许必要的校准项存在}
$$

这一步主要解决 P6 与 P7。

### 3.7 真正引入 stock-stock relational operator

如果长期目标是 “stock prediction research platform”，则必须引入真正的股票关系建模模块，而不是只在因子维做交互。  
推荐的对象是：

$$
\mathcal{G}_d = (\mathcal{V}_d, \mathcal{E}_d)
$$

其中边可以来自：

- 行业层级
- 风格暴露相近性
- 收益协方差 / residual correlation
- 同主题或事件暴露

然后增加 relational expert：

$$
\mathcal{A}_{\text{stock}}(H_d, \mathcal{G}_d)
$$

使整体算子升级为：

$$
\mathcal{K}_{i,d}
=
\sum a^t T
+
\sum a^f F
+
\sum a^{tf} C
+
\sum a^{s} S
$$

这一步不是为了堆模块，而是为了让 “cross-sectional” 这个术语在代码层面成立。  
它主要解决 P8。

---

## 4. 研究路线图

### Phase A. 研究清洗层

目标：在不追求更强模型前，先让对象定义干净。

- 固定真实研究对象：day-global state + stock-local residual
- 拆分 shared latent 的职责
- 去掉 batch summary 对 forward 的核心依赖
- 用 ranking-consistent objective 重写训练目标

这一步的成功标准不是绝对收益，而是：

$$
\text{机制可辨识性} \uparrow,\qquad
\text{实验可复现性} \uparrow
$$

### Phase B. 算子扩展层

目标：补齐当前 separable operator 的表达瓶颈。

- 在 time / factor 双专家之外，引入少量低秩 cross experts
- 把 temporal readout 从 last-step 扩展到 full-window
- 增加 expert specialization 约束

成功标准：

$$
\text{Full operator family} > \text{separable-only family}
$$

且提升在跨 regime bucket 上稳定存在。

### Phase C. 真正的 panel intelligence 层

目标：让模型具备市场状态、股票关系、个股局部状态三层智能。

- global market field
- stock-local controller
- stock-stock relational expert

成功标准：

- 不仅整体 RankIC 提升；
- 在高拥挤、高共振 regime 下尤其提升；
- 路由和读出行为与市场状态变化方向一致。

### Phase D. 长期 research platform 层

目标：把模型创新、状态资产、协议、证据生成彻底解耦。

- State assets 独立版本化
- Operator family 独立实验化
- Objectives 独立组合化
- Evidence generation 自动化

此时平台就不再是 “一个模型脚本”，而是：

$$
\text{market state lab}
\times
\text{panel operator lab}
\times
\text{evidence engine}
$$

---

## 5. 最终判断

当前代码最值得肯定的，不是已经做成了 SOTA，而是它已经碰到了一个真正重要的问题：

$$
\text{non-stationarity should change the computation path, not only the input}
$$

这是正确的方向。

但当前版本的主要短板也很明确：

1. 算子族还太分离；
2. 路由仍是 day-global；
3. 市场 summary 受采样器污染；
4. 一个 latent 负责过多机制；
5. 读出只看最后时刻；
6. 训练目标与研究目标不完全同向；
7. `cross-sectional` 叙事在代码层面还不充分成立。

因此，真正的高水平方案，不是继续堆新 trick，而是完成一次对象重定义：

$$
\text{从 “带 regime 的 MoE” }
\rightarrow
\text{“层级状态场驱动的 panel neural operator”}
$$

这条路线更难，但它才是长期工程能够走向一流研究的方向。
