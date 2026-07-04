# JEPA 式 Latent 预测预训练 — 完整设计与分析

**Status**: 设计定稿（design-complete）。**尚未立卡** — 立卡时机：`h-20260610-001/002` 出 n=3 结论、且 weight-EMA 基建（gen-gap 线 #1）落地之后（target encoder 与 weight-EMA 共享同一份 EMA 代码）。
**Date**: 2026-06-10（整合自当日算法实验规划会话的四轮分析）
**Scope**: RST-MoE（commit `3675090` 谱系）/ Qlib Alpha158；下游监督任务 = csi300 t+5 ranking；预训练语料 = csi800（无 label）。
**Related**:
- 卡片：`experiments.md` 的 `h-20260610-001`（DropExtremeLabel）、`-002`（股票轴第三 expert）、`-003`（多 horizon 辅助头，deferred，若本设计立项则并入其 Stage-2）
- Probe 基建来源：`scripts/pool_readout_forensics.py`（frozen backbone + ridge head 的加载与评估路径）
- 前置发现（memory `project_rst_moe`）：gen-gap=0.12 capacity/sampler 不敏感且 finetune-hurts-valid；readout 双轴 settled；pool 158 因子等弱冗余；模型 time-dominated

---

## 0. TL;DR

**做什么**：用现有 QuantMoEModel backbone 作 encoder，在 csi800×2008–2020-03 的**无 label**数据上预训练：从过去 8 天窗口的表示 `z_ctx`，预测未来 k∈{1,5,10} 天**表示变化的截面相对值** `d_k`（在 latent 空间预测，不重建输入，不预测 label）。然后两级评估：frozen ridge probe（分钟级生死关）→ SSL-init 监督微调（paired n=3→n=6）。

**为什么**：gen-gap=0.12 的本质是监督饥饿（每股每天 1 个噪声标量），模型靠记忆 2008–2020 时代特异结构降 train loss。JEPA 把每样本监督提升为 64 维结构预测 ×3 horizon，且目标不含任何 label——学到的先验在构造上无法过拟合 valid 窗口的 label 实现。

**关键设计决策一览**：

| 决策 | 选择 | 一句话理由 | 详见 |
|---|---|---|---|
| 预测空间 | latent（JEPA），非重建/非对比 | 收益 ~99% 方差不可预测，重建式给噪声建模；对比式负样本会推开同业结构 | §1 |
| 身份捷径 | **predict-the-delta** + 截面 CS-zscore | 直接预测 z_{t+k} 会学成行业分类器；差分消静态身份，CS 消市场共模 | §3.2 |
| 防塌缩 | EMA target + stop-grad + predictor 不对称（主）+ VICReg variance hinge（兜底 λ=0.01） | 本项目有 collapse 前科（gate six-nines、τ），双保险 | §5.4 |
| loss | SmoothL1(β=1)，目标逐维单位方差 | 事件日 Δz 厚尾，L2 会被单票垄断梯度；cosine 丢模长（事件强度） | §5.3 |
| 头号监控 | **R²_latent = 1 − MSE**（目标单位方差 ⇒ 闭式基线） | "未来相对状态变化被解释的比例"，免费可读 | §5.6 |
| 保险丝臂 | pinned-dims（target 拼上已实现收益/波动的 CSZ 值） | 防 JEPA"丢弃不可预测成分"的自由把 alpha 一起丢了 | §3.4 |
| 评估 | Stage-1 frozen probe 双 kill 线 → Stage-2 init 微调 | "finetune-hurts" 是 converged 模型的结论，init 不在其内；两臂都跑 | §7 |

**预算**：实现 2–3 天；预训练 12–20 GPU-hr（1–2 通宵 @4070S）；probe 分钟级；Stage-2 ≈4 GPU-hr。
**诚实预期**：Stage-1 过线（probe ≥0.070）主观概率 ~25–35%；但失败的副产品是"该数据中无监督可学动力学结构的上限"——与 pool forensics 同级的结论性知识。

---

## 1. 动机与思想：为什么是 JEPA

### 1.1 出发点：学习 = 学会预测，但"预测什么"决定一切

LeCun 世界模型论纲（2022 position paper）的核心命题：原始信号里绝大部分内容不可预测，强迫模型预测全部细节 = 把容量浪费在噪声上。三大自监督范式对应三种处理：

**① 生成式/重建式（MAE、自回归）** — 在输入空间预测 y 本身，必须为 y 的每个细节分配概率质量。
> 对金融是灾难：日频收益可解释方差 ~0.6%（本项目 IC 0.078 ⇒ R²≈0.006）。重建 loss 会**奖励** encoder 忠实编码 idiosyncratic 噪声。

**② 对比式（InfoNCE/SimCLR）** — 拉近正对、推开负对，回避建模噪声，但负样本语义由你定义。
> 本项目的 batch = 同日 300 只股票，天然负样本是"同日其他股票"——InfoNCE 会把**同行业相似股票**在表示空间推开，而同业结构恰是我们想要的信息（002 卡注入的就是它）。负样本设计与研究目标正面冲突。

**③ JEPA** — 在**表示空间**预测，能量函数视角：

```
E(x, y) = ‖ g_φ( f_θ(x), z ) − f_ξ(y) ‖²
            ↑predictor  ↑context     ↑target encoder
```

关键性质在 f_ξ 侧：**target encoder 自己决定 y 的哪些方面值得被表示**。y 中不可预测的成分若被编码，能量压不下去，梯度自然把 f_ξ 推向"只保留可预测的抽象"。**模型自己选择抽象层级，把预测不了的丢掉**——金融数据是这个性质收益最大的场景。

### 1.2 三个已有 settled 发现如何"选择"了这个配方

| settled 发现 | 对配方的指向 |
|---|---|
| 收益信噪比 ~0.6%（IC 0.078） | 排除重建式；latent 预测的"丢弃噪声"自由是刚需 |
| 同业/截面结构是目标信息（002 卡、time-dominated 发现的反面） | 排除对比式（负样本破坏同业几何） |
| finetune-hurts-valid（time-readout 卡的副结论） | frozen-encoder + linear probe 恰是 JEPA 原生评估法；probe 基建已被 pool forensics 写好 |

### 1.3 塌缩问题（免费午餐的代价）

能量有平凡解：f_ξ(·)≡常数 ⇒ 能量处处为零。三种破法：对比式（负样本，代价见上）、正则式（VICReg：逐维方差下限 + 维度去相关）、**架构式（BYOL/I-JEPA 路线）**：target encoder 不接收梯度（stop-grad）+ 是 online encoder 的 EMA 慢拷贝 + 只有 online 侧带 predictor 的不对称。理论解释（Tian et al. 2021）：EMA+predictor 构成隐式双时间尺度动力系统，不动点非塌缩。
**本设计取架构式为主 + VICReg variance hinge 小系数兜底**（collapse 前科项目值得双保险，§5.4）。

### 1.4 I-JEPA 配方拆解（理解通用结构用）

I-JEPA（图像，2023）：context block → context encoder；多个 target block → EMA target encoder 出 latent；predictor 拿 context 表示 + **位置 token（条件变量 z）**预测对应 latent；smooth-L1 对齐。可迁移的三个要点：
1. target encoder 看**完整**信息、context encoder 看**部分**信息——信息不对称制造预测难度；
2. **条件变量 z 是配方的自由度**：图像里是空间位置，视频里是时空位置，**本设计换成 horizon**；
3. **predictor 故意浅** ⇒ 可预测结构被迫压进 encoder 的表示，而非藏在 predictor 容量里。probe 冻结的是 encoder，predictor 用完即弃。

---

## 2. 到底学的是什么（最优解分析）

### 2.1 predictor 学条件期望，encoder 学条件充分统计量

固定目标看 predictor：L2 意义下最优解是**条件期望**

```
g*(z_ctx, e_k) = E[ d_k | z_ctx ]，   残余 loss = E[ Var(d_k | z_ctx) ]
```

再看 encoder 的梯度（方法的灵魂）：z_ctx 被训练去**最小化这个条件方差**，即 f_θ 的有效目标是——

> 把过去 8 天窗口里"对未来相对状态变化有预测力"的信息无损搬进 64 维 z；对预测无贡献的输入成分**得不到任何保留它的梯度**（在 dropout/容量限制下自然被挤出）。

术语化：z_ctx 被塑形为 d_k 的**条件充分统计量**；高斯近似下等价于最大化 I(z_ctx; d_k)。
**这是"学的是什么"的第一层答案：学的不是某个预测值，而是"哪些历史特征携带前瞻信息"这个判别本身。**

**"无损"的精确含义（重要，防过度解读；2026-06-10 设计评审修订）**：充分性是**相对 x_ctx 所含信息**而言的，不是相对 d_k 本身。正交分解（tower property）：

```
E‖d_k − g(z_ctx)‖²  =  E‖E[d_k|x_ctx] − g(z_ctx)‖²   +   E[ Var(d_k | x_ctx) ]
                        ↑ 可消减项（表示+predictor 逼近误差）   ↑ 不可消减底（horizon 内事件）
```

- 第二项是任何时点-a 模型的信息论下界：(a, a+k] 内的新闻/资金流不在 x_ctx 里，**信息物理上不在那**。I(z_ctx; d_k) ≤ I(x_ctx; d_k) 永远成立（encoder 不创造信息，只决定保留什么）。
- 训练唯一能压的是第一项；"无损" = 把它压到 0，即 **E[d_k|z_ctx] = E[d_k|x_ctx]**（均值充分性）——z 保留 x_ctx **关于 d_k** 的全部信息，不是保留 x_ctx 的全部信息，更不是预测 d_k 本身。
- 监督任务面对同一上限（t 时点预测 t+5 收益，horizon 内实现只能留作噪声）。**本设计不承诺突破 I(x_ctx; future)，承诺在该上限内找到比监督训练更好的条件期望参数化（少记忆 label 噪声）**——这正是 §4.6 预期 R²_latent 仅 0.05–0.15、收益子空间仅 ~0.006 的原因：第二项占绝对主导的世界就长这样。
- **64 维容量不是瓶颈**：要装的是条件期望映射的像，I(x_ctx; d_k) 每样本仅 bits 量级（per-dim R²≈0.1 的高斯粗算 ≈7 bits/样本）；且监督模型本就通过同一个 64 维 pooled 瓶颈打分（pool forensics：slot-IC max ≈ 全模型 IC，readout 处不缺容量）。约束是数据里的信息量，不是容器。

### 2.2 bootstrap 不动点 = "找一个让世界最可预测的坐标系"

f_ξ 是 f_θ 的 EMA，"未来状态"的定义本身被慢慢重写——系统是自举的（与 TD-learning 同构）：表示既要**可被过去预测**，又要**作为目标有信息量**。收敛点：一个使 z_t → z_{t+k} 拥有最大可预测成分的表示空间。动力系统类比 = Koopman 算子：找一组"可观测量"（latent 维）使演化在其上近似封闭，predictor 是学出的演化算子。
**与监督学习的根本区别**：监督把表示塑形成"对一个外加标量的回归特征"；JEPA 塑形的是"动力学最规整的状态变量"——学的是"状态"这个概念的定义。

### 2.3 微观实例（一只股票一天）

锚定日 a，股票 i 处于"放量突破"态（高换手 + 突破类因子激活），当日大盘普涨：
1. f_ξ 编码 a 与 a+5 两个窗口作差得 δ_5：含 (i) 大盘 5 天共同漂移 + (ii) i 的"突破→延续/衰竭"状态迁移；
2. CS-zscore 把 (i) 扣进 μ；剩下 d_5 = "i 的状态迁移相对同日市场超出多少"；
3. 若"放量突破态"的后续迁移历史上有规律，predictor 部分命中 d_5 → 条件方差下降 → **encoder 收到的梯度 = 把"突破态的可延续性特征"写进 z_ctx**；
4. 若 i 当天状态是纯噪声，E[d_5|z_ctx]≈0，predictor 输出 ≈0，该样本对 encoder 无塑形力——**噪声被目标函数自动忽略，而非被强行拟合**。与监督 MSE 的根本差别在此：监督 loss 强迫对每个噪声 label 回归，gap=0.12 的记忆行为由此而来。

### 2.4 与 gen-gap 的关系（为什么这是第三种攻击面）

监督训练的假设空间里，"记住 2008–2020 每个时代的 label 噪声实现"是 train loss 的合法下降方向（= 0.12 gap 的来源）。JEPA 的 64 维目标相当于**数据依赖正则**：z 必须落在"动力学可预测流形"上，能记忆 label 噪声的参数子空间在预训练中得不到支持。Stage-2 从该流形出发 = 假设空间预先收缩到"至少描述了真实市场动力学"的区域。
对比另两个 gen-gap lever：weight-EMA 改**解的几何**（flat minima），DropExtremeLabel 改**监督信号的纯度**，JEPA 改**学什么**——三者正交可叠加。

---

## 3. 本问题的建模（x / y / z 怎么选）

### 3.1 基本映射

| JEPA 元素 | 本项目实例 |
|---|---|
| 样本单元 | (股票 i, 锚定日 a) |
| context x | 因子窗口 `x[a−7..a]` ∈ R^{8×158} —— 与现有 backbone 输入**一字不差** |
| context encoder f_θ | **现有 QuantMoEModel 原封不动**（tokenizer → FiLM/time-emb → 2×MoE → final_norm → factor_pooling）→ pooled z ∈ R^64 |
| target y | 同股票未来窗口 `x[a+k−7..a+k]`，k∈{1,5,10} |
| target encoder f_ξ | f_θ 的 EMA 拷贝，stop-grad |
| 条件 z | horizon embedding `e_k`（`nn.Embedding(3, d)`） |
| predictor g_φ | MLP([z_ctx ; e_k]) → R^64，两层 64→128→64 |

工程红利：f_θ 就是 QuantMoEModel 本体 ⇒ 预训练权重直接 load 进 `QlibQuantMoE`，下游零适配；EMA 机制与 weight-EMA（gen-gap #1）共享实现。

### 3.2 关键决策：身份捷径与消去手术（金融 JEPA ≠ 图像 JEPA 的核心点）

把表示分解：`z_{i,t} ≈ s_i + m_t + r_{i,t}`（静态身份 + 市场共同项 + 个股动态）。

**失败模式**：直接预测 z_{i,a+k} 时，最可预测成分是 s_i（行业/市值/波动风格）——encoder 只要编码"这是茅台"，predictor 就能完美预测"5 天后还是茅台"，loss 很低、表示零 alpha。图像 I-JEPA 靠 masking 阻断捷径；**时序股票数据 mask 不掉身份**（任何 8 天窗口都能恢复它），必须在目标构造里做手术：

```
① 差分      δ = z_{i,a+k} − z_{i,a} = (m_{a+k} − m_a) + (r_{i,a+k} − r_{i,a})   # s_i 精确消失
② 截面去均值 δ − mean_batch(δ)                                                  # 共同漂移 (m_{a+k}−m_a) 消失
③ 逐维标准化 ÷ std_batch(δ)                                                     # 尺度稳定（含目标侧防塌缩，§5.2）
```

最终目标语义：**"该股未来 k 日的表示变化，相对同日全市场的表示变化，超出/落后多少"——alpha 问题的无标签同构体。**

### 3.3 机械重叠论证：label 就藏在 target 里（本设计最重要的"不是愿望是机制"论据）

target 窗口 `x[a+k−7..a+k]` 的因子是价格/量的函数、回看最长 60 天。ROC 类因子在 a+k 日的取值，分子分母**横跨锚定日 a**——机械地编码了 (a, a+k] 的已实现收益。于是：

```
监督 label  = CSZ(t+5 收益)
JEPA target = CSZ(Δ表示)，其输入机械地含该收益
⇒ label ≈ JEPA target 的一个（被 encoder 加工的）低维投影
```

**所以这是"带 64 维自动发现辅助任务的超多任务预训练，其中一个子任务近似就是监督任务本身"**——迁移不靠运气。

### 3.4 对应风险（可丢弃性）与保险丝（pinned-dims，预登记 A/B 臂）

JEPA 的定义性自由——丢弃不可预测成分——恰可能丢掉 alpha：收益方向是 d_k 中**最难预测**的成分（R²~0.006），而波动/流动性状态的可预测性高一个量级；encoder 理性分配容量时会偏向后者（= "学成 regime nowcasting"失败模式的精确机制）。两道防线：

1. **64 维冗余**：alpha 子空间只需"存在于 z 中"，probe 的 ridge 能挑出来，不要求主导；
2. **pinned-dims 臂**：target 扩为 `d̃_k = [d_k ; CSZ(已实现收益_a→a+k) ; CSZ(已实现波动)]`，predictor 输出 66 维。被钉死的两维是**从原始数据算的固定目标，encoder 无法通过重定义表示来丢弃**——其梯度穿过 predictor 直灌 z_ctx，强制保留收益预测信息。合法性：已实现收益全在 train 段内，与监督训练用 label 同级合法。
   谱系视角：纯监督（现状，gap=0.12）←— pinned-JEPA —→ 纯 JEPA；pinning 即滑杆。**A/B 两臂（latent-only vs pinned）各一次预训练，都过 probe。**

### 3.5 消去手术后还剩什么可预测结构（predictor 的"猎物"清单）

| 结构 | 时间常数 | 进入 z 的形式 |
|---|---|---|
| 短期反转的延续/衰减 | ~1–3d | "超买/超卖态" + 衰减率 |
| 截面动量的持续 | ~5–20d | "相对强势态" |
| 波动聚集（GARCH 型） | ~5–10d | 高波动态 ⇒ 预测 ‖d_k‖ 大 |
| 因子极值的均值回复 | 因子各异 | "偏离截面均值的程度" |
| 流动性/换手状态转移 | ~5d | 放量态的衰减轨迹 |

每条都 label-free 可学、且与"什么股票未来 5 天相对走强"相交——**交集大小 = Stage-1 probe 要测的量**。

### 3.6 多 horizon 的角色

三个 e_k 共享 predictor 主干 ⇒ 强迫 z_ctx 同时携带"明天/一周/两周"的可预测信息 = return 期限结构的表示版。与 deferred 卡 `h-20260610-003`（多 horizon 辅助头）思想同源——**若本设计立项，003 并入其 Stage-2 微调臂**，不单独做。

### 3.7 信息集纪律：horizon 内的数据只进 target 侧，绝不进 prediction 侧

horizon (a, a+k] 的数据**确实被使用**——f_ξ 的 target 窗口 `[a+k−7..a+k]` 横跨它——但只用于**定义要预测什么**。这条纪律是机制成立的前提：

- 若 predictor（或 z_ctx）看到 horizon 内数据，任务从 forecasting 退化为 **nowcasting**：predictor 直接读出已实现变化，loss 不需要 z_ctx 携带前瞻信息即可归零 ⇒ encoder 梯度消失 ⇒ 机制瓦解。**信息不对称本身就是训练信号**——I-JEPA 的 context 看残缺图、target 看全图是同一条纪律，这里"时间之箭"替代了 masking。
- 同理拒绝自回归 rollout / teacher-forcing 变体（逐步预测 z_{a+1} 并喂回）：训练期喂真中间 latent = horizon 数据泄入条件集（exposure bias，train/inference 错配）；且本设计要的是**特征**不是模拟器——直接 k 步跳跃预测（V-JEPA 同款）更简单且对齐目的。
- 下游口径一致：真实打分发生在 t 时点、只有 ≤t 数据。预训练的信息集纪律 = 部署的信息集纪律。

---

## 4. Loss 完整建模

### 4.1 全式（batch = 锚定日 a 的截面，m≈300，D=64，K=3）

```
                 1
L_total  =  ─────────  Σ_k Σ_i  SmoothL1_β=1( ẑ_k^i − d_k^i )      ← L_pred（主项）
              K·m·D
          +  λ_v · (1/D) Σ_d  relu( γ − std_i(z_ctx[i,d]) )         ← L_var（方差铰链兜底，λ_v=0.01, γ=1.0）
          [ + λ_c · (1/D²) Σ_{d≠d'} Corr(z_ctx)_{dd'}²  ]           ← L_cov（去相关，默认关，塌缩迹象时再开）
```

target 构造（全程 no_grad）：

```
δ_k^i  = f_ξ(x_i[k:k+8]) − f_ξ(x_i[0:8])          # ① 差分：消静态身份 s_i
μ_k,d  = mean_i(δ_k^i[d])                          # ② 截面均值（逐维）
σ_k,d  = std_i(δ_k^i[d]).clamp_min(1e-4)           # ③ 截面标准差（逐维，clamp 防除零）
d_k^i  = (δ_k^i − μ_k) / σ_k                       # ④ 最终目标：逐维单位方差
```

### 4.2 每步在防什么

- **① 差分**：没有它，loss 最低成本解 = 把 z 变成行业/市值分类器（§3.2）。
- **② 去均值**：消 (m_{a+k}−m_a)。否则 predictor 梯度大头花在"预测大盘"——最不可预测、且对 RankIC 零贡献（截面排序平移不变）的成分。
- **③④ 标准化的三重身份**：
  (a) **语义对齐**——与 label 管道 CSZScoreNorm 同构，目标几何 = RankIC 的截面相对几何；
  (b) **尺度稳定**——latent 各维天然方差差数个量级且训练中漂移，防少数维垄断梯度；
  (c) **目标侧防塌缩**——若 f_ξ 开始塌（δ→0），标准化把 d 重新吹回单位方差，**目标永不退化为常数**；塌缩只表现为"目标变纯噪声、loss 顶在不可预测基线"（给出闭式监控量，§4.6）。

### 4.3 为什么 SmoothL1（vs MSE / cosine）

- vs MSE：即使 zscore 后，事件日（涨停/公告）的 δ 仍厚尾——L2 让单票垄断整天梯度。SmoothL1 = |e|<1 时 L2（高效）、>1 时 L1（稳健），与对收益用 Huber 同理；β=1 匹配单位方差目标。注意 `clip_outlier=True` 只截了**输入**尾部，状态迁移造成的 **Δz 厚尾仍在**。
- vs cosine（BYOL 用）：cosine 丢模长，但 ‖d_k‖ 有语义 = "状态相对同行移动了多少"（事件强度，波动预测的载体），保留。
- 诊断侧另记一份 **MSE（不进梯度）**：目标逐维单位方差 ⇒ "全预测 0" 的 MSE 恰为 1.0 ⇒ `R²_latent = 1 − MSE` 是免费的解释方差读数。

### 4.4 梯度流与防塌缩动力学

```
            梯度 ✓                      梯度 ✗ (stop-grad)
 x_ctx ──► f_θ ──► z_ctx ──► g_φ ──► ẑ_k ──┐
                                            ├──► SmoothL1
 x_tgt ──► f_ξ ──────────► d_k ────────────┘
            ▲
            └── 每 step EMA: ξ ← m·ξ + (1−m)·θ    (m: 0.996 → 1.0 cosine ramp)
```

- **g_φ（快变量）**：收敛到条件期望/中位数——学出的"演化算子"，**用完即弃**。实操：predictor lr = encoder 的 **10×**（双时间尺度让"predictor 近似最优"成立，encoder 的梯度才是干净的"降条件方差"方向）。
- **f_θ（慢变量）**：唯一梯度来源 = 让 z_ctx 携带更多对 d_k 的预测信息。
- **f_ξ（无梯度）**：若目标侧也通梯度，loss 最快下降方向是双方合谋塌缩（把 d 变常数）。stop-grad 剪断"把目标改简单"这条路；EMA 让目标成为慢速跟随者——不冻死（表示能进化）也追不上塌缩方向。**m ramp 理由**：早期 0.996 目标新鲜跟得上进化，后期 →1 目标冻结利于收敛。
- **L_var 铰链**：对 EMA+stop-grad 的输入侧兜底（防 z_ctx 自身维度死亡）。**系数刻意小（0.01）**——保险丝不是主机制，调大会扭曲表示几何。

### 4.5 归一化职责划分（FAQ：已有 RobustZScoreNorm，还要 CS-zscore 吗？→ 要）

现有管道本身就是两轴归一化的范本：`RobustZScoreNorm` 在 **infer_processors**（feature，全段 DK_I，`work_flow.py:85-91`），`CSZScoreNorm` 在 **learn_processors**（label，仅 train DK_L）。

| | 作用对象 | 统计量轴 | 目的 |
|---|---|---|---|
| RobustZScoreNorm（现有，不动） | 输入 feature | **全历史**（2008→2020-03 拟合一次，常数） | 158 因子量纲可比、输入良态 |
| CSZScoreNorm（现有，不动） | label | **每日截面**（逐日重算） | 监督目标 = 当日相对排序几何 |
| JEPA target CS-zscore（新增） | latent Δz（d_k） | **每日截面**（逐 batch 重算） | 同上——**d_k 就是 pretext 任务的 label** |

**对称律：JEPA 里 d_k 扮演 label，享受 label 待遇（CS）；喂 f_θ/f_ξ 的特征继续吃 feature 待遇（全局），零重叠零重复。**

feature 侧全局 z-score 替代不了 target 侧截面 z-score 的三个理由：
1. **市场共模消不掉**：RobustZScoreNorm 是 2008–2020 拟合的常数，按构造无法移除"某一天特有"的共同移动；大盘 ±2% 日全体价格衍生因子态一起平移，编码后是 Δz 的第一主方向。
2. **latent 尺度不受输入归一化保护**：网络不传递归一性（内部 LayerNorm 是对 D 维 per-token，非截面）；且 **Δz 尺度随训练阶段剧烈漂移**（早期权重快动 Δz 大、收敛后小），只有 per-day per-dim σ 能自适应跟住。
3. **逐维梯度均衡**：64 维天然方差差数量级。

⚠️ **反方向禁止**：不要给 **feature** 加 CS 归一化——特征里的截面共同结构（当日市场状态）是 `RegimeContextEncoder` 和 `layer_summary` 明确消费的输入信息，CS 掉它 = 拆掉架构的 regime 感知前提。CS 操作只属于 target 侧。

其他相关：预训练不读 label（learn_processors 留着无害）；probe/微调阶段 label 管道原样；pinned 臂的已实现收益目标用同款 CSZ 口径与监督 label 一致；batch=300 估计截面统计噪声 ~1/√300≈6% 可接受，薄日上采样重复股票对统计的偏置可忽略（在意可去重后算）。

### 4.6 监控量与预注册判据（写死进训练脚本，不许事后挑）

| 监控量 | 定义 | 含义与判据 |
|---|---|---|
| **R²_latent(k)** | `1 − MSE(ẑ_k, d_k)`（目标单位方差 ⇒ 基线 MSE₀=1 闭式） | **头号指标**：未来相对状态变化被解释比例。预期 R²(1)>R²(5)>R²(10)，总量 0.05–0.15；若 ≈0 → 没学到动力学，整线停 |
| L_pred vs L₀ | SmoothL1 全预测 0 基线 **L₀≈0.425**（单位高斯闭式：E[SmoothL1(0,N(0,1))]） | loss 不显著低于 0.425 = 还在瞎猜 |
| **σ_raw 中位数** | 标准化**前** δ_k 的逐维截面 std | **真正的塌缩信号**（标准化后 d 永远单位方差，会骗人）；持续 ↓ → f_ξ 在塌。**< 0.1 → 硬停** |
| eff-rank(Cov(z_ctx)) | 表示协方差有效秩 | **< 8（D=64）→ 维度塌缩，硬停** |
| 2019H2 frozen probe | train 末段（2019-07~2020-03）ridge rank_ic | SSL 的"valid 曲线"；趋势不升 → 设计迭代（最多 2 次）。**绝不用真 valid 选 checkpoint**（防模型选择泄漏，见 §8） |

**诚实预期设定**：R²_latent 大头来自波动/流动性维（高可预测），收益方向子空间贡献仅 ~0.006 量级（IC² 尺度）——**R² 高 ≠ probe 好**，两条曲线都看；probe 是金标准，R² 只回答"有没有在学东西"。

**R²(k) 的窗口重叠混杂（解读注意；2026-06-10 设计评审补充）**：context `[a−7..a]` 与 target `[a+k−7..a+k]` 重叠 **max(0, 8−k) 天**——k=1 共享 7 天，d_1 的可预测性有相当部分来自"已知旧日 a−7 滚出窗口"的**机械效应**（滚出影响完全可从 context 计算，非市场前瞻）；k=5 共享 3 天；**k=10 零重叠，是唯一的纯 forecasting 读数**。推论：(i) R²(k) 随 k 下降部分反映重叠收缩而非可预测性衰减，横向比较须扣此混杂，纯净读数看 R²(10) 与 pinned 维；(ii) 这不全是 bug——预测 roll-off 效应迫使 z 编码**逐日分辨的窗内时间结构**（而非 bag 摘要），对表示是有益压力；(iii) 若诊断显示 k=1 机械项垄断梯度，v2 旋钮 = horizon loss 权重上调 k=10（v1 等权）。

---

## 5. 实现蓝图

### 5.1 数据通道：一个 18 步窗口切全部 view（零新基建）

用现有 `TSDatasetH`，只改 `step_len: 8 → 18`（8 context + 10 max horizon）。每样本 (i, τ) 得 `[18, 158]`，位置 0..17 对应日 τ−17..τ：

```
position:   0  1  2  3  4  5  6  7 | 8  9  10 11 12 13 14 15 16 17
            └────── context ──────┘
            anchor = pos 7 (= 日 τ−10)
target k=1:    └────── pos 1..8 ─────┘            (止于 anchor+1)
target k=5:                └── pos 5..12 ──┘      (止于 anchor+5)
target k=10:                      └── pos 10..17 ──┘ (止于 anchor+10 = 日 τ)
```

要点（每条都有理由）：
- **模型 `context_len` 保持 8**——每个 view 进 backbone 都是 8 步，18 只是数据窗口，进模型前切片；
- batch 仍由 `FixedDailyBatchSampler` 按日成批 ⇒ 同 batch 共享 anchor 日 ⇒ **CS 统计直接在 batch 维做即正确截面操作**（日度 batch 设计白送的）；
- 防泄漏自动满足：train τ ≤ 2020-03-31 ⇒ 最远 target = τ 本身，全部 ≤ fit_end_time；
- 无 label：不读即可（handler 配置可不动）；
- 2008 年初短窗口由现有 Fillna 机制处理（与现 step_len=8 一致）；
- universe → csi800：`QIB_DATA_OVERRIDES_JSON` 改 `instruments`；macro 走 `use_external_macro=False`（internal regime 不依赖 pkl，免去 csi800 预计算依赖）。

### 5.2 代码计划：两个文件 ~400 行（不动 model_adapter 的监督循环）

**文件 1：`module/ssl/jepa.py`（~200 行）**

```python
class JEPAPredictor(nn.Module):
    """g_φ: [z_ctx ; e_k] -> ẑ_k。容量刻意小（2 层 MLP）——
    可预测结构必须压进 encoder 的表示，不能藏在 predictor 里（§1.4 要点 3）。"""
    def __init__(self, d_model=64, d_hidden=128, n_horizons=3, out_dim=None):
        super().__init__()
        self.h_emb = nn.Embedding(n_horizons, d_model)   # 条件变量：horizon embedding
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, out_dim or d_model),     # pinned 臂: out_dim = d_model + 2
        )
    def forward(self, z_ctx, k_idx):                     # [B,D], int
        e = self.h_emb.weight[k_idx].expand_as(z_ctx)
        return self.net(torch.cat([z_ctx, e], dim=-1))

class JEPAPretrainer:
    def __init__(self, net, lr=1e-4, pred_lr_mult=10.0, lambda_var=0.01, gamma=1.0):
        self.online = net                                          # f_θ = QuantMoEModel 本体
        self.target = copy.deepcopy(net).requires_grad_(False)     # f_ξ
        self.predictor = JEPAPredictor(...)
        # 双时间尺度：predictor lr ×10（§4.4）
        self.opt = AdamW([{'params': net.parameters(), 'lr': lr},
                          {'params': self.predictor.parameters(), 'lr': lr * pred_lr_mult}])

    @torch.no_grad()
    def ema_update(self, m):
        """ξ ← m·ξ + (1−m)·θ。m: 0.996→1.0 cosine ramp（早期目标新鲜，后期冻结）。
        ⚠️ buffer 也要同步（RegimeContextEncoder 等若有 running stats）。"""
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.lerp_(p_o, 1.0 - m)
        for b_t, b_o in zip(self.target.buffers(), self.online.buffers()):
            b_t.copy_(b_o)

    def step(self, x18):                                 # [B,18,N]，B = 一个交易日截面
        z_ctx = encode(self.online, x18[:, 0:8])         # [B,D] 唯一带梯度的前向
        with torch.no_grad():                            # —— target 构造（§4.1 ①–④）——
            z_a = encode(self.target, x18[:, 0:8])       # anchor 的 target-encoder 视角
            tgts = {}
            for j, k in enumerate((1, 5, 10)):
                d = encode(self.target, x18[:, k:k+8]) - z_a          # ① 差分消身份
                d = (d - d.mean(0)) / d.std(0).clamp_min(1e-4)        # ②③④ 截面 CS-zscore
                tgts[j] = d                              # 逐维单位方差 ⇒ MSE 基线 = 1.0
        loss, mse_diag = 0., {}
        for j in range(3):
            pred = self.predictor(z_ctx, j).float()      # fp32 cast（AMP 下 loss 稳定，house 惯例）
            loss = loss + F.smooth_l1_loss(pred, tgts[j].float())
            mse_diag[j] = F.mse_loss(pred.detach(), tgts[j].float()).item()  # R²_latent = 1 − mse
        # VICReg variance hinge：仅兜底，λ 小（§4.4）
        std = z_ctx.float().std(0)
        loss = loss + 0.01 * F.relu(1.0 - std).mean()
        return loss, mse_diag
```

每 step 前向计数：**5 次（1 带梯度 + 4 no-grad：ctx-online、ctx-target、3 个 target 窗口）≈ 监督 step 的 ~2.5× 成本**。

`encode()` 的暴露方式二选一：
- (a) **推荐**：`QuantMoEModel` 加 `encode_only=True` 分支（~10 行，flag 默认 off 不动现行为）——house 的 flag-guarded 模式（同 `temporal_readout`）；
- (b) `forward_pre_hook` 截 `factor_pooling` 输入输出——pool forensics 已用同款 hook，零模型改动。

**文件 2：`scripts/jepa_pretrain.py`（~200 行）** —— 独立训练脚本，house 形态（同 `time_readout_finetune.py` / `pool_readout_forensics.py`："独立脚本 + 复用部件"，不碰 `model_adapter.py` 3500 行监督循环）：

```
数据:   data_conf 深拷贝 + step_len=18 + instruments=csi800 + train-only segments
模型:   QuantMoEModel(model_config)   # context_len=8 不变, use_external_macro=False
采样:   FixedDailyBatchSampler(train_tsds, batch_size=300, seed=...)   # 直接 import
循环:   cosine + 5% warmup, 100 epochs, amp_fp16
        每 step: loss = pretrainer.step(x18); opt.step(); pretrainer.ema_update(m_t)
每 epoch 落盘:
  - online backbone 的 state_dict checkpoint
  - 塌缩诊断 CSV: σ_raw 中位数 / eff-rank(Cov(z)) / R²_latent(k) / loss vs L₀
  - 每 5 epoch: 2019H2 frozen-ridge probe rank_ic（SSL 的 valid 曲线）
硬停（预注册）: eff-rank < 8 或 σ_raw 中位数 < 0.1 → 标记 collapse 退出
```

### 5.3 下游对接（复用两套现有基建）

- **Stage-1 frozen probe**：冻结 backbone → train 段 pooled z 拟 ridge（α 扫 {1e-3..10}）→ valid `daily_rank_ic`。**代码直接复用 `scripts/pool_readout_forensics.py` 的加载 + ridge 路径**（它干的就是这件事）。
- **Stage-2 SSL-init 微调**：adapter 加 flag-guarded `trainer_config["init_state_dict_path"]`（~15 行：训练前 `net.load_state_dict(torch.load(p), strict=False)`；head/predictor 不载入，head 重新训练）。

### 5.4 训练配方汇总表

| 项 | 值 | 理由 |
|---|---|---|
| encoder lr / predictor lr | 1e-4 / 1e-3（×10） | 双时间尺度（§4.4） |
| schedule | cosine + 5% warmup, 100 epochs | 对齐现有惯例 |
| EMA m | 0.996 → 1.0 cosine ramp | §4.4 |
| batch | 300/日（FixedDailyBatchSampler） | CS 统计正确性白送 |
| precision | amp_fp16，loss 处 fp32 cast | house 惯例（readout 实验同款处理） |
| SmoothL1 β | 1.0 | 匹配单位方差目标 |
| λ_var / γ | 0.01 / 1.0 | 保险丝定位 |
| σ clamp | 1e-4 | 防除零 |

---

## 6. 评估阶梯与 kill 条件（每级带 kill，省卡）

| Stage | 内容 | 参照 | kill / pass |
|---|---|---|---|
| **0** 训练中 | 塌缩监控 + 2019H2 probe 趋势 | — | eff-rank<8 或 σ_raw<0.1 硬停；probe 无上升 → 设计迭代（最多 2 次） |
| **1** frozen probe（分钟级，生死关） | 冻结 encoder + ridge → valid rank_ic | ① random-init backbone probe（下界，预期 ~0.02–0.04）② supervised full_135 同协议 probe（上参照，~0.073–0.081，forensics 已有）③ 绝对线 | **kill: probe < random+0.02 或 < 0.060**；**pass: ≥ 0.070**（无监督逼近监督 = 直接有趣） |
| **2** SSL-init 微调 | SSL 权重 init + 完整监督训练（叠 weight-EMA 更佳），paired vs scratch | full_135 | n=3 Δ<+0.003 → kill；过 → n=6 四元组（house 规则） |

**与 "finetune-hurts-valid" 发现的区分**：那是 **converged 模型的 continue-finetune**；这里是 **init**——不冲突。但两臂都跑（frozen+head / init+full-train），以分辨价值在表示还是初始化。

---

## 7. 防泄漏清单（/quant-leakage-audit 预演）

1. 预训练语料止于 **2020-03-31**（= 现 `fit_end_time`）；target 需 t+10 ⇒ 实际样本止于 2020-03 中旬。**2020-07 之后的数据对 encoder 必须是未来**。
2. `RobustZScoreNorm` 拟合边界与现行完全一致（fit 2008→2020-03），csi800 重新拟合时同边界。
3. 无 label 参与预训练；pinned 臂的已实现收益均在 train 段内（合法性 = 监督训练用 label 同级）。
4. **SSL checkpoint 选择只用 2019H2 probe**（train 末段），绝不用真 valid——否则重蹈 valid==test 的模型选择泄漏。
5. Stage-1/2 的 label 管道、segments、评估窗口与监督基线完全一致（paired 可比性）。

---

## 8. 预算、时序与依赖

| 项 | 量 |
|---|---|
| 实现 | 2–3 天（jepa.py + 脚本 + encode 分支 + init_state_dict flag） |
| 预训练 | ~2950 day-batch/epoch × 100e，5 前向/step ≈ **12–20 GPU-hr（1–2 通宵 @4070S）**；A/B 两臂 ×2 |
| Stage-1 probe | 分钟级 |
| Stage-2 | 6 runs ≈ 4 GPU-hr |

**时序与依赖**：`h-20260610-001/002` 先出 n=3 结论 → weight-EMA 基建落地（共享 EMA 代码）→ 本设计立卡开工。立卡时把 §4.6 预注册判据 + §6 双 kill 线 + latent-only/pinned 两臂写进 falsifier。

---

## 9. 诚实的概率评估与价值不对称

- **主要风险**：A 股日频 Alpha158 的可预测 latent 结构可能太薄——Δz 可预测部分剩下的多是慢变量（波动/流动性 regime），probe 学到 regime nowcasting 而非截面 alpha。**Stage-1 过线（≥0.070）主观概率 ~25–35%**。
- **价值不对称**：(a) 全失败成本 = 1–2 通宵 GPU + 2–3 天实现；(b) Stage-1 失败的副产品 = "该数据中无监督可学动力学结构的上限"——与 pool forensics 同级的**结论性知识**，永久关掉一整类提案；(c) 过线则是唯一同时攻 gen-gap 与信息利用率的 lever，且与 001/002/EMA 全可叠加（叠加路径：SSL-init + stock-expert + DropExtreme + weight-EMA）。

---

## 10. 已知噪声源与 v2 扩展

- **停牌 stale 窗口**：未来窗口内停牌的股票被 Fillna 填成 stale ⇒ Δz≈0 的假目标。v1 接受（量小、方向偏保守）；v2 用交易日历过滤 target 窗口无效样本。
- **薄日上采样重复**：CS 统计微偏，可忽略；在意则去重后算 μ/σ。
- v2 候选：context 侧随机 mask 20% 因子列做增广（防单因子依赖）；同业 relational target（需 002 的截面机制先落地）；σ-加权逐维 loss（防超可预测维垄断梯度）。

## 附录 A：为什么这不是又一个 readout/pool 类提案（与已证伪 lever 的类别区分）

已死的 lever（readout 双轴、pool 重加权、τ-promotion、capacity）全属同一类：**固定的信息和固定的训练动力学下，重新分配/重新聚合已有表示**。本设计属于另一类：**改变表示被学出来的目标函数本身**（学什么），与 weight-EMA（解的几何）、DropExtremeLabel（监督纯度）、stock-expert（新信息通路）共同构成对 gen-gap 墙的四个正交攻击面。pool forensics 的"158 因子等弱冗余、equal-weight ~IC-optimal"结论不约束本设计——它约束的是聚合算子，不是表示学习目标。

## 附录 B：参考文献指针

- LeCun (2022) *A Path Towards Autonomous Machine Intelligence*（JEPA 论纲、能量视角）
- Assran et al. (2023) *I-JEPA*（配方原型：context/target block + 位置条件 + EMA target）
- Bardes et al. (2024) *V-JEPA*（时空版）
- Grill et al. (2020) *BYOL*（EMA + stop-grad + predictor 不对称防塌缩）
- Bardes et al. (2022) *VICReg*（variance/covariance 正则）
- Tian et al. (2021) *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs*（双时间尺度防塌缩的理论解释）
