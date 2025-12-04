loss_ic – Cross-Sectional IC Loss 

losses

定义：1 − ρ(pred, label)，其中 ρ 是当日截面上的 Pearson 相关系数。

取值范围（理论）：[0, 2]，但实际常在 [0, 1] 左右。

含义：

loss_ic ≈ 0：模型预测和真实收益高度正相关（IC → 1），非常优秀。

loss_ic ≈ 1：IC 接近 0，说明预测接近随机。

loss_ic > 1：IC 为负，有系统性反向信号（可以用在多空翻转策略，但与当前优化目标相悖）。

观察点：

训练初期 loss_ic 会快速下降；

若后期震荡在某个区间，说明 capacity / 数据噪声 / overfit 已经成为瓶颈。

loss_rank – Top-Bottom RankNet Loss 

losses

定义：只对 label Top-K 与 Bottom-K 股票构造 pairwise logistic loss：
E[log(1 + exp(p_bot − p_top))]

取值范围：[0, +∞)，但典型在 [0, 1] 左右。

含义：

越接近 0，说明“头部股票预测得分普遍高于尾部”越充分；

较大值表明模型在极端多空两端排序做得不好，即使整体 IC 还可以。

研究价值：弥补 IC 对“tail ordering”不敏感的问题，更贴近实际 long-short 策略的盈亏结构。

loss_huber – Robust Regression Loss 

losses

定义：对 pred vs label 的 Huber loss；小残差是 L2，大残差切到 L1。

取值范围：[0, +∞)；越小越好。

含义：

控制模型不要在少数极端收益上“发疯”（尤其 A 股这种 fat-tail 市场），提升稳健性。

观察点：如果 loss_ic 好但 loss_huber 很大，可能 model 在极端行情下误差很大，risk 部门会不高兴。

loss_aux – MoE Router Z-Loss

定义：sum_layer E[(logsumexp(router_logits))^2] * router_z_loss_coef。

取值范围：非负，越小说明 router logits 的范数较小。

含义：

用来防止 MoE router 的 pre-softmax logits 爆炸，保持 softmax 较为稳定；

借鉴自 Switch Transformer 系列的 Z-loss，用于提高训练稳定性。

观察：

太大：router 输出非常极端，可能引起梯度爆炸/ collapse；

过小且 gate_entropy 同时接近 0，可能说明 router logits 接近常数但 softmax 又极端，值得检查。

loss_sparsity – STG Feature Selection Regularization

定义：reg_loss = mean(z)，z 是每个因子的 gate 概率；乘以 selection_reg_lambda 加入总损失。

取值范围：[0, 1]；越小越 sparse。

含义：

近似等于“平均有多少比例的因子被激活”；

控制因子数目 vs 噪声 trade-off：过大→全因子堆噪声；过小→信息丢失。

观察：

与 active_feat_ratio 的水平强相关，后者可以看作更直接的 interpretability 指标。

3.2 结构诊断指标（Structural Diagnostics）

gate_entropy – Router Decision Entropy 

moe_block

定义：对 router 的 gate 分布 p = softmax(logits)，计算
H(p) = −∑ p_i log p_i，并在 batch 上取平均。

对于 2-expert gate，理论范围 [0, ln 2 ≈ 0.693]。

含义：

≈ 0：router 非常“决绝”，几乎总是把样本判给某一个 expert（time 或 factor），高度 specialization；

≈ 0.69：router 近似均匀，两条路都在用，更像 soft ensemble。

研究角度：

长期极低 entropy 且 time_ratio 接近 0 或 1 → collapse：模型退化成“只有时间专家”或“只有因子专家”，说明 MoE 没有真正起作用；

合理区间是 中高 entropy + 时间/截面比例有明显随 regime 变化，这在 MoE 理论分析里被视为“健康的专家分工”。

time_ratio – Expected Time-Expert Usage

定义：E[p_time]，即对 batch 和（可能）层的平均，p_time 是 gate 分布中“时间专家”的概率。

范围：[0,1]。

含义：

趋近 1：router 倾向于用 time_expert 做“大部分工作”，模型更像纯时序模型；

趋近 0：router 主要依赖 factor_expert，模型更像截面关系网络；

~0.5：两条路权重均衡。

对 “gate-over-time” 曲线的解释：

在波动性高、趋势强的 regime，理论上 time_ratio 上升（趋势占主导）；

在 sector rotation、风格切换强的 regime，time_ratio 下降，让 factor_expert 捕捉截面轮动。

这类 pattern 如果在图上出现，是“可以写在 KDD/WWW 的 qualitative evidence”。

active_feat_ratio – Effective Feature Utilization Ratio

定义：mean(selected_mask)，其中 mask 来自 STG gate z 的 sigmoid 输出。

范围：[0,1]。

含义：

≈ 1：几乎所有因子都被充分使用（dense）；

≈ 0.1：平均只有 10% 的因子在多数样本上“开灯”，其余被抑制；

接近 0：过度稀疏，模型会丢掉很多信息。

研究意义：

配合 get_feature_importance() 输出的 mu.sigmoid()，可以直接画 bar plot：

高权重因子对应长期稳定的 alpha；

低权重因子可以视为“冗余/噪声候选”，给 factor library 做 pruning。