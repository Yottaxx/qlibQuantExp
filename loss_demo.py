# ranking_losses_demo.py
# -*- coding: utf-8 -*-
"""
用一个极简例子演示：
- pointwise: MSE 回归式损失
- pairwise: pairwise logistic ranking loss
- listwise: NDCG-based listwise 指标 & loss（主要是评估用）
- ListNet: listwise top-1 概率交叉熵损失
- ListMLE: 基于 Plackett-Luce 的排列似然损失

约定：
- 一个 "query" = 一天的截面（同一个日期的所有股票）
- labels: 真实 y（比如未来收益）
- scores: 模型打分（越大排名越靠前）
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


# =========================
# 1. Pointwise: MSE 回归损失
# =========================
def pointwise_mse_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Pointwise: 把每个 (x_i, y_i) 当成独立样本，做回归。
    这里简单用 MSE。

    Args:
        scores: [n]，模型预测的分数（可以直接当作 predicted y）
        labels: [n]，真实标签 y

    Returns:
        标量 loss（越小越好）
    """
    return F.mse_loss(scores, labels)


# =========================
# 2. Pairwise: logistic loss
# =========================
def pairwise_logistic_loss(
    scores: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss（类似 RankNet）.

    做法：
    - 枚举所有 pair (i, j)，只保留 y_i > y_j 的成对样本
    - 对每对，比较 s_i - s_j，希望差值越大越好
    - 损失：log(1 + exp(-(s_i - s_j)))

    Args:
        scores: [n]
        labels: [n]

    Returns:
        标量 loss（越小越好）
    """
    n = scores.size(0)
    # [n, n] 的 label 差值矩阵：y_i - y_j
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # [n, n]
    # 只保留 y_i > y_j 的 pair（有方向）
    pair_mask = label_diff > 0

    if pair_mask.sum() == 0:
        # 没有严格有序的 pair（比如所有 label 相等），返回 0
        return torch.tensor(0.0, dtype=scores.dtype, device=scores.device)

    # 对应的 score 差值：s_i - s_j
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # [n, n]
    score_diff = score_diff[pair_mask]  # [num_pairs]

    # logistic loss: log(1 + exp(-Δs))
    loss = torch.log1p(torch.exp(-score_diff))
    # 或者用更数值稳定的 F.softplus(-score_diff)
    # loss = F.softplus(-score_diff)

    return loss.mean()


# =====================================
# 3. Listwise（这里用 NDCG 作为 listwise 指标）
# =====================================
def _dcg_at_k(rels_sorted: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    简化版 DCG@k:
    DCG = sum_{i=1..k} rel_i / log2(1 + i)

    这里假设 rels_sorted 已经按“预测排序”或“真实最佳排序”排好序。
    """
    if k is None:
        k = rels_sorted.size(0)
    k = min(k, rels_sorted.size(0))

    device = rels_sorted.device
    dtype = rels_sorted.dtype

    idx = torch.arange(1, k + 1, device=device, dtype=dtype)  # [1..k]
    discounts = 1.0 / torch.log2(idx + 1.0)  # 1 / log2(1 + rank)
    dcg = (rels_sorted[:k] * discounts).sum()
    return dcg


def listwise_ndcg_loss(
    scores: torch.Tensor, labels: torch.Tensor, k: Optional[int] = None, eps: float = 1e-8
) -> torch.Tensor:
    """
    用 NDCG@k 做一个 listwise 指标，并用 (1 - NDCG) 当作“loss”。

    注意：
    - 这是基于排序的指标，对 scores 做 argsort，**不可导或梯度不稳定**，
      通常用来评估而不是直接当训练 loss。
    - 真正的 listwise 训练一般用 ListNet / ListMLE 等“可导 surrogate”。

    Args:
        scores: [n]
        labels: [n]
        k: 只看前 k 名；若为 None 则看全列表

    Returns:
        标量 "loss" = 1 - NDCG（越小越好）
    """
    # 1) 预测排序下的 DCG
    sorted_idx_pred = torch.argsort(scores, descending=True)
    labels_sorted_pred = labels[sorted_idx_pred]
    dcg = _dcg_at_k(labels_sorted_pred, k)

    # 2) 理想排序（按 labels 排）下的 IDCG
    sorted_idx_true = torch.argsort(labels, descending=True)
    labels_sorted_true = labels[sorted_idx_true]
    idcg = _dcg_at_k(labels_sorted_true, k)

    if idcg.item() <= 0:
        # 全 0 等情况，NDCG 没有意义
        return torch.tensor(0.0, dtype=scores.dtype, device=scores.device)

    ndcg = dcg / (idcg + eps)
    return 1.0 - ndcg  # 当作一个 "loss"


# =========================
# 4. ListNet: top-1 概率交叉熵
# =========================
def listnet_top1_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    ListNet (top-1) listwise loss.

    思路：
    - 从 labels 构造“理想 top-1 概率分布”：P_true(i) ∝ exp(y_i)
    - 从 scores 构造“模型 top-1 概率分布”：P_pred(i) ∝ exp(s_i)
    - 用交叉熵：CE(P_true || P_pred)

    Args:
        scores: [n]
        labels: [n]

    Returns:
        标量 listwise loss（越小越好）
    """
    # 理想分布（用 softmax(y)）
    p_true = F.softmax(labels, dim=0)           # [n]
    # 模型分布（softmax(s)）
    log_p_pred = F.log_softmax(scores, dim=0)   # [n]

    loss = -(p_true * log_p_pred).sum()
    return loss


# =========================
# 5. ListMLE: 排列概率 NLL
# =========================
def listmle_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    ListMLE listwise loss（基于 Plackett-Luce）.

    步骤：
    - 按 labels 从大到小得到真实排序 π_true
    - 按这个顺序排列 scores -> s_sorted
    - 定义排列概率：
        P(π_true | s) = ∏_{k=1..n} exp(s_{π_k}) / ∑_{j>=k} exp(s_{π_j})
    - loss = -log P(π_true | s)

    Args:
        scores: [n]
        labels: [n]

    Returns:
        标量 listwise loss（越小越好）
    """
    device = scores.device
    dtype = scores.dtype

    n = scores.size(0)
    # 按 labels 从大到小排序，得到“真实排名”
    _, indices = torch.sort(labels, descending=True)
    s_sorted = scores[indices]  # [n]

    # 直接按定义求：
    # loss = -∑_{k=1..n} [s_k - log ∑_{j=k..n} exp(s_j)]
    loss = torch.tensor(0.0, dtype=dtype, device=device)
    for k in range(n):
        tail = s_sorted[k:]  # 第 k 名及之后的那些
        log_sum_exp_tail = torch.logsumexp(tail, dim=0)
        loss = loss - (tail[0] - log_sum_exp_tail)

    # 也可以不除 n；这里除 n 只是让数值尺度更温和
    return loss / n


# =========================
# 6. Demo: 用 3 只股票的极简例子对比
# =========================
def demo():
    """
    构造一个极简截面：
    - 真实收益：S1 > S2 > S3
    - M1：排序正确，但数值跟 y 不完全贴
    - M2：数值更贴近 y，但把前两名顺序搞反

    看看：
    - MSE (pointwise) 会偏爱谁？
    - Pairwise / ListNet / ListMLE / Listwise NDCG 会偏爱谁？
    """
    # 真实收益
    labels = torch.tensor([3.0, 2.0, 1.0])  # S1, S2, S3

    # 模型 1：排序正确
    scores_m1 = torch.tensor([3.0, 2.0, 0.0])

    # 模型 2：排序略乱（S2 稍微高于 S1），但数值更贴近
    scores_m2 = torch.tensor([2.5, 2.6, 1.5])

    models = {
        "M1 (排序正确)": scores_m1,
        "M2 (排序略乱)": scores_m2,
    }

    for name, scores in models.items():
        print("=" * 60)
        print(f"{name}")
        print(f"  labels: {labels.tolist()}")
        print(f"  scores: {scores.tolist()}")

        pw_mse = pointwise_mse_loss(scores, labels).item()
        pw_pair = pairwise_logistic_loss(scores, labels).item()
        lw_ndcg = listwise_ndcg_loss(scores, labels, k=None).item()
        ln_loss = listnet_top1_loss(scores, labels).item()
        lmle_loss = listmle_loss(scores, labels).item()

        print(f"  Pointwise MSE loss         : {pw_mse:.6f}")
        print(f"  Pairwise logistic loss     : {pw_pair:.6f}")
        print(f"  Listwise (1 - NDCG)        : {lw_ndcg:.6f}")
        print(f"  ListNet top-1 loss         : {ln_loss:.6f}")
        print(f"  ListMLE loss               : {lmle_loss:.6f}")
        print()

    print("=" * 60)
    print("观察：")
    print("- MSE 更关心“数值贴不贴 y”，可能更喜欢 M2；")
    print("- Pairwise / Listwise 一般更偏向“排序正确”的 M1。")


if __name__ == "__main__":
    demo()
