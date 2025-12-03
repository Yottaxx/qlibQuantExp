import torch
import torch.nn.functional as F


class QuantLossFunctions:
    @staticmethod
    def cs_ic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Cross-Sectional IC Loss (Pearson Correlation).
        Input: pred [B], target [B] (Batch represents stocks in a single day)
        """
        # 扁平化确保是 1D 向量
        pred = pred.flatten()
        target = target.flatten()

        # 极端情况兜底
        if pred.numel() < 2: return torch.tensor(0.0, device=pred.device)

        # 归一化 (Z-Score)
        pred_n = (pred - pred.mean()) / (pred.std() + eps)
        target_n = (target - target.mean()) / (target.std() + eps)

        # IC = E[X*Y] (after norm)
        ic = (pred_n * target_n).mean()
        return 1.0 - ic

    @staticmethod
    def ranknet_topbottom_loss(pred: torch.Tensor, target: torch.Tensor, topk: int = 10) -> torch.Tensor:
        """
        Pairwise RankNet Loss focused on Top-K vs Bottom-K stocks.
        """
        pred = pred.flatten()
        target = target.flatten()

        # 如果样本不够 topk * 2，跳过
        if pred.numel() < 2 * topk: return torch.tensor(0.0, device=pred.device)

        # 1. 依据真实 Label 找到头部和尾部股票的索引
        _, top_idx = torch.topk(target, k=topk, largest=True)
        _, bot_idx = torch.topk(target, k=topk, largest=False)

        # 2. 取出预测分
        p_top = pred[top_idx].unsqueeze(1)  # [K, 1]
        p_bot = pred[bot_idx].unsqueeze(0)  # [1, K]

        # 3. 优化目标: p_top > p_bot
        # Loss = log(1 + exp(p_bot - p_top))
        return F.softplus(p_bot - p_top).mean()

    @staticmethod
    def cs_huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        return F.huber_loss(pred.flatten(), target.flatten(), delta=delta)