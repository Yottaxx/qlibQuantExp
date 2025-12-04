import torch
import torch.nn.functional as F


class QuantLossFunctions:
    @staticmethod
    def cs_ic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Cross-Sectional IC Loss (Pearson Correlation).
        通常作为 metric 使用，这里返回 -IC（便于当作“要最小化的量”）。
        pred, target: [B] (同一日的股票截面).
        """
        pred = pred.flatten()
        target = target.flatten()

        mask = torch.isfinite(pred) & torch.isfinite(target)
        if mask.sum() < 2:
            return torch.tensor(0.0, device=pred.device)

        x = pred[mask] - pred[mask].mean()
        y = target[mask] - target[mask].mean()
        cov = (x * y).mean()
        var_x = (x * x).mean()
        var_y = (y * y).mean()
        ic = cov / (torch.sqrt(var_x * var_y) + eps)
        return -ic  # 作为“loss”时：越小越好 => IC 越大

    @staticmethod
    def ranknet_topbottom_loss(pred: torch.Tensor, target: torch.Tensor, topk: int = 5) -> torch.Tensor:
        """
        Pairwise RankNet Loss on top/bottom K names.
        pred, target: [B]
        """
        pred = pred.flatten()
        target = target.flatten()

        n = pred.numel()
        if n < 2 or topk <= 0:
            return torch.tensor(0.0, device=pred.device)

        k = min(topk, n)
        _, top_idx = torch.topk(target, k=k, largest=True)
        _, bot_idx = torch.topk(target, k=k, largest=False)

        p_top = pred[top_idx].unsqueeze(1)  # [K,1]
        p_bot = pred[bot_idx].unsqueeze(0)  # [1,K]

        # log(1 + exp(p_bot - p_top)) = softplus(p_bot - p_top)
        return F.softplus(p_bot - p_top).mean()

    @staticmethod
    def cs_huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """截面回归型 Huber Loss，用于监控/辅助."""
        return F.huber_loss(pred.flatten(), target.flatten(), delta=delta)

    @staticmethod
    def listmle_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        ListMLE loss (list-wise ranking).
        - pred, target: [B]，B 是当日采样到的一篮子股票
        - target 只用于确定排序（你的 label 已经是 CSRankNorm 后的“单调变换”）

        公式: L = - sum_i log P(y_i | y_{>=i})
            = sum_i [ logsumexp_{j>=i} s_j - s_i ]
        其中 s 是按照 target 降序排序后的 pred（可选温度 tau 调节平滑度）。
        """
        pred = pred.flatten()
        target = target.flatten()
        n = pred.numel()
        if n < 2:
            return torch.tensor(0.0, device=pred.device)

        # 1) 按 target 降序排序
        _, indices = torch.sort(target, descending=True)
        s = pred[indices]  # [n]

        # 2) 温度 + 数值稳定
        s = s / tau
        s = s - s.max()

        # 3) 逐前缀 logsumexp
        rev_s = s.flip(0)
        log_cumsumexp = torch.logcumsumexp(rev_s, dim=0).flip(0)  # [n]

        loss = (log_cumsumexp - s).sum() / n
        return loss
