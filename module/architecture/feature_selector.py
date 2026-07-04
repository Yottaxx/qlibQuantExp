import torch
from torch import nn


# --- B. 可微特征选择器 (STG) ---
class DifferentiableFeatureSelector(nn.Module):
    def __init__(self, num_features, sigma=0.5):
        super().__init__()
        # 初始化 logit，使其对应的概率在 0.5 附近
        self.mu = nn.Parameter(torch.randn(num_features) * 0.01)
        self.noise_std = sigma

    def sample_mask(self, temperature=0.1, training: bool | None = None):
        if training is None:
            training = self.training

        if training:
            noise = torch.randn_like(self.mu) * self.noise_std
            logits = self.mu + noise
            z = torch.sigmoid(logits / temperature)
        else:
            z = torch.sigmoid(self.mu)

        # 正则化 Loss (L1 Norm)
        # Fix: 使用 sum(dim=-1).mean() 替代 .mean()
        # .mean() 会随着特征数量 N 增加而变小 (1/N)，导致对多特征模型约束过弱
        # .sum(dim=-1).mean() 代表“平均每个样本选择了多少个特征”，对 N 不敏感
        reg_loss = z.sum(dim=-1).mean()

        return z, reg_loss

    def forward(self, x, temperature=0.1, training=True, *, apply_mask: bool = True):
        # x: [B, T, N, D]
        z, reg_loss = self.sample_mask(temperature=temperature, training=training)

        if apply_mask:
            # Apply Gate (factor axis)
            z_broadcast = z.to(dtype=x.dtype, device=x.device).view(1, 1, -1, 1)
            x = x * z_broadcast

        return x, reg_loss, z
