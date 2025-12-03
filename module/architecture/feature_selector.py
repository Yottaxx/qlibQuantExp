import torch
from torch import nn


# --- B. 可微特征选择器 (STG) ---
class DifferentiableFeatureSelector(nn.Module):
    def __init__(self, num_features, sigma=0.5):
        super().__init__()
        # 初始化 logit，使其对应的概率在 0.5 附近
        self.mu = nn.Parameter(torch.randn(num_features) * 0.01)
        self.noise_std = sigma

    def forward(self, x, temperature=0.1, training=True):
        # x: [B, T, N, D]
        if training:
            noise = torch.randn_like(self.mu) * self.noise_std
            logits = self.mu + noise
            z = torch.sigmoid(logits / temperature)
        else:
            z = torch.sigmoid(self.mu)

        # 正则化 Loss (L1 Norm)
        reg_loss = torch.mean(z)

        # Apply Gate
        z_broadcast = z.view(1, 1, -1, 1)
        x_masked = x * z_broadcast

        return x_masked, reg_loss, z
