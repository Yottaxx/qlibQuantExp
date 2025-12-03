import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from module.utils.model_configuration import QuantMoEConfig
from module.architecture.parallel_attention import ParallelAttention


class RegimeAdaptiveMoEBlock(nn.Module):
    """Regime-adaptive MoE block with spatio-temporal disentanglement.

    Inputs:
      x: [B, T, N, D]
      regime_embedding: [B, D]
      attn_bias: None or Tensor or (bias_time, bias_factor)
        - bias_time  expected shape [1, H, T, T]
        - bias_factor expected shape [1, H, N, N]
    """

    def __init__(self, config: QuantMoEConfig):
        super().__init__()
        self.config = config

        # Router: route between {time_expert, factor_expert}
        self.router = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),
        )

        self.time_expert = ParallelAttention(config)
        self.factor_expert = ParallelAttention(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, regime_embedding: torch.Tensor, attn_bias=None):
        B, T, N, D = x.shape
        residual = x

        x = self.norm1(x)

        router_logits = self.router(regime_embedding)  # [B,2]
        gate_weights = F.softmax(router_logits, dim=-1)  # [B,2]

        # diagnostics
        z_loss = (torch.logsumexp(router_logits, dim=-1) ** 2).mean()
        entropy = -(gate_weights * torch.log(gate_weights + 1e-9)).sum(-1).mean()
        time_ratio = gate_weights[:, 0].mean()

        w_time = gate_weights[:, 0].view(B, 1, 1, 1)
        w_factor = gate_weights[:, 1].view(B, 1, 1, 1)

        bias_time = bias_factor = None
        if isinstance(attn_bias, (tuple, list)) and len(attn_bias) == 2:
            bias_time, bias_factor = attn_bias
        else:
            bias_time = attn_bias

        # 1) Time expert: per-factor temporal modeling
        h_time = rearrange(x, "b t n d -> (b n) t d")
        out_time = self.time_expert(h_time, bias_time)
        out_time = rearrange(out_time, "(b n) t d -> b t n d", b=B, n=N)

        # 2) Factor expert: per-time cross-sectional modeling
        h_factor = rearrange(x, "b t n d -> (b t) n d")
        out_factor = self.factor_expert(h_factor, bias_factor)
        out_factor = rearrange(out_factor, "(b t) n d -> b t n d", b=B, t=T)

        fused = w_time * out_time + w_factor * out_factor
        x = residual + fused
        x = x + self.ffn(self.norm2(x))

        diag = {
            "z_loss": z_loss,
            "entropy": entropy,
            "time_ratio": time_ratio,
            "weights": gate_weights,
        }
        return x, diag
