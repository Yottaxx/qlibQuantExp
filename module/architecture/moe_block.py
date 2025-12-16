# module/architecture/moe_block.py

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
        - bias_time   expected shape [1, H, T, T]
        - bias_factor expected shape [1, H, N, N]
    """

    def __init__(self, config: QuantMoEConfig):
        super().__init__()
        self.config = config
        self.use_layer_summary = bool(getattr(config, "router_use_layer_summary", False))

        d_model = int(config.d_model)
        router_in = d_model * (2 if self.use_layer_summary else 1)

        # Optional: per-layer market summary (same for the daily cross-section batch)
        # summary = proj([mean(stock_repr), std(stock_repr)]) where stock_repr = mean over factor tokens
        self.layer_summary_proj = None
        if self.use_layer_summary:
            self.layer_summary_proj = nn.Linear(2 * d_model, d_model, bias=False)

        self.router = nn.Sequential(
            nn.Linear(router_in, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
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

    def forward(
        self,
        x: torch.Tensor,
        regime_embedding: torch.Tensor,
        attn_bias=None,
        return_attn: bool = False,
    ):
        B, T, N, D = x.shape
        residual = x

        x = self.norm1(x)

        # Build router input
        router_input = regime_embedding
        if self.use_layer_summary:
            # x: [B,T,N,D] after norm; summarize market state at this layer
            # - last time step -> [B,N,D]
            # - per-stock factor pooling -> [B,D]
            # - market mean/std over stocks -> [D],[D] (broadcast back to [B,D])
            h_last = x[:, -1, :, :]  # [B,N,D]
            per_stock = h_last.mean(dim=1)  # [B,D]
            m_mean = per_stock.mean(dim=0)
            m_std = per_stock.std(dim=0, unbiased=False)
            summary = torch.cat([m_mean, m_std], dim=-1)  # [2D]
            summary = self.layer_summary_proj(summary).unsqueeze(0).expand(B, -1)  # [B,D]
            router_input = torch.cat([regime_embedding, summary], dim=-1)  # [B,2D]

        # Router exploration: optional logit noise + temperature scaling
        # - noise encourages exploration early in training
        # - temperature controls sharpness (lower => sharper)
        router_logits_raw = self.router(router_input)  # [B, 2]
        router_logits = router_logits_raw
        noise_std = float(getattr(self.config, "router_noise", 0.0) or 0.0)
        if self.training and noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * noise_std

        temperature = float(getattr(self.config, "router_temperature", 1.0) or 1.0)
        temperature = max(temperature, 1e-3)
        gate_weights = F.softmax(router_logits / temperature, dim=-1)  # [B, 2]

        # diagnostics
        # z-loss is computed on raw (un-noised, un-tempered) logits for stability
        z_loss = (torch.logsumexp(router_logits_raw, dim=-1) ** 2).mean()
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
        if return_attn:
            out_time, attn_time = self.time_expert(h_time, bias_time, return_attn=True)
        else:
            out_time = self.time_expert(h_time, bias_time, return_attn=False)
            attn_time = None
        out_time = rearrange(out_time, "(b n) t d -> b t n d", b=B, n=N)

        # 2) Factor expert: per-time cross-sectional modeling
        h_factor = rearrange(x, "b t n d -> (b t) n d")
        if return_attn:
            out_factor, attn_factor = self.factor_expert(h_factor, bias_factor, return_attn=True)
        else:
            out_factor = self.factor_expert(h_factor, bias_factor, return_attn=False)
            attn_factor = None
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

        attn_dict = None
        if return_attn and (attn_time is not None or attn_factor is not None):
            attn_dict = {"time": attn_time, "factor": attn_factor}

        return x, diag, attn_dict
