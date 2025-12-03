# module/architecture/parallel_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from module.utils.model_configuration import QuantMoEConfig


class ParallelAttention(nn.Module):
    """Multi-head self-attention for a sequence.

    Used twice:
      - Time expert:  sequence length = T, batch = B*N
      - Factor expert: sequence length = N, batch = B*T
    """

    def __init__(self, config: QuantMoEConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = int(config.n_heads)
        self.head_dim = int(config.d_model // config.n_heads)
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        # x: [B, L, D]
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.n_heads)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if attn_bias is not None:
            # expected: [1, H, L, L] or broadcastable
            scores = scores + attn_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)  # [B, H, L, L]

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.out_proj(out)

        if return_attn:
            return out, attn  # attn: [B, H, L, L]
        return out
