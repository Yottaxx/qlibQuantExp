# module/architecture/parallel_attention.py

import torch
import torch.nn as nn
from module.utils.model_configuration import QuantMoEConfig


class ParallelAttention(nn.Module):
    """Official multi-head self-attention (PyTorch).

    Used twice:
      - Time expert:   sequence length = T, batch = B*N
      - Factor expert: sequence length = N, batch = B*T

    API:
      x: [B, L, D]
      attn_bias: additive bias, expected broadcastable to [B, H, L, L] or [1, H, L, L] or [L, L]
      return_attn: if True, also return attn weights [B, H, L, L] (or best-effort if torch version is old)
    """

    def __init__(self, config: QuantMoEConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = int(config.n_heads)
        self.d_model = int(config.d_model)

        # Official implementation: in-proj + out-proj + dropout handled inside
        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=float(config.dropout),
            bias=False,
            batch_first=True,  # so we can keep x as [B, L, D]
        )

    def _build_attn_mask(
        self,
        attn_bias: torch.Tensor,
        B: int,
        L: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert additive attn_bias into PyTorch MHA attn_mask.
        PyTorch supports:
          - 2D: [L, L]
          - 3D: [B*H, L, L] (per batch and per head)
        """
        bias = attn_bias.to(device=device, dtype=dtype)

        if bias.dim() == 2:
            # [L, L]
            if bias.shape != (L, L):
                raise ValueError(f"attn_bias 2D must be [L,L]={L,L}, got {tuple(bias.shape)}")
            return bias.contiguous()

        if bias.dim() == 3:
            # could be [H, L, L] or [B, L, L]
            if bias.shape[-2:] != (L, L):
                raise ValueError(f"attn_bias 3D last dims must be [L,L]={L,L}, got {tuple(bias.shape)}")

            if bias.shape[0] == self.n_heads:
                # [H,L,L] -> [B*H,L,L]
                bias = bias.unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,L,L]
                return bias.reshape(B * self.n_heads, L, L).contiguous()

            if bias.shape[0] == B:
                # [B,L,L] -> [B*H,L,L]
                bias = bias.unsqueeze(1).expand(B, self.n_heads, L, L)  # [B,H,L,L]
                return bias.reshape(B * self.n_heads, L, L).contiguous()

            raise ValueError(
                f"attn_bias 3D first dim must be H={self.n_heads} or B={B}, got {bias.shape[0]}"
            )

        if bias.dim() == 4:
            # [1,H,L,L] or [B,H,L,L] or broadcastable variants
            if bias.shape[-2:] != (L, L):
                raise ValueError(f"attn_bias 4D last dims must be [L,L]={L,L}, got {tuple(bias.shape)}")

            # Expand batch if needed
            if bias.shape[0] == 1:
                bias = bias.expand(B, -1, -1, -1)  # [B,?,L,L]
            elif bias.shape[0] != B:
                raise ValueError(f"attn_bias 4D first dim must be 1 or B={B}, got {bias.shape[0]}")

            # Expand heads if needed
            if bias.shape[1] == 1:
                bias = bias.expand(B, self.n_heads, L, L)
            elif bias.shape[1] != self.n_heads:
                raise ValueError(f"attn_bias 4D head dim must be 1 or H={self.n_heads}, got {bias.shape[1]}")

            return bias.reshape(B * self.n_heads, L, L).contiguous()

        raise ValueError(f"attn_bias must be 2D/3D/4D tensor, got dim={bias.dim()}")

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        # x: [B, L, D]
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim D={self.d_model}, got {D}")

        attn_mask = None
        if attn_bias is not None:
            attn_mask = self._build_attn_mask(attn_bias, B, L, x.dtype, x.device)

        # PyTorch MHA: returns (attn_output, attn_weights or None)
        # - need_weights=True to get weights
        # - average_attn_weights=False to get [B, H, L, L] (if supported)
        if return_attn:
            try:
                out, attn = self.mha(
                    x, x, x,
                    attn_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                # attn: [B, H, L, L]
                return out, attn
            except TypeError:
                # Older torch may not have average_attn_weights; returns [B, L, L] averaged over heads
                out, attn_avg = self.mha(
                    x, x, x,
                    attn_mask=attn_mask,
                    need_weights=True,
                )
                # best-effort: expand to [B, 1, L, L]
                return out, attn_avg.unsqueeze(1)

        out, _ = self.mha(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return out
