# module/architecture/parallel_attention.py

import os

import torch
import torch.nn as nn
from module.utils.model_configuration import QuantMoEConfig

# CUDA grid-dim limit (2^16-1). PyTorch's fused / memory-efficient SDPA kernels launch a grid whose
# x-dimension scales with the MHA batch size; above this they raise "invalid configuration argument"
# (empirically B=63200 works, B=94800 fails on an RTX 40-series). The time expert reshapes to
# batch = B*N, so a wide feature set (e.g. L-4 CS-rank append => N=316, batch=300 => 94800) trips it
# while the default N=158 (47400) does not.
#
# Fix: when the batch exceeds the limit we split the attention over the batch dimension into chunks
# <= _SDPA_CHUNK and run each through the normal (memory-efficient) backend, then concatenate. Because
# self-attention is independent per batch element this is numerically identical to a single call, so
# existing (narrower) runs are byte-unchanged and we do NOT fall back to the memory-heavy MATH kernel
# (which materializes the full LxL score matrix per element and caused OOM at N=316, batch=300).
_SDPA_GRID_LIMIT = 65535
# 4096 keeps the attention fwd+bwd working set minimal for the B*N=94800 time expert (measured ~5GB
# reserved already at 8192; smaller only trims further), vs ~9GB at 32768 (which tips a 12GB card over
# once the rest of the model + optimizer + allocator fragmentation pile on) and ~23GB for the MATH
# kernel. Smaller = more (cheap, seq-len-8) kernel launches but strictly less peak memory and less
# fragmentation churn; equivalence to a single call is exact (self-attn is independent per batch
# element). Overridable via QIB_SDPA_CHUNK for cards with more/less headroom.
_SDPA_CHUNK = int(os.environ.get("QIB_SDPA_CHUNK", "4096"))


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

        # Guard the CUDA SDPA grid-dim limit (see module header). When the batch exceeds the limit we
        # split over the batch dim and run each chunk through the normal backend; self-attention is
        # independent per batch element so this is numerically identical to a single call and keeps
        # the memory-efficient kernel (no MATH fallback -> no OOM). attn_mask is [B*H, L, L], so it is
        # chunked in lockstep by the same batch factor.
        if x.is_cuda and B > _SDPA_GRID_LIMIT:
            H = self.n_heads
            outs = []
            for start in range(0, B, _SDPA_CHUNK):
                end = min(start + _SDPA_CHUNK, B)
                xc = x[start:end]
                mc = None
                if attn_mask is not None:
                    mc = attn_mask[start * H:end * H]
                oc, _ = self.mha(xc, xc, xc, attn_mask=mc, need_weights=False)
                outs.append(oc)
            return torch.cat(outs, dim=0)

        out, _ = self.mha(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return out
