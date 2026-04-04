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

        # [Large Batch Safety]
        # Calculate memory footprint of the attention matrix: B * n_heads * L * L * 2 (bytes/half)
        # If this exceeds reasonable limits (e.g. 1GB or 2^30 elements), we must chunk.
        # Current threshold: ~0.5 billion elements to stay safe from int32 overflow and OOM.
        # 500,000,000 elements * 2 bytes = 1GB per chunk.
        
        # Also check just the input size (B * L * D) to be safe.
        
        max_elements = 500_000_000  # Conservative limit for memory (int32 / OOM)
        max_batch_dim = 10000       # Conservative limit for CUDA grid Y dimension (allows n_heads up to ~6)

        attn_elements = B * self.n_heads * L * L
        
        # Chunking needed if EITHER memory or grid limit is threatened
        if (attn_elements > max_elements or B > max_batch_dim) and B > 1:
            # Determine chunk size
            # 1. Memory constraint
            denom = self.n_heads * L * L
            chunk_b_mem = max(1, int(max_elements / denom))
            
            # 2. Grid constraint
            chunk_b_grid = max_batch_dim
            
            # Combined
            chunk_b = min(chunk_b_mem, chunk_b_grid, B)

            # print(f">>> [ParallelAttention] Chunking large batch: B={B} -> chunks of {chunk_b}")

            out_list = []
            attn_list = []
            
            for start in range(0, B, chunk_b):
                end = min(start + chunk_b, B)
                x_chunk = x[start:end]
                
                # Slice attn_bias if it matches B dimension (not common for ALiBi, but good for robustness)
                bias_chunk = None
                if attn_bias is not None:
                    # Logic matches _build_attn_mask logic for checking B dimension
                    # attn_bias could be 3D [B, L, L] or 4D [B, H, L, L]
                    if attn_bias.ndim == 3 and attn_bias.shape[0] == B:
                        bias_chunk = attn_bias[start:end]
                    elif attn_bias.ndim == 4 and attn_bias.shape[0] == B:
                        bias_chunk = attn_bias[start:end]
                    else:
                        # Broadcastable (1, H, ...) or (L, L) -> keep as is
                        bias_chunk = attn_bias
                
                o_c, a_c = self._forward_single_chunk(x_chunk, bias_chunk, return_attn=return_attn)
                out_list.append(o_c)
                if return_attn:
                     attn_list.append(a_c)
            
            out = torch.cat(out_list, dim=0)
            attn = torch.cat(attn_list, dim=0) if return_attn else None
            
            if return_attn:
                return out, attn
            return out

        # No chunking needed
        out, attn = self._forward_single_chunk(x, attn_bias, return_attn=return_attn)
        if return_attn:
            return out, attn
        return out

    def _forward_single_chunk(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None,
        return_attn: bool = False,
    ):
        B, L, D = x.shape
        attn_mask = None
        if attn_bias is not None:
            attn_mask = self._build_attn_mask(attn_bias, B, L, x.dtype, x.device)

        # PyTorch MHA: returns (attn_output, attn_weights or None)
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
                # Older torch compatibility
                out, attn_avg = self.mha(
                    x, x, x,
                    attn_mask=attn_mask,
                    need_weights=True,
                )
                return out, attn_avg.unsqueeze(1)

        out, _ = self.mha(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return out, None
