import math
import torch
from einops import rearrange


def get_slopes(n_heads: int):
    """ALiBi slopes from the original formulation (stable for any n_heads)."""

    def _power_of_2_slopes(n: int):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return _power_of_2_slopes(n_heads)

    # closest power of two
    closest = 2 ** math.floor(math.log2(n_heads))
    slopes = _power_of_2_slopes(closest)
    extra = get_slopes(2 * closest)[0::2][: n_heads - closest]
    return slopes + extra


def build_bidirectional_alibi_bias(batch_size: int, seq_len: int, n_heads: int, device: torch.device):
    """Bidirectional ALiBi bias based on |i - j|.

    Returns shape: [1, H, L, L] (broadcastable to [B, H, L, L]).
    `batch_size` is kept for backward compatibility (unused).
    """
    pos = torch.arange(seq_len, device=device)
    rel = torch.abs(pos[:, None] - pos[None, :]).to(torch.float32)  # [L, L]

    slopes = torch.tensor(get_slopes(n_heads), device=device, dtype=torch.float32)
    slopes = rearrange(slopes, "h -> 1 h 1 1")  # [1, H, 1, 1]

    return -slopes * rel[None, None, :, :]  # [1, H, L, L]
