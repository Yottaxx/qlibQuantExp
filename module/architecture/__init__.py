import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Preserves direction and relative magnitude better than LayerNorm by skipping mean subtraction.
    Commonly used in modern MoE and Transformer architectures.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # Calculate RMS: sqrt(mean(x^2))
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale
