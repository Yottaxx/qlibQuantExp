"""
Attention-based Pooling Module for Factor Aggregation.

This module uses PyTorch's official MultiheadAttention implementation
for aggregating factor representations into stock-level scores.
"""
import torch
import torch.nn as nn
from module.architecture import RMSNorm


class AttentionPooling(nn.Module):
    """
    Multi-head Attention Pooling for factor aggregation.
    
    Uses PyTorch's nn.MultiheadAttention with a learnable query vector
    to attend over factor representations, providing adaptive weighting
    that can focus on relevant factors.
    
    Args:
        d_model: Hidden dimension of factor representations
        n_heads: Number of attention heads (default: 1 for simplicity)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Learnable query vector for pooling (will be expanded to batch size)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Use PyTorch's official MultiheadAttention implementation
        # This handles Q/K/V projections, scaled dot-product attention, and output projection internally
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=False,  # No bias for cleaner implementation
            batch_first=True,  # Use [B, L, D] format
            add_zero_attn=False,  # Don't add zero attention
        )
        
        # Output normalization
        self.norm = RMSNorm(d_model)
        
        # Initialize query with larger random values to break symmetry and prevent attention collapse
        nn.init.normal_(self.query, std=1.0)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] factor representations (B=batch, N=num_factors, D=d_model)
            
        Returns:
            pooled: [B, D] pooled representation
            attention_weights: [B, N] attention weights (for interpretability)
        """
        B, N, D = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"
        
        # Expand learnable query to batch size: [1, 1, D] -> [B, 1, D]
        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        
        # Use PyTorch's MultiheadAttention
        # Query: learnable query vector [B, 1, D]
        # Key & Value: factor representations [B, N, D]
        # Note: MHA internally handles Q/K/V projections and scaled dot-product attention
        attn_output, attn_weights = self.mha(
            query=q,  # [B, 1, D]
            key=x,    # [B, N, D]
            value=x,  # [B, N, D]
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights for better interpretability
        )
        # attn_output: [B, 1, D]
        # attn_weights: [B, n_heads, 1, N] (if average_attn_weights=False) or [B, 1, N] (if True)
        
        # Note: MHA already performs scaled dot-product attention with scale = sqrt(d_head),
        # which provides regularization effects similar to temperature scaling.
        
        # Squeeze output: [B, 1, D] -> [B, D]
        pooled = attn_output.squeeze(1)  # [B, D]
        pooled = self.norm(pooled)
        
        # Extract attention weights for interpretability
        # Average over heads if multi-head, squeeze query dimension
        if attn_weights.dim() == 4:  # [B, H, 1, N] - per-head weights
            attention_weights = attn_weights.squeeze(2).mean(dim=1)  # [B, N]
        elif attn_weights.dim() == 3:  # [B, 1, N] - averaged weights
            attention_weights = attn_weights.squeeze(1)  # [B, N]
        else:  # Fallback: [B, N] - already squeezed
            attention_weights = attn_weights
        
        return pooled, attention_weights


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling with multiple strategies.
    
    Combines attention pooling with residual mean pooling for robustness.
    """
    
    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for attention pooling (1-alpha) for mean pooling
        """
        super().__init__()
        self.attention_pool = AttentionPooling(d_model, n_heads, dropout)
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D]
            
        Returns:
            pooled: [B, D]
            attention_weights: [B, N]
        """
        attn_pooled, attn_weights = self.attention_pool(x)
        mean_pooled = x.mean(dim=1)  # [B, D]
        
        # Weighted combination
        pooled = self.alpha * attn_pooled + (1 - self.alpha) * mean_pooled
        
        return pooled, attn_weights

