import torch
import torch.nn as nn

class RegimeAdaptivePooling(nn.Module):
    """
    Regime-aware factor pooling strategy that unifies Scheme A (Adaptive Alpha) 
    and Scheme B (Conditioned Query) into a single flexible module.

    Modes:
    - "static": Standard attention pooling with static query and fixed alpha.
    - "adaptive_alpha": Static query, but alpha (attn vs mean weight) is regime-dependent.
    - "conditioned_query": Query is generated from regime, alpha is fixed.
    - "full": Both query and alpha are regime-dependent.

    Scheme A (Adaptive Alpha):
        alpha = base_alpha + scale * sigmoid(MLP(regime))
        pooled = alpha * attn_pooled + (1 - alpha) * mean_pooled

    Scheme B (Conditioned Query):
        query = MLP(regime)
        attn_output = MHA(query, x, x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        pooling_mode: str = "adaptive_alpha",
        base_alpha: float = 0.7,
        alpha_scale: float = 0.3,
        pooling_d_ff: int | None = None,
        use_layer_summary: bool = False,  # New: Condition on layer summary
    ):
        """
        Args:
            d_model: Feature dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
            pooling_mode: One of ["static", "adaptive_alpha", "conditioned_query", "full"].
            base_alpha: Base weight for attention pooling (default 0.7).
            alpha_scale: Scaling factor for adaptive alpha adjustment (default 0.3).
                         Effective alpha range: [base - scale, base + scale].
            pooling_d_ff: Dimension of Feed Forward Network after attention. 
                          If None, defaults to d_model.
            use_layer_summary: If True, condtions query/alpha on concatenation of 
                               [regime_embedding, layer_summary] (dim=2*d_model).
        """
        super().__init__()
        self.d_model = d_model
        self.pooling_mode = pooling_mode.lower()
        self.base_alpha = base_alpha
        self.alpha_scale = alpha_scale
        self.use_layer_summary = use_layer_summary
        
        d_ff = pooling_d_ff if pooling_d_ff is not None else d_model
        
        # Input dimension for regime-dependent modules
        # If use_layer_summary is True, we concatenate regime (D) + summary (D) -> 2D
        self.cond_dim = d_model * 2 if use_layer_summary else d_model

        valid_modes = {"static", "adaptive_alpha", "conditioned_query", "full"}
        if self.pooling_mode not in valid_modes:
            raise ValueError(f"pooling_mode must be one of {valid_modes}, got {self.pooling_mode}")

        # --- Components for Attention Mechanism ---
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
            add_zero_attn=False,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # --- New: Feed Forward Network ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Implementation Logic:
        # If static query (static / adaptive_alpha): learnable parameter
        # If dynamic query (conditioned_query / full): projection from regime
        
        self.use_dynamic_query = self.pooling_mode in {"conditioned_query", "full"}
        if self.use_dynamic_query:
            # Scheme B: Regime -> Query
            self.query_norm = nn.LayerNorm(self.cond_dim)
            self.query_proj = nn.Linear(self.cond_dim, d_model, bias=False)
        else:
            # Static: Learnable parameter
            self.query = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.normal_(self.query, std=0.02)

        # --- Components for Alpha Mechanism ---
        self.mean_norm = nn.LayerNorm(d_model)

        self.use_adaptive_alpha = self.pooling_mode in {"adaptive_alpha", "full"}
        if self.use_adaptive_alpha:
            # Scheme A: Regime -> Alpha adjustment
            self.alpha_head = nn.Sequential(
                nn.Linear(self.cond_dim, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
            )
        else:
            # Static alpha is handled in forward logic directly using self.base_alpha
            pass

    def forward(
        self, 
        x: torch.Tensor, 
        regime_embedding: torch.Tensor,
        layer_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] factor representations
            regime_embedding: [B, D] regime context vector
            layer_summary: [B, D] optional layer summary stats (if use_layer_summary=True)

        Returns:
            pooled: [B, D] aggregated representation
            attention_weights: [B, N] (averaged over heads if n_heads > 1)
            alpha: [B, 1] applied alpha values
        """
        B, N, D = x.shape
        
        # Prepare Conditioning Vector
        if self.use_layer_summary:
            if layer_summary is None:
                raise ValueError("use_layer_summary=True but layer_summary is None")
            # Concatenate [Regime, Summary] -> [B, 2D]
            cond_vector = torch.cat([regime_embedding, layer_summary], dim=-1)
        else:
            cond_vector = regime_embedding
        
        # 1. Query Preparation
        if self.use_dynamic_query:
            # Dynamic Query from Cond Vector
            q_in = self.query_norm(cond_vector)
            query = self.query_proj(q_in).unsqueeze(1) # [B, 1, D]
        else:
            # Static Learnable Query
            query = self.query.expand(B, -1, -1) # [B, 1, D]

        # 2. Attention Pooling
        # Need weights for interpretability
        attn_output, attn_weights_raw = self.mha(
            query=query,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,  # Get per-head [B, H, 1, N]
        )
        
        # Process attention weights for return [B, N]
        if attn_weights_raw.dim() == 4:
            attention_weights = attn_weights_raw.squeeze(2).mean(dim=1) # [B, N]
        else:
            attention_weights = attn_weights_raw.squeeze(1)

        # Apply Norm to MHA output (no residual for cross-attention)
        attn_out = self.attn_norm(attn_output)  # [B, 1, D]
        
        # Pre-LN FFN with residual
        attn_out = attn_out + self.ffn(self.ffn_norm(attn_out))
        
        attn_pooled = attn_out.squeeze(1) # [B, D]

        # 3. Mean Pooling (Fallback)
        mean_pooled = self.mean_norm(x.mean(dim=1)) # [B, D]

        # 4. Alpha Calculation
        if self.use_adaptive_alpha:
            # Adaptive Alpha
            alpha_logit = self.alpha_head(cond_vector) # [B, 1]
            # Map logit to [-1, 1] range via tanh, or [0, 1] via sigmoid?
            # Analysis suggested: base_alpha + scale * (sigmoid(eps) - 0.5) * 2
            # Let's use a cleaner approach:
            # sigmoid(x) -> [0, 1]. (2*sig - 1) -> [-1, 1].
            # alpha = base + scale * delta
            alpha_delta = torch.tanh(alpha_logit) # [-1, 1]
            alpha = self.base_alpha + self.alpha_scale * alpha_delta
            
            # Clamp for safety, though design usually keeps it safe.
            # Avoid negative or >1 alpha if base+scale > 1 or base-scale < 0
            alpha = alpha.clamp(0.0, 1.0)
        else:
            # Static Alpha (efficient tensor creation)
            alpha = x.new_full((B, 1), self.base_alpha)

        # 5. Fusion
        pooled = alpha * attn_pooled + (1 - alpha) * mean_pooled
        
        return pooled, attention_weights, alpha
