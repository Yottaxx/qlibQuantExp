import torch
import torch.nn as nn


class RegimeAdaptivePooling(nn.Module):
    """
    Regime-aware factor pooling with optional hierarchical global/local conditioning.
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
        summary_source: str = "none",
        *,
        use_hierarchical_state_field: bool = False,
        d_global_state: int | None = None,
        d_local_state: int | None = None,
        use_global_state: bool = True,
        use_local_state: bool = True,
        state_fusion_mode: str = "sum_norm",
        local_pooling_scale_init: float = 0.1,
        local_pooling_alpha_scale_init: float = 0.1,
        local_scale_learnable: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pooling_mode = str(pooling_mode or "adaptive_alpha").strip().lower()
        self.base_alpha = float(base_alpha)
        self.alpha_scale = float(alpha_scale)
        self.summary_source = str(summary_source or "none").strip().lower()
        self.use_hierarchical_state_field = bool(use_hierarchical_state_field)
        self.use_global_state = bool(use_global_state)
        self.use_local_state = bool(use_local_state)
        self.d_global_state = int(d_global_state if d_global_state is not None else d_model)
        self.d_local_state = int(d_local_state if d_local_state is not None else d_model)
        self.state_fusion_mode = str(state_fusion_mode or "sum_norm").strip().lower()
        if self.state_fusion_mode not in {"sum_norm", "branch_mlp_v1", "bounded_sum_v1"}:
            raise ValueError(f"Unsupported state_fusion_mode: {self.state_fusion_mode}")
        self.use_branch_fusion = self.use_hierarchical_state_field and self.state_fusion_mode == "branch_mlp_v1"
        self.use_bounded_sum = self.use_hierarchical_state_field and self.state_fusion_mode == "bounded_sum_v1"

        valid_sources = {"none", "batch", "day_asset"}
        if self.summary_source not in valid_sources:
            raise ValueError(f"summary_source must be one of {valid_sources}, got {self.summary_source}")
        self.use_summary_context = self.summary_source != "none"

        d_ff = pooling_d_ff if pooling_d_ff is not None else d_model
        valid_modes = {"static", "adaptive_alpha", "conditioned_query", "full"}
        if self.pooling_mode not in valid_modes:
            raise ValueError(f"pooling_mode must be one of {valid_modes}, got {self.pooling_mode}")

        if self.use_hierarchical_state_field:
            self.global_proj = nn.Linear(self.d_global_state, d_model, bias=False) if self.use_global_state else None
            self.local_proj = nn.Linear(self.d_local_state, d_model, bias=False) if self.use_local_state else None
            self.summary_proj = nn.Linear(d_model, d_model, bias=False) if self.use_summary_context else None
            self.cond_norm = nn.LayerNorm(d_model)
            self.cond_dim = d_model
            self.branch_fusion = None
            self.branch_day_norm = None
            self.local_pooling_scale = None
            self.local_pooling_alpha_scale = None
            if self.use_branch_fusion:
                self.branch_fusion = nn.Sequential(
                    nn.Linear(d_model * 4, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                )
                self.branch_day_norm = nn.LayerNorm(d_model)
            elif self.use_bounded_sum:
                pooling_init = torch.tensor(float(local_pooling_scale_init), dtype=torch.float32)
                alpha_init = torch.tensor(float(local_pooling_alpha_scale_init), dtype=torch.float32)
                if bool(local_scale_learnable):
                    self.local_pooling_scale = nn.Parameter(pooling_init)
                    self.local_pooling_alpha_scale = nn.Parameter(alpha_init)
                else:
                    if hasattr(self, "local_pooling_scale"):
                        delattr(self, "local_pooling_scale")
                    if hasattr(self, "local_pooling_alpha_scale"):
                        delattr(self, "local_pooling_alpha_scale")
                    self.register_buffer("local_pooling_scale", pooling_init)
                    self.register_buffer("local_pooling_alpha_scale", alpha_init)
        else:
            self.global_proj = None
            self.local_proj = None
            self.summary_proj = None
            self.cond_norm = None
            self.branch_fusion = None
            self.branch_day_norm = None
            self.local_pooling_scale = None
            self.local_pooling_alpha_scale = None
            self.cond_dim = d_model * 2 if self.use_summary_context else d_model

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
            add_zero_attn=False,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.use_dynamic_query = self.pooling_mode in {"conditioned_query", "full"}
        self.query_norm = None
        self.query_proj = None
        self.query_day_proj = None
        self.query_local_proj = None
        self.query_branch_proj = None
        self.query = None
        if self.use_dynamic_query:
            if self.use_branch_fusion:
                self.query_day_proj = nn.Linear(d_model, d_model, bias=False)
                self.query_local_proj = nn.Linear(d_model, d_model, bias=False)
                self.query_branch_proj = nn.Linear(d_model, d_model, bias=False)
            else:
                self.query_norm = nn.LayerNorm(self.cond_dim)
                self.query_proj = nn.Linear(self.cond_dim, d_model, bias=False)
        else:
            self.query = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.normal_(self.query, std=0.02)

        self.mean_norm = nn.LayerNorm(d_model)
        self.use_adaptive_alpha = self.pooling_mode in {"adaptive_alpha", "full"}
        self.alpha_head = None
        self.alpha_day_head = None
        self.alpha_local_head = None
        self.alpha_branch_head = None
        if self.use_adaptive_alpha:
            if self.use_branch_fusion:
                self.alpha_day_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.GELU(),
                    nn.Linear(d_model // 4, 1),
                )
                self.alpha_local_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.GELU(),
                    nn.Linear(d_model // 4, 1),
                )
                self.alpha_branch_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.GELU(),
                    nn.Linear(d_model // 4, 1),
                )
            else:
                self.alpha_head = nn.Sequential(
                    nn.Linear(self.cond_dim, d_model // 4),
                    nn.GELU(),
                    nn.Linear(d_model // 4, 1),
                )

    def _build_condition(
        self,
        global_state: torch.Tensor,
        local_state: torch.Tensor | None,
        summary_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor] | None]:
        if self.use_hierarchical_state_field:
            parts = []
            global_sensitivity = 0.0
            local_sensitivity = 0.0
            g_proj = None
            l_proj = None
            u_proj = None
            if self.use_global_state:
                if global_state is None:
                    raise ValueError("Hierarchical pooling expects global_state.")
                g_proj = self.global_proj(global_state)
                parts.append(g_proj)
                global_sensitivity = float(g_proj.detach().norm(dim=-1).mean().item())
            if self.use_local_state:
                if local_state is None:
                    raise ValueError("Hierarchical pooling expects local_state.")
                l_proj = self.local_proj(local_state)
                parts.append(l_proj)
                local_sensitivity = float(l_proj.detach().norm(dim=-1).mean().item())
            if self.use_summary_context:
                if summary_context is None:
                    raise ValueError(f"summary_source='{self.summary_source}' but summary_context is None")
                u_proj = self.summary_proj(summary_context)
                parts.append(u_proj)
            if not parts:
                raise ValueError("Hierarchical pooling received no active conditioning branch.")
            branches = None
            if self.use_branch_fusion:
                zero = parts[0].new_zeros(parts[0].shape)
                g = g_proj if g_proj is not None else zero
                l = l_proj if l_proj is not None else zero
                u = u_proj if u_proj is not None else zero
                cond_vector = self.branch_fusion(torch.cat([g, l, u, g * l], dim=-1))
                branches = {"global": g, "local": l, "summary": u}
            elif self.use_bounded_sum:
                zero = parts[0].new_zeros(parts[0].shape)
                g = g_proj if g_proj is not None else zero
                l = l_proj if l_proj is not None else zero
                u = u_proj if u_proj is not None else zero
                scale = self.local_pooling_scale.to(device=g.device, dtype=g.dtype)
                cond_vector = self.cond_norm(g + u + scale * l)
                branches = {"global": g, "local": l, "summary": u}
            else:
                cond_vector = self.cond_norm(torch.stack(parts, dim=0).sum(dim=0))
            diag = {
                "pool_global_sensitivity": float(global_sensitivity),
                "pool_local_sensitivity": float(local_sensitivity),
                "local_pooling_scale": (
                    float(self.local_pooling_scale.detach().item()) if self.local_pooling_scale is not None else 0.0
                ),
                "local_pooling_alpha_scale": (
                    float(self.local_pooling_alpha_scale.detach().item())
                    if self.local_pooling_alpha_scale is not None
                    else 0.0
                ),
            }
            return cond_vector, diag, branches

        if self.use_summary_context:
            if summary_context is None:
                raise ValueError(f"summary_source='{self.summary_source}' but summary_context is None")
            cond_vector = torch.cat([global_state, summary_context], dim=-1)
        else:
            cond_vector = global_state
        return cond_vector, {"pool_global_sensitivity": 0.0, "pool_local_sensitivity": 0.0}, None

    def forward(
        self,
        x: torch.Tensor,
        global_state: torch.Tensor,
        local_state: torch.Tensor | None = None,
        summary_context: torch.Tensor | None = None,
        pooling_mode_override: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        B, N, _ = x.shape
        cond_vector, diag, branches = self._build_condition(global_state, local_state, summary_context)
        diag.setdefault("query_local_norm", 0.0)
        diag.setdefault("query_branch_norm", 0.0)
        diag.setdefault("alpha_local_logit_std", 0.0)
        diag.setdefault("alpha_branch_logit_std", 0.0)

        if self.use_dynamic_query:
            if self.use_branch_fusion:
                if branches is None or self.branch_day_norm is None:
                    raise RuntimeError("branch_mlp_v1 pooling is missing branch projections.")
                day_cond = self.branch_day_norm(branches["global"] + branches["summary"])
                query_day = self.query_day_proj(day_cond)
                query_local = self.query_local_proj(branches["local"])
                query_branch = self.query_branch_proj(cond_vector)
                query = (query_day + query_branch + query_local).unsqueeze(1)
                diag["query_local_norm"] = float(query_local.detach().norm(dim=-1).mean().item())
                diag["query_branch_norm"] = float(query_branch.detach().norm(dim=-1).mean().item())
            elif self.use_bounded_sum and branches is not None:
                q_in = self.query_norm(cond_vector)
                query = self.query_proj(q_in).unsqueeze(1)
                scale = self.local_pooling_scale.to(device=cond_vector.device, dtype=cond_vector.dtype)
                diag["query_local_norm"] = float((scale * branches["local"]).detach().norm(dim=-1).mean().item())
            else:
                q_in = self.query_norm(cond_vector)
                query = self.query_proj(q_in).unsqueeze(1)
        else:
            query = self.query.expand(B, -1, -1)

        attn_output, attn_weights_raw = self.mha(
            query=query,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
        )

        if attn_weights_raw.dim() == 4:
            attention_weights = attn_weights_raw.squeeze(2).mean(dim=1)
        else:
            attention_weights = attn_weights_raw.squeeze(1)

        attn_out = query + attn_output  # residual from query
        attn_out = self.attn_norm(attn_out)
        attn_out = attn_out + self.ffn(self.ffn_norm(attn_out))
        attn_pooled = attn_out.squeeze(1)

        mean_pooled = self.mean_norm(x.mean(dim=1))

        if self.use_adaptive_alpha:
            if self.use_branch_fusion:
                if branches is None or self.branch_day_norm is None:
                    raise RuntimeError("branch_mlp_v1 pooling is missing branch projections.")
                day_cond = self.branch_day_norm(branches["global"] + branches["summary"])
                alpha_local_logit = self.alpha_local_head(branches["local"])
                alpha_branch_logit = self.alpha_branch_head(cond_vector)
                alpha_logit = self.alpha_day_head(day_cond) + alpha_branch_logit + alpha_local_logit
                diag["alpha_local_logit_std"] = float(alpha_local_logit.detach().std(unbiased=False).item())
                diag["alpha_branch_logit_std"] = float(alpha_branch_logit.detach().std(unbiased=False).item())
            elif self.use_bounded_sum and branches is not None:
                scale = self.local_pooling_alpha_scale.to(device=cond_vector.device, dtype=cond_vector.dtype)
                alpha_cond = self.cond_norm(branches["global"] + branches["summary"] + scale * branches["local"])
                alpha_day_cond = self.cond_norm(branches["global"] + branches["summary"])
                alpha_logit = self.alpha_head(alpha_cond)
                alpha_day_logit = self.alpha_head(alpha_day_cond)
                diag["alpha_local_logit_std"] = float(
                    (alpha_logit - alpha_day_logit).detach().std(unbiased=False).item()
                )
            else:
                alpha_logit = self.alpha_head(cond_vector)
            alpha_delta = torch.tanh(alpha_logit)
            alpha = self.base_alpha + self.alpha_scale * alpha_delta
            alpha = alpha.clamp(0.0, 1.0)
        else:
            alpha = x.new_full((B, 1), self.base_alpha)

        override = str(pooling_mode_override or "").strip().lower()
        if override:
            if override == "attn_only":
                pooled = attn_pooled
            elif override == "mean_only":
                pooled = mean_pooled
            else:
                raise ValueError(f"Unsupported pooling_mode_override: {pooling_mode_override}")
        else:
            pooled = alpha * attn_pooled + (1 - alpha) * mean_pooled
        return pooled, attention_weights, alpha, diag
