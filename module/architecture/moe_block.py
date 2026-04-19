import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from module.architecture.parallel_attention import ParallelAttention
from module.utils.model_configuration import QuantMoEConfig


class RegimeAdaptiveMoEBlock(nn.Module):
    """Regime-adaptive MoE block with optional hierarchical conditioning."""

    def __init__(self, config: QuantMoEConfig):
        super().__init__()
        self.config = config
        self.use_hierarchical_state_field = bool(getattr(config, "use_hierarchical_state_field", False))
        self.router_summary_source = str(getattr(config, "router_summary_source", "none") or "none").strip().lower()
        self.use_summary_context = self.router_summary_source != "none"
        self.router_use_global_state = bool(getattr(config, "router_use_global_state", True))
        self.router_use_local_state = bool(getattr(config, "router_use_local_state", True))
        self.state_fusion_mode = str(getattr(config, "state_fusion_mode", "sum_norm") or "sum_norm").strip().lower()
        if self.state_fusion_mode not in {"sum_norm", "branch_mlp_v1", "bounded_sum_v1"}:
            raise ValueError(f"Unsupported state_fusion_mode: {self.state_fusion_mode}")
        self.use_bounded_sum = self.use_hierarchical_state_field and self.state_fusion_mode == "bounded_sum_v1"

        d_model = int(config.d_model)
        d_global_state = int(getattr(config, "d_global_state", d_model) or d_model)
        d_local_state = int(getattr(config, "d_local_state", d_model) or d_model)

        self.layer_summary_proj = None
        if self.router_summary_source == "batch":
            self.layer_summary_proj = nn.Linear(2 * d_model, d_model, bias=False)

        if self.use_hierarchical_state_field:
            self.global_router_proj = (
                nn.Linear(d_global_state, d_model, bias=False) if self.router_use_global_state else None
            )
            self.local_router_proj = (
                nn.Linear(d_local_state, d_model, bias=False) if self.router_use_local_state else None
            )
            self.summary_router_proj = (
                nn.Linear(d_model, d_model, bias=False) if self.use_summary_context else None
            )
            self.router_cond_norm = nn.LayerNorm(d_model)
            self.router_day_norm = nn.LayerNorm(d_model) if self.state_fusion_mode == "branch_mlp_v1" else None
            self.router_branch_fusion = None
            self.router_day = None
            self.router_local_logits = None
            self.local_router_scale = None
            if self.state_fusion_mode == "branch_mlp_v1":
                self.router_branch_fusion = nn.Sequential(
                    nn.Linear(d_model * 4, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                )
                self.router_day = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 2),
                )
                self.router_local_logits = nn.Linear(d_model, 2, bias=False)
            elif self.use_bounded_sum:
                init = torch.tensor(float(getattr(config, "local_router_scale_init", 0.1)), dtype=torch.float32)
                if bool(getattr(config, "local_scale_learnable", True)):
                    self.local_router_scale = nn.Parameter(init)
                else:
                    if hasattr(self, "local_router_scale"):
                        delattr(self, "local_router_scale")
                    self.register_buffer("local_router_scale", init)
            router_in = d_model
        else:
            self.global_router_proj = None
            self.local_router_proj = None
            self.summary_router_proj = None
            self.router_cond_norm = None
            self.router_day_norm = None
            self.router_branch_fusion = None
            self.router_day = None
            self.router_local_logits = None
            self.local_router_scale = None
            router_in = d_model * (2 if self.use_summary_context else 1)

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

    def _resolve_summary_context(self, x: torch.Tensor, summary_context: torch.Tensor | None) -> torch.Tensor | None:
        B, _, N, D = x.shape
        if self.router_summary_source == "batch":
            h_last = x[:, -1, :, :]
            per_stock = h_last.mean(dim=1)
            m_mean = per_stock.mean(dim=0)
            m_std = per_stock.std(dim=0, unbiased=False)
            batch_summary = torch.cat([m_mean, m_std], dim=-1)
            return self.layer_summary_proj(batch_summary).unsqueeze(0).expand(B, -1)
        if self.router_summary_source == "day_asset":
            if summary_context is None:
                raise ValueError("router_summary_source='day_asset' but summary_context is None")
            return summary_context
        if self.router_summary_source == "none":
            return None
        raise ValueError(f"Unsupported router_summary_source: {self.router_summary_source}")

    def forward(
        self,
        x: torch.Tensor,
        global_state: torch.Tensor,
        local_state: torch.Tensor | None = None,
        attn_bias=None,
        return_attn: bool = False,
        *,
        summary_context: torch.Tensor | None = None,
        factor_film: tuple[torch.Tensor, torch.Tensor] | None = None,
        feature_mask: torch.Tensor | None = None,
        expert_mode: str | None = None,
    ):
        B, T, N, D = x.shape
        residual = x
        x = self.norm1(x)

        summary_context = self._resolve_summary_context(x, summary_context)
        router_global_sensitivity = 0.0
        router_local_sensitivity = 0.0
        router_local_logit_std = 0.0
        local_router_scale_value = 0.0

        if self.use_hierarchical_state_field:
            cond_parts = []
            g_router = None
            l_router = None
            u_router = None
            if self.router_use_global_state:
                if global_state is None:
                    raise ValueError("Hierarchical router expects global_state.")
                g_router = self.global_router_proj(global_state)
                cond_parts.append(g_router)
                router_global_sensitivity = float(g_router.detach().norm(dim=-1).mean().item())
            if self.router_use_local_state:
                if local_state is None:
                    raise ValueError("Hierarchical router expects local_state.")
                l_router = self.local_router_proj(local_state)
                cond_parts.append(l_router)
                router_local_sensitivity = float(l_router.detach().norm(dim=-1).mean().item())
            if summary_context is not None and self.summary_router_proj is not None:
                u_router = self.summary_router_proj(summary_context)
                cond_parts.append(u_router)
            if not cond_parts:
                raise ValueError("Hierarchical router received no active conditioning branch.")
            if self.state_fusion_mode == "branch_mlp_v1":
                zero = x.new_zeros(B, D)
                g = g_router if g_router is not None else zero
                l = l_router if l_router is not None else zero
                u = u_router if u_router is not None else zero
                day_cond = self.router_day_norm(g + u)
                branch_input = torch.cat([g, l, u, g * l], dim=-1)
                router_input = self.router_branch_fusion(branch_input)
                local_logits = self.router_local_logits(l)
                router_local_logit_std = float(local_logits.detach().std(unbiased=False).item())
                router_logits_raw = self.router(router_input) + self.router_day(day_cond) + local_logits
            elif self.use_bounded_sum:
                zero = x.new_zeros(B, D)
                g = g_router if g_router is not None else zero
                l = l_router if l_router is not None else zero
                u = u_router if u_router is not None else zero
                scale = self.local_router_scale.to(device=x.device, dtype=x.dtype)
                local_router_scale_value = float(scale.detach().item())
                router_input = self.router_cond_norm(g + u + scale * l)
                router_logits_raw = self.router(router_input)
                with torch.no_grad():
                    router_input_day = self.router_cond_norm(g + u)
                    local_logits_delta = self.router(router_input) - self.router(router_input_day)
                    router_local_logit_std = float(local_logits_delta.detach().std(unbiased=False).item())
            else:
                router_input = self.router_cond_norm(torch.stack(cond_parts, dim=0).sum(dim=0))
                router_logits_raw = self.router(router_input)
        else:
            if global_state is None:
                raise ValueError("Legacy router expects regime/global state.")
            router_input = global_state
            if summary_context is not None:
                router_input = torch.cat([global_state, summary_context], dim=-1)
            router_logits_raw = self.router(router_input)
        router_logits = router_logits_raw
        noise_std = float(getattr(self.config, "router_noise", 0.0) or 0.0)
        if self.training and noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * noise_std

        temperature = float(getattr(self.config, "router_temperature", 1.0) or 1.0)
        temperature = max(temperature, 1e-3)
        gate_weights = F.softmax(router_logits / temperature, dim=-1)
        mode = str(expert_mode or "").strip().lower()
        if mode:
            if mode == "time_only":
                gate_weights = torch.zeros_like(gate_weights)
                gate_weights[:, 0] = 1.0
            elif mode == "factor_only":
                gate_weights = torch.zeros_like(gate_weights)
                gate_weights[:, 1] = 1.0
            else:
                raise ValueError(f"Unsupported expert_mode: {expert_mode}")

        z_loss = (torch.logsumexp(router_logits_raw, dim=-1) ** 2).mean()
        entropy = -(gate_weights * torch.log(gate_weights + 1e-9)).sum(-1).mean()
        time_ratio = gate_weights[:, 0].mean()

        w_time = gate_weights[:, 0].view(B, 1, 1, 1)
        w_factor = gate_weights[:, 1].view(B, 1, 1, 1)

        if factor_film is not None:
            gamma, beta = factor_film
            if gamma.ndim != 3 or beta.ndim != 3:
                raise ValueError(
                    f"factor_film expects (gamma,beta) with shape [B,N,D]; got {tuple(gamma.shape)}, {tuple(beta.shape)}"
                )
            if tuple(gamma.shape) != (B, N, D) or tuple(beta.shape) != (B, N, D):
                raise ValueError(
                    f"factor_film expects (gamma,beta) with shape [B,N,D]={B,N,D}; got {tuple(gamma.shape)}, {tuple(beta.shape)}"
                )
            x = x * gamma.view(B, 1, N, D) + beta.view(B, 1, N, D)

        if feature_mask is not None:
            feature_mask = feature_mask.to(device=x.device, dtype=x.dtype)
            if feature_mask.ndim == 1:
                if int(feature_mask.shape[0]) != int(N):
                    raise ValueError(f"feature_mask must have length N={N}, got {int(feature_mask.shape[0])}")
                x = x * feature_mask.view(1, 1, N, 1)
            elif feature_mask.ndim == 2:
                if tuple(feature_mask.shape) != (B, N):
                    raise ValueError(f"feature_mask 2D must be [B,N]={B,N}, got {tuple(feature_mask.shape)}")
                x = x * feature_mask.view(B, 1, N, 1)
            else:
                raise ValueError(f"feature_mask must be 1D/2D, got shape {tuple(feature_mask.shape)}")

        bias_time = bias_factor = None
        if isinstance(attn_bias, (tuple, list)) and len(attn_bias) == 2:
            bias_time, bias_factor = attn_bias
        else:
            bias_time = attn_bias

        h_time = rearrange(x, "b t n d -> (b n) t d")
        if return_attn:
            out_time, attn_time = self.time_expert(h_time, bias_time, return_attn=True)
        else:
            out_time = self.time_expert(h_time, bias_time, return_attn=False)
            attn_time = None
        out_time = rearrange(out_time, "(b n) t d -> b t n d", b=B, n=N)

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
            "router_global_sensitivity": float(router_global_sensitivity),
            "router_local_sensitivity": float(router_local_sensitivity),
            "router_local_logit_std": float(router_local_logit_std),
            "local_router_scale": float(local_router_scale_value),
        }

        attn_dict = None
        if return_attn and (attn_time is not None or attn_factor is not None):
            attn_dict = {"time": attn_time, "factor": attn_factor}

        return x, diag, attn_dict
