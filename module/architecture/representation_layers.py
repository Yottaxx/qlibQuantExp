import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.architecture import RMSNorm
from einops import rearrange


class LearnedTemporalPooling(nn.Module):
    """Pool the time axis for each stock token."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        mode: str = "last",
        gru_residual_scale_init: float = 0.05,
        gru_residual_scale_learnable: bool = True,
        temporal_mhc_mix_init: float = 0.05,
        temporal_mhc_mix_max: float = 0.25,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.mode = str(mode or "last").strip().lower()
        valid_modes = {"last", "learned_query_mha_v1", "gru_v1", "gru_residual_v1", "gru_mhc_lite_v1"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported temporal_pooling_mode: {self.mode}")

        self.query = None
        self.mha = None
        self.gru = None
        self.norm = None
        if self.mode == "learned_query_mha_v1":
            self.query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
            self.mha = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=int(n_heads),
                dropout=float(dropout),
                bias=False,
                batch_first=True,
            )
            self.norm = RMSNorm(self.d_model)
        elif self.mode in {"gru_v1", "gru_residual_v1", "gru_mhc_lite_v1"}:
            self.gru = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=1,
                batch_first=True,
            )
            self.norm = RMSNorm(self.d_model)
            if self.mode == "gru_residual_v1":
                scale = torch.tensor(float(gru_residual_scale_init), dtype=torch.float32)
                if bool(gru_residual_scale_learnable):
                    self.gru_residual_scale = nn.Parameter(scale)
                else:
                    self.register_buffer("gru_residual_scale", scale)
            elif self.mode == "gru_mhc_lite_v1":
                mix_max = float(temporal_mhc_mix_max)
                if not 0.0 < mix_max <= 1.0:
                    raise ValueError("temporal_mhc_mix_max must be in (0, 1].")
                mix_init = min(max(float(temporal_mhc_mix_init), 1e-6), mix_max - 1e-6)
                mix_ratio = torch.tensor(mix_init / mix_max, dtype=torch.float32)
                mix_logit = torch.logit(mix_ratio)
                self.temporal_mhc_mix_logit = nn.Parameter(mix_logit)
                self.register_buffer("temporal_mhc_mix_max", torch.tensor(mix_max, dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if h.ndim != 4:
            raise ValueError(f"LearnedTemporalPooling expects [B,T,N,D], got {tuple(h.shape)}")
        B, T, N, D = h.shape
        h_last = h[:, -1, :, :]
        if self.mode == "last":
            return h_last, {
                "temporal_pool_attention_entropy": 0.0,
                "temporal_pool_last_residual_ratio": 0.0,
                "temporal_gru_output_norm": 0.0,
                "temporal_gru_output_std": 0.0,
                "temporal_gru_last_cosine": 0.0,
                "temporal_gru_residual_scale": 0.0,
                "temporal_mhc_mix": 0.0,
                "temporal_mhc_mix_max": 0.0,
            }

        if self.mode in {"gru_v1", "gru_residual_v1", "gru_mhc_lite_v1"}:
            x = rearrange(h, "b t n d -> (b n) t d")
            _, h_n = self.gru(x)
            pooled = h_n[-1].reshape(B, N, D)
            if self.mode == "gru_residual_v1":
                scale = self.gru_residual_scale.to(device=h.device, dtype=h.dtype)
                residual = scale * pooled
                out = self.norm(h_last + residual)
                residual_ratio = (
                    residual.detach().norm(dim=-1).mean()
                    / h_last.detach().norm(dim=-1).mean().clamp_min(1e-9)
                )
                scale_value = float(scale.detach().item())
                mhc_mix_value = 0.0
                mhc_mix_max_value = 0.0
            elif self.mode == "gru_mhc_lite_v1":
                mix_max = self.temporal_mhc_mix_max.to(device=h.device, dtype=h.dtype)
                mix = mix_max * torch.sigmoid(self.temporal_mhc_mix_logit.to(device=h.device, dtype=h.dtype))
                residual = mix * (pooled - h_last)
                out = self.norm(h_last + residual)
                residual_ratio = (
                    residual.detach().norm(dim=-1).mean()
                    / h_last.detach().norm(dim=-1).mean().clamp_min(1e-9)
                )
                scale_value = 0.0
                mhc_mix_value = float(mix.detach().item())
                mhc_mix_max_value = float(mix_max.detach().item())
            else:
                out = self.norm(pooled)
                residual_ratio = h.new_tensor(0.0)
                scale_value = 0.0
                mhc_mix_value = 0.0
                mhc_mix_max_value = 0.0

            with torch.no_grad():
                out_det = out.detach()
                last_det = h_last.detach()
                cosine = F.cosine_similarity(out_det.reshape(B * N, D), last_det.reshape(B * N, D), dim=-1)
                
                # 1. Std Ratio: How much variance does the temporal pooling preserve or compress?
                h_std = last_det.std(dim=-1).mean()
                out_std = out_det.std(dim=-1).mean()
                std_ratio = (out_std / h_std.clamp_min(1e-9)).item()
                
                # 2. Temporal Persistence: Correlation between h_t and h_{t-1}
                h_norm = F.normalize(h.detach(), dim=-1)
                persistence = (h_norm[:, 1:, :] * h_norm[:, :-1, :]).sum(dim=-1).mean().item()

            return out, {
                "temporal_pool_attention_entropy": 0.0,
                "temporal_pool_last_residual_ratio": float(residual_ratio.item()),
                "temporal_gru_output_norm": float(out_det.norm(dim=-1).mean().item()),
                "temporal_gru_output_std": float(out_det.std(unbiased=False).item()),
                "temporal_gru_last_cosine": float(cosine.mean().item()),
                "temporal_std_ratio": float(std_ratio),
                "temporal_persistence": float(persistence),
                "temporal_gru_residual_scale": scale_value,
                "temporal_mhc_mix": mhc_mix_value,
                "temporal_mhc_mix_max": mhc_mix_max_value,
            }

        x = rearrange(h, "b t n d -> (b n) t d")
        query = self.query.expand(B * N, -1, -1)
        pooled, attn = self.mha(
            query=query,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
        )
        pooled = pooled.squeeze(1).reshape(B, N, D)
        out = self.norm(h_last + pooled)

        with torch.no_grad():
            attn_prob = attn.squeeze(2) if attn.dim() == 4 else attn
            entropy = -(attn_prob * torch.log(attn_prob.clamp_min(1e-9))).sum(dim=-1)
            entropy = entropy / math.log(max(int(T), 2))
            residual_ratio = pooled.detach().norm(dim=-1).mean() / h_last.detach().norm(dim=-1).mean().clamp_min(1e-9)

        return out, {
            "temporal_pool_attention_entropy": float(entropy.mean().item()),
            "temporal_pool_last_residual_ratio": float(residual_ratio.item()),
            "temporal_gru_output_norm": 0.0,
            "temporal_gru_output_std": 0.0,
            "temporal_gru_last_cosine": 0.0,
            "temporal_gru_residual_scale": 0.0,
            "temporal_mhc_mix": 0.0,
            "temporal_mhc_mix_max": 0.0,
        }


class CrossStockContextLayer(nn.Module):
    """Broadcast a learned cross-stock context back to every factor token."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        residual_scale_init: float = 0.1,
        scale_learnable: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(n_heads),
            dropout=float(dropout),
            bias=False,
            batch_first=True,
        )
        self.pre_norm = RMSNorm(self.d_model)
        self.norm = RMSNorm(self.d_model)
        scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if bool(scale_learnable):
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)

    def forward(self, h_last: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if h_last.ndim != 3:
            raise ValueError(f"CrossStockContextLayer expects [B,N,D], got {tuple(h_last.shape)}")
        B, N, _ = h_last.shape
        if B <= 1:
            return h_last, {
                "cross_stock_scale": float(self.residual_scale.detach().item()),
                "cross_stock_token_std": 0.0,
                "cross_stock_context_norm": 0.0,
                "cross_stock_attention_entropy": 0.0,
            }

        h_norm = self.pre_norm(h_last)
        stock_token = h_norm.mean(dim=1).unsqueeze(0)  # [1,B,D]
        ctx, attn = self.mha(
            query=stock_token,
            key=stock_token,
            value=stock_token,
            need_weights=True,
            average_attn_weights=False,
        )
        ctx = ctx.squeeze(0)
        scale = self.residual_scale.to(device=h_last.device, dtype=h_last.dtype)
        out = self.norm(h_last + scale * ctx.unsqueeze(1))

        with torch.no_grad():
            attn_prob = attn.squeeze(0)
            entropy = -(attn_prob * torch.log(attn_prob.clamp_min(1e-9))).sum(dim=-1)
            entropy = entropy / math.log(max(int(B), 2))
            token_std = stock_token.squeeze(0).detach().std(dim=0, unbiased=False).mean()
            context_norm = ctx.detach().norm(dim=-1).mean()

        return out, {
            "cross_stock_scale": float(scale.detach().item()),
            "cross_stock_token_std": float(token_std.item()),
            "cross_stock_context_norm": float(context_norm.item()),
            "cross_stock_attention_entropy": float(entropy.mean().item()),
        }


class InnerCrossStockContextLayer(nn.Module):
    """Inject cross-stock context inside the 4D MoE representation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
        scale_learnable: bool = True,
        mode: str = "day_token",
        diag_factor_samples: int = 8,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.mode = str(mode or "day_token").strip().lower()
        if self.mode not in {"day_token", "stock_factor_v1", "stock_time_factor_v1"}:
            raise ValueError(f"Unsupported inner_cross_stock_mode: {self.mode}")
        self.diag_factor_samples = max(0, int(diag_factor_samples))
        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(n_heads),
            dropout=float(dropout),
            bias=False,
            batch_first=True,
        )
        self.pre_norm = RMSNorm(self.d_model)
        self.norm = RMSNorm(self.d_model)
        scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if bool(scale_learnable):
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)

    def _empty_diag(self) -> dict[str, float]:
        return {
            "inner_cross_stock_scale": float(self.residual_scale.detach().item()),
            "inner_cross_stock_token_std": 0.0,
            "inner_cross_stock_context_norm": 0.0,
            "inner_cross_stock_attention_entropy": 0.0,
        }

    def _forward_day_token(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        B, _, _, _ = h.shape
        h_norm = self.pre_norm(h)
        stock_token = h_norm.mean(dim=(1, 2)).unsqueeze(0)  # [1,B,D]
        ctx, attn = self.mha(
            query=stock_token,
            key=stock_token,
            value=stock_token,
            need_weights=True,
            average_attn_weights=False,
        )
        ctx = ctx.squeeze(0)
        scale = self.residual_scale.to(device=h.device, dtype=h.dtype)
        out = self.norm(h + scale * ctx[:, None, None, :])

        with torch.no_grad():
            attn_prob = attn.squeeze(0)
            entropy = -(attn_prob * torch.log(attn_prob.clamp_min(1e-9))).sum(dim=-1)
            entropy = entropy / math.log(max(int(B), 2))
            token_std = stock_token.squeeze(0).detach().std(dim=0, unbiased=False).mean()
            context_norm = ctx.detach().norm(dim=-1).mean()

        return out, {
            "inner_cross_stock_scale": float(scale.detach().item()),
            "inner_cross_stock_token_std": float(token_std.item()),
            "inner_cross_stock_context_norm": float(context_norm.item()),
            "inner_cross_stock_attention_entropy": float(entropy.mean().item()),
        }

    def _forward_stock_factor(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        B, _, N, _ = h.shape
        h_norm = self.pre_norm(h)
        stock_factor_token = h_norm.mean(dim=1)  # [B,N,D]
        relation_tokens = rearrange(stock_factor_token, "b n d -> n b d")
        ctx, _ = self.mha(
            query=relation_tokens,
            key=relation_tokens,
            value=relation_tokens,
            need_weights=False,
        )
        ctx = rearrange(ctx, "n b d -> b n d")
        scale = self.residual_scale.to(device=h.device, dtype=h.dtype)
        out = self.norm(h + scale * ctx[:, None, :, :])

        with torch.no_grad():
            token_std = stock_factor_token.detach().std(dim=0, unbiased=False).mean()
            context_norm = ctx.detach().norm(dim=-1).mean()
            entropy = h.new_tensor(0.0)
            k = min(int(self.diag_factor_samples), int(N))
            if k > 0:
                if k == int(N):
                    idx = torch.arange(int(N), device=h.device)
                else:
                    idx = torch.linspace(0, int(N) - 1, steps=k, device=h.device).round().long().unique()
                diag_tokens = relation_tokens.index_select(0, idx)
                _, attn = self.mha(
                    query=diag_tokens,
                    key=diag_tokens,
                    value=diag_tokens,
                    need_weights=True,
                    average_attn_weights=False,
                )
                entropy = -(attn * torch.log(attn.clamp_min(1e-9))).sum(dim=-1)
                entropy = entropy / math.log(max(int(B), 2))

        return out, {
            "inner_cross_stock_scale": float(scale.detach().item()),
            "inner_cross_stock_token_std": float(token_std.item()),
            "inner_cross_stock_context_norm": float(context_norm.item()),
            "inner_cross_stock_attention_entropy": float(entropy.mean().item()),
        }

    def _forward_stock_time_factor(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """All (time, factor) pairs per stock attend across stocks.

        relation_tokens shape: [(T*N), B, D]  — each of the T*N slots
        attends across B stocks independently.
        """
        B, T, N, D = h.shape
        h_norm = self.pre_norm(h)
        relation_tokens = rearrange(h_norm, "b t n d -> (t n) b d")  # [(T*N), B, D]
        ctx, _ = self.mha(
            query=relation_tokens,
            key=relation_tokens,
            value=relation_tokens,
            need_weights=False,
        )
        ctx = rearrange(ctx, "(t n) b d -> b t n d", t=T, n=N)
        scale = self.residual_scale.to(device=h.device, dtype=h.dtype)
        out = self.norm(h + scale * ctx)

        with torch.no_grad():
            token_std = relation_tokens.detach().std(dim=1, unbiased=False).mean()
            context_norm = ctx.detach().norm(dim=-1).mean()
            entropy = h.new_tensor(0.0)
            TN = T * N
            k = min(int(self.diag_factor_samples), TN)
            if k > 0:
                if k == TN:
                    idx = torch.arange(TN, device=h.device)
                else:
                    idx = torch.linspace(0, TN - 1, steps=k, device=h.device).round().long().unique()
                diag_tokens = relation_tokens.index_select(0, idx)
                _, attn = self.mha(
                    query=diag_tokens,
                    key=diag_tokens,
                    value=diag_tokens,
                    need_weights=True,
                    average_attn_weights=False,
                )
                entropy = -(attn * torch.log(attn.clamp_min(1e-9))).sum(dim=-1)
                entropy = entropy / math.log(max(int(B), 2))

        return out, {
            "inner_cross_stock_scale": float(scale.detach().item()),
            "inner_cross_stock_token_std": float(token_std.item()),
            "inner_cross_stock_context_norm": float(context_norm.item()),
            "inner_cross_stock_attention_entropy": float(entropy.mean().item()),
        }

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if h.ndim != 4:
            raise ValueError(f"InnerCrossStockContextLayer expects [B,T,N,D], got {tuple(h.shape)}")
        B, _, _, _ = h.shape
        if B <= 1:
            return h, self._empty_diag()
        if self.mode == "stock_factor_v1":
            return self._forward_stock_factor(h)
        if self.mode == "stock_time_factor_v1":
            return self._forward_stock_time_factor(h)
        return self._forward_day_token(h)
