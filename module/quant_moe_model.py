import math

# module/quant_moe_model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from module.architecture.feature_selector import DifferentiableFeatureSelector
from module.utils.losses import QuantLossFunctions
from module.utils.model_configuration import QuantMoEConfig, QuantModelOutput
from module.architecture.moe_block import RegimeAdaptiveMoEBlock
from module.architecture.regime_encoder import RegimeContextEncoder
from module.architecture.regime_adaptive_embedding import RegimeAdaptiveFactorGate, RegimeAdaptiveTimeEmbedding
from module.architecture.attention_pooling import AdaptivePooling
from module.utils.utils import build_bidirectional_alibi_bias


class QuantMoEModel(PreTrainedModel):
    """
    RST-MoE 主模型：
    - 因子维度 N：num_alphas
    - 序列长度 T：context_len
    - 时序/截面解耦：RegimeAdaptiveMoEBlock
    """
    config_class = QuantMoEConfig

    def __init__(self, config: QuantMoEConfig):
        super().__init__(config)
        self.config = config

        d_model = config.d_model
        num_alphas = config.num_alphas

        # 1) 数值 + 因子 ID 嵌入
        self.val_proj = nn.Linear(1, d_model)
        self.factor_id_emb = nn.Embedding(num_alphas, d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # 1.5) Regime-adaptive time embedding & factor gate (lightweight, ablatable)
        self.time_embedding = None
        if bool(getattr(config, "use_regime_time_embedding", False)):
            self.time_embedding = RegimeAdaptiveTimeEmbedding(
                d_model=int(config.d_model),
                max_len=int(config.context_len),
                tau_min=float(getattr(config, "time_tau_min", 0.5)),
                tau_max=float(getattr(config, "time_tau_max", 50.0)),
                tau_init=float(getattr(config, "time_tau_init", 5.0)),
                init_std=float(getattr(config, "time_emb_init_std", 0.02)),
                normalize_decay=bool(getattr(config, "time_decay_normalize", True)),
            )

        self.factor_gate = None
        if bool(getattr(config, "use_regime_factor_gate", False)):
            self.factor_gate = RegimeAdaptiveFactorGate(
                d_model=int(config.d_model),
                gate_scale=float(getattr(config, "factor_gate_scale", 0.5)),
                shift_scale=float(getattr(config, "factor_gate_shift_scale", 0.0)),
            )

        # 2) 可微特征选择
        if config.use_feature_selection:
            self.feature_selector = DifferentiableFeatureSelector(
                num_features=num_alphas,
                sigma=config.selection_noise_std,
            )
        else:
            self.feature_selector = None

        # 3) Regime 编码器（内部 x 统计 + 可选宏观）
        self.regime_encoder = RegimeContextEncoder(
            d_model=config.d_model,
            use_external_macro=config.use_external_macro,
            d_macro=config.d_macro_input,
            internal_mode=getattr(config, "regime_internal_mode", "short"),
            internal_lag=getattr(config, "regime_internal_lag", 1),
            internal_use_batch_stats=getattr(config, "regime_internal_use_batch_stats", True),
            internal_tail_threshold=getattr(config, "regime_internal_tail_threshold", 2.0),
        )

        # 4) MoE 主干
        self.layers = nn.ModuleList([RegimeAdaptiveMoEBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # 5) Attention-based pooling for factor aggregation
        # 使用自适应 attention pooling 替代简单的 softmax pooling
        # 结合 attention 和 mean pooling，提供更稳健的因子聚合
        self.factor_pooling = AdaptivePooling(
            d_model=d_model,
            n_heads=1,
            dropout=config.dropout,
            alpha=config.pooling_alpha,
        )
        
        # 6) Stock score head
        self.head = nn.Linear(d_model, 1)
        
        # HF 标准初始化
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """
        HF-style weight init (executed by `self.post_init()`).

        Goals
        -----
        - Make embedding/value projection scales consistent (avoid `Linear(1,d)` default init dominating).
        - Keep identity-start for FiLM/gating projections (ResNet-style stability).
        """
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        if init_std <= 0:
            init_std = 0.02

        # Preserve identity-start modules (e.g., FiLM projections).
        if getattr(module, "_rstmoe_zero_init", False):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if getattr(module, "padding_idx", None) is not None:
                module.weight.data[module.padding_idx].zero_()
            return

        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.MultiheadAttention):
            # nn.MultiheadAttention keeps in-proj weights as a single Parameter.
            # out_proj is a Linear submodule and will be initialized by the nn.Linear branch above.
            if getattr(module, "in_proj_weight", None) is not None:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=init_std)
            if getattr(module, "in_proj_bias", None) is not None:
                nn.init.zeros_(module.in_proj_bias)
            if getattr(module, "bias_k", None) is not None:
                nn.init.normal_(module.bias_k, mean=0.0, std=init_std)
            if getattr(module, "bias_v", None) is not None:
                nn.init.normal_(module.bias_v, mean=0.0, std=init_std)
            return

    def forward(
        self,
        x: torch.Tensor,
        factor_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        macro_features: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
        attn_layers: list[int] | None = None,
    ) -> QuantModelOutput:
        """
        Args:
            x: [B, T, N]  (时间窗口 × 因子)
            factor_ids: [N] long
            labels: [B] (一个样本一个 label; 已是 CSRankNorm 后的值)
            macro_features: [B, d_macro] optional external macro features
            
        Note:
            Regime signal is computed from internal statistics of x (via RegimeContextEncoder),
            not from external date IDs. This allows the model to adaptively learn market regimes
            from factor patterns rather than relying on calendar dates.
        """
        device = x.device
        B, T, N = x.shape

        if factor_ids is None:
            factor_ids = torch.arange(N, device=device)
        factor_ids = factor_ids.to(device)

        # 0) Regime embedding（只看 x/macro，不看 label；也不被 time embedding 污染）
        regime = self.regime_encoder(x, macro_features)  # [B, D]

        # 1) value + factor embedding
        factor_table = self.factor_id_emb(factor_ids.long())  # [N,D]
        h = self.val_proj(x.unsqueeze(-1)) + factor_table.view(1, 1, N, -1)

        diag_metrics: dict[str, float] = {}
        factor_film = None

        # 1.5) Regime-adaptive time embedding (adds a clean time signal for short windows)
        if self.time_embedding is not None:
            time_emb, tau = self.time_embedding(regime, T)  # [B,T,D], [B,1]
            h = h + time_emb.unsqueeze(2)  # [B,T,1,D] -> broadcast over N
            tau_mean = float(tau.detach().mean().item())
            diag_metrics["time_tau"] = tau_mean
            diag_metrics["time_half_life"] = tau_mean * math.log(2.0)

        # 1.6) Regime-adaptive factor reweighting (per-sample, per-factor)
        if self.factor_gate is not None:
            gamma, beta = self.factor_gate(regime, factor_table)  # [B,N,D], [B,N,D]
            factor_film = (gamma, beta)

            # Diagnostics: summarize FiLM strength on a scalar per factor.
            gate_det = gamma.detach().mean(dim=-1)  # [B,N], in ~[1-scale, 1+scale]
            diag_metrics["factor_gate_mean"] = float(gate_det.mean().item())
            diag_metrics["factor_gate_std"] = float(gate_det.std(unbiased=False).item())

            # Concentration: use |gamma-1| as "importance" to see whether the model is actively modulating a few factors.
            eps = 1e-12
            imp = (gate_det - 1.0).abs().clamp_min(eps)
            denom = imp.sum(dim=-1, keepdim=True).clamp_min(eps)
            p = imp / denom  # [B,N]

            entropy = -(p * (p + eps).log()).sum(dim=-1)  # [B]
            entropy_norm = entropy / math.log(max(int(N), 2))
            diag_metrics["factor_gate_entropy"] = float(entropy_norm.mean().item())

            k5 = min(5, int(N))
            k10 = min(10, int(N))
            diag_metrics["factor_gate_topk_mass_5"] = float(p.topk(k5, dim=-1).values.sum(dim=-1).mean().item())
            diag_metrics["factor_gate_topk_mass_10"] = float(p.topk(k10, dim=-1).values.sum(dim=-1).mean().item())

        # 2) 可微特征选择
        reg_loss = torch.tensor(0.0, device=device)
        mask = None
        feature_mask = None
        if self.feature_selector is not None:
            z, reg_loss = self.feature_selector.sample_mask(
                temperature=self.config.selection_temperature,
                training=self.training,
            )
            # Apply after LayerNorm inside each block (FiLM-friendly, not canceled by Pre-LN).
            feature_mask = z
            mask = z.detach()

        h = self.emb_dropout(h)

        # 4) ALiBi bias（time axis only）
        attn_bias = None
        if self.config.use_alibi:
            bias_time = build_bidirectional_alibi_bias(B, T, self.config.n_heads, device)
            # NOTE: factor-ALiBi is intentionally disabled (factor axis has no natural order).
            attn_bias = (bias_time, None)

        # 5) MoE blocks
        z_losses = []
        entropies = []
        time_ratios = []
        gates_list = []
        attn_maps: dict[str, dict[str, torch.Tensor]] = {}

        for idx, layer in enumerate(self.layers):
            need_attn = return_attn and (attn_layers is None or idx in attn_layers)
            h, diag, layer_attn = layer(
                h,
                regime_embedding=regime,
                attn_bias=attn_bias,
                return_attn=need_attn,
                factor_film=factor_film,
                feature_mask=feature_mask,
            )
            z_losses.append(diag["z_loss"])
            entropies.append(diag["entropy"])
            time_ratios.append(diag["time_ratio"])
            gates_list.append(diag["weights"])

            if need_attn and layer_attn is not None:
                attn_maps[f"layer_{idx}"] = layer_attn

        h = self.final_norm(h)

        # 6) Factor aggregation & stock scoring
        # 使用最后一时间步的因子表示 [B, N, D]
        h_last = h[:, -1, :, :]  # [B, N, D]
        
        # Attention-based pooling: 自适应加权聚合因子表示
        h_pooled, factor_attention_weights = self.factor_pooling(h_last)  # [B, D], [B, N]
        
        # Stock score prediction
        stock_score = self.head(h_pooled).squeeze(-1)  # [B]
        
        # factor_attention_weights 用于返回（可用于可解释性分析）
        factor_logits = factor_attention_weights  # [B, N]

        # 7) Loss & metrics
        total_loss: torch.Tensor | None = None
        metrics: dict[str, float] = {}
        valid_ratio = 0.0

        if labels is not None:
            if labels.ndim == 0:
                labels = labels.view(1, 1)
            elif labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            else:
                labels = labels.reshape(labels.shape[0], -1)

            if labels.shape[-1] != 1:
                raise RuntimeError(
                    f"QuantMoEModel expects a single label per sample; got shape {tuple(labels.shape)}"
                )
            labels = labels.squeeze(-1)
            valid = torch.isfinite(labels)
            valid_ratio = float(valid.float().mean().item())

            if valid.sum().item() >= 2:
                y = labels[valid]
                p = stock_score[valid]

                w = self.config.loss_weights

                # 主 loss：ListMLE（list-wise）
                l_listmle = QuantLossFunctions.listmle_loss(
                    p,
                    y,
                    tau=getattr(self.config, "listmle_tau", 1.0),
                )

                # IC 作为 metric，只做监控，不进 total_loss
                l_ic = QuantLossFunctions.cs_ic_loss(p, y)

                # 其他辅助 loss（可选）
                l_rank = None
                if w.get("rank", 0.0) != 0.0:
                    l_rank = QuantLossFunctions.ranknet_topbottom_loss(p, y, self.config.rank_topk)

                l_huber = None
                if w.get("huber", 0.0) != 0.0:
                    l_huber = QuantLossFunctions.cs_huber_loss(p, y, self.config.huber_delta)

                # MoE router z-loss & 特征稀疏正则
                # 直接用 router_z_loss_coef 和 selection_reg_lambda，不再经过 loss_weights 二次缩放
                l_aux = (
                    torch.stack(z_losses).mean() * self.config.router_z_loss_coef
                    if z_losses
                    else torch.tensor(0.0, device=device)
                )
                l_reg = reg_loss * self.config.selection_reg_lambda

                # ★ total_loss：ListMLE + 辅助 loss
                total_loss = (
                    w.get("listmle", 1.0) * l_listmle
                    + l_aux  # 直接加，不再乘 w["aux"]
                    + l_reg  # 直接加，不再乘 w["reg"]
                )
                if l_rank is not None:
                    total_loss = total_loss + w.get("rank", 0.0) * l_rank
                if l_huber is not None:
                    total_loss = total_loss + w.get("huber", 0.0) * l_huber

                metrics = {
                    "loss_total": float(total_loss.detach().item()),
                    "loss_listmle": float(l_listmle.detach().item()),
                    "loss_ic": float(l_ic.detach().item()),
                    "loss_aux": float(l_aux.detach().item()),
                    "loss_sparsity": float(l_reg.detach().item()),
                    "valid_ratio": float(valid_ratio),
                }
                if l_rank is not None:
                    metrics["loss_rank"] = float(l_rank.detach().item())
                if l_huber is not None:
                    metrics["loss_huber"] = float(l_huber.detach().item())
            else:
                metrics = {"valid_ratio": float(valid_ratio)}
        else:
            metrics = {}

        if diag_metrics:
            metrics = {**diag_metrics, **metrics}

        avg_entropy = float(torch.stack(entropies).mean().detach().item()) if entropies else None
        avg_time_ratio = float(torch.stack(time_ratios).mean().detach().item()) if time_ratios else None

        return QuantModelOutput(
            loss=total_loss,
            logits=factor_logits,
            gate_weights=gates_list,
            metrics=metrics,
            avg_gate_entropy=avg_entropy,
            avg_time_ratio=avg_time_ratio,
            selected_mask=mask,
            attn_maps=attn_maps if return_attn else None,
            scores=stock_score,
        )
