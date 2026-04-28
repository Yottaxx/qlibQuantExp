import math

import torch
import torch.nn as nn
from module.architecture import RMSNorm
from transformers import PreTrainedModel

from module.architecture.feature_selector import DifferentiableFeatureSelector
from module.architecture.feature_tokenizer import FeatureTokenizer
from module.architecture.hierarchical_state import GlobalStateEncoder, LocalStateEncoder
from module.architecture.moe_block import RegimeAdaptiveMoEBlock
from module.architecture.regime_adaptive_embedding import RegimeAdaptiveFactorGate, RegimeAdaptiveTimeEmbedding
from module.architecture.regime_adaptive_pooling import RegimeAdaptivePooling, SimpleStaticFactorPooling
from module.architecture.representation_layers import (
    CrossStockContextLayer,
    InnerCrossStockContextLayer,
    LearnedTemporalPooling,
)
from module.architecture.regime_encoder import RegimeContextEncoder, compute_legacy_internal_stats
from module.utils.losses import QuantLossFunctions
from module.utils.model_configuration import QuantMoEConfig, QuantModelOutput
from module.utils.utils import build_bidirectional_alibi_bias


class QuantMoEModel(PreTrainedModel):
    """
    RST-MoE main model with optional deterministic global-local hyper-state.
    """

    config_class = QuantMoEConfig

    def __init__(self, config: QuantMoEConfig):
        super().__init__(config)
        self.config = config

        d_model = int(config.d_model)
        num_alphas = int(config.num_alphas)
        self.router_summary_source = str(getattr(config, "router_summary_source", "none") or "none").strip().lower()
        self.pooling_summary_source = str(getattr(config, "pooling_summary_source", "none") or "none").strip().lower()
        self.use_hierarchical_state_field = bool(getattr(config, "use_hierarchical_state_field", False))
        self.global_state_use_macro = bool(getattr(config, "global_state_use_macro", True))
        self.global_state_use_day_summary = bool(getattr(config, "global_state_use_day_summary", True))
        self.global_state_use_internal_state = bool(getattr(config, "global_state_use_internal_state", False))
        self.global_state_use_x = bool(getattr(config, "global_state_use_x", True))
        self.local_state_use_internal_state = bool(getattr(config, "local_state_use_internal_state", False))
        self.router_use_internal_batch_state = bool(getattr(config, "router_use_internal_batch_state", False))
        self.use_internal_state_features = bool(
            self.global_state_use_internal_state
            or self.local_state_use_internal_state
            or self.router_use_internal_batch_state
        )
        hierarchical_uses_day_summary = self.use_hierarchical_state_field and self.global_state_use_day_summary
        self.uses_day_summary_asset = bool(
            hierarchical_uses_day_summary
            or self.router_summary_source == "day_asset"
            or self.pooling_summary_source == "day_asset"
        )

        self.value_embedding_type = str(getattr(config, "value_embedding_type", "shared_linear")).strip().lower()
        self.feature_tokenizer_add_factor_id = bool(getattr(config, "feature_tokenizer_add_factor_id", False))
        self.val_proj = None
        self.feature_tokenizer = None
        if self.value_embedding_type == "feature_tokenizer":
            init_std = float(getattr(config, "feature_tokenizer_init_std", config.initializer_range))
            if init_std <= 0:
                init_std = float(getattr(config, "initializer_range", 0.02) or 0.02)
            self.feature_tokenizer = FeatureTokenizer(
                num_features=num_alphas,
                d_model=d_model,
                bias=bool(getattr(config, "feature_tokenizer_bias", True)),
                init_std=init_std,
            )
        else:
            self.val_proj = nn.Linear(1, d_model)
        self.factor_id_emb = nn.Embedding(num_alphas, d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.feature_selector = None
        if config.use_feature_selection:
            self.feature_selector = DifferentiableFeatureSelector(
                num_features=num_alphas,
                sigma=config.selection_noise_std,
            )

        self.day_summary_encoder = None
        if self.uses_day_summary_asset:
            d_day_summary_input = int(getattr(config, "d_day_summary_input", 0) or 0)
            if d_day_summary_input <= 0:
                raise ValueError("day summary mode requires d_day_summary_input > 0")
            self.day_summary_encoder = nn.Sequential(
                nn.Linear(d_day_summary_input, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                RMSNorm(d_model),
            )

        self.global_state_encoder = None
        self.local_state_encoder = None
        self.regime_encoder = None
        if self.use_hierarchical_state_field:
            self.global_state_encoder = GlobalStateEncoder(
                d_macro_input=int(getattr(config, "d_macro_input", 0) or 0),
                d_day_summary_input=d_model,
                d_internal_state_input=int(getattr(config, "d_internal_state_input", 4) or 4),
                d_x_input=int(config.context_len) * int(config.num_alphas),
                d_global_state=int(getattr(config, "d_global_state", d_model) or d_model),
                use_macro=self.global_state_use_macro,
                use_day_summary=self.global_state_use_day_summary,
                use_internal_state=self.global_state_use_internal_state,
                use_x=self.global_state_use_x,
                dropout=float(getattr(config, "regime_macro_dropout", 0.0) or 0.0),
            )
            self.local_state_encoder = LocalStateEncoder(
                d_global_state=int(getattr(config, "d_global_state", d_model) or d_model),
                d_local_state=int(getattr(config, "d_local_state", d_model) or d_model),
                input_mode=str(getattr(config, "local_state_input_mode", "last_mean_std_trend_vol")),
                num_alphas=num_alphas,
                d_internal_state_input=int(getattr(config, "d_internal_state_input", 4) or 4),
                use_internal_state=self.local_state_use_internal_state,
                dropout=float(getattr(config, "dropout", 0.0) or 0.0),
            )
        else:
            if bool(getattr(config, "use_external_macro", False)) and int(getattr(config, "d_macro_input", 0) or 0) <= 0:
                raise ValueError(
                    "Legacy regime encoder with use_external_macro=True requires d_macro_input > 0. "
                    "Either provide market-state features or set use_external_macro=False."
                )
            self.regime_encoder = RegimeContextEncoder(
                d_model=config.d_model,
                use_external_macro=config.use_external_macro,
                d_macro=config.d_macro_input,
                internal_mode=getattr(config, "regime_internal_mode", "short"),
                internal_lag=getattr(config, "regime_internal_lag", 1),
                internal_use_batch_stats=getattr(config, "regime_internal_use_batch_stats", True),
                internal_tail_threshold=getattr(config, "regime_internal_tail_threshold", 2.0),
                dropout=float(getattr(config, "regime_macro_dropout", 0.0) or 0.0),
            )

        time_condition_dim = int(getattr(config, "d_global_state", d_model) or d_model)
        if not self.use_hierarchical_state_field:
            time_condition_dim = d_model
        self.time_embedding = None
        if bool(getattr(config, "use_regime_time_embedding", False)):
            self.time_embedding = RegimeAdaptiveTimeEmbedding(
                d_model=d_model,
                max_len=int(config.context_len),
                condition_dim=time_condition_dim,
                tau_min=float(getattr(config, "time_tau_min", 0.5)),
                tau_max=float(getattr(config, "time_tau_max", 50.0)),
                tau_init=float(getattr(config, "time_tau_init", 5.0)),
                init_std=float(getattr(config, "time_emb_init_std", 0.02)),
                normalize_decay=bool(getattr(config, "time_decay_normalize", True)),
            )

        self.factor_gate = None
        if bool(getattr(config, "use_regime_factor_gate", False)):
            factor_gate_kwargs = dict(
                d_model=d_model,
                global_dim=int(getattr(config, "d_global_state", d_model) or d_model),
                local_dim=int(getattr(config, "d_local_state", d_model) or d_model),
                use_global_state=bool(getattr(config, "film_use_global_state", True)),
                use_local_state=bool(getattr(config, "film_use_local_state", True)) and self.use_hierarchical_state_field,
                gate_scale=float(getattr(config, "factor_gate_scale", 0.5)),
                shift_scale=float(getattr(config, "factor_gate_shift_scale", 0.0)),
                local_scale_init=(
                    float(getattr(config, "local_film_scale_init", 0.1))
                    if str(getattr(config, "state_fusion_mode", "sum_norm") or "sum_norm").strip().lower()
                    == "bounded_sum_v1"
                    else 1.0
                ),
                local_scale_learnable=(
                    bool(getattr(config, "local_scale_learnable", True))
                    if str(getattr(config, "state_fusion_mode", "sum_norm") or "sum_norm").strip().lower()
                    == "bounded_sum_v1"
                    else False
                ),
            )
            if not self.use_hierarchical_state_field:
                factor_gate_kwargs.update(
                    global_dim=d_model,
                    local_dim=0,
                    use_global_state=True,
                    use_local_state=False,
                )
            self.factor_gate = RegimeAdaptiveFactorGate(**factor_gate_kwargs)

        self.layers = nn.ModuleList([RegimeAdaptiveMoEBlock(config) for _ in range(config.n_layers)])
        self.inner_cross_stock_layers = None
        if bool(getattr(config, "use_inner_cross_stock_attention", False)):
            self.inner_cross_stock_layers = nn.ModuleList(
                [
                    InnerCrossStockContextLayer(
                        d_model=d_model,
                        n_heads=int(getattr(config, "inner_cross_stock_attention_heads", config.n_heads)),
                        dropout=float(getattr(config, "dropout", 0.0) or 0.0),
                        residual_scale_init=float(getattr(config, "inner_cross_stock_residual_scale_init", 0.05)),
                        scale_learnable=bool(getattr(config, "inner_cross_stock_scale_learnable", True)),
                        mode=str(getattr(config, "inner_cross_stock_mode", "day_token") or "day_token"),
                        diag_factor_samples=int(getattr(config, "inner_cross_stock_diag_factor_samples", 8) or 0),
                    )
                    for _ in range(config.n_layers)
                ]
            )
        self.final_norm = RMSNorm(d_model)
        self.temporal_pooling = LearnedTemporalPooling(
            d_model=d_model,
            n_heads=int(getattr(config, "temporal_pooling_heads", config.n_heads)),
            dropout=float(getattr(config, "dropout", 0.0) or 0.0),
            mode=str(getattr(config, "temporal_pooling_mode", "last") or "last"),
            gru_residual_scale_init=float(getattr(config, "temporal_gru_residual_scale_init", 0.05)),
            gru_residual_scale_learnable=bool(getattr(config, "temporal_gru_residual_scale_learnable", True)),
            temporal_mhc_mix_init=float(getattr(config, "temporal_mhc_mix_init", 0.05)),
            temporal_mhc_mix_max=float(getattr(config, "temporal_mhc_mix_max", 0.25)),
        )
        self.cross_stock_attention = None
        if bool(getattr(config, "use_cross_stock_attention", False)):
            self.cross_stock_attention = CrossStockContextLayer(
                d_model=d_model,
                n_heads=int(getattr(config, "cross_stock_attention_heads", config.n_heads)),
                dropout=float(getattr(config, "dropout", 0.0) or 0.0),
                residual_scale_init=float(getattr(config, "cross_stock_residual_scale_init", 0.1)),
                scale_learnable=bool(getattr(config, "cross_stock_scale_learnable", True)),
            )
        pooling_mode = str(getattr(config, "pooling_mode", "adaptive_alpha") or "adaptive_alpha").strip().lower()
        if pooling_mode == "simple_static":
            self.factor_pooling = SimpleStaticFactorPooling(
                d_model=d_model,
                n_heads=int(getattr(config, "n_heads", 4)),
                dropout=config.dropout,
                alpha=float(getattr(config, "pooling_alpha", 0.7)),
            )
        else:
            self.factor_pooling = RegimeAdaptivePooling(
                d_model=d_model,
                n_heads=int(getattr(config, "n_heads", 4)),
                dropout=config.dropout,
                pooling_mode=pooling_mode,
                base_alpha=config.pooling_alpha,
                alpha_scale=config.pooling_alpha_scale,
                pooling_d_ff=config.pooling_d_ff,
                pooling_num_queries=int(getattr(config, "pooling_num_queries", 1) or 1),
                summary_source=self.pooling_summary_source,
                use_hierarchical_state_field=self.use_hierarchical_state_field,
                d_global_state=int(getattr(config, "d_global_state", d_model) or d_model),
                d_local_state=int(getattr(config, "d_local_state", d_model) or d_model),
                use_global_state=bool(getattr(config, "pooling_use_global_state", True)),
                use_local_state=bool(getattr(config, "pooling_use_local_state", True)) and self.use_hierarchical_state_field,
                state_fusion_mode=str(getattr(config, "state_fusion_mode", "sum_norm") or "sum_norm"),
                local_pooling_scale_init=float(getattr(config, "local_pooling_scale_init", 0.1)),
                local_pooling_alpha_scale_init=float(getattr(config, "local_pooling_alpha_scale_init", 0.1)),
                local_scale_learnable=bool(getattr(config, "local_scale_learnable", True)),
            )

        self.pooling_summary_proj = None
        if self.pooling_summary_source == "batch":
            self.pooling_summary_proj = nn.Linear(2 * d_model, d_model, bias=False)

        self.head = nn.Linear(int(getattr(self.factor_pooling, "output_dim", d_model)), 1)

        self.post_init()
        with torch.no_grad():
            head_init_std = float(getattr(self.config, "head_init_std", self.config.initializer_range) or 0.02)
            if head_init_std <= 0:
                head_init_std = float(getattr(self.config, "initializer_range", 0.02) or 0.02)
            nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def _init_weights(self, module: nn.Module) -> None:
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        if init_std <= 0:
            init_std = 0.02

        if getattr(module, "_rstmoe_router_init", False):
            if isinstance(module, nn.Linear):
                # Router final layer needs larger variance to break symmetry and prevent logit collapse.
                # Default to 1.0 if not specified, which is much larger than 0.02.
                router_std = float(getattr(self.config, "router_init_std", 1.0))
                nn.init.normal_(module.weight, mean=0.0, std=router_std)
                if module.bias is not None:
                    # Optional: slight bias offset could also help, but zero is safer for now.
                    nn.init.zeros_(module.bias)
            return

        if getattr(module, "_rstmoe_zero_init", False):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            return

        if isinstance(module, FeatureTokenizer):
            module.reset_parameters()
            return

        if isinstance(module, nn.Linear):
            if module.in_features == 1:
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if getattr(module, "padding_idx", None) is not None:
                module.weight.data[module.padding_idx].zero_()
            return

        if isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)
            return

        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            return

        if isinstance(module, nn.MultiheadAttention):
            # Use scaled normal initialization (1 / sqrt(d_model)) instead of the global 0.02
            # to prevent projection logits from collapsing, which leads to Attention Entropy = 0.999
            mha_std = 1.0 / math.sqrt(self.config.d_model)
            if getattr(module, "in_proj_weight", None) is not None:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=mha_std)
            if getattr(module, "in_proj_bias", None) is not None:
                nn.init.zeros_(module.in_proj_bias)
            if getattr(module, "bias_k", None) is not None:
                nn.init.normal_(module.bias_k, mean=0.0, std=mha_std)
            if getattr(module, "bias_v", None) is not None:
                nn.init.normal_(module.bias_v, mean=0.0, std=mha_std)
            return

    def _encode_day_summary(
        self,
        day_summary_features: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.uses_day_summary_asset:
            return None, None
        if day_summary_features is None:
            raise ValueError("day summary mode requires day_summary_features")
        if self.day_summary_encoder is None:
            raise RuntimeError("day_summary_encoder is not initialized.")
        if day_summary_features.ndim != 2:
            raise ValueError(
                f"day_summary_features must be 2D [B,D], got shape {tuple(day_summary_features.shape)}"
            )
        if int(day_summary_features.shape[0]) != int(batch_size):
            raise ValueError(
                f"day_summary_features batch mismatch: expected B={int(batch_size)}, "
                f"got shape {tuple(day_summary_features.shape)}"
            )
        day_state = self.day_summary_encoder(day_summary_features)
        return day_state, day_state

    def forward(
        self,
        x: torch.Tensor,
        factor_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        macro_features: torch.Tensor | None = None,
        day_summary_features: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
        attn_layers: list[int] | None = None,
        local_state_mode: str | None = None,
        expert_mode: str | None = None,
        pooling_mode_override: str | None = None,
    ) -> QuantModelOutput:
        device = x.device
        B, T, N = x.shape
        router_internal_state = None
        if self.use_internal_state_features:
            router_internal_state = compute_legacy_internal_stats(
                x,
                internal_mode=getattr(self.config, "router_internal_mode", "long"),
                internal_lag=int(getattr(self.config, "router_internal_lag", 5)),
                internal_use_batch_stats=bool(getattr(self.config, "router_internal_use_batch_stats", True)),
                internal_tail_threshold=float(getattr(self.config, "router_internal_tail_threshold", 2.0)),
            )

        if factor_ids is None:
            factor_ids = torch.arange(N, device=device)
        factor_ids = factor_ids.to(device)

        num_alphas = int(getattr(self.factor_id_emb, "num_embeddings", 0) or 0)
        if num_alphas > 0 and int(N) > num_alphas:
            raise RuntimeError(
                f"Input x has N={int(N)} factor channels, but the model was initialized with num_alphas={num_alphas}. "
                "This usually means extra channels leaked into x, or num_alphas mismatches the data."
            )
        if factor_ids.numel() != int(N):
            raise RuntimeError(
                "factor_ids must have the same number of elements as x.shape[2] (N). "
                f"Got x.shape={tuple(x.shape)}, factor_ids.shape={tuple(factor_ids.shape)}, "
                f"factor_ids.numel()={int(factor_ids.numel())}."
            )
        factor_ids = factor_ids.reshape(int(N))

        day_summary_embedding, day_summary_day = self._encode_day_summary(day_summary_features, batch_size=B)

        if self.use_hierarchical_state_field:
            if self.global_state_encoder is None or self.local_state_encoder is None:
                raise RuntimeError("Hierarchical state field is enabled but encoders are not initialized.")
            global_state, global_state_day = self.global_state_encoder(
                batch_size=B,
                macro_features=macro_features if self.global_state_use_macro else None,
                day_summary_embedding=day_summary_embedding if self.global_state_use_day_summary else None,
                internal_state=router_internal_state if self.global_state_use_internal_state else None,
                x_features=x if self.global_state_use_x else None,
            )
            local_state, local_stats = self.local_state_encoder(
                x,
                global_state,
                internal_state=router_internal_state if self.local_state_use_internal_state else None,
            )
            if local_state_mode is not None:
                mode = str(local_state_mode).strip().lower()
                if mode == "zero":
                    local_state = torch.zeros_like(local_state)
                elif mode == "shuffle":
                    if int(local_state.shape[0]) > 1:
                        local_state = torch.roll(local_state, shifts=1, dims=0)
                else:
                    raise ValueError(f"Unsupported local_state_mode: {local_state_mode}")
        else:
            if self.regime_encoder is None:
                raise RuntimeError("Legacy regime_encoder is not initialized.")
            global_state = self.regime_encoder(x, macro_features)
            global_state_day = global_state.mean(dim=0, keepdim=True)
            local_state = None
            local_stats = None

        diag_metrics: dict[str, float] = {
            "global_state_norm": float(global_state_day.detach().norm(dim=-1).mean().item()),
            "global_state_cross_sectional_variance": float(
                global_state.detach().var(dim=0, unbiased=False).mean().item()
            ),
            "local_state_norm": 0.0,
            "local_state_cross_sectional_variance": 0.0,
            "time_ratio_stock_std": 0.0,
            "pooling_alpha_stock_std": 0.0,
            "router_global_sensitivity": 0.0,
            "router_local_sensitivity": 0.0,
            "film_global_sensitivity": 0.0,
            "film_local_sensitivity": 0.0,
            "pool_global_sensitivity": 0.0,
            "pool_local_sensitivity": 0.0,
            "router_local_logit_std": 0.0,
            "query_local_norm": 0.0,
            "query_branch_norm": 0.0,
            "alpha_local_logit_std": 0.0,
            "alpha_branch_logit_std": 0.0,
            "local_router_scale": 0.0,
            "local_pooling_scale": 0.0,
            "local_pooling_alpha_scale": 0.0,
            "local_film_scale": 0.0,
            "router_internal_sensitivity": 0.0,
            "router_internal_logit_std": 0.0,
            "router_internal_scale": 0.0,
            "router_internal_crowding": 0.0,
            "router_internal_pc1_ratio": 0.0,
            "router_internal_drift": 0.0,
            "router_internal_tail": 0.0,
            "inner_cross_stock_scale": 0.0,
            "inner_cross_stock_token_std": 0.0,
            "inner_cross_stock_context_norm": 0.0,
            "inner_cross_stock_attention_entropy": 0.0,
        }
        if router_internal_state is not None:
            ris = router_internal_state.detach()
            diag_metrics["router_internal_crowding"] = float(ris[:, 0].mean().item())
            diag_metrics["router_internal_pc1_ratio"] = float(ris[:, 1].mean().item())
            diag_metrics["router_internal_drift"] = float(ris[:, 2].mean().item())
            diag_metrics["router_internal_tail"] = float(ris[:, 3].mean().item())
        if local_state is not None:
            diag_metrics["local_state_norm"] = float(local_state.detach().norm(dim=-1).mean().item())
            diag_metrics["local_state_cross_sectional_variance"] = float(
                local_state.detach().var(dim=0, unbiased=False).mean().item()
            )

        factor_table = self.factor_id_emb(factor_ids.long())
        if self.value_embedding_type == "feature_tokenizer":
            if self.feature_tokenizer is None:
                raise RuntimeError("feature_tokenizer is enabled but not initialized.")
            h = self.feature_tokenizer(x)
            if self.feature_tokenizer_add_factor_id:
                h = h + factor_table.unsqueeze(0).unsqueeze(0)
        else:
            if self.val_proj is None:
                raise RuntimeError("shared_linear embedding is enabled but val_proj is missing.")
            h = self.val_proj(x.unsqueeze(-1)) + factor_table.unsqueeze(0).unsqueeze(0)

        if self.time_embedding is not None:
            time_emb, tau = self.time_embedding(global_state, T)
            h = h + time_emb.unsqueeze(2)
            tau_mean = float(tau.detach().mean().item())
            diag_metrics["time_tau"] = tau_mean
            diag_metrics["time_half_life"] = tau_mean * math.log(2.0)

        factor_film = None
        if self.factor_gate is not None:
            gamma, beta, film_diag = self.factor_gate(global_state, local_state, factor_table)
            factor_film = (gamma, beta)
            diag_metrics.update(film_diag)

            gate_det = gamma.detach().mean(dim=-1)
            diag_metrics["factor_gate_mean"] = float(gate_det.mean().item())
            diag_metrics["factor_gate_std"] = float(gate_det.std(unbiased=False).item())

            eps = 1e-12
            imp = (gate_det - 1.0).abs().clamp_min(eps)
            denom = imp.sum(dim=-1, keepdim=True).clamp_min(eps)
            p = imp / denom
            entropy = -(p * (p + eps).log()).sum(dim=-1)
            entropy_norm = entropy / math.log(max(int(N), 2))
            diag_metrics["factor_gate_entropy"] = float(entropy_norm.mean().item())
            k5 = min(5, int(N))
            k10 = min(10, int(N))
            diag_metrics["factor_gate_topk_mass_5"] = float(p.topk(k5, dim=-1).values.sum(dim=-1).mean().item())
            diag_metrics["factor_gate_topk_mass_10"] = float(p.topk(k10, dim=-1).values.sum(dim=-1).mean().item())

        reg_loss = torch.tensor(0.0, device=device)
        mask = None
        feature_mask = None
        if self.feature_selector is not None:
            z, reg_loss = self.feature_selector.sample_mask(
                temperature=self.config.selection_temperature,
                training=self.training,
            )
            feature_mask = z
            mask = z.detach()

        h = self.emb_dropout(h)

        attn_bias = None
        if self.config.use_alibi:
            bias_time = build_bidirectional_alibi_bias(B, T, self.config.n_heads, device)
            attn_bias = (bias_time, None)

        z_losses = []
        usage_losses = []
        entropies = []
        time_ratios = []
        gates_list = []
        router_global_sens = []
        router_local_sens = []
        router_local_logit_stds = []
        router_stock_logit_stds = []
        router_stock_scales = []
        router_internal_sens = []
        router_internal_logit_stds = []
        router_internal_scales = []
        router_usage_entropies = []
        router_usage_imbalances = []
        inner_cross_stock_scales = []
        inner_cross_stock_token_stds = []
        inner_cross_stock_context_norms = []
        inner_cross_stock_entropies = []
        attn_maps: dict[str, dict[str, torch.Tensor]] = {}

        for idx, layer in enumerate(self.layers):
            need_attn = return_attn and (attn_layers is None or idx in attn_layers)
            h, diag, layer_attn = layer(
                h,
                global_state=global_state,
                local_state=local_state,
                attn_bias=attn_bias,
                return_attn=need_attn,
                summary_context=day_summary_embedding,
                router_internal_state=router_internal_state,
                factor_film=factor_film,
                feature_mask=feature_mask,
                expert_mode=expert_mode,
            )
            z_losses.append(diag["z_loss"])
            usage_losses.append(diag.get("usage_entropy_loss", diag["z_loss"] * 0.0))
            entropies.append(diag["entropy"])
            time_ratios.append(diag["time_ratio"])
            gates_list.append(diag["weights"])
            router_global_sens.append(float(diag.get("router_global_sensitivity", 0.0)))
            router_local_sens.append(float(diag.get("router_local_sensitivity", 0.0)))
            router_local_logit_stds.append(float(diag.get("router_local_logit_std", 0.0)))
            router_stock_logit_stds.append(float(diag.get("router_stock_logit_std", 0.0)))
            router_stock_scales.append(float(diag.get("router_stock_scale", 0.0)))
            router_internal_sens.append(float(diag.get("router_internal_sensitivity", 0.0)))
            router_internal_logit_stds.append(float(diag.get("router_internal_logit_std", 0.0)))
            router_internal_scales.append(float(diag.get("router_internal_scale", 0.0)))
            router_usage_entropies.append(float(diag.get("router_usage_entropy", 0.0)))
            router_usage_imbalances.append(float(diag.get("router_usage_imbalance", 0.0)))
            if "local_router_scale" in diag:
                diag_metrics["local_router_scale"] = float(diag.get("local_router_scale", 0.0))

            if need_attn and layer_attn is not None:
                attn_maps[f"layer_{idx}"] = layer_attn

            if self.inner_cross_stock_layers is not None:
                h, inner_diag = self.inner_cross_stock_layers[idx](h)
                inner_cross_stock_scales.append(float(inner_diag.get("inner_cross_stock_scale", 0.0)))
                inner_cross_stock_token_stds.append(float(inner_diag.get("inner_cross_stock_token_std", 0.0)))
                inner_cross_stock_context_norms.append(float(inner_diag.get("inner_cross_stock_context_norm", 0.0)))
                inner_cross_stock_entropies.append(
                    float(inner_diag.get("inner_cross_stock_attention_entropy", 0.0))
                )

        if router_global_sens:
            diag_metrics["router_global_sensitivity"] = float(sum(router_global_sens) / len(router_global_sens))
        if router_local_sens:
            diag_metrics["router_local_sensitivity"] = float(sum(router_local_sens) / len(router_local_sens))
        if router_local_logit_stds:
            diag_metrics["router_local_logit_std"] = float(
                sum(router_local_logit_stds) / len(router_local_logit_stds)
            )
        if router_stock_logit_stds:
            diag_metrics["router_stock_logit_std"] = float(
                sum(router_stock_logit_stds) / len(router_stock_logit_stds)
            )
        if router_stock_scales:
            diag_metrics["router_stock_scale"] = float(sum(router_stock_scales) / len(router_stock_scales))
        if router_internal_sens:
            diag_metrics["router_internal_sensitivity"] = float(
                sum(router_internal_sens) / len(router_internal_sens)
            )
        if router_internal_logit_stds:
            diag_metrics["router_internal_logit_std"] = float(
                sum(router_internal_logit_stds) / len(router_internal_logit_stds)
            )
        if router_internal_scales:
            diag_metrics["router_internal_scale"] = float(sum(router_internal_scales) / len(router_internal_scales))
        if router_usage_entropies:
            diag_metrics["router_usage_entropy"] = float(sum(router_usage_entropies) / len(router_usage_entropies))
        if router_usage_imbalances:
            diag_metrics["router_usage_imbalance"] = float(
                sum(router_usage_imbalances) / len(router_usage_imbalances)
            )
        if inner_cross_stock_scales:
            diag_metrics["inner_cross_stock_scale"] = float(
                sum(inner_cross_stock_scales) / len(inner_cross_stock_scales)
            )
            diag_metrics["inner_cross_stock_token_std"] = float(
                sum(inner_cross_stock_token_stds) / len(inner_cross_stock_token_stds)
            )
            diag_metrics["inner_cross_stock_context_norm"] = float(
                sum(inner_cross_stock_context_norms) / len(inner_cross_stock_context_norms)
            )
            diag_metrics["inner_cross_stock_attention_entropy"] = float(
                sum(inner_cross_stock_entropies) / len(inner_cross_stock_entropies)
            )

        h = self.final_norm(h)
        h_last, temporal_diag = self.temporal_pooling(h)
        diag_metrics.update(temporal_diag)
        if self.cross_stock_attention is not None:
            h_last, cross_stock_diag = self.cross_stock_attention(h_last)
            diag_metrics.update(cross_stock_diag)
        else:
            diag_metrics.update(
                {
                    "cross_stock_scale": 0.0,
                    "cross_stock_token_std": 0.0,
                    "cross_stock_context_norm": 0.0,
                    "cross_stock_attention_entropy": 0.0,
                }
            )

        summary_context = None
        if self.pooling_summary_source == "batch":
            per_stock = h_last.mean(dim=1)
            m_mean = per_stock.mean(dim=0)
            m_std = per_stock.std(dim=0, unbiased=False)
            summary_raw = torch.cat([m_mean, m_std], dim=-1)
            summary_context = self.pooling_summary_proj(summary_raw).unsqueeze(0).expand(B, -1)
        elif self.pooling_summary_source == "day_asset":
            summary_context = day_summary_embedding

        h_pooled, factor_attention_weights, alpha_val, pooling_diag = self.factor_pooling(
            h_last,
            global_state=global_state,
            local_state=local_state,
            summary_context=summary_context,
            pooling_mode_override=pooling_mode_override,
        )
        diag_metrics.update(pooling_diag)

        stock_score = self.head(h_pooled).squeeze(-1)
        factor_pool_weights = factor_attention_weights

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
                l_listmle = QuantLossFunctions.listmle_loss(
                    p,
                    y,
                    tau=getattr(self.config, "listmle_tau", 1.0),
                )
                l_ic = QuantLossFunctions.cs_ic_loss(p, y)
                l_mse = QuantLossFunctions.cs_mse_loss(
                    p,
                    y,
                    normalize=bool(getattr(self.config, "mse_normalize", False)),
                )

                l_rank = None
                if w.get("rank", 0.0) != 0.0:
                    l_rank = QuantLossFunctions.ranknet_topbottom_loss(p, y, self.config.rank_topk)

                l_huber = None
                if w.get("huber", 0.0) != 0.0:
                    l_huber = QuantLossFunctions.cs_huber_loss(p, y, self.config.huber_delta)

                aux_type = str(getattr(self.config, "router_aux_loss_type", "usage_entropy") or "usage_entropy").strip().lower()
                z_stability = (
                    torch.stack(z_losses).mean() * float(getattr(self.config, "router_z_stability_coef", 1e-3))
                    if z_losses
                    else torch.tensor(0.0, device=device)
                )
                if aux_type == "usage_entropy":
                    l_aux = (
                        torch.stack(usage_losses).mean()
                        * float(getattr(self.config, "router_usage_entropy_coef", 0.05))
                        if usage_losses
                        else torch.tensor(0.0, device=device)
                    )
                    l_aux = l_aux + z_stability
                else:
                    l_aux = (
                        torch.stack(z_losses).mean() * self.config.router_z_loss_coef
                        if z_losses
                        else torch.tensor(0.0, device=device)
                    )
                l_reg = reg_loss * self.config.selection_reg_lambda

                main_loss = str(getattr(self.config, "main_loss", "listmle")).lower()
                if main_loss == "mle":
                    main_loss = "listmle"
                loss_map = {
                    "listmle": l_listmle,
                    "mse": l_mse,
                    "ic": l_ic,
                }
                if main_loss not in loss_map:
                    raise RuntimeError(
                        f"Unsupported main_loss: {main_loss}. Supported: {sorted(loss_map)}"
                    )
                l_main = loss_map[main_loss]

                total_loss = w.get(main_loss, 1.0) * l_main + l_aux + l_reg
                if l_rank is not None:
                    total_loss = total_loss + w.get("rank", 0.0) * l_rank
                if l_huber is not None:
                    total_loss = total_loss + w.get("huber", 0.0) * l_huber

                metrics = {
                    "loss_total": float(total_loss.detach().item()),
                    "loss_main": float(l_main.detach().item()),
                    "loss_listmle": float(l_listmle.detach().item()),
                    "loss_mse": float(l_mse.detach().item()),
                    "loss_ic": float(l_ic.detach().item()),
                    "loss_aux": float(l_aux.detach().item()),
                    "loss_router_z_stability": float(z_stability.detach().item()),
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

        if alpha_val is not None:
            alpha_mean = float(alpha_val.mean().item())
            alpha_std = float(alpha_val.std(unbiased=False).item())
            metrics["pooling_alpha_mean"] = alpha_mean
            metrics["pooling_alpha_std"] = alpha_std
            diag_metrics["pooling_alpha_stock_std"] = alpha_std

        if gates_list:
            gate_tensor = torch.stack(gates_list, dim=0)
            time_ratio_stock = gate_tensor[:, :, 0].mean(dim=0)
            diag_metrics["time_ratio_stock_std"] = float(time_ratio_stock.std(unbiased=False).item())

        if diag_metrics:
            metrics = {**diag_metrics, **metrics}

        avg_entropy = float(torch.stack(entropies).mean().detach().item()) if entropies else None
        avg_time_ratio = float(torch.stack(time_ratios).mean().detach().item()) if time_ratios else None

        return QuantModelOutput(
            loss=total_loss,
            logits=stock_score,
            gate_weights=gates_list,
            metrics=metrics,
            avg_gate_entropy=avg_entropy,
            avg_time_ratio=avg_time_ratio,
            selected_mask=mask,
            attn_maps=attn_maps if return_attn else None,
            factor_pool_weights=factor_pool_weights,
            scores=stock_score,
        )
