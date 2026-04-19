import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from transformers import PretrainedConfig
from transformers.utils import ModelOutput


# ==========================================
# 1. 配置类 (Configuration)
# ==========================================
class QuantMoEConfig(PretrainedConfig):
    model_type = "quant_moe"

    def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 4,
            d_ff: int = 128,
            dropout: float = 0.1,
            initializer_range: float = 0.02,
            num_alphas: int = 64,
            context_len: int = 32,
            # value embedding
            value_embedding_type: str = "shared_linear",
            feature_tokenizer_bias: bool = True,
            feature_tokenizer_add_factor_id: bool = False,
            feature_tokenizer_init_std: float = 0.02,
            # regime-adaptive embeddings (lightweight, recommended)
            use_regime_time_embedding: bool = True,
            time_tau_min: float = 0.5,
            time_tau_max: float = 50.0,
            time_tau_init: float = 5.0,
            time_emb_init_std: float = 0.02,
            time_decay_normalize: bool = True,
            use_regime_factor_gate: bool = True,
            factor_gate_scale: float = 0.5,
            factor_gate_shift_scale: float = 0.0,
            # router (MoE gate)
            router_noise: float = 0.01,       # logit noise std (training only)
            router_temperature: float = 1.0, # softmax temperature (lower => sharper)
            router_z_loss_coef: float = 0.01,  # 防止 collapse，比原 1e-3 更安全
            router_aux_loss_type: str = "usage_entropy",
            router_usage_entropy_coef: float = 0.01,
            router_use_stock_token: bool = True,
            router_stock_scale_init: float = 0.1,
            router_stock_scale_learnable: bool = True,
            router_use_layer_summary: bool = False,  # add per-layer market summary token to router input
            router_summary_source: Optional[str] = None,
            temporal_pooling_mode: str = "learned_query_mha_v1",
            temporal_pooling_heads: int = 4,
            use_cross_stock_attention: bool = True,
            cross_stock_attention_heads: int = 4,
            cross_stock_residual_scale_init: float = 0.1,
            cross_stock_scale_learnable: bool = True,
            use_alibi: bool = False,
            use_feature_selection: bool = False,
            selection_reg_lambda: float = 1e-5,  # 修复后降低（原 1e-3 会过强）
            selection_temperature: float = 0.1,
            selection_noise_std: float = 0.5,
            # Loss & ranking
            main_loss: str = "mse",
            loss_weights: Optional[Dict[str, float]] = None,
            mse_normalize: bool = False,
            rank_topk: int = 5,
            huber_delta: float = 1.0,
            listmle_tau: float = 0.8,  # ★ 新增：ListMLE 温度
            # context encoder
            use_external_macro: bool = True,
            d_macro_input: int = 0,
            d_day_summary_input: int = 0,
            regime_macro_dropout: float = 0.0,
            use_hierarchical_state_field: bool = False,
            d_global_state: Optional[int] = None,
            d_local_state: Optional[int] = None,
            global_state_use_macro: bool = True,
            global_state_use_day_summary: bool = True,
            local_state_input_mode: str = "last_mean_std_trend_vol",
            state_fusion_mode: str = "sum_norm",
            local_router_scale_init: float = 0.1,
            local_pooling_scale_init: float = 0.1,
            local_pooling_alpha_scale_init: float = 0.1,
            local_film_scale_init: float = 0.1,
            local_scale_learnable: bool = True,
            router_use_global_state: bool = True,
            router_use_local_state: bool = True,
            film_use_global_state: bool = True,
            film_use_local_state: bool = True,
            pooling_use_global_state: bool = True,
            pooling_use_local_state: bool = True,
            # internal regime stats (used when use_external_macro=False)
            regime_internal_mode: str = "long",  # "short" (t+1-ish) or "long" (t+5-ish)
            regime_internal_lag: int = 5,         # effective when mode="long"
            regime_internal_use_batch_stats: bool = False,
            regime_internal_tail_threshold: float = 2.0,
            # pooling
            pooling_alpha: float = 0.25,  # Base weight for attention vs mean pooling
            pooling_mode: str = "adaptive_alpha",  # "static", "adaptive_alpha", "conditioned_query", "full"
            pooling_alpha_scale: float = 0.15,      # Scaling factor for adaptive alpha
            pooling_d_ff: Optional[int] = None,    # FFN dimension (None = d_model)
            pooling_num_queries: int = 4,
            pooling_use_layer_summary: bool = False, # If True, condition on layer summary
            pooling_summary_source: Optional[str] = None,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.num_alphas = num_alphas
        self.context_len = context_len

        value_embedding_type = str(value_embedding_type).strip().lower()
        embedding_alias = {
            "shared": "shared_linear",
            "shared_linear": "shared_linear",
            "feature_tokenizer": "feature_tokenizer",
            "ft_tokenizer": "feature_tokenizer",
            "ft": "feature_tokenizer",
        }
        value_embedding_type = embedding_alias.get(value_embedding_type, value_embedding_type)
        allowed_embedding = {"shared_linear", "feature_tokenizer"}
        if value_embedding_type not in allowed_embedding:
            raise ValueError(
                f"Unsupported value_embedding_type: {value_embedding_type}. "
                f"Supported: {sorted(allowed_embedding)}"
            )
        self.value_embedding_type = value_embedding_type
        self.feature_tokenizer_bias = bool(feature_tokenizer_bias)
        self.feature_tokenizer_add_factor_id = bool(feature_tokenizer_add_factor_id)
        self.feature_tokenizer_init_std = float(feature_tokenizer_init_std)

        # regime-adaptive embeddings
        self.use_regime_time_embedding = use_regime_time_embedding
        self.time_tau_min = time_tau_min
        self.time_tau_max = time_tau_max
        self.time_tau_init = time_tau_init
        self.time_emb_init_std = time_emb_init_std
        self.time_decay_normalize = time_decay_normalize

        self.use_regime_factor_gate = use_regime_factor_gate
        self.factor_gate_scale = factor_gate_scale
        self.factor_gate_shift_scale = factor_gate_shift_scale

        self.router_noise = router_noise
        self.router_temperature = router_temperature
        self.router_z_loss_coef = router_z_loss_coef
        router_aux_loss_type = str(router_aux_loss_type or "z_loss").strip().lower()
        if router_aux_loss_type not in {"z_loss", "usage_entropy"}:
            raise ValueError("router_aux_loss_type must be one of: z_loss, usage_entropy.")
        self.router_aux_loss_type = router_aux_loss_type
        self.router_usage_entropy_coef = float(router_usage_entropy_coef)
        self.router_use_stock_token = bool(router_use_stock_token)
        self.router_stock_scale_init = float(router_stock_scale_init)
        self.router_stock_scale_learnable = bool(router_stock_scale_learnable)
        self.router_use_layer_summary = bool(router_use_layer_summary)
        self.temporal_pooling_mode = str(temporal_pooling_mode or "last").strip().lower()
        if self.temporal_pooling_mode not in {"last", "learned_query_mha_v1"}:
            raise ValueError("temporal_pooling_mode must be one of: last, learned_query_mha_v1.")
        self.temporal_pooling_heads = int(temporal_pooling_heads)
        self.use_cross_stock_attention = bool(use_cross_stock_attention)
        self.cross_stock_attention_heads = int(cross_stock_attention_heads)
        self.cross_stock_residual_scale_init = float(cross_stock_residual_scale_init)
        self.cross_stock_scale_learnable = bool(cross_stock_scale_learnable)
        self.use_alibi = use_alibi

        def _resolve_summary_source(explicit: Optional[str], legacy_flag: bool, *, field: str) -> str:
            if explicit is None:
                source = "batch" if bool(legacy_flag) else "none"
            else:
                source = str(explicit).strip().lower()
            valid_sources = {"none", "batch", "day_asset"}
            if source not in valid_sources:
                raise ValueError(
                    f"Unsupported {field}: {source}. Supported: {sorted(valid_sources)}"
                )
            return source

        self.router_summary_source = _resolve_summary_source(
            router_summary_source,
            self.router_use_layer_summary,
            field="router_summary_source",
        )

        self.use_feature_selection = use_feature_selection
        self.selection_reg_lambda = selection_reg_lambda
        self.selection_temperature = selection_temperature

        self.selection_noise_std = selection_noise_std
        main_loss = str(main_loss).lower().strip()
        if main_loss == "mle":
            main_loss = "listmle"
        allowed_main_loss = {"mse", "ic", "listmle"}
        if main_loss not in allowed_main_loss:
            raise ValueError(
                f"Unsupported main_loss: {main_loss}. Supported: {sorted(allowed_main_loss)}"
            )
        self.main_loss = main_loss
        # 默认 Loss 权重
        # 注意：aux 和 reg 已移除，直接由 router_z_loss_coef 和 selection_reg_lambda 控制
        self.loss_weights = loss_weights if loss_weights is not None else {
            "listmle": 1.0,
            "mse": 1.0,
            "ic": 1.0,
            "rank": 0.0,
            "huber": 0.0,
        }

        self.listmle_tau = listmle_tau
        self.rank_topk = rank_topk
        self.huber_delta = huber_delta
        self.mse_normalize = bool(mse_normalize)

        self.use_external_macro = use_external_macro
        self.d_macro_input = d_macro_input
        self.d_day_summary_input = int(d_day_summary_input or 0)
        self.regime_macro_dropout = regime_macro_dropout
        self.use_hierarchical_state_field = bool(use_hierarchical_state_field)
        self.d_global_state = int(d_model if d_global_state is None else d_global_state)
        self.d_local_state = int(d_model if d_local_state is None else d_local_state)
        self.global_state_use_macro = bool(global_state_use_macro)
        self.global_state_use_day_summary = bool(global_state_use_day_summary)
        self.local_state_input_mode = str(local_state_input_mode or "last_mean_std_trend_vol").strip().lower()
        self.state_fusion_mode = str(state_fusion_mode or "sum_norm").strip().lower()
        if self.state_fusion_mode not in {"sum_norm", "branch_mlp_v1", "bounded_sum_v1"}:
            raise ValueError("state_fusion_mode must be one of: sum_norm, branch_mlp_v1, bounded_sum_v1.")
        self.local_router_scale_init = float(local_router_scale_init)
        self.local_pooling_scale_init = float(local_pooling_scale_init)
        self.local_pooling_alpha_scale_init = float(local_pooling_alpha_scale_init)
        self.local_film_scale_init = float(local_film_scale_init)
        self.local_scale_learnable = bool(local_scale_learnable)
        self.router_use_global_state = bool(router_use_global_state)
        self.router_use_local_state = bool(router_use_local_state)
        self.film_use_global_state = bool(film_use_global_state)
        self.film_use_local_state = bool(film_use_local_state)
        self.pooling_use_global_state = bool(pooling_use_global_state)
        self.pooling_use_local_state = bool(pooling_use_local_state)
        if self.use_hierarchical_state_field:
            if self.d_global_state <= 0 or self.d_local_state <= 0:
                raise ValueError("Hierarchical state field requires d_global_state > 0 and d_local_state > 0.")
            if not (self.global_state_use_macro or self.global_state_use_day_summary):
                raise ValueError(
                    "Hierarchical state field requires at least one global-state input branch "
                    "(global_state_use_macro or global_state_use_day_summary)."
                )

        self.regime_internal_mode = regime_internal_mode
        self.regime_internal_lag = regime_internal_lag
        self.regime_internal_use_batch_stats = regime_internal_use_batch_stats
        self.regime_internal_tail_threshold = regime_internal_tail_threshold
        
        self.pooling_alpha = pooling_alpha
        self.pooling_mode = pooling_mode
        self.pooling_alpha_scale = pooling_alpha_scale
        self.pooling_d_ff = pooling_d_ff
        self.pooling_num_queries = int(pooling_num_queries)
        if self.pooling_num_queries <= 0:
            raise ValueError("pooling_num_queries must be positive.")
        self.pooling_use_layer_summary = bool(pooling_use_layer_summary)
        self.pooling_summary_source = _resolve_summary_source(
            pooling_summary_source,
            self.pooling_use_layer_summary,
            field="pooling_summary_source",
        )

        valid_pooling_modes = {"static", "adaptive_alpha", "conditioned_query", "full"}
        if self.pooling_mode not in valid_pooling_modes:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}. Supported: {valid_pooling_modes}")


# ==========================================
# 2. 输出数据结构 (Output Dataclass)
# ==========================================
@dataclass
class QuantModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    # Model prediction logits (kept for HF-style compatibility).
    # For RST-MoE, this is the per-sample stock score prediction: [B]
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    # 每层 router gate 权重: List[num_layers] of [B, 2]
    gate_weights: Optional[List[torch.FloatTensor]] = None

    # 详细 Loss 组件 (用于监控)
    metrics: Optional[Dict[str, float]] = None

    # 诊断信息
    avg_gate_entropy: Optional[float] = None
    avg_time_ratio: Optional[float] = None
    selected_mask: Optional[torch.FloatTensor] = None

    # 新增: 注意力图 (只在需要时填充)
    # 约定: { "layer_0": {"time": Tensor, "factor": Tensor}, ... }
    attn_maps: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    # Attention pooling weights over factors (for interpretability): [B, N]
    factor_pool_weights: Optional[torch.FloatTensor] = None

    # scores 用于predict
    scores: Optional[torch.FloatTensor] = None
