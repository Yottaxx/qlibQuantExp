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
            router_noise: float = 0.1,       # logit noise std (training only)
            router_temperature: float = 1.0, # softmax temperature (lower => sharper)
            router_z_loss_coef: float = 0.01,  # 防止 collapse，比原 1e-3 更安全
            router_use_layer_summary: bool = True,  # add per-layer market summary token to router input
            use_alibi: bool = False,
            use_feature_selection: bool = False,
            selection_reg_lambda: float = 1e-5,  # 修复后降低（原 1e-3 会过强）
            selection_temperature: float = 0.1,
            selection_noise_std: float = 0.5,
            # Loss & ranking
            main_loss: str = "ic",
            loss_weights: Optional[Dict[str, float]] = None,
            rank_topk: int = 5,
            huber_delta: float = 1.0,
            listmle_tau: float = 0.8,  # ★ 新增：ListMLE 温度
            # context encoder
            use_external_macro: bool = True,
            d_macro_input: int = 0,
            regime_macro_dropout: float = 0.0,
            # internal regime stats (used when use_external_macro=False)
            regime_internal_mode: str = "long",  # "short" (t+1-ish) or "long" (t+5-ish)
            regime_internal_lag: int = 5,         # effective when mode="long"
            regime_internal_use_batch_stats: bool = True,
            regime_internal_tail_threshold: float = 2.0,
            # pooling
            pooling_alpha: float = 0.7,  # Weight for attention vs mean pooling
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
        self.router_use_layer_summary = router_use_layer_summary
        self.use_alibi = use_alibi

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

        self.use_external_macro = use_external_macro
        self.d_macro_input = d_macro_input
        self.regime_macro_dropout = regime_macro_dropout

        self.regime_internal_mode = regime_internal_mode
        self.regime_internal_lag = regime_internal_lag
        self.regime_internal_use_batch_stats = regime_internal_use_batch_stats
        self.regime_internal_tail_threshold = regime_internal_tail_threshold
        
        self.pooling_alpha = pooling_alpha


# ==========================================
# 2. 输出数据结构 (Output Dataclass)
# ==========================================
@dataclass
class QuantModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
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

    # scores 用于predict
    scores: Optional[torch.FloatTensor] = None
