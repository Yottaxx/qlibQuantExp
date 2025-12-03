import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput


# ==========================================
# 1. 配置类 (Configuration)
# ==========================================
class QuantMoEConfig(PretrainedConfig):
    model_type = "quant_moe"

    def __init__(
            self,
            d_model: int = 256,
            n_heads: int = 4,
            n_layers: int = 4,
            d_ff: int = 1024,
            dropout: float = 0.1,
            num_alphas: int = 50,  # 因子数量 (N)
            context_len: int = 100,  # 时间窗口 (T)
            num_dates: int = 5000,  # 最大日期ID (用于 Regime Gating)

            # MoE & Routing
            router_noise: float = 0.1,  # Noisy Gating 标准差
            router_z_loss_coef: float = 1e-3,  # Z-Loss 系数 (防坍塌)

            # Attention
            use_alibi: bool = True,  # 是否使用 ALiBi

            # Feature Selection (STG)
            use_feature_selection: bool = True,
            selection_reg_lambda: float = 0.01,  # L1 正则系数 (稀疏度控制)
            selection_temperature: float = 0.1,  # Gumbel Softmax 温度

            selection_noise_std: float = 0.5,  # STG noise std (feature gate)
            # Loss Weights & Params
            loss_weights: Optional[Dict[str, float]] = None,
            rank_topk: int = 5,  # RankNet 关注头部 K 只股票
            huber_delta: float = 1.0,  # Huber Loss 阈值

            #context encoder
            use_external_macro: bool = False,
            d_macro_input: int = 0,  # 外部宏观数据的维度

            **kwargs
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_alphas = num_alphas
        self.context_len = context_len
        self.num_dates = num_dates

        self.router_noise = router_noise
        self.router_z_loss_coef = router_z_loss_coef
        self.use_alibi = use_alibi

        self.use_feature_selection = use_feature_selection
        self.selection_reg_lambda = selection_reg_lambda
        self.selection_temperature = selection_temperature

        self.selection_noise_std = selection_noise_std
        # 默认 Loss 权重
        self.loss_weights = loss_weights if loss_weights is not None else {
            "ic": 1.0,
            "huber": 0.1,
            "rank": 0.1,
            "aux": 1.0,  # Z-Loss
            "reg": 1.0  # Feature Selection L1
        }
        self.rank_topk = rank_topk
        self.huber_delta = huber_delta

        self.use_external_macro = use_external_macro
        self.d_macro_input = d_macro_input


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