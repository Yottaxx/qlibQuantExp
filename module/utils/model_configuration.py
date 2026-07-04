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
            time_tau_mlp_out_scale: float = 0.01,
            use_regime_factor_gate: bool = True,
            factor_gate_scale: float = 0.5,
            factor_gate_shift_scale: float = 0.0,
            # router (MoE gate)
            router_noise: float = 0.01,       # logit noise std (training only)
            router_temperature: float = 1.0, # softmax temperature (lower => sharper)
            router_z_loss_coef: float = 0.01,  # 防止 collapse，比原 1e-3 更安全
            router_use_layer_summary: bool = False,  # add per-layer market summary token to router input
            router_mode: str = "learned",
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
            regime_macro_dropout: float = 0.0,
            # internal regime stats (used when use_external_macro=False)
            regime_internal_mode: str = "long",  # "short" (t+1-ish) or "long" (t+5-ish)
            regime_internal_lag: int = 5,         # effective when mode="long"
            regime_internal_use_batch_stats: bool = False,
            regime_internal_tail_threshold: float = 2.0,
            # pooling
            pooling_alpha: float = 0.7,  # Weight for attention vs mean pooling
            pool_n_heads: int = 1,       # attention-pool heads (1=current; >1 for pool-forensics R1)
            # temporal readout (time-readout-bonus-20260607): learned temporal aggregation over T
            # BEFORE the (unchanged) factor pool. ""=current last-step behavior.
            temporal_readout: str = "",
            # init-ablation (task #14): init of the temporal-AGGREGATION params (tr_A/tr_A_N/tr_collapse).
            # "onehot_last"=z==h[:,-1] (control-nesting); "uniform_mean"=1/T+noise (sum-preserving diverse).
            temporal_readout_init: str = "onehot_last",
            # d1pma/duals tr_gate initial value (0.0=>g=0.5 current; -2=>g~0.12; +2=>g~0.88). Gate-init sweep.
            temporal_readout_gate_init: float = 0.0,
            # h-20260610-002: third stock-axis expert (cross-sectional MHA over the daily batch),
            # symmetric peer of time/factor experts, router 2->3. Default off => baseline unchanged.
            use_stock_expert: bool = False,
            # h-20260610-002 follow-up: subtract the per-day cross-sectional mean from the stock
            # expert output, so it can only contribute the rank-changing relative component (kills
            # the loss-neutral uniform-attention attractor). Default off => baseline unchanged.
            stock_expert_demean: bool = False,
            # h-20260610-002 follow-up #2: center the stock expert INPUT across the daily cross-
            # section before projection (centers q/k/v) so attention keys carry only the stock-
            # relative signal, not the common-mode time/factor embedding. Default off.
            stock_expert_xs_center: bool = False,
            # h-20260617: normalize the stock-expert OUTPUT before de-mean/fusion to close the
            # "V-amplification escape" (uniform attention + ||Wv||||Wo||->88 instead of sharpening).
            # Modes: "none"(default,baseline) | "rms" | "unit" | "ln" | "ln_affine" | "block_rms"
            #        | "cap:<float>". Scale-invariant modes (rms/unit/ln) foreclose the escape;
            # ln_affine REOPENS it via gamma (diagnostic only). Default "none" => baseline unchanged.
            stock_expert_out_norm: str = "none",
            # h-20260624: cross-stock attention as a PRE-MoE MAIN-PATH residual (NOT a routed expert).
            # x = x + gamma * StockAttn(LN(x)); NO de-mean / NO out_norm => removes the V-escape driver
            # that killed the routed-expert form (entropy froze ~1.0 + ||out||->65-100 under de-mean).
            # gamma is a learnable scalar init from gamma_init: 0.0 = ReZero opt-in (branch starts a
            # no-op; only engages if cross-stock context helps), 1.0 = full-on from step 1. Router stays
            # 2-way (time/factor). Pair with expert_no_wd_scope="stock_backbone" to keep warm-q's wd-free
            # q/k. Default off => baseline unchanged.
            stock_backbone: bool = False,
            stock_backbone_gamma_init: float = 0.0,
            # h-20260619: cross-stock self-attention at the READOUT (post-pool, pre-head), ungated/
            # mainpath/in-series. Migration target of warm-q: removes all 3 warm-q pathologies at once
            # (no de-mean => no V-escape; QK-norm+temp => cold-query immune; ungated residual =>
            # load-bearing, no router exit). R1=qknorm on (the bet); R0=qknorm off (fixed 1/sqrt(d),
            # placement-only control); gated=cold-sigmoid control (R2). Default off => baseline-identical.
            use_readout_stock_attn: bool = False,
            readout_stock_attn_qknorm: bool = True,
            readout_stock_attn_gated: bool = False,
            readout_stock_attn_heads: int = 0,          # 0 => reuse n_heads
            readout_stock_attn_temp_init: float = 4.0,  # mild => entropy starts ~1, DROP = sharpened
            readout_stock_attn_ffn: bool = False,
            # h-20260704 C1: use the CONTRAST operator c_i=Wv·u_i−Σ_j a_ij·Wv·u_j instead of the standard
            # pooling o=Σ_j a_ij·Wv·u_j at the readout. Uniform attention ⇒ c_i=Wv·(u_i−mean(u)) = the
            # pure cross-sectional-demeaned coordinate (common-mode-free, V-escape-immune, CS-blind
            # backbone cannot produce it). Default off => R1 pooling behavior unchanged.
            readout_stock_attn_contrast: bool = False,
            # ---- Portfolio-IR auxiliary loss (h-20260627-001, L-6; default OFF => anchor byte-identical) ----
            # Soft long-short-return / IC-family aux on the per-day book; MSE stays the main loss.
            # NOTE the variance denominator is DETACHED (no gradient on variance) => honestly NOT a true Sharpe.
            ir_aux_lambda: float = 0.0,          # 0 => disabled; target weight reached after the ramp
            ir_aux_ramp_steps: int = 5000,       # linear ramp 0->ir_aux_lambda over this many PER-DAY FORWARD
                                                 # steps (microbatches; ir_step++ per forward, NOT optimizer
                                                 # steps -> with grad_accum_steps=K it is K*optimizer-steps)
            ir_aux_var_eps: float = 1e-6,
            ir_aux_ema_decay: float = 0.99,      # EMA decay for the detached book-return mean/var
            # ---- Memory: gradient checkpointing of the MoE layers (default OFF => byte-identical) ----
            # When ON, each RegimeAdaptiveMoEBlock forward is wrapped in torch.utils.checkpoint
            # (use_reentrant=False, RNG preserved => dropout masks match => numerically exact). Only the
            # layer INPUT [B,T,N,D] is kept; the attention/FFN activations are recomputed in backward.
            # Needed for wide-feature runs (e.g. L-4 CS-rank append => N=316) on a 12GB card where the
            # full activation graph exceeds VRAM. No effect at eval (eval runs under no_grad).
            use_grad_checkpoint: bool = False,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.use_grad_checkpoint = bool(use_grad_checkpoint)

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
        self.time_tau_mlp_out_scale = float(time_tau_mlp_out_scale)

        self.use_regime_factor_gate = use_regime_factor_gate
        self.factor_gate_scale = factor_gate_scale
        self.factor_gate_shift_scale = factor_gate_shift_scale

        self.router_noise = router_noise
        self.router_temperature = router_temperature
        self.router_z_loss_coef = router_z_loss_coef
        self.router_use_layer_summary = router_use_layer_summary
        router_mode = str(router_mode or "learned").strip().lower()
        router_alias = {
            "learn": "learned",
            "learned": "learned",
            "fixed": "fixed_05",
            "fixed05": "fixed_05",
            "fixed_05": "fixed_05",
            "fixed_0.5": "fixed_05",
            "uniform": "fixed_05",
        }
        router_mode = router_alias.get(router_mode, router_mode)
        if router_mode not in {"learned", "fixed_05"}:
            raise ValueError("Unsupported router_mode: %s. Supported: learned, fixed_05" % router_mode)
        self.router_mode = router_mode
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
        self.mse_normalize = bool(mse_normalize)

        self.use_external_macro = use_external_macro
        self.d_macro_input = d_macro_input
        self.regime_macro_dropout = regime_macro_dropout

        self.regime_internal_mode = regime_internal_mode
        self.regime_internal_lag = regime_internal_lag
        self.regime_internal_use_batch_stats = regime_internal_use_batch_stats
        self.regime_internal_tail_threshold = regime_internal_tail_threshold
        
        self.pooling_alpha = pooling_alpha
        self.pool_n_heads = int(pool_n_heads)

        # temporal readout design (see quant_moe_model.forward). Allow-list guards typos.
        temporal_readout = str(temporal_readout or "").strip().lower()
        allowed_temporal_readout = {"", "d3cid", "d3cin", "d3mix", "d1pma", "duala", "dualb"}
        if temporal_readout not in allowed_temporal_readout:
            raise ValueError(
                "Unsupported temporal_readout: %s. Supported: %s"
                % (temporal_readout, sorted(allowed_temporal_readout))
            )
        self.temporal_readout = temporal_readout

        # init-ablation knobs (task #14). Allow-list guards typos; gate_init is a free float.
        temporal_readout_init = str(temporal_readout_init or "onehot_last").strip().lower()
        allowed_tr_init = {"onehot_last", "uniform_mean"}
        if temporal_readout_init not in allowed_tr_init:
            raise ValueError(
                "Unsupported temporal_readout_init: %s. Supported: %s"
                % (temporal_readout_init, sorted(allowed_tr_init))
            )
        self.temporal_readout_init = temporal_readout_init
        self.temporal_readout_gate_init = float(temporal_readout_gate_init)
        self.use_stock_expert = bool(use_stock_expert)
        self.stock_expert_demean = bool(stock_expert_demean)
        self.stock_expert_xs_center = bool(stock_expert_xs_center)
        self.stock_expert_out_norm = str(stock_expert_out_norm or "none").strip().lower()
        self.stock_backbone = bool(stock_backbone)
        self.stock_backbone_gamma_init = float(stock_backbone_gamma_init)
        self.use_readout_stock_attn = bool(use_readout_stock_attn)
        self.readout_stock_attn_qknorm = bool(readout_stock_attn_qknorm)
        self.readout_stock_attn_gated = bool(readout_stock_attn_gated)
        self.readout_stock_attn_heads = int(readout_stock_attn_heads)
        self.readout_stock_attn_temp_init = float(readout_stock_attn_temp_init)
        self.readout_stock_attn_ffn = bool(readout_stock_attn_ffn)
        self.readout_stock_attn_contrast = bool(readout_stock_attn_contrast)

        # Portfolio-IR auxiliary loss (h-20260627-001, L-6)
        self.ir_aux_lambda = float(ir_aux_lambda)
        self.ir_aux_ramp_steps = int(ir_aux_ramp_steps)
        self.ir_aux_var_eps = float(ir_aux_var_eps)
        self.ir_aux_ema_decay = float(ir_aux_ema_decay)


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

    # Required diagnostic matrix support. These are optional and only populated
    # when the corresponding module is enabled.
    router_logit_margins: Optional[List[torch.FloatTensor]] = None
    router_entropy_values: Optional[List[torch.FloatTensor]] = None
    time_tau_values: Optional[torch.FloatTensor] = None
    factor_gate_importance: Optional[torch.FloatTensor] = None
    film_gamma_strength_by_factor: Optional[torch.FloatTensor] = None
    film_beta_strength_by_factor: Optional[torch.FloatTensor] = None

    # scores 用于predict
    scores: Optional[torch.FloatTensor] = None
