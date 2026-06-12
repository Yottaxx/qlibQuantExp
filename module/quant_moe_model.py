import math

# module/quant_moe_model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from module.architecture.feature_selector import DifferentiableFeatureSelector
from module.architecture.feature_tokenizer import FeatureTokenizer
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
                tau_mlp_out_scale=float(getattr(config, "time_tau_mlp_out_scale", 0.01)),
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
            dropout=float(getattr(config, "regime_macro_dropout", 0.0) or 0.0),
        )

        # 4) MoE 主干
        self.layers = nn.ModuleList([RegimeAdaptiveMoEBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # 5) Attention-based pooling for factor aggregation
        # 使用自适应 attention pooling 替代简单的 softmax pooling
        # 结合 attention 和 mean pooling，提供更稳健的因子聚合
        self.factor_pooling = AdaptivePooling(
            d_model=d_model,
            n_heads=int(getattr(config, "pool_n_heads", 1)),
            dropout=config.dropout,
            alpha=config.pooling_alpha,
        )
        
        # 6) Stock score head
        self.head = nn.Linear(d_model, 1)

        # 5.5) Optional temporal readout (time-readout-bonus-20260607): a learned temporal aggregation
        # over T applied BEFORE the (unchanged) factor pool. All designs identity-start at z==h[:,-1]
        # (one-hot-last for the linears, gated-residual for the attentions) so a from-scratch retrain
        # begins exactly at the last-step control and is free to move off it. See forward() for the
        # tensor flow; identity-start overrides for nn.Linear/LayerNorm are re-applied AFTER post_init.
        self.temporal_readout = str(getattr(config, "temporal_readout", "") or "").strip().lower()
        _tr = self.temporal_readout
        _T = int(config.context_len)
        _N = int(config.num_alphas)
        # init-ablation (task #14). _ti: aggregation init; _gi: attention gate init.
        self.temporal_readout_init = str(getattr(config, "temporal_readout_init", "onehot_last") or "onehot_last").strip().lower()
        _ti = self.temporal_readout_init
        _gi = float(getattr(config, "temporal_readout_gate_init", 0.0) or 0.0)
        _uniform = (_ti == "uniform_mean")
        # Dedicated generator for the uniform-init noise so it does NOT consume the global RNG stream —
        # keeps the backbone (post_init draws) IDENTICAL between onehot and uniform runs => clean A/B.
        _tg = torch.Generator().manual_seed(20260607)

        def _agg_uniform(shape):
            # sum-preserving diverse init: 1/T + small noise (Σ≈1 ⇒ z-scale = h[:,-1]-scale). NOT Kaiming.
            return torch.full(shape, 1.0 / _T) + 0.02 * torch.randn(shape, generator=_tg)

        if _tr == "d3cid":
            if _uniform:
                _A = _agg_uniform((_T, d_model))
            else:
                _A = torch.zeros(_T, d_model); _A[-1, :] = 1.0
            self.tr_A = nn.Parameter(_A)                          # [T,D] per-channel linear over T
        elif _tr == "d3cin":
            if _uniform:
                _AN = _agg_uniform((_N, _T))
            else:
                _AN = torch.zeros(_N, _T); _AN[:, -1] = 1.0
            self.tr_A_N = nn.Parameter(_AN)                       # [N,T] per-factor linear over T
        elif _tr == "d3mix":
            # TSMixer: time-mix (Linear T->T) + channel-mix (Linear D->2D->D), each residual + pre-LN,
            # then a learned linear temporal collapse (T->1). Identity-start (zero tr_tw/tr_cw2 +
            # one-hot tr_collapse) is applied AFTER post_init since it re-inits nn.Linear/LayerNorm.
            self.tr_ln_t = nn.LayerNorm(d_model)
            self.tr_tw = nn.Linear(_T, _T)
            self.tr_ln_c = nn.LayerNorm(d_model)
            self.tr_cw1 = nn.Linear(d_model, 2 * d_model)
            self.tr_cw2 = nn.Linear(2 * d_model, d_model)
            if _uniform:
                _c = _agg_uniform((_T,))
            else:
                _c = torch.zeros(_T); _c[-1] = 1.0
            self.tr_collapse = nn.Parameter(_c)                  # [T] linear temporal collapse
        elif _tr in ("d1pma", "duala", "dualb"):
            # gated attention over T: single learnable query, uniform bias (=0), gate (gate_init -> g=sigmoid).
            self.tr_q = nn.Parameter(torch.randn(d_model) * 0.02)
            self.tr_b_t = nn.Parameter(torch.zeros(_T))
            self.tr_gate = nn.Parameter(torch.full((1,), _gi))   # 0->g=0.5; -2->g~0.12; +2->g~0.88
            if _tr in ("duala", "dualb"):
                self.tr_head2 = nn.Linear(2 * d_model, 1)        # [2D->1]; zF-half zeroed after post_init
            if _tr == "dualb":
                self.tr_qN = nn.Parameter(torch.randn(d_model) * 0.02)
                self.tr_bN = nn.Parameter(torch.zeros(_N))

        # HF 标准初始化
        self.post_init()

        # [Optimized Init] Match head output scale to label variance (~1.0)
        # Prevents "scale seeking" in early epochs.
        with torch.no_grad():
            nn.init.kaiming_normal_(self.head.weight, mode="fan_in", nonlinearity="linear")
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

            # Temporal-readout identity-start overrides. MUST come after post_init() because it
            # re-inits every nn.Linear/LayerNorm; bare Parameters (tr_A/tr_A_N/tr_q/...) survive it.
            _tr = getattr(self, "temporal_readout", "")
            if _tr == "d3mix":
                # Zero the residual-branch OUTPUT linears so the mixing block is an exact no-op at init
                # (Xt=X, Xc=X) — init-INDEPENDENT. The collapse init (onehot_last vs uniform_mean) is set
                # in __init__ and survives post_init (bare Parameter); only re-state the one-hot-last form
                # (uniform_mean must be left intact, else it gets clobbered back to control).
                nn.init.zeros_(self.tr_tw.weight); nn.init.zeros_(self.tr_tw.bias)
                nn.init.zeros_(self.tr_cw2.weight); nn.init.zeros_(self.tr_cw2.bias)
                if getattr(self, "temporal_readout_init", "onehot_last") != "uniform_mean":
                    self.tr_collapse.zero_(); self.tr_collapse[-1] = 1.0
            elif _tr in ("duala", "dualb"):
                # Zero the zF half of the dual head so the score starts from the time branch only.
                self.tr_head2.weight[:, d_model:] = 0.0
                if self.tr_head2.bias is not None:
                    nn.init.zeros_(self.tr_head2.bias)

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

        if isinstance(module, FeatureTokenizer):
            module.reset_parameters()
            return

        if isinstance(module, nn.Linear):
            # [Optimized Init] val_proj (1->D): Boost std to 0.1 to preserve signal
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
        router_override: str | None = None,
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

        num_alphas = int(getattr(self.factor_id_emb, "num_embeddings", 0) or 0)
        if num_alphas > 0 and int(N) > num_alphas:
            raise RuntimeError(
                f"Input x has N={int(N)} factor channels, but the model was initialized with num_alphas={num_alphas}. "
                "This usually means extra channels (e.g., packed label) leaked into x, or num_alphas mismatches the data."
            )

        if factor_ids.numel() != int(N):
            raise RuntimeError(
                "factor_ids must have the same number of elements as x.shape[2] (N). "
                f"Got x.shape={tuple(x.shape)}, factor_ids.shape={tuple(factor_ids.shape)}, "
                f"factor_ids.numel()={int(factor_ids.numel())}."
            )
        factor_ids = factor_ids.reshape(int(N))

        # 0) Regime embedding（只看 x/macro，不看 label；也不被 time embedding 污染）
        regime = self.regime_encoder(x, macro_features)  # [B, D]

        # 1) value + factor embedding
        factor_table = self.factor_id_emb(factor_ids.long())  # [N,D]
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

        diag_metrics: dict[str, float] = {}
        factor_film = None
        time_tau_values = None
        factor_gate_importance = None
        film_gamma_strength_by_factor = None
        film_beta_strength_by_factor = None

        # 1.5) Regime-adaptive time embedding (adds a clean time signal for short windows)
        if self.time_embedding is not None:
            time_emb, tau = self.time_embedding(regime, T)  # [B,T,D], [B,1]
            emb_norm = float(time_emb.detach().float().norm(dim=-1).mean().item())
            base_norm = float(h.detach().float().norm(dim=-1).mean().item())
            diag_metrics["time_embedding_norm"] = emb_norm
            diag_metrics["time_embedding_to_value_norm_ratio"] = emb_norm / max(base_norm, 1e-12)
            h = h + time_emb.unsqueeze(2)  # [B,T,1,D] -> broadcast over N
            tau_det = tau.detach().float().view(-1)
            time_tau_values = tau_det
            tau_mean = float(tau_det.mean().item())
            diag_metrics["time_tau"] = tau_mean
            diag_metrics["time_half_life"] = tau_mean * math.log(2.0)
            diag_metrics["time_tau_std"] = float(tau_det.std(unbiased=False).item())
            diag_metrics["time_tau_p10"] = float(torch.quantile(tau_det, 0.10).item())
            diag_metrics["time_tau_p90"] = float(torch.quantile(tau_det, 0.90).item())
            tau_span = float(getattr(self.config, "time_tau_max", 0.0) or 0.0) - float(
                getattr(self.config, "time_tau_min", 0.0) or 0.0
            )
            if tau_span > 0:
                diag_metrics["time_tau_range_util"] = (
                    diag_metrics["time_tau_p90"] - diag_metrics["time_tau_p10"]
                ) / tau_span

        # 1.6) Regime-adaptive factor reweighting (per-sample, per-factor)
        if self.factor_gate is not None:
            gamma, beta = self.factor_gate(regime, factor_table)  # [B,N,D], [B,N,D]
            factor_film = (gamma, beta)

            # Diagnostics: summarize FiLM strength on a scalar per factor.
            gate_det = gamma.detach().mean(dim=-1)  # [B,N], in ~[1-scale, 1+scale]
            gamma_strength = (gamma.detach() - 1.0).abs()
            beta_strength = beta.detach().abs()
            film_gamma_strength_by_factor = gamma_strength.mean(dim=-1)  # [B,N]
            film_beta_strength_by_factor = beta_strength.mean(dim=-1)  # [B,N]
            diag_metrics["film_gamma_strength"] = float(gamma_strength.mean().item())
            diag_metrics["film_beta_strength"] = float(beta_strength.mean().item())
            diag_metrics["factor_gate_mean"] = float(gate_det.mean().item())
            diag_metrics["factor_gate_std"] = float(gate_det.std(unbiased=False).item())

            # Concentration: use |gamma-1| as "importance" to see whether the model is actively modulating a few factors.
            eps = 1e-12
            imp = (gate_det - 1.0).abs().clamp_min(eps)
            denom = imp.sum(dim=-1, keepdim=True).clamp_min(eps)
            p = imp / denom  # [B,N]
            factor_gate_importance = p.detach()

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
        router_logit_margins = []
        router_entropy_values = []
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
                router_override=router_override,
            )
            z_losses.append(diag["z_loss"])
            entropies.append(diag["entropy"])
            time_ratios.append(diag["time_ratio"])
            gates_list.append(diag["weights"])
            if "logit_margin_per_sample" in diag:
                router_logit_margins.append(diag["logit_margin_per_sample"].detach())
            if "entropy_per_sample" in diag:
                router_entropy_values.append(diag["entropy_per_sample"].detach())

            gw = diag["weights"].detach()
            tr_layer = gw[:, 0]
            ent_layer = diag.get("entropy_per_sample", None)
            margin_layer = diag.get("logit_margin_per_sample", None)
            diag_metrics[f"router_layer_{idx}_time_ratio"] = float(tr_layer.mean().item())
            diag_metrics[f"router_layer_{idx}_factor_ratio"] = float((1.0 - tr_layer).mean().item())
            diag_metrics[f"router_layer_{idx}_collapse_ratio"] = float(
                ((tr_layer < 0.1) | (tr_layer > 0.9)).float().mean().item()
            )
            if ent_layer is not None:
                diag_metrics[f"router_layer_{idx}_entropy_norm"] = float(
                    (ent_layer.detach().float().mean() / math.log(2.0)).item()
                )
            if margin_layer is not None:
                diag_metrics[f"router_layer_{idx}_logit_margin"] = float(
                    margin_layer.detach().float().mean().item()
                )

            for metric_name in (
                "time_expert_norm",
                "factor_expert_norm",
                "time_contrib_norm",
                "factor_contrib_norm",
                "contrib_norm_ratio",
                "expert_cosine",
                "time_winner_ratio",
            ):
                if metric_name in diag:
                    diag_metrics[f"expert_layer_{idx}_{metric_name}"] = float(
                        diag[metric_name].detach().float().item()
                        if isinstance(diag[metric_name], torch.Tensor)
                        else diag[metric_name]
                    )

            if need_attn and layer_attn is not None:
                attn_maps[f"layer_{idx}"] = layer_attn

        h = self.final_norm(h)

        # 6) Factor aggregation & stock scoring
        # Optional temporal-readout override (time-readout-bonus-20260607). Each design produces a
        # temporal aggregation z[B,N,D] -> factor_pooling(z) -> head. Default (attr unset) preserves
        # the exact current last-step + factor-pool behavior. Identity-start: every design begins at
        # z==h[:,-1] (linears) or a gated blend with last-step (attentions). fp32-cast logits before
        # softmax/sigmoid (AMP); the linear designs need no cast. diag_metrics monitor "did it move?".
        _tr = getattr(self, "temporal_readout", "") or ""
        _D = h.shape[-1]
        stock_score_override = None
        if _tr == "d3cid":
            # per-channel(D) linear over T. tr_A one-hot-last => z == h[:,-1] (exact identity-start).
            z = torch.einsum("btnd,td->bnd", h, self.tr_A)             # [B,N,D]
            h_pooled, factor_attention_weights = self.factor_pooling(z)
            with torch.no_grad():
                _wabs = self.tr_A.detach().float().abs().mean(dim=1)   # [T]
                _wsum = _wabs.sum().clamp_min(1e-9)
                diag_metrics["tr_W_last_frac"] = float((_wabs[-1] / _wsum).item())
                diag_metrics["tr_W_nonlast_mass"] = float((1.0 - _wabs[-1] / _wsum).item())
        elif _tr == "d3cin":
            # per-factor(N) linear over T (shared across D). tr_A_N one-hot-last => z == h[:,-1].
            _Nf = h.shape[2]
            z = torch.einsum("btnd,nt->bnd", h, self.tr_A_N[:_Nf, :])  # [B,N,D]
            h_pooled, factor_attention_weights = self.factor_pooling(z)
            with torch.no_grad():
                _wabs = self.tr_A_N[:_Nf, :].detach().float().abs().mean(dim=0)  # [T]
                _wsum = _wabs.sum().clamp_min(1e-9)
                diag_metrics["tr_W_last_frac"] = float((_wabs[-1] / _wsum).item())
                diag_metrics["tr_W_nonlast_mass"] = float((1.0 - _wabs[-1] / _wsum).item())
        elif _tr == "d3mix":
            # TSMixer: time-mix (residual) + channel-mix (residual) -> learned linear collapse.
            X = h                                                      # [B,T,N,D]
            Xt = X + self.tr_tw(self.tr_ln_t(X).transpose(1, 3)).transpose(1, 3)   # time-mix over T
            Xc = Xt + self.tr_cw2(torch.relu(self.tr_cw1(self.tr_ln_c(Xt))))       # channel-mix over D
            z = torch.einsum("btnd,t->bnd", Xc, self.tr_collapse)      # [B,N,D] (linear collapse T->1)
            h_pooled, factor_attention_weights = self.factor_pooling(z)
            with torch.no_grad():
                _c = self.tr_collapse.detach().float().abs()
                diag_metrics["tr_collapse_last_frac"] = float((_c[-1] / _c.sum().clamp_min(1e-9)).item())
                _xn = X.detach().float().norm().clamp_min(1e-9)
                diag_metrics["tr_timemix_gain"] = float(((Xt - X).detach().float().norm() / _xn).item())
                _xtn = Xt.detach().float().norm().clamp_min(1e-9)
                diag_metrics["tr_chanmix_gain"] = float(((Xc - Xt).detach().float().norm() / _xtn).item())
        elif _tr in ("d1pma", "duala", "dualb"):
            # gated attention over T (shared temporal weights per factor), gated-residual w/ last-step.
            h_time = h.mean(dim=2)                                     # [B,T,D] (mean over N)
            logits = (h_time.float() @ self.tr_q.float()) / (_D ** 0.5) + self.tr_b_t.float()  # [B,T] fp32
            a = torch.softmax(logits, dim=1).to(h.dtype)              # [B,T]
            attn = torch.einsum("bt,btnd->bnd", a, h)                  # [B,N,D]
            g = torch.sigmoid(self.tr_gate.float()).to(h.dtype)       # scalar (init g=0.5)
            z_T = (1.0 - g) * h[:, -1, :, :] + g * attn                # [B,N,D] gated-residual
            zTp, factor_attention_weights = self.factor_pooling(z_T)   # [B,D]
            with torch.no_grad():
                diag_metrics["tr_gate_g"] = float(g.float().item())
                diag_metrics["tr_attn_last_frac"] = float(a[:, -1].float().mean().item())
                _p = a.float().clamp_min(1e-9)
                diag_metrics["tr_attn_entropy"] = float((-(_p * _p.log()).sum(dim=1)).mean().item())
            if _tr == "d1pma":
                h_pooled = zTp
            else:
                # dual branches: factor branch zF + concat -> tr_head2 (score_override).
                if _tr == "duala":
                    zF = h.mean(dim=(1, 2))                            # [B,D] global mean over (T,N)
                else:  # dualb: attention over N on the time-mean reps
                    _Nf = h.shape[2]
                    h_fac = h.mean(dim=1)                              # [B,N,D] (mean over T)
                    lN = (h_fac.float() @ self.tr_qN.float()) / (_D ** 0.5) + self.tr_bN[:_Nf].float()  # [B,N] fp32
                    aN = torch.softmax(lN, dim=1).to(h.dtype)         # [B,N]
                    zF = torch.einsum("bn,bnd->bd", aN, h_fac)        # [B,D]
                stock_score_override = self.tr_head2(torch.cat([zTp, zF], dim=-1)).squeeze(-1)  # [B]
                h_pooled = zTp
                with torch.no_grad():
                    diag_metrics["tr_head2_zT_norm"] = float(zTp.detach().float().norm(dim=-1).mean().item())
                    diag_metrics["tr_head2_zF_norm"] = float(zF.detach().float().norm(dim=-1).mean().item())
        else:
            # 使用最后一时间步的因子表示 [B, N, D]
            h_last = h[:, -1, :, :]  # [B, N, D]
            h_pooled, factor_attention_weights = self.factor_pooling(h_last)  # [B, D], [B, N]

        # Stock score prediction
        if stock_score_override is not None:
            stock_score = stock_score_override                        # [B] (duals: tr_head2 over [zTp,zF])
        else:
            stock_score = self.head(h_pooled).squeeze(-1)  # [B]

        # factor_attention_weights 用于返回（可用于可解释性分析）
        factor_pool_weights = factor_attention_weights  # [B, N] or None

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

                # 基础 loss 组件
                l_listmle = QuantLossFunctions.listmle_loss(
                    p,
                    y,
                    tau=getattr(self.config, "listmle_tau", 1.0),
                )

                # IC / MSE（可作为主 loss 或监控）
                l_ic = QuantLossFunctions.cs_ic_loss(p, y)
                l_mse = QuantLossFunctions.cs_mse_loss(
                    p,
                    y,
                    normalize=bool(getattr(self.config, "mse_normalize", False)),
                )

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

                # ★ total_loss：主 loss + 辅助 loss
                total_loss = (
                    w.get(main_loss, 1.0) * l_main
                    + l_aux  # 直接加，不再乘 w["aux"]
                    + l_reg  # 直接加，不再乘 w["reg"]
                )
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
            logits=stock_score,
            gate_weights=gates_list,
            metrics=metrics,
            avg_gate_entropy=avg_entropy,
            avg_time_ratio=avg_time_ratio,
            selected_mask=mask,
            attn_maps=attn_maps if return_attn else None,
            factor_pool_weights=factor_pool_weights,
            router_logit_margins=router_logit_margins if router_logit_margins else None,
            router_entropy_values=router_entropy_values if router_entropy_values else None,
            time_tau_values=time_tau_values,
            factor_gate_importance=factor_gate_importance,
            film_gamma_strength_by_factor=film_gamma_strength_by_factor,
            film_beta_strength_by_factor=film_beta_strength_by_factor,
            scores=stock_score,
        )
