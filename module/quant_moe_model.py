import torch
import torch.nn as nn
from transformers import PreTrainedModel

from module.architecture.feature_selector import DifferentiableFeatureSelector
from module.utils.losses import QuantLossFunctions
from module.utils.model_configuration import QuantMoEConfig, QuantModelOutput
from module.architecture.moe_block import RegimeAdaptiveMoEBlock
from module.architecture.regime_encoder import RegimeContextEncoder


class QuantMoEModel(PreTrainedModel):
    config_class = QuantMoEConfig

    def __init__(self, config: QuantMoEConfig):
        super().__init__(config)
        self.config = config

        # Embeddings
        self.val_proj = nn.Linear(1, config.d_model)
        self.factor_id_emb = nn.Embedding(config.num_alphas, config.d_model)

        self.emb_dropout = nn.Dropout(config.dropout)

        # Optional: feature selection (STG-like gate)
        self.feature_selector = (
            DifferentiableFeatureSelector(config.num_alphas, sigma=config.selection_noise_std)
            if config.use_feature_selection
            else None
        )

        # Regime encoder (market context)
        self.regime_encoder = RegimeContextEncoder(config.d_model)

        # Backbone
        self.layers = nn.ModuleList([RegimeAdaptiveMoEBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

        # Head
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x, factor_ids, date_ids=None, labels=None, macro_features=None):
        """Forward.

        Args:
            x: [B, T, N] (T window, N factors/features)
            factor_ids: [N] long
            labels: [B] (one label per stock/day cross-section sample)
        """
        B, T, N = x.shape
        device = x.device

        # 1) Embed values + factor id embedding
        h = self.val_proj(x.unsqueeze(-1)) + self.factor_id_emb(factor_ids.long()).view(1, 1, N, -1)

        # 2) Feature selection (optional)
        reg_loss = torch.tensor(0.0, device=device)
        mask = None
        if self.feature_selector is not None:
            h, reg_loss, z = self.feature_selector(h, self.config.selection_temperature, self.training)
            mask = z.detach()  # [N]

        h = self.emb_dropout(h)

        # 3) Regime embedding from raw x (avoid look-ahead)
        regime = self.regime_encoder(x, macro_features)  # [B, D]

        # 4) ALiBi: build per-axis biases to avoid shape mismatch (T vs N)
        attn_bias = None
        if self.config.use_alibi:
            from module.utils.utils import build_bidirectional_alibi_bias
            bias_time = build_bidirectional_alibi_bias(B, T, self.config.n_heads, device)   # [1,H,T,T]
            bias_factor = build_bidirectional_alibi_bias(B, N, self.config.n_heads, device) # [1,H,N,N]
            attn_bias = (bias_time, bias_factor)

        # 5) MoE blocks
        z_losses = []
        entropies = []
        time_ratios = []
        gates_list = []

        for blk in self.layers:
            h, diag = blk(h, regime, attn_bias)
            z_losses.append(diag["z_loss"])
            entropies.append(diag["entropy"])
            time_ratios.append(diag["time_ratio"])
            gates_list.append(diag["weights"])

        h = self.final_norm(h)

        # Predict per-factor at the last timestep
        h_last = h[:, -1, :, :]  # [B, N, D]
        factor_logits = self.head(h_last).squeeze(-1)  # [B, N]
        stock_score = factor_logits.mean(dim=1)  # [B]

        # 6) Loss
        total_loss = None
        metrics: dict = {}

        if labels is not None:
            labels = labels.squeeze()
            # explicit NaN/Inf guard (adapter may also mask, but model is the source of truth)
            valid = torch.isfinite(labels)
            valid_ratio = valid.float().mean().item()

            if valid.sum().item() >= 2:
                y = labels[valid]
                p = stock_score[valid]

                w = self.config.loss_weights
                l_ic = QuantLossFunctions.cs_ic_loss(p, y)
                l_rank = QuantLossFunctions.ranknet_topbottom_loss(p, y, self.config.rank_topk)
                l_huber = QuantLossFunctions.cs_huber_loss(p, y, self.config.huber_delta)

                l_aux = torch.stack(z_losses).mean() * self.config.router_z_loss_coef if z_losses else torch.tensor(0.0, device=device)
                l_reg = reg_loss * self.config.selection_reg_lambda

                total_loss = (
                    w.get("ic", 1.0) * l_ic +
                    w.get("rank", 1.0) * l_rank +
                    w.get("huber", 0.0) * l_huber +
                    l_aux +
                    l_reg
                )

                metrics = {
                    "loss_total": float(total_loss.detach().item()),
                    "loss_ic": float(l_ic.detach().item()),
                    "loss_rank": float(l_rank.detach().item()),
                    "loss_huber": float(l_huber.detach().item()),
                    "loss_aux": float(l_aux.detach().item()),
                    "loss_sparsity": float(l_reg.detach().item()),
                    "valid_ratio": float(valid_ratio),
                }
            else:
                # no valid labels => skip optimization
                total_loss = None
                metrics = {"valid_ratio": float(valid_ratio)}

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
        )
