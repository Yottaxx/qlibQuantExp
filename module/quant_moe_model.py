# module/quant_moe_model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from module.architecture.feature_selector import DifferentiableFeatureSelector
from module.utils.losses import QuantLossFunctions
from module.utils.model_configuration import QuantMoEConfig, QuantModelOutput
from module.architecture.moe_block import RegimeAdaptiveMoEBlock
from module.architecture.regime_encoder import RegimeContextEncoder
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
        )

        # 4) MoE 主干
        self.layers = nn.ModuleList([RegimeAdaptiveMoEBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # 5) 因子 head，最后对因子取均值得到股票打分
        self.head = nn.Linear(d_model, 1)
        self.pool_proj= nn.Linear(d_model, 1)
        # HF 标准初始化
        self.post_init()

    def forward(
        self,
        x: torch.Tensor,
        factor_ids: torch.Tensor,
        date_ids: torch.Tensor | None = None,
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
        """
        device = x.device
        B, T, N = x.shape

        if factor_ids is None:
            factor_ids = torch.arange(N, device=device)
        factor_ids = factor_ids.to(device)

        # 1) value + factor embedding
        h = self.val_proj(x.unsqueeze(-1)) + self.factor_id_emb(factor_ids.long()).view(1, 1, N, -1)

        # 2) 可微特征选择
        reg_loss = torch.tensor(0.0, device=device)
        mask = None
        if self.feature_selector is not None:
            h, reg_loss, z = self.feature_selector(
                h,
                temperature=self.config.selection_temperature,
                training=self.training,
            )
            mask = z.detach()

        h = self.emb_dropout(h)

        # 3) Regime embedding（只看 x，不看 label）
        regime = self.regime_encoder(x, macro_features)  # [B, D]

        # 4) ALiBi bias（时间 / 因子两个维度）
        attn_bias = None
        if self.config.use_alibi:
            bias_time = build_bidirectional_alibi_bias(B, T, self.config.n_heads, device)
            bias_factor = build_bidirectional_alibi_bias(B, N, self.config.n_heads, device)
            attn_bias = (bias_time, bias_factor)

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
                return_attn=need_attn
            )
            z_losses.append(diag["z_loss"])
            entropies.append(diag["entropy"])
            time_ratios.append(diag["time_ratio"])
            gates_list.append(diag["weights"])

            if need_attn and layer_attn is not None:
                attn_maps[f"layer_{idx}"] = layer_attn

        h = self.final_norm(h)

        # 6) 因子预测 & 股票打分
        h_last = h[:, -1, :, :]  # [B, N, D]
        # factor_logits = self.head(h_last).squeeze(-1)  # [B, N]
        # stock_score = factor_logits.mean(dim=1)        # [B]

        factor_logits = torch.softmax(self.pool_proj(h_last).squeeze(-1), dim=1)  # [B, N]
        h_pool = (h_last * factor_logits.unsqueeze(-1)).sum(dim=1)  # [B, D]
        stock_score = self.head(h_pool).squeeze(-1)  # [B]

        # 7) Loss & metrics
        total_loss: torch.Tensor | None = None
        metrics: dict[str, float] = {}
        valid_ratio = 0.0

        print("factor_logits std:", factor_logits.std().item(),
              "stock_score std:", stock_score.std().item())

        if labels is not None:
            labels = labels.squeeze()
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
                l_aux = (
                    torch.stack(z_losses).mean() * self.config.router_z_loss_coef
                    if z_losses
                    else torch.tensor(0.0, device=device)
                )
                l_reg = reg_loss * self.config.selection_reg_lambda

                # ★ total_loss：只用 ListMLE + 正则 / z-loss
                total_loss = (
                    w.get("listmle", 1.0) * l_listmle
                    + w.get("aux", 1.0) * l_aux
                    + w.get("reg", 1.0) * l_reg
                )

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
        )
