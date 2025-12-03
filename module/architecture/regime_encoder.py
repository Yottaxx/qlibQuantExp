import torch.nn as nn
import torch
# ==========================================
# 新增模块: 市场状态编码器 (Regime Encoder)
# ==========================================
class RegimeContextEncoder(nn.Module):
    def __init__(self, d_model, use_external_macro=False, d_macro=0):
        super().__init__()
        self.use_external = use_external_macro

        # 输入维度取决于是否使用外部宏观数据
        # 如果用 Internal，我们提取 4 个基础统计特征：
        # 1. Cross-sectional Volatility (市场散度)
        # 2. Temporal Momentum Strength (趋势强度)
        # 3. Max Drawdown in window (回撤幅度)
        # 4. Instant Volatility (瞬时波动)
        self.d_input = d_macro if use_external_macro else 4

        # 将低维统计特征映射到 d_model
        self.encoder = nn.Sequential(
            nn.Linear(self.d_input, d_model // 2),
            nn.Tanh(),  # Tanh 适合处理统计值的归一化
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)  # 必须 Norm，防止统计值波动过大
        )

    def compute_internal_stats(self, x):
        """
        从输入 x [Batch, T, N] 中实时提取市场状态
        """
        # x 已经经过归一化，通常是 z-score

        # 1. 截面波动率 (反映市场分歧程度) [B, T] -> Mean -> [B, 1]
        # 如果所有因子同涨同跌，std 小；如果有分歧，std 大
        cs_vol = x.std(dim=-1).mean(dim=-1, keepdim=True)

        # 2. 时序动量强度 (反映趋势性)
        # abs(Last - First)
        trend_strength = (x[:, -1, :] - x[:, 0, :]).abs().mean(dim=-1, keepdim=True)

        # 3. 瞬时波动 (时序上的变动幅度)
        # mean(abs(diff))
        temp_vol = x.diff(dim=1).abs().mean(dim=(1, 2)).unsqueeze(-1)

        # 4. 极端值程度 (反映肥尾/黑天鹅)
        # max(abs(x))
        extreme_val = x.abs().max(dim=1)[0].max(dim=1)[0].unsqueeze(-1)

        # 拼接: [B, 4]
        # 为了数值稳定性，建议取 log 或者 tanh，这里简单处理
        stats = torch.cat([cs_vol, trend_strength, temp_vol, torch.log1p(extreme_val)], dim=-1)
        return stats

    def forward(self, x, macro_features=None):
        if self.use_external:
            assert macro_features is not None, "Configured for external macro but None provided"
            feats = macro_features
        else:
            feats = self.compute_internal_stats(x)

        # [B, d_input] -> [B, d_model]
        context_emb = self.encoder(feats)
        return context_emb

