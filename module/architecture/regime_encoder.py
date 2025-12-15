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
        从输入 x [Batch, T, N] 中实时提取市场状态。
        
        注意：x 已经经过 RobustZScoreNorm 归一化（z-score），
        因此统计特征的计算需要考虑归一化后的分布特性。
        
        Returns:
            stats: [B, 4] 市场状态统计特征
        """
        # x 已经经过 RobustZScoreNorm 归一化，均值≈0，标准差≈1
        
        # 1. 截面波动率 (反映市场分歧程度)
        # 对于 z-score 归一化后的数据，std 反映因子间的分歧程度
        # 使用 robust std (median absolute deviation) 提高稳定性
        cs_vol = x.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True)  # [B, 1]
        # 归一化到合理范围（z-score 后 std 通常在 [0, 3] 之间）
        cs_vol = torch.clamp(cs_vol, 0.0, 5.0)
        
        # 2. 时序动量强度 (反映趋势性)
        # 对于归一化数据，使用相对变化而非绝对值
        trend_raw = (x[:, -1, :] - x[:, 0, :])  # [B, N]
        trend_strength = trend_raw.abs().mean(dim=-1, keepdim=True)  # [B, 1]
        # 归一化：z-score 后，趋势强度通常在 [0, 2] 之间
        trend_strength = torch.clamp(trend_strength, 0.0, 5.0)
        
        # 3. 瞬时波动 (时序上的变动幅度)
        # 计算时序差分，对归一化数据更稳定
        if x.shape[1] > 1:
            diff = x.diff(dim=1)  # [B, T-1, N]
            temp_vol = diff.abs().mean(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
        else:
            # 单时间步，使用截面波动作为替代
            temp_vol = cs_vol
        # 归一化：差分后的波动通常在 [0, 2] 之间
        temp_vol = torch.clamp(temp_vol, 0.0, 5.0)
        
        # 4. 极端值程度 (反映肥尾/黑天鹅)
        # 对于 z-score 归一化数据，极端值通常 > 2 或 < -2
        # 使用 robust 方法：计算超过阈值的比例
        threshold = 2.0  # z-score 的 2-sigma 阈值
        extreme_mask = x.abs() > threshold  # [B, T, N]
        extreme_ratio = extreme_mask.float().mean(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
        # 归一化到 [0, 1] 范围
        extreme_val = torch.clamp(extreme_ratio, 0.0, 1.0)
        
        # 拼接统计特征: [B, 4]
        # 所有特征都已归一化到合理范围，提高数值稳定性
        stats = torch.cat([cs_vol, trend_strength, temp_vol, extreme_val], dim=-1)
        
        # 最终安全检查：确保没有 NaN 或 Inf
        stats = torch.where(torch.isfinite(stats), stats, torch.zeros_like(stats))
        
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

