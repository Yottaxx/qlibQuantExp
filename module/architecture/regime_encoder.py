import torch
import torch.nn as nn
from module.architecture import RMSNorm
from contextlib import nullcontext


def _corr_crowding_stats_from_matrix(
    X: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable legacy crowding stats for a single daily cross-section."""
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape [B, N], got {tuple(X.shape)}")

    B, N = X.shape
    device = X.device
    if B < 3 or N < 2:
        crowding = torch.zeros((), device=device)
        pc1_ratio = torch.ones((), device=device) * (1.0 if N == 1 else 0.0)
        return crowding, pc1_ratio

    X = torch.where(torch.isfinite(X), X, torch.zeros_like(X))
    mu = X.mean(dim=0, keepdim=True)
    X0 = X - mu
    var = (X0 * X0).mean(dim=0, keepdim=True)
    std = var.sqrt().clamp_min(float(eps))
    Xn = X0 / std

    denom = float(max(B - 1, 1))
    corr = (Xn.transpose(0, 1) @ Xn) / denom
    corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
    corr = corr.clamp(-1.0, 1.0)

    abs_corr = corr.abs()
    off_diag_sum = abs_corr.sum() - abs_corr.diagonal().sum()
    crowding = off_diag_sum / float(N * (N - 1))

    try:
        evals = torch.linalg.eigvalsh(corr.detach())
        evals = torch.where(torch.isfinite(evals), evals, torch.zeros_like(evals))
        evals = evals.clamp_min(0.0)
        raw_pc1_ratio = evals[-1] / evals.sum().clamp_min(float(eps))
        pc1_floor = 1.0 / float(max(N, 1))
        pc1_ratio = (raw_pc1_ratio - pc1_floor) / (1.0 - pc1_floor + float(eps))
        pc1_ratio = pc1_ratio.clamp(0.0, 1.0)
    except Exception:
        pc1_ratio = torch.tensor(0.5, device=device)

    return crowding, pc1_ratio


def compute_legacy_internal_stats(
    x: torch.Tensor,
    *,
    internal_mode: str = "short",
    internal_lag: int = 1,
    internal_use_batch_stats: bool = True,
    internal_tail_threshold: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Stateless version of the legacy internal regime statistics."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [B, T, N], got {tuple(x.shape)}")

    try:
        autocast_off = torch.autocast(device_type=x.device.type, enabled=False)
    except Exception:
        autocast_off = torch.cuda.amp.autocast(enabled=False) if x.device.type == "cuda" else nullcontext()

    with autocast_off:
        x = x.float()

        if not bool(internal_use_batch_stats):
            cs_vol = x.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True).clamp(0.0, 5.0)
            trend_raw = x[:, -1, :] - x[:, 0, :]
            trend_strength = trend_raw.abs().mean(dim=-1, keepdim=True).clamp(0.0, 5.0)
            if x.shape[1] > 1:
                diff = x.diff(dim=1)
                temp_vol = diff.abs().mean(dim=(1, 2)).unsqueeze(-1)
            else:
                temp_vol = cs_vol
            temp_vol = temp_vol.clamp(0.0, 5.0)
            extreme_ratio = (x.abs() > float(internal_tail_threshold)).float().mean(dim=(1, 2)).unsqueeze(-1)
            extreme_val = extreme_ratio.clamp(0.0, 1.0)
            stats = torch.cat([cs_vol, trend_strength, temp_vol, extreme_val], dim=-1)
            return torch.where(torch.isfinite(stats), stats, torch.zeros_like(stats))

        B, T, _ = x.shape
        if T < 1:
            raise ValueError("T must be >= 1")

        lag = 1 if str(internal_mode).strip().lower() == "short" else max(int(internal_lag), 1)
        lag = min(lag, max(T - 1, 1))
        t0 = -1 - lag

        X_end = x[:, -1, :]
        X_start = x[:, t0, :] if T > 1 else X_end

        crowd_end, pc1_end = _corr_crowding_stats_from_matrix(X_end, eps=float(eps))
        crowd_start, _ = _corr_crowding_stats_from_matrix(X_start, eps=float(eps))
        tail = (X_end.abs() > float(internal_tail_threshold)).float().mean()

        stats = torch.stack(
            [
                crowd_end.clamp(0.0, 1.0),
                pc1_end.clamp(0.0, 1.0),
                (crowd_end - crowd_start).abs().clamp(0.0, 1.0),
                tail.clamp(0.0, 1.0),
            ],
            dim=-1,
        ).unsqueeze(0)
        stats = stats.expand(B, -1)
        return torch.where(torch.isfinite(stats), stats, torch.zeros_like(stats))

# ==========================================
# 新增模块: 市场状态编码器 (Regime Encoder)
# ==========================================
class RegimeContextEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_external_macro: bool = False,
        d_macro: int = 0,
        *,
        internal_mode: str = "short",
        internal_lag: int = 1,
        internal_use_batch_stats: bool = True,
        internal_tail_threshold: float = 2.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.use_external = use_external_macro
        self.internal_mode = internal_mode
        self.internal_lag = int(internal_lag)
        self.internal_use_batch_stats = bool(internal_use_batch_stats)
        self.internal_tail_threshold = float(internal_tail_threshold)
        self.eps = float(eps)

        # 输入维度取决于是否使用外部宏观数据
        # 如果用 Internal，我们提取 4 个基础统计特征：
        # 1. Crowding level (因子相关结构强度)
        # 2. Market-mode strength (PC1 explained ratio)
        # 3. Regime drift over lag (跨周期变化强度)
        # 4. Tail / shock intensity (极端值比例)
        self.d_input = d_macro if use_external_macro else 4

        # 将低维统计特征映射到 d_model
        drop_p = float(dropout) if dropout is not None else 0.0
        self.encoder = nn.Sequential(
            nn.Linear(self.d_input, d_model * 2),
            nn.Tanh(),  # Tanh 适合处理统计值的归一化s
            nn.Dropout(drop_p),
            nn.Linear(d_model * 2, d_model),
            RMSNorm(d_model)  # 必须 Norm，防止统计值波动过大
        )

    def _corr_crowding_stats(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: [B, N] factor matrix for a single date (cross-section of stocks).
        Returns:
            crowding: scalar in [0, 1] (mean abs off-diagonal correlation)
            pc1_ratio: scalar in (0, 1] (top eigenvalue / trace of correlation)
        """
        return _corr_crowding_stats_from_matrix(X, eps=self.eps)

    def _compute_internal_stats_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback: compute per-sample stats from x [B, T, N].
        This is less aligned with "market regime" but works when batches are not daily cross-sections.
        """
        cs_vol = x.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True)  # [B, 1]
        cs_vol = torch.clamp(cs_vol, 0.0, 5.0)

        trend_raw = (x[:, -1, :] - x[:, 0, :])  # [B, N]
        trend_strength = trend_raw.abs().mean(dim=-1, keepdim=True)  # [B, 1]
        trend_strength = torch.clamp(trend_strength, 0.0, 5.0)

        if x.shape[1] > 1:
            diff = x.diff(dim=1)  # [B, T-1, N]
            temp_vol = diff.abs().mean(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
        else:
            temp_vol = cs_vol
        temp_vol = torch.clamp(temp_vol, 0.0, 5.0)

        threshold = self.internal_tail_threshold
        extreme_mask = x.abs() > threshold  # [B, T, N]
        extreme_ratio = extreme_mask.float().mean(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
        extreme_val = torch.clamp(extreme_ratio, 0.0, 1.0)

        stats = torch.cat([cs_vol, trend_strength, temp_vol, extreme_val], dim=-1)
        stats = torch.where(torch.isfinite(stats), stats, torch.zeros_like(stats))
        return stats

    def compute_internal_stats(self, x: torch.Tensor) -> torch.Tensor:
        """
        从输入 x [B, T, N] 中实时提取市场状态。
        
        - 对于本项目的“日度截面 batch”（B=stocks），推荐 internal_use_batch_stats=True，
          以 batch 内截面估计“市场层面”状态，并广播到每个样本。
        - 对于非日度截面 batch，可切换到 per-sample fallback。
        
        Returns:
            stats: [B, 4] 市场状态统计特征
        """
        return compute_legacy_internal_stats(
            x,
            internal_mode=self.internal_mode,
            internal_lag=self.internal_lag,
            internal_use_batch_stats=self.internal_use_batch_stats,
            internal_tail_threshold=self.internal_tail_threshold,
            eps=self.eps,
        )

    def forward(self, x, macro_features=None):
        if self.use_external:
            assert macro_features is not None, "Configured for external macro but None provided"
            feats = macro_features
        else:
            feats = self.compute_internal_stats(x)

        # [B, d_input] -> [B, d_model]
        context_emb = self.encoder(feats)
        return context_emb
