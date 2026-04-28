import torch
import torch.nn as nn
from module.architecture import RMSNorm


class XPanelStateEncoder(nn.Module):
    """
    Encode each stock's full time x factor panel into a state branch.

    This keeps the global state exposed to the full raw [T, N] stock panel
    without collapsing the daily cross-section before the conditioning MLP.
    """

    def __init__(
        self,
        *,
        d_x_input: int,
        d_state: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_x_input = int(d_x_input)
        self.d_state = int(d_state)
        if self.d_x_input <= 0:
            raise ValueError("XPanelStateEncoder requires d_x_input > 0.")
        if self.d_state <= 0:
            raise ValueError("XPanelStateEncoder requires d_state > 0.")

        self.stock_encoder = nn.Sequential(
            RMSNorm(self.d_x_input),
            nn.Linear(self.d_x_input, self.d_state),
            nn.GELU(),
            nn.Dropout(float(dropout or 0.0)),
            RMSNorm(self.d_state),
        )

    def forward(self, x_features: torch.Tensor) -> torch.Tensor:
        if x_features.ndim != 3:
            raise ValueError(
                "XPanelStateEncoder expects x_features [B,T,N], "
                f"got shape {tuple(x_features.shape)}."
            )
        x_clean = torch.where(torch.isfinite(x_features), x_features, torch.zeros_like(x_features))
        x_flat = x_clean.reshape(x_clean.shape[0], -1)
        if int(x_flat.shape[-1]) != self.d_x_input:
            raise ValueError(
                f"XPanelStateEncoder expects flattened dim {self.d_x_input}, "
                f"got {int(x_flat.shape[-1])} from shape {tuple(x_features.shape)}."
            )

        stock_state = self.stock_encoder(x_flat)  # [B, Dg]
        return stock_state


class GlobalStateEncoder(nn.Module):
    """
    Stable global state encoder.

    Daily inputs may be repeated across the same-date stock batch, while the
    x-panel branch can remain stock-specific. The returned `global_state` is
    [B, Dg]; the second return value keeps the same [B, Dg] tensor for
    diagnostics without collapsing the cross-section.
    """

    def __init__(
        self,
        *,
        d_macro_input: int,
        d_day_summary_input: int,
        d_internal_state_input: int = 0,
        d_x_input: int = 0,
        d_global_state: int,
        use_macro: bool = True,
        use_day_summary: bool = True,
        use_internal_state: bool = False,
        use_x: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_macro = bool(use_macro)
        self.use_day_summary = bool(use_day_summary)
        self.use_internal_state = bool(use_internal_state)
        self.use_x = bool(use_x)
        self.d_macro_input = int(d_macro_input or 0)
        self.d_day_summary_input = int(d_day_summary_input or 0)
        self.d_internal_state_input = int(d_internal_state_input or 0)
        self.d_x_input = int(d_x_input or 0)
        self.d_global_state = int(d_global_state)

        self.x_panel_encoder = None
        if self.use_x:
            self.x_panel_encoder = XPanelStateEncoder(
                d_x_input=self.d_x_input,
                d_state=self.d_global_state,
                dropout=dropout,
            )

        in_dim = 0
        if self.use_macro:
            in_dim += self.d_macro_input
        if self.use_day_summary:
            in_dim += self.d_day_summary_input
        if self.use_internal_state:
            in_dim += self.d_internal_state_input
        if self.use_x:
            in_dim += self.d_global_state
        if in_dim <= 0:
            raise ValueError("GlobalStateEncoder requires at least one enabled input branch.")

        hidden = max(self.d_global_state * 2, 64)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout or 0.0)),
            nn.Linear(hidden, self.d_global_state),
            RMSNorm(self.d_global_state),
        )

    def forward(
        self,
        *,
        batch_size: int,
        macro_features: torch.Tensor | None = None,
        day_summary_embedding: torch.Tensor | None = None,
        internal_state: torch.Tensor | None = None,
        x_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = []
        if self.use_macro:
            if macro_features is None:
                raise ValueError("GlobalStateEncoder expects macro_features but got None.")
            parts.append(macro_features)
        if self.use_day_summary:
            if day_summary_embedding is None:
                raise ValueError("GlobalStateEncoder expects day_summary_embedding but got None.")
            parts.append(day_summary_embedding)
        if self.use_internal_state:
            if internal_state is None:
                raise ValueError("GlobalStateEncoder expects internal_state but got None.")
            if internal_state.ndim != 2 or int(internal_state.shape[-1]) != self.d_internal_state_input:
                raise ValueError(
                    "GlobalStateEncoder expects internal_state [B,Dinternal], "
                    f"got shape {tuple(internal_state.shape)}."
            )
            parts.append(internal_state)
        if self.use_x:
            if x_features is None:
                raise ValueError("GlobalStateEncoder expects x_features but got None.")
            if self.x_panel_encoder is None:
                raise RuntimeError("GlobalStateEncoder use_x=True but x_panel_encoder is not initialized.")
            parts.append(self.x_panel_encoder(x_features))
        if not parts:
            raise ValueError("GlobalStateEncoder received no active input branch.")
        for part in parts:
            if part.ndim != 2:
                raise ValueError(f"GlobalStateEncoder expects 2D branch inputs, got shape {tuple(part.shape)}")
            if int(part.shape[0]) != int(batch_size):
                raise ValueError(
                    f"GlobalStateEncoder branch batch mismatch: expected B={int(batch_size)}, "
                    f"got shape {tuple(part.shape)}."
                )

        global_input = torch.cat(parts, dim=-1)
        global_state = self.encoder(global_input)  # [B, Dg]
        return global_state, global_state


class LocalStateEncoder(nn.Module):
    """
    Stock-local deterministic state encoder built from raw panel statistics.
    """

    SUPPORTED_INPUT_MODES = {"last_mean_std_trend_vol", "rich_stats_v1", "factor_projection_v1"}

    def __init__(
        self,
        *,
        d_global_state: int,
        d_local_state: int,
        input_mode: str = "last_mean_std_trend_vol",
        num_alphas: int | None = None,
        d_internal_state_input: int = 0,
        use_internal_state: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_mode = str(input_mode or "last_mean_std_trend_vol").strip().lower()
        if self.input_mode not in self.SUPPORTED_INPUT_MODES:
            raise ValueError(
                f"Unsupported local_state_input_mode: {self.input_mode}. "
                f"Supported: {sorted(self.SUPPORTED_INPUT_MODES)}"
            )
        self.d_global_state = int(d_global_state)
        self.d_local_state = int(d_local_state)
        self.num_alphas = int(num_alphas or 0)
        self.use_internal_state = bool(use_internal_state)
        self.d_internal_state_input = int(d_internal_state_input or 0)
        self.base_stats_dim = 6

        self.vector_projs = None
        if self.input_mode == "last_mean_std_trend_vol":
            self.stats_dim = self.base_stats_dim
            encoder_in = self.stats_dim + self.d_global_state
        elif self.input_mode == "rich_stats_v1":
            self.stats_dim = self.base_stats_dim + 32
            encoder_in = self.stats_dim + self.d_global_state
        else:
            if self.num_alphas <= 0:
                raise ValueError("factor_projection_v1 requires num_alphas > 0.")
            branch_dim = max(16, self.d_local_state // 2)
            self.vector_projs = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        RMSNorm(self.num_alphas),
                        nn.Linear(self.num_alphas, branch_dim),
                        nn.GELU(),
                    )
                    for name in ("last", "window_mean", "trend", "vol")
                }
            )
            self.stats_dim = self.base_stats_dim
            encoder_in = 4 * branch_dim + self.stats_dim + self.d_global_state
        if self.use_internal_state:
            if self.d_internal_state_input <= 0:
                raise ValueError("LocalStateEncoder internal-state branch requires d_internal_state_input > 0.")
            encoder_in += self.d_internal_state_input

        hidden = max(self.d_local_state * 2, 64)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout or 0.0)),
            nn.Linear(hidden, self.d_local_state),
            RMSNorm(self.d_local_state),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
        """Compute mean over valid (finite) elements only."""
        x_clean = torch.where(valid, x, torch.zeros_like(x))
        count = valid.float().sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
        return x_clean.sum(dim=dim, keepdim=keepdim) / count

    @staticmethod
    def _masked_std(x: torch.Tensor, valid: torch.Tensor, mean: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
        """Compute std over valid (finite) elements only (biased, i.e. ddof=0)."""
        diff = torch.where(valid, x - mean, torch.zeros_like(x))
        count = valid.float().sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
        return (diff.square().sum(dim=dim, keepdim=keepdim) / count).sqrt()

    @staticmethod
    def _compute_stats(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"LocalStateEncoder expects x with shape [B,T,N], got {tuple(x.shape)}")
        valid = torch.isfinite(x)  # [B, T, N]
        x_clean = torch.where(valid, x, torch.zeros_like(x))

        # --- last timestep stats (across N factors) ---
        last = x_clean[:, -1, :]          # [B, N]
        last_valid = valid[:, -1, :]      # [B, N]
        last_mean = LocalStateEncoder._masked_mean(last, last_valid, dim=-1, keepdim=True)
        last_std = LocalStateEncoder._masked_std(last, last_valid, last_mean, dim=-1, keepdim=True)

        # --- full window stats (across T*N) ---
        window_mean_2d = LocalStateEncoder._masked_mean(
            x_clean.reshape(x.shape[0], -1),
            valid.reshape(x.shape[0], -1),
            dim=-1, keepdim=True,
        )  # [B, 1]
        window_std = LocalStateEncoder._masked_std(
            x_clean.reshape(x.shape[0], -1),
            valid.reshape(x.shape[0], -1),
            window_mean_2d,
            dim=-1, keepdim=True,
        )  # [B, 1]
        window_mean = window_mean_2d

        # --- trend: difference between last and first timestep ---
        first = x_clean[:, 0, :]          # [B, N]
        first_valid = valid[:, 0, :]
        both_valid = last_valid & first_valid  # [B, N]
        trend_raw = torch.where(both_valid, last - first, torch.zeros_like(last))
        trend_count = both_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        trend_mean = trend_raw.sum(dim=-1, keepdim=True) / trend_count

        # --- temporal volatility: mean absolute diff across time ---
        if x.shape[1] > 1:
            diffs = x_clean[:, 1:, :] - x_clean[:, :-1, :]        # [B, T-1, N]
            diff_valid = valid[:, 1:, :] & valid[:, :-1, :]        # both steps must be valid
            diffs_masked = torch.where(diff_valid, diffs.abs(), torch.zeros_like(diffs))
            diff_count = diff_valid.float().sum(dim=(1, 2)).clamp_min(1.0)  # [B]
            temp_absdiff = (diffs_masked.sum(dim=(1, 2)) / diff_count).unsqueeze(-1)  # [B, 1]
        else:
            temp_absdiff = torch.zeros_like(last_mean)

        return torch.cat(
            [last_mean, last_std, window_mean, window_std, trend_mean, temp_absdiff],
            dim=-1,
        )

    @staticmethod
    def _moment_stats(flat: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        mean = flat.mean(dim=-1, keepdim=True)
        std = flat.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        z = (flat - mean) / std
        skew = z.pow(3).mean(dim=-1, keepdim=True)
        kurt = z.pow(4).mean(dim=-1, keepdim=True)
        return skew, kurt

    @staticmethod
    def _quantiles(flat: torch.Tensor) -> torch.Tensor:
        quantile_input = flat.float()
        q = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=flat.device, dtype=quantile_input.dtype)
        return torch.quantile(quantile_input, q, dim=-1).transpose(0, 1).to(dtype=flat.dtype)

    @staticmethod
    def _compute_rich_stats(x: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x)
        x_clean = torch.where(valid, x, torch.zeros_like(x))
        B, T, N = x_clean.shape
        base = LocalStateEncoder._compute_stats(x)

        last = x_clean[:, -1, :]
        flat = x_clean.reshape(B, -1)
        last_q = LocalStateEncoder._quantiles(last)
        window_q = LocalStateEncoder._quantiles(flat)
        last_skew, last_kurt = LocalStateEncoder._moment_stats(last)
        window_skew, window_kurt = LocalStateEncoder._moment_stats(flat)

        first = x_clean[:, 0, :]
        trend = last - first
        trend_q = LocalStateEncoder._quantiles(trend)
        trend_mean = trend.mean(dim=-1, keepdim=True)
        trend_std = trend.std(dim=-1, unbiased=False, keepdim=True)

        if T > 1:
            diffs = (x_clean[:, 1:, :] - x_clean[:, :-1, :]).abs().reshape(B, -1)
        else:
            diffs = torch.zeros_like(flat)
        diff_q = LocalStateEncoder._quantiles(diffs)
        diff_mean = diffs.mean(dim=-1, keepdim=True)
        diff_std = diffs.std(dim=-1, unbiased=False, keepdim=True)

        pos_breadth = (last > 0).to(last.dtype).mean(dim=-1, keepdim=True)
        neg_breadth = (last < 0).to(last.dtype).mean(dim=-1, keepdim=True)
        k = max(1, int(round(float(N) * 0.05)))
        denom = last.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        top_pos_mass = last.clamp_min(0).topk(k, dim=-1).values.sum(dim=-1, keepdim=True) / denom
        bottom_neg_mass = (-last.clamp_max(0)).topk(k, dim=-1).values.sum(dim=-1, keepdim=True) / denom

        return torch.cat(
            [
                base,
                last_q,
                window_q,
                last_skew,
                last_kurt,
                window_skew,
                window_kurt,
                trend_mean,
                trend_std,
                trend_q,
                diff_mean,
                diff_std,
                diff_q,
                pos_breadth,
                neg_breadth,
                top_pos_mass,
                bottom_neg_mass,
            ],
            dim=-1,
        )

    @staticmethod
    def _compute_factor_vectors(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = torch.isfinite(x)
        x_clean = torch.where(valid, x, torch.zeros_like(x))
        last_vec = x_clean[:, -1, :]
        window_count = valid.float().sum(dim=1).clamp_min(1.0)
        window_mean_vec = x_clean.sum(dim=1) / window_count
        first_vec = x_clean[:, 0, :]
        trend_vec = last_vec - first_vec
        if x.shape[1] > 1:
            diff = (x_clean[:, 1:, :] - x_clean[:, :-1, :]).abs()
            diff_valid = (valid[:, 1:, :] & valid[:, :-1, :]).float()
            vol_vec = diff.sum(dim=1) / diff_valid.sum(dim=1).clamp_min(1.0)
        else:
            vol_vec = torch.zeros_like(last_vec)
        return last_vec, window_mean_vec, trend_vec, vol_vec

    def forward(
        self,
        x: torch.Tensor,
        global_state: torch.Tensor,
        internal_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_mode == "rich_stats_v1":
            stats = self._compute_rich_stats(x)
        else:
            stats = self._compute_stats(x)
        if global_state.ndim != 2:
            raise ValueError(f"LocalStateEncoder expects global_state [B,Dg], got {tuple(global_state.shape)}")
        if stats.shape[0] != global_state.shape[0]:
            raise ValueError(
                f"LocalStateEncoder batch mismatch: stats.shape={tuple(stats.shape)}, "
                f"global_state.shape={tuple(global_state.shape)}"
            )
        internal_part = None
        if self.use_internal_state:
            if internal_state is None:
                raise ValueError("LocalStateEncoder expects internal_state but got None.")
            if internal_state.ndim != 2 or int(internal_state.shape[0]) != int(stats.shape[0]):
                raise ValueError(
                    "LocalStateEncoder expects internal_state [B,Dinternal], "
                    f"got shape {tuple(internal_state.shape)}."
                )
            if int(internal_state.shape[-1]) != self.d_internal_state_input:
                raise ValueError(
                    f"LocalStateEncoder internal_state expects D={self.d_internal_state_input}, "
                    f"got {int(internal_state.shape[-1])}."
                )
            internal_part = internal_state.to(device=stats.device, dtype=stats.dtype)
        if self.input_mode == "factor_projection_v1":
            if self.vector_projs is None:
                raise RuntimeError("factor_projection_v1 is configured without vector projections.")
            if int(x.shape[-1]) != self.num_alphas:
                raise ValueError(
                    f"factor_projection_v1 expects N={self.num_alphas}, got x.shape={tuple(x.shape)}."
                )
            last_vec, window_mean_vec, trend_vec, vol_vec = self._compute_factor_vectors(x)
            local_input = torch.cat(
                [
                    self.vector_projs["last"](last_vec),
                    self.vector_projs["window_mean"](window_mean_vec),
                    self.vector_projs["trend"](trend_vec),
                    self.vector_projs["vol"](vol_vec),
                    stats,
                    global_state,
                ],
                dim=-1,
            )
        else:
            local_input = torch.cat([stats, global_state], dim=-1)
        if internal_part is not None:
            local_input = torch.cat([local_input, internal_part], dim=-1)
        local_state = self.encoder(local_input)
        return local_state, stats
