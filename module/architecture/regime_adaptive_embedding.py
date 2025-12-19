import math

import torch
from torch import nn
import torch.nn.functional as F


def _inv_softplus(x: float) -> float:
    # inverse of softplus for x>0: softplus(y)=log(1+exp(y))
    x = float(x)
    if x <= 0:
        return -20.0
    return math.log(math.expm1(x))


class RegimeAdaptiveTimeEmbedding(nn.Module):
    """
    Regime-adaptive time embedding for short windows (T ~ 8-32).

    We learn a base lag embedding table p[t] and modulate it with a per-sample
    time-scale tau(regime) (predicted by a small MLP) via an exponential decay over lag:
        w_lag = exp(-lag / tau)

    The decay weights are normalized to keep mean(w)=1 to avoid scale drift across regimes.
    """

    def __init__(
        self,
        *,
        d_model: int,
        max_len: int,
        tau_min: float = 0.5,
        tau_max: float = 50.0,
        tau_init: float = 5.0,
        init_std: float = 0.02,
        tau_mlp_hidden: int | None = None,
        tau_mlp_out_scale: float = 0.01,
        normalize_decay: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.normalize_decay = bool(normalize_decay)
        self.eps = float(eps)

        self.pos = nn.Parameter(torch.empty(self.max_len, self.d_model))
        init_std = float(init_std)
        if init_std > 0:
            nn.init.normal_(self.pos, std=init_std)
        else:
            nn.init.zeros_(self.pos)

        # τ(r): MLP with "base + scaled delta" parameterization
        # - base initialized to yield τ≈tau_init (after softplus + tau_min)
        # - delta starts small (LayerScale-style) to keep early training stable while allowing gradients to flow
        hidden = int(tau_mlp_hidden) if tau_mlp_hidden is not None else max(16, self.d_model // 2)
        self.tau_norm = nn.LayerNorm(self.d_model)
        self.tau_fc1 = nn.Linear(self.d_model, hidden)
        self.tau_fc2 = nn.Linear(hidden, 1)
        self.tau_mlp_out_scale = float(tau_mlp_out_scale)

        tau0 = _inv_softplus(max(float(tau_init) - self.tau_min, 1e-6))
        self.tau_base = nn.Parameter(torch.tensor(tau0, dtype=torch.float32))

    def forward(self, regime_embedding: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            regime_embedding: [B, D]
            seq_len: T (<= max_len)
        Returns:
            time_emb: [B, T, D]
            tau: [B, 1]
        """
        if regime_embedding.ndim != 2:
            raise ValueError(f"Expected regime_embedding [B,D], got {tuple(regime_embedding.shape)}")

        T = int(seq_len)
        if T <= 0:
            raise ValueError(f"seq_len must be >0, got {T}")
        if T > self.max_len:
            raise ValueError(f"seq_len={T} exceeds max_len={self.max_len}; increase context_len/max_len.")

        h = self.tau_norm(regime_embedding)
        h = F.gelu(self.tau_fc1(h))
        tau_raw = self.tau_base.to(device=h.device, dtype=h.dtype) + self.tau_fc2(h) * self.tau_mlp_out_scale  # [B,1]
        tau = F.softplus(tau_raw) + self.tau_min  # [B,1]
        if self.tau_max > 0:
            tau = tau.clamp_max(self.tau_max)

        # lag=0 at the last time step, lag increases into the past
        lags = torch.arange(T - 1, -1, -1, device=regime_embedding.device, dtype=regime_embedding.dtype)  # [T]
        w = torch.exp(-lags.view(1, T) / tau)  # [B,T]
        if self.normalize_decay:
            w = w / (w.mean(dim=-1, keepdim=True) + self.eps)

        pos = self.pos[:T].to(dtype=regime_embedding.dtype, device=regime_embedding.device)  # [T,D]
        time_emb = w.unsqueeze(-1) * pos.unsqueeze(0)  # [B,T,D]
        return time_emb, tau


class RegimeAdaptiveFactorGate(nn.Module):
    """
    Regime-adaptive factor FiLM (per factor-ID, permutation equivariant).

    Applies per-sample, per-factor modulation on the *post-LN* activations:
        y = x_norm * gamma + beta

    We use a tanh-bilinear interaction between regime embedding r_b and factor embedding e_n:
        s(b,n,d) = (W r_b)[d] * LN(e_n)[d] / sqrt(D)
        gamma = 1 + scale * tanh(s)
        beta  = shift_scale * tanh(s_beta)

    Initialization: projections are zero-initialized so gamma≈1 and beta≈0 at start,
    keeping the network close to an identity mapping (ResNet-style stability).
    """

    def __init__(self, *, d_model: int, gate_scale: float = 0.5, shift_scale: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.gate_scale = float(gate_scale)
        self.shift_scale = float(shift_scale)

        self.regime_norm = nn.LayerNorm(self.d_model)
        self.factor_norm = nn.LayerNorm(self.d_model)

        self.proj_gamma = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj_beta = nn.Linear(self.d_model, self.d_model, bias=False)

        # Identity-start: keep FiLM near no-op initially.
        nn.init.zeros_(self.proj_gamma.weight)
        nn.init.zeros_(self.proj_beta.weight)
        # Tell HF-style init to preserve identity (QuantMoEModel._init_weights checks this flag).
        setattr(self.proj_gamma, "_rstmoe_zero_init", True)
        setattr(self.proj_beta, "_rstmoe_zero_init", True)

    def forward(self, regime_embedding: torch.Tensor, factor_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            regime_embedding:  [B, D]
            factor_embeddings: [N, D]
        Returns:
            gamma: [B, N, D]
            beta:  [B, N, D]
        """
        if regime_embedding.ndim != 2:
            raise ValueError(f"Expected regime_embedding [B,D], got {tuple(regime_embedding.shape)}")
        if factor_embeddings.ndim != 2:
            raise ValueError(f"Expected factor_embeddings [N,D], got {tuple(factor_embeddings.shape)}")
        if int(factor_embeddings.shape[1]) != self.d_model:
            raise ValueError(f"factor_embeddings last dim must be D={self.d_model}, got {factor_embeddings.shape[1]}")

        r = self.regime_norm(regime_embedding)
        e = self.factor_norm(factor_embeddings)

        rg = self.proj_gamma(r)  # [B,D]
        sb = (rg.unsqueeze(1) * e.unsqueeze(0)) / math.sqrt(self.d_model)  # [B,N,D]
        gamma = 1.0 + self.gate_scale * torch.tanh(sb)

        beta = torch.zeros_like(gamma)
        if self.shift_scale != 0.0:
            rb = self.proj_beta(r)  # [B,D]
            sb2 = (rb.unsqueeze(1) * e.unsqueeze(0)) / math.sqrt(self.d_model)  # [B,N,D]
            beta = self.shift_scale * torch.tanh(sb2)

        return gamma, beta
