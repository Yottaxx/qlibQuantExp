import math

import torch
import torch.nn.functional as F
from module.architecture import RMSNorm
from torch import nn


def _inv_softplus(x: float) -> float:
    x = float(x)
    if x <= 0:
        return -20.0
    return math.log(math.expm1(x))


class RegimeAdaptiveTimeEmbedding(nn.Module):
    """
    Regime-adaptive time embedding for short windows.

    The embedding output lives in `d_model`, while the conditioning input can use
    a different dimension via `condition_dim`.
    """

    def __init__(
        self,
        *,
        d_model: int,
        max_len: int,
        condition_dim: int | None = None,
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
        self.condition_dim = int(condition_dim if condition_dim is not None else d_model)
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

        hidden = int(tau_mlp_hidden) if tau_mlp_hidden is not None else max(16, self.condition_dim // 2)
        self.tau_norm = RMSNorm(self.condition_dim)
        self.tau_fc1 = nn.Linear(self.condition_dim, hidden)
        self.tau_fc2 = nn.Linear(hidden, 1)
        self.tau_mlp_out_scale = float(tau_mlp_out_scale)

        tau0 = _inv_softplus(max(float(tau_init) - self.tau_min, 1e-6))
        self.tau_base = nn.Parameter(torch.tensor(tau0, dtype=torch.float32))

    def forward(self, regime_embedding: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if regime_embedding.ndim != 2:
            raise ValueError(f"Expected regime_embedding [B,D], got {tuple(regime_embedding.shape)}")

        T = int(seq_len)
        if T <= 0:
            raise ValueError(f"seq_len must be >0, got {T}")
        if T > self.max_len:
            raise ValueError(f"seq_len={T} exceeds max_len={self.max_len}; increase context_len/max_len.")

        h = self.tau_norm(regime_embedding)
        h = F.gelu(self.tau_fc1(h))
        tau_raw = self.tau_base.to(device=h.device, dtype=h.dtype) + self.tau_fc2(h) * self.tau_mlp_out_scale
        tau = F.softplus(tau_raw) + self.tau_min
        if self.tau_max > 0:
            tau = tau.clamp_max(self.tau_max)

        lags = torch.arange(T - 1, -1, -1, device=regime_embedding.device, dtype=regime_embedding.dtype)
        decay_logits = -lags.view(1, T) / tau
        w = torch.exp(decay_logits)
        if self.normalize_decay:
            # Keep total time-embedding energy comparable across tau while avoiding
            # the large mean-normalization amplification seen for short windows.
            w = F.softmax(decay_logits, dim=-1) * float(T)

        pos = self.pos[:T].to(dtype=regime_embedding.dtype, device=regime_embedding.device)
        time_emb = w.unsqueeze(-1) * pos.unsqueeze(0)
        return time_emb, tau


class RegimeAdaptiveFactorGate(nn.Module):
    """
    Regime-adaptive factor FiLM with explicit global/local conditioning.
    """

    def __init__(
        self,
        *,
        d_model: int,
        global_dim: int,
        local_dim: int = 0,
        use_global_state: bool = True,
        use_local_state: bool = False,
        gate_scale: float = 0.5,
        shift_scale: float = 0.0,
        local_scale_init: float = 0.1,
        local_scale_learnable: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.global_dim = int(global_dim)
        self.local_dim = int(local_dim or 0)
        self.use_global_state = bool(use_global_state)
        self.use_local_state = bool(use_local_state)
        self.gate_scale = float(gate_scale)
        self.shift_scale = float(shift_scale)
        self.local_scale_enabled = bool(self.use_local_state)

        if not (self.use_global_state or self.use_local_state):
            raise ValueError("RegimeAdaptiveFactorGate requires at least one enabled state branch.")

        self.factor_norm = RMSNorm(self.d_model)
        self.global_norm = RMSNorm(self.global_dim) if self.use_global_state else None
        self.local_norm = RMSNorm(self.local_dim) if self.use_local_state else None
        self.gamma_norm = RMSNorm(self.d_model)
        self.beta_norm = RMSNorm(self.d_model)

        self.proj_gamma_global = (
            nn.Linear(self.global_dim, self.d_model, bias=False) if self.use_global_state else None
        )
        self.proj_gamma_local = (
            nn.Linear(self.local_dim, self.d_model, bias=False) if self.use_local_state else None
        )
        self.proj_beta_global = (
            nn.Linear(self.global_dim, self.d_model, bias=False) if self.use_global_state else None
        )
        self.proj_beta_local = (
            nn.Linear(self.local_dim, self.d_model, bias=False) if self.use_local_state else None
        )
        if self.local_scale_enabled:
            init = torch.tensor(float(local_scale_init), dtype=torch.float32)
            if bool(local_scale_learnable):
                self.local_film_scale = nn.Parameter(init)
            else:
                self.register_buffer("local_film_scale", init)
        else:
            self.local_film_scale = None

        for proj in (
            self.proj_gamma_global,
            self.proj_gamma_local,
            self.proj_beta_global,
            self.proj_beta_local,
        ):
            if proj is None:
                continue
            nn.init.zeros_(proj.weight)
            setattr(proj, "_rstmoe_zero_init", True)

    def forward(
        self,
        global_state: torch.Tensor | None,
        local_state: torch.Tensor | None,
        factor_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if factor_embeddings.ndim != 2:
            raise ValueError(f"Expected factor_embeddings [N,D], got {tuple(factor_embeddings.shape)}")
        if int(factor_embeddings.shape[1]) != self.d_model:
            raise ValueError(f"factor_embeddings last dim must be D={self.d_model}, got {factor_embeddings.shape[1]}")

        e = self.factor_norm(factor_embeddings)
        gamma_parts = []
        beta_parts = []
        global_sensitivity = 0.0
        local_sensitivity = 0.0

        if self.use_global_state:
            if global_state is None or global_state.ndim != 2:
                raise ValueError("RegimeAdaptiveFactorGate expects global_state [B,Dg].")
            g = self.global_norm(global_state)
            g_gamma = self.proj_gamma_global(g)
            gamma_parts.append(g_gamma)
            global_sensitivity = float(g_gamma.detach().norm(dim=-1).mean().item())
            if self.shift_scale != 0.0 and self.proj_beta_global is not None:
                beta_parts.append(self.proj_beta_global(g))

        if self.use_local_state:
            if local_state is None or local_state.ndim != 2:
                raise ValueError("RegimeAdaptiveFactorGate expects local_state [B,Dl].")
            l = self.local_norm(local_state)
            l_gamma = self.proj_gamma_local(l)
            local_scale = self.local_film_scale.to(device=l_gamma.device, dtype=l_gamma.dtype)
            l_gamma_scaled = local_scale * l_gamma
            gamma_parts.append(l_gamma_scaled)
            local_sensitivity = float(l_gamma.detach().norm(dim=-1).mean().item())
            if self.shift_scale != 0.0 and self.proj_beta_local is not None:
                beta_parts.append(local_scale * self.proj_beta_local(l))

        if not gamma_parts:
            raise ValueError("RegimeAdaptiveFactorGate received no active conditioning branch.")

        gamma_cond = self.gamma_norm(torch.stack(gamma_parts, dim=0).sum(dim=0))
        sb = (gamma_cond.unsqueeze(1) * e.unsqueeze(0)) / math.sqrt(self.d_model)
        gamma = 1.0 + self.gate_scale * torch.tanh(sb)

        beta = torch.zeros_like(gamma)
        if self.shift_scale != 0.0 and beta_parts:
            beta_cond = self.beta_norm(torch.stack(beta_parts, dim=0).sum(dim=0))
            sb2 = (beta_cond.unsqueeze(1) * e.unsqueeze(0)) / math.sqrt(self.d_model)
            beta = self.shift_scale * torch.tanh(sb2)

        diag = {
            "film_global_sensitivity": float(global_sensitivity),
            "film_local_sensitivity": float(local_sensitivity),
            "local_film_scale": (
                float(self.local_film_scale.detach().item()) if self.local_film_scale is not None else 0.0
            ),
        }
        return gamma, beta, diag
