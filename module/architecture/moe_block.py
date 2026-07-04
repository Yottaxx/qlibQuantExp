# module/architecture/moe_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from module.utils.model_configuration import QuantMoEConfig
from module.architecture.parallel_attention import ParallelAttention


class RegimeAdaptiveMoEBlock(nn.Module):
    """Regime-adaptive MoE block with spatio-temporal disentanglement.

    Inputs:
      x: [B, T, N, D]
      regime_embedding: [B, D]
      attn_bias: None or Tensor or (bias_time, bias_factor)
        - bias_time   expected shape [1, H, T, T]
        - bias_factor expected shape [1, H, N, N]
    """

    def __init__(self, config: QuantMoEConfig):
        super().__init__()
        self.config = config
        self.use_layer_summary = bool(getattr(config, "router_use_layer_summary", False))
        self.router_mode = str(getattr(config, "router_mode", "learned") or "learned").strip().lower()

        d_model = int(config.d_model)
        router_in = d_model * (2 if self.use_layer_summary else 1)

        # Optional: per-layer market summary (same for the daily cross-section batch)
        # summary = proj([mean(stock_repr), std(stock_repr)]) where stock_repr = mean over factor tokens
        self.layer_summary_proj = None
        if self.use_layer_summary:
            self.layer_summary_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # h-20260610-002: optional third stock-axis expert — a SYMMETRIC PEER of time/factor
        # (same class, same post_init init; router widened 2->3 => initial gate ~1/3 each).
        # NO identity-start/zero-gate: experts compete on equal footing from scratch (owner design
        # decision; scalar gates freeze at init — a zero-start expert could never rise).
        self.use_stock_expert = bool(getattr(config, "use_stock_expert", False))
        self.n_experts = 3 if self.use_stock_expert else 2

        self.router = nn.Sequential(
            nn.Linear(router_in, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.n_experts),
        )

        self.time_expert = ParallelAttention(config)
        self.factor_expert = ParallelAttention(config)
        if self.use_stock_expert:
            self.stock_expert = ParallelAttention(config)

        # h-20260624: cross-stock attention on the MAIN PATH (pre-MoE residual), gated by a learnable
        # scalar gamma. Distinct from use_stock_expert (which ROUTES it as a 3rd peer): here it always
        # enriches x and the 2-way router (time/factor) is unchanged. NO de-mean / NO out_norm => no
        # V-escape driver. gamma_init: 0.0 = ReZero opt-in, 1.0 = full-on. n_experts is unaffected.
        self.stock_backbone = bool(getattr(config, "stock_backbone", False))
        if self.stock_backbone:
            self.stock_backbone_attn = ParallelAttention(config)
            self.stock_backbone_norm = nn.LayerNorm(config.d_model)
            _g0 = float(getattr(config, "stock_backbone_gamma_init", 0.0))
            self.stock_backbone_gamma = nn.Parameter(torch.tensor(_g0, dtype=torch.float32))

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        regime_embedding: torch.Tensor,
        attn_bias=None,
        return_attn: bool = False,
        *,
        factor_film: tuple[torch.Tensor, torch.Tensor] | None = None,
        feature_mask: torch.Tensor | None = None,
        router_override: str | None = None,
    ):
        B, T, N, D = x.shape

        # h-20260624: main-path cross-stock enrichment BEFORE residual capture, so it flows through
        # BOTH the residual stream and the two experts. stocks(B) on the seq axis (daily cross-section
        # — VALIDITY INVARIANT: batch = one day, guaranteed by the daily samplers / per-day eval).
        # NO de-mean, NO out_norm: x += gamma * StockAttn(LN(x)). gamma=0 => exact baseline at init.
        stock_bb_entropy = None
        if self.stock_backbone:
            h_sb = rearrange(self.stock_backbone_norm(x), "b t n d -> (t n) b d")
            sb_out = self.stock_backbone_attn(h_sb, None, return_attn=False)
            sb_out = rearrange(sb_out, "(t n) b d -> b t n d", t=T, n=N)
            x = x + self.stock_backbone_gamma.to(x.dtype) * sb_out
            if not self.training:
                # eval-only entropy probe (subsampled: last-step x 8 strided factor slots => attn
                # [n_sub,H,B,B]); descriptor only — measures whether the cold query un-froze.
                with torch.no_grad():
                    n_sub = min(8, N)
                    n_idx = torch.linspace(0, N - 1, n_sub, device=x.device).long()
                    g_idx = (T - 1) * N + n_idx  # (t n) is t-major => last-step groups
                    _, attn_sb = self.stock_backbone_attn(h_sb[g_idx], None, return_attn=True)
                    a_sb = attn_sb.float().clamp_min(1e-9)
                    stock_bb_entropy = (-(a_sb * a_sb.log()).sum(-1).mean()) / torch.log(
                        torch.tensor(float(B), device=x.device)
                    )

        residual = x

        x = self.norm1(x)

        # Build router input
        router_input = regime_embedding
        if self.use_layer_summary:
            # x: [B,T,N,D] after norm; summarize market state at this layer
            # - last time step -> [B,N,D]
            # - per-stock factor pooling -> [B,D]
            # - market mean/std over stocks -> [D],[D] (broadcast back to [B,D])
            h_last = x[:, -1, :, :]  # [B,N,D]
            per_stock = h_last.mean(dim=1)  # [B,D]
            m_mean = per_stock.mean(dim=0)
            m_std = per_stock.std(dim=0, unbiased=False)
            summary = torch.cat([m_mean, m_std], dim=-1)  # [2D]
            summary = self.layer_summary_proj(summary).unsqueeze(0).expand(B, -1)  # [B,D]
            router_input = torch.cat([regime_embedding, summary], dim=-1)  # [B,2D]

        E = self.n_experts
        if self.router_mode == "fixed_05":
            router_logits_raw = torch.zeros((B, E), device=x.device, dtype=x.dtype)
            gate_weights = torch.full((B, E), 1.0 / E, device=x.device, dtype=x.dtype)
        else:
            # Router exploration: optional logit noise + temperature scaling
            # - noise encourages exploration early in training
            # - temperature controls sharpness (lower => sharper)
            router_logits_raw = self.router(router_input)  # [B, E]
            router_logits = router_logits_raw
            noise_std = float(getattr(self.config, "router_noise", 0.0) or 0.0)
            if self.training and noise_std > 0:
                router_logits = router_logits + torch.randn_like(router_logits) * noise_std

            temperature = float(getattr(self.config, "router_temperature", 1.0) or 1.0)
            temperature = max(temperature, 1e-3)
            gate_weights = F.softmax(router_logits / temperature, dim=-1)  # [B, E]

        if router_override is not None:
            mode = str(router_override).strip().lower()
            if mode in {"", "none", "default", "learned"}:
                mode = "none"
            if mode == "time":
                gate_weights = torch.zeros((B, E), device=x.device, dtype=x.dtype)
                gate_weights[:, 0] = 1.0
            elif mode == "factor":
                gate_weights = torch.zeros((B, E), device=x.device, dtype=x.dtype)
                gate_weights[:, 1] = 1.0
            elif mode == "stock":
                if E < 3:
                    raise ValueError("router_override='stock' requires use_stock_expert=True")
                gate_weights = torch.zeros((B, E), device=x.device, dtype=x.dtype)
                gate_weights[:, 2] = 1.0
            elif mode == "no_stock":
                # leave-one-out probe: zero the stock gate, renormalize the LEARNED time/factor
                # weights — measures the stock expert's marginal contribution at eval (no retrain).
                if E < 3:
                    raise ValueError("router_override='no_stock' requires use_stock_expert=True")
                gw = gate_weights.clone()
                gw[:, 2] = 0.0
                gate_weights = gw / gw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            elif mode in {"uniform", "fixed_05", "fixed", "0.5"}:
                gate_weights = torch.full((B, E), 1.0 / E, device=x.device, dtype=x.dtype)
            elif mode != "none":
                raise ValueError(
                    f"Unsupported router_override={router_override!r}; "
                    "supported: None, time, factor, stock, no_stock, uniform"
                )

        # diagnostics
        # z-loss is computed on raw (un-noised, un-tempered) logits for stability
        if self.router_mode == "fixed_05":
            z_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            z_loss = (torch.logsumexp(router_logits_raw, dim=-1) ** 2).mean()
        entropy_per_sample = -(gate_weights * torch.log(gate_weights + 1e-9)).sum(-1)
        entropy = entropy_per_sample.mean()
        time_ratio = gate_weights[:, 0].mean()
        # MONITORING ONLY (routing itself stays soft/dense): router decisiveness = top1-top2 logit
        # gap. For E=2 this is numerically identical to the old |l0-l1|; for E=3 |l0-l1| would
        # ignore the stock logit and lose the metric's meaning.
        _top2 = router_logits_raw.topk(2, dim=-1).values
        logit_margin_per_sample = (_top2[:, 0] - _top2[:, 1]).abs()

        w_time = gate_weights[:, 0].view(B, 1, 1, 1)
        w_factor = gate_weights[:, 1].view(B, 1, 1, 1)
        w_stock = gate_weights[:, 2].view(B, 1, 1, 1) if E >= 3 else None

        # Apply factor-level conditioning after LayerNorm (FiLM/AdaLN style) so it is not canceled by Pre-LN.
        if factor_film is not None:
            gamma, beta = factor_film
            if gamma.ndim != 3 or beta.ndim != 3:
                raise ValueError(
                    f"factor_film expects (gamma,beta) with shape [B,N,D]; got {tuple(gamma.shape)}, {tuple(beta.shape)}"
                )
            if tuple(gamma.shape) != (B, N, D) or tuple(beta.shape) != (B, N, D):
                raise ValueError(
                    f"factor_film expects (gamma,beta) with shape [B,N,D]={B,N,D}; got {tuple(gamma.shape)}, {tuple(beta.shape)}"
                )
            x = x * gamma.view(B, 1, N, D) + beta.view(B, 1, N, D)

        # Feature selection mask (global factor gate) should apply last, so masked features stay masked even with FiLM shift.
        if feature_mask is not None:
            feature_mask = feature_mask.to(device=x.device, dtype=x.dtype)
            if feature_mask.ndim == 1:
                if int(feature_mask.shape[0]) != int(N):
                    raise ValueError(f"feature_mask must have length N={N}, got {int(feature_mask.shape[0])}")
                x = x * feature_mask.view(1, 1, N, 1)
            elif feature_mask.ndim == 2:
                if tuple(feature_mask.shape) != (B, N):
                    raise ValueError(f"feature_mask 2D must be [B,N]={B,N}, got {tuple(feature_mask.shape)}")
                x = x * feature_mask.view(B, 1, N, 1)
            else:
                raise ValueError(f"feature_mask must be 1D/2D, got shape {tuple(feature_mask.shape)}")

        bias_time = bias_factor = None
        if isinstance(attn_bias, (tuple, list)) and len(attn_bias) == 2:
            bias_time, bias_factor = attn_bias
        else:
            bias_time = attn_bias

        # 1) Time expert: per-factor temporal modeling
        h_time = rearrange(x, "b t n d -> (b n) t d")
        if return_attn:
            out_time, attn_time = self.time_expert(h_time, bias_time, return_attn=True)
        else:
            out_time = self.time_expert(h_time, bias_time, return_attn=False)
            attn_time = None
        out_time = rearrange(out_time, "(b n) t d -> b t n d", b=B, n=N)

        # 2) Factor expert: per-time cross-sectional modeling
        h_factor = rearrange(x, "b t n d -> (b t) n d")
        if return_attn:
            out_factor, attn_factor = self.factor_expert(h_factor, bias_factor, return_attn=True)
        else:
            out_factor = self.factor_expert(h_factor, bias_factor, return_attn=False)
            attn_factor = None
        out_factor = rearrange(out_factor, "(b t) n d -> b t n d", b=B, t=T)

        # 3) Stock expert (h-20260610-002): cross-STOCK attention within the daily batch.
        # VALIDITY INVARIANT: only meaningful when the batch is a single day's cross-section
        # (guaranteed by the daily samplers in train and the per-day predict/eval loops).
        # bias=None: stocks are unordered (permutation-equivariant — no positional bias).
        out_stock = None
        if self.use_stock_expert:
            h_stock = rearrange(x, "b t n d -> (t n) b d")
            if getattr(self.config, "stock_expert_xs_center", False):
                # Cross-sectional centering BEFORE projection: subtract the per-(t,n) mean over
                # the stock axis (dim=1 of [(t n), B, D]). Projection is linear, so this centers
                # q, k AND v across stocks -> the attention operates on the stock-RELATIVE
                # component, projecting out the large common-mode time/factor embedding that
                # otherwise homogenizes cross-stock keys and pins softmax to uniform.
                h_stock = h_stock - h_stock.mean(dim=1, keepdim=True)
            out_stock = self.stock_expert(h_stock, None, return_attn=False)
            out_stock = rearrange(out_stock, "(t n) b d -> b t n d", t=T, n=N)
            # h-20260617: cap the stock-expert OUTPUT magnitude BEFORE de-mean to foreclose the
            # V-amplification escape (uniform attention + ||Wv||||Wo||->88). Scale-invariant modes
            # remove magnitude as a degree of freedom => the only way to survive the de-mean is to
            # SHARPEN attention (Path A) or the router abandons the expert (honest death).
            # MUST run before de-mean: de-mean of a uniform field is ~0, normalizing 0/||0|| is
            # ill-conditioned, so always normalize first.
            _onorm = getattr(self.config, "stock_expert_out_norm", "none")
            if _onorm and _onorm != "none":
                _eps = 1e-6
                if _onorm == "rms":
                    # RMSNorm over D, per (b,t,n) token; NO learnable scale (parameter-free =>
                    # nothing to re-grow into a renamed gain). Recommended.
                    out_stock = out_stock * torch.rsqrt(
                        out_stock.pow(2).mean(dim=-1, keepdim=True) + _eps
                    )
                elif _onorm == "unit":
                    # per-token unit L2 over D (hardest scale-invariant cap).
                    out_stock = out_stock / out_stock.norm(dim=-1, keepdim=True).clamp_min(_eps)
                elif _onorm == "ln":
                    # LayerNorm over D, NO affine (mean-center + unit-var, parameter-free).
                    _mu = out_stock.mean(dim=-1, keepdim=True)
                    _var = out_stock.var(dim=-1, keepdim=True, unbiased=False)
                    out_stock = (out_stock - _mu) * torch.rsqrt(_var + _eps)
                elif _onorm == "ln_affine":
                    # LayerNorm WITH affine — INCLUDED ONLY to demonstrate Outcome C (gamma
                    # reopens the escape). Do NOT promote. Lazily build the affine on first use.
                    if not hasattr(self, "_stock_out_ln"):
                        self._stock_out_ln = nn.LayerNorm(D).to(out_stock.device)
                    out_stock = self._stock_out_ln(out_stock)
                elif _onorm == "block_rms":
                    # §1b gentle variant: one scalar RMS over the whole (B,D) cross-stock block
                    # per (t,n); allows one stock to dominate but caps TOTAL cross-stock energy.
                    _scale = out_stock.pow(2).mean(dim=(0, 3), keepdim=True).sqrt()  # [1,T,N,1]
                    out_stock = out_stock / (_scale + _eps)
                elif _onorm.startswith("cap:"):
                    # variant (d): clip ceiling only; below cap the V-escape is still local-cheapest.
                    _c = float(_onorm.split(":", 1)[1])
                    _n = out_stock.norm(dim=-1, keepdim=True).clamp_min(_eps)
                    out_stock = out_stock * (_c / _n).clamp_max(1.0)
                else:
                    raise ValueError(f"Unsupported stock_expert_out_norm={_onorm!r}")
            # h-20260610-002 follow-up: de-mean across the daily cross-section (B axis) so the
            # expert can ONLY contribute the rank-changing relative component. A uniform-attention
            # output is a per-day constant => mean-subtracted to ~0 => no longer a loss-neutral
            # parking spot; deviating from uniform becomes the only way to reduce loss.
            if getattr(self.config, "stock_expert_demean", False):
                out_stock = out_stock - out_stock.mean(dim=0, keepdim=True)

        eps = 1e-12
        time_flat = out_time.detach().float().reshape(B, -1)
        factor_flat = out_factor.detach().float().reshape(B, -1)
        time_norm_sample = time_flat.norm(dim=1)
        factor_norm_sample = factor_flat.norm(dim=1)
        time_contrib_sample = (w_time.detach().float() * out_time.detach().float()).reshape(B, -1).norm(dim=1)
        factor_contrib_sample = (w_factor.detach().float() * out_factor.detach().float()).reshape(B, -1).norm(dim=1)
        expert_cosine_sample = F.cosine_similarity(time_flat, factor_flat, dim=1, eps=eps)

        fused = w_time * out_time + w_factor * out_factor
        if out_stock is not None:
            fused = fused + w_stock * out_stock
        x = residual + fused
        x = x + self.ffn(self.norm2(x))

        diag = {
            "z_loss": z_loss,
            "entropy": entropy,
            "entropy_per_sample": entropy_per_sample,
            "time_ratio": time_ratio,
            "logit_margin": logit_margin_per_sample.mean(),
            "logit_margin_per_sample": logit_margin_per_sample,
            "weights": gate_weights,
            "time_expert_norm": time_norm_sample.mean(),
            "factor_expert_norm": factor_norm_sample.mean(),
            "time_contrib_norm": time_contrib_sample.mean(),
            "factor_contrib_norm": factor_contrib_sample.mean(),
            "contrib_norm_ratio": (time_contrib_sample / factor_contrib_sample.clamp_min(eps)).mean(),
            "expert_cosine": expert_cosine_sample.mean(),
            "time_winner_ratio": (time_contrib_sample > factor_contrib_sample).float().mean(),
        }

        if out_stock is not None:
            # L2 mechanism-liveness diagnostics (norm/gate stats — cheap, every pass)
            stock_flat = out_stock.detach().float().reshape(B, -1)
            stock_norm_sample = stock_flat.norm(dim=1)
            stock_contrib_sample = (w_stock.detach().float() * out_stock.detach().float()).reshape(B, -1).norm(dim=1)
            diag["stock_ratio"] = gate_weights[:, 2].mean()
            diag["stock_expert_norm"] = stock_norm_sample.mean()
            diag["stock_contrib_norm"] = stock_contrib_sample.mean()
            diag["stock_winner_ratio"] = (
                (stock_contrib_sample > torch.maximum(time_contrib_sample, factor_contrib_sample)).float().mean()
            )
            # orthogonality triangle: is the stock direction REDUNDANT with time/factor? (finding #5:
            # an expert's value is decorrelation — cos~1 means no new information even at high share)
            diag["expert_cosine_ts"] = F.cosine_similarity(time_flat, stock_flat, dim=1, eps=eps).mean()
            diag["expert_cosine_fs"] = F.cosine_similarity(factor_flat, stock_flat, dim=1, eps=eps).mean()

            # L3 structure descriptors (eval-only, SUBSAMPLED: t=last x 8 strided factor slots =>
            # attn [8,H,B,B] ~ 2.9M elems — never materialize the full 1264-group probs).
            # Descriptors only — NEVER verdicts (pool lesson: weight stats are blind to usefulness).
            if not self.training:
                with torch.no_grad():
                    n_sub = min(8, N)
                    n_idx = torch.linspace(0, N - 1, n_sub, device=x.device).long()
                    g_idx = (T - 1) * N + n_idx  # (t n) is t-major => last-step groups
                    sub = rearrange(x, "b t n d -> (t n) b d")[g_idx]  # [n_sub, B, D]
                    _, attn_s = self.stock_expert(sub, None, return_attn=True)  # [n_sub,H,B,B]
                    a = attn_s.float().clamp_min(1e-9)
                    ent = -(a * a.log()).sum(-1).mean()
                    diag["stock_attn_entropy_norm"] = ent / torch.log(
                        torch.tensor(float(B), device=x.device)
                    )
                    diag["stock_attn_self_frac"] = a.diagonal(dim1=-2, dim2=-1).mean()  # 1/B==uniform, ->1==self-collapse
                    col_mass = a.mean(dim=(0, 1, 2))  # [B] incoming-attention mass per stock
                    k = max(1, int(0.05 * B))
                    diag["stock_attn_hub_top5_share"] = col_mass.topk(k).values.sum() / col_mass.sum().clamp_min(eps)

        if self.stock_backbone:
            # mechanism signals: did gamma move off its init? did the cold query un-freeze (entropy)?
            diag["stock_backbone_gamma"] = self.stock_backbone_gamma.detach()
            if stock_bb_entropy is not None:
                diag["stock_backbone_attn_entropy_norm"] = stock_bb_entropy

        attn_dict = None
        if return_attn and (attn_time is not None or attn_factor is not None):
            attn_dict = {"time": attn_time, "factor": attn_factor}

        return x, diag, attn_dict
