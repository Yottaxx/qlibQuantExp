"""Cross-stock self-attention at the READOUT (h-20260619).

Placed post-pool / pre-head on `h_pooled : [B, D]`, where B is ONE trading day's cross-section
(an unordered set of stocks). This is the migration target of the warm-q series: it removes all
three warm-q pathologies at once.

  P1 (loss-neutral / RankIC!=IR)  -> ungated, in-series, MANDATORY residual. No router exit; the
        layer feeds the cross-sectional ranking loss directly, so uniform attention is a repeller
        (a flat cross-section is the degenerate max-ListMLE), not a free parking spot.
  P2 (cold-query collapse, ||Wq||->0 under WD pins softmax uniform) -> QK-norm + learnable temp.
        Logits are temp * cos(q,k): magnitude-free, so weight decay can no longer force uniform;
        sharpening rides a single well-conditioned scalar with direct gradient from the loss.
  P3 (V-amplification escape, ||Wv||||Wo||->88) -> NO de-mean, NO output-norm. The escape exists
        only when a scale-invariant constraint is present to game; the mandatory/replacing residual
        needs none, so it cannot exist here.

Variants:
  qknorm=True  -> R1 (the bet);  qknorm=False -> R0 (fixed 1/sqrt(d_head) scale, placement-only ctrl)
  gated=True   -> R2 control (cold sigmoid residual gate; predicted to freeze near init -> dead axis)

`out_proj` is zero-initialized (identity-start): at step 0 the readout is byte-identical to baseline
and the layer "turns on" only as gradient grows out_proj -- a clean warm-start with NO frozen scalar
gate (out_proj is a full matrix, rich gradient, cannot freeze the way a cold sigmoid does).

The forward `override` hook (None | "uniform" | "zero") mirrors moe_block's `router_override` and
exists for the eval leave-one-out ablation (uniformize / disable the layer without retraining).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossStockBlock(nn.Module):
    def __init__(self, d_model, n_heads, qknorm=True, gated=False, temp_init=4.0,
                 use_ffn=False, dropout=0.0, contrast=False):
        super().__init__()
        n_heads = int(n_heads) if int(n_heads) > 0 else 1
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = int(d_model)
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.qknorm = bool(qknorm)
        self.gated = bool(gated)
        # h-20260704 C1 CONTRAST operator: replace the pooling output o=Σ_j a_ij·Wv·u_j with the
        # CONTRAST c_i = Wv·u_i − Σ_j a_ij·Wv·u_j (each stock minus its attention-weighted peer set).
        # Motivation (ledger 82): uniform attention ⇒ Σ_j a_ij·Wv·u_j = mean_j(Wv·u_j) ⇒ c_i = Wv·(u_i −
        # mean(u)) = the pure cross-sectional demeaned coordinate ⇒ a common-mode-free, V-escape-immune
        # signal that the CS-blind backbone (ledger 84) cannot produce per-stock. Repeller: a flat
        # cross-section ⇒ c_i≈0 ⇒ no free parking spot (unlike the standard pooling attractor). This is
        # the only structurally-distinct untried cross-stock variant after R1(pooling)=n1-parity.
        self.contrast = bool(contrast)

        self.ln_in = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(float(dropout))

        if self.qknorm:
            # learnable temperature stored as log-scale (positive, clamped). Mild init => entropy
            # starts ~1 (near-uniform) so a measured DROP is the falsifiable "it sharpened" signal.
            self.log_temp = nn.Parameter(torch.tensor(math.log(float(temp_init))))
        if self.gated:
            self.gate = nn.Parameter(torch.full((1,), -2.0))  # cold sigmoid(-2)~0.12 (R2 control)

        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.ln_ffn = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 2 * d_model), nn.GELU(),
                nn.Linear(2 * d_model, d_model), nn.Dropout(float(dropout)),
            )

        self.reset_identity_start()
        # eval-only descriptors (NEVER verdicts — weight stats are blind to usefulness)
        self.last_entropy_norm = None
        self.last_out_norm = None

    def reset_identity_start(self):
        # zero out_proj => o=0 at init => readout starts exactly at baseline. Re-call AFTER the model's
        # post_init (HF _init_weights re-randomizes all nn.Linear, clobbering this).
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, u, override=None):
        # u: [B, D], one trading day's cross-section. override in {None,"uniform","zero"}.
        if u.dim() != 2:
            raise RuntimeError(f"CrossStockBlock expects [B,D]; got {tuple(u.shape)}")
        B, D = u.shape
        x = self.ln_in(u)
        q = self.q_proj(x).view(B, self.n_heads, self.d_head).transpose(0, 1)  # [H,B,dh]
        k = self.k_proj(x).view(B, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v_proj(x).view(B, self.n_heads, self.d_head).transpose(0, 1)

        if self.qknorm:
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            scale = self.log_temp.float().exp().clamp(max=100.0)
        else:
            q = q.float()
            k = k.float()
            scale = 1.0 / math.sqrt(self.d_head)

        logits = torch.einsum("hid,hjd->hij", q, k) * scale  # [H,B,B] fp32 (AMP-safe, B~300)
        if override == "uniform":
            attn = torch.full_like(logits, 1.0 / max(B, 1))
        else:
            attn = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("hij,hjd->hid", attn.to(v.dtype), v)  # [H,B,dh]
        if self.contrast:
            # c_i = Wv·u_i − Σ_j a_ij·Wv·u_j : the stock's own value minus its attention-weighted peers.
            o = v - pooled
        else:
            o = pooled
        o = o.transpose(0, 1).reshape(B, D)
        o = self.out_proj(o)
        o = self.drop(o)
        if override == "zero":
            o = torch.zeros_like(o)
        if self.gated:
            o = torch.sigmoid(self.gate.to(o.dtype)) * o

        u = u + o  # MANDATORY residual (no gate in R0/R1)
        if self.use_ffn:
            u = u + self.ffn(self.ln_ffn(u))

        if not self.training:
            with torch.no_grad():
                a = attn.float().clamp_min(1e-9)
                ent = -(a * a.log()).sum(-1).mean()
                self.last_entropy_norm = float((ent / math.log(max(B, 2))).item())
                self.last_out_norm = float(o.detach().float().norm(dim=-1).mean().item())
        return u
