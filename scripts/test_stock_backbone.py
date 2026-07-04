"""Unit smoke for h-20260624 stock-backbone (main-path cross-stock residual).

Validates the moe_block change WITHOUT the full pipeline:
  - block builds with stock_backbone for gamma_init in {0.0, 1.0}
  - forward (train + eval) returns correct [B,T,N,D] shape
  - backward populates stock_backbone_gamma.grad (=> gamma can move off init)
  - gamma=0 init => stock branch contributes EXACTLY 0 (baseline-identical at init)
  - no_wd substring 'stock_backbone' catches the attention proj tensors (warm-q parity)
  - diag carries stock_backbone_gamma (train) + ..._attn_entropy_norm (eval)
Run: python scripts/test_stock_backbone.py
"""
import torch
from module.utils.model_configuration import QuantMoEConfig
from module.architecture.moe_block import RegimeAdaptiveMoEBlock

torch.manual_seed(0)
B, T, N, D = 16, 8, 12, 64  # B = one day's cross-section


def build(gamma_init):
    cfg = QuantMoEConfig(d_model=D, n_heads=4, d_ff=128, dropout=0.0,
                         stock_backbone=True, stock_backbone_gamma_init=gamma_init)
    return RegimeAdaptiveMoEBlock(cfg)


def run(block, train):
    block.train(train)
    x = torch.randn(B, T, N, D, requires_grad=True)
    reg = torch.randn(B, D)
    out, diag, _ = block(x, reg, return_attn=False)
    return x, out, diag


fails = []

# 1) gamma=0 : builds, shapes, backward, grad on gamma, branch is a no-op at init
blk0 = build(0.0)
x, out, diag = run(blk0, train=True)
if out.shape != (B, T, N, D):
    fails.append(f"shape {out.shape} != {(B,T,N,D)}")
out.float().pow(2).mean().backward()
if blk0.stock_backbone_gamma.grad is None:
    fails.append("gamma.grad is None (train) -> gamma cannot move")
else:
    print(f"[ok] gamma=0 grad = {blk0.stock_backbone_gamma.grad.item():+.4e} (nonzero => can leave 0)")
if "stock_backbone_gamma" not in diag:
    fails.append("diag missing stock_backbone_gamma (train)")

# gamma=0 init => x + 0*sb_out == x going into the rest; verify the branch term is exactly 0
with torch.no_grad():
    from einops import rearrange
    xin = torch.randn(B, T, N, D)
    h_sb = rearrange(blk0.stock_backbone_norm(xin), "b t n d -> (t n) b d")
    sb = blk0.stock_backbone_attn(h_sb, None, return_attn=False)
    contrib = (blk0.stock_backbone_gamma * rearrange(sb, "(t n) b d -> b t n d", t=T, n=N))
    if contrib.abs().max().item() != 0.0:
        fails.append(f"gamma=0 branch contrib not exactly 0 (max={contrib.abs().max().item()})")
    else:
        print("[ok] gamma=0 => branch contributes exactly 0 (baseline-identical at init)")

# 2) eval path produces the entropy descriptor
_, _, diag_e = run(blk0, train=False)
if "stock_backbone_attn_entropy_norm" not in diag_e:
    fails.append("diag missing stock_backbone_attn_entropy_norm (eval)")
else:
    print(f"[ok] eval entropy_norm = {float(diag_e['stock_backbone_attn_entropy_norm']):.4f}")

# 3) gamma=1 : builds + runs, branch is active
blk1 = build(1.0)
_, out1, diag1 = run(blk1, train=True)
if float(diag1["stock_backbone_gamma"]) != 1.0:
    fails.append(f"gamma_init=1.0 not respected (got {float(diag1['stock_backbone_gamma'])})")
else:
    print("[ok] gamma=1 builds & forwards; gamma==1.0 at init")

# 4) no_wd substring match: 'stock_backbone' must catch the attn proj tensors
names = [n for n, _ in blk1.named_parameters() if "stock_backbone" in n]
proj = [n for n in names if "in_proj" in n or "out_proj" in n]
if not proj:
    fails.append(f"no_wd scope 'stock_backbone' catches no proj tensors; names={names}")
else:
    print(f"[ok] no_wd 'stock_backbone' catches {len(names)} tensors incl proj: {proj}")

# 5) router stays 2-way (use_stock_expert False)
if blk1.n_experts != 2:
    fails.append(f"n_experts={blk1.n_experts} != 2 (router should stay 2-way)")
else:
    print("[ok] router stays 2-way (n_experts=2)")

print()
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("ALL PASS")
