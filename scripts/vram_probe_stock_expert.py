#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VRAM probe for h-20260610-002 (stock expert) on the 4070S BEFORE implementation.

Measures, at real shapes (B=300, T=8, N=158, D=64, fp16 autocast, TRAINING fwd+bwd):
  1) current anchor model (2-expert, d1pma g0.12) peak allocated  — the baseline;
  2) isolated stock-attention ParallelAttention on [(T*N)=1264, 300, 64] fwd+bwd — the delta driver;
  3) baseline + simulated 3rd expert (model fwd+bwd with an EXTRA ParallelAttention on the rearranged
     input added to the loss) — an upper-ish bound on the 3-expert peak without implementing 002.
Reports peaks + which SDPA backend serves the stock shape.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from module.utils.model_configuration import QuantMoEConfig  # noqa: E402
from module.quant_moe_model import QuantMoEModel  # noqa: E402
from module.architecture.parallel_attention import ParallelAttention  # noqa: E402

B, T, N = 300, 8, 158
BASE = dict(
    context_len=T, num_alphas=N,
    use_regime_time_embedding=True, use_regime_factor_gate=True,
    router_use_layer_summary=True, router_mode="learned",
    time_tau_mlp_out_scale=0.5, use_external_macro=False,
    temporal_readout="d1pma", temporal_readout_gate_init=-2.0,
)


def gb(x):
    return f"{x / 1024**3:.2f} GB"


def peak_of(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"device={torch.cuda.get_device_name(0)} total={gb(total)} torch={torch.__version__}")

    cfg = QuantMoEConfig(**BASE)
    x = torch.randn(B, T, N, device=dev)
    fids = torch.arange(N, device=dev)
    labels = torch.randn(B, device=dev)

    # 1) anchor model training step (fwd+bwd, fp16 autocast)
    model = QuantMoEModel(cfg).to(dev).train()

    def anchor_step():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x, fids, labels=labels)
        out.loss.float().backward()

    p1 = peak_of(anchor_step)
    print(f"[1] anchor 2-expert train step peak: {gb(p1)}")

    # 2) isolated stock attention [(T*N), B, D] fwd+bwd
    pa = ParallelAttention(cfg).to(dev).train()
    xs = torch.randn(T * N, B, int(cfg.d_model), device=dev, requires_grad=True)

    def stock_attn_step():
        pa.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = pa(xs)  # need_weights=False path
        out.float().sum().backward()

    p2 = peak_of(stock_attn_step)
    print(f"[2] isolated stock-attention [(T*N)={T*N}, B={B}, D={int(cfg.d_model)}] fwd+bwd peak: {gb(p2)}")

    # which SDPA backend serves the stock shape (fp16, no mask)?
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        q = torch.randn(T * N, int(cfg.n_heads), B, int(cfg.d_model) // int(cfg.n_heads),
                        device=dev, dtype=torch.float16)
        for name, bk in [("FLASH", SDPBackend.FLASH_ATTENTION),
                         ("MEM_EFF", SDPBackend.EFFICIENT_ATTENTION),
                         ("MATH", SDPBackend.MATH)]:
            try:
                with sdpa_kernel([bk]):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
                print(f"    SDPA backend {name}: OK")
            except Exception as e:
                print(f"    SDPA backend {name}: unavailable ({type(e).__name__})")
    except Exception as e:
        print(f"    (sdpa_kernel probe skipped: {e})")

    # 3) anchor + simulated 3rd expert (upper-ish bound for the 3-expert peak)
    pa2 = ParallelAttention(cfg).to(dev).train()

    def combo_step():
        model.zero_grad(set_to_none=True)
        pa2.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x, fids, labels=labels)
            h = torch.randn(B, T, N, int(cfg.d_model), device=dev, dtype=torch.float16, requires_grad=True)
            hs = h.permute(1, 2, 0, 3).reshape(T * N, B, int(cfg.d_model))
            extra = pa2(hs)
            loss = out.loss.float() + extra.float().mean()
        loss.backward()

    p3 = peak_of(combo_step)
    print(f"[3] anchor + simulated stock expert (x2 layers ~= +{gb(2*(p2 - 0))} naive) combined peak: {gb(p3)}")

    free_now = total - torch.cuda.memory_reserved()
    print(f"\nsummary: anchor={gb(p1)}, +sim-3rd-expert={gb(p3)} (delta {gb(max(0, p3 - p1))}); "
          f"card total={gb(total)}")
    print("NOTE: real 3-expert adds ONE stock attention per layer (n_layers=2); the [3] sim adds one "
          "global — scale delta x2 for a conservative bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
