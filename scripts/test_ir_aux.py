# -*- coding: utf-8 -*-
"""Unit test for the Portfolio-IR auxiliary loss (h-20260627-001, L-6).
Verifies the pre-registered leakage invariants + default-off byte-identity.
Run: quantEnv python scripts/test_ir_aux.py
"""
import sys
sys.path.insert(0, r"C:\Users\60585\PycharmProjects\qibMacV2")
import torch
from module.utils.model_configuration import QuantMoEConfig
from module.quant_moe_model import QuantMoEModel


def make(lam, seed=1):
    cfg = QuantMoEConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, num_alphas=8,
                         context_len=4, use_external_macro=False, dropout=0.0,
                         main_loss="mse", mse_normalize=False,
                         ir_aux_lambda=lam, ir_aux_ramp_steps=1, ir_aux_ema_decay=0.9)
    torch.manual_seed(seed)            # identical param init regardless of lam (ir adds no Parameters)
    return QuantMoEModel(cfg)


torch.manual_seed(7)
B, T, N = 128, 4, 8
x = torch.randn(B, T, N); fids = torch.arange(N); y = torch.randn(B)

res = []
def check(name, cond):
    cond = bool(cond); res.append((name, cond))
    print(("PASS" if cond else "FAIL"), "-", name)

# 1) EVAL-mode byte-identity: ir is NOT added in eval => lam must be irrelevant; buffers FROZEN (the key guard)
m_off, m_on = make(0.0), make(0.08)
m_off.eval(); m_on.eval()
with torch.no_grad():
    l_off_e = m_off(x, fids, labels=y).loss.item()
    l_on_e = m_on(x, fids, labels=y).loss.item()
check("eval loss identical off-vs-on (ir frozen + NOT added in eval)", abs(l_off_e - l_on_e) < 1e-7)
check("ir_step frozen in eval", int(m_on.ir_step.item()) == 0)
check("ir_ema_r frozen in eval", abs(float(m_on.ir_ema_r.item())) < 1e-12)

# 2) TRAIN-mode: ir is active (added to loss, buffers update, gradient flows through p)
m_off2, m_on2 = make(0.0), make(0.08)
m_off2.train(); m_on2.train()
o_off = m_off2(x, fids, labels=y)
o_on = m_on2(x, fids, labels=y)
check("train loss differs off-vs-on (ir contributes to total_loss)", abs(o_off.loss.item() - o_on.loss.item()) > 1e-6)
check("total_loss is 0-dim scalar (l_ir not broadcasting)", o_on.loss.dim() == 0)
check("ir_step incremented in train", int(m_on2.ir_step.item()) == 1)
check("ir_ema_r updated in train (EMA moved off init)", abs(float(m_on2.ir_ema_r.item())) > 0.0)
o_on.loss.backward()
g = m_on2.head.weight.grad
check("gradient flows to head (finite, nonzero)", g is not None and bool(torch.isfinite(g).all()) and g.abs().sum().item() > 0)

# 3) DEFAULT-OFF (lam=0): buffers never touched even in train => anchor byte-identical path
m_off3 = make(0.0); m_off3.train(); _ = m_off3(x, fids, labels=y)
check("default-off: ir_step stays 0 in train (no-op path)", int(m_off3.ir_step.item()) == 0)

# 4) MF-1 regression (code-review wf_94c0eb30): fp16 autocast + degenerate (all-equal) cross-section
#    must NOT produce NaN nor poison the EMA buffers. Identical input rows -> collapsed score cross-section
#    -> pc=0 -> the OLD fp16 path did 0/0=NaN and the unconditional EMA update poisoned the buffers forever.
x_deg = x[:1].repeat(B, 1, 1)            # identical rows => pc ~ 0 (degenerate book)
if torch.cuda.is_available():
    dev = "cuda"
    m_deg = make(0.08).to(dev); m_deg.train()
    xd, fd, yd = x_deg.to(dev), fids.to(dev), y.to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        o_deg = m_deg(xd, fd, labels=yd)
        o_nxt = m_deg(x.to(dev), fd, labels=yd)   # a normal step right after must still be finite
    check("fp16 degenerate day: total_loss finite (no 0/0 NaN)", torch.isfinite(o_deg.loss))
    check("fp16 degenerate day: ir_ema_r NOT poisoned", bool(torch.isfinite(m_deg.ir_ema_r).all()))
    check("fp16 degenerate day: ir_ema_r2 NOT poisoned", bool(torch.isfinite(m_deg.ir_ema_r2).all()))
    check("fp16 step AFTER degenerate day still finite (buffers not stuck NaN)", torch.isfinite(o_nxt.loss))
else:
    m_deg = make(0.08); m_deg.train()
    o_deg = m_deg(x_deg, fids, labels=y)
    check("cpu degenerate day: total_loss finite", torch.isfinite(o_deg.loss))
    check("cpu degenerate day: ir_ema_r/r2 finite", bool(torch.isfinite(m_deg.ir_ema_r).all() and torch.isfinite(m_deg.ir_ema_r2).all()))

ok = all(c for _, c in res)
print("\n" + ("ALL %d INVARIANTS PASS" % len(res) if ok else "SOME INVARIANTS FAILED"))
sys.exit(0 if ok else 1)
