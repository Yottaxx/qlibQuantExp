#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only verification of the 6 temporal_readout designs (time-readout-bonus-20260607).

For each design we build the model, COPY the shared backbone weights from a control model
(temporal_readout=""), and run a synthetic forward. Then assert:
  - stock_score.shape == [B] and finite; factor_pool_weights is not None; design diag keys emitted.
  - LINEAR designs (d3cid/d3cin/d3mix): identity-start — stock_score == control within 1e-4
    (one-hot-last / zero-residual init => z == h[:,-1] => exactly the last-step control).
  - ATTENTION designs (d1pma/duala/dualb): finite+shape only (gated blend, g=0.5; not exact identity).
No training, no data, no checkpoints — pure forward on random inputs.
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

B, T, N = 4, 8, 158
BASE = dict(
    context_len=T, num_alphas=N,
    use_regime_time_embedding=True, use_regime_factor_gate=True,
    router_use_layer_summary=True, router_mode="learned",
    time_tau_mlp_out_scale=0.5, use_external_macro=False,
)
DESIGNS = ["d3cid", "d3cin", "d3mix", "d1pma", "duala", "dualb"]
LINEAR = {"d3cid", "d3cin", "d3mix"}
DIAG = {
    "d3cid": ["tr_W_last_frac"],
    "d3cin": ["tr_W_last_frac"],
    "d3mix": ["tr_collapse_last_frac", "tr_timemix_gain", "tr_chanmix_gain"],
    "d1pma": ["tr_gate_g", "tr_attn_last_frac", "tr_attn_entropy"],
    "duala": ["tr_gate_g", "tr_head2_zT_norm", "tr_head2_zF_norm"],
    "dualb": ["tr_gate_g", "tr_head2_zT_norm", "tr_head2_zF_norm"],
}


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(B, T, N)
    fids = torch.arange(N)

    torch.manual_seed(0)
    ctrl = QuantMoEModel(QuantMoEConfig(**BASE, temporal_readout="")).eval()
    with torch.no_grad():
        s_ctrl = ctrl(x, fids).logits.float()
    csd = ctrl.state_dict()

    fail = 0
    for d in DESIGNS:
        m = QuantMoEModel(QuantMoEConfig(**BASE, temporal_readout=d)).eval()
        msd = m.state_dict()
        shared = {k: v for k, v in csd.items() if k in msd and v.shape == msd[k].shape}
        m.load_state_dict({**msd, **shared}, strict=False)  # backbone<-ctrl; tr_* keep identity-start
        with torch.no_grad():
            out = m(x, fids)
        s = out.logits.float()
        mt = out.metrics or {}
        ok = True
        if tuple(s.shape) != (B,):
            ok = False; print(f"[{d}] SHAPE FAIL got {tuple(s.shape)}")
        if out.factor_pool_weights is None:
            ok = False; print(f"[{d}] factor_pool_weights is None FAIL")
        if not torch.isfinite(s).all():
            ok = False; print(f"[{d}] non-finite score FAIL")
        miss = [k for k in DIAG[d] if k not in mt]
        if miss:
            ok = False; print(f"[{d}] missing diag {miss} FAIL")
        dshow = {k: round(float(mt.get(k, float("nan"))), 4) for k in DIAG[d]}
        if d in LINEAR:
            diff = (s - s_ctrl).abs().max().item()
            idok = diff < 1e-4
            ok = ok and idok
            print(f"[{d:5s}] identity-start max|Δ vs control|={diff:.2e} {'OK' if idok else 'FAIL'} | diag={dshow}")
        else:
            # Attention designs: assert the identity-start invariants the spec calls out as easy to
            # get wrong — gate g==0.5 (NO +20 bias; tr_gate=0) and, for duals, the zF-half of tr_head2
            # zeroed (score starts from the time branch only). A regression here must FAIL, not pass.
            g = float(mt.get("tr_gate_g", float("nan")))
            gok = abs(g - 0.5) < 1e-3
            extra = ""
            if d in ("duala", "dualb"):
                dm = int(m.config.d_model)
                zf = float(m.tr_head2.weight.detach()[:, dm:].abs().max().item())
                zfok = zf < 1e-6
                gok = gok and zfok
                extra = f", zF-half|w|max={zf:.0e} {'OK' if zfok else 'FAIL'}"
            ok = ok and gok
            print(f"[{d:5s}] finite+shape, g={g:.4f} {'OK' if abs(g - 0.5) < 1e-3 else 'FAIL(g!=0.5)'}{extra} | diag={dshow}")
        if not ok:
            fail += 1

    # ---- init-ablation checks (task #14): uniform_mean linears + d1pma gate sweep ----
    print("\n--- init-ablation (uniform_mean linears + d1pma gate sweep) ---")
    LIN_PROFILE = {"d3cid": "tr_W_last_frac", "d3cin": "tr_W_last_frac", "d3mix": "tr_collapse_last_frac"}
    for d in ["d3cid", "d3cin", "d3mix"]:
        # Build onehot & uniform with the SAME global seed; backbone (non-tr_ params) must be IDENTICAL
        # (uniform noise uses a separate RNG generator) => the A/B isolates only the readout init.
        torch.manual_seed(0)
        m_oh = QuantMoEModel(QuantMoEConfig(**BASE, temporal_readout=d, temporal_readout_init="onehot_last")).eval()
        torch.manual_seed(0)
        m_un = QuantMoEModel(QuantMoEConfig(**BASE, temporal_readout=d, temporal_readout_init="uniform_mean")).eval()
        oh_sd, un_sd = m_oh.state_dict(), m_un.state_dict()
        bb_ident = all(torch.equal(oh_sd[k], un_sd[k]) for k in oh_sd if not k.startswith("tr_"))
        # uniform forward (ctrl backbone copied): should NOT nest control, profile ≈ 1/T.
        msd = m_un.state_dict()
        shared = {k: v for k, v in csd.items() if k in msd and v.shape == msd[k].shape}
        m_un.load_state_dict({**msd, **shared}, strict=False)
        with torch.no_grad():
            out = m_un(x, fids)
        s = out.logits.float()
        prof = float((out.metrics or {}).get(LIN_PROFILE[d], float("nan")))
        diff_ctrl = (s - s_ctrl).abs().max().item()
        not_ctrl = diff_ctrl > 1e-3
        prof_ok = abs(prof - 1.0 / T) < 0.05
        fin_ok = bool(torch.isfinite(s).all())
        good = bb_ident and not_ctrl and prof_ok and fin_ok
        if not good:
            fail += 1
        print(f"[{d:5s} uniform] backbone==onehot:{bb_ident} | NOT==control(Δ={diff_ctrl:.2e}):{not_ctrl} "
              f"| {LIN_PROFILE[d]}={prof:.3f}(≈{1.0/T:.3f}):{prof_ok} | finite:{fin_ok} -> {'OK' if good else 'FAIL'}")
    for gi, gexp in [(-2.0, 0.119), (2.0, 0.881)]:
        m = QuantMoEModel(QuantMoEConfig(**BASE, temporal_readout="d1pma", temporal_readout_gate_init=gi)).eval()
        msd = m.state_dict()
        shared = {k: v for k, v in csd.items() if k in msd and v.shape == msd[k].shape}
        m.load_state_dict({**msd, **shared}, strict=False)
        with torch.no_grad():
            out = m(x, fids)
        g = float((out.metrics or {}).get("tr_gate_g", float("nan")))
        gok = abs(g - gexp) < 0.01 and bool(torch.isfinite(out.logits).all())
        if not gok:
            fail += 1
        print(f"[d1pma gate_init={gi:+.0f}] g={g:.4f} (expect {gexp:.3f}) -> {'OK' if gok else 'FAIL'}")

    print(f"\n{'ALL PASS' if fail == 0 else str(fail) + ' CHECK(S) FAILED'} "
          f"(onehot identity <1e-4; uniform starts off-control at ~1/T w/ identical backbone; gate sweep g≈0.12/0.88)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
