#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step-A temporal-readout SCREEN (frozen backbone, cached reps, trained readout, 8 epochs).

Loads a faithful tau_scale_05 backbone, FREEZES it, caches the post-block reps via a hook on
net.final_norm, collapses factors by mean-over-N (proven optimal), and TRAINS several temporal
readout designs (8 epochs each) on the cached reps to ask: does a learned-over-T readout beat the
last-step readout? Confound-aware LOWER BOUND (backbone not reshaped) — Step-A screen only.

Identity-start CORRECTION (do NOT saturate softmax): softmax readouts use a GATED-RESIDUAL
  pooled = (1-g)*h_time[:,-1] + g*attn_over_T(h_time)   ; g=sigmoid(gate), gate init 0 (g=0.5, responsive)
so the attention is responsive (b_t=0 uniform, can concentrate OR spread) and last-step is always
nested (g->0). Linear readout (D3-CI/D) keeps a one-hot-last linear init (no saturation).

Runs ALL designs in ONE process per backbone (shared cache). Designs:
  baseline (last-step) | d1pma | duala | dualb | d3cid   (d3cin needs full [B,T,N,D] -> deferred)

Run (quantEnv):
  python scripts/time_readout_finetune.py --seed 42 \
    --load-model mlruns/867178867749867261/917808d52d9a4a13b36c6e20a4155d55/artifacts/model \
    --out-dir analysis/readout_redesign/stepA
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-model", required=True)
    ap.add_argument("--out-dir", default="analysis/readout_redesign/stepA")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train-stride", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8192)
    return ap.parse_args()


def _rankdata(a):
    order = a.argsort(kind="mergesort"); r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1, dtype=float)
    s = a[order]; i = 0; n = len(a)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            r[order[i:j]] = r[order[i:j]].mean()
        i = j
    return r


def _spear(p, y):
    if p.size < 3 or np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    rp = _rankdata(p) - (len(p) + 1) / 2.0
    ry = _rankdata(y) - (len(y) + 1) / 2.0
    den = math.sqrt(float((rp * rp).sum()) * float((ry * ry).sum()))
    return float((rp * ry).sum() / den) if den > 1e-12 else float("nan")


def daily_ric(pred, label, day):
    out = []
    for d in np.unique(day):
        m = day == d
        if m.sum() < 3:
            continue
        yv, pv = label[m], pred[m]
        f = np.isfinite(yv) & np.isfinite(pv)
        if f.sum() >= 3:
            ic = _spear(pv[f], yv[f])
            if np.isfinite(ic):
                out.append(ic)
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def main():
    a = _args(); seed = int(a.seed)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["QIB_RUN_SETTING"] = f"timereadout_seed{seed}"
    os.environ["QIB_TRAINER_OVERRIDES_JSON"] = json.dumps({"seed": seed, "use_tqdm": False})

    import torch, torch.nn as nn, pandas as pd, pickle
    _repo = str(Path(__file__).resolve().parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    import work_flow as wf
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed); np.random.seed(seed)

    out_root = Path(a.out_dir) / f"seed{seed}"; out_root.mkdir(parents=True, exist_ok=True)
    logs: List[str] = []

    def log(m):
        print(m, flush=True); logs.append(m)

    log(f"[setup] seed={seed} out={out_root}")
    dataset = init_instance_by_config(wf.data_conf)
    with open(a.load_model, "rb") as f:
        model = pickle.load(f)
    tc = getattr(model, "trainer_config", {}) or {}
    for k, v in {"_market_state": None, "market_state_path": tc.get("market_state_path", "data/market_state_csi300.pkl"),
                 "market_state_shift": tc.get("market_state_shift", 0), "market_state_strict": tc.get("market_state_strict", True),
                 "num_workers": tc.get("num_workers", 0), "use_tqdm": False}.items():
        if not hasattr(model, k):
            setattr(model, k, v)
    if hasattr(model, "_resolve_device"):
        try:
            model.device = model._resolve_device(getattr(model, "device_request", "auto"))
            model.net = model.net.to(model.device)
        except Exception:
            pass
    model._ensure_market_state()
    net = model.net.eval()
    for p in net.parameters():
        p.requires_grad_(False)            # FREEZE backbone
    device = model.device
    num_alphas = int(model._get_num_alphas())
    f_ids = torch.arange(num_alphas, device=device)
    T = int(getattr(net.config, "context_len", 8)); D = int(getattr(net.config, "d_model", 64))
    log(f"[load] backbone (frozen) T={T} D={D} device={device}")

    cap: Dict[str, torch.Tensor] = {}
    h = net.final_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o.detach()))

    def iterseg(segment, dk, stride=1):
        tsds = dataset.prepare(segment, col_set=["feature", "label"], data_key=dk)
        loader = model._make_daily_chunk_loader(tsds, with_label=True)
        di = 0
        with torch.no_grad():
            for b in loader:
                if not (isinstance(b, (tuple, list)) and len(b) == 4):
                    continue
                di += 1
                if stride > 1 and (di % stride != 0):
                    continue
                bx, by, bmac, day = b
                bx = model._coerce_bx_feature_dim(bx, expected_dim=num_alphas, context="tr")
                bxt = torch.nan_to_num(bx, 0.0).to(device).float()
                mac = None if bmac is None else torch.nan_to_num(bmac, 0.0).to(device).float()
                out = net(bxt, f_ids, labels=None, macro_features=mac)
                hh = cap.get("h")                          # [B,T,N,D]
                h_time = hh.mean(dim=2)                     # [B,T,D]
                h_fac = hh.mean(dim=1)                      # [B,N,D]
                yv = None if by is None else by.view(-1).float().cpu().numpy()
                sc = out.scores.view(-1).float().cpu().numpy()
                yield h_time, h_fac, yv, sc, pd.to_datetime(day).normalize()

    # ---- cache valid + train ----
    log("[cache] valid ...")
    vT, vF, vy, vs, vd = [], [], [], [], []
    for ht, hf, yv, sc, day in iterseg("valid", DataHandlerLP.DK_I, 1):
        B = ht.shape[0]
        vT.append(ht.half().cpu()); vF.append(hf.half().cpu())
        vy.append(yv); vs.append(sc); vd.append(np.full(B, np.datetime64(day, "D")))
    VY = np.concatenate(vy); VS = np.concatenate(vs); VD = np.concatenate(vd)
    model_ric, ndv = daily_ric(VS, VY, VD)
    log(f"[sanity] model valid daily_rank_ic={model_ric:.6f} ({ndv} days)")

    log("[cache] train ...")
    tT, tF, ty, td = [], [], [], []
    for ht, hf, yv, sc, day in iterseg("train", DataHandlerLP.DK_L, int(a.train_stride)):
        if yv is None:
            continue
        m = np.isfinite(yv)
        if m.sum() < 3:
            continue
        mt = torch.from_numpy(m).to(ht.device)
        tT.append(ht[mt].half().cpu()); tF.append(hf[mt].half().cpu())
        ty.append(yv[m].astype(np.float32)); td.append(np.full(int(m.sum()), np.datetime64(day, "D")))
    XT = torch.cat(tT).to(device).float(); XF = torch.cat(tF).to(device).float()
    Y = torch.from_numpy(np.concatenate(ty)).to(device).float(); DAY = np.concatenate(td)
    M = XT.shape[0]
    log(f"[cache] train {M} samples / {len(np.unique(DAY))} days ; valid {len(VY)}")
    h.remove()

    # day-blocked CV hold split (searchsorted, robust)
    uniq = np.unique(DAY); rng = np.random.RandomState(123); perm = rng.permutation(len(uniq))
    flag = np.zeros(len(uniq), bool); flag[perm[:max(1, int(0.2 * len(uniq)))]] = True
    is_hold = flag[np.searchsorted(uniq, DAY)]
    fit_idx = torch.from_numpy(np.where(~is_hold)[0]).to(device)
    hold_idx = torch.from_numpy(np.where(is_hold)[0]).to(device)
    Yh_np = Y[hold_idx].cpu().numpy(); DAYh = DAY[is_hold]

    sqrtD = math.sqrt(D)

    # ---------- readout modules (all end at score[B]) ----------
    def attn_over(X, q, b):  # X[B,L,Dd], q[Dd], b[L] -> [B,Dd]
        logits = torch.einsum("bld,d->bl", X, q) / sqrtD + b
        return torch.einsum("bl,bld->bd", torch.softmax(logits, 1), X)

    class Baseline(nn.Module):
        def __init__(s):
            super().__init__(); s.norm = nn.LayerNorm(D); s.head = nn.Linear(D, 1)
        def forward(s, ht, hf):
            return s.head(s.norm(ht[:, -1, :])).squeeze(-1)

    class D1PMA(nn.Module):
        def __init__(s):
            super().__init__()
            s.q = nn.Parameter(torch.randn(D) * 0.02); s.b_t = nn.Parameter(torch.zeros(T))
            s.gate = nn.Parameter(torch.zeros(1)); s.norm = nn.LayerNorm(D); s.head = nn.Linear(D, 1)
        def pool(s, ht):
            attn = attn_over(ht, s.q, s.b_t); g = torch.sigmoid(s.gate)
            return (1 - g) * ht[:, -1, :] + g * attn
        def forward(s, ht, hf):
            return s.head(s.norm(s.pool(ht))).squeeze(-1)

    class DualA(nn.Module):
        def __init__(s):
            super().__init__()
            s.q = nn.Parameter(torch.randn(D) * 0.02); s.b_t = nn.Parameter(torch.zeros(T)); s.gate = nn.Parameter(torch.zeros(1))
            s.nT = nn.LayerNorm(D); s.nF = nn.LayerNorm(D); s.head = nn.Linear(2 * D, 1)
            with torch.no_grad():
                s.head.weight[:, D:] = 0.0
        def forward(s, ht, hf):
            attn = attn_over(ht, s.q, s.b_t); g = torch.sigmoid(s.gate)
            zT = (1 - g) * ht[:, -1, :] + g * attn
            zF = ht.mean(dim=1)                              # global mean over T (= over T,N)
            return s.head(torch.cat([s.nT(zT), s.nF(zF)], 1)).squeeze(-1)

    class DualB(nn.Module):
        def __init__(s):
            super().__init__()
            s.q = nn.Parameter(torch.randn(D) * 0.02); s.b_t = nn.Parameter(torch.zeros(T)); s.gate = nn.Parameter(torch.zeros(1))
            s.qN = nn.Parameter(torch.randn(D) * 0.02); s.bN = nn.Parameter(torch.zeros(num_alphas))
            s.nT = nn.LayerNorm(D); s.nF = nn.LayerNorm(D); s.head = nn.Linear(2 * D, 1)
            with torch.no_grad():
                s.head.weight[:, D:] = 0.0
        def forward(s, ht, hf):
            attn = attn_over(ht, s.q, s.b_t); g = torch.sigmoid(s.gate)
            zT = (1 - g) * ht[:, -1, :] + g * attn
            zF = attn_over(hf, s.qN, s.bN)                   # attention over N on mean_T reps
            return s.head(torch.cat([s.nT(zT), s.nF(zF)], 1)).squeeze(-1)

    class D3CID(nn.Module):
        def __init__(s):
            super().__init__()
            w = torch.zeros(D, T); w[:, -1] = 1.0           # one-hot-last (linear, no saturation)
            s.W = nn.Parameter(w); s.norm = nn.LayerNorm(D); s.head = nn.Linear(D, 1)
        def forward(s, ht, hf):
            pooled = torch.einsum("btd,dt->bd", ht, s.W)
            return s.head(s.norm(pooled)).squeeze(-1)

    designs = {"baseline": Baseline, "d1pma": D1PMA, "duala": DualA, "dualb": DualB, "d3cid": D3CID}

    def train_design(name, Mod):
        mod = Mod().to(device)
        opt = torch.optim.Adam(mod.parameters(), lr=a.lr)
        n_fit = fit_idx.shape[0]
        best = (-1e9, None)
        for ep in range(int(a.epochs)):
            mod.train()
            permf = fit_idx[torch.randperm(n_fit, device=device)]
            for i in range(0, n_fit, a.batch):
                idx = permf[i:i + a.batch]
                opt.zero_grad()
                pred = mod(XT[idx], XF[idx])
                loss = ((pred - Y[idx]) ** 2).mean()
                loss.backward(); opt.step()
            mod.eval()
            with torch.no_grad():
                ph = mod(XT[hold_idx], XF[hold_idx]).cpu().numpy()
            ric, _ = daily_ric(ph, Yh_np, DAYh)
            if np.isfinite(ric) and ric > best[0]:
                best = (ric, {k: v.detach().clone() for k, v in mod.state_dict().items()})
        if best[1] is not None:
            mod.load_state_dict(best[1])
        # eval valid (stream)
        mod.eval(); preds = []
        with torch.no_grad():
            for ht, hf in zip(vT, vF):
                preds.append(mod(ht.to(device).float(), hf.to(device).float()).cpu().numpy())
        vric, _ = daily_ric(np.concatenate(preds), VY, VD)
        diag = {}
        if name == "d1pma":
            diag["gate_g"] = float(torch.sigmoid(mod.gate).item())
            with torch.no_grad():
                lg = torch.einsum("bld,d->bl", XT[hold_idx], mod.q) / sqrtD + mod.b_t
                diag["attn_over_T"] = [round(x, 3) for x in torch.softmax(lg, 1).mean(0).cpu().numpy().tolist()]
        if name in ("duala", "dualb"):
            with torch.no_grad():
                diag["head_zT_norm"] = float(mod.head.weight[0, :D].norm().item())
                diag["head_zF_norm"] = float(mod.head.weight[0, D:].norm().item())
        if name == "d3cid":
            with torch.no_grad():
                wabs = mod.W.abs().mean(dim=0)            # [T] mean |weight| per timestep
                diag["W_per_t"] = [round(x, 3) for x in wabs.cpu().numpy().tolist()]
                diag["W_last_frac"] = float((wabs[-1] / wabs.sum().clamp_min(1e-9)).item())
        return vric, best[0], diag

    results = {"seed": seed, "base": "tau_scale_05", "loaded_from": a.load_model,
               "model_valid_rank_ic": model_ric, "T": T, "D": D, "designs": {}}
    log("[train] designs (8ep, frozen reps) ...")
    base_ric = None
    for name, Mod in designs.items():
        t0 = time.time(); vric, holdric, diag = train_design(name, Mod)
        if name == "baseline":
            base_ric = vric
        results["designs"][name] = {"valid_rank_ic": vric, "hold_rank_ic": holdric, "diag": diag,
                                    "bonus_vs_baseline": (None if base_ric is None else float(vric - base_ric))}
        log(f"  {name:9s} valid_ric={vric:.5f} hold={holdric:.5f} bonus={'' if base_ric is None else f'{vric-base_ric:+.5f}'} {diag} ({time.time()-t0:.0f}s)")
    # recompute bonuses now that baseline known
    b = results["designs"]["baseline"]["valid_rank_ic"]
    for nm, d in results["designs"].items():
        d["bonus_vs_baseline"] = float(d["valid_rank_ic"] - b)
        d["gate_pass"] = bool(d["bonus_vs_baseline"] >= 0.003)
    results["baseline_rank_ic"] = b
    results["any_design_passes"] = bool(any(results["designs"][n]["gate_pass"] for n in designs if n != "baseline"))
    (out_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_root / "log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    log(f"[done] baseline={b:.5f} model={model_ric:.5f} any_pass={results['any_design_passes']} -> {out_root/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
