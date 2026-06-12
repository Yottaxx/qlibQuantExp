#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step-B temporal-readout VERDICT: readout + top-MoE-block continue-finetune (un-confounded).

Loads a tau_scale_05 backbone, unfreezes {last MoE block, final_norm, readout, head}, and
continue-finetunes 8 epochs. Two arms (run separately, bonus = d3cid - control):
  --arm control : last-step readout (default), top-block finetuned  -> isolates "finetuning the top"
  --arm d3cid   : per-channel-D linear-over-T readout, top-block finetuned -> "readout + finetuning"
Identity-start: d3cid tr_W = one-hot-last (linear, no saturation) so step0 ~ the loaded model.
baseline = the loaded model's valid rank_ic (before any finetune). bonus = best_finetuned - baseline.

Run (quantEnv):
  python scripts/time_readout_stepB.py --seed 42 --arm d3cid --epochs 8 \
    --load-model mlruns/867178867749867261/917808d52d9a4a13b36c6e20a4155d55/artifacts/model \
    --out-dir analysis/readout_redesign/stepB
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
from typing import List
import numpy as np


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", choices=["control", "d3cid"], required=True)
    ap.add_argument("--load-model", required=True)
    ap.add_argument("--out-dir", default="analysis/readout_redesign/stepB")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-train-batches", type=int, default=0, help="cap batches/epoch (0=unlimited; >0 for smoke)")
    return ap.parse_args()


def _rankdata(a):
    o = a.argsort(kind="mergesort"); r = np.empty(len(a), float); r[o] = np.arange(1, len(a) + 1, dtype=float)
    s = a[o]; i = 0; n = len(a)
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            r[o[i:j]] = r[o[i:j]].mean()
        i = j
    return r


def _spear(p, y):
    if p.size < 3 or np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    rp = _rankdata(p) - (len(p) + 1) / 2.0; ry = _rankdata(y) - (len(y) + 1) / 2.0
    den = math.sqrt(float((rp * rp).sum()) * float((ry * ry).sum()))
    return float((rp * ry).sum() / den) if den > 1e-12 else float("nan")


def daily_ric(pred, label, day):
    out = []
    for d in np.unique(day):
        m = day == d
        if m.sum() < 3:
            continue
        yv, pv = label[m], pred[m]; f = np.isfinite(yv) & np.isfinite(pv)
        if f.sum() >= 3:
            ic = _spear(pv[f], yv[f])
            if np.isfinite(ic):
                out.append(ic)
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def main():
    a = _args(); seed = int(a.seed)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["QIB_RUN_SETTING"] = f"stepB_{a.arm}_seed{seed}"
    os.environ["QIB_TRAINER_OVERRIDES_JSON"] = json.dumps({"seed": seed, "use_tqdm": False})

    import torch, torch.nn as nn, pandas as pd, pickle
    _repo = str(Path(__file__).resolve().parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    import work_flow as wf
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed); np.random.seed(seed)

    out_root = Path(a.out_dir) / f"{a.arm}_seed{seed}"; out_root.mkdir(parents=True, exist_ok=True)
    logs: List[str] = []

    def log(m):
        print(m, flush=True); logs.append(m)

    log(f"[setup] arm={a.arm} seed={seed} out={out_root}")
    dataset = init_instance_by_config(wf.data_conf)
    with open(a.load_model, "rb") as f:
        model = pickle.load(f)
    tc = getattr(model, "trainer_config", {}) or {}
    for k, v in {"_market_state": None, "market_state_path": tc.get("market_state_path", "data/market_state_csi300.pkl"),
                 "market_state_shift": tc.get("market_state_shift", 0), "market_state_strict": tc.get("market_state_strict", True),
                 "num_workers": tc.get("num_workers", 0), "use_tqdm": False, "random_seed": seed,
                 "train_sampler_mode": tc.get("train_sampler_mode", "sampled_daily"), "batch_size": tc.get("batch_size", 300)}.items():
        if not hasattr(model, k):
            setattr(model, k, v)
    if hasattr(model, "_resolve_device"):
        try:
            model.device = model._resolve_device(getattr(model, "device_request", "auto")); model.net = model.net.to(model.device)
        except Exception:
            pass
    model._ensure_market_state()
    net = model.net; device = model.device
    num_alphas = int(model._get_num_alphas()); f_ids = torch.arange(num_alphas, device=device)
    T = int(getattr(net.config, "context_len", 8)); D = int(getattr(net.config, "d_model", 64))

    # ---- wire the readout (temporal aggregation BEFORE the unchanged factor pool) ----
    if a.arm == "d3cid":
        A = torch.zeros(T, D, device=device); A[-1, :] = 1.0          # one-hot-last -> z==h[:,-1] -> step0==model
        net.tr_A = nn.Parameter(A)
        net.temporal_readout = "d3cid"
    else:
        net.temporal_readout = ""
    # ---- freeze all but {last MoE block, final_norm, factor_pooling, head}(+tr_A for d3cid) ----
    for p in net.parameters():
        p.requires_grad_(False)
    trainable = []
    for mod in [net.layers[-1], net.final_norm, net.factor_pooling, net.head]:
        for p in mod.parameters():
            p.requires_grad_(True); trainable.append(p)
    if a.arm == "d3cid":
        net.tr_A.requires_grad_(True); trainable.append(net.tr_A)
    ntr = sum(p.numel() for p in trainable)
    log(f"[wire] arm={a.arm} trainable params={ntr} (last block + final_norm + factor_pool + head{' + tr_A' if a.arm=='d3cid' else ''})")

    def eval_valid():
        net.eval(); sc, yy, dd = [], [], []
        tsds = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        loader = model._make_daily_chunk_loader(tsds, with_label=True)
        with torch.no_grad():
            for b in loader:
                if not (isinstance(b, (tuple, list)) and len(b) == 4):
                    continue
                bx, by, bmac, day = b
                bx = model._coerce_bx_feature_dim(bx, expected_dim=num_alphas, context="ev")
                bxt = torch.nan_to_num(bx, 0.0).to(device).float()
                mac = None if bmac is None else torch.nan_to_num(bmac, 0.0).to(device).float()
                out = net(bxt, f_ids, labels=None, macro_features=mac)
                sc.append(out.scores.view(-1).float().cpu().numpy())
                yy.append(by.view(-1).float().cpu().numpy())
                dd.append(np.full(out.scores.shape[0], np.datetime64(pd.to_datetime(day).normalize(), "D")))
        return daily_ric(np.concatenate(sc), np.concatenate(yy), np.concatenate(dd))[0]

    base = eval_valid()
    log(f"[baseline] step0 valid rank_ic = {base:.6f} (loaded model; d3cid step0 ~ model via one-hot-last)")

    # ---- continue-finetune ----
    train_tsds = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    opt = torch.optim.Adam(trainable, lr=a.lr)
    history = [{"epoch": 0, "valid_rank_ic": base}]
    best = (base, 0)
    t0 = time.time()
    for ep in range(1, int(a.epochs) + 1):
        net.train()
        if hasattr(model, "_make_daily_loader"):
            loader = model._make_daily_loader(train_tsds, shuffle=True, train=True)
        nb = 0; nskip = 0
        for b in loader:
            if isinstance(b, (tuple, list)) and len(b) == 3:
                bx, by, bmac = b
            elif isinstance(b, (tuple, list)) and len(b) == 2:
                bx, by = b; bmac = None
            else:
                continue
            bx = model._coerce_bx_feature_dim(bx, expected_dim=num_alphas, context="tr")
            bxt = torch.nan_to_num(bx, 0.0).to(device).float()
            byt = None if by is None else by.to(device).float()
            mac = None if bmac is None else torch.nan_to_num(bmac, 0.0).to(device).float()
            out = net(bxt, f_ids, labels=byt, macro_features=mac)
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                nskip += 1; continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0); opt.step()
            nb += 1
            if a.max_train_batches and nb >= a.max_train_batches:
                break
        vric = eval_valid()
        history.append({"epoch": ep, "valid_rank_ic": vric})
        if np.isfinite(vric) and vric > best[0]:
            best = (vric, ep)
        log(f"[ep {ep}] valid_rank_ic={vric:.6f} (batches={nb} skip={nskip}) best={best[0]:.6f}@{best[1]} [{time.time()-t0:.0f}s]")

    res = {"arm": a.arm, "seed": seed, "base": "tau_scale_05", "loaded_from": a.load_model,
           "trainable_params": ntr, "baseline_step0_rank_ic": base,
           "best_valid_rank_ic": best[0], "best_epoch": best[1],
           "bonus_vs_step0": float(best[0] - base), "history": history}
    if a.arm == "d3cid":
        with torch.no_grad():
            wabs = net.tr_A.abs().mean(dim=1)                        # [T] mean |A| per timestep
            res["W_per_t"] = [round(x, 3) for x in wabs.cpu().numpy().tolist()]
            res["W_last_frac"] = float((wabs[-1] / wabs.sum().clamp_min(1e-9)).item())
    (out_root / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_root / "log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    log(f"[done] arm={a.arm} base={base:.5f} best={best[0]:.5f}@{best[1]} bonus={best[0]-base:+.5f} -> {out_root/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
