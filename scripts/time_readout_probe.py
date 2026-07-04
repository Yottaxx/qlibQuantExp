#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-1 TIME-AXIS readout probe (frozen-backbone ceiling for the temporal readout).

The factor POOL at readout is settled-dead (pool-readout-gate-20260606: mean is ~IC-optimal),
and the model is time-dominated. This probe asks the complementary question on FROZEN loaded
backbones: is the last-step temporal readout `h[:, -1]` (under bidirectional ALiBi) leaving
signal on the table that a better time-aggregation would recover?

Seam: forward_hook on `net.final_norm` captures the FULL post-block hidden h [B, T, N, D]
(quant_moe_model.py:398, just before the `h[:, -1]` slice at :402). Pool over N by mean
(factor pool is dead -> mean is the right factor collapse) -> h_time [B, T, D]. Then compare
temporal-aggregation arms on the frozen reps (ridge / hold-selected Adam, daily_rank_ic, +0.003
gate vs the last-step arm). KEEPS both MoE experts (the factor EXPERT's in-block decorrelation
is realized in these reps); only the READOUT time-aggregation varies.

Arms: T0 last-step (parity = current model readout) | T-mean | T-attn (learnable query over T,
hold-selected) | T-flatten (concat all T) | T-each (per-timestep marginal IC).

Run (quantEnv), loaded backbone (no training):
  python scripts/time_readout_probe.py --seed 42 --label full_135 \
    --load-model mlruns/325032212672679181/26e088e3282a4170a59a53f0fa8c8319/artifacts/model \
    --out-dir analysis/time_readout_probe
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-1 time-axis readout probe (frozen backbone).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    ap.add_argument("--load-model", required=True, help="qlib-saved model artifact (mlruns/.../artifacts/model)")
    ap.add_argument("--out-dir", default="analysis/time_readout_probe")
    ap.add_argument("--train-stride", type=int, default=16)
    ap.add_argument("--attn-steps", type=int, default=600)
    return ap.parse_args()


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    a_sorted = a[order]
    i, n = 0, len(a)
    while i < n:
        j = i + 1
        while j < n and a_sorted[j] == a_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _spearman(p: np.ndarray, y: np.ndarray) -> float:
    if p.size < 3 or np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    rp = _rankdata(p) - (len(p) + 1) / 2.0
    ry = _rankdata(y) - (len(y) + 1) / 2.0
    denom = math.sqrt(float((rp * rp).sum()) * float((ry * ry).sum()))
    return float((rp * ry).sum() / denom) if denom > 1e-12 else float("nan")


def daily_rank_ic(pred: np.ndarray, label: np.ndarray, day_id: np.ndarray) -> Tuple[float, int]:
    out: List[float] = []
    for d in np.unique(day_id):
        m = day_id == d
        if m.sum() < 3:
            continue
        yv, pv = label[m], pred[m]
        fin = np.isfinite(yv) & np.isfinite(pv)
        if fin.sum() < 3:
            continue
        ic = _spearman(pv[fin], yv[fin])
        if np.isfinite(ic):
            out.append(ic)
    return (float(np.mean(out)), len(out)) if out else (float("nan"), 0)


def main() -> int:
    args = _parse_args()
    seed = int(args.seed)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["QIB_RUN_SETTING"] = f"timeprobe_seed{seed}"
    os.environ["QIB_TRAINER_OVERRIDES_JSON"] = json.dumps({"seed": seed, "use_tqdm": False})

    import torch
    import pandas as pd
    _repo = str(Path(__file__).resolve().parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    import work_flow as wf
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out_root = Path(args.out_dir) / f"seed{seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(m: str) -> None:
        print(m, flush=True)
        log_lines.append(m)

    log(f"[setup] seed={seed} label={args.label} out={out_root}")
    dataset = init_instance_by_config(wf.data_conf)

    import pickle
    with open(args.load_model, "rb") as f:
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
    log(f"[load] backbone from {args.load_model} (device={getattr(model,'device','?')})")
    model._ensure_market_state()
    net = model.net
    net.eval()
    device = model.device
    num_alphas = int(model._get_num_alphas())
    f_ids = torch.arange(num_alphas, device=device)
    eps = 1e-12

    # ----- hook final_norm to capture full post-block hidden h [B, T, N, D] -----
    cap: Dict[str, torch.Tensor] = {}

    def _hook(_m, _inp, out):
        cap["h"] = out.detach()

    handle = net.final_norm.register_forward_hook(_hook)

    def _lnpool(v: torch.Tensor) -> torch.Tensor:  # per-sample LN over last dim
        mu_ = v.mean(dim=-1, keepdim=True)
        sd_ = v.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (v - mu_) / sd_

    def iter_segment(segment: str, data_key: str, stride: int = 1):
        tsds = dataset.prepare(segment, col_set=["feature", "label"], data_key=data_key)
        loader = model._make_daily_chunk_loader(tsds, with_label=True)
        di = 0
        with torch.no_grad():
            for batch in loader:
                if not (isinstance(batch, (tuple, list)) and len(batch) == 4):
                    continue
                di += 1
                if stride > 1 and (di % stride != 0):
                    continue
                bx, by, bmacro, day = batch
                bx = model._coerce_bx_feature_dim(bx, expected_dim=num_alphas, context="timeprobe")
                bx_t = torch.nan_to_num(bx, 0.0).to(device).float()
                macro_t = None if bmacro is None else torch.nan_to_num(bmacro, 0.0).to(device).float()
                out = net(bx_t, f_ids, labels=None, macro_features=macro_t)
                h_full = cap.get("h")                       # [B,T,N,D]
                h_time = h_full.mean(dim=2)                 # mean over N -> [B,T,D]  (factor pool = dead -> mean)
                yv = None if by is None else by.view(-1).float().cpu().numpy()
                sc = out.scores.view(-1).float().cpu().numpy()
                yield h_time, yv, sc, pd.to_datetime(day).normalize()

    T = int(getattr(net.config, "context_len", 8))
    D = int(getattr(net.config, "d_model", 64))
    results: Dict = {"seed": seed, "config_label": args.label, "universe": "csi300", "T": T, "D": D,
                     "loaded_from": args.load_model}

    # ----- valid sanity + cache h_time -----
    log("[pass] cache valid h_time + model sanity ...")
    val_h: List[torch.Tensor] = []
    val_y: List[np.ndarray] = []
    val_d: List[np.ndarray] = []
    val_s: List[np.ndarray] = []
    for h_time, yv, sc, day in iter_segment("valid", DataHandlerLP.DK_I, stride=1):
        if h_time is None:
            continue
        B = h_time.shape[0]
        val_h.append(h_time.half().cpu())
        val_y.append(yv if yv is not None else np.full(B, np.nan))
        val_s.append(sc)
        val_d.append(np.full(B, np.datetime64(day, "D")))
    vY = np.concatenate(val_y); vS = np.concatenate(val_s); vDay = np.concatenate(val_d)
    model_ric, n_days = daily_rank_ic(vS, vY, vDay)
    results["sanity_valid_rank_ic"] = model_ric
    results["valid_n_days"] = n_days
    log(f"[sanity] model valid daily_rank_ic={model_ric:.6f} over {n_days} days")

    # ----- cache train (subsampled) -----
    log("[pass] cache train h_time ...")
    Xtr_list, ytr_list, dtr_list = [], [], []
    for h_time, yv, sc, day in iter_segment("train", DataHandlerLP.DK_L, stride=int(args.train_stride)):
        if h_time is None or yv is None:
            continue
        m = np.isfinite(yv)
        if m.sum() < 3:
            continue
        mt = torch.from_numpy(m).to(h_time.device)
        Xtr_list.append(h_time[mt].half().cpu())
        ytr_list.append(yv[m].astype(np.float32))
        dtr_list.append(np.full(int(m.sum()), np.datetime64(day, "D")))
    Xtr = torch.cat(Xtr_list).to(device).float()            # [M,T,D]
    ytr = np.concatenate(ytr_list); dtr = np.concatenate(dtr_list)
    ytr_t = torch.from_numpy(ytr).to(device).float()
    M = Xtr.shape[0]
    log(f"[probe] train cache: {M} samples over {len(np.unique(dtr))} days")

    # day-blocked CV split (searchsorted — robust to datetime64)
    uniq = np.unique(dtr)
    rng = np.random.RandomState(12345)
    perm = rng.permutation(len(uniq))
    n_hold = max(1, int(0.2 * len(uniq)))
    flag = np.zeros(len(uniq), dtype=bool); flag[perm[:n_hold]] = True
    is_hold = flag[np.searchsorted(uniq, dtr)]
    fit_mask = torch.from_numpy(~is_hold).to(device)
    hold_mask = torch.from_numpy(is_hold).to(device)
    dhold = dtr[is_hold]; yhold_np = ytr_t[hold_mask].cpu().numpy()

    def _ridge(X, y, lam):
        d = X.shape[1]
        return torch.linalg.solve(X.t() @ X + lam * torch.eye(d, device=X.device), X.t() @ y)

    def _fit_cv(feat):
        muf = feat[fit_mask].mean(0, keepdim=True); sdf = feat[fit_mask].std(0, keepdim=True).clamp_min(1e-6)
        fs = (feat - muf) / sdf
        Xf, yf = fs[fit_mask], ytr_t[fit_mask]
        Xh = fs[hold_mask]
        base = float((Xf.t() @ Xf).diagonal().mean().item())
        best = (-1e9, None)
        for mult in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            w = _ridge(Xf, yf, base * mult + 1e-6)
            ric, _ = daily_rank_ic((Xh @ w).cpu().numpy(), yhold_np, dhold)
            if np.isfinite(ric) and ric > best[0]:
                best = (ric, base * mult + 1e-6)
        w_full = _ridge(fs, ytr_t, best[1])
        return muf, sdf, w_full

    # feature builders on [.,T,D] -> feature tensor
    def f_last(X):   return _lnpool(X[:, -1, :])
    def f_mean(X):   return _lnpool(X.mean(dim=1))
    def f_flat(X):   return torch.cat([_lnpool(X[:, t, :]) for t in range(T)], dim=1)

    arms = {"T0_last": f_last, "T_mean": f_mean, "T_flatten": f_flat}
    fitted = {name: _fit_cv(fn(Xtr)) for name, fn in arms.items()}
    log("[probe] T0/T-mean/T-flatten fit done")

    # T-attn: learnable query over T (hold-selected), warm uniform
    wd = 1e-4
    fit_idx = torch.from_numpy(np.asarray(~is_hold).nonzero()[0]).to(device)
    hold_idx = torch.from_numpy(np.asarray(is_hold).nonzero()[0]).to(device)
    Xfit, yfit = Xtr[fit_idx].contiguous(), ytr_t[fit_idx]
    Xhold = Xtr[hold_idx].contiguous()
    # warm head from T0_last fit (rank-equivalent init)
    mu0, sd0, w0 = fitted["T0_last"]
    q = torch.zeros(D, device=device, requires_grad=True)
    head = (w0.detach() / sd0.squeeze(0)).clone().requires_grad_(True)

    def _attn_feat(X, q_):
        logits = torch.einsum("mtd,d->mt", X, q_)           # [m,T]
        a = torch.softmax(logits, dim=1)
        pooled = torch.einsum("mt,mtd->md", a, X)           # [m,D]
        return _lnpool(pooled)
    optK = torch.optim.Adam([q, head], lr=5e-3)

    def _hr():
        with torch.no_grad():
            ph = (_attn_feat(Xhold, q) @ head).cpu().numpy()
        return daily_rank_ic(ph, yhold_np, dhold)[0]
    best = (_hr(), q.detach().clone(), head.detach().clone())
    for step in range(int(args.attn_steps)):
        optK.zero_grad()
        pred = _attn_feat(Xfit, q) @ head
        loss = ((pred - yfit) ** 2).mean() + wd * (head * head).sum()
        loss.backward(); optK.step()
        if step % 20 == 0 or step == int(args.attn_steps) - 1:
            r = _hr()
            if np.isfinite(r) and r > best[0]:
                best = (r, q.detach().clone(), head.detach().clone())
    q_b, head_b = best[1], best[2]
    with torch.no_grad():
        lg = torch.einsum("mtd,d->mt", Xhold, q_b)
        attn_w = torch.softmax(lg, dim=1).mean(dim=0)        # avg attention over T
        cs_std = float(torch.softmax(lg, dim=1).std(dim=0).mean().item())
    log(f"[probe] T-attn hold_ric={best[0]:.5f} attn_over_T={[round(x,3) for x in attn_w.cpu().numpy().tolist()]} cross_stock_std={cs_std:.4f}")

    # T-each: per-timestep marginal ridge
    per_t = []
    for t in range(T):
        mu_t, sd_t, w_t = _fit_cv(_lnpool(Xtr[:, t, :]))
        per_t.append((mu_t, sd_t, w_t))

    # ----- eval on valid -----
    log("[pass] eval temporal arms on valid ...")
    preds = {name: [] for name in list(arms.keys()) + ["T_attn"]}
    per_t_pred = [[] for _ in range(T)]
    yv_all, dv_all = [], []
    with torch.no_grad():
        for hh, yv, dday in zip(val_h, val_y, val_d):
            X = hh.to(device).float()                        # [B,T,D]
            for name, fn in arms.items():
                muf, sdf, w = fitted[name]
                preds[name].append((((fn(X) - muf) / sdf) @ w).cpu().numpy())
            preds["T_attn"].append((_attn_feat(X, q_b) @ head_b).cpu().numpy())
            for t in range(T):
                mu_t, sd_t, w_t = per_t[t]
                per_t_pred[t].append((((_lnpool(X[:, t, :]) - mu_t) / sd_t) @ w_t).cpu().numpy())
            yv_all.append(yv); dv_all.append(np.full(X.shape[0], dday[0]))
    yvv = np.concatenate(yv_all); dvv = np.concatenate(dv_all)
    ric = {name: daily_rank_ic(np.concatenate(p), yvv, dvv)[0] for name, p in preds.items()}
    per_t_ric = [daily_rank_ic(np.concatenate(per_t_pred[t]), yvv, dvv)[0] for t in range(T)]

    base = ric["T0_last"]
    GATE = 0.003
    probe = {
        "T0_last_rank_ic": ric["T0_last"],
        "T_mean_rank_ic": ric["T_mean"], "T_mean_gap": float(ric["T_mean"] - base),
        "T_flatten_rank_ic": ric["T_flatten"], "T_flatten_gap": float(ric["T_flatten"] - base),
        "T_attn_rank_ic": ric["T_attn"], "T_attn_gap": float(ric["T_attn"] - base),
        "T_attn_avg_weights": [float(x) for x in attn_w.cpu().numpy().tolist()],
        "T_attn_cross_stock_std": cs_std,
        "per_timestep_marginal_rank_ic": [float(x) for x in per_t_ric],
        "argmax_informative_timestep": int(np.nanargmax(per_t_ric)),
    }
    probe["gate_T_mean"] = bool(probe["T_mean_gap"] >= GATE)
    probe["gate_T_flatten"] = bool(probe["T_flatten_gap"] >= GATE)
    probe["gate_T_attn"] = bool(probe["T_attn_gap"] >= GATE)
    probe["any_time_readout_helps"] = bool(probe["gate_T_mean"] or probe["gate_T_flatten"] or probe["gate_T_attn"])
    results["TIME_PROBE"] = probe
    log(f"[TIME-PROBE] T0_last={base:.5f} | T_mean gap={probe['T_mean_gap']:+.5f} | "
        f"T_flatten gap={probe['T_flatten_gap']:+.5f} | T_attn gap={probe['T_attn_gap']:+.5f} | "
        f"best_t={probe['argmax_informative_timestep']} (last=T-1={T-1}) | any_helps={probe['any_time_readout_helps']}")
    log(f"[TIME-PROBE] per-timestep marginal rank_ic: {[round(x,4) for x in per_t_ric]}")

    handle.remove()
    (out_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_root / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"[done] wrote {out_root/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
