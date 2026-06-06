#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pool/readout forensics for RST-MoE (rank-1 KEY-SVD-GATE + rank-2 PROBE-CEILING).

Trains one faithful `full_135` backbone (seed S), then runs two inference-only
forensics on the FROZEN backbone's factor-pool input `h_last` ([B, N, D], the
tensor fed into `net.factor_pooling`, captured via a forward_pre_hook):

  KEY-SVD-GATE (rank 1): is the six-nines factor-pool entropy forced by INPUT
    geometry (keys near-constant across the N factor axis -> world B, no operator
    fix can help) or is it an OPERATOR pathology (a better-aligned/sharper query
    could peak -> world A)? Decomposes the model's *actual* attention logits and
    compares to the *best achievable* logit spread of an optimally-aligned unit
    query at the same norm. Cross-checks raw h_last and the value channel V.

  PROBE-CEILING (rank 2): on the frozen reps, does a capacity-matched
    convex-reweight over the N factor axis (Arm A) beat the mean-pool parity
    baseline (Arm0) by >= +0.003 daily rank_ic, and do the 158 per-slot marginal
    ICs disperse (top/bottom-decile >= 2x) with a non-degenerate slot-signal
    covariance? If not -> equal-weight pooling is ~IC-optimal here (world C,
    rational-equilibrium null), and sharpening the pool is forbidden from helping.

Everything runs fp32 (no autocast) so the SVD spectra are not polluted by amp.
Memory-bounded: train features are day-subsampled & cached fp16; valid is streamed.

Run (quantEnv):
  python scripts/pool_readout_forensics.py --seed 42 --out-dir analysis/pool_readout_forensics
  python scripts/pool_readout_forensics.py --smoke   # 1-epoch tiny-window pipeline check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RST-MoE pool/readout forensics.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="analysis/pool_readout_forensics")
    ap.add_argument("--train-stride", type=int, default=16,
                    help="Keep every k-th train day for the probe cache (~180 days).")
    ap.add_argument("--valid-stride", type=int, default=1,
                    help="Keep every k-th valid day (1 = all).")
    ap.add_argument("--svd-max-samples", type=int, default=40000,
                    help="Cap on valid samples used for the per-sample key SVD spectra.")
    ap.add_argument("--arma-steps", type=int, default=600,
                    help="Adam steps for the Arm A convex-reweight joint fit.")
    ap.add_argument("--epochs", type=int, default=16,
                    help="Hard epoch cap for the backbone (promoted best-valid is ~epoch 16).")
    ap.add_argument("--load-model", default="",
                    help="Path to a qlib-saved model artifact (mlruns/.../artifacts/model). "
                         "If set, load this faithful backbone instead of training.")
    ap.add_argument("--label", default="",
                    help="Config label stored in results.json (e.g. full_135, tau_scale_05).")
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch tiny-window pipeline validation.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# small numeric helpers (numpy / torch)
# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (ties -> mean rank), matches scipy.stats.rankdata('average')."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    a_sorted = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and a_sorted[j] == a_sorted[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(p: np.ndarray, y: np.ndarray) -> float:
    if p.size < 3:
        return float("nan")
    if np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    rp = _rankdata(p)
    ry = _rankdata(y)
    rp = rp - rp.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rp * rp).sum()) * float((ry * ry).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((rp * ry).sum() / denom)


def daily_rank_ic(pred: np.ndarray, label: np.ndarray, day_id: np.ndarray) -> Tuple[float, int]:
    """Mean over days of the cross-sectional Spearman(pred, label). Returns (mean, n_days)."""
    out: List[float] = []
    for d in np.unique(day_id):
        m = day_id == d
        if m.sum() < 3:
            continue
        yv = label[m]
        pv = pred[m]
        fin = np.isfinite(yv) & np.isfinite(pv)
        if fin.sum() < 3:
            continue
        ic = _spearman(pv[fin], yv[fin])
        if np.isfinite(ic):
            out.append(ic)
    if not out:
        return float("nan"), 0
    return float(np.mean(out)), len(out)


def main() -> int:
    args = _parse_args()
    seed = int(args.seed)

    # --- env overrides MUST be set before importing work_flow ---
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["QIB_RUN_SETTING"] = f"poolforensic_full135_seed{seed}"
    trainer_over: Dict = {"seed": seed, "use_tqdm": False}
    if args.smoke:
        trainer_over.update({
            "n_epochs": 1, "min_epochs": 1, "consecutive_k": 1,
            "train_stop_threshold": None,
        })
    else:
        # faithful but BOUNDED: restore the best-valid_rank_ic backbone, hard-cap epochs
        # (the promoted full_135 peaks at ~epoch 16; train-loss threshold 1.35 ~never triggers).
        trainer_over.update({
            "checkpoint_metric": "valid_rank_ic",
            "checkpoint_mode": "max",
            "n_epochs": int(args.epochs),
            "min_epochs": int(args.epochs),
            "train_stop_threshold": None,
            "early_stop": 0,
        })
    os.environ["QIB_TRAINER_OVERRIDES_JSON"] = json.dumps(trainer_over)
    if args.smoke:
        os.environ["QIB_DATA_OVERRIDES_JSON"] = json.dumps({
            "kwargs": {
                "handler": {"kwargs": {
                    "start_time": "2018-01-01", "end_time": "2020-10-15",
                    "fit_start_time": "2018-01-01", "fit_end_time": "2019-06-30",
                }},
                "segments": {
                    "train": ["2018-01-01", "2019-06-30"],
                    "valid": ["2020-07-01", "2020-10-15"],
                    "test": ["2020-07-01", "2020-10-15"],
                },
            }
        })

    import torch
    import pandas as pd
    # repo root (parent of scripts/) must be importable for `work_flow`
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    import work_flow as wf
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP

    torch.backends.cuda.matmul.allow_tf32 = False  # clean fp32 for SVD
    torch.backends.cudnn.allow_tf32 = False

    out_root = Path(args.out_dir) / (f"smoke_seed{seed}" if args.smoke else f"seed{seed}")
    out_root.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[setup] seed={seed} smoke={args.smoke} out={out_root}")

    dataset = init_instance_by_config(wf.data_conf)

    if args.load_model:
        # Load a faithful pre-trained backbone (qlib-pickled QlibQuantMoE) — no training.
        import pickle
        t0 = time.time()
        with open(args.load_model, "rb") as f:
            model = pickle.load(f)
        if getattr(model, "net", None) is None:
            raise RuntimeError(f"loaded object has no .net: {args.load_model}")
        # patch lazy/private attributes the CURRENT class methods expect but an older
        # pickle may lack (the wrapper class drifted; the net weights are what matter).
        tc = getattr(model, "trainer_config", {}) or {}
        _lazy_defaults = {
            "_market_state": None,
            "market_state_path": tc.get("market_state_path", "data/market_state_csi300.pkl"),
            "market_state_shift": tc.get("market_state_shift", 0),
            "market_state_strict": tc.get("market_state_strict", True),
            "num_workers": tc.get("num_workers", 0),
            "use_tqdm": False,
        }
        for _k, _v in _lazy_defaults.items():
            if not hasattr(model, _k):
                setattr(model, _k, _v)
        # ensure device + market state are consistent with this session
        if hasattr(model, "_resolve_device"):
            try:
                model.device = model._resolve_device(getattr(model, "device_request", "auto"))
                model.net = model.net.to(model.device)
            except Exception:
                pass
        log(f"[load] loaded backbone from {args.load_model} in {time.time() - t0:.1f}s "
            f"(device={getattr(model, 'device', '?')})")
    else:
        wf._apply_env_overrides()
        model = init_instance_by_config(wf.model_conf)
        t0 = time.time()
        model.fit(dataset)
        log(f"[train] fit done in {time.time() - t0:.1f}s")

    model._ensure_market_state()
    net = model.net
    net.eval()
    device = model.device
    num_alphas = int(model._get_num_alphas())
    f_ids = torch.arange(num_alphas, device=device)
    eps = 1e-12

    # ----- extract pool weights (single head, bias=False) -----
    ap = net.factor_pooling.attention_pool
    inw = ap.mha.in_proj_weight.detach().float()  # [3D, D]
    D = int(inw.shape[1])
    Wq = inw[0:D]
    Wk = inw[D:2 * D]
    Wv = inw[2 * D:3 * D]
    Wo = ap.mha.out_proj.weight.detach().float()  # [D, D]
    q0 = ap.query.detach().float().reshape(1, D)   # learnable query [1, D]
    q_proj = q0 @ Wq.t()                            # projected query [1, D]
    n_pool_heads = int(getattr(ap, "n_heads", 1) or 1)
    d_head = D // n_pool_heads
    scale = 1.0 / math.sqrt(d_head)
    pool_alpha = float(getattr(net.factor_pooling, "alpha", 0.7))
    qnorm = float(q_proj.norm().item())
    log(f"[pool] D={D} n_pool_heads={n_pool_heads} d_head={d_head} scale={scale:.5f} "
        f"alpha={pool_alpha} ||q_proj||={qnorm:.4f}")

    # ----- forward_pre_hook to capture h_last (input to factor_pooling) -----
    cap: Dict[str, torch.Tensor] = {}

    def _pre_hook(_mod, inp):
        cap["h"] = inp[0].detach()

    handle = net.factor_pooling.register_forward_pre_hook(_pre_hook)

    def iter_segment(segment: str, data_key: str, stride: int = 1):
        """Yield (h_last[B,N,D] on device fp32, y[B] np, scores[B] np, day) per day batch."""
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
                bx = model._coerce_bx_feature_dim(bx, expected_dim=num_alphas, context="forensic")
                bx_t = torch.nan_to_num(bx, 0.0).to(device).float()
                macro_t = None if bmacro is None else torch.nan_to_num(bmacro, 0.0).to(device).float()
                out = net(bx_t, f_ids, labels=None, macro_features=macro_t)  # fp32, no autocast
                h = cap.get("h")
                yv = None if by is None else by.view(-1).float().cpu().numpy()
                sc = out.scores.view(-1).float().cpu().numpy()
                yield h, yv, sc, pd.to_datetime(day).normalize()

    results: Dict = {
        "seed": seed, "smoke": bool(args.smoke), "universe": "csi300",
        "config_label": args.label, "loaded_from": args.load_model or None,
        "N": num_alphas, "D": D, "pool_alpha": pool_alpha,
        "n_pool_heads": n_pool_heads, "q_proj_norm": qnorm,
    }

    # =====================================================================
    # PASS 0 — sanity: model valid rank_ic on the fp32 forward
    # =====================================================================
    log("[pass] valid sanity + KEY-SVD-GATE ...")

    def _entropy_norm(w: torch.Tensor, n: int) -> torch.Tensor:
        # w: [B, N] softmax weights -> normalized entropy in [0,1]
        ent = -(w * (w.clamp_min(eps)).log()).sum(dim=1)
        return ent / math.log(max(int(n), 2))

    svd_keys: List[np.ndarray] = []     # per-sample [sigma ratio, partratio, effrank, s1_over_common]
    cur_ent_acc: List[float] = []
    best_ent_acc: List[float] = []
    cur_logit_std_acc: List[float] = []
    best_logit_std_acc: List[float] = []
    s1_over_common_acc: List[float] = []
    ratio_acc: List[float] = []
    effrank_acc: List[float] = []
    part_acc: List[float] = []
    raw_ratio_acc: List[float] = []     # raw h_last across-factor SVD
    val_ratio_acc: List[float] = []     # value channel across-factor SVD
    val_s1_over_common_acc: List[float] = []
    crossstock_w_std_acc: List[float] = []
    model_w_entropy_acc: List[float] = []  # entropy of the *blended* pool weights from output
    # peakability sweep: target logit-std applied to the UNIT-normalized aligned top key-variation
    # direction. Pure shape descriptor (independent of the current cold query). entropy at target
    # logit-std ~2-3 is a normal attention sharpness; if it stays ~1 the contrast is too diffuse.
    TEMP_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]
    best_ent_T_acc: Dict[float, List[float]] = {t: [] for t in TEMP_GRID}
    svd_n = 0

    # valid feature cache for probe eval (kept on device, fp16 to bound memory)
    val_h: List[torch.Tensor] = []
    val_y: List[np.ndarray] = []
    val_score: List[np.ndarray] = []
    val_day: List[np.ndarray] = []

    for h, yv, sc, day in iter_segment("valid", DataHandlerLP.DK_I, stride=int(args.valid_stride)):
        if h is None:
            continue
        B, N, Dd = h.shape
        # --- keys & current attention ---
        K = h @ Wk.t()                                   # [B,N,D]
        Kc = K - K.mean(dim=1, keepdim=True)             # center across factor axis
        # current model logits (q_proj . k_n) * scale
        logits = (K @ q_proj.t()).squeeze(-1) * scale    # [B,N]
        logits_c = logits - logits.mean(dim=1, keepdim=True)
        w_cur = torch.softmax(logits, dim=1)
        cur_ent = _entropy_norm(w_cur, N)
        cur_logit_std = logits_c.std(dim=1)
        # --- SVD of centered keys (subsampled for the spectra) ---
        take = B
        if svd_n + B > int(args.svd_max_samples):
            take = max(0, int(args.svd_max_samples) - svd_n)
        if take > 0:
            Kc_s = Kc[:take]
            try:
                U, S, Vh = torch.linalg.svd(Kc_s, full_matrices=False)  # S [t,k], Vh [t,k,D]
                s1 = S[:, 0].clamp_min(eps)
                s2 = S[:, 1].clamp_min(0.0) if S.shape[1] > 1 else torch.zeros_like(s1)
                ratio = (s2 / s1)
                p2 = S.pow(2)
                part = (S.sum(dim=1).pow(2)) / p2.sum(dim=1).clamp_min(eps)
                pn = p2 / p2.sum(dim=1, keepdim=True).clamp_min(eps)
                effrank = torch.exp(-(pn * pn.clamp_min(eps).log()).sum(dim=1))
                kbar = K[:take].mean(dim=1)              # [t,D] common key
                common_norm = kbar.norm(dim=1).clamp_min(eps)
                s1_over_common = (s1 / math.sqrt(N)) / common_norm
                # best achievable logits: align query DIRECTION with the top key-variation dir
                # (at current ||q_proj||). Then sweep temperature to separate alignment from magnitude.
                u1 = Vh[:, 0, :]                          # [t,D] top right singular vec (D-space)
                logits_best = (Kc_s @ (u1 * qnorm).unsqueeze(-1)).squeeze(-1) * scale  # [t,N]
                logits_best_c = logits_best - logits_best.mean(dim=1, keepdim=True)
                best_logit_std = logits_best_c.std(dim=1)
                best_ent = _entropy_norm(torch.softmax(logits_best, dim=1), N)  # at current ||q||/temperature
                # peakability descriptor: UNIT-normalize the aligned logits (remove the cold-query
                # magnitude confound), then set target logit-std = T. Measures the SHAPE of the top
                # key-variation direction: if entropy stays ~1 even at logit-std 5, the contrast is
                # too diffuse to peak; if it drops, the direction is peakable by some operator.
                logits_unit = logits_best_c / best_logit_std.clamp_min(eps).unsqueeze(1)
                for t in TEMP_GRID:
                    w_t = torch.softmax(logits_unit * t, dim=1)
                    best_ent_T_acc[t].extend(_entropy_norm(w_t, N).detach().cpu().numpy().tolist())
                ratio_acc.extend(ratio.detach().cpu().numpy().tolist())
                effrank_acc.extend(effrank.detach().cpu().numpy().tolist())
                part_acc.extend(part.detach().cpu().numpy().tolist())
                s1_over_common_acc.extend(s1_over_common.detach().cpu().numpy().tolist())
                best_ent_acc.extend(best_ent.detach().cpu().numpy().tolist())
                best_logit_std_acc.extend(best_logit_std.detach().cpu().numpy().tolist())
                # raw h_last across-factor SVD ratio
                Hc = (h[:take] - h[:take].mean(dim=1, keepdim=True))
                Sr = torch.linalg.svdvals(Hc)
                raw_ratio_acc.extend((Sr[:, 1] / Sr[:, 0].clamp_min(eps)).detach().cpu().numpy().tolist())
                # value channel across-factor SVD + magnitude
                V = h[:take] @ Wv.t()
                Vc = V - V.mean(dim=1, keepdim=True)
                Sv = torch.linalg.svdvals(Vc)
                val_ratio_acc.extend((Sv[:, 1] / Sv[:, 0].clamp_min(eps)).detach().cpu().numpy().tolist())
                vbar = V.mean(dim=1).norm(dim=1).clamp_min(eps)
                val_s1_over_common_acc.extend(((Sv[:, 0] / math.sqrt(N)) / vbar).detach().cpu().numpy().tolist())
                svd_n += take
            except Exception as e:  # pragma: no cover
                log(f"[warn] SVD failed on a batch: {e}")
        cur_ent_acc.extend(cur_ent.detach().cpu().numpy().tolist())
        cur_logit_std_acc.extend(cur_logit_std.detach().cpu().numpy().tolist())
        # cross-stock variation of attention weights (per factor std over stocks, mean over factors)
        if B >= 2:
            crossstock_w_std_acc.append(float(w_cur.std(dim=0).mean().item()))
        # cache valid features for probe eval
        val_h.append(h.half().cpu())
        val_y.append(yv if yv is not None else np.full(B, np.nan))
        val_score.append(sc)
        di_day = np.full(B, np.datetime64(day, "D"))
        val_day.append(di_day)

    # sanity: model valid rank_ic
    vY = np.concatenate(val_y) if val_y else np.array([])
    vS = np.concatenate(val_score) if val_score else np.array([])
    vDay = np.concatenate(val_day) if val_day else np.array([])
    model_ric, n_days = daily_rank_ic(vS, vY, vDay)
    results["sanity_valid_rank_ic"] = model_ric
    results["valid_n_days"] = n_days
    log(f"[sanity] model valid daily_rank_ic={model_ric:.6f} over {n_days} days "
        f"(promoted anchor ~0.0768)")

    def _mean(x: List[float]) -> float:
        a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
        return float(a.mean()) if a.size else float("nan")

    def _median(x: List[float]) -> float:
        a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
        return float(np.median(a)) if a.size else float("nan")

    key_svd = {
        "key_sigma2_over_sigma1_mean": _mean(ratio_acc),
        "key_sigma2_over_sigma1_median": _median(ratio_acc),
        "key_effective_rank_mean": _mean(effrank_acc),
        "key_participation_ratio_mean": _mean(part_acc),
        "key_s1_over_common_norm_mean": _mean(s1_over_common_acc),  # world-B indicator (~0 => keys ~constant)
        "current_logit_std_mean": _mean(cur_logit_std_acc),
        "current_softmax_entropy_norm_mean": _mean(cur_ent_acc),    # should reproduce the six-nines
        "best_achievable_logit_std_mean_current_temp": _mean(best_logit_std_acc),
        "best_achievable_entropy_norm_current_temp": _mean(best_ent_acc),   # aligned query, current ||q_proj|| & scale
        "best_achievable_entropy_norm_by_temp": {f"T{t:g}": _mean(best_ent_T_acc[t]) for t in TEMP_GRID},
        "raw_hlast_sigma2_over_sigma1_mean": _mean(raw_ratio_acc),
        "value_sigma2_over_sigma1_mean": _mean(val_ratio_acc),
        "value_s1_over_common_norm_mean": _mean(val_s1_over_common_acc),  # world-C: value bank varies across factors?
        "crossstock_attn_weight_std_mean": _mean(crossstock_w_std_acc),
        "svd_samples_used": int(svd_n),
    }
    # ---- world verdict (structure-first; A vs C deferred to PROBE-CEILING) ----
    #   key_s1_over_common = size of the top across-factor key contrast vs the common key.
    #   If ~0 the keys are near-constant across factors -> NO operator can build a useful contrast
    #   (world B, input-forced). Otherwise the six-nines is an OPERATOR state (cold/near-zero query
    #   x cold 1/8 temperature), and whether un-collapsing HELPS is what PROBE-CEILING settles
    #   (world A = sharpening helps; world C = mean-pool is already ~optimal).
    cur_e = key_svd["current_softmax_entropy_norm_mean"]
    s1c = key_svd["key_s1_over_common_norm_mean"]
    peak3 = key_svd["best_achievable_entropy_norm_by_temp"].get("T3", float("nan"))
    if np.isfinite(s1c) and s1c <= 0.05:
        verdict = "B_input_forced"            # keys ~constant across factors; sharpening is structurally impossible
    elif np.isfinite(s1c) and s1c >= 0.10:
        verdict = "operator_state_collapse"   # real across-factor contrast exists; six-nines is a cold-operator artifact -> A/C via PROBE
    else:
        verdict = "inconclusive_escalate"
    key_svd["world_verdict"] = verdict
    key_svd["peakable_contrast"] = bool(np.isfinite(peak3) and peak3 <= 0.90)
    results["KEY_SVD_GATE"] = key_svd
    by_temp = key_svd["best_achievable_entropy_norm_by_temp"]
    log(f"[KEY-SVD] cur_ent={cur_e:.5f} cur_logit_std={key_svd['current_logit_std_mean']:.4g} "
        f"s1/common={s1c:.4f} peakability_ent[std1,std3]=[{by_temp.get('T1',float('nan')):.4f},"
        f"{by_temp.get('T3',float('nan')):.4f}] -> verdict={verdict}")

    # =====================================================================
    # PROBE-CEILING — cache train, fit Arm0/A/B, eval on valid
    # =====================================================================
    log("[pass] train cache for PROBE-CEILING ...")
    Xtr_list: List[torch.Tensor] = []
    ytr_list: List[np.ndarray] = []
    dtr_list: List[np.ndarray] = []
    for h, yv, sc, day in iter_segment("train", DataHandlerLP.DK_L, stride=int(args.train_stride)):
        if h is None or yv is None:
            continue
        m = np.isfinite(yv)
        if m.sum() < 3:
            continue
        mt = torch.from_numpy(m).to(h.device)
        Xtr_list.append(h[mt].half().cpu())
        ytr_list.append(yv[m].astype(np.float32))
        dtr_list.append(np.full(int(m.sum()), np.datetime64(day, "D")))
    if not Xtr_list:
        log("[error] no train cache collected")
        handle.remove()
        return 2
    Xtr = torch.cat(Xtr_list).to(device).float()        # [M,N,D]
    ytr = np.concatenate(ytr_list)
    dtr = np.concatenate(dtr_list)
    Mtr = Xtr.shape[0]
    log(f"[probe] train cache: {Mtr} samples over {len(np.unique(dtr))} days "
        f"(~{Xtr.element_size()*Xtr.nelement()/1e9:.2f} GB fp32 on device)")
    ytr_t = torch.from_numpy(ytr).to(device).float()

    def _lnpool(v: torch.Tensor) -> torch.Tensor:
        # per-sample LayerNorm (no affine) over the D dims, mirroring the model's final
        # pool LayerNorm so Arm0/ArmA isolate REWEIGHTING from the LN nonlinearity.
        mu_ = v.mean(dim=1, keepdim=True)
        sd_ = v.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (v - mu_) / sd_

    # mean-pool -> per-sample LN -> per-dim standardization (train stats on the LN'd pool)
    Mtr_pool = _lnpool(Xtr.mean(dim=1))                   # [M,D]
    mu = Mtr_pool.mean(dim=0, keepdim=True)
    sd = Mtr_pool.std(dim=0, keepdim=True).clamp_min(1e-6)

    def _standardize(v: torch.Tensor) -> torch.Tensor:
        return (v - mu) / sd

    # honest internal CV split by DAY for lambda selection.
    # NOTE: build is_hold via searchsorted (NOT set-of-.tolist(), which converts datetime64 ->
    # python datetime and silently makes membership never match -> empty hold split -> nan CV).
    uniq_days = np.unique(dtr)                      # sorted
    rng = np.random.RandomState(12345)
    perm = rng.permutation(len(uniq_days))
    n_hold = max(1, int(0.2 * len(uniq_days)))
    hold_day_flag = np.zeros(len(uniq_days), dtype=bool)
    hold_day_flag[perm[:n_hold]] = True
    is_hold = hold_day_flag[np.searchsorted(uniq_days, dtr)]
    assert 0 < int(is_hold.sum()) < len(is_hold), f"bad hold split: {int(is_hold.sum())}/{len(is_hold)}"
    fit_mask = torch.from_numpy(~is_hold).to(device)
    hold_mask = torch.from_numpy(is_hold).to(device)

    def _ridge_fit(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
        # X [n,d], returns w [d]
        d = X.shape[1]
        XtX = X.t() @ X
        Xty = X.t() @ y
        A = XtX + lam * torch.eye(d, device=X.device)
        return torch.linalg.solve(A, Xty)

    def _pick_lambda(Xs: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        Xf, yf = Xs[fit_mask], y[fit_mask]
        Xh, yh = Xs[hold_mask], y[hold_mask]
        dh = dtr[is_hold]
        base = float((Xf.t() @ Xf).diagonal().mean().item())
        best = (-1e9, base * 1e-2, None)
        for mult in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            lam = base * mult + 1e-6
            w = _ridge_fit(Xf, yf, lam)
            ph = (Xh @ w).detach().cpu().numpy()
            ric, _ = daily_rank_ic(ph, yh.detach().cpu().numpy(), dh)
            if np.isfinite(ric) and ric > best[0]:
                best = (ric, lam, w)
        # refit on full train at chosen lambda
        w_full = _ridge_fit(Xs, y, best[1])
        return best[1], w_full

    # ----- Arm 0: parity (mean-pool ridge) -----
    Xs0 = _standardize(Mtr_pool)
    lam0, w0 = _pick_lambda(Xs0, ytr_t)
    log(f"[probe] Arm0 (mean-pool ridge) lambda={lam0:.4g}")

    # ----- Arm B: 158 marginal slot ridges -----
    slot_mu = Xtr.mean(dim=0, keepdim=True)              # [1,N,D]
    slot_sd = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xtr_std = (Xtr - slot_mu) / slot_sd                  # [M,N,D] standardized per (slot,dim)
    base_b = float((Xtr_std[:, 0, :].t() @ Xtr_std[:, 0, :]).diagonal().mean().item())
    lam_b = base_b * 1e-2 + 1e-6
    slot_w = torch.zeros(num_alphas, D, device=device)
    for n in range(num_alphas):
        Xn = Xtr_std[:, n, :]
        slot_w[n] = _ridge_fit(Xn, ytr_t, lam_b)
    log("[probe] Arm B (158 slot ridges) fit done")

    del Xtr_std
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # shared fit / hold splits (by day) reused by every iterative arm
    wd = 1e-4
    fit_idx = torch.from_numpy(np.asarray(~is_hold).nonzero()[0]).to(device)
    hold_idx = torch.from_numpy(np.asarray(is_hold).nonzero()[0]).to(device)
    Xfit = Xtr[fit_idx].contiguous(); yfit = ytr_t[fit_idx]
    Xhold = Xtr[hold_idx].contiguous()
    yhold_np = ytr_t[hold_idx].detach().cpu().numpy()
    dhold = dtr[is_hold]

    def _attn_feat(X: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        # input-conditional attention pool: keys = h_n, k learnable queries -> concat of k LN(pooled) blocks.
        # Q=0 => uniform attention => mean-pool (so the arm's hypothesis space CONTAINS Arm0).
        logits = torch.einsum("mnd,kd->mkn", X, Q)         # [m,k,N]
        a_ = torch.softmax(logits, dim=2)
        pooled = torch.einsum("mkn,mnd->mkd", a_, X)       # [m,k,D]
        return torch.cat([_lnpool(pooled[:, j, :]) for j in range(Q.shape[0])], dim=1)

    # ----- Arm A: STATIC convex reweight (theta in R^N), warm uniform=Arm0, HOLD-SELECTED -----
    theta = torch.zeros(num_alphas, device=device, requires_grad=True)
    headA = w0.detach().clone().requires_grad_(True)
    optA = torch.optim.Adam([theta, headA], lr=5e-3)

    def _holdric_static(th, hd):
        with torch.no_grad():
            pooled = torch.einsum("mnd,n->md", Xhold, torch.softmax(th, dim=0))
            ph = (_standardize(_lnpool(pooled)) @ hd).detach().cpu().numpy()
        r, _ = daily_rank_ic(ph, yhold_np, dhold)
        return r
    bestA = (_holdric_static(theta, headA), theta.detach().clone(), headA.detach().clone())
    nstep = int(args.arma_steps)
    for step in range(nstep):
        optA.zero_grad()
        pooled = torch.einsum("mnd,n->md", Xfit, torch.softmax(theta, dim=0))
        pred = _standardize(_lnpool(pooled)) @ headA
        loss = ((pred - yfit) ** 2).mean() + wd * (headA * headA).sum()
        loss.backward(); optA.step()
        if step % 20 == 0 or step == nstep - 1:
            r = _holdric_static(theta, headA)
            if np.isfinite(r) and r > bestA[0]:
                bestA = (r, theta.detach().clone(), headA.detach().clone())
    with torch.no_grad():
        a_final = torch.softmax(theta, dim=0).detach()        # final-step (unselected)
        a_hold = torch.softmax(bestA[1], dim=0).detach()      # hold-selected
    headA_final = headA.detach(); headA_hold = bestA[2]
    a_entropy = float((-(a_final * a_final.clamp_min(eps).log()).sum() / math.log(num_alphas)).item())
    a_top10 = float(a_final.topk(min(10, num_alphas)).values.sum().item())
    log(f"[probe] ArmA static: final_entropy={a_entropy:.4f} top10={a_top10:.4f} hold_ric={bestA[0]:.5f}")

    # ----- Arm A-prime (k=1) & Arm C (k=4): INPUT-CONDITIONAL attention poolers (R1/R3 function class) -----
    def _fit_attn(k, steps=600, lr=5e-3):
        Q = torch.zeros(k, D, device=device, requires_grad=True)        # warm uniform
        h0 = torch.zeros(k * D, device=device)
        h0[:D] = (w0.detach() / sd.squeeze(0))                          # block0 ~ Arm0 (rank-equiv, LN-only)
        head_k = h0.clone().requires_grad_(True)
        optK = torch.optim.Adam([Q, head_k], lr=lr)

        def _hr():
            with torch.no_grad():
                ph = (_attn_feat(Xhold, Q) @ head_k).detach().cpu().numpy()
            r, _ = daily_rank_ic(ph, yhold_np, dhold)
            return r
        best = (_hr(), Q.detach().clone(), head_k.detach().clone())
        for step in range(steps):
            optK.zero_grad()
            pred = _attn_feat(Xfit, Q) @ head_k
            loss = ((pred - yfit) ** 2).mean() + wd * (head_k * head_k).sum()
            loss.backward(); optK.step()
            if step % 20 == 0 or step == steps - 1:
                r = _hr()
                if np.isfinite(r) and r > best[0]:
                    best = (r, Q.detach().clone(), head_k.detach().clone())
        with torch.no_grad():
            lg = torch.einsum("mnd,kd->mkn", Xhold, best[1])
            cs_std = float(torch.softmax(lg, dim=2)[:, 0, :].std(dim=0).mean().item())
        return best[1], best[2], best[0], cs_std

    Qp, headp, holdp, csstd_p = _fit_attn(1)
    Qc, headc, holdc, _csc = _fit_attn(4)
    log(f"[probe] ArmA-prime(k1) hold_ric={holdp:.5f} cross_stock_wstd={csstd_p:.4f} | ArmC(k4) hold_ric={holdc:.5f}")

    # ----- Arm-ASP: 2nd-moment (cross-factor std over the W_v value bank) — SEPARATE lever -----
    Vtr = torch.einsum("mnd,ed->mne", Xtr, Wv)                          # value bank [M,N,D]
    meanV = Vtr.mean(dim=1); stdV = Vtr.std(dim=1, unbiased=False)      # [M,D]
    del Vtr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def _cv_generic(feat):
        muf = feat[fit_mask].mean(0, keepdim=True)
        sdf = feat[fit_mask].std(0, keepdim=True).clamp_min(1e-6)
        _, w = _pick_lambda((feat - muf) / sdf, ytr_t)
        return (muf, sdf, w)
    asp_base = _cv_generic(_lnpool(meanV))                              # mean(V) ridge (ASP baseline)
    asp_full = _cv_generic(torch.cat([_lnpool(meanV), _lnpool(stdV)], dim=1))  # concat[mean,std]
    log("[probe] Arm-ASP (2nd-moment) fit done")

    # free the big train tensors before the valid-eval pass
    del Xtr, Xfit, Xhold, yfit, Mtr_pool, Xs0, meanV, stdV
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =====================================================================
    # eval Arm0/A/B on valid (stream from cached val_h)
    # =====================================================================
    log("[pass] eval probe arms on valid ...")
    buf = {k: [] for k in ("p0", "paF", "paH", "pap", "pac", "aspb", "aspf")}
    yv_all: List[np.ndarray] = []
    dv_all: List[np.ndarray] = []
    slot_sig_chunks: List[np.ndarray] = []
    slot_ic_sums = np.zeros(num_alphas)
    slot_ic_cnt = np.zeros(num_alphas)
    with torch.no_grad():
        for hh, yv, sc, dday in zip(val_h, val_y, val_score, val_day):
            h = hh.to(device).float()                    # [B,N,D]
            B = h.shape[0]
            m = np.isfinite(yv)
            # Arm0 (mean-pool)
            buf["p0"].append((_standardize(_lnpool(h.mean(dim=1))) @ w0).cpu().numpy())
            # Arm A static: final-step (unselected) and hold-selected
            buf["paF"].append((_standardize(_lnpool(torch.einsum("mnd,n->md", h, a_final))) @ headA_final).cpu().numpy())
            buf["paH"].append((_standardize(_lnpool(torch.einsum("mnd,n->md", h, a_hold))) @ headA_hold).cpu().numpy())
            # Arm A-prime (k1) + Arm C (k4): input-conditional attention poolers
            buf["pap"].append((_attn_feat(h, Qp) @ headp).cpu().numpy())
            buf["pac"].append((_attn_feat(h, Qc) @ headc).cpu().numpy())
            # Arm-ASP: 2nd-moment over value bank
            V = torch.einsum("mnd,ed->mne", h, Wv); mV = V.mean(dim=1); sV = V.std(dim=1, unbiased=False)
            fb = _lnpool(mV); ff = torch.cat([_lnpool(mV), _lnpool(sV)], dim=1)
            buf["aspb"].append((((fb - asp_base[0]) / asp_base[1]) @ asp_base[2]).cpu().numpy())
            buf["aspf"].append((((ff - asp_full[0]) / asp_full[1]) @ asp_full[2]).cpu().numpy())
            yv_all.append(yv); dv_all.append(np.full(B, dday[0]))
            # Arm B slot signals
            hs = (h - slot_mu) / slot_sd
            slot_sig = torch.einsum("bnd,nd->bn", hs, slot_w).cpu().numpy()
            for n in range(num_alphas):
                fin = m & np.isfinite(slot_sig[:, n])
                if fin.sum() >= 3:
                    ic = _spearman(slot_sig[fin, n], yv[fin])
                    if np.isfinite(ic):
                        slot_ic_sums[n] += ic
                        slot_ic_cnt[n] += 1
            if m.sum() >= 2:
                slot_sig_chunks.append(slot_sig[m])

    yvv = np.concatenate(yv_all); dvv = np.concatenate(dv_all)

    def _ric(key):
        r, _ = daily_rank_ic(np.concatenate(buf[key]), yvv, dvv)
        return float(r)
    ric0 = _ric("p0"); ricaF = _ric("paF"); ricaH = _ric("paH")
    ricap = _ric("pap"); ricac = _ric("pac"); raspb = _ric("aspb"); raspf = _ric("aspf")

    slot_ic = slot_ic_sums / np.maximum(slot_ic_cnt, 1)
    slot_ic_abs = np.abs(slot_ic)
    order = np.sort(slot_ic_abs)
    k = max(1, num_alphas // 10)
    top_dec = float(order[-k:].mean()); bot_dec = float(order[:k].mean())
    decile_ratio = float(top_dec / max(bot_dec, 1e-6))
    S = np.concatenate(slot_sig_chunks, axis=0) if slot_sig_chunks else np.zeros((2, num_alphas))
    S = S - S.mean(axis=0, keepdims=True)
    C = (S.T @ S) / max(S.shape[0] - 1, 1)
    evals = np.clip(np.linalg.eigvalsh(C), 0, None)
    pe = evals / max(evals.sum(), 1e-12)
    eff_rank_cov = float(np.exp(-(pe[pe > 0] * np.log(pe[pe > 0])).sum()))

    GATE = 0.003
    probe = {
        "arm0_meanpool_rank_ic": ric0,
        # static convex reweight (the original ArmA): final-step + hold-selected (magnitude-caveat fix)
        "armA_static_final_rank_ic": ricaF,
        "armA_static_hold_rank_ic": ricaH,
        "armA_minus_arm0": float(ricaF - ric0),                # back-compat (final-step)
        "armA_hold_minus_arm0": float(ricaH - ric0),
        "armA_weight_entropy_norm": a_entropy,
        "armA_weight_top10_mass": a_top10,
        # INPUT-CONDITIONAL poolers (the R1/R3 function class) — THE GATE
        "armAprime_conditional_rank_ic": ricap,
        "armAprime_minus_arm0": float(ricap - ric0),
        "armAprime_cross_stock_wstd": float(csstd_p),
        "armC_pma4_rank_ic": ricac,
        "armC_minus_arm0": float(ricac - ric0),
        # 2nd-moment ASP (separate lever, not R1/R2/R3)
        "arm_asp_base_rank_ic": raspb,
        "arm_asp_full_rank_ic": raspf,
        "asp_std_contribution": float(raspf - raspb),
        "asp_full_minus_arm0": float(raspf - ric0),
        # Arm B marginal slot structure
        "armB_slot_ic_top_decile_abs_mean": top_dec,
        "armB_slot_ic_bottom_decile_abs_mean": bot_dec,
        "armB_slot_ic_decile_ratio": decile_ratio,
        "armB_slot_ic_abs_median": float(np.median(slot_ic_abs)),
        "armB_slot_ic_abs_max": float(np.max(slot_ic_abs)),
        "armB_slot_signal_cov_effective_rank": eff_rank_cov,
        "lambda0": float(lam0),
    }
    # per-seed gate flags (aggregate decides majority across seeds)
    probe["gate_static_helps"] = bool((ricaH - ric0) >= GATE)
    probe["gate_conditional_helps"] = bool((ricap - ric0) >= GATE)
    probe["gate_pma4_helps"] = bool((ricac - ric0) >= GATE)
    probe["gate_asp_helps"] = bool((raspf - raspb) >= GATE)
    probe["any_reweight_or_pool_helps"] = bool(
        probe["gate_static_helps"] or probe["gate_conditional_helps"] or probe["gate_pma4_helps"])
    # legacy field (kept for the existing aggregator): static-reweight can-help
    probe["sharpening_can_help"] = bool(probe["gate_static_helps"] and decile_ratio >= 2.0)
    probe["rational_equilibrium_null_supported"] = bool(not probe["any_reweight_or_pool_helps"])
    results["PROBE_CEILING"] = probe
    log(f"[PROBE] arm0={ric0:.5f} | ArmA(static hold)={ricaH:.5f} gap={ricaH-ric0:+.5f} | "
        f"ArmA'(cond k1)={ricap:.5f} gap={ricap-ric0:+.5f} | ArmC(k4)={ricac:.5f} gap={ricac-ric0:+.5f} | "
        f"ASP std-contrib={raspf-raspb:+.5f} | any_helps={probe['any_reweight_or_pool_helps']}")

    handle.remove()

    # save slot ICs for later plotting
    np.save(out_root / "slot_ic.npy", slot_ic)
    (out_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_root / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"[done] wrote {out_root/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
