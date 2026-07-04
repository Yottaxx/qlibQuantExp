#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STOCK-axis KEY-SVD forensics for RST-MoE (cross-stock common-mode probe).

The stock-axis analog of scripts/pool_readout_forensics.py's FACTOR-axis KEY-SVD.

The MoE block (module/architecture/moe_block.py) has time/factor experts plus an
optional stock expert that would attend over the daily cross-section (B stocks).
The stock attention collapses to UNIFORM (entropy_norm=1.0). Hypothesis: the time
embedding + factor_id_emb are added IDENTICALLY across stocks (a common-mode vector
~1.3x the per-stock value norm), so the cross-stock representation =
  (large common vector, shared by all stocks) + (small stock-relative residual)
and the residual is swamped -> any stock-attention query sees ~constant keys ->
uniform. We quantify the common-vs-relative split and whether the residual is
"peakable" by a perfectly-aligned query.

We capture each MoE block's post-norm representation x [B,T,N,D] (the tensor the
experts consume, hooked on block.norm1's OUTPUT) and SVD across the STOCK axis (B)
at the last time step, over a strided subset of factor slots. Per layer we report
s1_over_common, common_energy_frac, eff_rank, s2/s1, and a peakability entropy@T
grid; and a one-line world-A (common-mode + peakable residual -> key-centering +
warm query fixes it) vs world-B (residual structureless -> abandon) verdict.

Faithfulness gate: the fp32 forward's valid daily rank_ic must land in the g012
anchor band (~0.078-0.082); if far off, the backbone didn't load -> debug first.

CPU-ONLY by construction (CUDA_VISIBLE_DEVICES="" set before torch import) so it
does not contend with a GPU job that may be running.

Run (quantEnv):
  python scripts/stock_key_svd.py --valid-stride 3
"""
from __future__ import annotations

# --- belt-and-suspenders: force CPU BEFORE importing torch ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_MODEL = ("mlruns/716849652326531066/"
                 "2e8e24fe541a4a31949275b387da2ef2/artifacts/model")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RST-MoE stock-axis KEY-SVD forensics (CPU).")
    ap.add_argument("--load-model", default=DEFAULT_MODEL,
                    help="Path to the qlib-saved backbone (g012 anchor seed42).")
    ap.add_argument("--out-dir", default="analysis/stock_expert")
    ap.add_argument("--valid-stride", type=int, default=3,
                    help="Keep every k-th valid day to stay fast.")
    ap.add_argument("--n-factor-slots", type=int, default=8,
                    help="~How many strided factor slots n to sample per day.")
    ap.add_argument("--label", default="g012_anchor_seed42")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# numeric helpers (numpy) — mirror the template
# ---------------------------------------------------------------------------
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
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(p: np.ndarray, y: np.ndarray) -> float:
    if p.size < 3:
        return float("nan")
    if np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    rp = _rankdata(p) - _rankdata(p).mean()
    ry = _rankdata(y) - _rankdata(y).mean()
    denom = math.sqrt(float((rp * rp).sum()) * float((ry * ry).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((rp * ry).sum() / denom)


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
    if not out:
        return float("nan"), 0
    return float(np.mean(out)), len(out)


def main() -> int:
    args = _parse_args()

    import torch
    import pandas as pd
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    import work_flow as wf
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP

    torch.backends.cuda.matmul.allow_tf32 = False  # clean fp32 for SVD
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[setup] CPU-only stock-axis KEY-SVD; out={out_root} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    cpu = torch.device("cpu")

    # ----- data -----
    dataset = init_instance_by_config(wf.data_conf)

    # ----- load faithful backbone (qlib-pickled QlibQuantMoE) — no training -----
    # The artifact was saved with CUDA storages; under CUDA_VISIBLE_DEVICES="" the
    # nested torch.load (called by Storage._load_from_bytes inside the pickle) would
    # try to restore onto cuda:0 and fail. Patch torch.load to default map_location
    # to CPU for the duration of the pickle load.
    import pickle
    t0 = time.time()
    _orig_torch_load = torch.load

    def _cpu_torch_load(*a, **kw):
        kw.setdefault("map_location", cpu)
        return _orig_torch_load(*a, **kw)

    torch.load = _cpu_torch_load
    try:
        with open(args.load_model, "rb") as f:
            model = pickle.load(f)
    finally:
        torch.load = _orig_torch_load
    if getattr(model, "net", None) is None:
        raise RuntimeError(f"loaded object has no .net: {args.load_model}")
    # patch lazy/private attributes a possibly-older pickle may lack
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

    # ----- FORCE CPU so we do not contend with a GPU job -----
    model.device = cpu
    model.net = model.net.cpu()
    log(f"[load] loaded backbone from {args.load_model} in {time.time() - t0:.1f}s "
        f"(forced device={model.device})")

    model._ensure_market_state()
    # ensure any cached market-state tensor lives on CPU
    ms = getattr(model, "_market_state", None)
    if ms is not None:
        try:
            for _a in ("tensor", "data", "values"):
                _v = getattr(ms, _a, None)
                if isinstance(_v, torch.Tensor) and _v.device.type != "cpu":
                    setattr(ms, _a, _v.cpu())
        except Exception:
            pass

    net = model.net
    net.eval()

    # patch lazy/plain attributes on the pickled MoE blocks that the CURRENT class
    # __init__ sets but an older pickle predates (n_experts, use_stock_expert, etc.).
    # These are derived from config; the net weights are unchanged.
    for blk in net.layers:
        cfg = getattr(blk, "config", None)
        if not hasattr(blk, "use_stock_expert"):
            blk.use_stock_expert = bool(getattr(cfg, "use_stock_expert", False)) if cfg else False
        if not hasattr(blk, "n_experts"):
            blk.n_experts = 3 if blk.use_stock_expert else 2
        if not hasattr(blk, "use_layer_summary"):
            blk.use_layer_summary = bool(getattr(cfg, "router_use_layer_summary", False)) if cfg else False
        if not hasattr(blk, "router_mode"):
            blk.router_mode = str(getattr(cfg, "router_mode", "learned") or "learned").strip().lower() if cfg else "learned"

    device = cpu
    num_alphas = int(model._get_num_alphas())
    f_ids = torch.arange(num_alphas, device=device)
    eps = 1e-12

    # ----- model geometry -----
    D = int(getattr(net, "d_model", getattr(model, "d_model", 64)) or 64)
    n_heads = int(getattr(getattr(net, "config", None), "n_heads", 4) or 4)
    d_head = D // max(n_heads, 1)
    scale = 1.0 / math.sqrt(d_head)
    blocks = list(net.layers)  # ModuleList of RegimeAdaptiveMoEBlock
    n_layers = len(blocks)
    log(f"[geom] D={D} n_heads={n_heads} d_head={d_head} scale={scale:.5f} "
        f"n_layers={n_layers} num_alphas={num_alphas}")

    # ----- forward_hooks on EACH block's norm1 OUTPUT -> post-norm x [B,T,N,D] -----
    cap: Dict[int, torch.Tensor] = {}

    def _make_hook(li: int):
        def _hook(_mod, _inp, out):
            # norm1 output is the post-norm x the experts consume
            t = out[0] if isinstance(out, (tuple, list)) else out
            cap[li] = t.detach()
        return _hook

    handles = [blk.norm1.register_forward_hook(_make_hook(i)) for i, blk in enumerate(blocks)]

    def iter_segment(segment: str, data_key: str, stride: int = 1):
        """Yield (caps {layer: x[B,T,N,D]}, y[B] np, scores[B] np, day) per day batch."""
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
                cap.clear()
                out = net(bx_t, f_ids, labels=None, macro_features=macro_t)  # fp32, no autocast
                caps = {li: t for li, t in cap.items()}
                yv = None if by is None else by.view(-1).float().cpu().numpy()
                sc = out.scores.view(-1).float().cpu().numpy()
                yield caps, yv, sc, pd.to_datetime(day).normalize()

    # ----- per-layer accumulators -----
    TEMP_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]

    def _new_acc() -> Dict:
        return {
            "s1_over_common": [], "eff_rank": [], "s2_over_s1": [],
            "common_energy_frac": [], "common_norm": [], "s1": [],
            "peak_ent": {t: [] for t in TEMP_GRID},
            "n_slots": 0, "n_days": 0,
        }

    acc: Dict[int, Dict] = {li: _new_acc() for li in range(n_layers)}

    val_y: List[np.ndarray] = []
    val_score: List[np.ndarray] = []
    val_day: List[np.ndarray] = []

    def _entropy_norm(w: torch.Tensor, n: int) -> torch.Tensor:
        ent = -(w * (w.clamp_min(eps)).log()).sum(dim=-1)
        return ent / math.log(max(int(n), 2))

    log(f"[pass] streaming valid (stride={args.valid_stride}) ...")
    for caps, yv, sc, day in iter_segment("valid", DataHandlerLP.DK_I, stride=int(args.valid_stride)):
        # sanity cache (faithfulness gate)
        B0 = sc.shape[0]
        val_y.append(yv if yv is not None else np.full(B0, np.nan))
        val_score.append(sc)
        val_day.append(np.full(B0, np.datetime64(day, "D")))

        for li in range(n_layers):
            x = caps.get(li)
            if x is None:
                continue
            B, T, N, Dd = x.shape
            if B < 3:
                continue  # need >=3 stocks for a meaningful cross-stock SVD
            # LAST time step; strided subset of ~n factor slots
            xt = x[:, T - 1, :, :]                         # [B,N,D]
            step = max(1, N // max(int(args.n_factor_slots), 1))
            slot_idx = list(range(0, N, step))[: int(args.n_factor_slots)]
            acc[li]["n_days"] += 1
            for n in slot_idx:
                X = xt[:, n, :]                             # [B,D] cross-stock matrix
                common = X.mean(dim=0)                      # [D] common-mode (across stocks)
                common_norm = float(common.norm().clamp_min(eps).item())
                Xc = X - common.unsqueeze(0)               # centered across stocks
                try:
                    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  # S [k], Vh [k,D]
                except Exception as e:  # pragma: no cover
                    log(f"[warn] SVD failed layer{li} slot{n}: {e}")
                    continue
                s = S.clamp_min(0.0)
                s1 = float(s[0].clamp_min(eps).item())
                s2 = float(s[1].item()) if s.shape[0] > 1 else 0.0
                # PRIMARY world-B indicator: top per-stock contrast amplitude vs common norm
                s1_over_common = (s1 / math.sqrt(B)) / common_norm
                # effective rank = exp(entropy of normalized squared singular values)
                p2 = s.pow(2)
                pn = p2 / p2.sum().clamp_min(eps)
                eff_rank = float(torch.exp(-(pn * pn.clamp_min(eps).log()).sum()).item())
                # fraction of total cross-stock energy that is common-mode
                resid_energy = float(p2.sum().item())       # sum of s_i^2 (centered, across stocks)
                common_energy = B * (common_norm ** 2)
                common_energy_frac = common_energy / max(common_energy + resid_energy, eps)
                # PEAKABILITY: query perfectly aligned to top cross-stock variation u1 (D-space)
                u1 = Vh[0, :]                               # [D] top right singular vec
                logits_best = (Xc @ u1) * scale            # [B]
                logits_best_c = logits_best - logits_best.mean()
                lstd = float(logits_best_c.std().clamp_min(eps).item())
                logits_unit = logits_best_c / lstd          # unit-std -> remove magnitude
                for t in TEMP_GRID:
                    w_t = torch.softmax(logits_unit * t, dim=0)  # [B] over stock keys
                    acc[li]["peak_ent"][t].append(float(_entropy_norm(w_t, B).item()))
                acc[li]["s1_over_common"].append(s1_over_common)
                acc[li]["eff_rank"].append(eff_rank)
                acc[li]["s2_over_s1"].append(s2 / s1)
                acc[li]["common_energy_frac"].append(common_energy_frac)
                acc[li]["common_norm"].append(common_norm)
                acc[li]["s1"].append(s1)
                acc[li]["n_slots"] += 1

    for h in handles:
        h.remove()

    # ----- faithfulness gate -----
    vY = np.concatenate(val_y) if val_y else np.array([])
    vS = np.concatenate(val_score) if val_score else np.array([])
    vDay = np.concatenate(val_day) if val_day else np.array([])
    model_ric, n_days = daily_rank_ic(vS, vY, vDay)
    faithful = bool(np.isfinite(model_ric) and 0.070 <= model_ric <= 0.090)
    log(f"[sanity] model valid daily_rank_ic={model_ric:.6f} over {n_days} days "
        f"(g012 anchor band ~0.078-0.082); FAITHFUL={faithful}")

    def _mean(x: List[float]) -> float:
        a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
        return float(a.mean()) if a.size else float("nan")

    def _median(x: List[float]) -> float:
        a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
        return float(np.median(a)) if a.size else float("nan")

    # ----- per-layer summary + verdict -----
    per_layer: Dict[str, Dict] = {}
    for li in range(n_layers):
        a = acc[li]
        s1c = _mean(a["s1_over_common"])
        cef = _mean(a["common_energy_frac"])
        peak_by_T = {f"T{t:g}": _mean(a["peak_ent"][t]) for t in TEMP_GRID}
        peak3 = peak_by_T.get("T3", float("nan"))
        common_mode_dominated = bool(
            (np.isfinite(s1c) and s1c <= 0.10) or (np.isfinite(cef) and cef >= 0.80)
        )
        residual_peakable = bool(np.isfinite(peak3) and peak3 <= 0.90)
        if common_mode_dominated and residual_peakable:
            verdict = "A_common_mode_peakable_residual"  # key-centering + warm query fixes it
        elif common_mode_dominated and not residual_peakable:
            verdict = "B_residual_structureless"          # abandon stock attention
        elif not common_mode_dominated:
            verdict = "not_common_mode_dominated"         # stock keys already vary; uniformity is operator-state
        else:
            verdict = "inconclusive"
        per_layer[f"layer{li}"] = {
            "s1_over_common_mean": s1c,
            "s1_over_common_median": _median(a["s1_over_common"]),
            "common_energy_frac_mean": cef,
            "common_energy_frac_median": _median(a["common_energy_frac"]),
            "eff_rank_mean": _mean(a["eff_rank"]),
            "s2_over_s1_mean": _mean(a["s2_over_s1"]),
            "common_norm_mean": _mean(a["common_norm"]),
            "s1_mean": _mean(a["s1"]),
            "peakability_entropy_norm_by_T": peak_by_T,
            "common_mode_dominated": common_mode_dominated,
            "residual_peakable": residual_peakable,
            "world_verdict": verdict,
            "n_slots": int(a["n_slots"]),
            "n_days": int(a["n_days"]),
        }
        log(f"[layer{li}] s1/common={s1c:.4f} common_energy_frac={cef:.4f} "
            f"eff_rank={per_layer[f'layer{li}']['eff_rank_mean']:.3f} "
            f"s2/s1={per_layer[f'layer{li}']['s2_over_s1_mean']:.4f} "
            f"peak_ent@T3={peak3:.4f} -> {verdict}")

    results = {
        "config_label": args.label,
        "loaded_from": args.load_model,
        "device": "cpu",
        "universe": "csi300",
        "D": D, "n_heads": n_heads, "d_head": d_head, "scale": scale,
        "n_layers": n_layers, "num_alphas": num_alphas,
        "valid_stride": int(args.valid_stride),
        "n_factor_slots": int(args.n_factor_slots),
        "sanity_valid_rank_ic": model_ric,
        "valid_n_days": n_days,
        "faithful": faithful,
        "temp_grid": TEMP_GRID,
        "STOCK_KEY_SVD": per_layer,
    }

    out_json = out_root / "stock_key_svd.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"[write] {out_json}")

    # ----- markdown report -----
    md: List[str] = []
    md.append("# STOCK-axis KEY-SVD forensics (cross-stock common-mode probe)\n")
    md.append(f"- backbone: `{args.load_model}` ({args.label})")
    md.append(f"- device: CPU (CUDA_VISIBLE_DEVICES=\"\")")
    md.append(f"- D={D}, n_heads={n_heads}, d_head={d_head}, scale={scale:.4f}, "
              f"n_layers={n_layers}, num_alphas={num_alphas}")
    md.append(f"- valid_stride={args.valid_stride}, n_factor_slots={args.n_factor_slots}")
    md.append(f"- **faithfulness gate**: valid daily rank_ic = **{model_ric:.6f}** "
              f"over {n_days} days (g012 band ~0.078-0.082) -> "
              f"{'PASS' if faithful else 'FAIL — debug load before trusting SVD'}\n")
    md.append("## Per-layer cross-stock geometry (mean over slots/days)\n")
    md.append("| layer | s1/common | common_energy_frac | eff_rank | s2/s1 | "
              "peak_ent@T0.5 | @T1 | @T2 | @T3 | @T5 | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for li in range(n_layers):
        L = per_layer[f"layer{li}"]
        pk = L["peakability_entropy_norm_by_T"]
        md.append(
            f"| {li} | {L['s1_over_common_mean']:.4f} | {L['common_energy_frac_mean']:.4f} | "
            f"{L['eff_rank_mean']:.3f} | {L['s2_over_s1_mean']:.4f} | "
            f"{pk.get('T0.5', float('nan')):.4f} | {pk.get('T1', float('nan')):.4f} | "
            f"{pk.get('T2', float('nan')):.4f} | {pk.get('T3', float('nan')):.4f} | "
            f"{pk.get('T5', float('nan')):.4f} | {L['world_verdict']} |"
        )
    md.append("")
    md.append("## Thresholds / reading")
    md.append("- **common-mode dominated** if `s1_over_common <~ 0.1` OR "
              "`common_energy_frac > ~0.8` (cross-stock keys ~= constant -> uniform attention).")
    md.append("- **residual peakable** if `peak_ent@T3 < ~0.9` (a perfectly-aligned warm "
              "query COULD un-uniform the residual).")
    md.append("- **world-A** = common-mode + peakable residual -> key-centering + warm query fixes it.")
    md.append("- **world-B** = residual structureless / not peakable -> abandon stock attention.")
    md.append("")
    # overall verdict = deepest layer (the one feeding the readout) with fallback
    last = per_layer[f"layer{n_layers - 1}"]
    md.append(f"## Overall verdict (final layer {n_layers - 1}): **{last['world_verdict']}**")
    md.append(f"- s1_over_common={last['s1_over_common_mean']:.4f}, "
              f"common_energy_frac={last['common_energy_frac_mean']:.4f}, "
              f"peak_ent@T3={last['peakability_entropy_norm_by_T'].get('T3', float('nan')):.4f}")
    md.append("")

    out_md = out_root / "STOCK_KEY_SVD.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    log(f"[write] {out_md}")

    if not faithful:
        log("[ERROR] faithfulness gate FAILED — backbone likely did not load; "
            "SVD numbers are NOT trustworthy.")
        return 3
    log("[done] stock-axis KEY-SVD complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
