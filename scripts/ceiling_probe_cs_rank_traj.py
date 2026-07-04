# -*- coding: utf-8 -*-
"""
CS-RANK CEILING PROBE — TRAJECTORY arm (decisive L-4 test).

ceiling_probe_cs_rank_window.py already answered the DAY-T question: on top of the raw 8x158 window,
does adding the CROSS-SECTIONAL RANK AT DAY T help? Answer: +0.0033 (marginal, below MDE=0.005).

But that under-tests what L-4 (CSRankAppend + the model's 8-step window) actually feeds the network.
CSRankAppend ranks EVERY day, then the model's sliding window stacks 8 consecutive days => the model
sees the full 8-STEP CS-RANK TRAJECTORY (how a stock's cross-sectional percentile MOVES over the
window), not just its day-T snapshot rank. A rank trajectory encodes cross-sectional momentum/reversal
that a single day-T rank cannot. This probe asks the fair, model-matched question:

  RW        = raw 8x158 window (1264)                          -> reproduces L-1 variant-B ~0.0592
  RW+CSRt   = raw window (1264) + 8-step CS-rank window (1264) = 2528  -> the true L-4 ceiling test

If RW+CSRt beats RW by >= MDE (0.005), the CS-rank TRAJECTORY has headroom the day-T probe missed and
L-4 is worth the GPU. If it lands in (0, 0.003], L-4's signal is redundant with the raw window even as
a trajectory => cheap kill, pivot to an architecture lever.

Leakage-clean protocol identical to the day-T probe: PIT per-day rank on UNCLIPPED robust-z features,
fit on train only, tune num_boost/alpha on an internal 2019 slice by RankIC, >=3 LGBM seeds, no
per-test-day refit. Writes analysis/tau_scale05_diagnosis/ceiling_probe_cs_rank_traj_RESULT.{json,md}.
"""
import os, json, time, traceback
import numpy as np
import pandas as pd
from pathlib import Path
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = Path(r"C:\Users\60585\PycharmProjects\qibMacV2\analysis\tau_scale05_diagnosis")
MODEL_REF = 0.0786; RAW_WINDOW_CEIL = 0.0592; DAYT_LIFT = 0.0033; MDE = 0.005; WIN = 8

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

if SMOKE:
    HS, HE = "2018-01-01", "2020-12-31"; TRAIN_END="2019-06-30"; INT_FIT_END="2019-03-31"; INT_EVAL_START="2019-04-01"; TEST_START, TEST_END="2020-01-01","2020-06-30"
else:
    HS, HE = "2008-01-01", "2022-12-31"; TRAIN_END="2020-03-31"; INT_FIT_END="2018-12-31"; INT_EVAL_START="2019-01-01"; TEST_START, TEST_END="2020-07-01","2022-12-31"
TRAIN_START = HS

def daily_rank_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ics = [sub["p"].corr(sub["y"], method="spearman") for _, sub in df.groupby(level="datetime") if len(sub) >= 5]
    ics = np.array([x for x in ics if np.isfinite(x)], float)
    if ics.size == 0: return {"rank_ic": float("nan"), "n_days": 0}
    return {"rank_ic": float(ics.mean()), "n_days": int(ics.size)}

def make_handler(clip):
    return init_instance_by_config({"class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": {
        "start_time": HS, "end_time": HE, "fit_start_time": HS, "fit_end_time": TRAIN_END, "instruments": "csi300",
        "infer_processors": [{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": clip}},
                             {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
        "learn_processors": [{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label", "method": "robust"}}],
        "label": ["Ref($close, -5) / Ref($close, -1) - 1"]}})

def build_window(Xi):
    """8-step flattened window, end-row = day T, within a single instrument (L-1 variant-B builder)."""
    from numpy.lib.stride_tricks import sliding_window_view
    Xs = Xi.sort_index(level=["instrument", "datetime"]); arr = Xs.to_numpy(np.float32); idx = Xs.index
    inst = idx.get_level_values("instrument").to_numpy(); N = arr.shape[0]
    change = np.empty(N, bool); change[0] = True; change[1:] = inst[1:] != inst[:-1]
    grp_start = np.maximum.accumulate(np.where(change, np.arange(N), 0))
    sw = sliding_window_view(arr, WIN, axis=0); ends = np.arange(WIN - 1, N)
    same = (ends - grp_start[ends]) >= (WIN - 1)
    Xb = sw[same].transpose(0, 2, 1).reshape(int(same.sum()), WIN * arr.shape[1]).astype(np.float32)
    idx_b = idx[ends][same]
    return Xb, idx_b

def main():
    t0 = time.time(); log(f"qlib.init (SMOKE={SMOKE})")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, kernels=1)
    log("clipped handler (raw window src)..."); h = make_handler(True)
    Xi = h.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32)
    yi_raw = h.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0].astype(np.float32)
    yl_z = h.fetch(col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0].astype(np.float32)
    F = Xi.shape[1]; log("clipped %.1fs" % (time.time()-t0), Xi.shape)

    log("unclipped handler (CS-rank src)..."); hnc = make_handler(False)
    Xnc = hnc.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32).reindex(Xi.index)
    # Per-day cross-sectional pct-rank-0.5 (matches CSRankAppend on unclipped robust-z features).
    CSR = (Xnc.groupby(level="datetime").rank(pct=True).astype(np.float32) - np.float32(0.5)); del Xnc
    log("CS-rank frame %.1fs" % (time.time()-t0), CSR.shape)

    log("building raw window..."); Xb, idx_b = build_window(Xi); del Xi
    Xb_df = pd.DataFrame(Xb, index=idx_b); del Xb
    Xb_df.columns = [f"w{i}" for i in range(Xb_df.shape[1])]
    log("raw window", Xb_df.shape)

    # KEY DIFFERENCE vs the day-T probe: window the CS-rank frame too => 8-step rank trajectory (1264).
    log("building CS-rank TRAJECTORY window...")
    CSRb, idx_csr = build_window(CSR); del CSR
    CSRb_df = pd.DataFrame(CSRb, index=idx_csr); del CSRb
    CSRb_df.columns = [f"c{i}" for i in range(CSRb_df.shape[1])]
    CSRb_df = CSRb_df.reindex(idx_b)  # align to raw-window rows
    log("CS-rank trajectory window", CSRb_df.shape)

    dt_b = idx_b.get_level_values("datetime")
    yb_z = yl_z.reindex(idx_b); yb_raw = yi_raw.reindex(idx_b)
    trb = ((dt_b >= TRAIN_START) & (dt_b <= TRAIN_END) & yb_z.notna().to_numpy())
    fitb = ((dt_b >= TRAIN_START) & (dt_b <= INT_FIT_END) & yb_z.notna().to_numpy())
    evb = ((dt_b >= INT_EVAL_START) & (dt_b <= TRAIN_END) & yb_z.notna().to_numpy())
    teb = ((dt_b >= TEST_START) & (dt_b <= TEST_END) & yb_raw.notna().to_numpy())

    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    def fit_score(Xf, yf, Xe, ye, Xfu, yfu, Xt, yt, tag):
        out = {}; best = (None, -1e9)
        for a in [1.0,10.,100.,1e3,1e4,1e5,1e6]:
            r = Ridge(alpha=a).fit(Xf, yf)
            ic = daily_rank_ic(pd.Series(r.predict(Xe), index=Xe.index), ye)["rank_ic"]
            if np.isfinite(ic) and ic > best[1]: best = (a, ic)
        r = Ridge(alpha=best[0]).fit(Xfu, yfu)
        out["ridge"] = {**daily_rank_ic(pd.Series(r.predict(Xt), index=Xt.index), yt), "alpha": best[0], "int_eval_rank_ic": best[1]}
        log(f"  [{tag}] ridge a={best[0]:g} int={best[1]:.5f} test={out['ridge']['rank_ic']:.5f}")
        p = dict(objective="regression", learning_rate=0.03, num_leaves=63, feature_fraction=0.6,
                 bagging_fraction=0.7, bagging_freq=1, min_child_samples=200, max_depth=-1, n_jobs=-1, verbosity=-1)
        dfit = lgb.Dataset(Xf, label=yf); best_n, best_nic = 300, -1e9
        for n in [100, 300, 600, 1000]:
            m = lgb.train(dict(p, seed=0), dfit, num_boost_round=n)
            ic = daily_rank_ic(pd.Series(m.predict(Xe), index=Xe.index), ye)["rank_ic"]
            log(f"  [{tag}] lgbm n={n} int={ic:.5f}")
            if np.isfinite(ic) and ic > best_nic: best_nic, best_n = ic, n
        dfu = lgb.Dataset(Xfu, label=yfu); sics = []
        for s in [0, 1, 2]:
            m = lgb.train(dict(p, seed=s, bagging_seed=s, feature_fraction_seed=s), dfu, num_boost_round=best_n)
            sics.append(daily_rank_ic(pd.Series(m.predict(Xt), index=Xt.index), yt)["rank_ic"])
            log(f"  [{tag}] lgbm s={s} n={best_n} test={sics[-1]:.5f}")
        out["lgbm"] = {"seed_rank_ics": sics, "rank_ic_mean": float(np.nanmean(sics)), "rank_ic_std": float(np.nanstd(sics)), "best_n": best_n, "int_eval_rank_ic": best_nic}
        out["ceiling"] = float(np.nanmax([out["ridge"]["rank_ic"], out["lgbm"]["rank_ic_mean"]]))
        log(f"  [{tag}] CEILING={out['ceiling']:.5f}"); return out

    results = {"meta": {"smoke": SMOKE, "model_ref": MODEL_REF, "raw_window_ceiling": RAW_WINDOW_CEIL,
                        "dayT_lift_reference": DAYT_LIFT, "F": F, "WIN": WIN}, "arms": {}}
    SKIP_RW = os.environ.get("SKIP_RW", "0") == "1"
    if SKIP_RW:
        results["arms"]["RW"] = {"ceiling": 0.05922, "note": "reused from prior window-probe run (LGBM 0.05817/0.05913/0.06036)"}
        log("ARM RW skipped (reuse known 0.05922)")
    else:
        log("ARM RW (raw window, reproduce 0.0592)")
        results["arms"]["RW"] = fit_score(Xb_df[fitb], yb_z[fitb], Xb_df[evb], yb_raw[evb], Xb_df[trb], yb_z[trb], Xb_df[teb], yb_raw[teb], "RW")
    json.dump(results, open(OUT / "ceiling_probe_cs_rank_traj_RESULT.json", "w"), indent=2)

    log("ARM RW+CSRt (raw window + 8-step CS-rank trajectory, the true L-4 test)")
    Xrc = pd.concat([Xb_df, CSRb_df], axis=1); del Xb_df, CSRb_df
    import gc; gc.collect()
    log("combined design", Xrc.shape)
    results["arms"]["RW+CSRt"] = fit_score(Xrc[fitb], yb_z[fitb], Xrc[evb], yb_raw[evb], Xrc[trb], yb_z[trb], Xrc[teb], yb_raw[teb], "RW+CSRt")
    json.dump(results, open(OUT / "ceiling_probe_cs_rank_traj_RESULT.json", "w"), indent=2)

    RW = results["arms"]["RW"]["ceiling"]; RWC = results["arms"]["RW+CSRt"]["ceiling"]; lift = RWC - RW
    zone = ("CS_TRAJ_HAS_HEADROOM_OVER_WINDOW__L4_go" if lift >= MDE else
            "CS_TRAJ_REDUNDANT_WITH_WINDOW__L4_no_go" if lift <= 0.003 else "MODAL__marginal")
    results["verdict"] = {"RW_ceiling": RW, "RWplusCSRt_ceiling": RWC, "traj_lift": lift,
                          "dayT_lift_reference": DAYT_LIFT, "model_ref": MODEL_REF, "zone": zone}
    json.dump(results, open(OUT / "ceiling_probe_cs_rank_traj_RESULT.json", "w"), indent=2)
    md = ["# CS-rank Ceiling Probe — TRAJECTORY arm (decisive L-4 test)", "",
          f"- SMOKE={SMOKE} F={F} WIN={WIN} model_ref={MODEL_REF} raw_window(L-1 B)={RAW_WINDOW_CEIL} dayT_lift_ref={DAYT_LIFT}",
          f"- **RW={RW:.5f}  RW+CSRt={RWC:.5f}  traj_lift={lift:+.5f}** (day-T lift was {DAYT_LIFT:+.4f})",
          f"- **VERDICT: {zone}**", ""]
    for nm, v in results["arms"].items():
        if "ridge" not in v:
            md += [f"## {nm}  ceiling={v['ceiling']:.5f}", f"- {v.get('note','')}", ""]; continue
        md += [f"## {nm}  ceiling={v['ceiling']:.5f}", f"- ridge {v['ridge']['rank_ic']:.5f} (alpha={v['ridge']['alpha']})",
               f"- lgbm {v['lgbm']['rank_ic_mean']:.5f} ± {v['lgbm']['rank_ic_std']:.5f} (n={v['lgbm']['best_n']}, seeds {['%.5f'%x for x in v['lgbm']['seed_rank_ics']]})", ""]
    (OUT / "ceiling_probe_cs_rank_traj_RESULT.md").write_text("\n".join(md), encoding="utf-8")
    log("DONE %.1fs -> %s" % (time.time()-t0, zone)); log("verdict:", json.dumps(results["verdict"]))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
