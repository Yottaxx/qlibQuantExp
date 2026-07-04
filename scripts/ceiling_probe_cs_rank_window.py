# -*- coding: utf-8 -*-
"""
CS-RANK CEILING PROBE — WINDOW arm (model-matched). Tightens ceiling_probe_cs_rank.py by addressing its one
confound: the snapshot R+CSR lift (+0.0085) is measured WITHOUT the 8-step temporal window the RST-MoE model
has. The fair, model-matched question is: on top of the raw 8x158 window (L-1 variant-B = 0.0592), does adding
the day-T cross-sectional rank coordinate still lift the leakage-clean GBDT ceiling? If CS-rank were merely
re-encoding momentum/reversal already in the window, the windowed lift would vanish.

ARMS:
  RW      = raw 8x158 window (1264)             -> reproduces L-1 variant-B ~0.0592
  RW+CSR  = raw window (1264) + day-T CS-rank (158) = 1422  -> the model-matched L-4 test

CS-rank: per-day pct-rank-0.5 on UNCLIPPED robust-z features (PIT, full daily universe), aligned to each
window's END row (day T). Same leakage-clean fit_score as L-1 (fit on train only, tune on 2019..2020-03
RankIC, >=3 LGBM seeds, no per-test-day refit). Writes analysis/tau_scale05_diagnosis/
ceiling_probe_cs_rank_window_RESULT.{json,md}.
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
MODEL_REF = 0.0786; RAW_WINDOW_CEIL = 0.0592; MDE = 0.005; WIN = 8

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
    """L-1 variant-B window builder: 8-step flattened, end-row = day T, within a single instrument."""
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
    log("clipped handler..."); h = make_handler(True)
    Xi = h.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32)
    yi_raw = h.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0].astype(np.float32)
    yl_z = h.fetch(col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0].astype(np.float32)
    F = Xi.shape[1]; log("clipped %.1fs" % (time.time()-t0), Xi.shape)
    log("unclipped handler (CS-rank src)..."); hnc = make_handler(False)
    Xnc = hnc.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32).reindex(Xi.index)
    CSR = (Xnc.groupby(level="datetime").rank(pct=True).astype(np.float32) - np.float32(0.5)); del Xnc
    log("CS-rank %.1fs" % (time.time()-t0))

    log("building raw window..."); Xb, idx_b = build_window(Xi); del Xi
    Xb_df = pd.DataFrame(Xb, index=idx_b); del Xb
    Xb_df.columns = [f"w{i}" for i in range(Xb_df.shape[1])]   # string col names (sklearn rejects mixed int/str)
    log("raw window", Xb_df.shape)
    CSR_b = CSR.reindex(idx_b); CSR_b.columns = [f"{c}_csr" for c in CSR_b.columns]; del CSR
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

    results = {"meta": {"smoke": SMOKE, "model_ref": MODEL_REF, "raw_window_ceiling": RAW_WINDOW_CEIL, "F": F, "WIN": WIN}, "arms": {}}
    SKIP_RW = os.environ.get("SKIP_RW", "0") == "1"
    if SKIP_RW:
        results["arms"]["RW"] = {"ceiling": 0.05922, "note": "reused from prior run log (LGBM 0.05817/0.05913/0.06036); reproduces L-1 variant-B 0.0592 exactly"}
        log("ARM RW skipped (reuse known 0.05922)")
    else:
        log("ARM RW (raw window, reproduce 0.0592)")
        results["arms"]["RW"] = fit_score(Xb_df[fitb], yb_z[fitb], Xb_df[evb], yb_raw[evb], Xb_df[trb], yb_z[trb], Xb_df[teb], yb_raw[teb], "RW")
    json.dump(results, open(OUT / "ceiling_probe_cs_rank_window_RESULT.json", "w"), indent=2)
    log("ARM RW+CSR (raw window + day-T CS-rank, model-matched L-4 test)")
    Xrc = pd.concat([Xb_df, CSR_b], axis=1); del Xb_df, CSR_b
    import gc; gc.collect()
    results["arms"]["RW+CSR"] = fit_score(Xrc[fitb], yb_z[fitb], Xrc[evb], yb_raw[evb], Xrc[trb], yb_z[trb], Xrc[teb], yb_raw[teb], "RW+CSR")
    json.dump(results, open(OUT / "ceiling_probe_cs_rank_window_RESULT.json", "w"), indent=2)

    RW = results["arms"]["RW"]["ceiling"]; RWC = results["arms"]["RW+CSR"]["ceiling"]; lift = RWC - RW
    zone = ("CS_INPUT_HAS_HEADROOM_OVER_WINDOW__L4_go" if lift >= MDE else
            "CS_INPUT_REDUNDANT_WITH_WINDOW__L4_no_go" if lift <= 0.003 else "MODAL__marginal")
    results["verdict"] = {"RW_ceiling": RW, "RWplusCSR_ceiling": RWC, "windowed_lift": lift, "model_ref": MODEL_REF, "zone": zone}
    json.dump(results, open(OUT / "ceiling_probe_cs_rank_window_RESULT.json", "w"), indent=2)
    md = ["# CS-rank Ceiling Probe — WINDOW arm (model-matched)", "",
          f"- SMOKE={SMOKE} F={F} WIN={WIN} model_ref={MODEL_REF} raw_window(L-1 B)={RAW_WINDOW_CEIL}",
          f"- **RW={RW:.5f}  RW+CSR={RWC:.5f}  windowed_lift={lift:+.5f}**", f"- **VERDICT: {zone}**", ""]
    for nm, v in results["arms"].items():
        if "ridge" not in v:
            md += [f"## {nm}  ceiling={v['ceiling']:.5f}", f"- {v.get('note','')}", ""]; continue
        md += [f"## {nm}  ceiling={v['ceiling']:.5f}", f"- ridge {v['ridge']['rank_ic']:.5f} (alpha={v['ridge']['alpha']})",
               f"- lgbm {v['lgbm']['rank_ic_mean']:.5f} ± {v['lgbm']['rank_ic_std']:.5f} (n={v['lgbm']['best_n']}, seeds {['%.5f'%x for x in v['lgbm']['seed_rank_ics']]})", ""]
    (OUT / "ceiling_probe_cs_rank_window_RESULT.md").write_text("\n".join(md), encoding="utf-8")
    log("DONE %.1fs -> %s" % (time.time()-t0, zone)); log("verdict:", json.dumps(results["verdict"]))

if __name__ == "__main__":
    main()
