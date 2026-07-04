# -*- coding: utf-8 -*-
"""
CS-rank ceiling-probe extension (IC-reopen gate, memory reframe IC_REOPEN_20260627 / wf_137057be-799).

Question: my earlier L-1 verdict ("data-capped, model 0.0786 > classic ceiling 0.0592") only bounded what a
strong learner extracts from the SAME PER-STOCK features the model sees. It NEVER tested CROSS-SECTIONAL input.
The architecture is proven CS-blind, and CS-rank (a stock's rank vs TODAY's peers) is literally absent from the
input. So: does APPENDING per-day CS-rank channels raise the extractable ceiling above 0.0592 (toward/past the
model's honest ~0.072-0.078)? If yes -> the IC track reopens via L-4 (CS-rank INPUT channels). If no -> CS-rank
adds nothing even to a tree, kill L-4.

Design (leakage-clean): point-in-time per-day CS-rank-pct of each Alpha158 feature, APPENDED to the global
RobustZScore features (keep globals -> regime info safe; NOT wholesale CS-norm). Same train-only fit / no
per-test-day refit / internal-eval-tuned discipline as the L-1 probe. CPU only.
"""
import os, json, time
import numpy as np, pandas as pd
from pathlib import Path
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from sklearn.linear_model import Ridge
import lightgbm as lgb

OUT = Path(r"C:\Users\60585\PycharmProjects\qibMacV2\analysis\tau_scale05_diagnosis")
MODEL_REF = 0.0786
SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    HS, HE, TREND, IFE, IES, TS, TE = "2018-01-01", "2020-12-31", "2019-06-30", "2019-03-31", "2019-04-01", "2020-01-01", "2020-06-30"
else:
    HS, HE, TREND, IFE, IES, TS, TE = "2008-01-01", "2022-12-31", "2020-03-31", "2018-12-31", "2019-01-01", "2020-07-01", "2022-12-31"

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def daily_rank_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ics = [sub["p"].corr(sub["y"], method="spearman") for _, sub in df.groupby(level="datetime") if len(sub) >= 5]
    ics = np.array([x for x in ics if np.isfinite(x)], float)
    return float(ics.mean()) if ics.size else float("nan")

def fit_score(X_fit, y_fit, X_eval, yeval_raw, X_full, y_full, X_te, yte_raw, tag):
    out = {}
    best = (None, -1e9)
    for a in [1.0, 10., 100., 1e3, 1e4, 1e5, 1e6]:
        ic = daily_rank_ic(pd.Series(Ridge(alpha=a).fit(X_fit, y_fit).predict(X_eval), index=X_eval.index), yeval_raw)
        if np.isfinite(ic) and ic > best[1]: best = (a, ic)
    r = Ridge(alpha=best[0]).fit(X_full, y_full)
    out["ridge"] = round(daily_rank_ic(pd.Series(r.predict(X_te), index=X_te.index), yte_raw), 5)
    log(f"  [{tag}] ridge alpha={best[0]:g} test={out['ridge']}")
    params = dict(objective="regression", learning_rate=0.03, num_leaves=63, feature_fraction=0.6,
                  bagging_fraction=0.7, bagging_freq=1, min_child_samples=200, n_jobs=-1, verbosity=-1)
    dfit = lgb.Dataset(X_fit, label=y_fit)
    bn, bnic = 300, -1e9
    for n in [100, 300, 600, 1000]:
        m = lgb.train(dict(params, seed=0), dfit, num_boost_round=n)
        ic = daily_rank_ic(pd.Series(m.predict(X_eval), index=X_eval.index), yeval_raw)
        if np.isfinite(ic) and ic > bnic: bnic, bn = ic, n
    dfull = lgb.Dataset(X_full, label=y_full); sics = []
    for s in [0, 1, 2]:
        m = lgb.train(dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s), dfull, num_boost_round=bn)
        sics.append(daily_rank_ic(pd.Series(m.predict(X_te), index=X_te.index), yte_raw))
    out["lgbm_mean"] = round(float(np.nanmean(sics)), 5)
    out["lgbm_seeds"] = [round(x, 5) for x in sics]
    out["best_n"] = bn
    out["ceiling"] = round(max(out["ridge"], out["lgbm_mean"]), 5)
    log(f"  [{tag}] lgbm n={bn} mean={out['lgbm_mean']} seeds={out['lgbm_seeds']} CEILING={out['ceiling']}")
    return out

def main():
    t0 = time.time()
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, kernels=1)
    hc = {"class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": {
        "start_time": HS, "end_time": HE, "fit_start_time": HS, "fit_end_time": TREND, "instruments": "csi300",
        "infer_processors": [{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                             {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
        "learn_processors": [{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label", "method": "robust"}}],
        "label": ["Ref($close, -5) / Ref($close, -1) - 1"]}}
    log("building handler..."); h = init_instance_by_config(hc); log("handler %.0fs" % (time.time()-t0))
    Xi = h.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32)
    yi = h.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0].astype(np.float32)
    Xl = h.fetch(col_set="feature", data_key=DataHandlerLP.DK_L).astype(np.float32)
    yl = h.fetch(col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0].astype(np.float32)
    log("DK_I", Xi.shape, "DK_L", Xl.shape)

    # point-in-time per-day CS-rank-pct (centered to [-0.5,0.5]); appended to globals. Caveat: computed on the
    # clipped robust-z feature (tail names tie at +/-3); a pre-clip raw rank is a future refinement.
    def csrank(X):
        log("  building CS-rank (groupby-datetime rank-pct) for", X.shape, "...")
        r = X.groupby(level="datetime").rank(pct=True) - 0.5
        r.columns = [f"{c}__csr" for c in X.columns]
        return r.astype(np.float32)
    Xi_cr = csrank(Xi); Xl_cr = csrank(Xl)
    log("CS-rank built %.0fs" % (time.time()-t0))

    dti = Xi.index.get_level_values("datetime"); dtl = Xl.index.get_level_values("datetime")
    trf = (dtl >= HS) & (dtl <= IFE); ev = (dtl >= IES) & (dtl <= TREND); tr = (dtl >= HS) & (dtl <= TREND); te = (dti >= TS) & (dti <= TE)
    yev = yi.reindex(Xl[ev].index)

    res = {"meta": {"smoke": SMOKE, "model_ref": MODEL_REF, "F_base": Xi.shape[1]}, "variants": {}}
    # BASE = global snapshot (re-confirm ~0.0535); CSRANK = global ++ CS-rank appended.
    XiC = pd.concat([Xi, Xi_cr], axis=1); XlC = pd.concat([Xl, Xl_cr], axis=1)
    log("BASE (global snapshot, F=%d)" % Xi.shape[1])
    res["variants"]["BASE_global"] = fit_score(Xl[trf], yl[trf], Xl[ev], yev, Xl[tr], yl[tr], Xi[te], yi[te], "BASE")
    log("CSRANK (global ++ CS-rank appended, F=%d)" % XiC.shape[1])
    res["variants"]["GLOBAL_plus_CSRANK"] = fit_score(XlC[trf], yl[trf], XlC[ev], yev, XlC[tr], yl[tr], XiC[te], yi[te], "CSR")

    base_c = res["variants"]["BASE_global"]["ceiling"]; csr_c = res["variants"]["GLOBAL_plus_CSRANK"]["ceiling"]
    lift = round(csr_c - base_c, 5); vs_model = round(csr_c - MODEL_REF, 5)
    if csr_c >= MODEL_REF - 0.003: verdict = "IC_REOPENS_STRONG (CS-rank ceiling ~matches/beats model -> L-4 has real headroom)"
    elif lift >= 0.005: verdict = "IC_REOPENS_MODAL (CS-rank lifts the classic ceiling >=+0.005 -> L-4 worth a GPU n=6)"
    else: verdict = "CS_RANK_ADDS_LITTLE (lift <+0.005 even to a tree -> L-4 low-EV, IC stays ~capped)"
    res["verdict"] = {"base_ceiling": base_c, "csrank_ceiling": csr_c, "lift": lift, "csrank_vs_model": vs_model, "zone": verdict}
    json.dump(res, open(OUT / "ceiling_probe_csrank_RESULT.json", "w"), indent=2)
    log("DONE %.0fs base=%.5f csrank=%.5f lift=%+.5f -> %s" % (time.time()-t0, base_c, csr_c, lift, verdict))

if __name__ == "__main__":
    main()
