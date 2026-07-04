# -*- coding: utf-8 -*-
"""
CS-RANK CEILING PROBE (extends scripts/ceiling_probe_raw_alpha158.py) — the DECISIVE cheap test for
whether lever L-4 (cross-sectional INPUT representation) has headroom.

WHY: the L-1 probe found classic-learner RankIC max 0.0592 < model 0.0786 and the team read it as
"DATA_SIGNAL_CEILING -> stop IC". That probe used ONLY the per-stock RobustZScoreNorm features and
NEVER tested a cross-sectional input. A GBDT ceiling BELOW the model is a LOWER bound on signal, not an
upper bound. This probe adds the per-day cross-sectional RELATIVE COORDINATE (CS-rank) the model is
structurally blind to and asks: does it lift the classic-learner ceiling above 0.0592?

  - If R+CSR ceiling >> R ceiling (and >= ~0.0592 + MDE) -> cross-sectional signal exists that the
    per-stock features do not expose -> L-4 has demonstrated headroom -> worth GPU (n=6 paired L-4).
  - If R+CSR ~= R ceiling -> the relative coordinate adds nothing on this data -> L-4 bounded-dead, and
    the data-cap verdict is RESCUED into a genuine ceiling test (the falsification the original skipped).

ARMS (snapshot, day-T; CS-rank is inherently a day-T cross-sectional op):
  R     = raw 158 clipped robust-z feature        (reproduces L-1 variant-A ~0.0535 LGBM / 0.0444 ridge)
  CSR   = CS-rank 158 (per-day pct-rank - 0.5)     (the pure relative coordinate)
  R+CSR = concat(R, CSR) = 316                      (the L-4 APPEND test: does CS-rank lift the ceiling?)
  [optional, RUN_CSZ=1] CSZ / R+CSZ                 (per-day cross-sectional z as an alternative coord)

LEAKAGE DISCIPLINE (inherited from L-1, code-identical fit_score):
  * Fit ONLY on train 2008-01..2020-03. NO per-test-day refit. Tune (ridge alpha / LGBM n) ONLY on the
    train-internal slice 2019-01..2020-03 RankIC. >=3 LGBM seeds.
  * CS-rank/CS-z are computed PER-DAY point-in-time on day-T peers only (groupby datetime) -> no lookahead.
    Computed on UNCLIPPED robust-z (clip_outlier=False) so the +/-3 clip does not collapse tail ranks.
    Ranking is invariant to the per-feature monotone global transform, so rank(unclipped robust-z) ==
    rank(raw expression) within a day; only the clip would have introduced tail ties.
  * CS-rank is taken over the FULL daily universe (DK_I, before label-drop) -> correct PIT cross-section.

Env: SMOKE=1 (short 2018..2020-06 pipeline check), RUN_CSZ=1 (add CS-z arms). Read-only on data; writes
only analysis/tau_scale05_diagnosis/ceiling_probe_cs_rank_RESULT.{json,md}.
"""
import os, sys, json, time, traceback
import numpy as np
import pandas as pd
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

SMOKE = os.environ.get("SMOKE", "0") == "1"
RUN_CSZ = os.environ.get("RUN_CSZ", "0") == "1"
OUT = Path(r"C:\Users\60585\PycharmProjects\qibMacV2\analysis\tau_scale05_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_REF = 0.0786          # RST-MoE anchor (selection-biased headline)
RAW_WINDOW_CEIL = 0.0592    # L-1 variant-B (8x158 window) ceiling -- the number to beat
MDE = 0.005                 # n=6 minimum-detectable-effect floor; used for the headroom verdict

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

if SMOKE:
    HANDLER_START, HANDLER_END = "2018-01-01", "2020-12-31"
    TRAIN_END = "2019-06-30"; INT_EVAL_START = "2019-04-01"
    TEST_START, TEST_END = "2020-01-01", "2020-06-30"
    INT_FIT_END = "2019-03-31"
else:
    HANDLER_START, HANDLER_END = "2008-01-01", "2022-12-31"
    TRAIN_END = "2020-03-31"; INT_EVAL_START = "2019-01-01"
    TEST_START, TEST_END = "2020-07-01", "2022-12-31"
    INT_FIT_END = "2018-12-31"
TRAIN_START = HANDLER_START

def daily_rank_ic(pred: pd.Series, label: pd.Series) -> dict:
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ics = []
    for _, sub in df.groupby(level="datetime"):
        if len(sub) < 5:
            continue
        ics.append(sub["p"].corr(sub["y"], method="spearman"))
    ics = np.array([x for x in ics if np.isfinite(x)], dtype=float)
    if ics.size == 0:
        return {"rank_ic": float("nan"), "icir": float("nan"), "n_days": 0}
    return {"rank_ic": float(ics.mean()), "icir": float(ics.mean()/ics.std()) if ics.std()>0 else float("nan"),
            "n_days": int(ics.size)}

def make_handler(clip: bool):
    conf = {
        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": HANDLER_START, "end_time": HANDLER_END,
            "fit_start_time": HANDLER_START, "fit_end_time": TRAIN_END,
            "instruments": "csi300",
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": clip}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label", "method": "robust"}},
            ],
            "label": ["Ref($close, -5) / Ref($close, -1) - 1"],
        },
    }
    return init_instance_by_config(conf)

def main():
    t0 = time.time()
    log(f"qlib.init (SMOKE={SMOKE} RUN_CSZ={RUN_CSZ})")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, kernels=1)

    log("building CLIPPED Alpha158 handler (raw arm + labels)...")
    h_clip = make_handler(clip=True)
    Xi = h_clip.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32)   # clipped robust-z, DK_I
    yi_raw = h_clip.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0].astype(np.float32)
    Xl = h_clip.fetch(col_set="feature", data_key=DataHandlerLP.DK_L).astype(np.float32)   # clipped robust-z, DK_L
    yl_z = h_clip.fetch(col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0].astype(np.float32)
    F = Xi.shape[1]
    log("clipped built %.1fs  DK_I" % (time.time()-t0), Xi.shape, "DK_L", Xl.shape, "F=%d" % F)

    log("building UNCLIPPED Alpha158 handler (CS-rank source; expressions cached -> should be fast)...")
    h_nc = make_handler(clip=False)
    Xnc = h_nc.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32)    # UNclipped robust-z, DK_I
    Xnc = Xnc.reindex(Xi.index)
    log("unclipped built %.1fs" % (time.time()-t0), Xnc.shape)

    # ---- cross-sectional coordinates (per-day, PIT, full universe) ----
    log("computing CS-rank (per-day pct-rank on unclipped features)...")
    CSR = Xnc.groupby(level="datetime").rank(pct=True).astype(np.float32) - np.float32(0.5)
    CSR.columns = [f"{c}_csr" for c in CSR.columns]
    log("CS-rank done %.1fs" % (time.time()-t0), CSR.shape, "range[", float(np.nanmin(CSR.values)), float(np.nanmax(CSR.values)), "]")
    CSZ = None
    if RUN_CSZ:
        log("computing CS-z (per-day cross-sectional z on unclipped features)...")
        g = Xnc.groupby(level="datetime")
        CSZ = ((Xnc - g.transform("mean")) / (g.transform("std") + 1e-6)).clip(-5, 5).astype(np.float32)
        CSZ.columns = [f"{c}_csz" for c in CSZ.columns]
        log("CS-z done %.1fs" % (time.time()-t0))
    del Xnc

    dti = Xi.index.get_level_values("datetime")
    dtl = Xl.index.get_level_values("datetime")
    tr = (dtl >= TRAIN_START) & (dtl <= TRAIN_END)
    fitm = (dtl >= TRAIN_START) & (dtl <= INT_FIT_END)
    ev = (dtl >= INT_EVAL_START) & (dtl <= TRAIN_END)
    te = (dti >= TEST_START) & (dti <= TEST_END)
    yev_raw = yi_raw.reindex(Xl[ev].index)

    results = {"meta": {"smoke": SMOKE, "run_csz": RUN_CSZ, "model_ref": MODEL_REF,
                        "raw_window_ceiling": RAW_WINDOW_CEIL, "mde": MDE, "F": F,
                        "segments": {"train": [TRAIN_START, TRAIN_END], "int_eval": [INT_EVAL_START, TRAIN_END],
                                     "test": [TEST_START, TEST_END]}}, "arms": {}}

    from sklearn.linear_model import Ridge
    import lightgbm as lgb

    def fit_score(X_fit, y_fit, X_eval, yeval_raw, X_full, y_full, X_te, yte_raw, tag):
        out = {}
        best = (None, -1e9)
        for alpha in [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
            r = Ridge(alpha=alpha).fit(X_fit, y_fit)
            ic = daily_rank_ic(pd.Series(r.predict(X_eval), index=X_eval.index), yeval_raw)["rank_ic"]
            if np.isfinite(ic) and ic > best[1]:
                best = (alpha, ic)
        alpha = best[0]
        r = Ridge(alpha=alpha).fit(X_full, y_full)
        out["ridge"] = {**daily_rank_ic(pd.Series(r.predict(X_te), index=X_te.index), yte_raw), "alpha": alpha,
                        "int_eval_rank_ic": best[1]}
        log(f"  [{tag}] ridge alpha={alpha:g} int_eval={best[1]:.5f} test={out['ridge']['rank_ic']:.5f}")
        params = dict(objective="regression", learning_rate=0.03, num_leaves=63,
                      feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
                      min_child_samples=200, max_depth=-1, n_jobs=-1, verbosity=-1)
        dfit = lgb.Dataset(X_fit, label=y_fit)
        best_n, best_nic = 300, -1e9
        for n in [100, 300, 600, 1000]:
            m = lgb.train(dict(params, seed=0), dfit, num_boost_round=n)
            ic = daily_rank_ic(pd.Series(m.predict(X_eval), index=X_eval.index), yeval_raw)["rank_ic"]
            log(f"  [{tag}] lgbm n={n} int_eval={ic:.5f}")
            if np.isfinite(ic) and ic > best_nic:
                best_nic, best_n = ic, n
        dfull = lgb.Dataset(X_full, label=y_full)
        seed_ics = []
        for seed in [0, 1, 2]:
            m = lgb.train(dict(params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed),
                          dfull, num_boost_round=best_n)
            seed_ics.append(daily_rank_ic(pd.Series(m.predict(X_te), index=X_te.index), yte_raw)["rank_ic"])
            log(f"  [{tag}] lgbm seed={seed} n={best_n} test={seed_ics[-1]:.5f}")
        out["lgbm"] = {"seed_rank_ics": seed_ics, "rank_ic_mean": float(np.nanmean(seed_ics)),
                       "rank_ic_std": float(np.nanstd(seed_ics)), "best_n": best_n, "int_eval_rank_ic": best_nic}
        out["ceiling"] = float(np.nanmax([out["ridge"]["rank_ic"], out["lgbm"]["rank_ic_mean"]]))
        log(f"  [{tag}] CEILING={out['ceiling']:.5f}")
        return out

    def run_arm(name, Xa_i):
        Xa_l = Xa_i.reindex(Xl.index)
        results["arms"][name] = fit_score(Xa_l[fitm], yl_z[fitm], Xa_l[ev], yev_raw,
                                          Xa_l[tr], yl_z[tr], Xa_i[te], yi_raw[te], name)
        json.dump(results, open(OUT / "ceiling_probe_cs_rank_RESULT.json", "w"), indent=2)

    log("ARM R (raw 158, reproduce baseline)"); run_arm("R", Xi)
    log("ARM CSR (CS-rank 158)"); run_arm("CSR", CSR)
    log("ARM R+CSR (316, the L-4 append test)"); run_arm("R+CSR", pd.concat([Xi, CSR], axis=1))
    if RUN_CSZ and CSZ is not None:
        log("ARM CSZ (CS-z 158)"); run_arm("CSZ", CSZ)
        log("ARM R+CSZ (316)"); run_arm("R+CSZ", pd.concat([Xi, CSZ], axis=1))

    # ---- verdict: does the cross-sectional coordinate lift the ceiling? ----
    R_ceil = results["arms"]["R"]["ceiling"]
    RCSR_ceil = results["arms"]["R+CSR"]["ceiling"]
    CSR_ceil = results["arms"]["CSR"]["ceiling"]
    lift_vs_R = RCSR_ceil - R_ceil
    lift_vs_window = RCSR_ceil - RAW_WINDOW_CEIL
    if lift_vs_R >= MDE or RCSR_ceil >= RAW_WINDOW_CEIL + MDE:
        zone = "CS_INPUT_HAS_HEADROOM__L4_reopens"
    elif lift_vs_R <= 0.003 and RCSR_ceil <= RAW_WINDOW_CEIL + 0.003:
        zone = "CS_INPUT_REDUNDANT__L4_bounded_dead"
    else:
        zone = "MODAL__inconclusive_consider_window_or_n6"
    results["verdict"] = {"R_ceiling": R_ceil, "CSR_ceiling": CSR_ceil, "RplusCSR_ceiling": RCSR_ceil,
                          "lift_vs_R_snapshot": lift_vs_R, "lift_vs_raw_window_0p0592": lift_vs_window,
                          "model_ref": MODEL_REF, "zone": zone}
    json.dump(results, open(OUT / "ceiling_probe_cs_rank_RESULT.json", "w"), indent=2)

    md = ["# CS-rank Ceiling Probe — does the cross-sectional coordinate lift the classic-learner ceiling?", "",
          f"- SMOKE={SMOKE}  F={F}  model_ref={MODEL_REF}  raw_window_ceiling(L-1 B)={RAW_WINDOW_CEIL}",
          f"- train {TRAIN_START}..{TRAIN_END} | int-eval {INT_EVAL_START}..{TRAIN_END} | test {TEST_START}..{TEST_END}",
          f"- **R={R_ceil:.5f}  CSR={CSR_ceil:.5f}  R+CSR={RCSR_ceil:.5f}**",
          f"- **lift(R+CSR vs R snapshot) = {lift_vs_R:+.5f} | lift(R+CSR vs raw-window 0.0592) = {lift_vs_window:+.5f}**",
          f"- **VERDICT: {results['verdict']['zone']}**", ""]
    for name, v in results["arms"].items():
        md += [f"## {name}  ceiling={v['ceiling']:.5f}",
               f"- ridge {v['ridge']['rank_ic']:.5f} (alpha={v['ridge']['alpha']}, int-eval {v['ridge']['int_eval_rank_ic']:.5f})",
               f"- lgbm {v['lgbm']['rank_ic_mean']:.5f} ± {v['lgbm']['rank_ic_std']:.5f} (n={v['lgbm']['best_n']}, seeds {['%.5f'%x for x in v['lgbm']['seed_rank_ics']]})", ""]
    (OUT / "ceiling_probe_cs_rank_RESULT.md").write_text("\n".join(md), encoding="utf-8")
    log("DONE in %.1fs -> %s" % (time.time()-t0, zone))
    log("verdict:", json.dumps(results["verdict"]))

if __name__ == "__main__":
    main()
