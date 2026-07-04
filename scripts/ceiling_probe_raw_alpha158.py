# -*- coding: utf-8 -*-
"""
L-1 model-agnostic CEILING PROBE on RAW Alpha158 (REVISED per analysis/tau_scale05_diagnosis/MEMO_20260626.md).

Question it answers: is the RST-MoE backbone's ~0.0786 test RankIC a DATA/SIGNAL ceiling,
or is it RST-MoE-specific (a strong non-RST-MoE learner on the SAME raw features does much better)?

Why this is not the prior PROBE-CEILING kill: that fit ridge on the backbone's FROZEN pooled hidden
state (backbone-conditioned, circular). This fits on the RAW DK_I/DK_L Alpha158 features the model sees.

STRICT anti-leakage discipline:
  * Fit ONLY on the train segment (2008-01-01 .. 2020-03-31). NO per-test-day refit.
  * Hyperparameters tuned ONLY on a train-internal eval slice (2019-01-01 .. 2020-03-31); NEVER on test.
  * "per-day" means daily-Spearman EVALUATION only.
  * >=3 LightGBM seeds.
Two information variants:
  A = day-T snapshot (158 features)                     -> info SUBSET of the model
  B = flattened 8-step window (8 x 158 = 1264 features) -> info-MATCHED to the model input
A flat-yet-low A with a higher B would indicate input-starvation, not a ceiling; running both guards it.

Features byte-match the model: DK_L (train, label CS-z-robust) / DK_I (test, label RAW 5d return),
RobustZScoreNorm(clip_outlier)+Fillna inherited via qlib append semantics. Matches work_flow.py data_conf.

Env toggles: SMOKE=1 (short 2019..2020-06 range, quick pipeline check), RUN_B=0 (skip window variant).
Read-only on data; writes only to analysis/tau_scale05_diagnosis/.
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
RUN_B = os.environ.get("RUN_B", "1") == "1"
OUT = Path(r"C:\Users\60585\PycharmProjects\qibMacV2\analysis\tau_scale05_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_REF = 0.0786  # RST-MoE anchor test RankIC (selection-biased headline; the thing the ceiling is compared against)

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---- segments (anti-leakage) ----
if SMOKE:
    HANDLER_START, HANDLER_END = "2018-01-01", "2020-12-31"
    TRAIN_END = "2019-06-30"; INT_FIT_END = "2019-03-31"; INT_EVAL_START = "2019-04-01"
    TEST_START, TEST_END = "2020-01-01", "2020-06-30"
else:
    HANDLER_START, HANDLER_END = "2008-01-01", "2022-12-31"
    TRAIN_END = "2020-03-31"; INT_FIT_END = "2018-12-31"; INT_EVAL_START = "2019-01-01"
    TEST_START, TEST_END = "2020-07-01", "2022-12-31"
TRAIN_START = HANDLER_START

def daily_rank_ic(pred: pd.Series, label: pd.Series) -> dict:
    """Mean daily Spearman(pred,label) + ICIR over the eval window. pred,label indexed by (datetime,instrument)."""
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

def main():
    t0 = time.time()
    log(f"qlib.init (SMOKE={SMOKE} RUN_B={RUN_B})")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, kernels=1)

    handler_conf = {
        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": HANDLER_START, "end_time": HANDLER_END,
            "fit_start_time": HANDLER_START, "fit_end_time": TRAIN_END,
            "instruments": "csi300",
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label", "method": "robust"}},
            ],
            "label": ["Ref($close, -5) / Ref($close, -1) - 1"],
        },
    }
    log("building Alpha158 handler (expression eval; first build is the slow step)...")
    handler = init_instance_by_config(handler_conf)
    log("handler built in %.1fs" % (time.time()-t0))

    # canonical col_set API (flat columns; matches model_adapter dataset.prepare usage)
    Xi = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32)  # robust-z feat
    yi_raw = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0].astype(np.float32)  # RAW 5d return
    Xl = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_L).astype(np.float32)  # robust-z feat
    yl_z = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0].astype(np.float32)  # CS-z robust
    log("DK_I feat", Xi.shape, "DK_L feat", Xl.shape, "yi_raw", yi_raw.shape, "yl_z", yl_z.shape)
    feat_names = list(Xi.columns)
    F = len(feat_names)
    log(f"n_features F={F}")

    dti = Xi.index.get_level_values("datetime")
    dtl = Xl.index.get_level_values("datetime")

    results = {"meta": {"smoke": SMOKE, "run_b": RUN_B, "model_ref": MODEL_REF, "F": F,
                        "segments": {"train": [TRAIN_START, TRAIN_END], "int_eval": [INT_EVAL_START, TRAIN_END],
                                     "test": [TEST_START, TEST_END]}}, "variants": {}}

    from sklearn.linear_model import Ridge
    import lightgbm as lgb

    def fit_score(X_fit, y_fit, X_eval, yeval_raw, X_full, y_full, X_te, yte_raw, tag):
        """Ridge (alpha tuned on internal-eval RankIC) + LGBM x3 seeds (n_estimators tuned on internal-eval
        RankIC, NOT MSE — stopping on MSE under-powers tree ranking). All tuning on the train-internal slice,
        never on test. Refit on FULL train, score test daily-Spearman RankIC."""
        out = {}
        # ---------- Ridge: alpha tuned on internal-eval RankIC ----------
        best = (None, -1e9)
        for alpha in [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]:
            r = Ridge(alpha=alpha).fit(X_fit, y_fit)
            p = pd.Series(r.predict(X_eval), index=X_eval.index)
            ic = daily_rank_ic(p, yeval_raw)["rank_ic"]
            if np.isfinite(ic) and ic > best[1]:
                best = (alpha, ic)
        alpha = best[0]
        r = Ridge(alpha=alpha).fit(X_full, y_full)               # refit on FULL train
        pte = pd.Series(r.predict(X_te), index=X_te.index)
        out["ridge"] = {**daily_rank_ic(pte, yte_raw), "alpha": alpha, "int_eval_rank_ic": best[1]}
        log(f"  [{tag}] ridge alpha={alpha:g} int_eval={best[1]:.5f} test_rank_ic={out['ridge']['rank_ic']:.5f}")
        # ---------- LightGBM: n_estimators tuned on internal-eval RankIC (seed0), then 3 seeds on full train ----------
        params = dict(objective="regression", learning_rate=0.03, num_leaves=63,
                      feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
                      min_child_samples=200, max_depth=-1, n_jobs=-1, verbosity=-1)
        dfit = lgb.Dataset(X_fit, label=y_fit)
        best_n, best_nic = 300, -1e9
        for n in [100, 300, 600, 1000]:
            m = lgb.train(dict(params, seed=0), dfit, num_boost_round=n)
            ic = daily_rank_ic(pd.Series(m.predict(X_eval), index=X_eval.index), yeval_raw)["rank_ic"]
            log(f"  [{tag}] lgbm n={n} int_eval_rank_ic={ic:.5f}")
            if np.isfinite(ic) and ic > best_nic:
                best_nic, best_n = ic, n
        dfull = lgb.Dataset(X_full, label=y_full)
        seed_ics = []
        for seed in [0, 1, 2]:
            m = lgb.train(dict(params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed),
                          dfull, num_boost_round=best_n)
            ic = daily_rank_ic(pd.Series(m.predict(X_te), index=X_te.index), yte_raw)["rank_ic"]
            seed_ics.append(ic)
            log(f"  [{tag}] lgbm seed={seed} n={best_n} test_rank_ic={ic:.5f}")
        out["lgbm"] = {"seed_rank_ics": seed_ics, "rank_ic_mean": float(np.nanmean(seed_ics)),
                       "rank_ic_std": float(np.nanstd(seed_ics)), "best_n": best_n, "int_eval_rank_ic": best_nic}
        out["ceiling"] = float(np.nanmax([out["ridge"]["rank_ic"], out["lgbm"]["rank_ic_mean"]]))
        log(f"  [{tag}] lgbm mean={out['lgbm']['rank_ic_mean']:.5f}+/-{out['lgbm']['rank_ic_std']:.5f}  CEILING={out['ceiling']:.5f}")
        return out

    # ================= VARIANT A: snapshot =================
    log("VARIANT A (snapshot, F=%d)" % F)
    tr = (dtl >= TRAIN_START) & (dtl <= TRAIN_END)
    fit = (dtl >= TRAIN_START) & (dtl <= INT_FIT_END)
    ev = (dtl >= INT_EVAL_START) & (dtl <= TRAIN_END)
    te = (dti >= TEST_START) & (dti <= TEST_END)
    Xev = Xl[ev]
    yev_raw_aligned = yi_raw.reindex(Xev.index)   # raw label for RankIC-based internal selection
    results["variants"]["A_snapshot"] = fit_score(
        Xl[fit], yl_z[fit], Xev, yev_raw_aligned,
        Xl[tr], yl_z[tr], Xi[te], yi_raw[te], "A")
    json.dump(results, open(OUT / "ceiling_probe_RESULT.json", "w"), indent=2)

    # ================= VARIANT B: 8-step window =================
    if RUN_B:
        try:
            log("VARIANT B (8-step window, F=%d -> %d)" % (F, 8*F))
            from numpy.lib.stride_tricks import sliding_window_view
            WIN = 8
            # build windows on FULL DK_I feature frame (sorted by instrument,datetime), end-row = day T
            Xs = Xi.sort_index(level=["instrument", "datetime"])
            arr = Xs.to_numpy(np.float32)
            idx = Xs.index
            inst = idx.get_level_values("instrument").to_numpy()
            N = arr.shape[0]
            change = np.empty(N, bool); change[0] = True; change[1:] = inst[1:] != inst[:-1]
            grp_start = np.maximum.accumulate(np.where(change, np.arange(N), 0))
            sw = sliding_window_view(arr, WIN, axis=0)          # [N-WIN+1, F, WIN], end-row = j+WIN-1
            ends = np.arange(WIN - 1, N)
            same = (ends - grp_start[ends]) >= (WIN - 1)         # full window within one instrument
            Xb_all = sw[same].transpose(0, 2, 1).reshape(same.sum(), WIN * F).astype(np.float32)
            idx_b = idx[ends][same]
            del sw, arr
            dt_b = idx_b.get_level_values("datetime")
            Xb_df = pd.DataFrame(Xb_all, index=idx_b)
            del Xb_all
            log("window matrix", Xb_df.shape)
            # targets: train CS-z label (reindex from dfl), test raw label (from dfi)
            yb_z = yl_z.reindex(idx_b)
            yb_raw = yi_raw.reindex(idx_b)
            trb = (dt_b >= TRAIN_START) & (dt_b <= TRAIN_END) & yb_z.notna().to_numpy()
            fitb = (dt_b >= TRAIN_START) & (dt_b <= INT_FIT_END) & yb_z.notna().to_numpy()
            evb = (dt_b >= INT_EVAL_START) & (dt_b <= TRAIN_END) & yb_z.notna().to_numpy()
            teb = (dt_b >= TEST_START) & (dt_b <= TEST_END) & yb_raw.notna().to_numpy()
            results["variants"]["B_window"] = fit_score(
                Xb_df[fitb], yb_z[fitb], Xb_df[evb], yb_raw[evb],
                Xb_df[trb], yb_z[trb], Xb_df[teb], yb_raw[teb], "B")
            json.dump(results, open(OUT / "ceiling_probe_RESULT.json", "w"), indent=2)
        except Exception as e:
            log("VARIANT B FAILED:", repr(e))
            results["variants"]["B_window"] = {"error": repr(e), "traceback": traceback.format_exc()}

    # ================= VERDICT (3-zone) =================
    ceils = [v["ceiling"] for v in results["variants"].values() if isinstance(v, dict) and "ceiling" in v]
    max_ceil = float(np.nanmax(ceils)) if ceils else float("nan")
    if max_ceil >= MODEL_REF + 0.02:
        verdict = "ARCHITECTURE_TRACK_REOPENS"
    elif max_ceil <= MODEL_REF + 0.005:
        verdict = "DATA_SIGNAL_CEILING__redirect_to_portfolio_IR"
    else:
        verdict = "REAL_BUT_MODAL__keep_architecture_downprioritized"
    results["verdict"] = {"max_ceiling": max_ceil, "model_ref": MODEL_REF, "delta": max_ceil - MODEL_REF, "zone": verdict}
    json.dump(results, open(OUT / "ceiling_probe_RESULT.json", "w"), indent=2)

    # markdown summary
    md = ["# L-1 Ceiling Probe — RAW Alpha158 (model-agnostic)", "",
          f"- SMOKE={SMOKE}  RUN_B={RUN_B}  F={F}  model_ref(RankIC)={MODEL_REF}",
          f"- train {TRAIN_START}..{TRAIN_END} | int-eval {INT_EVAL_START}..{TRAIN_END} | test {TEST_START}..{TEST_END}",
          f"- **max ceiling = {max_ceil:.5f}  (Δ vs model {max_ceil-MODEL_REF:+.5f})  → {verdict}**", ""]
    for name, v in results["variants"].items():
        if "ceiling" not in v:
            md.append(f"## {name}: ERROR {v.get('error')}"); continue
        md += [f"## {name}  ceiling={v['ceiling']:.5f}",
               f"- ridge: test RankIC {v['ridge']['rank_ic']:.5f} (alpha={v['ridge']['alpha']}, int-eval {v['ridge']['int_eval_rank_ic']:.5f})",
               f"- lgbm: test RankIC {v['lgbm']['rank_ic_mean']:.5f} ± {v['lgbm']['rank_ic_std']:.5f} (n={v['lgbm']['best_n']}, seeds {['%.5f'%x for x in v['lgbm']['seed_rank_ics']]})", ""]
    (OUT / "ceiling_probe_RESULT.md").write_text("\n".join(md), encoding="utf-8")
    log("DONE in %.1fs -> %s" % (time.time()-t0, verdict))
    log("results:", json.dumps(results["verdict"]))

if __name__ == "__main__":
    main()
