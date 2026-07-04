# -*- coding: utf-8 -*-
"""
SEED-ENSEMBLE IC PROBE (zero-GPU, MSE-preserving). Tests a standard, deployable IC lever the project has
NOT explicitly tried: averaging the PREDICTIONS of independently-trained seeds. (Memory only dismissed
within-run EMA/SWA, a DIFFERENT and weaker thing; cross-seed prediction ensembling reduces the per-seed
estimation noise that the project itself measures as rank_ic std ~0.0033.)

Loads each saved seed run's artifacts/pred.pkl + label.pkl from mlruns (no retrain), computes per-seed
daily-Spearman RankIC, then the ensemble RankIC for:
  - score-mean   (average raw predictions)
  - rank-mean    (average per-day ranks; scale-robust, usually best for a rank metric)
and the ensemble-size scaling curve k=1..N. Honest baseline = MEAN single-seed (NOT the selection-inflated
best seed). Also reports the section-5 IC-aligned monitors (decile monotonicity, precision@30 aligned to
topk=30) for the ensemble vs mean-single-seed.

Settings probed: tau_scale_05 (n=6, the named config) + readout_full_g012_d1pma (n=3, the baseline anchor).
Writes analysis/tau_scale05_diagnosis/ensemble_seed_probe_RESULT.{json,md}.
"""
import os, re, glob, json, time
import numpy as np
import pandas as pd

ROOT = r"C:\Users\60585\PycharmProjects\qibMacV2"
MLRUNS = os.path.join(ROOT, "mlruns")
OUT = os.path.join(ROOT, "analysis", "tau_scale05_diagnosis")
SETTINGS = ["tau_scale_05", "readout_full_g012_d1pma"]
TOPK = 30

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def canon(rs):
    return re.sub(r"_scale\d+", "", re.sub(r"_seed\d+", "", rs))

def read_param(run, name):
    p = os.path.join(run, "params", name)
    return open(p).read().strip() if os.path.exists(p) else None

def find_runs(target):
    """canon(run_setting)==target -> {seed: run_dir}, latest-by-mtime per seed."""
    out = {}
    for run in glob.glob(os.path.join(MLRUNS, "*", "*")):
        if not os.path.isdir(os.path.join(run, "artifacts")):
            continue
        rs = read_param(run, "kwargs.trainer_config.run_setting")
        if not rs or canon(rs) != target:
            continue
        if not os.path.exists(os.path.join(run, "artifacts", "pred.pkl")):
            continue
        try:
            seed = int(read_param(run, "kwargs.trainer_config.seed"))
        except Exception:
            continue
        prev = out.get(seed)
        if prev is None or os.path.getmtime(run) > prev[1]:
            out[seed] = (run, os.path.getmtime(run))
    return {s: v[0] for s, v in sorted(out.items())}

def load_series(path):
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj.astype(float)

def daily_rank_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ics = [s["p"].corr(s["y"], method="spearman") for _, s in df.groupby(level="datetime") if len(s) >= 5]
    ics = np.array([x for x in ics if np.isfinite(x)], float)
    return (float(ics.mean()), float(ics.mean()/ics.std()) if ics.std() > 0 else float("nan"), int(ics.size))

def daily_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ics = [s["p"].corr(s["y"]) for _, s in df.groupby(level="datetime") if len(s) >= 5]
    ics = np.array([x for x in ics if np.isfinite(x)], float)
    return (float(ics.mean()), float(ics.mean()/ics.std()) if ics.std() > 0 else float("nan"))

def decile_monotonicity(pred, label, q=10):
    """Spearman(decile_index, mean_fwd_return) averaged over days; +precision@TOPK."""
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    mons, precs = [], []
    for _, s in df.groupby(level="datetime"):
        if len(s) < q * 2:
            continue
        dec = pd.qcut(s["p"].rank(method="first"), q, labels=False)
        dmean = s.groupby(dec)["y"].mean()
        if dmean.notna().sum() >= q - 1:
            mons.append(pd.Series(dmean.index).corr(pd.Series(dmean.values), method="spearman"))
        k = min(TOPK, len(s) // 2)
        top = s.nlargest(k, "p")["y"]; thr = s["y"].quantile(1 - k / len(s))
        precs.append(float((top >= thr).mean()))
    return (float(np.nanmean(mons)) if mons else float("nan"),
            float(np.nanmean(precs)) if precs else float("nan"))

def rank_frame(preds):
    """per-day rank-percentile of each seed's prediction; returns aligned mean-rank series for any subset."""
    ranked = {}
    for s, p in preds.items():
        ranked[s] = p.groupby(level="datetime").rank(pct=True)
    return pd.DataFrame(ranked)

# credible IC settings to pool into a DIVERSE cross-setting ensemble (exclude KILLED stock-expert variants
# that HURT IC — h2_*, gbb_* — which would drag the ensemble down)
DIVERSE = ["base", "tau_scale_05", "tau_scale_10", "readout_full_g012_d1pma", "full_135"]

def main():
    results = {"settings": {}}
    diverse_pool = {}   # (setting,seed) -> pred ; collected for the cross-setting ensemble
    diverse_label = None
    for setting in SETTINGS + [s for s in DIVERSE if s not in SETTINGS]:
        runs = find_runs(setting)
        log(f"{setting}: {len(runs)} seed runs -> {list(runs)}")
        if len(runs) < 1:
            continue
        preds, label = {}, None
        per_seed = {}
        for seed, run in runs.items():
            p = load_series(os.path.join(run, "artifacts", "pred.pkl"))
            y = load_series(os.path.join(run, "artifacts", "label.pkl"))
            preds[seed] = p
            if label is None:
                label = y
            ric, ricir, nd = daily_rank_ic(p, y)
            ic, icir = daily_ic(p, y)
            per_seed[seed] = {"rank_ic": ric, "rank_icir": ricir, "ic": ic, "icir": icir, "n_days": nd}
            log(f"  seed {seed}: rank_ic={ric:.5f} ic={ic:.5f}")
            if setting in DIVERSE:
                diverse_pool[(setting, seed)] = p
                if diverse_label is None:
                    diverse_label = y
        if setting not in SETTINGS or len(runs) < 2:
            continue
        seeds = sorted(preds)
        # align all preds + label on common index
        P = pd.DataFrame({s: preds[s] for s in seeds}).dropna()
        lab = label.reindex(P.index)
        score_mean = P.mean(axis=1)
        RF = rank_frame({s: preds[s] for s in seeds}).reindex(P.index)
        rank_mean = RF.mean(axis=1)
        sm_ric, sm_ricir, _ = daily_rank_ic(score_mean, lab)
        rm_ric, rm_ricir, _ = daily_rank_ic(rank_mean, lab)
        sm_ic, sm_icir = daily_ic(score_mean, lab)
        per_ric = [per_seed[s]["rank_ic"] for s in seeds]
        mean_single = float(np.mean(per_ric)); best_single = float(np.max(per_ric))
        # ensemble-size scaling: average first k seeds (rank-mean), k=1..N
        scaling = []
        for k in range(1, len(seeds) + 1):
            sub = RF[seeds[:k]].mean(axis=1)
            scaling.append(round(daily_rank_ic(sub, lab)[0], 5))
        # section-5 monitors: ensemble (rank-mean) vs mean-single proxy (use best seed's monitors as ref)
        ens_mon, ens_prec = decile_monotonicity(rank_mean, lab)
        s0 = seeds[int(np.argmax(per_ric))]
        bs_mon, bs_prec = decile_monotonicity(preds[s0], lab)
        results["settings"][setting] = {
            "seeds": seeds, "per_seed": per_seed,
            "mean_single_rank_ic": mean_single, "best_single_rank_ic": best_single,
            "ensemble_score_mean_rank_ic": sm_ric, "ensemble_score_mean_rank_icir": sm_ricir, "ensemble_score_mean_ic": sm_ic,
            "ensemble_rank_mean_rank_ic": rm_ric, "ensemble_rank_mean_rank_icir": rm_ricir,
            "lift_vs_mean_single": rm_ric - mean_single, "lift_vs_best_single": rm_ric - best_single,
            "scaling_rank_mean_k1toN": scaling,
            "monitors": {"ensemble_decile_monotonicity": ens_mon, "ensemble_precision@30": ens_prec,
                         "bestseed_decile_monotonicity": bs_mon, "bestseed_precision@30": bs_prec},
        }
        log(f"  >> mean_single={mean_single:.5f} best={best_single:.5f} | score-mean={sm_ric:.5f} rank-mean={rm_ric:.5f} "
            f"| lift_vs_mean={rm_ric-mean_single:+.5f} lift_vs_best={rm_ric-best_single:+.5f}")
        log(f"  >> scaling k=1..N: {scaling}")
        json.dump(results, open(os.path.join(OUT, "ensemble_seed_probe_RESULT.json"), "w"), indent=2)

    # ===== DIVERSE cross-setting ensemble (more decorrelated base learners than same-config seeds) =====
    if len(diverse_pool) >= 2:
        keys = sorted(diverse_pool, key=lambda k: (k[0], k[1]))
        per_run_ric = {k: daily_rank_ic(diverse_pool[k], diverse_label.reindex(diverse_pool[k].index))[0] for k in keys}
        best_run = max(per_run_ric, key=per_run_ric.get); best_run_ric = per_run_ric[best_run]
        RFall = pd.DataFrame({f"{s}|{sd}": diverse_pool[(s, sd)].groupby(level="datetime").rank(pct=True) for (s, sd) in keys}).dropna()
        lab_d = diverse_label.reindex(RFall.index)
        all_ens = RFall.mean(axis=1)
        all_ric, all_ricir, _ = daily_rank_ic(all_ens, lab_d)
        best_per_setting = {}
        for k in keys:
            s = k[0]
            if s not in best_per_setting or per_run_ric[k] > per_run_ric[best_per_setting[s]]:
                best_per_setting[s] = k
        sel = list(best_per_setting.values())
        RFsel = pd.DataFrame({f"{s}|{sd}": diverse_pool[(s, sd)].groupby(level="datetime").rank(pct=True) for (s, sd) in sel}).dropna()
        sel_ric = daily_rank_ic(RFsel.mean(axis=1), diverse_label.reindex(RFsel.index))[0]
        # UNBIASED equal-weight-per-setting: average each setting's seed-ranks, then weight settings equally (no test-selection)
        per_set_rank = {}
        for s in sorted(set(k[0] for k in keys)):
            cols = [diverse_pool[k].groupby(level="datetime").rank(pct=True) for k in keys if k[0] == s]
            per_set_rank[s] = pd.concat(cols, axis=1).mean(axis=1)
        EQW = pd.DataFrame(per_set_rank).dropna()
        eqw_ric = daily_rank_ic(EQW.mean(axis=1), diverse_label.reindex(EQW.index))[0]
        d_mon, d_prec = decile_monotonicity(all_ens, lab_d)
        results["diverse_ensemble"] = {
            "n_runs": len(keys), "settings": sorted(set(s for s, _ in keys)),
            "best_single_run": f"{best_run[0]}|{best_run[1]}", "best_single_rank_ic": best_run_ric,
            "all_runs_rank_mean_rank_ic": all_ric, "all_runs_rank_icir": all_ricir,
            "best_per_setting_rank_mean_rank_ic": sel_ric, "best_per_setting_runs": [f"{s}|{sd}" for s, sd in sel],
            "eqw_per_setting_rank_ic": eqw_ric, "lift_eqw_vs_best_single": eqw_ric - best_run_ric,
            "lift_allruns_vs_best_single": all_ric - best_run_ric, "lift_bestpersetting_vs_best_single": sel_ric - best_run_ric,
            "monitors": {"decile_monotonicity": d_mon, "precision@30": d_prec},
        }
        log(f"DIVERSE: {len(keys)} runs across {sorted(set(s for s,_ in keys))}")
        log(f"  best single run = {best_run[0]}|{best_run[1]} rank_ic={best_run_ric:.5f}")
        log(f"  all-runs rank-mean = {all_ric:.5f} (lift vs best single {all_ric-best_run_ric:+.5f})")
        log(f"  eqw-per-setting (UNBIASED, all seeds) = {eqw_ric:.5f} (lift vs best single {eqw_ric-best_run_ric:+.5f})")
        log(f"  best-per-setting (test-selected) = {sel_ric:.5f} (lift vs best single {sel_ric-best_run_ric:+.5f})")
        json.dump(results, open(os.path.join(OUT, "ensemble_seed_probe_RESULT.json"), "w"), indent=2)

    md = ["# Seed-ensemble IC probe (zero-GPU, MSE-preserving)", ""]
    de = results.get("diverse_ensemble")
    if de:
        md += ["## DIVERSE cross-setting ensemble",
               f"- pool: {de['n_runs']} runs across {de['settings']}",
               f"- best single run: {de['best_single_run']} rank_ic {de['best_single_rank_ic']:.5f}",
               f"- **all-runs rank-mean {de['all_runs_rank_mean_rank_ic']:.5f} (lift vs best single {de['lift_allruns_vs_best_single']:+.5f}, ICIR {de['all_runs_rank_icir']:.4f})**",
               f"- **best-per-setting rank-mean {de['best_per_setting_rank_mean_rank_ic']:.5f} (lift vs best single {de['lift_bestpersetting_vs_best_single']:+.5f})**",
               f"- monitors: decile-monotonicity {de['monitors']['decile_monotonicity']:.4f} / precision@30 {de['monitors']['precision@30']:.4f}", ""]
    for setting, r in results["settings"].items():
        if "error" in r:
            md += [f"## {setting}: {r['error']}", ""]; continue
        md += [f"## {setting}  (n={len(r['seeds'])}, seeds {r['seeds']})",
               f"- per-seed rank_ic: {[round(r['per_seed'][s]['rank_ic'],5) for s in r['seeds']]}",
               f"- **mean-single {r['mean_single_rank_ic']:.5f} | best-single {r['best_single_rank_ic']:.5f}**",
               f"- **ensemble score-mean {r['ensemble_score_mean_rank_ic']:.5f} | rank-mean {r['ensemble_rank_mean_rank_ic']:.5f}**",
               f"- **lift(rank-mean vs MEAN-single) = {r['lift_vs_mean_single']:+.5f} | lift vs BEST-single = {r['lift_vs_best_single']:+.5f}**",
               f"- ensemble rank_icir {r['ensemble_rank_mean_rank_icir']:.4f} | score-mean ic {r['ensemble_score_mean_ic']:.5f}",
               f"- ensemble-size scaling (rank-mean, k=1..N): {r['scaling_rank_mean_k1toN']}",
               f"- monitors: ensemble decile-monotonicity {r['monitors']['ensemble_decile_monotonicity']:.4f} / precision@30 {r['monitors']['ensemble_precision@30']:.4f} "
               f"(vs best-seed {r['monitors']['bestseed_decile_monotonicity']:.4f} / {r['monitors']['bestseed_precision@30']:.4f})", ""]
    open(os.path.join(OUT, "ensemble_seed_probe_RESULT.md"), "w", encoding="utf-8").write("\n".join(md))
    log("DONE")

if __name__ == "__main__":
    main()
