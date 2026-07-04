# -*- coding: utf-8 -*-
"""
IC-MONITOR PANEL (zero-GPU) — the section-5 metrics missing from the training pipeline, computed post-hoc on
saved pred.pkl/label.pkl. Makes any future IC lever (L-4 CS-rank, L-6 portfolio-IR, ensembling) actually
MEASURABLE and attributable. Reuses the run-discovery from build_master_table / ensemble_seed_probe.

Panel (all daily-cross-sectional, averaged over the 611 test days):
  HEADLINE      IC, RankIC, ICIR, RankICIR  (sanity vs MLflow)
  DECILE        Q1..Q10 mean fwd-return + decile monotonicity (Spearman(decile,return)) + long-short spread Q10-Q1
  TOPK (k=30)   precision@30, recall@30, NDCG@30  (aligned to the topk=30,n_drop=5 trading rule; money is on 30 names)
  DISPERSION    std(p), std(y), std(p)/std(y), MSE-implied std(p)*=IC*std(y), shrinkage flag
                (diagnoses: is low IC a prediction-SHRINKAGE artifact, or a true signal ceiling?)
  STABILITY     per-quarter RankIC mean + worst quarter (regime fragility, no market_state pkl needed)

Usage:
  python scripts/ic_monitors.py --setting tau_scale_05            # ensemble of that setting's seeds
  python scripts/ic_monitors.py --setting tau_scale_05 --seed 42  # single seed
  python scripts/ic_monitors.py --diverse                         # the unbiased cross-setting ensemble
Writes analysis/tau_scale05_diagnosis/ic_monitors_<tag>.{json,md}.
"""
import os, re, glob, json, argparse
import numpy as np
import pandas as pd

ROOT = r"C:\Users\60585\PycharmProjects\qibMacV2"
MLRUNS = os.path.join(ROOT, "mlruns")
OUT = os.path.join(ROOT, "analysis", "tau_scale05_diagnosis")
DIVERSE = ["base", "tau_scale_05", "tau_scale_10", "readout_full_g012_d1pma", "full_135"]
K = 30

def canon(rs): return re.sub(r"_scale\d+", "", re.sub(r"_seed\d+", "", rs))
def read_param(run, name):
    p = os.path.join(run, "params", name)
    return open(p).read().strip() if os.path.exists(p) else None

def find_runs(target):
    out = {}
    for run in glob.glob(os.path.join(MLRUNS, "*", "*")):
        if not os.path.exists(os.path.join(run, "artifacts", "pred.pkl")): continue
        rs = read_param(run, "kwargs.trainer_config.run_setting")
        if not rs or canon(rs) != target: continue
        try: seed = int(read_param(run, "kwargs.trainer_config.seed"))
        except Exception: continue
        prev = out.get(seed)
        if prev is None or os.path.getmtime(run) > prev[1]:
            out[seed] = (run, os.path.getmtime(run))
    return {s: v[0] for s, v in sorted(out.items())}

def load_series(path):
    obj = pd.read_pickle(path)
    return (obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj).astype(float)

def _daily(df, fn, min_n=5):
    out = [fn(s) for _, s in df.groupby(level="datetime") if len(s) >= min_n]
    return np.array([x for x in out if np.isfinite(x)], float)

def headline(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ric = _daily(df, lambda s: s["p"].corr(s["y"], method="spearman"))
    ic = _daily(df, lambda s: s["p"].corr(s["y"]))
    return {"rank_ic": float(ric.mean()), "rank_icir": float(ric.mean()/ric.std()),
            "ic": float(ic.mean()), "icir": float(ic.mean()/ic.std()), "n_days": int(ric.size)}

def decile(pred, label, q=10):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    bucket_means, mons, spreads = {i: [] for i in range(q)}, [], []
    for _, s in df.groupby(level="datetime"):
        if len(s) < q * 2: continue
        d = pd.qcut(s["p"].rank(method="first"), q, labels=False)
        dm = s.groupby(d)["y"].mean()
        for i in range(q):
            if i in dm.index: bucket_means[i].append(dm[i])
        if dm.notna().sum() >= q - 1:
            mons.append(pd.Series(dm.index, dtype=float).corr(pd.Series(dm.values), method="spearman"))
        if (q - 1) in dm.index and 0 in dm.index:
            spreads.append(dm[q - 1] - dm[0])
    return {"decile_mean_return": [float(np.mean(bucket_means[i])) if bucket_means[i] else float("nan") for i in range(q)],
            "monotonicity_spearman": float(np.nanmean(mons)),
            "long_short_spread_Q10_Q1": float(np.nanmean(spreads))}

def topk(pred, label, k=K):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    precs, recs, ndcgs = [], [], []
    for _, s in df.groupby(level="datetime"):
        n = len(s)
        if n < 2 * k: continue
        pred_top = set(s.nlargest(k, "p").index)
        true_top = set(s.nlargest(k, "y").index)
        hit = len(pred_top & true_top)
        precs.append(hit / k); recs.append(hit / k)
        rel = s["y"].rank(pct=True)                          # non-negative relevance = label percentile
        order = s["p"].rank(ascending=False, method="first")
        topk_rel = rel[order <= k].sort_values(ascending=False)  # not the right order; recompute by pred
        sp = s.assign(rel=rel).nlargest(k, "p")["rel"].values
        si = s.assign(rel=rel).nlargest(k, "y")["rel"].values
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((sp * disc).sum()); idcg = float((si * disc).sum())
        ndcgs.append(dcg / idcg if idcg > 0 else np.nan)
    return {"precision@%d" % k: float(np.nanmean(precs)), "recall@%d" % k: float(np.nanmean(recs)),
            "ndcg@%d" % k: float(np.nanmean(ndcgs))}

def dispersion(pred, label):
    """Calibration diagnostic on the TRAINING-target scale. The model was MSE-trained on CSZScoreNorm(label)
    (per-day unit variance), so standardize the label per-day before comparing. At the MSE optimum
    std(p)* = IC*std(y_z) = IC (since std(y_z)=1). std(p) >> IC => cardinal scores over-confident (RankIC-
    irrelevant, rank-based strategy); std(p) << IC => shrinkage that would cap even the cardinal IC."""
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    df["yz"] = df.groupby(level="datetime")["y"].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    std_p = float(_daily(df, lambda s: s["p"].std()).mean())
    IC = float(_daily(df, lambda s: s["p"].corr(s["yz"])).mean())
    opt = IC                                              # MSE optimum on unit-variance target
    ratio = std_p / opt if opt else float("nan")
    flag = ("over-dispersed vs MSE-optimum (cardinal scores over-confident; RankIC-IRRELEVANT) => IC is a real "
            "signal ceiling, NOT a shrinkage artifact" if ratio > 1.5 else
            "under-dispersed (shrinkage caps even cardinal IC)" if ratio < 0.7 else "MSE-calibrated")
    return {"std_pred_native": std_p, "target_std": 1.0, "mse_optimal_std_pred": opt,
            "realized_ic": IC, "ratio_actual_over_optimal": ratio, "dispersion_flag": flag}

def stability(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    df = df.reset_index()
    df["q"] = pd.PeriodIndex(df["datetime"], freq="Q").astype(str)
    qric = {}
    for qtr, g in df.groupby("q"):
        ics = [s["p"].corr(s["y"], method="spearman") for _, s in g.groupby("datetime") if len(s) >= 5]
        ics = [x for x in ics if np.isfinite(x)]
        if ics: qric[qtr] = float(np.mean(ics))
    if not qric: return {}
    worst = min(qric, key=qric.get)
    return {"per_quarter_rank_ic": qric, "worst_quarter": worst, "worst_quarter_rank_ic": qric[worst],
            "n_quarters": len(qric), "frac_quarters_positive": float(np.mean([v > 0 for v in qric.values()]))}

def build_pred(args):
    if args.diverse:
        pool, label = {}, None
        for setting in DIVERSE:
            for seed, run in find_runs(setting).items():
                pool[(setting, seed)] = load_series(os.path.join(run, "artifacts", "pred.pkl"))
                if label is None: label = load_series(os.path.join(run, "artifacts", "label.pkl"))
        RF = pd.DataFrame({f"{s}|{sd}": pool[(s, sd)].groupby(level="datetime").rank(pct=True) for (s, sd) in pool}).dropna()
        return RF.mean(axis=1), label.reindex(RF.index), "diverse_ensemble"
    runs = find_runs(args.setting)
    if not runs: raise SystemExit(f"no runs for {args.setting}")
    if args.seed is not None:
        run = runs[args.seed]
        return (load_series(os.path.join(run, "artifacts", "pred.pkl")),
                load_series(os.path.join(run, "artifacts", "label.pkl")), f"{args.setting}_seed{args.seed}")
    label = None; ranks = {}
    for seed, run in runs.items():
        p = load_series(os.path.join(run, "artifacts", "pred.pkl"))
        ranks[seed] = p.groupby(level="datetime").rank(pct=True)
        if label is None: label = load_series(os.path.join(run, "artifacts", "label.pkl"))
    RF = pd.DataFrame(ranks).dropna()
    return RF.mean(axis=1), label.reindex(RF.index), f"{args.setting}_ens{len(runs)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="tau_scale_05")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--diverse", action="store_true")
    args = ap.parse_args()
    pred, label, tag = build_pred(args)
    panel = {"tag": tag, "headline": headline(pred, label), "decile": decile(pred, label),
             "topk": topk(pred, label), "dispersion": dispersion(pred, label), "stability": stability(pred, label)}
    json.dump(panel, open(os.path.join(OUT, f"ic_monitors_{tag}.json"), "w"), indent=2)
    h, d, t, dp, st = panel["headline"], panel["decile"], panel["topk"], panel["dispersion"], panel["stability"]
    md = [f"# IC-monitor panel — {tag}", "",
          f"## Headline  rank_ic {h['rank_ic']:.5f} | rank_icir {h['rank_icir']:.3f} | ic {h['ic']:.5f} | icir {h['icir']:.3f}  (n={h['n_days']})",
          "",
          f"## Decile (Q1..Q10 mean fwd-return)", f"- {[round(x,5) for x in d['decile_mean_return']]}",
          f"- monotonicity (Spearman decile->return) **{d['monotonicity_spearman']:.4f}** | long-short Q10-Q1 **{d['long_short_spread_Q10_Q1']:.5f}**", "",
          f"## Top-{K} (aligned to trading rule)",
          f"- precision@{K} **{t['precision@%d'%K]:.4f}** | recall@{K} {t['recall@%d'%K]:.4f} | ndcg@{K} **{t['ndcg@%d'%K]:.4f}**", "",
          f"## Dispersion anchor (is low IC a shrinkage artifact?) — on the per-day unit-variance target",
          f"- std(pred) {dp['std_pred_native']:.4f} vs MSE-optimal std(pred)*=IC={dp['mse_optimal_std_pred']:.4f} "
          f"(target std {dp['target_std']:.1f}) | ratio actual/optimal {dp['ratio_actual_over_optimal']:.2f}",
          f"- **{dp['dispersion_flag']}**", "",
          f"## Stability  worst quarter {st.get('worst_quarter','?')} rank_ic {st.get('worst_quarter_rank_ic',float('nan')):.4f} | "
          f"{st.get('frac_quarters_positive',float('nan')):.2f} of {st.get('n_quarters','?')} quarters positive"]
    open(os.path.join(OUT, f"ic_monitors_{tag}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))

if __name__ == "__main__":
    main()
