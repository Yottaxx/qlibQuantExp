# -*- coding: utf-8 -*-
"""
CS-RANK CEILING PROBE — WINDOWED-TRAJECTORY arm (model-matched, TS extension of ceiling_probe_cs_rank_window.py).

OPEN QUESTION (never tested): the RST-MoE model consumes an 8-step window, so if CS-rank is appended as *input
channels* the model sees the FULL 8-STEP CS-RANK TRAJECTORY (8x158=1264 cols), not just day-T. The sibling script
ceiling_probe_cs_rank_window.py measured only the day-T-only variant (RW+CSR = 0.06248, +0.0033 over the raw
window 0.05922). This script adds the trajectory arm and asks: does the full windowed CS-rank lift the
leakage-clean ceiling MORE than the day-T-only +0.0033?

ARMS:
  RW          = raw 8x158 window (1264)                               -> L-1 variant-B (reused 0.05922 via SKIP_RW)
  RW+CSR_TS   = raw window (1264) + 8-step windowed CS-rank (1264)    -> 2528 cols, the trajectory L-4 test

CS-rank: per-day pct-rank-0.5 on UNCLIPPED robust-z features (PIT, same-day peers only, NO shift), then the SAME
sliding-window (build_window's exact math) is applied to the CS-rank matrix so each window's END row aligns to day
T exactly like the raw window (identical instrument-group-boundary handling; a window never crosses instruments).
Leakage-clean fit_score identical to the sibling: fit LGBM/ridge on train only (2008-01..2020-03), tune alpha / n
on the internal-eval slice (2019-01..2020-03) only, >=3 LGBM seeds, NO per-test-day refit, daily-Spearman RankIC.

The day-T-only RW+CSR ceiling (0.06248) is NOT recomputed here; it is read from the sibling RESULT json for the
direct trajectory-vs-day-T comparison. Writes analysis/tau_scale05_diagnosis/ceiling_probe_cs_rank_window_ts_RESULT
.{json,md}.

Memory note: the full 2528-col matrix is ~11GB and the raw+CSR windows would be another ~11GB if both were
materialized, exceeding available RAM. So instead of building two 1264-wide window arrays via build_window and
concatenating (the sibling's approach, fine at 1422 cols), this script keeps ONLY the two 158-col base matrices and
gathers each window step directly into preallocated train/test matrices (peak ~13GB). `window_ends` reproduces
build_window's exact group-boundary / end-index logic and `gather_window` reproduces its exact step-major flatten
order; `_selftest_window` asserts byte-identity to build_window on synthetic data at startup. The MODELING protocol
(grids, seeds, ceiling=max(ridge,lgbm)) is byte-for-byte the sibling's.
"""
import os, json, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = Path(r"C:\Users\60585\PycharmProjects\qibMacV2\analysis\tau_scale05_diagnosis")
SIBLING = OUT / "ceiling_probe_cs_rank_window_RESULT.json"   # day-T RW+CSR reference lives here
RESULT_JSON = OUT / ("ceiling_probe_cs_rank_window_ts_SMOKE.json" if SMOKE else "ceiling_probe_cs_rank_window_ts_RESULT.json")
RESULT_MD = OUT / ("ceiling_probe_cs_rank_window_ts_SMOKE.md" if SMOKE else "ceiling_probe_cs_rank_window_ts_RESULT.md")
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
    """L-1 variant-B window builder (VERBATIM from ceiling_probe_cs_rank_window.py): 8-step flattened, end-row =
    day T, within a single instrument. Kept as the reference definition of the windowing; the production path uses
    window_ends + gather_window which reproduce this exactly (asserted in _selftest_window) but at far lower peak
    RAM (never materializes the full WIN*F-wide array)."""
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

def window_ends(idx_sorted):
    """The end-index half of build_window: given a MultiIndex already sorted by [instrument, datetime], return
    (ends_valid, idx_b) where ends_valid[k] is the row position (in the sorted array) of the END of window k, and
    idx_b = idx_sorted[ends_valid]. A window is valid iff all WIN rows lie within one instrument's run -- identical
    group-boundary handling to build_window (windows never cross instruments)."""
    inst = idx_sorted.get_level_values("instrument").to_numpy(); N = len(idx_sorted)
    change = np.empty(N, bool); change[0] = True; change[1:] = inst[1:] != inst[:-1]
    grp_start = np.maximum.accumulate(np.where(change, np.arange(N), 0))
    ends = np.arange(WIN - 1, N)
    same = (ends - grp_start[ends]) >= (WIN - 1)
    ends_valid = ends[same]
    return ends_valid, idx_sorted[ends_valid]

def gather_window(arr_sorted, ends_valid, sel_pos, F, out, col_off):
    """The flatten half of build_window: for the selected window rows sel_pos (positions into ends_valid), write the
    step-major-flattened 8-step window into out[:, col_off:col_off+WIN*F]. Column (step*F + f) == arr_sorted[end -
    (WIN-1) + step, f], byte-identical to build_window's sliding_window_view().transpose(0,2,1).reshape(). Gathers
    one (n_sel, F) step at a time -> peak transient ~ n_sel*F*4 bytes, never the full WIN*F width."""
    e = ends_valid[sel_pos]
    for step in range(WIN):
        src = e - (WIN - 1) + step
        out[:, col_off + step * F: col_off + (step + 1) * F] = arr_sorted[src, :]

def _selftest_window():
    """Assert window_ends + gather_window reproduce build_window byte-for-byte (flatten order, group boundaries,
    end index) on tiny 3-instrument synthetic data with runs shorter than WIN (to exercise the boundary drop)."""
    dts = pd.date_range("2020-01-01", periods=14)
    tuples, lens = [], {"A": 14, "B": 5, "C": 11}   # B has < WIN rows -> yields zero valid windows
    for inst, n in lens.items():
        for d in dts[:n]: tuples.append((d, inst))
    mi = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
    F = 3; arr = np.arange(len(mi) * F, dtype=np.float32).reshape(len(mi), F)
    df = pd.DataFrame(arr, index=mi)
    Xb_ref, idxb_ref = build_window(df)
    Xs = df.sort_index(level=["instrument", "datetime"]); arr_s = Xs.to_numpy(np.float32)
    ends_valid, idx_b = window_ends(Xs.index)
    out = np.empty((idx_b.size, WIN * F), np.float32)
    gather_window(arr_s, ends_valid, np.arange(idx_b.size), F, out, 0)
    assert idx_b.equals(idxb_ref), "selftest: idx_b mismatch"
    assert out.shape == Xb_ref.shape and np.array_equal(out, Xb_ref), "selftest: window content mismatch"
    log("selftest OK: fused windowing == build_window (n_windows=%d, incl. sub-WIN instrument dropped)" % idx_b.size)

def main():
    t0 = time.time()
    _selftest_window()
    log(f"qlib.init (SMOKE={SMOKE})")
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN, kernels=1)

    log("clipped handler (raw window src)...")
    h = make_handler(True)
    Xi = h.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32)
    yi_raw = h.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0].astype(np.float32)
    yl_z = h.fetch(col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0].astype(np.float32)
    F = Xi.shape[1]; log("clipped %.1fs" % (time.time() - t0), Xi.shape)

    log("unclipped handler (CS-rank src)...")
    hnc = make_handler(False)
    Xnc = hnc.fetch(col_set="feature", data_key=DataHandlerLP.DK_I).astype(np.float32).reindex(Xi.index)
    # PIT day-level cross-sectional rank on UNCLIPPED features: same-day peers only, no shift.
    CSR = (Xnc.groupby(level="datetime").rank(pct=True).astype(np.float32) - np.float32(0.5)); del Xnc, hnc; gc.collect()
    log("CS-rank (day-level pct-rank-0.5, PIT) %.1fs" % (time.time() - t0), CSR.shape)

    # --- sort both base matrices by [instrument, datetime] (build_window's canonical order) ---
    log("sorting base matrices...")
    Xs = Xi.sort_index(level=["instrument", "datetime"]); del Xi
    arr_raw = Xs.to_numpy(np.float32); idx = Xs.index; del Xs; gc.collect()
    CSRs = CSR.sort_index(level=["instrument", "datetime"]); del CSR
    arr_csr = CSRs.to_numpy(np.float32)
    assert CSRs.index.equals(idx), "CS-rank index must match the raw-feature index exactly (aligned rows)"
    del CSRs; gc.collect()
    assert arr_csr.shape == arr_raw.shape

    # --- window end index (shared by both arms; a window never crosses instruments) ---
    ends_valid, idx_b = window_ends(idx)
    Fw = WIN * F  # 1264
    log("window rows n_b=%d  Fw=%d  (RW+CSR_TS width=%d)" % (idx_b.size, Fw, 2 * Fw))

    dt_b = idx_b.get_level_values("datetime")
    yb_z = yl_z.reindex(idx_b); yb_raw = yi_raw.reindex(idx_b); del yl_z, yi_raw
    znf = yb_z.notna().to_numpy(); rnf = yb_raw.notna().to_numpy()
    trb = ((dt_b >= TRAIN_START) & (dt_b <= TRAIN_END) & znf)          # full train (fit + internal-eval)
    teb = ((dt_b >= TEST_START) & (dt_b <= TEST_END) & rnf)            # held-out test
    sel_tr = np.nonzero(trb)[0]; sel_te = np.nonzero(teb)[0]           # positions into idx_b (window rows)
    log("mask rows  train=%d  test=%d" % (sel_tr.size, sel_te.size))

    idx_tr = idx_b[sel_tr]; idx_te = idx_b[sel_te]
    yz_tr = yb_z.to_numpy(np.float32)[sel_tr]              # train fit target (CSZScoreNorm robust z-label)
    yraw_tr = yb_raw.iloc[sel_tr]                          # raw label on train (for internal-eval IC)
    yraw_te = yb_raw.iloc[sel_te]                          # raw label on test  (for test IC)
    dt_tr = idx_tr.get_level_values("datetime")
    fit_local = np.asarray((dt_tr >= TRAIN_START) & (dt_tr <= INT_FIT_END))       # 2008-01..2018-12
    eval_local = np.asarray((dt_tr >= INT_EVAL_START) & (dt_tr <= TRAIN_END))     # 2019-01..2020-03
    sel_fit = sel_tr[fit_local]; sel_eval = sel_tr[eval_local]        # idx_b positions for internal fit/eval
    idx_ev = idx_tr[eval_local]; ye = yraw_tr[eval_local]
    log("internal split  fit=%d  eval=%d" % (sel_fit.size, sel_eval.size))

    # gather the 2528-col (or raw-only 1264-col) feature matrix for a set of window rows, directly from the two
    # 158-col base arrays -> peak stays ~= base(1.4GB) + one materialized matrix, never two full-width copies.
    def gather_rows(sel_pos, raw_only=False):
        w = Fw if raw_only else 2 * Fw
        out = np.empty((sel_pos.size, w), np.float32)
        gather_window(arr_raw, ends_valid, sel_pos, F, out, 0)
        if not raw_only:
            gather_window(arr_csr, ends_valid, sel_pos, F, out, Fw)
        return out

    from sklearn.linear_model import Ridge
    import lightgbm as lgb

    def fit_score_lean(tag, raw_only=False):
        """Byte-for-byte the sibling fit_score's modeling: ridge alpha grid + lgbm n grid tuned on the internal
        eval slice, refit on full train, >=3 lgbm seeds on test, ceiling=max(ridge, lgbm_mean). Only the memory
        layout differs: the internal fit/eval matrices are gathered, tuned on, then FREED before the full-train
        matrix is gathered for the final refit (so the 8GB fit copy and the 9GB full-train copy never coexist).
        The fit/eval/train/test row sets and labels are identical to the sibling's."""
        out = {}
        # ---- Phase A: tune on internal fit/eval, then free those matrices ----
        Xf = gather_rows(sel_fit, raw_only); yf = yz_tr[fit_local]
        Xe = gather_rows(sel_eval, raw_only)
        best = (None, -1e9)
        for a in [1.0, 10., 100., 1e3, 1e4, 1e5, 1e6]:
            r = Ridge(alpha=a).fit(Xf, yf)
            ic = daily_rank_ic(pd.Series(r.predict(Xe), index=idx_ev), ye)["rank_ic"]
            if np.isfinite(ic) and ic > best[1]: best = (a, ic)
        best_alpha = best[0]; ridge_int = best[1]
        p = dict(objective="regression", learning_rate=0.03, num_leaves=63, feature_fraction=0.6,
                 bagging_fraction=0.7, bagging_freq=1, min_child_samples=200, max_depth=-1, n_jobs=-1, verbosity=-1)
        dfit = lgb.Dataset(Xf, label=yf); best_n, best_nic = 300, -1e9
        for n in [100, 300, 600, 1000]:
            m = lgb.train(dict(p, seed=0), dfit, num_boost_round=n)
            ic = daily_rank_ic(pd.Series(m.predict(Xe), index=idx_ev), ye)["rank_ic"]
            log(f"  [{tag}] lgbm n={n} int={ic:.5f}")
            if np.isfinite(ic) and ic > best_nic: best_nic, best_n = ic, n
        del dfit, Xf, Xe, yf; gc.collect()
        # ---- Phase B: refit on full train, score on test (free the ~9GB train matrix ASAP) ----
        Xtr_a = gather_rows(sel_tr, raw_only)
        r = Ridge(alpha=best_alpha).fit(Xtr_a, yz_tr)
        Xte_a = gather_rows(sel_te, raw_only)
        out["ridge"] = {**daily_rank_ic(pd.Series(r.predict(Xte_a), index=idx_te), yraw_te), "alpha": best_alpha, "int_eval_rank_ic": ridge_int}
        log(f"  [{tag}] ridge a={best_alpha:g} int={ridge_int:.5f} test={out['ridge']['rank_ic']:.5f}")
        del r, Xte_a; gc.collect()
        dfu = lgb.Dataset(Xtr_a, label=yz_tr, free_raw_data=True); dfu.construct()
        del Xtr_a; gc.collect()                     # bins are built; drop the dense 9GB train matrix now
        Xte_a = gather_rows(sel_te, raw_only); sics = []
        for s in [0, 1, 2]:
            m = lgb.train(dict(p, seed=s, bagging_seed=s, feature_fraction_seed=s), dfu, num_boost_round=best_n)
            sics.append(daily_rank_ic(pd.Series(m.predict(Xte_a), index=idx_te), yraw_te)["rank_ic"])
            log(f"  [{tag}] lgbm s={s} n={best_n} test={sics[-1]:.5f}")
        del dfu, Xte_a; gc.collect()
        out["lgbm"] = {"seed_rank_ics": sics, "rank_ic_mean": float(np.nanmean(sics)), "rank_ic_std": float(np.nanstd(sics)), "best_n": best_n, "int_eval_rank_ic": best_nic}
        out["ceiling"] = float(np.nanmax([out["ridge"]["rank_ic"], out["lgbm"]["rank_ic_mean"]]))
        log(f"  [{tag}] CEILING={out['ceiling']:.5f}"); return out

    results = {"meta": {"smoke": SMOKE, "model_ref": MODEL_REF, "raw_window_ceiling": RAW_WINDOW_CEIL, "F": F, "WIN": WIN,
                        "cols_RWplusCSR_TS": int(2 * Fw), "n_train": int(sel_tr.size), "n_test": int(sel_te.size)}, "arms": {}}

    # RW reference: reuse the known L-1 variant-B ceiling (SKIP_RW default; recompute only if explicitly asked)
    SKIP_RW = os.environ.get("SKIP_RW", "1") == "1"
    if SKIP_RW:
        results["arms"]["RW"] = {"ceiling": 0.05922, "note": "reused from sibling run log (LGBM 0.05817/0.05913/0.06036); reproduces L-1 variant-B 0.0592 exactly"}
        log("ARM RW skipped (reuse known 0.05922)")
    else:
        log("ARM RW (raw window only, recompute)")
        results["arms"]["RW"] = fit_score_lean("RW", raw_only=True)
    json.dump(results, open(RESULT_JSON, "w"), indent=2)

    # RW+CSR_TS: raw window (1264) + 8-step windowed CS-rank trajectory (1264) = 2528 cols
    log("ARM RW+CSR_TS (raw window + 8-step windowed CS-rank trajectory, 2528 cols)")
    results["arms"]["RW+CSR_TS"] = fit_score_lean("RW+CSR_TS")
    json.dump(results, open(RESULT_JSON, "w"), indent=2)

    # day-T RW+CSR reference (read from sibling; not recomputed)
    dayT = None
    try:
        sib = json.load(open(SIBLING))
        dayT = float(sib["arms"]["RW+CSR"]["ceiling"])
    except Exception as e:
        log("WARN could not read sibling day-T RW+CSR:", repr(e))

    RW = results["arms"]["RW"]["ceiling"]; RWCTS = results["arms"]["RW+CSR_TS"]["ceiling"]
    lift = RWCTS - RW
    zone = ("TS_CS_HAS_HEADROOM__L4_go" if lift >= MDE else
            "TS_CS_REDUNDANT__L4_marginal" if lift <= 0.003 else "MODAL")
    verdict = {"RW_ceiling": RW, "RWplusCSR_TS_ceiling": RWCTS, "windowed_ts_lift": lift,
               "RWplusCSR_dayT_ceiling": dayT,
               "lift_over_dayT": (RWCTS - dayT) if dayT is not None else None,
               "dayT_lift_over_RW": (dayT - RW) if dayT is not None else None,
               "model_ref": MODEL_REF, "mde": MDE, "zone": zone}
    results["verdict"] = verdict
    json.dump(results, open(RESULT_JSON, "w"), indent=2)

    md = ["# CS-rank Ceiling Probe — WINDOWED-TRAJECTORY arm (model-matched)", "",
          f"- SMOKE={SMOKE} F={F} WIN={WIN} cols(RW+CSR_TS)={2*Fw} model_ref={MODEL_REF} raw_window(L-1 B)={RAW_WINDOW_CEIL}",
          f"- **RW={RW:.5f}  RW+CSR_TS={RWCTS:.5f}  windowed_ts_lift={lift:+.5f}**",
          f"- day-T RW+CSR (sibling) = {dayT if dayT is None else round(dayT,5)}  | lift_TS_over_dayT = {verdict['lift_over_dayT'] if dayT is None else round(verdict['lift_over_dayT'],5)}",
          f"- **VERDICT: {zone}**", ""]
    for nm, v in results["arms"].items():
        if "ridge" not in v:
            md += [f"## {nm}  ceiling={v['ceiling']:.5f}", f"- {v.get('note','')}", ""]; continue
        md += [f"## {nm}  ceiling={v['ceiling']:.5f}", f"- ridge {v['ridge']['rank_ic']:.5f} (alpha={v['ridge']['alpha']}, int_eval={v['ridge']['int_eval_rank_ic']:.5f})",
               f"- lgbm {v['lgbm']['rank_ic_mean']:.5f} ± {v['lgbm']['rank_ic_std']:.5f} (n={v['lgbm']['best_n']}, seeds {['%.5f'%x for x in v['lgbm']['seed_rank_ics']]})", ""]
    RESULT_MD.write_text("\n".join(md), encoding="utf-8")
    log("DONE %.1fs -> %s" % (time.time() - t0, zone))
    print("FINAL_VERDICT " + json.dumps(verdict), flush=True)

if __name__ == "__main__":
    main()
