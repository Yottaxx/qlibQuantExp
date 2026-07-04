#!/usr/bin/env python
"""Build ONE master comparison table across all important settings, with AND without cost.

Scans mlruns/, reads each run's run_setting + seed (params/kwargs.trainer_config.*),
canonicalizes the setting name (strips _seed<N> and _scale<NN> suffixes), pulls the 7
decision metrics, groups by canonical setting and prints per-seed + n-seed mean.

Metrics (last logged value per run):
  rankIC                 = "Rank IC"
  AR_wc / AR_nc          = 1day.excess_return_{with,without}_cost.annualized_return
  IR_wc / IR_nc          = 1day.excess_return_{with,without}_cost.information_ratio
  MDD_wc / MDD_nc        = 1day.excess_return_{with,without}_cost.max_drawdown

Usage:
  python scripts/build_master_table.py            # curated IMPORTANT settings only
  python scripts/build_master_table.py --all      # every canonical setting found
  python scripts/build_master_table.py --csv analysis/master_table.csv
"""
import os, re, glob, sys, statistics as st, csv as csvmod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS = os.path.join(ROOT, "mlruns")

# Curated set of canonical settings to show (order = display order). Edit freely.
# Match is by canonical name (after stripping _seed*/_scale*). 'warm-q' will appear as
# 'h2_warmq' once seeds 44/45/46 finish logging to MLflow.
IMPORTANT = [
    "base",                       # original baseline
    "tau_scale_10",               # tau, no down-scale (scale=1.0) reference
    "tau_scale_05",               # *** standalone tau0.5 control (old anchor) ***
    "readout_full_g012_d1pma",    # *** tau0.5 + d1pma g0.12 (BASELINE ANCHOR) ***
    "full_135",                   # production full config
    "h2_demean",                  # stock-expert: demean only
    "h2_stockexp",                # stock-expert: plain
    "h12_combo",                  # stock-expert: combo
    "h2_warmq",                   # warm-q stock-expert (KILLED n=5)
    "gbb_g0",                     # *** stock-backbone main-path residual, gamma_init=0.0 (ReZero) ***
    "gbb_g1",                     # *** stock-backbone main-path residual, gamma_init=1.0 (full-on) ***
]

# predictive-quality metrics (IC family) + portfolio metrics (with/without cost)
PRED = {
    "IC": "IC",
    "ICIR": "ICIR",
    "rankIC": "Rank IC",
    "rankICIR": "Rank ICIR",
}
METRICS = {
    "AR_wc": "1day.excess_return_with_cost.annualized_return",
    "AR_nc": "1day.excess_return_without_cost.annualized_return",
    "IR_wc": "1day.excess_return_with_cost.information_ratio",
    "IR_nc": "1day.excess_return_without_cost.information_ratio",
    "MDD_wc": "1day.excess_return_with_cost.max_drawdown",
    "MDD_nc": "1day.excess_return_without_cost.max_drawdown",
}


def last_val(metric_dir, name):
    p = os.path.join(metric_dir, name)
    if not os.path.exists(p):
        return float("nan")
    try:
        return float(open(p).read().strip().splitlines()[-1].split()[1])
    except Exception:
        return float("nan")


def read_param(run, name):
    p = os.path.join(run, "params", name)
    if not os.path.exists(p):
        return None
    return open(p).read().strip()


def canon(run_setting):
    s = re.sub(r"_seed\d+", "", run_setting)
    s = re.sub(r"_scale\d+", "", s)
    return s


def scan():
    groups = {}  # canon -> {seed: {metrics}}
    for run in glob.glob(os.path.join(MLRUNS, "*", "*")):
        if not os.path.isdir(os.path.join(run, "metrics")):
            continue
        rs = read_param(run, "kwargs.trainer_config.run_setting")
        if not rs:
            continue
        seed = read_param(run, "kwargs.trainer_config.seed")
        try:
            seed = int(seed)
        except Exception:
            seed = -1
        md = os.path.join(run, "metrics")
        # require at least one portfolio metric present (finished run)
        row = {}
        for k, v in PRED.items():
            row[k] = last_val(md, v)
        for k, v in METRICS.items():
            row[k] = last_val(md, v)
        c = canon(rs)
        # keep the latest-by-mtime run if a (canon,seed) repeats
        prev = groups.setdefault(c, {}).get(seed)
        if prev is None or os.path.getmtime(run) > prev["_mtime"]:
            row["_mtime"] = os.path.getmtime(run)
            row["_run"] = os.path.relpath(run, ROOT)
            groups[c][seed] = row
    return groups


COLS = ["IC", "ICIR", "rankIC", "rankICIR",
        "AR_wc", "AR_nc", "IR_wc", "IR_nc", "MDD_wc", "MDD_nc"]


def fmt(v):
    return f"{v:>8.4f}" if v == v else f"{'   --   ':>8}"


def main():
    show_all = "--all" in sys.argv
    csv_path = None
    if "--csv" in sys.argv:
        csv_path = sys.argv[sys.argv.index("--csv") + 1]

    groups = scan()
    order = sorted(groups) if show_all else [s for s in IMPORTANT if s in groups]
    missing = [s for s in IMPORTANT if s not in groups]

    hdr = f"{'setting':<30}{'n':>2}  " + "".join(f"{c:>9}" for c in COLS)
    print(hdr); print("-" * len(hdr))
    csv_rows = []
    for c in order:
        seeds = sorted(k for k in groups[c] if k >= 0)
        rows = [groups[c][s] for s in seeds]
        means = {col: st.mean([r[col] for r in rows if r[col] == r[col]] or [float("nan")]) for col in COLS}
        seed_str = ",".join(map(str, seeds))
        line = f"{c:<30}{len(seeds):>2}  " + "".join(fmt(means[col]) for col in COLS)
        print(line)
        csv_rows.append({"setting": c, "n": len(seeds), "seeds": seed_str,
                         **{col: means[col] for col in COLS}})
    print("-" * len(hdr))
    print("cost drag: AR_wc-AR_nc, IR halves under cost. MDD_wc usually deeper than MDD_nc.")
    if not show_all and missing:
        print(f"\nNOT YET IN MLFLOW (waiting / not run): {', '.join(missing)}")
    print(f"\n(scanned {sum(len(v) for v in groups.values())} runs across {len(groups)} canonical settings; "
          f"use --all to list every setting)")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=["setting", "n", "seeds"] + COLS)
            w.writeheader(); w.writerows(csv_rows)
        print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
