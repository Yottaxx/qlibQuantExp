#!/usr/bin/env bash
# Pool/readout forensics on FAITHFUL pre-trained backbones (NO retraining), two config
# families x 6 seeds = 12 loads:
#   full_135      = tau-frozen baseline (scale=0.01)  -> experiment 325032212672679181
#   tau_scale_05  = tau unclamp scale=0.5             -> experiment 867178867749867261
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="analysis/pool_readout_forensics"
LOG="logs/pool_forensics_loaded.log"

run_one () {  # label  outsubdir  seed  exp  runid
  local label="$1" sub="$2" s="$3" exp="$4" rid="$5"
  echo "" >> "$LOG"; echo "########## $label seed $s ##########" >> "$LOG"
  "$PY" -u scripts/pool_readout_forensics.py --seed "$s" --label "$label" \
    --load-model "mlruns/$exp/$rid/artifacts/model" \
    --out-dir "$ROOT/$sub" >> "$LOG" 2>&1
  echo "${label}_seed${s}_EXIT=$?" >> "$LOG"
}

echo "=== two-family loaded pool forensics started $(date) ===" > "$LOG"

# ---- full_135 (tau-frozen baseline) ----
run_one full_135 full_135 42 325032212672679181 26e088e3282a4170a59a53f0fa8c8319
run_one full_135 full_135 43 325032212672679181 77e0c2a3d77b4076aa90e5dbc8b9d1e2
run_one full_135 full_135 44 325032212672679181 f58411c85d1845b2b7e249643b440486
run_one full_135 full_135 45 325032212672679181 da8e3a0c00364ae3b98afdca608f3d76
run_one full_135 full_135 46 325032212672679181 fbd6e6ac44e54e93b0bfb67fea287594
run_one full_135 full_135 47 325032212672679181 ddce1ee1e361477d8a7f40bb3456288d

# ---- tau_scale_05 ----
run_one tau_scale_05 tau_scale_05 42 867178867749867261 917808d52d9a4a13b36c6e20a4155d55
run_one tau_scale_05 tau_scale_05 43 867178867749867261 659cb7dccb784796b674257e4d13baf8
run_one tau_scale_05 tau_scale_05 44 867178867749867261 21a05e198fd34b43bd0d3a14f63e8e2b
run_one tau_scale_05 tau_scale_05 45 867178867749867261 af816bff348943b8b1674b15913adc37
run_one tau_scale_05 tau_scale_05 46 867178867749867261 dd07c8e7206b401ba231ba079a124e84
run_one tau_scale_05 tau_scale_05 47 867178867749867261 8144a4a501174ac8b1ddee49dfdf7288

# ---- per-family aggregation (n=6 each) ----
echo "" >> "$LOG"; echo "########## AGGREGATE full_135 (n=6) ##########" >> "$LOG"
"$PY" -u scripts/aggregate_pool_forensics.py --root "$ROOT/full_135" --seeds 42,43,44,45,46,47 >> "$LOG" 2>&1
echo "" >> "$LOG"; echo "########## AGGREGATE tau_scale_05 (n=6) ##########" >> "$LOG"
"$PY" -u scripts/aggregate_pool_forensics.py --root "$ROOT/tau_scale_05" --seeds 42,43,44,45,46,47 >> "$LOG" 2>&1

# ---- combined cross-family archived report ----
echo "" >> "$LOG"; echo "########## COMBINED REPORT ##########" >> "$LOG"
"$PY" -u scripts/pool_forensics_combined_report.py --root "$ROOT" \
  --families full_135,tau_scale_05 --seeds 42,43,44,45,46,47 >> "$LOG" 2>&1
echo "COMBINED_EXIT=$?" >> "$LOG"
echo "=== done $(date) ===" >> "$LOG"
