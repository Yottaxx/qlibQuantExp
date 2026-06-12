#!/usr/bin/env bash
# FROM-SCRATCH retrain of the d3cid temporal-readout model (Tier-B gold), scale=0.5 settings,
# 25 epochs, checkpoint=valid_rank_ic:max. Seeds 42-44. Full work_flow.py pipeline -> RankIC +
# IR_with_cost + MaxDD + diagnostics (incl. tr_W_last_frac monitoring). Control = the existing
# scale=0.5 backbones (same protocol). SERIAL (one GPU process at a time), stable temp.
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
mkdir -p "$ROOT/tmp_run" logs/d3cid_scale05
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
export QIB_MODEL_OVERRIDES_JSON='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d3cid","use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'

M="logs/d3cid_scale05/master.log"
echo "==== d3cid scale05 from-scratch (25ep, ckpt=valid_rank_ic, seeds 42-44) start $(date) ====" > "$M"
for s in 42 43 44; do
  echo "" >> "$M"; echo "#### seed $s start $(date) ####" >> "$M"
  export QIB_RUN_SETTING="d3cid_scale05_seed${s}"
  export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$s"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
  "$PY" -u work_flow.py > "logs/d3cid_scale05/seed${s}.log" 2>&1
  echo "seed_${s}_EXIT=$? $(date)" >> "$M"
done
echo "" >> "$M"; echo "#### COMPARE ####" >> "$M"
"$PY" -u scripts/compare_d3cid_vs_control.py --seeds 42,43,44 >> "$M" 2>&1
echo "COMPARE_EXIT=$?" >> "$M"
echo "==== d3cid scale05 done $(date) ====" >> "$M"
