#!/usr/bin/env bash
# Temporal-readout MATRIX at scale=0.5 (time-readout-bonus-20260607). Each design runs through the
# FULL work_flow.py pipeline FROM SCRATCH, SERIALLY (one GPU/qlib process at a time — concurrency
# previously crashed via joblib temp contention), with a stable temp dir.
#
#   MODE=smoke -> Phase 0 GATE: n_epochs=1 on a short data window. Prove each design BUILD+TRAIN+
#                COMPLETE (diagnostic_matrix.csv written, factor_pool_weights non-None, diag keys).
#   MODE=full  -> Phase 1: n_epochs=25, ckpt=valid_rank_ic:max, then compare vs scale=0.5 control.
#
# Usage:
#   MODE=smoke bash scripts/run_readout_matrix_scale05.sh
#   MODE=full  bash scripts/run_readout_matrix_scale05.sh
#   DESIGNS="d3mix d1pma" SEEDS="42 43" MODE=full bash scripts/run_readout_matrix_scale05.sh
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
MODE="${MODE:-smoke}"
DESIGNS="${DESIGNS:-d3cid d3cin d3mix d1pma duala dualb}"
SEEDS="${SEEDS:-42}"
OUTDIR="logs/readout_matrix_scale05"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8

if [ "$MODE" = "smoke" ]; then
  EPOCHS=1
  # Short window so an epoch is fast; normalization still fits on the full handler fit window.
  export QIB_DATA_OVERRIDES_JSON='{"kwargs":{"segments":{"train":["2019-04-01","2020-03-31"],"valid":["2020-07-01","2020-12-31"],"test":["2020-07-01","2020-12-31"]}}}'
  export QIB_PORT_OVERRIDES_JSON='{"backtest":{"end_time":"2020-12-31"}}'
else
  EPOCHS=25
  unset QIB_DATA_OVERRIDES_JSON 2>/dev/null || true
  unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
fi

M="$OUTDIR/master_${MODE}.log"
echo "==== readout matrix ($MODE, ${EPOCHS}ep) designs=[$DESIGNS] seeds=[$SEEDS] start $(date) ====" > "$M"
for d in $DESIGNS; do
  for s in $SEEDS; do
    echo "" >> "$M"; echo "#### $d seed $s ($MODE) start $(date) ####" >> "$M"
    export QIB_RUN_SETTING="readout_${MODE}_${d}_seed${s}_scale05"
    export QIB_MODEL_OVERRIDES_JSON='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"'"$d"'","use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'
    export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$s"',"n_epochs":'"$EPOCHS"',"min_epochs":'"$EPOCHS"',"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
    "$PY" -u work_flow.py > "$OUTDIR/${MODE}_${d}_seed${s}.log" 2>&1
    echo "${d}_seed${s}_EXIT=$? $(date)" >> "$M"
  done
done
if [ "$MODE" = "full" ]; then
  echo "" >> "$M"; echo "#### COMPARE ####" >> "$M"
  "$PY" -u scripts/compare_readout_matrix_vs_control.py --designs "$(echo $DESIGNS | tr ' ' ',')" --seeds "$(echo $SEEDS | tr ' ' ',')" >> "$M" 2>&1
  echo "COMPARE_EXIT=$?" >> "$M"
fi
echo "==== readout matrix ($MODE) done $(date) ====" >> "$M"
