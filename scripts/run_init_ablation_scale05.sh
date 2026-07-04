#!/usr/bin/env bash
# Init-ablation arm (task #14): linear uniform_mean vs onehot_last + d1pma gate sweep.
# seed42, scale=0.5, 25ep, ckpt=valid_rank_ic:max. SERIAL, stable temp.
# RESILIENT to intermittent external-CUDA kills: GPU-free precheck before each launch + retry up to N×.
# 6 new cells (~30h). Reused (NOT here): onehot d3cid/d3mix + the existing d1pma (=g0.5) from the prior matrix.
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
OUTDIR="logs/init_ablation_scale05"
SEED="${SEED:-42}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
MAX_RETRY="${MAX_RETRY:-3}"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
M="$OUTDIR/master.log"

# cell = "label|temporal_readout|extra_model_json_fragment"
CELLS=(
  "onehot_d3cin|d3cin|\"temporal_readout_init\":\"onehot_last\""
  "uniform_d3cid|d3cid|\"temporal_readout_init\":\"uniform_mean\""
  "uniform_d3cin|d3cin|\"temporal_readout_init\":\"uniform_mean\""
  "uniform_d3mix|d3mix|\"temporal_readout_init\":\"uniform_mean\""
  "g012_d1pma|d1pma|\"temporal_readout_gate_init\":-2.0"
  "g088_d1pma|d1pma|\"temporal_readout_gate_init\":2.0"
)

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9'; }
wait_for_gpu() {
  local f
  while :; do
    f=$(gpu_free_mib)
    if [ -n "$f" ] && [ "$f" -ge "$MIN_FREE_MIB" ]; then return 0; fi
    echo "   [gpu-wait] free=${f:-?}MiB < ${MIN_FREE_MIB}MiB; sleep 120s $(date)" >> "$M"
    sleep 120
  done
}

echo "==== init-ablation (seed${SEED}, 25ep, min_free=${MIN_FREE_MIB}MiB, retry=${MAX_RETRY}) start $(date) ====" > "$M"
for cell in "${CELLS[@]}"; do
  IFS='|' read -r label tr extra <<< "$cell"
  export QIB_RUN_SETTING="readout_full_${label}_seed${SEED}_scale05"
  export QIB_MODEL_OVERRIDES_JSON="{\"time_tau_mlp_out_scale\":0.5,\"temporal_readout\":\"$tr\",${extra},\"use_regime_time_embedding\":true,\"use_regime_factor_gate\":true,\"router_use_layer_summary\":true,\"router_mode\":\"learned\"}"
  export QIB_TRAINER_OVERRIDES_JSON="{\"seed\":${SEED},\"n_epochs\":25,\"min_epochs\":25,\"consecutive_k\":999999,\"train_stop_threshold\":null,\"checkpoint_metric\":\"valid_rank_ic\",\"checkpoint_mode\":\"max\",\"use_tqdm\":false}"
  attempt=1; ec=1
  while [ "$attempt" -le "$MAX_RETRY" ]; do
    echo "" >> "$M"; echo "#### $label attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
    wait_for_gpu
    "$PY" -u work_flow.py > "$OUTDIR/${label}_seed${SEED}.log" 2>&1
    ec=$?
    echo "${label}_attempt${attempt}_EXIT=$ec $(date)" >> "$M"
    if [ "$ec" -eq 0 ]; then break; fi
    attempt=$((attempt + 1)); sleep 30
  done
  echo "${label}_FINAL_EXIT=$ec $(date)" >> "$M"
done
echo "" >> "$M"; echo "#### COMPARE ####" >> "$M"
"$PY" -u scripts/compare_init_ablation.py --seed "$SEED" >> "$M" 2>&1
echo "COMPARE_EXIT=$? $(date)" >> "$M"
echo "==== init-ablation done $(date) ====" >> "$M"
