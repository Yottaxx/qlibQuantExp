#!/usr/bin/env bash
# g0.12 d1pma 3-seed kill-check: seeds 43,44 (seed42 exists from the init-ablation arm).
# Pre-registered contract: kill if paired ΔRankIC(g012−control) n=3 mean < +0.003 OR IR/MaxDD clearly
# worse -> attention readout closed for good. GPU-free precheck + 3x retry per cell (external-CUDA kills).
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
OUTDIR="logs/init_ablation_scale05"
SEEDS="${SEEDS:-43 44}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
MAX_RETRY="${MAX_RETRY:-3}"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
M="$OUTDIR/master_g012_n3.log"

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

echo "==== g012 n=3 kill-check (seeds $SEEDS) start $(date) ====" > "$M"
for s in $SEEDS; do
  export QIB_RUN_SETTING="readout_full_g012_d1pma_seed${s}_scale05"
  export QIB_MODEL_OVERRIDES_JSON='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'
  export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$s"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
  attempt=1; ec=1
  while [ "$attempt" -le "$MAX_RETRY" ]; do
    echo "" >> "$M"; echo "#### g012 seed $s attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
    wait_for_gpu
    "$PY" -u work_flow.py > "$OUTDIR/g012_d1pma_seed${s}.log" 2>&1
    ec=$?
    echo "g012_seed${s}_attempt${attempt}_EXIT=$ec $(date)" >> "$M"
    if [ "$ec" -eq 0 ]; then break; fi
    attempt=$((attempt + 1)); sleep 30
  done
  echo "g012_seed${s}_FINAL_EXIT=$ec $(date)" >> "$M"
done
echo "" >> "$M"; echo "#### COMPARE (paired n=3) ####" >> "$M"
"$PY" -u scripts/compare_g012_n3.py >> "$M" 2>&1
echo "COMPARE_EXIT=$? $(date)" >> "$M"
echo "==== g012 n=3 done $(date) ====" >> "$M"
