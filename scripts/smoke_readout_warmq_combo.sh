#!/usr/bin/env bash
# SMOKE (h-20260619): COMBINED arm — warm-q block stock_expert (demean + no_wd) AND the readout
# cross-stock block (QK-norm). Control vs the clean readout-only R1.
#   block : use_stock_expert + stock_expert_demean (+ trainer stock_expert_no_wd)  -> warm-q mechanism
#   readout: use_readout_stock_attn + qknorm                                        -> R1 mechanism
# Purpose: does stacking the (pathological) block mechanism UNDER the readout help, hurt, or wash out?
# CAVEAT: this re-imports warm-q's V-escape (P3) at the block; a gain here is NOT attributable between
# block and readout. Interpret only relative to R1 (readout-only) and warm-q (block-only).
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
SEED="${SEED:-42}"
MAX_RETRY="${MAX_RETRY:-1}"
OUTDIR="logs/smoke_readout_warmq_combo"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
export QIB_QLIB_KERNELS="${QIB_QLIB_KERNELS:-1}"
M="$OUTDIR/master_seed${SEED}.log"

ANCHOR_M='"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"'

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9'; }
wait_for_gpu() {
  local need="$1" f
  while :; do
    f=$(gpu_free_mib)
    if [ -n "$f" ] && [ "$f" -ge "$need" ]; then return 0; fi
    echo "   [gpu-wait] free=${f:-?}MiB < ${need}MiB; sleep 120s $(date)" >> "$M"
    sleep 120
  done
}

export QIB_RUN_SETTING="readout_warmqcombo_seed${SEED}_scale05"
export QIB_MODEL_OVERRIDES_JSON="{${ANCHOR_M},\"use_stock_expert\":true,\"stock_expert_demean\":true,\"use_readout_stock_attn\":true,\"readout_stock_attn_qknorm\":true}"
unset QIB_DATA_OVERRIDES_JSON 2>/dev/null || true
unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$SEED"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false,"stock_expert_no_wd":true}'

echo "==== SMOKE readout+warmq-combo seed $SEED start $(date) ====" > "$M"
attempt=1; ec=1
while [ "$attempt" -le "$MAX_RETRY" ]; do
  echo "#### attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
  wait_for_gpu 9000
  "$PY" -u work_flow.py > "$OUTDIR/readout_warmqcombo_seed${SEED}.log" 2>&1
  ec=$?
  echo "attempt${attempt}_EXIT=$ec $(date)" >> "$M"
  if [ "$ec" -eq 0 ]; then break; fi
  attempt=$((attempt + 1)); sleep 30
done
echo "FINAL_EXIT=$ec $(date)" >> "$M"
echo "==== done $(date); read: readout_stock_attn_entropy_norm, stock_expert_norm(escape?), rank_ic ====" >> "$M"
