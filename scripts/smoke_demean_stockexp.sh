#!/usr/bin/env bash
# SMOKE (h-20260610-002 follow-up): does de-mean break the uniform-attention collapse?
# Single seed (42), 25ep, g012 anchor + use_stock_expert + stock_expert_demean.
# Compare its eval diagnostics vs the existing NON-demean h2_stockexp seed42 run:
#   - non-demean (logged): stock_attn_entropy_norm=1.0000, self_frac=0.0033(=1/B), rank_ic 0.07817
# READ (falsifiable):
#   PASS-A  entropy_norm < 0.999                      => loss-neutral hypothesis CONFIRMED (attn learned structure)
#   PASS-B  entropy_norm~1.0 BUT stock_ratio<0.05 AND stock_contrib_norm->0  => honest abandonment (no cross-sec alpha)
#   FAIL    entropy_norm~1.0 AND stock_ratio still ~0.4  => another loss-neutral channel; re-investigate
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
OUTDIR="logs/smoke_demean_stockexp"
SEED="${SEED:-42}"
MAX_RETRY="${MAX_RETRY:-3}"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
M="$OUTDIR/master.log"

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

export QIB_RUN_SETTING="h2_demean_seed${SEED}_scale05"
export QIB_MODEL_OVERRIDES_JSON="{${ANCHOR_M},\"use_stock_expert\":true,\"stock_expert_demean\":true}"
unset QIB_DATA_OVERRIDES_JSON 2>/dev/null || true
unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$SEED"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'

echo "==== SMOKE h2_demean seed $SEED start $(date) ====" > "$M"
attempt=1; ec=1
while [ "$attempt" -le "$MAX_RETRY" ]; do
  echo "#### attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
  wait_for_gpu 9000
  "$PY" -u work_flow.py > "$OUTDIR/h2_demean_seed${SEED}.log" 2>&1
  ec=$?
  echo "attempt${attempt}_EXIT=$ec $(date)" >> "$M"
  if [ "$ec" -eq 0 ]; then break; fi
  attempt=$((attempt + 1)); sleep 30
done
echo "FINAL_EXIT=$ec $(date)" >> "$M"
echo "==== done $(date); grep the run log / MLflow for stock_attn_entropy_norm, stock_attn_self_frac, stock_ratio, stock_contrib_norm ====" >> "$M"
