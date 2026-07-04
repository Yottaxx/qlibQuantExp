#!/usr/bin/env bash
# SMOKE (h-20260610-002 follow-up #3): can the cold stock-query WARM when freed from weight decay?
# KEY-SVD verdict: cross-stock keys are structured & peakable (s1/common~0.8, eff_rank>1, peak_ent@T3<0.9)
# => collapse is OPERATOR-STATE (cold query x cold 1/d_head temp), NOT input-forced.
# Lever (gradient-derived): warm the query via Cov_j(g_ij, kr_j).
#   de-mean              -> makes the output rank-relevant => inflates Cov(g,kr) (the warming gradient)
#   stock Wq/Wk no-wd    -> stops weight decay from pinning ||Wq||~0 before the signal accumulates
#     (tau ~ ||Wq||^-1 redundancy: this IS "set a suitable temperature", the only form that survives)
# Single seed 42, 25ep, g012 anchor.
# READ (falsifiable, vs original h2: entropy 1.0000 / stock_expert_norm ~0.05 frozen):
#   PASS  stock_attn_entropy_norm < 0.999 (query warmed) AND rank_ic >= anchor at best epoch
#   WORLD-C  query warms (entropy < 0.999) BUT rank_ic flat  -> peakable != promotes; abandon w/ factor pool
#   KILL  entropy still ~1.0 (query stayed cold even wd-free) OR norm explodes / NaN
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
OUTDIR="logs/smoke_warmq_stockexp"
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

export QIB_RUN_SETTING="h2_warmq_seed${SEED}_scale05"
export QIB_MODEL_OVERRIDES_JSON="{${ANCHOR_M},\"use_stock_expert\":true,\"stock_expert_demean\":true}"
unset QIB_DATA_OVERRIDES_JSON 2>/dev/null || true
unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
# stock_expert_no_wd lives in the TRAINER overrides (optimizer param-group split in model_adapter)
export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$SEED"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false,"stock_expert_no_wd":true}'

echo "==== SMOKE h2_warmq seed $SEED start $(date) ====" > "$M"
attempt=1; ec=1
while [ "$attempt" -le "$MAX_RETRY" ]; do
  echo "#### attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
  wait_for_gpu 9000
  "$PY" -u work_flow.py > "$OUTDIR/h2_warmq_seed${SEED}.log" 2>&1
  ec=$?
  echo "attempt${attempt}_EXIT=$ec $(date)" >> "$M"
  if [ "$ec" -eq 0 ]; then break; fi
  attempt=$((attempt + 1)); sleep 30
done
echo "FINAL_EXIT=$ec $(date)" >> "$M"
echo "==== done $(date); watch stock_attn_entropy_norm (warm?), stock_expert_norm (stable?), rank_ic ====" >> "$M"
