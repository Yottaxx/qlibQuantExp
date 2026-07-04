#!/usr/bin/env bash
# SMOKE (h-20260617): the OUTPUT-NORM disambiguator. Pin the stock-expert output magnitude with
# parameter-free RMSNorm (over D, no scale) BEFORE de-mean, so the V-amplification escape is closed
# and SHARPENING (Path A) is the ONLY remaining way to satisfy the de-mean gradient.
#
# Single-variable delta vs the warm-q baseline (demean + no_wd, which V-escaped: entropy frozen
# 1.0000, stock_expert_norm -> 65..88). The ONLY change here is stock_expert_out_norm="rms".
#   - demean  ON : de-mean gradient still rewards cross-stock variation in the output
#   - no_wd   ON : Wq/Wk free to warm (gives Path A its best shot; not pinned by weight decay)
#   - rms     ON : output magnitude is scale-invariant => Path B (||Wv||||Wo||->88) is foreclosed
# Unit-tested: 50x weight inflation moves out-norm 17751x (none) vs ~2x (rms). V-escape closed.
#
# READ (falsifiable, A3 §4 decision table):
#   OUTCOME A (PROMOTE): stock_attn_entropy_norm drops < 0.999 (trends down) AND stock_expert_norm
#                        stays O(1) (NOT ->88) AND rank_ic >= anchor  -> signal was real, V-escape
#                        was masking it. Widen seeds, hand to ablation plan.
#   OUTCOME B (HONEST DEATH): entropy frozen ~0.9999, norm O(1), stock_ratio -> <0.05, rank_ic ~=
#                        anchor -> no recoverable cross-stock signal; stock axis dies cleanly.
#   OUTCOME C (BUG): stock_expert_norm RE-CLIMBS -> a learnable scale leaked in; reject variant.
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
OUTDIR="logs/smoke_rms_stockexp"
SEED="${SEED:-42}"
MAX_RETRY="${MAX_RETRY:-3}"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
export QIB_QLIB_KERNELS="${QIB_QLIB_KERNELS:-1}"   # serial qlib data loading -> no WinError5 pool-respawn crash
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

export QIB_RUN_SETTING="h2_rms_seed${SEED}_scale05"
export QIB_MODEL_OVERRIDES_JSON="{${ANCHOR_M},\"use_stock_expert\":true,\"stock_expert_demean\":true,\"stock_expert_out_norm\":\"rms\"}"
unset QIB_DATA_OVERRIDES_JSON 2>/dev/null || true
unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
# stock_expert_no_wd lives in the TRAINER overrides (optimizer param-group split in model_adapter)
export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$SEED"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false,"stock_expert_no_wd":true}'

echo "==== SMOKE h2_rms seed $SEED start $(date) ====" > "$M"
attempt=1; ec=1
while [ "$attempt" -le "$MAX_RETRY" ]; do
  echo "#### attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
  wait_for_gpu 9000
  "$PY" -u work_flow.py > "$OUTDIR/h2_rms_seed${SEED}.log" 2>&1
  ec=$?
  echo "attempt${attempt}_EXIT=$ec $(date)" >> "$M"
  if [ "$ec" -eq 0 ]; then break; fi
  attempt=$((attempt + 1)); sleep 30
done
echo "FINAL_EXIT=$ec $(date)" >> "$M"
echo "==== done $(date); read: stock_attn_entropy_norm(<0.999?=sharpen), stock_expert_norm(O(1)?), stock_ratio, rank_ic ====" >> "$M"
