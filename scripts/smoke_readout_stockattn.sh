#!/usr/bin/env bash
# SMOKE (h-20260619): READOUT cross-stock attention — the warm-q migration target.
# Post-pool/pre-head, ungated/mainpath/in-series. Removes all 3 warm-q pathologies at once:
#   no de-mean        -> no V-escape (P3 cannot exist; no scale-invariant constraint to game)
#   QK-norm + temp    -> cold-query immune (P2; logits are temp*cos, magnitude-free)  [ARM=r1 only]
#   ungated residual  -> load-bearing (P1; no router exit, uniform is a repeller under ranking loss)
#
# ARM=r1 (default): qknorm ON  (the bet)
# ARM=r0          : qknorm OFF (fixed 1/sqrt(d_head); placement-only control). R0 & R1 share an
#                   identical backbone (clean A/B: R0-vs-R1 isolates the QK-norm effect).
# Reference to beat: warm-q (test RankIC 0.0831/0.0790 seed42/43) AND anchor (RankIC 0.0816, IR 1.54).
#
# READ (falsifiable):
#   PROMOTE  readout_stock_attn_entropy_norm drops <0.999 & stable AND valid_rank_ic >= anchor & warm-q
#            AND readout_stock_attn_norm stays O(1) (immune by construction — tripwire)
#   HONEST DEATH  entropy stays ~1.0, out_proj/out_norm stay ~0, rank_ic == anchor -> no readout
#            cross-stock signal; layer turns itself off cleanly (the right negative)
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
ARM="${ARM:-r1}"
SEED="${SEED:-42}"
MAX_RETRY="${MAX_RETRY:-1}"
OUTDIR="logs/smoke_readout_stockattn"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
export QIB_QLIB_KERNELS="${QIB_QLIB_KERNELS:-1}"   # serial qlib data -> no WinError5 backtest crash
M="$OUTDIR/master_${ARM}_seed${SEED}.log"

ANCHOR_M='"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"'

if [ "$ARM" = "r0" ]; then QKN="false"; else QKN="true"; fi

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

export QIB_RUN_SETTING="readout_${ARM}_seed${SEED}_scale05"
export QIB_MODEL_OVERRIDES_JSON="{${ANCHOR_M},\"use_readout_stock_attn\":true,\"readout_stock_attn_qknorm\":${QKN}}"
unset QIB_DATA_OVERRIDES_JSON 2>/dev/null || true
unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
# NO stock_expert flags, NO no_wd (retired) — this is anchor + readout cross-stock block only.
export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$SEED"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'

echo "==== SMOKE readout ARM=$ARM (qknorm=$QKN) seed $SEED start $(date) ====" > "$M"
attempt=1; ec=1
while [ "$attempt" -le "$MAX_RETRY" ]; do
  echo "#### attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
  wait_for_gpu 9000
  "$PY" -u work_flow.py > "$OUTDIR/readout_${ARM}_seed${SEED}.log" 2>&1
  ec=$?
  echo "attempt${attempt}_EXIT=$ec $(date)" >> "$M"
  if [ "$ec" -eq 0 ]; then break; fi
  attempt=$((attempt + 1)); sleep 30
done
echo "FINAL_EXIT=$ec $(date)" >> "$M"
echo "==== done $(date); read: readout_stock_attn_entropy_norm(<0.999?), readout_stock_attn_norm(O(1)?), rank_ic ====" >> "$M"
