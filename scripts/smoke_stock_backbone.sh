#!/usr/bin/env bash
# h-20260624: stock-backbone = cross-stock attention as a PRE-MoE MAIN-PATH residual (NOT a routed
# expert). x = x + gamma*StockAttn(LN(x)); NO de-mean / NO out_norm (removes the V-escape driver that
# killed the routed warm-q form: entropy froze ~1.0 + ||out||->65-100). gamma learnable scalar init
# from GAMMA_INIT; router stays 2-way; stock attn proj kept wd-free (expert_no_wd_scope=stock_backbone).
# Params: SEED (default 42), GAMMA_INIT (default 0.0). g012 anchor base, 25ep.
# READ vs g012 anchor (rankIC 0.0784 / IR_wc 1.75 / MDD -0.076):
#   PROMOTE   IR_wc >= anchor AND MaxDD_wc not worse (do NOT promote on rankIC alone — finding #3/#8)
#   WATCH     stock_backbone_gamma (moves/grows? or decays to ~0 = model ignores it)
#             stock_backbone_attn_entropy_norm (<0.999 = cold query un-froze; ~1.0 = no cross-stock structure)
#   KILL      gamma -> ~0 AND/OR entropy ~1.0 AND portfolio flat => stock axis settled-DEAD
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
OUTDIR="logs/smoke_stock_backbone"
SEED="${SEED:-42}"
GAMMA_INIT="${GAMMA_INIT:-0.0}"
MAX_RETRY="${MAX_RETRY:-3}"
case "$GAMMA_INIT" in
  0.0|0) GTAG="g0" ;;
  1.0|1) GTAG="g1" ;;
  *)     GTAG="g${GAMMA_INIT}" ;;
esac
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

export QIB_RUN_SETTING="gbb_${GTAG}_seed${SEED}_scale05"
export QIB_MODEL_OVERRIDES_JSON="{${ANCHOR_M},\"stock_backbone\":true,\"stock_backbone_gamma_init\":${GAMMA_INIT}}"
unset QIB_DATA_OVERRIDES_JSON 2>/dev/null || true
unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
# free the stock-backbone attention from weight decay (warm-q parity: MHA in_proj is q/k/v fused).
export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$SEED"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false,"expert_no_wd_scope":"stock_backbone"}'

echo "==== stock_backbone ${GTAG} seed $SEED start $(date) (gamma_init=${GAMMA_INIT}) ====" >> "$M"
attempt=1; ec=1
while [ "$attempt" -le "$MAX_RETRY" ]; do
  echo "#### ${GTAG} seed${SEED} attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
  wait_for_gpu 9000
  "$PY" -u work_flow.py > "$OUTDIR/sbb_${GTAG}_seed${SEED}.log" 2>&1
  ec=$?
  echo "${GTAG}_seed${SEED}_attempt${attempt}_EXIT=$ec $(date)" >> "$M"
  if [ "$ec" -eq 0 ]; then break; fi
  attempt=$((attempt + 1)); sleep 30
done
echo "FINAL_EXIT ${GTAG}_seed${SEED}=$ec $(date)" >> "$M"
echo "==== done ${GTAG} seed${SEED} $(date); watch stock_backbone_gamma, stock_backbone_attn_entropy_norm, rank_ic/IR ====" >> "$M"
