#!/usr/bin/env bash
# Card h-20260629-001: normalized-MSE (mse_normalize=True => per-day CS z-score of pred+target
# => pure per-day Pearson-IC loss). 2 single-seed (s42, 25ep) treatment runs vs EXISTING
# same-code controls; each arm differs from its control ONLY in mse_normalize.
#   ArmA mseN_g012_d1pma_s42  (anchor d1pma + mseN, frictions-OFF) vs gradaccum_ctrl_k1_25ep_s42 (rank_ic 0.0803)
#   ArmB mseN_tau05_roff_s42  (readout-OFF tau0.5 + mseN, COST-ON) vs l6_ir000_s42 (rank_ic 0.0796, IR_wc 2.864)
# work_flow.py UNTOUCHED (env overrides only). Smoke (ArmA mseN 1ep) fail-fast guards the never-run normalize path.
set -u
cd /c/Users/60585/PycharmProjects/qibMacV2
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
LOGDIR="logs/mse_normalize_s42"
mkdir -p "$LOGDIR"

# ArmA = anchor (g012_d1pma) + mse_normalize. ArmB = l6_ir000 model (readout OFF, ir_aux off) + mse_normalize.
MODEL_A='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned","mse_normalize":true}'
MODEL_B='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"","use_readout_stock_attn":false,"use_stock_expert":false,"main_loss":"mse","loss_weights":{"mse":1.0},"ir_aux_lambda":0.0,"ir_aux_ramp_steps":5000,"ir_aux_var_eps":2.5e-3,"ir_aux_ema_decay":0.99,"mse_normalize":true}'
PORT_COST='{"backtest":{"exchange_kwargs":{"open_cost":0.0005,"close_cost":0.0015,"min_cost":5,"limit_threshold":0.095}}}'
TR_25='{"seed":42,"n_epochs":25,"min_epochs":25,"grad_accum_steps":1,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
TR_1='{"seed":42,"n_epochs":1,"min_epochs":1,"grad_accum_steps":1,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'

run () {
  local setting="$1" model="$2" trainer="$3" port="${4:-}"
  echo "=== $(date '+%F %T') START ${setting} ==="
  if [ -n "$port" ]; then
    QIB_RUN_SETTING="${setting}" QIB_MODEL_OVERRIDES_JSON="${model}" QIB_TRAINER_OVERRIDES_JSON="${trainer}" QIB_PORT_OVERRIDES_JSON="${port}" "$PY" work_flow.py > "${LOGDIR}/${setting}.log" 2>&1
  else
    QIB_RUN_SETTING="${setting}" QIB_MODEL_OVERRIDES_JSON="${model}" QIB_TRAINER_OVERRIDES_JSON="${trainer}" "$PY" work_flow.py > "${LOGDIR}/${setting}.log" 2>&1
  fi
  local rc=$?
  echo "=== $(date '+%F %T') END ${setting} rc=${rc} ==="
  return $rc
}

# 0) SMOKE — never-run mse_normalize path, 1ep, fail-fast.
run "mseN_smoke_a_1ep_s42" "$MODEL_A" "$TR_1"
if [ $? -ne 0 ]; then echo "!!! SMOKE FAILED (rc!=0) — aborting; mse_normalize path crashed. Inspect ${LOGDIR}/mseN_smoke_a_1ep_s42.log"; exit 1; fi
echo ">>> smoke OK; launching both arms"

# A) anchor d1pma + mse_normalize (frictions-OFF; control = gradaccum_ctrl_k1_25ep_s42)
run "mseN_g012_d1pma_s42" "$MODEL_A" "$TR_25"

# B) readout-OFF tau0.5 + mse_normalize (COST-ON; control = l6_ir000_s42)
run "mseN_tau05_roff_s42" "$MODEL_B" "$TR_25" "$PORT_COST"

echo "=== $(date '+%F %T') BOTH ARMS DONE (mse_normalize, seed 42) ==="
