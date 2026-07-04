#!/usr/bin/env bash
# L-6 portfolio-IR aux — n=1 PAIRED (card h-20260627-001), owner-chosen minimal first read.
# seed 42 x {treatment lambda=0.08, control lambda=0}, readout OFF, 25ep, FULL window, COST-ON backtest.
set -u
cd /c/Users/60585/PycharmProjects/qibMacV2
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
LOGDIR="analysis/tau_scale05_diagnosis/n3_logs"
mkdir -p "$LOGDIR"
COMMON='"time_tau_mlp_out_scale":0.5,"temporal_readout":"","use_readout_stock_attn":false,"use_stock_expert":false,"main_loss":"mse","loss_weights":{"mse":1.0}'
PORT='{"backtest":{"exchange_kwargs":{"open_cost":0.0005,"close_cost":0.0015,"min_cost":5,"limit_threshold":0.095}}}'
run () {
  local arm="$1" lam="$2" seed="$3"; local setting="l6_${arm}_s${seed}"
  echo "=== $(date '+%F %T') START ${setting} ==="
  QIB_RUN_SETTING="${setting}" \
  QIB_MODEL_OVERRIDES_JSON="{${COMMON},\"ir_aux_lambda\":${lam},\"ir_aux_ramp_steps\":5000,\"ir_aux_var_eps\":2.5e-3,\"ir_aux_ema_decay\":0.99}" \
  QIB_TRAINER_OVERRIDES_JSON="{\"seed\":${seed},\"n_epochs\":25,\"min_epochs\":25,\"consecutive_k\":999999,\"train_stop_threshold\":null,\"checkpoint_metric\":\"valid_rank_ic\",\"checkpoint_mode\":\"max\",\"use_tqdm\":false}" \
  QIB_PORT_OVERRIDES_JSON="${PORT}" \
  "$PY" work_flow.py > "${LOGDIR}/${setting}.log" 2>&1
  echo "=== $(date '+%F %T') END ${setting} rc=$? ==="
}
run ir008 0.08 42   # treatment
run ir000 0.0  42   # paired control
echo "=== $(date '+%F %T') N1 PAIRED DONE ==="
