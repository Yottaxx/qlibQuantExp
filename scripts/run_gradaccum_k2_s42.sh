#!/usr/bin/env bash
# Card h-20260628-001: cross-day gradient accumulation (K=2), step-matched.
# PAIRED single-seed (seed 42) on the CURRENT working tree (drift-controlled fresh anchor).
#   control   = anchor EXACT: K=1, 25ep   -> 25*D optimizer steps
#   treatment = K=2, 50ep                 -> 50*D/2 = 25*D optimizer steps (STEP-MATCHED)
# Both arms: identical model config (tau0.5 + d1pma g0.12); differ ONLY in
# {grad_accum_steps, n_epochs, min_epochs}. Frictions-OFF backtest (anchor default);
# primary read = valid_rank_ic (cost-independent). work_flow.py UNTOUCHED (env overrides only).
# Why a fresh control: working tree is +1548 lines vs anchor SHA b1df36d -> the archived
# anchor seed42=0.08163 is on OLD code; only a same-code paired control isolates grad-accum.
set -u
cd /c/Users/60585/PycharmProjects/qibMacV2
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
LOGDIR="logs/gradaccum_k2_s42"
mkdir -p "$LOGDIR"

# Anchor model config (baseline_g012_scale05) — IDENTICAL for both arms. ir_aux_* unset => L-6 inert.
MODEL='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'

run () {
  local setting="$1" trainer="$2"
  echo "=== $(date '+%F %T') START ${setting} ==="
  QIB_RUN_SETTING="${setting}" \
  QIB_MODEL_OVERRIDES_JSON="${MODEL}" \
  QIB_TRAINER_OVERRIDES_JSON="${trainer}" \
  "$PY" work_flow.py > "${LOGDIR}/${setting}.log" 2>&1
  local rc=$?
  echo "=== $(date '+%F %T') END ${setting} rc=${rc} ==="
  return $rc
}

# 0) SMOKE — exercise the never-run K=2 accumulation path for 1 epoch; abort all if it crashes.
run "gradaccum_smoke_k2_1ep_s42" '{"seed":42,"n_epochs":1,"min_epochs":1,"grad_accum_steps":2,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
if [ $? -ne 0 ]; then echo "!!! SMOKE FAILED (rc!=0) — aborting paired run; K=2 path crashed. Inspect ${LOGDIR}/gradaccum_smoke_k2_1ep_s42.log"; exit 1; fi
echo ">>> smoke OK; launching paired run"

# 1) CONTROL — anchor exact (K=1, 25ep) on current code => drift-matched fresh anchor.
run "gradaccum_ctrl_k1_25ep_s42" '{"seed":42,"n_epochs":25,"min_epochs":25,"grad_accum_steps":1,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'

# 2) TREATMENT — cross-day grad accum K=2, 50ep (step-matched to control).
run "gradaccum_treat_k2_50ep_s42" '{"seed":42,"n_epochs":50,"min_epochs":50,"grad_accum_steps":2,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'

echo "=== $(date '+%F %T') PAIRED RUN DONE (control + treatment, seed 42) ==="
