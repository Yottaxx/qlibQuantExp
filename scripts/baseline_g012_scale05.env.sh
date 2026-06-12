#!/usr/bin/env bash
# BASELINE ANCHOR env (designated 2026-06-12): τ scale=0.5 + d1pma g=0.12.
# Source this, set SEED, then run work_flow.py to reproduce the anchor exactly.
#   source scripts/baseline_g012_scale05.env.sh; export SEED=42; python work_flow.py
# Docs + data manifest: analysis/readout_redesign/archive_g012_baseline/BASELINE_ANCHOR.md
# NOTE: anchor-at-parity, not a promotion (g012_n3_RESULT.md). Unset temporal_readout => old control exactly.
SEED="${SEED:-42}"
export QIB_RUN_SETTING="baseline_g012_scale05_seed${SEED}"
export QIB_MODEL_OVERRIDES_JSON='{"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"}'
export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$SEED"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
