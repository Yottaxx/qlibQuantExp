#!/usr/bin/env bash
# WEEKEND AUTONOMOUS PLAN (owner 2026-06-12): on the g012 anchor (tau0.5 + d1pma gate_init=-2):
#   Stage A: h-20260610-001 DropExtremeLabel 2.5%/side  x seeds 42/43/44
#   Stage B: h-20260610-002 stock expert (3-expert MoE) x seeds 42/43/44
#   Stage C: 001+002 combined                            x seeds 42/43/44
# 25ep, ckpt=valid_rank_ic:max, STRICTLY SERIAL. Kill-resilient: GPU-free precheck per cell
# (Stage A >=7000 MiB, B/C >=9000 MiB — measured 3-expert peak 8.47 GB) + 3x retry.
# Compares vs the ANCHOR runs after each stage (analysis/weekend_001_002/*.md).
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
OUTDIR="logs/weekend_001_002"
SEEDS="${SEEDS:-42 43 44}"
MAX_RETRY="${MAX_RETRY:-3}"
mkdir -p "$ROOT/tmp_run" "$OUTDIR"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"; export PYTHONIOENCODING=utf-8
M="$OUTDIR/master.log"

ANCHOR_M='"time_tau_mlp_out_scale":0.5,"temporal_readout":"d1pma","temporal_readout_gate_init":-2.0,"use_regime_time_embedding":true,"use_regime_factor_gate":true,"router_use_layer_summary":true,"router_mode":"learned"'
LP='{"kwargs":{"handler":{"kwargs":{"learn_processors":[{"class":"DropnaLabel"},{"class":"DropExtremeLabel","module_path":"module.utils.processors","kwargs":{"fields_group":"label","percent":0.025}},{"class":"CSZScoreNorm","kwargs":{"fields_group":"label","method":"robust"}}]}}}}'

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

run_cell() { # run_cell <label> <model_json> <data_json_or_empty> <seed> <min_free>
  local label="$1" mjson="$2" djson="$3" s="$4" need="$5" attempt=1 ec=1
  export QIB_RUN_SETTING="${label}_seed${s}_scale05"
  export QIB_MODEL_OVERRIDES_JSON="$mjson"
  if [ -n "$djson" ]; then export QIB_DATA_OVERRIDES_JSON="$djson"; else unset QIB_DATA_OVERRIDES_JSON; fi
  unset QIB_PORT_OVERRIDES_JSON 2>/dev/null || true
  export QIB_TRAINER_OVERRIDES_JSON='{"seed":'"$s"',"n_epochs":25,"min_epochs":25,"consecutive_k":999999,"train_stop_threshold":null,"checkpoint_metric":"valid_rank_ic","checkpoint_mode":"max","use_tqdm":false}'
  while [ "$attempt" -le "$MAX_RETRY" ]; do
    echo "" >> "$M"; echo "#### ${label} seed $s attempt $attempt/$MAX_RETRY start $(date) ####" >> "$M"
    wait_for_gpu "$need"
    "$PY" -u work_flow.py > "$OUTDIR/${label}_seed${s}.log" 2>&1
    ec=$?
    echo "${label}_seed${s}_attempt${attempt}_EXIT=$ec $(date)" >> "$M"
    if [ "$ec" -eq 0 ]; then break; fi
    attempt=$((attempt + 1)); sleep 30
  done
  echo "${label}_seed${s}_FINAL_EXIT=$ec $(date)" >> "$M"
}

echo "==== WEEKEND 001/002 matrix (seeds $SEEDS) start $(date) ====" > "$M"

echo "" >> "$M"; echo "======== STAGE A: h1_dropext (001) ========" >> "$M"
for s in $SEEDS; do run_cell "h1_dropext" "{${ANCHOR_M}}" "$LP" "$s" 7000; done
"$PY" -u scripts/compare_weekend_vs_anchor.py --stage h1_dropext --seeds 42,43,44 >> "$M" 2>&1

echo "" >> "$M"; echo "======== STAGE B: h2_stockexp (002) ========" >> "$M"
for s in $SEEDS; do run_cell "h2_stockexp" "{${ANCHOR_M},\"use_stock_expert\":true}" "" "$s" 9000; done
"$PY" -u scripts/compare_weekend_vs_anchor.py --stage h2_stockexp --seeds 42,43,44 >> "$M" 2>&1

echo "" >> "$M"; echo "======== STAGE C: h12_combo (001+002) ========" >> "$M"
for s in $SEEDS; do run_cell "h12_combo" "{${ANCHOR_M},\"use_stock_expert\":true}" "$LP" "$s" 9000; done
"$PY" -u scripts/compare_weekend_vs_anchor.py --stage h12_combo --seeds 42,43,44 >> "$M" 2>&1

"$PY" -u scripts/compare_weekend_vs_anchor.py --stage all --seeds 42,43,44 >> "$M" 2>&1
echo "==== WEEKEND matrix done $(date) ====" >> "$M"
