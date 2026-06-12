#!/usr/bin/env bash
# Step-A time-readout SCREEN on tau_scale_05 backbones (seeds 42-47), frozen + cached, 8ep.
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="analysis/readout_redesign/stepA"
LOG="logs/time_readout_stepA.log"
EXP="mlruns/867178867749867261"   # tau_scale_05

run_one () {  # seed runid
  echo "" >> "$LOG"; echo "########## scale05 seed $1 ##########" >> "$LOG"
  "$PY" -u scripts/time_readout_finetune.py --seed "$1" \
    --load-model "$EXP/$2/artifacts/model" --out-dir "$ROOT" >> "$LOG" 2>&1
  echo "seed_${1}_EXIT=$?" >> "$LOG"
}

echo "=== Step-A time-readout screen started $(date) ===" > "$LOG"
run_one 42 917808d52d9a4a13b36c6e20a4155d55
run_one 43 659cb7dccb784796b674257e4d13baf8
run_one 44 21a05e198fd34b43bd0d3a14f63e8e2b
run_one 45 af816bff348943b8b1674b15913adc37
run_one 46 dd07c8e7206b401ba231ba079a124e84
run_one 47 8144a4a501174ac8b1ddee49dfdf7288

echo "" >> "$LOG"; echo "########## AGGREGATE ##########" >> "$LOG"
"$PY" -u scripts/aggregate_time_readout.py --root "$ROOT" --seeds 42,43,44,45,46,47 >> "$LOG" 2>&1
echo "AGG_EXIT=$?" >> "$LOG"
echo "=== Step-A done $(date) ===" >> "$LOG"
