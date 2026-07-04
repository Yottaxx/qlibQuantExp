#!/usr/bin/env bash
# Step-B time-readout VERDICT: readout+top-block continue-finetune, paired control vs d3cid.
# n=3 (seeds 42-44) for overnight feasibility (~7h). Extend to n=6 if promising.
set -u
PY="/c/Users/60585/miniconda3/envs/quantEnv/python.exe"
ROOT="analysis/readout_redesign/stepB"
LOG="logs/time_readout_stepB.log"
EXP="mlruns/867178867749867261"

declare -A RUN=( [42]="917808d52d9a4a13b36c6e20a4155d55" [43]="659cb7dccb784796b674257e4d13baf8" [44]="21a05e198fd34b43bd0d3a14f63e8e2b" )

run_one () {  # arm seed
  echo "" >> "$LOG"; echo "########## $1 seed $2 ##########" >> "$LOG"
  "$PY" -u scripts/time_readout_stepB.py --seed "$2" --arm "$1" --epochs 8 \
    --load-model "$EXP/${RUN[$2]}/artifacts/model" --out-dir "$ROOT" >> "$LOG" 2>&1
  echo "${1}_seed${2}_EXIT=$?" >> "$LOG"
}

echo "=== Step-B time-readout finetune started $(date) ===" > "$LOG"
for s in 42 43 44; do
  run_one control "$s"
  run_one d3cid "$s"
done
echo "" >> "$LOG"; echo "########## AGGREGATE ##########" >> "$LOG"
"$PY" -u scripts/aggregate_time_readout_stepB.py --root "$ROOT" --seeds 42,43,44 >> "$LOG" 2>&1
echo "AGG_EXIT=$?" >> "$LOG"
echo "=== Step-B done $(date) ===" >> "$LOG"
