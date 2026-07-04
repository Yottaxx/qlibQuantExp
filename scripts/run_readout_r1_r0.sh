#!/usr/bin/env bash
# Clean serial orchestrator: R1 (qknorm, the bet) then R0 (fixed-scale, placement control), seed 42.
# ONE process, no competing launchers. kernels=1 makes the qlib backtest crash-proof on Windows.
# R1 first so the decisive read is available soonest; R0 follows to attribute placement-vs-QKnorm.
set -u
echo "==== run_readout_r1_r0 start $(date) ===="
for arm in r1 r0; do
  echo "---- launching ARM=$arm seed 42 $(date) ----"
  ARM=$arm SEED=42 bash scripts/smoke_readout_stockattn.sh
  echo "---- ARM=$arm returned $(date) ----"
done
echo "==== run_readout_r1_r0 done $(date) ===="
