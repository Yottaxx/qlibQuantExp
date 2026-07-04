#!/usr/bin/env bash
# Clean serial orchestrator for the n=3 warm-q kill-check: run seed 43 then seed 44,
# ONE process, no competing launchers (replaces the tangle left by a dead agent).
# Each inner launcher has its own GPU precheck + retry; they run strictly sequentially.
set -u
echo "==== run_warmq_43_44 start $(date) ===="
for s in 43 44; do
  echo "---- launching seed $s $(date) ----"
  SEED=$s bash scripts/smoke_warmq_stockexp.sh
  echo "---- seed $s returned $(date) ----"
done
echo "==== run_warmq_43_44 done $(date) ===="
