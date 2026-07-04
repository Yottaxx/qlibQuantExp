#!/usr/bin/env bash
# Clean serial orchestrator for the RMS out-norm disambiguator: seed 42 then seed 43, ONE process,
# no competing launchers. Each inner launcher has its own GPU precheck + retry; strictly sequential.
# kernels=1 (set in smoke_rms_stockexp.sh) makes the qlib backtest crash-proof on Windows.
set -u
echo "==== run_rms_42_43 start $(date) ===="
for s in 42 43; do
  echo "---- launching seed $s $(date) ----"
  SEED=$s bash scripts/smoke_rms_stockexp.sh
  echo "---- seed $s returned $(date) ----"
done
echo "==== run_rms_42_43 done $(date) ===="
