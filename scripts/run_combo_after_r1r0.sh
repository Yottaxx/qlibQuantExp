#!/usr/bin/env bash
# Queued orchestrator: wait until the readout R1->R0 orchestrator finishes, THEN run the
# warm-q-block + readout COMBINED arm (seed 42). Strict serialization (no GPU contention):
#   (1) block on the r1_r0 orchestrator's "done" marker in its log
#   (2) short settle, then the inner smoke's own wait_for_gpu 9000 holds until the GPU is actually free
set -u
R1R0_LOG="logs/smoke_readout_stockattn/orchestrator.log"
echo "==== run_combo_after_r1r0 armed $(date); waiting for R1->R0 to finish ===="
while ! grep -q "run_readout_r1_r0 done" "$R1R0_LOG" 2>/dev/null; do sleep 120; done
echo "---- R1->R0 done marker seen $(date); settle 60s then launch (inner wait_for_gpu backstops) ----"
sleep 60
SEED=42 bash scripts/smoke_readout_warmq_combo.sh
echo "==== run_combo_after_r1r0 done $(date) ===="
