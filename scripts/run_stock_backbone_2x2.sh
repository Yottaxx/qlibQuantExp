#!/usr/bin/env bash
# h-20260624 2x2: stock-backbone main-path residual, 2 arms (gamma_init 0.0 / 1.0) x 2 seeds (42/43).
# Serial, one process at a time; each inner launcher has its own wait_for_gpu 9000 + retry.
# ArmA gbb_g0 = ReZero opt-in (branch starts no-op); ArmB gbb_g1 = full-on from step 1.
set -u
OUTDIR="logs/smoke_stock_backbone"
mkdir -p "$OUTDIR"
ORCH="$OUTDIR/orchestrator_2x2.log"
echo "==== run_stock_backbone_2x2 start $(date) ====" | tee -a "$ORCH"
for g in 0.0 1.0; do
  for s in 42 43; do
    echo "---- launching gamma_init=$g seed=$s $(date) ----" | tee -a "$ORCH"
    GAMMA_INIT=$g SEED=$s bash scripts/smoke_stock_backbone.sh
    echo "---- returned gamma_init=$g seed=$s ec=$? $(date) ----" | tee -a "$ORCH"
  done
done
echo "==== run_stock_backbone_2x2 done $(date) ====" | tee -a "$ORCH"
