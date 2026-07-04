#!/usr/bin/env bash
# Serial orchestrator for the warm-q 5-seed sweep TAIL: run seeds 44, 45, 46 one process at a time
# (seeds 42 @6/17 and 43 @6/19 already ran => 42..46 = the full n=5 set).
# Each inner launcher (smoke_warmq_stockexp.sh) has its OWN wait_for_gpu 9000 + retry, so this
# orchestrator naturally QUEUES behind the running readout_warmqcombo run: it sits in gpu-wait until
# the combo frees >=9000MiB, then starts seed 44, then 45, then 46 — strictly sequential, no overlap.
set -u
OUTDIR="logs/smoke_warmq_stockexp"
mkdir -p "$OUTDIR"
ORCH="$OUTDIR/orchestrator_44_45_46.log"
echo "==== run_warmq_44_45_46 start $(date) ====" | tee -a "$ORCH"
for s in 44 45 46; do
  echo "---- launching seed $s $(date) ----" | tee -a "$ORCH"
  SEED=$s bash scripts/smoke_warmq_stockexp.sh
  echo "---- seed $s returned ec=$? $(date) ----" | tee -a "$ORCH"
done
echo "==== run_warmq_44_45_46 done $(date) ====" | tee -a "$ORCH"
