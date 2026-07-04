#!/usr/bin/env bash
# Time-gated launcher: idle until 14:00 today, then run the warm-q tail sweep (seeds 44,45,46).
# Launched detached NOW so the 2pm start does NOT depend on any agent being awake. GPU is free,
# so at 14:00 the inner smoke_warmq_stockexp.sh starts immediately (its wait_for_gpu is a no-op).
set -u
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
cd "$ROOT" || exit 1
OUTDIR="logs/smoke_warmq_stockexp"; mkdir -p "$OUTDIR"
GATE="$OUTDIR/gate_2pm.log"
echo "==== gate start $(date); will launch sweep at 14:00 ====" > "$GATE"
while (( 10#$(date +%H%M) < 1400 )); do
  echo "[gate] $(date) waiting for 14:00 ..." >> "$GATE"
  sleep 120
done
echo "==== 14:00 reached $(date); launching warm-q 44/45/46 ====" >> "$GATE"
bash scripts/run_warmq_44_45_46.sh >> "$GATE" 2>&1
echo "==== gate: sweep returned ec=$? $(date) ====" >> "$GATE"
