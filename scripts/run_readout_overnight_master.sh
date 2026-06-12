#!/usr/bin/env bash
# SERIALIZED overnight master: Step-A screen (n=6) -> Step-B finetune (n=3). One GPU/qlib
# process at a time (no concurrency). Joblib temp pinned to a stable repo-local dir to avoid
# the volatile claude temp that caused cross-process FileNotFoundError. continue-on-error.
set -u
ROOT="C:/Users/60585/PycharmProjects/qibMacV2"
mkdir -p "$ROOT/tmp_run"
export TMP="$ROOT/tmp_run"; export TEMP="$TMP"; export TMPDIR="$TMP"; export JOBLIB_TEMP_FOLDER="$TMP"
export PYTHONIOENCODING=utf-8
M="logs/readout_overnight_master.log"
echo "==== MASTER start $(date) (TMP=$TMP) ====" > "$M"

echo "" >> "$M"; echo "#### PHASE 1: Step-A screen (frozen, n=6 seeds 42-47) ####" >> "$M"
bash scripts/run_time_readout_stepA.sh >> "$M" 2>&1
echo "PHASE1_EXIT=$? (see logs/time_readout_stepA.log)" >> "$M"

echo "" >> "$M"; echo "#### PHASE 2: Step-B finetune (readout+top-block, control vs d3cid, n=3 seeds 42-44) ####" >> "$M"
bash scripts/run_time_readout_stepB.sh >> "$M" 2>&1
echo "PHASE2_EXIT=$? (see logs/time_readout_stepB.log)" >> "$M"

echo "" >> "$M"; echo "==== MASTER done $(date) ====" >> "$M"
echo "RESULT: analysis/readout_redesign/RESULT_overnight.md" >> "$M"
