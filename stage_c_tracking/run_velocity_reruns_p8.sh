#!/bin/bash
# Phase 6.3: re-run trackers WITH Kalman velocity emission -> tracking_results_vel/
# Robust: parallelism 8 with staggered starts to avoid NFS table-load thrash
# (the prior 20-way batch died from concurrent nuScenes table loads).
# Idempotent: skips any output JSON that already exists. Verifies each output
# is non-empty.
set -u
PY=/workspace/miniconda3/envs/bevfusion/bin/python
DET=/workspace/detections
SCORES=/workspace/scores_bev_real
OUT=/workspace/tracking_results_vel
DATA=/workspace/mmdetection3d/data/nuscenes
LOG=/workspace/logs
TRACKER=/workspace/stage_c_tracking/tracker_fork
PARALLEL=${1:-8}
STAGGER=${2:-3}
mkdir -p "$OUT"

CONDS="clean_val_sev0 beamsreducing_val_sev1 beamsreducing_val_sev2 beamsreducing_val_sev3 missingcamera_val_sev1 missingcamera_val_sev2 missingcamera_val_sev3 motionblur_val_sev1 motionblur_val_sev2 motionblur_val_sev3"

run_one() {
  local row="$1" c="$2" extra="$3" outf="$4"
  if [ -f "$outf" ] && [ -s "$outf" ]; then
    echo "[skip] $outf exists ($(wc -c < "$outf") bytes)"; return 0
  fi
  local detf="$DET/$c/pred_instances_3d/pred_instances_3d/results_nusc.json"
  if [ ! -f "$detf" ]; then
    # fall back to alternate layout
    detf="$DET/$c/pred_instances_3d/results_nusc.json"
  fi
  if [ ! -f "$detf" ]; then echo "[ERR] no detection JSON for $c"; return 1; fi
  echo "[launch] $outf"
  ( cd "$TRACKER" && $PY -u main.py val 2 m 11 greedy true nuscenes "$OUT" \
      --detection-file "$detf" --data-root "$DATA" --output "$outf" $extra \
      > "$LOG/vol_${row}_${c}.log" 2>&1 )
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$outf" ]; then
    echo "[FAIL] $outf rc=$rc (see $LOG/vol_${row}_${c}.log)"; return $rc
  fi
  echo "[done] $outf ($(wc -c < "$outf") bytes)"
  return 0
}
export -f run_one
export PY DET SCORES OUT DATA LOG TRACKER

# Build the job list (rowA = no score; rowB = kappa3 + score file)
declare -a JOBS
for c in $CONDS; do
  JOBS+=("rowA|$c||$OUT/rowA_${c}.json")
  JOBS+=("rowB|$c|--score-file $SCORES/${c}_scores.npz --kappa 3.0|$OUT/rowB_${c}_kappa3.json")
done

# Launch with staggered starts, cap at $PARALLEL concurrent
running=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r row c extra outf <<< "$job"
  # wait if at cap
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 2; done
  (
    if [ "$row" = "rowA" ]; then
      run_one rowA "$c" "" "$outf"
    else
      run_one rowB "$c" "--score-file $SCORES/${c}_scores.npz --kappa 3.0" "$outf"
    fi
  ) &
  running=$((running+1))
  sleep "$STAGGER"
done
wait
echo "=== VELOCITY RE-RUNS ALL DONE ==="; date
ls -la "$OUT"
