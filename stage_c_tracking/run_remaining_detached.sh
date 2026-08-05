#!/bin/bash
# Detached Stage C finisher: Phase 6.3 (rebuild + det eval) + Phase 7 (kappa sweep).
# Run with:  setsid nohup bash run_remaining_detached.sh > LOG 2>&1 < /dev/null &
# Self-gates on velocity re-runs finishing; sequential (peak mem ~32 GiB < 57.74).
set -u
PY=/workspace/miniconda3/envs/bevfusion/bin/python
STC=/workspace/stage_c_tracking
DET=/workspace/detections
SCORES=/workspace/scores_bev_real
TRACK_OUT=/workspace/tracking_results
TRK_VEL=/workspace/tracking_results_vel
DATA=/workspace/mmdetection3d/data/nuscenes
LOG=/workspace/logs

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

# ---- Phase 0: wait for all 20 velocity re-runs to finish -------------------
CONDS="clean_val_sev0 beamsreducing_val_sev1 beamsreducing_val_sev2 beamsreducing_val_sev3 missingcamera_val_sev1 missingcamera_val_sev2 missingcamera_val_sev3 motionblur_val_sev1 motionblur_val_sev2 motionblur_val_sev3"
count_vel(){ local n=0; for c in $CONDS; do for r in rowA rowB; do suf=""; [ $r = rowB ] && suf="_kappa3"; [ -s "$TRK_VEL/${r}_${c}${suf}.json" ] && n=$((n+1)); done; done; echo $n; }
log "waiting for velocity re-runs (need 20, have $(count_vel))..."
for i in $(seq 1 60); do   # up to 30 min
  [ "$(count_vel)" -ge 20 ] && break
  sleep 30
done
log "velocity files present: $(count_vel)/20"
[ "$(count_vel)" -lt 20 ] && { log "ABORT: velocity re-runs incomplete"; exit 1; }

# ---- Phase 1: rebuild detections_vel with REAL velocities (--force) --------
log "=== Phase 6.3a: rebuild detections_vel/ with real Kalman velocities ==="
$PY -u $STC/rebuild_detections_vel.py --force 2>&1 | tee $LOG/rebuild_dvel.log
log "detections_vel files: $(ls /workspace/detections_vel/*/results_nusc.json 2>/dev/null | wc -l)/20"

# ---- Phase 2: detection eval (mAP/NDS/mAVE) load-once ----------------------
log "=== Phase 6.3b: detection eval on velocity-grafted files ==="
$PY -u $STC/batch_eval.py det 2>&1 | tee $LOG/det_eval_batch.log
log "det_metrics present: $(ls /workspace/results/*_det_metrics.json 2>/dev/null | wc -l)/20"

# ---- Phase 3: kappa-sweep tracker runs (kappa=1,5 on clean + motionblur) --
log "=== Phase 7a: kappa-sweep tracker runs (parallel=3) ==="
det_for(){ local c=$1; local f="$DET/$c/pred_instances_3d/pred_instances_3d/results_nusc.json"; [ -f "$f" ] || f="$DET/$c/pred_instances_3d/results_nusc.json"; echo "$f"; }
run_kappa(){
  local c=$1 k=$2; local out="$TRACK_OUT/rowB_${c}_kappa${k}.json"
  [ -s "$out" ] && { log "skip $out"; return 0; }
  local detf; detf=$(det_for "$c")
  log "launch kappa=$k $c"
  ( cd $STC/tracker_fork && $PY -u main.py val 2 m 11 greedy true nuscenes "$(dirname "$out")" \
      --detection-file "$detf" --data-root "$DATA" --output "$out" \
      --score-file "$SCORES/${c}_scores.npz" --kappa "$k" \
      > "$LOG/kappa${k}_${c}.log" 2>&1 )
  [ -s "$out" ] && log "done $out" || log "FAIL $out"
}
KAPPA_CONDS="clean_val_sev0 motionblur_val_sev1 motionblur_val_sev2 motionblur_val_sev3"
running=0
for k in 1 5; do
  for c in $KAPPA_CONDS; do
    while [ "$(pgrep -f 'main.py val' | wc -l)" -ge 3 ]; do sleep 3; done
    run_kappa "$c" "$k" &
    running=$((running+1)); sleep 4
  done
done
wait
log "kappa tracker runs done. kappa tracking files: $(ls $TRACK_OUT/rowB_*_kappa1.json $TRACK_OUT/rowB_*_kappa5.json 2>/dev/null | wc -l)/8"

# ---- Phase 4: eval the new kappa1/kappa5 tracking (load-once, idempotent) --
log "=== Phase 7b: tracking eval for kappa1/kappa5 ==="
$PY -u $STC/batch_eval.py track 2>&1 | tee $LOG/track_eval_kappa.log
log "kappa1 metrics: $(ls /workspace/results/rowB_*_kappa1_metrics.json 2>/dev/null | wc -l), kappa5 metrics: $(ls /workspace/results/rowB_*_kappa5_metrics.json 2>/dev/null | wc -l)"

# ---- Phase 5: assemble kappa sweep table + freeze kappa -------------------
log "=== Phase 7c: assemble kappa sweep ==="
$PY -u $STC/kappa_sweep.py --kappas 1 3 5 --conditions clean_val_sev0 motionblur_val_sev1 motionblur_val_sev2 motionblur_val_sev3 2>&1 | tee $LOG/kappa_sweep.log

log "================ ALL DETACHED PHASES COMPLETE ================"
log "det_metrics: $(ls /workspace/results/*_det_metrics.json 2>/dev/null | wc -l)/20"
log "kappa1 metrics: $(ls /workspace/results/rowB_*_kappa1_metrics.json 2>/dev/null | wc -l)/4"
log "kappa5 metrics: $(ls /workspace/results/rowB_*_kappa5_metrics.json 2>/dev/null | wc -l)/4"
log "kappa_sweep.json: $([ -f /workspace/results/kappa_sweep.json ] && echo yes || echo NO)"
log "DONE."
