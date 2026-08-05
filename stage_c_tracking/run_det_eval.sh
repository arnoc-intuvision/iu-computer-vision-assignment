#!/bin/bash
# Phase 6.3 detection eval for one velocity-grafted detection JSON.
# Usage: run_det_eval.sh <name>   where name = dir under /workspace/detections_vel
set -u
PY=/workspace/miniconda3/envs/bevfusion/bin/python
VDE=/workspace/stage_c_tracking/velocity_detection_eval.py
OUT=/workspace/detections_vel
RES=/workspace/results
LOG=/workspace/logs
name="$1"
rj="$OUT/$name/results_nusc.json"
mout="$RES/${name}_det_metrics.json"
odir="$RES/${name}_det_eval"
if [ -f "$mout" ]; then echo "skip $name"; exit 0; fi
if [ ! -f "$rj" ]; then echo "MISSING det json $rj"; exit 2; fi
$PY -u "$VDE" eval --result-path "$rj" --output-dir "$odir" --metrics-out "$mout" \
  > "$LOG/det_eval_$name.log" 2>&1
rc=$?
echo "done $name rc=$rc"
exit $rc
