#!/bin/bash
# Phase 2.1 — Regenerate corrupted val sensor data via MultiCorrupt converters.
#
# The corrupted val data was deleted after Stage A caching. This regenerates it
# on the val split (1353 samples / 34 scenes) using the ORIGINAL converters
# (which hardcode nuscenes_infos_val.pkl — NOT the _train.py patched copies).
#
# Output: /workspace/corrupted_val/<corruption>/sev<N>/{samples,sweeps}/...
#
# Budget: beamsreducing ~1.5h, missingcamera ~1.5h, motionblur ~4.5h = ~7.5h.
# Run with: nohup bash regenerate_corrupted_val.sh > /workspace/logs/regen_corrupted_val.log 2>&1 &
set -e
set -u

PY=/workspace/miniconda3/envs/bevfusion/bin/python
ROOT=/workspace/mmdetection3d/data/nuscenes
DST_BASE=/workspace/corrupted_val
CONV=/workspace/MultiCorrupt/converter
CPUS=100

mkdir -p "$DST_BASE" /workspace/logs

echo "=== Phase 2.1: Regenerate corrupted val data ==="
date

# --- beamsreducing (LiDAR only) ---
for sev in 1 2 3; do
  echo "=== beamsreducing sev$sev ==="; date
  $PY "$CONV/lidar_converter.py" -a beamsreducing -f $sev -s True \
    -r "$ROOT" -d "$DST_BASE/beamsreducing/sev$sev" -c $CPUS
  # verify a few files
  n=$(ls "$DST_BASE/beamsreducing/sev$sev/samples/LIDAR_TOP/" 2>/dev/null | wc -l)
  echo "beamsreducing sev$sev: $n LIDAR_TOP files"
  if [ "$n" -lt 1353 ]; then echo "ERROR: expected 1353, got $n"; exit 1; fi
done

# --- missingcamera (camera only) ---
for sev in 1 2 3; do
  echo "=== missingcamera sev$sev ==="; date
  $PY "$CONV/img_converter.py" -a missingcamera -f $sev \
    -r "$ROOT" -d "$DST_BASE/missingcamera/sev$sev" -c $CPUS
  n=$(find "$DST_BASE/missingcamera/sev$sev/samples" -name "*.jpg" 2>/dev/null | wc -l)
  echo "missingcamera sev$sev: $n jpg files"
  if [ "$n" -lt 8118 ]; then echo "ERROR: expected ~8118 (1353*6), got $n"; exit 1; fi
done

# --- motionblur (cross-modal: img + lidar into same dir) ---
for sev in 1 2 3; do
  echo "=== motionblur sev$sev (img) ==="; date
  $PY "$CONV/img_converter.py" -a motionblur -f $sev \
    -r "$ROOT" -d "$DST_BASE/motionblur/sev$sev" -c $CPUS
  echo "=== motionblur sev$sev (lidar) ==="; date
  $PY "$CONV/lidar_converter.py" -a motionblur -f $sev -s True \
    -r "$ROOT" -d "$DST_BASE/motionblur/sev$sev" -c $CPUS
  n_lidar=$(ls "$DST_BASE/motionblur/sev$sev/samples/LIDAR_TOP/" 2>/dev/null | wc -l)
  n_img=$(find "$DST_BASE/motionblur/sev$sev/samples" -name "*.jpg" 2>/dev/null | wc -l)
  echo "motionblur sev$sev: $n_lidar LIDAR, $n_img jpg"
  if [ "$n_lidar" -lt 1353 ]; then echo "ERROR lidar: expected 1353, got $n_lidar"; exit 1; fi
  if [ "$n_img" -lt 8118 ]; then echo "ERROR img: expected ~8118, got $n_img"; exit 1; fi
done

echo "=== Phase 2.1 DONE ==="
date
