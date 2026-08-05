#!/usr/bin/env python
"""Phase 6 — Ablation matrix: Row A (fixed-R) vs Row B (reliability R(t)).

Runs the vendored Chiu tracker across all 10 val conditions for:
  - Row A: fixed R (no score file) — the baseline.
  - Row B: reliability-adaptive R(t) with kappa=3 and the bev_real score file.

Then evaluates each tracking result with eval_tracking.py (partial_val patch).

Outputs:
  /workspace/tracking_results/rowA_<condition>.json
  /workspace/tracking_results/rowB_<condition>_kappa3.json
  /workspace/results/rowA_<condition>_metrics.json
  /workspace/results/rowB_<condition>_kappa3_metrics.json

Usage:
    python run_ablation_matrix.py --row A
    python run_ablation_matrix.py --row B
    python run_ablation_matrix.py --row both
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List

PY = "/workspace/miniconda3/envs/bevfusion/bin/python"
TRACKER_DIR = "/workspace/stage_c_tracking/tracker_fork"
EVAL_PY = "/workspace/stage_c_tracking/eval_tracking.py"
DETECTIONS_DIR = "/workspace/detections"
SCORES_DIR = "/workspace/scores_bev_real"
TRACKING_RESULTS_DIR = "/workspace/tracking_results"
RESULTS_DIR = "/workspace/results"
DATA_ROOT = "/workspace/mmdetection3d/data/nuscenes"

# Chiu proposed-method config (run.sh line 22): val 2 m 11 greedy true nuscenes
COVARIANCE_ID = "2"
MATCH_DISTANCE = "m"
MATCH_THRESHOLD = "11"
MATCH_ALGORITHM = "greedy"
USE_ANGULAR_VELOCITY = "true"
KAPPA = 3.0

CONDITIONS = [
    "clean_val_sev0",
    "beamsreducing_val_sev1", "beamsreducing_val_sev2", "beamsreducing_val_sev3",
    "missingcamera_val_sev1", "missingcamera_val_sev2", "missingcamera_val_sev3",
    "motionblur_val_sev1", "motionblur_val_sev2", "motionblur_val_sev3",
]


def run_tracker(condition: str, row: str, kappa: float = KAPPA) -> str:
    """Run the tracker for one condition. Returns the output tracking JSON path."""
    detection_file = os.path.join(DETECTIONS_DIR, condition, "pred_instances_3d", "pred_instances_3d", "results_nusc.json")
    if not os.path.exists(detection_file):
        raise FileNotFoundError(f"detection JSON missing: {detection_file}")

    if row == "A":
        # baseline: no score file -> R(t) = R_base
        out_path = os.path.join(TRACKING_RESULTS_DIR, f"rowA_{condition}.json")
        score_file_arg = []
    else:  # Row B
        score_file = os.path.join(SCORES_DIR, f"{condition}_scores.npz")
        if not os.path.exists(score_file):
            raise FileNotFoundError(f"score file missing: {score_file}")
        out_path = os.path.join(TRACKING_RESULTS_DIR, f"rowB_{condition}_kappa{int(kappa)}.json")
        score_file_arg = ["--score-file", score_file, "--kappa", str(kappa)]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[tracker] {row} {condition} -> {out_path} EXISTS, skipping")
        return out_path

    cmd = [
        PY, os.path.join(TRACKER_DIR, "main.py"),
        "val", COVARIANCE_ID, MATCH_DISTANCE, MATCH_THRESHOLD,
        MATCH_ALGORITHM, USE_ANGULAR_VELOCITY, "nuscenes",
        os.path.dirname(out_path),  # save_root
        "--detection-file", detection_file,
        "--data-root", DATA_ROOT,
        "--output", out_path,
    ] + score_file_arg

    print(f"\n[tracker] {row} {condition} -> {out_path}")
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=TRACKER_DIR)
    if proc.returncode != 0:
        raise RuntimeError(f"tracker failed for {row} {condition}")
    if not os.path.exists(out_path):
        raise RuntimeError(f"tracker output missing: {out_path}")
    return out_path


def eval_tracking(tracking_json: str, row: str, condition: str, kappa: float = KAPPA) -> str:
    """Evaluate one tracking JSON. Returns the metrics JSON path."""
    suffix = f"_kappa{int(kappa)}" if row == "B" else ""
    out_dir = os.path.join(RESULTS_DIR, f"{row}_{condition}{suffix}_eval")
    metrics_out = os.path.join(RESULTS_DIR, f"{row}_{condition}{suffix}_metrics.json")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [PY, EVAL_PY,
           "--result-path", tracking_json,
           "--output-dir", out_dir,
           "--metrics-out", metrics_out]
    print(f"[eval] {row} {condition} -> {metrics_out}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed for {row} {condition}")
    return metrics_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", choices=["A", "B", "both"], default="both")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    args = parser.parse_args()

    conds = args.conditions if args.conditions else CONDITIONS
    rows = ["A", "B"] if args.row == "both" else [args.row]

    os.makedirs(TRACKING_RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    failed = []
    for row in rows:
        for cond in conds:
            try:
                tracking_json = run_tracker(cond, row, args.kappa)
                eval_tracking(tracking_json, row, cond, args.kappa)
            except Exception as e:
                print(f"[ERROR] {row} {cond}: {e}")
                failed.append(f"{row}_{cond}")

    print("\n=== ablation matrix summary ===")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print("  ALL OK")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()