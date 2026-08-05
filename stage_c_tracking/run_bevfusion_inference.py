#!/usr/bin/env python
"""Phase 2.3 — Run BEVFusion inference with persisted detection JSONs.

For each of the 10 val conditions, runs mmdetection3d/tools/test.py with
test_evaluator.jsonfile_prefix set to persist results_nusc.json (the
nuScenes-format detection file the tracker ingests).

Outputs: /workspace/detections/<condition>/pred_instances_3d/results_nusc.json

Usage:
    python run_bevfusion_inference.py --condition clean_val_sev0
    python run_bevfusion_inference.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

PY = "/workspace/miniconda3/envs/bevfusion/bin/python"
BEV_CONFIG = "/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
BEV_CKPT = "/workspace/mmdetection3d/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
TEST_PY = "/workspace/mmdetection3d/tools/test.py"
EVAL_ROOT_BASE = "/workspace/eval_roots_new"
DETECTIONS_DIR = "/workspace/detections"
LOGS_DIR = "/workspace/logs"

CONDITIONS = [
    "clean_val_sev0",
    "beamsreducing_val_sev1", "beamsreducing_val_sev2", "beamsreducing_val_sev3",
    "missingcamera_val_sev1", "missingcamera_val_sev2", "missingcamera_val_sev3",
    "motionblur_val_sev1", "motionblur_val_sev2", "motionblur_val_sev3",
]


def run_one(condition: str, verify_first: bool = False) -> int:
    eval_root = os.path.join(EVAL_ROOT_BASE, condition)
    if not os.path.isdir(eval_root):
        print(f"[ERROR] eval root missing: {eval_root}")
        return 1

    out_dir = os.path.join(DETECTIONS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)
    jsonfile_prefix = os.path.join(out_dir, "pred_instances_3d")
    os.makedirs(jsonfile_prefix, exist_ok=True)

    log_path = os.path.join(LOGS_DIR, f"det_{condition}.log")

    cmd = [
        PY, TEST_PY, BEV_CONFIG, BEV_CKPT,
        "--cfg-options",
        f"test_dataloader.dataset.data_root={eval_root}/",
        f"test_evaluator.jsonfile_prefix={jsonfile_prefix}",
    ]
    print(f"[inference] {condition}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd="/workspace/mmdetection3d")

    # verify the JSON was written (evaluator appends 'pred_instances_3d' to the prefix)
    results_json = os.path.join(jsonfile_prefix, "pred_instances_3d", "results_nusc.json")
    if not os.path.exists(results_json):
        print(f"[ERROR] {condition}: results_nusc.json NOT written (check {log_path})")
        return 1
    size = os.path.getsize(results_json)
    if size < 1_000_000:
        print(f"[ERROR] {condition}: results_nusc.json too small ({size} bytes < 1MB)")
        return 1
    print(f"[inference] {condition}: OK ({size} bytes) -> {results_json}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default=None, help="one condition")
    parser.add_argument("--all", action="store_true", help="run all 10 conditions")
    parser.add_argument("--conditions", nargs="+", default=None, help="specific conditions")
    args = parser.parse_args()

    if args.all:
        conds = CONDITIONS
    elif args.conditions:
        conds = args.conditions
    elif args.condition:
        conds = [args.condition]
    else:
        parser.error("specify --condition, --conditions, or --all")

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(DETECTIONS_DIR, exist_ok=True)

    failed = []
    for i, cond in enumerate(conds):
        print(f"\n=== [{i+1}/{len(conds)}] {cond} ===")
        rc = run_one(cond)
        if rc != 0:
            failed.append(cond)
            # if the FIRST condition fails, abort early (per plan: verify first before rest)
            if i == 0:
                print(f"[ABORT] first condition {cond} failed; stopping.")
                break

    print("\n=== inference summary ===")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print("  ALL OK")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()