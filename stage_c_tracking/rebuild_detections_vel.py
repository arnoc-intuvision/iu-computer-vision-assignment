#!/usr/bin/env python
"""Phase 6.3 — rebuild /workspace/detections_vel/ from REAL Kalman velocities.

The earlier detections_vel graft used tracking_results/ which had all-zero
velocities (pre-velocity-emission tracker output). This rebuilds them from
tracking_results_vel/ (the re-runs that emit kf.x[7]/kf.x[8] smoothed velocity).

For each (row, condition) it grafts tracker velocities onto the original
BEVFusion detection JSON, producing detections_vel/<row>_<cond>[_kappa3]/
results_nusc.json. Only `velocity` is replaced; box set, scores, sizes,
rotations are preserved so the mAVE delta isolates velocity-estimation quality.

Idempotent: pass --force to overwrite existing files.

Usage:
    python rebuild_detections_vel.py            # graft all missing
    python rebuild_detections_vel.py --force    # rebuild all
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from velocity_detection_eval import convert_velocity  # noqa: E402

DET_DIR = "/workspace/detections"
TRACK_VEL_DIR = "/workspace/tracking_results_vel"
DET_VEL_DIR = "/workspace/detections_vel"

CONDITIONS = [
    "clean_val_sev0",
    "beamsreducing_val_sev1", "beamsreducing_val_sev2", "beamsreducing_val_sev3",
    "missingcamera_val_sev1", "missingcamera_val_sev2", "missingcamera_val_sev3",
    "motionblur_val_sev1", "motionblur_val_sev2", "motionblur_val_sev3",
]


def det_json_for(cond):
    p = os.path.join(DET_DIR, cond, "pred_instances_3d", "pred_instances_3d",
                     "results_nusc.json")
    if os.path.exists(p):
        return p
    p = os.path.join(DET_DIR, cond, "pred_instances_3d", "results_nusc.json")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing grafts")
    args = ap.parse_args()

    jobs = []
    for cond in CONDITIONS:
        jobs.append(("rowA", cond, os.path.join(TRACK_VEL_DIR, f"rowA_{cond}.json")))
        jobs.append(("rowB", cond, os.path.join(TRACK_VEL_DIR, f"rowB_{cond}_kappa3.json")))

    missing_track = []
    for row, cond, track_json in jobs:
        out_json = os.path.join(DET_VEL_DIR, f"{row}_{cond}" +
                                ("_kappa3" if row == "rowB" else ""), "results_nusc.json")
        if os.path.exists(out_json) and os.path.getsize(out_json) > 0 and not args.force:
            print(f"[skip] {os.path.basename(os.path.dirname(out_json))} (exists)")
            continue
        if not os.path.exists(track_json):
            print(f"[MISSING] {track_json} (velocity rerun not done yet)")
            missing_track.append(track_json)
            continue
        det_json = det_json_for(cond)
        if not os.path.exists(det_json):
            print(f"[ERR] no detection JSON for {cond}: {det_json}")
            continue
        stats = convert_velocity(det_json, track_json, out_json)
        print(f"  match_rate={stats['match_rate']:.1%}")

    if missing_track:
        print(f"\n[WARN] {len(missing_track)} tracking_results_vel files still missing:")
        for m in missing_track:
            print(f"  {m}")
        sys.exit(1)
    print("\n[rebuild] all detections_vel grafts complete.")


if __name__ == "__main__":
    main()
