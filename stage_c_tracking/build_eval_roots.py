"""Build eval roots for all 10 val conditions (Phase 2.2).

For clean: use /workspace/mmdetection3d/data/nuscenes directly.
For corrupted: create a hybrid root (clean symlinks + corrupted sensor dirs)
  at /workspace/eval_roots_new/<condition>/.

The corrupted data lives at /workspace/corrupted_val/<corruption>/sev<N>/
with subdirs samples/<SENSOR>/ (corrupted keyframes). Sweeps stay clean
(matching the original Stage A behavior — the converters' sweep loop is a
no-op with the dev-1.x info pkl format).

Usage:
    python build_eval_roots.py            # build all 10
    python build_eval_roots.py --verify   # verify all 10 exist + sensor dirs
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

CLEAN_ROOT = "/workspace/mmdetection3d/data/nuscenes"
EVAL_ROOT_BASE = "/workspace/eval_roots_new"
CORRUPTED_BASE = "/workspace/corrupted_val"

# condition -> (corruption_dir_relative, sensor_subdirs_to_overlay)
# beamsreducing/motionblur corrupt LIDAR; missingcamera/motionblur corrupt CAM.
# motionblur corrupts BOTH (img + lidar into the same dir).
CONDITIONS: Dict[str, Dict] = {
    "clean_val_sev0": {
        "corrupted_dir": None,  # use clean root directly
        "sensors": [],
    },
    "beamsreducing_val_sev1": {"corrupted_dir": "beamsreducing/sev1", "sensors": ["LIDAR_TOP"]},
    "beamsreducing_val_sev2": {"corrupted_dir": "beamsreducing/sev2", "sensors": ["LIDAR_TOP"]},
    "beamsreducing_val_sev3": {"corrupted_dir": "beamsreducing/sev3", "sensors": ["LIDAR_TOP"]},
    "missingcamera_val_sev1": {"corrupted_dir": "missingcamera/sev1",
                                "sensors": ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                                            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]},
    "missingcamera_val_sev2": {"corrupted_dir": "missingcamera/sev2",
                                "sensors": ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                                            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]},
    "missingcamera_val_sev3": {"corrupted_dir": "missingcamera/sev3",
                                "sensors": ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                                            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]},
    "motionblur_val_sev1": {"corrupted_dir": "motionblur/sev1",
                            "sensors": ["LIDAR_TOP", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                                         "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]},
    "motionblur_val_sev2": {"corrupted_dir": "motionblur/sev2",
                            "sensors": ["LIDAR_TOP", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                                         "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]},
    "motionblur_val_sev3": {"corrupted_dir": "motionblur/sev3",
                            "sensors": ["LIDAR_TOP", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                                         "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]},
}


def build_eval_root(condition: str, spec: Dict, force: bool = False) -> str:
    """Build a hybrid eval root for one condition. Returns the eval root path."""
    eval_root = os.path.join(EVAL_ROOT_BASE, condition)

    if spec["corrupted_dir"] is None:
        # clean: just symlink the whole clean root
        if force and os.path.exists(eval_root):
            os.system(f"rm -rf {eval_root}")
        os.makedirs(EVAL_ROOT_BASE, exist_ok=True)
        if not os.path.exists(eval_root):
            os.symlink(CLEAN_ROOT, eval_root)
        return eval_root

    corrupted_dir = os.path.join(CORRUPTED_BASE, spec["corrupted_dir"])
    if not os.path.isdir(corrupted_dir):
        raise FileNotFoundError(f"corrupted dir missing: {corrupted_dir}")

    if force and os.path.exists(eval_root):
        os.system(f"rm -rf {eval_root}")
    os.makedirs(eval_root, exist_ok=True)

    # symlink everything from clean root
    for entry in os.listdir(CLEAN_ROOT):
        src = os.path.join(CLEAN_ROOT, entry)
        dst = os.path.join(eval_root, entry)
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        os.symlink(src, dst)

    # overlay corrupted sensor dirs
    for sensor in spec["sensors"]:
        # e.g. samples/LIDAR_TOP
        for parent in ["samples", "sweeps"]:
            corrupted_sensor_dir = os.path.join(corrupted_dir, parent, sensor)
            if not os.path.isdir(corrupted_sensor_dir):
                continue
            # create parent dir in eval root (replace symlink with real dir)
            parent_link = os.path.join(eval_root, parent)
            if os.path.islink(parent_link):
                real_parent = os.path.join(eval_root, parent)
                os.remove(parent_link)
                os.makedirs(real_parent, exist_ok=True)
                # re-symlink the other (clean) sensors into the real parent
                clean_parent = os.path.join(CLEAN_ROOT, parent)
                for clean_sensor in os.listdir(clean_parent):
                    if clean_sensor == sensor:
                        continue
                    src = os.path.join(clean_parent, clean_sensor)
                    dst = os.path.join(real_parent, clean_sensor)
                    if not os.path.exists(dst):
                        os.symlink(src, dst)
            # now symlink the corrupted sensor dir
            sensor_dst = os.path.join(eval_root, parent, sensor)
            if os.path.exists(sensor_dst) or os.path.islink(sensor_dst):
                os.remove(sensor_dst)
            os.symlink(corrupted_sensor_dir, sensor_dst)

    return eval_root


def verify_eval_root(condition: str, spec: Dict) -> bool:
    """Verify the eval root has all expected sensor dirs."""
    eval_root = os.path.join(EVAL_ROOT_BASE, condition)
    if not os.path.isdir(eval_root):
        print(f"  {condition}: MISSING eval root {eval_root}")
        return False
    ok = True
    # check the 7 sample sensor dirs exist
    for sensor in ["LIDAR_TOP", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                   "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]:
        p = os.path.join(eval_root, "samples", sensor)
        if not os.path.isdir(p):
            print(f"  {condition}: MISSING samples/{sensor}")
            ok = False
    # check val pkl exists
    if not os.path.exists(os.path.join(eval_root, "nuscenes_infos_val.pkl")):
        print(f"  {condition}: MISSING nuscenes_infos_val.pkl")
        ok = False
    if ok:
        print(f"  {condition}: OK")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if exists")
    parser.add_argument("--verify", action="store_true", help="only verify")
    args = parser.parse_args()

    if args.verify:
        print("=== verifying eval roots ===")
        all_ok = True
        for cond, spec in CONDITIONS.items():
            if not verify_eval_root(cond, spec):
                all_ok = False
        print(f"[{'PASS' if all_ok else 'FAIL'}]")
        sys.exit(0 if all_ok else 1)

    print("=== building eval roots ===")
    for cond, spec in CONDITIONS.items():
        try:
            path = build_eval_root(cond, spec, force=args.force)
            print(f"  {cond}: -> {path}")
            verify_eval_root(cond, spec)
        except FileNotFoundError as e:
            print(f"  {cond}: SKIP ({e})")


if __name__ == "__main__":
    main()