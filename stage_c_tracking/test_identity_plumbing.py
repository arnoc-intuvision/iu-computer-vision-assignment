"""Smoke test that the R(t) code path reproduces the baseline tracker when the score is 1."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

PY = "/workspace/miniconda3/envs/bevfusion/bin/python"
TRACKER_DIR = "/workspace/stage_c_tracking/tracker_fork"
DATA_ROOT = "/workspace/mmdetection3d/data/nuscenes"


def make_synthetic_detections(out_path: str, n_scenes: int = 2, frames_per_scene: int = 5):
    """Create a tiny synthetic detection JSON using real sample tokens."""
    from nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATA_ROOT, verbose=False)

    # take the first n_scenes scenes that have samples in our partial val
    import nuscenes.eval.common.loaders as L
    splits = L.create_splits_scenes()
    val_scenes = splits["val"][:n_scenes]

    results = {}
    for scene_name in val_scenes:
        # find the scene
        scene = None
        for s in nusc.scene:
            if s["name"] == scene_name:
                scene = s
                break
        if scene is None:
            continue
        token = scene["first_sample_token"]
        x = 100.0
        for f in range(frames_per_scene):
            if token == "":
                break
            # one synthetic car detection, moving in x
            results[token] = [{
                "sample_token": token,
                "translation": [x, 50.0, 0.5],
                "size": [2.0, 4.0, 1.5],  # w, l, h
                "rotation": [1.0, 0.0, 0.0, 0.0],  # identity quaternion
                "velocity": [0.0, 0.0],
                "detection_name": "car",
                "detection_score": 0.9,
                "attribute_name": "",
            }]
            x += 0.5
            token = nusc.get("sample", token)["next"]

    meta = {"use_camera": True, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    print(f"[synthetic] wrote {len(results)} frames to {out_path}")
    return list(results.keys())


def make_identity_score_file(out_path: str, tokens):
    s_t = np.ones(len(tokens), dtype=np.float32)
    np.savez_compressed(out_path,
                        tokens=np.array(tokens, dtype=str),
                        s_t=s_t,
                        sample_idx=np.arange(len(tokens)),
                        true_severity=np.zeros(len(tokens)),
                        reliability_target=np.ones(len(tokens)),
                        is_decoy=np.zeros(len(tokens), dtype=bool),
                        corruption_type=np.array("identity"),
                        model_hash=np.array("identity"))


def run_tracker(detection_file, output, score_file=None, kappa=3.0):
    cmd = [PY, os.path.join(TRACKER_DIR, "main.py"),
           "val", "2", "m", "11", "greedy", "true", "nuscenes",
           os.path.dirname(output),
           "--detection-file", detection_file,
           "--data-root", DATA_ROOT,
           "--output", output]
    if score_file:
        cmd += ["--score-file", score_file, "--kappa", str(kappa)]
    proc = subprocess.run(cmd, cwd=TRACKER_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:])
        raise RuntimeError(f"tracker failed: {proc.returncode}")
    return output


def main():
    with tempfile.TemporaryDirectory() as td:
        det_file = os.path.join(td, "synthetic_dets.json")
        tokens = make_synthetic_detections(det_file)

        id_score = os.path.join(td, "identity_scores.npz")
        make_identity_score_file(id_score, tokens)

        baseline_out = os.path.join(td, "baseline.json")
        identity_out = os.path.join(td, "identity.json")

        print("\n=== run 1: baseline (no score file) ===")
        run_tracker(det_file, baseline_out)

        print("\n=== run 2: identity (s_t=1.0) ===")
        run_tracker(det_file, identity_out, score_file=id_score, kappa=3.0)

        # byte-identical diff
        with open(baseline_out) as f:
            b = json.load(f)
        with open(identity_out) as f:
            i = json.load(f)

        # compare results (sort for determinism)
        b_str = json.dumps(b, sort_keys=True)
        i_str = json.dumps(i, sort_keys=True)

        if b_str == i_str:
            print("\n[PASS] baseline == identity (byte-identical). R(t) plumbing OK.")
            return 0
        else:
            print("\n[FAIL] baseline != identity")
            # show diff
            b_tokens = set(b["results"].keys())
            i_tokens = set(i["results"].keys())
            print(f"  baseline tokens: {len(b_tokens)}, identity tokens: {len(i_tokens)}")
            for tok in b_tokens & i_tokens:
                if b["results"][tok] != i["results"][tok]:
                    print(f"  DIFF at token {tok}:")
                    print(f"    baseline: {b['results'][tok]}")
                    print(f"    identity: {i['results'][tok]}")
                    break
            return 1


if __name__ == "__main__":
    sys.exit(main())