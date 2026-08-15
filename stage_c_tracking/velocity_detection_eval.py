#!/usr/bin/env python
"""Replace detection velocities with tracker-smoothed velocities and run the nuScenes
detection evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np


# nuScenes TP distance threshold for the 7 detection classes (center distance,
# 2-D). Boxes closer than this are considered matched. Used for velocity grafting.
DIST_TH_TP = 2.0


def patch_partial_val(data_root: str, version: str = "v1.0-trainval"):
    """Monkeypatch create_splits_scenes so 'val' -> partial_val (34 scenes)."""
    import nuscenes.eval.common.loaders as nusc_loaders

    splits_path = os.path.join(data_root, version, "splits.json")
    with open(splits_path) as f:
        _partial_val_scenes = json.load(f)["partial_val"]

    _original_create_splits_scenes = nusc_loaders.create_splits_scenes

    def _patched_create_splits_scenes(verbose: bool = False):
        splits = _original_create_splits_scenes(verbose=verbose)
        splits["val"] = _partial_val_scenes
        return splits

    nusc_loaders.create_splits_scenes = _patched_create_splits_scenes
    return _partial_val_scenes


def convert_velocity(det_json: str, track_json: str, out_json: str,
                     dist_th: float = DIST_TH_TP) -> dict:
    """Replace detection velocities with tracker Kalman velocities."""
    with open(det_json) as f:
        det = json.load(f)
    with open(track_json) as f:
        trk = json.load(f)

    det_results: Dict[str, list] = det["results"]
    trk_results: Dict[str, list] = trk.get("results", {})

    n_total = 0
    n_matched = 0

    for sample_token, dets in det_results.items():
        tracks = trk_results.get(sample_token, [])
        # group tracks by class for fast nearest lookup
        by_class: Dict[str, List[tuple]] = {}
        for t in tracks:
            cls = t.get("tracking_name")
            tx, ty = float(t["translation"][0]), float(t["translation"][1])
            vx, vy = float(t["velocity"][0]), float(t["velocity"][1])
            by_class.setdefault(cls, []).append((tx, ty, vx, vy))

        for d in dets:
            n_total += 1
            cls = d.get("detection_name")
            cands = by_class.get(cls)
            if not cands:
                continue
            dx, dy = float(d["translation"][0]), float(d["translation"][1])
            best_d2 = dist_th * dist_th
            best = None
            for (tx, ty, vx, vy) in cands:
                d2 = (tx - dx) ** 2 + (ty - dy) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = (vx, vy)
            if best is not None:
                d["velocity"] = [best[0], best[1]]
                n_matched += 1

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(det, f)
    rate = (n_matched / n_total) if n_total else 0.0
    print(f"[convert] {os.path.basename(out_json)}: matched {n_matched}/{n_total} "
          f"detections ({rate:.1%})")
    return {"n_matched": n_matched, "n_total": n_total, "match_rate": rate}


def evaluate_detection(result_path: str, output_dir: str,
                       data_root: str = "/workspace/mmdetection3d/data/nuscenes",
                       version: str = "v1.0-trainval",
                       eval_set: str = "val") -> Dict:
    """Run nuScenes DetectionEval on a detection-format JSON (partial_val)."""
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    partial_val_scenes = patch_partial_val(data_root, version)
    config = config_factory("detection_cvpr_2019")
    os.makedirs(output_dir, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    nusc_eval = DetectionEval(
        nusc=nusc,
        config=config,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    metrics = nusc_eval.main(render_curves=False)

    # read the summary written by the evaluator for per-class detail
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # nuScenes DetectionEval.main() returns mean_ap/nd_score; TP errors live in
    # metrics_summary['tp_errors'] keyed by error type (trans/scale/orient/vel/
    # attr), already averaged over supporting classes.
    te = summary.get("tp_errors", {}) or {}

    def _f(x):
        return float(x) if isinstance(x, (int, float)) else float("nan")

    result = {
        "result_path": result_path,
        "output_dir": output_dir,
        "n_scenes": len(partial_val_scenes),
        "map": _f(summary.get("mean_ap")),
        "nds": _f(summary.get("nd_score")),
        "mate": _f(te.get("trans_err")),
        "mase": _f(te.get("scale_err")),
        "maoe": _f(te.get("orient_err")),
        "mave": _f(te.get("vel_err")),
        "maae": _f(te.get("attr_err")),
        "tp": int(metrics.get("tp", -1)) if str(metrics.get("tp", "")).isdigit() else metrics.get("tp"),
        "fp": int(metrics.get("fp", -1)) if str(metrics.get("fp", "")).isdigit() else metrics.get("fp"),
        "fn": int(metrics.get("fn", -1)) if str(metrics.get("fn", "")).isdigit() else metrics.get("fn"),
        "summary": summary,
    }

    print("\n" + "=" * 60)
    print(f"DETECTION EVAL: {os.path.basename(result_path)}")
    print("=" * 60)
    print(f"  scenes: {result['n_scenes']}")
    print(f"  mAP:    {result['map']:.4f}")
    print(f"  NDS:    {result['nds']:.4f}")
    print(f"  mATE:   {result['mate']:.4f}")
    print(f"  mASE:   {result['mase']:.4f}")
    print(f"  mAOE:   {result['maoe']:.4f}")
    print(f"  mAVE:   {result['mave']:.4f}")
    print(f"  mAAE:   {result['maae']:.4f}")
    print("=" * 60)
    return result


def main():
    parser = argparse.ArgumentParser(description="velocity graft + detection eval")
    sub = parser.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("convert", help="graft tracker velocities onto a detection JSON")
    pc.add_argument("--det-json", required=True)
    pc.add_argument("--track-json", required=True)
    pc.add_argument("--out-json", required=True)
    pc.add_argument("--dist-th", type=float, default=DIST_TH_TP)

    pe = sub.add_parser("eval", help="run DetectionEval on a detection-format JSON")
    pe.add_argument("--result-path", required=True)
    pe.add_argument("--output-dir", required=True)
    pe.add_argument("--metrics-out", default=None)
    pe.add_argument("--data-root", default="/workspace/mmdetection3d/data/nuscenes")
    pe.add_argument("--version", default="v1.0-trainval")
    pe.add_argument("--eval-set", default="val")

    args = parser.parse_args()

    if args.mode == "convert":
        convert_velocity(args.det_json, args.track_json, args.out_json, args.dist_th)
    elif args.mode == "eval":
        result = evaluate_detection(
            result_path=args.result_path,
            output_dir=args.output_dir,
            data_root=args.data_root,
            version=args.version,
            eval_set=args.eval_set,
        )
        metrics_out = args.metrics_out or os.path.join(args.output_dir, "metrics_extracted.json")
        with open(metrics_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[eval] metrics -> {metrics_out}")


if __name__ == "__main__":
    main()
