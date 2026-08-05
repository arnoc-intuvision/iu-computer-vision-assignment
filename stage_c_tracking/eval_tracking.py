"""Stage C tracking evaluation wrapper.

Wraps nuscenes.eval.tracking.TrackingEval with the partial_val monkeypatch
(same pattern as mmdet3d's nuscenes_metric.py lines 240-260) so the official
150-scene 'val' split resolves to our 34 on-disk scenes.

Usage:
    python eval_tracking.py --result-path <tracking.json> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np


def patch_partial_val(data_root: str, version: str = "v1.0-trainval"):
    """Monkeypatch create_splits_scenes so 'val' -> partial_val (34 scenes).

    Same pattern as mmdet3d/evaluation/metrics/nuscenes_metric.py lines 240-260.
    Must be called BEFORE constructing TrackingEval (which calls load_gt ->
    create_splits_scenes at eval time).
    """
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
    print(f"[eval_tracking] patched create_splits_scenes: val -> {len(_partial_val_scenes)} scenes")
    return _partial_val_scenes


def evaluate_tracking(
    result_path: str,
    output_dir: str,
    data_root: str = "/workspace/mmdetection3d/data/nuscenes",
    version: str = "v1.0-trainval",
    eval_set: str = "val",
) -> Dict:
    """Run TrackingEval and return the metrics summary.

    Returns dict with AMOTA, AMOTP, IDS, FRAG, MOTA, MOTP, etc.
    """
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval

    # patch BEFORE constructing TrackingEval
    partial_val_scenes = patch_partial_val(data_root, version)

    config = config_factory("tracking_nips_2019")

    os.makedirs(output_dir, exist_ok=True)

    nusc_eval = TrackingEval(
        config=config,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        nusc_version=version,
        nusc_dataroot=data_root,
        verbose=True,
    )
    metrics = nusc_eval.main(render_curves=False)

    # read the metrics_summary.json written by the evaluator
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    # extract the headline metrics
    result = {
        "result_path": result_path,
        "output_dir": output_dir,
        "n_scenes": len(partial_val_scenes),
        "amota": float(metrics.get("amota", float("nan"))),
        "amotp": float(metrics.get("amotp", float("nan"))),
        "motar": float(metrics.get("motar", float("nan"))),
        "mota": float(metrics.get("mota", float("nan"))),
        "motp": float(metrics.get("motp", float("nan"))),
        "ids": int(metrics.get("ids", -1)),
        "frag": int(metrics.get("frag", -1)),
        "recall": float(metrics.get("recall", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "hyp_count": int(metrics.get("hyp_count", -1)),
        "summary": summary,
    }

    print("\n" + "=" * 60)
    print(f"TRACKING EVAL: {os.path.basename(result_path)}")
    print("=" * 60)
    print(f"  scenes: {result['n_scenes']}")
    print(f"  AMOTA:  {result['amota']:.4f}")
    print(f"  AMOTP:  {result['amotp']:.4f}")
    print(f"  MOTA:   {result['mota']:.4f}")
    print(f"  MOTP:   {result['motp']:.4f}")
    print(f"  IDS:    {result['ids']}")
    print(f"  FRAG:   {result['frag']}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  Prec:   {result['precision']:.4f}")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage C tracking eval (partial_val)")
    parser.add_argument("--result-path", required=True, help="tracking results JSON")
    parser.add_argument("--output-dir", required=True, help="eval output dir")
    parser.add_argument("--data-root", default="/workspace/mmdetection3d/data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--metrics-out", default=None,
                        help="write metrics JSON here (default: <output-dir>/metrics_extracted.json)")
    args = parser.parse_args()

    result = evaluate_tracking(
        result_path=args.result_path,
        output_dir=args.output_dir,
        data_root=args.data_root,
        version=args.version,
        eval_set=args.eval_set,
    )

    metrics_out = args.metrics_out or os.path.join(args.output_dir, "metrics_extracted.json")
    with open(metrics_out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[eval_tracking] metrics -> {metrics_out}")


if __name__ == "__main__":
    main()