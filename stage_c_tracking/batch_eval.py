#!/usr/bin/env python
"""Batched Stage C evaluation — load nuScenes tables ONCE, eval many.

The prior per-condition parallel eval runs died from NFS thrash when ~30
processes each loaded the full v1.0-trainval tables at once. This driver loads
the tables a single time per mode and loops over conditions in-process, so the
expensive table parse happens once.

Modes:
  track  Evaluate tracking_results/<prefix>_<cond>[_kappaK].json with the
         nuScenes tracking eval (tracking_nips_2019) + partial_val patch.
         Writes results/<prefix>_<cond>[_kappaK]_metrics.json.
  det    Evaluate detection-format JSONs (Phase 6.3 velocity-grafted files in
         /workspace/detections_vel/) with the nuScenes detection eval
         (detection_cvpr_2019) + partial_val patch. Writes
         results/<name>_det_metrics.json.

Idempotent: skips any condition whose metrics JSON already exists.

Usage:
    python batch_eval.py track            # all missing tracking metrics
    python batch_eval.py track --only rowB --kappa 3
    python batch_eval.py det              # all missing det (velocity) metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

RESULTS_DIR = "/workspace/results"
TRACKING_DIR = "/workspace/tracking_results"
DET_VEL_DIR = "/workspace/detections_vel"
DATA_ROOT = "/workspace/mmdetection3d/data/nuscenes"
VERSION = "v1.0-trainval"

CONDITIONS = [
    "clean_val_sev0",
    "beamsreducing_val_sev1", "beamsreducing_val_sev2", "beamsreducing_val_sev3",
    "missingcamera_val_sev1", "missingcamera_val_sev2", "missingcamera_val_sev3",
    "motionblur_val_sev1", "motionblur_val_sev2", "motionblur_val_sev3",
]


def _install_nuscenes_cache():
    """Monkeypatch nuscenes.NuScenes to a per-(version,dataroot) singleton.

    TrackingEval constructs its own NuScenes(...) internally; this makes the
    2nd+ construction in the same process return the already-loaded DB (the DB
    is read-only during eval, so sharing is safe).
    """
    import nuscenes

    _real = nuscenes.NuScenes
    _cache = {}

    def _cached(version, dataroot=DATA_ROOT, *args, **kwargs):
        key = (version, dataroot)
        if key not in _cache:
            _cache[key] = _real(version=version, dataroot=dataroot, *args, **kwargs)
        else:
            print(f"[nusc-cache] reuse loaded DB {version} @ {dataroot}")
        return _cache[key]

    nuscenes.NuScenes = _cached
    # some callers import NuScenes via from nuscenes import NuScenes already
    import nuscenes.eval.tracking.evaluate as te
    import nuscenes.eval.detection.evaluate as de
    te.NuScenes = _cached
    de.NuScenes = _cached


def patch_partial_val(data_root=DATA_ROOT, version=VERSION):
    import nuscenes.eval.common.loaders as L
    splits_path = os.path.join(data_root, version, "splits.json")
    with open(splits_path) as f:
        partial = json.load(f)["partial_val"]
    _orig = L.create_splits_scenes

    def _patched(verbose=False):
        s = _orig(verbose=verbose)
        s["val"] = partial
        return s

    L.create_splits_scenes = _patched
    return partial


# --------------------------------------------------------------------------- track
def eval_one_track(tracking_json, output_dir, metrics_out):
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval

    partial = patch_partial_val()
    config = config_factory("tracking_nips_2019")
    os.makedirs(output_dir, exist_ok=True)
    nusc_eval = TrackingEval(
        config=config, result_path=tracking_json, eval_set="val",
        output_dir=output_dir, nusc_version=VERSION, nusc_dataroot=DATA_ROOT,
        verbose=False,
    )
    metrics = nusc_eval.main(render_curves=False)
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    def _g(k, cast=float):
        v = metrics.get(k)
        try:
            return cast(v)
        except (TypeError, ValueError):
            return v

    result = {
        "result_path": tracking_json, "output_dir": output_dir,
        "n_scenes": len(partial),
        "amota": _g("amota"), "amotp": _g("amotp"), "motar": _g("motar"),
        "mota": _g("mota"), "motp": _g("motp"),
        "ids": _g("ids", int), "frag": _g("frag", int),
        "recall": _g("recall"), "precision": _g("precision"),
        "hyp_count": _g("hyp_count", int), "summary": summary,
    }
    with open(metrics_out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [track] {os.path.basename(tracking_json):<45} "
          f"AMOTA={result['amota']:.4f} AMOTP={result['amotp']:.4f} "
          f"MOTA={result['mota']:.4f} IDS={result['ids']} FRAG={result['frag']}")


def run_track_mode(only_prefix=None, kappa=None):
    """Discover tracking JSONs missing metrics and eval them."""
    _install_nuscenes_cache()
    # candidate tracking jsons: rowA_<cond>.json, rowB_<cond>_kappaK.json
    tasks = []
    patterns = []
    if only_prefix:
        patterns.append(os.path.join(TRACKING_DIR, f"{only_prefix}_*.json"))
    else:
        patterns.append(os.path.join(TRACKING_DIR, "rowA_*.json"))
        patterns.append(os.path.join(TRACKING_DIR, "rowB_*.json"))
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            base = os.path.basename(path)
            if base in seen:
                continue
            seen.add(base)
            # derive metrics name: rowA_<cond>_metrics.json or rowB_<cond>_kappaK_metrics.json
            mname = base[:-5] + "_metrics.json"  # strip .json
            if kappa is not None and only_prefix == "rowB":
                # only this kappa
                if f"_kappa{int(kappa)}" not in mname:
                    continue
            mout = os.path.join(RESULTS_DIR, mname)
            if os.path.exists(mout):
                print(f"  [skip] {base} (metrics exist)")
                continue
            odir = os.path.join(RESULTS_DIR, mname[:-len("_metrics.json")] + "_eval")
            tasks.append((path, odir, mout))

    if not tasks:
        print("[track] nothing to do.")
        return
    print(f"[track] {len(tasks)} tracking evals (tables load ONCE)...")
    t0 = time.time()
    for i, (path, odir, mout) in enumerate(tasks, 1):
        print(f"[track] ({i}/{len(tasks)}) {os.path.basename(path)}")
        eval_one_track(path, odir, mout)
    print(f"[track] done in {time.time()-t0:.1f}s")


# ----------------------------------------------------------------------------- det
def eval_one_det(nusc, result_json, output_dir, metrics_out):
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    patch_partial_val()
    config = config_factory("detection_cvpr_2019")
    os.makedirs(output_dir, exist_ok=True)
    nusc_eval = DetectionEval(
        nusc=nusc, config=config, result_path=result_json, eval_set="val",
        output_dir=output_dir, verbose=False,
    )
    metrics = nusc_eval.main(render_curves=False)
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    # nuScenes DetectionEval.main() returns mean_ap/nd_score, and TP errors live
    # in metrics_summary['tp_errors'] keyed by error type (trans/scale/orient/
    # vel/attr), already averaged over the classes that support each error.
    te = summary.get("tp_errors", {}) or {}

    def _f(x):
        return float(x) if isinstance(x, (int, float)) else float("nan")

    result = {
        "result_path": result_json, "output_dir": output_dir,
        "map": _f(summary.get("mean_ap")),
        "nds": _f(summary.get("nd_score")),
        "mate": _f(te.get("trans_err")),
        "mase": _f(te.get("scale_err")),
        "maoe": _f(te.get("orient_err")),
        "mave": _f(te.get("vel_err")),
        "maae": _f(te.get("attr_err")),
        "summary": summary,
    }
    with open(metrics_out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [det] {os.path.basename(os.path.dirname(result_json)):<35} "
          f"mAP={result['map']:.4f} NDS={result['nds']:.4f} mAVE={result['mave']:.4f}")


def run_det_mode():
    """Eval all velocity-grafted detection JSONs in /workspace/detections_vel/."""
    from nuscenes import NuScenes
    partial = patch_partial_val()
    print(f"[det] loading NuScenes once ({VERSION})...")
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)

    tasks = []
    for result_json in sorted(glob.glob(os.path.join(DET_VEL_DIR, "*", "results_nusc.json"))):
        name = os.path.basename(os.path.dirname(result_json))  # e.g. rowA_clean_val_sev0
        mout = os.path.join(RESULTS_DIR, f"{name}_det_metrics.json")
        if os.path.exists(mout):
            print(f"  [skip] {name} (metrics exist)")
            continue
        odir = os.path.join(RESULTS_DIR, f"{name}_det_eval")
        tasks.append((result_json, odir, mout, name))

    if not tasks:
        print("[det] nothing to do.")
        return
    print(f"[det] {len(tasks)} detection evals (tables loaded)...")
    t0 = time.time()
    for i, (rj, odir, mout, name) in enumerate(tasks, 1):
        print(f"[det] ({i}/{len(tasks)}) {name}")
        eval_one_det(nusc, rj, odir, mout)
    print(f"[det] done in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["track", "det"])
    ap.add_argument("--only", default=None, help="track mode: prefix filter (rowA/rowB)")
    ap.add_argument("--kappa", type=float, default=None)
    args = ap.parse_args()
    if args.mode == "track":
        run_track_mode(only_prefix=args.only, kappa=args.kappa)
    else:
        run_det_mode()


if __name__ == "__main__":
    main()
