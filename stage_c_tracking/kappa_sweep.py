#!/usr/bin/env python
"""Sweep kappa over {1, 3, 5} on clean and motion blur and report pooled AMOTA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# reuse run_ablation_matrix machinery
sys.path.insert(0, os.path.dirname(__file__))
from run_ablation_matrix import run_tracker, eval_tracking, KAPPA  # noqa

SWEEP_CONDITIONS = ["clean_val_sev0", "motionblur_val_sev1", "motionblur_val_sev2", "motionblur_val_sev3"]
KAPPAS = [1.0, 3.0, 5.0]
RESULTS_DIR = "/workspace/results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappas", nargs="+", type=float, default=KAPPAS)
    parser.add_argument("--conditions", nargs="+", default=SWEEP_CONDITIONS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    failed = []
    sweep_results = []

    for kappa in args.kappas:
        for cond in args.conditions:
            try:
                metrics_path = os.path.join(
                    RESULTS_DIR, f"rowB_{cond}_kappa{int(kappa)}_metrics.json")
                if os.path.exists(metrics_path):
                    print(f"[skip] kappa={kappa} {cond} (metrics exist)")
                    with open(metrics_path) as f:
                        m = json.load(f)
                else:
                    tracking_json = run_tracker(cond, "B", kappa)
                    metrics_path = eval_tracking(tracking_json, "B", cond, kappa)
                    with open(metrics_path) as f:
                        m = json.load(f)
                # include mAVE/mAP/NDS from the velocity-grafted det eval
                det_path = os.path.join(
                    RESULTS_DIR, f"rowB_{cond}_kappa{int(kappa)}_det_metrics.json")
                det = json.load(open(det_path)) if os.path.exists(det_path) else {}
                sweep_results.append({
                    "kappa": kappa,
                    "condition": cond,
                    "amota": m["amota"],
                    "amotp": m["amotp"],
                    "mota": m["mota"],
                    "motp": m["motp"],
                    "ids": m["ids"],
                    "frag": m["frag"],
                    "mave": det.get("mave"),
                    "map": det.get("map"),
                    "nds": det.get("nds"),
                })
            except Exception as e:
                print(f"[ERROR] kappa={kappa} {cond}: {e}")
                failed.append(f"kappa{kappa}_{cond}")

    # write sweep table
    sweep_path = os.path.join(RESULTS_DIR, "kappa_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n=== kappa sweep -> {sweep_path} ===")
    print(f"{'kappa':>6} {'condition':<25} {'AMOTA':>8} {'AMOTP':>8} {'MOTA':>8} {'IDS':>5} {'FRAG':>5} {'mAVE':>7}")
    for r in sweep_results:
        mave = f"{r['mave']:.3f}" if r.get('mave') is not None else "—"
        print(f"{r['kappa']:>6} {r['condition']:<25} {r['amota']:>8.4f} {r['amotp']:>8.4f} "
              f"{r['mota']:>8.4f} {r['ids']:>5} {r['frag']:>5} {mave:>7}")

    # select kappa by pooled AMOTA
    by_kappa = {}
    for r in sweep_results:
        by_kappa.setdefault(r["kappa"], []).append(r["amota"])
    pooled = {k: sum(v) / len(v) for k, v in by_kappa.items()}
    best_kappa = max(pooled, key=pooled.get)
    print(f"\npooled AMOTA by kappa: {pooled}")
    print(f"selected kappa (frozen): {best_kappa}")
    with open(os.path.join(RESULTS_DIR, "kappa_frozen.json"), "w") as f:
        json.dump({"kappa": best_kappa, "pooled_amota": pooled}, f, indent=2)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()