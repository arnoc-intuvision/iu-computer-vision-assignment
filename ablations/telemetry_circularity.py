"""Telemetry-circularity ablation (Stage C §9, negative-result evidence).

Retrains the LCRE under several feature subsets and reports the decoy MAE
ratio (§5.4) for each. This is the reproducible evidence for the circularity
finding documented in stage_b_training_report.md:

    telemetry-only : decoy ratio ~8.23  (fully circular)
    full 1017-dim  : decoy ratio ~5.32  (circular via synth channels)
    drop_nolag     : decoy ratio ~1.25  (passes the 1.5 gate)
    bev_real       : decoy ratio ~0.93  (no synth telemetry -> anti-circular)

The bev_real variant is the one selected for Stage C. This script regenerates
the other three variants on demand so the comparison is reproducible from the
committed code, without keeping the stale checkpoints on disk.

Usage:
    python telemetry_circularity.py                 # all 4 variants
    python telemetry_circularity.py --variants bev_real drop_nolag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

# Reuse the committed Stage B training + boundary-check code.
sys.path.insert(0, "/workspace/scores")
from lcre_train import train  # noqa: E402
from run_boundary_checks import predict_heldout, check_5_4_decoy_discrimination  # noqa: E402
from lcre_dataset import ABLATION_SPECS, prepare_data  # noqa: E402


VARIANTS = ["telemetry", "full", "drop_nolag", "bev_real"]


def run_one(variant: str, epochs: int, clean_weight: float, device: str) -> Dict:
    """Train one variant and return its decoy-discrimination summary."""
    print(f"\n{'=' * 60}\n[telemetry_circularity] training variant: {variant}\n{'=' * 60}")
    # The train() function writes lcre_model_<ablation>[_<tag>].pt to /workspace/scores/.
    # For 'full' it writes lcre_model.pt (no suffix). We tag everything to avoid
    # clobbering the canonical bev_real checkpoint.
    tag = variant if variant != "bev_real" else "bev_real_circ"
    result = train(
        ablation=variant,
        epochs=epochs,
        batch_size=2048,
        clean_weight=clean_weight,
        tag=tag,
        device_str=device,
    )
    model = result["model"]
    model.eval()
    scaler = result["scaler"]
    heldout_data = result["heldout_data"]
    col_subset = result["config"]["col_subset"]

    pred = predict_heldout(model, heldout_data, scaler, col_subset, device)
    decoy = check_5_4_decoy_discrimination(pred, threshold=1.5)

    summary = {
        "variant": variant,
        "input_dim": result["config"]["input_dim"],
        "best_held_mse": result["best_held_mse"],
        "decoy_ratio": decoy["ratio"],
        "decoy_mae": decoy["mae_decoy"],
        "non_decoy_mae": decoy["mae_non_decoy"],
        "passed_gate": decoy["passed"],
        "model_path": result["model_path"],
    }
    print(f"[telemetry_circularity] {variant}: decoy_ratio={summary['decoy_ratio']:.4f} "
          f"passed={summary['passed_gate']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Telemetry-circularity ablation")
    parser.add_argument("--variants", nargs="+", default=VARIANTS,
                        choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--clean-weight", type=float, default=3.0,
                        help="Clean-frame loss weight (3.0 matches the bev_real recipe)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="/workspace/results/telemetry_circularity.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summaries: List[Dict] = []
    for v in args.variants:
        summaries.append(run_one(v, args.epochs, args.clean_weight, args.device))

    print("\n" + "=" * 60)
    print("TELEMETRY CIRCULARITY SUMMARY")
    print("=" * 60)
    print(f"{'variant':<14} {'dim':>5} {'held_mse':>10} {'decoy_ratio':>12} {'gate':>6}")
    for s in summaries:
        print(f"{s['variant']:<14} {s['input_dim']:>5} {s['best_held_mse']:>10.5f} "
              f"{s['decoy_ratio']:>12.4f} {'PASS' if s['passed_gate'] else 'FAIL':>6}")
    print("=" * 60)

    with open(args.out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[telemetry_circularity] -> {args.out}")


if __name__ == "__main__":
    main()
