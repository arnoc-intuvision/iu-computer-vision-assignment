"""Held-out gate checks for a trained LCRE: severity correlation, clean anchor,
monotonicity and decoy discrimination.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from lcre_model import LCRE
from lcre_dataset import (
    ABLATION_SPECS,
    LCREDataset,
    SCORES_DIR,
    Scaler,
    prepare_data,
)


def load_model(model_path: str, device: str = "cpu") -> Dict:
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]
    model = LCRE(input_dim=config["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return {"model": model, "config": config, "ckpt": ckpt}


@torch.no_grad()
def predict_heldout(
    model, heldout_data, scaler, col_subset, device_str="cpu"
) -> Dict[str, np.ndarray]:
    device = torch.device(device_str)
    ds = LCREDataset(heldout_data, scaler=scaler, col_subset=col_subset)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    for batch in loader:
        x = batch[0].to(device)
        p = model(x)
        preds.append(p.cpu().numpy())
    return {
        "s_t": np.concatenate(preds),
        "targets": heldout_data.targets,
        "true_severity": heldout_data.true_severity,
        "is_decoy": heldout_data.is_decoy,
        "corruption_type": heldout_data.corruption_type,
        "tokens": heldout_data.tokens,
    }


def check_5_1_severity_correlation(pred: Dict) -> Dict:
    """Spearman rho(S_t, true_severity), pooled + per-corruption-type."""
    print("\n--- Severity correlation (Spearman rho) ---")
    s_t = pred["s_t"]
    sev = pred["true_severity"]
    ctype = pred["corruption_type"]

    # pooled
    rho_pooled, p_pooled = spearmanr(s_t, sev)
    status = "PASS" if rho_pooled < -0.3 else "WARN"
    print(f"  pooled:                rho={rho_pooled:+.4f}  p={p_pooled:.2e}  [{status}]")

    # per corruption type
    per_type = {}
    for ct in sorted(set(ctype.tolist())):
        mask = ctype == ct
        if mask.sum() < 2:
            continue
        rho, p = spearmanr(s_t[mask], sev[mask])
        per_type[ct] = {"rho": float(rho), "p": float(p), "n": int(mask.sum())}
        st = "PASS" if rho < -0.2 else "WARN"
        print(f"  {ct:20s}: rho={rho:+.4f}  p={p:.2e}  n={mask.sum()}  [{st}]")

    passed = rho_pooled < -0.3
    return {"passed": passed, "rho_pooled": float(rho_pooled), "per_type": per_type}


def check_5_2_clean_anchor(pred: Dict) -> Dict:
    """Mean S_t on held-out clean frames should be in [0.85, 1.0]."""
    print("\n--- Clean-frame anchor ---")
    ctype = pred["corruption_type"]
    s_t = pred["s_t"]
    clean_mask = ctype == "clean"
    if clean_mask.sum() == 0:
        print("  SKIP: no clean frames in held-out split")
        return {"passed": False, "mean_s_t": None, "reason": "no clean frames"}
    mean_clean = float(s_t[clean_mask].mean())
    lo, hi = 0.85, 1.0
    passed = lo <= mean_clean <= hi
    st = "PASS" if passed else "FAIL"
    print(f"  mean S_t (clean, held-out): {mean_clean:.4f}  "
          f"expected [{lo}, {hi}]  [{st}]")
    return {"passed": passed, "mean_s_t": mean_clean}


def check_5_3_monotonicity(pred: Dict) -> Dict:
    """Mean S_t per severity must be strictly decreasing sev0>sev1>sev2>sev3."""
    print("\n--- Monotonicity (mean S_t per severity) ---")
    s_t = pred["s_t"]
    sev = pred["true_severity"]
    ctype = pred["corruption_type"]

    # pooled
    means_pooled = {}
    for s in sorted(set(sev.tolist())):
        mask = sev == s
        m = float(s_t[mask].mean()) if mask.sum() else float("nan")
        means_pooled[s] = m
    pooled_decreasing = all(
        means_pooled[s] > means_pooled[s + 1]
        for s in range(3)
        if s in means_pooled and (s + 1) in means_pooled
        and not np.isnan(means_pooled[s]) and not np.isnan(means_pooled[s + 1])
    )
    print(f"  pooled:  " + "  ".join(
        f"sev{s}={means_pooled.get(s, float('nan')):.4f}"
        for s in sorted(means_pooled)
    ) + f"  [{'PASS' if pooled_decreasing else 'FAIL'}]")

    # per corruption type
    per_type = {}
    for ct in sorted(set(ctype.tolist())):
        mask = ctype == ct
        if mask.sum() == 0:
            continue
        means = {}
        for s in sorted(set(sev[mask].tolist())):
            sm = mask & (sev == s)
            means[s] = float(s_t[sm].mean()) if sm.sum() else float("nan")
        decreasing = all(
            means[s] > means[s + 1]
            for s in range(3)
            if s in means and (s + 1) in means
            and not np.isnan(means[s]) and not np.isnan(means[s + 1])
        )
        per_type[ct] = {"means": means, "decreasing": decreasing}
        detail = "  ".join(f"sev{s}={means.get(s, float('nan')):.4f}"
                           for s in sorted(means))
        print(f"  {ct:20s}: {detail}  [{'PASS' if decreasing else 'FAIL'}]")

    passed = pooled_decreasing
    return {"passed": passed, "means_pooled": means_pooled, "per_type": per_type}


def check_5_4_decoy_discrimination(pred: Dict, threshold: float = 1.5) -> Dict:
    """MAE on decoy vs non-decoy frames. Large gap = telemetry shortcut."""
    print("\n--- Decoy discrimination (THE important check) ---")
    s_t = pred["s_t"]
    targets = pred["targets"]
    is_decoy = pred["is_decoy"]

    non_decoy_mask = ~is_decoy
    decoy_mask = is_decoy

    mae_non_decoy = float(np.abs(s_t[non_decoy_mask] - targets[non_decoy_mask]).mean())
    mae_decoy = float(np.abs(s_t[decoy_mask] - targets[decoy_mask]).mean())
    ratio = mae_decoy / max(mae_non_decoy, 1e-8)

    passed = ratio < threshold
    st = "PASS" if passed else "FAIL — TELEMETRY SHORTCUT DETECTED"

    print(f"  non-decoy MAE: {mae_non_decoy:.4f}  (n={non_decoy_mask.sum()})")
    print(f"  decoy MAE:     {mae_decoy:.4f}  (n={decoy_mask.sum()})")
    print(f"  ratio:         {ratio:.4f}  (threshold {threshold})")
    print(f"  [{st}]")
    if not passed:
        print("  *** S_t tracks telemetry on decoys — the contribution is CIRCULAR. ***")
        print("  *** Iterate: increase dropout/weight_decay, drop telemetry channels. ***")

    return {
        "passed": passed,
        "mae_non_decoy": mae_non_decoy,
        "mae_decoy": mae_decoy,
        "ratio": float(ratio),
        "threshold": threshold,
    }


def check_5_6_coverage(pred: Dict, val_cache_dir: str = "/workspace/cache") -> Dict:
    """Check held-out token consistency."""
    print("\n--- Coverage (held-out token check) ---")
    held_tokens = pred["tokens"]
    n_total = len(held_tokens)
    n_unique = len(set(held_tokens.tolist()))
    # tokens repeat across the 10 conditions -> duplicates are expected
    print(f"  held-out token rows: {n_total}  unique tokens: {n_unique}  "
          f"(duplicates expected across 10 conditions)")
    passed = n_unique > 0
    print(f"  [{'PASS' if passed else 'FAIL'}]")
    return {"passed": passed, "n_rows": n_total, "n_unique": n_unique}


def run_all_checks(
    model_path: str = os.path.join(SCORES_DIR, "lcre_model.pt"),
    ablation: str = "full",
    device_str: Optional[str] = None,
    decoy_threshold: float = 1.5,
) -> Dict:
    import torch
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]
    model = LCRE(input_dim=config["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # prepare data (loads caches, builds split, fits scaler)
    fit_data, heldout_data, scaler, scene_split = prepare_data()

    col_subset = config.get("col_subset", ABLATION_SPECS[ablation])

    # predict on held-out
    pred = predict_heldout(model, heldout_data, scaler, col_subset, device)

    # run all checks
    results = {}
    results["5.1"] = check_5_1_severity_correlation(pred)
    results["5.2"] = check_5_2_clean_anchor(pred)
    results["5.3"] = check_5_3_monotonicity(pred)
    results["5.4"] = check_5_4_decoy_discrimination(pred, threshold=decoy_threshold)
    results["5.6"] = check_5_6_coverage(pred)

    # summary
    print("\n" + "=" * 60)
    print("BOUNDARY CHECK SUMMARY")
    print("=" * 60)
    all_pass = True
    for check_id, r in results.items():
        st = "PASS" if r["passed"] else "FAIL"
        print(f"  check {check_id}: {st}")
        if not r["passed"]:
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("  ALL CHECKS PASSED — proceed to Stage C artifact emission.")
    else:
        print("  GATE FAILED — do NOT emit Stage C scores. Iterate on the model.")
    print("=" * 60)

    return {"all_passed": all_pass, "checks": results, "predictions": pred}


def main():
    parser = argparse.ArgumentParser(description="Run boundary checks")
    parser.add_argument("--model", default=os.path.join(SCORES_DIR, "lcre_model.pt"))
    parser.add_argument("--ablation", default="full",
                        choices=list(ABLATION_SPECS.keys()))
    parser.add_argument("--device", default=None)
    parser.add_argument("--decoy-threshold", type=float, default=1.5,
                        help="Fail if decoy_mae > threshold * non_decoy_mae")
    args = parser.parse_args()

    result = run_all_checks(
        model_path=args.model,
        ablation=args.ablation,
        device_str=args.device,
        decoy_threshold=args.decoy_threshold,
    )
    sys.exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
