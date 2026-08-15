#!/usr/bin/env python
"""Assemble the Stage C metric files into the markdown result tables."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

RESULTS_DIR = "/workspace/results"
SCORES_DIR = "/workspace/scores_bev_real"

CONDITIONS = [
    "clean_val_sev0",
    "beamsreducing_val_sev1", "beamsreducing_val_sev2", "beamsreducing_val_sev3",
    "missingcamera_val_sev1", "missingcamera_val_sev2", "missingcamera_val_sev3",
    "motionblur_val_sev1", "motionblur_val_sev2", "motionblur_val_sev3",
]

# corruption -> [sev label] for ordering
CORRUPTION_SEVS = {
    "clean": ["clean_val_sev0"],
    "beamsreducing": ["beamsreducing_val_sev1", "beamsreducing_val_sev2", "beamsreducing_val_sev3"],
    "missingcamera": ["missingcamera_val_sev1", "missingcamera_val_sev2", "missingcamera_val_sev3"],
    "motionblur": ["motionblur_val_sev1", "motionblur_val_sev2", "motionblur_val_sev3"],
}

PRETTY = {
    "clean_val_sev0": "clean",
    "beamsreducing_val_sev1": "beamsreducing sev1",
    "beamsreducing_val_sev2": "beamsreducing sev2",
    "beamsreducing_val_sev3": "beamsreducing sev3",
    "missingcamera_val_sev1": "missingcamera sev1",
    "missingcamera_val_sev2": "missingcamera sev2",
    "missingcamera_val_sev3": "missingcamera sev3",
    "motionblur_val_sev1": "motionblur sev1",
    "motionblur_val_sev2": "motionblur sev2",
    "motionblur_val_sev3": "motionblur sev3",
}


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def row_a_path(cond):
    return os.path.join(RESULTS_DIR, f"rowA_{cond}_metrics.json")


def row_b_path(cond, kappa=3):
    return os.path.join(RESULTS_DIR, f"rowB_{cond}_kappa{kappa}_metrics.json")


def det_metrics_path(row, cond, kappa=3):
    """velocity-grafted detection metrics (mAP/NDS/mAVE)."""
    row = row.upper()
    suffix = f"_kappa{kappa}" if row == "B" else ""
    return os.path.join(RESULTS_DIR, f"row{row}_{cond}{suffix}_det_metrics.json")


def load_det(row, cond, kappa=3):
    return load_metrics(det_metrics_path(row, cond, kappa))


def build_ablation_matrix(kappa=3):
    lines = []
    lines.append("# Stage C — Ablation Matrix (Row A vs Row B)\n")
    lines.append(f"Row A = fixed-R baseline. Row B = reliability-adaptive R(t), kappa={kappa}.\n")
    lines.append("Metrics from nuScenes `tracking_nips_2019` eval on the 34-scene partial val split.\n")
    lines.append("`d` = Row B - Row A (positive = Row B better for AMOTA/MOTA/recall; "
                 "negative = better for AMOTP/MOTP/IDS/FRAG where lower is better).\n")

    a_rows = {c: load_metrics(row_a_path(c)) for c in CONDITIONS}
    b_rows = {c: load_metrics(row_b_path(c, kappa)) for c in CONDITIONS}

    metrics = ["amota", "amotp", "mota", "motp", "recall", "ids", "frag"]
    metric_names = {
        "amota": "AMOTA (up)",
        "amotp": "AMOTP (down)",
        "mota": "MOTA (up)",
        "motp": "MOTP (down)",
        "recall": "Recall (up)",
        "ids": "IDS (down)",
        "frag": "FRAG (down)",
    }

    # one table per metric
    for m in metrics:
        lines.append(f"\n## {metric_names[m]}\n")
        lines.append("| condition | Row A | Row B | d (B-A) |")
        lines.append("|---|---|---|---|")
        for c in CONDITIONS:
            a = a_rows[c]
            b = b_rows[c]
            av = a[m] if a else None
            bv = b[m] if b else None
            if av is not None and bv is not None:
                d = bv - av
                ds = fmt(d)
            else:
                ds = "—"
            nd = 0 if m in ("ids", "frag") else 4
            lines.append(f"| {PRETTY[c]} | {fmt(av, nd)} | {fmt(bv, nd)} | {ds} |")

    # summary delta table (pooled over corruption sevs)
    lines.append("\n## Summary — mean delta over corrupted conditions (Row B - Row A)\n")
    lines.append("| corruption | AMOTA d | AMOTP d | MOTA d | MOTP d | IDS d | FRAG d |")
    lines.append("|---|---|---|---|---|---|---|")
    for corr, conds in CORRUPTION_SEVS.items():
        if corr == "clean":
            continue
        ds = {m: [] for m in metrics}
        for c in conds:
            a = a_rows[c]
            b = b_rows[c]
            if a and b:
                for m in metrics:
                    ds[m].append(b[m] - a[m])
        cells = []
        for m in metrics:
            if ds[m]:
                cells.append(fmt(np.mean(ds[m]), 0 if m in ("ids", "frag") else 4))
            else:
                cells.append("—")
        lines.append(f"| {corr} | " + " | ".join(cells) + " |")

    # Detection metrics (velocity-smoothed mAP/NDS/mAVE)
    det_rows_a = {c: load_det("A", c) for c in CONDITIONS}
    det_rows_b = {c: load_det("B", c, kappa) for c in CONDITIONS}
    det_metrics = ["map", "nds", "mave", "mate"]
    det_names = {"map": "mAP (up)", "nds": "NDS (up)",
                 "mave": "mAVE (down)", "mate": "mATE (down)"}
    lines.append("\n## Detection metrics (velocity-smoothed, )\n")
    lines.append("nuScenes detection eval on JSONs with tracker-smoothed velocities. "
                 "mAP is identical across rows (box set unchanged); only mAVE (and thus "
                 "NDS) can differ.\n")
    # Detector-velocity baseline reference + caveat (frame-verified degradation).
    det_base = load_metrics(os.path.join(RESULTS_DIR, "baseline_detector_clean_det_metrics.json"))
    clean_trk = det_rows_a.get("clean_val_sev0")
    if det_base and clean_trk:
        ratio = (f"{clean_trk['mave']/det_base['mave']:.1f}x"
                 if det_base.get("mave") and clean_trk.get("mave") else "—")
        lines.append(
            f">  **Velocity caveat:** every mAVE/NDS value below uses the *CV "
            f"tracker's* velocity, NOT the detector's. The detector's own clean "
            f"velocity (Stage A baseline, `baseline_detector_clean_det_metrics.json`) "
            f"is **mAVE={det_base['mave']:.3f}, NDS={det_base['nds']:.4f}** (matches "
            f": 0.309 / 0.7154). Grafting the tracker velocity degrades clean mAVE "
            f"to {clean_trk['mave']:.3f} (**{ratio} worse**) and NDS to "
            f"{clean_trk['nds']:.4f}. The write-back was verified frame-correct "
            f"(global frame, m/s; median angle(det_v, trk_v) = 1.0°, std 82°) — the "
            f"loss is genuine CV-filter noise (spurious motion on ~40k static "
            f"objects). Row A vs Row B remain internally valid (both use tracker "
            f"velocity); there is no velocity-improvement win over the detector. "
            f"See `README.md` for the full caveat.\n")
    for m in det_metrics:
        lines.append(f"\n### {det_names[m]}\n")
        lines.append("| condition | Row A | Row B | d (B-A) |")
        lines.append("|---|---|---|---|")
        for c in CONDITIONS:
            a = det_rows_a[c]; b = det_rows_b[c]
            av = a[m] if a else None; bv = b[m] if b else None
            ds = fmt(bv - av) if (av is not None and bv is not None
                                  and not (isinstance(av, float) and np.isnan(av))
                                  and not (isinstance(bv, float) and np.isnan(bv))) else "—"
            lines.append(f"| {PRETTY[c]} | {fmt(av)} | {fmt(bv)} | {ds} |")

    out = "\n".join(lines) + "\n"
    path = os.path.join(RESULTS_DIR, "ablation_matrix.md")
    with open(path, "w") as f:
        f.write(out)
    print(f"[assemble] -> {path}")

    # also dump a machine-readable json
    matrix = {}
    for c in CONDITIONS:
        a = a_rows[c]; b = b_rows[c]
        da = det_rows_a[c]; db = det_rows_b[c]
        matrix[c] = {
            "rowA": {m: (a[m] if a else None) for m in metrics},
            "rowB": {m: (b[m] if b else None) for m in metrics},
            "det_rowA": {m: (da[m] if da else None) for m in det_metrics},
            "det_rowB": {m: (db[m] if db else None) for m in det_metrics},
        }
    with open(os.path.join(RESULTS_DIR, "ablation_matrix.json"), "w") as f:
        json.dump(matrix, f, indent=2)

    missing = [c for c in CONDITIONS if a_rows[c] is None or b_rows[c] is None
               or det_rows_a[c] is None or det_rows_b[c] is None]
    return missing


def build_r_inflation(kappa=3):
    lines = []
    lines.append("# Stage C — R(t)/R_base Inflation per Condition\n")
    lines.append(f"R(t)/R_base = 1 + kappa*(1 - S_t), kappa={kappa}. "
                 "S_t statistics from the bev_real score files.\n")
    lines.append("| condition | S_t mean | S_t std | S_t min | S_t max | R(t)/R_base mean | R(t)/R_base max |")
    lines.append("|---|---|---|---|---|---|---|")
    summary = []
    for c in CONDITIONS:
        sf = os.path.join(SCORES_DIR, f"{c}_scores.npz")
        if not os.path.exists(sf):
            lines.append(f"| {PRETTY[c]} | — | — | — | — | — | — |")
            continue
        d = np.load(sf, allow_pickle=True)
        st = d["s_t"].astype(float)
        rmean = 1 + kappa * (1 - st.mean())
        rmax = 1 + kappa * (1 - st.min())
        lines.append(f"| {PRETTY[c]} | {st.mean():.4f} | {st.std():.4f} | "
                     f"{st.min():.4f} | {st.max():.4f} | {rmean:.3f} | {rmax:.3f} |")
        summary.append({"condition": c, "s_t_mean": float(st.mean()),
                        "s_t_std": float(st.std()), "r_ratio_mean": float(rmean)})
    out = "\n".join(lines) + "\n"
    path = os.path.join(RESULTS_DIR, "r_inflation_table.md")
    with open(path, "w") as f:
        f.write(out)
    print(f"[assemble] -> {path}")
    with open(os.path.join(RESULTS_DIR, "r_inflation_table.json"), "w") as f:
        json.dump(summary, f, indent=2)


def build_kappa_sweep():
    sweep_path = os.path.join(RESULTS_DIR, "kappa_sweep.json")
    if not os.path.exists(sweep_path):
        print("[assemble] no kappa_sweep.json yet, skipping kappa_sweep.md")
        return
    with open(sweep_path) as f:
        sweep = json.load(f)
    frozen_path = os.path.join(RESULTS_DIR, "kappa_frozen.json")
    frozen = json.load(open(frozen_path)) if os.path.exists(frozen_path) else {}

    lines = []
    lines.append("# Stage C — kappa Sweep ()\n")
    lines.append("Sweep over kappa on clean + motionblur. kappa frozen by pooled AMOTA.\n")
    lines.append("| kappa | condition | AMOTA | AMOTP | MOTA | MOTP | IDS | FRAG | mAVE |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sweep:
        mave = fmt(r.get("mave"), 3) if r.get("mave") is not None else "—"
        lines.append(f"| {r['kappa']} | {PRETTY.get(r['condition'], r['condition'])} | "
                     f"{fmt(r['amota'])} | {fmt(r['amotp'])} | {fmt(r['mota'])} | "
                     f"{fmt(r['motp'])} | {r['ids']} | {r['frag']} | {mave} |")
    if frozen:
        pooled = frozen.get("pooled_amota", {})
        lines.append("\n**Pooled AMOTA by kappa:** " +
                     ", ".join(f"k={k} -> {v:.4f}" for k, v in sorted(pooled.items())))
        lines.append(f"\n**Frozen kappa = {frozen.get('kappa')}** "
                     f"(highest pooled AMOTA).\n")
    out = "\n".join(lines) + "\n"
    path = os.path.join(RESULTS_DIR, "kappa_sweep.md")
    with open(path, "w") as f:
        f.write(out)
    print(f"[assemble] -> {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    missing = build_ablation_matrix(kappa=3)
    build_r_inflation(kappa=3)
    build_kappa_sweep()
    if missing:
        print(f"[assemble] WARNING: missing metrics for: {missing}")
    else:
        print("[assemble] all 20 cells present.")


if __name__ == "__main__":
    main()
