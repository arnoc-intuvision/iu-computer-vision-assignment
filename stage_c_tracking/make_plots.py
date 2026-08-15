#!/usr/bin/env python
"""Generate the Stage C result plots from the metric files and score files."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/workspace/results"
SCORES_DIR = "/workspace/scores_bev_real"

CORRUPTIONS = {
    "beamsreducing": [1, 2, 3],
    "missingcamera": [1, 2, 3],
    "motionblur": [1, 2, 3],
}
COLORS = {"beamsreducing": "C0", "missingcamera": "C1", "motionblur": "C2"}
CLEAN_KAPPA = 3


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def row_a(cond):
    return load_metrics(os.path.join(RESULTS_DIR, f"rowA_{cond}_metrics.json"))


def row_b(cond, kappa=CLEAN_KAPPA):
    return load_metrics(os.path.join(RESULTS_DIR, f"rowB_{cond}_kappa{kappa}_metrics.json"))


def det_a(cond):
    return load_metrics(os.path.join(RESULTS_DIR, f"rowA_{cond}_det_metrics.json"))


def det_b(cond, kappa=CLEAN_KAPPA):
    return load_metrics(os.path.join(RESULTS_DIR, f"rowB_{cond}_kappa{kappa}_det_metrics.json"))


def plot_metric_vs_severity(metric, ylabel, fname, lower_better=False, kappa=3):
    fig, ax = plt.subplots(figsize=(7, 5))
    # clean baseline points (sev 0) for reference
    clean_a = row_a("clean_val_sev0")
    clean_b = row_b("clean_val_sev0", kappa)
    for corr, sevs in CORRUPTIONS.items():
        xa, ya, xb, yb = [], [], [], []
        for s in sevs:
            cond = f"{corr}_val_sev{s}"
            a = row_a(cond)
            b = row_b(cond, kappa)
            if a:
                xa.append(s); ya.append(a[metric])
            if b:
                xb.append(s); yb.append(b[metric])
        col = COLORS[corr]
        if xa:
            ax.plot(xa, ya, "o--", color=col, label=f"{corr} Row A", alpha=0.7)
        if xb:
            ax.plot(xb, yb, "s-", color=col, label=f"{corr} Row B (k={kappa})")
    if clean_a:
        ax.axhline(clean_a[metric], color="gray", ls=":", alpha=0.6,
                   label=f"clean Row A ({clean_a[metric]:.3f})")
    if clean_b:
        ax.axhline(clean_b[metric], color="gray", ls="-.", alpha=0.4,
                   label=f"clean Row B ({clean_b[metric]:.3f})")
    ax.set_xlabel("corruption severity")
    ax.set_ylabel(ylabel)
    better = "lower is better" if lower_better else "higher is better"
    ax.set_title(f"{ylabel} vs severity  ({better})")
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plots] -> {path}")


def plot_st_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = []
    data = []
    labels = []
    pos = 1
    # clean
    sf = os.path.join(SCORES_DIR, "clean_val_sev0_scores.npz")
    if os.path.exists(sf):
        st = np.load(sf, allow_pickle=True)["s_t"].astype(float)
        ax.boxplot([st], positions=[pos], widths=0.6, showfliers=False)
        labels.append("clean\nsev0")
        pos += 1
    pos += 0.5  # gap
    for corr, sevs in CORRUPTIONS.items():
        for s in sevs:
            sf = os.path.join(SCORES_DIR, f"{corr}_val_sev{s}_scores.npz")
            if os.path.exists(sf):
                st = np.load(sf, allow_pickle=True)["s_t"].astype(float)
                ax.boxplot([st], positions=[pos], widths=0.6, showfliers=False,
                           patch_artist=True,
                           boxprops=dict(facecolor=COLORS[corr], alpha=0.4))
                labels.append(f"{corr[:5]}\nsev{s}")
                pos += 1
        pos += 0.5
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("S_t (bev_real)")
    ax.set_title("Reliability score S_t distribution per condition (Stage B -> Stage C)")
    ax.axhline(1.0, color="green", ls=":", alpha=0.5, label="S_t=1 (identity, R(t)=R_base)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_st_vs_severity.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plots] -> {path}")


def plot_mave_vs_severity(kappa=3):
    """mAVE (velocity error) Row A vs Row B vs severity."""
    fig, ax = plt.subplots(figsize=(7, 5))
    clean_a = det_a("clean_val_sev0"); clean_b = det_b("clean_val_sev0", kappa)
    for corr, sevs in CORRUPTIONS.items():
        xa, ya, xb, yb = [], [], [], []
        for s in sevs:
            cond = f"{corr}_val_sev{s}"
            a = det_a(cond); b = det_b(cond, kappa)
            if a and not (isinstance(a.get("mave"), float) and np.isnan(a["mave"])):
                xa.append(s); ya.append(a["mave"])
            if b and not (isinstance(b.get("mave"), float) and np.isnan(b["mave"])):
                xb.append(s); yb.append(b["mave"])
        col = COLORS[corr]
        if xa:
            ax.plot(xa, ya, "o--", color=col, label=f"{corr} Row A", alpha=0.7)
        if xb:
            ax.plot(xb, yb, "s-", color=col, label=f"{corr} Row B (k={kappa})")
    if clean_a and not np.isnan(clean_a.get("mave", float("nan"))):
        ax.axhline(clean_a["mave"], color="gray", ls=":", alpha=0.6,
                   label=f"clean Row A ({clean_a['mave']:.3f})")
    ax.set_xlabel("corruption severity"); ax.set_ylabel("mAVE (lower better)")
    ax.set_title("mAVE (velocity error) vs severity"); ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_mave_vs_severity.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[plots] -> {path}")


def plot_kappa_sweep():
    sweep_path = os.path.join(RESULTS_DIR, "kappa_sweep.json")
    if not os.path.exists(sweep_path):
        return
    sweep = json.load(open(sweep_path))
    fig, ax = plt.subplots(figsize=(6, 5))
    by_cond = {}
    for r in sweep:
        by_cond.setdefault(r["condition"], {}).setdefault(r["kappa"], r["amota"])
    for cond, kd in by_cond.items():
        ks = sorted(kd.keys())
        ax.plot(ks, [kd[k] for k in ks], "o-", label=cond)
    ax.set_xlabel("kappa")
    ax.set_ylabel("AMOTA")
    ax.set_title("kappa sweep — AMOTA (clean + motionblur)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_kappa_sweep.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plots] -> {path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_metric_vs_severity("amota", "AMOTA", "plot_amota_vs_severity.png")
    plot_metric_vs_severity("mota", "MOTA", "plot_mota_vs_severity.png")
    plot_metric_vs_severity("amotp", "AMOTP", "plot_amotp_vs_severity.png", lower_better=True)
    # IDS / FRAG combined
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in [(axes[0], "ids", "IDS (lower better)"),
                              (axes[1], "frag", "FRAG (lower better)")]:
        for corr, sevs in CORRUPTIONS.items():
            for tag, loader in [("Row A", row_a), ("Row B", row_b)]:
                xs, ys = [], []
                for s in sevs:
                    m = loader(f"{corr}_val_sev{s}")
                    if m:
                        xs.append(s); ys.append(m[metric])
                ls = "--" if tag == "Row A" else "-"
                ax.plot(xs, ys, ls, color=COLORS[corr],
                        label=f"{corr} {tag}", alpha=0.7 if tag == "Row A" else 1.0)
        ax.set_xlabel("severity"); ax.set_ylabel(metric)
        ax.set_title(title); ax.set_xticks([1, 2, 3])
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_idsfrag_vs_severity.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[plots] -> {path}")
    plot_st_distribution()
    plot_mave_vs_severity()
    plot_kappa_sweep()


if __name__ == "__main__":
    main()
