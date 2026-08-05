# Reliability-Adaptive Kalman Filtering for Temporally Robust 3D Detection under Sensor Degradation

**Module:** Computer Vision for Autonomous Systems (DLMAIEFSCVAS02) — Task 3, Written Assignment
**Backbone (frozen):** BEVFusion (LiDAR-camera 3D detection), evaluated on a 34-scene partial nuScenes val split under three sensor-corruption families.

This repository contains the scripts that constitute my contribution to an enhanced
**BEVFusion + Kalman-filter** pipeline. The idea: train a small per-frame **reliability
score S_t** from BEV encoder statistics + hardware telemetry (with no label-derived
inputs), then feed it into a Constant-Velocity Kalman tracker by scaling the measurement
noise:

> **R(t) = R_base · (1 + κ · (1 − S_t))**

The work is organised in three stages.

---

## Pipeline overview

| Stage | What it does | Key scripts |
|---|---|---|
| **A — Feature caching** | A non-invasive forward hook extracts pooled BEV encoder statistics per frame; a hardware-telemetry generator emits anti-circular synthetic + real telemetry channels. Cached as `.npz` (inputs for Stage B). | `stage_a_caching/` |
| **B — LCRE (Learned Cross-modal Reliability Estimator)** | A small MLP (~277k params) maps BEV stats + telemetry → a scalar S_t ∈ (0,1) that tracks injected corruption severity. Includes an anti-circularity "decoy gate" and a feature-ablation study. | `stage_b_lcre/` |
| **C — Reliability-adaptive Kalman tracking** | A vendored CV-KF tracker (Chiu et al. 2020) with the single R(t) modification wired in. Runs the Row A (fixed-R) vs Row B (adaptive R(t)) ablation matrix + κ sweep under the nuScenes tracking eval. | `stage_c_tracking/` |

### What is frozen vs. trained
- **Frozen backbone:** BEVFusion — re-run for inference only; its weights/checkout are unchanged.
- **Trained by me (Stage B):** the **LCRE** model only. The variant deployed in Stage C is
  `bev_real` (1013-dim input = 1008 BEV + 3 real telemetry + 2 NaN indicators; **no
  synthetic / label-derived telemetry**). Checkpoint: `stage_b_lcre/lcre_model_bev_real.pt`.
- **Not trained:** the tracker — only the single R(t) line was added to the vendored tracker.

### The R(t) rule and frozen κ
`KalmanBoxTracker` stores `R_base` (per-class, from training-set stats) at track birth.
Each frame, **before** `kf.update()`, `set_reliability(s_t, κ)` rescales
`self.kf.R = self.R_base * (1 + κ·(1−S_t))`. S_t is **per-frame** (looked up once per
`sample_token`, applied to every active track). `assert scalar > 0` guards positive-definiteness.
κ was swept over {1, 3, 5} and **frozen at κ = 1.0** by pooled validation AMOTA (lower is
better; the main matrix is reported at κ = 3, the pre-sweep default — see `results/`).

---

## Results summary (full detail in `results/README.md`)

| Hypothesis | Claim | Status |
|---|---|---|
| **H1** — identity | R(t) plumbing is side-effect-free: S_t≡1 ⇒ byte-identical to the fixed-R baseline | ✅ **PASS** (verified byte-identical on clean **and** beamsreducing sev3) |
| **H2** — robustness | reliability-adaptive R(t) improves tracking under corruption | ❌ **NOT SUPPORTED** — Row B < Row A on AMOTA/MOTA in every cell; larger degradation at higher severity/κ |
| **H3** — specificity | sensor-state R(t) vs per-detection-confidence R(t) | ⏸ optional control, not run |

The positive, reproducible contribution is the **Stage B diagnostic itself**: a corruption-severity
signal (S_t ↔ severity, Spearman ρ ≈ −0.94) learned from BEV features + real telemetry only,
passing the anti-circularity decoy gate — plus an honest **negative result** on consuming that
signal via measurement-noise adaptation in a CV Kalman filter. Mean AMOTA Δ (B−A) over the 9
corrupted conditions = **−0.0198**. See `results/ablation_matrix.md` for the full matrix and
`results/README.md` for the velocity/mAVE caveat and interpretation.

---

## Repository layout

```
iu-computer-vision-assignment/
├── README.md                       # this file
├── stage_b_training_report.md      # Stage B methodology, decoy-gate analysis, ablation study
├── stage_c_implementation_plan.md  # Stage C design, R(t) wiring, phase/gate plan
├── stage_a_caching/                # BEV forward hook + telemetry generator + cache ops
├── stage_b_lcre/                   # LCRE model/dataset/training + chosen checkpoint (bev_real)
├── stage_c_tracking/
│   ├── tracker_fork/               # vendored Chiu CV-KF + the single R(t) modification
│   ├── eval_tracking.py            # nuScenes TrackingEval + partial_val split monkeypatch
│   ├── run_ablation_matrix.py      # Row A + Row B across 10 conditions
│   ├── kappa_sweep.py              # κ ∈ {1,3,5} sweep
│   ├── assemble_metrics.py         # → ablation_matrix.md, r_inflation_table.md, kappa_sweep.md
│   └── make_plots.py               # → results/*.png
├── ablations/
│   └── telemetry_circularity.py    # reproducible negative-result evidence (decoy ratios)
└── results/                        # aggregate tables (.md/.json) + plots + results README
```

The committed Stage B artifacts (`stage_b_lcre/scene_split.json`, `lcre_scaler.npz`,
`lcre_model_bev_real.pt`) are intentionally tracked (they are small and required to reproduce
the S_t scores).

---

## Attribution & dependencies (not included in this repo)

- **BEVFusion / MMDetection3D** — the frozen detection backbone (MIT-style licence). Not
  included; the scripts under `stage_c_tracking/` invoke its `tools/test.py` for inference.
- **Chiu et al. (2020)** — `mahalanobis_3d_multi_object_tracking` (the CV-KF tracker). Vendored
  under `stage_c_tracking/tracker_fork/` (with sklearn→scipy and path-arg compatibility fixes);
  the only algorithmic change I made is the R(t) line documented in `main.py`'s header comment.
- **MultiCorrupt** — the corruption generators used to build the corrupted val data. Not included.
- **nuScenes dataset** — not included (large, licensed).

## A note on paths

These scripts were authored and run on a RunPod instance whose persistent volume is mounted at
`/workspace`. They therefore contain absolute `/workspace/...` paths (and reference the
`bevfusion` conda env). They are provided as documentation of the **actual experimental runs** —
including the exact reproduction commands — rather than as a turnkey cross-machine pipeline.
`results/README.md` contains the full reproduction command sequence.
