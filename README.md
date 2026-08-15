# Reliability-Adaptive Kalman Filtering for Temporally Robust 3D Detection under Sensor Degradation

Companion code for a written assignment in *Computer Vision for Autonomous Systems*
(DLMAIEFSCVAS02), MSc Applied Artificial Intelligence.

The study asks whether an **upstream, cross-modal estimate of sensor health** — as
distinct from a detector's per-box confidence — can usefully drive the
measurement-noise covariance of a classical Kalman tracker. A small MLP reads
pooled bird's-eye-view (BEV) statistics from a frozen BEVFusion detector plus three
real nuScenes telemetry channels, and emits one scalar per frame, `S_t ∈ (0, 1]`.
That scalar scales the tracker's measurement noise:

```
R(t) = R_base · (1 + κ · (1 − S_t))
```

When the sensors look degraded, `R` inflates, the Kalman gain shrinks, and the
filter leans on its own motion prediction instead of the incoming detection.

## Result summary

| Hypothesis | Claim | Outcome |
|---|---|---|
| **H1** — identity | With `S_t ≡ 1` the fork is byte-identical to the fixed-`R` baseline | Holds (verified on clean and beams-reducing severity 3) |
| **H2** — robustness | Reliability-adaptive `R(t)` improves tracking under corruption | Not supported — AMOTA is lower in all ten conditions |

The reliability signal itself is recoverable: `S_t` falls monotonically with
injected corruption severity (Spearman ρ ≈ −0.94 on held-out frames) using only BEV
features and real vehicle metadata, and passes the anti-circularity decoy gate.
Consuming it through uniform measurement-noise inflation, however, degrades tracking
accuracy — mean ΔAMOTA over the nine corrupted conditions is **−0.0198**. The
velocity metrics require care: both tracker configurations report Kalman-smoothed
velocities that are roughly 2.9× worse than the detector's own output, so the
Column A versus Column B comparison is internally valid but is not a win over not
tracking at all. Full tables are in [`results/`](results/).

## Pipeline

| Stage | What it does | Directory |
|---|---|---|
| **A** — feature caching | A forward hook extracts pooled BEV encoder statistics and telemetry per frame during a normal inference pass, cached as `.npz`. No extra GPU passes. | `stage_a_caching/` |
| **B** — reliability estimator | A ~277k-parameter MLP maps those features to `S_t`. Deployed variant `bev_real`: 1,008 BEV statistics + 3 real telemetry channels + 2 missing-value indicators. | `stage_b_lcre/` |
| **C** — adaptive tracking | A vendored constant-velocity Kalman tracker with the single `R(t)` modification, run across the fixed-`R` and adaptive ablation matrix and a κ sweep. | `stage_c_tracking/` |

**Frozen:** the BEVFusion backbone (inference only, weights unchanged).
**Trained:** the reliability estimator, and nothing else.
**Modified:** one line in the tracker, where `R` is set.

κ was swept over {1, 3, 5} and frozen at **κ = 1.0** by pooled validation AMOTA; the
main ablation matrix is reported at κ = 3, the pre-sweep default, with the full
sweep in `results/kappa_sweep.md`.

## Repository layout

```
stage_a_caching/       BEV forward hook, telemetry generator, cache inspection
stage_b_lcre/          estimator model, dataset assembly, training, deployed checkpoint
  stage_b_training_report.md   training configuration and feature ablation
stage_c_tracking/
  tracker_fork/        vendored CV Kalman tracker plus the single R(t) change
  run_ablation_matrix.py   fixed-R and adaptive rows across 10 conditions
  kappa_sweep.py           κ ∈ {1, 3, 5}
  batch_eval.py            nuScenes tracking and detection evaluation
  assemble_metrics.py      builds the markdown result tables
ablations/
  telemetry_circularity.py four-variant decoy-gate comparison
results/               aggregate tables, plots, per-condition metrics
```

## Reproduction

Full command sequence, including the H1 identity check and the κ sweep, is in
[`results/README.md`](results/README.md).

```bash
python stage_b_lcre/train_lcre.py --ablation bev_real --epochs 100 \
    --batch-size 2048 --clean-weight 3.0
python stage_b_lcre/boundary_checks.py --model scores/lcre_model_bev_real.pt
python stage_b_lcre/emit_scores.py  --model scores/lcre_model_bev_real.pt \
    --out-dir /workspace/scores_bev_real/
python stage_c_tracking/run_ablation_matrix.py --row both
python stage_c_tracking/batch_eval.py track
python stage_c_tracking/kappa_sweep.py --kappas 1 3 5
python stage_c_tracking/assemble_metrics.py
```

These scripts were authored and run on a RunPod instance with a persistent volume at
`/workspace`, and contain absolute `/workspace/...` paths. They document the actual
experimental runs rather than providing a turnkey cross-machine pipeline.

Environment: Python 3.10, PyTorch 2.1.0, CUDA 11.8, MMDetection3D 1.4.0,
nuScenes devkit 1.1.11, on an RTX 4000 Ada (20 GB).

## Acknowledgements and attribution

This work builds directly on the following. All credit for these methods, datasets
and implementations belongs to their authors; only the reliability estimator and the
single `R(t)` substitution in the tracker are my own contribution.

**Detection backbone — BEVFusion.** Used frozen, inference only.
Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D., & Han, S. (2023).
*BEVFusion: Multi-task multi-sensor fusion with unified bird's-eye view representation.*
ICRA 2023, 2774–2781. https://doi.org/10.1109/ICRA48891.2023.10160968 ·
https://github.com/mit-han-lab/bevfusion

**Tracker — vendored and modified.** `stage_c_tracking/tracker_fork/` is derived from
the probabilistic 3D multi-object tracker of Chiu et al., with compatibility fixes
(sklearn → scipy, path arguments). The only algorithmic change is the `R(t)` line,
documented in the header of `main.py`.
Chiu, H.-K., Prioletti, A., Li, J., & Bohg, J. (2020). *Probabilistic 3D
multi-object tracking for autonomous driving.* arXiv:2001.05673 ·
https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking

**Tracking baseline and metrics.** AMOTA and AMOTP as used throughout are defined by:
Weng, X., Wang, J., Held, D., & Kitani, K. (2020). *3D multi-object tracking: A
baseline and new evaluation metrics (AB3DMOT).* IROS 2020, 10359–10366.
https://doi.org/10.1109/IROS45743.2020.9341164 ·
https://github.com/xinshuoweng/AB3DMOT

**Dataset and evaluation code — nuScenes.** Dataset, detection and tracking
evaluation, and the devkit used to compute every metric reported here.
Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., Krishnan, A.,
Pan, Y., Baldan, G., & Beijbom, O. (2020). *nuScenes: A multimodal dataset for
autonomous driving.* CVPR 2020, 11618–11628.
https://doi.org/10.1109/CVPR42600.2020.01164 ·
https://github.com/nutonomy/nuscenes-devkit

**Corruption benchmark — MultiCorrupt.** Source of the beams-reducing,
missing-camera and motion-blur corruptions at three severities.
Beemelmanns, T., Zhang, Q., Geller, C., & Eckstein, L. (2024). *MultiCorrupt: A
multi-modal robustness dataset and benchmark of LiDAR-camera fusion for 3D object
detection.* IEEE IV 2024, 3255–3261. https://doi.org/10.1109/IV55156.2024.10588664 ·
https://github.com/ika-rwth-aachen/MultiCorrupt

**Framework — MMDetection3D / OpenMMLab.** Detector configuration, inference
tooling and data pipeline.
MMDetection3D Contributors. (2020). *OpenMMLab's next-generation platform for
general 3D object detection* (v1.4.0). https://github.com/open-mmlab/mmdetection3d

**Prior art on confidence-adaptive noise.** The `R(t)` formulation is positioned
against the noise-scale-adaptive Kalman filter introduced in GIAOTracker
(Du et al., 2021, arXiv:2202.11983), which scales the covariance by per-detection
confidence rather than by an upstream sensor-state estimate.

BEVFusion, MMDetection3D, MultiCorrupt, the nuScenes devkit and the nuScenes dataset
are **not redistributed here**; the scripts invoke them from their own installations.
Each remains subject to its own licence and dataset terms.

## Committed artifacts

`stage_b_lcre/scene_split.json`, `lcre_scaler.npz` and `lcre_model_bev_real.pt` are
tracked deliberately: they are small and required to reproduce the `S_t` scores
without retraining.
