# Stage C — Reliability-Adaptive Kalman Tracking: Results

## TL;DR (result)

We tested whether a **per-frame reliability score S_t** (from the Stage B LCRE
model, trained with no label-derived inputs) can improve a Constant-Velocity
Kalman tracker under sensor corruption, by scaling the measurement noise:

> **R(t) = R_base · (1 + κ · (1 − S_t))**

The score is well-calibrated — S_t tracks corruption severity monotonically
(clean S̄_t = 0.956 → beamsreducing sev3 S̄_t = 0.008). **But feeding it into the
Kalman filter does not improve tracking.** It consistently *hurts* AMOTA/MOTA and
increases track fragmentation, with larger degradation at higher severity and
higher κ. The κ sweep freezes κ = **1.0** (least inflation) as the best option,
yet even κ = 1 underperforms the fixed-R baseline. The velocity story is also
negative: grafting the CV tracker's velocity *degrades* mAVE 2.9× vs the
detector's own (0.31→0.90), so there is no velocity-improvement win either
(verified frame-correct — see the Phase 6.3 caveat below).

This is a **null result for H2**, reported honestly per the study plan (§8). The
positive, reproducible contribution is the Stage B diagnostic itself: a
corruption-severity signal that correlates strongly with severity
(Spearman ρ ≈ −0.94) using only BEV features + real hardware telemetry — plus a
clear negative result on one natural way to consume such a signal downstream.

| Hypothesis | Status |
|---|---|
| **H1** — R(t) plumbing is side-effect-free (identity S_t≡1 ⇒ byte-identical to baseline) | ✅ **PASS** (both clean & beamsreducing sev3, byte-identical) |
| **H2** — reliability-adaptive R(t) improves tracking under corruption | ❌ **NOT SUPPORTED** (Row B < Row A on every condition) |
| **H3** — Row C confidence-driven R(t) control | ⏸ OPTIONAL, not run (Phases 1–7 banked) |

---

## What is frozen vs. trained

- **Frozen backbone:** BEVFusion (LiDAR-camera fusion), checkpoint unchanged.
  Re-run only for inference to persist `results_nusc.json` per val condition.
- **Trained component (Stage B):** the **LCRE** reliability model only
  (`lcre_train.py --ablation bev_real`, 1013-dim input = 1008 BEV + 3 real
  telemetry + 2 NaN indicators; **no synthetic/label-derived telemetry**).
- **Not trained:** the tracker (vendored Chiu et al. 2020 CV-KF,
  `covariance_id=2`, global-frame, Mahalanobis association) — only the single
  R(t) line was added.

---

## The R(t) rule and frozen κ

- `KalmanBoxTracker` stores `self.R_base` at track birth (per-class, from
  `covariance.py` training stats). Each frame, **before** `kf.update()`,
  `set_reliability(s_t, κ)` rescales `self.kf.R = self.R_base * (1 + κ·(1−S_t))`.
- **S_t is per-frame, not per-track** — looked up once per `sample_token` and
  applied to every active track.
- `assert scalar > 0` guards positive-definiteness (R_base is diagonal-positive;
  positive scaling preserves PD).
- **κ = 1.0 (frozen)** by pooled validation AMOTA over clean + motionblur:
  κ=1 → 0.563, κ=3 → 0.549, κ=5 → 0.540. Lower κ is better; fixed-R (Row A)
  still beats all of them. The main matrix is reported at κ = 3 (the pre-sweep
  default), with the full sweep in the appendix.

---

## Key results (Row A = fixed-R baseline, Row B = adaptive R(t) at κ = 3)

**Tracking (nuScenes `tracking_nips_2019`, 34-scene partial val):**

| condition | AMOTA A | AMOTA B | Δ | FRAG A | FRAG B |
|---|---|---|---|---|---|
| clean | 0.6352 | 0.6300 | −0.0052 | 176 | 179 |
| missingcamera sev3 | 0.6195 | 0.5765 | **−0.0429** | 179 | 237 |
| motionblur sev3 | 0.4790 | 0.4490 | −0.0300 | 199 | 252 |
| beamsreducing sev3 | 0.0534 | 0.0411 | −0.0123 | — | — |

Mean AMOTA Δ (B−A) over the 9 corrupted conditions = **−0.0198**. Row B is worse
on AMOTA, AMOTP, MOTA in **every** cell, and increases FRAG/IDS.

**Detection / velocity (nuScenes `detection_cvpr_2019`, velocity-smoothed):**
- Baseline detector (Row A on clean, **original detector velocity**): **mAP 0.6912,
  NDS 0.7170, mAVE 0.3101** — matches the Stage A §5 reference (0.309 / 0.7154).
- mAP is identical across all rows (the box set is unchanged by velocity grafting).
- mAVE (the only field the graft changes): **grafting the CV tracker's velocity
  degrades mAVE 2.9×** (clean 0.310 → 0.900) and NDS (0.717 → 0.658).

> ⚠️ **Phase 6.3 caveat — the velocity write-back is correct but the premise is
> backwards.** The write-back was verified frame-correct: the angle between the
> detector's and the tracker's velocity (on matched moving objects) has median
> 1.0° (std 82°), i.e. no systematic rotation — the tracker runs in the global
> frame, same as the detector, same units (m/s). The 2.9× mAVE degradation is
> *genuine*: the detector's learned velocity head outputs v≈0 for static objects,
> whereas the constant-velocity Kalman filter integrates detection-position jitter
> into spurious motion (≈0.5 m/s on ~40k static objects). Conclusions:
> - The **Row A vs Row B mAVE comparison is internally valid** (both use tracker
>   velocity; the relative Δ is real).
> - But there is **no "tracking improves velocity estimation" story**: both rows'
>   tracker velocities are ~2.8–2.9× worse than the detector's own (0.31). Row B's
>   marginal mAVE edge over Row A on camera corruptions (mean Δ −0.022) is a small
>   difference within an already-degraded velocity estimate, not a win over the
>   detector.
> - The detection **baseline reference** (original detector velocity) is stored as
>   `baseline_detector_clean_det_metrics.json`.

---

## Interpretation of the null result

Inflating R under-reweights corrupted detections during association/update,
over-trusting the constant-velocity motion model. As severity rises, S_t falls
and R grows (up to ~4× at sev3, see `r_inflation_table.md`); the tracker then
breaks track continuity (more fragmentation/identity switches) rather than
degrading gracefully. The well-calibrated signal is therefore not consumed
beneficially by *measurement-noise adaptation* in this tracker — a useful
negative finding that redirects toward alternative consumption mechanisms
(association gating, adaptive track birth/death, or detection-level confidence,
à la the optional Row C / H3).

---

## Directory structure

```
/workspace/
├── scores_bev_real/           # 11 score files (10 conditions + identity_scores.npz)
├── detections/                # 10 BEVFusion results_nusc.json (tracker input)
├── detections_vel/            # velocity-grafted detection JSONs (Phase 6.3)
├── tracking_results/          # Row A / Row B tracker output (+ κ1/κ5 sweep runs)
├── tracking_results_vel/      # tracker re-runs emitting Kalman velocities
├── results/                   # <-- THIS: tables, plots, per-condition metrics
├── stage_b_lcre/              # LCRE train/emit/boundary scripts + model
└── stage_c_tracking/
    ├── tracker_fork/          # vendored Chiu tracker + R(t) modification
    ├── batch_eval.py          # load-once tracking & detection eval
    ├── eval_tracking.py       # TrackingEval + partial_val monkeypatch
    ├── run_ablation_matrix.py # Row A + Row B across 10 conditions
    ├── kappa_sweep.py         # κ ∈ {1,3,5} on clean + motionblur
    ├── rebuild_detections_vel.py
    ├── assemble_metrics.py    # -> ablation_matrix.md, r_inflation_table.md, kappa_sweep.md
    └── make_plots.py          # -> *.png
```

## Result artifacts

- `ablation_matrix.md` — {AMOTA, AMOTP, MOTA, MOTP, Recall, IDS, FRAG} × {Row A,
  Row B} × 10 conditions, plus the velocity-smoothed {mAP, NDS, mAVE, mATE}
  block. (Machine-readable: `ablation_matrix.json`.)
- `kappa_sweep.md` — κ × {clean, motionblur sev1–3} × {AMOTA, AMOTP, MOTA, MOTP,
  IDS, FRAG, mAVE}; frozen κ. (`kappa_sweep.json`, `kappa_frozen.json`.)
- `r_inflation_table.md` — S_t statistics and R(t)/R_base per condition.
- Plots: `plot_amota_vs_severity.png`, `plot_mota_vs_severity.png`,
  `plot_amotp_vs_severity.png`, `plot_idsfrag_vs_severity.png`,
  `plot_mave_vs_severity.png`, `plot_st_vs_severity.png`, `plot_kappa_sweep.png`.

---

## How to reproduce

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate bevfusion

# Phase 1 — bev_real scores (gate)
python stage_b_lcre/train_lcre.py --ablation bev_real --epochs 100 --batch-size 2048 --clean-weight 3.0
python stage_b_lcre/boundary_checks.py --model scores/lcre_model_bev_real.pt
python stage_b_lcre/emit_scores.py --model scores/lcre_model_bev_real.pt --out-dir /workspace/scores_bev_real/

# Phase 4 — H1 identity check (must be byte-identical on BOTH conditions)
cd stage_c_tracking/tracker_fork
python main.py val 2 m 11 greedy true nuscenes /workspace/tracking_results \
  --detection-file /workspace/detections/clean_val_sev0/pred_instances_3d/pred_instances_3d/results_nusc.json \
  --data-root /workspace/mmdetection3d/data/nuscenes \
  --output /workspace/tracking_results/identity_clean.json \
  --score-file /workspace/scores_bev_real/identity_scores.npz --kappa 3.0
# (repeat for beamsreducing sev3, then diff against the no-score baseline)

# Phase 6 — ablation matrix (track, then eval load-once)
python stage_c_tracking/run_ablation_matrix.py --row both
python stage_c_tracking/batch_eval.py track

# Phase 6.3 — velocity graft + detection eval
python stage_c_tracking/run_velocity_reruns_p8.sh 3 4      # parallel=3, staggered
python stage_c_tracking/rebuild_detections_vel.py --force
python stage_c_tracking/batch_eval.py det

# Phase 7 — κ sweep
python stage_c_tracking/kappa_sweep.py --kappas 1 3 5 \
  --conditions clean_val_sev0 motionblur_val_sev1 motionblur_val_sev2 motionblur_val_sev3

# Phase 9 — tables + plots
python stage_c_tracking/assemble_metrics.py
python stage_c_tracking/make_plots.py
```

## Limitations

- **34-scene partial val split** (vs. the 150-scene official val): AMOTA is
  integrated over recall, so the curve is noisier here. Absolute numbers are not
  comparable to published full-val figures; the **relative Δ (Row B − Row A)** is
  the meaningful quantity, and it is consistently negative.
- Only the R(t) measurement-noise mechanism was tested; association gating and
  per-detection-confidence (Row C) variants were out of scope (optional Phase 8).
- κ was swept only on clean + motionblur and frozen globally (not per-corruption),
  per the plan.
