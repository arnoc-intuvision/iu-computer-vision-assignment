# Stage B — Reliability Estimator: Training and Ablation Report

**Deployed model:** `lcre_model_bev_real.pt` (`bev_real` variant)
**Supervision target:** `reliability_target = 1 − severity / 3`
**Consumed by:** Stage C, via `R(t) = R_base · (1 + κ · (1 − S_t))`

This report records the training configuration of the reliability estimator and the
feature ablation that determined which input subset was deployed. Tracking results
are reported separately in `results/`.

## 1. Summary

The estimator is a small MLP that maps per-frame features to a scalar `S_t ∈ (0, 1)`.
Four input subsets were compared. Accuracy and circularity trade off directly: the
subsets containing synthetic, severity-derived telemetry achieve the lowest held-out
error but fail the anti-circularity decoy gate, because the model recovers the
severity label from the telemetry rather than from sensor evidence.

The deployed subset, `bev_real`, excludes synthetic telemetry entirely. It is the
least accurate variant that passes the gate, and was selected on that basis: an
estimator that reads its own supervision target from its inputs cannot support any
downstream claim.

## 2. Architecture and training

| Property | Value |
|---|---|
| Layers | Linear(1013, 256) → BatchNorm1d → ReLU → Dropout(0.2) → Linear(256, 64) → BatchNorm1d → ReLU → Dropout(0.2) → Linear(64, 1) → Sigmoid |
| Parameters | 276,737 |
| Loss | MSE against `reliability_target` |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Batch size | 2,048; clean-frame loss weight 3.0 |
| Early stopping | Patience 10 on held-out MSE |
| Seed | 20260101 |
| Training length | 35 epochs; best held-out MSE 0.0181 at epoch 25 |

The sigmoid head asymptotes at 0 and 1 without reaching them, which keeps `R(t)`
finite at both extremes: `R(t) → R_base·(1 + κ)` as `S_t → 0`, and `R(t) = R_base`
exactly at `S_t = 1`.

## 3. Data pipeline

Scenes are recovered with a timestamp-gap heuristic and split scene-wise, with the
same split applied to all ten conditions to prevent scene-geometry leakage.

| Property | Value |
|---|---|
| Total scenes | 114 (89 fit, 25 held out) |
| Samples per condition | 4,525 fit, 1,190 held out |
| Split seed | 20260101 |

Input layout (1,013 dims):

| Block | Dims | Description |
|---|---|---|
| `pooled_cam` | 240 | 80 camera channels × (mean, variance, occupancy) |
| `pooled_lidar` | 768 | 256 LiDAR channels × (mean, variance, occupancy) |
| `telemetry_real` | 3 | Timestamp jitter, calibration residual, ego-motion magnitude |
| NaN indicators | 2 | Missing-value flags for `telemetry_real` channels 0 and 2 |

`telemetry_real` channels 0 and 2 are undefined on the first frame of each scene
(~2% of rows); these are imputed with the fit-split median and flagged. The
`StandardScaler` is fitted on the fit split only; indicator columns bypass scaling.

## 4. Feature ablation and the decoy gate

The decoy gate measures whether the estimator recovers severity from telemetry
rather than from sensor evidence. On a held-out fraction of frames the synthetic
telemetry reports a deliberately mismatched severity; a model relying on it is
disproportionately wrong on those frames. The pass threshold is a decoy MAE ratio
below 1.5.

| Variant | Input dims | Held-out MSE | Decoy ratio | Gate |
|---|---|---|---|---|
| Telemetry only | 9 | 0.0079 | 8.23 | Fail |
| Full | 1,017 | 0.0020 | 5.32 | Fail |
| `drop_nolag` | 1,015 | 0.0045 | 1.25 | Pass |
| BEV only | 1,010 | 0.0178 | 0.94 | Pass |
| **`bev_real` (deployed)** | **1,013** | **0.0181** | **0.93** | **Pass** |

Three observations follow. Telemetry alone is accurate on honest frames and collapses
on decoys — the profile of a confidence-like signal rather than a sensor-state one.
BEV features alone are noisier but unaffected by decoys, since pooled BEV statistics
reflect actual frame content. Adding the three real metadata channels to BEV changes
held-out error negligibly, so the deployed model's signal is carried by the BEV
features.

Regularisation was tested as an alternative to removing the offending channels and
did not resolve the shortcut: raising dropout to 0.5 and weight decay to 2e-3 moved
the full model's decoy ratio only from 5.32 to 4.39. The shortcut is a feature
salience problem, not overfitting.

## 5. Held-out checks (deployed model)

| Check | Result | Status |
|---|---|---|
| Severity correlation | Spearman ρ ≈ −0.94 pooled | Pass |
| Clean-frame anchor | Mean `S_t` = 0.956, expected [0.85, 1.0] | Pass |
| Monotonicity | `S_t` decreases with severity for all three corruption types | Pass |
| Decoy discrimination | Ratio 0.93, threshold < 1.5 | Pass |
| Coverage | Every `sample_token` in each validation cache has a matching score | Pass |

Per-condition `S_t` statistics and the resulting `R(t)` inflation are in
`results/r_inflation_table.md`.

## 6. Artifacts

| Artifact | Description |
|---|---|
| `lcre_model_bev_real.pt` | Deployed state dict and configuration |
| `lcre_scaler.npz` | Fitted feature scaler |
| `scene_split.json` | Fit / held-out split metadata |
| `train_log_bev_real.csv` | Training log for the deployed model |
| `<condition>_scores.npz` | Per-condition validation scores (10 files) |
| `identity_scores.npz` | Identity artifact with `S_t ≡ 1.0` |

Each score file contains `tokens`, `s_t`, `sample_idx`, `true_severity`,
`reliability_target`, `is_decoy`, `corruption_type` and `model_hash`. The model hash
prevents pairing scores with the wrong condition at Stage C.
