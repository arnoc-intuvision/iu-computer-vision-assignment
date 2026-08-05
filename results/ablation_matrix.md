# Stage C — Ablation Matrix (Row A vs Row B)

Row A = fixed-R baseline. Row B = reliability-adaptive R(t), kappa=3.

Metrics from nuScenes `tracking_nips_2019` eval on the 34-scene partial val split.

`d` = Row B - Row A (positive = Row B better for AMOTA/MOTA/recall; negative = better for AMOTP/MOTP/IDS/FRAG where lower is better).


## AMOTA (up)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.6352 | 0.6300 | -0.0052 |
| beamsreducing sev1 | 0.5277 | 0.5258 | -0.0019 |
| beamsreducing sev2 | 0.2554 | 0.2506 | -0.0049 |
| beamsreducing sev3 | 0.0534 | 0.0411 | -0.0123 |
| missingcamera sev1 | 0.6381 | 0.6204 | -0.0176 |
| missingcamera sev2 | 0.6238 | 0.5942 | -0.0297 |
| missingcamera sev3 | 0.6195 | 0.5765 | -0.0429 |
| motionblur sev1 | 0.6081 | 0.5975 | -0.0106 |
| motionblur sev2 | 0.5469 | 0.5188 | -0.0281 |
| motionblur sev3 | 0.4790 | 0.4490 | -0.0300 |

## AMOTP (down)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.6887 | 0.6918 | 0.0031 |
| beamsreducing sev1 | 0.8648 | 0.8778 | 0.0131 |
| beamsreducing sev2 | 1.2397 | 1.2558 | 0.0162 |
| beamsreducing sev3 | 1.6530 | 1.6810 | 0.0280 |
| missingcamera sev1 | 0.7024 | 0.7338 | 0.0314 |
| missingcamera sev2 | 0.7320 | 0.7738 | 0.0418 |
| missingcamera sev3 | 0.7436 | 0.8028 | 0.0591 |
| motionblur sev1 | 0.7303 | 0.7577 | 0.0273 |
| motionblur sev2 | 0.8163 | 0.8536 | 0.0373 |
| motionblur sev3 | 0.8749 | 0.9447 | 0.0698 |

## MOTA (up)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.5850 | 0.5811 | -0.0039 |
| beamsreducing sev1 | 0.4777 | 0.4876 | 0.0099 |
| beamsreducing sev2 | 0.2496 | 0.2596 | 0.0101 |
| beamsreducing sev3 | 0.0675 | 0.0562 | -0.0113 |
| missingcamera sev1 | 0.5964 | 0.5758 | -0.0206 |
| missingcamera sev2 | 0.5858 | 0.5580 | -0.0278 |
| missingcamera sev3 | 0.5757 | 0.5328 | -0.0429 |
| motionblur sev1 | 0.5550 | 0.5504 | -0.0046 |
| motionblur sev2 | 0.4943 | 0.4674 | -0.0270 |
| motionblur sev3 | 0.4280 | 0.4113 | -0.0167 |

## MOTP (down)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.3504 | 0.3534 | 0.0030 |
| beamsreducing sev1 | 0.3677 | 0.3981 | 0.0304 |
| beamsreducing sev2 | 0.4394 | 0.5005 | 0.0611 |
| beamsreducing sev3 | 0.5906 | 0.6511 | 0.0606 |
| missingcamera sev1 | 0.3338 | 0.3544 | 0.0206 |
| missingcamera sev2 | 0.3492 | 0.3779 | 0.0287 |
| missingcamera sev3 | 0.3493 | 0.3947 | 0.0454 |
| motionblur sev1 | 0.3395 | 0.3489 | 0.0094 |
| motionblur sev2 | 0.3589 | 0.3806 | 0.0217 |
| motionblur sev3 | 0.3727 | 0.4196 | 0.0469 |

## Recall (up)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.7420 | 0.7412 | -0.0008 |
| beamsreducing sev1 | 0.6098 | 0.6247 | 0.0149 |
| beamsreducing sev2 | 0.3674 | 0.3804 | 0.0131 |
| beamsreducing sev3 | 0.1968 | 0.2075 | 0.0107 |
| missingcamera sev1 | 0.7237 | 0.7133 | -0.0104 |
| missingcamera sev2 | 0.7150 | 0.7084 | -0.0066 |
| missingcamera sev3 | 0.6991 | 0.6691 | -0.0301 |
| motionblur sev1 | 0.6926 | 0.6768 | -0.0157 |
| motionblur sev2 | 0.6069 | 0.5929 | -0.0141 |
| motionblur sev3 | 0.5401 | 0.5373 | -0.0028 |

## IDS (down)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 255 | 257 | 2.0000 |
| beamsreducing sev1 | 249 | 298 | 49.0000 |
| beamsreducing sev2 | 321 | 255 | -66.0000 |
| beamsreducing sev3 | 707 | 792 | 85.0000 |
| missingcamera sev1 | 270 | 276 | 6.0000 |
| missingcamera sev2 | 247 | 247 | 0.0000 |
| missingcamera sev3 | 265 | 268 | 3.0000 |
| motionblur sev1 | 246 | 243 | -3.0000 |
| motionblur sev2 | 247 | 284 | 37.0000 |
| motionblur sev3 | 286 | 297 | 11.0000 |

## FRAG (down)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 176 | 179 | 3.0000 |
| beamsreducing sev1 | 175 | 225 | 50.0000 |
| beamsreducing sev2 | 248 | 218 | -30.0000 |
| beamsreducing sev3 | 307 | 338 | 31.0000 |
| missingcamera sev1 | 176 | 212 | 36.0000 |
| missingcamera sev2 | 188 | 220 | 32.0000 |
| missingcamera sev3 | 179 | 237 | 58.0000 |
| motionblur sev1 | 172 | 181 | 9.0000 |
| motionblur sev2 | 189 | 231 | 42.0000 |
| motionblur sev3 | 199 | 252 | 53.0000 |

## Summary — mean delta over corrupted conditions (Row B - Row A)

| corruption | AMOTA d | AMOTP d | MOTA d | MOTP d | IDS d | FRAG d |
|---|---|---|---|---|---|---|
| beamsreducing | -0.0064 | 0.0191 | 0.0029 | 0.0507 | 0.0129 | 23 | 17 |
| missingcamera | -0.0301 | 0.0441 | -0.0304 | 0.0316 | -0.0157 | 3 | 42 |
| motionblur | -0.0229 | 0.0448 | -0.0161 | 0.0260 | -0.0109 | 15 | 35 |

## Detection metrics (velocity-smoothed, Phase 6.3)

nuScenes detection eval on JSONs with tracker-smoothed velocities. mAP is identical across rows (box set unchanged); only mAVE (and thus NDS) can differ.

> ⚠️ **Velocity caveat:** every mAVE/NDS value below uses the *CV tracker's* velocity, NOT the detector's. The detector's own clean velocity (Stage A baseline, `baseline_detector_clean_det_metrics.json`) is **mAVE=0.310, NDS=0.7170** (matches §5: 0.309 / 0.7154). Grafting the tracker velocity degrades clean mAVE to 0.900 (**2.9x worse**) and NDS to 0.6581. The write-back was verified frame-correct (global frame, m/s; median angle(det_v, trk_v) = 1.0°, std 82°) — the loss is genuine CV-filter noise (spurious motion on ~40k static objects). Row A vs Row B remain internally valid (both use tracker velocity); there is no velocity-improvement win over the detector. See `README.md` for the full caveat.


### mAP (up)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.6912 | 0.6912 | 0.0000 |
| beamsreducing sev1 | 0.5954 | 0.5954 | 0.0000 |
| beamsreducing sev2 | 0.3543 | 0.3543 | 0.0000 |
| beamsreducing sev3 | 0.1267 | 0.1267 | 0.0000 |
| missingcamera sev1 | 0.6774 | 0.6774 | 0.0000 |
| missingcamera sev2 | 0.6671 | 0.6671 | 0.0000 |
| missingcamera sev3 | 0.6543 | 0.6543 | 0.0000 |
| motionblur sev1 | 0.6534 | 0.6534 | 0.0000 |
| motionblur sev2 | 0.5632 | 0.5632 | 0.0000 |
| motionblur sev3 | 0.4679 | 0.4679 | 0.0000 |

### NDS (up)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.6581 | 0.6580 | -0.0000 |
| beamsreducing sev1 | 0.5950 | 0.5962 | 0.0013 |
| beamsreducing sev2 | 0.4407 | 0.4421 | 0.0014 |
| beamsreducing sev3 | 0.2872 | 0.2872 | 0.0000 |
| missingcamera sev1 | 0.6516 | 0.6524 | 0.0008 |
| missingcamera sev2 | 0.6454 | 0.6484 | 0.0030 |
| missingcamera sev3 | 0.6359 | 0.6386 | 0.0026 |
| motionblur sev1 | 0.6285 | 0.6289 | 0.0004 |
| motionblur sev2 | 0.5718 | 0.5742 | 0.0024 |
| motionblur sev3 | 0.5078 | 0.5078 | 0.0000 |

### mAVE (down)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.8997 | 0.9002 | 0.0005 |
| beamsreducing sev1 | 0.9304 | 0.9177 | -0.0127 |
| beamsreducing sev2 | 0.9901 | 0.9761 | -0.0139 |
| beamsreducing sev3 | 1.1339 | 1.1577 | 0.0238 |
| missingcamera sev1 | 0.8829 | 0.8748 | -0.0081 |
| missingcamera sev2 | 0.9002 | 0.8700 | -0.0302 |
| missingcamera sev3 | 0.9101 | 0.8837 | -0.0265 |
| motionblur sev1 | 0.9419 | 0.9375 | -0.0044 |
| motionblur sev2 | 0.9799 | 0.9556 | -0.0243 |
| motionblur sev3 | 1.0486 | 1.0115 | -0.0371 |

### mATE (down)

| condition | Row A | Row B | d (B-A) |
|---|---|---|---|
| clean | 0.2567 | 0.2567 | 0.0000 |
| beamsreducing sev1 | 0.2913 | 0.2913 | 0.0000 |
| beamsreducing sev2 | 0.3843 | 0.3843 | 0.0000 |
| beamsreducing sev3 | 0.5184 | 0.5184 | 0.0000 |
| missingcamera sev1 | 0.2593 | 0.2593 | 0.0000 |
| missingcamera sev2 | 0.2632 | 0.2632 | 0.0000 |
| missingcamera sev3 | 0.2690 | 0.2690 | 0.0000 |
| motionblur sev1 | 0.2752 | 0.2752 | 0.0000 |
| motionblur sev2 | 0.3025 | 0.3025 | 0.0000 |
| motionblur sev3 | 0.3400 | 0.3400 | 0.0000 |
