# Plan: Stage C — Reliability-Adaptive Kalman Tracking

**TL;DR:** Re-emit `bev_real` Sₜ scores (retraining from committed code for provenance), vendor the Chiu et al. (2020) CV-KF tracker (`eddyhkchiu/mahalanobis_3d_multi_object_tracking`), wire the single R(t)=R_base·(1+κ·(1−Sₜ)) modification, pass the identity (H1) and baseline-reproduction (§4) hard gates, then run the Row A vs Row B ablation matrix across 4 conditions × 3 severities + κ sweep. The tracker ingests nuScenes detection JSON, which must be regenerated and persisted — but first the **corrupted val sensor data itself must be regenerated** (it was deleted after Stage A caching; only the `.npz` caches and clean val survive). Evaluation uses the nuScenes devkit `eval.tracking` module with the same `partial_val` split monkeypatch already used for detection eval.

---

## Critical findings from discovery (read before executing)

1. **Raw detection JSON files DO NOT EXIST on disk, AND the corrupted val sensor data was deleted after Stage A caching.** The `eval_results/` and `work_dirs/` dirs contain only metric-summary JSONs, not the raw `results_nusc.json` per-sample detection files (`NuScenesMetric.format_results` writes to a `tempfile.TemporaryDirectory()` deleted after eval when `jsonfile_prefix` is None). Worse, the `eval_roots/` symlinks point at non-existent directories — only the clean val data and the `.npz` caches survive. **Phase 2 must regenerate the corrupted val sensor data first** (via the MultiCorrupt converters on the val split), then build eval roots, then run BEVFusion inference with `test_evaluator.jsonfile_prefix` to persist `results_nusc.json` for all 10 val conditions. Budget ~10h total (corrupted val regeneration ~7.5h + inference ~2.5h); `motionblur` alone is ~1.5h/severity at val scale.

2. **filterpy is NOT installed** in the `bevfusion` conda env. Chiu's tracker imports `from filterpy.kalman import KalmanFilter`. Must `pip install filterpy` (or vendor a minimal KF). filterpy is pure-python, lightweight, and the simplest path.

3. **`sklearn.utils.linear_assignment_` is REMOVED** in sklearn 1.3.2 (installed). Chiu's `main.py` and `get_nuscenes_stats.py` use it. Replace with `scipy.optimize.linear_sum_assignment` (available, scipy 1.10.1 installed).

4. **Chiu's tracker operates in GLOBAL coordinates** — no ego-motion compensation needed (unlike AB3DMOT's KITTI version). Detections from BEVFusion's `results_nusc.json` are already in the global frame (translation is global). The tracker loops scene-by-scene, sample-by-sample in chronological order via `nusc.get('sample', token)['next']`.

5. **The R modification point** is in `KalmanBoxTracker`. R_base is set per-class at track birth (`covariance_id==2`, `self.kf.R = covariance.R[tracking_name]`, a 7×7 diagonal matrix from training-set stats in `covariance.py`). For R(t), we must scale `self.kf.R` by `(1+κ·(1−Sₜ))` **per frame before `self.kf.update()`** — R is set once at birth but must be rescaled each update step with the current frame's Sₜ.

6. **The partial_val split monkeypatch** exists in `mmdetection3d/mmdet3d/evaluation/metrics/nuscenes_metric.py` (lines ~240-260) for detection eval. The tracking eval (`nuscenes.eval.tracking.evaluate.TrackingEval`) calls `load_gt(nusc, self.eval_set, TrackingBox)` → `create_splits_scenes()` from `nuscenes.eval.common.loaders`. Must replicate the same monkeypatch on `nuscenes.eval.common.loaders.create_splits_scenes` before constructing `TrackingEval`. The `partial_val` key already exists in `/workspace/data/nuscenes/v1.0-trainval/splits.json` (34 scenes).

7. **Existing Stage B artifacts to DELETE** (per user instruction + §9): `lcre_model.pt`, `lcre_model_bev.pt`, `lcre_model_bev_real_bev_real.pt`, `lcre_model_drop_nolag_drop_nolag.pt`, `lcre_model_reg1/2/3.pt`, `lcre_model_telemetry.pt`, all `train_log_*.csv` except the new bev_real one, and all 11 score files in `/workspace/scores/` (they're from `drop_nolag`). Keep `lcre_train.py`, `lcre_dataset.py`, `lcre_model.py`, `emit_scores.py`, `run_boundary_checks.py`, `scene_split.json`, `lcre_scaler.npz` (the scaler is split-dependent and reusable — but verify it's from the committed split).

8. **The `bev_real` ablation spec** is already defined in `lcre_dataset.py`: `BEV_PLUS_REAL_COLS` = 1008 BEV + 3 real telemetry + 2 indicators = 1013 dims. Train via `python lcre_train.py --ablation bev_real`.

---

## Steps

### Phase 0 — Repository hygiene (§9) — *parallel with Phase 1*

**0.1 Delete stale Stage B artifacts** — remove all checkpoints/logs/scores from non-bev_real variants:
- Delete: `scores/lcre_model.pt`, `lcre_model_bev.pt`, `lcre_model_bev_real_bev_real.pt`, `lcre_model_drop_nolag_drop_nolag.pt`, `lcre_model_reg1.pt`, `lcre_model_reg2.pt`, `lcre_model_reg3.pt`, `lcre_model_telemetry.pt`
- Delete: `scores/train_log_bev.csv`, `train_log_bev_real_bev_real.csv`, `train_log_drop_nolag_drop_nolag.csv`, `train_log_reg1.csv`, `train_log_reg2.csv`, `train_log_reg3.csv`, `train_log_telemetry.csv`
- Delete: all 11 score files in `scores/` (`*_scores.npz`, `identity_scores.npz`) — they're from `drop_nolag`
- Keep: `lcre_train.py`, `lcre_dataset.py`, `lcre_model.py`, `emit_scores.py`, `run_boundary_checks.py`, `scene_split.json`, `lcre_scaler.npz`
- ⚠️ Verify `scene_split.json` and `lcre_scaler.npz` are consistent with the committed `lcre_dataset.py` (same seed 20260101, same NaN handling). If in doubt, regenerate by running the data-prep path of `lcre_train.py` once.

**0.2 Create Stage C directory structure** (§9):
```
/workspace/
├── stage_a_caching/           # move stage_a hook + scripts here (or symlink)
├── stage_b_lcre/               # move lcre_*.py + emit_scores.py here
│   ├── train_lcre.py
│   ├── emit_scores.py
│   └── boundary_checks.py
├── ablations/
│   └── telemetry_circularity.py   # regenerate the negative-result evidence
├── stage_c_tracking/
│   ├── tracker_fork/           # vendored Chiu tracker + R(t) modification
│   ├── run_ablation_matrix.py
│   ├── kappa_sweep.py
│   └── eval_tracking.py        # partial_val monkeypatch + TrackingEval wrapper
└── results/                    # tables + plots only
```
- Move `scores/lcre_*.py`, `emit_scores.py`, `run_boundary_checks.py` → `stage_b_lcre/` (rename `lcre_train.py` → `train_lcre.py`, `run_boundary_checks.py` → `boundary_checks.py`). Update internal `sys.path` and `SCORES_DIR` references.
- Move `mmdetection3d/tools/cv_assign_stage_a_cache_hook.py` + `hw_telemetry_generator.py` + `check_cache_matrix.sh` + `inspect_cache.py` → `stage_a_caching/` (or leave in place with symlinks — they're imported by `lcre_dataset.py`).
- Create `ablations/telemetry_circularity.py` — a script that retrains the telemetry-only, full, reg1/2/3, and BEV-only variants and reports decoy ratios (8.23 / 5.32 / 5.32→4.39 / 0.93). This is the reproducible evidence for the circularity finding.

### Phase 1 — Re-emit bev_real scores (§1) — **GATE**

**1.1 Retrain bev_real** — *depends on 0.1*
- Run: `python stage_b_lcre/train_lcre.py --ablation bev_real --epochs 100 --batch-size 2048 --clean-weight 3.0`
- This produces `scores/lcre_model_bev_real.pt` (1013-dim: 1008 BEV + 3 real telemetry + 2 NaN indicators, NO synthetic telemetry).
- Verify the model config: `input_dim=1013`, `ablation="bev_real"`, `col_subset=BEV_PLUS_REAL_COLS`.

**1.2 Run boundary checks on bev_real** — *depends on 1.1*
- Run: `python stage_b_lcre/boundary_checks.py --model scores/lcre_model_bev_real.pt`
- **Sanity-check (not exact targets)** against §1 reference values:
  - Decoy ratio ≈ 0.93 (if >1.4, synthetic telemetry leaked back in — STOP)
  - Held-out MSE ≈ 0.0178
  - Clean mean Sₜ ≈ 0.908
  - Per-corruption Sₜ means: beamsreducing {0.657, 0.351, 0.011}, missingcamera {0.585, 0.314, 0.175}, motionblur {0.665, 0.309, 0.059}
  - Spearman ρ pooled ≈ −0.94, per-type all < −0.9
  - Monotonicity: sev0 > sev1 > sev2 > sev3 per corruption type
- **If decoy ratio > 1.5: STOP.** The model has learned a telemetry shortcut. Iterate (the `bev_real` spec excludes all synthetic telemetry, so this should pass — if it doesn't, the BEV features alone are insufficient and the contribution claim needs revisiting).

**1.3 Emit 11 score files** — *depends on 1.2 passing*
- Run: `python stage_b_lcre/emit_scores.py --model scores/lcre_model_bev_real.pt --out-dir /workspace/scores_bev_real/`
- Produces 10 × `<corruption>_val_sev<N>_scores.npz` + `identity_scores.npz` in a clean `scores_bev_real/` dir.
- Each file contains: `tokens`, `s_t`, `sample_idx`, `true_severity`, `reliability_target`, `is_decoy`, `corruption_type`, `model_hash`.
- Verify coverage (§5.6): every token in each val cache has a matching score.
- Verify `identity_scores.npz`: all `s_t == 1.0` exactly.

**GATE: Do not proceed to Phase 2 until the 11 bev_real score files exist and boundary checks reproduce approximately the §1 reference values.**

⚠️ **Serialize Phase 1 before Phase 2 — do NOT run them in parallel.** Phase 1 is ~30 minutes and is a hard gate; Phase 2 is ~10 hours of GPU. If `bev_real` doesn't reproduce (e.g., decoy ratio comes back at 1.4 instead of ≈0.93), it means synthetic telemetry leaked back into the feature selection and the whole Stage C premise needs revisiting — you want to know that *before* burning 10 hours of compute on detection JSONs you'd then have to throw away. Confirm the boundary checks pass, then commit to Phase 2.

### Phase 2 — Regenerate corrupted val data + detection JSONs (tracker input) — *after Phase 1 gate*

⚠️ **The corrupted val data was deleted after Stage A caching.** The `eval_roots/` symlinks point at non-existent directories — only the clean val data and the `.npz` caches survive. Phase 2 must **regenerate the corrupted val sensor data first** (via the MultiCorrupt converters), then build eval roots, then run BEVFusion inference. This is the dominant cost of Stage C.

**2.1 Regenerate corrupted val sensor data** — *depends on 0 (and Phase 1 gate)*
- For each corruption × severity (9 combinations), run the MultiCorrupt converters on the **val** split (1353 samples / 34 scenes):
  - ⚠️ **Use the ORIGINAL converters** (`converter/img_converter.py` / `converter/lidar_converter.py`), **NOT the `_train.py` patched copies**. The originals hardcode `nuscenes_infos_val.pkl`, which is what you want here; the `_train.py` copies were patched to read the train pkl and would produce the wrong split.
  - `beamsreducing` (LiDAR): `converter/lidar_converter.py`, `-s True` for sweeps. ~30 min/severity.
  - `missingcamera` (camera): `converter/img_converter.py`, `-s True`. ~30 min/severity.
  - `motionblur` (cross-modal): **two** converter runs (img + lidar) into the *same* `-d` directory (they write disjoint subdirs, merge additively). **~1.5h/severity** — the image blur kernel dominates. Budget accordingly.
- Converter defaults: `--seed 1000`, `--sweep False`. ⚠️ `beamsreducing` and `motionblur` need `-s True` for sweeps. ⚠️ The argparse `type=bool` trap: `-s False` evaluates as **truthy**, so always write `-s True` explicitly — never `-s False`.
- Output: corrupted sensor files under per-condition dirs (e.g. `/workspace/corrupted_val/beamsreducing_sev1/`).
- 💾 **Disk:** the val corrupted data is ~42 GB. Current usage is ~206 GB of the 300 GB quota — that fits, but delete each condition's bulk data after its detection JSON is persisted (Phase 2.3) if it gets tight. (Use the RunPod dashboard "Volume usage" or `du -sh /workspace` for real usage; `df -h /workspace` reports the underlying MooseFS cluster, not the quota.)
- Budget: beamsreducing ~1.5h (3×30min), missingcamera ~1.5h, motionblur ~4.5h (3×1.5h) = **~7.5h** for corrupted val regeneration. This is the single most expensive step of Stage C.

**2.2 Build eval roots for all 10 val conditions** — *depends on 2.1*
- The existing `eval_roots/` dir has `*_1/`, `*_2/`, `*_3/` for beamsreducing, missingcamera, motionblur, but their symlinks point at the deleted corrupted data — **rebuild them** against the regenerated dirs from 2.1.
- Clean val uses `/workspace/mmdetection3d/data/nuscenes/` directly (this data survives).
- For each corrupted condition, `build_eval_root` (the shell function in `build_eval_root.sh`) creates the hybrid root (clean symlinks + corrupted sensor dirs). Source the script in any batch script that uses it.
- ⚠️ Network filesystem race: after `build_eval_root`, run `sync && sleep 5` and verify all 7 `samples/` sensor dirs exist before launching inference.

**2.3 Run BEVFusion inference with persisted JSON** — *depends on 2.2*
- For each of the 10 val conditions, run `tools/test.py` with `test_evaluator.jsonfile_prefix` set to persist `results_nusc.json`:
  ```bash
  python mmdetection3d/tools/test.py $BEV_CONFIG $BEV_CKPT \
    --cfg-options test_dataloader.dataset.data_root=<eval_root>/ \
      test_evaluator.jsonfile_prefix=/workspace/detections/<condition>/ \
    2>&1 | tee /workspace/logs/det_<condition>.log
  ```
- This writes `/workspace/detections/<condition>/pred_instances_3d/results_nusc.json` (the nuScenes-format detection file the tracker ingests).
- ⚠️ `test_evaluator` is a separate config node from `test_dataloader.dataset`. Setting `jsonfile_prefix` makes the evaluator persist the JSON even though it also runs the detection eval (which we can ignore or use to re-verify the §5 baselines).
- Budget: ~12-15 min per condition × 10 = ~2-2.5h GPU. Run via `nohup` batch script.
- Verify each `results_nusc.json` has `meta` + `results` keys, and `results` has 1353 sample_tokens matching `nuscenes_infos_val.pkl`.
- ⚠️ **Verify the FIRST condition's JSON appears on disk before launching the rest.** If `jsonfile_prefix` is misconfigured or the flag name is wrong for this MMDet3D version, you'd otherwise discover it 10 hours later with nothing on disk. Run condition #1 (clean), confirm `/workspace/detections/clean_val_sev0/pred_instances_3d/results_nusc.json` exists and is non-empty, *then* launch the remaining 9.
- ⚠️ **Size check each JSON.** Each `results_nusc.json` contains every predicted box for 1353 samples — expect tens-to-hundreds of MB. If one comes out at a few KB, it's empty or truncated (the evaluator ran but wrote nothing, or the path was wrong). Assert `os.path.getsize(path) > 1_000_000` (1 MB floor) before accepting a condition as done.

**Phase 2 total budget: ~7.5h (corrupted val regeneration) + ~2.5h (inference) ≈ ~10h.** Plan for an overnight `nohup` run. Verify each JSON as it completes (don't wait for all 10 to find a format bug) — and specifically verify the first one before committing to the rest.

### Phase 3 — Vendor and modify the tracker (§2) — *depends on 0.2, 2.2*

**3.1 Install filterpy** — *depends on 0.2*
- `pip install filterpy` in the `bevfusion` env. Pure-python, lightweight (~100KB). No CUDA deps.
- Verify: `python -c "from filterpy.kalman import KalmanFilter; print('ok')"`
- ⚠️ This must be reinstalled each pod restart (not in `/workspace`). Add to `init.sh`.

**3.2 Vendor Chiu tracker** — *depends on 3.1*
- Clone `eddyhkchiu/mahalanobis_3d_multi_object_tracking` into `stage_c_tracking/tracker_fork/` (or copy the relevant files: `main.py`, `covariance.py`, `utils.py`, `evaluate_nuscenes.py`).
- Apply compatibility fixes:
  - Replace `from sklearn.utils.linear_assignment_ import linear_assignment` with `from scipy.optimize import linear_sum_assignment` and adapt the call signature (scipy returns row_ind, col_ind separately).
  - Replace hardcoded paths (`/juno/u/hkchiu/...`) with configurable args.
  - The `numba@jit` decorators on `poly_area`, `box3d_vol`, etc. — keep (numba 0.58.1 installed) but they'll fall back to pure-python if numba fails.
- Keep the `covariance_id=2` path (nuScenes-specific P/Q/R from training stats in `covariance.py`).

**3.3 Wire R(t) — the single modification** — *depends on 3.2*
- In `KalmanBoxTracker.__init__`: store `self.R_base = self.kf.R.copy()` (the per-class R from `covariance.R[tracking_name]`).
- Add a method `set_reliability(self, s_t, kappa)`:
  ```python
  def set_reliability(self, s_t, kappa):
      scalar = 1.0 + kappa * (1.0 - s_t)
      assert scalar > 0, f"R(t) scalar must be positive, got {scalar}"
      self.kf.R = self.R_base * scalar
  ```
  ⚠️ **Do NOT special-case `scalar == 1.0`.** `R_base * 1.0` is exact in IEEE 754 — there is no floating-point drift to avoid. Branching to `R_base.copy()` would bypass the arithmetic the identity check exists to test. If the identity check fails, that's a real bug (most likely `self.kf.R = self.kf.R * scalar` accumulating instead of `self.kf.R = self.R_base * scalar`) — fix the bug, don't hide it.
- In `AB3DMOT.update()`, before calling `trk.update(dets[d,:][0], ...)`, call `trk.set_reliability(s_t_for_current_frame, kappa)`.
- **Sₜ is per-frame, not per-track.** Look it up once per `sample_token` and apply the same scalar to every active track's R before that frame's update. Do NOT freeze Sₜ at track birth — each frame re-looks-up Sₜ by its `sample_token` and rescales every active track's R from `R_base`.
- The `s_t_for_current_frame` comes from the score file, looked up by `sample_token`.
- Add a `--score-file` arg to `track_nuscenes()` that loads the score `.npz` and builds a `{sample_token: s_t}` dict. When `--score-file` is None or points to `identity_scores.npz`, R(t) = R_base exactly (H1).
- **Change nothing else** — not the association metric (Mahalanobis with `trks_S = H·P·Hᵀ + R`), not track birth/death, not the motion model (CV with state [x,y,z,θ,l,w,h,ẋ,ẏ,ż]).

**3.4 Assert positive-definiteness** — *depends on 3.3*
- R_base is diagonal with positive entries (from `covariance.py`), so PD. Scaling by a positive scalar preserves PD. The `assert scalar > 0` in `set_reliability` is the code-level check.
- Add a one-time assertion at tracker init: `assert np.all(np.diag(self.R_base) > 0)`.

### Phase 4 — Identity check / H1 (§3) — **HARD GATE** — *depends on 3.3, 1.3, 2.2*

Run the identity check on **TWO conditions**: clean *and* beamsreducing sev3. Clean alone barely exercises the R(t) path (few tracks, all near R_base); beamsreducing sev3 has many tracks under heavy degradation, so the per-frame `set_reliability` call is exercised across a wide Sₜ range — a much stronger test that the plumbing introduces no side effects.

**4.1 Run tracker with identity_scores.npz on BOTH conditions**
- Clean: `python stage_c_tracking/tracker_fork/main.py val 2 m 11 greedy true nuscenes --score-file /workspace/scores_bev_real/identity_scores.npz --detection-file /workspace/detections/clean_val_sev0/pred_instances_3d/results_nusc.json --output /workspace/tracking_results/identity_clean.json`
- Beamsreducing sev3: `... --detection-file /workspace/detections/beamsreducing_val_sev3/pred_instances_3d/results_nusc.json --output /workspace/tracking_results/identity_beams3.json`
- `covariance_id=2` (nuScenes stats), `match_distance=m` (Mahalanobis), `match_threshold=11`, `match_algorithm=greedy`, `use_angular_velocity=true` (the Chiu proposed-method config from `run.sh` line 22).
- The identity artifact has Sₜ≡1.0 for every token, so `1+κ·(1−1.0)=1` and R(t)=R_base exactly on both conditions.

**4.2 Run unmodified baseline tracker on BOTH conditions**
- Run the same two conditions but with `--score-file None` (or a code path that skips the `set_reliability` call entirely — R stays at R_base). Outputs: `baseline_clean.json`, `baseline_beams3.json`.

**4.3 Byte-identical diff on BOTH conditions**
- `diff <(python -m json.tool identity_clean.json) <(python -m json.tool baseline_clean.json)` — must be empty.
- `diff <(python -m json.tool identity_beams3.json) <(python -m json.tool baseline_beams3.json)` — must be empty.
- If either differs by even one track: STOP. Debug the R(t) plumbing. The identity artifact forces `1+κ·(1−1.0)=1`, R(t)=R_base exactly. Any difference is a real bug — most likely `self.kf.R = self.kf.R * scalar` accumulating across frames instead of `self.kf.R = self.R_base * scalar` rescaling from the stored base each frame. Fix the bug; do not special-case `scalar == 1.0` to hide it.

**HARD GATE: Do not proceed until BOTH identity checks are byte-identical.**

### Phase 5 — Baseline reproduction (§4) — **HARD GATE** — *depends on 4.3*

**5.1 Run fixed-R tracker on clean val detections**
- Same as 4.2 (the unmodified baseline). Output: `tracking_results/baseline_clean.json`.

**5.2 Evaluate with nuScenes tracking eval**
- Build `stage_c_tracking/eval_tracking.py` that:
  1. Monkeypatches `nuscenes.eval.common.loaders.create_splits_scenes` to return `partial_val` from `splits.json` (same pattern as `nuscenes_metric.py` lines 240-260).
  2. Constructs `TrackingEval(config=tracking_nips_2019, result_path=..., eval_set='val', output_dir=..., nusc_version='v1.0-trainval', nusc_dataroot='/workspace/mmdetection3d/data/nuscenes')`.
  3. Runs `nusc_eval.main(render_curves=False)`.
  4. Reads `metrics_summary.json` → extracts AMOTA, AMOTP, IDS, FRAG, MOTA, MOTP.
- Verify the evaluator sees exactly 34 scenes / 1353 samples (assert in the eval wrapper).

**5.3 Sanity-check the AMOTA**
- The fixed-R CV-KF AMOTA on clean data must land near the tracker's published nuScenes-val figure, adjusted for (a) BEVFusion detections vs the original paper's Megvii detections, and (b) the 34-scene partial subset.
- Chiu et al. report AMOTA ≈ 0.748 on full val with Megvii detections. With BEVFusion (a stronger detector) on 34 scenes, expect a different but plausible figure. The key is **internal consistency**: the baseline must be a stable anchor, not a specific number.
- **If AMOTA is implausibly low (e.g., <0.3) or the evaluator errors: STOP.** Debug the split, the detection JSON format, or the tracker before trusting any Row B result.

**HARD GATE: Do not proceed to Phase 6 until the baseline reproduces a plausible AMOTA.**

### Phase 6 — Ablation matrix (§6.2) — *depends on 5.3*

**6.1 Row A: fixed-R baseline across all conditions**
- Run the unmodified tracker (fixed R, no score file) on all 10 val detection JSONs:
  - clean_val_sev0, beamsreducing_val_sev{1,2,3}, missingcamera_val_sev{1,2,3}, motionblur_val_sev{1,2,3}
- Output: `tracking_results/rowA_<condition>.json` (10 files).
- Evaluate each with `eval_tracking.py` → `results/rowA_<condition>_metrics.json`.

**6.2 Row B: reliability-adaptive R(t) across all conditions**
- Run the modified tracker with the corresponding `scores_bev_real/<condition>_scores.npz` and κ=3 (midpoint, pre-frozen pending the sweep).
- For clean: use `clean_val_sev0_scores.npz` (the model's natural clean predictions, mean Sₜ≈0.908 — this is the real clean case, NOT the identity artifact).
- Output: `tracking_results/rowB_<condition>_kappa3.json` (10 files).
- Evaluate each → `results/rowB_<condition>_kappa3_metrics.json`.

**6.3 mAVE / NDS reporting (§5)**
- For each tracking result, write the tracker's smoothed per-object velocities back into a detection-format results file (replace `velocity` in the detection JSON with the tracker's estimated velocity per `tracking_id`).
- Re-run the **detection** eval (`nuscenes_metric.py` path, which already has the partial_val patch) on this modified file → mAVE, NDS.
- This shows whether tracking improves velocity estimation (the mAVE story).

**6.4 Assemble the ablation matrix table**
- For each cell: {AMOTA ↑, AMOTP ↓, mAVE ↓, NDS ↑}.
- Rows: A (fixed R), B (reliability R(t) at κ=3).
- Columns: clean, beamsreducing sev{1,2,3}, missingcamera sev{1,2,3}, motionblur sev{1,2,3}.
- Report Row A vs B deltas under corruption (H2).
- ⚠️ **Keep H1 and clean-preservation separate.** They are different things:
  - **H1 (identity)** = `identity_scores.npz` (Sₜ≡1.0) → byte-identical to Row A → **hard gate** (Phase 4). This is the code-level proof that the R(t) plumbing introduces no side effects.
  - **Clean-preservation** = Row B on clean using the *real* `clean_val_sev0_scores.npz` (mean Sₜ=0.908, so R inflates 1.28× at κ=3) → the tracker **WILL differ** from Row A on clean → this is a **reported finding, not a gate**. Report the magnitude of the clean delta honestly; do not treat it as a pass/fail check. The §8 success threshold (Row B clean within ±0.5 mAP / comparable AMOTA of Row A) is a *soft* criterion for the write-up, not a hard gate that blocks Phase 7.

### Phase 7 — κ sweep (§6.3) — *depends on 6.2*

**7.1 Sweep κ ∈ {1, 3, 5} on clean + one corruption**
- Run Row B with κ={1,5} on clean + motionblur (the most promising case per §7).
- κ=3 already done in 6.2.
- Select κ by validation AMOTA (pooled across the two conditions), then **freeze κ**.
- Do NOT tune κ per corruption.

**7.2 Report the sweep in the appendix**
- Table: κ × {clean, motionblur} × {AMOTA, AMOTP, mAVE}.

### Phase 8 — Row C (H3 control, OPTIONAL) — *only if 1-7 banked*

**8.1 Implement per-detection-confidence R(t)**
- Replace the Sₜ-driven scalar with a per-detection-confidence scalar: `R(t) = R_base · (1 + κ · (1 − confidence))` where `confidence` is the detection_score of the matched detection.
- This is the NSA-KF / GIAOTracker mechanism (confidence → R).

**8.2 Run Row C across all conditions**
- Same conditions as Row A/B, κ=3 (frozen from 7.1).
- Compare Row B vs Row C on motionblur (the expected discriminator — Sₜ drops on blur while confidence stays high).

### Phase 9 — Results, plots, README (§9) — *depends on 6, 7*

**9.1 Generate result tables**
- `results/ablation_matrix.md` — the core {AMOTA, AMOTP, mAVE, NDS} × {A, B} × {10 conditions} table.
- `results/kappa_sweep.md` — the κ sweep appendix table.
- `results/r_inflation_table.md` — the R(t)/R_base table per condition (§6.4).

**9.2 Generate plots**
- `results/plot_amota_vs_severity.png` — AMOTA vs severity for Row A vs B, per corruption type.
- `results/plot_mave_reduction.png` — mAVE reduction (Row A − Row B) vs severity, per corruption.
- `results/plot_st_vs_severity.png` — Sₜ distribution per severity (from Stage B, for context).

**9.3 Write README.md**
- State: frozen backbone (BEVFusion), what is trained (only the LCRE), what is not.
- How to reproduce each reported table (exact commands).
- The R(t) rule and κ value.
- The H1/H2/H3 results (or null result + redirection per §8).

---

## Relevant files

### Existing (to modify/move)
- `/workspace/scores/lcre_train.py` → `stage_b_lcre/train_lcre.py` — retrain `bev_real` via `--ablation bev_real`; the `ABLATION_SPECS["bev_real"]` = `BEV_PLUS_REAL_COLS` (1013 dims) is already defined.
- `/workspace/scores/lcre_dataset.py` → `stage_b_lcre/lcre_dataset.py` — defines `BEV_PLUS_REAL_COLS`, `VAL_CONDITIONS`, `assemble_features`, `Scaler`; the `bev_real` spec excludes all 4 synthetic telemetry columns (keeps 1008 BEV + 3 real + 2 indicators).
- `/workspace/scores/lcre_model.py` → `stage_b_lcre/lcre_model.py` — `LCRE` MLP with sigmoid head; `input_dim` parameterized (pass 1013 for bev_real).
- `/workspace/scores/emit_scores.py` → `stage_b_lcre/emit_scores.py` — emits per-condition score `.npz` + `identity_scores.npz`; add `--out-dir /workspace/scores_bev_real/` default.
- `/workspace/scores/run_boundary_checks.py` → `stage_b_lcre/boundary_checks.py` — §5.1-5.4 checks; run on `bev_real` checkpoint.
- `/workspace/mmdetection3d/mmdet3d/evaluation/metrics/nuscenes_metric.py` (lines 240-260) — **reference for the `partial_val` monkeypatch pattern** to replicate in `eval_tracking.py`.
- `/workspace/mmdetection3d/tools/test.py` — re-run with `test_evaluator.jsonfile_prefix` to persist `results_nusc.json`.
- `/workspace/build_eval_root.sh` — `build_eval_root` shell function; source in batch scripts.
- `/workspace/data/nuscenes/v1.0-trainval/splits.json` — contains `partial_val` key (34 scenes); read by both detection and tracking eval monkeypatches.
- `/workspace/mmdetection3d/data/nuscenes/val_scene_names.txt` — the 34 val scene names (cross-check against `splits.json`).

### Existing (reference, do not modify)
- `/workspace/cache/` — the 20 Stage A `.npz` caches (verified 20/20). Do NOT regenerate.
- `/workspace/scores/scene_split.json` — scene-wise fit/heldout split (seed 20260101). Reuse if consistent; else regenerate.
- `/workspace/scores/lcre_scaler.npz` — StandardScaler fit on train fit-split. Reuse if consistent; else regenerate.
- `/workspace/miniconda3/envs/bevfusion/lib/python3.8/site-packages/nuscenes/eval/tracking/` — `TrackingEval`, `TrackingBox`, `create_tracks`, `TrackingConfig`; the `tracking_nips_2019.json` config (7 classes, dist_th_tp=2.0, min_recall=0.1).
- `/workspace/miniconda3/envs/bevfusion/lib/python3.8/site-packages/nuscenes/eval/common/loaders.py` — `create_splits_scenes`, `load_prediction`, `load_gt`; the monkeypatch target.

### To create (Stage C)
- `/workspace/stage_c_tracking/tracker_fork/main.py` — vendored from `eddyhkchiu/mahalanobis_3d_multi_object_tracking/main.py`, with: (a) `sklearn.linear_assignment_` → `scipy.linear_sum_assignment`, (b) configurable paths, (c) `--score-file` arg + `set_reliability(s_t, kappa)` in `KalmanBoxTracker`, (d) R(t) applied per-frame before `kf.update()`.
- `/workspace/stage_c_tracking/tracker_fork/covariance.py` — vendored unchanged (contains the nuScenes P/Q/R per-class stats, `covariance_id==2`).
- `/workspace/stage_c_tracking/tracker_fork/utils.py` — vendored (`mkdir_if_missing`, `load_list_from_folder`, etc.).
- `/workspace/stage_c_tracking/eval_tracking.py` — wraps `TrackingEval` with the `partial_val` monkeypatch; reads `metrics_summary.json`.
- `/workspace/stage_c_tracking/run_ablation_matrix.py` — orchestrates Row A + Row B across 10 conditions.
- `/workspace/stage_c_tracking/kappa_sweep.py` — sweeps κ ∈ {1, 3, 5} on clean + motionblur.
- `/workspace/ablations/telemetry_circularity.py` — retrains telemetry-only/full/reg/BEV-only variants, reports decoy ratios (the negative-result evidence).
- `/workspace/detections/<condition>/pred_instances_3d/results_nusc.json` — 10 regenerated detection JSON files (tracker input).
- `/workspace/scores_bev_real/` — 11 bev_real score files (10 per-condition + identity).
- `/workspace/tracking_results/` — tracker output JSONs (rowA_*, rowB_*, identity_*).
- `/workspace/results/` — final tables + plots.

---

## Verification

1. **Phase 1 gate**: `python stage_b_lcre/boundary_checks.py --model scores/lcre_model_bev_real.pt` — decoy ratio < 1.5, Spearman ρ < −0.9, monotonicity holds, clean mean Sₜ ∈ [0.85, 1.0]. Compare per-corruption Sₜ means to §1 reference (approximate match).
2. **Phase 1 coverage**: every token in each of the 10 val caches has a matching `s_t` in `scores_bev_real/<condition>_scores.npz`. No orphans.
3. **Phase 2**: corrupted val sensor data regenerated for all 9 corruption×severity combinations (verify files exist under each per-condition dir before inference). Each `detections/<condition>/pred_instances_3d/results_nusc.json` has 1353 sample_tokens matching `nuscenes_infos_val.pkl` AND `os.path.getsize(path) > 1_000_000` (a few-KB file means the evaluator wrote nothing — don't accept it). **Verify the first condition's JSON (clean) before launching the other 9** — catches `jsonfile_prefix` misconfiguration early instead of 10 hours later. Re-verify detection mAP/NDS on clean matches §5 (0.6893/0.7154 ± noise) — confirms the detection JSON is correct.
4. **Phase 3**: `python -c "from filterpy.kalman import KalmanFilter"` succeeds. Tracker runs on a single condition without error.
5. **Phase 4 HARD GATE**: `diff <(identity_clean.json) <(baseline_clean.json)` AND `diff <(identity_beams3.json) <(baseline_beams3.json)` are both empty — byte-identical output on BOTH conditions. If either differs, STOP.
6. **Phase 5 HARD GATE**: `eval_tracking.py` on `baseline_clean.json` reports a plausible AMOTA (not <0.3, not errored). Evaluator sees exactly 34 scenes / 1353 samples (assert in wrapper).
7. **Phase 6**: Row A and Row B metrics computed for all 10 conditions. **H1 (identity, Phase 4) is the hard gate; clean-preservation is a reported finding, not a gate.** Row B on clean uses the real `clean_val_sev0_scores.npz` (mean Sₜ=0.908 → R inflates 1.28× at κ=3), so it **will** differ from Row A on clean — report the magnitude of this delta honestly. The §8 ±0.5 mAP / comparable AMOTA threshold is a soft write-up criterion, not a pass/fail gate that blocks Phase 7.
8. **Phase 7**: κ sweep table complete; κ frozen.
9. **Phase 9**: README states frozen backbone, trained component (LCRE only), R(t) rule, κ value, and reproduction commands for every reported table.

---

## Decisions

- **Tracker**: Vendor Chiu et al. (2020) `mahalanobis_3d_multi_object_tracking` (user choice). Uses `filterpy.kalman.KalmanFilter` (must `pip install`), operates in global coordinates (no ego-motion compensation), reads nuScenes detection JSON directly, uses `covariance_id=2` (nuScenes-specific P/Q/R from training stats).
- **bev_real model**: Retrain from scratch via committed `lcre_train.py --ablation bev_real` for full reproducibility (user choice). Delete all stale checkpoints/logs/scores from other variants.
- **R(t) application point**: Per-frame in `KalmanBoxTracker`, scaling `self.kf.R = self.R_base * (1 + κ·(1−Sₜ))` before `kf.update()`. R_base is stored at track birth; the scalar is recomputed each frame from the current sample_token's Sₜ. **Sₜ is per-frame, not per-track** — look it up once per `sample_token` and apply the same scalar to every active track's R before that frame's update; do not freeze Sₜ at track birth.
- **Identity check**: Use `identity_scores.npz` (Sₜ≡1.0) through the SAME tracker path. **Do NOT special-case `scalar == 1.0`** — `R_base * 1.0` is exact in IEEE 754, so branching to `R_base.copy()` would bypass the arithmetic the check exists to test. Run on TWO conditions (clean + beamsreducing sev3); both must be byte-identical to the unmodified baseline.
- **κ**: Sweep {1, 3, 5} on clean + motionblur, freeze by validation AMOTA. Default κ=3 for the main matrix (pre-sweep).
- **Eval**: nuScenes devkit `eval.tracking.TrackingEval` with `tracking_nips_2019` config, `partial_val` monkeypatch on `create_splits_scenes` (same pattern as detection eval).
- **mAVE/NDS**: Write tracker-smoothed velocities back into detection-format JSON, re-run detection eval (which already has the partial_val patch).
- **Null result is valid** (§8): If Row B shows no gain over Row A anywhere, report it honestly. The Stage B diagnostic (Sₜ correlates with severity at ρ≈−0.94, no label-derived inputs) + the negative tracking result is still a reportable contribution.
- **Scope included**: Phases 1-7 (bev_real re-emission, tracker fork, H1+baseline gates, Row A/B matrix, κ sweep). Phase 8 (Row C / H3) is optional, built only if 1-7 banked.
- **Scope excluded**: Regenerating Stage A caches (frozen, verified 20/20). Modifying the BEVFusion backbone (frozen). Tuning κ per corruption. Any change to the tracker beyond the R(t) line.

---

## Further Considerations

1. **filterpy reinstall on pod restart**: filterpy installs to container root (`/`), which is wiped on pod stop/start. Add `pip install filterpy` to `init.sh` alongside the other reinstall steps. Alternatively, `pip install --target /workspace/.local filterpy` and add to PYTHONPATH — but this can cause version conflicts; the `init.sh` approach is cleaner.
2. **Corrupted val regeneration + detection JSON cost**: ~10h total (~7.5h corrupted val sensor regeneration via MultiCorrupt converters + ~2.5h BEVFusion inference). This is the single most expensive step of Stage C — `motionblur` alone is ~1.5h/severity at val scale. Run as a single `nohup` batch script overnight. Verify each JSON as it completes (don't wait for all 10 to find a format bug).
3. **Chiu tracker `linear_assignment` compatibility**: The scipy `linear_sum_assignment` returns `(row_ind, col_ind)` not the `np.array([[r,c],...])` format. The vendored `greedy_match` function in `main.py` may also need adaptation. Test on a single small scene first.
4. **The 34-scene subset and tracking metrics**: AMOTA integrates over recall thresholds. With only 34 scenes (vs 150), the recall curve is noisier. Report this as a limitation. The relative delta (Row B − Row A) is more meaningful than the absolute AMOTA.
