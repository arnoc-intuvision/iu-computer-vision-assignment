"""
Stage A caching hook for the reliability-adaptive Kalman tracking study.

Attaches forward hooks to a frozen BEVFusion model during a normal
`tools/test.py` inference pass and, for every sample, caches:

    [ 1008 pooled BEV stats ]  (80 cam + 256 lidar) x (mean, var, occupancy)
    [ 4 synthetic telemetry ]  from telemetry_generator (severity-derived)
    [ 3 real metadata telem ]  timestamp jitter, calib residual, ego-motion

keyed by sample_token, to a single .npz per run. The run corresponds to ONE
(corruption_type, severity) because that is how MultiCorrupt datasets are laid
out; both are passed in as constants for the whole run.

Integration (one run = one corruption x severity):
---------------------------------------------------
In tools/test.py, after the runner/model is built and BEFORE runner.test():

    from stage_a_cache_hook import StageACache
    cache = StageACache(
        model=runner.model,
        corruption_type="beamsreducing",   # or "clean"
        severity=2,                          # 0 for clean
        out_path="/workspace/cache/beamsreducing_sev2.npz",
    )
    cache.attach()
    runner.test()
    cache.save()

The hook fires inside the existing inference pass -- NO extra GPU passes.

Metadata-key robustness:
------------------------
MMDet3D's `data_sample.metainfo` key names drift across versions. This module
tries several known key paths and records NaN (not a crash) if a field is
missing, so a long caching run never dies on a key mismatch. Run
`probe_metainfo(runner)` once first (see bottom) to confirm the real keys on
your instance, then, if needed, adjust METAINFO_KEYS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from telemetry_generator import TelemetryGenerator, TelemetryConfig, SYNTH_CHANNELS

# Real metadata-anchored telemetry channel order (FIXED, appended after synthetic).
REAL_CHANNELS = (
    "meta_timestamp_jitter",     # inter-frame dt deviation from nominal 0.5s (2 Hz keyframes)
    "meta_calib_residual",       # calibration translation-norm magnitude proxy
    "meta_egomotion_magnitude",  # consecutive-frame ego translation magnitude
)

# nuScenes keyframe nominal period (2 Hz) in seconds; timestamps are microseconds.
NUSC_NOMINAL_DT_S = 0.5
NUSC_TIMESTAMP_UNIT = 1e-6  # microseconds -> seconds


# ----- pooled BEV feature statistics ---------------------------------------

def pool_feature_map(feat: torch.Tensor, occ_threshold: float = 1e-6) -> np.ndarray:
    """Global per-channel mean / var / occupancy over a [B, C, H, W] BEV map.

    Returns a 1D float32 array of length 3*C for a single-sample batch
    (B is expected to be 1 at test time). Order: [mean(C), var(C), occ(C)].
    """
    assert feat.dim() == 4, f"expected [B,C,H,W], got {tuple(feat.shape)}"
    b = feat.shape[0]
    assert b == 1, f"caching assumes batch size 1 at test time, got B={b}"
    f = feat[0].float()                       # [C, H, W]
    mean = f.mean(dim=(1, 2))                  # [C]
    var = f.var(dim=(1, 2), unbiased=False)   # [C]
    occ = (f.abs() > occ_threshold).float().mean(dim=(1, 2))  # [C] fraction active
    return torch.cat([mean, var, occ]).detach().cpu().numpy().astype(np.float32)


# ----- real metadata channels ----------------------------------------------

# Candidate key paths tried in order; first hit wins. Tuples are nested lookups.
METAINFO_KEYS = {
    "token": [("token",), ("sample_idx",), ("lidar_path",)],
    "timestamp": [("timestamp",), ("lidar_points", "timestamp")],
    "ego2global": [("ego2global",), ("ego2global_translation",)],
    # calibration: lidar->ego or a cam2img matrix; we use translation norm as a proxy
    "lidar2ego": [("lidar2ego",), ("lidar_points", "lidar2ego"),
                  ("lidar2cam",), ("cam2img",)],
}


def _first_key(metainfo: dict, candidates: List[tuple]):
    """Return the value at the first candidate key-path that exists, else None."""
    for path in candidates:
        cur = metainfo
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _translation_from(mat_or_vec) -> Optional[np.ndarray]:
    """Extract a 3-translation from either a 4x4 matrix or a length-3 vector."""
    if mat_or_vec is None:
        return None
    arr = np.asarray(mat_or_vec, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr[:3, 3]
    if arr.shape in [(3,), (3, 1), (1, 3)]:
        return arr.reshape(3)
    return None


@dataclass
class MetadataState:
    """Carries per-scene state needed for inter-frame metadata channels
    (timestamp jitter and ego-motion both need the previous frame)."""
    prev_timestamp_s: Optional[float] = None
    prev_ego_translation: Optional[np.ndarray] = None

    def reset(self):
        self.prev_timestamp_s = None
        self.prev_ego_translation = None


def compute_real_channels(metainfo: dict, state: MetadataState) -> Tuple[np.ndarray, str]:
    """Compute the 3 real telemetry channels for one sample, updating state.

    Returns (vec3 float32, token_str). Missing fields -> NaN in that slot,
    never a crash.
    """
    token_val = _first_key(metainfo, METAINFO_KEYS["token"])
    token = str(token_val) if token_val is not None else "UNKNOWN"

    # --- timestamp jitter: deviation of inter-frame dt from nominal 0.5s ---
    ts_raw = _first_key(metainfo, METAINFO_KEYS["timestamp"])
    timestamp_jitter = np.nan
    ts_s = None
    if ts_raw is not None:
        ts_s = float(ts_raw) * NUSC_TIMESTAMP_UNIT
        if state.prev_timestamp_s is not None:
            dt = ts_s - state.prev_timestamp_s
            timestamp_jitter = abs(dt - NUSC_NOMINAL_DT_S)

    # --- calibration residual: translation-norm of the calibration matrix ---
    calib = _first_key(metainfo, METAINFO_KEYS["lidar2ego"])
    calib_trans = _translation_from(calib)
    calib_residual = float(np.linalg.norm(calib_trans)) if calib_trans is not None else np.nan

    # --- ego-motion magnitude: consecutive-frame ego translation distance ---
    ego = _first_key(metainfo, METAINFO_KEYS["ego2global"])
    ego_trans = _translation_from(ego)
    egomotion = np.nan
    if ego_trans is not None and state.prev_ego_translation is not None:
        egomotion = float(np.linalg.norm(ego_trans - state.prev_ego_translation))

    # update state for next frame
    if ts_s is not None:
        state.prev_timestamp_s = ts_s
    if ego_trans is not None:
        state.prev_ego_translation = ego_trans

    vec = np.array([timestamp_jitter, calib_residual, egomotion], dtype=np.float32)
    return vec, token


# ----- the cache orchestrator ----------------------------------------------

@dataclass
class StageACache:
    model: object
    corruption_type: str
    severity: int
    out_path: str
    cfg: TelemetryConfig = field(default_factory=TelemetryConfig)
    occ_threshold: float = 1e-6

    # populated at attach()
    _cam_feat: dict = field(default_factory=dict, init=False)
    _lidar_feat: dict = field(default_factory=dict, init=False)
    _handles: list = field(default_factory=list, init=False)

    # per-scene stateful generators / metadata trackers
    _telemetry_gen: Optional[TelemetryGenerator] = field(default=None, init=False)
    _meta_state: MetadataState = field(default_factory=MetadataState, init=False)
    _current_scene: Optional[str] = field(default=None, init=False)

    # accumulated records
    _records: Dict[str, dict] = field(default_factory=dict, init=False)

    # ----- hook plumbing -----
    def _cam_hook(self, module, inp, out):
        self._cam_feat["feat"] = out.detach()

    def _lidar_hook(self, module, inp, out):
        self._lidar_feat["feat"] = out.detach()

    def attach(self):
        m = self.model
        # unwrap DataParallel / MMDistributedDataParallel if present
        core = getattr(m, "module", m)
        vt = core.view_transform
        pme = core.pts_middle_encoder
        self._handles.append(vt.register_forward_hook(self._cam_hook))
        self._handles.append(pme.register_forward_hook(self._lidar_hook))
        # wrap the model's predict/forward so we can grab metainfo per sample
        self._wrap_predict(core)
        return self

    def _wrap_predict(self, core):
        """Wrap the detector's predict() to intercept data_samples (metainfo)
        after the forward pass has populated the feature hooks."""
        orig_predict = core.predict

        def wrapped_predict(batch_inputs_dict, batch_data_samples, **kw):
            result = orig_predict(batch_inputs_dict, batch_data_samples, **kw)
            # after predict, both feature hooks have fired for this sample
            for ds in batch_data_samples:
                self._record_sample(ds.metainfo)
            return result

        core.predict = wrapped_predict
        self._orig_predict = orig_predict
        self._core = core

    # ----- per-sample record assembly -----
    def _scene_of(self, metainfo: dict) -> str:
        """Best-effort scene identifier for resetting per-scene state.
        Falls back to a monotonic run of tokens if no scene token exists."""
        for key in ("scene_token", "scene_name", "scene_idx"):
            if key in metainfo:
                return str(metainfo[key])
        # nuScenes sample_data tokens don't carry scene directly here; if absent
        # we treat the whole run as one scene (constant severity anyway).
        return "RUN"

    def _record_sample(self, metainfo: dict):
        # resolve scene; reset stateful trackers at scene boundaries
        scene = self._scene_of(metainfo)
        if scene != self._current_scene:
            self._current_scene = scene
            self._telemetry_gen = TelemetryGenerator(
                self.corruption_type, severity=self.severity, cfg=self.cfg)
            self._meta_state.reset()

        # real metadata channels (also yields the canonical token)
        real_vec, token = compute_real_channels(metainfo, self._meta_state)

        # synthetic telemetry
        tel = self._telemetry_gen.step(token)
        synth_vec = tel["telemetry"]

        # pooled BEV stats from the two captured feature maps
        cam = self._cam_feat.get("feat")
        lidar = self._lidar_feat.get("feat")
        if cam is None or lidar is None:
            # a modality was absent this frame (e.g. missingcamera at high severity):
            # record NaNs of the right width so the row still exists
            cam_stats = np.full(3 * 80, np.nan, dtype=np.float32) if cam is None \
                else pool_feature_map(cam, self.occ_threshold)
            lidar_stats = np.full(3 * 256, np.nan, dtype=np.float32) if lidar is None \
                else pool_feature_map(lidar, self.occ_threshold)
        else:
            cam_stats = pool_feature_map(cam, self.occ_threshold)
            lidar_stats = pool_feature_map(lidar, self.occ_threshold)

        self._records[token] = {
            "pooled_cam": cam_stats,          # 240
            "pooled_lidar": lidar_stats,      # 768
            "telemetry_synth": synth_vec,     # 4
            "telemetry_real": real_vec,       # 3
            "reliability_target": tel["reliability_target"],
            "true_severity": np.int64(self.severity),
            "is_decoy": bool(tel["is_decoy"]),
        }
        # clear captured feats so a dropped modality next frame is detectable
        self._cam_feat.clear()
        self._lidar_feat.clear()

    # ----- teardown / save -----
    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if getattr(self, "_core", None) is not None:
            self._core.predict = self._orig_predict

    def save(self):
        self.detach()
        tokens = list(self._records.keys())
        if not tokens:
            raise RuntimeError("no records captured -- did the hooks attach and predict run?")

        pooled_cam = np.stack([self._records[t]["pooled_cam"] for t in tokens])
        pooled_lidar = np.stack([self._records[t]["pooled_lidar"] for t in tokens])
        tel_synth = np.stack([self._records[t]["telemetry_synth"] for t in tokens])
        tel_real = np.stack([self._records[t]["telemetry_real"] for t in tokens])
        targets = np.array([self._records[t]["reliability_target"] for t in tokens], dtype=np.float32)
        sev = np.array([self._records[t]["true_severity"] for t in tokens], dtype=np.int64)
        decoy = np.array([self._records[t]["is_decoy"] for t in tokens], dtype=bool)

        np.savez_compressed(
            self.out_path,
            tokens=np.array(tokens),
            pooled_cam=pooled_cam,            # [N, 240]
            pooled_lidar=pooled_lidar,        # [N, 768]
            telemetry_synth=tel_synth,        # [N, 4]
            telemetry_real=tel_real,          # [N, 3]
            reliability_target=targets,       # [N]
            true_severity=sev,                # [N]
            is_decoy=decoy,                   # [N]
            synth_channel_names=np.array(SYNTH_CHANNELS),
            real_channel_names=np.array(REAL_CHANNELS),
            corruption_type=np.array(self.corruption_type),
        )
        print(f"[StageACache] saved {len(tokens)} records -> {self.out_path}")
        return self.out_path


# ----- one-time metainfo probe (run before a real caching pass) -------------

def probe_metainfo(runner, n: int = 2):
    """Print the metainfo keys of the first n test samples so you can confirm
    the real key names on your MMDet3D version before committing to a full run.
    Call after runner is built; does not run inference."""
    ds = runner.test_dataloader.dataset
    for i in range(min(n, len(ds))):
        item = ds[i]
        ds_obj = item["data_samples"]
        mi = ds_obj.metainfo if hasattr(ds_obj, "metainfo") else ds_obj
        print(f"--- sample {i} metainfo keys ---")
        print(sorted(mi.keys()) if isinstance(mi, dict) else type(mi))
        for k in ("token", "timestamp", "ego2global", "lidar2ego", "lidar2cam",
                  "cam2img", "scene_token", "sample_idx"):
            if isinstance(mi, dict) and k in mi:
                v = mi[k]
                shape = getattr(v, "shape", None)
                print(f"  {k}: type={type(v).__name__} shape={shape} "
                      f"value={str(v)[:60]}")