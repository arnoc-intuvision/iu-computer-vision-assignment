"""Scene-wise split, feature assembly and scaling for the LCRE training set."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Reuse the exact scene-boundary heuristic and constants from Stage A.
import sys
sys.path.insert(0, "/workspace/mmdetection3d/tools")
from cv_assign_stage_a_cache_hook import (  # noqa: E402
    load_meta_table,
    NUSC_NOMINAL_DT_S,
)

# Constants -- must match Stage C inference exactly.

CACHE_DIR = "/workspace/cache"
SCORES_DIR = "/workspace/scores"
INFO_PKL_TRAIN = "/workspace/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl"
INFO_PKL_VAL = "/workspace/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl"

SPLIT_SEED = 20260101  # matches TelemetryConfig.master_seed
NOMINAL_DT = NUSC_NOMINAL_DT_S  # 0.5 s

# The 10 train conditions (cache basenames without .npz).
TRAIN_CONDITIONS: List[str] = [
    "clean_train_sev0",
    "beamsreducing_train_sev1",
    "beamsreducing_train_sev2",
    "beamsreducing_train_sev3",
    "missingcamera_train_sev1",
    "missingcamera_train_sev2",
    "missingcamera_train_sev3",
    "motionblur_train_sev1",
    "motionblur_train_sev2",
    "motionblur_train_sev3",
]

# The 10 val conditions.
VAL_CONDITIONS: List[str] = [
    "clean_val_sev0",
    "beamsreducing_val_sev1",
    "beamsreducing_val_sev2",
    "beamsreducing_val_sev3",
    "missingcamera_val_sev1",
    "missingcamera_val_sev2",
    "missingcamera_val_sev3",
    "motionblur_val_sev1",
    "motionblur_val_sev2",
    "motionblur_val_sev3",
]

# Feature layout offsets (0-indexed into the 1015-dim pre-indicator vector).
CAM_OFFSET = 0
CAM_WIDTH = 240
LIDAR_OFFSET = CAM_WIDTH            # 240
LIDAR_WIDTH = 768
SYNTH_OFFSET = CAM_WIDTH + LIDAR_WIDTH  # 1008
SYNTH_WIDTH = 4
REAL_OFFSET = SYNTH_OFFSET + SYNTH_WIDTH  # 1012
REAL_WIDTH = 3

BASE_FEATURE_DIM = CAM_WIDTH + LIDAR_WIDTH + SYNTH_WIDTH + REAL_WIDTH  # 1015

# telemetry_real channels that are NaN on the first frame of every scene.
# Indices into the 3-dim telemetry_real block (meta_timestamp_jitter=0,
# meta_egomotion_magnitude=2).
NAN_REAL_CHANNELS = [0, 2]
N_INDICATORS = len(NAN_REAL_CHANNELS)  # 2

FULL_INPUT_DIM = BASE_FEATURE_DIM + N_INDICATORS  # 1017

# Column index of the two indicator columns in the full 1017-dim vector.
INDICATOR_COLS = [BASE_FEATURE_DIM + i for i in range(N_INDICATORS)]  # [1015, 1016]

# Ablation column subsets (into the full 1017-dim vector).
ABLATION_BEV_COLS = list(range(CAM_OFFSET, SYNTH_OFFSET)) + INDICATOR_COLS          # 1010
ABLATION_TELEMETRY_COLS = list(range(SYNTH_OFFSET, BASE_FEATURE_DIM)) + INDICATOR_COLS  # 9
ABLATION_FULL_COLS = list(range(FULL_INPUT_DIM))                                      # 1017

# Decoy-shortcut mitigation variants (failure protocol step b).
# Synthetic telemetry channels are the ONLY ones that carry the decoy-swapped
# severity (real channels are genuine metadata, never decoy-swapped). Channels
# with NO lag (A=col 1008, C=col 1010) directly reflect the current frame's
# decoy value -> most circular. Lagged channels (B=1009 lag1, D=1011 lag2)
# report prior frames' values -> less directly tied to the current decoy swap.
SYNTH_NO_LAG_COLS = [SYNTH_OFFSET + 0, SYNTH_OFFSET + 2]   # A, C (most circular)
SYNTH_LAG_COLS = [SYNTH_OFFSET + 1, SYNTH_OFFSET + 3]       # B, D (lagged)

# Drop the 2 most-circular (no-lag) synthetic channels, keep lagged synth + real.
DROP_NO_LAG_SYNTH_COLS = (
    [c for c in range(FULL_INPUT_DIM)
     if c not in SYNTH_NO_LAG_COLS]
)  # 1015

# Drop ALL synthetic channels -> BEV + real telemetry only (no decoy-swapped input).
BEV_PLUS_REAL_COLS = (
    list(range(CAM_OFFSET, SYNTH_OFFSET))           # BEV (1008)
    + list(range(REAL_OFFSET, BASE_FEATURE_DIM))     # real telemetry (3)
    + INDICATOR_COLS                                  # 2 indicators
)  # 1013

ABLATION_SPECS: Dict[str, List[int]] = {
    "full": ABLATION_FULL_COLS,
    "bev": ABLATION_BEV_COLS,
    "telemetry": ABLATION_TELEMETRY_COLS,
    "drop_nolag": DROP_NO_LAG_SYNTH_COLS,     # drop 2 no-lag synth channels
    "bev_real": BEV_PLUS_REAL_COLS,            # BEV + real telemetry, no synth
}


# - scene-wise split

def recover_scenes(info_pkl: str) -> Dict[int, str]:
    """Return sample_idx -> scene_id using the timestamp-gap heuristic."""
    meta_table = load_meta_table(info_pkl)
    scene_map: Dict[int, str] = {}
    for idx, meta in meta_table.items():
        scene_map[idx] = meta.scene_prefix
    return scene_map


def build_scene_split(
    info_pkl: str = INFO_PKL_TRAIN,
    seed: int = SPLIT_SEED,
    n_heldout_scenes: int = 25,
    out_path: str = os.path.join(SCORES_DIR, "scene_split.json"),
) -> Dict:
    """Build and persist the scene-wise fit/heldout split."""
    scene_map = recover_scenes(info_pkl)

    # group sample_idx by scene
    scenes: Dict[str, List[int]] = {}
    for idx, scene in scene_map.items():
        scenes.setdefault(scene, []).append(int(idx))

    scene_ids = sorted(scenes.keys())
    n_total = len(scene_ids)

    # cap n_heldout if there are fewer scenes than requested
    n_heldout = min(n_heldout_scenes, n_total)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    heldout_scene_ids = [scene_ids[p] for p in perm[:n_heldout]]
    fit_scene_ids = [scene_ids[p] for p in perm[n_heldout:]]

    heldout = sorted(int(i) for s in heldout_scene_ids for i in scenes[s])
    fit = sorted(int(i) for s in fit_scene_ids for i in scenes[s])

    result = {
        "fit": fit,
        "heldout": heldout,
        "seed": seed,
        "n_scenes_total": n_total,
        "n_heldout": n_heldout,
        "heldout_scene_ids": heldout_scene_ids,
        "fit_scene_ids": fit_scene_ids,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    return result


def load_scene_split(path: str = os.path.join(SCORES_DIR, "scene_split.json")) -> Dict:
    with open(path) as f:
        return json.load(f)


def verify_scene_split(split: Dict, info_pkl: str = INFO_PKL_TRAIN) -> None:
    """Assert no sample_idx appears in both splits and the fraction is ~18%."""
    fit = set(split["fit"])
    heldout = set(split["heldout"])
    overlap = fit & heldout
    assert not overlap, f"scene split leakage: {len(overlap)} sample_idx in both"
    total = len(fit) + len(heldout)
    frac = len(heldout) / total
    assert 0.10 < frac < 0.30, f"heldout fraction {frac:.3f} outside [0.10, 0.30]"
    # reproducibility: scene count should match info pkl
    scene_map = recover_scenes(info_pkl)
    assert total == len(scene_map), f"split total {total} != info pkl {len(scene_map)}"
    print(f"[scene_split] OK: {len(fit)} fit / {len(heldout)} heldout "
          f"({frac:.1%}), {split['n_scenes_total']} scenes, "
          f"{split['n_heldout']} heldout scenes")


# - feature assembly, NaN handling, scaler

def _load_cache(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def assemble_raw_features(cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate [pooled_cam | pooled_lidar | telemetry_synth | telemetry_real]."""
    cam = cache["pooled_cam"].astype(np.float32)
    lidar = cache["pooled_lidar"].astype(np.float32)
    synth = cache["telemetry_synth"].astype(np.float32)
    real = cache["telemetry_real"].astype(np.float32)
    return np.concatenate([cam, lidar, synth, real], axis=1)


@dataclass
class Scaler:
    """StandardScaler-like with NaN-aware imputation for indicator channels."""

    mean_: np.ndarray
    scale_: np.ndarray
    n_features_in_: int
    indicator_cols: List[int]
    nan_real_channels: List[int]
    impute_values: np.ndarray  # median per NaN real channel, for imputation

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply imputation + standardisation. X is (N, 1017)."""
        X = X.copy()
        # impute NaN in the real channels (before standardisation)
        for i, ch in enumerate(self.nan_real_channels):
            col = REAL_OFFSET + ch
            mask = np.isnan(X[:, col])
            X[mask, col] = self.impute_values[i]
        # standardise (indicator cols pass through)
        X = (X - self.mean_) / self.scale_
        return X

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            mean_=self.mean_.astype(np.float64),
            scale_=self.scale_.astype(np.float64),
            n_features_in_=np.int64(self.n_features_in_),
            indicator_cols=np.array(self.indicator_cols, dtype=np.int64),
            nan_real_channels=np.array(self.nan_real_channels, dtype=np.int64),
            impute_values=self.impute_values.astype(np.float64),
        )

    @classmethod
    def load(cls, path: str) -> "Scaler":
        d = np.load(path, allow_pickle=True)
        return cls(
            mean_=d["mean_"].astype(np.float64),
            scale_=d["scale_"].astype(np.float64),
            n_features_in_=int(d["n_features_in_"]),
            indicator_cols=[int(x) for x in d["indicator_cols"]],
            nan_real_channels=[int(x) for x in d["nan_real_channels"]],
            impute_values=d["impute_values"].astype(np.float64),
        )


def fit_scaler(X_fit: np.ndarray) -> Scaler:
    """Fit a StandardScaler on the fit split (1017-dim, with NaN indicators)."""
    assert X_fit.shape[1] == FULL_INPUT_DIM, (
        f"expected {FULL_INPUT_DIM} dims, got {X_fit.shape[1]}"
    )

    X = X_fit.copy()

    # compute imputation medians from non-NaN values in the fit split
    impute_values = np.zeros(len(NAN_REAL_CHANNELS), dtype=np.float64)
    for i, ch in enumerate(NAN_REAL_CHANNELS):
        col = REAL_OFFSET + ch
        vals = X[:, col]
        valid = vals[~np.isnan(vals)]
        impute_values[i] = float(np.median(valid)) if len(valid) else 0.0
        X[np.isnan(X[:, col]), col] = impute_values[i]

    # standardise per-dimension, EXCEPT indicator columns (pass through)
    mean_ = np.zeros(FULL_INPUT_DIM, dtype=np.float64)
    scale_ = np.ones(FULL_INPUT_DIM, dtype=np.float64)
    non_indicator = [c for c in range(FULL_INPUT_DIM) if c not in INDICATOR_COLS]
    mean_[non_indicator] = X[:, non_indicator].mean(axis=0)
    std = X[:, non_indicator].std(axis=0)
    # guard against zero std (constant columns)
    scale_[non_indicator] = np.where(std > 1e-8, std, 1.0)

    return Scaler(
        mean_=mean_,
        scale_=scale_,
        n_features_in_=FULL_INPUT_DIM,
        indicator_cols=list(INDICATOR_COLS),
        nan_real_channels=list(NAN_REAL_CHANNELS),
        impute_values=impute_values,
    )


def add_indicators(X_raw: np.ndarray) -> np.ndarray:
    """Append 2 binary NaN-indicator columns to the 1015-dim raw vector."""
    n = X_raw.shape[0]
    indicators = np.zeros((n, N_INDICATORS), dtype=np.float32)
    for i, ch in enumerate(NAN_REAL_CHANNELS):
        col = REAL_OFFSET + ch
        indicators[:, i] = np.isnan(X_raw[:, col]).astype(np.float32)
    return np.concatenate([X_raw, indicators], axis=1)


def assemble_features(cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Full 1017-dim feature: raw 1015 + 2 NaN indicators (NaN preserved)."""
    X_raw = assemble_raw_features(cache)
    return add_indicators(X_raw)


# Cache loading + stacking across conditions

@dataclass
class StackedData:
    """All 10 train caches stacked into flat arrays."""
    X: np.ndarray              # (N_total, 1017)
    targets: np.ndarray        # (N_total,) reliability_target
    true_severity: np.ndarray  # (N_total,)
    is_decoy: np.ndarray       # (N_total,) bool
    tokens: np.ndarray         # (N_total,)
    sample_idx: np.ndarray     # (N_total,)
    corruption_type: np.ndarray  # (N_total,) str
    condition: np.ndarray      # (N_total,) str cache basename


def load_all_train_caches(
    cache_dir: str = CACHE_DIR,
    conditions: Sequence[str] = TRAIN_CONDITIONS,
) -> StackedData:
    """Load and stack all 10 train caches into a single StackedData."""
    chunks: List[dict] = []
    for cond in conditions:
        path = os.path.join(cache_dir, cond + ".npz")
        cache = _load_cache(path)
        X = assemble_features(cache)
        chunks.append({
            "X": X,
            "targets": cache["reliability_target"].astype(np.float32),
            "true_severity": cache["true_severity"].astype(np.int64),
            "is_decoy": cache["is_decoy"].astype(bool),
            "tokens": cache["tokens"].astype(str),
            "sample_idx": cache["sample_idx"].astype(np.int64),
            "corruption_type": np.array(
                [str(cache["corruption_type"])] * len(cache["tokens"])
            ),
            "condition": np.array([cond] * len(cache["tokens"])),
        })

    def _cat(key):
        return np.concatenate([c[key] for c in chunks], axis=0)

    return StackedData(
        X=_cat("X"),
        targets=_cat("targets"),
        true_severity=_cat("true_severity"),
        is_decoy=_cat("is_decoy"),
        tokens=_cat("tokens"),
        sample_idx=_cat("sample_idx"),
        corruption_type=_cat("corruption_type"),
        condition=_cat("condition"),
    )


def split_by_scene(
    data: StackedData, scene_split: Dict
) -> Tuple[StackedData, StackedData]:
    """Partition StackedData into fit / heldout using the scene split."""
    fit_idx = set(scene_split["fit"])
    heldout_idx = set(scene_split["heldout"])

    fit_mask = np.array([int(s) in fit_idx for s in data.sample_idx], dtype=bool)
    held_mask = np.array([int(s) in heldout_idx for s in data.sample_idx], dtype=bool)

    # every sample_idx must be in exactly one split
    n_both = int((fit_mask & held_mask).sum())
    n_neither = int((~(fit_mask | held_mask)).sum())
    assert n_both == 0, f"{n_both} samples in both splits"
    assert n_neither == 0, f"{n_neither} samples in neither split"

    def _select(mask):
        return StackedData(
            X=data.X[mask],
            targets=data.targets[mask],
            true_severity=data.true_severity[mask],
            is_decoy=data.is_decoy[mask],
            tokens=data.tokens[mask],
            sample_idx=data.sample_idx[mask],
            corruption_type=data.corruption_type[mask],
            condition=data.condition[mask],
        )

    return _select(fit_mask), _select(held_mask)


# Torch Dataset

class LCREDataset(Dataset):
    """torch Dataset over a StackedData block, with lazy scaler transform."""

    def __init__(
        self,
        data: StackedData,
        scaler: Optional[Scaler] = None,
        col_subset: Optional[List[int]] = None,
    ) -> None:
        self.scaler = scaler
        self.col_subset = col_subset
        # pre-transform if scaler is available (transform is cheap, do once)
        if scaler is not None:
            self.X = scaler.transform(data.X).astype(np.float32)
        else:
            self.X = data.X.astype(np.float32)
        if col_subset is not None:
            self.X = self.X[:, col_subset]
        self.targets = data.targets.astype(np.float32)
        self.true_severity = data.true_severity.astype(np.int64)
        self.is_decoy = data.is_decoy.astype(bool)
        self.tokens = data.tokens
        self.sample_idx = data.sample_idx
        self.corruption_type = data.corruption_type

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple:
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.targets[idx]),
            torch.tensor(self.true_severity[idx], dtype=torch.long),
            torch.tensor(self.is_decoy[idx], dtype=torch.bool),
            self.tokens[idx],
            int(self.sample_idx[idx]),
            str(self.corruption_type[idx]),
        )


# Convenience: build everything end-to-end

def prepare_data(
    use_existing_split: bool = True,
) -> Tuple[StackedData, StackedData, Scaler, Dict]:
    """Load caches, build/verify split, fit scaler."""
    split_path = os.path.join(SCORES_DIR, "scene_split.json")
    if use_existing_split and os.path.exists(split_path):
        scene_split = load_scene_split(split_path)
    else:
        scene_split = build_scene_split(out_path=split_path)
    verify_scene_split(scene_split)

    print("[data] loading 10 train caches ...")
    all_data = load_all_train_caches()
    print(f"[data] total samples: {len(all_data.targets)}")

    fit_data, heldout_data = split_by_scene(all_data, scene_split)
    print(f"[data] fit={len(fit_data.targets)}  heldout={len(heldout_data.targets)}")

    # fit scaler on the fit split only
    scaler = fit_scaler(fit_data.X)

    # verify no NaN remains after transform
    Xt = scaler.transform(fit_data.X)
    assert not np.isnan(Xt).any(), "NaN remains after scaler transform (fit)"
    Xh = scaler.transform(heldout_data.X)
    assert not np.isnan(Xh).any(), "NaN remains after scaler transform (heldout)"

    # verify indicators are binary
    for c in INDICATOR_COLS:
        vals = set(np.unique(fit_data.X[:, c]).tolist())
        assert vals <= {0.0, 1.0}, f"indicator col {c} not binary: {vals}"

    return fit_data, heldout_data, scaler, scene_split


def save_scaler(scaler: Scaler, path: str = os.path.join(SCORES_DIR, "lcre_scaler.npz")) -> None:
    scaler.save(path)
    print(f"[scaler] saved -> {path}")
