"""
Hardware-telemetry generator for the reliability-adaptive Kalman tracking study.

Produces a per-frame synthetic hardware-telemetry vector for camera and LiDAR,
derived from the MultiCorrupt (corruption_type, severity) label but deliberately
NOT trivially invertible back to that label, per the anti-circularity discipline
in the Empirical Study Plan (Sec. 4.3).

Design summary
--------------
Each telemetry channel is severity-driven through a DISTINCT noisy monotonic
map g(.) -- not identity, and not the same shape reused -- so that no linear
readout of the vector recovers severity, and so that the four synthetic channels
disagree at the extremes rather than being monotone transforms of one another.

Three anti-leakage mechanisms (Sec. 4.3):
  (i)   per-channel noisy monotonic maps g(.) (saturating / sqrt / quadratic / log)
  (ii)  1-2 frame lag on some channels (hardware self-report latency)
  (iii) a held-out DECOY fraction of frames carrying mismatched telemetry
        (corrupted frame -> nominal-looking telemetry, and vice versa),
        forcing the Stage-B estimator to also use the BEV feature evidence.

Real metadata-anchored channels (timestamp jitter, calibration residual,
ego-motion magnitude) are handled SEPARATELY at hook time from nuScenes
metadata -- they are never severity-derived and never decoy-swapped, so they
are not produced here. This module emits only the synthetic, severity-derived
channels; the hook concatenates the real ones alongside.

Channels emitted (synthetic, severity-derived)
----------------------------------------------
  LiDAR:
    A  point_return_rate_dev   g(s) = 1 - exp(-s / tau)      (saturating), no lag
    B  beam_health_index       g(s) = sqrt(s / s_max)         (sqrt),       lag 1
  Camera:
    C  exposure_gain_fault      g(s) = (s / s_max) ** 2        (quadratic),  no lag
    D  dynamic_range_degrad     g(s) = log1p(s) / log1p(s_max)(log),        lag 2

All maps return ~0 at s=0 (clean) and ~1 at s=s_max (worst), then have
independent Gaussian noise added. Values are NOT clipped to [0,1]: real
telemetry overshoots, and clipping would reintroduce a hard, invertible ceiling.

Reproducibility
---------------
Noise, lag buffers, and decoy selection are all driven by a deterministic
per-frame seed derived from sample_token, so a re-run reproduces byte-identical
telemetry -- important for the Sec. 6 boundary checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# MultiCorrupt severity scheme: 0 = clean, worst = 3 (verified against repo).
SEVERITY_MAX = 3

# Channel order is FIXED and must match the Stage-B feature layout.
SYNTH_CHANNELS = (
    "lidar_point_return_rate_dev",   # A
    "lidar_beam_health_index",       # B
    "cam_exposure_gain_fault",       # C
    "cam_dynamic_range_degrad",      # D
)


@dataclass
class TelemetryConfig:
    severity_max: int = SEVERITY_MAX

    # g(.) shape constant for the saturating LiDAR channel A.
    tau: float = 1.5

    # Per-channel independent-noise std (added AFTER the monotonic map).
    # Different magnitudes per channel so noise floors differ too.
    noise_std_A: float = 0.06
    noise_std_B: float = 0.09
    noise_std_C: float = 0.05
    noise_std_D: float = 0.08

    # Frame lag per channel (hardware self-report latency), in frames.
    lag_A: int = 0
    lag_B: int = 1
    lag_C: int = 0
    lag_D: int = 2

    # Fraction of frames that are decoys (mismatched telemetry/label).
    decoy_fraction: float = 0.08

    # Master seed folded into every per-frame seed for global reproducibility.
    master_seed: int = 20260101


def frame_rng(sample_token: str, master_seed: int, salt: str) -> np.random.Generator:
    """Deterministic per-frame RNG keyed by sample_token (+ a salt string).

    Using a hash of the token means the same frame always draws the same noise
    regardless of iteration order, which keeps the whole pipeline reproducible.
    """
    h = hashlib.sha256(f"{master_seed}:{salt}:{sample_token}".encode()).digest()
    seed = int.from_bytes(h[:8], "little")
    return np.random.default_rng(seed)


def is_decoy(sample_token: str, cfg: TelemetryConfig) -> bool:
    """Deterministically mark ~decoy_fraction of frames as decoys.

    Independent of severity so the decoy set is fixed per frame across all runs.
    """
    rng = frame_rng(sample_token, cfg.master_seed, salt="decoy")
    return bool(rng.random() < cfg.decoy_fraction)


# ---- the four distinct monotonic maps g(.) ---------------------------------

def g_saturating(s: float, cfg: TelemetryConfig) -> float:
    # A: 1 - exp(-s/tau); ~0 at s=0, approaches (but never reaches) ~1.
    return 1.0 - np.exp(-s / cfg.tau)


def g_sqrt(s: float, cfg: TelemetryConfig) -> float:
    # B: sqrt(s / s_max); concave, rises fast early.
    return np.sqrt(s / cfg.severity_max)


def g_quadratic(s: float, cfg: TelemetryConfig) -> float:
    # C: (s / s_max)^2; convex, rises slow early then steep.
    return (s / cfg.severity_max) ** 2


def g_log(s: float, cfg: TelemetryConfig) -> float:
    # D: log1p(s)/log1p(s_max); concave, normalised to ~1 at s_max.
    return np.log1p(s) / np.log1p(cfg.severity_max)




@dataclass
class TelemetryGenerator:
    """Stateful per-scene telemetry generator.

    Stateful because the lag mechanism needs the severity history of prior
    frames in the SAME scene. Instantiate one generator per scene, then call
    `step()` once per frame in temporal order.
    """
    corruption_type: str
    severity: int                       # the injected severity for THIS run (0..3)
    cfg: TelemetryConfig = field(default_factory=TelemetryConfig)

    # severity history for this scene, newest last; index -1 is current frame
    _severity_history: list = field(default_factory=list, init=False)

    def _effective_severity_for_channel(self, lag: int) -> float:
        """Return the severity value a lagged channel should report this frame:
        the severity from `lag` frames ago, or the oldest available if the scene
        just started (self-report hasn't 'caught up' yet)."""
        if lag == 0 or len(self._severity_history) <= lag:
            # not enough history yet -> report the earliest known severity
            idx = 0 if len(self._severity_history) <= lag else -1
            return float(self._severity_history[idx]) if self._severity_history else 0.0
        return float(self._severity_history[-1 - lag])

    def step(self, sample_token: str) -> dict:
        """Produce one frame's telemetry.

        Returns a dict with the synthetic channel vector, the decoy flag, and
        the reliability target -- ready for the hook to merge with pooled BEV
        stats and real metadata channels into the per-token .npz.
        """
        cfg = self.cfg

        # true severity for this frame (constant across a single-corruption run)
        true_sev = float(self.severity)

        # decide decoy status, then compute the severity the telemetry PRETENDS
        decoy = is_decoy(sample_token, cfg)
        if decoy:
            drng = frame_rng(sample_token, cfg.master_seed, salt="decoy_value")
            if true_sev > 0:
                reported_sev = 0.0                      # degraded frame, nominal telemetry
            else:
                reported_sev = float(drng.integers(1, cfg.severity_max + 1))  # clean frame, faulty telemetry
        else:
            reported_sev = true_sev

        # push the REPORTED severity into history so lag operates on what the
        # hardware would actually have been self-reporting frame to frame
        self._severity_history.append(reported_sev)

        rng = frame_rng(sample_token, cfg.master_seed, salt="noise")

        # resolve per-channel lagged effective severities
        sev_A = self._effective_severity_for_channel(cfg.lag_A)
        sev_B = self._effective_severity_for_channel(cfg.lag_B)
        sev_C = self._effective_severity_for_channel(cfg.lag_C)
        sev_D = self._effective_severity_for_channel(cfg.lag_D)

        a = g_saturating(sev_A, cfg) + rng.normal(0.0, cfg.noise_std_A)
        b = g_sqrt(sev_B, cfg)       + rng.normal(0.0, cfg.noise_std_B)
        c = g_quadratic(sev_C, cfg)  + rng.normal(0.0, cfg.noise_std_C)
        d = g_log(sev_D, cfg)        + rng.normal(0.0, cfg.noise_std_D)
        vec = np.array([a, b, c, d], dtype=np.float32)

        # reliability TARGET is derived from the TRUE severity, never the decoy
        # value -- the label the estimator must learn is the ground truth, and
        # the decoy frames are precisely where telemetry alone would mislead it.
        reliability_target = np.float32(1.0 - true_sev / cfg.severity_max)

        return {
            "telemetry": vec,                       # shape (4,), channel order = SYNTH_CHANNELS
            "channel_names": SYNTH_CHANNELS,
            "reliability_target": reliability_target,
            "true_severity": np.int64(self.severity),
            "corruption_type": self.corruption_type,
            "is_decoy": bool(decoy),
        }