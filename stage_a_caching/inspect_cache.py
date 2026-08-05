"""
Inspect and validate a Stage A cache .npz before committing to all seven
caching passes.

Usage:
    python inspect_cache.py /workspace/cache/beamsreducing_sev2.npz

Checks, in order (fails loud on the first structural problem):
  1. file loads and has all expected arrays
  2. record count > 0 and all arrays share the same N
  3. pooled_cam width == 240, pooled_lidar width == 768
  4. telemetry_real is not entirely NaN  (metadata-key / predict-wrap sanity)
  5. reports NaN rates per block so a dropped modality / missing key is visible
  6. reports decoy fraction, severity/target consistency, and channel ranges

Then prints a small human-readable summary so you can eyeball that the
numbers look like real telemetry, not garbage.
"""

import sys
import numpy as np

EXPECTED_ARRAYS = [
    "tokens", "pooled_cam", "pooled_lidar", "telemetry_synth",
    "telemetry_real", "reliability_target", "true_severity", "is_decoy",
    "synth_channel_names", "real_channel_names", "corruption_type",
]

CAM_WIDTH = 240      # 80 channels x (mean, var, occupancy)
LIDAR_WIDTH = 768    # 256 channels x (mean, var, occupancy)


def fail(msg):
    print(f"\n  ✗ FAIL: {msg}\n")
    sys.exit(1)


def ok(msg):
    print(f"  ✓ {msg}")


def nan_rate(arr):
    return float(np.isnan(arr).mean()) if arr.size else float("nan")


def main(path):
    print(f"\n=== inspecting {path} ===\n")

    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        fail(f"could not load .npz: {e}")

    # 1. structural completeness
    missing = [k for k in EXPECTED_ARRAYS if k not in d.files]
    if missing:
        fail(f"missing arrays: {missing}")
    ok(f"all {len(EXPECTED_ARRAYS)} expected arrays present")

    tokens = d["tokens"]
    n = len(tokens)

    # 2. non-empty and consistent N
    if n == 0:
        fail("zero records -- hooks likely didn't fire or predict() wasn't wrapped")
    per_row = ["pooled_cam", "pooled_lidar", "telemetry_synth",
               "telemetry_real", "reliability_target", "true_severity", "is_decoy"]
    for k in per_row:
        if len(d[k]) != n:
            fail(f"{k} has {len(d[k])} rows but tokens has {n}")
    ok(f"{n} records, all per-row arrays aligned")

    # unique tokens?
    n_unique = len(set(tokens.tolist()))
    if n_unique != n:
        print(f"  ⚠ WARNING: {n - n_unique} duplicate tokens "
              f"({n_unique} unique of {n}) -- check the caching loop iterates once/token")
    else:
        ok(f"all {n} tokens unique")

    # 3. pooled widths
    if d["pooled_cam"].shape[1] != CAM_WIDTH:
        fail(f"pooled_cam width {d['pooled_cam'].shape[1]} != {CAM_WIDTH}")
    if d["pooled_lidar"].shape[1] != LIDAR_WIDTH:
        fail(f"pooled_lidar width {d['pooled_lidar'].shape[1]} != {LIDAR_WIDTH}")
    ok(f"pooled widths correct (cam {CAM_WIDTH}, lidar {LIDAR_WIDTH}, "
       f"total {CAM_WIDTH + LIDAR_WIDTH})")

    # 4. real telemetry not entirely NaN (the key sanity for metadata plumbing)
    real = d["telemetry_real"]
    if np.isnan(real).all():
        fail("telemetry_real is ALL NaN -- metadata keys wrong "
             "(run probe_metainfo and fix METAINFO_KEYS) or predict() not wrapped")
    ok("telemetry_real has real values (metadata plumbing works)")

    # 5. NaN rates per block -- diagnostic, not fatal
    print("\n--- NaN rates (per block) ---")
    print(f"  pooled_cam        : {nan_rate(d['pooled_cam']):.3f}")
    print(f"  pooled_lidar      : {nan_rate(d['pooled_lidar']):.3f}")
    print(f"  telemetry_synth   : {nan_rate(d['telemetry_synth']):.3f}")
    print(f"  telemetry_real    : {nan_rate(real):.3f}")
    real_names = [str(x) for x in d["real_channel_names"]]
    for i, name in enumerate(real_names):
        print(f"    {name:28s}: {nan_rate(real[:, i]):.3f}")
    # first frame of each scene legitimately has NaN jitter/egomotion; a high
    # rate on those two is expected & fine. A high rate on calib_residual is NOT.

    # 6. consistency + ranges
    print("\n--- consistency & ranges ---")
    sev = d["true_severity"]
    tgt = d["reliability_target"]
    expected_tgt = 1.0 - sev.astype(np.float32) / 3.0
    if not np.allclose(tgt, expected_tgt, atol=1e-5):
        print("  ⚠ WARNING: reliability_target != 1 - severity/3 for some rows")
    else:
        ok("reliability_target == 1 - severity/3 for all rows")

    print(f"  severity values   : {sorted(set(sev.tolist()))}")
    print(f"  corruption_type   : {d['corruption_type']}")
    print(f"  decoy fraction    : {d['is_decoy'].mean():.4f}")

    synth = d["telemetry_synth"]
    synth_names = [str(x) for x in d["synth_channel_names"]]
    print("\n  synthetic telemetry channel ranges:")
    for i, name in enumerate(synth_names):
        col = synth[:, i]
        print(f"    {name:30s}: min={col.min():+.3f} "
              f"mean={col.mean():+.3f} max={col.max():+.3f}")

    print("\n  pooled occupancy sanity (should be in [0,1]):")
    cam_occ = d["pooled_cam"][:, 160:240]      # last 80 = cam occupancy block
    lidar_occ = d["pooled_lidar"][:, 512:768]  # last 256 = lidar occupancy block
    print(f"    cam occupancy    : mean={np.nanmean(cam_occ):.4f} "
          f"max={np.nanmax(cam_occ):.4f}")
    print(f"    lidar occupancy  : mean={np.nanmean(lidar_occ):.4f} "
          f"max={np.nanmax(lidar_occ):.4f}")

    print(f"\n  first 3 tokens: {[str(t) for t in tokens[:3]]}")
    print("\n=== inspection complete ===")
    print("If all checks passed and ranges look sane, you're clear to run the")
    print("remaining caching passes.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python inspect_cache.py <path_to_cache.npz>")
        sys.exit(1)
    main(sys.argv[1])