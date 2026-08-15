"""Run a trained LCRE over the validation caches and write per-condition score files."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lcre_model import LCRE
from lcre_dataset import (
    ABLATION_SPECS,
    CACHE_DIR,
    FULL_INPUT_DIM,
    INDICATOR_COLS,
    REAL_OFFSET,
    SCORES_DIR,
    Scaler,
    VAL_CONDITIONS,
    assemble_features,
    _load_cache,
)


def parse_condition(cond: str) -> tuple:
    """Parse e.g. 'beamsreducing_val_sev2' -> ('beamsreducing', 2)."""
    parts = cond.split("_")
    # corruption may be multi-word? No — all are single words here.
    # format: <corruption>_val_sev<N>
    corruption = parts[0]
    sev = int(parts[-1].replace("sev", ""))
    return corruption, sev


@torch.no_grad()
def predict_cache(
    model, scaler, cache_path: str, col_subset, device
) -> Dict[str, np.ndarray]:
    """Run LCRE over one cache, return dict with s_t and metadata."""
    cache = _load_cache(cache_path)
    X = assemble_features(cache)  # (N, 1017)
    Xt = scaler.transform(X).astype(np.float32)
    if col_subset is not None:
        Xt = Xt[:, col_subset]

    x = torch.from_numpy(Xt).to(device)
    # batch to avoid OOM on large caches
    preds = []
    bs = 1024
    for i in range(0, len(x), bs):
        p = model(x[i:i + bs])
        preds.append(p.cpu().numpy())
    s_t = np.concatenate(preds).astype(np.float32)

    return {
        "s_t": s_t,
        "tokens": cache["tokens"].astype(str),
        "sample_idx": cache["sample_idx"].astype(np.int64),
        "true_severity": cache["true_severity"].astype(np.int64),
        "reliability_target": cache["reliability_target"].astype(np.float32),
        "is_decoy": cache["is_decoy"].astype(bool),
        "corruption_type": str(cache["corruption_type"]),
    }


def emit_score_file(
    model, scaler, model_hash: str, col_subset,
    condition: str, cache_dir: str, out_dir: str, device
) -> str:
    """Emit one per-condition val score file."""
    cache_path = os.path.join(cache_dir, condition + ".npz")
    out_name = condition + "_scores.npz"
    out_path = os.path.join(out_dir, out_name)

    result = predict_cache(model, scaler, cache_path, col_subset, device)

    np.savez_compressed(
        out_path,
        tokens=np.array(result["tokens"]),
        s_t=result["s_t"],
        sample_idx=result["sample_idx"],
        true_severity=result["true_severity"],
        reliability_target=result["reliability_target"],
        is_decoy=result["is_decoy"],
        corruption_type=np.array(result["corruption_type"]),
        model_hash=np.array(model_hash),
    )

    print(f"  {condition:30s} -> {out_name}  "
          f"(N={len(result['s_t'])}, "
          f"s_t range=[{result['s_t'].min():.4f}, {result['s_t'].max():.4f}])")
    return out_path


def emit_identity_file(out_dir: str, cache_dir: str = CACHE_DIR) -> str:
    """Emit identity_scores.npz: same tokens as clean_val_sev0, s_t = 1.0."""
    cache_path = os.path.join(cache_dir, "clean_val_sev0.npz")
    cache = _load_cache(cache_path)
    n = len(cache["tokens"])
    s_t = np.ones(n, dtype=np.float32)

    out_path = os.path.join(out_dir, "identity_scores.npz")
    np.savez_compressed(
        out_path,
        tokens=np.array(cache["tokens"].astype(str)),
        s_t=s_t,
        sample_idx=cache["sample_idx"].astype(np.int64),
        true_severity=cache["true_severity"].astype(np.int64),
        reliability_target=cache["reliability_target"].astype(np.float32),
        is_decoy=cache["is_decoy"].astype(bool),
        corruption_type=np.array("identity"),
        model_hash=np.array("identity"),
    )
    print(f"  {'identity (H1)':30s} -> identity_scores.npz  "
          f"(N={n}, s_t = 1.0 for all)")
    return out_path


def verify_coverage(out_dir: str, cache_dir: str = CACHE_DIR) -> None:
    """: every token in val caches has a matching score."""
    print("\n--- Coverage verification ---")
    all_ok = True
    for cond in VAL_CONDITIONS:
        cache = _load_cache(os.path.join(cache_dir, cond + ".npz"))
        cache_tokens = set(cache["tokens"].astype(str).tolist())
        score_path = os.path.join(out_dir, cond + "_scores.npz")
        if not os.path.exists(score_path):
            print(f"  {cond}: MISSING score file")
            all_ok = False
            continue
        d = np.load(score_path, allow_pickle=True)
        score_tokens = set(d["tokens"].astype(str).tolist())
        if cache_tokens != score_tokens:
            missing = cache_tokens - score_tokens
            extra = score_tokens - cache_tokens
            print(f"  {cond}: MISMATCH (missing={len(missing)}, extra={len(extra)})")
            all_ok = False
        else:
            print(f"  {cond}: OK ({len(score_tokens)} tokens match)")
    print(f"  [{'PASS' if all_ok else 'FAIL'}]")


def main():
    parser = argparse.ArgumentParser(description="Emit Stage C score artifacts")
    parser.add_argument("--model", default=os.path.join(SCORES_DIR, "lcre_model.pt"))
    parser.add_argument("--scaler", default=os.path.join(SCORES_DIR, "lcre_scaler.npz"))
    parser.add_argument("--out-dir", default=SCORES_DIR)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-identity", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # load model + scaler
    ckpt = torch.load(args.model, map_location=device)
    config = ckpt["config"]
    model = LCRE(input_dim=config["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    scaler = Scaler.load(args.scaler)
    model_hash = config["model_hash"]
    col_subset = config.get("col_subset", None)

    print(f"=== emitting Stage C score files ===")
    print(f"  model: {args.model}  (hash={model_hash})")
    print(f"  input_dim={config['input_dim']}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # emit 10 val score files
    for cond in VAL_CONDITIONS:
        emit_score_file(
            model, scaler, model_hash, col_subset,
            cond, args.cache_dir, args.out_dir, device,
        )

    # emit identity file
    if not args.skip_identity:
        emit_identity_file(args.out_dir, args.cache_dir)

    # verify coverage
    verify_coverage(args.out_dir, args.cache_dir)

    # verify s_t in (0, 1) for all score files
    print("\n--- s_t range check ---")
    for cond in VAL_CONDITIONS:
        path = os.path.join(args.out_dir, cond + "_scores.npz")
        d = np.load(path, allow_pickle=True)
        s = d["s_t"]
        lo, hi = float(s.min()), float(s.max())
        ok = 0.0 < lo and hi < 1.0 + 1e-6
        print(f"  {cond:30s}: [{lo:.4f}, {hi:.4f}]  [{'OK' if ok else 'WARN'}]")

    # identity check
    idpath = os.path.join(args.out_dir, "identity_scores.npz")
    if os.path.exists(idpath):
        d = np.load(idpath, allow_pickle=True)
        all_one = np.allclose(d["s_t"], 1.0)
        print(f"  {'identity_scores':30s}: all s_t == 1.0  [{'OK' if all_one else 'FAIL'}]")

    print("\n=== emission complete ===")


if __name__ == "__main__":
    main()
