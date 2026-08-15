"""Train the LCRE on the cached Stage A features with a scene-wise held-out split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from lcre_model import LCRE, count_params
from lcre_dataset import (
    ABLATION_SPECS,
    FULL_INPUT_DIM,
    LCREDataset,
    SCORES_DIR,
    Scaler,
    prepare_data,
    save_scaler,
    INDICATOR_COLS,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sample_weights(
    true_severity: np.ndarray, clean_weight: float
) -> np.ndarray:
    """Per-sample loss weights. Clean (sev 0) frames weighted x``clean_weight``,."""
    w = np.ones(len(true_severity), dtype=np.float32)
    w[true_severity == 0] = clean_weight
    return w


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mse, preds, targets, true_severity, is_decoy)."""
    model.eval()
    preds_list, tgt_list, sev_list, decoy_list = [], [], [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        x, target, sev, decoy, _, _, _ = batch
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, target)
        total_loss += loss.item() * len(target)
        n += len(target)
        preds_list.append(pred.cpu().numpy())
        tgt_list.append(target.cpu().numpy())
        sev_list.append(sev.numpy())
        decoy_list.append(decoy.numpy())
    mse = total_loss / max(n, 1)
    return (
        mse,
        np.concatenate(preds_list),
        np.concatenate(tgt_list),
        np.concatenate(sev_list),
        np.concatenate(decoy_list),
    )


class WeightedDataset(torch.utils.data.Dataset):
    """Wraps LCREDataset to also yield a per-sample weight index."""

    def __init__(self, base: LCREDataset, weights: np.ndarray) -> None:
        self.base = base
        self.weights = weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        return item + (torch.tensor(self.weights[idx]),)


def model_hash(state_dict: Dict) -> str:
    """Short hash of the model state_dict for artifact self-description."""
    buf = b""
    for k in sorted(state_dict.keys()):
        buf += k.encode()
        buf += state_dict[k].cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()[:12]


def train(
    ablation: str = "full",
    epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    clean_weight: float = 1.0,
    dropout: float = 0.2,
    seed: int = 20260101,
    device_str: Optional[str] = None,
    tag: str = "",
) -> Dict:
    """Train one LCRE variant. Returns a dict of results + artifacts."""
    set_seed(seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device={device}  ablation={ablation}  clean_weight={clean_weight}")

    # data
    fit_data, heldout_data, scaler, scene_split = prepare_data()
    save_scaler(scaler)

    col_subset = ABLATION_SPECS[ablation]
    input_dim = len(col_subset)
    print(f"[train] input_dim={input_dim}  (ablation={ablation})")

    # transform once, then stage on GPU as a single tensor
    X_fit = scaler.transform(fit_data.X)[:, col_subset].astype(np.float32)
    y_fit = fit_data.targets.astype(np.float32)
    X_held = scaler.transform(heldout_data.X)[:, col_subset].astype(np.float32)
    y_held = heldout_data.targets.astype(np.float32)
    w_fit = compute_sample_weights(fit_data.true_severity, clean_weight)

    # stage on GPU (the whole 45k samples fit in ~185MB on a 16GB A4000)
    X_fit_t = torch.from_numpy(X_fit).to(device)
    y_fit_t = torch.from_numpy(y_fit).to(device)
    w_fit_t = torch.from_numpy(w_fit).to(device)
    X_held_t = torch.from_numpy(X_held).to(device)
    y_held_t = torch.from_numpy(y_held).to(device)

    n_fit = X_fit_t.shape[0]
    n_held = X_held_t.shape[0]
    print(f"[train] staged on GPU: fit={n_fit}  held={n_held}")

    # model
    model = LCRE(input_dim=input_dim, dropout=dropout).to(device)
    n_params = count_params(model)
    print(f"[train] model params: {n_params:,}  dropout={dropout}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, verbose=False
    )

    # training loop
    ablation_suffix = "" if ablation == "full" else f"_{ablation}"
    tag_suffix = f"_{tag}" if tag else ""
    suffix = f"{ablation_suffix}{tag_suffix}"
    log_path = os.path.join(SCORES_DIR, f"train_log{suffix}.csv")
    model_path = os.path.join(SCORES_DIR, f"lcre_model{suffix}.pt")

    best_held_mse = float("inf")
    best_state = None
    best_preds = best_tgt = best_sev = best_decoy = None
    no_improve = 0
    log_rows: List[Dict] = []

    print(f"[train] starting training: {epochs} epochs, patience={patience}, "
          f"batch_size={batch_size}")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_fit, device=device)
        train_loss = 0.0
        for i in range(0, n_fit, batch_size):
            idx = perm[i:i + batch_size]
            x = X_fit_t[idx]
            target = y_fit_t[idx]  # reliability_target = 1 - severity/3
            w = w_fit_t[idx]
            pred = model(x)  # S_t ∈ (0,1) via sigmoid head
            # MSE loss: S_t → reliability_target.
            # S_t feeds Stage C's R(t) = R_base·(1+κ·(1−S_t)) ().
            # Sigmoid keeps S_t in (0,1), so R(t) is bounded and
            # identity (S_t=1 → R(t)=R_base) is preserved for H1.
            loss = ((pred - target) ** 2 * w).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
        train_loss /= max(n_fit, 1)

        # evaluate held-out (forward in batches, no grad)
        model.eval()
        with torch.no_grad():
            held_pred = model(X_held_t)
            held_mse = ((held_pred - y_held_t) ** 2).mean().item()

        scheduler.step(held_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        log_rows.append({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "heldout_loss": f"{held_mse:.6f}",
            "lr": f"{current_lr:.2e}",
        })
        print(f"  epoch {epoch:3d}  train_mse={train_loss:.5f}  "
              f"held_mse={held_mse:.5f}  lr={current_lr:.2e}")

        if held_mse < best_held_mse - 1e-7:
            best_held_mse = held_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds = held_pred.cpu().numpy()
            best_tgt = y_held
            best_sev = heldout_data.true_severity
            best_decoy = heldout_data.is_decoy
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stopping at epoch {epoch} (patience {patience})")
                break

    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s  best held_mse={best_held_mse:.6f}")

    # save artifacts
    # write train log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "heldout_loss", "lr"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train] log -> {log_path}")

    # save model (state_dict + config + hash)
    mhash = model_hash(best_state)
    config = {
        "input_dim": input_dim,
        "ablation": ablation,
        "tag": tag,
        "hidden1": 256,
        "hidden2": 64,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "col_subset": col_subset,
        "clean_weight": clean_weight,
        "seed": seed,
        "best_held_mse": float(best_held_mse),
        "model_hash": mhash,
        # R(t) supervision contract (Empirical Study Plan ):
        #   R(t) = R_base · (1 + κ · (1 − S_t)),  κ > 0, S_t ∈ (0, 1]
        # S_t is trained to reliability_target = 1 − severity/3.
        # κ is a Stage C hyperparameter (sweep {1,3,5}), not trained here.
        "r_rule": "R(t) = R_base * (1 + kappa * (1 - S_t))",
        "r_rule_kappa": "Stage C hyperparameter (sweep {1,3,5})",
        "s_t_domain": "(0, 1]",
        "supervision_target": "reliability_target = 1 - severity/3",
        "identity_condition": "S_t = 1.0 => R(t) = R_base (H1 check)",
    }
    torch.save({"state_dict": best_state, "config": config}, model_path)
    print(f"[train] model -> {model_path}  (hash={mhash})")

    return {
        "model": model,
        "best_state": best_state,
        "best_held_mse": best_held_mse,
        "held_preds": best_preds,
        "held_tgt": best_tgt,
        "held_sev": best_sev,
        "held_decoy": best_decoy,
        "config": config,
        "scaler": scaler,
        "scene_split": scene_split,
        "heldout_data": heldout_data,
        "log_path": log_path,
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train LCRE (Stage B)")
    parser.add_argument("--ablation", choices=list(ABLATION_SPECS.keys()),
                        default="full",
                        help="Feature subset: full (1017), bev (1010), telemetry (9)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="In-GPU batch size (large keeps the small MLP's GPU busy)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--clean-weight", type=float, default=1.0,
                        help="Loss weight for clean (sev0) frames (1.0 = no weighting)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability (increase to fight telemetry shortcut)")
    parser.add_argument("--tag", type=str, default="",
                        help="Filename suffix for this variant (e.g. 'reg' -> lcre_model_reg.pt)")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    result = train(
        ablation=args.ablation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        clean_weight=args.clean_weight,
        dropout=args.dropout,
        seed=args.seed,
        device_str=args.device,
        tag=args.tag,
    )

    print(f"\n=== training complete ({args.ablation}) ===")
    print(f"  best held-out MSE: {result['best_held_mse']:.6f}")
    print(f"  model: {result['model_path']}")
    print(f"  hash:  {result['config']['model_hash']}")


if __name__ == "__main__":
    main()
