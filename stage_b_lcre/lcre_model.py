"""LCRE (Learned Cross-modal Reliability Estimator) model definition.

A deliberately small MLP (~270k params) that ingests a 1017-dim per-frame
feature vector and emits a continuous scalar S_t in (0, 1) via a sigmoid head.

Architecture (per stage_b_implementation_plan.md, Phase 2 step 3):
    Linear(1017, 256) -> BatchNorm1d -> ReLU -> Dropout(0.2)
    Linear(256, 64)   -> BatchNorm1d -> ReLU -> Dropout(0.2)
    Linear(64, 1)     -> sigmoid     -> squeeze

The sigmoid head asymptotes at (0, 1) so R(t) stays finite at both extremes
(required for the H1 identity check in Stage C).

Stage C imports this module + the saved state_dict + lcre_scaler.npz for
inference. The model is the ONLY trainable component in the pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LCRE(nn.Module):
    """Learned Cross-modal Reliability Estimator.

    Parameters
    ----------
    input_dim : int
        Width of the per-frame feature vector. Default 1017 for the full
        model (1008 pooled BEV + 4 synthetic telemetry + 3 real telemetry
        + 2 NaN indicators). Pass a smaller value for ablation variants.
    hidden1 : int
        First hidden layer width (default 256).
    hidden2 : int
        Second hidden layer width (default 64).
    dropout : float
        Dropout probability (default 0.2).
    """

    def __init__(
        self,
        input_dim: int = 1017,
        hidden1: int = 256,
        hidden2: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return S_t in (0, 1), shape (batch,)."""
        logits = self.net(x)
        s_t = torch.sigmoid(logits).squeeze(-1)
        return s_t

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
