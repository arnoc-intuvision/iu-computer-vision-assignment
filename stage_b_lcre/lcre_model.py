"""LCRE model: a small MLP mapping per-frame features to a reliability scalar."""

from __future__ import annotations

import torch
import torch.nn as nn


class LCRE(nn.Module):
    """Learned Cross-modal Reliability Estimator."""

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
