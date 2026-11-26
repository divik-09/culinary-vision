from __future__ import annotations

import torch
from torch import nn


class TemporalGRU(nn.Module):
    """Compact sequence classifier over frame-level culinary visual features."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        sequence, _ = self.gru(x)
        # Pool over time to reduce sensitivity to the exact boundary frame.
        pooled = sequence.mean(dim=1)
        return self.head(pooled)
