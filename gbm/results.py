"""Result dataclasses for GBM simulations."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SimulationResult:
    time_grid: torch.Tensor
    prices: torch.Tensor
    states: torch.Tensor


@dataclass
class SimulationFrame:
    """Represents a streaming update for a set of GBM paths."""

    time_index: int
    time_value: float
    prices: torch.Tensor
    states: torch.Tensor
    drift: torch.Tensor
    volatility: torch.Tensor
