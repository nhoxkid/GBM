"""Configuration helpers for regime-switching GBM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _as_tensor(data: torch.Tensor | Sequence[float], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(data, dtype=dtype, device=device)
    if tensor.dim() == 0:
        raise ValueError("Tensor data must not be scalar.")
    return tensor


@dataclass
class RegimeConfig:
    """Encapsulates Markov-regime parameters for GBM."""

    transition_matrix: torch.Tensor
    drift: torch.Tensor
    volatility: torch.Tensor
    state_labels: Sequence[str]
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        transition = _as_tensor(self.transition_matrix, dtype=self.dtype, device=self.device)
        drift = _as_tensor(self.drift, dtype=self.dtype, device=self.device)
        volatility = _as_tensor(self.volatility, dtype=self.dtype, device=self.device)

        if transition.dim() != 2 or transition.shape[0] != transition.shape[1]:
            raise ValueError("Transition matrix must be square.")
        n_states = transition.shape[0]
        if drift.shape != (n_states,):
            raise ValueError("Drift vector must match number of states.")
        if volatility.shape != (n_states,):
            raise ValueError("Volatility vector must match number of states.")
        if len(self.state_labels) != n_states:
            raise ValueError("State labels must match number of states.")
        if torch.any(transition < 0):
            raise ValueError("Transition probabilities must be non-negative.")
        row_sums = transition.sum(dim=1)
        ones = torch.ones_like(row_sums)
        if not torch.allclose(row_sums, ones, atol=1e-10, rtol=0.0):
            raise ValueError("Transition matrix rows must sum to 1.")
        if torch.any(volatility <= 0):
            raise ValueError("Volatility values must be strictly positive.")

        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "volatility", volatility)

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "RegimeConfig":
        """Return a copy of the config moved to a new device/dtype."""
        target_device = device or self.device
        target_dtype = dtype or self.dtype
        return RegimeConfig(
            transition_matrix=self.transition_matrix.to(device=target_device, dtype=target_dtype),
            drift=self.drift.to(device=target_device, dtype=target_dtype),
            volatility=self.volatility.to(device=target_device, dtype=target_dtype),
            state_labels=self.state_labels,
            device=target_device,
            dtype=target_dtype,
        )


def build_default_config(*, device: torch.device, dtype: torch.dtype = torch.float64) -> RegimeConfig:
    """Factory for a sample three-regime configuration."""
    transition_matrix = [
        [0.82, 0.15, 0.03],
        [0.20, 0.70, 0.10],
        [0.10, 0.25, 0.65],
    ]
    drift = [0.08, 0.03, -0.02]
    volatility = [0.15, 0.30, 0.45]
    labels = ("bull", "neutral", "bear")
    return RegimeConfig(
        transition_matrix=transition_matrix,
        drift=drift,
        volatility=volatility,
        state_labels=labels,
        device=device,
        dtype=dtype,
    )
