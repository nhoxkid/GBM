"""Finite-state Markov chain simulation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import RegimeConfig


@dataclass
class RegimeMarkovChain:
    """Samples state paths for a regime-switching diffusion."""

    config: RegimeConfig
    generator: torch.Generator | None = None

    def __post_init__(self) -> None:
        if self.generator is None:
            generator = torch.Generator(device=self.config.device)
            generator.manual_seed(torch.seed())
            object.__setattr__(self, "generator", generator)

    def simulate(self, *, initial_state: int, n_paths: int, n_steps: int) -> torch.Tensor:
        if not 0 <= initial_state < self.config.transition_matrix.shape[0]:
            raise ValueError("Initial state index out of bounds.")
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("Number of paths and steps must be positive.")

        states = torch.empty(
            (n_paths, n_steps + 1),
            dtype=torch.long,
            device=self.config.device,
        )
        states[:, 0] = initial_state

        transition = self.config.transition_matrix
        for step in range(n_steps):
            current = states[:, step]
            probs = transition.index_select(0, current)
            samples = torch.multinomial(probs, num_samples=1, generator=self.generator)
            states[:, step + 1] = samples.squeeze(1)

        return states
