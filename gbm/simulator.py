"""GBM path simulation using PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .config import RegimeConfig
from .markov import RegimeMarkovChain


@dataclass
class SimulationResult:
    time_grid: torch.Tensor
    prices: torch.Tensor
    states: torch.Tensor


class GBMSimulator:
    """Simulates regime-switching GBM paths with Ito discretisation."""

    def __init__(self, config: RegimeConfig, generator: torch.Generator | None = None) -> None:
        self.config = config
        if generator is None:
            generator = torch.Generator(device=config.device)
            generator.manual_seed(torch.seed())
        self.generator = generator

    def simulate(
        self,
        *,
        n_paths: int,
        n_steps: int,
        horizon: float,
        s0: float,
        initial_state: int,
    ) -> SimulationResult:
        if horizon <= 0:
            raise ValueError("Time horizon must be positive.")
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if n_paths <= 0:
            raise ValueError("Number of paths must be positive.")

        dt = horizon / n_steps
        sqrt_dt = math.sqrt(dt)

        chain = RegimeMarkovChain(config=self.config, generator=self.generator)
        states = chain.simulate(initial_state=initial_state, n_paths=n_paths, n_steps=n_steps)

        prices = torch.empty(
            (n_paths, n_steps + 1),
            dtype=self.config.dtype,
            device=self.config.device,
        )
        prices[:, 0] = s0

        shocks = torch.randn(
            (n_paths, n_steps),
            generator=self.generator,
            dtype=self.config.dtype,
            device=self.config.device,
        ) * sqrt_dt

        for step in range(n_steps):
            idx = states[:, step]
            mu = self.config.drift.index_select(0, idx)
            sigma = self.config.volatility.index_select(0, idx)
            increment = (mu - 0.5 * sigma.pow(2)) * dt + sigma * shocks[:, step]
            prices[:, step + 1] = prices[:, step] * torch.exp(increment)

        time_grid = torch.linspace(
            0.0,
            horizon,
            n_steps + 1,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        return SimulationResult(time_grid=time_grid, prices=prices, states=states)
