"""GBM path simulation using PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import torch

from .config import RegimeConfig
from .markov import RegimeMarkovChain


@dataclass
class SimulationResult:
    time_grid: torch.Tensor
    prices: torch.Tensor
    states: torch.Tensor


class StreamingSimulation:
    """Incrementally evaluates regime-switching GBM paths for real-time visualisation."""

    def __init__(
        self,
        *,
        config: RegimeConfig,
        generator: torch.Generator,
        n_paths: int,
        n_steps: int,
        horizon: float,
        s0: float,
        initial_state: int,
    ) -> None:
        if horizon <= 0:
            raise ValueError("Time horizon must be positive.")
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if n_paths <= 0:
            raise ValueError("Number of paths must be positive.")
        if not 0 <= initial_state < config.transition_matrix.shape[0]:
            raise ValueError("Initial state index out of bounds.")

        self.config = config
        self.generator = generator
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.horizon = horizon
        self.s0 = s0
        self.initial_state = initial_state

        self.dt = horizon / n_steps
        self.sqrt_dt = math.sqrt(self.dt)

        self.time_grid = torch.linspace(
            0.0,
            horizon,
            n_steps + 1,
            dtype=config.dtype,
            device=config.device,
        )
        self._time_cpu = self.time_grid.detach().cpu().numpy()

        self._current_step = 0
        self._completed = False
        self._result_cache: Optional[SimulationResult] = None

        self._prices = torch.full(
            (n_paths,),
            fill_value=s0,
            dtype=config.dtype,
            device=config.device,
        )
        self._states = torch.full(
            (n_paths,),
            fill_value=initial_state,
            dtype=torch.long,
            device=config.device,
        )

        self._price_history = [self._prices.clone()]
        self._state_history = [self._states.clone()]

        self._visual_count = 0
        self._visual_buffer: Optional[np.ndarray] = None

    def set_visual_count(self, count: int) -> int:
        actual = min(max(count, 0), self.n_paths)
        self._visual_count = actual
        if actual == 0:
            self._visual_buffer = None
            return 0

        buffer = np.empty((actual, self.n_steps + 1), dtype=float)
        for idx, prices in enumerate(self._price_history):
            buffer[:, idx] = prices[:actual].detach().cpu().numpy()
        self._visual_buffer = buffer
        return actual

    def _advance(self) -> None:
        if self._current_step >= self.n_steps:
            self._completed = True
            return

        mu = self.config.drift.index_select(0, self._states)
        sigma = self.config.volatility.index_select(0, self._states)
        shocks = torch.randn(
            (self.n_paths,),
            generator=self.generator,
            dtype=self.config.dtype,
            device=self.config.device,
        ) * self.sqrt_dt
        increment = (mu - 0.5 * sigma.pow(2)) * self.dt + sigma * shocks
        self._prices = self._prices * torch.exp(increment)
        self._price_history.append(self._prices.clone())

        probs = self.config.transition_matrix.index_select(0, self._states)
        next_states = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(1)
        self._states = next_states
        self._state_history.append(self._states.clone())

        self._current_step += 1

        if self._visual_buffer is not None:
            self._visual_buffer[:, self._current_step] = (
                self._prices[: self._visual_count].detach().cpu().numpy()
            )

        if self._current_step >= self.n_steps:
            self._completed = True

    def ensure_step(self, step: int) -> None:
        target = min(step, self.n_steps)
        while self._current_step < target:
            self._advance()

    def ensure_complete(self) -> None:
        if not self._completed:
            self.ensure_step(self.n_steps)

    def get_visual_snapshot(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        if self._visual_count == 0:
            self.ensure_step(frame)
            return self._time_cpu[: frame + 1], np.empty((0, frame + 1))

        self.ensure_step(frame)
        if self._visual_buffer is None:
            raise RuntimeError("Visual buffer not initialised. Call set_visual_count first.")
        return self._time_cpu[: frame + 1], self._visual_buffer[:, : frame + 1]

    def to_result(self) -> SimulationResult:
        if self._result_cache is None:
            self.ensure_complete()
            prices = torch.stack(self._price_history, dim=0).transpose(0, 1).contiguous()
            states = torch.stack(self._state_history, dim=0).transpose(0, 1).contiguous()
            self._result_cache = SimulationResult(
                time_grid=self.time_grid,
                prices=prices,
                states=states,
            )
        return self._result_cache


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

    def simulate_stream(
        self,
        *,
        n_paths: int,
        n_steps: int,
        horizon: float,
        s0: float,
        initial_state: int,
    ) -> StreamingSimulation:
        return StreamingSimulation(
            config=self.config,
            generator=self.generator,
            n_paths=n_paths,
            n_steps=n_steps,
            horizon=horizon,
            s0=s0,
            initial_state=initial_state,
        )
