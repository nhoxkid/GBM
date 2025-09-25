"""Streaming utilities for GBM simulations."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .config import RegimeConfig
from .distributions import sample_parameters
from .results import SimulationFrame, SimulationResult


class StreamingSimulation:
    """Incrementally evaluates regime-switching GBM paths for visualisation."""

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
        infinite: bool = False,
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
        self.max_steps = n_steps
        self.horizon = horizon
        self.infinite = infinite
        self.s0 = s0
        self.initial_state = initial_state

        self.dt = horizon / n_steps
        self.sqrt_dt = math.sqrt(self.dt)

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
        self._time_values = [0.0]

        self._visual_count = 0

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def is_complete(self) -> bool:
        return self._completed

    def set_visual_count(self, count: int) -> int:
        actual = min(max(count, 0), self.n_paths)
        self._visual_count = actual
        return actual

    def step(self) -> SimulationFrame:
        if not self.infinite and self._current_step >= self.max_steps:
            self._completed = True
            raise StopIteration("Simulation already completed.")

        mu, sigma = sample_parameters(self.config, self._states, self.generator)
        shocks = torch.randn(
            (self.n_paths,),
            generator=self.generator,
            dtype=self.config.dtype,
            device=self.config.device,
        ) * self.sqrt_dt
        increment = (mu - 0.5 * sigma.pow(2)) * self.dt + sigma * shocks
        self._prices = self._prices * torch.exp(increment)

        probs = self.config.transition_matrix.index_select(0, self._states)
        next_states = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(1)
        self._states = next_states

        self._current_step += 1
        if not self.infinite and self._current_step >= self.max_steps:
            self._completed = True

        current_time = self._time_values[-1] + self.dt
        self._time_values.append(current_time)
        self._price_history.append(self._prices.clone())
        self._state_history.append(self._states.clone())

        return SimulationFrame(
            time_index=self._current_step,
            time_value=float(current_time),
            prices=self._prices.clone(),
            states=self._states.clone(),
            drift=mu.clone(),
            volatility=sigma.clone(),
        )

    def ensure_step(self, step: int) -> None:
        while self._current_step < step:
            try:
                self.step()
            except StopIteration:
                break

    def ensure_complete(self) -> None:
        if self.infinite:
            raise RuntimeError("Infinite simulations do not complete automatically.")
        if not self._completed:
            self.ensure_step(self.max_steps)

    def get_visual_snapshot(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        if frame > self._current_step:
            self.ensure_step(frame)

        times = np.array(self._time_values[: frame + 1], dtype=float)
        if self._visual_count == 0:
            return times, np.empty((0, frame + 1))

        stacked = torch.stack(self._price_history[: frame + 1], dim=0).transpose(0, 1)
        sample = stacked[: self._visual_count].detach().cpu().numpy()
        return times, sample

    def to_result(self) -> SimulationResult:
        if self._result_cache is None:
            time_grid = torch.tensor(
                self._time_values,
                dtype=self.config.dtype,
                device=self.config.device,
            )
            prices = torch.stack(self._price_history, dim=0).transpose(0, 1).contiguous()
            states = torch.stack(self._state_history, dim=0).transpose(0, 1).contiguous()
            self._result_cache = SimulationResult(
                time_grid=time_grid,
                prices=prices,
                states=states,
            )
        return self._result_cache
