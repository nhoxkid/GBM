"""Runtime helpers shared by the CLI and Qt interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from .config import RegimeConfig, build_default_config
from .distributions import attach_randomness
from .simulator import GBMSimulator

DEFAULT_DRIFT_STD: tuple[float, ...] = (0.015, 0.010, 0.020)
DEFAULT_VOLATILITY_CV: tuple[float, ...] = (0.12, 0.18, 0.25)


@dataclass(frozen=True)
class SimulationContext:
    """Holds the configured simulator and metadata for a run."""

    config: RegimeConfig
    simulator: GBMSimulator
    save_dir: Path


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def precision_to_dtype(precision: str) -> torch.dtype:
    return torch.float64 if precision == "float64" else torch.float32


def manual_seed_or_random(generator: torch.Generator, seed: Optional[int]) -> None:
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(torch.seed())


def expand_sequence(
    raw: Optional[Sequence[float]],
    default: Sequence[float],
    length: int,
    name: str,
) -> list[float]:
    """Expand or validate a per-state sequence."""
    if raw is None:
        values = list(default)
    else:
        if len(raw) == 1:
            values = [float(raw[0])] * length
        elif len(raw) == length:
            values = [float(v) for v in raw]
        else:
            raise ValueError(f"Expected 1 or {length} values for {name}, got {len(raw)}.")
    return values


def create_simulation_context(
    *,
    device: str,
    precision: str,
    seed: Optional[int],
    deterministic_params: bool,
    drift_std: Optional[Sequence[float]],
    volatility_cv: Optional[Sequence[float]],
    save_dir: Path,
) -> SimulationContext:
    resolved_device = resolve_device(device)
    dtype = precision_to_dtype(precision)

    generator = torch.Generator(device=resolved_device)
    manual_seed_or_random(generator, seed)

    config = build_default_config(device=resolved_device, dtype=dtype)

    if not deterministic_params:
        n_states = config.transition_matrix.shape[0]
        drift_std_values = expand_sequence(
            drift_std,
            DEFAULT_DRIFT_STD,
            n_states,
            "drift standard deviation",
        )
        vol_cv_values = expand_sequence(
            volatility_cv,
            DEFAULT_VOLATILITY_CV,
            n_states,
            "volatility coefficient of variation",
        )
        attach_randomness(config, drift_std=drift_std_values, volatility_cv=vol_cv_values)

    simulator = GBMSimulator(config=config, generator=generator)
    return SimulationContext(config=config, simulator=simulator, save_dir=save_dir.expanduser())


def fmt(value: float) -> str:
    return f"{value:.6f}"
